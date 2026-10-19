"""Context compression: fit conversation history into a token budget.

Pieces:

* :func:`truncate_to_tokens` - token-bounded truncation at word boundaries.
* :class:`ExtractiveSummarizer` - offline, deterministic sentence-extraction summarizer.
* :class:`LLMSummarizer` - delegates to any :class:`~memory_agent.interfaces.LLM`, with fallback.
* :class:`ContextCompressor` - keeps recent turns verbatim and summarizes older ones.
* :class:`RollingSummary` - folds evicted short-term messages into a bounded running summary.

All budgets are measured with a pluggable :data:`~memory_agent.tokens.TokenCounter`, which is
assumed to be (roughly) monotone in text length.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .embeddings import tokenize
from .interfaces import LLM, Summarizer
from .tokens import MESSAGE_OVERHEAD, TokenCounter, estimate_tokens, message_tokens
from .types import Message, Role

ELLIPSIS = "…"
PRIOR_SUMMARY_MARKER = "[Earlier summary]"

_SPEAKERS: dict[str, str] = {"user": "User", "assistant": "Assistant"}
_PREFIX_RE = re.compile(r"^\s*(User|Assistant):\s*(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")
_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|im|i've|i'd|i'll|my|me|mine|myself)\b")
_DIGIT_RE = re.compile(r"\d")

_FILLER_WORDS = frozenset(
    "hi hello hey hiya yo greetings thanks thank thx ty you ok okay k sure great cool bye "
    "goodbye see ya later good morning afternoon evening night welcome yes no yeah yep yup nope "
    "nice awesome please glad help how are can i today doing well fine sounds got it alright "
    "np lol haha there what's up whats is am the a an so to with for that's thats perfect "
    "wonderful excellent anything else again much very really".split()
)

SELF_DISCLOSURE_BOOST = 1.5
DIGIT_BOOST = 1.3
PRIOR_SUMMARY_BOOST = 1.2
FILLER_PENALTY = 0.2


# --------------------------------------------------------------------------- truncation


def _longest_fitting_prefix(text: str, fits: Any) -> int:
    """Largest ``n`` in ``[0, len(text)]`` with ``fits(n)``; assumes ``fits`` is monotone."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def truncate_to_tokens(text: str, max_tokens: int, counter: TokenCounter = estimate_tokens) -> str:
    """Return ``text`` cut so that ``counter(result) <= max_tokens``.

    Text within budget is returned unchanged. Otherwise the cut prefers a word boundary and an
    ellipsis (counted within the budget) marks the truncation. ``max_tokens <= 0`` yields ``""``.
    """
    if max_tokens <= 0 or not text:
        return ""
    if counter(text) <= max_tokens:
        return text

    if counter(ELLIPSIS) > max_tokens:
        # Not even the marker fits: return the largest bare prefix that does.
        n = _longest_fitting_prefix(text, lambda k: counter(text[:k]) <= max_tokens)
        return text[:n].rstrip()

    n = _longest_fitting_prefix(text, lambda k: counter(text[:k].rstrip() + ELLIPSIS) <= max_tokens)
    prefix = text[:n]
    if n < len(text) and prefix and not text[n].isspace() and not prefix[-1].isspace():
        # We would cut mid-word: back off to the last whitespace when there is one.
        cut = max(prefix.rfind(" "), prefix.rfind("\n"), prefix.rfind("\t"))
        if cut > 0:
            prefix = prefix[:cut]
    result = prefix.rstrip() + ELLIPSIS
    # Guard against non-monotone counters.
    while counter(result) > max_tokens and len(result) > len(ELLIPSIS):
        result = result[: -len(ELLIPSIS) - 1].rstrip() + ELLIPSIS
    return result if counter(result) <= max_tokens else ""


# --------------------------------------------------------------------------- helpers


def prior_summary_message(text: str) -> Message:
    """Wrap an existing summary as a pseudo-message so summarizers can carry it forward."""
    return Message(
        role="assistant",
        content=f"{PRIOR_SUMMARY_MARKER}\n{text}",
        timestamp=0.0,
        metadata={"summary": True},
    )


def _is_prior_summary(message: Message) -> bool:
    return bool(message.metadata.get("summary")) and message.content.startswith(
        PRIOR_SUMMARY_MARKER
    )


def render_transcript(messages: Sequence[Message]) -> str:
    """Render user/assistant messages as ``Speaker: text`` lines (system messages skipped)."""
    lines: list[str] = []
    for m in messages:
        if m.role == "system":
            continue
        if _is_prior_summary(m):
            body = m.content[len(PRIOR_SUMMARY_MARKER) :].strip()
            if body:
                lines.append(f"Earlier summary:\n{body}")
            continue
        lines.append(f"{_SPEAKERS[m.role]}: {m.content.strip()}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _Sentence:
    index: int
    role: Role
    text: str
    prior: bool

    def render(self) -> str:
        return f"{_SPEAKERS[self.role]}: {self.text}"


def _split_sentences(messages: Sequence[Message]) -> list[_Sentence]:
    out: list[_Sentence] = []
    for m in messages:
        if m.role not in _SPEAKERS:
            continue
        prior = _is_prior_summary(m)
        for line in m.content.splitlines():
            line = line.strip()
            if not line or line == PRIOR_SUMMARY_MARKER:
                continue
            role: Role = m.role
            match = _PREFIX_RE.match(line)
            if match:
                role = "user" if match.group(1) == "User" else "assistant"
                line = match.group(2).strip()
            for sentence in _SENTENCE_SPLIT_RE.split(line):
                sentence = sentence.strip()
                if sentence:
                    out.append(_Sentence(len(out), role, sentence, prior))
    return out


def _is_filler(text: str) -> bool:
    words = _WORD_RE.findall(text.lower())
    if not words or len(words) > 8:
        return False
    filler = sum(1 for w in words if w in _FILLER_WORDS)
    return filler / len(words) >= 0.6


# --------------------------------------------------------------------------- summarizers


class ExtractiveSummarizer:
    """Deterministic extractive summarizer (satisfies :class:`Summarizer`).

    Sentences are scored by normalized term frequency across the window, boosted for user
    self-disclosure ("I ...", "my ...") and digits, and penalized for greetings/filler. The best
    sentences that fit the budget are emitted chronologically as ``User: ...`` /
    ``Assistant: ...`` lines.
    """

    def __init__(self, token_counter: TokenCounter = estimate_tokens) -> None:
        self.token_counter = token_counter

    def score_sentences(self, messages: Sequence[Message]) -> list[tuple[float, str]]:
        """``(score, rendered line)`` for every candidate sentence, in chronological order."""
        sentences = _split_sentences(messages)
        return [
            (score, s.render()) for s, score in zip(sentences, self._scores(sentences), strict=True)
        ]

    def _scores(self, sentences: Sequence[_Sentence]) -> list[float]:
        token_lists = [tokenize(s.text) for s in sentences]
        tf = Counter(t for tokens in token_lists for t in tokens)
        max_tf = max(tf.values(), default=1)
        scores: list[float] = []
        for s, tokens in zip(sentences, token_lists, strict=True):
            unique = set(tokens)
            if unique:
                base = 1.0 + sum(tf[t] / max_tf for t in unique) / len(unique)
            else:
                base = 0.5
            lowered = s.text.lower()
            if s.role == "user" and _FIRST_PERSON_RE.search(lowered):
                base *= SELF_DISCLOSURE_BOOST
            if _DIGIT_RE.search(s.text):
                base *= DIGIT_BOOST
            if s.prior:
                base *= PRIOR_SUMMARY_BOOST
            if _is_filler(s.text):
                base *= FILLER_PENALTY
            scores.append(base)
        return scores

    def summarize(self, messages: Sequence[Message], max_tokens: int) -> str:
        if max_tokens <= 0 or not messages:
            return ""
        sentences = _split_sentences(messages)
        if not sentences:
            return ""
        count = self.token_counter
        scores = self._scores(sentences)
        ranked = sorted(sentences, key=lambda s: (-scores[s.index], s.index))

        chosen: list[_Sentence] = []
        for candidate in ranked:
            trial = sorted([*chosen, candidate], key=lambda s: s.index)
            if count("\n".join(s.render() for s in trial)) <= max_tokens:
                chosen = trial

        if not chosen:
            return truncate_to_tokens(ranked[0].render(), max_tokens, count)
        return truncate_to_tokens("\n".join(s.render() for s in chosen), max_tokens, count)


DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "You compress conversations for a memory-enabled assistant. Write a concise, factual "
    "summary that preserves what the user said about themselves (names, preferences, plans, "
    "numbers, dates) and any commitments the assistant made. Do not invent details."
)


class LLMSummarizer:
    """Summarizer backed by an :class:`LLM`, falling back to another summarizer on failure."""

    def __init__(
        self,
        llm: LLM,
        fallback: Summarizer | None = None,
        token_counter: TokenCounter = estimate_tokens,
        system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.token_counter = token_counter
        self.fallback: Summarizer = fallback or ExtractiveSummarizer(token_counter)
        self.system_prompt = system_prompt

    def build_prompt(self, messages: Sequence[Message], max_tokens: int) -> str:
        transcript = render_transcript(messages)
        return (
            f"Conversation:\n{transcript}\n\n"
            f"Summarize the conversation above in at most {max_tokens} tokens. "
            "Keep facts the user shared about themselves and any open tasks."
        )

    def summarize(self, messages: Sequence[Message], max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        relevant = [m for m in messages if m.role != "system"]
        if not relevant:
            return ""
        prompt = Message(role="user", content=self.build_prompt(relevant, max_tokens))
        try:
            output = self.llm.complete(self.system_prompt, [prompt])
        except Exception:  # noqa: BLE001 - any provider failure triggers the fallback
            output = ""
        output = (output or "").strip()
        if not output:
            output = self.fallback.summarize(messages, max_tokens).strip()
        return truncate_to_tokens(output, max_tokens, self.token_counter)


# --------------------------------------------------------------------------- compressor


@dataclass
class CompressedContext:
    """Result of :meth:`ContextCompressor.compress`.

    ``tokens`` counts the summary as one message-equivalent (including overhead) when it is
    non-empty, plus every kept message.
    """

    summary: str
    messages: list[Message] = field(default_factory=list)
    compressed_count: int = 0
    tokens: int = 0


class ContextCompressor:
    """Keeps the most recent turns verbatim and summarizes the rest into a token budget."""

    def __init__(
        self,
        summarizer: Summarizer | None = None,
        token_counter: TokenCounter = estimate_tokens,
        summary_ratio: float = 0.3,
        min_recent: int = 2,
    ) -> None:
        if not 0.0 <= summary_ratio < 1.0:
            raise ValueError("summary_ratio must be in [0, 1)")
        if min_recent < 0:
            raise ValueError("min_recent must be >= 0")
        self.token_counter = token_counter
        self.summarizer: Summarizer = summarizer or ExtractiveSummarizer(token_counter)
        self.summary_ratio = summary_ratio
        self.min_recent = min_recent

    def summary_tokens(self, summary: str) -> int:
        """Cost of a summary as one message-equivalent (0 when empty)."""
        return self.token_counter(summary) + MESSAGE_OVERHEAD if summary else 0

    def _summarize(self, dropped: Sequence[Message], existing: str, available: int) -> str:
        if available <= 0:
            return ""
        inputs = ([prior_summary_message(existing)] if existing else []) + list(dropped)
        if not inputs:
            return ""
        text = (self.summarizer.summarize(inputs, available) or "").strip()
        return truncate_to_tokens(text, available, self.token_counter)

    def compress(
        self,
        messages: Sequence[Message],
        budget_tokens: int,
        existing_summary: str = "",
    ) -> CompressedContext:
        """Fit ``existing_summary`` + ``messages`` into ``budget_tokens`` (never exceeded)."""
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        msgs = list(messages)
        existing = (existing_summary or "").strip()
        costs = [message_tokens(m, self.token_counter) for m in msgs]

        total = self.summary_tokens(existing) + sum(costs)
        if total <= budget_tokens:
            return CompressedContext(existing, msgs, 0, total)

        if not msgs:
            summary = self._summarize([], existing, budget_tokens - MESSAGE_OVERHEAD)
            return CompressedContext(summary, [], 0, self.summary_tokens(summary))

        if costs[-1] > budget_tokens:
            return self._truncate_newest(msgs, budget_tokens)

        # 1. Longest recent suffix that fits outside the summary reserve.
        reserve = int(budget_tokens * self.summary_ratio)
        kept, used = self._suffix(costs, budget_tokens - reserve, len(costs))
        # 2. Guarantee min_recent when they fit in the whole budget.
        target = min(self.min_recent, len(msgs))
        if kept < target:
            kept, used = self._suffix(costs, budget_tokens, target)

        # 3. Summarize everything older (plus the prior summary) into the remaining budget.
        split = len(msgs) - kept
        summary = self._summarize(msgs[:split], existing, budget_tokens - used - MESSAGE_OVERHEAD)

        # 4. Give leftover budget back to messages, then re-summarize the smaller remainder.
        leftover = budget_tokens - used - self.summary_tokens(summary)
        extra = 0
        while split - extra > 0 and costs[split - extra - 1] <= leftover:
            leftover -= costs[split - extra - 1]
            used += costs[split - extra - 1]
            extra += 1
        if extra:
            split -= extra
            summary = self._summarize(
                msgs[:split], existing, budget_tokens - used - MESSAGE_OVERHEAD
            )

        return CompressedContext(
            summary=summary,
            messages=msgs[split:],
            compressed_count=split,
            tokens=used + self.summary_tokens(summary),
        )

    @staticmethod
    def _suffix(costs: Sequence[int], budget: int, limit: int) -> tuple[int, int]:
        """Longest suffix (at most ``limit`` items) whose total cost fits ``budget``."""
        kept = used = 0
        for cost in reversed(costs):
            if kept >= limit or used + cost > budget:
                break
            used += cost
            kept += 1
        return kept, used

    def _truncate_newest(self, msgs: list[Message], budget_tokens: int) -> CompressedContext:
        newest = msgs[-1]
        content = truncate_to_tokens(
            newest.content, budget_tokens - MESSAGE_OVERHEAD, self.token_counter
        )
        if not content:
            return CompressedContext("", [], len(msgs), 0)
        clipped = replace(newest, content=content, metadata={**newest.metadata, "truncated": True})
        return CompressedContext(
            "", [clipped], len(msgs) - 1, message_tokens(clipped, self.token_counter)
        )


# --------------------------------------------------------------------------- rolling summary


class RollingSummary:
    """Running summary of evicted messages, bounded by ``max_tokens``.

    Instances are callable with a list of messages, so they can be passed directly as a
    short-term buffer ``on_evict`` callback.
    """

    def __init__(
        self,
        summarizer: Summarizer | None = None,
        max_tokens: int = 300,
        text: str = "",
        token_counter: TokenCounter = estimate_tokens,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.token_counter = token_counter
        self.summarizer: Summarizer = summarizer or ExtractiveSummarizer(token_counter)
        self.max_tokens = max_tokens
        self.text = text

    def absorb(self, messages: Sequence[Message] | Message) -> str:
        """Fold newly evicted ``messages`` into the summary; returns the updated text."""
        batch = [messages] if isinstance(messages, Message) else list(messages)
        if not any(m.role != "system" for m in batch):
            return self.text
        inputs = ([prior_summary_message(self.text)] if self.text else []) + batch
        updated = (self.summarizer.summarize(inputs, self.max_tokens) or "").strip()
        self.text = truncate_to_tokens(updated, self.max_tokens, self.token_counter)
        return self.text

    def __call__(self, messages: Sequence[Message] | Message) -> str:
        return self.absorb(messages)

    def reset(self) -> None:
        self.text = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "max_tokens": self.max_tokens}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        summarizer: Summarizer | None = None,
        token_counter: TokenCounter = estimate_tokens,
    ) -> RollingSummary:
        return cls(
            summarizer=summarizer,
            max_tokens=int(data.get("max_tokens", 300)),
            text=str(data.get("text", "")),
            token_counter=token_counter,
        )


__all__ = [
    "CompressedContext",
    "ContextCompressor",
    "ExtractiveSummarizer",
    "LLMSummarizer",
    "RollingSummary",
    "prior_summary_message",
    "render_transcript",
    "truncate_to_tokens",
]
