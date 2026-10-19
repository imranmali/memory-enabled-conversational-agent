"""Language-model backends for the agent.

* :class:`OfflineLLM` - deterministic, network-free model grounded in the memory sections of
  the system prompt. It is the default model for the CLI, demos and tests.
* :class:`AnthropicLLM` - Claude via the optional ``anthropic`` SDK (imported lazily).
* :func:`build_llm` - select a backend by name.

The section headers the agent writes into its system prompt are defined here so that the
agent (which renders them) and :class:`OfflineLLM` (which reads them) always agree.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .facts import extract_facts
from .interfaces import LLM
from .types import Message

__all__ = [
    "FACTS_HEADER",
    "MEMORIES_HEADER",
    "PREVIOUS_SESSION_HEADER",
    "REFUSAL_MESSAGE",
    "SECTION_HEADERS",
    "SUMMARY_HEADER",
    "AnthropicLLM",
    "OfflineLLM",
    "build_llm",
    "parse_sections",
    "to_second_person",
]

FACTS_HEADER = "## Known facts about the user"
MEMORIES_HEADER = "## Relevant memories"
SUMMARY_HEADER = "## Conversation summary"
PREVIOUS_SESSION_HEADER = "## Previous session"
SECTION_HEADERS = (FACTS_HEADER, PREVIOUS_SESSION_HEADER, MEMORIES_HEADER, SUMMARY_HEADER)

#: Prefix of the "topics" line the agent writes in the previous-session section.
TOPICS_PREFIX = "Topics:"

REFUSAL_MESSAGE = "I'm sorry, but I can't help with that request."


def parse_sections(system: str) -> dict[str, list[str]]:
    """Split a system prompt into its memory sections.

    Returns a mapping from each header in :data:`SECTION_HEADERS` to its non-blank lines (with a
    leading ``"- "`` bullet removed). Text before the first header is ignored.
    """
    sections: dict[str, list[str]] = {header: [] for header in SECTION_HEADERS}
    current: str | None = None
    for raw in (system or "").splitlines():
        line = raw.strip()
        if line in sections:
            current = line
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is None or not line:
            continue
        sections[current].append(line[2:].strip() if line.startswith("- ") else line)
    return sections


# --------------------------------------------------------------------------- offline model

_KIND_TAG = re.compile(r"^\((?P<kind>fact|message|summary)\)\s*(?P<text>.*)$")
_SECOND_PERSON_FIXED = (
    (re.compile(r"^User's\s+"), "your "),
    (re.compile(r"^User is\s+"), "you are "),
    (re.compile(r"^User was\s+"), "you were "),
    (re.compile(r"^User has\s+"), "you have "),
    (re.compile(r"^User does\s+"), "you do "),
)
_SECOND_PERSON_VERB = re.compile(
    r"^User\s+(?:(?P<adv>really|also|absolutely|totally)\s+)?(?P<v>\w+)"
)


def _base_form(verb: str) -> str:
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):
        return verb[:-1]
    return verb


def to_second_person(fact: str) -> str:
    """Rephrase a third-person user fact for the user (``User loves X`` → ``you love X``)."""
    text = fact.strip()
    for pattern, repl in _SECOND_PERSON_FIXED:
        if pattern.match(text):
            text = pattern.sub(repl, text, count=1)
            break
    else:
        match = _SECOND_PERSON_VERB.match(text)
        if match:
            adverb = f"{match.group('adv')} " if match.group("adv") else ""
            text = f"you {adverb}{_base_form(match.group('v'))}" + text[match.end() :]
    text = re.sub(r"\btheir\b", "your", text)
    text = re.sub(r"\bthemselves\b", "yourself", text)
    return re.sub(r"\bUser's\b", "your", text)


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


_REMEMBER_ME = re.compile(
    r"\b(?:what|anything)\s+(?:else\s+)?(?:do|did|can)\s+you\s+(?:remember|know|recall)\b"
    r"|\bwhat\s+have\s+i\s+told\s+you\b|\bdo\s+you\s+remember\s+me\b"
    r"|\btell\s+me\s+(?:what\s+you\s+know\s+)?about\s+(?:me|myself)\b",
    re.IGNORECASE,
)
_NAME_QUESTION = re.compile(
    r"\bwhat(?:'s|\s+is)\s+my\s+name\b|\bdo\s+you\s+know\s+my\s+name\b|\bwho\s+am\s+i\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(?:hi|hello|hey|hiya|yo|good\s+(?:morning|afternoon|evening)|greetings)"
    r"(?:\s+there)?[\s!.,]*$",
    re.IGNORECASE,
)
_NAME_FACT = re.compile(r"^User's name is (?P<name>.+)$")
_CALLED_FACT = re.compile(r"^User prefers to be called (?P<name>.+)$")


class OfflineLLM:
    """Deterministic rule-based model that answers from the prompt's memory sections.

    It satisfies :class:`~memory_agent.interfaces.LLM`, needs no network, and is what the CLI
    uses by default. Intents, in priority order: "what do you remember about me", "what is my
    name", acknowledging newly shared facts, greeting, referencing the top relevant memory,
    and a generic reply. With no user message it produces a session greeting.
    """

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        sections = parse_sections(system)
        facts = sections[FACTS_HEADER]
        name = self._name(facts)
        user_turns = [m for m in messages if m.role == "user"]
        if not user_turns:
            return self._greeting(name, sections)
        text = user_turns[-1].content.strip()

        if _REMEMBER_ME.search(text):
            return self._what_i_remember(facts, sections)
        if _NAME_QUESTION.search(text):
            if name:
                return f"Your name is {name}."
            return "I don't know your name yet. What should I call you?"
        new_facts = extract_facts(text)
        if new_facts:
            return self._acknowledge(new_facts)
        if _GREETING.match(text):
            return self._greeting(name, sections)
        memory = self._top_memory(sections[MEMORIES_HEADER], text)
        if memory:
            return memory
        address = f", {name}" if name else ""
        return (
            f"Thanks for sharing{address}. I'm a simple offline assistant, but I'm listening "
            "and I'll keep track of what matters to you. Tell me more!"
        )

    # -- helpers --------------------------------------------------------------------------------

    @staticmethod
    def _name(facts: Sequence[str]) -> str | None:
        name = None
        for fact in facts:  # facts are rendered oldest first: the last match is the latest
            match = _CALLED_FACT.match(fact) or _NAME_FACT.match(fact)
            if match:
                name = match.group("name")
        return name

    @staticmethod
    def _previous(sections: dict[str, list[str]]) -> tuple[list[str], list[str]]:
        topics: list[str] = []
        lines: list[str] = []
        for line in sections[PREVIOUS_SESSION_HEADER]:
            if line.startswith(TOPICS_PREFIX):
                topics = [t.strip() for t in line[len(TOPICS_PREFIX) :].split(",") if t.strip()]
            else:
                lines.append(line)
        return topics, lines

    def _previous_sentence(self, sections: dict[str, list[str]]) -> str:
        topics, lines = self._previous(sections)
        if topics:
            return f"Last time we talked about {_join(topics)}."
        if lines:
            return f"Last time: {lines[0]}"
        return ""

    def _greeting(self, name: str | None, sections: dict[str, list[str]]) -> str:
        previous = self._previous_sentence(sections)
        if previous:
            hello = f"Welcome back, {name}!" if name else "Welcome back!"
            return f"{hello} {previous} How can I help today?"
        hello = f"Hello, {name}!" if name else "Hello!"
        return (
            f"{hello} I'm your memory-enabled assistant. Tell me about yourself and I'll remember."
        )

    def _what_i_remember(self, facts: Sequence[str], sections: dict[str, list[str]]) -> str:
        previous = self._previous_sentence(sections)
        if not facts:
            reply = "I don't know much about you yet. Tell me about yourself and I'll remember it."
            return f"{reply} {previous}".strip()
        known = [to_second_person(f) for f in facts]
        reply = f"Here's what I remember about you: {_join(known)}."
        return f"{reply} {previous}".strip()

    @staticmethod
    def _acknowledge(new_facts: Sequence[str]) -> str:
        parts: list[str] = []
        others: list[str] = []
        for fact in new_facts:
            match = _NAME_FACT.match(fact) or _CALLED_FACT.match(fact)
            if match and not parts:
                parts.append(f"Nice to meet you, {match.group('name')}!")
            elif not match:
                others.append(to_second_person(fact))
        if others:
            parts.append(f"Got it — I'll remember that {_join(others)}.")
        return " ".join(parts)

    @staticmethod
    def _top_memory(memories: Sequence[str], text: str) -> str:
        for line in memories:
            match = _KIND_TAG.match(line)
            kind, content = (match.group("kind"), match.group("text")) if match else ("", line)
            if not content or content.strip().lower() == text.lower():
                continue
            if kind == "fact":
                return f"That reminds me: I remember that {to_second_person(content)}."
            if kind == "summary":
                return f"That reminds me of when we talked before: {content}"
            return f'That reminds me: earlier you mentioned "{content}".'
        return ""


# --------------------------------------------------------------------------- Anthropic model

FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MODEL = "claude-opus-5-5"
_INSTALL_HINT = (
    "AnthropicLLM requires the optional 'anthropic' package. Install it with: "
    "pip install 'memory-enabled-conversational-agent[anthropic]'"
)
_START_TURN = "Hello."
_CONTINUE_TURN = "Please continue."


def _default_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return anthropic.Anthropic()


class AnthropicLLM:
    """Claude through the Anthropic Messages API; satisfies :class:`~memory_agent.interfaces.LLM`.

    By default it opts into server-side refusal fallbacks (``fallbacks="default"`` with the
    ``server-side-fallback-2026-07-01`` beta) and sets ``output_config.effort``. It never sends
    ``thinking`` or ``temperature`` (rejected by the default model). The SDK is imported only
    when no ``client`` is injected.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 16000,
        effort: str | None = "medium",
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        self.client = client if client is not None else _default_client()

    @staticmethod
    def to_api_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
        """User/assistant turns as API dicts: merged, starting and ending with a user turn."""
        turns: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system" or not message.content.strip():
                continue
            if turns and turns[-1]["role"] == message.role:
                turns[-1]["content"] += "\n\n" + message.content
            else:
                turns.append({"role": message.role, "content": message.content})
        if not turns or turns[0]["role"] != "user":
            turns.insert(0, {"role": "user", "content": _START_TURN})
        if turns[-1]["role"] != "user":  # assistant prefill is rejected by current models
            turns.append({"role": "user", "content": _CONTINUE_TURN})
        return turns

    def request_kwargs(self, system: str, messages: Sequence[Message]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self.to_api_messages(messages),
        }
        if self.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        kwargs = self.request_kwargs(system, messages)
        if self.fallbacks:
            response = self.client.beta.messages.create(**kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        if getattr(response, "stop_reason", None) == "refusal":
            return REFUSAL_MESSAGE
        blocks = getattr(response, "content", None) or []
        return "".join(
            getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
        ).strip()


# --------------------------------------------------------------------------- factory

LLM_NAMES = ("offline", "anthropic")


def build_llm(name: str, **kwargs: Any) -> LLM:
    """Create a model by name: ``"offline"`` or ``"anthropic"`` (kwargs go to its constructor)."""
    key = (name or "").strip().lower()
    if key == "offline":
        return OfflineLLM(**kwargs)
    if key == "anthropic":
        return AnthropicLLM(**kwargs)
    raise ValueError(f"unknown LLM {name!r}; choose one of: {', '.join(LLM_NAMES)}")
