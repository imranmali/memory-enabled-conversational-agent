"""The memory-enabled conversational agent.

:class:`MemoryAgent` ties the memory components together for one user:

1. extract facts from the user's message and remember them (superseding single-valued facts),
2. recall relevant long-term memories,
3. compress short-term history (plus the rolling summary of evicted turns) into the budget
   left after the system prompt,
4. render the system prompt (persona + memory sections) so prompt + messages fit
   ``context_budget_tokens``,
5. call the LLM, buffer both turns, store the user message long-term and count it in the
   session.

Sessions end with a stored summary, topics on the session record, and a sync.
"""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .compression import (
    ContextCompressor,
    ExtractiveSummarizer,
    RollingSummary,
    prior_summary_message,
    truncate_to_tokens,
)
from .embeddings import tokenize
from .facts import extract_facts, fact_key
from .interfaces import LLM, Summarizer
from .llm import (
    FACTS_HEADER,
    MEMORIES_HEADER,
    PREVIOUS_SESSION_HEADER,
    SUMMARY_HEADER,
    TOPICS_PREFIX,
    OfflineLLM,
)
from .memory import LongTermMemory
from .scoring import estimate_importance
from .short_term import ShortTermBuffer
from .stores import InMemoryVectorStore, SQLiteVectorStore
from .sync import FileSyncBackend, MemorySynchronizer, SessionInfo, SessionManager, SyncReport
from .tokens import estimate_tokens, total_tokens
from .types import Message, ScoredMemory

__all__ = ["AgentConfig", "AgentReply", "MemoryAgent", "default_device_id"]

DEFAULT_PERSONA = (
    "You are a warm, helpful assistant with long-term memory. Use what you know about the user "
    "naturally and accurately; never invent facts about them. If you don't know something "
    "about the user, say so."
)

#: Fraction of the context budget each system-prompt part may use at most.
PERSONA_SHARE = 0.15
FACTS_SHARE = 0.20
PREVIOUS_SHARE = 0.10
MEMORIES_SHARE = 0.20
MIN_CONTEXT_BUDGET = 200
FACT_IMPORTANCE = 0.8
SUMMARY_IMPORTANCE = 0.6
MAX_TOPICS = 5
SYNC_STATE_PREFIX = "sync-state-"

_RESERVED_DEVICE = "local"
_TOPIC_STOPWORDS = frozenset(
    "about after again also always am any anything back been being can could did does doing "
    "done dont don't each even ever every from get got going good great have having hello help "
    "here hey hi how i'd i'll i'm i've into it's just know last let like likes love loves make "
    "many more most much name need never nice not now okay one only other our out over please "
    "really remember right said say see should some something still such sure tell thank thanks "
    "than that's then there these thing things think through time today told too user very want "
    "well what's would yeah yes you'd you're".split()
)


def default_device_id() -> str:
    """A sync device id derived from the host name (never the reserved ``"local"``)."""
    try:
        host = socket.gethostname()
    except OSError:  # pragma: no cover - extremely unusual
        host = ""
    device = re.sub(r"[^A-Za-z0-9_.-]+", "-", host).strip("-.")
    if not device or device.lower() == _RESERVED_DEVICE:
        return "device-1"
    return device


@dataclass
class AgentConfig:
    """Tunable behaviour of :class:`MemoryAgent`."""

    persona: str = DEFAULT_PERSONA
    context_budget_tokens: int = 3000
    recall_k: int = 5
    min_recall_score: float = 0.15
    min_recall_similarity: float = 0.1
    short_term_max_messages: int = 20
    short_term_max_tokens: int = 1500
    summary_max_tokens: int = 300
    store_messages: bool = True
    min_message_importance: float = 0.15
    device_id: str = field(default_factory=default_device_id)

    def __post_init__(self) -> None:
        if self.context_budget_tokens < MIN_CONTEXT_BUDGET:
            raise ValueError(f"context_budget_tokens must be at least {MIN_CONTEXT_BUDGET}")
        if self.recall_k < 0:
            raise ValueError("recall_k must be >= 0")
        if self.summary_max_tokens <= 0:
            raise ValueError("summary_max_tokens must be positive")
        if not self.device_id or self.device_id == _RESERVED_DEVICE:
            raise ValueError(f"device_id must be non-empty and not {_RESERVED_DEVICE!r}")


@dataclass
class AgentReply:
    """What :meth:`MemoryAgent.chat` produced for one user message."""

    text: str
    recalled: list[ScoredMemory] = field(default_factory=list)
    facts_learned: list[str] = field(default_factory=list)
    context_tokens: int = 0
    compressed: bool = False


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _fit_section(header: str, lines: Sequence[str], cap: int) -> str:
    """Render ``header`` + bullet ``lines`` within ``cap`` tokens (later lines dropped first)."""
    if not lines or cap <= estimate_tokens(header) + 2:
        return ""
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([header, *kept, f"- {line}"])
        if estimate_tokens(candidate) > cap:
            break
        kept.append(f"- {line}")
    if not kept:
        room = cap - estimate_tokens(header + "\n- ")
        first = truncate_to_tokens(lines[0], room)
        if not first:
            return ""
        kept.append(f"- {first}")
    return "\n".join([header, *kept])


def _fit_block(header: str, body: str, cap: int) -> str:
    if not body.strip():
        return ""
    room = cap - estimate_tokens(header + "\n")
    text = truncate_to_tokens(body.strip(), room)
    return f"{header}\n{text}" if text else ""


class MemoryAgent:
    """A conversational agent that remembers its user within and across sessions and devices."""

    def __init__(
        self,
        user_id: str,
        llm: LLM | None = None,
        memory: LongTermMemory | None = None,
        config: AgentConfig | None = None,
        session_manager: SessionManager | None = None,
        synchronizer: MemorySynchronizer | None = None,
        summarizer: Summarizer | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        self.user_id = user_id
        self.config = config or AgentConfig()
        self.clock = clock
        self.llm: LLM = llm or OfflineLLM()
        self.memory = memory or LongTermMemory(InMemoryVectorStore(clock=clock), clock=clock)
        self.session_manager = session_manager
        self.synchronizer = synchronizer
        self.summarizer: Summarizer = summarizer or ExtractiveSummarizer()
        self.rolling_summary = RollingSummary(
            self.summarizer, max_tokens=self.config.summary_max_tokens
        )
        self.buffer = ShortTermBuffer(
            max_messages=self.config.short_term_max_messages,
            max_tokens=self.config.short_term_max_tokens,
            on_evict=self.rolling_summary,
        )
        self.compressor = ContextCompressor(self.summarizer)
        self.session: SessionInfo | None = None
        self.previous_session: SessionInfo | None = None
        self.last_system_prompt = ""
        self.last_messages: list[Message] = []
        self.last_sync: SyncReport | None = None
        self.data_dir: Path | None = None
        self._sync_state_path: Path | None = None
        self._closables: list[Any] = []
        self._closed = False

    # -- sessions -------------------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        return self.session.session_id if self.session is not None else None

    def start_session(self) -> SessionInfo | None:
        """Begin a conversation: sync, load the previous session, open a new session record."""
        if self.session is not None:
            return self.session.copy()
        self.buffer.clear()
        self.rolling_summary.reset()
        self.sync()
        if self.session_manager is None:
            return None
        self.previous_session = self.session_manager.last_session(self.user_id)
        self.session = self.session_manager.start_session(self.user_id)
        return self.session.copy()

    def end_session(self) -> SessionInfo | None:
        """Summarize the conversation, store it, close the session record and sync."""
        summary = self._session_summary()
        topics = self._topics()
        if summary:
            self.memory.remember(
                self.user_id,
                summary,
                kind="summary",
                importance=SUMMARY_IMPORTANCE,
                session_id=self.session_id,
                metadata={"topics": topics},
            )
        info: SessionInfo | None = None
        if self.session_manager is not None and self.session is not None:
            info = self.session_manager.end_session(
                self.session.session_id, summary=summary, topics=topics
            )
            self.previous_session = info
        self.session = None
        self.buffer.clear()
        self.rolling_summary.reset()
        self.sync()
        return info

    def _session_summary(self) -> str:
        inputs: list[Message] = []
        if self.rolling_summary.text:
            inputs.append(prior_summary_message(self.rolling_summary.text))
        inputs.extend(self.buffer.messages())
        if not inputs:
            return ""
        limit = self.config.summary_max_tokens
        return truncate_to_tokens(self.summarizer.summarize(inputs, limit).strip(), limit)

    def _topics(self) -> list[str]:
        texts = [m.content for m in self.buffer.messages() if m.role == "user"]
        texts.extend(
            line.split(":", 1)[1]
            for line in self.rolling_summary.text.splitlines()
            if line.startswith("User:")
        )
        names = {
            token
            for fact in self.known_facts()
            if fact_key(fact) in ("name", "preferred-name")
            for token in tokenize(fact)
        }
        counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for token in (t for text in texts for t in tokenize(text)):
            if len(token) < 3 or token in _TOPIC_STOPWORDS or token in names or token.isdigit():
                continue
            counts[token] += 1
            first_seen.setdefault(token, len(first_seen))
        ranked = sorted(counts, key=lambda t: (-counts[t], first_seen[t]))
        return ranked[:MAX_TOPICS]

    # -- conversation ---------------------------------------------------------------------------

    def greeting(self) -> str:
        """An opening line for the session (mentions the previous session when known)."""
        system = self._system_prompt([], "", self.config.context_budget_tokens)
        return self.llm.complete(system, [])

    def chat(self, text: str) -> AgentReply:
        """Handle one user message and return the agent's reply with memory diagnostics."""
        text = (text or "").strip()
        if not text:
            raise ValueError("message must not be empty")
        if self.session is None and self.session_manager is not None:
            self.start_session()
        session_id = self.session_id

        facts = extract_facts(text)
        learned_ids = {self._learn(fact, session_id) for fact in facts}

        current = _normalize(text)
        recalled = self.memory.recall(
            self.user_id,
            text,
            k=self.config.recall_k,
            kinds=("fact", "message", "summary"),
            min_score=self.config.min_recall_score,
            min_similarity=self.config.min_recall_similarity,
            exclude=lambda item: item.id in learned_ids or _normalize(item.content) == current,
        )

        user_message = Message(
            role="user", content=text, timestamp=float(self.clock()), metadata=self._meta()
        )
        budget = self.config.context_budget_tokens
        base_system = self._system_prompt(recalled, "", budget)
        summary_overhead = estimate_tokens(f"\n\n{SUMMARY_HEADER}\n")
        conversation_budget = budget - estimate_tokens(base_system) - summary_overhead
        compressed = self.compressor.compress(
            [*self.buffer.messages(), user_message],
            max(1, conversation_budget),
            existing_summary=self.rolling_summary.text,
        )
        system = base_system
        if compressed.summary:
            system = f"{base_system}\n\n{SUMMARY_HEADER}\n{compressed.summary}"
        messages = list(compressed.messages)
        context_tokens = estimate_tokens(system) + total_tokens(messages)

        reply_text = self.llm.complete(system, messages).strip()
        self.last_system_prompt = system
        self.last_messages = messages

        assistant_message = Message(
            role="assistant",
            content=reply_text,
            timestamp=float(self.clock()),
            metadata=self._meta(),
        )
        self.buffer.extend([user_message, assistant_message])
        if self.config.store_messages and (
            estimate_importance(text) >= self.config.min_message_importance
        ):
            self.memory.remember(self.user_id, text, kind="message", session_id=session_id)
        if self.session_manager is not None and session_id is not None:
            self.session_manager.record_message(session_id)
            self.session_manager.record_message(session_id)

        truncated = any(m.metadata.get("truncated") for m in messages)
        return AgentReply(
            text=reply_text,
            recalled=recalled,
            facts_learned=facts,
            context_tokens=context_tokens,
            compressed=compressed.compressed_count > 0 or bool(compressed.summary) or truncated,
        )

    def _meta(self) -> dict[str, Any]:
        return {"session_id": self.session_id} if self.session_id else {}

    def _learn(self, fact: str, session_id: str | None) -> str:
        key = fact_key(fact)
        if key is not None:
            for existing in self.memory.facts(self.user_id):
                if fact_key(existing.content) == key and existing.content != fact:
                    self.memory.forget(self.user_id, existing.id)
        item = self.memory.remember(
            self.user_id,
            fact,
            kind="fact",
            importance=FACT_IMPORTANCE,
            session_id=session_id,
            metadata={"source": "chat"},
        )
        return item.id

    def _system_prompt(self, recalled: Sequence[ScoredMemory], summary: str, budget: int) -> str:
        persona = truncate_to_tokens(self.config.persona.strip(), int(budget * PERSONA_SHARE))
        parts = [persona] if persona else []

        facts = sorted(
            self.memory.facts(self.user_id), key=lambda i: (i.updated_at or i.created_at, i.id)
        )
        # Keep the newest facts when the section overflows, but render oldest first.
        newest_first = [f.content for f in reversed(facts)]
        section = _fit_section(FACTS_HEADER, newest_first, int(budget * FACTS_SHARE))
        if section:
            header, *lines = section.splitlines()
            parts.append("\n".join([header, *reversed(lines)]))

        previous = self._previous_block()
        block = _fit_block(PREVIOUS_SESSION_HEADER, previous, int(budget * PREVIOUS_SHARE))
        if block:
            parts.append(block)

        memory_lines = [f"({s.item.kind}) {s.item.content}" for s in recalled]
        section = _fit_section(MEMORIES_HEADER, memory_lines, int(budget * MEMORIES_SHARE))
        if section:
            parts.append(section)

        if summary:
            parts.append(f"{SUMMARY_HEADER}\n{summary}")
        return "\n\n".join(parts)

    def _previous_block(self) -> str:
        info = self.previous_session
        if info is None:
            return ""
        lines = []
        if info.topics:
            lines.append(f"{TOPICS_PREFIX} {', '.join(info.topics)}")
        if info.summary:
            lines.append(info.summary)
        return "\n".join(lines)

    # -- memory management ----------------------------------------------------------------------

    def known_facts(self) -> list[str]:
        """Contents of the user's live fact memories, oldest first."""
        return [item.content for item in self.memory.facts(self.user_id)]

    def forget_everything(self) -> int:
        """Forget every long-term memory of the user and clear short-term state, then sync."""
        count = self.memory.forget_all(self.user_id)
        self.buffer.clear()
        self.rolling_summary.reset()
        self.sync()
        return count

    def sync(self) -> SyncReport | None:
        """Synchronize long-term memory with other devices (no-op without a synchronizer)."""
        if self.synchronizer is None:
            return None
        report = self.synchronizer.sync(self.user_id)
        self.last_sync = report
        self._save_sync_state()
        return report

    def _save_sync_state(self) -> None:
        if self._sync_state_path is None or self.synchronizer is None:
            return
        path = self._sync_state_path
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.synchronizer.state(), fh, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- construction & lifecycle ---------------------------------------------------------------

    @classmethod
    def from_data_dir(
        cls,
        user_id: str,
        data_dir: str | os.PathLike[str],
        llm: LLM | None = None,
        config: AgentConfig | None = None,
        device_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> MemoryAgent:
        """Agent with persistent SQLite memory and file-based sync under ``data_dir``.

        Layout: ``memory.sqlite3`` (this device's store), ``sync/`` (shared sync folder, which
        may be a synced/network directory) and ``sync-state-<device>.json`` (sync cursor).
        """
        config = config or AgentConfig()
        if device_id is not None:
            config = replace(config, device_id=device_id)
        root = Path(data_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        store = SQLiteVectorStore(root / "memory.sqlite3", clock=clock)
        backend = FileSyncBackend(root / "sync")
        synchronizer = MemorySynchronizer(store, backend, config.device_id)
        safe_device = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.device_id)
        state_path = root / f"{SYNC_STATE_PREFIX}{safe_device}.json"
        _load_sync_state(synchronizer, state_path)
        agent = cls(
            user_id,
            llm=llm,
            memory=LongTermMemory(store, clock=clock),
            config=config,
            session_manager=SessionManager(backend, config.device_id, clock=clock),
            synchronizer=synchronizer,
            clock=clock,
        )
        agent.data_dir = root
        agent._sync_state_path = state_path
        agent._closables.append(store)
        return agent

    def close(self) -> None:
        """End an active session (summary + sync) and release owned resources. Idempotent."""
        if self._closed:
            return
        try:
            if self.session is not None:
                self.end_session()
        finally:
            self._closed = True
            for resource in self._closables:
                resource.close()
            self._closables.clear()

    def __enter__(self) -> MemoryAgent:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _load_sync_state(synchronizer: MemorySynchronizer, path: Path) -> None:
    """Restore saved sync cursors; unreadable or foreign state just means a full re-sync."""
    if not path.exists():
        return
    try:
        synchronizer.load_state(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, AttributeError):
        return
