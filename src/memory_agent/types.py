"""Core data types shared by every memory component."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
MemoryKind = Literal["message", "fact", "summary"]


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


@dataclass
class Message:
    """A single conversational turn."""

    role: Role
    content: str
    timestamp: float = field(default_factory=now)
    id: str = field(default_factory=new_id)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(**data)


@dataclass
class MemoryItem:
    """A unit of long-term memory stored in a vector store.

    ``updated_at`` and ``origin`` drive last-write-wins conflict resolution during
    cross-session sync; ``deleted`` is a tombstone so deletions propagate.
    """

    user_id: str
    content: str
    embedding: list[float]
    kind: MemoryKind = "message"
    session_id: str | None = None
    importance: float = 0.5
    created_at: float = field(default_factory=now)
    last_accessed: float | None = None
    access_count: int = 0
    updated_at: float | None = None
    origin: str = "local"
    deleted: bool = False
    id: str = field(default_factory=new_id)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.updated_at is None:
            self.updated_at = self.created_at
        self.importance = min(1.0, max(0.0, float(self.importance)))

    def touch(self, at: float | None = None) -> None:
        """Record an access (used by recency/frequency scoring)."""
        self.last_accessed = now() if at is None else at
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        return cls(**data)


@dataclass
class ScoredMemory:
    """A recalled memory together with its relevance score breakdown."""

    item: MemoryItem
    score: float
    similarity: float
    components: dict[str, float] = field(default_factory=dict)
