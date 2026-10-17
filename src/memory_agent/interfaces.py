"""Protocols that decouple the memory components from one another.

Every concrete implementation in the package satisfies one of these, and users
can plug in their own (e.g. a hosted vector DB or a real LLM).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import MemoryItem, Message, ScoredMemory


@runtime_checkable
class VectorStore(Protocol):
    """Persistent store of :class:`MemoryItem` with cosine-similarity search."""

    def upsert(self, item: MemoryItem) -> None:
        """Insert or replace an item by id."""

    def get(self, item_id: str) -> MemoryItem | None:
        """Return the item (including tombstoned ones) or ``None``."""

    def delete(self, item_id: str) -> bool:
        """Tombstone an item (``deleted=True``, bump ``updated_at``). Returns False if unknown."""

    def search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        k: int = 10,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Top-``k`` live items for ``user_id`` by cosine similarity, highest first."""

    def all(self, user_id: str | None = None, include_deleted: bool = False) -> list[MemoryItem]:
        """All items, optionally filtered by user."""

    def __len__(self) -> int:
        """Number of live (non-deleted) items."""


@runtime_checkable
class Scorer(Protocol):
    def rank(
        self,
        candidates: Sequence[tuple[MemoryItem, float]],
        now: float | None = None,
        k: int | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredMemory]:
        """Re-rank ``(item, similarity)`` candidates into scored memories, best first."""


@runtime_checkable
class Summarizer(Protocol):
    def summarize(self, messages: Sequence[Message], max_tokens: int) -> str:
        """Condense messages into a summary of at most ~``max_tokens`` tokens."""


@runtime_checkable
class LLM(Protocol):
    def complete(self, system: str, messages: Sequence[Message]) -> str:
        """Generate the assistant reply for the given system prompt and conversation."""
