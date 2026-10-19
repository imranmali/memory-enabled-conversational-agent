"""Per-user long-term memory on top of any :class:`~memory_agent.interfaces.VectorStore`."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .embeddings import Embedder, HashingEmbedder
from .interfaces import Scorer, VectorStore
from .scoring import RelevanceScorer, deduplicate, estimate_importance, maximal_marginal_relevance
from .types import MemoryItem, MemoryKind, ScoredMemory

__all__ = ["LongTermMemory"]

#: Recall searches this many candidates per requested result before re-ranking.
POOL_FACTOR = 4

#: Default score boosts: distilled facts edge out the raw messages they were extracted from.
DEFAULT_KIND_BOOSTS = {"fact": 0.05}


class LongTermMemory:
    """Remember, reinforce, recall and forget a user's memories.

    * :meth:`remember` embeds content and stores it, or *reinforces* a near-duplicate of the
      same user and kind (cosine ≥ ``dedup_threshold``) instead of inserting a copy.
    * :meth:`recall` ranks a candidate pool with the scorer, de-duplicates, optionally
      diversifies with MMR, and records an access on every returned memory.
    * :meth:`forget` / :meth:`forget_all` tombstone memories so deletions sync to other devices.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
        scorer: Scorer | None = None,
        clock: Callable[[], float] = time.time,
        dedup_threshold: float = 0.92,
    ) -> None:
        if not 0.0 < dedup_threshold <= 1.0:
            raise ValueError("dedup_threshold must be in (0, 1]")
        self.store = store
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.scorer: Scorer = scorer or RelevanceScorer(kind_boosts=DEFAULT_KIND_BOOSTS)
        self.clock = clock
        self.dedup_threshold = dedup_threshold

    def embed(self, text: str) -> list[float]:
        return list(self.embedder.embed([text])[0])

    # -- writing --------------------------------------------------------------------------------

    def remember(
        self,
        user_id: str,
        content: str,
        kind: MemoryKind = "message",
        importance: float | None = None,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
        """Store ``content`` (or reinforce its near-duplicate) and return the stored item."""
        content = (content or "").strip()
        if not content:
            raise ValueError("cannot remember empty content")
        weight = estimate_importance(content) if importance is None else float(importance)
        embedding = self.embed(content)
        now = float(self.clock())

        nearest = self.store.search(user_id, embedding, k=1, kinds=[kind])
        if nearest and nearest[0][1] >= self.dedup_threshold:
            item = nearest[0][0]
            item.importance = max(item.importance, min(1.0, max(0.0, weight)))
            previous = item.updated_at if item.updated_at is not None else item.created_at
            item.updated_at = max(now, previous)
            item.last_accessed = max(now, item.last_accessed or item.created_at)
            if metadata:
                item.metadata.update(dict(metadata))
            self.store.upsert(item)
            return item

        item = MemoryItem(
            user_id=user_id,
            content=content,
            embedding=embedding,
            kind=kind,
            session_id=session_id,
            importance=weight,
            created_at=now,
            metadata=dict(metadata or {}),
        )
        self.store.upsert(item)
        return item

    # -- reading --------------------------------------------------------------------------------

    def recall(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        kinds: Sequence[str] | None = None,
        min_score: float = 0.0,
        diversify: bool = True,
        min_similarity: float | None = None,
        exclude: Callable[[MemoryItem], bool] | None = None,
    ) -> list[ScoredMemory]:
        """The ``k`` most relevant live memories of ``user_id`` for ``query``, best first.

        ``min_similarity`` drops candidates whose cosine similarity to the query is below it
        (so recency/importance alone cannot surface unrelated memories); ``exclude`` drops
        candidates for which it returns true. Returned memories are touched at ``clock()``.
        """
        if k <= 0 or not query or not query.strip():
            return []
        pool = self.store.search(user_id, self.embed(query), k=POOL_FACTOR * k, kinds=kinds)
        if min_similarity is not None:
            pool = [(item, sim) for item, sim in pool if sim >= min_similarity]
        if exclude is not None:
            pool = [(item, sim) for item, sim in pool if not exclude(item)]
        now = float(self.clock())
        ranked = deduplicate(self.scorer.rank(pool, now=now, min_score=min_score))
        if diversify:
            ranked = maximal_marginal_relevance(ranked, k)
        ranked = ranked[:k]
        for scored in ranked:
            scored.item.touch(now)
            self.store.upsert(scored.item)
        return ranked

    def memories(self, user_id: str, kinds: Sequence[str] | None = None) -> list[MemoryItem]:
        """Live memories of ``user_id`` (optionally of ``kinds``), oldest first."""
        wanted = None if kinds is None else set(kinds)
        return [i for i in self.store.all(user_id) if wanted is None or i.kind in wanted]

    def facts(self, user_id: str) -> list[MemoryItem]:
        """Live ``fact`` memories of ``user_id``, oldest first."""
        return self.memories(user_id, kinds=("fact",))

    def count(self, user_id: str) -> int:
        """Number of live memories of ``user_id``."""
        return len(self.store.all(user_id))

    # -- forgetting -----------------------------------------------------------------------------

    def forget(self, user_id: str, item_id: str) -> bool:
        """Tombstone one of ``user_id``'s memories; False if unknown, deleted or not theirs."""
        item = self.store.get(item_id)
        if item is None or item.deleted or item.user_id != user_id:
            return False
        return self.store.delete(item_id)

    def forget_all(self, user_id: str) -> int:
        """Tombstone every live memory of ``user_id``; return how many were forgotten."""
        return sum(1 for item in self.store.all(user_id) if self.store.delete(item.id))
