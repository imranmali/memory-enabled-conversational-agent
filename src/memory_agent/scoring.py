"""Relevance scoring for long-term memory recall.

Blends query similarity with recency, importance and access frequency into one
score in ``[0, 1]``, ranks recall candidates deterministically, diversifies results
with Maximal Marginal Relevance (MMR), de-duplicates near-identical memories, and
estimates a baseline importance for new memory text.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .embeddings import cosine_similarity
from .types import MemoryItem, ScoredMemory

__all__ = [
    "RelevanceScorer",
    "ScoringWeights",
    "deduplicate",
    "estimate_importance",
    "maximal_marginal_relevance",
]

DEFAULT_HALF_LIFE_SECONDS = 7 * 24 * 3600.0


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        return low
    return min(high, max(low, value))


@dataclass(frozen=True)
class ScoringWeights:
    """Relative weights of the score components. Must be non-negative and not all zero."""

    similarity: float = 0.6
    recency: float = 0.2
    importance: float = 0.15
    frequency: float = 0.05

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(math.isnan(v) or math.isinf(v) for v in values):
            raise ValueError("scoring weights must be finite")
        if any(v < 0 for v in values):
            raise ValueError("scoring weights must be non-negative")
        if sum(values) == 0:
            raise ValueError("at least one scoring weight must be positive")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.similarity, self.recency, self.importance, self.frequency)

    def normalized(self) -> ScoringWeights:
        """Return a copy whose weights sum to 1 (ratios preserved)."""
        total = sum(self.as_tuple())
        return ScoringWeights(
            similarity=self.similarity / total,
            recency=self.recency / total,
            importance=self.importance / total,
            frequency=self.frequency / total,
        )


class RelevanceScorer:
    """Scores and ranks ``(MemoryItem, similarity)`` candidates; satisfies ``Scorer``."""

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        half_life_seconds: float = DEFAULT_HALF_LIFE_SECONDS,
        frequency_saturation: int = 10,
        kind_boosts: Mapping[str, float] | None = None,
    ) -> None:
        if not half_life_seconds > 0 or math.isinf(half_life_seconds):
            raise ValueError("half_life_seconds must be a positive finite number")
        if frequency_saturation < 1:
            raise ValueError("frequency_saturation must be >= 1")
        boosts = dict(kind_boosts or {})
        for kind, boost in boosts.items():
            if not math.isfinite(boost):
                raise ValueError(f"kind boost for {kind!r} must be finite")
        self.weights = weights if weights is not None else ScoringWeights()
        self.half_life_seconds = float(half_life_seconds)
        self.frequency_saturation = int(frequency_saturation)
        self.kind_boosts: dict[str, float] = boosts
        self._normalized = self.weights.normalized()

    def recency(self, item: MemoryItem, now: float) -> float:
        """``0.5 ** (age / half_life)`` with age measured from the last access, floored at 0."""
        last = item.last_accessed if item.last_accessed is not None else item.created_at
        age = max(0.0, now - last)
        return _clip(0.5 ** (age / self.half_life_seconds))

    def frequency(self, item: MemoryItem) -> float:
        """Log-scaled access count reaching 1 at ``frequency_saturation`` accesses."""
        count = max(0, item.access_count)
        return min(1.0, math.log1p(count) / math.log1p(self.frequency_saturation))

    def score(self, item: MemoryItem, similarity: float, now: float | None = None) -> ScoredMemory:
        """Score one memory; ``components`` holds raw component values plus ``boost``."""
        at = time.time() if now is None else now
        sim = _clip(float(similarity))
        components = {
            "similarity": sim,
            "recency": self.recency(item, at),
            "importance": _clip(item.importance),
            "frequency": self.frequency(item),
            "boost": self.kind_boosts.get(item.kind, 0.0),
        }
        w = self._normalized
        weighted = (
            w.similarity * components["similarity"]
            + w.recency * components["recency"]
            + w.importance * components["importance"]
            + w.frequency * components["frequency"]
        )
        total = _clip(weighted + components["boost"])
        return ScoredMemory(item=item, score=total, similarity=sim, components=components)

    def rank(
        self,
        candidates: Sequence[tuple[MemoryItem, float]],
        now: float | None = None,
        k: int | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredMemory]:
        """Score, filter (tombstones, ``min_score``), sort best-first and truncate to ``k``."""
        at = time.time() if now is None else now
        scored = [
            s
            for s in (self.score(item, sim, at) for item, sim in candidates if not item.deleted)
            if s.score >= min_score
        ]
        scored.sort(key=_rank_key)
        if k is not None:
            scored = scored[: max(0, k)]
        return scored


def _rank_key(s: ScoredMemory) -> tuple[float, float, float, str]:
    return (-s.score, -s.similarity, -s.item.created_at, s.item.id)


def _embedding_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return cosine_similarity(a, b)


def _by_score(scored: Sequence[ScoredMemory]) -> list[ScoredMemory]:
    # Stable: equal scores keep their input order.
    return sorted(scored, key=lambda s: -s.score)


def maximal_marginal_relevance(
    scored: Sequence[ScoredMemory], k: int, lambda_: float = 0.7
) -> list[ScoredMemory]:
    """Diversity re-ranking: trade relevance (score) against redundancy with picked items.

    ``lambda_ = 1`` yields pure score order; lower values penalize memories whose
    embedding is close to an already-selected one.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be within [0, 1]")
    if k <= 0:
        return []
    pool = _by_score(scored)
    selected: list[ScoredMemory] = []
    max_sim = [0.0] * len(pool)
    while pool and len(selected) < k:
        best_idx = 0
        best_value = -math.inf
        for idx, cand in enumerate(pool):
            value = lambda_ * cand.score - (1.0 - lambda_) * max_sim[idx]
            if value > best_value:
                best_value = value
                best_idx = idx
        chosen = pool.pop(best_idx)
        max_sim.pop(best_idx)
        selected.append(chosen)
        for idx, cand in enumerate(pool):
            sim = _embedding_similarity(cand.item.embedding, chosen.item.embedding)
            if sim > max_sim[idx]:
                max_sim[idx] = sim
    return selected


def deduplicate(scored: Sequence[ScoredMemory], threshold: float = 0.95) -> list[ScoredMemory]:
    """Drop memories whose embedding cosine with a kept, higher-scored memory >= threshold."""
    kept: list[ScoredMemory] = []
    for cand in _by_score(scored):
        if all(
            _embedding_similarity(cand.item.embedding, k.item.embedding) < threshold for k in kept
        ):
            kept.append(cand)
    return kept


# --- importance heuristic -------------------------------------------------

BASELINE_IMPORTANCE = 0.3

_EXPLICIT_REMEMBER = re.compile(
    r"^\s*(?:please\s+)?(?:remember|don't forget|do not forget|note)\s+(?:that\b|this\b|:)"
    r"|\bremember that\b|\bplease remember\b"
)

# (pattern, weight) for self-disclosure / durable-fact cues.
_CUES: tuple[tuple[re.Pattern[str], float], ...] = tuple(
    (re.compile(p), w)
    for p, w in (
        (r"\bmy name is\b", 0.3),
        (r"\bcall me\b", 0.2),
        (r"\bi am\b", 0.2),
        (r"\bi'm\b", 0.2),
        (r"\bi live\b", 0.25),
        (r"\bi work\b", 0.25),
        (r"\bi prefer\b", 0.25),
        (r"\bi like\b", 0.2),
        (r"\bi love\b", 0.2),
        (r"\bi hate\b", 0.2),
        (r"\ballergic\b", 0.3),
        (r"\bbirthday\b", 0.25),
        (r"\bfavou?rite\b", 0.2),
        (r"\bgoals?\b", 0.15),
        (r"\bplan(?:ning)? to\b", 0.15),
        (r"\bremember\b", 0.15),
        (r"\balways\b", 0.1),
        (r"\bnever\b", 0.1),
    )
)
_MAX_CUE_BONUS = 0.5
_EXPLICIT_FLOOR = 0.85
_NUMBER_BONUS = 0.05
_QUESTION_PENALTY = 0.1
_SMALL_TALK_SCORE = 0.1

_NUMBER_OR_DATE = re.compile(
    r"\d|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday"
    r"|thursday|friday|saturday|sunday)\b"
)
_QUESTION_START = re.compile(
    r"^(?:what|who|whom|whose|when|where|why|how|which|is|are|am|do|does|did|can|could|"
    r"would|will|should|shall|may|might|have|has|had)\b"
)
_SMALL_TALK = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yo",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "ok",
        "okay",
        "k",
        "cool",
        "nice",
        "great",
        "sure",
        "yes",
        "no",
        "yep",
        "nope",
        "bye",
        "goodbye",
        "lol",
        "good morning",
        "good night",
        "ok thanks",
        "okay thanks",
        "thanks a lot",
        "hi there",
        "hello there",
    }
)


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("’", "'").replace("‘", "'").lower().split())


def estimate_importance(text: str) -> float:
    """Deterministic heuristic importance in ``[0, 1]`` for a piece of memory text.

    Baseline 0.3; self-disclosure and durable-fact cues raise it (bonus capped at 0.5),
    explicit "remember that ..." requests floor it at 0.85, numbers/dates add 0.05,
    pure questions lose 0.1 and short small talk drops to 0.1.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0
    bare = re.sub(r"[^a-z' ]+", " ", normalized)
    bare = " ".join(bare.split())
    if bare in _SMALL_TALK or (not bare and not _NUMBER_OR_DATE.search(normalized)):
        return _SMALL_TALK_SCORE

    cue_bonus = min(_MAX_CUE_BONUS, sum(w for p, w in _CUES if p.search(normalized)))
    score = BASELINE_IMPORTANCE + cue_bonus
    if _NUMBER_OR_DATE.search(normalized):
        score += _NUMBER_BONUS
    if _EXPLICIT_REMEMBER.search(normalized):
        score = max(score, _EXPLICIT_FLOOR + cue_bonus / 5)
    elif cue_bonus == 0:
        is_question = normalized.endswith("?") or bool(_QUESTION_START.match(normalized))
        if is_question:
            score -= _QUESTION_PENALTY
        elif len(bare.split()) <= 2:
            score -= _QUESTION_PENALTY / 2
    return round(_clip(score), 4)
