from __future__ import annotations

import math
import random

import pytest

from memory_agent import scoring
from memory_agent.interfaces import Scorer
from memory_agent.scoring import (
    BASELINE_IMPORTANCE,
    RelevanceScorer,
    ScoringWeights,
    deduplicate,
    estimate_importance,
    maximal_marginal_relevance,
)
from memory_agent.types import MemoryItem, ScoredMemory

NOW = 1_000_000.0
DAY = 24 * 3600.0
WEEK = 7 * DAY


def make_item(
    content: str = "x",
    *,
    embedding: list[float] | None = None,
    created_at: float = NOW,
    last_accessed: float | None = None,
    access_count: int = 0,
    importance: float = 0.5,
    kind: str = "message",
    deleted: bool = False,
    item_id: str | None = None,
) -> MemoryItem:
    kwargs = {}
    if item_id is not None:
        kwargs["id"] = item_id
    return MemoryItem(
        user_id="u1",
        content=content,
        embedding=embedding if embedding is not None else [1.0, 0.0],
        kind=kind,
        importance=importance,
        created_at=created_at,
        last_accessed=last_accessed,
        access_count=access_count,
        deleted=deleted,
        **kwargs,
    )


def scored(score: float, embedding: list[float], item_id: str) -> ScoredMemory:
    item = make_item(item_id, embedding=embedding, item_id=item_id)
    return ScoredMemory(item=item, score=score, similarity=score)


# --- 1. weights and configuration -----------------------------------------


class TestScoringWeights:
    def test_defaults(self) -> None:
        w = ScoringWeights()
        assert (w.similarity, w.recency, w.importance, w.frequency) == (0.6, 0.2, 0.15, 0.05)

    @pytest.mark.parametrize(
        "kwargs",
        [{"similarity": -0.1}, {"recency": -1}, {"importance": -0.5}, {"frequency": -0.01}],
    )
    def test_negative_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            ScoringWeights(**kwargs)

    def test_all_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScoringWeights(0, 0, 0, 0)

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScoringWeights(similarity=math.nan)

    def test_normalized(self) -> None:
        n = ScoringWeights(2, 1, 1, 0).normalized()
        assert n.as_tuple() == pytest.approx((0.5, 0.25, 0.25, 0.0))
        assert sum(n.as_tuple()) == pytest.approx(1.0)

    def test_default_normalized_sums_to_one(self) -> None:
        assert sum(ScoringWeights().normalized().as_tuple()) == pytest.approx(1.0)


class TestScorerConfig:
    @pytest.mark.parametrize("half_life", [0, -5, math.inf])
    def test_invalid_half_life(self, half_life: float) -> None:
        with pytest.raises(ValueError):
            RelevanceScorer(half_life_seconds=half_life)

    def test_invalid_frequency_saturation(self) -> None:
        with pytest.raises(ValueError):
            RelevanceScorer(frequency_saturation=0)

    @pytest.mark.parametrize("boost", [math.inf, math.nan])
    def test_non_finite_kind_boost(self, boost: float) -> None:
        with pytest.raises(ValueError):
            RelevanceScorer(kind_boosts={"fact": boost})

    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(RelevanceScorer(), Scorer)


# --- 2. components and composite score -------------------------------------


class TestRecency:
    def test_just_accessed(self) -> None:
        assert RelevanceScorer().recency(make_item(last_accessed=NOW), NOW) == 1.0

    def test_one_half_life(self) -> None:
        item = make_item(created_at=NOW - WEEK)
        assert RelevanceScorer().recency(item, NOW) == pytest.approx(0.5)

    def test_two_half_lives_custom(self) -> None:
        item = make_item(created_at=NOW - 2 * DAY)
        assert RelevanceScorer(half_life_seconds=DAY).recency(item, NOW) == pytest.approx(0.25)

    def test_future_access_is_one(self) -> None:
        item = make_item(created_at=NOW, last_accessed=NOW + 100)
        assert RelevanceScorer().recency(item, NOW) == 1.0

    def test_uses_last_accessed_not_created(self) -> None:
        item = make_item(created_at=NOW - 10 * WEEK, last_accessed=NOW)
        assert RelevanceScorer().recency(item, NOW) == 1.0

    def test_bounded(self) -> None:
        item = make_item(created_at=0.0)
        r = RelevanceScorer(half_life_seconds=1).recency(item, NOW)
        assert 0.0 <= r <= 1.0


class TestFrequency:
    def test_never_accessed(self) -> None:
        assert RelevanceScorer().frequency(make_item(access_count=0)) == 0.0

    @pytest.mark.parametrize("count", [10, 11, 1000])
    def test_saturated(self, count: int) -> None:
        assert RelevanceScorer().frequency(make_item(access_count=count)) == 1.0

    def test_monotonic_below_saturation(self) -> None:
        s = RelevanceScorer()
        values = [s.frequency(make_item(access_count=c)) for c in range(11)]
        assert all(a < b for a, b in zip(values, values[1:], strict=False))

    def test_formula(self) -> None:
        s = RelevanceScorer(frequency_saturation=3)
        assert s.frequency(make_item(access_count=1)) == pytest.approx(math.log(2) / math.log(4))


class TestScore:
    def test_weighted_sum(self) -> None:
        item = make_item(created_at=NOW - WEEK, importance=0.8, access_count=10)
        result = RelevanceScorer().score(item, 0.5, now=NOW)
        expected = 0.6 * 0.5 + 0.2 * 0.5 + 0.15 * 0.8 + 0.05 * 1.0
        assert result.score == pytest.approx(expected)
        assert result.item is item

    def test_weights_are_normalized(self) -> None:
        item = make_item(importance=0.0)
        s = RelevanceScorer(weights=ScoringWeights(2, 2, 0, 0))
        # similarity 0.5, recency 1.0 -> (0.5 + 1.0) / 2
        assert s.score(item, 0.5, now=NOW).score == pytest.approx(0.75)

    @pytest.mark.parametrize(("raw", "clipped"), [(-0.4, 0.0), (1.7, 1.0), (0.3, 0.3)])
    def test_similarity_clipped(self, raw: float, clipped: float) -> None:
        result = RelevanceScorer().score(make_item(), raw, now=NOW)
        assert result.similarity == clipped
        assert result.components["similarity"] == clipped

    def test_nan_similarity_treated_as_zero(self) -> None:
        result = RelevanceScorer().score(make_item(), math.nan, now=NOW)
        assert result.similarity == 0.0
        assert 0.0 <= result.score <= 1.0

    def test_breakdown_keys_and_values(self) -> None:
        item = make_item(importance=0.4, access_count=0)
        result = RelevanceScorer().score(item, 0.9, now=NOW)
        assert set(result.components) == {
            "similarity",
            "recency",
            "importance",
            "frequency",
            "boost",
        }
        assert result.components["recency"] == 1.0
        assert result.components["importance"] == pytest.approx(0.4)
        assert result.components["frequency"] == 0.0
        assert result.components["boost"] == 0.0

    def test_kind_boost_added(self) -> None:
        item = make_item(kind="fact", importance=0.0)
        base = RelevanceScorer().score(item, 0.2, now=NOW).score
        boosted = RelevanceScorer(kind_boosts={"fact": 0.1}).score(item, 0.2, now=NOW)
        assert boosted.score == pytest.approx(base + 0.1)
        assert boosted.components["boost"] == 0.1

    def test_kind_boost_only_for_matching_kind(self) -> None:
        item = make_item(kind="message")
        s = RelevanceScorer(kind_boosts={"fact": 0.3})
        assert s.score(item, 0.5, now=NOW).components["boost"] == 0.0

    def test_score_clipped_to_unit_interval(self) -> None:
        item = make_item(kind="fact", importance=1.0, access_count=50)
        high = RelevanceScorer(kind_boosts={"fact": 0.9}).score(item, 1.0, now=NOW)
        assert high.score == 1.0
        low = RelevanceScorer(kind_boosts={"fact": -5.0}).score(item, 1.0, now=NOW)
        assert low.score == 0.0

    def test_default_now(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(scoring.time, "time", lambda: NOW + WEEK)
        result = RelevanceScorer().score(make_item(created_at=NOW), 0.5)
        assert result.components["recency"] == pytest.approx(0.5)


# --- 3. ranking ---------------------------------------------------------------


class TestRank:
    def test_orders_by_score(self) -> None:
        a, b, c = make_item("a"), make_item("b"), make_item("c")
        ranked = RelevanceScorer().rank([(a, 0.2), (b, 0.9), (c, 0.5)], now=NOW)
        assert [s.item.content for s in ranked] == ["b", "c", "a"]

    def test_tombstones_excluded(self) -> None:
        live, dead = make_item("live"), make_item("dead", deleted=True)
        ranked = RelevanceScorer().rank([(dead, 1.0), (live, 0.1)], now=NOW)
        assert [s.item.content for s in ranked] == ["live"]

    def test_min_score_filter(self) -> None:
        s = RelevanceScorer()
        items = [(make_item(str(v)), v) for v in (0.1, 0.5, 0.9)]
        threshold = s.score(items[1][0], 0.5, now=NOW).score
        ranked = s.rank(items, now=NOW, min_score=threshold)
        assert [r.item.content for r in ranked] == ["0.9", "0.5"]
        assert all(r.score >= threshold for r in ranked)

    def test_truncate_to_k(self) -> None:
        items = [(make_item(str(i)), i / 10) for i in range(10)]
        ranked = RelevanceScorer().rank(items, now=NOW, k=3)
        assert [r.item.content for r in ranked] == ["9", "8", "7"]

    def test_k_zero_and_none(self) -> None:
        items = [(make_item(str(i)), i / 10) for i in range(4)]
        assert RelevanceScorer().rank(items, now=NOW, k=0) == []
        assert len(RelevanceScorer().rank(items, now=NOW)) == 4

    def test_empty_candidates(self) -> None:
        assert RelevanceScorer().rank([], now=NOW) == []

    def test_default_now(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(scoring.time, "time", lambda: NOW + WEEK)
        ranked = RelevanceScorer().rank([(make_item(created_at=NOW), 0.5)])
        assert ranked[0].components["recency"] == pytest.approx(0.5)

    def test_recency_breaks_similarity_tie(self) -> None:
        old = make_item("old", created_at=NOW - 4 * WEEK)
        new = make_item("new", created_at=NOW)
        ranked = RelevanceScorer().rank([(old, 0.7), (new, 0.7)], now=NOW)
        assert [r.item.content for r in ranked] == ["new", "old"]

    def test_deterministic_tie_break(self) -> None:
        # Equal scores: sim 0.5/imp 0.5 vs sim 0.25/imp 1.0 under similarity=importance weights.
        w = ScoringWeights(similarity=1, recency=0, importance=1, frequency=0)
        hi_sim = make_item("hi_sim", importance=0.5, created_at=NOW - 100, item_id="z")
        lo_sim = make_item("lo_sim", importance=1.0, created_at=NOW, item_id="a")
        same_newer = make_item("newer", importance=0.5, created_at=NOW - 50, item_id="y")
        same_b = make_item("id_b", importance=0.5, created_at=NOW - 100, item_id="b")
        cands = [(hi_sim, 0.5), (lo_sim, 0.0), (same_newer, 0.5), (same_b, 0.5)]
        s = RelevanceScorer(weights=w)
        expected = ["newer", "id_b", "hi_sim", "lo_sim"]
        for seed in range(5):
            shuffled = cands[:]
            random.Random(seed).shuffle(shuffled)
            ranked = s.rank(shuffled, now=NOW)
            assert len({round(r.score, 12) for r in ranked}) == 1
            assert [r.item.content for r in ranked] == expected


# --- 4. diversity and de-duplication ---------------------------------------------


class TestMMR:
    def test_near_duplicates_demoted(self) -> None:
        top = scored(0.9, [1.0, 0.0], "top")
        dup = scored(0.89, [0.999, 0.01], "dup")
        distinct = scored(0.8, [0.0, 1.0], "distinct")
        result = maximal_marginal_relevance([top, dup, distinct], k=3, lambda_=0.7)
        assert [r.item.id for r in result] == ["top", "distinct", "dup"]

    def test_lambda_one_keeps_score_order(self) -> None:
        items = [
            scored(0.5, [1.0, 0.0], "c"),
            scored(0.9, [1.0, 0.0], "a"),
            scored(0.7, [1.0, 0.0], "b"),
            scored(0.1, [0.0, 1.0], "d"),
        ]
        result = maximal_marginal_relevance(items, k=3, lambda_=1.0)
        assert [r.item.id for r in result] == ["a", "b", "c"]

    @pytest.mark.parametrize("lam", [-0.1, 1.1])
    def test_invalid_lambda(self, lam: float) -> None:
        with pytest.raises(ValueError):
            maximal_marginal_relevance([scored(0.5, [1.0], "a")], k=1, lambda_=lam)

    @pytest.mark.parametrize("k", [0, -2])
    def test_non_positive_k(self, k: int) -> None:
        assert maximal_marginal_relevance([scored(0.5, [1.0], "a")], k=k) == []

    def test_k_larger_than_input(self) -> None:
        items = [scored(0.5, [1.0, 0.0], "a"), scored(0.4, [0.0, 1.0], "b")]
        assert len(maximal_marginal_relevance(items, k=10)) == 2

    def test_incomparable_embeddings(self) -> None:
        items = [
            scored(0.9, [1.0, 0.0], "a"),
            scored(0.8, [], "empty"),
            scored(0.7, [1.0, 0.0, 0.0], "longer"),
        ]
        result = maximal_marginal_relevance(items, k=3, lambda_=0.5)
        assert [r.item.id for r in result] == ["a", "empty", "longer"]

    def test_empty_input(self) -> None:
        assert maximal_marginal_relevance([], k=3) == []


class TestDeduplicate:
    def test_duplicate_dropped(self) -> None:
        low = scored(0.4, [0.6, 0.8], "low")
        high = scored(0.9, [0.6, 0.8], "high")
        result = deduplicate([low, high])
        assert [r.item.id for r in result] == ["high"]

    def test_distinct_kept_in_score_order(self) -> None:
        items = [
            scored(0.3, [0.0, 1.0], "b"),
            scored(0.9, [1.0, 0.0], "a"),
            scored(0.6, [1.0, 1.0], "c"),
        ]
        result = deduplicate(items)
        assert [r.item.id for r in result] == ["a", "c", "b"]

    def test_threshold_respected(self) -> None:
        a = scored(0.9, [1.0, 0.0], "a")
        b = scored(0.8, [1.0, 1.0], "b")  # cosine ~0.707
        assert [r.item.id for r in deduplicate([a, b], threshold=0.7)] == ["a"]
        assert [r.item.id for r in deduplicate([a, b], threshold=0.8)] == ["a", "b"]

    def test_incomparable_embeddings_kept(self) -> None:
        items = [scored(0.9, [1.0, 0.0], "a"), scored(0.8, [], "b")]
        assert len(deduplicate(items)) == 2


# --- 5. importance estimation ---------------------------------------------------------


class TestEstimateImportance:
    def test_fact_outranks_small_talk(self) -> None:
        fact = estimate_importance("My name is Alice and I live in Paris")
        talk = estimate_importance("thanks")
        assert fact > talk
        assert fact > BASELINE_IMPORTANCE
        assert talk < BASELINE_IMPORTANCE

    @pytest.mark.parametrize(
        "text",
        [
            "Remember that my sister's birthday is on May 3rd",
            "remember that I take my coffee black",
            "Please remember that the deploy key rotates monthly.",
            "Remember this: the wifi password is on the fridge",
        ],
    )
    def test_explicit_remember(self, text: str) -> None:
        assert estimate_importance(text) >= 0.8

    @pytest.mark.parametrize(
        "text",
        [
            "I am allergic to peanuts",
            "I'm vegetarian",
            "I work at a hospital",
            "I prefer short answers",
            "My favourite colour is green",
            "I plan to run a marathon",
            "I hate cilantro",
        ],
    )
    def test_self_disclosure_above_baseline(self, text: str) -> None:
        assert estimate_importance(text) > BASELINE_IMPORTANCE

    def test_curly_apostrophe(self) -> None:
        assert estimate_importance("I’m a nurse") == estimate_importance("I'm a nurse")

    @pytest.mark.parametrize("text", ["What time is it?", "how does this work", "Is it sunny?"])
    def test_pure_question_lowered(self, text: str) -> None:
        assert estimate_importance(text) < BASELINE_IMPORTANCE

    def test_question_with_fact_not_penalized(self) -> None:
        assert estimate_importance("I live in Oslo, is it cold there?") > BASELINE_IMPORTANCE

    @pytest.mark.parametrize("text", ["hi", "Thanks!", "ok", "OK.", "hello there", "lol"])
    def test_small_talk_low(self, text: str) -> None:
        assert estimate_importance(text) <= 0.15

    def test_numbers_raise_slightly(self) -> None:
        plain = estimate_importance("The meeting moved to the big room")
        numbered = estimate_importance("The meeting moved to room 42")
        dated = estimate_importance("The meeting moved to March")
        assert plain == pytest.approx(BASELINE_IMPORTANCE)
        assert plain < numbered <= plain + 0.1
        assert plain < dated <= plain + 0.1

    def test_more_cues_never_lower(self) -> None:
        one = estimate_importance("I live in Berlin")
        two = estimate_importance("I live in Berlin and I work as a chef")
        assert two >= one

    def test_remember_beats_plain_fact(self) -> None:
        fact = estimate_importance("I live in Berlin")
        assert estimate_importance("Remember that I live in Berlin") > fact

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "hi", "Remember that " + "I love " * 50, "?", "123", "What?", "a" * 1000],
    )
    def test_deterministic_and_bounded(self, text: str) -> None:
        first = estimate_importance(text)
        assert 0.0 <= first <= 1.0
        assert all(estimate_importance(text) == first for _ in range(3))

    def test_empty_is_zero(self) -> None:
        assert estimate_importance("") == 0.0
