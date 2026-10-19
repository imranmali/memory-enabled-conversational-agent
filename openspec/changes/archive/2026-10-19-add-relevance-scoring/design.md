# Design

## Context

Long-term recall arrives from a `VectorStore.search` as `(MemoryItem, similarity)` pairs. The
`Scorer` protocol in `interfaces.py` defines `rank(candidates, now, k, min_score)` returning
`ScoredMemory` objects. `MemoryItem` already carries `importance`, `last_accessed`,
`access_count`, `kind`, `deleted` and `embedding`, so every scoring signal is available without
changing shared types. See proposal.md - Why for motivation.

## Goals / Non-Goals

**Goals:**
- One self-contained module (`scoring.py`) with pure, deterministic functions; time is always
  injectable via `now`.
- Score values and breakdowns that are easy to inspect and debug.

**Non-Goals:**
- Learned/trained ranking models or calling an LLM to judge importance.
- Mutating memories during scoring (touching `last_accessed`/`access_count` stays the
  caller's responsibility after recall).

## Decisions

- **Linear blend with normalized weights.** `ScoringWeights` is a frozen dataclass validated in
  `__post_init__`; `normalized()` divides by the sum so callers can express weights as ratios.
  A linear blend is transparent and lets the breakdown explain each score. Alternative:
  multiplicative combination - rejected because a single zero component (e.g. never accessed)
  would zero the whole score.
- **Exponential half-life recency** (`0.5 ** (age / half_life)`, default 7 days). Smooth,
  bounded in (0, 1], with an intuitive tuning knob. Alternative: linear decay with a cutoff -
  rejected as it introduces a cliff.
- **Log-saturating frequency** (`log1p(n) / log1p(saturation)`, capped at 1) so early accesses
  matter most and popular memories cannot dominate.
- **Kind boosts are additive after weighting** and the total is clipped to [0, 1], letting e.g.
  `fact` or `summary` memories be nudged up without retuning weights. The breakdown stores raw
  component values plus `boost` (not weighted contributions) so it is weight-agnostic.
- **Deterministic tie-break** via a sort key `(-score, -similarity, -created_at, id)`.
- **MMR over `ScoredMemory.score`** (not raw similarity) as the relevance term, using
  `embeddings.cosine_similarity` between item embeddings. The input is first stably sorted by
  score so lambda = 1 reproduces score order; ties in the MMR objective keep that order.
  Empty or mismatched embeddings are treated as similarity 0 instead of raising, since stores
  may mix embedders.
- **Greedy de-duplication** in score order against kept items only, O(n^2) which is fine for
  recall-sized candidate lists (tens of items).
- **Regex-based importance heuristic** with word-boundary cue patterns, a capped cue bonus
  (+0.25 for the first strong cue, smaller increments after, total capped), a strong floor for
  explicit "remember that", +0.05 for numbers/dates, and penalties for pure questions and
  short small talk. Curly apostrophes are normalized. Output rounded to 4 decimals.

## Risks / Trade-offs

- [Heuristic importance misjudges phrasing, e.g. "I am not sure?"] → It is only a baseline;
  callers can override `importance` explicitly, and cue weights are module constants.
- [Wall-clock default for `now` makes results time-dependent] → All APIs accept `now`; tests
  always inject it.
- [MMR/dedup are quadratic] → Applied only to small, already-retrieved candidate sets.
