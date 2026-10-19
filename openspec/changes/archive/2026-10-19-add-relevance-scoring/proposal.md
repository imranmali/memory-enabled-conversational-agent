# Proposal

## Why

Raw cosine similarity alone recalls stale, trivial or near-duplicate memories ahead of the
durable facts a user actually cares about. The agent needs a single, deterministic relevance
model that blends similarity with recency, importance and access frequency, keeps recall
diverse, and can assign an importance baseline to new memories at write time.

## What Changes

- Add configurable scoring weights (similarity, recency, importance, frequency) that are
  validated and normalized to sum to 1.
- Add a relevance scorer satisfying the existing `Scorer` protocol: exponential recency decay
  with a configurable half-life, saturating log-scaled access frequency, per-kind additive
  boosts, a per-component score breakdown, and deterministic ranking with `k`/`min_score`
  filtering that excludes tombstoned memories.
- Add Maximal Marginal Relevance (MMR) re-ranking so near-duplicate memories do not crowd
  recall results.
- Add embedding-based de-duplication of scored memories.
- Add a heuristic, deterministic importance estimator for new memory text (durable facts and
  explicit "remember that" requests score high; small talk and pure questions score low).

## Capabilities

### New Capabilities
- `relevance-scoring`: How recalled long-term memories are scored, ranked, diversified and
  de-duplicated, and how the importance of new memory text is estimated.

### Modified Capabilities

_None._

## Impact

- New module `src/memory_agent/scoring.py` and tests `tests/test_scoring.py`.
- Builds on `MemoryItem`/`ScoredMemory` (types.py), the `Scorer` protocol (interfaces.py) and
  `cosine_similarity` (embeddings.py); no shared-file changes.
- No new runtime dependencies (stdlib only).
