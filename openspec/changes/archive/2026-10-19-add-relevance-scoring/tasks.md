# Tasks

## 1. Weights and scorer configuration

- [x] 1.1 Implement `ScoringWeights` (defaults, negative/all-zero validation, `normalized()`) in `src/memory_agent/scoring.py` and verify with pytest tests for defaults, both ValueErrors and normalization in `tests/test_scoring.py`
- [x] 1.2 Implement `RelevanceScorer.__init__` validation (half-life, frequency saturation, finite kind boosts) and verify with pytest tests for each ValueError

## 2. Component scores and composite score

- [x] 2.1 Implement `RelevanceScorer.recency` and verify with pytest tests for age 0, one half-life and future access
- [x] 2.2 Implement `RelevanceScorer.frequency` and verify with pytest tests for zero, saturated and monotonic access counts
- [x] 2.3 Implement `RelevanceScorer.score` (similarity clipping, normalized weighted sum, kind boost, clipping, breakdown) and verify with pytest tests for weighted sum, clipping, breakdown keys and kind boosts

## 3. Ranking

- [x] 3.1 Implement `RelevanceScorer.rank` (tombstone exclusion, min_score, k, default `now`, deterministic tie-break) and verify with pytest tests for each scenario plus `isinstance(scorer, Scorer)`

## 4. Diversity and de-duplication

- [x] 4.1 Implement `maximal_marginal_relevance` and verify with pytest tests for near-duplicate demotion, lambda 1 order, invalid lambda, non-positive k and incomparable embeddings
- [x] 4.2 Implement `deduplicate` and verify with pytest tests for duplicate removal and distinct-item retention

## 5. Importance estimation

- [x] 5.1 Implement `estimate_importance` and verify with pytest tests for fact vs small talk, "remember that" >= 0.8, question penalty, number bonus, determinism and bounds

## 6. Quality gates

- [x] 6.1 Run `.venv/bin/pytest tests/test_scoring.py -q`, `.venv/bin/ruff check` and `.venv/bin/ruff format` on owned files and verify all are clean, and `openspec validate add-relevance-scoring --strict` passes

## Workflow follow-up

- Archive the change after review (done by the lead)
