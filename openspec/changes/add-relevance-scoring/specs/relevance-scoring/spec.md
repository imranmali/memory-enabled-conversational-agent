# Spec Delta

## Purpose

Defines how recalled long-term memories are scored, ranked, diversified and de-duplicated
against a query, and how a baseline importance is estimated for new memory text.

## ADDED Requirements

### Requirement: Scoring weights are validated and normalized
The system SHALL accept non-negative weights for similarity, recency, importance and frequency, defaulting to 0.6, 0.2, 0.15 and 0.05. It MUST reject any negative weight and an all-zero weight set with a ValueError, and SHALL expose a normalized copy whose weights sum to 1 while preserving their ratios.

#### Scenario: Default weights
- **WHEN** weights are created without arguments
- **THEN** they are similarity 0.6, recency 0.2, importance 0.15 and frequency 0.05

#### Scenario: Negative weight rejected
- **WHEN** weights are created with any negative value
- **THEN** a ValueError is raised

#### Scenario: All-zero weights rejected
- **WHEN** weights are created with every value equal to 0
- **THEN** a ValueError is raised

#### Scenario: Normalization
- **WHEN** weights 2, 1, 1, 0 are normalized
- **THEN** the result is 0.5, 0.25, 0.25, 0 and sums to 1

### Requirement: Scorer configuration is validated
The scorer SHALL reject a non-positive recency half-life, a frequency saturation below 1, and non-finite kind boosts with a ValueError.

#### Scenario: Invalid half-life
- **WHEN** a scorer is created with a half-life of 0 or less
- **THEN** a ValueError is raised

#### Scenario: Invalid frequency saturation
- **WHEN** a scorer is created with a frequency saturation of 0
- **THEN** a ValueError is raised

#### Scenario: Non-finite kind boost
- **WHEN** a scorer is created with a kind boost of infinity or NaN
- **THEN** a ValueError is raised

### Requirement: Recency decays exponentially with a half-life
The recency component SHALL equal 0.5 raised to (age / half-life), where age is the time since the memory was last accessed, floored at 0. It MUST lie in [0, 1].

#### Scenario: Just accessed
- **WHEN** a memory was last accessed exactly at the scoring time
- **THEN** its recency is 1.0

#### Scenario: One half-life old
- **WHEN** a memory was last accessed one half-life before the scoring time
- **THEN** its recency is 0.5

#### Scenario: Access time in the future
- **WHEN** a memory's last access is later than the scoring time
- **THEN** its recency is 1.0

### Requirement: Frequency saturates logarithmically
The frequency component SHALL equal log(1 + access_count) / log(1 + saturation), capped at 1, so it is 0 for never-accessed memories and reaches 1 at the saturation count.

#### Scenario: Never accessed
- **WHEN** a memory has an access count of 0
- **THEN** its frequency is 0

#### Scenario: Saturated
- **WHEN** a memory's access count is at or above the saturation count
- **THEN** its frequency is 1

#### Scenario: Monotonic below saturation
- **WHEN** two memories below saturation differ only in access count
- **THEN** the more-accessed memory has the strictly higher frequency

### Requirement: Composite score with breakdown
The score SHALL be the normalized-weighted sum of similarity (clipped to [0, 1]), recency, importance and frequency, plus the additive boost for the memory's kind (0 if none), clipped to [0, 1]. The result MUST report the clipped similarity and a breakdown with keys similarity, recency, importance, frequency and boost.

#### Scenario: Weighted sum
- **WHEN** a memory is scored with known component values and default weights
- **THEN** the score equals the weighted sum of the components

#### Scenario: Similarity clipping
- **WHEN** a memory is scored with similarity -0.4 or 1.7
- **THEN** the reported similarity is 0 or 1 respectively

#### Scenario: Breakdown keys
- **WHEN** any memory is scored
- **THEN** the breakdown contains exactly similarity, recency, importance, frequency and boost

#### Scenario: Kind boost applied and score clipped
- **WHEN** a kind boost is configured for the memory's kind
- **THEN** the boost is added to the weighted sum and the final score stays within [0, 1]

### Requirement: Ranking of recall candidates
Ranking SHALL score (memory, similarity) candidates, exclude tombstoned memories and those scoring below the minimum score, order by score descending, and keep at most k results when k is given. When no scoring time is given, the current wall-clock time SHALL be used.

#### Scenario: Tombstones excluded
- **WHEN** a candidate memory is marked deleted
- **THEN** it does not appear in the ranked results

#### Scenario: Minimum score filter
- **WHEN** a minimum score is given
- **THEN** only memories scoring at or above it are returned

#### Scenario: Truncate to k
- **WHEN** k is given and more candidates qualify
- **THEN** exactly the k best-scoring memories are returned in score order

#### Scenario: Default time
- **WHEN** no scoring time is given
- **THEN** recency is computed against the current wall-clock time

### Requirement: Deterministic tie-breaking
Memories with equal scores SHALL be ordered by higher similarity, then newer creation time, then ascending id, so ranking is independent of input order.

#### Scenario: Equal scores
- **WHEN** candidates have identical scores
- **THEN** their order follows similarity, then creation time, then id regardless of input order

### Requirement: Diversity re-ranking with MMR
The system SHALL re-rank scored memories with Maximal Marginal Relevance, iteratively picking the memory maximizing lambda * score - (1 - lambda) * its maximum embedding cosine to already-picked memories, returning at most k. Lambda MUST be in [0, 1] else ValueError; lambda 1 SHALL reproduce score order.

#### Scenario: Near-duplicates demoted
- **WHEN** the two top-scored memories are near-duplicates and a distinct memory scores slightly lower
- **THEN** MMR with lambda below 1 picks the distinct memory second

#### Scenario: Lambda one keeps score order
- **WHEN** MMR runs with lambda 1
- **THEN** the result equals the memories sorted by score, truncated to k

#### Scenario: Invalid lambda
- **WHEN** MMR runs with lambda outside [0, 1]
- **THEN** a ValueError is raised

#### Scenario: Non-positive k
- **WHEN** MMR runs with k of 0 or less
- **THEN** an empty list is returned

#### Scenario: Incomparable embeddings
- **WHEN** memories have empty or different-length embeddings
- **THEN** their mutual similarity is treated as 0 instead of raising

### Requirement: Embedding de-duplication
The system SHALL drop any scored memory whose embedding cosine with an already-kept, higher-scored memory is at or above the threshold (default 0.95), returning the kept memories in score order.

#### Scenario: Duplicate dropped
- **WHEN** two memories have identical embeddings
- **THEN** only the higher-scored one is kept

#### Scenario: Distinct memories kept
- **WHEN** memories' pairwise cosine is below the threshold
- **THEN** all of them are kept in score order

### Requirement: Heuristic importance estimation
The system SHALL deterministically estimate an importance in [0, 1] for memory text from a 0.3 baseline: raised for self-disclosure and durable-fact cues, raised to at least 0.8 for explicit "remember that" requests, raised slightly for numbers or dates, and lowered for pure questions and short small talk.

#### Scenario: Fact outranks small talk
- **WHEN** "My name is Alice and I live in Paris" and "thanks" are estimated
- **THEN** the fact scores higher than the small talk and above the baseline

#### Scenario: Explicit remember request
- **WHEN** text starts with "Remember that"
- **THEN** its importance is at least 0.8

#### Scenario: Pure question lowered
- **WHEN** a question without durable-fact cues is estimated
- **THEN** its importance is below the baseline

#### Scenario: Numbers raise slightly
- **WHEN** otherwise-neutral text contains a number or date
- **THEN** its importance is slightly above the same text without it

#### Scenario: Deterministic and bounded
- **WHEN** the same text is estimated repeatedly, including empty text
- **THEN** the result is identical each time and within [0, 1]
