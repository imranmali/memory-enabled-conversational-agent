# Spec Delta

## Purpose

Turns text into fixed-size, unit-length vectors offline and deterministically, and compares
vectors with cosine similarity, so semantic recall works without network access.

## ADDED Requirements

### Requirement: Deterministic fixed-size embeddings
The default embedder SHALL map any text to a vector of its configured dimension, SHALL return
identical vectors for identical text across processes, and SHALL L2-normalize non-empty vectors.

#### Scenario: Same text, same vector
- **WHEN** the same sentence is embedded twice by two embedder instances
- **THEN** the vectors are identical and have the configured dimension

#### Scenario: Unit length
- **WHEN** a non-empty sentence is embedded
- **THEN** the vector's Euclidean norm is 1 (within floating point tolerance)

#### Scenario: Invalid dimension
- **WHEN** an embedder is configured with a dimension of zero
- **THEN** configuration fails with an error

### Requirement: Semantic overlap increases similarity
Texts that share meaningful (non-stopword) terms SHALL be more similar than unrelated texts,
and inflected forms of a word SHALL contribute to the same feature.

#### Scenario: Related beats unrelated
- **WHEN** "I love hiking in the mountains" is compared to "hiking mountains is my favourite" and to "tax return deadline"
- **THEN** the first similarity is greater than the second

#### Scenario: Inflections match
- **WHEN** "hikes" and "hiking" are embedded
- **THEN** their cosine similarity is greater than zero

### Requirement: Cosine similarity
The system SHALL compute cosine similarity between equal-length vectors, return 0 when either
vector is all zeros, and reject vectors of different lengths.

#### Scenario: Identical and orthogonal vectors
- **WHEN** a vector is compared to itself and to an orthogonal vector
- **THEN** the similarities are 1 and 0

#### Scenario: Dimension mismatch
- **WHEN** vectors of length 2 and 3 are compared
- **THEN** an error is raised
