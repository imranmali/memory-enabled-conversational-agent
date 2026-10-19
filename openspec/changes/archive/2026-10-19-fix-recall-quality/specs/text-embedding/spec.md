# Spec Delta

## MODIFIED Requirements

### Requirement: Semantic overlap increases similarity
Texts that share meaningful (non-stopword) terms SHALL be more similar than unrelated texts,
and inflected forms of a word SHALL contribute to the same feature.

#### Scenario: Related beats unrelated
- **WHEN** "I love hiking in the mountains" is compared to "hiking mountains is my favourite" and to "tax return deadline"
- **THEN** the first similarity is greater than the second

#### Scenario: Inflections match
- **WHEN** "hikes" and "hiking" are embedded
- **THEN** their cosine similarity is greater than zero

#### Scenario: Base forms ending in e match their inflections
- **WHEN** "live", "lives", "lived" and "living" are each embedded
- **THEN** all four vectors are identical

#### Scenario: Third-person fact matches first-person question
- **WHEN** "User lives in Lisbon" is compared to "Where do I live?" and to "What is the weather?"
- **THEN** the first similarity is greater than the second
