# Spec Delta

## Purpose

Derives durable facts about the user (name, location, work, preferences, allergies, birthday,
favorites, explicit "remember that" requests) from conversational text, phrased in the third
person so they can be stored and shown back reliably.

## ADDED Requirements

### Requirement: Third-person fact extraction
The system SHALL extract durable user facts from text deterministically and phrase them in the third person (e.g. "My name is Ada" becomes "User's name is Ada", "I love hiking" becomes "User loves hiking", "I live in Paris" becomes "User lives in Paris"), stripping trailing punctuation from each fact.

#### Scenario: Name statement
- **WHEN** facts are extracted from "My name is Ada."
- **THEN** the result is exactly ["User's name is Ada"]

#### Scenario: Preference statement
- **WHEN** facts are extracted from "I love hiking!"
- **THEN** the result is exactly ["User loves hiking"]

#### Scenario: Other supported cues
- **WHEN** facts are extracted from statements of the form "call me X", "I'm from X", "I work at X", "I am a X", "I'm allergic to X", "my birthday is X" and "my favorite X is Y"
- **THEN** each yields one third-person fact containing X (and Y)

### Requirement: Multi-clause statements
The system SHALL split a sentence into separate facts when independent first-person clauses are joined by conjunctions or commas, while keeping a single clause with a compound object as one fact.

#### Scenario: Two clauses
- **WHEN** facts are extracted from "My name is Ada and I love hiking."
- **THEN** the result is ["User's name is Ada", "User loves hiking"] in that order

#### Scenario: Compound object stays together
- **WHEN** facts are extracted from "I love hiking and climbing"
- **THEN** the result is ["User loves hiking and climbing"]

### Requirement: Explicit remember requests
The system SHALL turn "remember that X" (and "please remember", "don't forget that") into the fact X with first-person pronouns normalized to refer to the user.

#### Scenario: Remember request with pronouns
- **WHEN** facts are extracted from "Please remember that my sister is called Jane"
- **THEN** the result is ["User's sister is called Jane"]

### Requirement: Questions and non-facts are ignored
The system SHALL NOT extract facts from questions, negated statements or text without a supported cue, and SHALL return an empty list for empty text.

#### Scenario: Question
- **WHEN** facts are extracted from "What is my name?"
- **THEN** the result is empty

#### Scenario: Small talk
- **WHEN** facts are extracted from "hello there" or ""
- **THEN** the result is empty

### Requirement: De-duplicated, ordered output
The system SHALL return each distinct fact once (case-insensitive), in order of first appearance.

#### Scenario: Repeated fact
- **WHEN** facts are extracted from "I love tea. I love tea. My name is Bo."
- **THEN** the result is ["User loves tea", "User's name is Bo"]

### Requirement: Single-valued fact keys
The system SHALL identify facts that hold one value at a time (name, preferred name, birthday, home location, a given favorite) by a key, so that a newer value can supersede an older one; other facts SHALL have no key.

#### Scenario: Same key for different names
- **WHEN** the keys of "User's name is Ada" and "User's name is Bob" are computed
- **THEN** they are equal and not empty

#### Scenario: Multi-valued fact
- **WHEN** the key of "User loves hiking" is computed
- **THEN** there is no key
