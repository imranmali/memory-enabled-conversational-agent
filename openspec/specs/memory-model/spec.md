# memory-model Specification

## Purpose
Defines the data model shared by every memory component: conversational messages, long-term
memory records with sync metadata, and scored recall results, plus token estimation.

## Requirements

### Requirement: Messages carry identity and time
Each conversational message SHALL have a role, content, a unique id and a creation timestamp,
and SHALL round-trip losslessly through a plain dictionary representation.

#### Scenario: Defaults are populated
- **WHEN** a message is created with only a role and content
- **THEN** it has a non-empty unique id and a timestamp close to the current time

#### Scenario: Dictionary round trip
- **WHEN** a message is converted to a dictionary and back
- **THEN** the result equals the original message

### Requirement: Memory records carry sync metadata
Each memory record SHALL carry an owner user id, content, embedding, kind, importance,
creation, last-access and last-update timestamps, an access count, an origin and a deletion
tombstone flag. Unset last-access and last-update timestamps SHALL default to the creation time.

#### Scenario: Timestamps default to creation time
- **WHEN** a memory record is created with a creation time and no other timestamps
- **THEN** its last-access and last-update timestamps equal the creation time

#### Scenario: Dictionary round trip
- **WHEN** a memory record is converted to a dictionary and back
- **THEN** the result equals the original record

### Requirement: Importance is bounded
A memory record's importance SHALL always lie in the range 0 to 1 inclusive; out-of-range
values SHALL be clamped.

#### Scenario: Out-of-range importance is clamped
- **WHEN** records are created with importance 5 and -1
- **THEN** their importance is 1 and 0 respectively

### Requirement: Accesses are recorded
Recording an access to a memory record SHALL increment its access count and set its last-access
time, without changing its last-update time.

#### Scenario: Touch a record
- **WHEN** a record with access count 0 is touched at time T
- **THEN** its access count is 1, its last-access time is T and its last-update time is unchanged

### Requirement: Token estimation
The system SHALL estimate token counts without external dependencies: empty text counts as
zero tokens, any non-empty text as at least one, and each message adds a fixed overhead.

#### Scenario: Empty and short text
- **WHEN** the token count of "" and of "a" is estimated
- **THEN** the results are 0 and 1

#### Scenario: Message overhead
- **WHEN** the tokens of a list of messages are totalled
- **THEN** the total equals the sum of each content estimate plus the per-message overhead
