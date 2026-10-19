# memory-recall Specification

## Purpose
Gives the agent a per-user long-term memory on top of any vector store: storing memories with
an importance, reinforcing near-duplicates, recalling the most relevant ones for a query, and
forgetting individual memories or everything about a user.

## Requirements

### Requirement: Remember content
The system SHALL store non-empty content for a user as a long-term memory of a given kind (message, fact or summary), timestamped by the injected clock, with an importance that defaults to the heuristic importance estimate of the content. Empty or whitespace-only content SHALL be rejected with an error.

#### Scenario: New memory
- **WHEN** "User loves hiking" is remembered as a fact for user "ada" at clock time 100
- **THEN** a memory with that content, kind fact, created_at 100 and an importance in [0, 1] is stored for "ada"

#### Scenario: Empty content
- **WHEN** "   " is remembered
- **THEN** a ValueError is raised and nothing is stored

### Requirement: Near-duplicate reinforcement
The system SHALL reinforce an existing live memory of the same user and kind whose embedding similarity to new content reaches the duplicate threshold, instead of inserting a duplicate: it keeps the existing id, raises the importance to the maximum of both, and advances its update time to the current clock.

#### Scenario: Repeated fact
- **WHEN** the same fact is remembered twice for a user, the second time with higher importance and a later clock
- **THEN** the user has one memory for it, with the original id, the higher importance and the later update time

#### Scenario: Different kind is not merged
- **WHEN** identical content is remembered once as a message and once as a fact
- **THEN** two memories exist

### Requirement: Relevance-ranked recall
The system SHALL recall at most k of a user's live memories for a non-empty query, ranked by the relevance scorer at the current clock, de-duplicated, optionally diversified, filtered by minimum score, minimum similarity, allowed kinds and an optional exclusion predicate. An empty query or non-positive k SHALL return no memories.

#### Scenario: Relevant memory ranks first
- **WHEN** a user has a fact about hiking and several unrelated chat messages, and recall runs for "hiking trip"
- **THEN** the hiking fact is the first result

#### Scenario: Other users are isolated
- **WHEN** recall runs for user "bob"
- **THEN** no memory of user "ada" is returned

#### Scenario: Empty query
- **WHEN** recall runs for "  "
- **THEN** an empty list is returned

### Requirement: Recall records access
The system SHALL record an access (access count and last-accessed time at the current clock) on every recalled memory and persist it, without changing the memory's update time.

#### Scenario: Access statistics
- **WHEN** a memory with access count 0 is recalled at clock time 500
- **THEN** the stored memory has access count 1, last accessed 500 and an unchanged update time

### Requirement: Forgetting
The system SHALL let a user's memory be forgotten by id (only by its owner) and all of a user's memories be forgotten at once, using tombstones so deletions propagate to other devices; forgotten memories SHALL no longer be recalled or counted.

#### Scenario: Forget one
- **WHEN** a user forgets one of their memory ids
- **THEN** the call returns true and the memory is no longer recalled

#### Scenario: Forget someone else's memory
- **WHEN** user "bob" forgets a memory id owned by "ada"
- **THEN** the call returns false and the memory remains

#### Scenario: Forget everything
- **WHEN** a user with three memories forgets everything
- **THEN** the call returns 3, the user's count is 0 and the store holds three tombstones

### Requirement: Listing facts
The system SHALL list a user's live fact memories, oldest first, and count a user's live memories.

#### Scenario: Facts only
- **WHEN** a user has two facts and one message
- **THEN** listing facts returns the two facts and the count is 3
