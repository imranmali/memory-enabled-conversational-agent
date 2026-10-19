# Spec Delta

## Purpose

Keeps the most recent conversation messages within a bounded message-count and token budget,
handing older messages off to callers (for summarization or long-term storage) as they age out.

## ADDED Requirements

### Requirement: Configurable buffer limits
The buffer SHALL accept an optional maximum message count and an optional maximum token budget. Each limit MUST be a positive integer or absent (unbounded); any other value MUST be rejected with a ValueError at construction.

#### Scenario: Defaults applied
- **WHEN** a buffer is created without arguments
- **THEN** it is limited to 20 messages and 2000 tokens

#### Scenario: Invalid limit rejected
- **WHEN** a buffer is created with a maximum message count or token budget of zero or less
- **THEN** a ValueError is raised

#### Scenario: Unbounded limits
- **WHEN** a buffer is created with both limits absent and many messages are added
- **THEN** no message is evicted

### Requirement: FIFO eviction on message-count limit
When adding a message causes the buffer to hold more messages than the maximum message count, the buffer SHALL evict the oldest messages first until the count is within the limit, and the add operation MUST return the evicted messages in chronological order.

#### Scenario: Oldest message evicted
- **WHEN** a buffer limited to 3 messages holds m1, m2, m3 and m4 is added
- **THEN** the add returns [m1] and the buffer holds m2, m3, m4

#### Scenario: Nothing evicted under the limit
- **WHEN** a message is added to a buffer below both limits
- **THEN** the add returns an empty list

### Requirement: FIFO eviction on token budget
The buffer SHALL account each message as its content token estimate plus the fixed per-message overhead, using the configured token counter. When the total exceeds the token budget, the oldest messages MUST be evicted until it fits, except that the newest message is always retained.

#### Scenario: Token budget triggers eviction
- **WHEN** the token total would exceed the budget after an add
- **THEN** the oldest messages are evicted until the total is within budget and the reported token count equals the sum of the retained messages

#### Scenario: Oversized newest message retained
- **WHEN** a single message larger than the whole token budget is added
- **THEN** every earlier message is evicted and the oversized message remains as the only buffered message

#### Scenario: Custom token counter
- **WHEN** the buffer is created with a custom token counter
- **THEN** token accounting and eviction use that counter

### Requirement: Eviction notification
The buffer SHALL invoke the eviction callback, if configured, exactly once per add or extend call that evicts messages, passing the evicted batch in chronological order. The callback MUST NOT be invoked when nothing is evicted or when the buffer is explicitly cleared.

#### Scenario: Callback receives batch
- **WHEN** an add evicts two messages
- **THEN** the callback is called once with those two messages, oldest first

#### Scenario: Callback not called without eviction
- **WHEN** an add evicts nothing
- **THEN** the callback is not called

#### Scenario: Clear does not notify
- **WHEN** the buffer is cleared
- **THEN** the removed messages are returned, the buffer is empty, and the callback is not called

### Requirement: Bulk add
The buffer SHALL support adding several messages in order with one call, returning every message evicted during that call in chronological order.

#### Scenario: Extend past the limit
- **WHEN** five messages are added in one call to a buffer limited to 3 messages
- **THEN** the call returns the first two messages and the buffer holds the last three

### Requirement: Buffer inspection
The buffer SHALL expose its messages oldest-first as a copy that callers can modify without affecting the buffer, the last n messages, the current token total, its length, and iteration over its messages.

#### Scenario: Messages returned as a copy
- **WHEN** a caller modifies the list returned by the messages accessor
- **THEN** the buffer contents are unchanged

#### Scenario: Last n messages
- **WHEN** the last 2 messages are requested from a buffer holding m1, m2, m3
- **THEN** m2 and m3 are returned in that order, and requesting zero or fewer returns an empty list

#### Scenario: Length and iteration
- **WHEN** the buffer holds three messages
- **THEN** its length is 3 and iterating yields them oldest-first

### Requirement: Serialization round-trip
The buffer SHALL serialize its limits and messages to a plain dictionary and SHALL be reconstructable from that dictionary, preserving limits, message order and message fields; non-serializable options such as the token counter and callback MAY be supplied again at reconstruction.

#### Scenario: Round-trip preserves state
- **WHEN** a buffer is serialized to a dictionary and reconstructed from it
- **THEN** the new buffer has the same limits, the same messages in the same order and the same token count

#### Scenario: Callback supplied on reconstruction
- **WHEN** a buffer is reconstructed with an eviction callback and later evicts a message
- **THEN** that callback receives the evicted message
