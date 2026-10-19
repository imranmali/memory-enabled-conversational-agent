# Spec Delta

## Purpose

Defines how the memory-enabled agent handles each conversational turn and each session: learning
facts, recalling relevant memories, keeping the prompt within a token budget, summarizing what
falls out of short-term memory, and persisting and synchronising what it learned.

## ADDED Requirements

### Requirement: Turn pipeline
For each non-empty user message the agent SHALL extract and remember facts (importance at least 0.8), recall relevant memories other than the current message itself, build a system prompt with the persona and memory sections, call the language model, add both turns to short-term memory and report the reply with the recalled memories, learned facts, context size and whether compression occurred. Empty messages SHALL be rejected.

#### Scenario: Learning a fact
- **WHEN** the user says "My name is Ada and I love hiking."
- **THEN** the reply reports both facts as learned, the agent's known facts include them, and the reply text greets Ada

#### Scenario: Empty message
- **WHEN** the user sends "   "
- **THEN** a ValueError is raised

#### Scenario: Recall excludes the current message
- **WHEN** the user repeats an earlier message verbatim
- **THEN** no recalled memory has exactly that message's content

### Requirement: Storing user messages
When message storage is enabled, the agent SHALL store each user message as a long-term message memory tagged with the session, except small talk whose estimated importance is below the configured minimum.

#### Scenario: Message stored
- **WHEN** the user says "My name is Ada and I love hiking."
- **THEN** a message memory with that content exists for the user

#### Scenario: Small talk is not stored
- **WHEN** the user says "hi"
- **THEN** no message memory is stored for it

### Requirement: Superseding single-valued facts
When a newly learned fact has the same single-valued key as an existing fact with different content, the agent SHALL forget the older fact.

#### Scenario: Name change
- **WHEN** the user says "My name is Ada" and later "My name is Bea"
- **THEN** the known facts contain "User's name is Bea" and not "User's name is Ada"

### Requirement: Context budget
The agent SHALL keep the system prompt plus the messages sent to the model within the configured context budget, truncating memory sections and compressing conversation history (using the rolling summary of evicted turns) as needed.

#### Scenario: Long conversation stays within budget
- **WHEN** a conversation of many long turns exceeds the short-term limits
- **THEN** every reply reports context tokens at most the budget and later replies report compression

### Requirement: Nothing lost on overflow
Turns evicted from short-term memory SHALL be folded into a rolling summary that is included in later prompts, and facts learned early SHALL remain available after any number of turns.

#### Scenario: Early fact after overflow
- **WHEN** a fact is shared in the first turn and is followed by more turns than the short-term buffer holds
- **THEN** the fact is still in the prompt's known facts and is recalled for a related question

### Requirement: Session lifecycle
With a session manager, the agent SHALL start a session on request (or on the first message), remember the user's most recent previous session for the prompt and greeting, count messages, and on ending store a bounded summary as a summary memory and close the session with that summary and its top topics, then synchronise if a synchronizer is configured.

#### Scenario: Previous session is used
- **WHEN** a session in which the user said they love hiking is ended and a new agent on the same data starts a session
- **THEN** the new agent's greeting mentions hiking

#### Scenario: Session summary stored
- **WHEN** a session with at least one message ends
- **THEN** a summary memory exists for the user and the ended session record carries the summary and topics

### Requirement: Privacy reset
The agent SHALL forget all of the user's long-term memories and clear its short-term state on request, returning the number of memories forgotten.

#### Scenario: Forget everything
- **WHEN** a user with learned facts asks the agent to forget everything
- **THEN** the returned count is positive and the agent knows no facts

### Requirement: Persistent data directory
The agent SHALL be constructible from a data directory that holds a SQLite memory database, a file-based sync folder and a per-device synchronisation state file that is reloaded on start and saved after every sync; it SHALL release its resources on close and work as a context manager.

#### Scenario: Restart keeps memories
- **WHEN** an agent built from a data directory learns a fact and is closed, and a new agent is built from the same directory
- **THEN** the new agent knows the fact

#### Scenario: Sync state saved
- **WHEN** an agent built from a data directory synchronises
- **THEN** a sync state file for its device exists in the data directory
