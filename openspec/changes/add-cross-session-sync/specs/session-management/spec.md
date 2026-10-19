# Spec Delta

## Purpose

Tracks the lifecycle and cross-device history of conversation sessions so a new session can recall
when and about what the user last talked to the agent.

## ADDED Requirements

### Requirement: Starting a session
The system SHALL start a session for a user with a unique session id, the creating device's id and a start time taken from the injected clock, and SHALL record it in the sync backend immediately.

#### Scenario: Start records the session
- **WHEN** a session is started for "alice" at clock time 1000 on device "laptop"
- **THEN** it is active, has started_at 1000, device "laptop", 0 messages, and appears in the backend's sessions for "alice"

#### Scenario: Session ids are unique
- **WHEN** two sessions are started
- **THEN** their session ids differ

### Requirement: Counting messages
Recording a message in an active session SHALL increment its message count and persist the new count to the backend.

#### Scenario: Three messages recorded
- **WHEN** three messages are recorded in a session
- **THEN** the session's message count is 3 in the manager and in the backend

### Requirement: Ending a session
Ending a session SHALL set its end time from the clock and store the given summary and topics; ending an already ended session or recording a message in it MUST raise a ValueError.

#### Scenario: End with summary and topics
- **WHEN** a session is ended at clock time 2000 with summary "Planned a trip" and topics ["travel"]
- **THEN** the stored session has ended_at 2000, that summary and those topics

#### Scenario: Ending twice
- **WHEN** an ended session is ended again or receives a message
- **THEN** a ValueError is raised

### Requirement: Unknown sessions
Looking up, recording into or ending a session id the manager does not know MUST raise a KeyError.

#### Scenario: Unknown id
- **WHEN** `get`, `record_message` or `end_session` is called with an unknown session id
- **THEN** a KeyError is raised

### Requirement: Cross-device session history
Listing a user's sessions SHALL return the sessions recorded by every device sharing the backend, newest start first, and only those of that user.

#### Scenario: Sessions from two devices
- **WHEN** device A starts a session at 100 and device B starts one at 200 for the same user
- **THEN** listing from either device returns B's session first, then A's

#### Scenario: Other users excluded
- **WHEN** sessions exist for "alice" and "bob"
- **THEN** listing "alice" returns none of "bob"'s sessions

### Requirement: Last finished session lookup
The system SHALL return the user's most recently ended session, optionally excluding a given session id, or nothing when no session has ended, so a new session can recall what was last discussed.

#### Scenario: Greeting from another device
- **WHEN** a session ended on device A with summary "Talked about Rust" and device B starts a new session
- **THEN** the last session for the user, excluding B's new session, is A's with that summary

#### Scenario: Active sessions ignored
- **WHEN** a user's only session is still active
- **THEN** the last-session lookup returns nothing

### Requirement: Session record serialization
A session record SHALL round-trip through a JSON-compatible mapping without loss, and the file backend MUST persist session records across re-instantiation.

#### Scenario: Round trip
- **WHEN** a session record is converted to a mapping, through JSON, and back
- **THEN** it equals the original

#### Scenario: Sessions survive restart
- **WHEN** a session is ended through a file backend and a new file backend opens the same directory
- **THEN** the new backend lists the ended session with its summary
