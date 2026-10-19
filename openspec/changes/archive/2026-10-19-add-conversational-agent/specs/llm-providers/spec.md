# Spec Delta

## Purpose

Provides interchangeable language-model backends for the agent: a deterministic offline model
that answers from the memory context in the system prompt, and an optional Anthropic-backed
model, selectable by name.

## ADDED Requirements

### Requirement: Shared prompt sections
The system SHALL define the memory section headers of the agent's system prompt ("## Known facts about the user", "## Relevant memories", "## Conversation summary", "## Previous session") in one place and parse a system prompt back into those sections.

#### Scenario: Parse sections
- **WHEN** a system prompt containing a facts section with two "- " bullet lines is parsed
- **THEN** the facts section yields the two lines without the bullet markers

### Requirement: Offline grounded model
The system SHALL provide an offline model that needs no network and returns the same reply for the same system prompt and messages, answering only from the memory sections of the system prompt.

#### Scenario: Deterministic
- **WHEN** the offline model completes the same input twice
- **THEN** both replies are identical

### Requirement: Offline answers about the user
The offline model SHALL answer "what do you remember/know about me" by listing the known facts (mentioning the previous session when present), answer "what is my name" from the name fact or say it does not know, and acknowledge facts shared in the latest user message.

#### Scenario: What do you remember
- **WHEN** the facts section contains "User loves hiking" and the user asks "What do you remember about me?"
- **THEN** the reply mentions hiking

#### Scenario: Name known
- **WHEN** the facts section contains "User's name is Ada" and the user asks "What's my name?"
- **THEN** the reply contains "Ada"

#### Scenario: Name unknown
- **WHEN** there are no facts and the user asks "What's my name?"
- **THEN** the reply says it does not know the name

#### Scenario: Acknowledge new facts
- **WHEN** the user says "My name is Ada and I love hiking."
- **THEN** the reply greets Ada by name and confirms it will remember the hiking preference

### Requirement: Offline use of memories and sessions
The offline model SHALL reference the top relevant memory when one is present, greet with a mention of the previous session when there is no user message yet, and otherwise give a generic helpful reply.

#### Scenario: Relevant memory
- **WHEN** the relevant memories section contains a memory about Paris and the user says "Any travel tips?"
- **THEN** the reply mentions Paris

#### Scenario: Greeting with previous session
- **WHEN** the model is called with no messages and a previous-session section mentioning hiking
- **THEN** the reply is a greeting that mentions hiking

### Requirement: Anthropic model requests
The Anthropic model SHALL send the system prompt and the user/assistant messages (system-role messages dropped, consecutive same-role turns merged, starting and ending with a user turn) with the configured model and max tokens, the server-side fallback beta with default fallbacks when enabled, and the configured effort when set, and SHALL NOT send thinking or temperature settings.

#### Scenario: Request with fallbacks
- **WHEN** an Anthropic model with defaults completes with a fake client
- **THEN** the client's beta create call receives model "claude-opus-5-5", max_tokens 16000, betas ["server-side-fallback-2026-07-01"], fallbacks "default" and output_config {"effort": "medium"}

#### Scenario: Request without fallbacks or effort
- **WHEN** fallbacks are disabled and effort is None
- **THEN** the non-beta create call is used without betas, fallbacks or output_config

#### Scenario: Leading assistant turn
- **WHEN** the conversation starts with an assistant message
- **THEN** a user turn is prepended so the first message sent has role user

### Requirement: Anthropic responses
The Anthropic model SHALL return the concatenated text of the response's text blocks, and a short polite refusal message when the response stop reason is "refusal".

#### Scenario: Text blocks
- **WHEN** the response contains a thinking block and two text blocks
- **THEN** only the text blocks' text is returned

#### Scenario: Refusal
- **WHEN** the response stop reason is "refusal"
- **THEN** a polite refusal string is returned without reading the content

### Requirement: Optional SDK dependency
The Anthropic SDK SHALL be imported only when an Anthropic model is created without an injected client; if it is not installed a clear ImportError SHALL explain how to install the optional extra.

#### Scenario: SDK missing
- **WHEN** an Anthropic model is created without a client and the SDK cannot be imported
- **THEN** an ImportError mentioning "memory-enabled-conversational-agent[anthropic]" is raised

### Requirement: Provider selection by name
The system SHALL build a model from the names "offline" or "anthropic" (case-insensitive) and reject any other name with a ValueError.

#### Scenario: Unknown provider
- **WHEN** a model named "gpt" is requested
- **THEN** a ValueError listing the supported names is raised
