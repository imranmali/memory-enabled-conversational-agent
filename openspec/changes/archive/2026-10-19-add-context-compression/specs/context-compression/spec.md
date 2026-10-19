# Spec Delta

## Purpose

Fits conversation history into a token budget by truncating text, summarizing older turns
(extractively or with an LLM), keeping recent turns verbatim, and maintaining a rolling summary of
evicted messages so earlier information carries forward.

## ADDED Requirements

### Requirement: Token-bounded truncation
The system SHALL truncate text so that its token count does not exceed a given maximum, preferring to cut at a word boundary and appending an ellipsis (counted within the budget) when text was shortened. Text already within the budget SHALL be returned unchanged, and a maximum of zero or less SHALL yield an empty string.

#### Scenario: Text within budget is unchanged
- **WHEN** text whose token count is at most the maximum is truncated
- **THEN** the identical text is returned

#### Scenario: Long text is shortened at a word boundary
- **WHEN** text exceeding the maximum is truncated
- **THEN** the result's token count is at most the maximum, ends with an ellipsis, and does not end in a partial word

#### Scenario: Non-positive budget
- **WHEN** text is truncated to a maximum of 0 or a negative number
- **THEN** an empty string is returned

### Requirement: Extractive summarization
The system SHALL provide an offline, deterministic summarizer that selects the highest-scoring sentences from user and assistant messages, emits them in original chronological order prefixed with the speaker ("User:" or "Assistant:"), one per line, and never exceeds the requested token budget. System messages SHALL be ignored.

#### Scenario: Speaker-prefixed chronological output
- **WHEN** a user and an assistant message are summarized with a generous budget
- **THEN** the summary contains a line starting "User:" before a line starting "Assistant:", in the order the messages occurred

#### Scenario: Budget respected
- **WHEN** many long messages are summarized with a small token budget
- **THEN** the summary's token count is at most that budget

#### Scenario: Empty input or non-positive budget
- **WHEN** an empty message list is summarized, or any messages are summarized with a budget of 0
- **THEN** an empty string is returned

#### Scenario: System messages ignored
- **WHEN** the input contains a system message
- **THEN** none of its content appears in the summary

#### Scenario: Deterministic output
- **WHEN** the same messages are summarized twice with the same budget
- **THEN** both summaries are identical

### Requirement: Salience-weighted sentence selection
The extractive summarizer SHALL score sentences by term frequency across the window, boost user self-disclosure statements (first person, e.g. "I ...", "my ...") and sentences containing digits, and penalize greetings and filler, so that under a tight budget salient facts are kept over pleasantries.

#### Scenario: Self-disclosure preferred over greeting
- **WHEN** messages containing a greeting and a user statement "my name is ..." are summarized with a budget that fits only one sentence
- **THEN** the summary contains the self-disclosure statement and not the greeting

#### Scenario: Numeric detail preferred over filler
- **WHEN** messages containing filler ("ok thanks") and a sentence with digits are summarized with a budget that fits only one sentence
- **THEN** the summary contains the sentence with digits

### Requirement: LLM-backed summarization with fallback
The system SHALL provide a summarizer that sends a transcript of the messages together with an instruction stating the token budget to an LLM, returns the LLM output truncated to the budget, and uses a fallback summarizer (extractive by default) when the LLM raises an error or returns empty output.

#### Scenario: LLM output used and truncated
- **WHEN** the LLM returns text longer than the budget
- **THEN** the summary is that text truncated to at most the budget, and the LLM received the transcript and the budget in its prompt

#### Scenario: Fallback on error
- **WHEN** the LLM raises an exception
- **THEN** the fallback summarizer's output is returned

#### Scenario: Fallback on empty output
- **WHEN** the LLM returns only whitespace
- **THEN** the fallback summarizer's output is returned

### Requirement: Budgeted context compression
The system SHALL compress a message list plus an optional existing summary into a result whose total tokens (summary counted as one message including overhead when non-empty, plus messages) never exceed the budget. A conversation already within budget SHALL be returned unchanged with zero messages compressed. A budget of zero or less SHALL raise a ValueError.

#### Scenario: Fits budget unchanged
- **WHEN** the existing summary and messages already fit the budget
- **THEN** the same messages and summary are returned with compressed count 0

#### Scenario: Over budget is compressed
- **WHEN** the messages exceed the budget
- **THEN** the result's tokens are at most the budget, older messages are summarized, and the compressed count equals the number of messages dropped

#### Scenario: Invalid budget
- **WHEN** compression is requested with a budget of 0
- **THEN** a ValueError is raised

### Requirement: Recent turns retained verbatim
When compressing, the system SHALL keep the most recent messages verbatim as a contiguous suffix in their original order, keeping at least the configured minimum of recent messages whenever they fit in the whole budget, and SHALL NOT mutate the input list.

#### Scenario: Most recent suffix kept in order
- **WHEN** a long conversation is compressed
- **THEN** the kept messages are the last N input messages in the same order, and the input list is unchanged

#### Scenario: Minimum recent messages kept
- **WHEN** the two newest messages fit within the budget but not within the non-summary share
- **THEN** both newest messages are still kept verbatim

### Requirement: Information carried forward in summaries
When compressing, the system SHALL summarize dropped messages together with the existing summary so that information from the prior summary can survive into the new summary.

#### Scenario: Prior summary content carried forward
- **WHEN** compression runs with an existing summary containing a distinctive fact and enough summary budget
- **THEN** the new summary still contains that fact

### Requirement: Oversized newest message truncated
If even the single newest message exceeds the budget, the system SHALL keep only that message with its content truncated to fit, with an empty summary, without altering the original message object.

#### Scenario: Single huge message
- **WHEN** the newest message alone exceeds the budget
- **THEN** the result holds one message whose content is a truncated version of the original, the summary is empty, and result tokens are at most the budget

### Requirement: Rolling summary of evicted messages
The system SHALL maintain a running summary bounded by a maximum token count that folds newly evicted messages into the prior summary, is callable directly with a message list (usable as a short-term buffer eviction callback), can be reset, and round-trips through a dictionary representation.

#### Scenario: Absorb updates bounded text
- **WHEN** evicted messages are absorbed, directly or by calling the object
- **THEN** the returned and stored summary text reflects those messages and its token count is at most the maximum

#### Scenario: Prior content folded in
- **WHEN** a second batch of messages is absorbed with a generous maximum
- **THEN** the summary still contains content from the first batch

#### Scenario: Reset and serialization
- **WHEN** the summary is serialized to a dict and restored, and later reset
- **THEN** the restored object has the same text and maximum, and after reset the text is empty
