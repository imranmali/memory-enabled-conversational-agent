# Spec Delta

## MODIFIED Requirements

### Requirement: Explicit remember requests
The system SHALL turn "remember that X" (and "please remember", "don't forget that") into the fact X with first-person pronouns normalized to refer to the user. When X is an embedded question (it starts with what, where, when, who, whom, whose, which, why, how, whether or if) no fact SHALL be produced.

#### Scenario: Remember request with pronouns
- **WHEN** facts are extracted from "Please remember that my sister is called Jane"
- **THEN** the result is ["User's sister is called Jane"]

#### Scenario: Remember request containing a question
- **WHEN** facts are extracted from "Remember what I'm allergic to." and from "Don't forget where I live"
- **THEN** both results are empty
