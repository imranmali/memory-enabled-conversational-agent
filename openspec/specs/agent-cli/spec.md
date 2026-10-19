# agent-cli Specification

## Purpose
Lets a person chat with the memory-enabled agent from a terminal or a pipe, inspect and manage
what it remembers, and have each run's session summarised and synchronised on exit.

## Requirements

### Requirement: Line-oriented chat
The command-line interface SHALL read one user message per line from standard input (interactive or piped), print each agent reply on a line prefixed "assistant> ", print an input prompt only when standard input is a terminal, and skip blank lines.

#### Scenario: Piped message
- **WHEN** "My name is Ada and I love hiking." is piped to the command for user "ada"
- **THEN** the output contains a line starting "assistant> " that mentions Ada

### Requirement: Options
The interface SHALL require a user id and accept a data directory (default "~/.memory-agent"), a model provider ("offline" by default, or "anthropic"), a model name and a device id (default derived from the host name, never "local").

#### Scenario: Missing user
- **WHEN** the command runs without a user id
- **THEN** it exits with a usage error

#### Scenario: Provider unavailable
- **WHEN** the anthropic provider is selected but its SDK is not installed
- **THEN** an error is printed and the exit status is non-zero

### Requirement: Memory commands
The interface SHALL support the commands /help, /facts, /memories [query], /forget-all, /sync, /stats and /quit (also /exit), and report unknown commands without sending them to the agent.

#### Scenario: Facts command
- **WHEN** a user who said "I love hiking" enters "/facts"
- **THEN** the output lists "User loves hiking"

#### Scenario: Unknown command
- **WHEN** "/bogus" is entered
- **THEN** the output says the command is unknown

### Requirement: Session end on exit
On /quit or end of input the interface SHALL end the session (storing its summary and synchronising) and close the agent, and exit with status 0.

#### Scenario: Remembered across runs
- **WHEN** one run receives "My name is Ada and I love hiking." then "/quit", and a second run with the same user and data directory receives "What do you remember about me?"
- **THEN** the second run's output mentions hiking
