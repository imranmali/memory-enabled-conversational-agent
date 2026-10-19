# Proposal

## Why

The memory building blocks (short-term buffer, vector stores, relevance scoring, context
compression, cross-session sync) exist but nothing ties them into an agent a person can talk to.
Without an integrating layer the project cannot deliver its goal: an assistant that remembers the
user within a conversation, across sessions and across devices.

## What Changes

- Add deterministic extraction of durable user facts ("My name is Ada" → "User's name is Ada")
  from free text, normalized to third person, with multi-clause splitting and de-duplication.
- Add a long-term memory facade over any vector store: remember (with importance estimation and
  near-duplicate reinforcement), recall (scored, de-duplicated, diversified, access-tracked),
  forget one item, forget everything for a user (tombstones that propagate via sync), list facts.
- Add LLM providers: an offline, deterministic model grounded in the memory sections of the
  system prompt (default, used by demos and tests) and an optional Anthropic-backed model whose
  SDK is imported lazily, plus a factory selecting a provider by name.
- Add the conversational agent that orchestrates fact extraction, recall, context compression
  within a token budget, prompt assembly, short-term buffering with a rolling summary of evicted
  turns, session tracking, end-of-session summaries and synchronisation; it can be built from a
  data directory with persistent SQLite memory and file-based sync.
- Add a `memory-agent` command-line chat (also `python -m memory_agent`) that works with piped
  input and offers memory-inspection commands.
- Export the public API of all modules from the package root.

## Capabilities

### New Capabilities
- `fact-extraction`: deriving durable, third-person user facts from conversational text.
- `memory-recall`: storing, reinforcing, recalling and forgetting a user's long-term memories.
- `llm-providers`: pluggable language-model backends (offline grounded model, Anthropic) and
  selection by name.
- `conversational-agent`: the turn pipeline and session lifecycle of the memory-enabled agent,
  including budgeted prompt assembly and persistence from a data directory.
- `agent-cli`: the interactive/piped command-line interface to the agent.

### Modified Capabilities

## Impact

- New modules `src/memory_agent/{facts,memory,llm,agent,cli,__main__}.py`; package root
  `__init__.py` exports the public API. The `memory-agent` console script (already declared in
  `pyproject.toml`) becomes functional.
- New tests `tests/test_{facts,memory,llm,agent,cli,integration}.py`.
- Builds on existing types, protocols, embeddings, tokens, short-term buffer, stores, scoring,
  compression and sync modules without changing them.
- No new required runtime dependencies; `anthropic` stays an optional extra.
