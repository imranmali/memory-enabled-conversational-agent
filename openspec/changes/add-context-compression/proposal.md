# Proposal

## Why

Long conversations eventually exceed the model's context window, and naively dropping old turns
loses facts the user already shared. The agent needs a dependency-free way to fit a conversation
into a token budget while carrying the essence of older turns forward as a summary.

## What Changes

- Add a token-aware text truncation helper that cuts at word boundaries and marks truncation.
- Add an offline, deterministic extractive summarizer that favours user self-disclosure and
  concrete details (numbers) and down-weights greetings and filler.
- Add an LLM-backed summarizer that delegates to any `LLM` implementation and falls back to the
  extractive summarizer on failure or empty output.
- Add a context compressor that keeps the most recent turns verbatim and summarizes older turns
  (together with any prior summary) so the result never exceeds a token budget.
- Add a rolling summary that folds evicted short-term messages into a bounded running summary and
  can be wired directly as a short-term buffer eviction callback; it is serializable.

## Capabilities

### New Capabilities
- `context-compression`: fitting conversation history into a token budget via truncation,
  summarization (extractive or LLM-backed), recent-turn retention and rolling summaries.

### Modified Capabilities

## Impact

- New module `src/memory_agent/compression.py` and tests `tests/test_compression.py`.
- Builds on existing shared types (`Message`), protocols (`Summarizer`, `LLM`), token estimation
  (`tokens.py`) and tokenization (`embeddings.tokenize`); no shared files change.
- No new runtime dependencies.
