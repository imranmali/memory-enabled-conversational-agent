# Proposal

## Why

The agent needs somewhere to keep the conversation it is currently having and somewhere to keep
what it has learned across sessions. Without a bounded short-term buffer, prompts grow without
limit; without a durable, searchable long-term store, recall and cross-session sync have nothing
to read from or write to. Both are prerequisites for compression, relevance scoring and sync.

## What Changes

- Add a bounded short-term conversation buffer that keeps recent messages within a message-count
  and token budget, evicts the oldest messages first, reports evicted messages to a callback (so
  they can be summarized or archived), and round-trips through a plain-dict representation.
- Add two long-term vector store implementations of the existing `VectorStore` protocol:
  - an in-memory store for tests and ephemeral sessions;
  - a SQLite-backed store (stdlib `sqlite3`) that persists across process restarts.
- Both stores enforce a fixed embedding dimension, isolate users, filter by memory kind, rank by
  cosine similarity with a deterministic tie-break, tombstone deletions (for sync propagation),
  and return defensive copies.
- No new runtime dependencies.

## Capabilities

### New Capabilities

- `short-term-memory`: bounded, FIFO-evicting buffer of recent conversation messages with token
  accounting, eviction notification and serialization.
- `long-term-memory-store`: durable per-user storage of memory items with cosine-similarity
  search, kind filtering, tombstone deletion and persistence.

### Modified Capabilities

<!-- None: no existing specs. -->

## Impact

- New modules: `src/memory_agent/short_term.py`, `src/memory_agent/stores.py`.
- New tests: `tests/test_short_term.py`, `tests/test_stores.py`.
- Builds on shared `types.Message`, `types.MemoryItem`, `interfaces.VectorStore`,
  `tokens.message_tokens` and `embeddings.cosine_similarity`; none of these change.
- Consumers (agent orchestration, compression, sync) can depend on these classes directly or via
  the `VectorStore` protocol.
