# Proposal

## Why

Every memory component (short-term buffer, vector store, scoring, compression, sync) needs to
agree on what a conversational turn and a memory record look like, how text becomes a vector,
and how tokens are counted. Defining these once, up front, lets the components be built in
parallel and swapped independently.

## What Changes

- Add the `memory_agent` Python package (src layout, zero required runtime dependencies).
- Add shared data types: conversational `Message`, long-term `MemoryItem` (with sync metadata:
  `updated_at`, `origin`, `deleted` tombstone) and `ScoredMemory`.
- Add an offline, deterministic text embedder plus cosine similarity.
- Add a dependency-free token estimator used for all context budgeting.
- Add protocols (`VectorStore`, `Scorer`, `Summarizer`, `LLM`) that decouple components.

## Capabilities

### New Capabilities
- `memory-model`: the shape, defaults and serialization of messages and memory records.
- `text-embedding`: turning text into fixed-size vectors and comparing them.

### Modified Capabilities
- None.

## Impact

- New package `src/memory_agent/` (`types.py`, `embeddings.py`, `tokens.py`, `interfaces.py`).
- New `pyproject.toml` with optional `anthropic` and `dev` extras.
