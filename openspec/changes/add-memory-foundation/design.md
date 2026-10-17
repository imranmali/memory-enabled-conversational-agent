# Design

## Context

Greenfield project. See proposal.md for motivation. Components will be implemented in
parallel, so the shared contracts must be fixed before any of them start.

## Goals / Non-Goals

**Goals:**
- Run and test entirely offline with only the Python standard library.
- Keep every component replaceable behind a small protocol.

**Non-Goals:**
- Exact tokenization for any particular model (an estimate is sufficient for budgeting).
- State-of-the-art embedding quality (a pluggable `Embedder` protocol covers that).

## Decisions

- **Dataclasses over pydantic**: zero dependencies and trivially serializable via `asdict`.
- **Signed feature hashing over unigrams + bigrams** for the default embedder: deterministic
  across processes (uses `blake2b`, not Python's salted `hash`), no vocabulary to persist,
  and good enough lexical recall. A light suffix stemmer lets "hikes"/"hiking" collide.
  Alternative: sentence-transformers — heavy and needs model downloads; kept as a plug-in.
- **Sync metadata lives on `MemoryItem`** (`updated_at`, `origin`, `deleted`) so any store
  can participate in last-write-wins replication without a side table.
- **`typing.Protocol` interfaces** (`VectorStore`, `Scorer`, `Summarizer`, `LLM`) instead of
  abstract base classes, so third-party objects fit structurally.
- **~4 characters per token** estimate with a fixed per-message overhead of 4.

## Risks / Trade-offs

- [Hash collisions blur unrelated terms] → 512 dimensions by default; dimension configurable.
- [Token estimate drifts from real tokenizers] → budgets are conservative; the counter is
  injectable wherever budgets are enforced.
