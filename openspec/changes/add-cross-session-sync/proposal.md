# Proposal

## Why

A user talks to the agent from several devices and across many sessions, but long-term memory
currently lives in one local vector store, so facts learned on a laptop are invisible on a phone
and nothing records what earlier sessions were about. We need a deterministic, offline-testable
way to replicate memories between devices and to remember sessions so a new conversation can pick
up where the last one ended.

## What Changes

- Add a conflict-resolution rule for two versions of the same memory: last-write-wins on
  `updated_at`, a deterministic and symmetric tie-break on `origin`, access statistics merged as
  maxima, and tombstones treated as ordinary versions so deletions propagate.
- Add a sync backend contract with a backend-assigned, monotonically increasing per-user revision
  (immune to device clock skew) offering push, incremental pull, and session records.
- Add two backends: an in-process backend and a directory-of-JSON-files backend with atomic
  writes, sanitised per-user file names and loud failure on corrupt files.
- Add a per-device synchronizer that pulls remote changes into any `VectorStore`, pushes only
  local changes, stamps locally created memories with the device id, reports pushed / pulled /
  conflict counts, and exposes its cursor state so the host app can persist it.
- Add session tracking: start / record message / end with summary and topics, cross-device
  listing, and lookup of the most recent ended session for "last time we talked about ..." greetings.

## Capabilities

### New Capabilities
- `cross-session-sync`: replication of a user's long-term memories between devices through a
  shared backend, including conflict resolution, revisions, durable storage and convergence.
- `session-management`: lifecycle and cross-device history of conversation sessions, including
  the lookup of the most recent finished session.

### Modified Capabilities
<!-- none -->

## Impact

- New module `src/memory_agent/sync.py` (stdlib only) and tests in `tests/test_sync.py`.
- Builds on `MemoryItem` (`updated_at`, `origin`, `deleted`) from `types.py` and the
  `VectorStore` protocol from `interfaces.py`; no shared file changes are required.
- No new runtime dependencies. Concrete vector stores are integrated later by the lead.
