# Design

## Context

Shared building blocks already exist: `types.Message`, `types.MemoryItem` (with `updated_at`,
`origin` and `deleted` for last-write-wins sync), the `interfaces.VectorStore` protocol,
`tokens.message_tokens` and `embeddings.cosine_similarity`. This change adds concrete short-term
and long-term storage on top of them without changing any shared module. Runtime must stay
stdlib-only and tests must be deterministic and offline. See proposal.md for motivation and the
two delta specs for required behavior.

## Goals / Non-Goals

**Goals:**

- `ShortTermBuffer` in `short_term.py` and `InMemoryVectorStore` / `SQLiteVectorStore` in
  `stores.py`, both stores structurally satisfying `VectorStore`.
- Injectable clock for tombstone timestamps so tests never sleep.
- One contract test suite parametrized over both stores so they cannot drift apart.

**Non-Goals:**

- Approximate nearest-neighbour indexing; search is an exact linear scan per user.
- Relevance re-ranking (recency/importance) and `touch` bookkeeping on recall; those belong to
  the scorer and the agent.
- Schema migrations; the SQLite schema is created if absent and versioned only implicitly.

## Decisions

### ShortTermBuffer: deque plus running token total

Messages live in a `collections.deque`; a parallel deque caches each message's token cost
(`tokens.message_tokens(msg, counter)`) so eviction and `token_count()` are O(1) per message
instead of re-counting. Eviction pops from the left while `len > max_messages` or
(`total > max_tokens` and `len > 1`) — the `len > 1` guard is what keeps the newest message even
when it alone exceeds the budget. `extend` evicts per message but gathers one batch and calls
`on_evict` once at the end, so the callback sees the whole chronological batch.
Alternative considered: re-scanning the list on each add — simpler but O(n) per add.

`clear()` returns removed messages without notifying, because clearing is an explicit caller
decision (e.g. session reset) rather than ageing-out.

`to_dict()` returns `{"max_messages", "max_tokens", "messages": [Message.to_dict(), ...]}`.
`from_dict(data, **kwargs)` uses stored limits unless overridden in kwargs (also the place to pass
`token_counter` / `on_evict`) and re-adds messages via `extend`, so limits are re-enforced if the
caller overrides them smaller (evictions are then reported to `on_evict`).

### Copy semantics

Both stores store and return copies built with `MemoryItem.from_dict(item.to_dict())`
(`dataclasses.asdict` deep-copies lists and dicts). This keeps callers from silently mutating the
store, matches what a database-backed store does naturally, and is cheap at expected sizes.
`search` also returns copies.

### Shared base for dimension, tombstone and ranking logic

A small private base class `_BaseVectorStore` holds the dimension check, tombstone timestamp rule
(`max(clock(), previous_updated_at + 1e-6)`) and ranking (`cosine_similarity`, sort key
`(-similarity, -created_at, id)`). Backends implement only row-level primitives (`_put`,
`_get`, `_candidates(user_id, kinds)`, `_iter`, `_count`). Alternative considered: two fully
independent classes — rejected because the contract-level behavior would be duplicated and could
diverge.

### SQLite layout

Single table `memories` with scalar columns for every `MemoryItem` field used in filtering
(`id` primary key, `user_id`, `kind`, `deleted`, `created_at`, `updated_at`, ...), `embedding` as a
BLOB of `array('d', vec).tobytes()` (compact, exact float64 round-trip), `metadata` as JSON text,
and an index on `(user_id, deleted)`. `INSERT OR REPLACE` implements upsert. A `meta` key/value
table records the fixed dimension so it survives reopen; if absent, the dimension is inferred from
any stored row. Connection uses `check_same_thread=False` and every operation runs under a
`threading.Lock`, giving safe use from multiple threads. Search loads the user's live candidate
rows (optionally filtered by kind in SQL) and scores them in Python.

Alternative considered: JSON-encoding embeddings — readable but ~2-3x larger and slower to parse.

Byte order: `array('d')` uses native byte order; files are therefore portable between machines of
the same endianness, which is acceptable for a local single-user store.

### Clock injection

Both stores take `clock: Callable[[], float] = time.time`, used only for tombstone `updated_at`.
Item creation times come from the caller's `MemoryItem`.

## Risks / Trade-offs

- [Linear scan per user] → Acceptable for personal-assistant scale (thousands of items); the
  `VectorStore` protocol lets a hosted ANN store be swapped in later.
- [Copy overhead on every get/search] → Small for expected sizes; correctness of isolation is
  worth more than the allocation cost.
- [Native-endian embedding blobs] → Documented; a store file is not meant to be shared across
  architectures. Sync between machines goes through `MemoryItem` dicts, not raw files.
- [Single lock serializes SQLite access] → Simple and correct; throughput is bounded by the
  single connection anyway.
