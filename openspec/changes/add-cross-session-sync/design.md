# Design

## Context

`MemoryItem` already carries `updated_at`, `origin` and a `deleted` tombstone flag, and every
store satisfies the `VectorStore` protocol (`upsert`, `get`, tombstoning `delete`, `all(user_id,
include_deleted=True)`). Concrete stores are developed in parallel, so sync must rely on the
protocol only and must not assume stores return fresh copies or shared references. Everything
lives in one stdlib-only module, `memory_agent/sync.py`.

## Goals / Non-Goals

**Goals:**
- A pure, deterministic `resolve_conflict` usable by both the backend and the device side.
- Change detection that is immune to device clock skew and never re-sends unchanged records.
- Backends simple enough to reason about and to replace with a hosted service later.

**Non-Goals:**
- Multi-process locking for `FileSyncBackend` (one writer process per directory is assumed;
  an in-process lock serialises threads).
- Vector-clock / CRDT merging of concurrent edits: last-write-wins is the accepted semantics.
- Compaction or garbage collection of tombstones.
- Network transport; a remote service only has to implement the `SyncBackend` protocol.

## Decisions

### Version identity = digest of the content-defining fields
A memory's *version* is a SHA-256 digest over its JSON form with `access_count` and
`last_accessed` removed. Equal digest means "same version". This lets the synchronizer detect
local edits even if the writer forgot to bump `updated_at`, and lets the backend tell whether a
push changed anything. The digest is also the final tie-breaker in `resolve_conflict`
(`updated_at`, then `origin`, then digest), making resolution a total order and therefore
symmetric even for pathological equal-timestamp/equal-origin pairs.
*Alternative:* a per-user `updated_at` watermark — rejected because device clocks skew and an
item merged in with an older timestamp would never be pushed.

### Access statistics are merged, never versioned
The resolved item takes `max(access_count)` and `max(last_accessed)`. Because they are not in the
digest, touching a memory never triggers a push on its own; the statistics travel opportunistically
whenever the record is exchanged. The backend still stores a raised maximum (and bumps the
revision) if a pushed version carries larger statistics, so they eventually converge.

### Backend revision per push, records tagged with the revision that last changed them
`push` validates all items first (all must belong to `user_id`), merges each with
`resolve_conflict`, and if any stored record changed (full-dict comparison) increments the user's
revision once and tags the changed records with it. A pushed version whose digest differs from the
stored one but *loses* also gets the stored winner re-tagged ("re-announced"): otherwise the pushing
device (e.g. one with a slow clock) would keep its losing copy forever, because nothing it can pull
would ever contain the winner. Pushing a version identical to the stored one never bumps. `pull(since)` returns copies of records whose
tag is greater than `since`, ordered by (revision, id). A shared `_BaseSyncBackend` implements this
over two hooks (`_load_log` / `_save_log`) so both backends share identical semantics.

### Device state: cursor + known versions per user
`MemorySynchronizer` keeps, per user, `cursor` (last backend revision fully applied) and `known`
(item id → digest last exchanged with the backend). `sync(user_id)`:
1. **Pull** since `cursor`. If the backend's revision is lower than `cursor`, the backend was reset:
   clear `known`, set `cursor = 0` and pull again. For each remote record: if the local copy is
   dirty (`digest(local) != known[id]`) and differs from the remote, count a conflict; upsert
   `resolve_conflict(local, remote)` when it differs from the local item (count as pulled); set
   `known[id] = digest(remote)`; advance `cursor`.
2. **Push** every item from `store.all(user_id, include_deleted=True)` whose digest differs from
   `known` (after stamping `origin="local"` → `device_id` and upserting the stamped copy). Update
   `known` for pushed items.
3. If the push advanced the revision beyond `cursor`, **pull again** from the old cursor. Our own
   echoed records match `known` and the local digest, so they cost nothing and are not counted;
   any record another device pushed concurrently is merged. This makes one `sync` complete and the
   next one a no-op.
*Alternative:* set `cursor` to the revision returned by push — rejected because it would silently
skip records pushed by another device between our pull and push.

`state()` returns `{"device_id": ..., "users": {uid: {"cursor": int, "known": {...}}}}`;
`load_state` deep-copies it and raises ValueError if `device_id` differs.

### Conflict counting
A conflict is counted only for a *true* concurrent change: the local item had an unpushed version
and the remote brought a different one. A plain remote update of an untouched local memory is
just a pull.

### FileSyncBackend layout
Two files per user in the directory: `<key>.json` (revision + records) and `<key>.sessions.json`
(session records), so recording a chat message does not rewrite the memory log. `<key>` is a slug
of `[A-Za-z0-9_-]` characters (max 40) followed by `-` and the first 16 hex chars of SHA-256 of the
raw user id: no separators or dots can survive, and the hash keeps "a/b" and "a_b" apart. Writes go
to a `tempfile.mkstemp` file in the same directory, are fsynced and `os.replace`d. Every load
validates the JSON structure and the stored `user_id`; any failure raises `SyncStorageError`
(a `RuntimeError`) before anything is written, so a corrupt file is never overwritten.

### SessionManager writes through
Every mutation (`start_session`, `record_message`, `end_session`) is written to the backend with
`put_session`, and the manager caches every session it has created or seen in `list_sessions`, so a
restarted manager can end a session after listing it. `list_sessions` sorts by `started_at` desc
(then id); `last_session` picks the max `ended_at` among ended sessions, skipping `exclude`.

## Risks / Trade-offs

- [Whole-file rewrite per push in `FileSyncBackend`] → acceptable for per-user memory volumes;
  the backend is a reference implementation and hosted backends can implement the protocol.
- [`known` grows with the number of memories] → it holds one short digest per id; tombstone
  compaction can prune it later.
- [Last-write-wins discards the losing concurrent edit] → documented semantics; the conflict count
  in `SyncReport` lets apps surface it.
- [Non-JSON metadata values] → the file backend requires JSON-compatible metadata (as do the
  stores); digests use `default=repr` so the in-memory path never crashes.
