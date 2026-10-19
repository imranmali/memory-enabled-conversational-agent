# Tasks

## 1. Conflict resolution

- [x] 1.1 Implement version digest and `resolve_conflict` (LWW on `updated_at`, origin tie-break, digest final tie-break, max access stats, id mismatch → ValueError) in `src/memory_agent/sync.py`; verify with tests for newer remote/local wins, origin tie-break symmetry, fully tied symmetry, access-stat merge and id mismatch
- [x] 1.2 Verify tombstone handling in `resolve_conflict` with tests for newer delete beats older edit and newer edit beats older delete

## 2. Sync backends

- [x] 2.1 Implement `SyncBackend` protocol, `SessionInfo`, `SyncStorageError` and the shared revision/merge logic with `InMemorySyncBackend`; verify with tests for revision advance, unchanged push keeps revision, stale push keeps newer record and re-announces it, mismatched user rejected atomically, incremental pull, unknown user pull and returned copies being isolated
- [x] 2.2 Implement `FileSyncBackend` (sanitised file names, atomic writes, structure validation); verify with tests for re-instantiation durability, path-traversal ids staying inside the directory, non-colliding similar ids, and corrupt file raising `SyncStorageError` without modifying the file
- [x] 2.3 Run the backend contract tests against both backends via pytest parametrisation and verify they pass

## 3. Device synchronizer

- [x] 3.1 Implement `SyncReport` and `MemorySynchronizer.sync` (pull/merge, origin stamping, push of changed items incl. tombstones, echo pull); verify with a dict-backed fake `VectorStore` in tests for two- and three-device convergence, idempotence, deletion propagation, concurrent-edit convergence with conflict count, losing-push correction, user isolation and origin stamping
- [x] 3.2 Implement device id validation, `state()` / `load_state()` and backend-reset recovery; verify with tests for invalid device ids, restored synchronizer no-op sync, foreign-device state rejection and re-push after backend reset

## 4. Session management

- [x] 4.1 Implement `SessionManager` (start, record_message, end_session, get, list_sessions, last_session) with injected clock; verify with tests for start persistence, unique ids, message counting, ending with summary/topics, double end and post-end message errors, unknown-id KeyErrors, cross-device newest-first listing, user exclusion, last-session lookup with exclude and active-only None
- [x] 4.2 Verify `SessionInfo` JSON round-trip and file-backend session persistence across re-instantiation with tests

## 5. Quality gates

- [x] 5.1 Run `.venv/bin/pytest tests/test_sync.py -q`, `.venv/bin/ruff check` and `.venv/bin/ruff format --check` on the owned files and verify all are clean
- [x] 5.2 Run `openspec validate add-cross-session-sync --strict` and verify it passes

## Workflow follow-up

- Archive the change after review (done by the lead)
