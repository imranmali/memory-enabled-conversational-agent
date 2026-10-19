# Tasks

## 1. Short-term buffer

- [x] 1.1 Implement `ShortTermBuffer` construction with limit validation in `src/memory_agent/short_term.py`; verify with pytest tests for defaults, invalid limits and unbounded limits in `tests/test_short_term.py`
- [x] 1.2 Implement `add`/`extend` with FIFO eviction on message count and token budget (newest message always retained, custom token counter honoured); verify with pytest tests for count eviction, token eviction, oversized newest message, custom counter and extend
- [x] 1.3 Implement `on_evict` notification (once per call, chronological batch, not on clear); verify with pytest tests for batch delivery, no-eviction and clear
- [x] 1.4 Implement `messages`, `last`, `token_count`, `clear`, `__len__`, `__iter__`; verify with pytest tests for copy semantics, `last(n)` edge cases, length and iteration
- [x] 1.5 Implement `to_dict`/`from_dict` round-trip; verify with pytest tests that limits, order, fields and token count survive and that a callback passed at reconstruction is used
- [x] 1.6 Run `.venv/bin/pytest tests/test_short_term.py -q` and `.venv/bin/ruff check`/`ruff format --check` on the module and tests; verify all are clean

## 2. Long-term vector stores

- [x] 2.1 Implement the shared store base (dimension check, tombstone timestamp rule, ranking) and `InMemoryVectorStore` in `src/memory_agent/stores.py`; verify with the parametrized contract tests in `tests/test_stores.py` running against the in-memory store
- [x] 2.2 Implement `SQLiteVectorStore` (single table, compact embedding blob, JSON metadata, user index, lock-guarded connection, persisted dimension); verify with the same parametrized contract tests running against the SQLite store
- [x] 2.3 Write the contract tests covering upsert/replace, defensive copies, unknown ids, dimension rules, tombstone deletion and timestamps, search ranking/isolation/kind filter/tie-break/non-positive k, listing, counting and `isinstance(store, VectorStore)`; verify they pass for both stores
- [x] 2.4 Write SQLite-specific tests for persistence across close/reopen (tombstones, metadata, embeddings), dimension recovery on reopen, context-manager close and concurrent upserts from threads, using `tmp_path`; verify they pass
- [x] 2.5 Run `.venv/bin/pytest tests/test_stores.py -q` and `.venv/bin/ruff check`/`ruff format --check` on the module and tests; verify all are clean

## 3. Integration check

- [x] 3.1 Run `openspec validate add-memory-stores --strict` and both test files together; verify validation passes and all tests are green

## Workflow follow-up

- Archive the change after review (done by the lead)
