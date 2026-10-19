# Tasks

## 1. Fact extraction

- [x] 1.1 Implement `extract_facts` and `fact_key` in `src/memory_agent/facts.py` (sentence/clause splitting, question skipping, ordered cues, third-person normalization, de-duplication) and verify it imports with `.venv/bin/python -c "import memory_agent.facts"`
- [x] 1.2 Add `tests/test_facts.py` covering every cue, multi-clause splitting, compound objects, remember-that pronouns, questions/negations/small talk, de-duplication and fact keys; verify with `.venv/bin/pytest tests/test_facts.py -q`

## 2. Long-term memory

- [x] 2.1 Implement `LongTermMemory` in `src/memory_agent/memory.py` (remember with importance default and reinforcement, recall pipeline with filters and access tracking, forget, forget_all, facts, memories, count) and verify it imports
- [x] 2.2 Add `tests/test_memory.py` with an injected clock covering new memories, empty content, reinforcement, kind separation, ranking, user isolation, filters, access stats without `updated_at` change, forgetting (owner only, all, tombstones) and listing; verify with `.venv/bin/pytest tests/test_memory.py -q`

## 3. LLM providers

- [x] 3.1 Implement section constants, `parse_sections`, `OfflineLLM`, `AnthropicLLM` (lazy SDK import, message normalisation, beta/fallback/effort kwargs, refusal handling) and `build_llm` in `src/memory_agent/llm.py`, and verify `OfflineLLM` satisfies `interfaces.LLM` via `isinstance`
- [x] 3.2 Add `tests/test_llm.py` covering offline intents (remember-me, name known/unknown, acknowledgement, memory reference, greeting, generic, determinism) and AnthropicLLM with a fake client asserting exact kwargs, message normalisation, text joining, refusal and the missing-SDK ImportError; verify with `.venv/bin/pytest tests/test_llm.py -q`

## 4. Conversational agent

- [x] 4.1 Implement `AgentConfig`, `AgentReply` and `MemoryAgent` in `src/memory_agent/agent.py` (turn pipeline, fact superseding, budgeted prompt assembly, rolling summary, sessions, end-of-session summary, sync, forget_everything, greeting, `from_data_dir` with sync-state persistence, close/context manager) and verify it imports
- [x] 4.2 Add `tests/test_agent.py` covering the turn pipeline, empty input, recall exclusion, superseding, budget compliance and compression, session lifecycle and summary memory, forget_everything, data-dir restart and sync-state file; verify with `.venv/bin/pytest tests/test_agent.py -q`

## 5. Command-line interface and package API

- [x] 5.1 Implement `src/memory_agent/cli.py` (`main(argv)`, options, commands, TTY-only prompt, session end on exit) and `src/memory_agent/__main__.py`, export the public API of all modules from `src/memory_agent/__init__.py`, and verify `printf 'hi\n/quit\n' | .venv/bin/memory-agent --user t --data-dir <tmp>` exits 0
- [x] 5.2 Add `tests/test_cli.py` running `main` in-process for each command, missing user, unavailable provider, `python -m memory_agent`, and the exact two-run CI smoke scenario; verify with `.venv/bin/pytest tests/test_cli.py -q`

## 6. Integration

- [x] 6.1 Add `tests/test_integration.py` covering cross-instance memory on one data dir (greeting and answers use the previous session), buffer overflow with early facts still recallable within budget, two devices with separate SQLite stores converging through a shared `FileSyncBackend` including delete propagation, and relevant-fact ranking; verify with `.venv/bin/pytest tests/test_integration.py -q`
- [x] 6.2 Run the full suite with coverage (`.venv/bin/pytest --cov --cov-fail-under=90`), `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .` and the CI CLI smoke commands, and verify all pass
- [x] 6.3 Run `openspec validate add-conversational-agent --strict` and verify it passes

## Workflow follow-up

- Archive the change after review (done by the lead)
