# Tasks

## 1. Truncation

- [x] 1.1 Implement `truncate_to_tokens` in `src/memory_agent/compression.py` (binary search, word boundary, ellipsis within budget, non-positive budget → "") and verify it imports cleanly with `.venv/bin/python -c "import memory_agent.compression"`
- [x] 1.2 Add pytest tests for unchanged short text, word-boundary truncation with ellipsis under budget, custom counters and non-positive budgets; verify with `.venv/bin/pytest tests/test_compression.py -q`

## 2. Extractive summarizer

- [x] 2.1 Implement `ExtractiveSummarizer` (sentence split, TF scoring with self-disclosure/digit boosts and filler penalty, chronological speaker-prefixed output, budget guard) and verify it satisfies `interfaces.Summarizer` via `isinstance`
- [x] 2.2 Add pytest tests for ordering/prefixes, budget, empty input, system messages, determinism, self-disclosure and digit preference; verify with `.venv/bin/pytest tests/test_compression.py -q`

## 3. LLM summarizer

- [x] 3.1 Implement `LLMSummarizer` (transcript + budget instruction to `llm.complete`, truncation, fallback on error/empty) and verify it satisfies `interfaces.Summarizer`
- [x] 3.2 Add pytest tests with fake LLMs for truncated output, prompt contents, fallback on exception and on empty output; verify with `.venv/bin/pytest tests/test_compression.py -q`

## 4. Context compressor

- [x] 4.1 Implement `CompressedContext` and `ContextCompressor.compress` (fit check, summary reserve, recent suffix with `min_recent`, carry-forward of existing summary, leftover give-back, oversized newest message truncation, ValueError on budget ≤ 0) and verify the module imports
- [x] 4.2 Add pytest tests for unchanged fit, compression under budget across many budgets, suffix order and non-mutation, min_recent, carry-forward, single huge message and invalid budget; verify with `.venv/bin/pytest tests/test_compression.py -q`

## 5. Rolling summary

- [x] 5.1 Implement `RollingSummary` (`absorb`, `__call__`, `reset`, `to_dict`/`from_dict`) and verify the module imports
- [x] 5.2 Add pytest tests for absorb/call bounded text, folding prior content, reset and dict round-trip; verify with `.venv/bin/pytest tests/test_compression.py -q`

## 6. Quality gates

- [x] 6.1 Run `.venv/bin/ruff check` and `.venv/bin/ruff format --check` on the compression module and tests and verify both are clean
- [x] 6.2 Run `openspec validate add-context-compression --strict` and verify it passes

## Workflow follow-up

- Archive the change after review (done by the lead)
