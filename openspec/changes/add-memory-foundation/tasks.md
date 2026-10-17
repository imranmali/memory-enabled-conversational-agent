# Tasks

## 1. Package scaffolding

- [x] 1.1 Create `pyproject.toml` (src layout, `dev`/`anthropic` extras, ruff + pytest config) and verify `pip install -e '.[dev]'` succeeds

## 2. Data model and tokens

- [x] 2.1 Implement `Message`, `MemoryItem`, `ScoredMemory` in `types.py` and verify `tests/test_types.py` passes
- [x] 2.2 Implement token estimation in `tokens.py` and verify `tests/test_tokens.py` passes

## 3. Embeddings and protocols

- [x] 3.1 Implement `HashingEmbedder` and `cosine_similarity` in `embeddings.py` and verify `tests/test_embeddings.py` passes
- [x] 3.2 Define `VectorStore`, `Scorer`, `Summarizer`, `LLM` protocols in `interfaces.py` and verify they import cleanly and `ruff check` passes

## Workflow follow-up

- Archive the change after review (done by the lead).
