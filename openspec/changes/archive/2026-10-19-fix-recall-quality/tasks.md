# Tasks

## 1. Stemming

- [x] 1.1 Fold a trailing "e" after suffix stripping in `embeddings._stem` and verify new `tests/test_embeddings.py` cases (live/lives/lived/living identical; "User lives in Lisbon" closer to "Where do I live?" than to an unrelated question) pass along with the existing embedding and scoring tests

## 2. Fact extraction

- [x] 2.1 Make the remember rule produce no fact for embedded questions and verify new `tests/test_facts.py` cases pass and an agent turn "Remember what I'm allergic to." learns no fact

## 3. Integration

- [x] 3.1 Run the full suite with coverage, `ruff check .`, `ruff format --check .` and `openspec validate --all --strict`, and verify all pass

## Workflow follow-up

- Archive the change after review (done by the lead).
