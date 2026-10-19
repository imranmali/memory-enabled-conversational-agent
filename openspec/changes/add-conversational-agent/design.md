# Design

## Context

See proposal.md - Why. All lower layers already exist and are reused unchanged: `Message`,
`MemoryItem`, `ScoredMemory` (`types.py`); the `VectorStore`, `Scorer`, `Summarizer`, `LLM`
protocols (`interfaces.py`); `HashingEmbedder` / `tokenize`; `estimate_tokens` /
`total_tokens`; `ShortTermBuffer` (with `on_evict`); `InMemoryVectorStore` /
`SQLiteVectorStore`; `RelevanceScorer`, `deduplicate`, `maximal_marginal_relevance`,
`estimate_importance`; `ContextCompressor`, `ExtractiveSummarizer`, `RollingSummary`,
`truncate_to_tokens`; `MemorySynchronizer`, `FileSyncBackend`, `SessionManager`. Constraints:
stdlib only at runtime, offline deterministic tests with injected clocks.

## Goals / Non-Goals

**Goals:**
- One orchestrating class (`MemoryAgent`) whose every collaborator is injectable.
- A hard, testable token-budget guarantee for each model call.
- An offline model good enough for the CLI demo and end-to-end tests.

**Non-Goals:**
- NLP-grade fact extraction (regex cues only; English only).
- Streaming replies, tool use, or multi-user conversations in one agent.
- Concurrent writers to one data directory.

## Decisions

- **Layering.** `facts.py` (pure functions) → `memory.py` (`LongTermMemory` over a
  `VectorStore`) → `llm.py` (providers + prompt section constants/parser) → `agent.py`
  (`MemoryAgent`) → `cli.py`. Each layer depends only on layers to its left plus existing
  modules, so each is unit-testable in isolation.
- **Fact extraction by ordered regex cues on clauses.** Text is split into sentences; questions
  (trailing "?" or interrogative opener) are skipped; sentences are split into clauses only on
  conjunctions/commas *followed by* a first-person opener ("and I …", ", my …"), which keeps
  "hiking and climbing" intact. The first matching cue per clause wins (remember-that, name,
  call-me, birthday, favorite, allergy, location, work, preference, "I am a"). Verbs are
  conjugated to the third person with a tiny rule set; possessives in values become "their".
  Alternative (an NLP library) rejected: runtime dependency and non-determinism.
- **Single-valued fact keys** (`fact_key`) let the agent supersede "User's name is Ada" with
  "User's name is Bea" by forgetting the older fact; multi-valued facts (likes) accumulate.
- **Reinforce instead of duplicate.** `remember` searches the same user+kind for the nearest
  item; at cosine ≥ 0.92 it raises importance, bumps `updated_at` (a new version that sync
  propagates) and refreshes `last_accessed`, keeping the id. Alternative (always insert and
  de-duplicate at recall) rejected: unbounded growth and noisy fact lists.
- **Recall = search pool (4k) → scorer → deduplicate → MMR → top k → touch.** Touching changes
  only access statistics, which `version_digest` ignores, so recall never causes sync traffic.
  A `min_similarity` filter (agent default 0.1) stops recency/importance alone from surfacing
  unrelated memories; an `exclude` predicate drops the current message and facts just learned.
  The default scorer gives `fact` memories a +0.05 kind boost so a distilled fact edges out the
  raw message it was extracted from. Small talk (importance below 0.15) is not stored.
- **Prompt sections as shared constants.** `llm.py` owns the headers and `parse_sections`;
  the agent renders them and `OfflineLLM` parses them, so the offline model is grounded in
  exactly what a real model would see. Facts are rendered oldest-first so "latest wins".
- **Budget arithmetic.** Sections get caps (persona 15%, facts 20%, previous session 10%,
  memories 20% of the budget) and are truncated line by line. The conversation budget is
  `budget − tokens(system without summary) − tokens(summary header)`. `ContextCompressor`
  counts its summary as `tokens(summary) + 4` and, since `ceil(a+b) ≤ ceil(a)+ceil(b)`, the
  final `tokens(system) + total_tokens(messages)` cannot exceed the budget. Config validation
  requires a budget ≥ 200 so the conversation share is always positive.
- **Rolling summary as `on_evict`.** Turns evicted from the buffer are folded into a
  `RollingSummary`, passed to the compressor as `existing_summary`; nothing silently vanishes.
- **Session end.** Summary = `ExtractiveSummarizer` over (rolling summary + buffer) bounded by
  `summary_max_tokens`; topics = top non-filler user keywords. Stored as a `summary` memory and
  on the session record, then `sync()`.
- **Data directory.** `memory.sqlite3` (`SQLiteVectorStore`), `sync/` (`FileSyncBackend`,
  shared by devices), `sync-state-<device>.json` (synchronizer cursor, saved atomically after
  each sync; unreadable or foreign state is ignored, which only costs a full re-pull). The
  synchronizer rejects the reserved origin "local", so device ids default to a sanitized host
  name (fallback "device-1").
- **AnthropicLLM.** `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"],
  fallbacks="default", output_config={"effort": effort})` by default; `client.messages.create`
  without beta fields when fallbacks are off. No `thinking`/`temperature` (rejected by the
  default model). The message list is normalised to start and end with a user turn (no
  prefill). `stop_reason == "refusal"` is checked before content. The SDK is imported inside
  the constructor only when no client is injected.
- **CLI** reads `stdin.readline()` (no readline dependency), prints prompts only on a TTY,
  and always closes the agent in `finally` so EOF/`/quit` end and sync the session.

## Risks / Trade-offs

- [Regex extraction misses or mis-phrases facts] → conservative cues, negations and questions
  ignored, explicit "remember that" as an escape hatch; tests pin the supported forms.
- [Hashing embeddings give weak semantic recall] → the embedder is injectable; known facts are
  always in the prompt regardless of recall.
- [Prompt sections may be truncated under small budgets] → oldest facts dropped first;
  budget validated.
- [Two processes sharing one data dir could race on SQLite / state file] → documented as
  unsupported (one writer per data dir), matching `FileSyncBackend`'s assumption.
