# Design

## Context

See proposal.md for motivation. The module must be stdlib-only and build on `Message`
(`types.py`), the `Summarizer`/`LLM` protocols (`interfaces.py`), `estimate_tokens`,
`message_tokens`, `total_tokens` and `MESSAGE_OVERHEAD` (`tokens.py`) and `tokenize`
(`embeddings.py`). Token counting is pluggable via a `TokenCounter` callable everywhere.

## Goals / Non-Goals

**Goals:**
- Hard token-budget guarantees that hold for any pluggable token counter that is monotone in
  text length.
- Deterministic, offline extractive summarization good enough to preserve user facts.
- Composable pieces: compressor and rolling summary accept any `Summarizer`.

**Non-Goals:**
- Abstractive summarization without an LLM; exact tokenizer parity with any model.
- Deciding *when* to compress (the agent/short-term buffer owns that).

## Decisions

- **Truncation by binary search on character prefix.** With an arbitrary counter we cannot map
  tokens to characters, so we binary-search the longest prefix `p` with
  `counter(p.rstrip() + "…") <= max_tokens`; if that would cut mid-word, back off to the last
  whitespace in the prefix (a single over-long word is cut mid-word). Alternative (chars ≈ 4×tokens) was rejected because it breaks with
  custom counters. If even "…" does not fit, return the largest prefix that fits without it.
- **Sentence splitting by regex** on `[.!?]` followed by whitespace plus newlines. Simple and
  deterministic; abbreviations may split oddly, which is acceptable for summaries.
- **Scoring**: mean term frequency (over `tokenize` of all candidate sentences) of the
  sentence's tokens, ×1.5 for user first-person sentences, ×1.3 if digits are present, ×0.2 for
  greetings/filler (short sentences dominated by a fixed filler vocabulary). Ties break by
  original position, keeping determinism.
- **Selection**: greedy by score, adding a sentence only if the joined output still fits;
  output re-sorted chronologically. As a last resort (no sentence fits) the top sentence is
  truncated with `truncate_to_tokens`. A final `truncate_to_tokens` pass guards the budget.
- **Prior summaries** fed to summarizers as a synthetic `Message(role="assistant",
  content="[Earlier summary] ...")` placed first; lines already prefixed with a speaker are kept
  as-is when the extractive summarizer re-summarizes (the "[Earlier summary]" marker is
  stripped and the sentences keep their original "User:"/"Assistant:" speaker), so speaker
  attribution and first-person boosts survive repeated folding.
- **Compression algorithm**: if everything fits, return unchanged. Else reserve
  `int(budget * summary_ratio)` for the summary (including overhead), fill the remainder with
  the longest recent suffix; if fewer than `min_recent` fit, extend toward `min_recent`
  greedily within the whole budget. Summarize dropped + existing summary into everything not
  used by kept messages (`budget - used - MESSAGE_OVERHEAD`, at least the reserve minus
  overhead). If the summary comes back shorter, the leftover is given back: the newest dropped
  messages that fit are restored verbatim and the smaller remainder is re-summarized into the
  new remaining budget. Summarizer output always passes through `truncate_to_tokens`, so
  third-party summarizers cannot break the budget. If the newest message alone exceeds the
  budget, its content is truncated in a copy (`dataclasses.replace`, metadata copied and marked
  `truncated`); older messages are dropped without a summary, and if not even the overhead fits
  the result is empty.
- **RollingSummary** is a plain class whose `__call__` delegates to `absorb`, so it can be passed
  as `on_evict` (it also accepts a single `Message`); serialization stores only `text` and
  `max_tokens` (the summarizer and token counter are code, re-supplied to `from_dict`).
- **LLMSummarizer** sends one user message (transcript as `User:`/`Assistant:` lines, prior
  summaries as an "Earlier summary:" block, plus an instruction naming the token budget) with a
  configurable `system_prompt`; any exception or blank output triggers the fallback, whose output
  is truncated too.

## Risks / Trade-offs

- [Non-monotone custom token counters could break binary search] → final verification loop
  shrinks the result until it fits; documented requirement that counters be monotone.
- [Extractive summaries can lose context across many folds] → summaries keep speaker prefixes
  and boost self-disclosure; an `LLMSummarizer` can be plugged in for higher quality.
- [Budget too small for message overhead] → compressor returns an empty message list with an
  empty summary rather than violating the budget.
