# Proposal

## Why

Trying the finished agent end to end showed two recall-quality defects. Third-person facts did
not match first-person questions on the verb, because the stemmer mapped "lives" to "liv" but
left "live" as "live". And a request such as "Remember what I'm allergic to." was stored as the
bogus fact "What I'm allergic to", which then polluted later prompts.

## What Changes

- Stemming folds a trailing "e" after suffix removal, so live/lives/lived/living (and
  love/loves, hike/hikes) map to one feature.
- "Remember …" requests are only turned into facts when the remembered content is a
  statement. Embedded questions ("remember what/where/when/who/how/why/which/whether/if …")
  produce no fact.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `text-embedding`: inflected forms, including base forms ending in "e", share a feature.
- `fact-extraction`: explicit remember requests exclude embedded questions.

## Impact

- `src/memory_agent/embeddings.py` (`_stem`), `src/memory_agent/facts.py` (remember rule).
- Embedding vectors change for words ending in "e". Memories persisted by the earlier version
  keep their old vectors until they are re-embedded, so recall of those memories is slightly
  degraded but nothing breaks.
