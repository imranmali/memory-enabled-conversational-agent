# Memory-Enabled Conversational Agent

[![CI](https://github.com/imranmali/memory-enabled-conversational-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/imranmali/memory-enabled-conversational-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Dependencies](https://img.shields.io/badge/runtime%20deps-none-brightgreen)

A conversational agent that **remembers its users**: within a conversation, across sessions,
and across devices.

- **Short-term buffer**: a token-bounded window of recent turns. Turns that overflow are folded
  into a rolling summary, so nothing is silently dropped.
- **Long-term vector recall**: facts, messages and session summaries stored as embeddings in an
  in-memory or SQLite vector store.
- **Relevance scoring**: recall ranks by semantic similarity, recency decay, importance and
  access frequency, then removes near-duplicates and diversifies the results (MMR).
- **Context compression**: the prompt always fits a token budget. Recent turns stay verbatim and
  older ones are summarized.
- **Cross-session sync**: last-write-wins replication with tombstones and per-user revisions,
  plus session history, so a new session (even on another device) picks up where the last one
  left off.

It runs fully offline with **zero runtime dependencies**: a deterministic hashing embedder and
an offline rule-based LLM are built in. Install the optional `anthropic` extra to plug in Claude.

---

## Quick start

```bash
git clone https://github.com/imranmali/memory-enabled-conversational-agent.git
cd memory-enabled-conversational-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",anthropic" to use Claude
```

Talk to it twice; the second run is a new session:

```console
$ memory-agent --user ada
assistant> Hello! I'm your memory-enabled assistant. Tell me about yourself and I'll remember.
you> My name is Ada and I love hiking.
assistant> Nice to meet you, Ada! Got it — I'll remember that you love hiking.
you> /quit
assistant> Goodbye! I'll remember our conversation.

$ memory-agent --user ada
assistant> Welcome back, Ada! Last time we talked about hiking. How can I help today?
you> What do you remember about me?
assistant> Here's what I remember about you: your name is Ada and you love hiking. Last time we talked about hiking.
you> /memories hiking
assistant> - [0.69] (fact) User loves hiking
           - [0.61] (message) My name is Ada and I love hiking.
           ...
```

Use Claude instead of the offline model (reads `ANTHROPIC_API_KEY` or an `ant auth login` profile):

```bash
memory-agent --user ada --llm anthropic            # default model: claude-opus-5-5
```

### CLI commands

| Command | What it does |
| --- | --- |
| `/facts` | List the facts the agent knows about you |
| `/memories [query]` | Show recalled memories with their score breakdown |
| `/sync` | Sync memories with the shared sync backend |
| `/stats` | Show memory counts, buffer size and session information |
| `/forget-all` | Delete (tombstone) every memory for this user |
| `/help`, `/quit` (or `/exit`, EOF) | Show help; end the session (summarize and sync) |

Options: `--user` (required), `--data-dir` (default `~/.memory-agent`), `--llm offline|anthropic`,
`--model`, `--device` (defaults to the hostname). Also runnable as `python -m memory_agent`.
Input can be piped, which is how CI smoke-tests two consecutive sessions.

## Library usage

```python
from memory_agent import MemoryAgent

# Session 1
with MemoryAgent.from_data_dir("ada", "./.memory", device_id="laptop") as agent:
    agent.start_session()
    reply = agent.chat("My name is Ada, I live in Lisbon and I'm allergic to peanuts.")
    print(reply.facts_learned)
    # ["User's name is Ada", 'User lives in Lisbon', 'User is allergic to peanuts']
    agent.end_session()  # stores a session summary and syncs

# Session 2: a fresh process, same data directory
with MemoryAgent.from_data_dir("ada", "./.memory", device_id="laptop") as agent:
    agent.start_session()
    print(agent.greeting())  # Welcome back, Ada! Last time we talked about ...
    reply = agent.chat("Where do I live?")
    print(reply.text)  # That reminds me: I remember that you live in Lisbon.
    for m in reply.recalled:  # ScoredMemory: item, score, similarity, components
        print(f"{m.score:.2f} [{m.item.kind}] {m.item.content}")
```

`reply` is an `AgentReply` with `text`, `recalled`, `facts_learned`, `context_tokens` and
`compressed`. Other useful calls are `agent.known_facts()`, `agent.sync()` and
`agent.forget_everything()`.

Every component sits behind a small protocol (`memory_agent.interfaces`), so you can swap in a
hosted vector database, a neural embedder or another LLM:

```python
from memory_agent import LongTermMemory, MemoryAgent, RelevanceScorer, ScoringWeights
from memory_agent.stores import SQLiteVectorStore

scorer = RelevanceScorer(
    weights=ScoringWeights(similarity=0.5, recency=0.3, importance=0.15, frequency=0.05),
    half_life_seconds=3 * 24 * 3600,
)
memory = LongTermMemory(SQLiteVectorStore("memory.sqlite3"), scorer=scorer)
agent = MemoryAgent("ada", memory=memory)
```

## Architecture

```
                     ┌──────────────────────────── MemoryAgent.chat(text) ────────────────────────────┐
user text ──► facts.extract_facts ──► LongTermMemory.remember(kind="fact")                            │
          │                                                                                         │
          ├─► LongTermMemory.recall ──► VectorStore.search ──► RelevanceScorer.rank ──► dedupe/MMR   │
          │                                                                                         │
          ├─► ShortTermBuffer ──(overflow)──► RollingSummary                                        │
          │                                                                                         │
          └─► ContextCompressor (budget) ──► system prompt: persona + facts + memories              │
                                                       + summary + previous session ──► LLM ──► reply
end_session ──► session summary ──► SessionManager ──► MemorySynchronizer ⇄ SyncBackend ⇄ other devices
```

| Module | Responsibility |
| --- | --- |
| `types.py`, `interfaces.py` | `Message`, `MemoryItem`, `ScoredMemory`; the `VectorStore` / `Scorer` / `Summarizer` / `LLM` protocols |
| `embeddings.py`, `tokens.py` | Offline hashing embedder, cosine similarity, token estimation |
| `short_term.py` | `ShortTermBuffer`: FIFO window bounded by message count and tokens, with an eviction callback |
| `stores.py` | `InMemoryVectorStore`, `SQLiteVectorStore` |
| `scoring.py` | `RelevanceScorer`, `ScoringWeights`, `deduplicate`, `maximal_marginal_relevance`, `estimate_importance` |
| `compression.py` | `ContextCompressor`, `ExtractiveSummarizer`, `LLMSummarizer`, `RollingSummary` |
| `sync.py` | `resolve_conflict`, `InMemorySyncBackend`, `FileSyncBackend`, `MemorySynchronizer`, `SessionManager` |
| `facts.py`, `memory.py` | Fact extraction; the `LongTermMemory` facade (remember / recall / forget) |
| `llm.py`, `agent.py`, `cli.py` | `OfflineLLM`, `AnthropicLLM`; `MemoryAgent`; the `memory-agent` CLI |

### How relevance is scored

```
score = w_sim · similarity + w_rec · 0.5^(age / half_life) + w_imp · importance
      + w_freq · log1p(accesses) / log1p(saturation) + kind_boost          (clipped to [0, 1])
```

Default weights are 0.6 / 0.2 / 0.15 / 0.05 with a 7-day half-life. Importance is estimated
when a memory is written: self-disclosures ("my name is…", "I'm allergic to…") and explicit
"remember that…" requests score high, and small talk scores low. Recall touches the returned
memories, so memories that keep proving useful stay fresh.

### How sync converges

Each `MemoryItem` carries `updated_at`, `origin` and a `deleted` tombstone. The backend gives
every user a monotonically increasing revision, so devices pull "everything after revision N"
without relying on clock agreement. Conflicts resolve last-write-wins on `updated_at`, with
`origin` as a deterministic tie-break; access statistics merge with `max`. Syncing again
without changes is a no-op, and deletes propagate.

## Development

```bash
pytest                                  # full test suite (offline)
pytest --cov --cov-fail-under=90        # with coverage, as CI runs it
ruff check . && ruff format --check .   # lint and format
openspec validate --all --strict        # spec validation (npm i -g @fission-ai/openspec)
```

CI (`.github/workflows/ci.yml`) runs on every push and pull request to `main`. It runs lint and
format checks, the test suite on Python 3.10–3.13 with a 90% coverage gate, a two-session CLI
smoke test, and strict OpenSpec validation.

## Spec-driven development with OpenSpec

The project was built with [OpenSpec](https://github.com/Fission-AI/OpenSpec), one change per
capability: propose (proposal, delta specs, design, tasks), implement with tests, then archive.
After archiving, the living specs are in [`openspec/specs/`](openspec/specs) and the change
history is in [`openspec/changes/archive/`](openspec/changes/archive).

| Change | Capabilities |
| --- | --- |
| `add-memory-foundation` | `memory-model`, `text-embedding` |
| `add-memory-stores` | `short-term-memory`, `long-term-memory-store` |
| `add-relevance-scoring` | `relevance-scoring` |
| `add-context-compression` | `context-compression` |
| `add-cross-session-sync` | `cross-session-sync`, `session-management` |
| `add-conversational-agent` | `memory-recall`, `fact-extraction`, `llm-providers`, `conversational-agent`, `agent-cli` |
| `fix-recall-quality` | modifies `text-embedding`, `fact-extraction` (found by end-to-end testing) |

To propose a new change with Claude Code, use `/opsx:propose`, then `/opsx:apply` and
`/opsx:archive`.

## License

MIT
