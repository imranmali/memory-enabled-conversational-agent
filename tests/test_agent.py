import json

import pytest

from memory_agent.agent import AgentConfig, AgentReply, MemoryAgent, default_device_id
from memory_agent.llm import FACTS_HEADER, MEMORIES_HEADER, PREVIOUS_SESSION_HEADER, SUMMARY_HEADER
from memory_agent.memory import LongTermMemory
from memory_agent.stores import InMemoryVectorStore
from memory_agent.sync import InMemorySyncBackend, MemorySynchronizer, SessionManager
from memory_agent.tokens import estimate_tokens, total_tokens


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class RecordingLLM:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, list(messages)))
        return self.reply


def make_agent(clock=None, llm=None, config=None, with_sessions=True, backend=None):
    clock = clock or Clock()
    backend = backend or InMemorySyncBackend()
    store = InMemoryVectorStore(clock=clock)
    memory = LongTermMemory(store, clock=clock)
    return MemoryAgent(
        "ada",
        llm=llm,
        memory=memory,
        config=config or AgentConfig(device_id="dev-a"),
        session_manager=SessionManager(backend, "dev-a", clock=clock) if with_sessions else None,
        synchronizer=MemorySynchronizer(store, backend, "dev-a") if with_sessions else None,
        clock=clock,
    )


# --------------------------------------------------------------------------- config


def test_default_device_id_is_never_local(monkeypatch):
    import memory_agent.agent as agent_module

    monkeypatch.setattr(agent_module.socket, "gethostname", lambda: "My Laptop!")
    assert default_device_id() == "My-Laptop"
    monkeypatch.setattr(agent_module.socket, "gethostname", lambda: "local")
    assert default_device_id() == "device-1"
    monkeypatch.setattr(agent_module.socket, "gethostname", lambda: "")
    assert default_device_id() == "device-1"
    assert AgentConfig().device_id != "local"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_budget_tokens": 100},
        {"recall_k": -1},
        {"summary_max_tokens": 0},
        {"device_id": "local"},
        {"device_id": ""},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AgentConfig(**kwargs)


def test_invalid_user_id():
    with pytest.raises(ValueError):
        MemoryAgent("  ")


# --------------------------------------------------------------------------- turn pipeline


def test_chat_learns_facts_and_greets():
    agent = make_agent()
    reply = agent.chat("My name is Ada and I love hiking.")
    assert isinstance(reply, AgentReply)
    assert reply.facts_learned == ["User's name is Ada", "User loves hiking"]
    assert agent.known_facts() == ["User's name is Ada", "User loves hiking"]
    assert "Nice to meet you, Ada!" in reply.text
    assert all(f.importance >= 0.8 for f in agent.memory.facts("ada"))
    # Session started lazily and both turns counted.
    assert agent.session is not None
    assert agent.session_manager.get(agent.session_id).message_count == 2
    # User message stored long-term; both turns buffered.
    assert [m.kind for m in agent.memory.memories("ada", kinds=["message"])] == ["message"]
    assert [m.role for m in agent.buffer.messages()] == ["user", "assistant"]


def test_default_agent_works_without_collaborators():
    agent = MemoryAgent("ada")
    assert agent.start_session() is None
    assert agent.chat("I love tea").text
    assert agent.sync() is None
    assert agent.end_session() is None
    assert agent.memory.memories("ada", kinds=["summary"])


def test_empty_message_rejected():
    with pytest.raises(ValueError):
        make_agent().chat("   ")


def test_prompt_contains_sections_and_recall_excludes_current_message():
    llm = RecordingLLM()
    agent = make_agent(llm=llm)
    agent.chat("I live in Paris")
    agent.chat("Paris has lovely museums and cafes")
    reply = agent.chat("Paris has lovely museums and cafes")
    assert all(s.item.content != "Paris has lovely museums and cafes" for s in reply.recalled)
    assert any(s.item.content == "User lives in Paris" for s in reply.recalled)
    system, messages = llm.calls[-1]
    assert FACTS_HEADER in system and "- User lives in Paris" in system
    assert MEMORIES_HEADER in system and "(fact) User lives in Paris" in system
    assert messages[-1].role == "user"
    assert messages[-1].content == "Paris has lovely museums and cafes"
    assert agent.last_system_prompt == system


def test_newly_learned_facts_not_recalled_in_same_turn():
    reply = make_agent().chat("I love hiking")
    assert reply.recalled == []


def test_small_talk_not_stored_long_term():
    agent = make_agent()
    agent.chat("hi")
    assert agent.memory.memories("ada", kinds=["message"]) == []
    agent2 = make_agent(config=AgentConfig(device_id="d", store_messages=False))
    agent2.chat("I went to the market today to buy apples")
    assert agent2.memory.memories("ada", kinds=["message"]) == []


def test_single_valued_fact_superseded():
    agent = make_agent()
    agent.chat("My name is Ada")
    agent.chat("I love tea")
    agent.chat("My name is Bea")
    assert agent.known_facts() == ["User loves tea", "User's name is Bea"]
    assert "Bea" in agent.chat("What's my name?").text


def test_context_budget_and_compression():
    config = AgentConfig(
        device_id="d",
        context_budget_tokens=400,
        short_term_max_messages=6,
        short_term_max_tokens=200,
        summary_max_tokens=60,
    )
    llm = RecordingLLM(reply="That is interesting, tell me more about it please. " * 3)
    agent = make_agent(llm=llm, config=config)
    agent.chat("My name is Ada and I love hiking.")
    replies = [
        agent.chat(f"Story part {i}: the river near the old mill was quiet and cold today. " * 2)
        for i in range(15)
    ]
    for reply, (system, messages) in zip(replies, llm.calls[1:], strict=True):
        assert reply.context_tokens <= 400
        assert reply.context_tokens == estimate_tokens(system) + total_tokens(messages)
    assert replies[-1].compressed
    assert agent.rolling_summary.text  # evicted turns were folded into the rolling summary
    system, _ = llm.calls[-1]
    assert SUMMARY_HEADER in system
    assert "- User's name is Ada" in system


def test_huge_message_is_truncated_within_budget():
    config = AgentConfig(device_id="d", context_budget_tokens=200)
    agent = make_agent(llm=RecordingLLM(), config=config)
    reply = agent.chat("word " * 2000)
    assert reply.context_tokens <= 200
    assert reply.compressed


def test_huge_persona_and_many_facts_are_truncated():
    config = AgentConfig(device_id="d", persona="persona " * 500, context_budget_tokens=300)
    llm = RecordingLLM()
    agent = make_agent(llm=llm, config=config)
    for i in range(40):
        agent.memory.remember("ada", f"User likes topic number {i} quite a lot", kind="fact")
    reply = agent.chat("tell me something")
    assert reply.context_tokens <= 300
    system, _ = llm.calls[-1]
    assert "topic number 39" in system  # newest facts kept
    assert "topic number 0 " not in system


def test_overlong_single_fact_is_cut():
    config = AgentConfig(device_id="d", context_budget_tokens=200)
    llm = RecordingLLM()
    agent = make_agent(llm=llm, config=config)
    agent.memory.remember("ada", "User likes " + "very " * 200 + "long things", kind="fact")
    assert agent.chat("hello there friend").context_tokens <= 200
    assert "- User likes very" in llm.calls[-1][0]


# --------------------------------------------------------------------------- sessions


def test_session_lifecycle_and_summary():
    backend = InMemorySyncBackend()
    clock = Clock()
    agent = make_agent(clock=clock, backend=backend)
    info = agent.start_session()
    assert info is not None and info.is_active
    assert agent.start_session().session_id == info.session_id  # idempotent while active
    agent.chat("My name is Ada and I love hiking in the mountains.")
    agent.chat("Mountains near Zurich are my favourite hiking spots.")
    ended = agent.end_session()
    assert ended is not None and not ended.is_active
    assert ended.summary and "hiking" in ended.summary.lower()
    assert "hiking" in ended.topics and "ada" not in ended.topics
    assert ended.message_count == 4
    summaries = agent.memory.memories("ada", kinds=["summary"])
    assert len(summaries) == 1 and summaries[0].session_id == ended.session_id
    assert agent.session is None and len(agent.buffer) == 0
    assert agent.last_sync is not None and agent.last_sync.pushed > 0
    assert agent.end_session() is None  # nothing active, nothing new to summarize

    # A new session sees the previous one.
    agent.start_session()
    assert agent.previous_session.session_id == ended.session_id
    greeting = agent.greeting()
    assert "Welcome back, Ada!" in greeting and "hiking" in greeting
    llm = RecordingLLM()
    agent.llm = llm
    agent.chat("hello")
    assert PREVIOUS_SESSION_HEADER in llm.calls[-1][0]


def test_forget_everything():
    agent = make_agent()
    agent.chat("My name is Ada and I love hiking.")
    assert agent.forget_everything() > 0
    assert agent.known_facts() == []
    assert len(agent.buffer) == 0
    assert "don't know your name" in agent.chat("What's my name?").text


# --------------------------------------------------------------------------- data dir


def test_from_data_dir_persists_and_saves_sync_state(tmp_path):
    with MemoryAgent.from_data_dir("ada", tmp_path / "data", device_id="laptop") as agent:
        agent.start_session()
        agent.chat("I love hiking")
        assert agent.config.device_id == "laptop"
    assert (tmp_path / "data" / "memory.sqlite3").exists()
    assert (tmp_path / "data" / "sync").is_dir()
    state_path = tmp_path / "data" / "sync-state-laptop.json"
    state = json.loads(state_path.read_text())
    assert state["device_id"] == "laptop"

    agent = MemoryAgent.from_data_dir("ada", tmp_path / "data", device_id="laptop")
    try:
        assert agent.known_facts() == ["User loves hiking"]
        assert agent.start_session() is not None
        assert agent.previous_session is not None
        report = agent.sync()
        assert report.pushed == 0  # state was restored: nothing to re-push
    finally:
        agent.close()
        agent.close()  # idempotent


def test_from_data_dir_ignores_bad_sync_state(tmp_path):
    (tmp_path / "sync-state-pc.json").write_text("{not json")
    agent = MemoryAgent.from_data_dir("ada", tmp_path, device_id="pc")
    assert agent.sync() is not None
    agent.close()
    (tmp_path / "sync-state-pc.json").write_text(json.dumps({"device_id": "other"}))
    agent = MemoryAgent.from_data_dir("ada", tmp_path, device_id="pc")
    agent.close()
    assert json.loads((tmp_path / "sync-state-pc.json").read_text())["device_id"] == "other"


def test_from_data_dir_rejects_reserved_device(tmp_path):
    with pytest.raises(ValueError):
        MemoryAgent.from_data_dir("ada", tmp_path, device_id="local")


def test_sync_state_write_failure_cleans_up(tmp_path, monkeypatch):
    agent = MemoryAgent.from_data_dir("ada", tmp_path, device_id="pc")
    import memory_agent.agent as agent_module

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(agent_module.os, "replace", boom)
    with pytest.raises(OSError):
        agent.sync()
    assert not list(tmp_path.glob(".sync-state-*.tmp"))
    monkeypatch.undo()
    agent.close()


def test_remember_question_does_not_create_fact():
    agent = MemoryAgent("u1")
    reply = agent.chat("Remember what I'm allergic to.")
    assert reply.facts_learned == []
    assert agent.known_facts() == []
