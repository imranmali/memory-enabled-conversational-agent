"""End-to-end behaviour of the agent on real stores and sync backends."""

from memory_agent import (
    AgentConfig,
    FileSyncBackend,
    LongTermMemory,
    MemoryAgent,
    MemorySynchronizer,
    SessionManager,
    SQLiteVectorStore,
)


class Clock:
    def __init__(self, t: float = 10_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def test_remembers_across_agent_instances(tmp_path):
    clock = Clock()
    first = MemoryAgent.from_data_dir("ada", tmp_path, device_id="laptop", clock=clock)
    first.start_session()
    assert "Ada" in first.chat("My name is Ada and I love hiking.").text
    first.chat("This weekend I am planning a hiking trip to the Alps.")
    first.close()

    second = MemoryAgent.from_data_dir("ada", tmp_path, device_id="laptop", clock=clock)
    try:
        second.start_session()
        greeting = second.greeting()
        assert greeting.startswith("Welcome back, Ada!")
        assert "hiking" in greeting
        reply = second.chat("What do you remember about me?")
        assert "your name is Ada" in reply.text and "you love hiking" in reply.text
        assert second.chat("What's my name?").text == "Your name is Ada."
        trip = second.chat("Any advice for the hiking trip to the Alps?")
        assert any("Alps" in s.item.content for s in trip.recalled)
    finally:
        second.close()


def test_long_conversation_overflows_buffer_but_keeps_facts_within_budget():
    clock = Clock()
    config = AgentConfig(
        device_id="d",
        context_budget_tokens=600,
        short_term_max_messages=8,
        short_term_max_tokens=300,
    )
    agent = MemoryAgent("ada", config=config, clock=clock)
    agent.chat("My name is Ada. I live in Lisbon and I'm allergic to peanuts.")
    budgets = []
    for i in range(40):
        reply = agent.chat(
            f"Turn {i}: today I reorganised the bookshelf and watered the plants again."
        )
        budgets.append(reply.context_tokens)
    assert max(budgets) <= 600
    assert len(agent.buffer) <= 8  # the first turn was evicted long ago
    assert agent.rolling_summary.text
    assert set(agent.known_facts()) >= {
        "User's name is Ada",
        "User lives in Lisbon",
        "User is allergic to peanuts",
    }
    question = agent.chat("Is there anything with peanuts I should avoid?")
    assert any(s.item.content == "User is allergic to peanuts" for s in question.recalled)
    assert question.context_tokens <= 600
    assert "peanuts" in agent.last_system_prompt
    where = agent.chat("Any good restaurants in Lisbon?")
    assert any(s.item.content == "User lives in Lisbon" for s in where.recalled)


def _device(tmp_path, name, shared, clock):
    store = SQLiteVectorStore(tmp_path / f"{name}.sqlite3", clock=clock)
    backend = FileSyncBackend(shared)
    agent = MemoryAgent(
        "ada",
        memory=LongTermMemory(store, clock=clock),
        config=AgentConfig(device_id=name),
        session_manager=SessionManager(backend, name, clock=clock),
        synchronizer=MemorySynchronizer(store, backend, name),
        clock=clock,
    )
    return agent, store


def test_two_devices_converge_through_shared_file_backend(tmp_path):
    clock = Clock()
    shared = tmp_path / "shared-sync"
    phone, phone_store = _device(tmp_path, "phone", shared, clock)
    laptop, laptop_store = _device(tmp_path, "laptop", shared, clock)
    try:
        phone.start_session()
        phone.chat("My name is Ada and I love hiking.")
        phone.end_session()  # summary + sync

        laptop.start_session()  # pulls the phone's memories
        assert set(laptop.known_facts()) == {"User's name is Ada", "User loves hiking"}
        assert "hiking" in laptop.greeting()
        laptop.chat("I'm allergic to peanuts")
        laptop.end_session()

        phone.sync()
        assert "User is allergic to peanuts" in phone.known_facts()
        assert {i.id for i in phone_store.all("ada")} == {i.id for i in laptop_store.all("ada")}

        # A delete on one device propagates to the other.
        hiking = next(i for i in laptop.memory.facts("ada") if i.content == "User loves hiking")
        assert laptop.memory.forget("ada", hiking.id)
        laptop.sync()
        phone.sync()
        assert "User loves hiking" not in phone.known_facts()
        assert phone_store.get(hiking.id).deleted

        # Forget-everything on the phone wipes the laptop too.
        assert phone.forget_everything() > 0
        laptop.sync()
        assert laptop.known_facts() == []
        assert laptop.memory.count("ada") == 0
    finally:
        phone_store.close()
        laptop_store.close()


def test_recall_ranks_relevant_fact_above_chatter():
    clock = Clock()
    agent = MemoryAgent("ada", clock=clock)
    agent.chat("I'm allergic to shellfish")
    for text in [
        "The bus was late again this morning",
        "I watched a documentary about volcanoes",
        "My laptop battery is draining quickly",
        "We should repaint the kitchen soon",
        "The football match last night was exciting",
    ]:
        agent.chat(text)
    results = agent.memory.recall("ada", "can I eat shellfish at the restaurant?", k=4)
    assert results[0].item.content == "User is allergic to shellfish"
    assert results[0].item.kind == "fact"
    relevant = {"User is allergic to shellfish", "I'm allergic to shellfish"}
    ranks = [s.item.content in relevant for s in results]
    assert ranks[: ranks.count(True)] == [True] * ranks.count(True)  # chatter ranks below
