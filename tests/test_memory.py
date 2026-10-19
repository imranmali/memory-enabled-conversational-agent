import pytest

from memory_agent.memory import LongTermMemory
from memory_agent.stores import InMemoryVectorStore, SQLiteVectorStore


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def memory(clock):
    return LongTermMemory(InMemoryVectorStore(clock=clock), clock=clock)


def test_remember_new_item(memory, clock):
    clock.t = 100.0
    item = memory.remember("ada", "  User loves hiking ", kind="fact", session_id="s1")
    assert item.content == "User loves hiking"
    assert item.kind == "fact"
    assert item.created_at == 100.0 == item.updated_at
    assert item.session_id == "s1"
    assert 0.0 <= item.importance <= 1.0
    stored = memory.store.get(item.id)
    assert stored is not None and stored.content == "User loves hiking"


def test_remember_default_and_explicit_importance(memory):
    small = memory.remember("ada", "ok")
    explicit = memory.remember("ada", "something unique here", importance=0.9)
    assert small.importance == pytest.approx(0.1)
    assert explicit.importance == pytest.approx(0.9)


@pytest.mark.parametrize("content", ["", "   ", None])
def test_remember_rejects_empty(memory, content):
    with pytest.raises(ValueError):
        memory.remember("ada", content)
    assert memory.count("ada") == 0


def test_invalid_threshold():
    with pytest.raises(ValueError):
        LongTermMemory(InMemoryVectorStore(), dedup_threshold=0)


def test_reinforces_near_duplicate(memory, clock):
    first = memory.remember("ada", "User loves hiking", kind="fact", importance=0.5)
    clock.t = 2000.0
    second = memory.remember(
        "ada", "User loves hiking", kind="fact", importance=0.8, metadata={"source": "x"}
    )
    assert second.id == first.id
    assert memory.count("ada") == 1
    stored = memory.store.get(first.id)
    assert stored.importance == pytest.approx(0.8)
    assert stored.updated_at == 2000.0
    assert stored.created_at == 1000.0
    assert stored.metadata == {"source": "x"}
    # Lower importance never lowers the stored value.
    memory.remember("ada", "User loves hiking", kind="fact", importance=0.1)
    assert memory.store.get(first.id).importance == pytest.approx(0.8)


def test_different_kind_or_user_not_merged(memory):
    memory.remember("ada", "I love hiking", kind="message")
    memory.remember("ada", "I love hiking", kind="fact")
    memory.remember("bob", "I love hiking", kind="fact")
    assert memory.count("ada") == 2
    assert memory.count("bob") == 1


def test_recall_ranks_relevant_first(memory):
    memory.remember("ada", "User loves hiking in the mountains", kind="fact")
    for text in ["the weather is nice", "pizza for dinner tonight", "my code compiles now"]:
        memory.remember("ada", text)
    results = memory.recall("ada", "hiking trip", k=3)
    assert results[0].item.content == "User loves hiking in the mountains"
    assert len(results) <= 3


def test_recall_isolates_users_and_handles_empty(memory):
    memory.remember("ada", "User loves hiking", kind="fact")
    assert memory.recall("bob", "hiking") == []
    assert memory.recall("ada", "   ") == []
    assert memory.recall("ada", "hiking", k=0) == []


def test_recall_filters(memory):
    memory.remember("ada", "User loves hiking", kind="fact")
    memory.remember("ada", "We went hiking yesterday", kind="message")
    assert {r.item.kind for r in memory.recall("ada", "hiking", kinds=["fact"])} == {"fact"}
    assert memory.recall("ada", "hiking", min_score=1.01) == []
    assert memory.recall("ada", "quantum chromodynamics", min_similarity=0.2) == []
    excluded = memory.recall("ada", "hiking", exclude=lambda item: item.kind == "message")
    assert [r.item.kind for r in excluded] == ["fact"]
    plain = memory.recall("ada", "hiking", diversify=False)
    assert len(plain) == 2


def test_recall_records_access_without_new_version(memory, clock):
    item = memory.remember("ada", "User loves hiking", kind="fact")
    clock.t = 500.0 + 1000.0
    [result] = memory.recall("ada", "hiking")
    assert result.item.id == item.id
    stored = memory.store.get(item.id)
    assert stored.access_count == 1
    assert stored.last_accessed == 1500.0
    assert stored.updated_at == item.updated_at


def test_forget_owner_only(memory):
    item = memory.remember("ada", "User loves hiking", kind="fact")
    assert memory.forget("bob", item.id) is False
    assert memory.forget("ada", "missing") is False
    assert memory.forget("ada", item.id) is True
    assert memory.forget("ada", item.id) is False  # already tombstoned
    assert memory.recall("ada", "hiking") == []
    assert memory.store.get(item.id).deleted is True


def test_forget_all_uses_tombstones(memory):
    for text in ["User loves tea", "User lives in Rome", "a message about cars"]:
        memory.remember("ada", text, kind="fact" if text.startswith("User") else "message")
    memory.remember("bob", "User loves coffee", kind="fact")
    assert memory.forget_all("ada") == 3
    assert memory.count("ada") == 0
    assert memory.count("bob") == 1
    assert sum(1 for i in memory.store.all("ada", include_deleted=True) if i.deleted) == 3


def test_facts_and_memories_listing(memory, clock):
    memory.remember("ada", "User loves tea", kind="fact")
    clock.t += 1
    memory.remember("ada", "a message about weather", kind="message")
    clock.t += 1
    memory.remember("ada", "User lives in Rome", kind="fact")
    assert [f.content for f in memory.facts("ada")] == ["User loves tea", "User lives in Rome"]
    assert memory.count("ada") == 3
    assert len(memory.memories("ada")) == 3
    assert [m.kind for m in memory.memories("ada", kinds=["message"])] == ["message"]


def test_works_with_sqlite_store(tmp_path, clock):
    store = SQLiteVectorStore(tmp_path / "m.sqlite3", clock=clock)
    memory = LongTermMemory(store, clock=clock)
    memory.remember("ada", "User loves hiking", kind="fact")
    store.close()
    reopened = LongTermMemory(SQLiteVectorStore(tmp_path / "m.sqlite3", clock=clock), clock=clock)
    assert [f.content for f in reopened.facts("ada")] == ["User loves hiking"]
    assert reopened.recall("ada", "hiking")[0].item.access_count == 1
    reopened.store.close()
