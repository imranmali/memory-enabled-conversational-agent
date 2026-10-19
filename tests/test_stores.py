from __future__ import annotations

import math
import threading

import pytest

from memory_agent.embeddings import HashingEmbedder, cosine_similarity
from memory_agent.interfaces import VectorStore
from memory_agent.stores import InMemoryVectorStore, SQLiteVectorStore
from memory_agent.types import MemoryItem


class FakeClock:
    def __init__(self, t: float = 5000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def item(
    id: str,
    embedding: list[float],
    user_id: str = "alice",
    kind: str = "message",
    created_at: float = 100.0,
    **kwargs,
) -> MemoryItem:
    return MemoryItem(
        id=id,
        user_id=user_id,
        content=f"content {id}",
        embedding=embedding,
        kind=kind,
        created_at=created_at,
        **kwargs,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(params=["memory", "sqlite"])
def make_store(request, tmp_path, clock):
    created = []

    def factory(dim=None):
        if request.param == "memory":
            store = InMemoryVectorStore(dim=dim, clock=clock)
        else:
            store = SQLiteVectorStore(tmp_path / f"store{len(created)}.db", dim=dim, clock=clock)
        created.append(store)
        return store

    yield factory
    for store in created:
        store.close()


@pytest.fixture
def store(make_store):
    return make_store()


# -- contract: protocol ----------------------------------------------------------------------


def test_satisfies_vector_store_protocol(store):
    assert isinstance(store, VectorStore)


def test_empty_store(store):
    assert len(store) == 0
    assert store.count() == 0
    assert store.all() == []
    assert store.dim is None
    assert store.search("alice", [1.0, 0.0], k=5) == []


# -- contract: upsert / get ------------------------------------------------------------------


def test_upsert_then_get(store):
    original = item(
        "a",
        [0.1, 0.2, 0.3],
        session_id="s1",
        importance=0.9,
        last_accessed=150.0,
        access_count=3,
        updated_at=160.0,
        origin="laptop",
        metadata={"tags": ["x", "y"], "n": 1},
        kind="fact",
    )
    store.upsert(original)
    got = store.get("a")
    assert got is not None
    assert got.to_dict() == original.to_dict()
    assert got is not original


def test_replace_by_id(store):
    store.upsert(item("a", [1.0, 0.0]))
    replacement = item("a", [0.0, 1.0])
    replacement.content = "new content"
    store.upsert(replacement)
    got = store.get("a")
    assert got.content == "new content"
    assert got.embedding == [0.0, 1.0]
    assert len(store) == 1
    assert len(store.all(include_deleted=True)) == 1


def test_defensive_copy_on_upsert(store):
    original = item("a", [1.0, 0.0], metadata={"tags": ["t"]})
    store.upsert(original)
    original.content = "mutated"
    original.embedding[0] = 99.0
    original.metadata["tags"].append("mutated")
    got = store.get("a")
    assert got.content == "content a"
    assert got.embedding == [1.0, 0.0]
    assert got.metadata == {"tags": ["t"]}


def test_defensive_copy_on_get_and_search(store):
    store.upsert(item("a", [1.0, 0.0], metadata={"k": "v"}))
    got = store.get("a")
    got.content = "mutated"
    got.metadata["k"] = "mutated"
    got.embedding.append(5.0)
    (hit, _), *_ = store.search("alice", [1.0, 0.0])
    hit.content = "also mutated"
    listed = store.all()[0]
    listed.deleted = True
    fresh = store.get("a")
    assert fresh.content == "content a"
    assert fresh.metadata == {"k": "v"}
    assert fresh.embedding == [1.0, 0.0]
    assert not fresh.deleted


def test_get_unknown_id(store):
    assert store.get("missing") is None
    store.upsert(item("a", [1.0]))
    assert store.get("missing") is None


def test_integer_embeddings_accepted(store):
    store.upsert(item("a", [1, 0, 0]))
    assert store.get("a").embedding == [1.0, 0.0, 0.0]


def test_empty_embedding_rejected(store):
    with pytest.raises(ValueError):
        store.upsert(item("a", []))
    assert store.dim is None


# -- contract: dimension --------------------------------------------------------------------


def test_dimension_fixed_by_first_item(store):
    store.upsert(item("a", [1.0, 0.0, 0.0]))
    assert store.dim == 3
    with pytest.raises(ValueError):
        store.upsert(item("b", [1.0, 0.0, 0.0, 0.0]))
    assert store.get("b") is None


def test_dimension_fixed_at_construction(make_store):
    store = make_store(dim=3)
    assert store.dim == 3
    with pytest.raises(ValueError):
        store.upsert(item("a", [1.0, 0.0]))
    store.upsert(item("a", [1.0, 0.0, 0.0]))
    assert len(store) == 1


@pytest.mark.parametrize("bad", [0, -2, 2.5, True])
def test_invalid_dim_rejected(make_store, bad):
    with pytest.raises(ValueError):
        make_store(dim=bad)


def test_query_dimension_mismatch(store):
    store.upsert(item("a", [1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        store.search("alice", [1.0, 0.0])


def test_query_dimension_mismatch_even_with_nonpositive_k(make_store):
    store = make_store(dim=3)
    with pytest.raises(ValueError):
        store.search("alice", [1.0], k=0)


# -- contract: delete -----------------------------------------------------------------------


def test_delete_marks_tombstone(store, clock):
    store.upsert(item("a", [1.0, 0.0], updated_at=200.0))
    clock.t = 6000.0
    assert store.delete("a") is True
    got = store.get("a")
    assert got.deleted is True
    assert got.updated_at == 6000.0
    assert got.content == "content a"
    assert len(store) == 0


def test_delete_update_time_strictly_increases(store, clock):
    store.upsert(item("a", [1.0, 0.0], updated_at=9000.0))
    clock.t = 10.0  # clock behind the item's update time (e.g. clock skew from sync)
    assert store.delete("a") is True
    assert store.get("a").updated_at == pytest.approx(9000.0 + 1e-6, abs=0, rel=1e-15)
    assert store.get("a").updated_at > 9000.0


def test_delete_twice_keeps_advancing(store, clock):
    store.upsert(item("a", [1.0, 0.0], updated_at=100.0))
    clock.t = 500.0
    store.delete("a")
    store.delete("a")
    assert store.get("a").updated_at > 500.0


def test_delete_unknown_id(store):
    store.upsert(item("a", [1.0]))
    assert store.delete("missing") is False
    assert len(store) == 1
    assert store.get("missing") is None


def test_upsert_revives_tombstone(store):
    store.upsert(item("a", [1.0, 0.0]))
    store.delete("a")
    store.upsert(item("a", [1.0, 0.0]))
    assert len(store) == 1
    assert store.search("alice", [1.0, 0.0])[0][0].id == "a"


# -- contract: search -----------------------------------------------------------------------


def test_search_ranked_by_similarity(store):
    a = item("a", [1.0, 0.1, 0.0])
    b = item("b", [1.0, 1.0, 0.0])
    c = item("c", [0.0, 0.0, 1.0])
    for it in (c, a, b):
        store.upsert(it)
    query = [1.0, 0.0, 0.0]
    results = store.search("alice", query, k=2)
    assert [r[0].id for r in results] == ["a", "b"]
    assert results[0][1] == pytest.approx(cosine_similarity(query, a.embedding))
    assert results[1][1] == pytest.approx(cosine_similarity(query, b.embedding))
    assert results[0][1] > results[1][1]
    full = store.search("alice", query)
    assert [r[0].id for r in full] == ["a", "b", "c"]
    assert full[2][1] == pytest.approx(0.0)


def test_search_negative_similarity_last(store):
    store.upsert(item("pos", [1.0, 0.0]))
    store.upsert(item("neg", [-1.0, 0.0]))
    store.upsert(item("zero", [0.0, 0.0]))
    results = store.search("alice", [1.0, 0.0])
    assert [r[0].id for r in results] == ["pos", "zero", "neg"]
    assert results[-1][1] == pytest.approx(-1.0)


def test_search_user_isolation(store):
    store.upsert(item("a1", [1.0, 0.0], user_id="alice"))
    store.upsert(item("b1", [1.0, 0.0], user_id="bob"))
    store.upsert(item("b2", [0.9, 0.1], user_id="bob"))
    alice = store.search("alice", [1.0, 0.0], k=10)
    assert [r[0].id for r in alice] == ["a1"]
    bob = store.search("bob", [1.0, 0.0], k=10)
    assert {r[0].user_id for r in bob} == {"bob"}
    assert store.search("carol", [1.0, 0.0]) == []


def test_search_kind_filter(store):
    store.upsert(item("m", [1.0, 0.0], kind="message"))
    store.upsert(item("f", [0.5, 0.5], kind="fact"))
    store.upsert(item("s", [0.9, 0.1], kind="summary"))
    facts = store.search("alice", [1.0, 0.0], kinds=["fact"])
    assert [r[0].id for r in facts] == ["f"]
    assert all(r[0].kind == "fact" for r in facts)
    two = store.search("alice", [1.0, 0.0], kinds=("fact", "summary"))
    assert [r[0].id for r in two] == ["s", "f"]
    assert store.search("alice", [1.0, 0.0], kinds=[]) == []


def test_search_excludes_deleted(store):
    store.upsert(item("a", [1.0, 0.0]))
    store.upsert(item("b", [0.5, 0.5]))
    store.delete("a")
    assert [r[0].id for r in store.search("alice", [1.0, 0.0])] == ["b"]


def test_search_tie_break(store):
    store.upsert(item("old", [1.0, 0.0], created_at=10.0))
    store.upsert(item("zeta", [2.0, 0.0], created_at=20.0))
    store.upsert(item("alpha", [3.0, 0.0], created_at=20.0))
    store.upsert(item("new", [4.0, 0.0], created_at=30.0))
    results = store.search("alice", [1.0, 0.0])
    sims = [r[1] for r in results]
    assert all(math.isclose(s, 1.0) for s in sims)
    assert [r[0].id for r in results] == ["new", "alpha", "zeta", "old"]


@pytest.mark.parametrize("k", [0, -1])
def test_search_nonpositive_k(store, k):
    store.upsert(item("a", [1.0, 0.0]))
    assert store.search("alice", [1.0, 0.0], k=k) == []


def test_search_with_real_embedder(store):
    emb = HashingEmbedder(dim=64)
    texts = {
        "pets": "my dog loves playing fetch in the park",
        "food": "I enjoy cooking spicy thai curry",
        "work": "the quarterly report is due friday",
    }
    for key, text in texts.items():
        store.upsert(item(key, emb.embed_one(text)))
    results = store.search("alice", emb.embed_one("what does the dog like to play"), k=1)
    assert results[0][0].id == "pets"


# -- contract: listing and counting ---------------------------------------------------------


def test_listing_excludes_tombstones_by_default(store):
    store.upsert(item("a", [1.0, 0.0], created_at=1.0))
    store.upsert(item("b", [0.0, 1.0], created_at=2.0))
    store.upsert(item("c", [1.0, 1.0], created_at=3.0))
    store.delete("b")
    assert [i.id for i in store.all()] == ["a", "c"]
    everything = store.all(include_deleted=True)
    assert [i.id for i in everything] == ["a", "b", "c"]
    assert [i.deleted for i in everything] == [False, True, False]
    assert len(store) == 2
    assert store.count() == 2


def test_per_user_listing_and_count(store):
    store.upsert(item("a1", [1.0, 0.0], user_id="alice", created_at=1.0))
    store.upsert(item("a2", [1.0, 0.0], user_id="alice", created_at=2.0))
    store.upsert(item("b1", [1.0, 0.0], user_id="bob", created_at=3.0))
    store.delete("a2")
    assert [i.id for i in store.all("alice")] == ["a1"]
    assert [i.id for i in store.all("alice", include_deleted=True)] == ["a1", "a2"]
    assert [i.id for i in store.all(user_id="bob")] == ["b1"]
    assert store.count("alice") == 1
    assert store.count("bob") == 1
    assert store.count("carol") == 0
    assert store.count() == 2


def test_context_manager_returns_store(make_store):
    store = make_store()
    with store as s:
        assert s is store
        s.upsert(item("a", [1.0]))


# -- SQLite-specific -------------------------------------------------------------------------


def test_sqlite_in_memory_default():
    with SQLiteVectorStore() as store:
        assert store.path == ":memory:"
        store.upsert(item("a", [1.0, 0.0]))
        assert len(store) == 1


def test_sqlite_persistence_across_reopen(tmp_path):
    path = tmp_path / "memories.db"
    clock = FakeClock(7000.0)
    items = [
        item(
            "a",
            [0.1, -0.2, 1e-300],
            kind="fact",
            session_id="s1",
            importance=0.75,
            last_accessed=123.456,
            access_count=4,
            origin="phone",
            metadata={"tags": ["x"], "nested": {"k": 1.5}, "none": None},
        ),
        item("b", [0.3, 0.3, 0.3], user_id="bob", created_at=101.0),
        item("c", [math.pi, math.e, -1.0], kind="summary", created_at=102.0),
    ]
    store = SQLiteVectorStore(path, clock=clock)
    for it in items:
        store.upsert(it)
    assert store.delete("c") is True
    expected = {i.id: i.to_dict() for i in store.all(include_deleted=True)}
    store.close()

    reopened = SQLiteVectorStore(str(path))
    try:
        got = {i.id: i.to_dict() for i in reopened.all(include_deleted=True)}
        assert got == expected
        assert got["c"]["deleted"] is True
        assert got["c"]["updated_at"] == 7000.0
        assert got["a"]["embedding"] == [0.1, -0.2, 1e-300]
        assert got["a"]["metadata"] == {"tags": ["x"], "nested": {"k": 1.5}, "none": None}
        assert len(reopened) == 2
        assert reopened.dim == 3
        assert [r[0].id for r in reopened.search("alice", [1.0, 0.0, 0.0])] == ["a"]
    finally:
        reopened.close()


def test_sqlite_dimension_recovered_on_reopen(tmp_path):
    path = tmp_path / "dim.db"
    with SQLiteVectorStore(path) as store:
        store.upsert(item("a", [1.0, 0.0, 0.0]))
    with SQLiteVectorStore(path) as store:
        assert store.dim == 3
        with pytest.raises(ValueError):
            store.upsert(item("b", [1.0, 0.0, 0.0, 0.0]))


def test_sqlite_dimension_inferred_from_rows_without_meta(tmp_path):
    path = tmp_path / "legacy.db"
    with SQLiteVectorStore(path) as store:
        store.upsert(item("a", [1.0, 0.0, 0.0, 0.0, 0.0]))
        with store._conn:
            store._conn.execute("DELETE FROM store_meta")
    with SQLiteVectorStore(path) as store:
        assert store.dim == 5


def test_sqlite_configured_dimension_persisted_without_items(tmp_path):
    path = tmp_path / "dimonly.db"
    SQLiteVectorStore(path, dim=4).close()
    with SQLiteVectorStore(path) as store:
        assert store.dim == 4


def test_sqlite_conflicting_dimension_on_reopen(tmp_path):
    path = tmp_path / "conflict.db"
    with SQLiteVectorStore(path) as store:
        store.upsert(item("a", [1.0, 0.0]))
    with pytest.raises(ValueError):
        SQLiteVectorStore(path, dim=5)
    with SQLiteVectorStore(path, dim=2) as store:
        assert store.dim == 2


def test_sqlite_context_manager_closes(tmp_path):
    import sqlite3

    path = tmp_path / "ctx.db"
    with SQLiteVectorStore(path) as store:
        store.upsert(item("a", [1.0, 0.0]))
    with pytest.raises(sqlite3.ProgrammingError):
        store.get("a")
    with SQLiteVectorStore(path) as again:
        assert again.get("a").content == "content a"


def test_sqlite_concurrent_upserts(tmp_path):
    with SQLiteVectorStore(tmp_path / "threads.db") as store:
        errors: list[BaseException] = []

        def worker(n: int) -> None:
            try:
                for j in range(25):
                    store.upsert(item(f"t{n}-{j}", [float(n), float(j) + 1.0]))
                    store.search("alice", [1.0, 0.0], k=3)
            except BaseException as exc:  # pragma: no cover - surfaced by assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(store) == 100


def test_sqlite_creates_user_index(tmp_path):
    with SQLiteVectorStore(tmp_path / "idx.db") as store:
        rows = store._conn.execute("PRAGMA index_list('memories')").fetchall()
        assert any("user" in row[1] for row in rows)


def test_repr():
    assert "dim=None" in repr(InMemoryVectorStore())
    with SQLiteVectorStore() as store:
        assert ":memory:" in repr(store)
