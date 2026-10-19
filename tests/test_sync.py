"""Tests for cross-session sync (memory_agent.sync)."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from memory_agent.interfaces import VectorStore
from memory_agent.sync import (
    FileSyncBackend,
    InMemorySyncBackend,
    MemorySynchronizer,
    SessionInfo,
    SessionManager,
    SyncBackend,
    SyncReport,
    SyncStorageError,
    resolve_conflict,
    version_digest,
)
from memory_agent.types import MemoryItem

# --------------------------------------------------------------------------- fakes & helpers


def clone(item: MemoryItem) -> MemoryItem:
    return MemoryItem.from_dict(item.to_dict())


class FakeStore:
    """Dict-backed VectorStore honouring the protocol; stores and returns copies."""

    def __init__(self, clock: float = 0.0) -> None:
        self._items: dict[str, MemoryItem] = {}
        self.clock = clock

    def upsert(self, item: MemoryItem) -> None:
        self._items[item.id] = clone(item)

    def get(self, item_id: str) -> MemoryItem | None:
        item = self._items.get(item_id)
        return None if item is None else clone(item)

    def delete(self, item_id: str) -> bool:
        item = self._items.get(item_id)
        if item is None:
            return False
        item.deleted = True
        item.updated_at = self.clock
        return True

    def search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        k: int = 10,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        def cos(a: Sequence[float], b: Sequence[float]) -> float:
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return (
                0.0
                if not na or not nb
                else sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
            )

        hits = [
            (clone(i), cos(i.embedding, query_embedding))
            for i in self._items.values()
            if i.user_id == user_id and not i.deleted and (kinds is None or i.kind in kinds)
        ]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]

    def all(self, user_id: str | None = None, include_deleted: bool = False) -> list[MemoryItem]:
        return [
            clone(i)
            for i in self._items.values()
            if (user_id is None or i.user_id == user_id) and (include_deleted or not i.deleted)
        ]

    def __len__(self) -> int:
        return sum(1 for i in self._items.values() if not i.deleted)


def mk(
    content: str,
    at: float = 100.0,
    user: str = "alice",
    item_id: str | None = None,
    origin: str = "local",
) -> MemoryItem:
    kwargs = {} if item_id is None else {"id": item_id}
    return MemoryItem(
        user_id=user, content=content, embedding=[1.0, 0.5], created_at=at, origin=origin, **kwargs
    )


def edit(store: FakeStore, item_id: str, content: str, at: float) -> None:
    item = store.get(item_id)
    assert item is not None
    item.content = content
    item.updated_at = at
    store.upsert(item)


def snapshot(store: FakeStore, user: str = "alice") -> dict[str, dict]:
    return {i.id: i.to_dict() for i in store.all(user, include_deleted=True)}


def live(store: FakeStore, user: str = "alice") -> dict[str, str]:
    return {i.id: i.content for i in store.all(user)}


class Device:
    def __init__(self, backend: SyncBackend, device_id: str) -> None:
        self.store = FakeStore()
        self.sync = MemorySynchronizer(self.store, backend, device_id)


@pytest.fixture(params=["memory", "file"])
def backend(request, tmp_path: Path) -> SyncBackend:
    if request.param == "memory":
        return InMemorySyncBackend()
    return FileSyncBackend(tmp_path / "sync")


def test_fake_store_and_backends_satisfy_protocols(tmp_path: Path) -> None:
    assert isinstance(FakeStore(), VectorStore)
    assert isinstance(InMemorySyncBackend(), SyncBackend)
    assert isinstance(FileSyncBackend(tmp_path), SyncBackend)


# --------------------------------------------------------------------------- 1. resolve_conflict


def test_newer_remote_edit_wins() -> None:
    local = mk("old", item_id="x")
    local.updated_at = 100.0
    remote = clone(local)
    remote.content, remote.updated_at = "new", 200.0
    assert resolve_conflict(local, remote).content == "new"


def test_newer_local_edit_wins() -> None:
    local = mk("mine", item_id="x")
    local.updated_at = 300.0
    remote = clone(local)
    remote.content, remote.updated_at = "theirs", 200.0
    assert resolve_conflict(local, remote).content == "mine"


def test_resolving_different_ids_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_conflict(mk("a", item_id="x"), mk("b", item_id="y"))


def test_equal_timestamps_resolved_by_origin_symmetrically() -> None:
    a = mk("from a", item_id="x", origin="device-a")
    b = mk("from b", item_id="x", origin="device-b")
    assert a.updated_at == b.updated_at
    assert resolve_conflict(a, b).content == "from b"
    assert resolve_conflict(b, a).content == "from b"


def test_fully_tied_versions_resolve_identically() -> None:
    a = mk("alpha", item_id="x", origin="dev")
    b = mk("beta", item_id="x", origin="dev")
    ab, ba = resolve_conflict(a, b), resolve_conflict(b, a)
    assert ab.to_dict() == ba.to_dict()
    assert ab.content in {"alpha", "beta"}


def test_access_stats_merge_without_affecting_winner() -> None:
    older = mk("older", item_id="x")
    older.updated_at, older.access_count, older.last_accessed = 100.0, 7, 500.0
    newer = clone(older)
    newer.content, newer.updated_at, newer.access_count, newer.last_accessed = (
        "newer",
        150.0,
        2,
        150.0,
    )
    for merged in (resolve_conflict(older, newer), resolve_conflict(newer, older)):
        assert merged.content == "newer"
        assert merged.access_count == 7
        assert merged.last_accessed == 500.0


def test_resolve_does_not_mutate_inputs() -> None:
    a = mk("a", item_id="x")
    b = clone(a)
    b.updated_at, b.access_count = 200.0, 3
    before = (a.to_dict(), b.to_dict())
    merged = resolve_conflict(a, b)
    merged.embedding.append(9.0)
    assert (a.to_dict(), b.to_dict()) == before


def test_version_digest_ignores_access_stats() -> None:
    a = mk("a", item_id="x")
    b = clone(a)
    b.touch(at=999.0)
    assert version_digest(a) == version_digest(b)
    b.content = "changed"
    assert version_digest(a) != version_digest(b)


def test_newer_delete_beats_older_edit() -> None:
    edit_v = mk("edited", item_id="x")
    edit_v.updated_at = 150.0
    tomb = clone(edit_v)
    tomb.deleted, tomb.updated_at = True, 200.0
    assert resolve_conflict(edit_v, tomb).deleted
    assert resolve_conflict(tomb, edit_v).deleted


def test_newer_edit_beats_older_delete() -> None:
    tomb = mk("gone", item_id="x")
    tomb.deleted, tomb.updated_at = True, 200.0
    edit_v = clone(tomb)
    edit_v.deleted, edit_v.content, edit_v.updated_at = False, "revived", 250.0
    merged = resolve_conflict(tomb, edit_v)
    assert not merged.deleted
    assert merged.content == "revived"


# --------------------------------------------------------------------------- 2. backends


def test_push_new_records_advances_revision(backend: SyncBackend) -> None:
    assert backend.pull("alice") == ([], 0)
    rev = backend.push("alice", [mk("hello", item_id="x")])
    assert rev > 0


def test_push_unchanged_records_keeps_revision(backend: SyncBackend) -> None:
    items = [mk("a", item_id="x"), mk("b", item_id="y")]
    rev1 = backend.push("alice", items)
    rev2 = backend.push("alice", [clone(i) for i in items])
    assert rev2 == rev1
    assert backend.push("alice", []) == rev1


def test_push_stale_version_keeps_newer_record(backend: SyncBackend) -> None:
    newer = mk("newer", item_id="x")
    newer.updated_at = 300.0
    stale = clone(newer)
    stale.content, stale.updated_at = "stale", 100.0
    rev = backend.push("alice", [newer])
    # The losing push re-announces the stored winner under a new revision...
    assert backend.push("alice", [stale]) == rev + 1
    items, latest = backend.pull("alice", rev)
    assert [i.content for i in items] == ["newer"]
    assert latest == rev + 1
    # ...but re-pushing the stored version itself changes nothing.
    assert backend.push("alice", [newer]) == rev + 1


def test_device_with_losing_edit_adopts_winner(backend: SyncBackend) -> None:
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("original", item_id="x", at=100.0))
    a.sync.sync("alice")
    b.sync.sync("alice")
    edit(a.store, "x", "A's newer edit", 300.0)
    a.sync.sync("alice")
    edit(b.store, "x", "B's edit with a slow clock", 200.0)
    b.sync.sync("alice")  # learns of A's edit while dirty -> keeps the winner
    assert b.store.get("x").content == "A's newer edit"
    # Now B edits again, but its clock is behind the backend's version: the push loses.
    edit(b.store, "x", "B stale again", 250.0)
    report = b.sync.sync("alice")
    assert report.pushed == 1 and report.pulled == 1
    assert b.store.get("x").content == "A's newer edit"
    a.sync.sync("alice")
    assert snapshot(a.store) == snapshot(b.store)
    assert b.sync.sync("alice").pushed == 0


def test_push_raises_access_stats_of_existing_version(backend: SyncBackend) -> None:
    item = mk("a", item_id="x")
    rev = backend.push("alice", [item])
    touched = clone(item)
    touched.touch(at=999.0)
    assert backend.push("alice", [touched]) == rev + 1
    items, _ = backend.pull("alice", rev)
    assert items[0].access_count == 1 and items[0].last_accessed == 999.0


def test_push_other_users_record_rejected_atomically(backend: SyncBackend) -> None:
    with pytest.raises(ValueError):
        backend.push("alice", [mk("ok", item_id="x"), mk("bob's", user="bob", item_id="y")])
    assert backend.pull("alice") == ([], 0)
    assert backend.pull("bob") == ([], 0)


def test_empty_user_id_rejected(backend: SyncBackend) -> None:
    with pytest.raises(ValueError):
        backend.pull("")


def test_incremental_pull(backend: SyncBackend) -> None:
    r1 = backend.push("alice", [mk("first", item_id="a")])
    r2 = backend.push("alice", [mk("second", item_id="b")])
    assert (r1, r2) == (1, 2)
    items, latest = backend.pull("alice", since_revision=1)
    assert [i.id for i in items] == ["b"]
    assert latest == 2
    items, _ = backend.pull("alice", since_revision=0)
    assert [i.id for i in items] == ["a", "b"]
    assert backend.pull("alice", since_revision=2) == ([], 2)


def test_unknown_user_pulls_nothing(backend: SyncBackend) -> None:
    backend.push("alice", [mk("x")])
    assert backend.pull("nobody", 0) == ([], 0)


def test_pulled_records_are_isolated_copies(backend: SyncBackend) -> None:
    pushed = mk("orig", item_id="x")
    backend.push("alice", [pushed])
    pushed.content = "mutated after push"
    items, _ = backend.pull("alice")
    items[0].content = "mutated after pull"
    items[0].embedding.append(42.0)
    again, _ = backend.pull("alice")
    assert again[0].content == "orig"
    assert again[0].embedding == [1.0, 0.5]


def test_file_backend_survives_reinstantiation(tmp_path: Path) -> None:
    first = FileSyncBackend(tmp_path)
    first.push("alice", [mk("one", item_id="a")])
    rev = first.push("alice", [mk("two", item_id="b")])
    second = FileSyncBackend(tmp_path)
    items, latest = second.pull("alice")
    assert latest == rev
    assert {i.id: i.content for i in items} == {"a": "one", "b": "two"}
    assert second.push("alice", items) == rev  # identical content is not a change
    assert not list(tmp_path.glob("*.tmp"))  # atomic writes leave no temp files behind


@pytest.mark.parametrize("user_id", ["../../etc/passwd", "/abs/path", "..", "a\\b", "üser .x"])
def test_file_backend_paths_stay_inside_directory(tmp_path: Path, user_id: str) -> None:
    root = tmp_path / "root"
    backend = FileSyncBackend(root)
    backend.push(user_id, [mk("secret", user=user_id)])
    backend.put_session(SessionInfo("s1", user_id, "dev", 1.0))
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written
    assert all(p.parent == root for p in written)
    assert backend.log_path(user_id).parent == root
    assert ".." not in backend.log_path(user_id).name.replace(".json", "")
    assert backend.pull(user_id)[0][0].content == "secret"


def test_file_backend_similar_ids_do_not_collide(tmp_path: Path) -> None:
    backend = FileSyncBackend(tmp_path)
    backend.push("a/b", [mk("slash", user="a/b")])
    backend.push("a_b", [mk("underscore", user="a_b")])
    assert backend.log_path("a/b") != backend.log_path("a_b")
    assert [i.content for i in backend.pull("a/b")[0]] == ["slash"]
    assert [i.content for i in backend.pull("a_b")[0]] == ["underscore"]


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        "[]",
        json.dumps({"user_id": "someone-else", "revision": 1, "records": []}),
        json.dumps({"user_id": "alice", "revision": "1", "records": []}),
        json.dumps({"user_id": "alice", "revision": 1, "records": [{"rev": 1, "item": {}}]}),
    ],
)
def test_corrupt_file_raises_and_is_untouched(tmp_path: Path, contents: str) -> None:
    backend = FileSyncBackend(tmp_path)
    path = backend.log_path("alice")
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(SyncStorageError) as excinfo:
        backend.pull("alice")
    assert str(path) in str(excinfo.value)
    with pytest.raises(SyncStorageError):
        backend.push("alice", [mk("new")])
    assert path.read_text(encoding="utf-8") == contents


def test_corrupt_sessions_file_raises(tmp_path: Path) -> None:
    backend = FileSyncBackend(tmp_path)
    path = backend.sessions_path("alice")
    path.write_text(json.dumps({"user_id": "alice", "sessions": [{"nope": 1}]}), "utf-8")
    with pytest.raises(SyncStorageError):
        backend.list_sessions("alice")
    with pytest.raises(SyncStorageError):
        backend.put_session(SessionInfo("s", "alice", "d", 1.0))


def test_failed_write_removes_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FileSyncBackend(tmp_path)
    backend.push("alice", [mk("kept", item_id="a")])

    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        backend.push("alice", [mk("lost", item_id="b")])
    monkeypatch.undo()
    assert not list(tmp_path.glob("*.tmp"))
    assert [i.content for i in FileSyncBackend(tmp_path).pull("alice")[0]] == ["kept"]


# --------------------------------------------------------------------------- 3. synchronizer


def test_two_devices_converge(backend: SyncBackend) -> None:
    a, b = Device(backend, "laptop"), Device(backend, "phone")
    for i in range(3):
        a.store.upsert(mk(f"a{i}", at=100.0 + i))
        b.store.upsert(mk(f"b{i}", at=200.0 + i))
    ra = a.sync.sync("alice")
    rb = b.sync.sync("alice")
    a.sync.sync("alice")
    assert (ra.pushed, rb.pulled, rb.pushed) == (3, 3, 3)
    assert snapshot(a.store) == snapshot(b.store)
    assert sorted(live(a.store).values()) == ["a0", "a1", "a2", "b0", "b1", "b2"]


def test_three_devices_converge(backend: SyncBackend) -> None:
    devices = [Device(backend, d) for d in ("a", "b", "c")]
    for n, dev in enumerate(devices):
        dev.store.upsert(mk(f"from {dev.sync.device_id}", at=float(n)))
    shared = mk("shared", item_id="s", at=1.0)
    devices[0].store.upsert(shared)
    for _ in range(2):
        for dev in devices:
            dev.sync.sync("alice")
    edit(devices[1].store, "s", "edited by b", 50.0)
    devices[2].store.clock = 60.0
    devices[2].store.delete(next(iter(i.id for i in devices[2].store.all() if i.id != "s")))
    for _ in range(2):
        for dev in devices:
            dev.sync.sync("alice")
    snaps = [snapshot(d.store) for d in devices]
    assert snaps[0] == snaps[1] == snaps[2]
    assert live(devices[0].store)["s"] == "edited by b"
    assert len(devices[0].store) == 3


def test_second_sync_is_noop(backend: SyncBackend) -> None:
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("x"))
    b.store.upsert(mk("y"))
    a.sync.sync("alice")
    b.sync.sync("alice")
    first = a.sync.sync("alice")
    assert first.pulled == 1
    again = a.sync.sync("alice")
    assert again == SyncReport(pushed=0, pulled=0, conflicts=0, revision=first.revision)
    rev_before = backend.pull("alice", 0)[1]
    assert b.sync.sync("alice") == SyncReport(0, 0, 0, rev_before)
    assert backend.pull("alice", 0)[1] == rev_before


def test_touch_alone_does_not_push() -> None:
    backend = InMemorySyncBackend()
    a = Device(backend, "a")
    a.store.upsert(mk("x", item_id="x"))
    rev = a.sync.sync("alice").revision
    item = a.store.get("x")
    item.touch(at=999.0)
    a.store.upsert(item)
    assert a.sync.sync("alice") == SyncReport(0, 0, 0, rev)


def test_deletion_propagates_and_is_not_resurrected(backend: SyncBackend) -> None:
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("doomed", item_id="x"))
    a.sync.sync("alice")
    b.sync.sync("alice")
    a.store.clock = 500.0
    assert a.store.delete("x")
    assert a.sync.sync("alice").pushed == 1
    report = b.sync.sync("alice")
    assert report.pulled == 1
    assert b.store.get("x").deleted
    assert live(b.store) == {}
    # A stale copy pushed later (older updated_at) must not bring it back.
    stale = mk("doomed", item_id="x", origin="a")
    backend.push("alice", [stale])
    b.sync.sync("alice")
    assert b.store.get("x").deleted


def test_remote_tombstone_for_unknown_item_is_stored() -> None:
    backend = InMemorySyncBackend()
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("brief", item_id="x"))
    a.store.clock = 200.0
    a.store.delete("x")
    a.sync.sync("alice")
    b.sync.sync("alice")
    assert b.store.get("x").deleted
    assert len(b.store) == 0


def test_concurrent_edits_converge_to_lww_winner(backend: SyncBackend) -> None:
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("original", item_id="x"))
    a.sync.sync("alice")
    b.sync.sync("alice")
    edit(a.store, "x", "A's edit", 200.0)
    edit(b.store, "x", "B's edit", 300.0)
    ra = a.sync.sync("alice")
    rb = b.sync.sync("alice")
    ra2 = a.sync.sync("alice")
    assert ra.conflicts == 0
    assert rb.conflicts == 1
    assert ra2.conflicts == 0 and ra2.pulled == 1
    assert a.store.get("x").content == b.store.get("x").content == "B's edit"
    assert snapshot(a.store) == snapshot(b.store)


def test_concurrent_edit_where_local_wins() -> None:
    backend = InMemorySyncBackend()
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("original", item_id="x"))
    a.sync.sync("alice")
    b.sync.sync("alice")
    edit(a.store, "x", "A late", 400.0)
    edit(b.store, "x", "B early", 300.0)
    b.sync.sync("alice")
    report = a.sync.sync("alice")
    assert report.conflicts == 1
    assert report.pushed == 1
    b.sync.sync("alice")
    assert a.store.get("x").content == b.store.get("x").content == "A late"


def test_users_are_isolated(backend: SyncBackend) -> None:
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("alice fact", user="alice"))
    a.store.upsert(mk("bob fact", user="bob"))
    report = a.sync.sync("alice")
    assert report.pushed == 1
    assert backend.pull("bob") == ([], 0)
    assert [i.origin for i in a.store.all("bob")] == ["local"]  # untouched
    b.store.upsert(mk("bob on b", user="bob"))
    b.sync.sync("bob")
    assert live(b.store, "alice") == {}
    assert sorted(live(b.store, "bob").values()) == ["bob on b"]
    assert [i.content for i in backend.pull("bob")[0]] == ["bob on b"]


def test_local_origin_is_stamped_with_device_id(backend: SyncBackend) -> None:
    dev = Device(backend, "phone")
    dev.store.upsert(mk("hi", item_id="x"))
    dev.store.upsert(mk("pre-stamped", item_id="y", origin="tablet"))
    dev.sync.sync("alice")
    remote = {i.id: i.origin for i in backend.pull("alice")[0]}
    assert remote == {"x": "phone", "y": "tablet"}
    assert dev.store.get("x").origin == "phone"
    assert dev.store.get("y").origin == "tablet"


@pytest.mark.parametrize("device_id", ["", "local"])
def test_invalid_device_id_rejected(device_id: str) -> None:
    with pytest.raises(ValueError):
        MemorySynchronizer(FakeStore(), InMemorySyncBackend(), device_id)


def test_restored_synchronizer_resumes_without_resending() -> None:
    backend = InMemorySyncBackend()
    a, b = Device(backend, "a"), Device(backend, "b")
    a.store.upsert(mk("x"))
    b.store.upsert(mk("y"))
    a.sync.sync("alice")
    b.sync.sync("alice")
    a.sync.sync("alice")
    saved = json.loads(json.dumps(a.sync.state()))
    restored = MemorySynchronizer(a.store, backend, "a")
    restored.load_state(saved)
    rev = backend.pull("alice")[1]
    assert restored.sync("alice") == SyncReport(0, 0, 0, rev)
    assert restored.state() == saved
    # Without restored state the device would re-push everything.
    fresh = MemorySynchronizer(a.store, backend, "a")
    assert fresh.sync("alice").pushed == 0  # versions match the backend so nothing changes...
    assert backend.pull("alice")[1] == rev  # ...and the revision is not bumped


def test_state_from_other_device_rejected() -> None:
    backend = InMemorySyncBackend()
    a = Device(backend, "a")
    a.store.upsert(mk("x"))
    a.sync.sync("alice")
    other = MemorySynchronizer(FakeStore(), backend, "b")
    with pytest.raises(ValueError):
        other.load_state(a.sync.state())


def test_recovers_from_backend_reset() -> None:
    old = InMemorySyncBackend()
    dev = Device(old, "a")
    for i in range(3):
        dev.store.upsert(mk(f"m{i}", at=float(i)))
    dev.sync.sync("alice")
    dev.sync.sync("alice")
    new = InMemorySyncBackend()
    dev.sync.backend = new
    report = dev.sync.sync("alice")
    assert report.pushed == 3
    assert sorted(i.content for i in new.pull("alice")[0]) == ["m0", "m1", "m2"]
    assert report.revision == new.pull("alice")[1]


def test_concurrent_push_between_pull_and_push_is_not_skipped() -> None:
    """A record pushed by another device mid-sync is collected by the echo pull."""
    backend = InMemorySyncBackend()
    a = Device(backend, "a")
    a.store.upsert(mk("from a", item_id="a1"))

    class Interleaving:
        def __init__(self, inner: InMemorySyncBackend) -> None:
            self.inner = inner
            self.fired = False

        def push(self, user_id, items):
            if not self.fired:
                self.fired = True
                self.inner.push(user_id, [mk("from b", item_id="b1", origin="b")])
            return self.inner.push(user_id, items)

        def pull(self, user_id, since_revision=0):
            return self.inner.pull(user_id, since_revision)

        def put_session(self, info):
            self.inner.put_session(info)

        def list_sessions(self, user_id):
            return self.inner.list_sessions(user_id)

    a.sync.backend = Interleaving(backend)
    report = a.sync.sync("alice")
    assert report.pushed == 1 and report.pulled == 1
    assert sorted(live(a.store).values()) == ["from a", "from b"]
    assert a.sync.sync("alice") == SyncReport(0, 0, 0, backend.pull("alice")[1])


# --------------------------------------------------------------------------- 4. sessions


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_start_session_records_in_backend(backend: SyncBackend) -> None:
    mgr = SessionManager(backend, "laptop", clock=Clock(1000.0))
    info = mgr.start_session("alice")
    assert info.is_active
    assert (info.started_at, info.device_id, info.message_count) == (1000.0, "laptop", 0)
    assert [s.session_id for s in backend.list_sessions("alice")] == [info.session_id]


def test_session_ids_are_unique() -> None:
    mgr = SessionManager(InMemorySyncBackend(), "d", clock=Clock())
    assert mgr.start_session("alice").session_id != mgr.start_session("alice").session_id


def test_record_message_counts_and_persists(backend: SyncBackend) -> None:
    mgr = SessionManager(backend, "d", clock=Clock())
    sid = mgr.start_session("alice").session_id
    assert [mgr.record_message(sid) for _ in range(3)] == [1, 2, 3]
    assert mgr.get(sid).message_count == 3
    assert backend.list_sessions("alice")[0].message_count == 3


def test_end_session_with_summary_and_topics(backend: SyncBackend) -> None:
    clock = Clock(1000.0)
    mgr = SessionManager(backend, "d", clock=clock)
    sid = mgr.start_session("alice").session_id
    clock.t = 2000.0
    ended = mgr.end_session(sid, summary="Planned a trip", topics=("travel",))
    assert (ended.ended_at, ended.summary, ended.topics) == (2000.0, "Planned a trip", ["travel"])
    stored = backend.list_sessions("alice")[0]
    assert (stored.ended_at, stored.summary, stored.topics) == (
        2000.0,
        "Planned a trip",
        ["travel"],
    )
    assert not stored.is_active


def test_ending_twice_or_recording_after_end_rejected() -> None:
    mgr = SessionManager(InMemorySyncBackend(), "d", clock=Clock())
    sid = mgr.start_session("alice").session_id
    mgr.end_session(sid)
    with pytest.raises(ValueError):
        mgr.end_session(sid)
    with pytest.raises(ValueError):
        mgr.record_message(sid)


def test_unknown_session_raises_key_error() -> None:
    mgr = SessionManager(InMemorySyncBackend(), "d", clock=Clock())
    with pytest.raises(KeyError):
        mgr.get("nope")
    with pytest.raises(KeyError):
        mgr.record_message("nope")
    with pytest.raises(KeyError):
        mgr.end_session("nope")


def test_returned_sessions_are_copies() -> None:
    mgr = SessionManager(InMemorySyncBackend(), "d", clock=Clock())
    info = mgr.start_session("alice")
    info.message_count = 99
    assert mgr.get(info.session_id).message_count == 0


def test_cross_device_listing_newest_first(backend: SyncBackend) -> None:
    a = SessionManager(backend, "A", clock=Clock(100.0))
    b = SessionManager(backend, "B", clock=Clock(200.0))
    sa = a.start_session("alice").session_id
    sb = b.start_session("alice").session_id
    for mgr in (a, b):
        assert [s.session_id for s in mgr.list_sessions("alice")] == [sb, sa]
    # Sessions seen through listing become manageable (e.g. after an app restart).
    a.end_session(sb, summary="closed from A")
    assert backend.list_sessions("alice")[0].summary == "closed from A"


def test_listing_excludes_other_users(backend: SyncBackend) -> None:
    mgr = SessionManager(backend, "d", clock=Clock())
    mgr.start_session("alice")
    bob = mgr.start_session("bob").session_id
    assert bob not in {s.session_id for s in mgr.list_sessions("alice")}
    assert [s.user_id for s in mgr.list_sessions("alice")] == ["alice"]


def test_last_session_greeting_from_another_device(backend: SyncBackend) -> None:
    clock_a, clock_b = Clock(100.0), Clock(500.0)
    a = SessionManager(backend, "A", clock=clock_a)
    b = SessionManager(backend, "B", clock=clock_b)
    old = a.start_session("alice").session_id
    clock_a.t = 150.0
    a.end_session(old, summary="Talked about Python")
    s = a.start_session("alice").session_id
    clock_a.t = 300.0
    a.end_session(s, summary="Talked about Rust", topics=["rust"])
    new = b.start_session("alice").session_id
    last = b.last_session("alice", exclude=new)
    assert last is not None
    assert (last.session_id, last.summary, last.device_id) == (s, "Talked about Rust", "A")
    assert b.last_session("alice", exclude=s).session_id == old


def test_last_session_ignores_active_sessions() -> None:
    mgr = SessionManager(InMemorySyncBackend(), "d", clock=Clock())
    assert mgr.last_session("alice") is None
    mgr.start_session("alice")
    assert mgr.last_session("alice") is None


def test_session_info_round_trip() -> None:
    info = SessionInfo("s1", "alice", "laptop", 1.5, 9.25, "summary", 4, ["a", "b"])
    restored = SessionInfo.from_dict(json.loads(json.dumps(info.to_dict())))
    assert restored == info
    minimal = SessionInfo.from_dict(
        {"session_id": "s", "user_id": "u", "device_id": "d", "started_at": 1}
    )
    assert (minimal.ended_at, minimal.summary, minimal.message_count, minimal.topics) == (
        None,
        "",
        0,
        [],
    )


def test_sessions_survive_file_backend_restart(tmp_path: Path) -> None:
    clock = Clock(10.0)
    mgr = SessionManager(FileSyncBackend(tmp_path), "d", clock=clock)
    sid = mgr.start_session("alice").session_id
    clock.t = 20.0
    mgr.end_session(sid, summary="Discussed sync", topics=["sync"])
    sessions = FileSyncBackend(tmp_path).list_sessions("alice")
    assert [(s.session_id, s.summary, s.ended_at) for s in sessions] == [
        (sid, "Discussed sync", 20.0)
    ]


def test_session_manager_requires_device_id() -> None:
    with pytest.raises(ValueError):
        SessionManager(InMemorySyncBackend(), "")
