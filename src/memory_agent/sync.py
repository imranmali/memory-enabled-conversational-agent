"""Cross-session / cross-device memory synchronisation and session tracking.

The pieces fit together like this:

* :func:`resolve_conflict` decides between two versions of the same :class:`MemoryItem`
  (last-write-wins on ``updated_at``, deterministic tie-breaks, access statistics merged).
* A :class:`SyncBackend` is the shared rendezvous point.  It assigns a monotonically increasing
  per-user *revision* to every change, so change detection never depends on device clocks.
  :class:`InMemorySyncBackend` and :class:`FileSyncBackend` are reference implementations.
* :class:`MemorySynchronizer` replicates one device's :class:`~memory_agent.interfaces.VectorStore`
  with a backend: pull remote changes since its cursor, merge, push local changes.
* :class:`SessionManager` records conversation sessions in the backend so any device can
  greet the user with "Last time we talked about ...".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .interfaces import VectorStore
from .types import MemoryItem, new_id

__all__ = [
    "FileSyncBackend",
    "InMemorySyncBackend",
    "MemorySynchronizer",
    "SessionInfo",
    "SessionManager",
    "SyncBackend",
    "SyncReport",
    "SyncStorageError",
    "resolve_conflict",
    "version_digest",
]

LOCAL_ORIGIN = "local"
_ACCESS_FIELDS = ("access_count", "last_accessed")
_FORMAT_VERSION = 1


class SyncStorageError(RuntimeError):
    """Raised when persisted sync data is unreadable or structurally invalid."""


# --------------------------------------------------------------------------- conflict resolution


def _clone(item: MemoryItem) -> MemoryItem:
    """Deep, independent copy of a memory item."""
    return MemoryItem.from_dict(item.to_dict())


def version_digest(item: MemoryItem) -> str:
    """Digest identifying an item's *version*: every field except the access statistics.

    Two items with equal digests are the same version; touching an item (``access_count`` /
    ``last_accessed``) does not create a new version.
    """
    data = item.to_dict()
    for name in _ACCESS_FIELDS:
        data.pop(name, None)
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _order_key(item: MemoryItem) -> tuple[float, str, str]:
    updated = item.updated_at if item.updated_at is not None else item.created_at
    return (float(updated), item.origin, version_digest(item))


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def resolve_conflict(local: MemoryItem, remote: MemoryItem) -> MemoryItem:
    """Merge two versions of the same memory into a new item.

    Content comes from the version with the later ``updated_at``; ties go to the larger
    ``origin`` and then to the larger version digest, so ``resolve_conflict(a, b)`` and
    ``resolve_conflict(b, a)`` always agree.  Tombstones (``deleted=True``) are ordinary
    versions.  The result carries the maximum ``access_count`` and ``last_accessed`` of both
    sides.  Neither input is modified.
    """
    if local.id != remote.id:
        raise ValueError(f"cannot resolve different memories: {local.id!r} vs {remote.id!r}")
    winner = remote if _order_key(remote) > _order_key(local) else local
    merged = _clone(winner)
    merged.access_count = max(local.access_count, remote.access_count)
    merged.last_accessed = _max_optional(local.last_accessed, remote.last_accessed)
    return merged


# --------------------------------------------------------------------------- sessions


@dataclass
class SessionInfo:
    """One conversation session as recorded in the sync backend."""

    session_id: str
    user_id: str
    device_id: str
    started_at: float
    ended_at: float | None = None
    summary: str = ""
    message_count: int = 0
    topics: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": self.summary,
            "message_count": self.message_count,
            "topics": list(self.topics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionInfo:
        ended = data.get("ended_at")
        return cls(
            session_id=str(data["session_id"]),
            user_id=str(data["user_id"]),
            device_id=str(data["device_id"]),
            started_at=float(data["started_at"]),
            ended_at=None if ended is None else float(ended),
            summary=str(data.get("summary", "")),
            message_count=int(data.get("message_count", 0)),
            topics=[str(t) for t in data.get("topics", [])],
        )

    def copy(self) -> SessionInfo:
        return SessionInfo.from_dict(self.to_dict())


def _session_sort_key(info: SessionInfo) -> tuple[float, str]:
    return (info.started_at, info.session_id)


# --------------------------------------------------------------------------- backends


@runtime_checkable
class SyncBackend(Protocol):
    """Shared store that devices synchronise through.

    The backend assigns a per-user revision that starts at 0 and only ever increases.
    """

    def push(self, user_id: str, items: Sequence[MemoryItem]) -> int:
        """Merge ``items`` (LWW) into the user's records; return the latest revision.

        The revision advances only if a stored record changed or a pushed version that differs
        from the stored one lost (the winner is then re-announced so the pusher pulls it).
        Pushing records identical to the stored versions never advances it.
        """

    def pull(self, user_id: str, since_revision: int = 0) -> tuple[list[MemoryItem], int]:
        """Records changed after ``since_revision`` (oldest first) and the latest revision."""

    def put_session(self, info: SessionInfo) -> None:
        """Insert or replace a session record."""

    def list_sessions(self, user_id: str) -> list[SessionInfo]:
        """All session records of ``user_id``, newest start first."""


def _check_user_id(user_id: str) -> None:
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id must be a non-empty string")


@dataclass
class _UserLog:
    revision: int = 0
    # item id -> (revision that last changed it, stored item)
    records: dict[str, tuple[int, MemoryItem]] = field(default_factory=dict)


class _BaseSyncBackend:
    """Revision / merge semantics shared by the reference backends."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    # storage hooks -----------------------------------------------------------------------
    def _load_log(self, user_id: str) -> _UserLog:  # pragma: no cover - abstract
        raise NotImplementedError

    def _save_log(self, user_id: str, log: _UserLog) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _load_sessions(self, user_id: str) -> dict[str, SessionInfo]:  # pragma: no cover
        raise NotImplementedError

    def _save_sessions(
        self, user_id: str, sessions: dict[str, SessionInfo]
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    # protocol ----------------------------------------------------------------------------
    def push(self, user_id: str, items: Sequence[MemoryItem]) -> int:
        _check_user_id(user_id)
        items = list(items)
        for item in items:
            if item.user_id != user_id:
                raise ValueError(
                    f"item {item.id!r} belongs to user {item.user_id!r}, not {user_id!r}"
                )
        with self._lock:
            log = self._load_log(user_id)
            changed: dict[str, MemoryItem] = {}
            for item in items:
                current = changed.get(item.id)
                if current is None and item.id in log.records:
                    current = log.records[item.id][1]
                if current is None:
                    changed[item.id] = _clone(item)
                    continue
                merged = resolve_conflict(current, item)
                # Re-announce the stored winner when a *different* pushed version lost, so the
                # pushing device pulls it back instead of keeping its losing copy forever.
                lost = version_digest(item) != version_digest(current)
                if lost or merged.to_dict() != current.to_dict():
                    changed[item.id] = merged
            if changed:
                log.revision += 1
                for item_id, merged in changed.items():
                    log.records[item_id] = (log.revision, merged)
                self._save_log(user_id, log)
            return log.revision

    def pull(self, user_id: str, since_revision: int = 0) -> tuple[list[MemoryItem], int]:
        _check_user_id(user_id)
        with self._lock:
            log = self._load_log(user_id)
            rows = sorted(
                (rev, item_id, item)
                for item_id, (rev, item) in log.records.items()
                if rev > since_revision
            )
            return [_clone(item) for _, _, item in rows], log.revision

    def put_session(self, info: SessionInfo) -> None:
        _check_user_id(info.user_id)
        with self._lock:
            sessions = self._load_sessions(info.user_id)
            sessions[info.session_id] = info.copy()
            self._save_sessions(info.user_id, sessions)

    def list_sessions(self, user_id: str) -> list[SessionInfo]:
        _check_user_id(user_id)
        with self._lock:
            sessions = self._load_sessions(user_id)
            ordered = sorted(sessions.values(), key=_session_sort_key, reverse=True)
            return [s.copy() for s in ordered]


class InMemorySyncBackend(_BaseSyncBackend):
    """Process-local backend; handy for tests and for several agents in one process."""

    def __init__(self) -> None:
        super().__init__()
        self._logs: dict[str, _UserLog] = {}
        self._sessions: dict[str, dict[str, SessionInfo]] = {}

    def _load_log(self, user_id: str) -> _UserLog:
        return self._logs.setdefault(user_id, _UserLog())

    def _save_log(self, user_id: str, log: _UserLog) -> None:
        self._logs[user_id] = log

    def _load_sessions(self, user_id: str) -> dict[str, SessionInfo]:
        return self._sessions.setdefault(user_id, {})

    def _save_sessions(self, user_id: str, sessions: dict[str, SessionInfo]) -> None:
        self._sessions[user_id] = sessions


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _user_file_key(user_id: str) -> str:
    """File-name-safe key: readable slug plus a hash so distinct ids never collide."""
    slug = _UNSAFE_CHARS.sub("_", user_id).strip("_")[:40] or "user"
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}"


class FileSyncBackend(_BaseSyncBackend):
    """Backend persisted as JSON files in a directory (e.g. a synced folder or network share).

    Per user it writes ``<key>.json`` (revision and memory records) and ``<key>.sessions.json``
    (session records), where ``<key>`` is derived from the user id so it can never escape the
    directory.  Writes are atomic (temp file + ``os.replace``).  Unreadable files raise
    :class:`SyncStorageError` and are never overwritten.  One writer process per directory is
    assumed; threads within a process are serialised.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # paths & raw io ----------------------------------------------------------------------
    def log_path(self, user_id: str) -> Path:
        return self.directory / f"{_user_file_key(user_id)}.json"

    def sessions_path(self, user_id: str) -> Path:
        return self.directory / f"{_user_file_key(user_id)}.sessions.json"

    def _read_json(self, path: Path, user_id: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SyncStorageError(f"cannot read sync file {path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("user_id") != user_id:
            raise SyncStorageError(f"sync file {path} is malformed or belongs to another user")
        return data

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # hooks -------------------------------------------------------------------------------
    def _load_log(self, user_id: str) -> _UserLog:
        path = self.log_path(user_id)
        data = self._read_json(path, user_id)
        if data is None:
            return _UserLog()
        try:
            revision = data["revision"]
            if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
                raise ValueError("revision must be a non-negative integer")
            log = _UserLog(revision=revision)
            for row in data["records"]:
                rev = row["rev"]
                if not isinstance(rev, int) or not 0 < rev <= revision:
                    raise ValueError(f"invalid record revision {rev!r}")
                item = MemoryItem.from_dict(dict(row["item"]))
                if item.user_id != user_id:
                    raise ValueError(f"record {item.id!r} belongs to another user")
                log.records[item.id] = (rev, item)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise SyncStorageError(f"sync file {path} is malformed: {exc}") from exc
        return log

    def _save_log(self, user_id: str, log: _UserLog) -> None:
        records = [
            {"rev": rev, "item": item.to_dict()} for _, (rev, item) in sorted(log.records.items())
        ]
        self._write_json(
            self.log_path(user_id),
            {
                "format": _FORMAT_VERSION,
                "user_id": user_id,
                "revision": log.revision,
                "records": records,
            },
        )

    def _load_sessions(self, user_id: str) -> dict[str, SessionInfo]:
        path = self.sessions_path(user_id)
        data = self._read_json(path, user_id)
        if data is None:
            return {}
        try:
            sessions = [SessionInfo.from_dict(row) for row in data["sessions"]]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise SyncStorageError(f"sync file {path} is malformed: {exc}") from exc
        return {s.session_id: s for s in sessions}

    def _save_sessions(self, user_id: str, sessions: dict[str, SessionInfo]) -> None:
        rows = [sessions[sid].to_dict() for sid in sorted(sessions)]
        self._write_json(
            self.sessions_path(user_id),
            {"format": _FORMAT_VERSION, "user_id": user_id, "sessions": rows},
        )


# --------------------------------------------------------------------------- synchronizer


@dataclass
class SyncReport:
    """Outcome of one :meth:`MemorySynchronizer.sync` call."""

    pushed: int = 0  # local records sent to the backend
    pulled: int = 0  # remote records that changed the local store
    conflicts: int = 0  # memories changed concurrently on this device and remotely
    revision: int = 0  # backend revision the device is now in sync with


@dataclass
class _UserCursor:
    cursor: int = 0
    known: dict[str, str] = field(default_factory=dict)  # item id -> version digest exchanged


class MemorySynchronizer:
    """Replicates one device's vector store with a :class:`SyncBackend`.

    Persist :meth:`state` between runs (and restore it with :meth:`load_state`) so a restarted
    app neither re-pushes nor re-pulls everything.
    """

    def __init__(self, store: VectorStore, backend: SyncBackend, device_id: str) -> None:
        if not isinstance(device_id, str) or not device_id or device_id == LOCAL_ORIGIN:
            raise ValueError(f"device_id must be a non-empty string other than {LOCAL_ORIGIN!r}")
        self.store = store
        self.backend = backend
        self.device_id = device_id
        self._users: dict[str, _UserCursor] = {}

    # state -------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of the per-user cursors."""
        return {
            "device_id": self.device_id,
            "users": {
                uid: {"cursor": st.cursor, "known": dict(st.known)}
                for uid, st in sorted(self._users.items())
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state` on this same device."""
        if state.get("device_id") != self.device_id:
            raise ValueError(
                f"sync state belongs to device {state.get('device_id')!r}, not {self.device_id!r}"
            )
        users: dict[str, _UserCursor] = {}
        for uid, raw in dict(state.get("users", {})).items():
            users[str(uid)] = _UserCursor(
                cursor=int(raw.get("cursor", 0)),
                known={str(k): str(v) for k, v in dict(raw.get("known", {})).items()},
            )
        self._users = users

    # sync --------------------------------------------------------------------------------
    def sync(self, user_id: str) -> SyncReport:
        """Pull remote changes, merge them, push local changes; return what happened."""
        _check_user_id(user_id)
        st = self._users.setdefault(user_id, _UserCursor())
        report = SyncReport()
        self._pull(user_id, st, report)
        latest = self._push(user_id, st, report)
        if latest > st.cursor:
            # Collect anything another device pushed concurrently; our own echoes are no-ops.
            self._pull(user_id, st, report)
        report.revision = st.cursor
        return report

    def _pull(self, user_id: str, st: _UserCursor, report: SyncReport) -> None:
        items, latest = self.backend.pull(user_id, st.cursor)
        if latest < st.cursor:
            # The backend lost history (reset/replaced): start over and re-push everything.
            st.cursor = 0
            st.known.clear()
            items, latest = self.backend.pull(user_id, 0)
        for remote in items:
            if remote.user_id != user_id:
                continue
            remote_digest = version_digest(remote)
            local = self.store.get(remote.id)
            if local is None:
                self.store.upsert(_clone(remote))
                report.pulled += 1
            else:
                local_digest = version_digest(local)
                local_dirty = local_digest != st.known.get(remote.id)
                if local_dirty and local_digest != remote_digest:
                    report.conflicts += 1
                merged = resolve_conflict(local, remote)
                if merged.to_dict() != local.to_dict():
                    self.store.upsert(merged)
                    report.pulled += 1
            st.known[remote.id] = remote_digest
        st.cursor = max(st.cursor, latest)

    def _push(self, user_id: str, st: _UserCursor, report: SyncReport) -> int:
        outgoing: list[tuple[MemoryItem, str]] = []
        for item in self.store.all(user_id, include_deleted=True):
            if item.user_id != user_id:
                continue
            if item.origin == LOCAL_ORIGIN:
                item = replace(_clone(item), origin=self.device_id)
                self.store.upsert(item)
            digest = version_digest(item)
            if st.known.get(item.id) != digest:
                outgoing.append((item, digest))
        if not outgoing:
            return st.cursor
        latest = self.backend.push(user_id, [item for item, _ in outgoing])
        for item, digest in outgoing:
            st.known[item.id] = digest
        report.pushed += len(outgoing)
        return latest


# --------------------------------------------------------------------------- session manager


class SessionManager:
    """Tracks conversation sessions and shares them across devices through a backend."""

    def __init__(
        self,
        backend: SyncBackend,
        device_id: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("device_id must be a non-empty string")
        self.backend = backend
        self.device_id = device_id
        self.clock = clock
        self._sessions: dict[str, SessionInfo] = {}

    def _require(self, session_id: str) -> SessionInfo:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown session {session_id!r}") from None

    def _require_active(self, session_id: str) -> SessionInfo:
        info = self._require(session_id)
        if not info.is_active:
            raise ValueError(f"session {session_id!r} has already ended")
        return info

    def start_session(self, user_id: str) -> SessionInfo:
        """Open a new session for ``user_id`` on this device."""
        _check_user_id(user_id)
        info = SessionInfo(
            session_id=new_id(),
            user_id=user_id,
            device_id=self.device_id,
            started_at=float(self.clock()),
        )
        self.backend.put_session(info)
        self._sessions[info.session_id] = info
        return info.copy()

    def record_message(self, session_id: str) -> int:
        """Count one more message in an active session; return the new count."""
        info = self._require_active(session_id)
        info.message_count += 1
        self.backend.put_session(info)
        return info.message_count

    def end_session(
        self, session_id: str, summary: str = "", topics: Iterable[str] | None = None
    ) -> SessionInfo:
        """Close an active session, recording what it was about."""
        info = self._require_active(session_id)
        info.ended_at = float(self.clock())
        info.summary = summary
        info.topics = [str(t) for t in topics] if topics is not None else []
        self.backend.put_session(info)
        return info.copy()

    def get(self, session_id: str) -> SessionInfo:
        """A session started here or seen via :meth:`list_sessions`; ``KeyError`` otherwise."""
        return self._require(session_id).copy()

    def list_sessions(self, user_id: str) -> list[SessionInfo]:
        """All sessions of ``user_id`` across devices, newest start first."""
        sessions = sorted(self.backend.list_sessions(user_id), key=_session_sort_key, reverse=True)
        for info in sessions:
            self._sessions[info.session_id] = info.copy()
        return [s.copy() for s in sessions]

    def last_session(self, user_id: str, exclude: str | None = None) -> SessionInfo | None:
        """The most recently *ended* session of ``user_id`` (optionally skipping one id)."""
        ended = [
            s
            for s in self.list_sessions(user_id)
            if s.ended_at is not None and s.session_id != exclude
        ]
        if not ended:
            return None
        return max(ended, key=lambda s: (s.ended_at, s.started_at, s.session_id))
