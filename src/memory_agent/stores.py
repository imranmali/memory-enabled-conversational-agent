"""Long-term memory stores implementing :class:`~memory_agent.interfaces.VectorStore`.

* :class:`InMemoryVectorStore` - dict-backed, for tests and ephemeral sessions.
* :class:`SQLiteVectorStore` - single-file (or ``:memory:``) SQLite database via stdlib
  ``sqlite3``; survives process restarts.

Both fix the embedding dimension (explicitly or from the first upsert), store and return
independent copies, tombstone deletions (``deleted=True`` plus a strictly increasing
``updated_at``) so they propagate during sync, and rank search results by exact cosine
similarity with a deterministic tie-break (newer ``created_at`` first, then ``id``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from array import array
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .embeddings import cosine_similarity
from .types import MemoryItem

Clock = Callable[[], float]

#: Minimum amount a tombstone advances ``updated_at`` so last-write-wins sees it as newer.
TOMBSTONE_EPSILON = 1e-6


def _copy(item: MemoryItem) -> MemoryItem:
    return MemoryItem.from_dict(item.to_dict())


def _check_dim(dim: int | None) -> int | None:
    if dim is None:
        return None
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive integer or None, got {dim!r}")
    return dim


class _BaseVectorStore:
    """Shared contract logic; subclasses implement the row-level primitives."""

    def __init__(self, dim: int | None, clock: Clock) -> None:
        self._dim = _check_dim(dim)
        self._clock = clock
        self._lock = threading.RLock()

    # -- primitives (called with the lock held) ----------------------------------------------

    def _put(self, item: MemoryItem) -> None:
        raise NotImplementedError

    def _get(self, item_id: str) -> MemoryItem | None:
        raise NotImplementedError

    def _candidates(self, user_id: str, kinds: Sequence[str] | None) -> Iterable[MemoryItem]:
        """Live items for ``user_id`` (optionally of ``kinds``); may be shared, never mutated."""
        raise NotImplementedError

    def _items(self, user_id: str | None, include_deleted: bool) -> list[MemoryItem]:
        raise NotImplementedError

    def _count(self, user_id: str | None) -> int:
        raise NotImplementedError

    def _on_dim_fixed(self, dim: int) -> None:
        """Hook for backends that persist the dimension."""

    # -- dimension -----------------------------------------------------------------------------

    @property
    def dim(self) -> int | None:
        """The fixed embedding dimension, or ``None`` until the first upsert."""
        return self._dim

    def _validate_dim(self, vector: Sequence[float], what: str) -> None:
        if self._dim is not None and len(vector) != self._dim:
            raise ValueError(
                f"{what} dimension {len(vector)} does not match store dimension {self._dim}"
            )

    # -- VectorStore API -------------------------------------------------------------------

    def upsert(self, item: MemoryItem) -> None:
        """Insert ``item`` or replace the stored item with the same id (stores a copy)."""
        if len(item.embedding) == 0:
            raise ValueError("embedding must not be empty")
        with self._lock:
            self._validate_dim(item.embedding, "embedding")
            stored = _copy(item)
            stored.embedding = [float(x) for x in stored.embedding]
            self._put(stored)
            if self._dim is None:
                self._dim = len(stored.embedding)
                self._on_dim_fixed(self._dim)

    def get(self, item_id: str) -> MemoryItem | None:
        """Return a copy of the item (tombstones included) or ``None``."""
        with self._lock:
            item = self._get(item_id)
            return None if item is None else _copy(item)

    def delete(self, item_id: str) -> bool:
        """Tombstone the item; ``updated_at`` becomes ``max(clock(), previous + 1e-6)``."""
        with self._lock:
            item = self._get(item_id)
            if item is None:
                return False
            item = _copy(item)
            previous = item.updated_at if item.updated_at is not None else item.created_at
            item.deleted = True
            item.updated_at = max(float(self._clock()), previous + TOMBSTONE_EPSILON)
            self._put(item)
            return True

    def search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        k: int = 10,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Top-``k`` live items for ``user_id`` by cosine similarity, highest first."""
        with self._lock:
            self._validate_dim(query_embedding, "query")
            if k <= 0:
                return []
            scored = [
                (item, cosine_similarity(query_embedding, item.embedding))
                for item in self._candidates(user_id, kinds)
            ]
        scored.sort(key=lambda pair: (-pair[1], -pair[0].created_at, pair[0].id))
        return [(_copy(item), sim) for item, sim in scored[:k]]

    def all(self, user_id: str | None = None, include_deleted: bool = False) -> list[MemoryItem]:
        """Copies of all items (oldest ``created_at`` first), optionally for one user."""
        with self._lock:
            items = self._items(user_id, include_deleted)
        items.sort(key=lambda i: (i.created_at, i.id))
        return [_copy(i) for i in items]

    def count(self, user_id: str | None = None) -> int:
        """Number of live items, optionally for one user."""
        with self._lock:
            return self._count(user_id)

    def __len__(self) -> int:
        return self.count()

    # -- lifecycle -------------------------------------------------------------------------------

    def close(self) -> None:
        """Release resources (no-op unless overridden)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class InMemoryVectorStore(_BaseVectorStore):
    """Dict-backed vector store; contents are lost when the object is discarded."""

    def __init__(self, dim: int | None = None, clock: Clock = time.time) -> None:
        super().__init__(dim, clock)
        self._data: dict[str, MemoryItem] = {}

    def _put(self, item: MemoryItem) -> None:
        self._data[item.id] = item

    def _get(self, item_id: str) -> MemoryItem | None:
        return self._data.get(item_id)

    def _candidates(self, user_id: str, kinds: Sequence[str] | None) -> Iterable[MemoryItem]:
        wanted = None if kinds is None else set(kinds)
        return [
            i
            for i in self._data.values()
            if i.user_id == user_id and not i.deleted and (wanted is None or i.kind in wanted)
        ]

    def _items(self, user_id: str | None, include_deleted: bool) -> list[MemoryItem]:
        return [
            i
            for i in self._data.values()
            if (user_id is None or i.user_id == user_id) and (include_deleted or not i.deleted)
        ]

    def _count(self, user_id: str | None) -> int:
        return len(self._items(user_id, include_deleted=False))

    def __repr__(self) -> str:
        return f"InMemoryVectorStore(items={len(self._data)}, dim={self._dim})"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    kind TEXT NOT NULL,
    session_id TEXT,
    importance REAL NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL,
    access_count INTEGER NOT NULL,
    updated_at REAL,
    origin TEXT NOT NULL,
    deleted INTEGER NOT NULL,
    metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id, deleted);
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_COLUMNS = (
    "id, user_id, content, embedding, kind, session_id, importance, created_at, "
    "last_accessed, access_count, updated_at, origin, deleted, metadata"
)


def _encode_embedding(vector: Sequence[float]) -> bytes:
    return array("d", vector).tobytes()


def _decode_embedding(blob: bytes) -> list[float]:
    values = array("d")
    values.frombytes(blob)
    return values.tolist()


def _row_to_item(row: tuple[Any, ...]) -> MemoryItem:
    (
        item_id,
        user_id,
        content,
        embedding,
        kind,
        session_id,
        importance,
        created_at,
        last_accessed,
        access_count,
        updated_at,
        origin,
        deleted,
        metadata,
    ) = row
    return MemoryItem(
        id=item_id,
        user_id=user_id,
        content=content,
        embedding=_decode_embedding(embedding),
        kind=kind,
        session_id=session_id,
        importance=importance,
        created_at=created_at,
        last_accessed=last_accessed,
        access_count=access_count,
        updated_at=updated_at,
        origin=origin,
        deleted=bool(deleted),
        metadata=json.loads(metadata),
    )


class SQLiteVectorStore(_BaseVectorStore):
    """SQLite-backed vector store persisted to ``path`` (``":memory:"`` for a transient DB).

    Embeddings are stored as native-endian float64 blobs and metadata as JSON (so metadata
    must be JSON-serializable; tuples come back as lists). The connection is shared across
    threads and every operation is serialized by a lock. Search loads the user's live rows
    and computes exact cosine similarity in Python.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] = ":memory:",
        dim: int | None = None,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(dim, clock)
        self.path = os.fspath(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            with self._conn:
                self._conn.executescript(_SCHEMA)
            stored = self._stored_dim()
            if stored is not None and self._dim is not None and stored != self._dim:
                self._conn.close()
                raise ValueError(f"store at {self.path!r} has dimension {stored}, not {self._dim}")
            if stored is not None:
                self._dim = stored
            elif self._dim is not None:
                self._on_dim_fixed(self._dim)

    def _stored_dim(self) -> int | None:
        row = self._conn.execute("SELECT value FROM store_meta WHERE key = 'dim'").fetchone()
        if row is not None:
            return int(row[0])
        row = self._conn.execute("SELECT length(embedding) FROM memories LIMIT 1").fetchone()
        if row is not None:
            return int(row[0]) // array("d").itemsize
        return None

    def _on_dim_fixed(self, dim: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('dim', ?)", (str(dim),)
            )

    def _put(self, item: MemoryItem) -> None:
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO memories ({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.user_id,
                    item.content,
                    _encode_embedding(item.embedding),
                    item.kind,
                    item.session_id,
                    item.importance,
                    item.created_at,
                    item.last_accessed,
                    item.access_count,
                    item.updated_at,
                    item.origin,
                    int(item.deleted),
                    json.dumps(item.metadata),
                ),
            )

    def _get(self, item_id: str) -> MemoryItem | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (item_id,)
        ).fetchone()
        return None if row is None else _row_to_item(row)

    def _candidates(self, user_id: str, kinds: Sequence[str] | None) -> Iterable[MemoryItem]:
        sql = f"SELECT {_COLUMNS} FROM memories WHERE user_id = ? AND deleted = 0"
        params: list[Any] = [user_id]
        if kinds is not None:
            kinds = list(kinds)
            if not kinds:
                return []
            sql += f" AND kind IN ({', '.join('?' for _ in kinds)})"
            params.extend(kinds)
        return [_row_to_item(r) for r in self._conn.execute(sql, params)]

    def _items(self, user_id: str | None, include_deleted: bool) -> list[MemoryItem]:
        sql = f"SELECT {_COLUMNS} FROM memories"
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if not include_deleted:
            clauses.append("deleted = 0")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [_row_to_item(r) for r in self._conn.execute(sql, params)]

    def _count(self, user_id: str | None) -> int:
        if user_id is None:
            row = self._conn.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE deleted = 0 AND user_id = ?", (user_id,)
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the database connection. Further operations raise ``sqlite3.ProgrammingError``."""
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:
        return f"SQLiteVectorStore(path={self.path!r}, dim={self._dim})"
