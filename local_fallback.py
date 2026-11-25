"""Durable offline memory store and replayable remote-operation outbox.

This module has no Hindsight, embedding, or Ollama imports. The Hindsight provider
wires it into retain/replay and offline recall paths. Retrieved text is data, not
trusted instructions.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal


class AmbiguousSourceError(ValueError):
    """A source resolves to more than one local/remote document mapping."""


class UnknownSourceError(KeyError):
    """No document mapping exists for a source that must be replaced or removed."""


@dataclass(frozen=True)
class Document:
    document_id: str
    source_id: str
    text: str


@dataclass(frozen=True)
class Operation:
    operation_id: str
    sequence: int
    kind: Literal["retain", "replace", "delete"]
    source_id: str
    document_id: str
    payload: str | None
    status: str
    expected_remote_text: str | None
    remote_options: dict
    lease_owner: str | None = None
    lease_until: float | None = None


class LocalFallbackStore:
    """SQLite WAL/FULL local store with FTS5 and a leased FIFO outbox."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            db_path = Path(self.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                os.chmod(self.path, 0o600)
            else:
                os.fchmod(fd, 0o600)
                os.close(fd)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._secure_file_modes()
        self._db.execute("PRAGMA synchronous=FULL")
        self._fts_enabled = False
        self._initialize()
        self._secure_file_modes()

    def _secure_file_modes(self) -> None:
        """Keep the database and SQLite WAL sidecars private on existing stores too."""
        if self.path == ":memory:":
            return
        for filename in (self.path, f"{self.path}-wal", f"{self.path}-shm"):
            try:
                os.chmod(filename, 0o600)
            except FileNotFoundError:
                pass

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Serialize this connection and take a SQLite write lock across the full mutation."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                try:
                    self._db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            else:
                try:
                    self._db.execute("COMMIT")
                except BaseException:
                    try:
                        self._db.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise

    def _initialize(self) -> None:
        with self._lock:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0,1)),
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS source_mappings (
                    source_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    PRIMARY KEY (source_id, document_id),
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                );
                CREATE TABLE IF NOT EXISTS outbox (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL CHECK(kind IN ('retain','replace','delete')),
                    source_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    payload TEXT,
                    expected_remote_text TEXT,
                    remote_options TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','done')),
                    lease_owner TEXT,
                    lease_until REAL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                """
            )
            self._ensure_column("documents", "deleted", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("outbox", "expected_remote_text", "TEXT")
            self._ensure_column("outbox", "remote_options", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("outbox", "lease_owner", "TEXT")
            self._ensure_column("outbox", "lease_until", "REAL")
            self._db.execute(
                "UPDATE outbox SET expected_remote_text=payload "
                "WHERE expected_remote_text IS NULL AND kind IN ('retain','replace')"
            )
            try:
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS document_fts "
                    "USING fts5(document_id UNINDEXED, source_id UNINDEXED, text)"
                )
                self._fts_enabled = True
                self._db.execute(
                    "INSERT INTO document_fts(document_id,source_id,text) "
                    "SELECT d.document_id,d.source_id,d.text FROM documents d "
                    "WHERE d.deleted=0 AND NOT EXISTS "
                    "(SELECT 1 FROM document_fts f WHERE f.document_id=d.document_id)"
                )
                self._db.execute(
                    "DELETE FROM document_fts WHERE document_id IN "
                    "(SELECT document_id FROM documents WHERE deleted=1)"
                )
            except sqlite3.OperationalError:
                self._fts_enabled = False

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "LocalFallbackStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _mapped_document(self, source_id: str) -> str | None:
        rows = self._db.execute(
            "SELECT document_id FROM source_mappings WHERE source_id=? ORDER BY document_id", (source_id,)
        ).fetchall()
        if len(rows) > 1:
            raise AmbiguousSourceError(f"source {source_id!r} maps to multiple documents")
        return rows[0][0] if rows else None

    def _ensure_document_unshared(self, document_id: str, source_id: str) -> None:
        other = self._db.execute(
            "SELECT source_id FROM source_mappings WHERE document_id=? AND source_id<>? LIMIT 1",
            (document_id, source_id),
        ).fetchone()
        if other is not None:
            raise AmbiguousSourceError(f"document {document_id!r} is mapped to multiple sources")

    @staticmethod
    def _serialize_options(options: dict) -> str:
        if not isinstance(options, dict):
            raise TypeError("remote options must be a dictionary")
        return json.dumps(options, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _operation(self, row: sqlite3.Row | None) -> Operation | None:
        if row is None:
            return None
        values = dict(row)
        values["remote_options"] = json.loads(values["remote_options"] or "{}")
        values.setdefault("expected_remote_text", values.get("payload"))
        return Operation(**{key: values[key] for key in Operation.__dataclass_fields__})

    def _enqueue(
        self, kind: str, source_id: str, document_id: str, payload: str | None, options: dict,
        expected_remote_text: str | None = None,
    ) -> Operation:
        encoded_options = self._serialize_options(options)
        expected = (payload if expected_remote_text is None else expected_remote_text) if kind != "delete" else None
        latest = self._db.execute(
            "SELECT * FROM outbox WHERE source_id=? ORDER BY sequence DESC LIMIT 1", (source_id,)
        ).fetchone()
        retain_kinds = {"retain", "replace"}
        same_retain = kind in retain_kinds and latest is not None and latest["kind"] in retain_kinds
        same_pending_delete = (
            kind == "delete" and latest is not None and latest["kind"] == "delete" and latest["status"] == "pending"
        )
        if (
            latest is not None
            and (same_retain or same_pending_delete)
            and latest["document_id"] == document_id
            and latest["payload"] == payload
            and latest["expected_remote_text"] == expected
            and (latest["remote_options"] or "{}") == encoded_options
        ):
            return self._operation(latest)
        op_id = str(uuid.uuid4())
        cursor = self._db.execute(
            "INSERT INTO outbox(operation_id,kind,source_id,document_id,payload,expected_remote_text,remote_options) "
            "VALUES(?,?,?,?,?,?,?)",
            (op_id, kind, source_id, document_id, payload, expected, encoded_options),
        )
        row = self._db.execute("SELECT * FROM outbox WHERE sequence=?", (cursor.lastrowid,)).fetchone()
        return self._operation(row)

    def retain(
        self, source_id: str, text: str, *, document_id: str | None = None,
        remote_payload: str | None = None, expected_remote_text: str | None = None, **remote_options,
    ) -> Operation:
        """Persist a local mirror separately from the immutable remote write and readback."""
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        remote_payload = text if remote_payload is None else remote_payload
        expected_remote_text = remote_payload if expected_remote_text is None else expected_remote_text
        if not isinstance(remote_payload, str) or not remote_payload:
            raise ValueError("remote_payload must be a non-empty string")
        if not isinstance(expected_remote_text, str) or not expected_remote_text:
            raise ValueError("expected_remote_text must be a non-empty string")
        encoded_options = self._serialize_options(remote_options)
        with self._transaction():
            mapped_id = self._mapped_document(source_id)
            if mapped_id is not None:
                self._ensure_document_unshared(mapped_id, source_id)
                if document_id is not None and document_id != mapped_id:
                    raise AmbiguousSourceError("source is already mapped to a different document")
                document_id = mapped_id
            elif document_id is None:
                document_id = str(uuid.uuid4())
            else:
                self._ensure_document_unshared(document_id, source_id)
            kind: Literal["retain", "replace"] = "replace" if mapped_id else "retain"
            if mapped_id is None:
                self._db.execute(
                    "INSERT INTO documents(document_id,source_id,text) VALUES(?,?,?)", (document_id, source_id, text)
                )
                self._db.execute(
                    "INSERT INTO source_mappings(source_id,document_id) VALUES(?,?)", (source_id, document_id)
                )
            else:
                self._db.execute(
                    "UPDATE documents SET text=?,deleted=0,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                    "WHERE document_id=?", (text, document_id),
                )
            self._index_document(document_id, source_id, text)
            return self._enqueue(
                kind, source_id, document_id, remote_payload, json.loads(encoded_options), expected_remote_text
            )

    def remove(self, source_id: str, **remote_options) -> Operation:
        """Tombstone a uniquely mapped document immediately and queue remote deletion."""
        encoded_options = self._serialize_options(remote_options)
        with self._transaction():
            document_id = self._mapped_document(source_id)
            if document_id is None:
                raise UnknownSourceError(source_id)
            self._ensure_document_unshared(document_id, source_id)
            self._db.execute(
                "UPDATE documents SET deleted=1,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE document_id=?",
                (document_id,),
            )
            if self._fts_enabled:
                self._db.execute("DELETE FROM document_fts WHERE document_id=?", (document_id,))
            return self._enqueue("delete", source_id, document_id, None, json.loads(encoded_options))

    def add_mapping(self, source_id: str, document_id: str) -> None:
        """Record a mapping only if no other source owns the remote document ID."""
        with self._transaction():
            self._ensure_document_unshared(document_id, source_id)
            self._db.execute(
                "INSERT OR IGNORE INTO source_mappings(source_id,document_id) VALUES(?,?)", (source_id, document_id)
            )

    def _index_document(self, document_id: str, source_id: str, text: str) -> None:
        if self._fts_enabled:
            self._db.execute("DELETE FROM document_fts WHERE document_id=?", (document_id,))
            self._db.execute(
                "INSERT INTO document_fts(document_id,source_id,text) VALUES(?,?,?)", (document_id, source_id, text)
            )

    def get_document(self, source_id: str) -> Document | None:
        with self._lock:
            document_id = self._mapped_document(source_id)
            if document_id is None:
                return None
            row = self._db.execute(
                "SELECT document_id,source_id,text FROM documents WHERE document_id=? AND deleted=0", (document_id,)
            ).fetchone()
            return Document(row["document_id"], row["source_id"], row["text"]) if row else None

    def search(self, query: str, limit: int = 10) -> list[Document]:
        if not self._fts_enabled:
            return []
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []
        natural_query = " AND ".join(f'"{term}"' for term in terms)
        with self._lock:
            rows = self._db.execute(
                "SELECT d.document_id,d.source_id,d.text FROM document_fts f "
                "JOIN documents d USING(document_id) WHERE document_fts MATCH ? AND d.deleted=0 "
                "ORDER BY bm25(document_fts), d.document_id LIMIT ?",
                (natural_query, max(0, int(limit))),
            ).fetchall()
        return [Document(r["document_id"], r["source_id"], r["text"]) for r in rows]

    def next_operation(self) -> Operation | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM outbox WHERE status='pending' ORDER BY sequence LIMIT 1").fetchone()
            return self._operation(row)

    def operations(self) -> list[Operation]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM outbox ORDER BY sequence").fetchall()
            return [self._operation(row) for row in rows]

    def claim_next(self, owner_id: str, *, lease_seconds: float = 30.0) -> Operation | None:
        if not owner_id:
            raise ValueError("owner_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = time.time()
        with self._transaction():
            row = self._db.execute(
                "SELECT * FROM outbox WHERE status='pending' ORDER BY sequence LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            if row["lease_owner"] and row["lease_until"] is not None and row["lease_until"] > now:
                if row["lease_owner"] != owner_id:
                    return None
            expiry = now + lease_seconds
            self._db.execute(
                "UPDATE outbox SET lease_owner=?,lease_until=? WHERE operation_id=?",
                (owner_id, expiry, row["operation_id"]),
            )
            claimed = self._db.execute("SELECT * FROM outbox WHERE operation_id=?", (row["operation_id"],)).fetchone()
            return self._operation(claimed)

    def release(self, operation_id: str, *, owner_id: str) -> bool:
        now = time.time()
        with self._transaction():
            row = self._db.execute(
                "SELECT * FROM outbox WHERE operation_id=? AND status='pending'", (operation_id,)
            ).fetchone()
            if (
                row is None
                or not self._is_next_operation(operation_id)
                or row["lease_owner"] != owner_id
                or row["lease_until"] is None
                or row["lease_until"] <= now
            ):
                return False
            self._db.execute(
                "UPDATE outbox SET lease_owner=NULL,lease_until=NULL WHERE operation_id=?", (operation_id,)
            )
            return True

    def _is_next_operation(self, operation_id: str) -> bool:
        row = self._db.execute(
            "SELECT operation_id FROM outbox WHERE status='pending' ORDER BY sequence LIMIT 1"
        ).fetchone()
        return row is not None and row[0] == operation_id

    def _owned_head(self, operation_id: str, owner_id: str) -> sqlite3.Row | None:
        now = time.time()
        row = self._db.execute(
            "SELECT * FROM outbox WHERE operation_id=? AND status='pending'", (operation_id,)
        ).fetchone()
        if (
            row is None
            or not self._is_next_operation(operation_id)
            or row["lease_owner"] != owner_id
            or row["lease_until"] is None
            or row["lease_until"] <= now
        ):
            return None
        return row

    def complete_retain(
        self, operation_id: str, remote_document_id: str, remote_text: str, *, owner_id: str
    ) -> bool:
        """Ack the owned FIFO retain only on exact target ID and text readback."""
        with self._transaction():
            row = self._owned_head(operation_id, owner_id)
            if row is None or row["kind"] == "delete":
                return False
            if remote_document_id != row["document_id"] or remote_text != row["expected_remote_text"]:
                return False
            self._db.execute(
                "UPDATE outbox SET status='done',lease_owner=NULL,lease_until=NULL WHERE operation_id=?",
                (operation_id,),
            )
            return True

    def complete_delete(self, operation_id: str, *, remote_absent: bool, owner_id: str) -> bool:
        """Ack delete only after absence readback, preserving later local replacements."""
        if remote_absent is not True:
            return False
        with self._transaction():
            row = self._owned_head(operation_id, owner_id)
            if row is None or row["kind"] != "delete":
                return False
            self._db.execute(
                "UPDATE outbox SET status='done',lease_owner=NULL,lease_until=NULL WHERE operation_id=?",
                (operation_id,),
            )
            latest = self._db.execute(
                "SELECT kind FROM outbox WHERE document_id=? ORDER BY sequence DESC LIMIT 1", (row["document_id"],)
            ).fetchone()
            if latest is None or latest["kind"] == "delete":
                document_id = row["document_id"]
                self._db.execute(
                    "UPDATE documents SET deleted=1,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                    "WHERE document_id=?",
                    (document_id,),
                )
                if self._fts_enabled:
                    self._db.execute("DELETE FROM document_fts WHERE document_id=?", (document_id,))
                self._db.execute("DELETE FROM source_mappings WHERE document_id=?", (document_id,))
                self._db.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
                self._db.execute(
                    "UPDATE outbox SET payload=NULL,expected_remote_text=NULL,remote_options='{}' "
                    "WHERE document_id=? AND status='done'",
                    (document_id,),
                )
            return True
