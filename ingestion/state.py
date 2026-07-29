"""
Ingestion State — Abstract checkpoint store interface and SQLite implementation.

Provides two capabilities:
1. **File-hash tracking** — records the SHA-256 of each ingested file so
   unchanged files can be skipped on subsequent runs (incremental ingestion).
2. **Run checkpointing** — records per-chunk indexing status so interrupted
   runs can be resumed without re-embedding already-indexed chunks.

Design
------
* ``BaseCheckpointStore`` is a pure-Python ABC with no I/O — trivially
  mockable in tests.
* ``SQLiteCheckpointStore`` is the default zero-dependency implementation
  backed by ``sqlite3`` (stdlib).
* Future backends (Redis, Postgres) implement ``BaseCheckpointStore`` and
  are injected via the ``IngestionPipeline`` constructor — no pipeline logic
  changes required.

Usage::

    from ingestion.state import SQLiteCheckpointStore

    store = SQLiteCheckpointStore(".ingestion_state.db")

    # Incremental skip
    if not store.is_file_changed("/data/report.pdf"):
        continue

    # After successful ingestion of a file
    store.mark_file_done("/data/report.pdf", sha256_hex)

    # Checkpointing a run
    run_id = store.begin_run()
    store.checkpoint_chunk(run_id, chunk_id, "indexed")

    # Resume: get chunks not yet indexed in a prior run
    pending = store.get_pending_chunks(run_id, all_chunk_ids)
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# How many bytes to hash for the file-change check (fast, catches most edits)
_HASH_READ_BYTES = 65_536  # 64 KB


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseCheckpointStore(ABC):
    """Abstract interface for file-hash tracking and run checkpointing.

    Any class implementing this interface can be injected into
    :class:`~ingestion.pipeline.IngestionPipeline` to provide incremental
    ingestion and resume-on-failure capabilities.
    """

    @abstractmethod
    def is_file_changed(self, path: str) -> bool:
        """Return ``True`` if *path* should be re-ingested.

        A file is considered changed if:
        - it has never been seen before, or
        - its current content hash differs from the stored hash.
        """
        pass

    @abstractmethod
    def mark_file_done(self, path: str, sha256: str) -> None:
        """Record that *path* was successfully ingested with hash *sha256*."""
        pass

    @abstractmethod
    def begin_run(self) -> str:
        """Create a new ingestion run record and return its unique ``run_id``."""
        pass

    @abstractmethod
    def checkpoint_chunk(self, run_id: str, chunk_id: str, status: str) -> None:
        """Record the indexing status of *chunk_id* within *run_id*.

        Parameters
        ----------
        run_id:
            UUID string returned by :meth:`begin_run`.
        chunk_id:
            The ``chunk_id`` field of a :class:`~domain.models.Chunk`.
        status:
            Suggested values: ``"indexed"`` or ``"failed"``.
        """
        pass

    @abstractmethod
    def get_pending_chunks(self, run_id: str, all_chunk_ids: List[str]) -> List[str]:
        """Return chunk IDs from *all_chunk_ids* not yet marked ``"indexed"`` in *run_id*.

        Used to resume an interrupted run without re-processing already-indexed chunks.
        """
        pass


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

class SQLiteCheckpointStore(BaseCheckpointStore):
    """SQLite-backed checkpoint store using only the Python standard library.

    Thread-safe for concurrent reads; writes are serialised by SQLite's
    default journal locking.  Suitable for single-node deployments.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created automatically if it
        does not exist.
    """

    # DDL executed once on first connection
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS file_hashes (
        path        TEXT PRIMARY KEY,
        sha256      TEXT NOT NULL,
        ingested_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS ingestion_runs (
        run_id      TEXT PRIMARY KEY,
        started_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunk_progress (
        run_id      TEXT NOT NULL,
        chunk_id    TEXT NOT NULL,
        status      TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        PRIMARY KEY (run_id, chunk_id)
    );
    """

    def __init__(self, db_path: str = ".ingestion_state.db") -> None:
        self.db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------
    # BaseCheckpointStore implementation
    # ------------------------------------------------------------------

    def is_file_changed(self, path: str) -> bool:
        """Return True if the file at *path* has changed since last ingestion."""
        current_hash = _sha256_file(path)
        if current_hash is None:
            # File unreadable — treat as changed so it gets an error on load
            return True

        with self._connect() as conn:
            row = conn.execute(
                "SELECT sha256 FROM file_hashes WHERE path = ?", (path,)
            ).fetchone()

        if row is None:
            return True  # Never seen before

        return row[0] != current_hash

    def mark_file_done(self, path: str, sha256: Optional[str] = None) -> None:
        """Record *path* as successfully ingested.

        Parameters
        ----------
        path:
            Absolute path to the file.
        sha256:
            Pre-computed hash.  If ``None``, the hash is computed here
            (slightly redundant but convenient for callers that don't cache it).
        """
        if sha256 is None:
            sha256 = _sha256_file(path) or ""
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO file_hashes (path, sha256, ingested_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256,
                                                    ingested_at=excluded.ingested_at""",
                (path, sha256, now),
            )

    def begin_run(self) -> str:
        """Create a new run record and return its ``run_id`` (UUID4 string)."""
        run_id = str(uuid.uuid4())
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ingestion_runs (run_id, started_at) VALUES (?, ?)",
                (run_id, now),
            )
        logger.info("state: began ingestion run run_id=%s", run_id)
        return run_id

    def checkpoint_chunk(self, run_id: str, chunk_id: str, status: str) -> None:
        """Upsert the indexing status of a single chunk."""
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO chunk_progress (run_id, chunk_id, status, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(run_id, chunk_id) DO UPDATE SET status=excluded.status,
                                                                updated_at=excluded.updated_at""",
                (run_id, chunk_id, status, now),
            )

    def get_pending_chunks(self, run_id: str, all_chunk_ids: List[str]) -> List[str]:
        """Return chunk IDs from *all_chunk_ids* that are not yet ``"indexed"``."""
        if not all_chunk_ids:
            return []

        with self._connect() as conn:
            placeholders = ",".join("?" * len(all_chunk_ids))
            done = {
                row[0]
                for row in conn.execute(
                    f"""SELECT chunk_id FROM chunk_progress
                        WHERE run_id = ? AND chunk_id IN ({placeholders}) AND status = 'indexed'""",
                    (run_id, *all_chunk_ids),
                )
            }

        return [cid for cid in all_chunk_ids if cid not in done]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @contextmanager
    def _connect(self):
        """Yield an auto-committing connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read performance
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> Optional[str]:
    """Compute SHA-256 of the first :data:`_HASH_READ_BYTES` bytes of *path*."""
    try:
        with open(path, "rb") as f:
            data = f.read(_HASH_READ_BYTES)
        return hashlib.sha256(data).hexdigest()
    except OSError as exc:
        logger.warning("state: could not hash file '%s': %s", path, exc)
        return None


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
