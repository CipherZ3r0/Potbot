"""
Embedding Cache — Abstract interface and SQLite implementation.

Caches chunk embeddings keyed by ``sha256(chunk.text)``.
This prevents identical text chunks from being sent through the heavy
embedding model on subsequent ingestion runs.

Usage::

    from ingestion.embed_cache import SQLiteEmbeddingCache

    cache = SQLiteEmbeddingCache(".embed_cache.db", max_entries=100000)
    
    vec = cache.get("hash_of_hello")
    if not vec:
        vec = model.encode("hello")
        cache.put("hash_of_hello", vec)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseEmbeddingCache(ABC):
    """Abstract Base Class for embedding caches."""

    @abstractmethod
    def get(self, text_hash: str) -> Optional[List[float]]:
        """Return the cached embedding for *text_hash*, or None if not found."""
        pass

    @abstractmethod
    def put(self, text_hash: str, vector: List[float]) -> None:
        """Store *vector* in the cache keyed by *text_hash*."""
        pass


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

class SQLiteEmbeddingCache(BaseEmbeddingCache):
    """SQLite-backed embedding cache with LRU eviction.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    max_entries:
        Maximum number of embeddings to keep.  ``<= 0`` means unlimited.
        When exceeded, the least recently requested embeddings are evicted.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS embeddings (
        text_hash   TEXT PRIMARY KEY,
        vector_json TEXT NOT NULL,
        last_accessed TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_last_accessed ON embeddings(last_accessed);
    """

    def __init__(self, db_path: str = ".embed_cache.db", max_entries: int = 100_000) -> None:
        self.db_path = db_path
        self.max_entries = max_entries
        self._init_schema()

    def get(self, text_hash: str) -> Optional[List[float]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector_json FROM embeddings WHERE text_hash = ?",
                (text_hash,)
            ).fetchone()

            if row:
                # Update last_accessed for LRU asynchronously (or here directly since SQLite WAL is fast)
                conn.execute(
                    "UPDATE embeddings SET last_accessed = strftime('%Y-%m-%d %H:%M:%f', 'now') WHERE text_hash = ?",
                    (text_hash,)
                )
                return json.loads(row[0])
            return None

    def put(self, text_hash: str, vector: List[float]) -> None:
        vec_json = json.dumps(vector)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO embeddings (text_hash, vector_json, last_accessed)
                   VALUES (?, ?, strftime('%Y-%m-%d %H:%M:%f', 'now'))
                   ON CONFLICT(text_hash) DO UPDATE SET vector_json=excluded.vector_json,
                                                        last_accessed=excluded.last_accessed""",
                (text_hash, vec_json),
            )
            
            # Enforce LRU eviction if needed
            if self.max_entries > 0:
                self._evict_if_needed(conn)

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        """Delete oldest entries if count exceeds max_entries."""
        # Fast approximate count check first to avoid heavy DELETE queries on every put
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if count > self.max_entries:
            # Delete entries keeping only the max_entries most recent
            delete_count = count - self.max_entries
            conn.execute(
                """DELETE FROM embeddings WHERE text_hash IN (
                       SELECT text_hash FROM embeddings
                       ORDER BY last_accessed ASC
                       LIMIT ?
                   )""",
                (delete_count,)
            )
            logger.info("embed_cache: evicted %d oldest entries", delete_count)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
