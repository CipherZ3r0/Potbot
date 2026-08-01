"""Unit tests for SQLiteEmbeddingCache."""

import pytest
import time
from ingestion.embed_cache import SQLiteEmbeddingCache


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "cache.db"
    return SQLiteEmbeddingCache(str(db_path), max_entries=5)


def test_put_and_get(cache):
    cache.put("h1", [0.1, 0.2, 0.3])
    vec = cache.get("h1")
    assert vec == [0.1, 0.2, 0.3]


def test_get_missing(cache):
    assert cache.get("missing") is None


def test_lru_eviction(cache):
    # Put 6 items in a cache with max_entries=5
    for i in range(1, 7):
        cache.put(f"h{i}", [float(i)])
        time.sleep(0.01)  # Ensure distinct timestamps
        
    # The oldest (h1) should be evicted
    assert cache.get("h1") is None
    
    # The rest should remain
    for i in range(2, 7):
        assert cache.get(f"h{i}") == [float(i)]


def test_update_existing(cache):
    cache.put("h1", [1.0])
    cache.put("h1", [2.0])
    assert cache.get("h1") == [2.0]
