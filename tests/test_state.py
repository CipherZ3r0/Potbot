"""Unit tests for SQLiteCheckpointStore."""

import pytest
import tempfile
import os
from ingestion.state import SQLiteCheckpointStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "state.db"
    return SQLiteCheckpointStore(str(db_path))


def test_is_file_changed_new_file(store, tmp_path):
    f = tmp_path / "new.txt"
    f.write_text("hello")
    assert store.is_file_changed(str(f)) is True


def test_is_file_changed_unchanged_file(store, tmp_path):
    f = tmp_path / "same.txt"
    f.write_text("hello")
    store.mark_file_done(str(f))
    assert store.is_file_changed(str(f)) is False


def test_is_file_changed_modified_file(store, tmp_path):
    f = tmp_path / "mod.txt"
    f.write_text("hello")
    store.mark_file_done(str(f))
    
    f.write_text("world")
    assert store.is_file_changed(str(f)) is True


def test_run_checkpoints(store):
    run_id = store.begin_run()
    assert run_id
    
    # All are pending initially
    pending = store.get_pending_chunks(run_id, ["c1", "c2", "c3"])
    assert len(pending) == 3
    
    # Checkpoint one
    store.checkpoint_chunk(run_id, "c1", "indexed")
    
    pending = store.get_pending_chunks(run_id, ["c1", "c2", "c3"])
    assert len(pending) == 2
    assert "c1" not in pending

def test_missing_file_handled_safely(store):
    assert store.is_file_changed("/nonexistent/file.txt") is True
