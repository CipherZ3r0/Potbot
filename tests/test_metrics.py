"""Unit tests for pipeline metrics."""

import pytest
import time
from ingestion.metrics import PipelineMetrics, StageTimer


def test_metrics_defaults():
    m = PipelineMetrics()
    assert m.files_found == 0
    assert m.wall_time_s == 0.0
    assert m.cache_hit_rate == 0.0

def test_metrics_wall_timer():
    m = PipelineMetrics()
    m.start_wall_timer()
    time.sleep(0.01)
    m.stop_wall_timer()
    assert m.wall_time_s >= 0.01

def test_metrics_computed_properties():
    m = PipelineMetrics()
    m.chunks_cached = 50
    m.chunks_embedded = 50
    m.docs_indexed = 100
    m.wall_time_s = 2.0
    
    assert m.cache_hit_rate == 0.5
    assert m.throughput_chunks_per_s == 50.0

def test_stage_timer():
    m = PipelineMetrics()
    with StageTimer(m, "load"):
        time.sleep(0.01)
    
    assert m.load_time_s >= 0.01
    assert m.chunk_time_s == 0.0

def test_metrics_as_dict():
    m = PipelineMetrics()
    m.chunks_produced = 10
    d = m.as_dict()
    assert "chunks_produced" in d
    assert d["chunks_produced"] == 10
    assert "cache_hit_rate" in d
    assert "_wall_start" not in d
