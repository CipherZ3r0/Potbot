"""
Pipeline Metrics — lightweight, zero-dependency counters and timers.

``PipelineMetrics`` accumulates per-stage statistics as the pipeline runs.
At the end of each stage (and of the full pipeline) a structured log line
is emitted so the data can be ingested by any log aggregator (Datadog,
Grafana Loki, CloudWatch, etc.) without code changes.

Design decisions
----------------
* Pure dataclass — no external deps, fully picklable, trivially testable.
* Structured emission uses ``logger.info`` with a JSON payload rather than
  custom metrics sinks, keeping the implementation dependency-free while
  remaining easy to hook into any observability stack later.
* ``StageTimer`` is a context-manager convenience to time a code block and
  write the result into the matching ``PipelineMetrics`` field.

Usage::

    from ingestion.metrics import PipelineMetrics, StageTimer

    metrics = PipelineMetrics()

    with StageTimer(metrics, "load"):
        docs = list(loader.stream_directory(path))

    metrics.docs_loaded = len(docs)
    metrics.emit("load")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Counters and timers accumulated across all ingestion stages.

    All numeric fields default to ``0`` so callers only need to set the
    ones that are relevant to their stage.
    """

    # ---- Load stage ----
    files_found: int = 0
    """Total files discovered in the source directory."""
    files_skipped: int = 0
    """Files skipped because their content hash has not changed (incremental mode)."""
    docs_loaded: int = 0
    """Document sections successfully loaded."""
    load_errors: int = 0
    """Files that raised an exception during loading."""
    load_time_s: float = 0.0

    # ---- Chunk stage ----
    chunks_produced: int = 0
    """Total chunks generated from all loaded documents."""
    chunk_time_s: float = 0.0

    # ---- Embed stage ----
    chunks_cached: int = 0
    """Chunks whose embedding was retrieved from cache (cache hit)."""
    chunks_embedded: int = 0
    """Chunks that required a model forward pass."""
    embed_errors: int = 0
    embed_time_s: float = 0.0

    # ---- Index stage ----
    docs_indexed: int = 0
    """Chunks successfully written to the vector store."""
    index_errors: int = 0
    index_time_s: float = 0.0

    # ---- Overall ----
    wall_time_s: float = 0.0
    """Total wall-clock time from pipeline start to finish."""

    # ---- Internal ----
    _wall_start: float = field(default=0.0, repr=False, compare=False)

    def start_wall_timer(self) -> None:
        """Record the pipeline start time."""
        self._wall_start = time.perf_counter()

    def stop_wall_timer(self) -> None:
        """Compute and store total wall-clock time."""
        if self._wall_start:
            self.wall_time_s = time.perf_counter() - self._wall_start

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of chunks served from the embedding cache (0–1)."""
        total = self.chunks_cached + self.chunks_embedded
        return self.chunks_cached / total if total else 0.0

    @property
    def throughput_chunks_per_s(self) -> float:
        """Overall chunks indexed per wall-clock second."""
        return self.docs_indexed / self.wall_time_s if self.wall_time_s else 0.0

    def as_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["cache_hit_rate"] = round(self.cache_hit_rate, 4)
        d["throughput_chunks_per_s"] = round(self.throughput_chunks_per_s, 2)
        return d

    def emit(self, stage: str = "pipeline") -> None:
        """Emit a structured JSON log line for the given stage."""
        payload = {"stage": stage, **self.as_dict()}
        logger.info("metrics %s", json.dumps(payload))

    def emit_summary(self) -> None:
        """Emit a concise human-readable summary at INFO level."""
        logger.info(
            "ingestion complete | files=%d skipped=%d docs=%d chunks=%d "
            "cached=%d embedded=%d indexed=%d errors_load=%d errors_embed=%d "
            "errors_index=%d wall=%.2fs throughput=%.1f chunks/s",
            self.files_found,
            self.files_skipped,
            self.docs_loaded,
            self.chunks_produced,
            self.chunks_cached,
            self.chunks_embedded,
            self.docs_indexed,
            self.load_errors,
            self.embed_errors,
            self.index_errors,
            self.wall_time_s,
            self.throughput_chunks_per_s,
        )


class StageTimer:
    """Context manager that times a code block and stores the result.

    Parameters
    ----------
    metrics:
        The :class:`PipelineMetrics` instance to update.
    stage:
        One of ``"load"``, ``"chunk"``, ``"embed"``, ``"index"``.
        The corresponding ``{stage}_time_s`` field is updated on exit.

    Example
    -------
    ::

        with StageTimer(metrics, "embed"):
            embeddings = model.encode(texts)
    """

    _FIELD_MAP = {
        "load": "load_time_s",
        "chunk": "chunk_time_s",
        "embed": "embed_time_s",
        "index": "index_time_s",
    }

    def __init__(self, metrics: PipelineMetrics, stage: str) -> None:
        self._metrics = metrics
        self._stage = stage
        self._field = self._FIELD_MAP.get(stage)
        self._start: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        elapsed = time.perf_counter() - self._start
        if self._field:
            current = getattr(self._metrics, self._field, 0.0)
            setattr(self._metrics, self._field, current + elapsed)
