"""
Ingestion Pipeline Configuration — PipelineConfig dataclass.

Centralises every runtime tuning knob for the ingestion pipeline.
Components receive a `PipelineConfig` instance via dependency injection;
they never read environment variables directly. This makes every component
trivially testable without monkeypatching the environment.

Usage::

    from ingestion.config import PipelineConfig

    # Production: reads from environment / config.py
    cfg = PipelineConfig.from_env()

    # Tests: override any field
    cfg = PipelineConfig(loader_workers=1, chunker_workers=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Runtime configuration for the ingestion pipeline.

    All fields have sensible defaults suitable for a single-node deployment.
    Override via :meth:`from_env` (reads ``config.py`` values) or by passing
    keyword arguments directly (useful in tests).

    Parallelism
    -----------
    loader_workers
        Number of threads in the ``ThreadPoolExecutor`` used for I/O-bound
        file loading.  Increase on machines with many CPU cores and fast
        storage.
    chunker_workers
        Number of processes in the ``ProcessPoolExecutor`` used for CPU-bound
        text chunking.  Keep at or below the physical CPU core count.
    embed_batch_size
        Chunks per forward pass through the embedding model.  Larger batches
        are more GPU-efficient but use more VRAM/RAM.
    index_bulk_size
        Chunks per Elasticsearch ``_bulk`` API call.  Larger values reduce
        round-trip overhead at the cost of higher per-request memory.
    queue_size
        Maximum items buffered in the inter-stage queue (back-pressure).
        ``0`` means unlimited — use with caution on very large datasets.

    Device
    ------
    device
        Preferred compute device for embedding inference.
        ``"auto"`` resolves at runtime: CUDA → MPS → CPU.

    State & Caching
    ---------------
    checkpoint_path
        Path to the SQLite database used for file-hash tracking and run
        checkpointing.
    embed_cache_enabled
        Whether to consult the embedding cache before calling the model.
    embed_cache_path
        Path to the SQLite database used for the embedding cache.
    embed_cache_max_entries
        LRU eviction limit.  ``<= 0`` means unlimited.
    """

    # ---- Parallelism ----
    loader_workers: int = 4
    chunker_workers: int = 2
    embed_batch_size: int = 64
    index_bulk_size: int = 200
    queue_size: int = 500

    # ---- Device ----
    device: str = "auto"

    # ---- State / Checkpointing ----
    checkpoint_path: str = ".ingestion_state.db"

    # ---- Embedding Cache ----
    embed_cache_enabled: bool = True
    embed_cache_path: str = ".embed_cache.db"
    embed_cache_max_entries: int = 100_000

    # ---- Chunking (mirrors top-level config) ----
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Construct a :class:`PipelineConfig` from ``config.py`` env values.

        Importing ``config`` here (not at module level) keeps this module
        importable even when ``python-dotenv`` is not installed, which is
        useful in minimal test environments.
        """
        import config as _cfg  # project-level config

        return cls(
            loader_workers=_cfg.INGESTION_LOADER_WORKERS,
            chunker_workers=_cfg.INGESTION_CHUNKER_WORKERS,
            embed_batch_size=_cfg.INGESTION_EMBED_BATCH_SIZE,
            index_bulk_size=_cfg.INGESTION_INDEX_BULK_SIZE,
            queue_size=_cfg.INGESTION_QUEUE_SIZE,
            device=_cfg.INGESTION_DEVICE,
            checkpoint_path=_cfg.INGESTION_CHECKPOINT_PATH,
            embed_cache_enabled=_cfg.EMBED_CACHE_ENABLED,
            embed_cache_path=_cfg.EMBED_CACHE_PATH,
            embed_cache_max_entries=_cfg.EMBED_CACHE_MAX_ENTRIES,
            chunk_size=_cfg.CHUNK_SIZE,
            chunk_overlap=_cfg.CHUNK_OVERLAP,
        )
