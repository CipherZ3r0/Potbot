"""
Ingestion Pipeline Orchestrator — Streaming producer-consumer architecture.

Changes from v1
---------------
* ``run()`` now chains ``stream_directory → stream_chunks → stream_embed →
  stream_index`` so documents flow through the pipeline instead of being
  accumulated in memory between stages.
* ``PipelineConfig`` is accepted via the constructor and passed to every
  component, enabling per-run tuning without touching global state.
* ``PipelineMetrics`` is created per ``run()`` call and returned as part of
  the summary dict (new keys are additive — existing keys preserved).
* ``run_uploaded_files()`` is unchanged and continues to work exactly as before.
* All original default component construction is preserved so the existing
  ``IngestionPipeline()`` zero-argument call still works.
"""

from __future__ import annotations

import logging
import tempfile
import os
from typing import Any, Dict, Optional

from ingestion.chunkers import CompositeChunker
from ingestion.config import PipelineConfig
from ingestion.embedders import BaseEmbedder, SentenceTransformerEmbedder
from ingestion.indexers import BaseVectorStore, ElasticsearchVectorStore
from ingestion.loaders import CompositeDocumentLoader
from ingestion.metrics import PipelineMetrics

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document parsing, chunking, embedding generation, and vector indexing.

    All components are injected via the constructor, making the pipeline
    trivially testable by passing mock objects.

    Parameters
    ----------
    loader:
        Document loader.  Defaults to a ``CompositeDocumentLoader`` with all
        built-in format loaders.
    chunker:
        Text chunker.  Defaults to a ``CompositeChunker``.
    embedder:
        Embedding service.  Defaults to a ``SentenceTransformerEmbedder``.
    vector_store:
        Vector store.  Defaults to an ``ElasticsearchVectorStore``.
    config:
        Runtime configuration controlling parallelism, batch sizes, device
        selection, and caching.  Defaults to ``PipelineConfig.from_env()``.
    state_store:
        Optional checkpoint/hash store for incremental ingestion (Phase 3).
        ``None`` disables incremental mode.
    """

    def __init__(
        self,
        loader: Optional[CompositeDocumentLoader] = None,
        chunker: Optional[CompositeChunker] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        config: Optional[PipelineConfig] = None,
        state_store=None,  # Optional[BaseCheckpointStore] — imported in Phase 3
    ) -> None:
        self.config = config or PipelineConfig.from_env()
        self.loader = loader or CompositeDocumentLoader(config=self.config)
        self.chunker = chunker or CompositeChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            config=self.config,
        )
        self.embedder = embedder or SentenceTransformerEmbedder(
            device=self.config.device,
            pipeline_config=self.config,
        )
        self.vector_store = vector_store or ElasticsearchVectorStore()
        self.state_store = state_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        folder_path: str,
        recreate_index: bool = False,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """Execute the end-to-end ingestion workflow.

        Documents stream through the pipeline in bounded memory: each stage
        operates on one batch at a time rather than accumulating the full
        dataset.

        Parameters
        ----------
        folder_path:
            Absolute path to the directory containing source documents.
        recreate_index:
            When ``True``, the vector store index is deleted and re-created
            before indexing.  Useful for full re-ingestion runs.
        incremental:
            When ``True`` and a ``state_store`` is configured, files whose
            content hash has not changed since the last run are skipped.
            Pass ``False`` to force re-ingestion of all files.

        Returns
        -------
        dict
            Summary containing at minimum ``status``, ``doc_count``,
            ``chunk_count``, ``indexed_count``, and ``total_in_index``.
            Also includes all :class:`~ingestion.metrics.PipelineMetrics`
            fields as additional keys (backward-compatible additive extension).
        """
        logger.info("IngestionPipeline.run: folder=%s incremental=%s", folder_path, incremental)
        metrics = PipelineMetrics()
        metrics.start_wall_timer()

        # Begin a checkpoint run so progress can be tracked / resumed
        run_id: Optional[str] = None
        if self.state_store is not None:
            run_id = self.state_store.begin_run()

        # Resolve state store: only use it if incremental mode is requested
        active_state_store = self.state_store if incremental else None

        # ----------------------------------------------------------------
        # Stage 1 → 4: Streaming chain
        # ----------------------------------------------------------------
        doc_stream = self.loader.stream_directory(
            folder_path,
            metrics=metrics,
            state_store=active_state_store,
        )

        chunk_stream = self.chunker.stream_chunks(doc_stream, metrics=metrics)

        embed_stream = self.embedder.stream_embed(
            chunk_stream,
            batch_size=self.config.embed_batch_size,
            metrics=metrics,
        )

        # Create / verify index before streaming data into it
        dim = self.embedder.get_dimension()
        self.vector_store.create_index(dimension=dim, recreate=recreate_index)

        indexed_count = self.vector_store.stream_index(
            embed_stream,
            bulk_size=self.config.index_bulk_size,
            metrics=metrics,
            state_store=active_state_store,
            run_id=run_id,
        )

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        metrics.stop_wall_timer()
        metrics.emit_summary()

        if metrics.files_found > 0 and metrics.files_skipped == metrics.files_found:
            status = "all_skipped"
        elif indexed_count == 0 and metrics.docs_loaded == 0:
            status = "no_documents"
        elif indexed_count == 0:
            status = "no_chunks"
        else:
            status = "success"

        es_stats = self.vector_store.get_stats()
        summary: Dict[str, Any] = {
            # --- Original keys (preserved for backward compat) ---
            "status": status,
            "doc_count": metrics.docs_loaded,
            "chunk_count": metrics.chunks_produced,
            "indexed_count": indexed_count,
            "total_in_index": es_stats.get("doc_count", 0),
            # --- New metric keys (additive) ---
            **metrics.as_dict(),
        }

        logger.info("IngestionPipeline complete: %s", summary)
        return summary

    def run_uploaded_files(
        self,
        uploaded_files: list,
        recreate_index: bool = False,
    ) -> Dict[str, Any]:
        """Save uploaded file objects to a temporary directory and run ingestion.

        This method is **unchanged** from v1 — it writes files to a temp dir
        and delegates to :meth:`run`.  ``incremental`` is implicitly ``False``
        because temp files have ephemeral paths that differ each call.
        """
        if not uploaded_files:
            return {"status": "no_documents", "doc_count": 0, "chunk_count": 0}

        with tempfile.TemporaryDirectory() as temp_dir:
            for uf in uploaded_files:
                file_path = os.path.join(temp_dir, uf.name)
                content = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
                with open(file_path, "wb") as f:
                    f.write(content)
            return self.run(
                folder_path=temp_dir,
                recreate_index=recreate_index,
                incremental=False,  # temp paths change per call; skip hash check
            )

    def resume_run(
        self,
        run_id: str,
        folder_path: str,
    ) -> Dict[str, Any]:
        """Resume a previously interrupted ingestion run.

        Re-runs the loader and chunker, but skips embedding and indexing for
        chunks that were already successfully indexed in the prior run.

        Parameters
        ----------
        run_id:
            The UUID of the interrupted run (found in the logs of the prior run).
        folder_path:
            The directory to ingest (must be the same as the prior run).
        """
        if self.state_store is None:
            raise ValueError("resume_run requires a configured state_store")

        logger.info("IngestionPipeline.resume_run: run_id=%s folder=%s", run_id, folder_path)
        metrics = PipelineMetrics()
        metrics.start_wall_timer()

        # Incremental loader skip is disabled for resumes because the file
        # hash would have been updated during the first run even if chunks failed.
        # We must re-chunk everything and filter at the chunk level.
        doc_stream = self.loader.stream_directory(
            folder_path,
            metrics=metrics,
            state_store=None,  # explicitly None
        )

        chunk_stream = self.chunker.stream_chunks(doc_stream, metrics=metrics)

        # Filter out chunks that were already indexed in this run_id
        def _filter_pending(chunks, batch_size=200):
            batch = []
            for c in chunks:
                batch.append(c)
                if len(batch) >= batch_size:
                    pending_ids = self.state_store.get_pending_chunks(run_id, [x.chunk_id for x in batch])
                    pending_set = set(pending_ids)
                    yield from (x for x in batch if x.chunk_id in pending_set)
                    batch.clear()
            if batch:
                pending_ids = self.state_store.get_pending_chunks(run_id, [x.chunk_id for x in batch])
                pending_set = set(pending_ids)
                yield from (x for x in batch if x.chunk_id in pending_set)

        filtered_chunk_stream = _filter_pending(chunk_stream)

        embed_stream = self.embedder.stream_embed(
            filtered_chunk_stream,
            batch_size=self.config.embed_batch_size,
            metrics=metrics,
        )

        dim = self.embedder.get_dimension()
        self.vector_store.create_index(dimension=dim, recreate=False)

        indexed_count = self.vector_store.stream_index(
            embed_stream,
            bulk_size=self.config.index_bulk_size,
            metrics=metrics,
            state_store=self.state_store,
            run_id=run_id,
        )

        metrics.stop_wall_timer()
        metrics.emit_summary()

        es_stats = self.vector_store.get_stats()
        summary = {
            "status": "success",
            "doc_count": metrics.docs_loaded,
            "chunk_count": metrics.chunks_produced,
            "indexed_count": indexed_count,
            "total_in_index": es_stats.get("doc_count", 0),
            **metrics.as_dict(),
        }

        logger.info("IngestionPipeline complete (resume): %s", summary)
        return summary
