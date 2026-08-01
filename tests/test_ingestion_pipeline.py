"""Unit tests for the IngestionPipeline orchestrator."""

import pytest
from unittest.mock import MagicMock
from ingestion.pipeline import IngestionPipeline


class TestIngestionPipeline:
    def test_run_success(self, sample_docs_dir, mock_embedder, mock_vector_store, minimal_pipeline_config):
        pipeline = IngestionPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=minimal_pipeline_config,
            state_store=None
        )
        
        summary = pipeline.run(sample_docs_dir, recreate_index=True, incremental=False)
        
        assert summary["status"] == "success"
        assert summary["doc_count"] > 0
        assert summary["chunk_count"] > 0
        assert summary["indexed_count"] > 0
        mock_vector_store.create_index.assert_called_once_with(dimension=384, recreate=True)

    def test_run_empty_folder(self, tmp_path, mock_embedder, mock_vector_store, minimal_pipeline_config):
        pipeline = IngestionPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=minimal_pipeline_config,
            state_store=None
        )
        
        summary = pipeline.run(str(tmp_path))
        assert summary["status"] == "no_documents"
        assert summary["doc_count"] == 0

    def test_incremental_skip_all(self, sample_docs_dir, mock_embedder, mock_vector_store, minimal_pipeline_config):
        mock_state = MagicMock()
        mock_state.is_file_changed.return_value = False  # Skip everything
        
        pipeline = IngestionPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=minimal_pipeline_config,
            state_store=mock_state
        )
        
        summary = pipeline.run(sample_docs_dir, incremental=True)
        assert summary["status"] == "all_skipped"
        assert summary["files_skipped"] == summary["files_found"]
        assert summary["doc_count"] == 0

    def test_resume_run(self, sample_docs_dir, mock_embedder, mock_vector_store, minimal_pipeline_config):
        mock_state = MagicMock()
        mock_state.get_pending_chunks.side_effect = lambda run, ids: ids[:1]  # Only 1 chunk needs indexing
        
        pipeline = IngestionPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=minimal_pipeline_config,
            state_store=mock_state
        )
        
        summary = pipeline.resume_run("test-run-id", sample_docs_dir)
        
        assert summary["status"] == "success"
        # Total chunks produced > 0, but fewer should be indexed
        assert summary["chunk_count"] > 0
        # It's mocked to only index 1 chunk per batch of chunks
        assert summary["indexed_count"] >= 1
