"""Unit tests for indexers — ElasticsearchVectorStore (mocked ES client)."""

import pytest
from unittest.mock import MagicMock, patch
from ingestion.indexers import ElasticsearchVectorStore
from ingestion.metrics import PipelineMetrics


class TestElasticsearchVectorStore:
    def _get_store(self):
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        store = ElasticsearchVectorStore()
        store.es = mock_es
        return store, mock_es

    def test_create_index(self):
        store, mock_es = self._get_store()
        mock_es.indices.exists.return_value = False
        store.create_index(dimension=384)
        mock_es.indices.create.assert_called_once()

    def test_recreate_index(self):
        store, mock_es = self._get_store()
        mock_es.indices.exists.return_value = True
        store.create_index(dimension=384, recreate=True)
        mock_es.indices.delete.assert_called_once()
        mock_es.indices.create.assert_called_once()

    @patch("ingestion.indexers.helpers")
    def test_index_chunks(self, mock_helpers, sample_chunks):
        store, mock_es = self._get_store()
        mock_helpers.bulk.return_value = (len(sample_chunks), [])
        
        count = store.index_chunks(sample_chunks)
        assert count == len(sample_chunks)
        mock_helpers.bulk.assert_called_once()

    @patch("ingestion.indexers.helpers")
    def test_stream_index(self, mock_helpers, sample_chunks):
        store, mock_es = self._get_store()
        def mock_bulk(client, actions, **kwargs):
            return (len(list(actions)), [])
        mock_helpers.bulk.side_effect = mock_bulk
        metrics = PipelineMetrics()
        
        count = store.stream_index(sample_chunks, bulk_size=2, metrics=metrics)
        assert count == 3
        assert mock_helpers.bulk.call_count == 2
        assert metrics.docs_indexed == 3

    def test_vector_search(self):
        store, mock_es = self._get_store()
        mock_es.search.return_value = {
            "hits": {"hits": [
                {"_source": {"chunk_id": "c1"}, "_score": 0.9}
            ]}
        }
        results = store.vector_search([0.1]*384, top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_get_stats(self):
        store, mock_es = self._get_store()
        mock_es.indices.exists.return_value = True
        mock_es.indices.stats.return_value = {
            "indices": {
                store.index_name: {"primaries": {"docs": {"count": 42}}}
            }
        }
        stats = store.get_stats()
        assert stats["exists"] is True
        assert stats["doc_count"] == 42
