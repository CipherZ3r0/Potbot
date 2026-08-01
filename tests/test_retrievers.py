"""Unit tests for search retrieval strategies (mocked Elasticsearch)."""

import pytest
from unittest.mock import MagicMock, patch
from domain.models import SearchResult
from rag.retrievers import (
    HybridSearchStrategy, SearchStrategyFactory,
    TextSearchStrategy, VectorSearchStrategy,
)


def _make_results(ids):
    return [SearchResult(chunk_id=cid, doc_id="d1", text=f"text_{cid}",
            file_name="f.txt", source_file="/f.txt", file_type=".txt", score=1.0 / (i + 1))
            for i, cid in enumerate(ids)]


class TestVectorSearchStrategy:
    def test_calls_vector_search(self):
        mock_emb = MagicMock()
        mock_emb.embed_text.return_value = [0.1] * 384
        mock_store = MagicMock()
        mock_store.vector_search.return_value = _make_results(["a", "b"])
        strategy = VectorSearchStrategy(embedder=mock_emb, vector_store=mock_store)
        results = strategy.search("query", top_k=5)
        assert len(results) == 2
        mock_store.vector_search.assert_called_once()


class TestTextSearchStrategy:
    def test_calls_text_search(self):
        mock_store = MagicMock()
        mock_store.text_search.return_value = _make_results(["x", "y"])
        strategy = TextSearchStrategy(vector_store=mock_store)
        results = strategy.search("query", top_k=5)
        assert len(results) == 2
        mock_store.text_search.assert_called_once()


class TestHybridSearchStrategy:
    def test_fuses_results_with_rrf(self):
        mock_emb = MagicMock()
        mock_emb.embed_text.return_value = [0.1] * 384
        mock_store = MagicMock()
        mock_store.vector_search.return_value = _make_results(["a", "b", "c"])
        mock_store.text_search.return_value = _make_results(["b", "c", "d"])
        strategy = HybridSearchStrategy(embedder=mock_emb, vector_store=mock_store)
        results = strategy.search("query", top_k=3)
        assert len(results) == 3
        # b and c appear in both, so they should be scored higher
        ids = [r.chunk_id for r in results]
        assert "b" in ids and "c" in ids

    def test_rrf_scores_are_set(self):
        mock_emb = MagicMock()
        mock_emb.embed_text.return_value = [0.1] * 384
        mock_store = MagicMock()
        mock_store.vector_search.return_value = _make_results(["a"])
        mock_store.text_search.return_value = _make_results(["a"])
        strategy = HybridSearchStrategy(embedder=mock_emb, vector_store=mock_store)
        results = strategy.search("q", top_k=1)
        assert results[0].rrf_score is not None
        assert results[0].rrf_score > 0


class TestSearchStrategyFactory:
    def test_vector(self):
        with patch("rag.retrievers.SentenceTransformerEmbedder"), \
             patch("rag.retrievers.ElasticsearchVectorStore"):
            s = SearchStrategyFactory.get_strategy("vector")
            assert isinstance(s, VectorSearchStrategy)

    def test_text(self):
        with patch("rag.retrievers.ElasticsearchVectorStore"):
            s = SearchStrategyFactory.get_strategy("text")
            assert isinstance(s, TextSearchStrategy)

    def test_hybrid(self):
        with patch("rag.retrievers.SentenceTransformerEmbedder"), \
             patch("rag.retrievers.ElasticsearchVectorStore"):
            s = SearchStrategyFactory.get_strategy("hybrid")
            assert isinstance(s, HybridSearchStrategy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown search strategy"):
            SearchStrategyFactory.get_strategy("nonexistent")
