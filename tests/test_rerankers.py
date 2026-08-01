"""Unit tests for rerankers — NoOpReranker and CrossEncoderReranker (mocked)."""

import pytest
from unittest.mock import MagicMock, patch
from domain.models import SearchResult
from rag.rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker


class TestNoOpReranker:
    def test_returns_top_n(self, sample_search_results):
        reranked = NoOpReranker().rerank("query", sample_search_results, top_n=2)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "r1"

    def test_top_n_larger_than_list(self, sample_search_results):
        reranked = NoOpReranker().rerank("query", sample_search_results, top_n=100)
        assert len(reranked) == 3

    def test_empty_results(self):
        assert NoOpReranker().rerank("query", [], top_n=5) == []


class TestCrossEncoderReranker:
    def test_reranks_by_score(self, sample_search_results):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker = CrossEncoderReranker()
        reranker._model = mock_model
        reranked = reranker.rerank("query", sample_search_results, top_n=2)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "r2"  # highest score 0.9
        assert reranked[0].rerank_score == pytest.approx(0.9)

    def test_empty_results(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", [], top_n=5) == []

    def test_single_result(self, sample_search_results):
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", sample_search_results[:1], top_n=1)
        assert len(result) == 1
