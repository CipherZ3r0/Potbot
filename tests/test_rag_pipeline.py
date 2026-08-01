"""Unit tests for the RAGPipeline orchestrator."""

import pytest
from unittest.mock import MagicMock
from rag.pipeline import RAGPipeline


class TestRAGPipeline:
    def test_query_flow(self, sample_search_results):
        mock_search = MagicMock()
        mock_search.search.return_value = sample_search_results
        
        mock_rerank = MagicMock()
        mock_rerank.rerank.return_value = sample_search_results[:2]
        
        mock_rewrite = MagicMock()
        mock_rewrite.rewrite.return_value = "rewritten"
        
        mock_prompt = MagicMock()
        mock_prompt.build_prompt.return_value = [{"role": "user", "content": "p"}]
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "answer": "response", "model": "test-model",
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "response_time_ms": 100
        }
        
        mock_repo = MagicMock()
        mock_repo.save_conversation.return_value = 123
        
        pipeline = RAGPipeline(
            search_strategy=mock_search,
            reranker=mock_rerank,
            query_rewriter=mock_rewrite,
            prompt_builder=mock_prompt,
            llm_provider=mock_llm,
            repository=mock_repo
        )
        
        resp = pipeline.query("q", use_reranking=True, use_query_rewriting=True, save_to_db=True)
        
        assert resp.answer == "response"
        assert resp.rewritten_query == "rewritten"
        assert len(resp.retrieved_docs) == 2
        assert resp.conversation_id == 123
        
        mock_search.search.assert_called_once()
        mock_rerank.rerank.assert_called_once()
        mock_rewrite.rewrite.assert_called_once()
        mock_prompt.build_prompt.assert_called_once()
        mock_llm.generate.assert_called_once()
        mock_repo.save_conversation.assert_called_once()

    def test_query_flow_disabled_features(self, sample_search_results):
        mock_search = MagicMock()
        mock_search.search.return_value = sample_search_results
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "answer": "response", "model": "test-model",
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "response_time_ms": 100
        }
        
        pipeline = RAGPipeline(
            search_strategy=mock_search,
            llm_provider=mock_llm,
            repository=None
        )
        
        # Test with built-in NoOp fallbacks via parameters
        resp = pipeline.query(
            "q", 
            use_reranking=False, 
            use_query_rewriting=False, 
            save_to_db=False
        )
        
        assert resp.rewritten_query is None
        assert len(resp.retrieved_docs) == 3  # Not reranked
