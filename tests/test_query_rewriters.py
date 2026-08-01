"""Unit tests for query rewriters — NoOp and LLMQueryRewriter (mocked)."""

import pytest
from unittest.mock import MagicMock, patch
from rag.query_rewriters import LLMQueryRewriter, NoOpQueryRewriter


class TestNoOpQueryRewriter:
    def test_returns_same_query(self):
        assert NoOpQueryRewriter().rewrite("what is PTO?") == "what is PTO?"

    def test_empty_query(self):
        assert NoOpQueryRewriter().rewrite("") == ""


class TestLLMQueryRewriter:
    def test_rewrites_query(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="PTO policy accrual days employee benefit"))]
        )
        with patch("rag.query_rewriters.Groq", return_value=mock_client):
            rewriter = LLMQueryRewriter(api_key="test_key")
            result = rewriter.rewrite("what is PTO?")
            assert "PTO" in result

    def test_fallback_on_error(self):
        with patch("rag.query_rewriters.Groq", side_effect=Exception("API error")):
            rewriter = LLMQueryRewriter(api_key="test_key")
            assert rewriter.rewrite("original query") == "original query"
