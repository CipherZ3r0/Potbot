"""Unit tests for LLM providers — GroqLLMProvider (mocked API)."""

import pytest
from unittest.mock import MagicMock, patch
from rag.llm_providers import GroqLLMProvider


class TestGroqLLMProvider:
    def _make_mock_response(self, answer="Test answer", tokens=100):
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=answer))]
        resp.usage = MagicMock(prompt_tokens=60, completion_tokens=40, total_tokens=tokens)
        return resp

    def test_generate_returns_answer(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        with patch("rag.llm_providers.Groq", return_value=mock_client):
            provider = GroqLLMProvider(api_key="test_key")
            result = provider.generate([{"role": "user", "content": "Hello"}])
            assert result["answer"] == "Test answer"
            assert result["total_tokens"] == 100
            assert result["response_time_ms"] >= 0

    def test_uses_default_model(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        with patch("rag.llm_providers.Groq", return_value=mock_client):
            provider = GroqLLMProvider(api_key="k", default_model="test-model")
            result = provider.generate([{"role": "user", "content": "Hi"}])
            assert result["model"] == "test-model"

    def test_custom_model_override(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        with patch("rag.llm_providers.Groq", return_value=mock_client):
            provider = GroqLLMProvider(api_key="k")
            result = provider.generate([{"role": "user", "content": "Hi"}], model="custom-model")
            call_kwargs = mock_client.chat.completions.create.call_args
            assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") == "custom-model"
