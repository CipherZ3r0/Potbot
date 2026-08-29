"""
Tests for LLM providers:  GroqLLMProvider, OllamaLLMProvider, create_llm_provider(),
and the refactored LLMQueryRewriter.

All external I/O (Groq SDK, HTTP) is mocked so the suite runs offline.
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_groq_response(content: str = "answer text", model: str = "llama3"):
    """Build a fake Groq SDK response object."""
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_ollama_response(content: str = "ollama answer", model: str = "llama3"):
    """Build a fake Ollama /v1/chat/completions JSON response dict."""
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


# ===========================================================================
# GroqLLMProvider tests (existing path — must remain unchanged)
# ===========================================================================

class TestGroqLLMProvider:
    def test_generate_returns_correct_structure(self):
        from rag.llm_providers import GroqLLMProvider

        provider = GroqLLMProvider(api_key="fake-key", default_model="llama3")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_groq_response("Hello!")
        provider._client = mock_client

        result = provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama3",
        )

        assert result["answer"] == "Hello!"
        assert result["model"] == "llama3"
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 5
        assert result["total_tokens"] == 15
        assert isinstance(result["response_time_ms"], int)

    def test_generate_uses_default_model_when_none_passed(self):
        from rag.llm_providers import GroqLLMProvider

        provider = GroqLLMProvider(api_key="fake-key", default_model="my-default-model")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_groq_response()
        provider._client = mock_client

        provider.generate(messages=[{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-default-model"

    def test_generate_handles_missing_usage_gracefully(self):
        from rag.llm_providers import GroqLLMProvider

        provider = GroqLLMProvider(api_key="fake-key", default_model="m")
        mock_client = MagicMock()
        # usage is None — simulates edge case
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="text"))],
            usage=None,
        )
        provider._client = mock_client

        result = provider.generate(messages=[])
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0


# ===========================================================================
# OllamaLLMProvider tests
# ===========================================================================

class TestOllamaLLMProvider:
    def _make_mock_response(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_generate_returns_correct_structure(self):
        from rag.llm_providers import OllamaLLMProvider

        provider = OllamaLLMProvider(
            base_url="http://localhost:11434", default_model="llama3"
        )

        with patch("rag.llm_providers.requests.post") as mock_post:
            mock_post.return_value = self._make_mock_response(
                _make_ollama_response("Ollama says hi!")
            )
            result = provider.generate(
                messages=[{"role": "user", "content": "Hi"}]
            )

        assert result["answer"] == "Ollama says hi!"
        assert result["model"] == "llama3"
        assert result["prompt_tokens"] == 8
        assert result["completion_tokens"] == 4
        assert result["total_tokens"] == 12
        assert isinstance(result["response_time_ms"], int)

    def test_generate_calls_correct_endpoint(self):
        from rag.llm_providers import OllamaLLMProvider

        provider = OllamaLLMProvider(
            base_url="http://myollama:11434", default_model="mistral"
        )

        with patch("rag.llm_providers.requests.post") as mock_post:
            mock_post.return_value = self._make_mock_response(
                _make_ollama_response(model="mistral")
            )
            provider.generate(messages=[{"role": "user", "content": "x"}])

        called_url = mock_post.call_args[0][0]
        assert called_url == "http://myollama:11434/v1/chat/completions"

    def test_generate_uses_override_model(self):
        from rag.llm_providers import OllamaLLMProvider

        provider = OllamaLLMProvider(
            base_url="http://localhost:11434", default_model="llama3"
        )

        with patch("rag.llm_providers.requests.post") as mock_post:
            mock_post.return_value = self._make_mock_response(
                _make_ollama_response(model="gemma2")
            )
            provider.generate(
                messages=[{"role": "user", "content": "x"}], model="gemma2"
            )

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "gemma2"

    def test_generate_graceful_when_usage_absent(self):
        from rag.llm_providers import OllamaLLMProvider

        provider = OllamaLLMProvider(
            base_url="http://localhost:11434", default_model="llama3"
        )
        data_no_usage = {
            "model": "llama3",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            # no "usage" key
        }

        with patch("rag.llm_providers.requests.post") as mock_post:
            mock_post.return_value = self._make_mock_response(data_no_usage)
            result = provider.generate(messages=[])

        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_connection_error_raises_runtime_error(self):
        import requests as req
        from rag.llm_providers import OllamaLLMProvider

        provider = OllamaLLMProvider(base_url="http://nowhere:11434", default_model="x")

        with patch("rag.llm_providers.requests.post") as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError("refused")
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                provider.generate(messages=[])

    def test_trailing_slash_stripped_from_base_url(self):
        from rag.llm_providers import OllamaLLMProvider

        provider = OllamaLLMProvider(
            base_url="http://localhost:11434/", default_model="llama3"
        )
        assert not provider.base_url.endswith("/")


# ===========================================================================
# create_llm_provider() factory tests
# ===========================================================================

class TestCreateLLMProvider:
    def test_returns_groq_by_default(self):
        from rag.llm_providers import create_llm_provider, GroqLLMProvider

        with patch("rag.llm_providers.config") as mock_cfg:
            mock_cfg.LLM_PROVIDER = "groq"
            mock_cfg.GROQ_API_KEY = ""
            mock_cfg.LLM_MODEL = "llama3"
            provider = create_llm_provider()

        assert isinstance(provider, GroqLLMProvider)

    def test_returns_ollama_when_configured(self):
        from rag.llm_providers import create_llm_provider, OllamaLLMProvider

        with patch("rag.llm_providers.config") as mock_cfg:
            mock_cfg.LLM_PROVIDER = "ollama"
            mock_cfg.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_cfg.OLLAMA_MODEL = "llama3"
            provider = create_llm_provider()

        assert isinstance(provider, OllamaLLMProvider)

    def test_explicit_override_beats_config(self):
        from rag.llm_providers import create_llm_provider, OllamaLLMProvider

        with patch("rag.llm_providers.config") as mock_cfg:
            mock_cfg.LLM_PROVIDER = "groq"        # config says groq
            mock_cfg.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_cfg.OLLAMA_MODEL = "llama3"
            provider = create_llm_provider(provider="ollama")  # explicit override

        assert isinstance(provider, OllamaLLMProvider)

    def test_raises_for_unknown_provider(self):
        from rag.llm_providers import create_llm_provider

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(provider="openai_future_provider")

    def test_case_insensitive(self):
        from rag.llm_providers import create_llm_provider, GroqLLMProvider

        with patch("rag.llm_providers.config") as mock_cfg:
            mock_cfg.GROQ_API_KEY = ""
            mock_cfg.LLM_MODEL = "llama3"
            mock_cfg.LLM_PROVIDER = "GROQ"
            provider = create_llm_provider(provider="GROQ")

        assert isinstance(provider, GroqLLMProvider)


# ===========================================================================
# LLMQueryRewriter tests (refactored DI path)
# ===========================================================================

class TestLLMQueryRewriter:
    def _make_provider(self, answer: str = "rewritten query"):
        """Return a mock BaseLLMProvider that returns ``answer``."""
        provider = MagicMock()
        provider.generate.return_value = {
            "answer": answer,
            "model": "test-model",
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "response_time_ms": 50,
        }
        return provider

    def test_rewrite_uses_injected_provider(self):
        from rag.query_rewriters import LLMQueryRewriter

        provider = self._make_provider("policy documents search")
        rewriter = LLMQueryRewriter(llm_provider=provider)
        result = rewriter.rewrite("what is the policy?")

        assert result == "policy documents search"
        provider.generate.assert_called_once()

    def test_rewrite_falls_back_on_provider_error(self):
        from rag.query_rewriters import LLMQueryRewriter

        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("network down")
        rewriter = LLMQueryRewriter(llm_provider=provider)

        result = rewriter.rewrite("original query")
        assert result == "original query"

    def test_rewrite_passes_correct_messages_to_provider(self):
        from rag.query_rewriters import LLMQueryRewriter

        provider = self._make_provider("better query")
        rewriter = LLMQueryRewriter(llm_provider=provider)
        rewriter.rewrite("how many vacation days?")

        call_kwargs = provider.generate.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][0]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]

    def test_rewrite_uses_default_provider_lazily(self):
        """When no provider is injected, it is created on first access."""
        from rag.query_rewriters import LLMQueryRewriter

        fake_provider = self._make_provider("lazy provider answer")

        with patch("rag.query_rewriters.create_llm_provider", return_value=fake_provider):
            rewriter = LLMQueryRewriter()
            result = rewriter.rewrite("test query")

        assert result == "lazy provider answer"

    def test_model_override_is_forwarded(self):
        from rag.query_rewriters import LLMQueryRewriter

        provider = self._make_provider("result")
        rewriter = LLMQueryRewriter(llm_provider=provider, model="custom-model")
        rewriter.rewrite("query")

        call_kwargs = provider.generate.call_args[1]
        assert call_kwargs.get("model") == "custom-model"


# ===========================================================================
# Backward-compatibility smoke-test: existing Groq pipeline path unchanged
# ===========================================================================

class TestGroqPathUnchanged:
    """Verify the existing RAGPipeline can still be instantiated with a GroqLLMProvider."""

    def test_pipeline_accepts_explicit_groq_provider(self):
        from rag.pipeline import RAGPipeline
        from rag.llm_providers import GroqLLMProvider

        groq_provider = GroqLLMProvider(api_key="fake", default_model="m")
        # RAGPipeline should accept it without errors
        pipeline = RAGPipeline(
            llm_provider=groq_provider,
            search_strategy=MagicMock(),
            reranker=MagicMock(),
            query_rewriter=MagicMock(),
            prompt_builder=MagicMock(),
            repository=MagicMock(),
        )
        assert pipeline.llm_provider is groq_provider
