"""
LLM Providers — Abstract interface, Groq implementation, Ollama implementation,
and a provider factory.

Design notes
------------
* BaseLLMProvider is the sole abstraction consumed by the rest of the codebase.
* OllamaLLMProvider uses Ollama's OpenAI-compatible REST endpoint
  (POST /v1/chat/completions) so the same interface can serve any
  OpenAI-compatible local runtime (LM Studio, vLLM, LocalAI, …) by just
  changing the base URL — no changes to RAG/business logic required.
* create_llm_provider() is the single entry-point for application code;
  individual providers are still importable for testing / explicit DI.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, Any, List

from groq import Groq
import requests

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM generation providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Generate response from LLM API and return answer text with usage metadata.

        Returns
        -------
        dict with keys:
            answer          : str   — generated text
            model           : str   — model identifier used
            prompt_tokens   : int
            completion_tokens: int
            total_tokens    : int
            response_time_ms: int
        """


# ---------------------------------------------------------------------------
# Groq implementation (unchanged)
# ---------------------------------------------------------------------------


class GroqLLMProvider(BaseLLMProvider):
    """Groq cloud API implementation of BaseLLMProvider."""

    def __init__(self, api_key: str = None, default_model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.default_model = default_model or config.LLM_MODEL
        self._client: Groq | None = None

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        target_model = model or self.default_model
        client = self._get_client()

        start_time = time.time()
        response = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_ms = int((time.time() - start_time) * 1000)

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        logger.info(
            f"Groq generation finished in {elapsed_ms}ms "
            f"using model={target_model} ({total_tokens} tokens)"
        )

        return {
            "answer": answer,
            "model": target_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_time_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Ollama implementation
# ---------------------------------------------------------------------------


class OllamaLLMProvider(BaseLLMProvider):
    """Local Ollama implementation of BaseLLMProvider.

    Uses Ollama's OpenAI-compatible endpoint (POST /v1/chat/completions).
    This endpoint is stable since Ollama v0.1.24 and mirrors the OpenAI API
    surface, making it straightforward to swap for other OpenAI-compatible
    runtimes (LM Studio, vLLM, LocalAI) by changing ``base_url`` alone.

    Usage metadata is normalised into the same BaseLLMProvider contract.
    Ollama returns ``prompt_eval_count`` / ``eval_count`` in the
    ``x-ollama-*`` extension fields as well as the standard OpenAI
    ``usage`` object — we prefer the standard ``usage`` object and fall
    back to zero gracefully when it is absent (older Ollama builds).
    """

    def __init__(self, base_url: str = None, default_model: str = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.default_model = default_model or config.OLLAMA_MODEL

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        target_model = model or self.default_model
        url = f"{self.base_url}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        start_time = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`) "
                "or set OLLAMA_BASE_URL correctly."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned an error: {exc.response.status_code} "
                f"{exc.response.text}"
            ) from exc

        elapsed_ms = int((time.time() - start_time) * 1000)
        data = resp.json()

        answer = data["choices"][0]["message"]["content"].strip()

        # Normalise usage — OpenAI-compatible format returned by Ollama ≥ 0.1.24
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", 0))

        logger.info(
            f"Ollama generation finished in {elapsed_ms}ms "
            f"using model={target_model} ({total_tokens} tokens)"
        )

        return {
            "answer": answer,
            "model": target_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_time_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP: Dict[str, type] = {
    "groq": GroqLLMProvider,
    "ollama": OllamaLLMProvider,
}


def create_llm_provider(provider: str = None) -> BaseLLMProvider:
    """Instantiate and return the configured LLM provider.

    Parameters
    ----------
    provider:
        Override string (``"groq"`` or ``"ollama"``).
        When omitted, ``config.LLM_PROVIDER`` is used (default: ``"groq"``).

    Raises
    ------
    ValueError
        If the requested provider name is not recognised.
    """
    name = (provider or config.LLM_PROVIDER).lower()
    cls = _PROVIDER_MAP.get(name)
    if cls is None:
        supported = ", ".join(_PROVIDER_MAP)
        raise ValueError(
            f"Unknown LLM provider '{name}'. Supported: {supported}. "
            "Set LLM_PROVIDER to one of these values."
        )
    logger.info(f"Using LLM provider: {name}")
    return cls()
