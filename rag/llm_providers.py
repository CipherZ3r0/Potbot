"""
LLM Providers — Abstract interface and Groq implementation for text generation.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, Any, List

from groq import Groq

import config

logger = logging.getLogger(__name__)


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
        """Generate response from LLM API and return answer text with usage metadata."""
        pass


class GroqLLMProvider(BaseLLMProvider):
    """Groq API implementation of BaseLLMProvider."""

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

        logger.info(f"Groq generation finished in {elapsed_ms}ms using model={target_model} ({total_tokens} tokens)")

        return {
            "answer": answer,
            "model": target_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_time_ms": elapsed_ms,
        }
