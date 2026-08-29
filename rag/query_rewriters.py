"""
Query Rewriter Services — Abstract interface & LLM implementation.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional

import config
from rag.llm_providers import BaseLLMProvider, create_llm_provider

logger = logging.getLogger(__name__)


class BaseQueryRewriter(ABC):
    """Abstract Base Class for query rewriting strategies."""

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Rewrite a user query into a search-optimized query."""
        pass


class NoOpQueryRewriter(BaseQueryRewriter):
    """Pass-through query rewriter."""

    def rewrite(self, query: str) -> str:
        return query


class LLMQueryRewriter(BaseQueryRewriter):
    """Query rewriter that uses any BaseLLMProvider for query expansion.

    The provider is injected at construction time, so it can be swapped or
    mocked in tests without changing this class.  When no provider is
    supplied, the configured default provider is used (Groq or Ollama,
    depending on ``LLM_PROVIDER``).
    """

    SYSTEM_PROMPT = (
        "You are a search query optimizer. Your job is to rewrite the user's question into a "
        "better search query that will retrieve the most relevant documents from an internal knowledge base.\n\n"
        "Rules:\n"
        "1. Expand abbreviations and add relevant search terms\n"
        "2. Keep it concise — output ONLY the rewritten query, nothing else\n"
        "3. Do NOT answer the question, only rewrite it\n"
        "4. If the query is already specific, return it as-is"
    )

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None, model: str = None):
        # Lazily create the provider so that importing this module never
        # performs network I/O or credential validation.
        self._provider = llm_provider
        self._model = model  # provider-level override (optional)

    @property
    def provider(self) -> BaseLLMProvider:
        if self._provider is None:
            self._provider = create_llm_provider()
        return self._provider

    def rewrite(self, query: str) -> str:
        try:
            result = self.provider.generate(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                model=self._model,   # None → provider uses its own default
                temperature=0.0,
                max_tokens=150,
            )
            rewritten = result["answer"]
            if rewritten:
                logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}. Falling back to original query.")
        return query
