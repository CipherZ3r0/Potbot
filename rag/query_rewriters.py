"""
Query Rewriter Services — Abstract interface & LLM implementation.
"""

from abc import ABC, abstractmethod
import logging

from groq import Groq

import config

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
    """Query rewriter using LLM for query expansion."""

    SYSTEM_PROMPT = (
        "You are a search query optimizer. Your job is to rewrite the user's question into a "
        "better search query that will retrieve the most relevant documents from an internal knowledge base.\n\n"
        "Rules:\n"
        "1. Expand abbreviations and add relevant search terms\n"
        "2. Keep it concise — output ONLY the rewritten query, nothing else\n"
        "3. Do NOT answer the question, only rewrite it\n"
        "4. If the query is already specific, return it as-is"
    )

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.LLM_MODEL

    def rewrite(self, query: str) -> str:
        try:
            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten:
                logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}. Falling back to original query.")
        return query
