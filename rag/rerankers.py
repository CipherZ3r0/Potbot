"""
Reranker Services — Abstract interface & CrossEncoder implementation for document re-scoring.
"""

from abc import ABC, abstractmethod
import logging
from typing import List

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

import config
from domain.models import SearchResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract Base Class for document re-ranking services."""

    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult], top_n: int) -> List[SearchResult]:
        """Re-rank search results based on query relevance."""
        pass


class NoOpReranker(BaseReranker):
    """Pass-through re-ranker when re-ranking is disabled."""

    def rerank(self, query: str, results: List[SearchResult], top_n: int) -> List[SearchResult]:
        return results[:top_n]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder re-ranker using a local sentence-transformers cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model loaded successfully")
        return self._model

    def rerank(self, query: str, results: List[SearchResult], top_n: int) -> List[SearchResult]:
        if not results:
            return []
        if len(results) <= 1:
            return results[:top_n]

        model = self._get_model()
        pairs = [(query, res.text) for res in results]
        scores = model.predict(pairs)

        for res, score in zip(results, scores):
            res.rerank_score = float(score)

        reranked = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        logger.info(f"Re-ranked {len(results)} search results → returning top {top_n}")
        return reranked[:top_n]
