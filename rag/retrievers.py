"""
Search Strategies — Strategy pattern for vector, text (BM25), and hybrid search.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Dict

import config
from domain.models import SearchResult
from ingestion.embedders import BaseEmbedder, SentenceTransformerEmbedder
from ingestion.indexers import BaseVectorStore, ElasticsearchVectorStore

logger = logging.getLogger(__name__)


class BaseSearchStrategy(ABC):
    """Abstract Base Class for search retrieval strategies."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        """Execute search given a query string and return top_k results."""
        pass


class VectorSearchStrategy(BaseSearchStrategy):
    """Dense vector kNN similarity search strategy."""

    def __init__(self, embedder: BaseEmbedder = None, vector_store: BaseVectorStore = None):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.vector_store = vector_store or ElasticsearchVectorStore()

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        query_vector = self.embedder.embed_text(query)
        return self.vector_store.vector_search(query_vector, top_k=top_k)


class TextSearchStrategy(BaseSearchStrategy):
    """Sparse BM25 keyword search strategy."""

    def __init__(self, vector_store: BaseVectorStore = None):
        self.vector_store = vector_store or ElasticsearchVectorStore()

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        return self.vector_store.text_search(query, top_k=top_k)


class HybridSearchStrategy(BaseSearchStrategy):
    """Hybrid search strategy utilizing Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        embedder: BaseEmbedder = None,
        vector_store: BaseVectorStore = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.vector_strategy = VectorSearchStrategy(embedder, vector_store)
        self.text_strategy = TextSearchStrategy(vector_store)
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        fetch_k = top_k * 3
        vector_results = self.vector_strategy.search(query, top_k=fetch_k)
        text_results = self.text_strategy.search(query, top_k=fetch_k)

        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}

        for rank, res in enumerate(vector_results):
            cid = res.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.vector_weight / (rank + self.rrf_k)
            result_map[cid] = res

        for rank, res in enumerate(text_results):
            cid = res.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.text_weight / (rank + self.rrf_k)
            result_map[cid] = res

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for cid in sorted_ids[:top_k]:
            res = result_map[cid]
            res.rrf_score = rrf_scores[cid]
            fused_results.append(res)

        return fused_results


class SearchStrategyFactory:
    """Factory for instantiating search strategies by name."""

    @staticmethod
    def get_strategy(method_name: str) -> BaseSearchStrategy:
        name = method_name.lower()
        if name == "vector":
            return VectorSearchStrategy()
        elif name == "text":
            return TextSearchStrategy()
        elif name == "hybrid":
            return HybridSearchStrategy()
        else:
            raise ValueError(f"Unknown search strategy: '{method_name}'. Choose 'vector', 'text', or 'hybrid'.")
