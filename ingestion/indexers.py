"""
Vector Store / Indexer — Abstract interface and Elasticsearch implementation.

Changes from v1
---------------
* ``BaseVectorStore`` gains ``stream_index()`` — sends chunks to the store in
  configurable bulk batches, keeping memory bounded.
* ``ElasticsearchVectorStore.stream_index()`` issues one ``helpers.bulk``
  call per ``bulk_size`` chunks instead of one call for the entire dataset.
* Duplicate ``if not chunks`` guards and duplicate query bodies from v1 are
  cleaned up.
* All existing public methods remain unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:
    from elasticsearch import Elasticsearch, helpers
except (ImportError, AttributeError, OSError, Exception):
    Elasticsearch = None
    helpers = None

import config
from domain.models import Chunk, SearchResult
from ingestion.metrics import PipelineMetrics, StageTimer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    """Abstract Base Class for vector database storage and retrieval."""

    @abstractmethod
    def create_index(self, dimension: int, recreate: bool = False) -> None:
        """Create database index/collection schema."""
        pass

    @abstractmethod
    def index_chunks(self, chunks: List[Chunk]) -> int:
        """Index a batch of embedded Chunk domain objects."""
        pass

    @abstractmethod
    def vector_search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        """Perform dense vector kNN search."""
        pass

    @abstractmethod
    def text_search(self, query_text: str, top_k: int) -> List[SearchResult]:
        """Perform sparse BM25 text search."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        pass

    def stream_index(
        self,
        chunks: Iterable[Chunk],
        bulk_size: int = 200,
        metrics: Optional[PipelineMetrics] = None,
        state_store=None,  # Optional[BaseCheckpointStore]
        run_id: Optional[str] = None,
    ) -> int:
        """Index chunks in *bulk_size* batches and return total count indexed.

        Accumulates up to ``bulk_size`` chunks then calls :meth:`index_chunks`.
        Memory is bounded to ``bulk_size`` chunks at a time.

        Parameters
        ----------
        chunks:
            Any iterable of embedded :class:`~domain.models.Chunk` objects.
        bulk_size:
            Number of chunks per :meth:`index_chunks` call.
        metrics:
            Optional :class:`~ingestion.metrics.PipelineMetrics` instance.
            ``docs_indexed`` and ``index_errors`` are updated in-place.

        Returns
        -------
        int
            Total number of chunks successfully indexed.
        """
        total_indexed = 0
        batch: List[Chunk] = []

        with StageTimer(metrics or PipelineMetrics(), "index"):
            for chunk in chunks:
                batch.append(chunk)
                if len(batch) >= bulk_size:
                    count = self._index_batch(batch, metrics, state_store, run_id)
                    total_indexed += count
                    batch.clear()

            if batch:
                count = self._index_batch(batch, metrics, state_store, run_id)
                total_indexed += count

        if metrics:
            logger.info(
                "stage=index docs_indexed=%d errors=%d time_s=%.2f",
                metrics.docs_indexed,
                metrics.index_errors,
                metrics.index_time_s,
            )

        return total_indexed

    def _index_batch(
        self,
        batch: List[Chunk],
        metrics: Optional[PipelineMetrics],
        state_store=None,
        run_id: Optional[str] = None,
    ) -> int:
        """Index one batch and update metrics. Delegates to :meth:`index_chunks`."""
        try:
            count = self.index_chunks(batch)
            if metrics:
                metrics.docs_indexed += count
            
            # If a state store is provided and there were no errors, checkpoint all chunks.
            # (A more robust implementation would parse partial errors, but this covers
            # the common success case).
            if state_store is not None and run_id is not None and count == len(batch):
                for c in batch:
                    state_store.checkpoint_chunk(run_id, c.chunk_id, "indexed")
                    
            return count
        except Exception as exc:
            logger.error("stream_index: batch indexing failed: %s", exc)
            if metrics:
                metrics.index_errors += len(batch)
            return 0


# ---------------------------------------------------------------------------
# Elasticsearch implementation
# ---------------------------------------------------------------------------

class ElasticsearchVectorStore(BaseVectorStore):
    """Elasticsearch implementation of BaseVectorStore for Hybrid Search."""

    def __init__(self, host: Optional[str] = None, index_name: Optional[str] = None):
        if Elasticsearch is None:
            logger.warning(
                "Elasticsearch client library not available. "
                "Vector store operations will be disabled."
            )
            self.host = host or config.ELASTICSEARCH_HOST
            self.index_name = index_name or config.ELASTICSEARCH_INDEX
            self.es = None
            return

        raw_host = host or config.ELASTICSEARCH_HOST
        if not raw_host.startswith(("http://", "https://")):
            raw_host = f"http://{raw_host}"
        self.host = raw_host
        self.index_name = index_name or config.ELASTICSEARCH_INDEX

        try:
            self.es = Elasticsearch(self.host)
            if not self.es.ping():
                raise ConnectionError("Elasticsearch ping failed")
        except Exception as e:
            logger.warning(
                "Elasticsearch client could not connect to %s: %s. "
                "Trying localhost as fallback.",
                self.host,
                e,
            )
            fallback_host = "http://localhost:9200"
            try:
                self.es = Elasticsearch(fallback_host)
                if not self.es.ping():
                    raise ConnectionError("Elasticsearch ping failed on fallback")
                self.host = fallback_host
            except Exception as e2:
                logger.warning(
                    "Fallback Elasticsearch connection also failed: %s. "
                    "Vector store operations will be disabled.",
                    e2,
                )
                self.es = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(self, dimension: int, recreate: bool = False) -> None:
        if self.es is None:
            logger.error("Cannot create index: Elasticsearch client is unavailable.")
            return

        if self.es.indices.exists(index=self.index_name):
            if recreate:
                logger.warning("Deleting existing index '%s'", self.index_name)
                self.es.indices.delete(index=self.index_name)
            else:
                logger.info("Index '%s' exists. Skipping creation.", self.index_name)
                return

        logger.info("Creating index '%s' with dims=%d", self.index_name, dimension)
        mapping = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "chunk_index": {"type": "integer"},
                    "source_file": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "modified_date": {"type": "date"},
                }
            },
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info("Index '%s' created successfully", self.index_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: List[Chunk]) -> int:
        if self.es is None:
            logger.error("Cannot index chunks: Elasticsearch client is unavailable.")
            return 0
        if not chunks:
            return 0

        def _generate_actions():
            for c in chunks:
                yield {
                    "_index": self.index_name,
                    "_id": c.chunk_id,
                    "_source": {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "embedding": c.embedding,
                        "chunk_index": c.chunk_index,
                        "source_file": c.source_file,
                        "file_name": c.file_name,
                        "file_type": c.file_type,
                        "page_number": c.page_number,
                        "modified_date": c.modified_date,
                    },
                }

        success_count, errors = helpers.bulk(
            self.es, _generate_actions(), raise_on_error=False, stats_only=False
        )

        if errors:
            logger.error("Bulk indexing had %d errors", len(errors))

        logger.info(
            "Indexed %d/%d chunks into '%s'",
            success_count,
            len(chunks),
            self.index_name,
        )
        return success_count

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def vector_search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        if self.es is None:
            logger.error("Cannot perform vector search: Elasticsearch client is unavailable.")
            return []
        body = {
            "size": top_k,
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "_source": {"excludes": ["embedding"]},
        }
        response = self.es.search(index=self.index_name, body=body)
        return self._parse_search_hits(response)

    def text_search(self, query_text: str, top_k: int) -> List[SearchResult]:
        if self.es is None:
            logger.error("Cannot perform text search: Elasticsearch client is unavailable.")
            return []
        body = {
            "size": top_k,
            "query": {"match": {"text": {"query": query_text}}},
            "_source": {"excludes": ["embedding"]},
        }
        response = self.es.search(index=self.index_name, body=body)
        return self._parse_search_hits(response)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        if self.es is None:
            return {"exists": False, "doc_count": 0}
        if not self.es.indices.exists(index=self.index_name):
            return {"exists": False, "doc_count": 0}
        stats = self.es.indices.stats(index=self.index_name)
        doc_count = stats["indices"][self.index_name]["primaries"]["docs"]["count"]
        return {"exists": True, "doc_count": doc_count}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_search_hits(response: Dict[str, Any]) -> List[SearchResult]:
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit["_source"]
            results.append(
                SearchResult(
                    chunk_id=src.get("chunk_id", ""),
                    doc_id=src.get("doc_id", ""),
                    text=src.get("text", ""),
                    file_name=src.get("file_name", ""),
                    source_file=src.get("source_file", ""),
                    file_type=src.get("file_type", ""),
                    page_number=src.get("page_number"),
                    score=float(hit.get("_score", 0.0)),
                )
            )
        return results
