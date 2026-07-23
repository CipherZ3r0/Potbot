"""
Vector Store / Indexer — Abstract interface and Elasticsearch implementation.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None
    helpers = None

import config
from domain.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


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


class ElasticsearchVectorStore(BaseVectorStore):
    """Elasticsearch implementation of BaseVectorStore for Hybrid Search."""

    def __init__(self, host: str = None, index_name: str = None):
        # Handle missing Elasticsearch library gracefully
        if Elasticsearch is None:
            logger.warning("Elasticsearch client library not available. Vector store operations will be disabled.")
            self.host = host or config.ELASTICSEARCH_HOST
            self.index_name = index_name or config.ELASTICSEARCH_INDEX
            self.es = None
        else:
            raw_host = host or config.ELASTICSEARCH_HOST
            # Ensure scheme is present
            if not raw_host.startswith(('http://', 'https://')):
                raw_host = f'http://{raw_host}'
            self.host = raw_host
            self.index_name = index_name or config.ELASTICSEARCH_INDEX
            try:
                self.es = Elasticsearch(self.host)
                if not self.es.ping():
                    raise ConnectionError('Elasticsearch ping failed')
            except Exception as e:
                logger.warning(f"Elasticsearch client could not connect to {self.host}: {e}. Trying localhost as fallback.")
                # Fallback to localhost
                fallback_host = 'http://localhost:9200'
                try:
                    self.es = Elasticsearch(fallback_host)
                    if not self.es.ping():
                        raise ConnectionError('Elasticsearch ping failed on fallback')
                    self.host = fallback_host
                except Exception as e2:
                    logger.warning(f"Fallback Elasticsearch connection also failed: {e2}. Vector store operations will be disabled.")
                    self.es = None
                    

    def create_index(self, dimension: int, recreate: bool = False) -> None:
        if self.es is None:
            logger.error("Cannot create index because Elasticsearch client is unavailable.")
            return
        if self.es.indices.exists(index=self.index_name):
            if recreate:
                logger.warning(f"Deleting existing index '{self.index_name}'")
                self.es.indices.delete(index=self.index_name)
            else:
                logger.info(f"Index '{self.index_name}' exists. Skipping creation.")
                return
        if self.es.indices.exists(index=self.index_name):
            if recreate:
                logger.warning(f"Deleting existing index '{self.index_name}'")
                self.es.indices.delete(index=self.index_name)
            else:
                logger.info(f"Index '{self.index_name}' exists. Skipping creation.")
                return

        logger.info(f"Creating index '{self.index_name}' with dims={dimension}")
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
        logger.info(f"Index '{self.index_name}' created successfully")

    def index_chunks(self, chunks: List[Chunk]) -> int:
        if self.es is None:
            logger.error("Cannot index chunks because Elasticsearch client is unavailable.")
            return 0
        if not chunks:
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
            logger.error(f"Bulk indexing had {len(errors)} errors")

        logger.info(f"Indexed {success_count}/{len(chunks)} chunks into '{self.index_name}'")
        return success_count

    def vector_search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        if self.es is None:
            logger.error("Cannot perform vector search because Elasticsearch client is unavailable.")
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
            logger.error("Cannot perform text search because Elasticsearch client is unavailable.")
            return []
        body = {
            "size": top_k,
            "query": {"match": {"text": {"query": query_text}}},
            "_source": {"excludes": ["embedding"]},
        }
        body = {
            "size": top_k,
            "query": {"match": {"text": {"query": query_text}}},
            "_source": {"excludes": ["embedding"]},
        }
        response = self.es.search(index=self.index_name, body=body)
        return self._parse_search_hits(response)

    def get_stats(self) -> Dict[str, Any]:
        if self.es is None:
            return {"exists": False, "doc_count": 0}
        if not self.es.indices.exists(index=self.index_name):
            return {"exists": False, "doc_count": 0}
        stats = self.es.indices.stats(index=self.index_name)
        doc_count = stats["indices"][self.index_name]["primaries"]["docs"]["count"]
        return {"exists": True, "doc_count": doc_count}

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
