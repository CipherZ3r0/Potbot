"""
Indexer — Manages Elasticsearch index creation and bulk document indexing.

Creates an index with both a text field (for BM25 keyword search) and a
dense_vector field (for kNN vector search), enabling hybrid retrieval.
"""

import logging
from elasticsearch import Elasticsearch, helpers

import config
from ingestion.embedder import get_embedding_dimension

logger = logging.getLogger(__name__)


def _get_client() -> Elasticsearch:
    """Create an Elasticsearch client."""
    return Elasticsearch(config.ELASTICSEARCH_HOST)


def create_index(es: Elasticsearch | None = None, recreate: bool = False) -> None:
    """
    Create the Elasticsearch index with the correct mapping.

    If `recreate` is True, deletes and rebuilds the existing index.
    """
    es = es or _get_client()
    index_name = config.ELASTICSEARCH_INDEX

    if es.indices.exists(index=index_name):
        if recreate:
            logger.warning(f"Deleting existing index '{index_name}'")
            es.indices.delete(index=index_name)
        else:
            logger.info(f"Index '{index_name}' already exists, skipping creation")
            return

    dim = get_embedding_dimension()
    logger.info(f"Creating index '{index_name}' with embedding dim={dim}")

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
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

    es.indices.create(index=index_name, body=mapping)
    logger.info(f"Index '{index_name}' created successfully")


def index_chunks(chunks: list[dict], es: Elasticsearch | None = None) -> int:
    """
    Bulk-index a list of chunk records into Elasticsearch.

    Each chunk dict must have: chunk_id, doc_id, text, embedding, and metadata fields.

    Returns the number of successfully indexed documents.
    """
    es = es or _get_client()
    index_name = config.ELASTICSEARCH_INDEX

    def _generate_actions():
        for chunk in chunks:
            yield {
                "_index": index_name,
                "_id": chunk["chunk_id"],
                "_source": {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "embedding": chunk["embedding"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "source_file": chunk.get("source_file", ""),
                    "file_name": chunk.get("file_name", ""),
                    "file_type": chunk.get("file_type", ""),
                    "page_number": chunk.get("page_number"),
                    "modified_date": chunk.get("modified_date"),
                },
            }

    success_count, errors = helpers.bulk(
        es, _generate_actions(), raise_on_error=False, stats_only=False
    )

    if errors:
        logger.error(f"Bulk indexing had {len(errors)} errors")
        for err in errors[:5]:  # Log first 5 errors
            logger.error(f"  Error: {err}")

    logger.info(
        f"Indexed {success_count}/{len(chunks)} chunks into '{index_name}'"
    )
    return success_count


def get_index_stats(es: Elasticsearch | None = None) -> dict:
    """Return basic stats about the current index."""
    es = es or _get_client()
    index_name = config.ELASTICSEARCH_INDEX

    if not es.indices.exists(index=index_name):
        return {"exists": False, "doc_count": 0}

    stats = es.indices.stats(index=index_name)
    doc_count = stats["indices"][index_name]["primaries"]["docs"]["count"]
    return {"exists": True, "doc_count": doc_count}


def delete_index(es: Elasticsearch | None = None) -> None:
    """Delete the index entirely."""
    es = es or _get_client()
    index_name = config.ELASTICSEARCH_INDEX
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        logger.info(f"Index '{index_name}' deleted")
