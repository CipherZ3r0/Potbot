"""
Retriever — Vector, text, and hybrid search against Elasticsearch.

Supports three retrieval modes:
  - vector: kNN cosine similarity search on embeddings
  - text: BM25 keyword search on text field
  - hybrid: Reciprocal Rank Fusion (RRF) combining both
"""

import logging
from elasticsearch import Elasticsearch

import config
from ingestion.embedder import embed_text

logger = logging.getLogger(__name__)


def _get_client() -> Elasticsearch:
    return Elasticsearch(config.ELASTICSEARCH_HOST)


def vector_search(
    query: str,
    top_k: int | None = None,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """
    Perform kNN vector similarity search.

    Embeds the query locally, then searches Elasticsearch's dense_vector field.
    """
    es = es or _get_client()
    top_k = top_k or config.TOP_K_RETRIEVAL

    query_vector = embed_text(query)

    body = {
        "size": top_k,
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 10,
        },
        "_source": {
            "excludes": ["embedding"]
        },
    }

    response = es.search(index=config.ELASTICSEARCH_INDEX, body=body)
    return _parse_hits(response)


def text_search(
    query: str,
    top_k: int | None = None,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """
    Perform BM25 keyword search on the text field.
    """
    es = es or _get_client()
    top_k = top_k or config.TOP_K_RETRIEVAL

    body = {
        "size": top_k,
        "query": {
            "match": {
                "text": {
                    "query": query,
                }
            }
        },
        "_source": {
            "excludes": ["embedding"]
        },
    }

    response = es.search(index=config.ELASTICSEARCH_INDEX, body=body)
    return _parse_hits(response)


def hybrid_search(
    query: str,
    top_k: int | None = None,
    es: Elasticsearch | None = None,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[dict]:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).

    Combines vector search and text search results by fusing their rankings.
    Each result gets a score of: sum(weight / (rank + 60)) across both lists.
    """
    es = es or _get_client()
    top_k = top_k or config.TOP_K_RETRIEVAL

    # Fetch more candidates from each source for better fusion
    fetch_k = top_k * 3

    vector_results = vector_search(query, top_k=fetch_k, es=es)
    text_results = text_search(query, top_k=fetch_k, es=es)

    # RRF fusion
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}
    rrf_k = 60  # Standard RRF constant

    for rank, result in enumerate(vector_results):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + vector_weight / (rank + rrf_k)
        result_map[chunk_id] = result

    for rank, result in enumerate(text_results):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + text_weight / (rank + rrf_k)
        result_map[chunk_id] = result

    # Sort by fused score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids[:top_k]:
        result = result_map[chunk_id]
        result["rrf_score"] = rrf_scores[chunk_id]
        results.append(result)

    return results


def _parse_hits(response: dict) -> list[dict]:
    """Parse Elasticsearch response into a clean list of result dicts."""
    results = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit["_source"]
        source["score"] = hit.get("_score", 0)
        results.append(source)
    return results


# Convenience dispatcher
SEARCH_METHODS = {
    "vector": vector_search,
    "text": text_search,
    "hybrid": hybrid_search,
}


def search(
    query: str,
    method: str = "hybrid",
    top_k: int | None = None,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """
    Unified search interface.

    Args:
        query: The user's question.
        method: One of 'vector', 'text', or 'hybrid'.
        top_k: Number of results to return.
        es: Optional Elasticsearch client.
    """
    search_fn = SEARCH_METHODS.get(method)
    if not search_fn:
        raise ValueError(f"Unknown search method: {method}. Use: {list(SEARCH_METHODS)}")
    return search_fn(query, top_k=top_k, es=es)
