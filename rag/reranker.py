"""
Re-ranker — Cross-encoder re-ranking of retrieved documents.

Uses a local cross-encoder model (sentence-transformers) to re-score
retrieved chunks by their relevance to the original query.
This runs entirely on-device.
"""

import logging

from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)

# Module-level cache
_cross_encoder: CrossEncoder | None = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder model loaded")
    return _cross_encoder


def rerank(
    query: str,
    results: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """
    Re-rank retrieved results using a cross-encoder.

    Args:
        query: The original user query.
        results: List of retrieval results (each must have a 'text' field).
        top_n: Number of top results to return after re-ranking.

    Returns:
        Re-ranked list of results, sorted by cross-encoder score (highest first).
        Each result gets a 'rerank_score' field added.
    """
    top_n = top_n or config.RERANK_TOP_N

    if not results:
        return []

    if len(results) <= 1:
        return results

    cross_encoder = _get_cross_encoder()

    # Create query-document pairs for scoring
    pairs = [(query, r["text"]) for r in results]
    scores = cross_encoder.predict(pairs)

    # Attach scores and sort
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    logger.info(
        f"Re-ranked {len(results)} results → returning top {top_n}. "
        f"Score range: [{reranked[-1]['rerank_score']:.3f}, {reranked[0]['rerank_score']:.3f}]"
    )

    return reranked[:top_n]
