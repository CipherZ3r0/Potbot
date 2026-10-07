"""
Retrieval Evaluation — Evaluates multiple retrieval approaches
using Hit Rate and MRR metrics against ground truth data.

Compares: vector-only, text-only, hybrid, and hybrid+reranking.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence, Union

import pandas as pd

import config
from rag.retrievers import SearchStrategyFactory
from rag.rerankers import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Metric depths evaluated for every method. Retrieval top_k is NOT changed;
# Hit@k/MRR@k are computed over whatever the (unchanged) strategy returns.
METRIC_K_VALUES = [1, 3, 5, 10]
MRR_K_VALUES = [5, 10]

# Methods evaluated by this harness (order preserved).
EVAL_METHODS = ["vector", "text", "hybrid", "hybrid_rerank"]


def _normalize_expected(expected_chunk_id: Union[str, Sequence[str]]) -> List[str]:
    """Accept either a single chunk_id or a sequence of chunk_ids."""
    if isinstance(expected_chunk_id, str):
        return [expected_chunk_id]
    return list(expected_chunk_id)


def _extract_chunk_id(result) -> str:
    """
    Extract the chunk_id from either a ``SearchResult`` dataclass or a dict.

    The retrieval strategies return ``@dataclass SearchResult`` objects
    (``domain.models``), so attribute access is the primary path. Dict-style
    results (``{"chunk_id": ...}``) are still supported for backends that
    produce plain dicts. Returns ``""`` when no chunk_id can be found.
    """
    if isinstance(result, dict):
        return result.get("chunk_id", "")
    chunk_id = getattr(result, "chunk_id", None)
    return chunk_id if chunk_id is not None else ""


def hit_rate_at_k(
    results: list, expected_chunk_id: Union[str, Sequence[str]], k: int = 5
) -> float:
    """
    Binary: 1.0 if any expected chunk_id appears in the top-K results.

    ``expected_chunk_id`` may be a single id or a sequence of ids (multiple
    relevant results). When several ids are supplied, a match on any of them
    counts as a hit.
    """
    result_ids = [_extract_chunk_id(r) for r in results[:k]]
    expected_ids = _normalize_expected(expected_chunk_id)
    if not expected_ids:
        return 0.0
    return 1.0 if any(e in result_ids for e in expected_ids) else 0.0


def mrr_at_k(
    results: list, expected_chunk_id: Union[str, Sequence[str]], k: int = 5
) -> float:
    """
    Reciprocal Rank: 1/rank of the highest-ranked relevant result in top-K.

    If ``expected_chunk_id`` is a sequence, MRR uses the first (highest-ranked)
    relevant result found — standard MRR for binary relevance.
    """
    expected_ids = _normalize_expected(expected_chunk_id)
    for rank, result in enumerate(results[:k], start=1):
        if _extract_chunk_id(result) in expected_ids:
            return 1.0 / rank
    return 0.0


def _compute_summary(data: dict, n_total_queries: int) -> dict:
    """Aggregate per-method metrics, separating successful vs failed queries.

    Failed evaluations are recorded explicitly and reported separately; they
    are NEVER folded into the metric denominator as zero-performance scores.
    """
    n_successful = len(data["hits"][5])
    n_errors = len(data["errors"])

    summary = {
        "n_queries": n_total_queries,
        "n_successful": n_successful,
        "n_errors": n_errors,
        "errors": data["errors"],
    }

    if n_successful > 0:
        for k in METRIC_K_VALUES:
            summary[f"hit_rate_at_{k}"] = sum(data["hits"][k]) / n_successful
        for k in MRR_K_VALUES:
            summary[f"mrr_at_{k}"] = sum(data["mrrs"][k]) / n_successful
        # Backwards-compatible aliases (historical Hit Rate@5 / MRR@5).
        summary["hit_rate"] = summary["hit_rate_at_5"]
        summary["mrr"] = summary["mrr_at_5"]
    else:
        summary["status"] = "no_successful_queries"

    return summary


def evaluate_retrieval(
    ground_truth_path: str = "data/ground_truth.json",
    output_path: str = "data/retrieval_eval_results.json",
    top_k: int = 5,
) -> dict:
    """
    Evaluate multiple retrieval methods against ground truth.

    Methods evaluated (unchanged):
      1. vector — kNN only
      2. text — BM25 only
      3. hybrid — RRF (vector + text)
      4. hybrid_rerank — RRF + cross-encoder re-ranking

    Retrieval depth is controlled by ``top_k`` (kept at its existing default
    of 5). Evaluation failures are recorded per query under ``errors`` and
    reported separately — they never masquerade as genuine 0.0 metrics.

    Returns a summary dict with per-method metrics plus evaluation errors.
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        logger.error("Run ground_truth_generator.py first")
        return {}

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    if not ground_truth:
        logger.error("Ground truth is empty")
        return {}

    logger.info(f"Evaluating {len(ground_truth)} queries across 4 retrieval methods...")

    all_results = {
        method: {
            "hits": {k: [] for k in METRIC_K_VALUES},
            "mrrs": {k: [] for k in MRR_K_VALUES},
            "errors": [],
        }
        for method in EVAL_METHODS
    }

    for i, gt in enumerate(ground_truth):
        question = gt["question"]
        expected_id = gt["chunk_id"]

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(ground_truth)}")

        for method in EVAL_METHODS:
            try:
                if method == "hybrid_rerank":
                    hybrid_strategy = SearchStrategyFactory.get_strategy("hybrid")
                    hybrid_results = hybrid_strategy.search(question, top_k=top_k * 2)
                    reranker = CrossEncoderReranker()
                    results = reranker.rerank(question, hybrid_results, top_n=top_k)
                else:
                    strategy = SearchStrategyFactory.get_strategy(method)
                    results = strategy.search(question, top_k=top_k)

                for k in METRIC_K_VALUES:
                    all_results[method]["hits"][k].append(
                        hit_rate_at_k(results, expected_id, k=k)
                    )
                for k in MRR_K_VALUES:
                    all_results[method]["mrrs"][k].append(
                        mrr_at_k(results, expected_id, k=k)
                    )
            except Exception as e:
                all_results[method]["errors"].append(
                    {
                        "query_index": i,
                        "question": str(question)[:200],
                        "method": method,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                logger.error(
                    "Evaluation failed for method=%s query_index=%d question=%r: %s",
                    method,
                    i,
                    str(question)[:120],
                    e,
                )

    # Compute summary
    summary = {}
    for method in EVAL_METHODS:
        summary[method] = _compute_summary(all_results[method], len(ground_truth))

    # Find best method, considering only methods with successful evaluations.
    eligible = [
        m for m in EVAL_METHODS if summary[m]["n_successful"] > 0
    ]
    if eligible:
        best_method = max(eligible, key=lambda m: summary[m]["mrr"])
        summary["best_method"] = best_method
    else:
        best_method = None
        summary["best_method"] = None

    # Log results table (skipping methods that produced no successful queries).
    logger.info("\n=== Retrieval Evaluation Results ===")
    table_rows = {
        m: {
            "Hit@1": summary[m].get("hit_rate_at_1"),
            "Hit@3": summary[m].get("hit_rate_at_3"),
            "Hit@5": summary[m].get("hit_rate_at_5"),
            "Hit@10": summary[m].get("hit_rate_at_10"),
            "MRR@5": summary[m].get("mrr_at_5"),
            "MRR@10": summary[m].get("mrr_at_10"),
        }
        for m in EVAL_METHODS
        if summary[m]["n_successful"] > 0
    }
    if table_rows:
        df = pd.DataFrame(table_rows).T
        logger.info(f"\n{df.to_string()}")
    logger.info(f"\nBest method: {best_method}")

    # Log an explicit, loud summary of any evaluation errors.
    total_errors = sum(summary[m]["n_errors"] for m in EVAL_METHODS)
    if total_errors:
        logger.error(
            "Evaluation errors: %d failed query evaluation(s). "
            "Affected methods are reported under 'errors' and excluded from metrics.",
            total_errors,
        )
        for m in EVAL_METHODS:
            if summary[m]["n_errors"]:
                logger.error(
                    "  %s: %d/%d queries failed (metrics computed over %d successful) — see 'errors' in output JSON.",
                    m,
                    summary[m]["n_errors"],
                    summary[m]["n_queries"],
                    summary[m]["n_successful"],
                )

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to '{output_path}'")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth.json"
    evaluate_retrieval(ground_truth_path=gt_path)