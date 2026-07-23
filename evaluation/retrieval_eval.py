"""
Retrieval Evaluation — Evaluates multiple retrieval approaches
using Hit Rate and MRR metrics against ground truth data.

Compares: vector-only, text-only, hybrid, and hybrid+reranking.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

import config
from rag.retrievers import SearchStrategyFactory
from rag.rerankers import CrossEncoderReranker

logger = logging.getLogger(__name__)


def hit_rate_at_k(results: list[dict], expected_chunk_id: str, k: int = 5) -> float:
    """
    Binary: 1.0 if the expected chunk_id is in the top-K results, else 0.0.
    """
    result_ids = [r.get("chunk_id", "") for r in results[:k]]
    return 1.0 if expected_chunk_id in result_ids else 0.0


def mrr_at_k(results: list[dict], expected_chunk_id: str, k: int = 5) -> float:
    """
    Reciprocal Rank: 1/rank if found in top-K, else 0.0.
    """
    for rank, result in enumerate(results[:k], start=1):
        if result.get("chunk_id", "") == expected_chunk_id:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    ground_truth_path: str = "data/ground_truth.json",
    output_path: str = "data/retrieval_eval_results.json",
    top_k: int = 5,
) -> dict:
    """
    Evaluate multiple retrieval methods against ground truth.

    Methods evaluated:
      1. vector — kNN only
      2. text — BM25 only
      3. hybrid — RRF (vector + text)
      4. hybrid_rerank — RRF + cross-encoder re-ranking

    Returns a summary dict with per-method metrics.
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

    methods = ["vector", "text", "hybrid"]
    all_results = {m: {"hit_rates": [], "mrrs": []} for m in methods}
    all_results["hybrid_rerank"] = {"hit_rates": [], "mrrs": []}

    for i, gt in enumerate(ground_truth):
        question = gt["question"]
        expected_id = gt["chunk_id"]

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(ground_truth)}")

        for method in methods:
            try:
                strategy = SearchStrategyFactory.get_strategy(method)
                results = strategy.search(question, top_k=top_k)
                hr = hit_rate_at_k(results, expected_id, k=top_k)
                mrr = mrr_at_k(results, expected_id, k=top_k)
                all_results[method]["hit_rates"].append(hr)
                all_results[method]["mrrs"].append(mrr)
            except Exception as e:
                logger.warning(f"Search failed for method={method}: {e}")
                all_results[method]["hit_rates"].append(0.0)
                all_results[method]["mrrs"].append(0.0)

        # Hybrid + re-ranking
        try:
            hybrid_strategy = SearchStrategyFactory.get_strategy("hybrid")
            hybrid_results = hybrid_strategy.search(question, top_k=top_k * 2)
            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(question, hybrid_results, top_n=top_k)
            hr = hit_rate_at_k(reranked, expected_id, k=top_k)
            mrr = mrr_at_k(reranked, expected_id, k=top_k)
            all_results["hybrid_rerank"]["hit_rates"].append(hr)
            all_results["hybrid_rerank"]["mrrs"].append(mrr)
        except Exception as e:
            logger.warning(f"Hybrid+rerank failed: {e}")
            all_results["hybrid_rerank"]["hit_rates"].append(0.0)
            all_results["hybrid_rerank"]["mrrs"].append(0.0)

    # Compute summary
    summary = {}
    for method, data in all_results.items():
        n = len(data["hit_rates"])
        summary[method] = {
            "hit_rate": sum(data["hit_rates"]) / n if n > 0 else 0,
            "mrr": sum(data["mrrs"]) / n if n > 0 else 0,
            "n_queries": n,
        }

    # Find best method
    best_method = max(summary, key=lambda m: summary[m]["mrr"])
    summary["best_method"] = best_method

    # Log results table
    logger.info("\n=== Retrieval Evaluation Results ===")
    df = pd.DataFrame(
        {m: {"Hit Rate": f"{v['hit_rate']:.3f}", "MRR": f"{v['mrr']:.3f}"}
         for m, v in summary.items() if m != "best_method"}
    ).T
    logger.info(f"\n{df.to_string()}")
    logger.info(f"\nBest method: {best_method}")

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
