"""
Eval-only experiment: hybrid_rerank candidate pool depth.

For each candidate pool size, retrieve that many hybrid (RRF) candidates,
rerank with the SAME cross-encoder, keep top 5, and score the SAME 26
ground-truth queries. RRF weights, embeddings, chunking, and query
rewriting are left unchanged; nothing here touches production retrieval
defaults or config.

Usage:
    ELASTICSEARCH_INDEX=potbot_eval \
        python -m evaluation.candidate_depth_experiment data/ground_truth.json
"""

import json
import sys
import time

import config  # noqa: F401  (loads env; ELASTICSEARCH_INDEX selects the eval index)
from rag.retrievers import SearchStrategyFactory
from rag.rerankers import CrossEncoderReranker
from evaluation.retrieval_eval import hit_rate_at_k, mrr_at_k

POOL_SIZES = [10, 20, 50, 80]
TOP_N = 5


def expected_ids(record):
    value = record["chunk_id"]
    return [value] if isinstance(value, str) else list(value)


def main():
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    reranker = CrossEncoderReranker()
    reranker._get_model()  # warm up once, before timing
    strategy = SearchStrategyFactory.get_strategy("hybrid")

    print("\n=== hybrid_rerank candidate-depth experiment ===")
    print(f"queries: {len(gt)} | rerank top_n: {TOP_N} | RRF unchanged "
          f"(vector=0.7, text=0.3, k=60) | model: {reranker.model_name}")
    header = ("candidate_pool | Hit@1 | Hit@3 | Hit@5 | MRR@5 | "
              "retrieval(ms) | rerank(ms) | total(ms)")
    print(header)
    print("-" * len(header))

    recovery = {pool: {} for pool in POOL_SIZES}
    for pool in POOL_SIZES:
        hit1 = hit3 = hit5 = mrr5 = 0.0
        retr = rerank = 0.0
        missed = []
        for i, rec in enumerate(gt):
            q = rec["question"]
            exp = expected_ids(rec)

            t0 = time.perf_counter()
            candidates = strategy.search(q, top_k=pool)
            t1 = time.perf_counter()
            final = reranker.rerank(q, candidates, top_n=TOP_N)
            t2 = time.perf_counter()

            retr += t1 - t0
            rerank += t2 - t1
            hit1 += hit_rate_at_k(final, exp, k=1)
            hit3 += hit_rate_at_k(final, exp, k=3)
            hit5 += hit_rate_at_k(final, exp, k=5)
            mrr5 += mrr_at_k(final, exp, k=5)
            if hit_rate_at_k(final, exp, k=5) < 1.0:
                missed.append(i)
            recovery[pool][i] = hit_rate_at_k(final, exp, k=5) == 1.0

        n = len(gt)
        print(f"{pool:>14} | {hit1/n:.3f} | {hit3/n:.3f} | {hit5/n:.3f} | "
              f"{mrr5/n:.3f} | {retr/n*1000:10.1f} | {rerank/n*1000:10.1f} | "
              f"{(retr+rerank)/n*1000:10.1f}")
        if missed:
            print(f"   missed_queries@{pool}: {missed}")

    print("\n=== recovery of Q13, Q17, Q19, Q22 (Hit@5) ===")
    for pool in POOL_SIZES:
        status = ", ".join(
            f"Q{q}={'OK' if recovery[pool].get(q) else 'miss'}"
            for q in (13, 17, 19, 22)
        )
        print(f"pool {pool:>4}: {status}")


if __name__ == "__main__":
    main()