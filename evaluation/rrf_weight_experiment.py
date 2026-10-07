"""
Eval-only experiment: hybrid_rerank RRF vector/text weight pairs.

For each (vector_weight, text_weight) pair, retrieve RETRIEVAL_CANDIDATE_POOL
(20) hybrid (RRF) candidates, rerank with the SAME cross-encoder, keep top 5,
and score the SAME 26 ground-truth queries. Everything else is unchanged:
embedding model, BM25, RRF k=60, reranker top-N, query rewriting disabled,
potbot_eval index. Production retrieval defaults and config are untouched.

Usage:
    ELASTICSEARCH_INDEX=potbot_eval \
        python -m evaluation.rrf_weight_experiment data/ground_truth.json
"""

import json
import sys
from pathlib import Path

import config  # noqa: F401  (loads env; ELASTICSEARCH_INDEX selects the eval index)
from rag.retrievers import HybridSearchStrategy
from rag.rerankers import CrossEncoderReranker
from evaluation.retrieval_eval import hit_rate_at_k, mrr_at_k

WEIGHT_PAIRS = [
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
]
BASELINE = WEIGHT_PAIRS[0]  # 0.7 / 0.3 (current production weights)
TOP_N = 5
CANDIDATE_POOL = config.RETRIEVAL_CANDIDATE_POOL
RRF_K = 60
TRACKED_QUERIES = (13, 17, 19)


def expected_ids(record):
    value = record["chunk_id"]
    return [value] if isinstance(value, str) else list(value)


def run_weight_pair(strategy, reranker, gt, vector_weight, text_weight):
    hit1 = hit3 = hit5 = mrr5 = 0.0
    per_query = {}
    for i, rec in enumerate(gt):
        q = rec["question"]
        exp = expected_ids(rec)
        candidates = strategy.search(q, top_k=CANDIDATE_POOL)
        final = reranker.rerank(q, candidates, top_n=TOP_N)
        hit1 += hit_rate_at_k(final, exp, k=1)
        hit3 += hit_rate_at_k(final, exp, k=3)
        hit5 += hit_rate_at_k(final, exp, k=5)
        mrr5 += mrr_at_k(final, exp, k=5)
        per_query[i] = {"hit@5": hit_rate_at_k(final, exp, k=5) == 1.0}

    n = len(gt)
    result = {
        "vector_weight": vector_weight,
        "text_weight": text_weight,
        "candidate_pool": CANDIDATE_POOL,
        "rrf_k": RRF_K,
        "top_n": TOP_N,
        "hit_rate_at_1": hit1 / n,
        "hit_rate_at_3": hit3 / n,
        "hit_rate_at_5": hit5 / n,
        "mrr_at_5": mrr5 / n,
        "tracked_queries": {q: per_query[q]["hit@5"] for q in TRACKED_QUERIES},
        "per_query_hit5": {str(i): v["hit@5"] for i, v in sorted(per_query.items())},
    }
    return result


def summarize(reference, results):
    ref_hits = set(i for i, ok in reference["per_query_hit5"].items() if ok)
    for result in results:
        hits = set(i for i, ok in result["per_query_hit5"].items() if ok)
        recovered = sorted(hits - ref_hits)
        regressed = sorted(ref_hits - hits)
        result["recovered_vs_baseline"] = [int(i) for i in recovered]
        result["regressed_vs_baseline"] = [int(i) for i in regressed]
        result["n_recovered"] = len(recovered)
        result["n_regressed"] = len(regressed)


def print_table(results):
    print("\n=== hybrid_rerank RRF weight experiment ===")
    print(f"queries: {len(results[0]['per_query_hit5'])} | candidate_pool: {CANDIDATE_POOL} "
          f"| rrf_k: {RRF_K} | rerank top_n: {TOP_N} | model: cross-encoder (ms-marco-MiniLM-L-6-v2)")
    header = ("v_w t_w | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Q13 | Q17 | Q19 | "
              "recovered | regressed")
    print(header)
    print("-" * len(header))
    for r in results:
        tq = r["tracked_queries"]
        status = " ".join("OK" if tq[q] else "ms" for q in TRACKED_QUERIES)
        print(f"{r['vector_weight']:.1f} {r['text_weight']:.1f} | "
              f"{r['hit_rate_at_1']:.3f} | {r['hit_rate_at_3']:.3f} | "
              f"{r['hit_rate_at_5']:.3f} | {r['mrr_at_5']:.3f} |  "
              f"{status}  | {r['n_recovered']:2d}@{r['recovered_vs_baseline']} | "
              f"{r['n_regressed']:2d}@{r['regressed_vs_baseline']}")


def main():
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    reranker = CrossEncoderReranker()
    reranker._get_model()  # warm up once

    results = []
    for vector_weight, text_weight in WEIGHT_PAIRS:
        strategy = HybridSearchStrategy(
            vector_weight=vector_weight, text_weight=text_weight, rrf_k=RRF_K
        )
        results.append(run_weight_pair(strategy, reranker, gt, vector_weight, text_weight))

    reference = results[BASELINE[0] == results[0]["vector_weight"] and
                        BASELINE[1] == results[0]["text_weight"]]
    summarize(reference, results)
    print_table(results)

    out = {
        "experiment": "rrf_weight_experiment",
        "description": (
            "hybrid_rerank over RETRIEVAL_CANDIDATE_POOL candidates with varying "
            "RRF vector/text weights; same embedder, BM25, cross-encoder, k=60, "
            "top_n=5, 26 GT queries, potbot_eval index."
        ),
        "baseline_weights": {"vector": BASELINE[0], "text": BASELINE[1]},
        "candidate_pool": CANDIDATE_POOL,
        "rrf_k": RRF_K,
        "top_n": TOP_N,
        "weight_pairs": results,
    }
    out_path = Path("data/rrf_weight_experiment_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to '{out_path}'")


if __name__ == "__main__":
    main()