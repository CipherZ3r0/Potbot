"""
Eval-only experiment: reranker model comparison for hybrid_rerank.

Retrieval is fixed and cached ONCE per query (hybrid RRF 0.5/0.5, k=60,
candidate pool RETRIEVAL_CANDIDATE_POOL=20, query rewriting off). Only the
cross-encoder reranker changes per run. Same 26 ground-truth queries, same
potbot_eval index. Nothing here touches production config or the reranker
default; existing evaluation artifacts are left untouched.

Usage:
    ELASTICSEARCH_INDEX=potbot_eval \
        python -m evaluation.reranker_comparison_experiment data/ground_truth.json
"""

import json
import sys
import time
from pathlib import Path

import config  # noqa: F401  (env: ELASTICSEARCH_INDEX selects eval index)
from rag.retrievers import HybridSearchStrategy
from rag.rerankers import CrossEncoderReranker
from evaluation.retrieval_eval import hit_rate_at_k, mrr_at_k

CANDIDATE_POOL = config.RETRIEVAL_CANDIDATE_POOL
TOP_N = 5
BASELINE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Baseline first (reference), then the candidates to compare.
MODELS = [
    BASELINE_MODEL,
    "cross-encoder/ms-marco-TinyBERT-L-6",
    "BAAI/bge-reranker-v2-m3",
]

# Queries that were persistent misses in earlier investigations.
TRACKED_QUERIES = (13, 17, 19)
Q17_GT_IDS = {"f0d276af1f0945eacd43bceb49e53fab", "6890b92519177cb2d84aab85bc2d2bc8"}


def expected_ids(record):
    value = record["chunk_id"]
    return [value] if isinstance(value, str) else list(value)


def rerank_one(reranker, query, candidates, top_n=TOP_N):
    t0 = time.perf_counter()
    final = reranker.rerank(query, candidates, top_n=top_n)
    dt = time.perf_counter() - t0
    return final, dt


def summarize(final_results, exp, tracked_query_idx):
    row = {
        "hit1": float(hit_rate_at_k(final_results, exp, k=1)),
        "hit3": float(hit_rate_at_k(final_results, exp, k=3)),
        "hit5": float(hit_rate_at_k(final_results, exp, k=5)),
        "mrr5": float(mrr_at_k(final_results, exp, k=5)),
        "hit5_bool": hit_rate_at_k(final_results, exp, k=5) == 1.0,
    }
    if tracked_query_idx == 17:
        best = None
        for rank, r in enumerate(final_results, start=1):
            if r.chunk_id in Q17_GT_IDS:
                best = {"rank": rank, "score": getattr(r, "rerank_score", None)}
                break
        row["q17_best_relevant"] = best
    return row


def main():
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # 1) Cache candidate sets per query once (fixed retrieval, reranker-agnostic).
    hybrid = HybridSearchStrategy()  # config weights: vector 0.5 / text 0.5, k=60
    candidates = []
    for i, rec in enumerate(gt):
        cand = hybrid.search(rec["question"], top_k=CANDIDATE_POOL)
        candidates.append([c for c in cand])

    # 2) Run every reranker over the SAME cached candidates.
    all_rows = {m: [] for m in MODELS}
    latencies = {m: [] for m in MODELS}
    errors = {m: [] for m in MODELS}

    for model_name in MODELS:
        print(f"\n=== reranker: {model_name} ===", flush=True)
        reranker = CrossEncoderReranker(model_name=model_name)
        reranker._get_model()  # load + (first predict warms the model)
        # warm-up call outside timing
        reranker.rerank("warm up warm up", candidates[0], top_n=1)

        for i, rec in enumerate(gt):
            exp = expected_ids(rec)
            try:
                final, dt = rerank_one(reranker, rec["question"], candidates[i])
                latencies[model_name].append(dt)
                row = summarize(final, exp, i)
                row["score"] = getattr(final[0], "rerank_score", None) if final else None
                row["top_id"] = final[0].chunk_id if final else None
                all_rows[model_name].append(row)
            except Exception as e:
                errors[model_name].append({"query_index": i, "error": f"{type(e).__name__}: {e}"})
                all_rows[model_name].append(
                    {"hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr5": 0.0,
                     "hit5_bool": False, "error": f"{type(e).__name__}: {e}"}
                )

    # 3) Aggregate + compare vs baseline.
    n = len(gt)
    results = []
    baseline_hits = set(i for i, r in enumerate(all_rows[BASELINE_MODEL]) if r["hit5_bool"])
    for model_name in MODELS:
        rows = all_rows[model_name]
        agg = {
            "model": model_name,
            "hit_rate_at_1": sum(r["hit1"] for r in rows) / n,
            "hit_rate_at_3": sum(r["hit3"] for r in rows) / n,
            "hit_rate_at_5": sum(r["hit5"] for r in rows) / n,
            "mrr_at_5": sum(r["mrr5"] for r in rows) / n,
            "n_errors": len(errors[model_name]),
            "errors": errors[model_name],
            "avg_rerank_ms": (sum(latencies[model_name]) / max(len(latencies[model_name]), 1)) * 1000,
            "tracked_queries": {q: rows[q]["hit5_bool"] for q in TRACKED_QUERIES},
            "q17_best_relevant": rows[17].get("q17_best_relevant"),
        }
        if model_name != BASELINE_MODEL:
            this_hits = set(i for i, r in enumerate(rows) if r["hit5_bool"])
            agg["recovered_vs_baseline"] = sorted(this_hits - baseline_hits)
            agg["regressed_vs_baseline"] = sorted(baseline_hits - this_hits)
        else:
            agg["recovered_vs_baseline"] = []
            agg["regressed_vs_baseline"] = []
        results.append(agg)

    # 4) Print table.
    print("\n=== reranker comparison (hybrid pool %d, top_n %d, %d queries) ==="
          % (CANDIDATE_POOL, TOP_N, n))
    header = ("model | H@1 | H@3 | H@5 | MRR@5 | err | avg_ms | Q13 | Q17 | Q19 | "
              "rec | reg")
    print(header)
    print("-" * len(header))
    for r in results:
        tq = r["tracked_queries"]
        status = " ".join("OK" if tq[q] else "ms" for q in TRACKED_QUERIES)
        name = r["model"].replace("cross-encoder/", "").replace("BAAI/", "")
        print(f"{name:22s} | {r['hit_rate_at_1']:.3f} | {r['hit_rate_at_3']:.3f} | "
              f"{r['hit_rate_at_5']:.3f} | {r['mrr_at_5']:.3f} | {r['n_errors']:3d} | "
              f"{r['avg_rerank_ms']:6.1f} | {status} | {len(r['recovered_vs_baseline']):2d} | "
              f"{len(r['regressed_vs_baseline']):2d}")
        if r["model"] != BASELINE_MODEL:
            if r["recovered_vs_baseline"]:
                print(f"   recovered: Q{r['recovered_vs_baseline']}")
            if r["regressed_vs_baseline"]:
                print(f"   regressed: Q{r['regressed_vs_baseline']}")
        print(f"   Q17 best relevant: {r['q17_best_relevant']}")

    # 5) Save NEW artifact (never overwrite existing eval results).
    out = {
        "experiment": "reranker_comparison_experiment",
        "description": (
            "Fixed retrieval (hybrid RRF 0.5/0.5 k=60, pool RETRIEVAL_CANDIDATE_POOL, "
            "query rewriting off) with only the cross-encoder reranker swapped; same 26 "
            "GT queries, potbot_eval index. Candidate sets cached once per query."
        ),
        "retrieval": {
            "candidate_pool": CANDIDATE_POOL,
            "vector_rrf_weight": config.VECTOR_RRF_WEIGHT,
            "text_rrf_weight": config.TEXT_RRF_WEIGHT,
            "rrf_k": config.RRF_K,
            "top_n": TOP_N,
        },
        "baseline_model": BASELINE_MODEL,
        "n_queries": n,
        "reranker_results": results,
    }
    out_path = Path("data/reranker_comparison_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to '{out_path}'")


if __name__ == "__main__":
    main()