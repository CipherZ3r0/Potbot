"""
Eval-only experiment: LLM query rewriting (hybrid_rerank, pool=20).

Configurations (all candidate_pool=20, rerank top_n=5, SAME 26 queries):
  A. original query  -> hybrid search  -> rerank with original query
  B. rewritten query -> hybrid search  -> rerank with ORIGINAL user query

The working LLM model (qwen/qwen3.8-27b) is injected ONLY here, as an
explicit llm_provider on the existing LLMQueryRewriter. Nothing in
.env / config.py / production retrieval code is touched.

Config C (original+rewritten combined retrieval) is NOT natively supported:
retriever.search() and RAGPipeline take a single query and always return top_k
results individually; fusing two candidate lists would be a new retrieval
behavior and is therefore skipped (evaluation-only).

Usage:
    ELASTICSEARCH_INDEX=potbot_eval \
        python -m evaluation.query_rewrite_experiment data/ground_truth.json
"""

import json
import sys
import time

import config  # noqa: F401  (loads env; ELASTICSEARCH_INDEX selects eval index)
from rag.retrievers import SearchStrategyFactory
from rag.rerankers import CrossEncoderReranker
from rag.query_rewriters import LLMQueryRewriter
from rag.llm_providers import GroqLLMProvider
from evaluation.retrieval_eval import hit_rate_at_k, mrr_at_k

POOL = 20
TOP_N = 5
REWRITE_MODEL = "qwen/qwen3.8-27b"


def expected_ids(record):
    value = record["chunk_id"]
    return [value] if isinstance(value, str) else list(value)


def main():
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    reranker = CrossEncoderReranker()
    reranker._get_model()
    strategy = SearchStrategyFactory.get_strategy("hybrid")
    rewriter = LLMQueryRewriter(
        llm_provider=GroqLLMProvider(default_model=REWRITE_MODEL)
    )

    results = {"A": {}, "B": {}}
    per_query = []

    for i, rec in enumerate(gt):
        orig = rec["question"]
        exp = expected_ids(rec)

        # probe rewrite separately so a failure never silently masks B
        try:
            t0 = time.perf_counter()
            rewritten = rewriter.rewrite(orig)
            t_rewrite = time.perf_counter() - t0
            rewrite_failed = False
        except Exception as e:
            rewritten = orig
            t_rewrite = 0.0
            rewrite_failed = True

        if orig.strip() == rewritten.strip():
            rewrite_unchanged = True
        else:
            rewrite_unchanged = False

        # A
        t0 = time.perf_counter()
        candA = strategy.search(orig, top_k=POOL)
        t1 = time.perf_counter()
        finalA = reranker.rerank(orig, candA, top_n=TOP_N)
        t2 = time.perf_counter()

        # B
        t3 = time.perf_counter()
        candB = strategy.search(rewritten, top_k=POOL)
        t4 = time.perf_counter()
        finalB = reranker.rerank(orig, candB, top_n=TOP_N)
        t5 = time.perf_counter()

        okA = hit_rate_at_k(finalA, exp, k=5) == 1.0
        okB = hit_rate_at_k(finalB, exp, k=5) == 1.0
        relevant_changed = okA != okB

        per_query.append({
            "index": i,
            "question": orig,
            "rewritten": rewritten,
            "rewrite_failed": rewrite_failed,
            "rewrite_unchanged": rewrite_unchanged,
            "rewrite_s": t_rewrite,
            "A": {"ok5": okA, "hits": {k: hit_rate_at_k(finalA, exp, k=k) for k in (1, 3, 5)},
                  "mrr5": mrr_at_k(finalA, exp, k=5),
                  "retr_s": t1 - t0, "rerank_s": t2 - t1},
            "B": {"ok5": okB, "hits": {k: hit_rate_at_k(finalB, exp, k=k) for k in (1, 3, 5)},
                  "mrr5": mrr_at_k(finalB, exp, k=5),
                  "retr_s": t4 - t3, "rerank_s": t5 - t4},
        })

    # Aggregate
    n = len(gt)
    for key in ("A", "B"):
        agg = results[key]
        agg["n_queries"] = n
        for k in (1, 3, 5):
            agg[f"hit_rate_at_{k}"] = sum(q[key]["hits"][k] for q in per_query) / n
        agg["mrr_at_5"] = sum(q[key]["mrr5"] for q in per_query) / n
        agg["mean_retrieval_s"] = sum(q[key]["retr_s"] for q in per_query) / n
        agg["mean_rerank_s"] = sum(q[key]["rerank_s"] for q in per_query) / n
        agg["mean_total_s"] = (
            sum(q[key]["retr_s"] for q in per_query) + sum(q[key]["rerank_s"] for q in per_query)
        ) / n

    results["B"]["rewrite_failures"] = sum(1 for q in per_query if q["rewrite_failed"])
    results["B"]["rewrites_unchanged"] = sum(1 for q in per_query if q["rewrite_unchanged"])
    results["n_queries_changed_relevant_set"] = sum(
        1 for q in per_query if q["A"]["ok5"] != q["B"]["ok5"]
    )

    # Report
    print("\n=== query-rewriting experiment (hybrid_rerank, pool=20) ===")
    print(f"queries: {n} | model: {REWRITE_MODEL} (injected, eval-only) | "
          f"rerank model: {reranker.model_name}")
    header = ("configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | "
              "retrieval(s) | rerank(s) | total(s) | rewrite failures")
    print(header)
    print("-" * len(header))
    for key, label in (("A", "A original                 "), ("B", "B rewritten (rerank@orig) ")):
        a = results[key]
        tot = a["mean_retrieval_s"] + a["mean_rerank_s"] + \
            (results["B"].get("rewrite_unchanged", 0) * 0) if key == "A" else a["mean_total_s"]
        fail = results["B"].get("rewrite_failures", "-")
        print(f"{label} | {a['hit_rate_at_1']:.3f} | {a['hit_rate_at_3']:.3f} | "
              f"{a['hit_rate_at_5']:.3f} | {a['mrr_at_5']:.3f} | "
              f"{a['mean_retrieval_s']:10.3f} | {a['mean_rerank_s']:9.3f} | "
              f"{tot:9.3f} | {fail}")

    print(f"\nqueries where rewriting changed relevant-retrieval outcome: "
          f"{results['n_queries_changed_relevant_set']}/{n}")

    print("\n=== special focus Q13 Q17 Q19 Q22 (Hit@5 / MRR@5) ===")
    for q in per_query:
        if q["index"] in (13, 17, 19, 22):
            print(f"Q{q['index']:>2} '{q['question'][:50]}'")
            print(f"    rewritten: '{q['rewritten']}'")
            print(f"    A hit5={q['A']['hits'][5]} mrr5={q['A']['mrr5']:.3f} | "
                  f"B hit5={q['B']['hits'][5]} mrr5={q['B']['mrr5']:.3f}")

    print("\n=== rewrite outputs for all queries ===")
    for q in per_query:
        mark = " (unchanged)" if q["rewrite_unchanged"] else ""
        print(f"Q{q['index']:>2}: {q['rewritten']}{mark}")

    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump({"results": results, "per_query": per_query}, f, indent=2, ensure_ascii=False)
        print(f"\nsaved -> {sys.argv[2]}")


if __name__ == "__main__":
    main()