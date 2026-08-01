"""
Potbot Offline Evaluation Harness

Runs a complete evaluation of the Retrieval and Generation pipeline using
synthetic ground-truth Q&A datasets. Generates a comprehensive JSON report
and Markdown summary.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

import config
from rag.retrievers import SearchStrategyFactory
from rag.rerankers import CrossEncoderReranker
from rag.prompt_builders import TemplatePromptBuilder
from rag.llm_providers import GroqLLMProvider

from evaluation.metrics_utils import (
    precision_at_k, recall_at_k, ndcg_at_k, average_precision,
    token_overlap_f1, simple_rouge_l
)

# Optional LLM judging
from evaluation.llm_eval import llm_judge_score, cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def evaluate_system(ground_truth_path: str = "data/ground_truth.json", top_k: int = 5, use_llm_judge: bool = True) -> Dict[str, Any]:
    """Run full system evaluation."""
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return {}

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    if not ground_truth:
        logger.error("Ground truth is empty.")
        return {}

    logger.info(f"Starting evaluation of {len(ground_truth)} queries...")

    # Load Embedder for Cosine Sim if doing LLM eval
    embed_model = None
    if use_llm_judge:
        logger.info("Loading Embedding Model for Cosine Similarity evaluation...")
        embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    strategies = ["vector", "text", "hybrid"]
    retrieval_metrics = {m: {"precisions": [], "recalls": [], "ndcgs": [], "aps": []} for m in strategies}
    retrieval_metrics["hybrid_rerank"] = {"precisions": [], "recalls": [], "ndcgs": [], "aps": []}

    generation_metrics = {
        style: {"f1s": [], "rouge_ls": [], "cos_sims": [], "relevances": [], "faithfulness": [], "completeness": [], "latencies": []}
        for style in ["concise", "detailed", "structured"]
    }

    # Evaluate all queries
    for i, gt in enumerate(ground_truth):
        query = gt["question"]
        expected_chunk_id = gt["chunk_id"]
        expected_answer = gt["expected_answer"]

        if (i + 1) % 5 == 0:
            logger.info(f"Progress: {i + 1}/{len(ground_truth)}")

        # ---------------------------------------------------------
        # 1. Retrieval Evaluation
        # ---------------------------------------------------------
        best_retrieved_results = None
        
        for method in strategies:
            try:
                strategy = SearchStrategyFactory.get_strategy(method)
                results = strategy.search(query, top_k=top_k)
                predicted_ids = [r.chunk_id for r in results]
                
                prec = precision_at_k([expected_chunk_id], predicted_ids, top_k)
                rec = recall_at_k([expected_chunk_id], predicted_ids, top_k)
                ndcg = ndcg_at_k([expected_chunk_id], predicted_ids, top_k)
                ap = average_precision([expected_chunk_id], predicted_ids)
                
                retrieval_metrics[method]["precisions"].append(prec)
                retrieval_metrics[method]["recalls"].append(rec)
                retrieval_metrics[method]["ndcgs"].append(ndcg)
                retrieval_metrics[method]["aps"].append(ap)
            except Exception as e:
                logger.warning(f"Retrieval failed for {method}: {e}")

        # Hybrid + Reranking
        try:
            hybrid_strategy = SearchStrategyFactory.get_strategy("hybrid")
            raw_results = hybrid_strategy.search(query, top_k=top_k * 2)
            reranker = CrossEncoderReranker()
            reranked_results = reranker.rerank(query, raw_results, top_n=top_k)
            
            predicted_ids = [r.chunk_id for r in reranked_results]
            prec = precision_at_k([expected_chunk_id], predicted_ids, top_k)
            rec = recall_at_k([expected_chunk_id], predicted_ids, top_k)
            ndcg = ndcg_at_k([expected_chunk_id], predicted_ids, top_k)
            ap = average_precision([expected_chunk_id], predicted_ids)
            
            retrieval_metrics["hybrid_rerank"]["precisions"].append(prec)
            retrieval_metrics["hybrid_rerank"]["recalls"].append(rec)
            retrieval_metrics["hybrid_rerank"]["ndcgs"].append(ndcg)
            retrieval_metrics["hybrid_rerank"]["aps"].append(ap)
            
            best_retrieved_results = reranked_results
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            best_retrieved_results = raw_results if 'raw_results' in locals() else []

        # ---------------------------------------------------------
        # 2. Generation Evaluation (using best retrieved chunks)
        # ---------------------------------------------------------
        if not best_retrieved_results:
            continue
            
        # Only evaluate generation on a subset to save LLM costs if it's large
        if i >= 20 and use_llm_judge: 
            continue
            
        for style in generation_metrics.keys():
            try:
                builder = TemplatePromptBuilder()
                prompt = builder.build_prompt(query, best_retrieved_results, style=style)
                
                llm = GroqLLMProvider()
                result = llm.generate(prompt)
                actual_answer = result["answer"]
                
                # Deterministic text metrics
                f1 = token_overlap_f1(expected_answer, actual_answer)
                rouge = simple_rouge_l(expected_answer, actual_answer)
                
                generation_metrics[style]["f1s"].append(f1)
                generation_metrics[style]["rouge_ls"].append(rouge)
                generation_metrics[style]["latencies"].append(result["response_time_ms"])
                
                # LLM-as-a-judge & Embedding metrics
                if use_llm_judge and embed_model:
                    emb_exp = embed_model.encode([expected_answer], normalize_embeddings=True)
                    emb_act = embed_model.encode([actual_answer], normalize_embeddings=True)
                    cos_sim = cosine_similarity(emb_exp[0].tolist(), emb_act[0].tolist())
                    
                    scores = llm_judge_score(query, expected_answer, actual_answer)
                    
                    generation_metrics[style]["cos_sims"].append(cos_sim)
                    generation_metrics[style]["relevances"].append(scores.get("relevance", 0))
                    generation_metrics[style]["faithfulness"].append(scores.get("faithfulness", 0))
                    generation_metrics[style]["completeness"].append(scores.get("completeness", 0))
                    
            except Exception as e:
                logger.warning(f"Generation failed for {style}: {e}")

    # Compile Final Report
    report = {
        "metadata": {
            "num_queries_retrieval": len(ground_truth),
            "num_queries_generation": min(20, len(ground_truth)),
            "top_k": top_k
        },
        "retrieval_benchmarks": {},
        "generation_benchmarks": {}
    }

    # Average retrieval metrics
    for method, metrics in retrieval_metrics.items():
        if metrics["precisions"]:
            report["retrieval_benchmarks"][method] = {
                "precision": float(np.mean(metrics["precisions"])),
                "recall": float(np.mean(metrics["recalls"])),
                "ndcg": float(np.mean(metrics["ndcgs"])),
                "map": float(np.mean(metrics["aps"])),
            }
            
    # Average generation metrics
    for style, metrics in generation_metrics.items():
        if metrics["f1s"]:
            stats = {
                "token_f1": float(np.mean(metrics["f1s"])),
                "rouge_l": float(np.mean(metrics["rouge_ls"])),
                "avg_latency_ms": float(np.mean(metrics["latencies"]))
            }
            if use_llm_judge and metrics["cos_sims"]:
                stats.update({
                    "cosine_similarity": float(np.mean(metrics["cos_sims"])),
                    "llm_relevance": float(np.mean(metrics["relevances"])),
                    "llm_faithfulness": float(np.mean(metrics["faithfulness"])),
                    "llm_completeness": float(np.mean(metrics["completeness"])),
                })
            report["generation_benchmarks"][style] = stats

    # Save JSON report
    out_path = Path("data/evaluation_report.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Evaluation complete. Report saved to {out_path}")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Check if ground truth exists, if not inform the user
    if not Path("data/ground_truth.json").exists():
        logger.error("No ground truth dataset found. Please ingest sample documents and run evaluation/ground_truth_generator.py first.")
        sys.exit(1)
        
    report = evaluate_system()
    
    # Print Markdown Summary
    print("\\n\\n### Retrieval Benchmarks")
    print("| Method | Precision@5 | Recall@5 | NDCG@5 | MAP |")
    print("|--------|-------------|----------|--------|-----|")
    for method, stats in report["retrieval_benchmarks"].items():
        print(f"| {method} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['ndcg']:.3f} | {stats['map']:.3f} |")
        
    print("\\n### Generation Benchmarks")
    print("| Style | ROUGE-L | Token F1 | Cosine Sim | Faithfulness | Latency (ms) |")
    print("|-------|---------|----------|------------|--------------|--------------|")
    for style, stats in report["generation_benchmarks"].items():
        cos = stats.get('cosine_similarity', 0)
        faith = stats.get('llm_faithfulness', 0)
        print(f"| {style} | {stats['rouge_l']:.3f} | {stats['token_f1']:.3f} | {cos:.3f} | {faith:.2f}/5.0 | {stats['avg_latency_ms']:.0f} |")
