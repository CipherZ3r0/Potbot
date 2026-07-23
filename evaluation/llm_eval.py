"""
LLM Evaluation — Evaluates generation quality across multiple prompt styles
using LLM-as-judge and cosine similarity scoring.

Compares: concise, detailed, and structured prompt styles.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
from rag.retrievers import SearchStrategyFactory
from rag.rerankers import CrossEncoderReranker
from rag.prompt_builders import TemplatePromptBuilder
from rag.llm_providers import GroqLLMProvider
from groq import Groq

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an impartial judge evaluating the quality of an AI assistant's answer.

Rate the answer on a scale of 1-5 for each criterion:
1. **Relevance**: Does the answer address the question? (1=off-topic, 5=perfectly relevant)
2. **Faithfulness**: Is the answer supported by the provided context? (1=hallucinated, 5=fully grounded)
3. **Completeness**: Does the answer cover all important aspects? (1=missing everything, 5=comprehensive)

Question: {question}
Expected Answer: {expected_answer}
AI Answer: {actual_answer}

Respond ONLY with a JSON object (no other text):
{{"relevance": <1-5>, "faithfulness": <1-5>, "completeness": <1-5>}}
"""


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def llm_judge_score(
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> dict:
    """
    Use an LLM to score answer quality on relevance, faithfulness, completeness.
    """
    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
        )

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        scores = json.loads(content)
        return {
            "relevance": scores.get("relevance", 0),
            "faithfulness": scores.get("faithfulness", 0),
            "completeness": scores.get("completeness", 0),
        }

    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {"relevance": 0, "faithfulness": 0, "completeness": 0}


def evaluate_llm(
    ground_truth_path: str = "data/ground_truth.json",
    output_path: str = "data/llm_eval_results.json",
    retrieval_method: str = "hybrid",
    use_reranking: bool = True,
) -> dict:
    """
    Evaluate LLM generation quality across multiple prompt styles.

    For each ground truth question:
      1. Retrieve context using the best retrieval method
      2. Generate answers with each prompt style (concise, detailed, structured)
      3. Score each answer using LLM-as-judge and cosine similarity

    Returns summary metrics per prompt style.
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return {}

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    if not ground_truth:
        logger.error("Ground truth is empty")
        return {}

    # Load embedding model for cosine similarity
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    prompt_styles = ["concise", "detailed", "structured"]
    results = {style: [] for style in prompt_styles}

    # Limit to avoid excessive API calls
    eval_set = ground_truth[:30]
    logger.info(
        f"Evaluating {len(eval_set)} queries × {len(prompt_styles)} prompt styles..."
    )

    for i, gt in enumerate(eval_set):
        question = gt["question"]
        expected_answer = gt["expected_answer"]

        if (i + 1) % 5 == 0:
            logger.info(f"  Progress: {i + 1}/{len(eval_set)}")

        # Retrieve context once
        try:
            strategy = SearchStrategyFactory.get_strategy(retrieval_method)
            context_results = strategy.search(question, top_k=config.TOP_K_RETRIEVAL)
            if use_reranking:
                reranker = CrossEncoderReranker()
                context_results = reranker.rerank(question, context_results, top_n=config.RERANK_TOP_N)
        except Exception as e:
            logger.warning(f"Retrieval failed for query {i}: {e}")
            continue

        for style in prompt_styles:
            try:
                # Generate answer
                prompt_builder = TemplatePromptBuilder()
                prompt = prompt_builder.build_prompt(question, context_results, style=style)
                
                llm = GroqLLMProvider()
                result = llm.generate(prompt)
                
                actual_answer = result["answer"]

                # Cosine similarity between expected and actual
                embeddings = embed_model.encode(
                    [expected_answer, actual_answer], normalize_embeddings=True
                )
                cos_sim = cosine_similarity(
                    embeddings[0].tolist(), embeddings[1].tolist()
                )

                # LLM-as-judge scoring
                judge_scores = llm_judge_score(
                    question, expected_answer, actual_answer
                )

                results[style].append({
                    "question": question,
                    "cosine_similarity": cos_sim,
                    "response_time_ms": result["response_time_ms"],
                    "total_tokens": result["total_tokens"],
                    **judge_scores,
                })

            except Exception as e:
                logger.warning(f"Evaluation failed for style={style}, query {i}: {e}")
                continue

    # Compute summary
    summary = {}
    for style, evals in results.items():
        if not evals:
            summary[style] = {"error": "no results"}
            continue

        df = pd.DataFrame(evals)
        summary[style] = {
            "avg_cosine_similarity": float(df["cosine_similarity"].mean()),
            "avg_relevance": float(df["relevance"].mean()),
            "avg_faithfulness": float(df["faithfulness"].mean()),
            "avg_completeness": float(df["completeness"].mean()),
            "avg_response_time_ms": float(df["response_time_ms"].mean()),
            "avg_total_tokens": float(df["total_tokens"].mean()),
            "n_queries": len(evals),
        }

    # Find best prompt style (by combined score)
    scored = {
        s: (v.get("avg_relevance", 0) + v.get("avg_faithfulness", 0) + v.get("avg_completeness", 0)) / 3
        for s, v in summary.items()
        if "error" not in v
    }
    best_style = max(scored, key=scored.get) if scored else "detailed"
    summary["best_style"] = best_style

    # Log results
    logger.info("\n=== LLM Evaluation Results ===")
    for style in prompt_styles:
        if style in summary and "error" not in summary[style]:
            s = summary[style]
            logger.info(
                f"  {style:12s} | CosSim={s['avg_cosine_similarity']:.3f} | "
                f"Rel={s['avg_relevance']:.1f} | Faith={s['avg_faithfulness']:.1f} | "
                f"Comp={s['avg_completeness']:.1f} | Time={s['avg_response_time_ms']:.0f}ms"
            )
    logger.info(f"\nBest prompt style: {best_style}")

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
    evaluate_llm(ground_truth_path=gt_path)
