"""
Ground Truth Generator — Creates synthetic Q&A pairs from document chunks
using the LLM. These are used to evaluate retrieval and generation quality.
"""

import json
import logging
import sys
from pathlib import Path

from groq import Groq
from elasticsearch import Elasticsearch

import config

logger = logging.getLogger(__name__)

GENERATION_PROMPT = """You are a question-answer generation expert. Given the following text chunk from an internal document, generate {n_questions} diverse question-answer pairs that could be answered using ONLY this text.

Rules:
1. Questions should be specific and answerable from the text alone
2. Answers should be concise and directly supported by the text
3. Vary question types: factual, procedural, definitional
4. Output valid JSON array with objects having "question" and "answer" fields

Text chunk (from file: {file_name}):
---
{text}
---

Output format (JSON array only, no other text):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""


def generate_ground_truth(
    n_questions_per_chunk: int = 2,
    max_chunks: int = 50,
    output_path: str = "data/ground_truth.json",
) -> list[dict]:
    """
    Generate synthetic ground-truth Q&A pairs from indexed chunks.

    Fetches chunks from Elasticsearch, then uses the LLM to generate
    questions and expected answers for each chunk.

    Args:
        n_questions_per_chunk: Number of Q&A pairs per chunk.
        max_chunks: Maximum number of chunks to process.
        output_path: Where to save the ground truth JSON.

    Returns:
        List of ground truth records.
    """
    es = Elasticsearch(config.ELASTICSEARCH_HOST)
    client = Groq(api_key=config.GROQ_API_KEY)

    # Fetch a diverse sample of chunks
    body = {
        "size": max_chunks,
        "query": {"match_all": {}},
        "_source": ["chunk_id", "text", "file_name", "source_file"],
    }
    response = es.search(index=config.ELASTICSEARCH_INDEX, body=body)
    chunks = [hit["_source"] for hit in response["hits"]["hits"]]

    if not chunks:
        logger.error("No chunks found in index. Run ingestion first.")
        return []

    logger.info(f"Generating ground truth from {len(chunks)} chunks...")
    ground_truth = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{len(chunks)}: {chunk['file_name']}")

        prompt = GENERATION_PROMPT.format(
            n_questions=n_questions_per_chunk,
            file_name=chunk.get("file_name", "unknown"),
            text=chunk["text"][:2000],  # Limit text length
        )

        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            qa_pairs = json.loads(content)

            for qa in qa_pairs:
                ground_truth.append({
                    "question": qa["question"],
                    "expected_answer": qa["answer"],
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk.get("source_file", ""),
                    "file_name": chunk.get("file_name", ""),
                })

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM output for chunk {i}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error generating Q&A for chunk {i}: {e}")
            continue

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Generated {len(ground_truth)} Q&A pairs, saved to '{output_path}'"
    )
    return ground_truth


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth.json"
    generate_ground_truth(output_path=output)
