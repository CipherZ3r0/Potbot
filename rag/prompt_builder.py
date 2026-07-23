"""
Prompt Builder — Constructs LLM prompts with retrieved context.

Provides multiple prompt templates for evaluation:
  - concise: Short, direct answers
  - detailed: Comprehensive answers with explanations
  - structured: Answers with source attribution
"""

import logging

logger = logging.getLogger(__name__)

# --- Prompt Templates ---

SYSTEM_PROMPTS = {
    "concise": (
        "You are a helpful internal knowledge assistant. Answer the user's question "
        "based ONLY on the provided context. Be concise and direct. If the context "
        "doesn't contain enough information to answer, say so clearly."
    ),
    "detailed": (
        "You are a helpful internal knowledge assistant. Answer the user's question "
        "based ONLY on the provided context. Provide a comprehensive and detailed "
        "answer, explaining your reasoning. If the context doesn't contain enough "
        "information to answer the question completely, explain what information is "
        "available and what is missing."
    ),
    "structured": (
        "You are a helpful internal knowledge assistant. Answer the user's question "
        "based ONLY on the provided context. Structure your response as follows:\n"
        "1. **Answer**: A direct answer to the question\n"
        "2. **Details**: Supporting details from the context\n"
        "3. **Sources**: Reference which document(s) the information came from\n\n"
        "If the context doesn't contain enough information, clearly state that."
    ),
}

CONTEXT_TEMPLATE = """--- Context Document {index} ---
Source: {file_name}
{page_info}
Content:
{text}
"""

USER_TEMPLATE = """Based on the following context documents, answer the question.

{context}

--- Question ---
{question}
"""


def build_context(results: list[dict]) -> str:
    """Format retrieved results into a context string for the prompt."""
    context_parts = []
    for i, result in enumerate(results, start=1):
        page_info = ""
        if result.get("page_number"):
            page_info = f"Page: {result['page_number']}"

        context_parts.append(
            CONTEXT_TEMPLATE.format(
                index=i,
                file_name=result.get("file_name", "Unknown"),
                page_info=page_info,
                text=result.get("text", ""),
            )
        )
    return "\n".join(context_parts)


def build_prompt(
    question: str,
    results: list[dict],
    prompt_style: str = "detailed",
) -> list[dict]:
    """
    Build a complete chat prompt (messages list) for the LLM.

    Args:
        question: The user's question.
        results: Retrieved context documents.
        prompt_style: One of 'concise', 'detailed', 'structured'.

    Returns:
        List of message dicts for the chat API: [system, user].
    """
    system_prompt = SYSTEM_PROMPTS.get(prompt_style, SYSTEM_PROMPTS["detailed"])
    context = build_context(results)

    user_message = USER_TEMPLATE.format(context=context, question=question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    logger.info(
        f"Built prompt: style={prompt_style}, context_docs={len(results)}, "
        f"total_chars={sum(len(m['content']) for m in messages)}"
    )

    return messages


def get_available_styles() -> list[str]:
    """Return list of available prompt styles."""
    return list(SYSTEM_PROMPTS.keys())
