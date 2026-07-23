"""
Query Rewriter — Uses Groq LLM to rewrite ambiguous user queries
into more effective search queries. (Bonus: user query rewriting)

Examples:
  "what's the vacation policy?" →
  "employee vacation leave policy rules days allowed annual PTO"
"""

import logging

from groq import Groq

import config

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a search query optimizer. Your job is to rewrite the user's question into a better search query that will retrieve the most relevant documents from an internal knowledge base.

Rules:
1. Expand abbreviations and add synonyms
2. Include key terms that are likely in the source documents
3. Keep it concise — output ONLY the rewritten query, nothing else
4. Do NOT answer the question, only rewrite it for better search retrieval
5. If the query is already clear and specific, return it as-is

Examples:
- Input: "what's the vacation policy?"
  Output: "employee vacation leave policy annual PTO days allowed rules"
- Input: "how do I reset my password?"
  Output: "password reset procedure steps account access recovery"
- Input: "Q3 revenue numbers"
  Output: "Q3 third quarter revenue financial results earnings report"
"""


def rewrite_query(query: str) -> str:
    """
    Rewrite a user query into a more search-friendly form using the LLM.

    Args:
        query: The original user question.

    Returns:
        A rewritten query optimized for retrieval search.
    """
    try:
        client = Groq(api_key=config.GROQ_API_KEY)

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=150,
        )

        rewritten = response.choices[0].message.content.strip()

        if rewritten:
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        else:
            logger.warning("Empty rewrite response, using original query")
            return query

    except Exception as e:
        logger.error(f"Query rewrite failed: {e}. Using original query.")
        return query
