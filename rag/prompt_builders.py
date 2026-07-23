"""
Prompt Builders — Abstract interface and template-based implementation for RAG prompts.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any

from domain.models import SearchResult

logger = logging.getLogger(__name__)


class BasePromptBuilder(ABC):
    """Abstract Base Class for building LLM context prompts."""

    @abstractmethod
    def build_prompt(
        self, question: str, search_results: List[SearchResult], style: str = "detailed"
    ) -> List[Dict[str, str]]:
        """Construct system & user messages list for chat completions API."""
        pass


class TemplatePromptBuilder(BasePromptBuilder):
    """Template-driven prompt builder supporting concise, detailed, and structured styles."""

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
            "information to answer completely, explain what information is available."
        ),
        "structured": (
            "You are a helpful internal knowledge assistant. Answer the user's question "
            "based ONLY on the provided context. Structure your response as follows:\n"
            "1. **Answer**: Direct answer to the question\n"
            "2. **Details**: Supporting details from the context\n"
            "3. **Sources**: Reference which document(s) the information came from\n"
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

    def build_prompt(
        self, question: str, search_results: List[SearchResult], style: str = "detailed"
    ) -> List[Dict[str, str]]:
        system_prompt = self.SYSTEM_PROMPTS.get(style, self.SYSTEM_PROMPTS["detailed"])

        context_parts = []
        for i, res in enumerate(search_results, start=1):
            page_info = f"Page: {res.page_number}" if res.page_number else ""
            context_parts.append(
                self.CONTEXT_TEMPLATE.format(
                    index=i,
                    file_name=res.file_name or "Unknown",
                    page_info=page_info,
                    text=res.text,
                )
            )
        context_str = "\n".join(context_parts)
        user_message = self.USER_TEMPLATE.format(context=context_str, question=question)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
