"""Unit tests for prompt builders — all 3 prompt styles."""

import pytest
from domain.models import SearchResult
from rag.prompt_builders import TemplatePromptBuilder


class TestTemplatePromptBuilder:
    def test_detailed_style(self, sample_search_results):
        msgs = TemplatePromptBuilder().build_prompt("How many PTO days?", sample_search_results, style="detailed")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "comprehensive" in msgs[0]["content"].lower()
        assert "PTO" in msgs[1]["content"]

    def test_concise_style(self, sample_search_results):
        msgs = TemplatePromptBuilder().build_prompt("Q?", sample_search_results, style="concise")
        assert "concise" in msgs[0]["content"].lower()

    def test_structured_style(self, sample_search_results):
        msgs = TemplatePromptBuilder().build_prompt("Q?", sample_search_results, style="structured")
        assert "Sources" in msgs[0]["content"]

    def test_unknown_style_defaults_to_detailed(self, sample_search_results):
        msgs = TemplatePromptBuilder().build_prompt("Q?", sample_search_results, style="unknown_xyz")
        assert "comprehensive" in msgs[0]["content"].lower()

    def test_empty_results(self):
        msgs = TemplatePromptBuilder().build_prompt("Q?", [], style="detailed")
        assert len(msgs) == 2

    def test_page_number_included(self):
        results = [SearchResult("c1", "d1", "text", "f.pdf", "/f.pdf", ".pdf", page_number=5, score=0.9)]
        msgs = TemplatePromptBuilder().build_prompt("Q?", results, style="detailed")
        assert "Page: 5" in msgs[1]["content"]

    def test_context_documents_numbered(self, sample_search_results):
        msgs = TemplatePromptBuilder().build_prompt("Q?", sample_search_results, style="detailed")
        assert "Context Document 1" in msgs[1]["content"]
        assert "Context Document 2" in msgs[1]["content"]
