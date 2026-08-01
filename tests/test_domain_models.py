"""
Unit tests for domain models — Document, Chunk, SearchResult, RAGResponse, FeedbackRecord.

Tests validate dataclass construction, default values, optional fields, and metadata.
"""

import pytest
from datetime import datetime, timezone

from domain.models import Chunk, Document, FeedbackRecord, RAGResponse, SearchResult


class TestDocument:
    """Tests for the Document dataclass."""

    def test_required_fields(self):
        doc = Document(
            text="Hello", source_file="/a.txt", file_name="a.txt", file_type=".txt"
        )
        assert doc.text == "Hello"
        assert doc.source_file == "/a.txt"
        assert doc.file_name == "a.txt"
        assert doc.file_type == ".txt"

    def test_optional_fields_default_none(self):
        doc = Document(text="", source_file="", file_name="", file_type="")
        assert doc.page_number is None
        assert doc.total_pages is None
        assert doc.modified_date is None
        assert doc.metadata == {}

    def test_metadata_isolation(self):
        """Two Document instances should not share the same metadata dict."""
        d1 = Document(text="a", source_file="", file_name="", file_type="")
        d2 = Document(text="b", source_file="", file_name="", file_type="")
        d1.metadata["key"] = "val"
        assert "key" not in d2.metadata

    def test_pdf_with_page_info(self):
        doc = Document(
            text="Page 1 content",
            source_file="/report.pdf",
            file_name="report.pdf",
            file_type=".pdf",
            page_number=1,
            total_pages=10,
        )
        assert doc.page_number == 1
        assert doc.total_pages == 10


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_required_fields(self):
        chunk = Chunk(
            chunk_id="c1", doc_id="d1", text="content",
            chunk_index=0, source_file="/a.txt",
            file_name="a.txt", file_type=".txt",
        )
        assert chunk.chunk_id == "c1"
        assert chunk.chunk_index == 0

    def test_embedding_default_none(self):
        chunk = Chunk(
            chunk_id="c2", doc_id="d1", text="text",
            chunk_index=1, source_file="", file_name="", file_type="",
        )
        assert chunk.embedding is None

    def test_embedding_assignment(self):
        chunk = Chunk(
            chunk_id="c3", doc_id="d1", text="text",
            chunk_index=0, source_file="", file_name="", file_type="",
            embedding=[0.1, 0.2, 0.3],
        )
        assert len(chunk.embedding) == 3
        assert chunk.embedding[0] == pytest.approx(0.1)


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_default_scores(self):
        sr = SearchResult(
            chunk_id="r1", doc_id="d1", text="result",
            file_name="f.txt", source_file="/f.txt", file_type=".txt",
        )
        assert sr.score == 0.0
        assert sr.rrf_score is None
        assert sr.rerank_score is None

    def test_score_assignment(self):
        sr = SearchResult(
            chunk_id="r2", doc_id="d1", text="result",
            file_name="f.txt", source_file="/f.txt", file_type=".txt",
            score=0.95, rrf_score=0.012, rerank_score=3.45,
        )
        assert sr.score == pytest.approx(0.95)
        assert sr.rrf_score == pytest.approx(0.012)
        assert sr.rerank_score == pytest.approx(3.45)


class TestRAGResponse:
    """Tests for the RAGResponse dataclass."""

    def test_construction(self, sample_search_results):
        resp = RAGResponse(
            answer="15 days",
            query="How many PTO days?",
            rewritten_query="PTO accrual days new employee",
            retrieved_docs=sample_search_results,
            model="llama-3.3-70b-versatile",
            prompt_style="detailed",
            retrieval_method="hybrid",
            response_time_ms=350,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert resp.answer == "15 days"
        assert len(resp.retrieved_docs) == 3
        assert resp.conversation_id is None

    def test_default_token_counts(self, sample_search_results):
        resp = RAGResponse(
            answer="answer", query="q", rewritten_query=None,
            retrieved_docs=[], model="m", prompt_style="concise",
            retrieval_method="vector", response_time_ms=100,
        )
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0


class TestFeedbackRecord:
    """Tests for the FeedbackRecord dataclass."""

    def test_construction(self):
        fb = FeedbackRecord(conversation_id=1, sentiment="positive", comment="Great!")
        assert fb.sentiment == "positive"
        assert fb.comment == "Great!"
        assert isinstance(fb.created_at, datetime)

    def test_negative_feedback(self):
        fb = FeedbackRecord(conversation_id=2, sentiment="negative")
        assert fb.sentiment == "negative"
        assert fb.comment is None

    def test_created_at_is_utc(self):
        fb = FeedbackRecord(conversation_id=3, sentiment="positive")
        assert fb.created_at.tzinfo is not None
