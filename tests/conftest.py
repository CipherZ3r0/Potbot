"""
Shared test fixtures for the potbot test suite.

All heavy dependencies (Elasticsearch, Groq, ML models) are mocked here,
making the entire suite runnable offline in < 30 seconds without any
external services.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so absolute imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.models import Chunk, Document, SearchResult, RAGResponse, FeedbackRecord
from ingestion.config import PipelineConfig


# ---------------------------------------------------------------------------
# Domain model factories
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document() -> Document:
    """Return a realistic plain-text Document."""
    return Document(
        text=(
            "Acme Corp provides 15 days of PTO per year for employees with 0–2 years of service. "
            "Employees with 2–5 years receive 20 days. Senior employees (5+ years) receive 25 days. "
            "Up to 5 unused days may roll over. PTO requests must be submitted 2 weeks in advance."
        ),
        source_file="/data/vacation_policy.txt",
        file_name="vacation_policy.txt",
        file_type=".txt",
        modified_date="2026-01-15T00:00:00+00:00",
    )


@pytest.fixture
def sample_markdown_document() -> Document:
    """Return a markdown Document with headers."""
    return Document(
        text=(
            "# Security Policy\n\n"
            "## Password Requirements\n"
            "Minimum 16 characters. Must include uppercase, lowercase, numbers.\n\n"
            "## MFA Policy\n"
            "Hardware tokens or authenticator apps required. SMS prohibited.\n"
        ),
        source_file="/data/security.md",
        file_name="security.md",
        file_type=".md",
    )


@pytest.fixture
def sample_code_document() -> Document:
    """Return a Python code Document."""
    return Document(
        text=(
            "class DataProcessor:\n"
            "    def __init__(self, batch_size=100):\n"
            "        self.batch_size = batch_size\n\n"
            "    def process(self, data):\n"
            "        return [x * 2 for x in data]\n\n"
            "def main():\n"
            "    dp = DataProcessor()\n"
            "    print(dp.process([1, 2, 3]))\n"
        ),
        source_file="/data/processor.py",
        file_name="processor.py",
        file_type=".py",
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    """Return a pre-built Chunk with embedding."""
    return Chunk(
        chunk_id="abc123",
        doc_id="doc001",
        text="Employees receive 15 days of PTO per year.",
        chunk_index=0,
        source_file="/data/vacation.txt",
        file_name="vacation.txt",
        file_type=".txt",
        embedding=[0.1] * 384,
    )


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Return a list of 3 Chunks with embeddings."""
    return [
        Chunk(
            chunk_id=f"chunk_{i}",
            doc_id="doc001",
            text=f"Chunk text number {i} about corporate policies.",
            chunk_index=i,
            source_file="/data/policy.txt",
            file_name="policy.txt",
            file_type=".txt",
            embedding=[float(i) / 10] * 384,
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Return a list of ranked SearchResult objects."""
    return [
        SearchResult(
            chunk_id="r1",
            doc_id="d1",
            text="PTO accrual rate is 15 days per year for new employees.",
            file_name="vacation.md",
            source_file="/data/vacation.md",
            file_type=".md",
            score=0.95,
        ),
        SearchResult(
            chunk_id="r2",
            doc_id="d1",
            text="Employees may roll over up to 5 unused PTO days.",
            file_name="vacation.md",
            source_file="/data/vacation.md",
            file_type=".md",
            page_number=2,
            score=0.88,
        ),
        SearchResult(
            chunk_id="r3",
            doc_id="d2",
            text="Password minimum length is 16 characters.",
            file_name="security.txt",
            source_file="/data/security.txt",
            file_type=".txt",
            score=0.72,
        ),
    ]


# ---------------------------------------------------------------------------
# Temporary file / directory helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path as a string."""
    return str(tmp_path)


@pytest.fixture
def sample_docs_dir(tmp_path) -> str:
    """Create a temporary directory with sample document files."""
    # Text file
    (tmp_path / "policy.txt").write_text(
        "All employees must follow the code of conduct. Violations result in disciplinary action.",
        encoding="utf-8",
    )
    # Markdown file
    (tmp_path / "guide.md").write_text(
        "# Onboarding Guide\n\n## Step 1\nComplete your profile.\n\n## Step 2\nRead the handbook.",
        encoding="utf-8",
    )
    # CSV file
    (tmp_path / "data.csv").write_text(
        "Name,Department,Salary\nAlice,Engineering,120000\nBob,Marketing,95000\n",
        encoding="utf-8",
    )
    # Python file
    (tmp_path / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    # Unsupported file (should be skipped)
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(tmp_path)


@pytest.fixture
def minimal_pipeline_config() -> PipelineConfig:
    """Return a PipelineConfig suitable for unit tests (minimal parallelism)."""
    return PipelineConfig(
        loader_workers=1,
        chunker_workers=1,
        embed_batch_size=4,
        index_bulk_size=10,
        queue_size=0,
        device="cpu",
        embed_cache_enabled=False,
        chunk_size=200,
        chunk_overlap=20,
    )


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder():
    """Return a MagicMock embedder that passes chunks through unchanged."""
    embedder = MagicMock()
    embedder.get_dimension.return_value = 384

    def _embed_chunks(chunks, batch_size=64):
        for c in chunks:
            c.embedding = [0.1] * 384
        return chunks

    embedder.embed_chunks.side_effect = _embed_chunks

    def _stream_embed(chunks, **kwargs):
        for c in chunks:
            c.embedding = [0.1] * 384
            yield c

    embedder.stream_embed.side_effect = _stream_embed
    embedder.embed_text.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def mock_vector_store():
    """Return a MagicMock vector store that counts indexed chunks."""
    store = MagicMock()
    store.get_stats.return_value = {"exists": True, "doc_count": 0}

    def _stream_index(chunks, **kwargs):
        return len(list(chunks))

    store.stream_index.side_effect = _stream_index
    store.create_index.return_value = None
    store.index_chunks.side_effect = lambda chunks: len(chunks)
    store.vector_search.return_value = []
    store.text_search.return_value = []
    return store
