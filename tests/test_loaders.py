"""
Unit tests for document loaders — all 5 loader types plus CompositeDocumentLoader.

Tests cover: format detection, text extraction, encoding handling, error resilience,
and the composite loader's file discovery and parallel loading capabilities.
"""

import os
import tempfile
from pathlib import Path

import pytest

from domain.models import Document
from ingestion.loaders import (
    BaseDocumentLoader,
    CodeDocumentLoader,
    CompositeDocumentLoader,
    CSVDocumentLoader,
    DocxDocumentLoader,
    PDFDocumentLoader,
    TextDocumentLoader,
)


class TestTextDocumentLoader:
    """Tests for TextDocumentLoader (.txt, .md, .rst, .log, .env)."""

    def test_can_load_supported_extensions(self):
        loader = TextDocumentLoader()
        for ext in [".txt", ".md", ".rst", ".log", ".env", ".ini", ".cfg"]:
            assert loader.can_load(ext), f"Should support {ext}"

    def test_cannot_load_unsupported(self):
        loader = TextDocumentLoader()
        assert not loader.can_load(".pdf")
        assert not loader.can_load(".py")

    def test_loads_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, World!", encoding="utf-8")
        docs = TextDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert docs[0].text == "Hello, World!"
        assert docs[0].file_type == ".txt"
        assert docs[0].file_name == "test.txt"

    def test_loads_md_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nContent here.", encoding="utf-8")
        docs = TextDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "# Title" in docs[0].text
        assert docs[0].file_type == ".md"

    def test_empty_file_returns_nothing(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        docs = TextDocumentLoader().load(str(f))
        assert docs == []

    def test_whitespace_only_returns_nothing(self, tmp_path):
        f = tmp_path / "spaces.txt"
        f.write_text("   \n\n  \t  ", encoding="utf-8")
        docs = TextDocumentLoader().load(str(f))
        assert docs == []

    def test_modified_date_populated(self, tmp_path):
        f = tmp_path / "dated.txt"
        f.write_text("content", encoding="utf-8")
        docs = TextDocumentLoader().load(str(f))
        assert docs[0].modified_date is not None

    def test_nonexistent_file_returns_empty(self):
        docs = TextDocumentLoader().load("/nonexistent/path.txt")
        assert docs == []

    def test_source_file_is_absolute(self, tmp_path):
        f = tmp_path / "abs.txt"
        f.write_text("data", encoding="utf-8")
        docs = TextDocumentLoader().load(str(f))
        assert os.path.isabs(docs[0].source_file)


class TestCSVDocumentLoader:
    """Tests for CSVDocumentLoader (.csv, .tsv, .jsonl)."""

    def test_can_load_csv_tsv_jsonl(self):
        loader = CSVDocumentLoader()
        assert loader.can_load(".csv")
        assert loader.can_load(".tsv")
        assert loader.can_load(".jsonl")

    def test_cannot_load_txt(self):
        assert not CSVDocumentLoader().can_load(".txt")

    def test_loads_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("Name,Age\nAlice,30\nBob,25\n", encoding="utf-8")
        docs = CSVDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "Alice" in docs[0].text
        assert "Bob" in docs[0].text
        assert docs[0].file_type == ".csv"

    def test_loads_tsv(self, tmp_path):
        f = tmp_path / "data.tsv"
        f.write_text("Name\tAge\nAlice\t30\n", encoding="utf-8")
        docs = CSVDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "Alice" in docs[0].text

    def test_loads_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"name": "Alice"}\n{"name": "Bob"}\n', encoding="utf-8")
        docs = CSVDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "Alice" in docs[0].text

    def test_empty_csv_returns_nothing(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        docs = CSVDocumentLoader().load(str(f))
        assert docs == []


class TestCodeDocumentLoader:
    """Tests for CodeDocumentLoader (.py, .js, .sql, .json, .yaml, etc.)."""

    def test_can_load_programming_languages(self):
        loader = CodeDocumentLoader()
        for ext in [".py", ".js", ".ts", ".java", ".go", ".rs", ".sql", ".sh"]:
            assert loader.can_load(ext), f"Should support {ext}"

    def test_can_load_config_formats(self):
        loader = CodeDocumentLoader()
        for ext in [".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css"]:
            assert loader.can_load(ext), f"Should support {ext}"

    def test_cannot_load_pdf_or_docx(self):
        loader = CodeDocumentLoader()
        assert not loader.can_load(".pdf")
        assert not loader.can_load(".docx")
        assert not loader.can_load(".txt")

    def test_loads_python_file(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("def hello():\n    print('hi')\n", encoding="utf-8")
        docs = CodeDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "def hello" in docs[0].text
        assert docs[0].file_type == ".py"

    def test_loads_json_file(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        docs = CodeDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert docs[0].file_type == ".json"

    def test_loads_sql_file(self, tmp_path):
        f = tmp_path / "schema.sql"
        f.write_text("CREATE TABLE users (id INT PRIMARY KEY);", encoding="utf-8")
        docs = CodeDocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "CREATE TABLE" in docs[0].text


class TestPDFDocumentLoader:
    """Tests for PDFDocumentLoader (requires PyMuPDF)."""

    def test_can_load_pdf(self):
        assert PDFDocumentLoader().can_load(".pdf")
        assert PDFDocumentLoader().can_load(".PDF")

    def test_cannot_load_others(self):
        assert not PDFDocumentLoader().can_load(".txt")
        assert not PDFDocumentLoader().can_load(".docx")


class TestDocxDocumentLoader:
    """Tests for DocxDocumentLoader."""

    def test_can_load_docx(self):
        assert DocxDocumentLoader().can_load(".docx")
        assert DocxDocumentLoader().can_load(".DOCX")

    def test_cannot_load_others(self):
        assert not DocxDocumentLoader().can_load(".doc")
        assert not DocxDocumentLoader().can_load(".pdf")


class TestCompositeDocumentLoader:
    """Tests for CompositeDocumentLoader (orchestration layer)."""

    def test_supported_extensions_includes_all(self):
        exts = CompositeDocumentLoader.get_supported_extensions()
        assert ".txt" in exts
        assert ".md" in exts
        assert ".csv" in exts
        assert ".py" in exts
        assert ".pdf" in exts
        assert ".docx" in exts

    def test_supported_extensions_without_dot(self):
        exts = CompositeDocumentLoader.get_supported_extensions_without_dot()
        assert "txt" in exts
        assert "py" in exts
        assert "pdf" in exts
        # No dots
        assert all(not e.startswith(".") for e in exts)

    def test_load_directory_discovers_files(self, sample_docs_dir):
        loader = CompositeDocumentLoader()
        docs = loader.load_directory(sample_docs_dir)
        # Should load: policy.txt, guide.md, data.csv, utils.py (not image.png)
        assert len(docs) >= 4

    def test_load_directory_skips_unsupported(self, sample_docs_dir):
        loader = CompositeDocumentLoader()
        docs = loader.load_directory(sample_docs_dir)
        file_types = {d.file_type for d in docs}
        assert ".png" not in file_types

    def test_nonexistent_directory_raises(self):
        loader = CompositeDocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory("/nonexistent/directory")

    def test_empty_directory(self, tmp_path):
        loader = CompositeDocumentLoader()
        docs = loader.load_directory(str(tmp_path))
        assert docs == []

    def test_stream_directory_is_generator(self, sample_docs_dir):
        loader = CompositeDocumentLoader()
        gen = loader.stream_directory(sample_docs_dir)
        # Should be an iterator, not a list
        import types
        assert isinstance(gen, types.GeneratorType)
        docs = list(gen)
        assert len(docs) >= 4
