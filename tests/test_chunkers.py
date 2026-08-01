"""
Unit tests for text chunkers — RecursiveCharacterChunker, MarkdownHeaderChunker,
CodeChunker, and CompositeChunker.
"""

import pytest
from domain.models import Chunk, Document
from ingestion.chunkers import (
    CodeChunker, CompositeChunker, MarkdownHeaderChunker,
    RecursiveCharacterChunker, _chunk_doc_worker,
)


class TestRecursiveCharacterChunker:
    def test_short_text_single_chunk(self):
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        doc = Document(text="Short text.", source_file="/a.txt", file_name="a.txt", file_type=".txt")
        assert len(chunker.chunk_document(doc)) == 1

    def test_long_text_multiple_chunks(self):
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        doc = Document(text="word " * 100, source_file="/a.txt", file_name="a.txt", file_type=".txt")
        assert len(chunker.chunk_document(doc)) > 1

    def test_chunks_have_correct_metadata(self, sample_document):
        chunks = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20).chunk_document(sample_document)
        for i, c in enumerate(chunks):
            assert c.source_file == sample_document.source_file
            assert c.file_name == sample_document.file_name
            assert c.chunk_index == i
            assert c.chunk_id and c.doc_id

    def test_chunk_ids_are_unique(self, sample_document):
        chunks = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20).chunk_document(sample_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_text_no_chunks(self):
        doc = Document(text="", source_file="/a.txt", file_name="a.txt", file_type=".txt")
        assert RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10).chunk_document(doc) == []

    def test_whitespace_only_no_chunks(self):
        doc = Document(text="   \n\n  ", source_file="/a.txt", file_name="a.txt", file_type=".txt")
        assert RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10).chunk_document(doc) == []


class TestMarkdownHeaderChunker:
    def test_splits_by_headers(self, sample_markdown_document):
        assert len(MarkdownHeaderChunker(chunk_size=500, chunk_overlap=20).chunk_document(sample_markdown_document)) >= 2

    def test_non_markdown_falls_back(self):
        doc = Document(text="Plain text.", source_file="/a.txt", file_name="a.txt", file_type=".txt")
        assert len(MarkdownHeaderChunker(chunk_size=500, chunk_overlap=20).chunk_document(doc)) >= 1

    def test_large_section_gets_sub_chunked(self):
        doc = Document(text="# Big\n" + "Long sentence. " * 50, source_file="/a.md", file_name="a.md", file_type=".md")
        assert len(MarkdownHeaderChunker(chunk_size=100, chunk_overlap=10).chunk_document(doc)) > 1


class TestCodeChunker:
    def test_splits_python_code(self, sample_code_document):
        chunks = CodeChunker(chunk_size=80, chunk_overlap=10).chunk_document(sample_code_document)
        assert len(chunks) >= 1
        assert all(c.file_type == ".py" for c in chunks)

    def test_code_extensions_set(self):
        assert {".py", ".js", ".sql", ".json", ".yaml"}.issubset(CodeChunker.CODE_EXTENSIONS)


class TestChunkDocWorker:
    def test_dispatches_markdown(self, sample_markdown_document):
        assert len(_chunk_doc_worker(sample_markdown_document, 500, 20)) >= 1

    def test_dispatches_code(self, sample_code_document):
        assert len(_chunk_doc_worker(sample_code_document, 500, 20)) >= 1

    def test_dispatches_txt(self, sample_document):
        assert len(_chunk_doc_worker(sample_document, 100, 10)) >= 1


class TestCompositeChunker:
    def test_chunk_documents_returns_list(self, sample_document):
        chunks = CompositeChunker(chunk_size=100, chunk_overlap=10).chunk_documents([sample_document])
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_input(self):
        assert CompositeChunker(chunk_size=100, chunk_overlap=10).chunk_documents([]) == []
