"""
Unit Tests for potbot domain models, loaders, chunkers, prompt builders, and rerankers.
"""

import unittest

from domain.models import Document, Chunk, SearchResult
from ingestion.loaders import TextDocumentLoader, PDFDocumentLoader, DocxDocumentLoader, CSVDocumentLoader
from ingestion.chunkers import RecursiveCharacterChunker, MarkdownHeaderChunker, CompositeChunker
from rag.prompt_builders import TemplatePromptBuilder
from rag.rerankers import NoOpReranker
from rag.query_rewriters import NoOpQueryRewriter


class TestpotbotPipeline(unittest.TestCase):

    def test_domain_models(self):
        doc = Document(
            text="Hello world",
            source_file="/tmp/test.txt",
            file_name="test.txt",
            file_type=".txt"
        )
        self.assertEqual(doc.text, "Hello world")
        self.assertEqual(doc.file_name, "test.txt")

        chunk = Chunk(
            chunk_id="abc",
            doc_id="123",
            text="Hello",
            chunk_index=0,
            source_file="/tmp/test.txt",
            file_name="test.txt",
            file_type=".txt"
        )
        self.assertEqual(chunk.chunk_id, "abc")

    def test_recursive_character_chunker(self):
        chunker = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=5)
        doc = Document(
            text="This is a test document with several words for testing chunking.",
            source_file="/tmp/doc.txt",
            file_name="doc.txt",
            file_type=".txt"
        )
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))

    def test_markdown_header_chunker(self):
        chunker = MarkdownHeaderChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(
            text="# Heading 1\nSection 1 text here.\n## Heading 2\nSection 2 text here.",
            source_file="/tmp/doc.md",
            file_name="doc.md",
            file_type=".md"
        )
        chunks = chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 2)

    def test_template_prompt_builder(self):
        builder = TemplatePromptBuilder()
        results = [
            SearchResult(
                chunk_id="1",
                doc_id="1",
                text="Vacation days: 15 per year.",
                file_name="vacation.md",
                source_file="/path/vacation.md",
                file_type=".md",
                page_number=1,
                score=0.9
            )
        ]
        messages = builder.build_prompt("How many vacation days?", results, style="detailed")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("15 per year", messages[1]["content"])

    def test_noop_reranker(self):
        reranker = NoOpReranker()
        results = [
            SearchResult("1", "1", "text 1", "f1", "p1", ".txt"),
            SearchResult("2", "2", "text 2", "f2", "p2", ".txt"),
        ]
        reranked = reranker.rerank("query", results, top_n=1)
        self.assertEqual(len(reranked), 1)
        self.assertEqual(reranked[0].chunk_id, "1")

    def test_noop_query_rewriter(self):
        rewriter = NoOpQueryRewriter()
        q = "what is the policy?"
        self.assertEqual(rewriter.rewrite(q), q)

    def test_run_uploaded_files(self):
        from unittest.mock import MagicMock
        from ingestion.pipeline import IngestionPipeline

        class DummyFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content
            def getvalue(self):
                return self.content

        dummy_file = DummyFile("test.txt", b"Sample uploaded document content for testing pipeline ingestion.")
        
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.side_effect = lambda chunks: chunks
        mock_embedder.get_dimension.return_value = 384

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"doc_count": 1}
        mock_store.index_chunks.return_value = 1

        pipeline = IngestionPipeline(embedder=mock_embedder, vector_store=mock_store)
        res = pipeline.run_uploaded_files([dummy_file])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["doc_count"], 1)


if __name__ == "__main__":
    unittest.main()

