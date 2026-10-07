"""
Unit Tests for potbot domain models, loaders, chunkers, prompt builders, and rerankers.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import config
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

    def test_chunk_no_collision_across_pages(self):
        chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
        docs = [
            Document(
                text="Page one text that is long enough to split.",
                source_file="/tmp/book.pdf",
                file_name="book.pdf",
                file_type=".pdf",
                page_number=page,
            )
            for page in (1, 2)
        ]
        ids_1 = [c.chunk_id for c in chunker.chunk_document(docs[0])]
        ids_2 = [c.chunk_id for c in chunker.chunk_document(docs[1])]
        self.assertTrue(ids_1 and ids_2)
        self.assertFalse(set(ids_1) & set(ids_2), "chunks from different pages must not collide")

    def test_chunk_ids_distinct_for_identical_text_on_different_pages(self):
        chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
        text = "Same content every page"
        make = lambda p: chunker.chunk_document(
            Document(
                text=text,
                source_file="/tmp/book.txt",
                file_name="book.txt",
                file_type=".txt",
                page_number=p,
            )
        )
        ids_1 = [c.chunk_id for c in make(1)]
        ids_2 = [c.chunk_id for c in make(2)]
        self.assertTrue(ids_1 and ids_2)
        self.assertFalse(set(ids_1) & set(ids_2))

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
        mock_embedder.stream_embed.side_effect = lambda chunks, **kw: iter(chunks)
        mock_embedder.get_dimension.return_value = 384

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"doc_count": 1}
        def mock_stream_index(chunks, **kwargs):
            return len(list(chunks))
        mock_store.stream_index.side_effect = mock_stream_index

        pipeline = IngestionPipeline(embedder=mock_embedder, vector_store=mock_store)
        res = pipeline.run_uploaded_files([dummy_file])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["doc_count"], 1)

    def test_code_document_loader_and_chunker(self):
        import tempfile
        import os
        from ingestion.loaders import CodeDocumentLoader, CompositeDocumentLoader
        from ingestion.chunkers import CodeChunker

        loader = CodeDocumentLoader()
        self.assertTrue(loader.can_load(".py"))
        self.assertTrue(loader.can_load(".json"))
        self.assertTrue(loader.can_load(".yaml"))
        self.assertTrue(loader.can_load(".sql"))

        # Test dynamic extension helper
        exts = CompositeDocumentLoader.get_supported_extensions_without_dot()
        self.assertIn("py", exts)
        self.assertIn("json", exts)
        self.assertIn("pdf", exts)

        # Create temporary python file
        py_content = "class DataProcessor:\n    def process(self):\n        pass\n\ndef main():\n    print('hello')\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(py_content)
            temp_path = f.name

        try:
            docs = loader.load(temp_path)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].file_type, ".py")

            chunker = CodeChunker(chunk_size=50, chunk_overlap=10)
            chunks = chunker.chunk_document(docs[0])
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(all(c.file_type == ".py" for c in chunks))
        finally:
            os.remove(temp_path)


class TestRetrievalCandidatePool(unittest.TestCase):

    def test_config_default_is_20(self):
        self.assertEqual(config.RETRIEVAL_CANDIDATE_POOL, 20)

        pool_from_env = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))
        self.assertEqual(config.RETRIEVAL_CANDIDATE_POOL, pool_from_env)

    def test_config_is_positive_int(self):
        self.assertIsInstance(config.RETRIEVAL_CANDIDATE_POOL, int)
        self.assertGreater(config.RETRIEVAL_CANDIDATE_POOL, 0)

    def _build_pipeline(self, search_strategy, reranker):
        from rag.pipeline import RAGPipeline
        from rag.llm_providers import GroqLLMProvider

        llm = MagicMock()
        llm.generate.return_value = {
            "answer": "ok",
            "model": "m",
            "response_time_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        return RAGPipeline(
            search_strategy=search_strategy,
            reranker=reranker,
            query_rewriter=mock_noop_rewriter(),
            prompt_builder=MagicMock(),
            llm_provider=llm,
            repository=MagicMock(),
        )

    def test_reranking_fetches_candidate_pool_not_top_k(self):
        strategy = MagicMock()
        strategy.search.return_value = []
        reranker = MagicMock()
        reranker.rerank.return_value = []
        pipeline = self._build_pipeline(strategy, reranker)

        with patch.object(config, "RETRIEVAL_CANDIDATE_POOL", 20):
            pipeline.query(
                "test query",
                use_reranking=True,
                use_query_rewriting=False,
                top_k=5,
                rerank_top_n=3,
                save_to_db=False,
            )

        strategy.search.assert_called_once()
        call_kwargs = strategy.search.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 20)
        reranker.rerank.assert_called_once()
        self.assertEqual(reranker.rerank.call_args[1]["top_n"], 3)

    def test_reranking_ignores_top_k_override(self):
        strategy = MagicMock()
        strategy.search.return_value = []
        reranker = MagicMock()
        reranker.rerank.return_value = []
        pipeline = self._build_pipeline(strategy, reranker)

        with patch.object(config, "RETRIEVAL_CANDIDATE_POOL", 20):
            pipeline.query(
                "test query",
                use_reranking=True,
                use_query_rewriting=False,
                top_k=99,
                rerank_top_n=3,
                save_to_db=False,
            )

        call_kwargs = strategy.search.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 20)
        self.assertEqual(reranker.rerank.call_args[1]["top_n"], 3)

    def test_no_reranking_uses_top_k(self):
        strategy = MagicMock()
        strategy.search.return_value = []
        reranker = MagicMock()
        pipeline = self._build_pipeline(strategy, reranker)

        with patch.object(config, "RETRIEVAL_CANDIDATE_POOL", 20):
            pipeline.query(
                "test query",
                use_reranking=False,
                use_query_rewriting=False,
                top_k=7,
                rerank_top_n=3,
                save_to_db=False,
            )

        call_kwargs = strategy.search.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 7)

    def test_pipeline_honours_configured_pool(self):
        strategy = MagicMock()
        strategy.search.return_value = []
        reranker = MagicMock()
        reranker.rerank.return_value = []
        pipeline = self._build_pipeline(strategy, reranker)

        with patch.object(config, "RETRIEVAL_CANDIDATE_POOL", 42):
            pipeline.query(
                "test query",
                use_reranking=True,
                use_query_rewriting=False,
                top_k=5,
                rerank_top_n=3,
                save_to_db=False,
            )

        call_kwargs = strategy.search.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 42)


class TestQueryRewritingDefault(unittest.TestCase):

    def _build_pipeline(self, search_strategy, llm_provider, reranker=None):
        from rag.pipeline import RAGPipeline

        return RAGPipeline(
            search_strategy=search_strategy,
            reranker=reranker or MagicMock(),
            query_rewriter=MagicMock(),
            prompt_builder=MagicMock(),
            llm_provider=llm_provider,
            repository=MagicMock(),
        )

    @staticmethod
    def _llm_provider(answer="ok"):
        llm = MagicMock()
        llm.generate.return_value = {
            "answer": answer,
            "model": "m",
            "response_time_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        return llm

    def test_config_flag_defaults_to_false(self):
        self.assertIs(config.USE_QUERY_REWRITING, False)

    def test_pipeline_query_default_is_false(self):
        import inspect
        from rag.pipeline import RAGPipeline

        param = inspect.signature(RAGPipeline.query).parameters["use_query_rewriting"]
        self.assertIs(param.default, False)

    def test_rewriting_disabled_by_default_uses_original_query(self):
        strategy = MagicMock()
        strategy.search.return_value = []
        llm = self._llm_provider("rewritten-query")
        pipeline = self._build_pipeline(strategy, llm)

        response = pipeline.query("original query", save_to_db=False)

        call_kwargs = strategy.search.call_args
        self.assertEqual(call_kwargs[0][0], "original query")  # NOT rewritten
        self.assertEqual(call_kwargs[1]["top_k"], config.RETRIEVAL_CANDIDATE_POOL)
        self.assertIsNone(response.rewritten_query)

    def test_explicit_opt_in_enables_rewriter(self):
        strategy = MagicMock()
        strategy.search.return_value = []
        rewriter_llm = self._llm_provider("rewritten-query")
        pipeline = self._build_pipeline(strategy, rewriter_llm)

        response = pipeline.query(
            "original query", use_query_rewriting=True, save_to_db=False
        )

        call_kwargs = strategy.search.call_args
        self.assertEqual(call_kwargs[0][0], "rewritten-query")
        self.assertEqual(response.rewritten_query, "rewritten-query")

    def test_rewriter_implementation_still_importable(self):
        from rag.query_rewriters import LLMQueryRewriter, NoOpQueryRewriter

        self.assertTrue(callable(LLMQueryRewriter))
        self.assertTrue(callable(NoOpQueryRewriter))


def mock_noop_rewriter():
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "test query"
    return rewriter


if __name__ == "__main__":
    unittest.main()


