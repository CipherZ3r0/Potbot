"""
Ingestion Pipeline Orchestrator — Clean OOP Pipeline using Dependency Injection.
"""

import logging
from typing import Dict, Any

from ingestion.loaders import CompositeDocumentLoader
from ingestion.chunkers import CompositeChunker
from ingestion.embedders import BaseEmbedder, SentenceTransformerEmbedder
from ingestion.indexers import BaseVectorStore, ElasticsearchVectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document parsing, chunking, embedding generation, and vector indexing."""

    def __init__(
        self,
        loader: CompositeDocumentLoader = None,
        chunker: CompositeChunker = None,
        embedder: BaseEmbedder = None,
        vector_store: BaseVectorStore = None,
    ):
        self.loader = loader or CompositeDocumentLoader()
        self.chunker = chunker or CompositeChunker()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.vector_store = vector_store or ElasticsearchVectorStore()

    def run(self, folder_path: str, recreate_index: bool = False) -> Dict[str, Any]:
        """Execute the end-to-end ingestion workflow."""
        logger.info(f"Starting IngestionPipeline for folder: {folder_path}")

        # 1. Load Documents
        documents = self.loader.load_directory(folder_path)
        if not documents:
            logger.warning("No documents found to ingest.")
            return {"status": "no_documents", "doc_count": 0, "chunk_count": 0}

        # 2. Chunk Documents
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks generated from documents.")
            return {"status": "no_chunks", "doc_count": len(documents), "chunk_count": 0}

        # 3. Generate Embeddings
        embedded_chunks = self.embedder.embed_chunks(chunks)

        # 4. Prepare & Index in Vector Store
        dim = self.embedder.get_dimension()
        self.vector_store.create_index(dimension=dim, recreate=recreate_index)
        indexed_count = self.vector_store.index_chunks(embedded_chunks)

        # 5. Return execution summary
        stats = self.vector_store.get_stats()
        summary = {
            "status": "success",
            "doc_count": len(documents),
            "chunk_count": len(embedded_chunks),
            "indexed_count": indexed_count,
            "total_in_index": stats.get("doc_count", 0),
        }
        logger.info(f"IngestionPipeline complete: {summary}")
        return summary

    def run_uploaded_files(self, uploaded_files: list, recreate_index: bool = False) -> Dict[str, Any]:
        """Save uploaded file objects to a temporary directory and run ingestion."""
        import tempfile
        import os

        if not uploaded_files:
            return {"status": "no_documents", "doc_count": 0, "chunk_count": 0}

        with tempfile.TemporaryDirectory() as temp_dir:
            for uf in uploaded_files:
                file_path = os.path.join(temp_dir, uf.name)
                # Handle bytes vs UploadedFile objects with getvalue() / read()
                content = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
                with open(file_path, "wb") as f:
                    f.write(content)
            return self.run(folder_path=temp_dir, recreate_index=recreate_index)

