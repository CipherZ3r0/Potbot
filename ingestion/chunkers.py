"""
Chunkers — Strategy pattern for splitting documents into chunks.

Changes from v1
---------------
* ``CompositeChunker`` now accepts an optional ``PipelineConfig`` and exposes
  ``stream_chunks()`` — a generator that uses a ``ProcessPoolExecutor`` to
  parallelise CPU-bound chunking across documents.
* ``chunk_documents()`` is preserved exactly — it delegates to ``stream_chunks()``.
* A module-level ``_chunk_doc_worker`` function is required so that
  ``ProcessPoolExecutor`` can pickle the work item (methods cannot be pickled).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import logging
import re
from typing import Iterable, Iterator, List, Optional

from domain.models import Document, Chunk
from ingestion.config import PipelineConfig
from ingestion.metrics import PipelineMetrics, StageTimer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Abstract Base Class for text chunkers."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split a Document into a list of Chunk domain models."""
        pass

    @staticmethod
    def _generate_chunk_id(source_file: str, index: int) -> str:
        raw = f"{source_file}::{index}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _generate_doc_id(source_file: str) -> str:
        return hashlib.md5(source_file.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Concrete chunkers (unchanged from v1)
# ---------------------------------------------------------------------------

class RecursiveCharacterChunker(BaseChunker):
    """Splits text recursively by paragraphs, sentences, and words."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in separators:
            if sep and sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""

                for part in parts:
                    candidate = f"{current}{sep}{part}" if current else part
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        if len(part) > self.chunk_size:
                            next_seps = separators[separators.index(sep) + 1:]
                            chunks.extend(self._split_text(part, next_seps))
                            current = ""
                        else:
                            current = part

                if current:
                    chunks.append(current)

                if self.chunk_overlap > 0 and len(chunks) > 1:
                    overlapped = [chunks[0]]
                    for i in range(1, len(chunks)):
                        prev_tail = chunks[i - 1][-self.chunk_overlap:]
                        overlapped.append(prev_tail + chunks[i])
                    return overlapped

                return chunks

        # Fallback to hard character split
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            c = text[i: i + self.chunk_size]
            if c.strip():
                chunks.append(c)
        return chunks

    def chunk_document(self, doc: Document) -> List[Chunk]:
        raw_chunks = self._split_text(doc.text, self.separators)
        doc_id = self._generate_doc_id(doc.source_file)
        result_chunks = []

        for idx, text in enumerate(raw_chunks):
            if text.strip():
                result_chunks.append(
                    Chunk(
                        chunk_id=self._generate_chunk_id(doc.source_file, idx),
                        doc_id=doc_id,
                        text=text.strip(),
                        chunk_index=idx,
                        source_file=doc.source_file,
                        file_name=doc.file_name,
                        file_type=doc.file_type,
                        page_number=doc.page_number,
                        modified_date=doc.modified_date,
                    )
                )
        return result_chunks


class MarkdownHeaderChunker(BaseChunker):
    """Splits Markdown documents by header sections."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.fallback_chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)

    def chunk_document(self, doc: Document) -> List[Chunk]:
        if doc.file_type != ".md":
            return self.fallback_chunker.chunk_document(doc)

        header_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        sections = header_pattern.split(doc.text)

        raw_sections = []
        current_header = ""

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            if header_pattern.match(sec):
                current_header = sec
                continue
            full_sec = f"{current_header}\n{sec}" if current_header else sec
            raw_sections.append(full_sec)

        doc_id = self._generate_doc_id(doc.source_file)
        chunks: List[Chunk] = []
        idx = 0

        for sec in raw_sections:
            if len(sec) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id=self._generate_chunk_id(doc.source_file, idx),
                        doc_id=doc_id,
                        text=sec,
                        chunk_index=idx,
                        source_file=doc.source_file,
                        file_name=doc.file_name,
                        file_type=doc.file_type,
                        page_number=doc.page_number,
                        modified_date=doc.modified_date,
                    )
                )
                idx += 1
            else:
                sub_doc = Document(
                    text=sec,
                    source_file=doc.source_file,
                    file_name=doc.file_name,
                    file_type=doc.file_type,
                    page_number=doc.page_number,
                    modified_date=doc.modified_date,
                )
                sub_chunks = self.fallback_chunker.chunk_document(sub_doc)
                for sc in sub_chunks:
                    sc.chunk_id = self._generate_chunk_id(doc.source_file, idx)
                    sc.chunk_index = idx
                    chunks.append(sc)
                    idx += 1

        return chunks if chunks else self.fallback_chunker.chunk_document(doc)


class CodeChunker(BaseChunker):
    """Splits source code and structured documents using language-aware boundaries."""

    CODE_EXTENSIONS = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".c", ".cpp", ".h", ".hpp",
        ".java", ".go", ".rs", ".rb", ".php", ".cs", ".sh", ".bash", ".zsh",
        ".sql", ".r", ".swift", ".kt", ".scala", ".lua",
        ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css"
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        code_separators = [
            "\nclass ", "\ndef ", "\nasync def ", "\nfunction ",
            "\npublic class ", "\nprivate ", "\npublic ", "\nfunc ", "\nfn ",
            "\n\n", "\n", " ", ""
        ]
        self.chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=code_separators,
        )

    def chunk_document(self, doc: Document) -> List[Chunk]:
        return self.chunker.chunk_document(doc)


# ---------------------------------------------------------------------------
# Module-level worker — required for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

def _chunk_doc_worker(doc: Document, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """Top-level function that can be pickled by ProcessPoolExecutor.

    Reconstructs the appropriate chunker based on document type so that
    chunker instances (which may hold compiled regex) are created fresh
    inside each worker process rather than being serialised.
    """
    if doc.file_type == ".md":
        chunker: BaseChunker = MarkdownHeaderChunker(chunk_size, chunk_overlap)
    elif doc.file_type in CodeChunker.CODE_EXTENSIONS:
        chunker = CodeChunker(chunk_size, chunk_overlap)
    else:
        chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)
    return chunker.chunk_document(doc)


# ---------------------------------------------------------------------------
# Composite chunker
# ---------------------------------------------------------------------------

class CompositeChunker:
    """Delegates chunking to the appropriate strategy based on document type.

    Parameters
    ----------
    chunk_size:
        Target character count per chunk (passed to all sub-chunkers).
    chunk_overlap:
        Overlap between adjacent chunks in characters.
    config:
        ``PipelineConfig`` controlling worker counts.  Defaults to
        ``PipelineConfig()`` if omitted.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = config or PipelineConfig()
        # Keep concrete instances for the batch API
        self.default_chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)
        self.md_chunker = MarkdownHeaderChunker(chunk_size, chunk_overlap)
        self.code_chunker = CodeChunker(chunk_size, chunk_overlap)


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents and return all chunks.

        **Backward-compatible** — preserves the original batch API.
        Delegates to :meth:`stream_chunks` and materialises the result.
        """
        return list(self.stream_chunks(documents))

    def stream_chunks(
        self,
        documents: Iterable[Document],
        metrics: Optional[PipelineMetrics] = None,
    ) -> Iterator[Chunk]:
        """Yield :class:`~domain.models.Chunk` objects as each document is chunked.

        Uses a ``ProcessPoolExecutor`` so that CPU-bound text splitting runs
        in parallel across worker processes.  Documents are submitted to the
        pool as they arrive from the upstream generator, maintaining the
        streaming property of the pipeline.

        Parameters
        ----------
        documents:
            Any iterable of :class:`~domain.models.Document` objects,
            including generators from :meth:`~ingestion.loaders.CompositeDocumentLoader.stream_directory`.
        metrics:
            Optional :class:`~ingestion.metrics.PipelineMetrics` instance.
            ``chunks_produced`` is updated in-place.
        """
        with StageTimer(metrics or PipelineMetrics(), "chunk"):
            with ProcessPoolExecutor(max_workers=self.config.chunker_workers) as pool:
                futures = {
                    pool.submit(
                        _chunk_doc_worker,
                        doc,
                        self.chunk_size,
                        self.chunk_overlap,
                    ): doc
                    for doc in documents
                }

                for future in as_completed(futures):
                    doc = futures[future]
                    try:
                        chunks = future.result()
                        if metrics:
                            metrics.chunks_produced += len(chunks)
                        yield from chunks
                    except Exception as exc:
                        logger.error(
                            "chunker: failed to chunk '%s': %s",
                            doc.source_file,
                            exc,
                        )

        if metrics:
            logger.info(
                "stage=chunk chunks_produced=%d time_s=%.2f",
                metrics.chunks_produced,
                metrics.chunk_time_s,
            )
