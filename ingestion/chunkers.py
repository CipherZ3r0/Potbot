"""
Chunkers — Strategy pattern for splitting documents into chunks.
"""

from abc import ABC, abstractmethod
import hashlib
import logging
import re
from typing import List

from domain.models import Document, Chunk

logger = logging.getLogger(__name__)


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


class RecursiveCharacterChunker(BaseChunker):
    """Splits text recursively by paragraphs, sentences, and words."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
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
                        prev_tail = chunks[i - 1][-self.chunk_overlap :]
                        overlapped.append(prev_tail + chunks[i])
                    return overlapped

                return chunks

        # Fallback to hard character split
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            c = text[i : i + self.chunk_size]
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
                # Sub-chunk section using recursive chunker
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


class CompositeChunker:
    """Delegates chunking based on document file type."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.default_chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)
        self.md_chunker = MarkdownHeaderChunker(chunk_size, chunk_overlap)

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in documents:
            if doc.file_type == ".md":
                chunks = self.md_chunker.chunk_document(doc)
            else:
                chunks = self.default_chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(f"CompositeChunker generated {len(all_chunks)} total chunks")
        return all_chunks
