"""
Chunker — Splits document text into overlapping chunks with metadata.

Supports:
  - Recursive character splitting (general purpose)
  - Markdown-aware splitting (for .md files)
"""

import hashlib
import logging
import re
from typing import Generator

logger = logging.getLogger(__name__)


def _recursive_split(
    text: str, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None
) -> list[str]:
    """
    Split text recursively by trying separators in order.
    Falls back to character-level splitting if no separator works.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator
    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If single part exceeds chunk_size, recurse with next separator
                    if len(part) > chunk_size:
                        remaining_seps = separators[separators.index(sep) + 1:]
                        chunks.extend(
                            _recursive_split(part, chunk_size, chunk_overlap, remaining_seps)
                        )
                        current = ""
                    else:
                        current = part

            if current:
                chunks.append(current)

            # Apply overlap
            if chunk_overlap > 0 and len(chunks) > 1:
                overlapped = [chunks[0]]
                for i in range(1, len(chunks)):
                    prev_tail = chunks[i - 1][-chunk_overlap:]
                    overlapped.append(prev_tail + chunks[i])
                return overlapped

            return chunks

    # Last resort: hard character split
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _markdown_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split markdown by headers first, then by size within each section.
    """
    # Split on markdown headers (##, ###, etc.)
    header_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    sections = header_pattern.split(text)

    chunks = []
    current_header = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if header_pattern.match(section):
            current_header = section
            continue

        # Prepend header to section for context
        full_section = f"{current_header}\n{section}" if current_header else section

        if len(full_section) <= chunk_size:
            chunks.append(full_section)
        else:
            # Further split large sections
            sub_chunks = _recursive_split(full_section, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)

    return chunks if chunks else _recursive_split(text, chunk_size, chunk_overlap)


def _generate_chunk_id(source_file: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from file path and index."""
    raw = f"{source_file}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def chunk_documents(
    documents: Generator[dict, None, None] | list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Generator[dict, None, None]:
    """
    Split documents into chunks and yield enriched chunk records.

    Each yielded dict has:
        - chunk_id: str — unique identifier for the chunk
        - doc_id: str — hash of the source file
        - text: str — chunk text content
        - chunk_index: int — position of this chunk within the document
        - source_file: str
        - file_name: str
        - file_type: str
        - page_number: int | None
        - modified_date: str
    """
    total_chunks = 0

    for doc in documents:
        text = doc["text"]
        file_type = doc.get("file_type", "")

        # Choose splitting strategy
        if file_type == ".md":
            chunks = _markdown_split(text, chunk_size, chunk_overlap)
        else:
            chunks = _recursive_split(text, chunk_size, chunk_overlap)

        doc_id = hashlib.md5(doc["source_file"].encode()).hexdigest()

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            total_chunks += 1
            yield {
                "chunk_id": _generate_chunk_id(doc["source_file"], i),
                "doc_id": doc_id,
                "text": chunk_text.strip(),
                "chunk_index": i,
                "source_file": doc["source_file"],
                "file_name": doc["file_name"],
                "file_type": file_type,
                "page_number": doc.get("page_number"),
                "modified_date": doc.get("modified_date", ""),
            }

    logger.info(f"Generated {total_chunks} chunks")
