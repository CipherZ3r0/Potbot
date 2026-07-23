"""
Document Loader — Recursively scans a folder and extracts text from
PDF, DOCX, TXT, MD, and CSV files with metadata.
"""

import os
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
import chardet
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def scan_folder(folder_path: str) -> list[str]:
    """Recursively find all supported files in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = []
    for root, _, filenames in os.walk(folder):
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fname))

    logger.info(f"Found {len(files)} supported files in '{folder_path}'")
    return sorted(files)


def _read_text_file(filepath: str) -> str:
    """Read a text file with encoding detection."""
    with open(filepath, "rb") as f:
        raw = f.read()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    return raw.decode(encoding, errors="replace")


def _extract_pdf(filepath: str) -> list[dict]:
    """Extract text from PDF, one record per page."""
    pages = []
    try:
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "page_number": page_num,
                    "total_pages": len(doc),
                })
        doc.close()
    except Exception as e:
        logger.error(f"Failed to read PDF '{filepath}': {e}")
    return pages


def _extract_docx(filepath: str) -> list[dict]:
    """Extract text from DOCX."""
    try:
        doc = DocxDocument(filepath)
        full_text = "\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )
        if full_text.strip():
            return [{"text": full_text, "page_number": None, "total_pages": None}]
    except Exception as e:
        logger.error(f"Failed to read DOCX '{filepath}': {e}")
    return []


def _extract_text(filepath: str) -> list[dict]:
    """Extract text from TXT or MD files."""
    try:
        text = _read_text_file(filepath)
        if text.strip():
            return [{"text": text, "page_number": None, "total_pages": None}]
    except Exception as e:
        logger.error(f"Failed to read text file '{filepath}': {e}")
    return []


def _extract_csv(filepath: str) -> list[dict]:
    """Extract text from CSV — each row becomes a text record."""
    records = []
    try:
        text = _read_text_file(filepath)
        reader = csv.DictReader(text.splitlines())
        rows = []
        for row in reader:
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            if row_text.strip():
                rows.append(row_text)
        if rows:
            # Combine all rows into a single document for chunking
            full_text = "\n".join(rows)
            records.append({
                "text": full_text,
                "page_number": None,
                "total_pages": None,
            })
    except Exception as e:
        logger.error(f"Failed to read CSV '{filepath}': {e}")
    return records


# Dispatcher for file types
_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_text,
    ".md": _extract_text,
    ".csv": _extract_csv,
}


def load_documents(folder_path: str) -> Generator[dict, None, None]:
    """
    Scan a folder and yield document records with metadata.

    Each yielded dict has:
        - text: str — the extracted text
        - source_file: str — absolute path to the file
        - file_name: str — basename of the file
        - file_type: str — file extension
        - page_number: int | None
        - total_pages: int | None
        - modified_date: str — ISO-formatted last-modified date
    """
    files = scan_folder(folder_path)
    total_docs = 0

    for filepath in files:
        ext = Path(filepath).suffix.lower()
        extractor = _EXTRACTORS.get(ext)
        if not extractor:
            continue

        stat = os.stat(filepath)
        modified_date = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()

        records = extractor(filepath)
        for record in records:
            total_docs += 1
            yield {
                "text": record["text"],
                "source_file": os.path.abspath(filepath),
                "file_name": os.path.basename(filepath),
                "file_type": ext,
                "page_number": record.get("page_number"),
                "total_pages": record.get("total_pages"),
                "modified_date": modified_date,
            }

    logger.info(f"Loaded {total_docs} document sections from '{folder_path}'")
