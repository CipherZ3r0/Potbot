"""
Document Loaders — Strategy & Factory patterns for loading multi-format documents.
"""

from abc import ABC, abstractmethod
import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

try:
    import chardet
except ImportError:
    chardet = None
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

from domain.models import Document

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """Abstract Base Class for file format loaders."""

    @abstractmethod
    def can_load(self, file_extension: str) -> bool:
        """Return True if this loader handles the given extension."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> List[Document]:
        """Extract text content and metadata from file into Document objects."""
        pass

    @staticmethod
    def _get_modified_date(filepath: str) -> str:
        stat = os.stat(filepath)
        return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF documents using PyMuPDF."""

    def can_load(self, file_extension: str) -> bool:
        return file_extension.lower() == ".pdf"

    def load(self, filepath: str) -> List[Document]:
        if fitz is None:
            logger.error("PyMuPDF (fitz) is not installed.")
            return []
        documents = []
        try:
            doc = fitz.open(filepath)
            modified_date = self._get_modified_date(filepath)
            file_name = os.path.basename(filepath)
            abs_path = os.path.abspath(filepath)

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    documents.append(
                        Document(
                            text=text,
                            source_file=abs_path,
                            file_name=file_name,
                            file_type=".pdf",
                            page_number=page_num,
                            total_pages=len(doc),
                            modified_date=modified_date,
                        )
                    )
            doc.close()
        except Exception as e:
            logger.error(f"Failed to read PDF '{filepath}': {e}")
        return documents


class DocxDocumentLoader(BaseDocumentLoader):
    """Loader for Microsoft Word (.docx) files."""

    def can_load(self, file_extension: str) -> bool:
        return file_extension.lower() == ".docx"

    def load(self, filepath: str) -> List[Document]:
        if DocxDocument is None:
            logger.error("python-docx is not installed.")
            return []
        try:
            doc = DocxDocument(filepath)
            full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
            if full_text.strip():
                return [
                    Document(
                        text=full_text,
                        source_file=os.path.abspath(filepath),
                        file_name=os.path.basename(filepath),
                        file_type=".docx",
                        modified_date=self._get_modified_date(filepath),
                    )
                ]
        except Exception as e:
            logger.error(f"Failed to read DOCX '{filepath}': {e}")
        return []


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for plain text (.txt) and Markdown (.md) files."""

    def can_load(self, file_extension: str) -> bool:
        return file_extension.lower() in {".txt", ".md"}

    def load(self, filepath: str) -> List[Document]:
        try:
            with open(filepath, "rb") as f:
                raw = f.read()
            encoding = "utf-8"
            if chardet is not None:
                encoding = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"
            text = raw.decode(encoding, errors="replace").strip()

            if text:
                ext = Path(filepath).suffix.lower()
                return [
                    Document(
                        text=text,
                        source_file=os.path.abspath(filepath),
                        file_name=os.path.basename(filepath),
                        file_type=ext,
                        modified_date=self._get_modified_date(filepath),
                    )
                ]
        except Exception as e:
            logger.error(f"Failed to read text file '{filepath}': {e}")
        return []


class CSVDocumentLoader(BaseDocumentLoader):
    """Loader for CSV documents."""

    def can_load(self, file_extension: str) -> bool:
        return file_extension.lower() == ".csv"

    def load(self, filepath: str) -> List[Document]:
        try:
            with open(filepath, "rb") as f:
                raw = f.read()
            encoding = "utf-8"
            if chardet is not None:
                encoding = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"
            text = raw.decode(encoding, errors="replace")

            reader = csv.DictReader(text.splitlines())
            rows = []
            for row in reader:
                row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                if row_str.strip():
                    rows.append(row_str)

            if rows:
                return [
                    Document(
                        text="\n".join(rows),
                        source_file=os.path.abspath(filepath),
                        file_name=os.path.basename(filepath),
                        file_type=".csv",
                        modified_date=self._get_modified_date(filepath),
                    )
                ]
        except Exception as e:
            logger.error(f"Failed to read CSV '{filepath}': {e}")
        return []


class CompositeDocumentLoader:
    """Composite Loader orchestrating specialized document loaders."""

    def __init__(self, loaders: List[BaseDocumentLoader] = None):
        self.loaders = loaders or [
            PDFDocumentLoader(),
            DocxDocumentLoader(),
            TextDocumentLoader(),
            CSVDocumentLoader(),
        ]

    def load_directory(self, folder_path: str) -> List[Document]:
        """Scan folder recursively and load all supported documents."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        loaded_documents: List[Document] = []
        for root, _, filenames in os.walk(folder):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                filepath = os.path.join(root, fname)
                for loader in self.loaders:
                    if loader.can_load(ext):
                        docs = loader.load(filepath)
                        loaded_documents.extend(docs)
                        break

        logger.info(f"CompositeDocumentLoader loaded {len(loaded_documents)} document sections from '{folder_path}'")
        return loaded_documents
