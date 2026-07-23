"""
Embedder Services — Abstract interface & SentenceTransformer implementation.
"""

from abc import ABC, abstractmethod
import logging
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import config
from domain.models import Chunk

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract Base Class for text embedding services."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> List[Chunk]:
        """Populate the 'embedding' property of Chunk objects."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return vector dimension size."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedding service using SentenceTransformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    def embed_text(self, text: str) -> List[float]:
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        vecs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return vecs.tolist()

    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> List[Chunk]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        vectors = self.embed_batch(texts, batch_size=batch_size)
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return chunks

    def get_dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
