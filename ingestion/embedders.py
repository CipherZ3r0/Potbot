"""
Embedder Services — Abstract interface & SentenceTransformer implementation.

Changes from v1
---------------
* ``BaseEmbedder`` gains ``stream_embed()`` — a generator that processes
  chunks in batches of ``batch_size``, bounding memory to one batch at a time.
* ``SentenceTransformerEmbedder`` gains a ``device`` parameter (resolved via
  :func:`~ingestion.device.resolve_device`) and accepts an optional
  embedding cache (Phase 4 hook — ``cache`` parameter is wired here as None
  by default so Phase 4 can activate it without touching the pipeline).
* All existing public methods (``embed_text``, ``embed_batch``,
  ``embed_chunks``, ``get_dimension``) are unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Iterable, Iterator, List, Optional

from abc import ABC, abstractmethod
import hashlib
import logging
from typing import Iterable, Iterator, List, Optional

import config
from domain.models import Chunk
from ingestion.backends import EmbeddingBackend, TorchEmbeddingBackend
from ingestion.config import PipelineConfig
from ingestion.embed_cache import BaseEmbeddingCache
from ingestion.metrics import PipelineMetrics, StageTimer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

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

    def stream_embed(
        self,
        chunks: Iterable[Chunk],
        batch_size: int = 64,
        metrics: Optional[PipelineMetrics] = None,
    ) -> Iterator[Chunk]:
        """Yield embedded chunks, processing *batch_size* at a time.

        Memory is bounded to one batch of ``batch_size`` chunks at a time,
        regardless of dataset size.  Each batch calls :meth:`embed_chunks`
        so subclass caching / backend logic is respected automatically.

        Parameters
        ----------
        chunks:
            Any iterable of :class:`~domain.models.Chunk` objects, including
            generators from upstream pipeline stages.
        batch_size:
            Number of chunks per model forward pass.
        metrics:
            Optional :class:`~ingestion.metrics.PipelineMetrics` instance.
            ``chunks_embedded`` is updated after each batch.
        """
        batch: List[Chunk] = []

        with StageTimer(metrics or PipelineMetrics(), "embed"):
            for chunk in chunks:
                batch.append(chunk)
                if len(batch) >= batch_size:
                    embedded = self.embed_chunks(batch, batch_size=batch_size)
                    if metrics:
                        metrics.chunks_embedded += len(embedded)
                    yield from embedded
                    batch.clear()

            if batch:
                embedded = self.embed_chunks(batch, batch_size=len(batch))
                if metrics:
                    metrics.chunks_embedded += len(embedded)
                yield from embedded

        if metrics:
            logger.info(
                "stage=embed chunks_embedded=%d cached=%d errors=%d time_s=%.2f",
                metrics.chunks_embedded,
                metrics.chunks_cached,
                metrics.embed_errors,
                metrics.embed_time_s,
            )


# ---------------------------------------------------------------------------
# SentenceTransformer implementation
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding service delegating to a pluggable EmbeddingBackend.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``config.EMBEDDING_MODEL``.
    device:
        Compute device string (``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``).
    cache:
        Optional embedding cache instance.  When provided, the cache is
        consulted before calling the backend.
    config:
        ``PipelineConfig`` — used only to read ``embed_batch_size`` as a
        default; callers can always override per call.
    backend:
        Optional ``EmbeddingBackend`` instance.  Defaults to a
        ``TorchEmbeddingBackend`` using *model_name* and *device*.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        cache: Optional[BaseEmbeddingCache] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._device_preference = device
        self._pipeline_config = pipeline_config or PipelineConfig()
        self._cache = cache
        self._backend = backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._backend = TorchEmbeddingBackend(
                model_name=self.model_name,
                device=self._device_preference,
            )
        return self._backend

    def _hash_text(self, text: str) -> str:
        """Return the SHA-256 hex digest of *text* for caching."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API (unchanged from v1)
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text], batch_size=1)[0]

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []
        backend = self._get_backend()
        return backend.encode(texts, batch_size=batch_size)

    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> List[Chunk]:
        """Embed a list of chunks, optionally consulting the cache first."""
        if not chunks:
            return []

        uncached = []
        # Fast path if cache is missing
        if self._cache is None:
            uncached = chunks
        else:
            # Check cache
            for chunk in chunks:
                h = self._hash_text(chunk.text)
                vec = self._cache.get(h)
                if vec:
                    chunk.embedding = vec
                else:
                    uncached.append(chunk)

        if uncached:
            texts = [c.text for c in uncached]
            vectors = self.embed_batch(texts, batch_size=batch_size)
            
            for chunk, vector in zip(uncached, vectors):
                chunk.embedding = vector
                
                # Write back to cache
                if self._cache is not None:
                    h = self._hash_text(chunk.text)
                    self._cache.put(h, vector)

        logger.info(
            "embedder: %d total chunks (%d cached, %d computed)",
            len(chunks),
            len(chunks) - len(uncached),
            len(uncached),
        )
        return chunks

    def get_dimension(self) -> int:
        return self._get_backend().dimension
