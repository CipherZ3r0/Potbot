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

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, AttributeError, OSError, Exception):
    SentenceTransformer = None

import config
from domain.models import Chunk
from ingestion.config import PipelineConfig
from ingestion.device import resolve_device
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
    """Local embedding service using SentenceTransformers.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``config.EMBEDDING_MODEL``.
    device:
        Compute device string (``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``).
        ``"auto"`` resolves at first model load via :func:`~ingestion.device.resolve_device`.
    cache:
        Optional embedding cache instance (injected in Phase 4).  When
        provided, the cache is consulted before calling the model.
        *Not used in this phase — placeholder for DI.*
    config:
        ``PipelineConfig`` — used only to read ``embed_batch_size`` as a
        default; callers can always override per call.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        cache=None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._device_preference = device
        self._resolved_device: Optional[str] = None
        self._model: Optional[SentenceTransformer] = None
        self._cache = cache  # EmbeddingCache | None — wired in Phase 4
        self._pipeline_config = pipeline_config or PipelineConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                )
            self._resolved_device = resolve_device(self._device_preference)
            logger.info(
                "embedder: loading model=%s device=%s",
                self.model_name,
                self._resolved_device,
            )
            self._model = SentenceTransformer(
                self.model_name, device=self._resolved_device
            )
            logger.info("embedder: model loaded successfully")
        return self._model

    # ------------------------------------------------------------------
    # Public API (unchanged from v1)
    # ------------------------------------------------------------------

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
        """Embed a list of chunks, optionally consulting the cache first.

        Cache lookup is a no-op in this phase (``self._cache`` is ``None``).
        Phase 4 injects a real cache here via the constructor.
        """
        if not chunks:
            return []

        uncached = chunks  # all chunks need embedding in this phase

        texts = [c.text for c in uncached]
        vectors = self.embed_batch(texts, batch_size=batch_size)
        for chunk, vector in zip(uncached, vectors):
            chunk.embedding = vector

        logger.info("embedder: embedded %d chunks", len(chunks))
        return chunks

    def get_dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
