"""
PyTorch / SentenceTransformers Backend.
"""

import logging
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, AttributeError, OSError, Exception):
    SentenceTransformer = None

from ingestion.backends.base import EmbeddingBackend
from ingestion.device import resolve_device

logger = logging.getLogger(__name__)


class TorchEmbeddingBackend(EmbeddingBackend):
    """PyTorch-based embedding backend using SentenceTransformers.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
    device:
        Compute device string.  ``"auto"`` probes CUDA → MPS → CPU.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed or could not be loaded. "
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        self._device_preference = device
        self._resolved_device: Optional[str] = None
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._resolved_device = resolve_device(self._device_preference)
            logger.info(
                "torch_backend: loading model=%s device=%s",
                self.model_name,
                self._resolved_device,
            )
            self._model = SentenceTransformer(
                self.model_name, device=self._resolved_device
            )
            logger.info("torch_backend: model loaded successfully")
        return self._model

    def encode(self, texts: List[str], batch_size: int) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        vecs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return vecs.tolist()

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    @property
    def device(self) -> str:
        # Resolve device without loading the model if possible,
        # but if we must load it to know, we do.
        if self._resolved_device is None:
            self._resolved_device = resolve_device(self._device_preference)
        return self._resolved_device
