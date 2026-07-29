"""
ONNX Runtime Backend (Optional).
"""

import logging
from typing import List, Optional

try:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
except (ImportError, AttributeError, OSError, Exception):
    ort = None

from ingestion.backends.base import EmbeddingBackend

logger = logging.getLogger(__name__)


class OnnxEmbeddingBackend(EmbeddingBackend):
    """ONNX Runtime-based embedding backend for high-performance inference.

    Requires ``onnxruntime`` and ``transformers`` (tokenizer only) to be installed.
    You must provide a model that has been exported to ONNX format.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file or a HuggingFace hub id (if local files exist).
    tokenizer_name:
        HuggingFace tokenizer identifier (usually matches the original PyTorch model).
    device:
        Compute device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        device: str = "cpu",
    ) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime and transformers are required for OnnxEmbeddingBackend. "
                "Run: pip install onnxruntime transformers"
            )
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self._device_preference = device
        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer = None

    def _get_session(self):
        if self._session is None:
            providers = ["CPUExecutionProvider"]
            if self._device_preference == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            logger.info("onnx_backend: loading model=%s providers=%s", self.model_path, providers)
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info("onnx_backend: model loaded successfully")
        return self._session, self._tokenizer

    def encode(self, texts: List[str], batch_size: int) -> List[List[float]]:
        if not texts:
            return []

        session, tokenizer = self._get_session()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            
            # Prepare ONNX inputs
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in encoded:
                inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

            # Forward pass
            outputs = session.run(None, inputs)
            token_embeddings = outputs[0]  # usually (batch, seq_len, hidden_size)
            
            # Mean pooling
            mask = inputs["attention_mask"][..., np.newaxis]
            sum_emb = np.sum(token_embeddings * mask, axis=1)
            sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_emb / sum_mask

            # L2 Normalize
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            normalized = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)
            
            all_embeddings.extend(normalized.tolist())

        return all_embeddings

    @property
    def dimension(self) -> int:
        session, _ = self._get_session()
        # Typically the last dimension of the output tensor
        return session.get_outputs()[0].shape[-1]

    @property
    def device(self) -> str:
        return self._device_preference
