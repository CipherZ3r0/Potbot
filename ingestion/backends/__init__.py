"""
Pluggable Embedding Backends.

This package isolates the heavy ML dependencies (PyTorch, ONNX) behind a
simple abstract interface (``EmbeddingBackend``).

Usage::

    from ingestion.backends import TorchEmbeddingBackend, OnnxEmbeddingBackend

    backend = TorchEmbeddingBackend("all-MiniLM-L6-v2", device="cuda")
    vectors = backend.encode(["hello world"], batch_size=32)
"""

from ingestion.backends.base import EmbeddingBackend
from ingestion.backends.torch_backend import TorchEmbeddingBackend
from ingestion.backends.onnx_backend import OnnxEmbeddingBackend

__all__ = ["EmbeddingBackend", "TorchEmbeddingBackend", "OnnxEmbeddingBackend"]
