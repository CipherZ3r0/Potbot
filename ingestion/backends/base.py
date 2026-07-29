"""
Abstract Embedding Backend Interface.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingBackend(ABC):
    """Abstract Base Class for heavy ML inference backends."""

    @abstractmethod
    def encode(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Run a forward pass on *texts* and return their vectors.

        Outputs must be L2-normalized.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the compute device (e.g., 'cpu', 'cuda', 'mps')."""
        pass
