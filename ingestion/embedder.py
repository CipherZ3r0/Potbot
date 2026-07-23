"""
Embedder — Generates vector embeddings using a local sentence-transformers model.

All embeddings are computed on-device, so document content never leaves the machine.
"""

import logging
from typing import Generator

from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

# Module-level cache for the model
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string and return the vector."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a batch of texts and return a list of vectors."""
    model = _get_model()
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    return embeddings.tolist()


def embed_chunks(
    chunks: Generator[dict, None, None] | list[dict],
    batch_size: int = 64,
) -> list[dict]:
    """
    Add embedding vectors to chunk records.

    Takes chunks from the chunker and adds an 'embedding' field to each.
    Processes in batches for efficiency.

    Returns a list (not generator) since we need to batch the embedding calls.
    """
    chunk_list = list(chunks)
    if not chunk_list:
        logger.warning("No chunks to embed")
        return []

    logger.info(f"Embedding {len(chunk_list)} chunks in batches of {batch_size}...")
    texts = [c["text"] for c in chunk_list]
    vectors = embed_texts(texts, batch_size=batch_size)

    for chunk, vector in zip(chunk_list, vectors):
        chunk["embedding"] = vector

    logger.info(f"Embedded {len(chunk_list)} chunks successfully")
    return chunk_list


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding model's output."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
