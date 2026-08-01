"""Unit tests for embedders — SentenceTransformerEmbedder."""

import pytest
from unittest.mock import MagicMock
from domain.models import Chunk
from ingestion.embedders import SentenceTransformerEmbedder


class TestSentenceTransformerEmbedder:
    def test_embed_text(self):
        mock_backend = MagicMock()
        mock_backend.encode.return_value = [[0.1] * 384]
        embedder = SentenceTransformerEmbedder(backend=mock_backend)
        vector = embedder.embed_text("test")
        assert len(vector) == 384
        mock_backend.encode.assert_called_once_with(["test"], batch_size=1)

    def test_embed_batch(self):
        mock_backend = MagicMock()
        mock_backend.encode.return_value = [[0.1] * 384, [0.2] * 384]
        embedder = SentenceTransformerEmbedder(backend=mock_backend)
        vectors = embedder.embed_batch(["a", "b"], batch_size=2)
        assert len(vectors) == 2

    def test_embed_chunks_uses_cache(self, sample_chunks):
        mock_backend = MagicMock()
        # Only one chunk should miss the cache
        mock_backend.encode.return_value = [[0.9] * 384]
        
        mock_cache = MagicMock()
        # chunk 0 and 1 hit cache, chunk 2 misses
        def cache_get(h):
            if not hasattr(cache_get, "calls"): cache_get.calls = 0
            cache_get.calls += 1
            if cache_get.calls == 3: return None
            return [0.1] * 384
        mock_cache.get.side_effect = cache_get
        
        embedder = SentenceTransformerEmbedder(backend=mock_backend, cache=mock_cache)
        chunks = embedder.embed_chunks(sample_chunks)
        
        assert len(chunks) == 3
        # Backend should only be called for the 1 missed chunk
        mock_backend.encode.assert_called_once()
        assert len(mock_backend.encode.call_args[0][0]) == 1

    def test_stream_embed(self, sample_chunks):
        mock_backend = MagicMock()
        mock_backend.encode.return_value = [[0.1] * 384] * 3
        embedder = SentenceTransformerEmbedder(backend=mock_backend)
        
        gen = embedder.stream_embed(sample_chunks, batch_size=2)
        import types
        assert isinstance(gen, types.GeneratorType)
        
        result = list(gen)
        assert len(result) == 3
        assert mock_backend.encode.call_count == 2  # batch of 2, then batch of 1
