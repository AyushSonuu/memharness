# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Tests for embedding functions."""

from __future__ import annotations

import pytest

from memharness.core.embedding import (
    create_huggingface_embedding_fn,
    default_embedding_fn,
)


def _check_huggingface_available() -> bool:
    """Check if langchain-huggingface is available."""
    try:
        import langchain_huggingface  # noqa: F401

        return True
    except ImportError:
        return False


class TestDefaultEmbedding:
    """Tests for default hash-based embedding."""

    def test_default_embedding_dimension(self):
        """Test that default embedding returns correct dimension."""
        text = "Hello, world!"
        embedding = default_embedding_fn(text)

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_default_embedding_normalized(self):
        """Test that default embedding is normalized."""
        text = "Test text"
        embedding = default_embedding_fn(text)

        # Calculate norm (should be ~1.0 for normalized vector)
        norm = sum(x * x for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_default_embedding_deterministic(self):
        """Test that default embedding is deterministic."""
        text = "Same text"
        embedding1 = default_embedding_fn(text)
        embedding2 = default_embedding_fn(text)

        assert embedding1 == embedding2

    def test_default_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        text1 = "First text"
        text2 = "Second text"

        embedding1 = default_embedding_fn(text1)
        embedding2 = default_embedding_fn(text2)

        assert embedding1 != embedding2

    def test_default_embedding_empty_string(self):
        """Test default embedding with empty string."""
        embedding = default_embedding_fn("")

        assert len(embedding) == 384
        # Empty string should still produce a valid embedding
        norm = sum(x * x for x in embedding) ** 0.5
        assert norm > 0


class TestHuggingFaceEmbedding:
    """Tests for HuggingFace embedding function."""

    def test_create_huggingface_embedding_fn_import_error(self):
        """Test that ImportError is raised if langchain-huggingface not installed."""
        # This test will pass if langchain-huggingface IS installed
        # (the function won't raise an error)
        # If not installed, it should raise ImportError
        try:
            embed_fn = create_huggingface_embedding_fn()
            # If we get here, the package is installed
            assert callable(embed_fn)
        except ImportError as e:
            # If package not installed, check error message
            assert "langchain-huggingface" in str(e)
            assert "memharness[embeddings]" in str(e)

    @pytest.mark.skipif(
        not _check_huggingface_available(),
        reason="langchain-huggingface not installed",
    )
    def test_huggingface_embedding_basic(self):
        """Test basic HuggingFace embedding functionality."""
        embed_fn = create_huggingface_embedding_fn()

        text = "Hello, world!"
        embedding = embed_fn(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384  # all-MiniLM-L6-v2 default dimension
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.skipif(
        not _check_huggingface_available(),
        reason="langchain-huggingface not installed",
    )
    def test_huggingface_embedding_deterministic(self):
        """Test that HuggingFace embedding is deterministic."""
        embed_fn = create_huggingface_embedding_fn()

        text = "Same text"
        embedding1 = embed_fn(text)
        embedding2 = embed_fn(text)

        # Should be identical (or very close due to floating point)
        assert len(embedding1) == len(embedding2)
        for v1, v2 in zip(embedding1, embedding2, strict=False):
            assert abs(v1 - v2) < 1e-6

    @pytest.mark.skipif(
        not _check_huggingface_available(),
        reason="langchain-huggingface not installed",
    )
    def test_huggingface_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        embed_fn = create_huggingface_embedding_fn()

        text1 = "First text"
        text2 = "Second text"

        embedding1 = embed_fn(text1)
        embedding2 = embed_fn(text2)

        # Embeddings should be different
        assert embedding1 != embedding2

    @pytest.mark.skipif(
        not _check_huggingface_available(),
        reason="langchain-huggingface not installed",
    )
    def test_huggingface_semantic_similarity(self):
        """Test that semantically similar texts have similar embeddings."""
        embed_fn = create_huggingface_embedding_fn()

        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on a mat"
        text3 = "Python programming language"

        emb1 = embed_fn(text1)
        emb2 = embed_fn(text2)
        emb3 = embed_fn(text3)

        # Calculate cosine similarities
        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_1_2 > sim_1_3
