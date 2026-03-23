# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Embedding functions for memharness."""

from __future__ import annotations

from collections.abc import Callable


def default_embedding_fn(text: str) -> list[float]:
    """
    Default embedding function that returns a simple hash-based embedding.

    This is a placeholder that should be replaced with a real embedding
    model in production. It generates a deterministic but low-quality
    embedding based on character hashing.

    Args:
        text: The text to embed.

    Returns:
        A 384-dimensional embedding vector.
    """
    # Simple hash-based embedding for development/testing
    # In production, this should be replaced with a real model
    import hashlib

    dimension = 384
    embedding = [0.0] * dimension

    # Hash the text and use it to seed pseudo-random values
    text_bytes = text.encode("utf-8")
    hash_bytes = hashlib.sha256(text_bytes).digest()

    # Generate embedding values from hash
    for i in range(dimension):
        byte_idx = i % len(hash_bytes)
        embedding[i] = (hash_bytes[byte_idx] - 128) / 128.0

    # Normalize the embedding
    norm = sum(x * x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding


def create_huggingface_embedding_fn(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Callable[[str], list[float]]:
    """
    Create an embedding function using HuggingFace via LangChain.

    This function requires the `langchain-huggingface` package to be installed.
    Install it with: pip install memharness[embeddings]

    Args:
        model_name: The HuggingFace model to use for embeddings.
            Default is 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions).

    Returns:
        A callable that takes text and returns an embedding vector.

    Raises:
        ImportError: If langchain-huggingface is not installed.

    Example:
        ```python
        from memharness.core.embedding import create_huggingface_embedding_fn

        # Create embedding function
        embed_fn = create_huggingface_embedding_fn()

        # Use with MemoryHarness
        harness = MemoryHarness(
            "sqlite:///memory.db",
            embedding_fn=embed_fn
        )
        ```
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        raise ImportError(
            "HuggingFace embeddings require langchain-huggingface. "
            "Install with: pip install memharness[embeddings]"
        ) from e

    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed(text: str) -> list[float]:
        """Embed text using HuggingFace model."""
        return embeddings.embed_query(text)

    return embed
