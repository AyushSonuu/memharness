# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Default embedding function for memharness."""

from __future__ import annotations


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
