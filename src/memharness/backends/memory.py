# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""In-memory backend for testing and development."""

from typing import Optional
from collections import defaultdict
import copy
import math


class InMemoryBackend:
    """In-memory storage backend for testing and development.

    This backend stores all data in memory using Python dictionaries.
    Data is lost when the process exits. Useful for:
    - Unit testing
    - Development and prototyping
    - Temporary storage needs

    Example:
        backend = InMemoryBackend()
        await backend.connect()
        await backend.write(("kb",), "key1", {"content": "hello"})
        result = await backend.read(("kb",), "key1")
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        self._storage: dict[tuple[str, ...], dict[str, dict]] = defaultdict(dict)
        self._embeddings: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(dict)
        self._connected: bool = False

    async def connect(self) -> None:
        """Initialize the backend (no-op for in-memory)."""
        self._connected = True

    async def disconnect(self) -> None:
        """Clear storage and mark as disconnected."""
        self._storage.clear()
        self._embeddings.clear()
        self._connected = False

    async def write(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: Optional[list[float]] = None
    ) -> str:
        """Write a memory to storage.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace
            value: Memory data as a dictionary
            embedding: Optional embedding vector

        Returns:
            The key of the written memory
        """
        # Deep copy to prevent external mutations
        self._storage[namespace][key] = copy.deepcopy(value)

        if embedding is not None:
            self._embeddings[namespace][key] = embedding.copy()

        return key

    async def read(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> Optional[dict]:
        """Read a single memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace

        Returns:
            Memory data dictionary if found, None otherwise
        """
        value = self._storage.get(namespace, {}).get(key)
        return copy.deepcopy(value) if value is not None else None

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        embedding: Optional[list[float]] = None,
        k: int = 10,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """Search memories using text query or embedding similarity.

        For in-memory backend:
        - If embedding provided, uses cosine similarity
        - Otherwise, does simple text matching on string values

        Args:
            namespace: Hierarchical namespace tuple
            query: Text query for search
            embedding: Optional embedding vector for semantic search
            k: Maximum number of results
            filters: Optional filters to apply

        Returns:
            List of matching memories ordered by relevance
        """
        memories = self._storage.get(namespace, {})

        if not memories:
            return []

        # Apply filters first
        filtered = {}
        for key, value in memories.items():
            if filters:
                if not self._matches_filters(value, filters):
                    continue
            filtered[key] = value

        if not filtered:
            return []

        # If embedding provided, use vector similarity
        if embedding is not None:
            ns_embeddings = self._embeddings.get(namespace, {})
            scored = []

            for key, value in filtered.items():
                stored_embedding = ns_embeddings.get(key)
                if stored_embedding:
                    score = self._cosine_similarity(embedding, stored_embedding)
                    scored.append((score, key, value))
                else:
                    # No embedding, lowest score
                    scored.append((0.0, key, value))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [copy.deepcopy(item[2]) for item in scored[:k]]

        # Text-based search: simple substring matching
        query_lower = query.lower()
        scored = []

        for key, value in filtered.items():
            score = self._text_match_score(value, query_lower)
            if score > 0:
                scored.append((score, key, value))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [copy.deepcopy(item[2]) for item in scored[:k]]

    async def list(
        self,
        namespace: tuple[str, ...],
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[dict]:
        """List memories with optional filtering and ordering.

        Args:
            namespace: Hierarchical namespace tuple
            filters: Optional filters to apply
            order_by: Field to order by (prefix with - for descending)
            limit: Maximum number of results

        Returns:
            List of memory dictionaries
        """
        memories = self._storage.get(namespace, {})

        if not memories:
            return []

        # Apply filters
        results = []
        for value in memories.values():
            if filters:
                if not self._matches_filters(value, filters):
                    continue
            results.append(copy.deepcopy(value))

        # Apply ordering
        if order_by:
            descending = order_by.startswith("-")
            field = order_by[1:] if descending else order_by

            results.sort(
                key=lambda x: x.get(field, ""),
                reverse=descending
            )

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return results

    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> bool:
        """Delete a memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        if namespace in self._storage and key in self._storage[namespace]:
            del self._storage[namespace][key]

            # Also remove embedding if present
            if namespace in self._embeddings and key in self._embeddings[namespace]:
                del self._embeddings[namespace][key]

            return True

        return False

    async def update(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: Optional[list[float]] = None
    ) -> bool:
        """Update an existing memory.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier
            value: New data to merge
            embedding: Optional new embedding

        Returns:
            True if updated, False if not found
        """
        if namespace not in self._storage or key not in self._storage[namespace]:
            return False

        # Merge with existing data
        existing = self._storage[namespace][key]
        existing.update(copy.deepcopy(value))

        if embedding is not None:
            self._embeddings[namespace][key] = embedding.copy()

        return True

    def _matches_filters(self, value: dict, filters: dict) -> bool:
        """Check if a value matches all filters.

        Args:
            value: Memory data
            filters: Filter criteria

        Returns:
            True if all filters match
        """
        for filter_key, filter_value in filters.items():
            if filter_key not in value:
                return False
            if value[filter_key] != filter_value:
                return False
        return True

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _text_match_score(self, value: dict, query: str) -> float:
        """Calculate text match score for a memory.

        Simple scoring: counts how many times query appears in string values.

        Args:
            value: Memory data
            query: Lowercase query string

        Returns:
            Match score (higher is better)
        """
        score = 0.0

        for v in value.values():
            if isinstance(v, str):
                text = v.lower()
                if query in text:
                    score += text.count(query)

        return score

    # Utility methods for testing

    def clear(self) -> None:
        """Clear all stored data without disconnecting."""
        self._storage.clear()
        self._embeddings.clear()

    def get_namespace_count(self, namespace: tuple[str, ...]) -> int:
        """Get count of memories in a namespace.

        Args:
            namespace: Hierarchical namespace tuple

        Returns:
            Number of memories in namespace
        """
        return len(self._storage.get(namespace, {}))

    def get_all_namespaces(self) -> list[tuple[str, ...]]:
        """Get all namespaces with data.

        Returns:
            List of namespace tuples
        """
        return list(self._storage.keys())
