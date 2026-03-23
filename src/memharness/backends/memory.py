# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""In-memory backend for testing and development."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from memharness.types import MemoryType, MemoryUnit


class InMemoryBackend:
    """
    Simple in-memory backend for development and testing.

    This backend stores all data in memory and is lost when the process ends.
    It provides basic similarity search using cosine similarity.
    """

    def __init__(self) -> None:
        self._storage: dict[str, MemoryUnit] = {}
        self._connected: bool = False

    async def connect(self) -> None:
        """Mark the backend as connected."""
        self._connected = True

    async def disconnect(self) -> None:
        """Mark the backend as disconnected."""
        self._connected = False

    async def store(self, unit: MemoryUnit) -> str:
        """Store a memory unit in memory."""
        self._storage[unit.id] = unit
        return unit.id

    async def get(self, memory_id: str) -> MemoryUnit | None:
        """Retrieve a memory unit by ID."""
        return self._storage.get(memory_id)

    async def search(
        self,
        query_embedding: list[float],
        memory_type: MemoryType | None = None,
        namespace: tuple[str, ...] | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[MemoryUnit]:
        """Search for memory units using cosine similarity."""
        candidates = []

        for unit in self._storage.values():
            # Filter by memory type
            if memory_type and unit.memory_type != memory_type:
                continue

            # Filter by namespace prefix
            if namespace and not self._namespace_matches(unit.namespace, namespace):
                continue

            # Apply metadata filters
            if filters and not self._matches_filters(unit.metadata, filters):
                continue

            # Calculate similarity if embeddings exist
            if unit.embedding and query_embedding:
                similarity = self._cosine_similarity(query_embedding, unit.embedding)
                candidates.append((similarity, unit))
            else:
                # No embedding, use 0 similarity but still include
                candidates.append((0.0, unit))

        # Sort by similarity descending and return top k
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [unit for _, unit in candidates[:k]]

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a memory unit."""
        if memory_id not in self._storage:
            return False

        unit = self._storage[memory_id]

        if "content" in updates:
            unit.content = updates["content"]
        if "metadata" in updates:
            unit.metadata.update(updates["metadata"])
        if "embedding" in updates:
            unit.embedding = updates["embedding"]

        unit.updated_at = datetime.now(UTC)
        return True

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory unit."""
        if memory_id in self._storage:
            del self._storage[memory_id]
            return True
        return False

    async def list_by_namespace(
        self,
        namespace: tuple[str, ...],
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[MemoryUnit]:
        """List memory units by namespace prefix."""
        results = []

        for unit in self._storage.values():
            if not self._namespace_matches(unit.namespace, namespace):
                continue
            if memory_type and unit.memory_type != memory_type:
                continue
            results.append(unit)
            if len(results) >= limit:
                break

        # Sort by created_at descending
        results.sort(key=lambda u: u.created_at, reverse=True)
        return results[:limit]

    def _namespace_matches(self, unit_ns: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
        """Check if a namespace starts with the given prefix."""
        if len(prefix) > len(unit_ns):
            return False
        return unit_ns[: len(prefix)] == prefix

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
