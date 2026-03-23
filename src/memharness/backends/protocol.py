# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Backend protocol definition for storage backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from memharness.types import MemoryType, MemoryUnit


@runtime_checkable
class BackendProtocol(Protocol):
    """
    Protocol defining the interface that all backends must implement.

    Backends are responsible for the actual storage and retrieval of memory units.
    They handle persistence, indexing, and search operations.
    """

    async def connect(self) -> None:
        """Establish connection to the backend storage."""
        ...

    async def disconnect(self) -> None:
        """Close connection to the backend storage."""
        ...

    async def store(self, unit: MemoryUnit) -> str:
        """Store a memory unit and return its ID."""
        ...

    async def get(self, memory_id: str) -> MemoryUnit | None:
        """Retrieve a memory unit by ID."""
        ...

    async def search(
        self,
        query_embedding: list[float],
        memory_type: MemoryType | None = None,
        namespace: tuple[str, ...] | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[MemoryUnit]:
        """Search for memory units by similarity."""
        ...

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a memory unit."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory unit."""
        ...

    async def list_by_namespace(
        self,
        namespace: tuple[str, ...],
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[MemoryUnit]:
        """List memory units by namespace prefix."""
        ...
