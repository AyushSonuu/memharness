# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Entity memory type mixin.

This module provides methods for managing entity memories
(named entities and relationships).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["EntityMixin"]


class EntityMixin(BaseMixin):
    """Mixin for entity memory operations."""

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        relationships: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Add an entity to memory.

        Args:
            name: The entity name (e.g., "John Smith", "OpenAI").
            entity_type: Type of entity (e.g., "person", "organization", "location").
            description: Description of the entity.
            relationships: Optional list of relationships, each as a dict with
                          "target" (entity name), "type" (relationship type).

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            entity_id = await harness.add_entity(
                name="Anthropic",
                entity_type="organization",
                description="AI safety company that created Claude",
                relationships=[
                    {"target": "Claude", "type": "created"},
                    {"target": "San Francisco", "type": "headquartered_in"}
                ]
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.ENTITY, entity_type)

        # Create searchable content
        content = f"{name}: {description}"
        embedding = await self._embed(content)

        meta = {
            "name": name,
            "entity_type": entity_type,
            "relationships": relationships or [],
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.ENTITY,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_entity(
        self,
        query: str,
        entity_type: str | None = None,
        k: int = 5,
    ) -> list[MemoryUnit]:
        """
        Search for entities by semantic similarity.

        Args:
            query: The search query.
            entity_type: Optional filter by entity type.
            k: Number of results to return.

        Returns:
            List of matching entity MemoryUnit objects.

        Example:
            ```python
            people = await harness.search_entity(
                query="AI researcher",
                entity_type="person",
                k=5
            )
            ```
        """
        from memharness.types import MemoryType

        query_embedding = await self._embed(query)

        filters = {}
        if entity_type:
            filters["entity_type"] = entity_type

        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.ENTITY,
            filters=filters if filters else None,
            k=k,
        )
