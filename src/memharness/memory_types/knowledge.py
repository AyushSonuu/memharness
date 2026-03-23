# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Knowledge memory type mixin.

This module provides methods for managing knowledge base memories
(facts and information).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["KnowledgeMixin"]


class KnowledgeMixin(BaseMixin):
    """Mixin for knowledge base memory operations."""

    async def add_knowledge(
        self,
        content: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add knowledge to the knowledge base.

        Args:
            content: The knowledge content (fact, information, etc.).
            source: Optional source of the knowledge (URL, document name, etc.).
            metadata: Optional additional metadata.

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            kb_id = await harness.add_knowledge(
                content="Python's GIL prevents true parallelism in CPU-bound threads.",
                source="Python Documentation",
                metadata={"category": "programming", "language": "python"}
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.KNOWLEDGE)
        embedding = await self._embed(content)

        meta = metadata or {}
        if source:
            meta["source"] = source

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.KNOWLEDGE,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_knowledge(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryUnit]:
        """
        Search the knowledge base by semantic similarity.

        Args:
            query: The search query.
            k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching MemoryUnit objects, ordered by relevance.

        Example:
            ```python
            results = await harness.search_knowledge(
                query="Python concurrency",
                k=3,
                filters={"category": "programming"}
            )
            ```
        """
        from memharness.types import MemoryType

        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.KNOWLEDGE,
            namespace=self._namespace_prefix + (MemoryType.KNOWLEDGE.value,)
            if self._namespace_prefix
            else None,
            filters=filters,
            k=k,
        )
