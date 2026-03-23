# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Generic memory operations mixin.

This module provides generic methods for memory operations that work
across all memory types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryType, MemoryUnit

__all__ = ["GenericMixin"]


class GenericMixin(BaseMixin):
    """Mixin for generic memory operations."""

    async def add(
        self,
        content: str,
        memory_type: str | None = None,
        namespace: tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a generic memory unit.

        Args:
            content: The memory content.
            memory_type: Optional memory type string. Defaults to "knowledge".
            namespace: Optional custom namespace.
            metadata: Optional metadata.

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            mem_id = await harness.add(
                content="Important note about the project",
                memory_type="knowledge",
                metadata={"importance": "high"}
            )
            ```
        """
        from memharness.types import MemoryType

        mem_type = MemoryType(memory_type) if memory_type else MemoryType.KNOWLEDGE
        ns = namespace if namespace else self._build_namespace(mem_type)

        embedding = await self._embed(content)

        unit = self._create_unit(
            content=content,
            memory_type=mem_type,
            namespace=ns,
            metadata=metadata,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search(
        self,
        query: str,
        memory_type: str | None = None,
        k: int = 10,
    ) -> list[MemoryUnit]:
        """
        Search across memories.

        Args:
            query: The search query.
            memory_type: Optional memory type to filter by.
            k: Number of results to return.

        Returns:
            List of matching MemoryUnit objects.

        Example:
            ```python
            results = await harness.search("Python programming", k=5)
            ```
        """
        from memharness.types import MemoryType

        query_embedding = await self._embed(query)
        mem_type = MemoryType(memory_type) if memory_type else None

        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=mem_type,
            k=k,
        )

    async def get(self, memory_id: str) -> MemoryUnit | None:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The MemoryUnit if found, None otherwise.

        Example:
            ```python
            memory = await harness.get("mem-123")
            if memory:
                print(memory.content)
            ```
        """
        return await self._backend.get(memory_id)

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a memory unit.

        Args:
            memory_id: The memory ID to update.
            content: Optional new content.
            metadata: Optional metadata to merge.

        Returns:
            True if updated successfully, False if not found.

        Example:
            ```python
            success = await harness.update(
                "mem-123",
                content="Updated content",
                metadata={"updated": True}
            )
            ```
        """
        updates: dict[str, Any] = {}

        if content is not None:
            updates["content"] = content
            updates["embedding"] = await self._embed(content)

        if metadata is not None:
            updates["metadata"] = metadata

        if not updates:
            return True  # Nothing to update

        return await self._backend.update(memory_id, updates)

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory unit.

        Args:
            memory_id: The memory ID to delete.

        Returns:
            True if deleted successfully, False if not found.

        Example:
            ```python
            deleted = await harness.delete("mem-123")
            ```
        """
        return await self._backend.delete(memory_id)

    async def clear_all(self) -> int:
        """
        Clear all memories from the backend.

        This method removes all stored memories. Use with caution!

        Returns:
            The number of memories deleted.

        Example:
            ```python
            count = await harness.clear_all()
            print(f"Deleted {count} memories")
            ```
        """
        from memharness.types import MemoryType

        # For in-memory backend, we can directly clear storage
        if hasattr(self._backend, "_storage"):
            count = len(self._backend._storage)
            self._backend._storage.clear()
            return count

        # For other backends, we need to list and delete all memories
        # This is inefficient but works for all backend types
        count = 0
        for memory_type in MemoryType:
            namespace = (
                self._namespace_prefix + (memory_type.value,)
                if self._namespace_prefix
                else (memory_type.value,)
            )
            memories = await self._backend.list_by_namespace(
                namespace=namespace,
                memory_type=memory_type,
                limit=100000,  # Large limit to get all
            )
            for memory in memories:
                await self._backend.delete(memory.id)
                count += 1

        return count

    async def search_all(
        self,
        query: str,
        k: int = 5,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemoryUnit]:
        """
        Search across multiple memory types.

        Args:
            query: Search query.
            k: Total number of results to return.
            memory_types: Optional list of memory types to search. If None, searches all.

        Returns:
            List of memory units from all searched types, sorted by relevance.

        Example:
            ```python
            results = await harness.search_all("Python", k=10)
            for result in results:
                print(f"{result.memory_type}: {result.content[:50]}")
            ```
        """
        from memharness.types import MemoryType

        if memory_types is None:
            # Search all vector-based types
            memory_types = [
                MemoryType.KNOWLEDGE,
                MemoryType.ENTITY,
                MemoryType.WORKFLOW,
                MemoryType.TOOLBOX,
                MemoryType.SKILLS,
                MemoryType.FILE,
            ]

        all_results = []
        per_type_k = max(1, k // len(memory_types))

        for mem_type in memory_types:
            try:
                results = await self.search(
                    query=query,
                    memory_type=mem_type.value,
                    k=per_type_k,
                )
                all_results.extend(results)
            except Exception:
                # Skip types that fail
                continue

        # Sort by relevance if available, otherwise by recency
        all_results.sort(
            key=lambda x: (
                getattr(x, "score", 0) if hasattr(x, "score") else 0,
                x.created_at,
            ),
            reverse=True,
        )

        return all_results[:k]

    async def get_stats(self) -> dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary containing statistics about stored memories.

        Example:
            ```python
            stats = await harness.get_stats()
            print(f"Total conversations: {stats['conversational']}")
            print(f"Total knowledge: {stats['knowledge']}")
            ```
        """
        from memharness.types import MemoryType

        stats = {}

        for mem_type in MemoryType:
            try:
                namespace = self._build_namespace(mem_type)
                results = await self._backend.list_by_namespace(
                    namespace=namespace,
                    memory_type=mem_type,
                    limit=10000,  # Large limit to count all
                )
                stats[mem_type.value] = len(results)
            except Exception:
                stats[mem_type.value] = 0

        stats["total"] = sum(stats.values())
        return stats

    async def clear_thread(self, thread_id: str) -> int:
        """
        Clear all memories associated with a specific thread.

        Args:
            thread_id: The thread ID to clear.

        Returns:
            Number of memories deleted.

        Example:
            ```python
            deleted = await harness.clear_thread("thread_123")
            print(f"Deleted {deleted} memories")
            ```
        """
        from memharness.types import MemoryType

        # Get all conversational memories for the thread
        # Thread ID is part of the namespace
        namespace = self._build_namespace(MemoryType.CONVERSATIONAL, thread_id)
        memories = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.CONVERSATIONAL,
            limit=10000,
        )

        # Delete each memory
        for memory in memories:
            await self._backend.delete(memory.id)

        return len(memories)
