# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Summary memory type mixin.

This module provides methods for managing summary memories
(compressed summaries with expansion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["SummaryMixin"]


class SummaryMixin(BaseMixin):
    """Mixin for summary memory operations."""

    async def add_summary(
        self,
        summary: str,
        source_ids: list[str],
        thread_id: str | None = None,
    ) -> str:
        """
        Add a summary that references source memories.

        Args:
            summary: The summary text.
            source_ids: List of memory IDs that this summary is derived from.
            thread_id: Optional thread ID if this summarizes a conversation.

        Returns:
            The ID of the created summary memory.

        Example:
            ```python
            summary_id = await harness.add_summary(
                summary="User discussed Python async programming and asked about GIL",
                source_ids=["msg-1", "msg-2", "msg-3"],
                thread_id="chat-123"
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.SUMMARY)
        if thread_id:
            namespace = self._build_namespace(MemoryType.SUMMARY, thread_id)

        embedding = await self._embed(summary)

        meta = {
            "source_ids": source_ids,
            "thread_id": thread_id,
        }

        unit = self._create_unit(
            content=summary,
            memory_type=MemoryType.SUMMARY,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def expand_summary(self, summary_id: str) -> list[MemoryUnit]:
        """
        Expand a summary to retrieve its source memories.

        Args:
            summary_id: The ID of the summary to expand.

        Returns:
            List of source MemoryUnit objects that the summary was derived from.

        Raises:
            KeyError: If the summary is not found.

        Example:
            ```python
            sources = await harness.expand_summary("summary-123")
            for source in sources:
                print(f"Source: {source.content[:100]}...")
            ```
        """
        summary = await self._backend.get(summary_id)
        if not summary:
            raise KeyError(f"Summary not found: {summary_id}")

        source_ids = summary.metadata.get("source_ids", [])
        sources = []

        for source_id in source_ids:
            source = await self._backend.get(source_id)
            if source:
                sources.append(source)

        return sources

    async def get_summaries_by_thread(self, thread_id: str) -> list[MemoryUnit]:
        """
        Retrieve all summaries for a conversation thread.

        Args:
            thread_id: The conversation thread ID.

        Returns:
            List of summary MemoryUnit objects for the thread,
            ordered from oldest to newest.

        Example:
            ```python
            summaries = await harness.get_summaries_by_thread("chat-123")
            for summary in summaries:
                print(f"Summary: {summary.content}")
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.SUMMARY, thread_id)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.SUMMARY,
            limit=100,
        )
        # Sort by created_at ascending (oldest first)
        results.sort(key=lambda u: u.created_at)
        return results
