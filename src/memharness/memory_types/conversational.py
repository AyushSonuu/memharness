# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Conversational memory type mixin.

This module provides methods for managing conversational memories
(chat history and dialogue).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["ConversationalMixin"]


class ConversationalMixin(BaseMixin):
    """Mixin for conversational memory operations."""

    async def add_conversational(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a conversational message to memory.

        Args:
            thread_id: Unique identifier for the conversation thread.
            role: The role of the speaker (e.g., "user", "assistant", "system").
            content: The message content.
            metadata: Optional additional metadata (e.g., timestamp, tool_calls).

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            msg_id = await harness.add_conversational(
                thread_id="chat-123",
                role="user",
                content="What's the weather like?",
                metadata={"timestamp": "2024-01-01T12:00:00Z"}
            )
            ```
        """
        from memharness.types import MemoryType

        self._check_connected()
        namespace = self._build_namespace(MemoryType.CONVERSATIONAL, thread_id)
        embedding = await self._embed(content)

        meta = metadata or {}
        meta["role"] = role
        meta["thread_id"] = thread_id

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.CONVERSATIONAL,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def get_conversational(
        self,
        thread_id: str,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        """
        Retrieve conversation history for a thread.

        Args:
            thread_id: The conversation thread ID.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of MemoryUnit objects representing the conversation,
            ordered from oldest to newest.

        Example:
            ```python
            messages = await harness.get_conversational("chat-123", limit=10)
            for msg in messages:
                print(f"{msg.metadata['role']}: {msg.content}")
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.CONVERSATIONAL, thread_id)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.CONVERSATIONAL,
            limit=limit,
        )
        # Sort by created_at ascending (oldest first)
        results.sort(key=lambda u: u.created_at)
        return results
