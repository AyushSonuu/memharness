# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Base mixin for memory type implementations.

This module provides the BaseMixin class containing utility methods
used by all memory type mixins.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memharness.backends.protocol import BackendProtocol
    from memharness.core.config import MemharnessConfig
    from memharness.types import MemoryType, MemoryUnit

__all__ = ["BaseMixin"]


class BaseMixin:
    """
    Base mixin providing utility methods for memory operations.

    This mixin contains helper methods used by all memory type mixins:
    - Connection checking
    - ID generation
    - Namespace building
    - Embedding generation
    - Memory unit creation
    """

    # Type hints for attributes that will be provided by MemoryHarness
    _backend: BackendProtocol
    _embedding_fn: Any
    _config: MemharnessConfig
    _namespace_prefix: tuple[str, ...]
    _connected: bool

    def _check_connected(self) -> None:
        """
        Check if the harness is connected to the backend.

        Raises:
            RuntimeError: If not connected to the backend.
        """
        if not self._connected:
            raise RuntimeError(
                "Not connected to backend. Call await harness.connect() first "
                "or use the async context manager: async with MemoryHarness(...) as harness:"
            )

    def _generate_id(self) -> str:
        """Generate a unique memory ID."""
        return str(uuid.uuid4())

    def _build_namespace(
        self,
        memory_type: MemoryType,
        *parts: str,
    ) -> tuple[str, ...]:
        """Build a full namespace including the prefix."""
        return self._namespace_prefix + (memory_type.value,) + parts

    async def _embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text."""
        if text is None:
            raise TypeError("Cannot embed None text")
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        return self._embedding_fn(text)

    def _create_unit(
        self,
        content: str,
        memory_type: MemoryType,
        namespace: tuple[str, ...],
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryUnit:
        """Create a new MemoryUnit with generated ID and timestamps."""
        # Import here to avoid circular imports
        from memharness.types import MemoryUnit

        now = datetime.now(UTC)
        return MemoryUnit(
            id=self._generate_id(),
            content=content,
            memory_type=memory_type,
            namespace=namespace,
            embedding=embedding,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
