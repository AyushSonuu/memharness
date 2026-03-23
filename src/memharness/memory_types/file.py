# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
File memory type mixin.

This module provides methods for managing file memories
(file metadata and content summaries).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["FileMixin"]


class FileMixin(BaseMixin):
    """Mixin for file memory operations."""

    async def add_file(
        self,
        path: str,
        content_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a file reference to memory.

        Args:
            path: Path to the file.
            content_summary: Optional summary of the file contents.
            metadata: Optional additional metadata (size, type, etc.).

        Returns:
            The ID of the created file memory.

        Example:
            ```python
            file_id = await harness.add_file(
                path="/src/main.py",
                content_summary="Main application entry point with FastAPI setup",
                metadata={"size": 2048, "language": "python"}
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.FILE)

        content = f"File: {path}"
        if content_summary:
            content += f"\n{content_summary}"

        embedding = await self._embed(content)

        meta = metadata or {}
        meta["path"] = path
        if content_summary:
            meta["content_summary"] = content_summary

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.FILE,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_files(
        self,
        query: str,
        k: int = 5,
    ) -> list[MemoryUnit]:
        """
        Search for files by content or path.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of matching file MemoryUnit objects.

        Example:
            ```python
            files = await harness.search_files("FastAPI application")
            for f in files:
                print(f"File: {f.metadata['path']}")
            ```
        """
        from memharness.types import MemoryType

        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.FILE,
            k=k,
        )
