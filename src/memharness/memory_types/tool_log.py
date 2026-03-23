# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tool log memory type mixin.

This module provides methods for managing tool log memories
(tool execution history).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["ToolLogMixin"]


class ToolLogMixin(BaseMixin):
    """Mixin for tool log memory operations."""

    async def add_tool_log(
        self,
        thread_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        status: str,
    ) -> str:
        """
        Log a tool execution.

        Args:
            thread_id: The conversation thread ID.
            tool_name: Name of the executed tool.
            args: Arguments passed to the tool.
            result: Result or output from the tool.
            status: Execution status ("success", "error", "timeout").

        Returns:
            The ID of the created log entry.

        Example:
            ```python
            log_id = await harness.add_tool_log(
                thread_id="chat-123",
                tool_name="github/create_issue",
                args={"title": "Bug fix", "body": "Fixed the bug"},
                result="Issue #42 created",
                status="success"
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.TOOL_LOG, thread_id)

        content = f"Tool: {tool_name}\nStatus: {status}\nResult: {result}"
        embedding = await self._embed(content)

        meta = {
            "thread_id": thread_id,
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "status": status,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.TOOL_LOG,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def get_tool_log(
        self,
        thread_id: str,
        limit: int = 20,
    ) -> list[MemoryUnit]:
        """
        Retrieve tool execution log for a thread.

        Args:
            thread_id: The conversation thread ID.
            limit: Maximum number of log entries to retrieve.

        Returns:
            List of tool log MemoryUnit objects, ordered from oldest to newest.

        Example:
            ```python
            logs = await harness.get_tool_log("chat-123", limit=10)
            for log in logs:
                print(f"{log.metadata['tool_name']}: {log.metadata['status']}")
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.TOOL_LOG, thread_id)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.TOOL_LOG,
            limit=limit,
        )
        results.sort(key=lambda u: u.created_at)
        return results

    async def log_tool_execution(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        output_result: dict[str, Any] | None = None,
        success: bool = True,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> str:
        """
        Log a tool execution.

        Args:
            tool_name: Name of the tool executed.
            input_params: Input parameters passed to the tool.
            output_result: Output result from the tool.
            success: Whether the execution was successful.
            duration_ms: Duration in milliseconds.
            error: Error message if execution failed.

        Returns:
            The ID of the created log entry.

        Example:
            ```python
            log_id = await harness.log_tool_execution(
                tool_name="github.create_pr",
                input_params={"title": "Fix bug", "body": "Description"},
                output_result={"pr_number": 123},
                success=True,
                duration_ms=500
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.TOOL_LOG, tool_name)

        # Construct log content
        content_parts = [f"Tool: {tool_name}"]
        content_parts.append(f"Status: {'success' if success else 'error'}")

        if duration_ms:
            content_parts.append(f"Duration: {duration_ms}ms")

        if input_params:
            content_parts.append(f"Input: {json.dumps(input_params, indent=2)}")

        if output_result:
            content_parts.append(f"Output: {json.dumps(output_result, indent=2)}")

        if error:
            content_parts.append(f"Error: {error}")

        content = "\n".join(content_parts)

        metadata = {
            "tool_name": tool_name,
            "status": "success" if success else "error",
            "input": input_params,
            "output": output_result,
            "duration_ms": duration_ms,
            "error": error,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.TOOL_LOG,
            namespace=namespace,
            metadata=metadata,
            embedding=None,  # Tool logs don't need embeddings
        )

        return await self._backend.store(unit)

    async def search_tool_logs(
        self,
        query: str,
        k: int = 10,
    ) -> list[MemoryUnit]:
        """
        Search tool execution logs.

        Args:
            query: Search query (tool name or partial match).
            k: Number of results to return.

        Returns:
            List of matching tool log memory units.

        Example:
            ```python
            logs = await harness.search_tool_logs("github")
            for log in logs:
                print(log.metadata.get("tool_name"))
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.TOOL_LOG)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.TOOL_LOG,
            limit=k,
        )

        # Filter by query if provided
        if query:
            filtered = [
                r
                for r in results
                if query.lower() in r.metadata.get("tool_name", "").lower()
                or query.lower() in r.content.lower()
            ]
            return filtered[:k]

        return results[:k]
