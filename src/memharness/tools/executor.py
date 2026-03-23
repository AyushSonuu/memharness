# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tool executor for memory operations.

This module provides the MemoryToolExecutor class that handles executing
memory tools called by AI agents, returning nicely formatted string results.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memharness import MemoryHarness


class MemoryToolExecutor:
    """
    Executes memory tools called by AI agents.

    This class maps tool names to their implementations and handles
    executing them with proper error handling and result formatting.

    The executor returns results as formatted strings that are easy
    for LLMs to understand and process.

    Attributes:
        memory: The MemoryHarness instance to operate on.

    Example:
        >>> executor = MemoryToolExecutor(memory_harness)
        >>> result = await executor.execute("memory_search", query="user preferences")
        >>> print(result)
        # Formatted search results
    """

    def __init__(self, memory: MemoryHarness) -> None:
        """
        Initialize the tool executor.

        Args:
            memory: The MemoryHarness instance to operate on.
        """
        self.memory = memory
        self._tools: dict[str, Callable[..., Coroutine[Any, Any, str]]] = {
            "memory_search": self._memory_search,
            "memory_read": self._memory_read,
            "memory_write": self._memory_write,
            "memory_stats": self._memory_stats,
            "memory_list": self._memory_list,
            "toolbox_tree": self._toolbox_tree,
            "toolbox_ls": self._toolbox_ls,
            "toolbox_grep": self._toolbox_grep,
            "toolbox_cat": self._toolbox_cat,
            "summary_expand": self._summary_expand,
        }

    @property
    def available_tools(self) -> list[str]:
        """Returns list of available tool names."""
        return list(self._tools.keys())

    async def execute(self, tool_name: str, **kwargs: Any) -> str:
        """
        Execute a memory tool and return result as formatted string.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Arguments to pass to the tool.

        Returns:
            Formatted string result suitable for LLM consumption.

        Raises:
            ValueError: If the tool name is unknown.
            Exception: Re-raises any exception from tool execution with context.
        """
        if tool_name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            raise ValueError(f"Unknown tool: '{tool_name}'. Available tools: {available}")

        try:
            return await self._tools[tool_name](**kwargs)
        except TypeError as e:
            # Provide helpful error for wrong arguments
            raise TypeError(f"Invalid arguments for tool '{tool_name}': {e}") from e
        except Exception as e:
            # Re-raise with context
            raise type(e)(f"Error executing tool '{tool_name}': {e}") from e

    # =========================================================================
    # Memory Tools
    # =========================================================================

    async def _memory_search(
        self,
        query: str,
        memory_type: str | None = None,
        k: int = 5,
    ) -> str:
        """
        Search memory for relevant information.

        Args:
            query: Natural language search query.
            memory_type: Optional type to filter by.
            k: Number of results to return.

        Returns:
            Formatted search results.
        """
        # Clamp k to valid range
        k = max(1, min(20, k))

        # Perform the search
        results = await self.memory.search(
            query=query,
            memory_type=memory_type,
            k=k,
        )

        if not results:
            return self._format_empty_results(
                "memory_search",
                f"No memories found matching: '{query}'"
                + (f" (type: {memory_type})" if memory_type else ""),
            )

        # Format results
        lines = [
            f"Found {len(results)} memories matching '{query}':",
            "",
        ]

        for i, result in enumerate(results, 1):
            lines.extend(self._format_memory_result(i, result))
            lines.append("")

        return "\n".join(lines)

    async def _memory_read(self, memory_id: str) -> str:
        """
        Read a specific memory by ID.

        Args:
            memory_id: The memory's unique identifier.

        Returns:
            Formatted memory content.
        """
        memory = await self.memory.get(memory_id)

        if memory is None:
            return self._format_empty_results(
                "memory_read",
                f"No memory found with ID: '{memory_id}'",
            )

        lines = [
            f"Memory [{memory_id}]",
            "=" * 50,
            f"Type: {memory.memory_type}",
            f"Created: {self._format_datetime(memory.created_at)}",
        ]

        if memory.updated_at:
            lines.append(f"Updated: {self._format_datetime(memory.updated_at)}")

        lines.extend(
            [
                "",
                "Content:",
                "-" * 30,
                memory.content,
            ]
        )

        if memory.metadata:
            lines.extend(
                [
                    "",
                    "Metadata:",
                    "-" * 30,
                    self._format_metadata(memory.metadata),
                ]
            )

        return "\n".join(lines)

    async def _memory_write(
        self,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Write new memory.

        Args:
            memory_type: Type of memory to write.
            content: Content to store.
            metadata: Optional metadata.

        Returns:
            Confirmation with memory ID.
        """
        # Handle different memory types with their specific APIs
        try:
            if memory_type == "knowledge_base":
                memory_id = await self.memory.add_knowledge(
                    content=content,
                    metadata=metadata or {},
                )
            elif memory_type == "entity":
                # For entity, we need name, type, and description
                # Parse from content or use defaults
                entity_name = metadata.get("name", "Entity") if metadata else "Entity"
                entity_type = metadata.get("type", "unknown") if metadata else "unknown"
                memory_id = await self.memory.add_entity(
                    name=entity_name,
                    entity_type=entity_type,
                    description=content,
                )
            elif memory_type == "workflow":
                # For workflow, we need task, steps, outcome
                task = metadata.get("task", "Workflow task") if metadata else "Workflow task"
                steps = metadata.get("steps", ["Step 1"]) if metadata else ["Step 1"]
                outcome = metadata.get("outcome", "Outcome") if metadata else "Outcome"
                memory_id = await self.memory.add_workflow(
                    task=task,
                    steps=steps,
                    outcome=outcome,
                    result=content,
                )
            elif memory_type == "skills":
                # For skills, we need skill name and description
                skill_name = metadata.get("name", "Skill") if metadata else "Skill"
                memory_id = await self.memory.add_skill(
                    skill_name=skill_name,
                    description=content,
                )
            else:
                valid_types = "knowledge_base, entity, workflow, skills"
                return f"Error: Invalid memory type '{memory_type}'. Valid types: {valid_types}"
        except Exception as e:
            return f"Error writing memory: {e}"

        lines = [
            "Memory written successfully!",
            "",
            f"  ID: {memory_id}",
            f"  Type: {memory_type}",
            f"  Content length: {len(content)} characters",
        ]

        if metadata:
            lines.append(f"  Metadata keys: {', '.join(metadata.keys())}")

        return "\n".join(lines)

    async def _memory_stats(self) -> str:
        """
        Get memory statistics.

        Returns:
            Formatted statistics overview.
        """
        stats = await self.memory.stats()

        lines = [
            "Memory Statistics",
            "=" * 50,
            "",
        ]

        # Overall stats
        total_count = stats.get("total_count", 0)
        total_size = stats.get("total_size_bytes", 0)

        lines.extend(
            [
                f"Total memories: {total_count:,}",
                f"Total size: {self._format_bytes(total_size)}",
                "",
                "By Type:",
                "-" * 30,
            ]
        )

        # Per-type stats
        type_stats = stats.get("by_type", {})
        for memory_type, type_info in sorted(type_stats.items()):
            count = type_info.get("count", 0)
            size = type_info.get("size_bytes", 0)
            lines.append(f"  {memory_type}: {count:,} items ({self._format_bytes(size)})")

        # Health metrics if available
        if "health" in stats:
            health = stats["health"]
            lines.extend(
                [
                    "",
                    "Health:",
                    "-" * 30,
                    f"  Status: {health.get('status', 'unknown')}",
                    f"  Last vacuum: {health.get('last_vacuum', 'never')}",
                    f"  Fragmentation: {health.get('fragmentation', 0):.1%}",
                ]
            )

        return "\n".join(lines)

    async def _memory_list(
        self,
        memory_type: str,
        limit: int = 10,
        offset: int = 0,
        order_by: str = "created_at",
    ) -> str:
        """
        List memories of a specific type.

        Args:
            memory_type: Type to list.
            limit: Max results.
            offset: Skip count.
            order_by: Sort field.

        Returns:
            Formatted memory list.
        """
        # Clamp limit
        limit = max(1, min(50, limit))
        offset = max(0, offset)

        memories = await self.memory.list(
            memory_type=memory_type,
            limit=limit,
            offset=offset,
            order_by=order_by,
        )

        if not memories:
            return self._format_empty_results(
                "memory_list",
                f"No memories found of type '{memory_type}'"
                + (f" (offset: {offset})" if offset > 0 else ""),
            )

        lines = [
            f"Listing {len(memories)} memories of type '{memory_type}'",
            f"(offset: {offset}, ordered by: {order_by})",
            "",
        ]

        for i, memory in enumerate(memories, offset + 1):
            preview = self._truncate(memory.content, 100)
            created = self._format_datetime(memory.created_at)
            lines.append(f"{i}. [{memory.id}] {preview}")
            lines.append(f"   Created: {created}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Toolbox Tools
    # =========================================================================

    async def _toolbox_tree(
        self,
        path: str = "/",
        depth: int = 3,
    ) -> str:
        """
        Display tree view of tools.

        Args:
            path: Starting path.
            depth: Max depth.

        Returns:
            Formatted tree.
        """
        depth = max(1, min(5, depth))

        # Use the harness's toolbox_tree method directly
        try:
            tree_str = await self.memory.toolbox_tree(path=path)
            return tree_str if tree_str else self._format_empty_results(
                "toolbox_tree",
                f"No tools found at path: '{path}'",
            )
        except Exception as e:
            return self._format_empty_results(
                "toolbox_tree",
                f"Error retrieving toolbox tree: {e}",
            )

    async def _toolbox_ls(
        self,
        server: str | None = None,
        verbose: bool = False,
    ) -> str:
        """
        List tools in a server.

        Args:
            server: Server to filter by.
            verbose: Include descriptions.

        Returns:
            Tool listing.
        """
        if not server:
            return self._format_empty_results("toolbox_ls", "Server parameter is required")

        try:
            tools = await self.memory.toolbox_ls(server=server)
            if not tools:
                return self._format_empty_results("toolbox_ls", f"No tools found in server '{server}'")

            lines = [f"[{server}] ({len(tools)} tools)"]
            for tool in tools:
                lines.append(f"  {tool}")
            return "\n".join(lines)
        except Exception as e:
            return self._format_empty_results("toolbox_ls", f"Error listing tools: {e}")

    async def _toolbox_grep(
        self,
        pattern: str,
        case_sensitive: bool = False,
    ) -> str:
        """
        Search tools by pattern.

        Args:
            pattern: Regex pattern.
            case_sensitive: Match case.

        Returns:
            Matching tools.
        """
        try:
            matches = await self.memory.toolbox_grep(pattern=pattern)

            if not matches:
                return self._format_empty_results(
                    "toolbox_grep",
                    f"No tools matching pattern: '{pattern}'",
                )

            lines = [
                f"Found {len(matches)} tools matching '{pattern}':",
                "",
            ]

            for tool in matches:
                tool_name = tool.get("name", "unnamed")
                server = tool.get("server", "unknown")
                desc = self._truncate(tool.get("description", ""), 60)
                lines.append(f"  [{server}] {tool_name}")
                lines.append(f"    {desc}")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return self._format_empty_results("toolbox_grep", f"Error searching tools: {e}")

    async def _toolbox_cat(
        self,
        tool_name: str,
        server: str | None = None,
    ) -> str:
        """
        Get full tool schema.

        Args:
            tool_name: Tool to look up.
            server: Server filter.

        Returns:
            Full tool documentation.
        """
        # Build the tool path
        if server:
            tool_path = f"{server}/{tool_name}"
        else:
            tool_path = tool_name

        try:
            tool = await self.memory.toolbox_cat(tool_path=tool_path)

            if not tool:
                return self._format_empty_results(
                    "toolbox_cat",
                    f"Tool not found: '{tool_path}'",
                )

            lines = [
                f"Tool: {tool.get('name', 'unnamed')}",
                "=" * 50,
                f"Server: {tool.get('server', 'unknown')}",
                "",
                "Description:",
                "-" * 30,
                tool.get("description", "No description available"),
                "",
            ]

            # Parameters
            params = tool.get("parameters", tool.get("input_schema", {}))
            if params:
                lines.extend(
                    [
                        "Parameters:",
                        "-" * 30,
                        self._format_tool_params(params),
                    ]
                )

            return "\n".join(lines)
        except Exception as e:
            return self._format_empty_results("toolbox_cat", f"Error retrieving tool: {e}")

    # =========================================================================
    # Summary Tools
    # =========================================================================

    async def _summary_expand(
        self,
        summary_id: str,
        include_metadata: bool = False,
    ) -> str:
        """
        Expand a summary to original messages.

        Args:
            summary_id: Summary to expand.
            include_metadata: Include message metadata.

        Returns:
            Original messages.
        """
        summary = await self.memory.get_summary(summary_id)

        if summary is None:
            return self._format_empty_results(
                "summary_expand",
                f"Summary not found: '{summary_id}'",
            )

        messages = summary.get("original_messages", [])

        if not messages:
            return self._format_empty_results(
                "summary_expand",
                f"No original messages found for summary '{summary_id}'",
            )

        lines = [
            f"Expanded Summary [{summary_id}]",
            f"Original messages: {len(messages)}",
            "=" * 50,
            "",
        ]

        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            lines.append(f"[{i}] {role.upper()}:")
            lines.append(content)

            if include_metadata and "metadata" in msg:
                lines.append(f"    Metadata: {json.dumps(msg['metadata'])}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Formatting Helpers
    # =========================================================================

    def _format_empty_results(self, tool: str, message: str) -> str:
        """Format a 'no results' message."""
        return f"[{tool}] {message}"

    def _format_memory_result(self, index: int, result: Any) -> list[str]:
        """Format a single memory search result."""
        # Result is a MemoryUnit object (dataclass)
        memory_id = result.id
        memory_type = result.memory_type.value if hasattr(result.memory_type, "value") else str(result.memory_type)
        content = result.content

        # If the result has a score (from search), show it
        score = getattr(result, "score", None)

        preview = self._truncate(content, 150)

        result_lines = [f"{index}. [{memory_type}] ID: {memory_id}"]
        if score is not None:
            result_lines[0] += f" (relevance: {score:.2f})"
        result_lines.append(f"   {preview}")

        return result_lines

    def _format_datetime(self, dt: datetime | str | None) -> str:
        """Format a datetime for display."""
        if dt is None:
            return "unknown"
        if isinstance(dt, str):
            return dt
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _format_bytes(self, size: int) -> str:
        """Format byte size for human reading."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _format_metadata(self, metadata: dict) -> str:
        """Format metadata dict for display."""
        return json.dumps(metadata, indent=2, default=str)

    def _format_tool_params(self, params: dict) -> str:
        """Format tool parameters for display."""
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        lines = []
        for name, schema in sorted(properties.items()):
            req_marker = "*" if name in required else " "
            param_type = schema.get("type", "any")
            desc = schema.get("description", "")
            default = schema.get("default")

            line = f"  {req_marker}{name} ({param_type})"
            if default is not None:
                line += f" [default: {default}]"
            lines.append(line)

            if desc:
                lines.append(f"      {desc}")

        if required:
            lines.append("")
            lines.append("  * = required")

        return "\n".join(lines)

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis."""
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _build_tool_tree(
        self,
        tools: list[dict],
        path: str,
    ) -> dict[str, Any]:
        """Build a tree structure from tools."""
        tree: dict[str, Any] = {}

        # Normalize path
        path = path.strip("/")
        path_parts = path.split("/") if path else []

        for tool in tools:
            server = tool.get("server", "unknown")
            name = tool.get("name", "unnamed")

            # Build path: /server/tool_name
            tool_path = [server, name]

            # Check if tool matches the requested path
            if path_parts:
                if tool_path[: len(path_parts)] != path_parts:
                    continue
                # Remove matched prefix
                tool_path = tool_path[len(path_parts) :]

            # Insert into tree
            current = tree
            for i, part in enumerate(tool_path):
                if i == len(tool_path) - 1:
                    # Leaf node (tool)
                    current[part] = tool
                else:
                    # Directory node
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        return tree

    def _render_tree(
        self,
        tree: dict,
        prefix: str = "",
        depth: int = 3,
        current_depth: int = 0,
    ) -> list[str]:
        """Render tree structure as lines."""
        if current_depth >= depth:
            return ["  " + prefix + "..."]

        lines = []
        items = sorted(tree.items())

        for i, (name, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "\\-- " if is_last else "|-- "
            child_prefix = "    " if is_last else "|   "

            if isinstance(value, dict) and "name" not in value:
                # Directory node
                lines.append(prefix + connector + name + "/")
                lines.extend(
                    self._render_tree(
                        value,
                        prefix + child_prefix,
                        depth,
                        current_depth + 1,
                    )
                )
            else:
                # Tool node
                lines.append(prefix + connector + name)

        return lines
