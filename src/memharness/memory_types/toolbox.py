# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Toolbox memory type mixin.

This module provides methods for managing toolbox memories
(tool definitions with VFS interface).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    pass

__all__ = ["ToolboxMixin"]


class ToolboxMixin(BaseMixin):
    """Mixin for toolbox memory operations with VFS interface."""

    # Type hint for toolbox cache (will be provided by MemoryHarness)
    _toolbox_cache: dict[str, dict[str, Any]]

    async def add_tool(
        self,
        server: str,
        tool_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> str:
        """
        Add a tool definition to the toolbox.

        Args:
            server: The server/namespace the tool belongs to (e.g., "github", "slack").
            tool_name: Name of the tool.
            description: Description of what the tool does.
            parameters: JSON Schema of the tool's parameters.

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            tool_id = await harness.add_tool(
                server="github",
                tool_name="create_issue",
                description="Create a new GitHub issue",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["title"]
                }
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.TOOLBOX, server)

        content = f"{server}/{tool_name}: {description}"
        embedding = await self._embed(content)

        meta = {
            "server": server,
            "tool_name": tool_name,
            "description": description,
            "parameters": parameters,
        }

        # Dedup: check if tool already exists by name
        existing = await self._backend.search(
            query_embedding=embedding,
            memory_type=MemoryType.TOOLBOX,
            namespace=namespace,
            k=5,
        )
        for e in existing:
            if e.metadata.get("tool_name") == tool_name and e.metadata.get("server") == server:
                # Update existing tool instead of creating duplicate
                await self._backend.update(
                    e.id,
                    {
                        "content": content,
                        "metadata": meta,
                        "embedding": embedding,
                    },
                )
                return e.id

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.TOOLBOX,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        result = await self._backend.store(unit)

        # Update VFS cache
        if server not in self._toolbox_cache:
            self._toolbox_cache[server] = {}
        self._toolbox_cache[server][tool_name] = {
            "id": result,
            "description": description,
            "parameters": parameters,
        }

        return result

    async def toolbox_tree(self, path: str = "/") -> str:
        """
        Get a tree view of the toolbox virtual filesystem.

        Args:
            path: The path to start from (default "/" for root).

        Returns:
            A tree-formatted string showing the toolbox structure.

        Example:
            ```python
            tree = await harness.toolbox_tree("/")
            print(tree)
            # /
            # ├── github/
            # │   ├── create_issue
            # │   └── list_prs
            # └── slack/
            #     └── send_message
            ```
        """
        from memharness.types import MemoryType

        # Build tree from all toolbox entries
        tools = await self._backend.list_by_namespace(
            namespace=self._namespace_prefix + (MemoryType.TOOLBOX.value,),
            memory_type=MemoryType.TOOLBOX,
            limit=1000,
        )

        # Organize by server
        servers: dict[str, list[str]] = {}
        for tool in tools:
            server = tool.metadata.get("server", "unknown")
            tool_name = tool.metadata.get("tool_name", "unknown")
            if server not in servers:
                servers[server] = []
            servers[server].append(tool_name)

        # Build tree string
        lines = [path]
        server_list = sorted(servers.keys())

        for i, server in enumerate(server_list):
            is_last_server = i == len(server_list) - 1
            prefix = "└── " if is_last_server else "├── "
            lines.append(f"{prefix}{server}/")

            tool_list = sorted(servers[server])
            for j, tool_name in enumerate(tool_list):
                is_last_tool = j == len(tool_list) - 1
                child_prefix = "    " if is_last_server else "│   "
                tool_prefix = "└── " if is_last_tool else "├── "
                lines.append(f"{child_prefix}{tool_prefix}{tool_name}")

        return "\n".join(lines)

    async def toolbox_ls(self, server: str) -> list[str]:
        """
        List all tools in a server/namespace.

        Args:
            server: The server name to list tools from.

        Returns:
            List of tool names in the server.

        Example:
            ```python
            tools = await harness.toolbox_ls("github")
            # ["create_issue", "list_prs", "create_pr"]
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.TOOLBOX, server)
        tools = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.TOOLBOX,
            limit=1000,
        )
        return [t.metadata.get("tool_name", "") for t in tools if t.metadata.get("tool_name")]

    async def toolbox_grep(self, pattern: str) -> list[dict[str, Any]]:
        """
        Search for tools matching a pattern.

        Args:
            pattern: Regex pattern to match against tool names and descriptions.

        Returns:
            List of matching tool info dicts with server, name, and description.

        Example:
            ```python
            matches = await harness.toolbox_grep("create.*")
            # [{"server": "github", "name": "create_issue", "description": "..."}]
            ```
        """
        from memharness.types import MemoryType

        tools = await self._backend.list_by_namespace(
            namespace=self._namespace_prefix + (MemoryType.TOOLBOX.value,),
            memory_type=MemoryType.TOOLBOX,
            limit=1000,
        )

        regex = re.compile(pattern, re.IGNORECASE)
        results = []

        for tool in tools:
            name = tool.metadata.get("tool_name", "")
            desc = tool.metadata.get("description", "")
            server = tool.metadata.get("server", "")

            if regex.search(name) or regex.search(desc):
                results.append(
                    {
                        "server": server,
                        "name": name,
                        "description": desc,
                    }
                )

        return results

    async def toolbox_cat(self, tool_path: str) -> dict[str, Any]:
        """
        Get full details of a tool.

        Args:
            tool_path: Path to the tool in format "server/tool_name".

        Returns:
            Dict with full tool information including parameters.

        Raises:
            ValueError: If tool_path format is invalid.
            KeyError: If tool is not found.

        Example:
            ```python
            tool_info = await harness.toolbox_cat("github/create_issue")
            # {
            #     "server": "github",
            #     "name": "create_issue",
            #     "description": "Create a new GitHub issue",
            #     "parameters": {...}
            # }
            ```
        """
        from memharness.types import MemoryType

        parts = tool_path.strip("/").split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid tool path: {tool_path}. Expected format: server/tool_name")

        server, tool_name = parts
        namespace = self._build_namespace(MemoryType.TOOLBOX, server)

        tools = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.TOOLBOX,
            limit=1000,
        )

        for tool in tools:
            if tool.metadata.get("tool_name") == tool_name:
                return {
                    "server": server,
                    "name": tool_name,
                    "description": tool.metadata.get("description", ""),
                    "parameters": tool.metadata.get("parameters", {}),
                }

        raise KeyError(f"Tool not found: {tool_path}")

    async def search_tools(self, query: str, k: int = 5) -> list:
        """Search for relevant tools using semantic similarity.

        This is the core Toolbox Pattern from L04: instead of stuffing all tools
        into the context, search the toolbox vector DB and return only the
        top-K most relevant tools for the current query.

        Args:
            query: Natural language query describing what tool is needed.
            k: Number of tools to return (default 5).

        Returns:
            List of MemoryUnit objects for matching tools.

        Example:
            ```python
            tools = await harness.search_tools("search the web")
            for t in tools:
                print(f"{t.metadata.get('tool_name')}: {t.content}")
            ```
        """
        from memharness.types import MemoryType

        embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=embedding,
            memory_type=MemoryType.TOOLBOX,
            namespace=self._namespace_prefix + (MemoryType.TOOLBOX.value,),
            k=k,
        )
