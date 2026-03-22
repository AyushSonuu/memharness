# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Memory tools for AI agents.

This module provides tools that AI agents can use to explore and manage
their own memory through a simple, LLM-friendly interface.

Tools available:
- memory_search: Search across memory types
- memory_read: Read specific memory by ID
- memory_write: Write new memory
- memory_stats: Get memory statistics
- memory_list: List memories by type
- toolbox_tree: VFS tree view of tools
- toolbox_ls: List tools in a server
- toolbox_grep: Search tools by pattern
- toolbox_cat: Get tool schema
- summary_expand: Expand a summary to original messages

Usage:
    from memharness.tools import get_memory_tools, MemoryToolExecutor

    # Get tool definitions for your agent
    tools = get_memory_tools(memory_harness)

    # Execute tool calls
    executor = MemoryToolExecutor(memory_harness)
    result = await executor.execute("memory_search", query="user preferences")
"""

from memharness.tools.definitions import (
    get_memory_tools,
    get_memory_tools_anthropic,
    get_tool_names,
)
from memharness.tools.executor import MemoryToolExecutor

__all__ = [
    "get_memory_tools",
    "get_memory_tools_anthropic",
    "get_tool_names",
    "MemoryToolExecutor",
]
