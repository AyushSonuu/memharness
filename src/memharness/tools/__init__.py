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
- memory_write: Write to ANY memory type (knowledge, entity, workflow, tool_log, etc.)
- toolbox_search: Discover tools (tree + grep combined)
- expand_summary: Expand a summary to original messages
- assemble_context: Assemble all relevant memory context
- summarize_conversation: Compress conversation history

Usage:
    from memharness.tools import get_memory_tools, MemoryToolExecutor

    # Get tool definitions for your agent
    tools = get_memory_tools(memory_harness)

    # Execute tool calls
    executor = MemoryToolExecutor(memory_harness)
    result = await executor.execute("memory_search", query="user preferences")
"""

from memharness.tools.definitions import (
    LANGCHAIN_AVAILABLE,
    AssembleContextTool,
    ExpandSummaryTool,
    MemoryReadTool,
    MemorySearchTool,
    MemoryWriteTool,
    SummarizeAndStoreTool,
    ToolboxSearchTool,
    get_memory_tools,
)
from memharness.tools.executor import MemoryToolExecutor

__all__ = [
    "get_memory_tools",
    "MemoryToolExecutor",
    "MemorySearchTool",
    "MemoryReadTool",
    "MemoryWriteTool",
    "ToolboxSearchTool",
    "ExpandSummaryTool",
    "AssembleContextTool",
    "SummarizeAndStoreTool",
    "LANGCHAIN_AVAILABLE",
]
