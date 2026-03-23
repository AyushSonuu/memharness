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
- toolbox_tree: VFS tree view of tools
- toolbox_grep: Search tools by pattern
- expand_summary: Expand a summary to original messages
- get_conversation_history: Get conversation history for a thread
- assemble_context: Assemble all relevant memory context
- summarize_and_store: Compress conversation history
- write_tool_log: Log tool executions
- write_workflow: Save task as reusable workflow

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
    ConversationHistoryTool,
    ExpandSummaryTool,
    MemoryReadTool,
    MemorySearchTool,
    MemoryStatsTool,
    MemoryWriteTool,
    SummarizeAndStoreTool,
    ToolboxGrepTool,
    ToolboxTreeTool,
    WriteToolLogTool,
    WriteWorkflowTool,
    get_memory_tools,
)
from memharness.tools.executor import MemoryToolExecutor

__all__ = [
    "get_memory_tools",
    "MemoryToolExecutor",
    "MemorySearchTool",
    "MemoryReadTool",
    "MemoryWriteTool",
    "MemoryStatsTool",
    "ToolboxTreeTool",
    "ToolboxGrepTool",
    "ExpandSummaryTool",
    "ConversationHistoryTool",
    "AssembleContextTool",
    "SummarizeAndStoreTool",
    "WriteToolLogTool",
    "WriteWorkflowTool",
    "LANGCHAIN_AVAILABLE",
]
