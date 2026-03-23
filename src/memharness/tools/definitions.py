# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tool definitions for memory exploration.

This module defines the tool schemas that can be provided to AI agents,
allowing them to explore and manage their own memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memharness import MemoryHarness


def get_memory_tools(memory: MemoryHarness) -> list[dict]:
    """
    Returns tool definitions that agents can use to explore their memory.

    These definitions follow the OpenAI/Anthropic tool calling format and can
    be directly passed to most LLM APIs.

    Tools:
        1. memory_search - Search across memory types using semantic similarity
        2. memory_read - Read specific memory by ID
        3. memory_write - Write new memory to a specific type
        4. memory_stats - Get memory statistics and usage
        5. memory_list - List memories by type with pagination
        6. toolbox_tree - VFS tree view of available tools
        7. toolbox_ls - List tools in a specific server
        8. toolbox_grep - Search tools by pattern
        9. toolbox_cat - Get detailed tool schema
        10. summary_expand - Expand a summary to original messages

    Args:
        memory: The MemoryHarness instance to create tools for.
                Currently unused but reserved for future dynamic tool generation.

    Returns:
        List of tool definition dictionaries in standard format.

    Example:
        >>> tools = get_memory_tools(memory_harness)
        >>> # Pass to your LLM API
        >>> response = llm.generate(messages, tools=tools)
    """
    return [
        # Memory Search
        {
            "name": "memory_search",
            "description": (
                "Search your memory for relevant information using semantic similarity. "
                "Returns the most relevant memories matching your query. "
                "Use this when you need to recall something but don't know the exact ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what you're looking for",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "conversational",
                            "knowledge_base",
                            "entity",
                            "workflow",
                            "skills",
                            "toolbox",
                            "summary",
                            "tool_log",
                            "file",
                            "persona",
                        ],
                        "description": "Type of memory to search. If not specified, searches all types.",
                    },
                    "k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return (1-20)",
                    },
                },
                "required": ["query"],
            },
        },
        # Memory Read
        {
            "name": "memory_read",
            "description": (
                "Read a specific memory by its ID. "
                "Use this when you have a memory ID from a previous search or reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique identifier of the memory to read",
                    },
                },
                "required": ["memory_id"],
            },
        },
        # Memory Write
        {
            "name": "memory_write",
            "description": (
                "Write new information to memory. "
                "Use this to persist important facts, learnings, or observations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "knowledge_base",
                            "entity",
                            "workflow",
                            "skills",
                        ],
                        "description": (
                            "Type of memory to write to. "
                            "knowledge_base: General facts and information. "
                            "entity: Information about specific people, organizations, or things. "
                            "workflow: Procedures, processes, and how-to information. "
                            "skills: Learned capabilities and techniques."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to store in memory",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata to attach (e.g., source, tags, confidence)",
                    },
                },
                "required": ["memory_type", "content"],
            },
        },
        # Memory Stats
        {
            "name": "memory_stats",
            "description": (
                "Get statistics about your memory usage. "
                "Shows counts, sizes, and health metrics for each memory type."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        # Memory List
        {
            "name": "memory_list",
            "description": (
                "List memories of a specific type with pagination. "
                "Use this to browse memories without searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "conversational",
                            "knowledge_base",
                            "entity",
                            "workflow",
                            "skills",
                            "toolbox",
                            "summary",
                            "tool_log",
                            "file",
                            "persona",
                        ],
                        "description": "Type of memory to list",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results (1-50)",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Number of results to skip for pagination",
                    },
                    "order_by": {
                        "type": "string",
                        "enum": ["created_at", "updated_at", "relevance"],
                        "default": "created_at",
                        "description": "Field to sort by",
                    },
                },
                "required": ["memory_type"],
            },
        },
        # Toolbox Tree
        {
            "name": "toolbox_tree",
            "description": (
                "Display a tree view of available tools in your toolbox. "
                "Shows tools organized by server/category in a hierarchical view."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": "/",
                        "description": "Path to start the tree from (e.g., '/' for root, '/mcp/filesystem')",
                    },
                    "depth": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum depth to display (1-5)",
                    },
                },
                "required": [],
            },
        },
        # Toolbox List
        {
            "name": "toolbox_ls",
            "description": (
                "List tools in a specific server or category. "
                "Similar to 'ls' command for exploring your toolbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Server name to list tools from (e.g., 'filesystem', 'github')",
                    },
                    "verbose": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include tool descriptions in output",
                    },
                },
                "required": [],
            },
        },
        # Toolbox Grep
        {
            "name": "toolbox_grep",
            "description": (
                "Search for tools by name or description pattern. "
                "Useful when you know part of a tool name or what it does."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for in tool names and descriptions",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether the search is case-sensitive",
                    },
                },
                "required": ["pattern"],
            },
        },
        # Toolbox Cat
        {
            "name": "toolbox_cat",
            "description": (
                "Get the full schema and documentation for a specific tool. "
                "Shows parameters, types, and usage examples."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool to get details for",
                    },
                    "server": {
                        "type": "string",
                        "description": "Server the tool belongs to (if ambiguous)",
                    },
                },
                "required": ["tool_name"],
            },
        },
        # Summary Expand
        {
            "name": "summary_expand",
            "description": (
                "Expand a conversation summary to see the original messages. "
                "Use this when you need full context from a summarized conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_id": {
                        "type": "string",
                        "description": "ID of the summary to expand",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include message metadata (timestamps, token counts)",
                    },
                },
                "required": ["summary_id"],
            },
        },
    ]


def get_memory_tools_anthropic(memory: MemoryHarness) -> list[dict]:
    """
    Returns tool definitions in Anthropic's Claude format.

    Anthropic uses a slightly different schema with 'input_schema' instead
    of 'parameters'.

    Args:
        memory: The MemoryHarness instance.

    Returns:
        List of tool definitions in Anthropic format.
    """
    tools = get_memory_tools(memory)
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in tools
    ]


def get_tool_names() -> list[str]:
    """
    Returns a list of all available memory tool names.

    Returns:
        List of tool name strings.
    """
    return [
        "memory_search",
        "memory_read",
        "memory_write",
        "memory_stats",
        "memory_list",
        "toolbox_tree",
        "toolbox_ls",
        "toolbox_grep",
        "toolbox_cat",
        "summary_expand",
    ]
