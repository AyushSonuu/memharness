# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Main MemoryHarness class - the primary entry point for memharness.

This module provides the MemoryHarness class which serves as the main interface
for all memory operations. Users interact with this class to store, retrieve,
search, and manage memories across different memory types and backends.

Example:
    async with MemoryHarness("sqlite:///memory.db") as harness:
        await harness.add_conversational("thread1", "user", "Hello!")
        messages = await harness.get_conversational("thread1")
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memharness.backends.protocol import BackendProtocol
from memharness.core.backend_factory import parse_backend
from memharness.core.config import MemharnessConfig
from memharness.core.context import ContextMixin
from memharness.core.embedding import default_embedding_fn
from memharness.memory_types import (
    ConversationalMixin,
    EntityMixin,
    FileMixin,
    GenericMixin,
    KnowledgeMixin,
    PersonaMixin,
    SummaryMixin,
    ToolboxMixin,
    ToolLogMixin,
    WorkflowMixin,
)
from memharness.types import MemoryType

__all__ = ["MemoryHarness"]


class MemoryHarness(
    ConversationalMixin,
    KnowledgeMixin,
    EntityMixin,
    WorkflowMixin,
    ToolboxMixin,
    SummaryMixin,
    ToolLogMixin,
    FileMixin,
    PersonaMixin,
    GenericMixin,
    ContextMixin,
):
    """
    The main entry point for memharness - a framework-agnostic memory layer for AI agents.

    MemoryHarness provides a unified interface for storing, retrieving, and searching
    memories across 9 different memory types. It supports multiple backends (PostgreSQL,
    SQLite, in-memory) and can be configured for various use cases.

    Memory Types:
        - Conversational: Chat history and dialogue
        - Knowledge: Facts and information
        - Entity: Named entities and relationships
        - Workflow: Task procedures and outcomes
        - Toolbox: Tool definitions with VFS interface
        - Summary: Compressed summaries with expansion
        - Tool Log: Tool execution history
        - File: File metadata and content summaries
        - Persona: User/agent persona blocks

    Example:
        ```python
        async with MemoryHarness("sqlite:///memory.db") as harness:
            # Add a conversation message
            await harness.add_conversational("thread1", "user", "Hello!")

            # Search knowledge base
            results = await harness.search_knowledge("Python async programming")

            # Assemble context for an agent
            context = await harness.assemble_context("Help with async", "thread1")
        ```

    Attributes:
        backend: The storage backend instance.
        config: Configuration settings.
        embedding_fn: Function to generate embeddings from text.
        namespace_prefix: Optional namespace prefix for all operations.
    """

    def __init__(
        self,
        backend: str | BackendProtocol = "memory://",
        embedding_fn: Callable[[str], list[float]] | None = None,
        config: MemharnessConfig | None = None,
        namespace_prefix: tuple[str, ...] | None = None,
    ) -> None:
        """
        Initialize a MemoryHarness instance.

        Args:
            backend: Either a connection string (e.g., "memory://", "sqlite:///db.sqlite",
                    "postgresql://...") or a BackendProtocol instance.
            embedding_fn: Optional function to generate embeddings. If not provided,
                         a simple hash-based function is used (not recommended for production).
            config: Optional configuration. Uses defaults if not provided.
            namespace_prefix: Optional namespace prefix applied to all operations.
        """
        # Parse backend string or use provided instance
        if isinstance(backend, str):
            self._backend: BackendProtocol = parse_backend(backend)
        else:
            self._backend = backend

        self._embedding_fn = embedding_fn or default_embedding_fn
        self._config = config or MemharnessConfig()
        self._namespace_prefix = namespace_prefix or ()
        self._connected = False

        # Toolbox VFS cache for tree/ls operations
        self._toolbox_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, path: str) -> MemoryHarness:
        """
        Create a MemoryHarness from a configuration file.

        Args:
            path: Path to a JSON or YAML configuration file.

        Returns:
            A configured MemoryHarness instance.

        Example:
            ```python
            harness = MemoryHarness.from_config("config/memharness.yaml")
            ```
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        content = file_path.read_text()

        if file_path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError as exc:
                raise ImportError("PyYAML required for YAML configs: pip install pyyaml") from exc
        else:
            data = json.loads(content)

        backend = data.get("backend", "memory://")
        config = MemharnessConfig.from_dict(data.get("config", {}))
        namespace_prefix = tuple(data.get("namespace_prefix", []))

        return cls(
            backend=backend,
            config=config,
            namespace_prefix=namespace_prefix if namespace_prefix else None,
        )

    @classmethod
    def from_env(cls) -> MemoryHarness:
        """
        Create a MemoryHarness from environment variables.

        Environment Variables:
            MEMHARNESS_BACKEND: Backend connection string (default: "memory://")
            MEMHARNESS_CONFIG_PATH: Path to config file (optional)
            MEMHARNESS_NAMESPACE: Comma-separated namespace prefix (optional)

        Returns:
            A configured MemoryHarness instance.

        Example:
            ```python
            # With MEMHARNESS_BACKEND=postgresql://localhost/memory
            harness = MemoryHarness.from_env()
            ```
        """
        # Check for config file first
        config_path = os.environ.get("MEMHARNESS_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            return cls.from_config(config_path)

        backend = os.environ.get("MEMHARNESS_BACKEND", "memory://")
        namespace_str = os.environ.get("MEMHARNESS_NAMESPACE", "")
        namespace_prefix = tuple(namespace_str.split(",")) if namespace_str else None

        return cls(
            backend=backend,
            namespace_prefix=namespace_prefix,
        )

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """
        Check if the harness is connected to the backend.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected

    async def connect(self) -> None:
        """
        Establish connection to the backend.

        This method should be called before performing any operations,
        unless using the async context manager.
        """
        await self._backend.connect()
        self._connected = True

    async def disconnect(self) -> None:
        """
        Close connection to the backend.

        This method should be called when done with the harness,
        unless using the async context manager.
        """
        await self._backend.disconnect()
        self._connected = False

    async def __aenter__(self) -> MemoryHarness:
        """Async context manager entry - connects to backend."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Async context manager exit - disconnects from backend."""
        await self.disconnect()

    # =========================================================================
    # Memory Tools (for agents)
    # =========================================================================

    def get_memory_tools(self) -> list[dict[str, Any]]:
        """
        Get tool definitions for agents to explore their memory.

        Returns a list of tool definitions that can be provided to an AI agent,
        allowing the agent to search, retrieve, and manage its own memories.

        Returns:
            List of tool definition dicts compatible with OpenAI/Anthropic format.

        Example:
            ```python
            tools = harness.get_memory_tools()
            # Pass to your LLM API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=tools
            )
            ```
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search through memories by semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "memory_type": {
                                "type": "string",
                                "enum": [t.value for t in MemoryType],
                                "description": "Optional filter by memory type",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results (default 5)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_get",
                    "description": "Retrieve a specific memory by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "The memory ID to retrieve",
                            }
                        },
                        "required": ["memory_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_add",
                    "description": "Add a new memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to remember",
                            },
                            "memory_type": {
                                "type": "string",
                                "enum": [t.value for t in MemoryType],
                                "description": "Type of memory (default: knowledge)",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "toolbox_tree",
                    "description": "View the toolbox structure as a tree",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to start from (default: /)",
                                "default": "/",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "toolbox_cat",
                    "description": "Get full details of a tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_path": {
                                "type": "string",
                                "description": "Path to tool in format: server/tool_name",
                            }
                        },
                        "required": ["tool_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "expand_summary",
                    "description": "Expand a summary to see its source memories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary_id": {
                                "type": "string",
                                "description": "The summary memory ID to expand",
                            }
                        },
                        "required": ["summary_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_conversation_history",
                    "description": "Retrieve conversation history for a thread",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thread_id": {
                                "type": "string",
                                "description": "The conversation thread ID",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum messages to retrieve (default 20)",
                                "default": 20,
                            },
                        },
                        "required": ["thread_id"],
                    },
                },
            },
        ]
