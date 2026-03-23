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
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memharness.backends.memory import InMemoryBackend
from memharness.backends.protocol import BackendProtocol
from memharness.core.embedding import default_embedding_fn
from memharness.types import MemoryType, MemoryUnit

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MemharnessConfig:
    """
    Configuration for the MemoryHarness.

    Attributes:
        default_embedding_model: Name of the default embedding model to use.
        default_k: Default number of results for search operations.
        max_context_tokens: Maximum tokens for context assembly.
        enable_ai_agents: Whether to enable embedded AI agents.
        toolbox_vfs_enabled: Whether to enable virtual filesystem for toolbox.
        persona_max_blocks: Maximum number of persona blocks.
        summary_expansion_depth: How deep to expand summaries.
    """

    default_embedding_model: str = "text-embedding-3-small"
    default_k: int = 10
    max_context_tokens: int = 4000
    enable_ai_agents: bool = True
    toolbox_vfs_enabled: bool = True
    persona_max_blocks: int = 10
    summary_expansion_depth: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemharnessConfig:
        """Create config from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: str) -> MemharnessConfig:
        """Load config from a JSON or YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        content = file_path.read_text()

        if file_path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML config files. Install with: pip install pyyaml"
                ) from exc
        else:
            data = json.loads(content)

        return cls.from_dict(data)


# =============================================================================
# Backend Factory
# =============================================================================


def _parse_backend(backend_uri: str) -> BackendProtocol:
    """
    Parse a backend URI and return the appropriate backend instance.

    Supported URIs:
        - "memory://" -> InMemoryBackend
        - "sqlite:///path/to/db.sqlite" -> SqliteBackend
        - "postgresql://user:pass@host:port/db" -> PostgresBackend

    Args:
        backend_uri: The backend connection string.

    Returns:
        An instance of the appropriate backend.

    Raises:
        ValueError: If the backend URI format is not recognized.
        ImportError: If required backend dependencies are not installed.
    """
    if backend_uri == "memory://" or backend_uri.startswith("memory://"):
        return InMemoryBackend()

    if backend_uri.startswith("sqlite:///"):
        # Extract path from sqlite:///path/to/db.sqlite
        db_path = backend_uri[10:]  # Remove "sqlite:///"
        try:
            from memharness.backends.sqlite import SqliteBackend

            return SqliteBackend(db_path)
        except ImportError as exc:
            raise ImportError(
                "SqliteBackend is not available. Ensure the sqlite backend module is installed."
            ) from exc

    if backend_uri.startswith("postgresql://") or backend_uri.startswith("postgres://"):
        try:
            from memharness.backends.postgres import PostgresBackend

            return PostgresBackend(backend_uri)
        except ImportError as exc:
            raise ImportError(
                "PostgresBackend is not available. Install with: pip install memharness[postgres]"
            ) from exc

    raise ValueError(
        f"Unrecognized backend URI: {backend_uri}. "
        "Supported formats: memory://, sqlite:///path, postgresql://..."
    )


# =============================================================================
# Main Harness Class
# =============================================================================


class MemoryHarness:
    """
    The main entry point for memharness - a framework-agnostic memory layer for AI agents.

    MemoryHarness provides a unified interface for storing, retrieving, and searching
    memories across 10 different memory types. It supports multiple backends (PostgreSQL,
    SQLite, in-memory) and can be configured for various use cases.

    Memory Types:
        - Conversational: Chat history and dialogue
        - Knowledge: Facts and information
        - Entity: Named entities and relationships
        - Workflow: Task procedures and outcomes
        - Toolbox: Tool definitions with VFS interface
        - Summary: Compressed summaries with expansion
        - Tool Log: Tool execution history
        - Skills: Learned capabilities
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
            self._backend: BackendProtocol = _parse_backend(backend)
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
    # Helper Methods
    # =========================================================================

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

    # =========================================================================
    # Conversational Memory
    # =========================================================================

    async def add_conversational(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a conversational message to memory.

        Args:
            thread_id: Unique identifier for the conversation thread.
            role: The role of the speaker (e.g., "user", "assistant", "system").
            content: The message content.
            metadata: Optional additional metadata (e.g., timestamp, tool_calls).

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            msg_id = await harness.add_conversational(
                thread_id="chat-123",
                role="user",
                content="What's the weather like?",
                metadata={"timestamp": "2024-01-01T12:00:00Z"}
            )
            ```
        """
        self._check_connected()
        namespace = self._build_namespace(MemoryType.CONVERSATIONAL, thread_id)
        embedding = await self._embed(content)

        meta = metadata or {}
        meta["role"] = role
        meta["thread_id"] = thread_id

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.CONVERSATIONAL,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def get_conversational(
        self,
        thread_id: str,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        """
        Retrieve conversation history for a thread.

        Args:
            thread_id: The conversation thread ID.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of MemoryUnit objects representing the conversation,
            ordered from oldest to newest.

        Example:
            ```python
            messages = await harness.get_conversational("chat-123", limit=10)
            for msg in messages:
                print(f"{msg.metadata['role']}: {msg.content}")
            ```
        """
        namespace = self._build_namespace(MemoryType.CONVERSATIONAL, thread_id)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.CONVERSATIONAL,
            limit=limit,
        )
        # Sort by created_at ascending (oldest first)
        results.sort(key=lambda u: u.created_at)
        return results

    # =========================================================================
    # Knowledge Base Memory
    # =========================================================================

    async def add_knowledge(
        self,
        content: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add knowledge to the knowledge base.

        Args:
            content: The knowledge content (fact, information, etc.).
            source: Optional source of the knowledge (URL, document name, etc.).
            metadata: Optional additional metadata.

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            kb_id = await harness.add_knowledge(
                content="Python's GIL prevents true parallelism in CPU-bound threads.",
                source="Python Documentation",
                metadata={"category": "programming", "language": "python"}
            )
            ```
        """
        namespace = self._build_namespace(MemoryType.KNOWLEDGE)
        embedding = await self._embed(content)

        meta = metadata or {}
        if source:
            meta["source"] = source

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.KNOWLEDGE,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_knowledge(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryUnit]:
        """
        Search the knowledge base by semantic similarity.

        Args:
            query: The search query.
            k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching MemoryUnit objects, ordered by relevance.

        Example:
            ```python
            results = await harness.search_knowledge(
                query="Python concurrency",
                k=3,
                filters={"category": "programming"}
            )
            ```
        """
        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.KNOWLEDGE,
            namespace=self._namespace_prefix + (MemoryType.KNOWLEDGE.value,)
            if self._namespace_prefix
            else None,
            filters=filters,
            k=k,
        )

    # =========================================================================
    # Entity Memory
    # =========================================================================

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        relationships: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Add an entity to memory.

        Args:
            name: The entity name (e.g., "John Smith", "OpenAI").
            entity_type: Type of entity (e.g., "person", "organization", "location").
            description: Description of the entity.
            relationships: Optional list of relationships, each as a dict with
                          "target" (entity name), "type" (relationship type).

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            entity_id = await harness.add_entity(
                name="Anthropic",
                entity_type="organization",
                description="AI safety company that created Claude",
                relationships=[
                    {"target": "Claude", "type": "created"},
                    {"target": "San Francisco", "type": "headquartered_in"}
                ]
            )
            ```
        """
        namespace = self._build_namespace(MemoryType.ENTITY, entity_type)

        # Create searchable content
        content = f"{name}: {description}"
        embedding = await self._embed(content)

        meta = {
            "name": name,
            "entity_type": entity_type,
            "relationships": relationships or [],
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.ENTITY,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_entity(
        self,
        query: str,
        entity_type: str | None = None,
        k: int = 5,
    ) -> list[MemoryUnit]:
        """
        Search for entities by semantic similarity.

        Args:
            query: The search query.
            entity_type: Optional filter by entity type.
            k: Number of results to return.

        Returns:
            List of matching entity MemoryUnit objects.

        Example:
            ```python
            people = await harness.search_entity(
                query="AI researcher",
                entity_type="person",
                k=5
            )
            ```
        """
        query_embedding = await self._embed(query)

        filters = {}
        if entity_type:
            filters["entity_type"] = entity_type

        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.ENTITY,
            filters=filters if filters else None,
            k=k,
        )

    # =========================================================================
    # Workflow Memory
    # =========================================================================

    async def add_workflow(
        self,
        task: str,
        steps: list[str],
        outcome: str,
        result: str | None = None,
    ) -> str:
        """
        Add a workflow/procedure to memory.

        Args:
            task: Description of the task this workflow accomplishes.
            steps: List of steps to complete the task.
            outcome: Expected outcome of the workflow.
            result: Optional actual result after execution.

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            wf_id = await harness.add_workflow(
                task="Deploy application to production",
                steps=["Run tests", "Build Docker image", "Push to registry", "Update k8s"],
                outcome="Application deployed and healthy",
                result="Deployed v2.1.0 successfully"
            )
            ```
        """
        namespace = self._build_namespace(MemoryType.WORKFLOW)

        # Create searchable content
        steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
        content = f"Task: {task}\nSteps:\n{steps_text}\nOutcome: {outcome}"
        if result:
            content += f"\nResult: {result}"

        embedding = await self._embed(content)

        meta = {
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "result": result,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.WORKFLOW,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_workflow(
        self,
        query: str,
        k: int = 3,
    ) -> list[MemoryUnit]:
        """
        Search for workflows by semantic similarity.

        Args:
            query: The search query (task description, keywords, etc.).
            k: Number of results to return.

        Returns:
            List of matching workflow MemoryUnit objects.

        Example:
            ```python
            workflows = await harness.search_workflow("deploy application", k=3)
            for wf in workflows:
                print(f"Task: {wf.metadata['task']}")
            ```
        """
        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.WORKFLOW,
            k=k,
        )

    # =========================================================================
    # Toolbox Memory (with VFS)
    # =========================================================================

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
        namespace = self._build_namespace(MemoryType.TOOLBOX, server)

        content = f"{server}/{tool_name}: {description}"
        embedding = await self._embed(content)

        meta = {
            "server": server,
            "tool_name": tool_name,
            "description": description,
            "parameters": parameters,
        }

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

    # =========================================================================
    # Summary Memory (with expansion)
    # =========================================================================

    async def add_summary(
        self,
        summary: str,
        source_ids: list[str],
        thread_id: str | None = None,
    ) -> str:
        """
        Add a summary that references source memories.

        Args:
            summary: The summary text.
            source_ids: List of memory IDs that this summary is derived from.
            thread_id: Optional thread ID if this summarizes a conversation.

        Returns:
            The ID of the created summary memory.

        Example:
            ```python
            summary_id = await harness.add_summary(
                summary="User discussed Python async programming and asked about GIL",
                source_ids=["msg-1", "msg-2", "msg-3"],
                thread_id="chat-123"
            )
            ```
        """
        namespace = self._build_namespace(MemoryType.SUMMARY)
        if thread_id:
            namespace = self._build_namespace(MemoryType.SUMMARY, thread_id)

        embedding = await self._embed(summary)

        meta = {
            "source_ids": source_ids,
            "thread_id": thread_id,
        }

        unit = self._create_unit(
            content=summary,
            memory_type=MemoryType.SUMMARY,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def expand_summary(self, summary_id: str) -> list[MemoryUnit]:
        """
        Expand a summary to retrieve its source memories.

        Args:
            summary_id: The ID of the summary to expand.

        Returns:
            List of source MemoryUnit objects that the summary was derived from.

        Raises:
            KeyError: If the summary is not found.

        Example:
            ```python
            sources = await harness.expand_summary("summary-123")
            for source in sources:
                print(f"Source: {source.content[:100]}...")
            ```
        """
        summary = await self._backend.get(summary_id)
        if not summary:
            raise KeyError(f"Summary not found: {summary_id}")

        source_ids = summary.metadata.get("source_ids", [])
        sources = []

        for source_id in source_ids:
            source = await self._backend.get(source_id)
            if source:
                sources.append(source)

        return sources

    # =========================================================================
    # Tool Log Memory
    # =========================================================================

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
        namespace = self._build_namespace(MemoryType.TOOL_LOG, thread_id)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.TOOL_LOG,
            limit=limit,
        )
        results.sort(key=lambda u: u.created_at)
        return results

    # =========================================================================
    # Skills Memory
    # =========================================================================

    async def add_skill(
        self,
        name: str,
        description: str,
        examples: list[str] | None = None,
        category: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Add a learned skill to memory.

        Args:
            name: Name of the skill.
            description: Description of what the skill does.
            examples: Optional list of example usages.
            category: Optional category for the skill.
            **kwargs: Additional skill attributes.

        Returns:
            The ID of the created skill memory.

        Example:
            ```python
            skill_id = await harness.add_skill(
                name="code_review",
                description="Review code for bugs, style issues, and improvements",
                examples=[
                    "Review this Python function for efficiency",
                    "Check this code for security vulnerabilities"
                ]
            )
            ```
        """
        namespace = self._build_namespace(MemoryType.SKILLS)

        content = f"Skill: {name}\n{description}"
        if category:
            content += f"\nCategory: {category}"
        if examples:
            content += "\nExamples:\n" + "\n".join(f"- {ex}" for ex in examples)

        embedding = await self._embed(content)

        meta = {
            "name": name,
            "description": description,
            "examples": examples or [],
            "category": category,
            **kwargs,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.SKILLS,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_skills(
        self,
        query: str,
        k: int = 3,
    ) -> list[MemoryUnit]:
        """
        Search for relevant skills.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of matching skill MemoryUnit objects.

        Example:
            ```python
            skills = await harness.search_skills("review code for bugs")
            for skill in skills:
                print(f"Skill: {skill.metadata['name']}")
            ```
        """
        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.SKILLS,
            k=k,
        )

    # =========================================================================
    # File Memory
    # =========================================================================

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
        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.FILE,
            k=k,
        )

    # =========================================================================
    # Persona Memory
    # =========================================================================

    async def add_persona(
        self,
        block_name: str,
        content: str,
    ) -> str:
        """
        Add or update a persona block.

        Args:
            block_name: Name of the persona block (e.g., "preferences", "background").
            content: The persona content.

        Returns:
            The ID of the created/updated persona block.

        Example:
            ```python
            await harness.add_persona(
                block_name="communication_style",
                content="Prefers concise, technical explanations with code examples"
            )
            ```
        """
        namespace = self._build_namespace(MemoryType.PERSONA, block_name)

        # Check if block already exists and delete it
        existing = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.PERSONA,
            limit=1,
        )
        for unit in existing:
            await self._backend.delete(unit.id)

        embedding = await self._embed(content)

        meta = {
            "block_name": block_name,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.PERSONA,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def get_persona(
        self,
        block_name: str | None = None,
    ) -> str:
        """
        Retrieve persona content.

        Args:
            block_name: Optional specific block name. If None, returns all blocks.

        Returns:
            The persona content as a string.

        Example:
            ```python
            # Get specific block
            style = await harness.get_persona("communication_style")

            # Get all persona blocks
            full_persona = await harness.get_persona()
            ```
        """
        if block_name:
            namespace = self._build_namespace(MemoryType.PERSONA, block_name)
            results = await self._backend.list_by_namespace(
                namespace=namespace,
                memory_type=MemoryType.PERSONA,
                limit=1,
            )
            return results[0].content if results else ""

        # Get all persona blocks
        namespace = self._build_namespace(MemoryType.PERSONA)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.PERSONA,
            limit=self._config.persona_max_blocks,
        )

        blocks = []
        for unit in results:
            block_name = unit.metadata.get("block_name", "unknown")
            blocks.append(f"## {block_name}\n{unit.content}")

        return "\n\n".join(blocks)

    async def set_persona(
        self,
        name: str,
        traits: list[str] | None = None,
        communication_style: str | None = None,
        domain_expertise: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Set the active persona for the agent.

        Args:
            name: Name of the persona.
            traits: List of personality traits.
            communication_style: Communication style description.
            domain_expertise: List of domain expertise areas.
            **kwargs: Additional persona attributes.

        Returns:
            The ID of the created persona.

        Example:
            ```python
            persona_id = await harness.set_persona(
                name="Technical Expert",
                traits=["concise", "technical", "helpful"],
                communication_style="professional",
                domain_expertise=["python", "devops"]
            )
            ```
        """
        # Construct persona content
        content_parts = [f"Persona: {name}"]

        if traits:
            content_parts.append(f"Traits: {', '.join(traits)}")

        if communication_style:
            content_parts.append(f"Communication Style: {communication_style}")

        if domain_expertise:
            content_parts.append(f"Domain Expertise: {', '.join(domain_expertise)}")

        for key, value in kwargs.items():
            content_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        content = "\n".join(content_parts)

        # Store as persona block
        return await self.add_persona(block_name=name, content=content)

    async def get_active_persona(self) -> MemoryUnit | None:
        """
        Get the active persona.

        Returns:
            The active persona as a MemoryUnit, or None if no persona is set.

        Example:
            ```python
            persona = await harness.get_active_persona()
            if persona:
                print(persona.content)
            ```
        """
        namespace = self._build_namespace(MemoryType.PERSONA)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.PERSONA,
            limit=1,
        )
        return results[0] if results else None

    # =========================================================================
    # Tool Log Memory
    # =========================================================================

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

    # =========================================================================
    # Multi-Type Operations
    # =========================================================================

    async def search_all(
        self,
        query: str,
        k: int = 5,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemoryUnit]:
        """
        Search across multiple memory types.

        Args:
            query: Search query.
            k: Total number of results to return.
            memory_types: Optional list of memory types to search. If None, searches all.

        Returns:
            List of memory units from all searched types, sorted by relevance.

        Example:
            ```python
            results = await harness.search_all("Python", k=10)
            for result in results:
                print(f"{result.memory_type}: {result.content[:50]}")
            ```
        """
        if memory_types is None:
            # Search all vector-based types
            memory_types = [
                MemoryType.KNOWLEDGE,
                MemoryType.ENTITY,
                MemoryType.WORKFLOW,
                MemoryType.TOOLBOX,
                MemoryType.SKILLS,
                MemoryType.FILE,
            ]

        all_results = []
        per_type_k = max(1, k // len(memory_types))

        for mem_type in memory_types:
            try:
                results = await self.search(
                    query=query,
                    memory_type=mem_type.value,
                    k=per_type_k,
                )
                all_results.extend(results)
            except Exception:
                # Skip types that fail
                continue

        # Sort by relevance if available, otherwise by recency
        all_results.sort(
            key=lambda x: (
                getattr(x, "score", 0) if hasattr(x, "score") else 0,
                x.created_at,
            ),
            reverse=True,
        )

        return all_results[:k]

    async def get_stats(self) -> dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary containing statistics about stored memories.

        Example:
            ```python
            stats = await harness.get_stats()
            print(f"Total conversations: {stats['conversational']}")
            print(f"Total knowledge: {stats['knowledge']}")
            ```
        """
        stats = {}

        for mem_type in MemoryType:
            try:
                namespace = self._build_namespace(mem_type)
                results = await self._backend.list_by_namespace(
                    namespace=namespace,
                    memory_type=mem_type,
                    limit=10000,  # Large limit to count all
                )
                stats[mem_type.value] = len(results)
            except Exception:
                stats[mem_type.value] = 0

        stats["total"] = sum(stats.values())
        return stats

    async def clear_thread(self, thread_id: str) -> int:
        """
        Clear all memories associated with a specific thread.

        Args:
            thread_id: The thread ID to clear.

        Returns:
            Number of memories deleted.

        Example:
            ```python
            deleted = await harness.clear_thread("thread_123")
            print(f"Deleted {deleted} memories")
            ```
        """
        # Get all conversational memories for the thread
        # Thread ID is part of the namespace
        namespace = self._build_namespace(MemoryType.CONVERSATIONAL, thread_id)
        memories = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.CONVERSATIONAL,
            limit=10000,
        )

        # Delete each memory
        for memory in memories:
            await self._backend.delete(memory.id)

        return len(memories)

    # =========================================================================
    # Generic Operations
    # =========================================================================

    async def add(
        self,
        content: str,
        memory_type: str | None = None,
        namespace: tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a generic memory unit.

        Args:
            content: The memory content.
            memory_type: Optional memory type string. Defaults to "knowledge".
            namespace: Optional custom namespace.
            metadata: Optional metadata.

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            mem_id = await harness.add(
                content="Important note about the project",
                memory_type="knowledge",
                metadata={"importance": "high"}
            )
            ```
        """
        mem_type = MemoryType(memory_type) if memory_type else MemoryType.KNOWLEDGE
        ns = namespace if namespace else self._build_namespace(mem_type)

        embedding = await self._embed(content)

        unit = self._create_unit(
            content=content,
            memory_type=mem_type,
            namespace=ns,
            metadata=metadata,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search(
        self,
        query: str,
        memory_type: str | None = None,
        k: int = 10,
    ) -> list[MemoryUnit]:
        """
        Search across memories.

        Args:
            query: The search query.
            memory_type: Optional memory type to filter by.
            k: Number of results to return.

        Returns:
            List of matching MemoryUnit objects.

        Example:
            ```python
            results = await harness.search("Python programming", k=5)
            ```
        """
        query_embedding = await self._embed(query)
        mem_type = MemoryType(memory_type) if memory_type else None

        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=mem_type,
            k=k,
        )

    async def get(self, memory_id: str) -> MemoryUnit | None:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The MemoryUnit if found, None otherwise.

        Example:
            ```python
            memory = await harness.get("mem-123")
            if memory:
                print(memory.content)
            ```
        """
        return await self._backend.get(memory_id)

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a memory unit.

        Args:
            memory_id: The memory ID to update.
            content: Optional new content.
            metadata: Optional metadata to merge.

        Returns:
            True if updated successfully, False if not found.

        Example:
            ```python
            success = await harness.update(
                "mem-123",
                content="Updated content",
                metadata={"updated": True}
            )
            ```
        """
        updates: dict[str, Any] = {}

        if content is not None:
            updates["content"] = content
            updates["embedding"] = await self._embed(content)

        if metadata is not None:
            updates["metadata"] = metadata

        if not updates:
            return True  # Nothing to update

        return await self._backend.update(memory_id, updates)

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory unit.

        Args:
            memory_id: The memory ID to delete.

        Returns:
            True if deleted successfully, False if not found.

        Example:
            ```python
            deleted = await harness.delete("mem-123")
            ```
        """
        return await self._backend.delete(memory_id)

    async def clear_all(self) -> int:
        """
        Clear all memories from the backend.

        This method removes all stored memories. Use with caution!

        Returns:
            The number of memories deleted.

        Example:
            ```python
            count = await harness.clear_all()
            print(f"Deleted {count} memories")
            ```
        """
        # For in-memory backend, we can directly clear storage
        if hasattr(self._backend, "_storage"):
            count = len(self._backend._storage)
            self._backend._storage.clear()
            return count

        # For other backends, we need to list and delete all memories
        # This is inefficient but works for all backend types
        count = 0
        for memory_type in MemoryType:
            namespace = (
                self._namespace_prefix + (memory_type.value,)
                if self._namespace_prefix
                else (memory_type.value,)
            )
            memories = await self._backend.list_by_namespace(
                namespace=namespace,
                memory_type=memory_type,
                limit=100000,  # Large limit to get all
            )
            for memory in memories:
                await self._backend.delete(memory.id)
                count += 1

        return count

    # =========================================================================
    # Context Assembly
    # =========================================================================

    async def assemble_context(
        self,
        query: str,
        thread_id: str,
        max_tokens: int = 4000,
    ) -> str:
        """
        Assemble relevant context for an agent query.

        This method gathers relevant memories from multiple sources:
        - Recent conversation history
        - Relevant knowledge base entries
        - Matching entities
        - Applicable workflows
        - Persona information

        Args:
            query: The query to assemble context for.
            thread_id: The conversation thread ID.
            max_tokens: Maximum tokens in the assembled context.

        Returns:
            A formatted context string ready to be used in a prompt.

        Example:
            ```python
            context = await harness.assemble_context(
                query="How do I deploy the application?",
                thread_id="chat-123",
                max_tokens=4000
            )
            # Use context in your prompt
            prompt = f"{context}\\n\\nUser: {query}"
            ```
        """
        sections = []
        estimated_tokens = 0
        chars_per_token = 4  # Rough estimate

        # 1. Persona (always include, typically small)
        persona = await self.get_persona()
        if persona:
            sections.append(f"## Persona\n{persona}")
            estimated_tokens += len(persona) // chars_per_token

        # 2. Recent conversation history
        if estimated_tokens < max_tokens:
            messages = await self.get_conversational(thread_id, limit=10)
            if messages:
                conv_text = "\n".join(
                    f"{m.metadata.get('role', 'unknown')}: {m.content}"
                    for m in messages[-5:]  # Last 5 messages
                )
                sections.append(f"## Recent Conversation\n{conv_text}")
                estimated_tokens += len(conv_text) // chars_per_token

        # 3. Relevant knowledge
        if estimated_tokens < max_tokens:
            knowledge = await self.search_knowledge(query, k=3)
            if knowledge:
                kb_text = "\n\n".join(f"- {k.content}" for k in knowledge)
                sections.append(f"## Relevant Knowledge\n{kb_text}")
                estimated_tokens += len(kb_text) // chars_per_token

        # 4. Relevant entities
        if estimated_tokens < max_tokens:
            entities = await self.search_entity(query, k=3)
            if entities:
                ent_text = "\n".join(
                    f"- {e.metadata.get('name', 'Unknown')}: {e.content}" for e in entities
                )
                sections.append(f"## Related Entities\n{ent_text}")
                estimated_tokens += len(ent_text) // chars_per_token

        # 5. Relevant workflows
        if estimated_tokens < max_tokens:
            workflows = await self.search_workflow(query, k=2)
            if workflows:
                wf_text = "\n\n".join(
                    f"**{w.metadata.get('task', 'Task')}**\n{w.content}" for w in workflows
                )
                sections.append(f"## Relevant Workflows\n{wf_text}")
                estimated_tokens += len(wf_text) // chars_per_token

        # 6. Relevant skills
        if estimated_tokens < max_tokens:
            skills = await self.search_skills(query, k=2)
            if skills:
                skills_text = "\n".join(
                    f"- {s.metadata.get('name', 'Skill')}: {s.metadata.get('description', '')}"
                    for s in skills
                )
                sections.append(f"## Available Skills\n{skills_text}")

        return "\n\n".join(sections)

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
                            "query": {"type": "string", "description": "The search query"},
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
                            "content": {"type": "string", "description": "The content to remember"},
                            "memory_type": {
                                "type": "string",
                                "enum": [t.value for t in MemoryType],
                                "description": "Type of memory (default: knowledge)",
                            },
                            "metadata": {"type": "object", "description": "Optional metadata"},
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
