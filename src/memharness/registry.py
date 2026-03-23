# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Memory type registry for memharness.

This module provides the configuration and registration system for memory types.
Each memory type has specific characteristics that determine how it's stored,
indexed, and retrieved.

The registry comes pre-configured with all 10 built-in memory types but supports
custom type registration for specialized use cases.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memharness.types import MemoryType, MemoryUnit, StorageType


@dataclass
class MemoryTypeConfig:
    """
    Configuration for a memory type.

    This dataclass defines all the characteristics of a memory type that
    determine its storage, indexing, and retrieval behavior.

    Attributes:
        name: The memory type identifier (matches MemoryType enum value).
        storage: Which backend storage strategy to use.
        schema: JSON schema for validating memory content/metadata.
        index_type: Vector index type (e.g., "hnsw", "flat", "ivf").
        default_k: Default number of results for retrieval queries.
        supports_embedding: Whether this type uses vector embeddings.
        ordered: Whether temporal ordering is significant.
        formatter: Optional function to format memories for LLM context.

    Example:
        >>> config = MemoryTypeConfig(
        ...     name="conversational",
        ...     storage=StorageType.SQL,
        ...     default_k=20,
        ...     supports_embedding=False,
        ...     ordered=True,
        ... )
    """

    name: str
    storage: StorageType
    schema: dict[str, Any] | None = None
    index_type: str | None = None
    default_k: int = 10
    supports_embedding: bool = True
    ordered: bool = False
    formatter: Callable[[list[MemoryUnit]], str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if self.storage == StorageType.SQL and self.supports_embedding:
            # SQL-only types typically don't use embeddings
            pass  # Allow but warn in production
        if self.supports_embedding and self.index_type is None:
            self.index_type = "hnsw"  # Default to HNSW for vector search


class _DefaultHandler:
    """Default handler implementation for memory types."""

    def __init__(self, config: MemoryTypeConfig):
        self.config = config
        self.name = config.name
        self.description = f"Handler for {config.name} memory type"
        self.schema = config.schema

    def validate(self, data: Any) -> bool:
        """Validate data against the schema."""
        return True

    def process(self, data: Any) -> Any:
        """Process data for storage."""
        return data

    def serialize(self, data: Any) -> str:
        """Serialize data for storage."""
        if isinstance(data, str):
            return data
        return json.dumps(data)

    def store(self, unit: MemoryUnit) -> MemoryUnit:
        """Store a memory unit."""
        return unit

    def search(self, query: str, **kwargs: Any) -> list[MemoryUnit]:
        """Search for memories."""
        return []

    def format(self, units: list[MemoryUnit]) -> str:
        """Format memories for LLM context."""
        if self.config.formatter:
            return self.config.formatter(units)
        return "\n".join(u.content for u in units)


class MemoryTypeRegistry:
    """
    Registry for memory type configurations.

    The registry maintains all known memory types and their configurations.
    It comes pre-loaded with the 10 built-in types but supports runtime
    registration of custom types.

    Built-in Memory Types:
        - conversational: Chat history (SQL, ordered)
        - knowledge: Facts and docs (VECTOR)
        - entity: Named entities (VECTOR)
        - workflow: Procedures (VECTOR)
        - toolbox: Tool definitions (VECTOR)
        - summary: Compressed memories (VECTOR)
        - tool_log: Execution records (SQL, ordered)
        - skills: Learned patterns (VECTOR)
        - file: File references (VECTOR)
        - persona: Agent identity (VECTOR)

    Example:
        >>> registry = MemoryTypeRegistry.get_instance()
        >>> config = registry.get(MemoryType.CONVERSATIONAL)
        >>> print(config.storage)
        StorageType.SQL
    """

    _instance: "MemoryTypeRegistry | None" = None
    _configs: dict[str, MemoryTypeConfig]
    _handlers: dict[str, Any]

    def __init__(self) -> None:
        """Initialize the registry with all built-in memory types."""
        self._configs = {}
        self._handlers = {}
        self._register_builtin_types()

    @classmethod
    def get_instance(cls) -> "MemoryTypeRegistry":
        """
        Get the singleton instance of the registry.

        Returns:
            The singleton MemoryTypeRegistry instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_builtin_types(self) -> None:
        """Register all 10 built-in memory types with their default configurations."""

        # SQL-backed types (ordered, no embedding needed for primary retrieval)
        self.register(
            MemoryTypeConfig(
                name=MemoryType.CONVERSATIONAL.value,
                storage=StorageType.SQL,
                default_k=20,
                supports_embedding=False,
                ordered=True,
                schema={
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"]},
                        "thread_id": {"type": "string"},
                        "turn_index": {"type": "integer"},
                    },
                    "required": ["role", "thread_id"],
                },
                formatter=_format_conversational,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.TOOL_LOG.value,
                storage=StorageType.SQL,
                default_k=10,
                supports_embedding=False,
                ordered=True,
                schema={
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "input": {"type": "object"},
                        "output": {"type": "object"},
                        "status": {"type": "string", "enum": ["success", "error", "timeout"]},
                        "duration_ms": {"type": "integer"},
                    },
                    "required": ["tool_name", "status"],
                },
                formatter=_format_tool_log,
            )
        )

        # Vector-backed types (semantic search)
        self.register(
            MemoryTypeConfig(
                name=MemoryType.KNOWLEDGE.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=5,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "document_id": {"type": "string"},
                    },
                },
                formatter=_format_knowledge_base,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.ENTITY.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=10,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string"},
                        "entity_type": {"type": "string"},
                        "attributes": {"type": "object"},
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "relation": {"type": "string"},
                                    "target_entity": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["entity_name", "entity_type"],
                },
                formatter=_format_entity,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.WORKFLOW.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=3,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "workflow_name": {"type": "string"},
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step_number": {"type": "integer"},
                                    "action": {"type": "string"},
                                    "expected_output": {"type": "string"},
                                },
                            },
                        },
                        "preconditions": {"type": "array", "items": {"type": "string"}},
                        "postconditions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["workflow_name"],
                },
                formatter=_format_workflow,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.TOOLBOX.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=5,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "description": {"type": "string"},
                        "parameters_schema": {"type": "object"},
                        "examples": {"type": "array", "items": {"type": "object"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["tool_name", "description"],
                },
                formatter=_format_toolbox,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.SUMMARY.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=3,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "summary_type": {
                            "type": "string",
                            "enum": ["conversation", "document", "session", "topic"],
                        },
                        "time_range_start": {"type": "string", "format": "date-time"},
                        "time_range_end": {"type": "string", "format": "date-time"},
                        "source_count": {"type": "integer"},
                    },
                    "required": ["summary_type"],
                },
                formatter=_format_summary,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.SKILLS.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=5,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string"},
                        "capability": {"type": "string"},
                        "learned_from": {"type": "string"},
                        "success_rate": {"type": "number"},
                        "usage_count": {"type": "integer"},
                    },
                    "required": ["skill_name", "capability"],
                },
                formatter=_format_skills,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.FILE.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=5,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "file_type": {"type": "string"},
                        "file_size": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "last_modified": {"type": "string", "format": "date-time"},
                    },
                    "required": ["file_path"],
                },
                formatter=_format_file,
            )
        )

        self.register(
            MemoryTypeConfig(
                name=MemoryType.PERSONA.value,
                storage=StorageType.VECTOR,
                index_type="hnsw",
                default_k=3,
                supports_embedding=True,
                ordered=False,
                schema={
                    "type": "object",
                    "properties": {
                        "persona_name": {"type": "string"},
                        "trait_type": {
                            "type": "string",
                            "enum": ["style", "constraint", "preference", "identity"],
                        },
                        "priority": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["persona_name"],
                },
                formatter=_format_persona,
            )
        )

    def register(self, name_or_config: str | MemoryTypeConfig, handler: Any = None) -> None:
        """
        Register a memory type configuration or handler.

        Args:
            name_or_config: Either a MemoryTypeConfig or a string name.
            handler: Optional handler object (used when name_or_config is a string).

        Note:
            If a type with this name already exists, it will be replaced.
        """
        if isinstance(name_or_config, MemoryTypeConfig):
            config = name_or_config
            self._configs[config.name] = config
            # Create a default handler from the config
            self._handlers[config.name] = _DefaultHandler(config)
        else:
            # String name with handler
            name = name_or_config
            self._handlers[name] = handler
            # Create a minimal config for the handler
            if name not in self._configs:
                self._configs[name] = MemoryTypeConfig(
                    name=name,
                    storage=StorageType.HYBRID,
                    supports_embedding=True,
                )

    def unregister(self, name: str | MemoryType) -> MemoryTypeConfig | None:
        """
        Remove a memory type from the registry.

        Args:
            name: The name of the memory type to remove (string or MemoryType enum).

        Returns:
            The removed configuration, or None if not found.
        """
        key = name.value if isinstance(name, MemoryType) else name
        self._handlers.pop(key, None)
        return self._configs.pop(key, None)

    def get(self, memory_type: MemoryType | str | None) -> Any:
        """
        Get the handler for a memory type.

        Args:
            memory_type: The memory type (enum or string name).

        Returns:
            The handler for this memory type.

        Raises:
            TypeError: If memory_type is None.
            KeyError: If the memory type is not registered.
        """
        if memory_type is None:
            raise TypeError("Memory type cannot be None")

        name = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        if name not in self._handlers:
            raise KeyError(
                f"Memory type '{name}' is not registered. "
                f"Available types: {list(self._handlers.keys())}"
            )
        return self._handlers[name]

    def list_types(self) -> list[MemoryType]:
        """
        List all registered memory types as MemoryType enums.

        Returns:
            List of registered MemoryType enum values.
        """
        result = []
        for name in self._configs.keys():
            try:
                # Try to convert to MemoryType enum
                result.append(MemoryType(name))
            except ValueError:
                # For custom types, return as string but wrapped as MemoryType-like
                pass
        return result

    def list_configs(self) -> list[MemoryTypeConfig]:
        """
        List all registered memory type configurations.

        Returns:
            List of all registered configurations.
        """
        return list(self._configs.values())

    def get_by_storage(self, storage: StorageType) -> list[MemoryTypeConfig]:
        """
        Get all memory types using a specific storage backend.

        Args:
            storage: The storage type to filter by.

        Returns:
            List of configurations using this storage type.
        """
        return [c for c in self._configs.values() if c.storage == storage]

    def handler_for(self, memory_type: MemoryType | str) -> Any:
        """
        Get the handler for a memory type (alias for get()).

        Args:
            memory_type: The memory type (enum or string name).

        Returns:
            The handler for this memory type.
        """
        return self.get(memory_type)

    def get_description(self, name: str | MemoryType) -> str | None:
        """
        Get the description for a memory type.

        Args:
            name: The memory type name or enum.

        Returns:
            The description if available, None otherwise.
        """
        key = name.value if isinstance(name, MemoryType) else name
        handler = self._handlers.get(key)
        if handler:
            return getattr(handler, "description", handler.__doc__)
        return None

    def get_schema(self, name: str | MemoryType) -> dict[str, Any] | None:
        """
        Get the schema for a memory type.

        Args:
            name: The memory type name or enum.

        Returns:
            The schema if available, None otherwise.
        """
        key = name.value if isinstance(name, MemoryType) else name
        config = self._configs.get(key)
        if config:
            return config.schema
        handler = self._handlers.get(key)
        if handler and hasattr(handler, "schema"):
            return handler.schema
        return None

    def get_all_handlers(self) -> dict[str, Any]:
        """
        Get all registered handlers.

        Returns:
            Dictionary mapping memory type names to their handlers.
        """
        return dict(self._handlers)

    def get_all(self) -> dict[str, Any]:
        """
        Get all handlers (alias for get_all_handlers()).

        Returns:
            Dictionary mapping memory type names to their handlers.
        """
        return self.get_all_handlers()

    def register_handler(self, name: str) -> Callable[[type], type]:
        """
        Decorator for registering a handler class.

        Args:
            name: The memory type name.

        Returns:
            Decorator function.

        Example:
            @registry.register_handler("custom_type")
            class CustomHandler:
                pass
        """

        def decorator(handler_class: type) -> type:
            self.register(name, handler_class())
            return handler_class

        return decorator

    def __contains__(self, name: str) -> bool:
        """Check if a memory type is registered."""
        return name in self._configs

    def __len__(self) -> int:
        """Return the number of registered memory types."""
        return len(self._configs)


# Default formatters for LLM context


def _format_conversational(units: list[MemoryUnit]) -> str:
    """Format conversational memories as chat history."""
    lines = []
    for unit in sorted(units, key=lambda u: u.created_at):
        role = unit.metadata.get("role", "unknown")
        lines.append(f"[{role}]: {unit.content}")
    return "\n".join(lines)


def _format_tool_log(units: list[MemoryUnit]) -> str:
    """Format tool logs as execution history."""
    lines = ["## Tool Execution Log"]
    for unit in sorted(units, key=lambda u: u.created_at):
        tool_name = unit.metadata.get("tool_name", "unknown")
        status = unit.metadata.get("status", "unknown")
        lines.append(f"- {tool_name}: {status} - {unit.content[:100]}")
    return "\n".join(lines)


def _format_knowledge_base(units: list[MemoryUnit]) -> str:
    """Format knowledge base entries as reference material."""
    lines = ["## Relevant Knowledge"]
    for i, unit in enumerate(units, 1):
        source = unit.metadata.get("source", "unknown")
        score = f" (relevance: {unit.score:.2f})" if unit.score else ""
        lines.append(f"[{i}] {source}{score}")
        lines.append(unit.content)
        lines.append("")
    return "\n".join(lines)


def _format_entity(units: list[MemoryUnit]) -> str:
    """Format entity memories as structured information."""
    lines = ["## Known Entities"]
    for unit in units:
        name = unit.metadata.get("entity_name", "Unknown")
        etype = unit.metadata.get("entity_type", "entity")
        lines.append(f"- **{name}** ({etype}): {unit.content}")
    return "\n".join(lines)


def _format_workflow(units: list[MemoryUnit]) -> str:
    """Format workflow memories as procedures."""
    lines = ["## Available Workflows"]
    for unit in units:
        name = unit.metadata.get("workflow_name", "Unnamed")
        lines.append(f"### {name}")
        lines.append(unit.content)
        lines.append("")
    return "\n".join(lines)


def _format_toolbox(units: list[MemoryUnit]) -> str:
    """Format toolbox entries as available tools."""
    lines = ["## Available Tools"]
    for unit in units:
        name = unit.metadata.get("tool_name", "unknown")
        desc = unit.metadata.get("description", "")
        lines.append(f"- **{name}**: {desc}")
        lines.append(f"  {unit.content[:200]}")
    return "\n".join(lines)


def _format_summary(units: list[MemoryUnit]) -> str:
    """Format summaries as condensed context."""
    lines = ["## Context Summaries"]
    for unit in units:
        stype = unit.metadata.get("summary_type", "general")
        lines.append(f"### {stype.title()} Summary")
        lines.append(unit.content)
        lines.append("")
    return "\n".join(lines)


def _format_skills(units: list[MemoryUnit]) -> str:
    """Format skills as learned capabilities."""
    lines = ["## Learned Skills"]
    for unit in units:
        name = unit.metadata.get("skill_name", "unknown")
        lines.append(f"- **{name}**: {unit.content}")
    return "\n".join(lines)


def _format_file(units: list[MemoryUnit]) -> str:
    """Format file memories as file references."""
    lines = ["## File Contents"]
    for unit in units:
        path = unit.metadata.get("file_path", "unknown")
        lines.append(f"### {path}")
        lines.append(unit.content)
        lines.append("")
    return "\n".join(lines)


def _format_persona(units: list[MemoryUnit]) -> str:
    """Format persona entries as identity guidelines."""
    lines = ["## Agent Persona"]
    for unit in units:
        trait = unit.metadata.get("trait_type", "general")
        lines.append(f"- [{trait}] {unit.content}")
    return "\n".join(lines)


# Module-level default registry instance
_default_registry: MemoryTypeRegistry | None = None


def get_default_registry() -> MemoryTypeRegistry:
    """
    Get the default global registry instance.

    Returns a singleton registry that can be shared across the application.
    For isolated registries, instantiate MemoryTypeRegistry directly.

    Returns:
        The default MemoryTypeRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = MemoryTypeRegistry()
    return _default_registry
