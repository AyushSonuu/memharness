# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Core data types for the memharness memory system.

This module defines the fundamental types used throughout memharness:
- MemoryType: Enum of the 10 supported memory categories
- MemoryUnit: The universal container for all memory entries (dataclass-based)
- StorageType: Backend storage strategies (SQL, VECTOR, HYBRID)
- TriggerType: Event triggers for memory agents
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Enumeration of all supported memory types."""

    CONVERSATIONAL = "conversational"
    KNOWLEDGE = "knowledge"
    ENTITY = "entity"
    WORKFLOW = "workflow"
    TOOLBOX = "toolbox"
    SUMMARY = "summary"
    TOOL_LOG = "tool_log"
    FILE = "file"
    PERSONA = "persona"


class StorageType(str, Enum):
    """
    Backend storage strategies for memory types.

    - SQL: Relational storage with exact-match queries (for ordered/log data)
    - VECTOR: Embedding-based semantic search (for knowledge retrieval)
    - HYBRID: Combined SQL + vector capabilities (for complex queries)
    """

    SQL = "sql"
    VECTOR = "vector"
    HYBRID = "hybrid"


class TriggerType(str, Enum):
    """
    Event triggers for memory agents.

    Agents can be invoked at different points in the memory lifecycle:

    - ON_WRITE: Triggered when new memory is added
    - ON_READ: Triggered when memory is retrieved
    - PRE_LLM: Before sending context to the LLM
    - POST_LLM: After receiving LLM response
    - SCHEDULED: Time-based periodic execution
    - POLICY: Triggered by policy rules (e.g., TTL expiration)
    - ON_DEMAND: Manual invocation by user or system
    """

    ON_WRITE = "on_write"
    ON_READ = "on_read"
    PRE_LLM = "pre_llm"
    POST_LLM = "post_llm"
    SCHEDULED = "scheduled"
    POLICY = "policy"
    ON_DEMAND = "on_demand"


@dataclass
class MemoryUnit:
    """
    A single unit of memory stored in the harness.

    This is the fundamental data structure representing any piece of memory,
    regardless of its type. Each MemoryUnit has a unique ID, content, type,
    and associated metadata.

    Attributes:
        content: The actual content/text of the memory.
        memory_type: The type of memory (e.g., conversational, knowledge).
        id: Unique identifier for this memory unit.
        namespace: Hierarchical namespace tuple for organization.
        embedding: Optional vector embedding for similarity search.
        metadata: Additional key-value metadata.
        created_at: Timestamp when the memory was created.
        updated_at: Timestamp when the memory was last updated.
        thread_id: Optional thread ID for conversational memories.
        parent_id: Optional parent memory ID for hierarchical relationships.
    """

    content: str
    memory_type: MemoryType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: tuple[str, ...] = field(default_factory=tuple)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    thread_id: str | None = None
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the MemoryUnit to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "namespace": list(self.namespace),
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "thread_id": self.thread_id,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryUnit:
        """Create a MemoryUnit from a dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            namespace=tuple(data.get("namespace", [])),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", datetime.now(UTC)),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at", datetime.now(UTC)),
            thread_id=data.get("thread_id"),
            parent_id=data.get("parent_id"),
        )

    def to_json(self) -> str:
        """Convert the MemoryUnit to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> MemoryUnit:
        """Create a MemoryUnit from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
