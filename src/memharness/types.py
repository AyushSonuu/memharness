# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Core data types for the memharness memory system.

This module defines the fundamental types used throughout memharness:
- MemoryType: Enum of the 10 supported memory categories
- MemoryUnit: The universal container for all memory entries
- StorageType: Backend storage strategies (SQL, VECTOR, HYBRID)
- TriggerType: Event triggers for memory agents
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """
    Enumeration of all memory types supported by memharness.

    Each memory type serves a distinct purpose in the agent's memory architecture:

    - CONVERSATIONAL: Chat history and dialogue context (ordered by time)
    - KNOWLEDGE: Factual information and documentation (semantic search)
    - ENTITY: Named entities with attributes and relationships
    - WORKFLOW: Procedural knowledge and multi-step processes
    - TOOLBOX: Tool definitions, schemas, and usage examples
    - SUMMARY: Compressed representations of other memories
    - TOOL_LOG: Execution records of tool calls (append-only)
    - SKILLS: Learned capabilities and reusable patterns
    - FILE: File content and metadata references
    - PERSONA: Agent identity, style, and behavioral guidelines
    """

    CONVERSATIONAL = "conversational"
    KNOWLEDGE = "knowledge"
    ENTITY = "entity"
    WORKFLOW = "workflow"
    TOOLBOX = "toolbox"
    SUMMARY = "summary"
    TOOL_LOG = "tool_log"
    SKILLS = "skills"
    FILE = "file"
    PERSONA = "persona"

    @property
    def uses_vector(self) -> bool:
        """Check if this memory type uses vector similarity search."""
        return self in {
            MemoryType.KNOWLEDGE,
            MemoryType.ENTITY,
            MemoryType.WORKFLOW,
            MemoryType.TOOLBOX,
            MemoryType.SUMMARY,
            MemoryType.SKILLS,
            MemoryType.FILE,
            MemoryType.PERSONA,
        }

    @property
    def table_name(self) -> str:
        """Get the database table name for this memory type."""
        return f"{self.value}_memory"


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


class MemoryUnit(BaseModel):
    """
    Universal container for all memory entries in memharness.

    MemoryUnit is the fundamental data structure that holds any type of memory.
    It provides a consistent interface regardless of the underlying memory type,
    enabling uniform operations across the entire memory system.

    Attributes:
        id: Unique identifier for this memory unit.
        namespace: Logical grouping (e.g., user_id, session_id, project_id).
        memory_type: The category of memory this unit represents.
        content: The actual memory content (text, structured data, etc.).
        embedding: Optional vector embedding for semantic search.
        metadata: Arbitrary key-value pairs for filtering and context.
        created_at: Timestamp when this memory was first created.
        updated_at: Timestamp of the last modification.
        expires_at: Optional TTL - memory will be garbage collected after this.
        score: Relevance score from retrieval (set during queries).
        source_ids: For summaries - IDs of memories this was derived from.

    Example:
        >>> unit = MemoryUnit(
        ...     namespace="user_123",
        ...     memory_type=MemoryType.CONVERSATIONAL,
        ...     content="Hello, how can I help you today?",
        ...     metadata={"role": "assistant", "thread_id": "t_abc"}
        ... )
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    namespace: str = Field(..., description="Logical grouping (user/session/project)")
    memory_type: MemoryType = Field(..., description="Category of this memory")
    content: str = Field(..., description="The memory content (text or serialized)")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding for semantic search"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary key-value metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    expires_at: datetime | None = Field(
        default=None, description="Optional TTL expiration"
    )
    score: float | None = Field(
        default=None, description="Relevance score from retrieval"
    )
    source_ids: list[UUID] | None = Field(
        default=None, description="Source memory IDs (for summaries)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "namespace": "user_123",
                "memory_type": "conversational",
                "content": "What is the capital of France?",
                "metadata": {"role": "user", "thread_id": "thread_abc"},
                "created_at": "2026-03-23T10:30:00Z",
                "updated_at": "2026-03-23T10:30:00Z",
            }
        }
    }

    def with_score(self, score: float) -> MemoryUnit:
        """Return a copy of this unit with the given relevance score."""
        return self.model_copy(update={"score": score})

    def touch(self) -> MemoryUnit:
        """Return a copy with updated_at set to now."""
        return self.model_copy(update={"updated_at": datetime.utcnow()})

    def is_expired(self) -> bool:
        """Check if this memory has passed its expiration time."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return self.model_dump(mode="json", exclude_none=True)

    def __repr__(self) -> str:
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return (
            f"MemoryUnit(id={str(self.id)[:8]}, type={self.memory_type.value}, "
            f"ns={self.namespace!r}, content={content_preview!r})"
        )


class SearchResult(BaseModel):
    """Result from a memory search operation."""

    unit: MemoryUnit
    score: float = 0.0
    distance: float | None = None

    @property
    def similarity(self) -> float:
        """Return similarity score (1 - distance for cosine distance)."""
        if self.distance is not None:
            return 1 - self.distance
        return self.score


class SearchFilter(BaseModel):
    """Filter criteria for memory searches."""

    namespace: str | None = None
    thread_id: str | None = None
    source: str | None = None
    entity_type: str | None = None
    tool_name: str | None = None
    metadata_filters: dict[str, Any] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
