"""Pydantic configuration models for memharness.

This module defines all configuration models used throughout the memharness package.
All models use Pydantic v2 for validation, serialization, and documentation.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationalConfig(BaseModel):
    """Configuration for conversational memory type.

    Controls thread limits, TTL behavior, and auto-summarization thresholds
    for conversational memories.
    """

    model_config = ConfigDict(frozen=True)

    max_messages_per_thread: Annotated[
        int, Field(ge=1, description="Maximum messages allowed per conversation thread")
    ] = 1000

    default_ttl: Annotated[
        str | None,
        Field(
            description="Default time-to-live for messages (e.g., '7d', '24h'). None means no expiry"
        ),
    ] = None

    auto_summarize_threshold: Annotated[
        int,
        Field(
            ge=1, description="Number of messages after which auto-summarization is triggered"
        ),
    ] = 50


class SummarizationTrigger(BaseModel):
    """A trigger condition for summarization.

    Defines when summarization should occur based on memory conditions.
    """

    model_config = ConfigDict(frozen=True)

    condition: Annotated[
        str, Field(description="Trigger condition expression (e.g., 'age > 7d', 'count > 100')")
    ]

    memory_type: Annotated[
        str, Field(description="Memory type this trigger applies to")
    ]


class SummarizationConfig(BaseModel):
    """Configuration for the summarization agent.

    Controls when and how memories are summarized, and what happens
    to original memories after summarization.
    """

    model_config = ConfigDict(frozen=True)

    enabled: Annotated[bool, Field(description="Whether summarization is enabled")] = True

    triggers: Annotated[
        list[SummarizationTrigger],
        Field(
            default_factory=list,
            description="List of trigger conditions for summarization",
        ),
    ] = []

    keep_originals: Annotated[
        bool, Field(description="Whether to keep original memories after summarization")
    ] = True

    originals_ttl: Annotated[
        str,
        Field(description="TTL for original memories after summarization (e.g., '365d')"),
    ] = "365d"


class ConsolidationConfig(BaseModel):
    """Configuration for the consolidation agent.

    Controls duplicate detection and merging of similar memories.
    """

    model_config = ConfigDict(frozen=True)

    enabled: Annotated[bool, Field(description="Whether consolidation is enabled")] = True

    schedule: Annotated[
        str, Field(description="Cron expression for consolidation schedule")
    ] = "0 3 * * *"

    similarity_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Similarity threshold (0.0-1.0) for considering memories as duplicates",
        ),
    ] = 0.9


class GCConfig(BaseModel):
    """Configuration for the garbage collection agent.

    Controls archival and deletion of old or stale memories.
    """

    model_config = ConfigDict(frozen=True)

    enabled: Annotated[bool, Field(description="Whether garbage collection is enabled")] = True

    schedule: Annotated[
        str, Field(description="Cron expression for GC schedule")
    ] = "0 4 * * 0"

    archive_after: Annotated[
        str,
        Field(description="Duration after which memories are archived (e.g., '90d')"),
    ] = "90d"

    delete_after: Annotated[
        str,
        Field(description="Duration after which archived memories are deleted (e.g., '365d')"),
    ] = "365d"


class EntityExtractionConfig(BaseModel):
    """Configuration for the entity extraction agent.

    Controls how entities are extracted from memories.
    """

    model_config = ConfigDict(frozen=True)

    enabled: Annotated[bool, Field(description="Whether entity extraction is enabled")] = True

    mode: Annotated[
        Literal["on_write", "batch", "disabled"],
        Field(description="Extraction mode: on_write (real-time), batch (scheduled), or disabled"),
    ] = "on_write"

    types: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["PERSON", "ORG", "PLACE", "CONCEPT"],
            description="Entity types to extract",
        ),
    ] = ["PERSON", "ORG", "PLACE", "CONCEPT"]


class ContextAssemblyConfig(BaseModel):
    """Configuration for context assembly.

    Controls how memories are assembled into context for agent consumption.
    """

    model_config = ConfigDict(frozen=True)

    default_max_tokens: Annotated[
        int, Field(ge=1, description="Default maximum tokens for assembled context")
    ] = 4000

    priorities: Annotated[
        dict[str, float],
        Field(
            default_factory=dict,
            description="Memory type to priority percentage mapping",
        ),
    ] = {}


class ToolDiscoveryConfig(BaseModel):
    """Configuration for tool discovery.

    Controls the self-exploration capabilities of agents.
    """

    model_config = ConfigDict(frozen=True)

    enabled: Annotated[bool, Field(description="Whether tool discovery is enabled")] = True

    max_iterations: Annotated[
        int, Field(ge=1, description="Maximum iterations for tool discovery exploration")
    ] = 10


class AgentConfig(BaseModel):
    """Configuration for all embedded AI agents.

    Aggregates configuration for all agent subsystems.
    """

    model_config = ConfigDict(frozen=True)

    summarizer: Annotated[
        SummarizationConfig,
        Field(default_factory=SummarizationConfig, description="Summarization agent config"),
    ] = SummarizationConfig()

    entity_extractor: Annotated[
        EntityExtractionConfig,
        Field(default_factory=EntityExtractionConfig, description="Entity extraction agent config"),
    ] = EntityExtractionConfig()

    consolidator: Annotated[
        ConsolidationConfig,
        Field(default_factory=ConsolidationConfig, description="Consolidation agent config"),
    ] = ConsolidationConfig()

    gc: Annotated[
        GCConfig, Field(default_factory=GCConfig, description="Garbage collection agent config")
    ] = GCConfig()

    context_assembly: Annotated[
        ContextAssemblyConfig,
        Field(default_factory=ContextAssemblyConfig, description="Context assembly config"),
    ] = ContextAssemblyConfig()

    tool_discovery: Annotated[
        ToolDiscoveryConfig,
        Field(default_factory=ToolDiscoveryConfig, description="Tool discovery config"),
    ] = ToolDiscoveryConfig()


class KnowledgeBaseConfig(BaseModel):
    """Configuration for knowledge base memory type."""

    model_config = ConfigDict(frozen=True)

    default_collection: Annotated[
        str, Field(description="Default collection name for KB memories")
    ] = "default"

    embedding_model: Annotated[
        str | None, Field(description="Embedding model to use for vectorization")
    ] = None


class EntityMemoryConfig(BaseModel):
    """Configuration for entity memory type."""

    model_config = ConfigDict(frozen=True)

    auto_link: Annotated[
        bool, Field(description="Automatically link related entities")
    ] = True

    max_relations_per_entity: Annotated[
        int, Field(ge=1, description="Maximum relations per entity")
    ] = 100


class WorkflowConfig(BaseModel):
    """Configuration for workflow memory type."""

    model_config = ConfigDict(frozen=True)

    max_steps_per_workflow: Annotated[
        int, Field(ge=1, description="Maximum steps per workflow")
    ] = 1000

    auto_archive_completed: Annotated[
        bool, Field(description="Auto-archive completed workflows")
    ] = True


class ToolboxConfig(BaseModel):
    """Configuration for toolbox memory type."""

    model_config = ConfigDict(frozen=True)

    cache_schemas: Annotated[
        bool, Field(description="Cache tool schemas for faster access")
    ] = True


class ToolLogConfig(BaseModel):
    """Configuration for tool log memory type."""

    model_config = ConfigDict(frozen=True)

    max_logs_per_tool: Annotated[
        int, Field(ge=1, description="Maximum log entries per tool")
    ] = 10000

    retention_days: Annotated[
        int, Field(ge=1, description="Days to retain tool logs")
    ] = 30


class SkillsConfig(BaseModel):
    """Configuration for skills memory type."""

    model_config = ConfigDict(frozen=True)

    auto_version: Annotated[
        bool, Field(description="Automatically version skill updates")
    ] = True


class FileMemoryConfig(BaseModel):
    """Configuration for file memory type."""

    model_config = ConfigDict(frozen=True)

    max_file_size_mb: Annotated[
        int, Field(ge=1, description="Maximum file size in MB")
    ] = 100

    allowed_extensions: Annotated[
        list[str] | None, Field(description="Allowed file extensions. None allows all")
    ] = None


class PersonaConfig(BaseModel):
    """Configuration for persona memory type."""

    model_config = ConfigDict(frozen=True)

    max_personas_per_agent: Annotated[
        int, Field(ge=1, description="Maximum personas per agent")
    ] = 10


class MemoryTypesConfig(BaseModel):
    """Configuration for all memory types."""

    model_config = ConfigDict(frozen=True)

    conversational: Annotated[
        ConversationalConfig,
        Field(default_factory=ConversationalConfig, description="Conversational memory config"),
    ] = ConversationalConfig()

    knowledge_base: Annotated[
        KnowledgeBaseConfig,
        Field(default_factory=KnowledgeBaseConfig, description="Knowledge base memory config"),
    ] = KnowledgeBaseConfig()

    entity: Annotated[
        EntityMemoryConfig,
        Field(default_factory=EntityMemoryConfig, description="Entity memory config"),
    ] = EntityMemoryConfig()

    workflow: Annotated[
        WorkflowConfig,
        Field(default_factory=WorkflowConfig, description="Workflow memory config"),
    ] = WorkflowConfig()

    toolbox: Annotated[
        ToolboxConfig,
        Field(default_factory=ToolboxConfig, description="Toolbox memory config"),
    ] = ToolboxConfig()

    tool_log: Annotated[
        ToolLogConfig,
        Field(default_factory=ToolLogConfig, description="Tool log memory config"),
    ] = ToolLogConfig()

    skills: Annotated[
        SkillsConfig,
        Field(default_factory=SkillsConfig, description="Skills memory config"),
    ] = SkillsConfig()

    file: Annotated[
        FileMemoryConfig,
        Field(default_factory=FileMemoryConfig, description="File memory config"),
    ] = FileMemoryConfig()

    persona: Annotated[
        PersonaConfig,
        Field(default_factory=PersonaConfig, description="Persona memory config"),
    ] = PersonaConfig()


class MemharnessConfig(BaseModel):
    """Root configuration for the memharness package.

    This is the main configuration object that contains all settings
    for the memory harness system.

    Example:
        ```python
        config = MemharnessConfig(
            backend="postgresql://localhost/memharness",
            memory_types=MemoryTypesConfig(
                conversational=ConversationalConfig(max_messages_per_thread=500)
            ),
            agents=AgentConfig(
                summarizer=SummarizationConfig(enabled=True)
            )
        )
        ```
    """

    model_config = ConfigDict(frozen=True)

    backend: Annotated[
        str,
        Field(
            description="Backend connection string (e.g., 'postgresql://...', 'sqlite://...')"
        ),
    ] = "sqlite:///memharness.db"

    memory_types: Annotated[
        MemoryTypesConfig,
        Field(default_factory=MemoryTypesConfig, description="Configuration for all memory types"),
    ] = MemoryTypesConfig()

    agents: Annotated[
        AgentConfig,
        Field(default_factory=AgentConfig, description="Configuration for embedded AI agents"),
    ] = AgentConfig()

    debug: Annotated[bool, Field(description="Enable debug mode")] = False

    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Field(description="Logging level"),
    ] = "INFO"


# Alias for backward compatibility
Config = MemharnessConfig
