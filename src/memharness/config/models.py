"""Pydantic configuration models for memharness.

This module defines all configuration models used throughout the memharness package.
All models use Pydantic v2 for validation, serialization, and documentation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationalConfig(BaseModel):
    """Configuration for conversational memory type."""

    model_config = ConfigDict(frozen=True)

    max_messages_per_thread: int = Field(
        default=1000, ge=1, description="Maximum messages allowed per conversation thread"
    )
    default_ttl: str | None = Field(
        default=None,
        description="Default TTL for messages (e.g., '7d', '24h'). None means no expiry",
    )
    auto_summarize_threshold: int = Field(
        default=50, ge=1, description="Number of messages after which auto-summarization triggers"
    )


class SummarizationTrigger(BaseModel):
    """A trigger condition for summarization."""

    model_config = ConfigDict(frozen=True)

    condition: str = Field(description="Trigger condition (e.g., 'age > 7d', 'count > 100')")
    memory_type: str = Field(description="Memory type this trigger applies to")


class SummarizationConfig(BaseModel):
    """Configuration for the summarization agent."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Whether summarization is enabled")
    triggers: list[SummarizationTrigger] = Field(
        default_factory=list, description="Trigger conditions"
    )
    keep_originals: bool = Field(
        default=True, description="Keep original memories after summarization"
    )
    originals_ttl: str = Field(default="365d", description="TTL for originals after summarization")


class ConsolidationConfig(BaseModel):
    """Configuration for the consolidation agent."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Whether consolidation is enabled")
    schedule: str = Field(
        default="0 3 * * *", description="Cron expression for consolidation schedule"
    )
    similarity_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Similarity threshold for duplicate detection"
    )


class GCConfig(BaseModel):
    """Configuration for the garbage collection agent."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Whether garbage collection is enabled")
    schedule: str = Field(default="0 4 * * 0", description="Cron expression for GC schedule")
    archive_after: str = Field(
        default="90d", description="Duration after which memories are archived"
    )
    delete_after: str = Field(
        default="365d", description="Duration after which archived memories are deleted"
    )


class EntityExtractionConfig(BaseModel):
    """Configuration for the entity extraction agent."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Whether entity extraction is enabled")
    mode: Literal["on_write", "batch", "disabled"] = Field(
        default="on_write", description="Extraction mode: on_write, batch, or disabled"
    )
    types: list[str] = Field(
        default_factory=lambda: ["PERSON", "ORG", "PLACE", "CONCEPT"],
        description="Entity types to extract",
    )


class ContextAssemblyConfig(BaseModel):
    """Configuration for context assembly."""

    model_config = ConfigDict(frozen=True)

    default_max_tokens: int = Field(
        default=4000, ge=1, description="Default max tokens for context"
    )
    priorities: dict[str, float] = Field(
        default_factory=dict, description="Memory type to priority percentage mapping"
    )


class ToolDiscoveryConfig(BaseModel):
    """Configuration for tool discovery."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Whether tool discovery is enabled")
    max_iterations: int = Field(default=10, ge=1, description="Max iterations for tool discovery")


class AgentConfig(BaseModel):
    """Configuration for all embedded AI agents."""

    model_config = ConfigDict(frozen=True)

    summarizer: SummarizationConfig = Field(default_factory=SummarizationConfig)
    entity_extractor: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)
    consolidator: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    gc: GCConfig = Field(default_factory=GCConfig)
    context_assembly: ContextAssemblyConfig = Field(default_factory=ContextAssemblyConfig)
    tool_discovery: ToolDiscoveryConfig = Field(default_factory=ToolDiscoveryConfig)


class KnowledgeBaseConfig(BaseModel):
    """Configuration for knowledge base memory type."""

    model_config = ConfigDict(frozen=True)

    default_collection: str = Field(default="default", description="Default collection name")
    embedding_model: str | None = Field(default=None, description="Embedding model to use")


class EntityMemoryConfig(BaseModel):
    """Configuration for entity memory type."""

    model_config = ConfigDict(frozen=True)

    auto_link: bool = Field(default=True, description="Automatically link related entities")
    max_relations_per_entity: int = Field(default=100, ge=1, description="Max relations per entity")


class WorkflowConfig(BaseModel):
    """Configuration for workflow memory type."""

    model_config = ConfigDict(frozen=True)

    max_steps_per_workflow: int = Field(default=1000, ge=1, description="Max steps per workflow")
    auto_archive_completed: bool = Field(
        default=True, description="Auto-archive completed workflows"
    )


class ToolboxConfig(BaseModel):
    """Configuration for toolbox memory type."""

    model_config = ConfigDict(frozen=True)

    cache_schemas: bool = Field(default=True, description="Cache tool schemas for faster access")


class ToolLogConfig(BaseModel):
    """Configuration for tool log memory type."""

    model_config = ConfigDict(frozen=True)

    max_logs_per_tool: int = Field(default=10000, ge=1, description="Max log entries per tool")
    retention_days: int = Field(default=30, ge=1, description="Days to retain tool logs")


class FileMemoryConfig(BaseModel):
    """Configuration for file memory type."""

    model_config = ConfigDict(frozen=True)

    max_file_size_mb: int = Field(default=100, ge=1, description="Max file size in MB")
    allowed_extensions: list[str] | None = Field(
        default=None, description="Allowed file extensions"
    )


class PersonaConfig(BaseModel):
    """Configuration for persona memory type."""

    model_config = ConfigDict(frozen=True)

    max_personas_per_agent: int = Field(default=10, ge=1, description="Max personas per agent")


class MemoryTypesConfig(BaseModel):
    """Configuration for all memory types."""

    model_config = ConfigDict(frozen=True)

    conversational: ConversationalConfig = Field(default_factory=ConversationalConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    entity: EntityMemoryConfig = Field(default_factory=EntityMemoryConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    toolbox: ToolboxConfig = Field(default_factory=ToolboxConfig)
    tool_log: ToolLogConfig = Field(default_factory=ToolLogConfig)
    file: FileMemoryConfig = Field(default_factory=FileMemoryConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)


class MemharnessConfig(BaseModel):
    """Root configuration for the memharness package."""

    model_config = ConfigDict(frozen=True)

    backend: str = Field(
        default="memory://",
        description="Backend connection string (e.g., 'postgresql://...', 'sqlite://...')",
    )
    memory_types: MemoryTypesConfig = Field(default_factory=MemoryTypesConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    @property
    def default_backend(self) -> str:
        """Alias for backend field for backward compatibility."""
        return self.backend

    @property
    def connection_string(self) -> str:
        """Alias for backend field for backward compatibility."""
        return self.backend

    @property
    def memory(self) -> MemoryTypesConfig:
        """Alias for memory_types for backward compatibility."""
        return self.memory_types

    @property
    def retention(self) -> dict[str, str | None]:
        """Get retention settings for all memory types.

        Returns a dict mapping memory type names to their TTL settings.
        """
        conversational_ttl = self.memory_types.conversational.default_ttl
        return {
            "conversational": conversational_ttl,
        }

    @classmethod
    def from_yaml(cls, path):
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            MemharnessConfig instance.

        Example:
            >>> config = MemharnessConfig.from_yaml("config.yaml")
        """
        # Import here to avoid circular dependency
        from memharness.config.loader import from_yaml as loader_from_yaml

        return loader_from_yaml(path)


# Alias for backward compatibility
Config = MemharnessConfig
