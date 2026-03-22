"""Configuration module for memharness.

This module provides configuration management for the memharness package,
including Pydantic models for validation and loaders for YAML/environment sources.

Quick Start:
    ```python
    from memharness.config import MemharnessConfig, from_yaml, from_env

    # Load from YAML
    config = from_yaml("config.yaml")

    # Load from environment variables
    config = from_env()

    # Create with defaults
    config = MemharnessConfig()

    # Create with custom values
    config = MemharnessConfig(
        backend="postgresql://localhost/memharness",
        debug=True,
    )
    ```

Duration Parsing:
    ```python
    from memharness.config import parse_duration

    delta = parse_duration("7d")   # 7 days
    delta = parse_duration("24h")  # 24 hours
    delta = parse_duration("30m")  # 30 minutes
    ```
"""

from memharness.config.loader import (
    ConfigLoadError,
    DurationParseError,
    from_env,
    from_yaml,
    from_yaml_with_env,
    get_default_config,
    parse_duration,
)
from memharness.config.models import (
    AgentConfig,
    ConsolidationConfig,
    ContextAssemblyConfig,
    ConversationalConfig,
    EntityExtractionConfig,
    EntityMemoryConfig,
    FileMemoryConfig,
    GCConfig,
    KnowledgeBaseConfig,
    MemharnessConfig,
    MemoryTypesConfig,
    PersonaConfig,
    SkillsConfig,
    SummarizationConfig,
    SummarizationTrigger,
    ToolboxConfig,
    ToolDiscoveryConfig,
    ToolLogConfig,
    WorkflowConfig,
)

# Backward compatibility alias
Config = MemharnessConfig

__all__ = [
    # Main config class
    "Config",
    "MemharnessConfig",
    # Memory type configs
    "ConversationalConfig",
    "EntityMemoryConfig",
    "FileMemoryConfig",
    "KnowledgeBaseConfig",
    "MemoryTypesConfig",
    "PersonaConfig",
    "SkillsConfig",
    "ToolboxConfig",
    "ToolLogConfig",
    "WorkflowConfig",
    # Agent configs
    "AgentConfig",
    "ConsolidationConfig",
    "ContextAssemblyConfig",
    "EntityExtractionConfig",
    "GCConfig",
    "SummarizationConfig",
    "SummarizationTrigger",
    "ToolDiscoveryConfig",
    # Loaders
    "from_env",
    "from_yaml",
    "from_yaml_with_env",
    "get_default_config",
    # Utilities
    "parse_duration",
    # Exceptions
    "ConfigLoadError",
    "DurationParseError",
]
