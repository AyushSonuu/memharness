"""Configuration loader for memharness.

This module provides utilities for loading configuration from various sources:
- YAML files
- Environment variables
- Default values

It also provides duration parsing utilities for TTL and schedule configurations.
"""

from __future__ import annotations

import os
import re
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

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

if TYPE_CHECKING:
    from collections.abc import Mapping


class ConfigLoadError(Exception):
    """Raised when configuration loading fails."""

    pass


class DurationParseError(Exception):
    """Raised when duration string parsing fails."""

    pass


# Duration unit mappings
_DURATION_UNITS: dict[str, int] = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "w": 604800,
    "week": 604800,
    "weeks": 604800,
}

# Pattern for parsing duration strings
_DURATION_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$", re.IGNORECASE
)


def parse_duration(duration_str: str) -> timedelta:
    """Parse a duration string into a timedelta.

    Supports various formats:
    - Seconds: "30s", "30sec", "30 seconds"
    - Minutes: "5m", "5min", "5 minutes"
    - Hours: "2h", "2hr", "2 hours"
    - Days: "7d", "7day", "7 days"
    - Weeks: "1w", "1week", "1 weeks"

    Args:
        duration_str: Duration string to parse (e.g., "7d", "24h", "30m")

    Returns:
        timedelta representing the duration

    Raises:
        DurationParseError: If the duration string cannot be parsed

    Examples:
        >>> parse_duration("7d")
        datetime.timedelta(days=7)
        >>> parse_duration("24h")
        datetime.timedelta(days=1)
        >>> parse_duration("30m")
        datetime.timedelta(seconds=1800)
    """
    if not duration_str or not isinstance(duration_str, str):
        raise DurationParseError(f"Invalid duration: {duration_str!r}")

    match = _DURATION_PATTERN.match(duration_str)
    if not match:
        raise DurationParseError(
            f"Cannot parse duration: {duration_str!r}. "
            "Expected format like '7d', '24h', '30m', '60s'"
        )

    value_str, unit = match.groups()
    value = float(value_str)
    unit_lower = unit.lower()

    if unit_lower not in _DURATION_UNITS:
        valid_units = ", ".join(sorted(set(_DURATION_UNITS.keys())))
        raise DurationParseError(
            f"Unknown duration unit: {unit!r}. Valid units: {valid_units}"
        )

    seconds = value * _DURATION_UNITS[unit_lower]
    return timedelta(seconds=seconds)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Values from override take precedence over base.
    Nested dictionaries are merged recursively.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _convert_triggers(triggers_data: list[dict[str, Any]]) -> list[SummarizationTrigger]:
    """Convert raw trigger dicts to SummarizationTrigger objects."""
    return [SummarizationTrigger(**t) for t in triggers_data]


def _build_config_from_dict(data: Mapping[str, Any]) -> MemharnessConfig:
    """Build MemharnessConfig from a dictionary.

    Args:
        data: Configuration dictionary

    Returns:
        MemharnessConfig instance

    Raises:
        ConfigLoadError: If validation fails
    """
    try:
        # Handle nested agent configs with triggers conversion
        agents_data = data.get("agents", {})
        if "summarizer" in agents_data and "triggers" in agents_data["summarizer"]:
            agents_data = dict(agents_data)
            summarizer_data = dict(agents_data["summarizer"])
            summarizer_data["triggers"] = _convert_triggers(summarizer_data["triggers"])
            agents_data["summarizer"] = summarizer_data

        config_data = dict(data)
        if agents_data:
            config_data["agents"] = agents_data

        return MemharnessConfig(**config_data)
    except ValidationError as e:
        raise ConfigLoadError(f"Configuration validation failed: {e}") from e


def from_yaml(path: str | Path) -> MemharnessConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        MemharnessConfig instance

    Raises:
        ConfigLoadError: If the file cannot be read or parsed

    Example:
        ```python
        config = from_yaml("config.yaml")
        ```

    Example YAML file:
        ```yaml
        backend: postgresql://localhost/memharness
        memory_types:
          conversational:
            max_messages_per_thread: 500
            default_ttl: 7d
        agents:
          summarizer:
            enabled: true
            triggers:
              - condition: "age > 7d"
                memory_type: conversational
        ```
    """
    path = Path(path)

    if not path.exists():
        raise ConfigLoadError(f"Configuration file not found: {path}")

    if not path.is_file():
        raise ConfigLoadError(f"Configuration path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Failed to parse YAML file {path}: {e}") from e
    except OSError as e:
        raise ConfigLoadError(f"Failed to read configuration file {path}: {e}") from e

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ConfigLoadError(
            f"Configuration file must contain a YAML mapping, got {type(data).__name__}"
        )

    return _build_config_from_dict(data)


# Environment variable prefix
_ENV_PREFIX = "MEMHARNESS_"


def _get_env_value(key: str, default: Any = None) -> str | None:
    """Get environment variable with prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _parse_bool(value: str | None) -> bool | None:
    """Parse boolean from string."""
    if value is None:
        return None
    value_lower = value.lower()
    if value_lower in ("true", "1", "yes", "on"):
        return True
    if value_lower in ("false", "0", "no", "off"):
        return False
    return None


def _parse_int(value: str | None) -> int | None:
    """Parse integer from string."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    """Parse float from string."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_list(value: str | None) -> list[str] | None:
    """Parse comma-separated list from string."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def from_env() -> MemharnessConfig:
    """Load configuration from environment variables.

    Environment variables use the MEMHARNESS_ prefix and underscore-separated
    nested keys.

    Supported environment variables:
        - MEMHARNESS_BACKEND: Backend connection string
        - MEMHARNESS_DEBUG: Enable debug mode (true/false)
        - MEMHARNESS_LOG_LEVEL: Logging level

        Conversational memory:
        - MEMHARNESS_CONVERSATIONAL_MAX_MESSAGES: Max messages per thread
        - MEMHARNESS_CONVERSATIONAL_DEFAULT_TTL: Default TTL
        - MEMHARNESS_CONVERSATIONAL_AUTO_SUMMARIZE_THRESHOLD: Auto-summarize threshold

        Agents:
        - MEMHARNESS_SUMMARIZER_ENABLED: Enable summarization (true/false)
        - MEMHARNESS_SUMMARIZER_KEEP_ORIGINALS: Keep originals (true/false)
        - MEMHARNESS_SUMMARIZER_ORIGINALS_TTL: TTL for originals

        - MEMHARNESS_CONSOLIDATION_ENABLED: Enable consolidation (true/false)
        - MEMHARNESS_CONSOLIDATION_SCHEDULE: Cron schedule
        - MEMHARNESS_CONSOLIDATION_SIMILARITY_THRESHOLD: Similarity threshold

        - MEMHARNESS_GC_ENABLED: Enable GC (true/false)
        - MEMHARNESS_GC_SCHEDULE: Cron schedule
        - MEMHARNESS_GC_ARCHIVE_AFTER: Archive duration
        - MEMHARNESS_GC_DELETE_AFTER: Delete duration

        - MEMHARNESS_ENTITY_EXTRACTION_ENABLED: Enable entity extraction (true/false)
        - MEMHARNESS_ENTITY_EXTRACTION_MODE: Extraction mode
        - MEMHARNESS_ENTITY_EXTRACTION_TYPES: Comma-separated entity types

        - MEMHARNESS_CONTEXT_ASSEMBLY_MAX_TOKENS: Max tokens
        - MEMHARNESS_TOOL_DISCOVERY_ENABLED: Enable tool discovery (true/false)
        - MEMHARNESS_TOOL_DISCOVERY_MAX_ITERATIONS: Max iterations

    Returns:
        MemharnessConfig instance built from environment variables

    Example:
        ```bash
        export MEMHARNESS_BACKEND="postgresql://localhost/memharness"
        export MEMHARNESS_DEBUG="true"
        export MEMHARNESS_CONVERSATIONAL_MAX_MESSAGES="500"
        ```

        ```python
        config = from_env()
        assert config.backend == "postgresql://localhost/memharness"
        assert config.debug is True
        ```
    """
    # Build config dict from environment variables
    config_data: dict[str, Any] = {}

    # Root level
    if backend := _get_env_value("BACKEND"):
        config_data["backend"] = backend

    if debug := _parse_bool(_get_env_value("DEBUG")):
        config_data["debug"] = debug

    if log_level := _get_env_value("LOG_LEVEL"):
        config_data["log_level"] = log_level

    # Conversational memory config
    conv_data: dict[str, Any] = {}
    if max_msgs := _parse_int(_get_env_value("CONVERSATIONAL_MAX_MESSAGES")):
        conv_data["max_messages_per_thread"] = max_msgs
    if ttl := _get_env_value("CONVERSATIONAL_DEFAULT_TTL"):
        conv_data["default_ttl"] = ttl
    if threshold := _parse_int(_get_env_value("CONVERSATIONAL_AUTO_SUMMARIZE_THRESHOLD")):
        conv_data["auto_summarize_threshold"] = threshold

    if conv_data:
        config_data.setdefault("memory_types", {})["conversational"] = conv_data

    # Agent configs
    agents_data: dict[str, Any] = {}

    # Summarizer
    summarizer_data: dict[str, Any] = {}
    if enabled := _parse_bool(_get_env_value("SUMMARIZER_ENABLED")):
        summarizer_data["enabled"] = enabled
    if keep := _parse_bool(_get_env_value("SUMMARIZER_KEEP_ORIGINALS")):
        summarizer_data["keep_originals"] = keep
    if ttl := _get_env_value("SUMMARIZER_ORIGINALS_TTL"):
        summarizer_data["originals_ttl"] = ttl
    if summarizer_data:
        agents_data["summarizer"] = summarizer_data

    # Consolidation
    consol_data: dict[str, Any] = {}
    if enabled := _parse_bool(_get_env_value("CONSOLIDATION_ENABLED")):
        consol_data["enabled"] = enabled
    if schedule := _get_env_value("CONSOLIDATION_SCHEDULE"):
        consol_data["schedule"] = schedule
    if threshold := _parse_float(_get_env_value("CONSOLIDATION_SIMILARITY_THRESHOLD")):
        consol_data["similarity_threshold"] = threshold
    if consol_data:
        agents_data["consolidator"] = consol_data

    # GC
    gc_data: dict[str, Any] = {}
    if enabled := _parse_bool(_get_env_value("GC_ENABLED")):
        gc_data["enabled"] = enabled
    if schedule := _get_env_value("GC_SCHEDULE"):
        gc_data["schedule"] = schedule
    if archive := _get_env_value("GC_ARCHIVE_AFTER"):
        gc_data["archive_after"] = archive
    if delete := _get_env_value("GC_DELETE_AFTER"):
        gc_data["delete_after"] = delete
    if gc_data:
        agents_data["gc"] = gc_data

    # Entity extraction
    entity_data: dict[str, Any] = {}
    if enabled := _parse_bool(_get_env_value("ENTITY_EXTRACTION_ENABLED")):
        entity_data["enabled"] = enabled
    if mode := _get_env_value("ENTITY_EXTRACTION_MODE"):
        entity_data["mode"] = mode
    if types := _parse_list(_get_env_value("ENTITY_EXTRACTION_TYPES")):
        entity_data["types"] = types
    if entity_data:
        agents_data["entity_extractor"] = entity_data

    # Context assembly
    ctx_data: dict[str, Any] = {}
    if max_tokens := _parse_int(_get_env_value("CONTEXT_ASSEMBLY_MAX_TOKENS")):
        ctx_data["default_max_tokens"] = max_tokens
    if ctx_data:
        agents_data["context_assembly"] = ctx_data

    # Tool discovery
    tool_data: dict[str, Any] = {}
    if enabled := _parse_bool(_get_env_value("TOOL_DISCOVERY_ENABLED")):
        tool_data["enabled"] = enabled
    if max_iter := _parse_int(_get_env_value("TOOL_DISCOVERY_MAX_ITERATIONS")):
        tool_data["max_iterations"] = max_iter
    if tool_data:
        agents_data["tool_discovery"] = tool_data

    if agents_data:
        config_data["agents"] = agents_data

    return _build_config_from_dict(config_data)


def from_yaml_with_env(path: str | Path) -> MemharnessConfig:
    """Load configuration from YAML with environment variable overrides.

    This function first loads configuration from a YAML file, then
    applies any environment variable overrides.

    Args:
        path: Path to the YAML configuration file

    Returns:
        MemharnessConfig instance with env overrides applied

    Example:
        ```python
        # config.yaml sets backend to "sqlite:///local.db"
        # MEMHARNESS_BACKEND env var is set to "postgresql://prod/db"
        config = from_yaml_with_env("config.yaml")
        # config.backend will be "postgresql://prod/db"
        ```
    """
    yaml_config = from_yaml(path)
    env_config = from_env()

    # Convert both to dicts and merge
    yaml_dict = yaml_config.model_dump()
    env_dict = env_config.model_dump()

    # Only include non-default values from env
    default_config = MemharnessConfig()
    default_dict = default_config.model_dump()

    # Filter env_dict to only include values that differ from defaults
    filtered_env: dict[str, Any] = {}
    for key, value in env_dict.items():
        if value != default_dict.get(key):
            filtered_env[key] = value

    merged = _deep_merge(yaml_dict, filtered_env)
    return _build_config_from_dict(merged)


def get_default_config() -> MemharnessConfig:
    """Get the default configuration.

    Returns:
        MemharnessConfig with all default values
    """
    return MemharnessConfig()


__all__ = [
    "ConfigLoadError",
    "DurationParseError",
    "from_env",
    "from_yaml",
    "from_yaml_with_env",
    "get_default_config",
    "parse_duration",
]
