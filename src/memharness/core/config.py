# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Configuration module for memharness.

This module provides the MemharnessConfig dataclass and utilities for
loading configuration from files or dictionaries.

Example:
    from memharness.core.config import MemharnessConfig

    # From dict
    config = MemharnessConfig.from_dict({"default_k": 5})

    # From file
    config = MemharnessConfig.from_file("config.yaml")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["MemharnessConfig"]


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
        """
        Create config from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            MemharnessConfig instance with values from dict.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: str) -> MemharnessConfig:
        """
        Load config from a JSON or YAML file.

        Args:
            path: Path to the configuration file (.json, .yaml, or .yml).

        Returns:
            MemharnessConfig instance loaded from file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ImportError: If YAML file is used but PyYAML is not installed.
        """
        file_path = Path(path)
        if not file_path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        content = file_path.read_text()

        if file_path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError as exc:
                msg = (
                    "PyYAML is required to load YAML config files. Install with: pip install pyyaml"
                )
                raise ImportError(msg) from exc
        else:
            data = json.loads(content)

        return cls.from_dict(data)
