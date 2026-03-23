# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
memharness - Framework-agnostic memory infrastructure for AI agents.

This package provides a complete memory layer for AI agents with:
- 10 memory types (Conversational, KB, Entity, Workflow, Toolbox, Summary, ToolLog, Skills, File, Persona)
- Multiple backends (PostgreSQL, SQLite, In-memory)
- Embedded AI agents for complex operations
- Self-exploration tools for agents
- Full configurability

Quick Start:
    from memharness import MemoryHarness

    memory = MemoryHarness("sqlite:///memory.db")
    await memory.add_conversational(thread_id="t1", role="user", content="Hello")
"""

from memharness.config import Config, MemharnessConfig
from memharness.core.harness import MemoryHarness
from memharness.registry import MemoryTypeRegistry
from memharness.types import MemoryType, MemoryUnit

__version__ = "0.2.0"
__author__ = "Ayush Sonuu"
__license__ = "MIT"

__all__ = [
    "MemoryHarness",
    "MemoryUnit",
    "MemoryType",
    "Config",
    "MemharnessConfig",
    "MemoryTypeRegistry",
    "__version__",
]
