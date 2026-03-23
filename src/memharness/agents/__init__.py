# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Memory agents for intelligent memory management.

This module provides AI agents that perform automatic maintenance,

All agents work WITHOUT an LLM (using deterministic fallbacks) but can
leverage LLMs when provided for more intelligent operations.
"""

from memharness.agents.agent_workflow import create_after_workflow
from memharness.agents.base import BaseMemoryAgent
from memharness.agents.consolidator import ConsolidatorAgent
from memharness.agents.context_assembler import AssembledContext, ContextAssemblyAgent
from memharness.agents.entity_extractor import EntityExtractorAgent
from memharness.agents.summarizer import SummarizerAgent

__all__ = [
    "BaseMemoryAgent",
    "SummarizerAgent",
    "EntityExtractorAgent",
    "ConsolidatorAgent",
    "AssembledContext",
    "ContextAssemblyAgent",
    "create_after_workflow",
]
