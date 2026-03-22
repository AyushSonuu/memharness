# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Embedded agents for intelligent memory management.

This module provides AI agents that run embedded within the memory system
to perform automatic maintenance, summarization, entity extraction, and more.

All agents work WITHOUT an LLM (using deterministic fallbacks) but can
leverage LLMs when provided for more intelligent operations.
"""

from memharness.agents.base import EmbeddedAgent, TriggerType, AgentConfig
from memharness.agents.summarizer import SummarizerAgent
from memharness.agents.entity_extractor import EntityExtractorAgent
from memharness.agents.consolidator import ConsolidatorAgent
from memharness.agents.gc import GCAgent
from memharness.agents.scheduler import AgentScheduler

__all__ = [
    # Base classes
    "EmbeddedAgent",
    "TriggerType",
    "AgentConfig",
    # Agents
    "SummarizerAgent",
    "EntityExtractorAgent",
    "ConsolidatorAgent",
    "GCAgent",
    # Scheduler
    "AgentScheduler",
]
