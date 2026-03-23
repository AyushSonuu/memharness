# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Base protocol for memory agents.

Defines the interface that all memory agents must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from memharness.core.harness import MemoryHarness


class BaseMemoryAgent(Protocol):
    """
    Protocol for memory agents.

    All agents must implement this interface to work with the memory harness.
    Agents should work WITHOUT an LLM (using heuristics) and optionally use
    an LLM when provided for more intelligent operations.
    """

    harness: MemoryHarness
    llm: BaseChatModel | None

    def __init__(self, harness: MemoryHarness, llm: BaseChatModel | None = None) -> None:
        """
        Initialize the agent.

        Args:
            harness: The MemoryHarness instance to operate on.
            llm: Optional LLM for intelligent operations (None = heuristic mode).
        """
        ...

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the agent's main logic.

        Args:
            **kwargs: Agent-specific arguments.

        Returns:
            Dictionary with results (keys depend on agent type).
        """
        ...
