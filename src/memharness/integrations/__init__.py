# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Integrations for third-party frameworks.

This module provides adapters and integrations for popular LLM frameworks:
- LangChain: MemharnessMemory for LangChain conversation memory
- LangGraph: MemharnessCheckpointer for LangGraph state persistence

Usage:
    # LangChain integration
    from memharness.integrations import MemharnessMemory

    memory = MemharnessMemory(harness=harness, thread_id="my-thread")
    chain = ConversationChain(llm=llm, memory=memory)

    # LangGraph integration
    from memharness.integrations import MemharnessCheckpointer

    checkpointer = MemharnessCheckpointer(harness=harness)
    graph = StateGraph(State, checkpointer=checkpointer)

Note: These integrations require optional dependencies.
Install with: pip install memharness[langchain] or pip install memharness[langgraph]
"""

from memharness.integrations.langchain import (
    LANGCHAIN_AVAILABLE,
    MemharnessChatHistory,
)
from memharness.integrations.langgraph import (
    LANGGRAPH_AVAILABLE,
    MemharnessCheckpointer,
)

__all__ = [
    "MemharnessChatHistory",
    "MemharnessCheckpointer",
    "LANGCHAIN_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
]
