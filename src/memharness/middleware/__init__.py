# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
LangChain middleware for memharness.

This module provides middleware components that can be used with LangChain's
create_agent to automatically inject memory context, persist conversations,
and extract entities from agent interactions.

Middleware classes:
- MemoryContextMiddleware: Injects relevant memory context before model calls
- MemoryPersistenceMiddleware: Stores interactions in conversational memory
- EntityExtractionMiddleware: Extracts and stores entities from responses

Example:
    ```python
    from memharness import MemoryHarness
    from memharness.middleware import (
        MemoryContextMiddleware,
        MemoryPersistenceMiddleware,
        EntityExtractionMiddleware,
    )
    from langchain.agents import create_agent

    harness = MemoryHarness('sqlite:///memory.db')
    await harness.connect()

    agent = create_agent(
        model='anthropic:claude-sonnet-4-6',
        tools=[...],
        middleware=[
            MemoryContextMiddleware(harness=harness, thread_id='main'),
            EntityExtractionMiddleware(harness=harness, thread_id='main'),
            MemoryPersistenceMiddleware(harness=harness, thread_id='main'),
        ],
    )
    ```
"""

try:
    from memharness.middleware.entity_extraction import EntityExtractionMiddleware
    from memharness.middleware.memory_context import MemoryContextMiddleware
    from memharness.middleware.persistence import MemoryPersistenceMiddleware

    MIDDLEWARE_AVAILABLE = True
except ImportError:
    MIDDLEWARE_AVAILABLE = False
    # Stub classes when langchain is not available
    MemoryContextMiddleware = None  # type: ignore[misc, assignment]
    MemoryPersistenceMiddleware = None  # type: ignore[misc, assignment]
    EntityExtractionMiddleware = None  # type: ignore[misc, assignment]

__all__ = [
    "MemoryContextMiddleware",
    "MemoryPersistenceMiddleware",
    "EntityExtractionMiddleware",
    "MIDDLEWARE_AVAILABLE",
]
