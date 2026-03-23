# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Memory context middleware for LangChain agents.

This middleware automatically injects relevant memory context as a system
message before each model call, ensuring the agent has access to relevant
historical information, knowledge, entities, workflows, and persona data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import SystemMessage
except ImportError as e:
    raise ImportError(
        "LangChain is required for middleware. Install with: pip install memharness[langchain]"
    ) from e

if TYPE_CHECKING:
    from langchain.agents.middleware.types import Runtime

    from memharness import MemoryHarness

__all__ = ["MemoryContextMiddleware"]


class MemoryContextMiddleware(AgentMiddleware):
    """
    Middleware that injects relevant memory context before model calls.

    This middleware retrieves relevant memories from the harness based on the
    user's query and prepends them as a system message, providing the model
    with contextual information from past conversations, knowledge base,
    entities, workflows, and more.

    Example:
        ```python
        from memharness import MemoryHarness
        from memharness.middleware import MemoryContextMiddleware
        from langchain.agents import create_agent

        harness = MemoryHarness('sqlite:///memory.db')
        await harness.connect()

        agent = create_agent(
            model='anthropic:claude-sonnet-4-6',
            tools=[...],
            middleware=[
                MemoryContextMiddleware(
                    harness=harness,
                    thread_id='main',
                    max_tokens=2000
                )
            ]
        )
        ```
    """

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
        max_tokens: int = 2000,
        prefix: str = "Relevant memory context:",
    ) -> None:
        """
        Initialize the memory context middleware.

        Args:
            harness: The MemoryHarness instance to retrieve context from.
            thread_id: The conversation thread ID for this agent session.
            max_tokens: Maximum tokens to include in the context (default: 2000).
            prefix: Prefix to add before the context in the system message.
        """
        super().__init__()
        self.harness = harness
        self.thread_id = thread_id
        self.max_tokens = max_tokens
        self.prefix = prefix

    async def abefore_model(
        self, state: dict[str, Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """
        Inject memory context before the model is called.

        Retrieves relevant memories based on the last user message and prepends
        them as a system message to provide context to the model.

        Args:
            state: The current agent state containing messages.
            runtime: The runtime context (unused).

        Returns:
            State updates with memory context as a system message, or None if
            no context is available.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get the last message as the query
        last_message = messages[-1]
        query = last_message.content if hasattr(last_message, "content") else str(last_message)

        if not query or not isinstance(query, str):
            return None

        # Assemble context from memory
        try:
            context = await self.harness.assemble_context(
                query=query,
                thread_id=self.thread_id,
                max_tokens=self.max_tokens,
            )
        except Exception:
            # If context assembly fails, continue without context
            return None

        if not context or not context.strip():
            return None

        # Prepend memory context as a system message
        context_msg = SystemMessage(content=f"{self.prefix}\n\n{context}")

        # Return state updates with the context message prepended
        return {"messages": [context_msg]}
