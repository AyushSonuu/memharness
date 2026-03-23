# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Memory persistence middleware for LangChain agents.

This middleware automatically stores agent interactions (user messages and
AI responses) in conversational memory after each model call, creating a
persistent conversation history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError as e:
    raise ImportError(
        "LangChain is required for middleware. Install with: pip install memharness[langchain]"
    ) from e

if TYPE_CHECKING:
    from langchain.agents.middleware.types import Runtime

    from memharness import MemoryHarness

__all__ = ["MemoryPersistenceMiddleware"]


class MemoryPersistenceMiddleware(AgentMiddleware):
    """
    Middleware that persists agent interactions to conversational memory.

    This middleware stores both user messages and AI responses in the
    conversational memory store after each model call, creating a persistent
    history that can be retrieved later for context.

    Example:
        ```python
        from memharness import MemoryHarness
        from memharness.middleware import MemoryPersistenceMiddleware
        from langchain.agents import create_agent

        harness = MemoryHarness('sqlite:///memory.db')
        await harness.connect()

        agent = create_agent(
            model='anthropic:claude-sonnet-4-6',
            tools=[...],
            middleware=[
                MemoryPersistenceMiddleware(
                    harness=harness,
                    thread_id='main'
                )
            ]
        )
        ```
    """

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
        store_user_messages: bool = True,
        store_ai_messages: bool = True,
    ) -> None:
        """
        Initialize the memory persistence middleware.

        Args:
            harness: The MemoryHarness instance to store messages in.
            thread_id: The conversation thread ID for this agent session.
            store_user_messages: Whether to store user messages (default: True).
            store_ai_messages: Whether to store AI responses (default: True).
        """
        super().__init__()
        self.harness = harness
        self.thread_id = thread_id
        self.store_user_messages = store_user_messages
        self.store_ai_messages = store_ai_messages
        self._stored_message_ids: set[str] = set()

    async def aafter_model(
        self, state: dict[str, Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """
        Store the interaction in conversational memory after the model responds.

        Extracts user and AI messages from the state and stores them in the
        harness's conversational memory for the configured thread.

        Args:
            state: The current agent state containing messages.
            runtime: The runtime context (unused).

        Returns:
            None (no state updates needed).
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        try:
            # Find the last user message and AI response that we haven't stored yet
            user_msg = None
            ai_msg = None

            # Iterate from the end to find the most recent pair
            for msg in reversed(messages):
                msg_id = getattr(msg, "id", None)
                if msg_id and msg_id in self._stored_message_ids:
                    # Already stored this message
                    continue

                if isinstance(msg, AIMessage) and ai_msg is None:
                    ai_msg = msg
                elif isinstance(msg, HumanMessage) and user_msg is None:
                    user_msg = msg

                # Stop once we have both
                if user_msg and ai_msg:
                    break

            # Store user message
            if user_msg and self.store_user_messages:
                content = user_msg.content if hasattr(user_msg, "content") else str(user_msg)
                if content and isinstance(content, str):
                    await self.harness.add_conversational(
                        thread_id=self.thread_id,
                        role="user",
                        content=content,
                    )
                    if hasattr(user_msg, "id") and user_msg.id:
                        self._stored_message_ids.add(user_msg.id)

            # Store AI response
            if ai_msg and self.store_ai_messages:
                content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
                if content and isinstance(content, str):
                    await self.harness.add_conversational(
                        thread_id=self.thread_id,
                        role="assistant",
                        content=content,
                    )
                    if hasattr(ai_msg, "id") and ai_msg.id:
                        self._stored_message_ids.add(ai_msg.id)

        except Exception:
            # If persistence fails, continue without raising
            # (don't break the agent flow)
            pass

        return None
