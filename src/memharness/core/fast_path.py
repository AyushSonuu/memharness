# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Fast path for user-facing agent interactions.

Handles the hot path: save message → assemble context → return.
No extraction, no summarization. Keep it fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memharness.agents.context_assembler import AssembledContext
    from memharness.core.harness import MemoryHarness

__all__ = ["AgentResponse", "FastPath"]


@dataclass
class AgentResponse:
    """Response from fast path processing.

    Attributes:
        content: The agent's response content.
        context: The assembled context used for the response.
        thread_id: The conversation thread ID.
    """

    content: str
    context: AssembledContext
    thread_id: str


class FastPath:
    """Fast path for user-facing agent interactions.

    Handles the hot path: save message → assemble context → return.
    No extraction, no summarization. Keep it fast.

    This class implements the deterministic, low-latency path for agent
    interactions. It focuses on:
    1. Saving user messages (deterministic write)
    2. Assembling context from memory (deterministic reads, sorted by updated_at)
    3. Returning context to LLM
    4. Saving assistant responses (deterministic write)

    All heavy processing (extraction, summarization, consolidation) is deferred
    to the slow path (background workers).

    Attributes:
        harness: The MemoryHarness instance to operate on.
        max_context_tokens: Maximum tokens for assembled context.

    Example:
        ```python
        from memharness import MemoryHarness
        from memharness.core.fast_path import FastPath

        harness = MemoryHarness('sqlite:///memory.db')
        await harness.connect()

        fast = FastPath(harness)

        # User sends message
        ctx = await fast.process_user_message('thread-1', 'How do I deploy?')
        messages = ctx.to_messages()  # feed to LLM

        # LLM responds
        response = '...'
        await fast.process_assistant_response('thread-1', response)
        ```
    """

    def __init__(self, harness: MemoryHarness, max_context_tokens: int = 4000) -> None:
        """Initialize the fast path.

        Args:
            harness: The MemoryHarness instance to operate on.
            max_context_tokens: Maximum tokens for assembled context (default: 4000).
        """
        self.harness = harness
        self.max_context_tokens = max_context_tokens

        # Lazy import to avoid circular dependency
        from memharness.agents.context_assembler import ContextAssemblyAgent

        self._ctx_agent = ContextAssemblyAgent(harness, max_tokens=max_context_tokens)

    async def process_user_message(self, thread_id: str, message: str) -> AssembledContext:
        """Process incoming user message. Returns assembled context for LLM.

        This method implements the fast path for user messages:
        1. Save user message to conversational memory (deterministic write)
        2. Assemble context from all memory types (deterministic reads)
        3. Return context for LLM consumption

        No extraction or summarization happens in this path. Those operations
        are deferred to the slow path (background workers).

        Args:
            thread_id: The conversation thread ID.
            message: The user's message.

        Returns:
            AssembledContext ready for LLM consumption.

        Example:
            ```python
            ctx = await fast.process_user_message('thread-1', 'How do I deploy?')
            prompt = ctx.to_prompt()
            # Or use LangChain messages:
            messages = ctx.to_messages()
            ```
        """
        # Assemble context (this internally saves the user message)
        ctx = await self._ctx_agent.assemble(query=message, thread_id=thread_id)
        return ctx

    async def process_assistant_response(self, thread_id: str, response: str) -> str:
        """Save assistant response. Returns the memory ID.

        This method saves the assistant's response to conversational memory.
        No extraction or processing happens here - it's a pure write operation.

        Args:
            thread_id: The conversation thread ID.
            response: The assistant's response content.

        Returns:
            The memory ID of the saved response.

        Example:
            ```python
            memory_id = await fast.process_assistant_response('thread-1', 'Deploy using...')
            ```
        """
        return await self.harness.add_conversational(thread_id, "assistant", response)
