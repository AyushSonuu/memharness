# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
LangChain integration for memharness.

This module provides LangChain-compatible memory classes that use memharness
as the backend storage. This allows you to use memharness's powerful memory
infrastructure with LangChain chains and agents.

Example:
    from langchain_core.chat_history import BaseChatMessageHistory
    from memharness import MemoryHarness
    from memharness.integrations import MemharnessChatHistory

    # Initialize memharness
    harness = MemoryHarness("sqlite:///memory.db")
    await harness.connect()

    # Create LangChain-compatible chat history
    history = MemharnessChatHistory(harness=harness, thread_id="conversation-1")

    # Use with LangChain
    await history.add_user_message("Hello!")
    await history.add_ai_message("Hi! How can I help?")
    messages = await history.aget_messages()

Note: Requires langchain-core to be installed.
Install with: pip install memharness[langchain]
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

# Optional dependency handling for LangChain
try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Provide stub classes for type checking when langchain is not installed
    BaseChatMessageHistory = object  # type: ignore[misc, assignment]
    BaseMessage = object  # type: ignore[misc, assignment]
    HumanMessage = None  # type: ignore[misc, assignment]
    AIMessage = None  # type: ignore[misc, assignment]
    SystemMessage = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from memharness import MemoryHarness
    from memharness.types import MemoryUnit


__all__ = [
    "MemharnessChatHistory",
    "MemharnessMemory",
    "LANGCHAIN_AVAILABLE",
]


class MemharnessChatHistory(BaseChatMessageHistory):
    """
    LangChain-compatible chat history using memharness backend.

    This class implements the BaseChatMessageHistory interface from langchain-core,
    allowing seamless integration with LangChain chains and agents.

    Attributes:
        harness: The MemoryHarness instance to use for storage.
        thread_id: The conversation thread identifier.

    Example:
        >>> harness = MemoryHarness("sqlite:///memory.db")
        >>> await harness.connect()
        >>> history = MemharnessChatHistory(harness=harness, thread_id="thread-1")
        >>>
        >>> # Add messages
        >>> await history.add_user_message("What is the weather?")
        >>> await history.add_ai_message("I don't have weather data.")
        >>>
        >>> # Get messages
        >>> messages = await history.aget_messages()
        >>> print(messages)
    """

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
    ) -> None:
        """
        Initialize MemharnessChatHistory.

        Args:
            harness: The MemoryHarness instance to use for storage.
            thread_id: The conversation thread identifier.

        Raises:
            ImportError: If langchain-core is not installed.
        """
        if not LANGCHAIN_AVAILABLE:
            msg = "langchain-core is not installed. Install with: pip install memharness[langchain]"
            raise ImportError(msg)

        self.harness = harness
        self.thread_id = thread_id

    @property
    def messages(self) -> list[BaseMessage]:
        """
        Synchronous property to get messages (required by BaseChatMessageHistory).

        Note: This runs the async method in a sync context.
        Prefer using aget_messages() in async contexts.

        Returns:
            List of LangChain BaseMessage objects.
        """
        return self._run_async(self.aget_messages())

    async def aget_messages(self) -> list[BaseMessage]:
        """
        Async method to get messages from memharness.

        Returns:
            List of LangChain BaseMessage objects.
        """
        memories = await self.harness.get_conversational(self.thread_id)
        return self._to_langchain_messages(memories)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Synchronous method to add messages.

        Args:
            messages: Sequence of BaseMessage objects to add.

        Note: This runs the async method in a sync context.
        Prefer using aadd_messages() in async contexts.
        """
        self._run_async(self.aadd_messages(messages))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Async method to add messages to memharness.

        Args:
            messages: Sequence of BaseMessage objects to add.
        """
        for message in messages:
            role = self._get_role_from_message(message)
            content = message.content
            if isinstance(content, list):
                # Handle complex content (images, etc.) by converting to string
                content = str(content)

            await self.harness.add_conversational(
                thread_id=self.thread_id,
                role=role,
                content=content,
            )

    def clear(self) -> None:
        """
        Synchronous method to clear conversation history.

        Note: This runs the async method in a sync context.
        Prefer using aclear() in async contexts.
        """
        self._run_async(self.aclear())

    async def aclear(self) -> None:
        """
        Async method to clear conversation history for this thread.
        """
        await self.harness.clear_thread(self.thread_id)

    def _to_langchain_messages(self, memories: list[MemoryUnit]) -> list[BaseMessage]:
        """
        Convert MemoryUnits to LangChain message objects.

        Args:
            memories: List of MemoryUnit objects from memharness.

        Returns:
            List of LangChain BaseMessage objects.
        """
        messages: list[BaseMessage] = []

        for mem in memories:
            role = mem.metadata.get("role", "user") if mem.metadata else "user"
            content = mem.content

            if role == "user" or role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "assistant" or role == "ai":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                # Default to human message for unknown roles
                messages.append(HumanMessage(content=content))

        return messages

    def _get_role_from_message(self, message: BaseMessage) -> str:
        """
        Extract role from LangChain message type.

        Args:
            message: LangChain BaseMessage object.

        Returns:
            Role string ("user", "assistant", or "system").
        """
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            # Default to user for unknown message types
            return "user"

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """
        Run an async coroutine in a sync context.

        This handles the complexity of running async code from sync methods,
        including when there's already an event loop running.

        Args:
            coro: The coroutine to run.

        Returns:
            The result of the coroutine.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(coro)

        # If we're in a running loop (e.g., Jupyter), use thread executor
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


# Backward compatibility: MemharnessMemory wraps MemharnessChatHistory
# with BaseMemory-like interface (save_context, load_memory_variables)


class MemharnessMemory:
    """Memory wrapper with BaseMemory-compatible interface.

    Provides save_context/load_memory_variables interface on top of
    MemharnessChatHistory for backward compatibility with older LangChain patterns.

    Args:
        harness: The MemoryHarness instance.
        thread_id: Conversation thread identifier.
        memory_key: Key for memory variables dict. Defaults to "history".
        return_messages: If True, return Message objects; if False, return string.
    """

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
        memory_key: str = "history",
        return_messages: bool = True,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            msg = "langchain-core required. Install: pip install memharness[langchain]"
            raise ImportError(msg)
        self.harness = harness
        self.thread_id = thread_id
        self.memory_key = memory_key
        self.return_messages = return_messages
        self._chat_history = MemharnessChatHistory(harness=harness, thread_id=thread_id)

    @property
    def memory_variables(self) -> list[str]:
        """Return the list of memory variable keys."""
        return [self.memory_key]

    async def aload_memory_variables(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load memory variables (async)."""
        messages = await self._chat_history.aget_messages()
        if self.return_messages:
            return {self.memory_key: messages}
        # Convert to string
        lines = []
        for msg in messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{role}: {msg.content}")
        return {self.memory_key: "\n".join(lines)}

    async def asave_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save conversation context (async)."""
        input_val = inputs.get("input", str(inputs))
        output_val = outputs.get("output", str(outputs))
        await self._chat_history.aadd_messages(
            [
                HumanMessage(content=input_val),
                AIMessage(content=output_val),
            ]
        )

    async def aclear(self) -> None:
        """Clear memory (async)."""
        await self._chat_history.aclear()
