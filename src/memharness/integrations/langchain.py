# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
LangChain integration for memharness.

This module provides LangChain-compatible memory classes that use memharness
as the backend storage. This allows you to use memharness's powerful memory
infrastructure with LangChain chains and agents.

Example:
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI
    from memharness import MemoryHarness
    from memharness.integrations import MemharnessMemory

    # Initialize memharness
    harness = MemoryHarness("sqlite:///memory.db")

    # Create LangChain-compatible memory
    memory = MemharnessMemory(harness=harness, thread_id="conversation-1")

    # Use with LangChain
    llm = ChatOpenAI()
    chain = ConversationChain(llm=llm, memory=memory)
    response = chain.predict(input="Hello!")

Note: Requires langchain to be installed.
Install with: pip install memharness[langchain]
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

# Optional dependency handling for LangChain
try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Provide stub classes for type checking when langchain is not installed
    BaseChatMemory = object  # type: ignore[misc, assignment]
    BaseMessage = object  # type: ignore[misc, assignment]
    HumanMessage = None  # type: ignore[misc, assignment]
    AIMessage = None  # type: ignore[misc, assignment]
    SystemMessage = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from memharness import MemoryHarness
    from memharness.types import MemoryUnit


__all__ = [
    "MemharnessMemory",
    "LANGCHAIN_AVAILABLE",
]


class MemharnessMemory(BaseChatMemory):
    """
    LangChain-compatible memory using memharness backend.

    This class bridges memharness's conversational memory with LangChain's
    memory interface, allowing seamless integration with LangChain chains
    and agents.

    Attributes:
        harness: The MemoryHarness instance to use for storage.
        thread_id: The conversation thread identifier.
        memory_key: The key used in the memory variables dict (default: "history").
        return_messages: If True, return LangChain Message objects; if False, return string.
        human_prefix: Prefix for human messages when formatting as string.
        ai_prefix: Prefix for AI messages when formatting as string.
        input_key: Key to look for user input in inputs dict.
        output_key: Key to look for AI output in outputs dict.

    Example:
        >>> harness = MemoryHarness("sqlite:///memory.db")
        >>> memory = MemharnessMemory(harness=harness, thread_id="thread-1")
        >>>
        >>> # Save context
        >>> memory.save_context(
        ...     inputs={"input": "What is the weather?"},
        ...     outputs={"output": "I don't have access to weather data."}
        ... )
        >>>
        >>> # Load memory
        >>> variables = memory.load_memory_variables({})
        >>> print(variables["history"])
    """

    # Instance attributes (not class-level Pydantic fields to avoid issues without langchain)
    harness: Any  # MemoryHarness instance
    thread_id: str
    memory_key: str
    return_messages: bool
    human_prefix: str
    ai_prefix: str
    input_key: str | None
    output_key: str | None

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
        memory_key: str = "history",
        return_messages: bool = True,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        input_key: str | None = None,
        output_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MemharnessMemory.

        Args:
            harness: The MemoryHarness instance to use for storage.
            thread_id: The conversation thread identifier.
            memory_key: The key used in the memory variables dict.
            return_messages: If True, return LangChain Message objects.
            human_prefix: Prefix for human messages when formatting as string.
            ai_prefix: Prefix for AI messages when formatting as string.
            input_key: Key to look for user input (auto-detected if None).
            output_key: Key to look for AI output (auto-detected if None).
            **kwargs: Additional arguments passed to BaseChatMemory.

        Raises:
            ImportError: If langchain is not installed.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is not installed. "
                "Install with: pip install memharness[langchain]"
            )

        super().__init__(**kwargs)
        self.harness = harness
        self.thread_id = thread_id
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.input_key = input_key
        self.output_key = output_key

    @property
    def memory_variables(self) -> list[str]:
        """
        Return the list of memory variable keys.

        Returns:
            List containing the memory_key.
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Load conversation history from memharness.

        This method retrieves all conversational memories for the thread
        and returns them in the format expected by LangChain.

        Args:
            inputs: The current inputs (not used, but required by interface).

        Returns:
            Dict with memory_key mapping to either List[BaseMessage] or str,
            depending on the return_messages setting.
        """
        memories = self._run_async(
            self.harness.get_conversational(self.thread_id)
        )

        if self.return_messages:
            return {self.memory_key: self._to_langchain_messages(memories)}
        else:
            return {self.memory_key: self._format_as_string(memories)}

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Async version of load_memory_variables.

        Args:
            inputs: The current inputs (not used, but required by interface).

        Returns:
            Dict with memory_key mapping to conversation history.
        """
        memories = await self.harness.get_conversational(self.thread_id)

        if self.return_messages:
            return {self.memory_key: self._to_langchain_messages(memories)}
        else:
            return {self.memory_key: self._format_as_string(memories)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """
        Save user input and AI output to memharness.

        This method extracts the user input and AI output from the provided
        dicts and stores them as conversational memories.

        Args:
            inputs: Dict containing user input (looks for 'input', 'question',
                   or 'human_input' keys).
            outputs: Dict containing AI output (looks for 'output', 'answer',
                    or 'response' keys).
        """
        # Extract user input
        user_input = self._get_input_value(inputs)
        if user_input:
            self._run_async(
                self.harness.add_conversational(
                    thread_id=self.thread_id,
                    role="user",
                    content=user_input,
                )
            )

        # Extract AI output
        ai_output = self._get_output_value(outputs)
        if ai_output:
            self._run_async(
                self.harness.add_conversational(
                    thread_id=self.thread_id,
                    role="assistant",
                    content=ai_output,
                )
            )

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> None:
        """
        Async version of save_context.

        Args:
            inputs: Dict containing user input.
            outputs: Dict containing AI output.
        """
        # Extract user input
        user_input = self._get_input_value(inputs)
        if user_input:
            await self.harness.add_conversational(
                thread_id=self.thread_id,
                role="user",
                content=user_input,
            )

        # Extract AI output
        ai_output = self._get_output_value(outputs)
        if ai_output:
            await self.harness.add_conversational(
                thread_id=self.thread_id,
                role="assistant",
                content=ai_output,
            )

    def clear(self) -> None:
        """
        Clear conversation history for this thread.

        Note: This operation requires the harness to support clearing
        conversational memory. If not supported, this is a no-op.
        """
        # Check if harness has a clear method
        if hasattr(self.harness, 'clear_conversational'):
            self._run_async(
                self.harness.clear_conversational(self.thread_id)
            )

    async def aclear(self) -> None:
        """
        Async version of clear.
        """
        if hasattr(self.harness, 'clear_conversational'):
            await self.harness.clear_conversational(self.thread_id)

    def _get_input_value(self, inputs: dict[str, Any]) -> str | None:
        """
        Extract user input from inputs dict.

        Args:
            inputs: The inputs dict.

        Returns:
            The user input string or None if not found.
        """
        if self.input_key:
            return inputs.get(self.input_key)

        # Try common input keys
        for key in ["input", "question", "human_input", "query", "text"]:
            if key in inputs:
                return inputs[key]

        # If only one key, use it
        if len(inputs) == 1:
            return list(inputs.values())[0]

        return None

    def _get_output_value(self, outputs: dict[str, Any]) -> str | None:
        """
        Extract AI output from outputs dict.

        Args:
            outputs: The outputs dict.

        Returns:
            The AI output string or None if not found.
        """
        if self.output_key:
            return outputs.get(self.output_key)

        # Try common output keys
        for key in ["output", "answer", "response", "text", "result"]:
            if key in outputs:
                return outputs[key]

        # If only one key, use it
        if len(outputs) == 1:
            return list(outputs.values())[0]

        return None

    def _to_langchain_messages(
        self, memories: list[MemoryUnit]
    ) -> list[BaseMessage]:
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

    def _format_as_string(self, memories: list[MemoryUnit]) -> str:
        """
        Format memories as a conversation string.

        Args:
            memories: List of MemoryUnit objects from memharness.

        Returns:
            Formatted conversation string.
        """
        lines: list[str] = []

        for mem in memories:
            role = mem.metadata.get("role", "user") if mem.metadata else "user"
            content = mem.content

            if role == "user" or role == "human":
                prefix = self.human_prefix
            elif role == "assistant" or role == "ai":
                prefix = self.ai_prefix
            else:
                prefix = role.capitalize()

            lines.append(f"{prefix}: {content}")

        return "\n".join(lines)

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
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(coro)

        # If we're in a running loop (e.g., Jupyter), use nest_asyncio pattern
        # or create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
