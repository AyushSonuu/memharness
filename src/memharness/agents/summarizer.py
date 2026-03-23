# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Summarizer agent for conversation summarization.

Generates concise summaries of conversation threads using LLM or heuristics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from memharness.core.harness import MemoryHarness


class SummarizerAgent:
    """
    Agent that summarizes conversation threads.

    Works in two modes:
    1. Without LLM: Uses heuristic summarization (first/last messages + count)
    2. With LLM: Generates intelligent summaries using LangChain

    Example:
        ```python
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("gpt-4o")
        agent = SummarizerAgent(harness, llm=llm)
        summary = await agent.summarize_thread("thread1", max_messages=50)
        ```
    """

    def __init__(self, harness: MemoryHarness, llm: BaseChatModel | None = None) -> None:
        """
        Initialize the summarizer agent.

        Args:
            harness: The MemoryHarness instance to operate on.
            llm: Optional LLM for intelligent summarization.
        """
        self.harness = harness
        self.llm = llm

    async def summarize_thread(self, thread_id: str, max_messages: int = 50) -> str:
        """
        Summarize a conversation thread.

        Args:
            thread_id: The thread ID to summarize.
            max_messages: Maximum number of recent messages to include.

        Returns:
            A summary string.
        """
        messages = await self.harness.get_conversational(thread_id, limit=max_messages)
        if not messages:
            return ""

        if not self.llm:
            return self._heuristic_summary(messages)

        return await self._llm_summary(messages)

    def _heuristic_summary(self, messages: list[Any]) -> str:
        """
        Create a simple heuristic summary.

        Args:
            messages: List of MemoryUnit objects.

        Returns:
            A heuristic summary string.
        """
        if not messages:
            return "No messages"

        total = len(messages)
        first_msg = messages[0].content[:100]
        last_msg = messages[-1].content[:100]

        return (
            f"Conversation with {total} message(s). "
            f"Started with: '{first_msg}...' "
            f"Latest: '{last_msg}...'"
        )

    async def _llm_summary(self, messages: list[Any]) -> str:
        """
        Generate an intelligent summary using LangChain.

        Args:
            messages: List of MemoryUnit objects.

        Returns:
            An LLM-generated summary.
        """
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError as e:
            raise ImportError("Install langchain-core: pip install memharness[langchain]") from e

        # Build conversation text
        conversation = "\n".join(f"{m.metadata.get('role', 'user')}: {m.content}" for m in messages)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a conversation summarizer. Summarize the "
                    "following conversation concisely in 2-3 sentences, "
                    "capturing the main topics and outcomes.",
                ),
                ("user", "{conversation}"),
            ]
        )

        # Build chain
        chain = prompt | self.llm | StrOutputParser()

        # Generate summary
        return await chain.ainvoke({"conversation": conversation})

    async def run(self, thread_id: str, max_messages: int = 50, **kwargs: Any) -> dict[str, Any]:
        """
        Run the summarizer agent.

        Args:
            thread_id: The thread ID to summarize.
            max_messages: Maximum number of messages to process.
            **kwargs: Additional arguments (ignored).

        Returns:
            Dictionary with 'summary' and 'message_count' keys.
        """
        messages = await self.harness.get_conversational(thread_id, limit=max_messages)
        summary = await self.summarize_thread(thread_id, max_messages)

        return {"summary": summary, "message_count": len(messages)}
