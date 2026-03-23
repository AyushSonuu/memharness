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

    async def summarize_thread(self, thread_id: str, max_messages: int = 50) -> dict[str, Any]:
        """
        Summarize a conversation thread and mark source messages.

        This method implements the full summarization pipeline from L05:
        1. Get unsummarized messages for the thread
        2. Create summary (heuristic or LLM)
        3. Store summary via harness.add_summary()
        4. Mark original messages with summary_id in their metadata
        5. Return summarization result with summary_id

        Args:
            thread_id: The thread ID to summarize.
            max_messages: Maximum number of recent messages to include.

        Returns:
            Dictionary with:
            - 'summarized': bool indicating if summarization occurred
            - 'summary_id': str ID of the created summary (if summarized)
            - 'summary_text': str summary text (if summarized)
            - 'messages_summarized': int count of messages processed
            - 'reason': str explanation if not summarized

        Example:
            ```python
            result = await summarizer.summarize_thread("chat-123", max_messages=50)
            if result['summarized']:
                print(f"Summarized {result['messages_summarized']} messages")
                print(f"Summary ID: {result['summary_id']}")
            ```
        """
        # 1. Get unsummarized messages
        messages = await self.harness.get_conversational(thread_id, limit=max_messages)

        if len(messages) < 10:
            return {
                "summarized": False,
                "reason": "too_few_messages",
                "messages_summarized": len(messages),
            }

        # 2. Create summary text
        if not self.llm:
            summary_text = self._heuristic_summary(messages)
        else:
            summary_text = await self._llm_summary(messages)

        # 3. Store summary via harness.add_summary()
        source_ids = [m.id for m in messages]
        summary_id = await self.harness.add_summary(
            summary=summary_text, source_ids=source_ids, thread_id=thread_id
        )

        # 4. Mark original messages with summary_id
        for msg in messages:
            msg.metadata["summary_id"] = summary_id
            await self.harness._backend.update(msg.id, {"metadata": msg.metadata})

        # 5. Return result
        return {
            "summarized": True,
            "summary_id": summary_id,
            "summary_text": summary_text,
            "messages_summarized": len(messages),
        }

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
            Dictionary with summarization results (see summarize_thread docstring).
        """
        return await self.summarize_thread(thread_id, max_messages)
