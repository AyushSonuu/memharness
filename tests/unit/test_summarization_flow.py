# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Test suite for summarization flow aligned with agent memory course L05.

Tests verify:
- get_conversational() excludes messages with summary_id
- SummarizerAgent marks messages with summary_id after summarization
- ContextAssemblyAgent loads summaries + unsummarized messages
- expand_summary() retrieves original messages (lossless compaction)
"""

from __future__ import annotations

import pytest

from memharness import MemoryHarness
from memharness.agents.context_assembler import ContextAssemblyAgent
from memharness.agents.summarizer import SummarizerAgent


class TestSummarizationFlow:
    """Test the full summarization flow from L05."""

    @pytest.fixture
    async def harness(self):
        """Create an in-memory harness for testing."""
        h = MemoryHarness("memory://")
        await h.connect()
        yield h
        await h.disconnect()

    @pytest.fixture
    async def populated_thread(self, harness):
        """Create a thread with 15 messages for testing."""
        thread_id = "test-thread-1"
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            await harness.add_conversational(
                thread_id=thread_id, role=role, content=f"Message {i + 1}"
            )
        return thread_id

    @pytest.mark.asyncio
    async def test_get_conversational_excludes_summarized(self, harness, populated_thread):
        """
        Test that get_conversational() excludes messages with summary_id set.

        Expected behavior (L05):
        - BEFORE summarization: get_conversational() returns all messages
        - AFTER summarization: get_conversational() returns only unsummarized messages
        """
        thread_id = populated_thread

        # BEFORE: All messages returned
        messages_before = await harness.get_conversational(thread_id)
        assert len(messages_before) == 15

        # Summarize the thread (without LLM, uses heuristic)
        summarizer = SummarizerAgent(harness, llm=None)
        result = await summarizer.summarize_thread(thread_id, max_messages=15)

        assert result["summarized"] is True
        assert result["messages_summarized"] == 15
        summary_id = result["summary_id"]
        assert summary_id is not None

        # AFTER: Only unsummarized messages returned (should be empty)
        messages_after = await harness.get_conversational(thread_id)
        assert len(messages_after) == 0

        # Verify the summary exists
        summary = await harness._backend.get(summary_id)
        assert summary is not None
        assert "Message 1" in summary.content or "15 message(s)" in summary.content

    @pytest.mark.asyncio
    async def test_summarizer_marks_messages(self, harness, populated_thread):
        """
        Test that SummarizerAgent correctly marks messages with summary_id.

        Expected behavior (L05 step 4):
        - After summarization, each original message has metadata['summary_id'] set
        """
        thread_id = populated_thread

        # Get messages before summarization
        messages_before = await harness.get_conversational(thread_id)
        original_ids = [m.id for m in messages_before]

        # Summarize
        summarizer = SummarizerAgent(harness, llm=None)
        result = await summarizer.summarize_thread(thread_id, max_messages=15)
        summary_id = result["summary_id"]

        # Verify each original message is marked
        for msg_id in original_ids:
            msg = await harness._backend.get(msg_id)
            assert msg is not None
            assert msg.metadata.get("summary_id") == summary_id

    @pytest.mark.asyncio
    async def test_context_assembly_with_summaries(self, harness, populated_thread):
        """
        Test that ContextAssemblyAgent loads summaries first + unsummarized messages.

        Expected behavior (L05/L06):
        - AssembledContext.summaries contains summary text
        - AssembledContext.conversation_history contains only unsummarized messages
        - to_messages() renders summaries as SystemMessage before conversation
        """
        thread_id = populated_thread

        # Summarize first 10 messages
        summarizer = SummarizerAgent(harness, llm=None)
        await summarizer.summarize_thread(thread_id, max_messages=10)

        # Add 5 new messages (unsummarized)
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            await harness.add_conversational(
                thread_id=thread_id, role=role, content=f"New message {i + 1}"
            )

        # Assemble context
        assembler = ContextAssemblyAgent(harness, max_tokens=4000)
        ctx = await assembler.assemble(query="Test query", thread_id=thread_id)

        # Verify summaries are present
        assert ctx.summaries != ""
        assert "[Summary ID:" in ctx.summaries

        # Verify conversation_history contains the remaining + new unsummarized messages
        # populated_thread has 15 messages total
        # summarize_thread with max_messages=10 gets first 10 and summarizes them
        # Remaining from original 15: 5 messages (messages 11-15)
        # Add 5 new messages
        # Total unsummarized: 10
        assert len(ctx.conversation_history) == 10

        # Verify to_messages() renders summaries as SystemMessage
        messages = ctx.to_messages()

        # Find the summary system message
        summary_msg = None
        for msg in messages:
            if hasattr(msg, "content") and "[Summary of earlier conversation]" in msg.content:
                summary_msg = msg
                break

        assert summary_msg is not None
        try:
            from langchain_core.messages import SystemMessage

            assert isinstance(summary_msg, SystemMessage)
        except ImportError:
            # If langchain not available, check dict format
            assert summary_msg["role"] == "system"

    @pytest.mark.asyncio
    async def test_expand_summary_retrieves_originals(self, harness, populated_thread):
        """
        Test that expand_summary() retrieves original messages (lossless).

        Expected behavior (L05 compaction):
        - Summary is stored with source_ids
        - expand_summary(id) returns all original MemoryUnits
        - Compaction is lossless — full content retrievable
        """
        thread_id = populated_thread

        # Get messages before summarization
        messages_before = await harness.get_conversational(thread_id)
        original_count = len(messages_before)
        original_contents = [m.content for m in messages_before]

        # Summarize
        summarizer = SummarizerAgent(harness, llm=None)
        result = await summarizer.summarize_thread(thread_id, max_messages=15)
        summary_id = result["summary_id"]

        # Expand the summary
        expanded = await harness.expand_summary(summary_id)

        # Verify all original messages are retrievable
        assert len(expanded) == original_count
        expanded_contents = [m.content for m in expanded]
        assert set(expanded_contents) == set(original_contents)

    @pytest.mark.asyncio
    async def test_partial_summarization(self, harness):
        """
        Test that partial summarization works correctly.

        Scenario:
        - 20 messages in thread
        - Summarize first 10
        - Add 5 new messages
        - get_conversational() returns only the 5 new + remaining 10 unsummarized
        """
        thread_id = "partial-thread"

        # Add 20 messages
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            await harness.add_conversational(
                thread_id=thread_id, role=role, content=f"Message {i + 1}"
            )

        # Get all before summarization
        all_messages = await harness.get_conversational(thread_id)
        assert len(all_messages) == 20

        # Summarize first 10 only
        messages_to_summarize = all_messages[:10]

        # Manually create summary for first 10
        source_ids = [m.id for m in messages_to_summarize]
        summary_text = "Summary of messages 1-10"
        summary_id = await harness.add_summary(
            summary=summary_text, source_ids=source_ids, thread_id=thread_id
        )

        # Mark the first 10 messages
        for msg in messages_to_summarize:
            msg.metadata["summary_id"] = summary_id
            await harness._backend.update(msg.id, {"metadata": msg.metadata})

        # get_conversational() should return only remaining 10
        unsummarized = await harness.get_conversational(thread_id)
        assert len(unsummarized) == 10
        assert unsummarized[0].content == "Message 11"

        # Add 5 new messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            await harness.add_conversational(
                thread_id=thread_id, role=role, content=f"New message {i + 1}"
            )

        # get_conversational() should return 10 + 5 = 15
        unsummarized_after = await harness.get_conversational(thread_id)
        assert len(unsummarized_after) == 15

    @pytest.mark.asyncio
    async def test_too_few_messages_not_summarized(self, harness):
        """
        Test that summarizer skips threads with too few messages.

        Expected: If < 10 messages, return {'summarized': False, 'reason': 'too_few_messages'}
        """
        thread_id = "small-thread"

        # Add only 5 messages
        for i in range(5):
            await harness.add_conversational(
                thread_id=thread_id, role="user", content=f"Message {i + 1}"
            )

        # Attempt to summarize
        summarizer = SummarizerAgent(harness, llm=None)
        result = await summarizer.summarize_thread(thread_id, max_messages=50)

        # Should NOT summarize
        assert result["summarized"] is False
        assert result["reason"] == "too_few_messages"
        assert result["messages_summarized"] == 5

        # Verify no summary was created
        summaries = await harness.get_summaries_by_thread(thread_id)
        assert len(summaries) == 0
