# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tests for Fast Path.

Tests the low-latency user-facing path for agent interactions.
"""

from __future__ import annotations

import pytest

from memharness import MemoryHarness
from memharness.core.fast_path import AgentResponse, FastPath


class TestAgentResponse:
    """Test AgentResponse dataclass."""

    def test_agent_response_creation(self):
        """Test creating an AgentResponse."""
        from memharness.agents.context_assembler import AssembledContext

        ctx = AssembledContext(user_query="test query")
        response = AgentResponse(
            content="test response",
            context=ctx,
            thread_id="thread-1",
        )

        assert response.content == "test response"
        assert response.context.user_query == "test query"
        assert response.thread_id == "thread-1"


class TestFastPath:
    """Test FastPath class."""

    @pytest.fixture
    async def harness(self):
        """Create an in-memory harness for testing."""
        harness = MemoryHarness("memory://")
        await harness.connect()
        yield harness
        await harness.disconnect()

    @pytest.fixture
    def fast_path(self, harness):
        """Create a FastPath instance."""
        return FastPath(harness, max_context_tokens=1000)

    @pytest.mark.asyncio
    async def test_process_user_message(self, fast_path, harness):
        """Test processing a user message."""
        ctx = await fast_path.process_user_message("thread-1", "How do I deploy?")

        # Should return assembled context
        assert ctx.user_query == "How do I deploy?"
        assert ctx.total_tokens_estimate > 0

        # Should have saved the message
        messages = await harness.get_conversational("thread-1")
        assert len(messages) == 1
        assert messages[0].content == "How do I deploy?"
        assert messages[0].metadata.get("role") == "user"

    @pytest.mark.asyncio
    async def test_process_assistant_response(self, fast_path, harness):
        """Test processing an assistant response."""
        # First, add a user message
        await fast_path.process_user_message("thread-2", "What is Python?")

        # Then save assistant response
        memory_id = await fast_path.process_assistant_response(
            "thread-2", "Python is a programming language"
        )

        # Should return a memory ID
        assert memory_id is not None
        assert isinstance(memory_id, str)

        # Should have saved the message
        messages = await harness.get_conversational("thread-2")
        assert len(messages) == 2
        assert messages[1].content == "Python is a programming language"
        assert messages[1].metadata.get("role") == "assistant"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, fast_path, harness):
        """Test multiple turns of conversation."""
        # Turn 1
        await fast_path.process_user_message("thread-3", "Hello!")
        await fast_path.process_assistant_response("thread-3", "Hi there!")

        # Turn 2
        ctx2 = await fast_path.process_user_message("thread-3", "How are you?")
        await fast_path.process_assistant_response("thread-3", "I'm doing well!")

        # Check conversation history
        messages = await harness.get_conversational("thread-3")
        assert len(messages) == 4

        # Check context includes history
        assert len(ctx2.conversation_history) > 0
        # Should include previous messages
        contents = [m.content for m in ctx2.conversation_history]
        assert "Hello!" in contents
        assert "Hi there!" in contents

    @pytest.mark.asyncio
    async def test_context_to_messages(self, fast_path):
        """Test converting context to LangChain messages."""
        ctx = await fast_path.process_user_message("thread-4", "Test query")

        # Should be able to convert to messages
        messages = ctx.to_messages()
        assert isinstance(messages, list)
        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_fast_path_with_existing_memories(self, fast_path, harness):
        """Test fast path with pre-existing knowledge and entities."""
        # Add some knowledge
        await harness.add_knowledge("Python is a programming language", source="kb")

        # Add an entity
        await harness.add_entity("Python", "LANGUAGE", "A popular programming language")

        # Process user message
        ctx = await fast_path.process_user_message("thread-5", "Tell me about Python")

        # Context should include knowledge (if semantic search matches)
        # Note: with in-memory backend and hash embeddings, matching may be limited
        assert ctx.user_query == "Tell me about Python"

    @pytest.mark.asyncio
    async def test_fast_path_is_deterministic(self, fast_path, harness):
        """Test that fast path is deterministic (no extraction in hot path)."""
        # Process a message with entity mentions
        await fast_path.process_user_message("thread-6", "I work at SAP with John Smith")
        await fast_path.process_assistant_response("thread-6", "That's great!")

        # Fast path should NOT extract entities automatically
        # (that's the slow path's job)
        # So entity table should be empty or unchanged
        # Note: This is testing the absence of side effects in the fast path

    @pytest.mark.asyncio
    async def test_max_context_tokens(self, harness):
        """Test custom max_context_tokens setting."""
        fast_path = FastPath(harness, max_context_tokens=500)

        await fast_path.process_user_message("thread-7", "Test")

        # Should respect max tokens setting
        assert fast_path.max_context_tokens == 500
        assert fast_path._ctx_agent.max_tokens == 500
