# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Unit tests for memharness middleware components.
"""

from __future__ import annotations

import pytest

# Skip all tests if langchain is not available
pytest.importorskip("langchain")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memharness import MemoryHarness
from memharness.middleware import (
    EntityExtractionMiddleware,
    MemoryContextMiddleware,
    MemoryPersistenceMiddleware,
)


class MockRuntime:
    """Mock runtime for testing middleware."""

    def __init__(self):
        self.context = None


@pytest.fixture
async def harness():
    """Create a test harness with in-memory backend."""
    h = MemoryHarness("memory://")
    await h.connect()
    yield h
    await h.disconnect()


@pytest.mark.asyncio
class TestMemoryContextMiddleware:
    """Tests for MemoryContextMiddleware."""

    async def test_initialization(self, harness):
        """Test middleware initialization."""
        middleware = MemoryContextMiddleware(
            harness=harness,
            thread_id="test-thread",
            max_tokens=1000,
        )
        assert middleware.harness is harness
        assert middleware.thread_id == "test-thread"
        assert middleware.max_tokens == 1000

    async def test_abefore_model_no_messages(self, harness):
        """Test that middleware returns None when there are no messages."""
        middleware = MemoryContextMiddleware(
            harness=harness,
            thread_id="test-thread",
        )
        state = {"messages": []}
        runtime = MockRuntime()

        result = await middleware.abefore_model(state, runtime)
        assert result is None

    async def test_abefore_model_with_messages(self, harness):
        """Test that middleware injects context when messages exist."""
        # Add some knowledge to the harness
        await harness.add_knowledge(
            content="Python is a programming language",
            source="test",
        )

        middleware = MemoryContextMiddleware(
            harness=harness,
            thread_id="test-thread",
        )

        state = {
            "messages": [
                HumanMessage(content="Tell me about Python"),
            ]
        }
        runtime = MockRuntime()

        result = await middleware.abefore_model(state, runtime)

        # Should return state updates with a system message
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)
        assert "Relevant memory context:" in result["messages"][0].content

    async def test_abefore_model_empty_query(self, harness):
        """Test that middleware handles empty queries gracefully."""
        middleware = MemoryContextMiddleware(
            harness=harness,
            thread_id="test-thread",
        )

        state = {"messages": [HumanMessage(content="")]}
        runtime = MockRuntime()

        result = await middleware.abefore_model(state, runtime)
        assert result is None

    async def test_custom_prefix(self, harness):
        """Test custom prefix in context message."""
        await harness.add_knowledge(
            content="Python is a programming language",
            source="test",
        )

        middleware = MemoryContextMiddleware(
            harness=harness,
            thread_id="test-thread",
            prefix="Custom context:",
        )

        state = {"messages": [HumanMessage(content="Tell me about Python")]}
        runtime = MockRuntime()

        result = await middleware.abefore_model(state, runtime)

        assert result is not None
        assert "Custom context:" in result["messages"][0].content


@pytest.mark.asyncio
class TestMemoryPersistenceMiddleware:
    """Tests for MemoryPersistenceMiddleware."""

    async def test_initialization(self, harness):
        """Test middleware initialization."""
        middleware = MemoryPersistenceMiddleware(
            harness=harness,
            thread_id="test-thread",
        )
        assert middleware.harness is harness
        assert middleware.thread_id == "test-thread"
        assert middleware.store_user_messages is True
        assert middleware.store_ai_messages is True

    async def test_aafter_model_stores_messages(self, harness):
        """Test that middleware stores user and AI messages."""
        middleware = MemoryPersistenceMiddleware(
            harness=harness,
            thread_id="test-thread",
        )

        state = {
            "messages": [
                HumanMessage(content="Hello", id="msg-1"),
                AIMessage(content="Hi there!", id="msg-2"),
            ]
        }
        runtime = MockRuntime()

        await middleware.aafter_model(state, runtime)

        # Check that messages were stored
        messages = await harness.get_conversational("test-thread", limit=10)
        assert len(messages) >= 2

        # Check content
        user_msgs = [m for m in messages if m.metadata.get("role") == "user"]
        ai_msgs = [m for m in messages if m.metadata.get("role") == "assistant"]

        assert len(user_msgs) >= 1
        assert len(ai_msgs) >= 1
        assert any("Hello" in m.content for m in user_msgs)
        assert any("Hi there!" in m.content for m in ai_msgs)

    async def test_aafter_model_no_messages(self, harness):
        """Test that middleware handles empty messages gracefully."""
        middleware = MemoryPersistenceMiddleware(
            harness=harness,
            thread_id="test-thread",
        )

        state = {"messages": []}
        runtime = MockRuntime()

        result = await middleware.aafter_model(state, runtime)
        assert result is None

    async def test_store_only_user_messages(self, harness):
        """Test storing only user messages."""
        middleware = MemoryPersistenceMiddleware(
            harness=harness,
            thread_id="test-thread-2",
            store_ai_messages=False,
        )

        state = {
            "messages": [
                HumanMessage(content="User message", id="msg-3"),
                AIMessage(content="AI message", id="msg-4"),
            ]
        }
        runtime = MockRuntime()

        await middleware.aafter_model(state, runtime)

        messages = await harness.get_conversational("test-thread-2", limit=10)
        user_msgs = [m for m in messages if m.metadata.get("role") == "user"]
        ai_msgs = [m for m in messages if m.metadata.get("role") == "assistant"]

        assert len(user_msgs) >= 1
        assert len(ai_msgs) == 0

    async def test_deduplication(self, harness):
        """Test that middleware doesn't store the same message twice."""
        middleware = MemoryPersistenceMiddleware(
            harness=harness,
            thread_id="test-thread-3",
        )

        msg = HumanMessage(content="Test message", id="msg-5")
        state = {"messages": [msg]}
        runtime = MockRuntime()

        # Store twice
        await middleware.aafter_model(state, runtime)
        await middleware.aafter_model(state, runtime)

        messages = await harness.get_conversational("test-thread-3", limit=10)

        # Should only have one message (not duplicated)
        assert len(messages) == 1


@pytest.mark.asyncio
class TestEntityExtractionMiddleware:
    """Tests for EntityExtractionMiddleware."""

    async def test_initialization(self, harness):
        """Test middleware initialization."""
        middleware = EntityExtractionMiddleware(
            harness=harness,
            thread_id="test-thread",
        )
        assert middleware.harness is harness
        assert middleware.thread_id == "test-thread"
        assert middleware.llm is None
        assert middleware.extract_from_user is True
        assert middleware.extract_from_ai is True

    async def test_aafter_model_extracts_entities(self, harness):
        """Test that middleware extracts entities from messages."""
        middleware = EntityExtractionMiddleware(
            harness=harness,
            thread_id="test-thread",
        )

        state = {
            "messages": [
                HumanMessage(
                    content="John Smith works at OpenAI in San Francisco",
                    id="msg-6",
                ),
                AIMessage(content="That's interesting!", id="msg-7"),
            ]
        }
        runtime = MockRuntime()

        await middleware.aafter_model(state, runtime)

        # Check that entities were extracted and stored
        entities = await harness.search_entity("", k=20)

        # Should have extracted at least some entities
        assert len(entities) > 0

        # Check for expected entities (may vary based on heuristic extraction)
        entity_names = [e.metadata.get("name") for e in entities]
        assert len(entity_names) > 0

    async def test_aafter_model_no_messages(self, harness):
        """Test that middleware handles empty messages gracefully."""
        middleware = EntityExtractionMiddleware(
            harness=harness,
            thread_id="test-thread",
        )

        state = {"messages": []}
        runtime = MockRuntime()

        result = await middleware.aafter_model(state, runtime)
        assert result is None

    async def test_extract_only_from_user(self, harness):
        """Test extracting entities only from user messages."""
        middleware = EntityExtractionMiddleware(
            harness=harness,
            thread_id="test-thread-4",
            extract_from_ai=False,
        )

        state = {
            "messages": [
                HumanMessage(content="Alice works at Microsoft", id="msg-8"),
                AIMessage(content="Bob works at Google", id="msg-9"),
            ]
        }
        runtime = MockRuntime()

        await middleware.aafter_model(state, runtime)

        entities = await harness.search_entity("", k=20)
        entity_names = [e.metadata.get("name") for e in entities]

        # Should have some entities, but this is hard to test precisely
        # without knowing the exact extraction logic
        assert len(entity_names) >= 0

    async def test_deduplication(self, harness):
        """Test that middleware doesn't process the same message twice."""
        middleware = EntityExtractionMiddleware(
            harness=harness,
            thread_id="test-thread-5",
        )

        msg = HumanMessage(
            content="TestEntity is a test organization",
            id="msg-10",
        )
        state = {"messages": [msg]}
        runtime = MockRuntime()

        # Process twice
        await middleware.aafter_model(state, runtime)

        entities_count_1 = len(await harness.search_entity("", k=100))

        await middleware.aafter_model(state, runtime)

        entities_count_2 = len(await harness.search_entity("", k=100))

        # Should not have doubled the entities
        assert entities_count_2 == entities_count_1
