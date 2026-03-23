# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Tests for memory agents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memharness.agents.consolidator import ConsolidatorAgent
from memharness.agents.entity_extractor import EntityExtractorAgent
from memharness.agents.summarizer import SummarizerAgent
from memharness.types import MemoryType, MemoryUnit


@pytest.fixture
def mock_harness():
    """Create a mock MemoryHarness for testing."""
    harness = MagicMock()
    harness.get_conversational = AsyncMock(return_value=[])
    harness.add_summary = AsyncMock(return_value="summary-123")
    harness._backend = MagicMock()
    harness._backend.update = AsyncMock(return_value=True)
    return harness


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        MemoryUnit(
            id="msg1",
            memory_type=MemoryType.CONVERSATIONAL,
            content="Hello, how can I help you?",
            embedding=[0.1] * 384,
            metadata={"role": "assistant"},
        ),
        MemoryUnit(
            id="msg2",
            memory_type=MemoryType.CONVERSATIONAL,
            content="I need help with Python async programming",
            embedding=[0.2] * 384,
            metadata={"role": "user"},
        ),
        MemoryUnit(
            id="msg3",
            memory_type=MemoryType.CONVERSATIONAL,
            content="Sure! Async programming in Python uses asyncio...",
            embedding=[0.3] * 384,
            metadata={"role": "assistant"},
        ),
    ]


class TestSummarizerAgent:
    """Tests for SummarizerAgent."""

    async def test_heuristic_summary(self, mock_harness):
        """Test summarizer without LLM (heuristic mode) with enough messages."""
        from memharness.types import MemoryType, MemoryUnit

        # Create 12 messages (> 10 threshold)
        messages = [
            MemoryUnit(
                id=f"msg{i}",
                memory_type=MemoryType.CONVERSATIONAL,
                content=f"Message {i}",
                embedding=[0.1 * i] * 384,
                metadata={"role": "user" if i % 2 == 0 else "assistant"},
            )
            for i in range(12)
        ]

        mock_harness.get_conversational.return_value = messages
        agent = SummarizerAgent(mock_harness)

        result = await agent.summarize_thread("thread1", max_messages=50)

        # Now returns a dict with summarization results
        assert result["summarized"] is True
        assert "12 message(s)" in result["summary_text"]
        assert "Message 0" in result["summary_text"]
        assert result["messages_summarized"] == 12
        assert result["summary_id"] == "summary-123"
        assert mock_harness.get_conversational.called
        assert mock_harness.add_summary.called
        assert mock_harness._backend.update.called

    async def test_empty_thread(self, mock_harness):
        """Test summarizer with empty thread."""
        mock_harness.get_conversational.return_value = []
        agent = SummarizerAgent(mock_harness)

        result = await agent.summarize_thread("thread1")

        # With 0 messages, should return not summarized
        assert result["summarized"] is False
        assert result["reason"] == "too_few_messages"

    async def test_run_method(self, mock_harness):
        """Test the run method with enough messages."""
        from memharness.types import MemoryType, MemoryUnit

        # Create 10 messages (exactly at threshold)
        messages = [
            MemoryUnit(
                id=f"msg{i}",
                memory_type=MemoryType.CONVERSATIONAL,
                content=f"Message {i}",
                embedding=[0.1 * i] * 384,
                metadata={"role": "user" if i % 2 == 0 else "assistant"},
            )
            for i in range(10)
        ]

        mock_harness.get_conversational.return_value = messages
        agent = SummarizerAgent(mock_harness)

        result = await agent.run("thread1", max_messages=50)

        # run() now delegates to summarize_thread(), returns dict
        assert result["summarized"] is True
        assert result["messages_summarized"] == 10
        assert result["summary_id"] == "summary-123"


class TestEntityExtractorAgent:
    """Tests for EntityExtractorAgent."""

    async def test_heuristic_extraction(self, mock_harness):
        """Test entity extractor without LLM (heuristic mode)."""
        agent = EntityExtractorAgent(mock_harness)

        text = "John Smith works at Acme Corp in New York. Contact: john@example.com"
        entities = await agent.extract_entities(text)

        assert "people" in entities
        assert "organizations" in entities
        assert "locations" in entities
        assert isinstance(entities["people"], list)

    async def test_empty_text(self, mock_harness):
        """Test entity extractor with empty text."""
        agent = EntityExtractorAgent(mock_harness)

        entities = await agent.extract_entities("")

        assert entities == {"people": [], "organizations": [], "locations": []}

    async def test_mentions_extraction(self, mock_harness):
        """Test extraction of @mentions."""
        agent = EntityExtractorAgent(mock_harness)

        text = "Hey @alice and @bob, check this out!"
        entities = await agent.extract_entities(text)

        assert "alice" in entities["people"] or "bob" in entities["people"]

    async def test_run_method(self, mock_harness):
        """Test the run method."""
        agent = EntityExtractorAgent(mock_harness)

        text = "John works at Acme"
        result = await agent.run(text)

        assert "entities" in result
        assert "total_extracted" in result
        assert isinstance(result["total_extracted"], int)


class TestConsolidatorAgent:
    """Tests for ConsolidatorAgent."""

    async def test_consolidate_memories(self, mock_harness):
        """Test memory consolidation."""
        agent = ConsolidatorAgent(mock_harness, threshold=0.85)

        result = await agent.consolidate_memories(min_memories=5)

        assert "merged" in result
        assert "deleted" in result
        assert isinstance(result["merged"], int)
        assert isinstance(result["deleted"], int)

    async def test_cosine_similarity(self, mock_harness):
        """Test cosine similarity calculation."""
        agent = ConsolidatorAgent(mock_harness)

        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]

        # Identical embeddings should have similarity ~1.0
        sim_identical = agent._cosine_similarity(emb1, emb2)
        assert sim_identical > 0.99

        # Orthogonal embeddings should have similarity ~0.0
        sim_orthogonal = agent._cosine_similarity(emb1, emb3)
        assert abs(sim_orthogonal) < 0.01

    async def test_run_method(self, mock_harness):
        """Test the run method."""
        agent = ConsolidatorAgent(mock_harness, threshold=0.9)

        result = await agent.run(min_memories=10)

        assert "merged" in result
        assert "deleted" in result
        assert "threshold" in result
        assert result["threshold"] == 0.9
