# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Tests for memory agents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memharness.agents.consolidator import ConsolidatorAgent
from memharness.agents.entity_extractor import EntityExtractorAgent
from memharness.agents.gc import GCAgent
from memharness.agents.summarizer import SummarizerAgent
from memharness.types import MemoryType, MemoryUnit


@pytest.fixture
def mock_harness():
    """Create a mock MemoryHarness for testing."""
    harness = MagicMock()
    harness.get_conversational = AsyncMock(return_value=[])
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

    async def test_heuristic_summary(self, mock_harness, sample_messages):
        """Test summarizer without LLM (heuristic mode)."""
        mock_harness.get_conversational.return_value = sample_messages
        agent = SummarizerAgent(mock_harness)

        summary = await agent.summarize_thread("thread1", max_messages=50)

        assert "3 message(s)" in summary
        assert "Hello" in summary
        assert mock_harness.get_conversational.called

    async def test_empty_thread(self, mock_harness):
        """Test summarizer with empty thread."""
        mock_harness.get_conversational.return_value = []
        agent = SummarizerAgent(mock_harness)

        summary = await agent.summarize_thread("thread1")

        assert summary == ""

    async def test_run_method(self, mock_harness, sample_messages):
        """Test the run method."""
        mock_harness.get_conversational.return_value = sample_messages
        agent = SummarizerAgent(mock_harness)

        result = await agent.run("thread1", max_messages=50)

        assert "summary" in result
        assert "message_count" in result
        assert result["message_count"] == 3


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


class TestGCAgent:
    """Tests for GCAgent."""

    async def test_archive_old_memories(self, mock_harness):
        """Test archiving old memories."""
        agent = GCAgent(mock_harness, archive_after_days=90)

        archived = await agent.archive_old_memories()

        assert isinstance(archived, int)
        assert archived >= 0

    async def test_delete_old_memories(self, mock_harness):
        """Test deleting old memories."""
        agent = GCAgent(mock_harness, delete_after_days=365)

        deleted = await agent.delete_old_memories()

        assert isinstance(deleted, int)
        assert deleted >= 0

    async def test_run_method(self, mock_harness):
        """Test the run method."""
        agent = GCAgent(mock_harness, archive_after_days=60, delete_after_days=180)

        result = await agent.run()

        assert "archived" in result
        assert "deleted" in result
        assert "archive_after_days" in result
        assert "delete_after_days" in result
        assert result["archive_after_days"] == 60
        assert result["delete_after_days"] == 180

    async def test_llm_not_used(self, mock_harness):
        """Test that GC agent works without LLM."""
        mock_llm = MagicMock()
        agent = GCAgent(mock_harness, llm=mock_llm)

        await agent.run()

        # LLM should not be called for GC agent
        assert not mock_llm.called
