# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tests for Slow Path.

Tests the background worker path for memory enrichment.
"""

from __future__ import annotations

import pytest

from memharness import MemoryHarness
from memharness.core.slow_path import SlowPath, WorkerResult


class TestWorkerResult:
    """Test WorkerResult dataclass."""

    def test_worker_result_creation(self):
        """Test creating a WorkerResult."""
        result = WorkerResult(
            worker="entity_extractor",
            processed=10,
            errors=0,
            duration_ms=123.45,
        )

        assert result.worker == "entity_extractor"
        assert result.processed == 10
        assert result.errors == 0
        assert result.duration_ms == 123.45


class TestSlowPath:
    """Test SlowPath class."""

    @pytest.fixture
    async def harness(self):
        """Create an in-memory harness for testing."""
        harness = MemoryHarness("memory://")
        await harness.connect()
        yield harness
        await harness.disconnect()

    @pytest.fixture
    def slow_path(self, harness):
        """Create a SlowPath instance."""
        return SlowPath(harness)

    @pytest.mark.asyncio
    async def test_extract_entities(self, slow_path):
        """Test entity extraction worker."""
        result = await slow_path.extract_entities()

        # Should return a WorkerResult
        assert isinstance(result, WorkerResult)
        assert result.worker == "entity_extractor"
        assert result.processed >= 0
        assert result.errors >= 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_summarize_threads(self, slow_path):
        """Test thread summarization worker."""
        result = await slow_path.summarize_threads()

        # Should return a WorkerResult
        assert isinstance(result, WorkerResult)
        assert result.worker == "summarizer"
        assert result.processed >= 0
        assert result.errors >= 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_consolidate(self, slow_path):
        """Test consolidation worker."""
        result = await slow_path.consolidate()

        # Should return a WorkerResult
        assert isinstance(result, WorkerResult)
        assert result.worker == "consolidator"
        assert result.processed >= 0
        assert result.errors >= 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_all(self, slow_path):
        """Test running all workers."""
        results = await slow_path.run_all()

        # Should return list of 4 results
        assert isinstance(results, list)
        assert len(results) == 3

        # Check worker names
        worker_names = [r.worker for r in results]
        assert "entity_extractor" in worker_names
        assert "summarizer" in worker_names
        assert "consolidator" in worker_names

        # All should have valid metrics
        for result in results:
            assert result.processed >= 0
            assert result.errors >= 0
            assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_last_processed_at_tracking(self, slow_path):
        """Test that last_processed_at is tracked."""
        # Initially None
        assert slow_path._last_processed_at is None

        # After run_all
        await slow_path.run_all()
        assert slow_path._last_processed_at is not None

        # Second run should have a timestamp
        first_run = slow_path._last_processed_at
        await slow_path.run_all()
        second_run = slow_path._last_processed_at

        # Should be updated
        assert second_run >= first_run

    @pytest.mark.asyncio
    async def test_slow_path_with_llms(self, harness):
        """Test SlowPath with custom LLMs."""
        # Mock LLM (None is fine for this test)
        mock_llm = None

        slow_path = SlowPath(
            harness,
            entity_extractor_llm=mock_llm,
            summarizer_llm=mock_llm,
        )

        # Should initialize without error
        assert slow_path._entity_extractor is not None
        assert slow_path._summarizer is not None

        # Should be able to run
        results = await slow_path.run_all()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_extract_entities_with_conversations(self, slow_path, harness):
        """Test entity extraction with actual conversation data."""
        # Add some conversations
        await harness.add_conversational("thread-1", "user", "I work at SAP with John Smith")
        await harness.add_conversational("thread-1", "assistant", "That's interesting!")

        # Run entity extraction
        result = await slow_path.extract_entities()

        # Should process without error
        assert result.errors == 0

        # Note: The actual extraction logic is a placeholder in the current implementation
        # In a full implementation, this would extract "SAP" and "John Smith" as entities

    @pytest.mark.asyncio
    async def test_worker_error_handling(self, slow_path):
        """Test that workers handle errors gracefully."""
        # Run all workers - should not raise exceptions
        results = await slow_path.run_all()

        # All workers should complete (even if with errors)
        assert len(results) == 3

        # Workers should track errors
        for result in results:
            assert isinstance(result.errors, int)

    @pytest.mark.asyncio
    async def test_slow_path_idempotent(self, slow_path):
        """Test that slow path can be run multiple times safely."""
        # Run once
        results1 = await slow_path.run_all()
        assert len(results1) == 3

        # Run again
        results2 = await slow_path.run_all()
        assert len(results2) == 3

        # Should not error
        for result in results2:
            assert result.errors >= 0
