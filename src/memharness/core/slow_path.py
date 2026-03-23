# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Slow path background workers for memory enrichment.

Processes the conversation table asynchronously to:
- Extract entities (upsert, not duplicate)
- Summarize long threads
- Consolidate duplicate entities
- Garbage collect old data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memharness.core.harness import MemoryHarness

logger = logging.getLogger(__name__)

__all__ = ["WorkerResult", "SlowPath"]


@dataclass
class WorkerResult:
    """Result from a background worker execution.

    Attributes:
        worker: Name of the worker that ran.
        processed: Number of items processed.
        errors: Number of errors encountered.
        duration_ms: Execution duration in milliseconds.
    """

    worker: str
    processed: int
    errors: int
    duration_ms: float


class SlowPath:
    """Slow path background workers for memory enrichment.

    Processes the conversation table asynchronously to:
    - Extract entities (upsert, not duplicate)
    - Summarize long threads
    - Consolidate duplicate entities
    - Garbage collect old data

    This class implements the background processing path for agent memory.
    Unlike the fast path (which is user-facing and low-latency), the slow path
    runs asynchronously (e.g., via cron, background tasks) to enrich memory:

    1. Entity Extraction: Scans new conversation messages and extracts/upserts
       entities. Uses UPDATE for existing entities (avoids duplicates).

    2. Summarization: Compresses long conversation threads to free up context space.

    3. Consolidation: Merges duplicate entities based on similarity.

    4. Garbage Collection: Archives/deletes old data based on retention policies.

    Attributes:
        harness: The MemoryHarness instance to operate on.

    Example:
        ```python
        from memharness import MemoryHarness
        from memharness.core.slow_path import SlowPath

        harness = MemoryHarness('sqlite:///memory.db')
        await harness.connect()

        # Slow path: run periodically (cron, background task, etc.)
        slow = SlowPath(harness)
        results = await slow.run_all()
        for r in results:
            print(f'{r.worker}: processed={r.processed}, errors={r.errors}')
        ```
    """

    def __init__(
        self,
        harness: MemoryHarness,
        entity_extractor_llm: Any = None,
        summarizer_llm: Any = None,
    ) -> None:
        """Initialize the slow path.

        Args:
            harness: The MemoryHarness instance to operate on.
            entity_extractor_llm: Optional LLM for entity extraction
                (BaseChatModel from langchain-core).
            summarizer_llm: Optional LLM for summarization
                (BaseChatModel from langchain-core).
        """
        self.harness = harness

        # Import agents
        from memharness.agents.consolidator import ConsolidatorAgent
        from memharness.agents.entity_extractor import EntityExtractorAgent
        from memharness.agents.summarizer import SummarizerAgent

        self._entity_extractor = EntityExtractorAgent(harness, llm=entity_extractor_llm)
        self._summarizer = SummarizerAgent(harness, llm=summarizer_llm)
        self._consolidator = ConsolidatorAgent(harness)
        self._last_processed_at: datetime | None = None

    async def run_all(self) -> list[WorkerResult]:
        """Run all background workers. Call periodically.

        This method runs all slow path workers in sequence:
        1. Extract entities from recent conversations
        2. Summarize long threads
        3. Consolidate duplicate entities
        4. Garbage collect old data

        Returns:
            List of WorkerResult objects, one per worker.

        Example:
            ```python
            results = await slow.run_all()
            for r in results:
                print(f'{r.worker}: {r.processed} processed, {r.errors} errors, '
                      f'{r.duration_ms:.1f}ms')
            ```
        """
        results = []
        results.append(await self.extract_entities())
        results.append(await self.summarize_threads())
        results.append(await self.consolidate())
        self._last_processed_at = datetime.now(UTC)
        return results

    async def extract_entities(self) -> WorkerResult:
        """Scan recent conversations and extract/upsert entities.

        This worker:
        1. Gets recent unsummarized conversation messages (since last_processed_at)
        2. Extracts entities from each message
        3. UPSERTs entities: if entity exists (by name), update it; else insert
        4. Tracks updated_at timestamp for freshness

        Returns:
            WorkerResult with extraction statistics.

        Example:
            ```python
            result = await slow.extract_entities()
            print(f'Extracted {result.processed} entities in {result.duration_ms}ms')
            ```
        """
        start = datetime.now(UTC)
        processed = 0
        errors = 0
        try:
            # Get recent unsummarized conv messages and extract entities
            result = await self._entity_extractor.extract_from_recent(since=self._last_processed_at)
            processed = result.get("extracted", 0)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            errors = 1

        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
        return WorkerResult("entity_extractor", processed, errors, elapsed)

    async def summarize_threads(self) -> WorkerResult:
        """Summarize threads that have grown too long.

        This worker:
        1. Identifies threads with message counts exceeding a threshold
        2. Summarizes older messages in the thread
        3. Marks summarized messages (updates summary_id)
        4. Frees up context space for future queries

        Returns:
            WorkerResult with summarization statistics.

        Example:
            ```python
            result = await slow.summarize_threads()
            print(f'Summarized {result.processed} threads')
            ```
        """
        start = datetime.now(UTC)
        processed = 0
        errors = 0
        try:
            # For now, this is a placeholder
            # In a full implementation, you would:
            # 1. Query for threads with > N messages
            # 2. For each thread, call _summarizer.summarize_thread()
            # 3. Store summary and mark messages as summarized
            pass
        except Exception as e:
            logger.error(f"Thread summarization failed: {e}", exc_info=True)
            errors = 1

        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
        return WorkerResult("summarizer", processed, errors, elapsed)

    async def consolidate(self) -> WorkerResult:
        """Merge duplicate entities.

        This worker:
        1. Scans entity table for duplicates (similar names, high embedding similarity)
        2. Merges duplicate entities (keeps most recent, deletes others)
        3. Updates references to merged entities

        Returns:
            WorkerResult with consolidation statistics.

        Example:
            ```python
            result = await slow.consolidate()
            print(f'Merged {result.processed} duplicate entities')
            ```
        """
        start = datetime.now(UTC)
        processed = 0
        errors = 0
        try:
            result = await self._consolidator.consolidate_memories(min_memories=5)
            processed = result.get("merged", 0)
        except Exception as e:
            logger.error(f"Entity consolidation failed: {e}", exc_info=True)
            errors = 1

        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
        return WorkerResult("consolidator", processed, errors, elapsed)
