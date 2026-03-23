# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Consolidation agent for merging semantically similar memories.

Finds and intelligently merges duplicate or highly similar memories
to reduce redundancy and improve retrieval quality.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from memharness.agents.base import AgentConfig, EmbeddedAgent, TriggerType

if TYPE_CHECKING:
    from memharness import MemoryHarness


@dataclass
class SimilarityMatch:
    """Represents a pair of similar memories."""

    memory_a_id: str
    memory_b_id: str
    similarity_score: float
    similarity_type: str  # "exact", "near_duplicate", "semantic"
    merged: bool = False


class ConsolidatorAgent(EmbeddedAgent):
    """
    Agent that finds and merges semantically similar memories.

    Features:
    - Detects exact and near-duplicate memories
    - Semantic similarity comparison
    - Intelligent merging preserving information
    - Runs on schedule (cron)

    Without LLM:
    - Uses text similarity metrics (Jaccard, edit distance)
    - Keyword overlap analysis
    - Structural comparison

    With LLM:
    - Semantic understanding of content
    - Intelligent merge decisions
    - Context-aware deduplication
    """

    trigger = TriggerType.SCHEDULED
    schedule = "0 * * * *"  # Every hour

    def __init__(
        self,
        memory: MemoryHarness,
        llm: Any | None = None,
        config: AgentConfig | None = None,
    ):
        super().__init__(memory, llm, config)

    @property
    def name(self) -> str:
        return "consolidator"

    async def run(
        self,
        memory_type: str | None = None,
        namespace: tuple[str, ...] | None = None,
        dry_run: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run the consolidation agent.

        Args:
            memory_type: Specific memory type to consolidate (None = all)
            namespace: Namespace to operate in
            dry_run: If True, report duplicates without merging

        Returns:
            Dictionary with consolidation results
        """
        started_at = datetime.now()
        errors: list[str] = []
        memories_processed = 0
        duplicates_found = 0
        memories_merged = 0
        matches: list[SimilarityMatch] = []

        try:
            # Get memories to process
            memories = await self._get_memories_to_consolidate(memory_type, namespace)

            if len(memories) < self.config.consolidator_min_memories:
                # Not enough memories to consolidate
                result = await self._create_result(
                    success=True,
                    started_at=started_at,
                    metadata={"reason": "insufficient_memories"},
                )
                return result.to_dict()

            # Find similar pairs
            matches = await self._find_similar_pairs(memories)
            duplicates_found = len(matches)

            if not dry_run:
                # Merge duplicates
                for match in matches:
                    if match.similarity_score >= self.config.consolidator_similarity_threshold:
                        try:
                            await self._merge_memories(
                                match.memory_a_id,
                                match.memory_b_id,
                                match.similarity_type,
                            )
                            match.merged = True
                            memories_merged += 1
                        except Exception as e:
                            errors.append(
                                f"Merge {match.memory_a_id} + {match.memory_b_id}: {str(e)}"
                            )

            memories_processed = len(memories)

            result = await self._create_result(
                success=len(errors) == 0,
                started_at=started_at,
                items_processed=memories_processed,
                items_updated=memories_merged,
                items_deleted=memories_merged,  # One memory deleted per merge
                errors=errors,
                metadata={
                    "mode": "llm" if self.has_llm else "heuristic",
                    "duplicates_found": duplicates_found,
                    "dry_run": dry_run,
                    "matches": [
                        {
                            "a": m.memory_a_id,
                            "b": m.memory_b_id,
                            "score": m.similarity_score,
                            "type": m.similarity_type,
                            "merged": m.merged,
                        }
                        for m in matches[:10]  # Limit to first 10 for reporting
                    ],
                },
            )
            await self._log_run(result)

            return result.to_dict()

        except Exception as e:
            result = await self._create_result(
                success=False,
                started_at=started_at,
                errors=[str(e)],
            )
            return result.to_dict()

    async def find_duplicates(
        self,
        memory_type: str | None = None,
        namespace: tuple[str, ...] | None = None,
        threshold: float | None = None,
    ) -> list[SimilarityMatch]:
        """
        Find duplicate memories without merging.

        Args:
            memory_type: Type of memories to check
            namespace: Namespace to search in
            threshold: Similarity threshold (default from config)

        Returns:
            List of similar memory pairs
        """
        threshold = threshold or self.config.consolidator_similarity_threshold
        memories = await self._get_memories_to_consolidate(memory_type, namespace)
        matches = await self._find_similar_pairs(memories)

        return [m for m in matches if m.similarity_score >= threshold]

    async def _get_memories_to_consolidate(
        self,
        memory_type: str | None,
        namespace: tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        """Get memories eligible for consolidation."""
        # Would call memory.search() or memory.list()
        # Filter by type if specified
        # Return list of memory dicts with id, content, metadata
        return []

    async def _find_similar_pairs(
        self,
        memories: list[dict[str, Any]],
    ) -> list[SimilarityMatch]:
        """Find pairs of similar memories."""
        matches: list[SimilarityMatch] = []

        # Group by type for more focused comparison
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for memory in memories:
            mem_type = memory.get("memory_type", "unknown")
            by_type[mem_type].append(memory)

        # Compare within each type
        for mem_type, type_memories in by_type.items():
            n = len(type_memories)
            for i in range(n):
                for j in range(i + 1, n):
                    mem_a = type_memories[i]
                    mem_b = type_memories[j]

                    similarity, sim_type = await self._calculate_similarity(
                        mem_a, mem_b
                    )

                    if similarity >= self.config.consolidator_similarity_threshold:
                        matches.append(
                            SimilarityMatch(
                                memory_a_id=mem_a["id"],
                                memory_b_id=mem_b["id"],
                                similarity_score=similarity,
                                similarity_type=sim_type,
                            )
                        )

        # Sort by similarity (highest first)
        matches.sort(key=lambda m: m.similarity_score, reverse=True)

        return matches

    async def _calculate_similarity(
        self,
        memory_a: dict[str, Any],
        memory_b: dict[str, Any],
    ) -> tuple[float, str]:
        """
        Calculate similarity between two memories.

        Returns (similarity_score, similarity_type).
        """
        content_a = memory_a.get("content", "")
        content_b = memory_b.get("content", "")

        # Check for exact match
        if content_a == content_b:
            return 1.0, "exact"

        # Check for near-exact match (whitespace/case differences)
        norm_a = self._normalize_text(content_a)
        norm_b = self._normalize_text(content_b)

        if norm_a == norm_b:
            return 0.99, "near_exact"

        if self.has_llm:
            # Use LLM for semantic comparison
            return await self._calculate_semantic_similarity(
                content_a, content_b, memory_a, memory_b
            )

        # Heuristic similarity
        return self._calculate_heuristic_similarity(
            content_a, content_b, norm_a, norm_b
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def _calculate_heuristic_similarity(
        self,
        content_a: str,
        content_b: str,
        norm_a: str,
        norm_b: str,
    ) -> tuple[float, str]:
        """Calculate similarity using heuristics."""
        # Method 1: Sequence matcher (edit distance based)
        seq_sim = SequenceMatcher(None, norm_a, norm_b).ratio()

        # Method 2: Jaccard similarity on words
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())

        if words_a or words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
        else:
            jaccard = 0.0

        # Method 3: N-gram overlap
        ngrams_a = self._get_ngrams(norm_a, 3)
        ngrams_b = self._get_ngrams(norm_b, 3)

        if ngrams_a or ngrams_b:
            ngram_sim = len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)
        else:
            ngram_sim = 0.0

        # Combine methods (weighted average)
        combined = (seq_sim * 0.4) + (jaccard * 0.3) + (ngram_sim * 0.3)

        return combined, "heuristic"

    def _get_ngrams(self, text: str, n: int) -> set[str]:
        """Get character n-grams from text."""
        if len(text) < n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    async def _calculate_semantic_similarity(
        self,
        content_a: str,
        content_b: str,
        memory_a: dict[str, Any],
        memory_b: dict[str, Any],
    ) -> tuple[float, str]:
        """Calculate semantic similarity using LLM."""
        prompt = f"""Compare these two pieces of content and rate their semantic similarity.
Consider:
1. Do they express the same idea or information?
2. Are they about the same topic/subject?
3. Would keeping both add redundant information?

Content A:
{content_a[:500]}

Content B:
{content_b[:500]}

Rate the semantic similarity from 0.0 (completely different) to 1.0 (semantically identical).
Respond with ONLY a number between 0.0 and 1.0."""

        try:
            if hasattr(self.llm, "generate"):
                response = await self.llm.generate(prompt)
            elif hasattr(self.llm, "complete"):
                response = await self.llm.complete(prompt)
            elif callable(self.llm):
                response = await self.llm(prompt)
                response = response if isinstance(response, str) else str(response)
            else:
                return self._calculate_heuristic_similarity(
                    content_a, content_b,
                    self._normalize_text(content_a),
                    self._normalize_text(content_b),
                )

            # Parse response
            response = response.strip()
            try:
                score = float(response)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                return score, "semantic"
            except ValueError:
                # Try to find a number in the response
                match = re.search(r"(\d+\.?\d*)", response)
                if match:
                    score = float(match.group(1))
                    if score > 1:
                        score = score / 100  # Handle percentage responses
                    return max(0.0, min(1.0, score)), "semantic"

        except Exception:
            pass

        # Fallback to heuristic
        return self._calculate_heuristic_similarity(
            content_a, content_b,
            self._normalize_text(content_a),
            self._normalize_text(content_b),
        )

    async def _merge_memories(
        self,
        memory_a_id: str,
        memory_b_id: str,
        similarity_type: str,
    ) -> None:
        """
        Merge two memories into one.

        Keeps memory_a and updates it with info from memory_b,
        then marks memory_b as merged (or deletes it).
        """
        # Would fetch both memories
        # memory_a = await self.memory.get(memory_a_id)
        # memory_b = await self.memory.get(memory_b_id)

        if self.has_llm:
            merged_content = await self._merge_with_llm(memory_a_id, memory_b_id)
        else:
            merged_content = await self._merge_heuristic(memory_a_id, memory_b_id)

        # Update memory_a with merged content
        # await self.memory.update(
        #     memory_a_id,
        #     content=merged_content,
        #     metadata={
        #         "merged_from": [memory_b_id],
        #         "merged_at": datetime.now().isoformat(),
        #     }
        # )

        # Mark memory_b as merged (soft delete)
        # await self.memory.update(
        #     memory_b_id,
        #     metadata={
        #         "merged_into": memory_a_id,
        #         "merged_at": datetime.now().isoformat(),
        #         "archived": True,
        #     }
        # )
        pass

    async def _merge_with_llm(
        self,
        memory_a_id: str,
        memory_b_id: str,
    ) -> str:
        """Merge memories using LLM."""
        # Would fetch memory content
        content_a = ""  # await self.memory.get(memory_a_id).content
        content_b = ""  # await self.memory.get(memory_b_id).content

        prompt = f"""Merge these two related pieces of information into a single, coherent entry.
Preserve all unique information from both. Remove redundancy.
Keep the most complete and accurate version of any conflicting information.

Content A:
{content_a}

Content B:
{content_b}

Merged content:"""

        try:
            if hasattr(self.llm, "generate"):
                return await self.llm.generate(prompt)
            elif hasattr(self.llm, "complete"):
                return await self.llm.complete(prompt)
            elif callable(self.llm):
                response = await self.llm(prompt)
                return response if isinstance(response, str) else str(response)
        except Exception:
            pass

        return await self._merge_heuristic(memory_a_id, memory_b_id)

    async def _merge_heuristic(
        self,
        memory_a_id: str,
        memory_b_id: str,
    ) -> str:
        """Merge memories using heuristics (keep longer/more complete)."""
        # Would fetch memory content
        content_a = ""  # await self.memory.get(memory_a_id).content
        content_b = ""  # await self.memory.get(memory_b_id).content

        # Simple heuristic: keep the longer one
        # Could be enhanced with more sophisticated merging
        if len(content_b) > len(content_a):
            return content_b
        return content_a
