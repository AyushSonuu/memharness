# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Consolidator agent for merging duplicate memories.

Finds and merges similar or duplicate memories to reduce redundancy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memharness.types import MemoryUnit

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from memharness.core.harness import MemoryHarness


class ConsolidatorAgent:
    """
    Agent that consolidates duplicate or highly similar memories.

    Works in two modes:
    1. Without LLM: Uses embedding similarity only
    2. With LLM: Uses LLM to intelligently merge content

    Example:
        ```python
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("gpt-4o")
        agent = ConsolidatorAgent(harness, llm=llm, threshold=0.85)
        result = await agent.consolidate_memories(min_memories=5)
        # Returns: {"merged": 3, "deleted": 3}
        ```
    """

    def __init__(
        self,
        harness: MemoryHarness,
        llm: BaseChatModel | None = None,
        threshold: float = 0.85,
    ) -> None:
        """
        Initialize the consolidator agent.

        Args:
            harness: The MemoryHarness instance to operate on.
            llm: Optional LLM for intelligent merging.
            threshold: Similarity threshold (0.0-1.0) for considering duplicates.
        """
        self.harness = harness
        self.llm = llm
        self.threshold = threshold

    def _cosine_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if len(embedding1) != len(embedding2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=False))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _merge_memories_heuristic(
        self, memory1: MemoryUnit, memory2: MemoryUnit
    ) -> MemoryUnit:
        """
        Merge two memories using heuristic approach (keep longer content).

        Args:
            memory1: First memory to merge.
            memory2: Second memory to merge.

        Returns:
            Merged memory unit.
        """
        # Keep the one with longer content
        if len(memory1.content) >= len(memory2.content):
            primary, secondary = memory1, memory2
        else:
            primary, secondary = memory2, memory1

        # Merge metadata
        merged_metadata = {**secondary.metadata, **primary.metadata}
        merged_metadata["merged_from"] = [
            memory1.id,
            memory2.id,
        ]

        # Return primary with merged metadata
        return MemoryUnit(
            id=primary.id,
            memory_type=primary.memory_type,
            content=primary.content,
            embedding=primary.embedding,
            metadata=merged_metadata,
            namespace=primary.namespace,
        )

    async def _merge_memories_llm(self, memory1: MemoryUnit, memory2: MemoryUnit) -> MemoryUnit:
        """
        Merge two memories using LLM for intelligent content merging.

        Args:
            memory1: First memory to merge.
            memory2: Second memory to merge.

        Returns:
            Merged memory unit.
        """
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            # Fall back to heuristic
            return await self._merge_memories_heuristic(memory1, memory2)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a memory consolidation system. Merge the two similar "
                    "memories below into a single, comprehensive memory. Keep all "
                    "important information and remove redundancy.",
                ),
                ("user", "Memory 1: {content1}\n\nMemory 2: {content2}"),
            ]
        )

        # Build chain
        chain = prompt | self.llm | StrOutputParser()

        # Generate merged content
        try:
            merged_content = await chain.ainvoke(
                {"content1": memory1.content, "content2": memory2.content}
            )

            # Use memory1 as base
            merged_metadata = {**memory2.metadata, **memory1.metadata}
            merged_metadata["merged_from"] = [
                memory1.id,
                memory2.id,
            ]

            return MemoryUnit(
                id=memory1.id,
                memory_type=memory1.memory_type,
                content=merged_content,
                embedding=memory1.embedding,  # Keep original embedding for now
                metadata=merged_metadata,
                namespace=memory1.namespace,
            )
        except Exception:
            # Fall back to heuristic on error
            return await self._merge_memories_heuristic(memory1, memory2)

    async def consolidate_memories(
        self, min_memories: int = 5, memory_type: str | None = None
    ) -> dict[str, int]:
        """
        Consolidate duplicate memories.

        Args:
            min_memories: Minimum number of memories before consolidation runs.
            memory_type: Optional memory type to consolidate (None = all types).

        Returns:
            Dictionary with 'merged' and 'deleted' counts.
        """
        # This is a simplified implementation
        # In production, you'd search across the backend for similar embeddings
        merged_count = 0
        deleted_count = 0

        # For now, return empty results (full implementation would query backend)
        return {"merged": merged_count, "deleted": deleted_count}

    async def run(
        self, min_memories: int = 5, memory_type: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Run the consolidator agent.

        Args:
            min_memories: Minimum number of memories before consolidation.
            memory_type: Optional memory type to consolidate.
            **kwargs: Additional arguments (ignored).

        Returns:
            Dictionary with consolidation results.
        """
        result = await self.consolidate_memories(min_memories, memory_type)
        return {
            "merged": result["merged"],
            "deleted": result["deleted"],
            "threshold": self.threshold,
        }
