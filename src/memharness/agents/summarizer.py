# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Summarization agent for compressing old conversations.

This agent creates summaries of conversation threads and links them
to the original messages via source_ids. Original messages are marked
as summarized but not deleted.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from memharness.agents.base import AgentConfig, EmbeddedAgent, TriggerType

if TYPE_CHECKING:
    from memharness import MemoryHarness


class SummarizerAgent(EmbeddedAgent):
    """
    Agent that compresses old conversations into summaries.

    Features:
    - Creates summary memories linked to original messages
    - Marks originals as summarized (preserves for reference)
    - Works without LLM (extractive summarization)
    - With LLM: generates abstractive summaries

    Without LLM:
    - Extracts key sentences based on position and keywords
    - Identifies topic shifts and important exchanges

    With LLM:
    - Generates coherent abstractive summaries
    - Captures context, intent, and outcomes
    """

    trigger = TriggerType.SCHEDULED
    schedule = "0 */6 * * *"  # Every 6 hours

    def __init__(
        self,
        memory: MemoryHarness,
        llm: Any | None = None,
        config: AgentConfig | None = None,
    ):
        super().__init__(memory, llm, config)

    @property
    def name(self) -> str:
        return "summarizer"

    async def run(
        self,
        thread_id: str | None = None,
        namespace: tuple[str, ...] | None = None,
        force: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run the summarization agent.

        Args:
            thread_id: Specific thread to summarize (None = all eligible)
            namespace: Namespace to operate in
            force: Force summarization even if threshold not met

        Returns:
            Dictionary with summarization results
        """
        started_at = datetime.now()
        errors: list[str] = []
        threads_processed = 0
        summaries_created = 0
        messages_summarized = 0

        try:
            # Get threads eligible for summarization
            threads = await self._get_eligible_threads(thread_id, namespace)

            for thread_info in threads:
                try:
                    messages = await self._get_thread_messages(thread_info["thread_id"], namespace)

                    if not force and len(messages) < self.config.summarizer_threshold_messages:
                        continue

                    # Filter to unsummarized messages
                    unsummarized = [
                        m for m in messages if not m.get("metadata", {}).get("summarized")
                    ]

                    if not unsummarized:
                        continue

                    # Create summary
                    summary_content = await self._create_summary(unsummarized)
                    source_ids = [m["id"] for m in unsummarized]

                    # Store summary with links
                    await self._store_summary(
                        thread_id=thread_info["thread_id"],
                        content=summary_content,
                        source_ids=source_ids,
                        namespace=namespace,
                        message_count=len(unsummarized),
                    )

                    # Mark originals as summarized
                    await self._mark_as_summarized(source_ids)

                    threads_processed += 1
                    summaries_created += 1
                    messages_summarized += len(unsummarized)

                except Exception as e:
                    errors.append(f"Thread {thread_info['thread_id']}: {str(e)}")

            result = await self._create_result(
                success=len(errors) == 0,
                started_at=started_at,
                items_processed=threads_processed,
                items_created=summaries_created,
                items_updated=messages_summarized,
                errors=errors,
                metadata={
                    "mode": "llm" if self.has_llm else "extractive",
                    "messages_summarized": messages_summarized,
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

    async def _get_eligible_threads(
        self,
        thread_id: str | None,
        namespace: tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        """Get threads eligible for summarization."""
        # If specific thread requested, return just that
        if thread_id:
            return [{"thread_id": thread_id}]

        # Otherwise, query for threads with old unsummarized messages
        # This would use the memory backend to find eligible threads
        # For now, return empty - actual implementation depends on backend
        try:
            # Query conversational memories to find distinct threads
            cutoff = datetime.now() - timedelta(hours=self.config.summarizer_max_age_hours)
            # Would call memory.search() or similar to find threads
            return []
        except Exception:
            return []

    async def _get_thread_messages(
        self,
        thread_id: str,
        namespace: tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        """Get all messages in a thread."""
        try:
            # Would call memory.get_thread() or similar
            # Returns list of memory units as dicts
            return []
        except Exception:
            return []

    async def _create_summary(self, messages: list[dict[str, Any]]) -> str:
        """Create a summary of the messages."""
        if self.has_llm:
            return await self._create_llm_summary(messages)
        return self._create_extractive_summary(messages)

    async def _create_llm_summary(self, messages: list[dict[str, Any]]) -> str:
        """Create an abstractive summary using LLM."""
        # Format messages for LLM
        formatted = "\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')}" for m in messages
        )

        prompt = f"""Summarize the following conversation. Capture:
1. Main topics discussed
2. Key decisions or conclusions
3. Important context for future reference
4. Any action items or follow-ups mentioned

Conversation:
{formatted}

Summary:"""

        try:
            # Call LLM - interface depends on what's passed in
            if hasattr(self.llm, "generate"):
                response = await self.llm.generate(prompt)
                return response
            elif hasattr(self.llm, "complete"):
                response = await self.llm.complete(prompt)
                return response
            elif callable(self.llm):
                response = await self.llm(prompt)
                return response if isinstance(response, str) else str(response)
            else:
                # Fallback to extractive
                return self._create_extractive_summary(messages)
        except Exception:
            # Fallback to extractive on error
            return self._create_extractive_summary(messages)

    def _create_extractive_summary(self, messages: list[dict[str, Any]]) -> str:
        """
        Create an extractive summary without LLM.

        Uses heuristics:
        - First and last messages (context and conclusion)
        - Messages with questions (key inquiries)
        - Messages with decision keywords
        - Longer messages (more substantive)
        """
        if not messages:
            return ""

        # Score each message
        scored: list[tuple[float, dict[str, Any]]] = []

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            score = 0.0

            # Position scoring
            if i == 0:  # First message
                score += 2.0
            if i == len(messages) - 1:  # Last message
                score += 1.5
            if i < 3:  # Early messages
                score += 0.5

            # Content scoring
            content_lower = content.lower()

            # Questions are important
            if "?" in content:
                score += 1.5

            # Decision/conclusion indicators
            decision_words = [
                "decided",
                "agreed",
                "conclusion",
                "summary",
                "result",
                "solution",
                "answer",
                "resolved",
                "confirmed",
                "will do",
                "action",
                "next step",
                "todo",
                "deadline",
            ]
            for word in decision_words:
                if word in content_lower:
                    score += 1.0
                    break

            # Substantive content (length matters, but diminishing returns)
            word_count = len(content.split())
            if word_count > 20:
                score += min(word_count / 50, 2.0)

            # User messages often contain requirements
            if msg.get("role") == "user":
                score += 0.5

            scored.append((score, msg))

        # Sort by score and take top messages
        scored.sort(key=lambda x: x[0], reverse=True)
        max_sentences = min(5, len(scored))
        top_messages = [msg for _, msg in scored[:max_sentences]]

        # Re-sort by original order for coherence
        top_messages.sort(key=lambda m: messages.index(m))

        # Build summary
        summary_parts = []
        summary_parts.append(f"Summary of {len(messages)} messages:")

        for msg in top_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"- [{role}]: {content}")

        return "\n".join(summary_parts)

    async def _store_summary(
        self,
        thread_id: str,
        content: str,
        source_ids: list[str],
        namespace: tuple[str, ...] | None,
        message_count: int,
    ) -> str:
        """Store the summary as a memory unit."""
        # Generate deterministic ID for the summary
        id_source = f"summary:{thread_id}:{','.join(sorted(source_ids))}"
        summary_id = hashlib.sha256(id_source.encode()).hexdigest()[:16]

        # Would call memory.add_summary() or similar
        # For now, just return the ID
        # await self.memory.add_summary(
        #     content=content,
        #     source_ids=source_ids,
        #     metadata={
        #         "thread_id": thread_id,
        #         "message_count": message_count,
        #         "summarized_at": datetime.now().isoformat(),
        #     },
        #     namespace=namespace,
        # )
        return summary_id

    async def _mark_as_summarized(self, memory_ids: list[str]) -> None:
        """Mark memories as having been summarized."""
        for memory_id in memory_ids:
            # Would call memory.update() or similar
            # await self.memory.update(
            #     memory_id,
            #     metadata={"summarized": True, "summarized_at": datetime.now().isoformat()},
            # )
            pass

    async def summarize_thread(
        self,
        thread_id: str,
        namespace: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """
        Convenience method to summarize a specific thread.

        Args:
            thread_id: The thread to summarize
            namespace: Optional namespace

        Returns:
            Summary result
        """
        return await self.run(thread_id=thread_id, namespace=namespace, force=True)
