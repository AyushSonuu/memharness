# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Memory workflow middleware.

This module provides MemoryWorkflowMiddleware — an AFTER-agent middleware
that handles all memory write operations:
- Entity extraction from agent responses
- Summarization when conversation threads get long
- Workflow saving when the agent uses tools
- Consolidation of duplicate entities (periodic)

This middleware is placed AFTER the agent call. It processes the agent's
output and enriches the memory database. The agent itself only gets
read-only tools.

Usage with LangChain:
    from memharness.agents.agent_workflow import MemoryWorkflowMiddleware

    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=get_read_tools(harness),
        middleware=[
            MemharnessContextMiddleware(harness, thread_id),    # BEFORE
            MemharnessConversationMiddleware(harness, thread_id), # BEFORE+AFTER
            MemoryWorkflowMiddleware(harness, thread_id),        # AFTER
        ],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memharness.core.harness import MemoryHarness

logger = logging.getLogger(__name__)

__all__ = ["MemoryWorkflowMiddleware", "WorkflowConfig"]


@dataclass
class WorkflowConfig:
    """Configuration for the memory workflow middleware.

    Attributes:
        summarize_after_messages: Summarize when thread exceeds this count.
        extract_entities: Whether to extract entities from responses.
        save_workflows: Whether to save tool execution patterns.
        consolidate_every_n: Run consolidation every N agent calls.
    """

    summarize_after_messages: int = 50
    extract_entities: bool = True
    save_workflows: bool = True
    consolidate_every_n: int = 20


class MemoryWorkflowMiddleware:
    """AFTER-agent middleware for memory management.

    Placed after the agent call, this middleware processes the agent's
    output and manages memory:

    1. Extract entities from the agent's response (regex, no LLM)
    2. Save tool execution steps as a reusable workflow
    3. Trigger summarization when conversation gets long
    4. Periodically consolidate duplicate entities

    The agent itself only gets read-only tools. This middleware
    handles all the writes.

    Args:
        harness: The MemoryHarness instance.
        thread_id: Conversation thread identifier.
        config: Optional WorkflowConfig for thresholds.

    Example:
        ```python
        middleware = MemoryWorkflowMiddleware(harness, "thread-1")
        agent = create_agent(
            model=...,
            tools=get_read_tools(harness),
            middleware=[context_mw, conversation_mw, middleware],
        )
        ```
    """

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.harness = harness
        self.thread_id = thread_id
        self.config = config or WorkflowConfig()
        self._call_count = 0
        self._current_steps: list[str] = []

    async def aafter_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Process agent output — extract entities, save workflow, summarize.

        This runs after every model call. It:
        1. Tracks tool calls as workflow steps
        2. On final answer: saves workflow, extracts entities, checks summarization
        """
        try:
            from langchain_core.messages import AIMessage, ToolMessage
        except ImportError:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]

        # Track tool calls as workflow steps
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                name = tc.get("name", "unknown")
                self._current_steps.append(f"{name}()")

            # Log tool executions
            for tc in last_msg.tool_calls:
                try:
                    await self.harness.add_tool_log(
                        tool_name=tc.get("name", "unknown"),
                        tool_input=str(tc.get("args", {}))[:500],
                        tool_output="",
                        status="success",
                        thread_id=self.thread_id,
                    )
                except Exception:
                    pass

            return None  # Not final answer yet, keep looping

        # Final answer (AIMessage with content, no tool calls)
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        self._call_count += 1

        # 1. Extract entities from response
        if self.config.extract_entities:
            await self._extract_entities(last_msg.content)

        # 2. Save workflow if tools were used
        if self.config.save_workflows and self._current_steps:
            await self._save_workflow(last_msg.content)
            self._current_steps = []

        # 3. Check if summarization needed
        await self._maybe_summarize()

        # 4. Periodic consolidation
        if self._call_count % self.config.consolidate_every_n == 0:
            await self._consolidate()

        return None

    async def _extract_entities(self, text: str) -> None:
        """Extract entities from text and upsert to entity table."""
        try:
            from memharness.agents.entity_extractor import EntityExtractorAgent

            extractor = EntityExtractorAgent(self.harness)
            entities = await extractor.extract_entities(text)
            for category, names in entities.items():
                for name in names:
                    await self.harness.add_entity(
                        name=name,
                        entity_type=category,
                        description=f"{category}: {name}",
                    )
        except Exception as e:
            logger.debug("Entity extraction failed: %s", e)

    async def _save_workflow(self, final_answer: str) -> None:
        """Save tool execution steps as a reusable workflow."""
        try:
            await self.harness.add_workflow(
                task=f"Thread {self.thread_id}",
                steps=self._current_steps,
                outcome=final_answer[:200],
            )
        except Exception as e:
            logger.debug("Workflow save failed: %s", e)

    async def _maybe_summarize(self) -> None:
        """Summarize conversation if it exceeds the threshold."""
        try:
            messages = await self.harness.get_conversational(
                self.thread_id, limit=self.config.summarize_after_messages + 1
            )
            if len(messages) >= self.config.summarize_after_messages:
                from memharness.agents.summarizer import SummarizerAgent

                summarizer = SummarizerAgent(self.harness)
                await summarizer.summarize_thread(self.thread_id)
                logger.info("Summarized thread %s (%d messages)", self.thread_id, len(messages))
        except Exception as e:
            logger.debug("Summarization check failed: %s", e)

    async def _consolidate(self) -> None:
        """Consolidate duplicate entities."""
        try:
            from memharness.agents.consolidator import ConsolidatorAgent

            consolidator = ConsolidatorAgent(self.harness)
            await consolidator.consolidate_memories()
            logger.debug("Consolidation completed")
        except Exception as e:
            logger.debug("Consolidation failed: %s", e)

    # LangChain middleware interface compatibility
    async def abefore_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """No-op — this middleware only runs AFTER the model."""
        return None
