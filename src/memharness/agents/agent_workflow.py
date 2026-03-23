# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""LangGraph workflow for AFTER-agent memory operations.

This module provides a LangGraph graph that handles all memory write
operations AFTER the main agent runs:
- save_response: persist assistant response to conversation table
- extract_entities: extract and upsert entities from response
- save_workflow: save tool execution steps as reusable workflow
- check_summarization: summarize if thread exceeds threshold

The graph does NOT contain an LLM agent. It only manages memory.

Usage as LangChain middleware:
    from memharness.agents.agent_workflow import MemoryWorkflowMiddleware

    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=get_read_tools(harness),
        middleware=[
            ContextMiddleware(harness, thread_id),       # BEFORE
            ConversationMiddleware(harness, thread_id),   # BEFORE+AFTER
            MemoryWorkflowMiddleware(harness, thread_id), # AFTER (this graph)
        ],
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from memharness.core.harness import MemoryHarness

logger = logging.getLogger(__name__)

__all__ = ["MemoryWorkflowMiddleware", "create_after_workflow"]


# =============================================================================
# Graph State
# =============================================================================


class AfterState(TypedDict):
    """State for the AFTER-agent workflow."""

    messages: Annotated[list[Any], add_messages]
    thread_id: str
    response_text: str
    steps: list[str]
    entities_extracted: int
    workflow_saved: bool
    summarized: bool


# =============================================================================
# Graph Nodes
# =============================================================================


async def save_response(state: AfterState, harness: MemoryHarness) -> dict[str, Any]:
    """Save assistant response to conversation table."""
    response = state.get("response_text", "")
    thread_id = state["thread_id"]

    if response:
        await harness.add_conversational(thread_id, "assistant", response)
        logger.debug("Saved response to thread %s", thread_id)

    return {}


async def extract_entities(state: AfterState, harness: MemoryHarness) -> dict[str, Any]:
    """Extract entities from response and upsert to entity table."""
    response = state.get("response_text", "")
    if not response:
        return {"entities_extracted": 0}

    count = 0
    try:
        from memharness.agents.entity_extractor import EntityExtractorAgent

        extractor = EntityExtractorAgent(harness)
        entities = await extractor.extract_entities(response)
        for category, names in entities.items():
            for name in names:
                await harness.add_entity(name, category, f"{category}: {name}")
                count += 1
    except Exception as e:
        logger.debug("Entity extraction failed: %s", e)

    return {"entities_extracted": count}


async def save_workflow(state: AfterState, harness: MemoryHarness) -> dict[str, Any]:
    """Save tool execution steps as a reusable workflow."""
    steps = state.get("steps", [])
    if not steps:
        return {"workflow_saved": False}

    response = state.get("response_text", "")
    thread_id = state["thread_id"]

    try:
        await harness.add_workflow(
            task=f"Thread {thread_id}",
            steps=steps,
            outcome=response[:200] if response else "completed",
        )
        logger.debug("Saved workflow with %d steps", len(steps))
        return {"workflow_saved": True}
    except Exception as e:
        logger.debug("Workflow save failed: %s", e)
        return {"workflow_saved": False}


async def check_summarization(
    state: AfterState, harness: MemoryHarness, threshold: int = 50
) -> dict[str, Any]:
    """Summarize conversation if thread exceeds threshold."""
    thread_id = state["thread_id"]

    try:
        messages = await harness.get_conversational(thread_id, limit=threshold + 1)
        if len(messages) >= threshold:
            from memharness.agents.summarizer import SummarizerAgent

            summarizer = SummarizerAgent(harness)
            await summarizer.summarize_thread(thread_id)
            logger.info("Summarized thread %s (%d messages)", thread_id, len(messages))
            return {"summarized": True}
    except Exception as e:
        logger.debug("Summarization check failed: %s", e)

    return {"summarized": False}


# =============================================================================
# Graph Builder
# =============================================================================


def create_after_workflow(
    harness: MemoryHarness,
    summarize_threshold: int = 50,
) -> StateGraph:
    """Create the AFTER-agent LangGraph workflow.

    Graph:
        START → save_response → extract_entities → save_workflow → check_summarization → END

    Args:
        harness: The MemoryHarness instance.
        summarize_threshold: Summarize when thread exceeds this message count.

    Returns:
        Compiled LangGraph StateGraph.
    """
    builder = StateGraph(AfterState)

    # Bind harness to node functions
    async def _save_response(state):
        return await save_response(state, harness)

    async def _extract_entities(state):
        return await extract_entities(state, harness)

    async def _save_workflow(state):
        return await save_workflow(state, harness)

    async def _check_summarization(state):
        return await check_summarization(state, harness, summarize_threshold)

    # Add nodes
    builder.add_node("save_response", _save_response)
    builder.add_node("extract_entities", _extract_entities)
    builder.add_node("save_workflow", _save_workflow)
    builder.add_node("check_summarization", _check_summarization)

    # Add edges
    builder.add_edge(START, "save_response")
    builder.add_edge("save_response", "extract_entities")
    builder.add_edge("extract_entities", "save_workflow")
    builder.add_edge("save_workflow", "check_summarization")
    builder.add_edge("check_summarization", END)

    return builder.compile()


# =============================================================================
# LangChain Middleware Wrapper
# =============================================================================


class MemoryWorkflowMiddleware:
    """Wraps the AFTER-agent LangGraph workflow as a LangChain middleware.

    BEFORE model: no-op
    AFTER model: runs the LangGraph workflow (save_response → extract_entities
                 → save_workflow → check_summarization)

    Args:
        harness: The MemoryHarness instance.
        thread_id: Conversation thread identifier.
        summarize_threshold: Summarize when thread exceeds this count (default 50).

    Usage:
        agent = create_agent(
            model="anthropic:claude-sonnet-4-6",
            tools=get_read_tools(harness),
            middleware=[
                ContextMiddleware(harness, thread_id),
                ConversationMiddleware(harness, thread_id),
                MemoryWorkflowMiddleware(harness, thread_id),
            ],
        )
    """

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
        summarize_threshold: int = 50,
    ) -> None:
        self.harness = harness
        self.thread_id = thread_id
        self._graph = create_after_workflow(harness, summarize_threshold)
        self._steps: list[str] = []

    async def abefore_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """No-op — this middleware only runs AFTER the model."""
        return None

    async def aafter_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Run the AFTER-agent LangGraph workflow."""
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]

        # Track tool calls
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                self._steps.append(f"{tc.get('name', 'unknown')}()")
            return None  # Not final answer yet

        # Final answer — run the graph
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        await self._graph.ainvoke(
            {
                "messages": messages,
                "thread_id": self.thread_id,
                "response_text": last_msg.content,
                "steps": self._steps,
                "entities_extracted": 0,
                "workflow_saved": False,
                "summarized": False,
            }
        )

        self._steps = []  # Reset for next turn
        return None
