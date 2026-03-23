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

Usage:
    from memharness.agents.agent_workflow import create_after_workflow

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

__all__ = ["create_after_workflow", "AfterState"]


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
    state: AfterState, harness: MemoryHarness, max_tokens: int = 4000, threshold: float = 0.8
) -> dict[str, Any]:
    """Summarize conversation if token usage exceeds threshold."""
    thread_id = state["thread_id"]

    try:
        messages = await harness.get_conversational(thread_id, limit=200)
        total_chars = sum(len(m.content) for m in messages)
        token_estimate = total_chars // 4  # standard approximation
        usage = token_estimate / max_tokens
        if usage >= threshold:
            from memharness.agents.summarizer import SummarizerAgent

            summarizer = SummarizerAgent(harness)
            await summarizer.summarize_thread(thread_id)
            logger.info("Summarized thread %s (%.0f%% token usage)", thread_id, usage * 100)
            return {"summarized": True}
    except Exception as e:
        logger.debug("Summarization check failed: %s", e)

    return {"summarized": False}


# =============================================================================
# Graph Builder
# =============================================================================


def create_after_workflow(
    harness: MemoryHarness,
    max_tokens: int = 4000,
    summarize_threshold: float = 0.8,
) -> StateGraph:
    """Create the AFTER-agent LangGraph workflow.

    Graph:
        START → save_response → extract_entities → save_workflow → check_summarization → END

    Args:
        harness: The MemoryHarness instance.
        max_tokens: Maximum context token budget (default 4000).
        summarize_threshold: Trigger summarization at this usage percentage (default 0.8 = 80%).

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
        return await check_summarization(state, harness, max_tokens, summarize_threshold)

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
