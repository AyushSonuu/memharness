# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Tests for MemoryWorkflowMiddleware and AFTER-agent LangGraph workflow."""

from __future__ import annotations

import pytest

from memharness import MemoryHarness
from memharness.agents.agent_workflow import (
    MemoryWorkflowMiddleware,
    create_after_workflow,
)


@pytest.fixture
async def harness():
    h = MemoryHarness("memory://")
    await h.connect()
    yield h
    await h.disconnect()


class TestCreateAfterWorkflow:
    async def test_creates_graph(self, harness):
        graph = create_after_workflow(harness)
        assert graph is not None

    async def test_graph_runs(self, harness):
        graph = create_after_workflow(harness)
        result = await graph.ainvoke(
            {
                "messages": [],
                "thread_id": "test-1",
                "response_text": "Alice works at Google in San Francisco.",
                "steps": [],
                "entities_extracted": 0,
                "workflow_saved": False,
                "summarized": False,
            }
        )
        assert result is not None

    async def test_graph_saves_response(self, harness):
        graph = create_after_workflow(harness)
        await graph.ainvoke(
            {
                "messages": [],
                "thread_id": "test-1",
                "response_text": "Test response",
                "steps": [],
                "entities_extracted": 0,
                "workflow_saved": False,
                "summarized": False,
            }
        )
        messages = await harness.get_conversational("test-1")
        assert any("Test response" in m.content for m in messages)

    async def test_graph_saves_workflow(self, harness):
        graph = create_after_workflow(harness)
        result = await graph.ainvoke(
            {
                "messages": [],
                "thread_id": "test-2",
                "response_text": "Done",
                "steps": ["search()", "read()"],
                "entities_extracted": 0,
                "workflow_saved": False,
                "summarized": False,
            }
        )
        assert result["workflow_saved"] is True

    async def test_graph_no_workflow_without_steps(self, harness):
        graph = create_after_workflow(harness)
        result = await graph.ainvoke(
            {
                "messages": [],
                "thread_id": "test-3",
                "response_text": "Simple answer",
                "steps": [],
                "entities_extracted": 0,
                "workflow_saved": False,
                "summarized": False,
            }
        )
        assert result["workflow_saved"] is False


class TestMemoryWorkflowMiddleware:
    async def test_init(self, harness):
        mw = MemoryWorkflowMiddleware(harness, "t1")
        assert mw.thread_id == "t1"
        assert mw._steps == []

    async def test_abefore_noop(self, harness):
        mw = MemoryWorkflowMiddleware(harness, "t1")
        result = await mw.abefore_model({"messages": []}, None)
        assert result is None

    async def test_aafter_empty(self, harness):
        mw = MemoryWorkflowMiddleware(harness, "t1")
        result = await mw.aafter_model({"messages": []}, None)
        assert result is None

    async def test_tracks_tool_calls(self, harness):
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        mw = MemoryWorkflowMiddleware(harness, "t1")
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "memory_search", "args": {}, "id": "tc1", "type": "tool_call"}],
        )
        await mw.aafter_model({"messages": [msg]}, None)
        assert len(mw._steps) == 1

    async def test_runs_graph_on_final_answer(self, harness):
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        mw = MemoryWorkflowMiddleware(harness, "t1")
        mw._steps = ["search()"]
        msg = AIMessage(content="Here is the answer.")
        await mw.aafter_model({"messages": [msg]}, None)
        # Steps reset after graph runs
        assert mw._steps == []
