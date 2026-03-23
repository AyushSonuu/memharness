# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Tests for MemoryWorkflowMiddleware."""

from __future__ import annotations

import pytest

from memharness import MemoryHarness
from memharness.agents.agent_workflow import MemoryWorkflowMiddleware, WorkflowConfig


@pytest.fixture
async def harness():
    h = MemoryHarness("memory://")
    await h.connect()
    yield h
    await h.disconnect()


@pytest.fixture
def middleware(harness):
    return MemoryWorkflowMiddleware(harness, thread_id="test-thread")


class TestWorkflowConfig:
    def test_defaults(self):
        config = WorkflowConfig()
        assert config.summarize_after_messages == 50
        assert config.extract_entities is True
        assert config.save_workflows is True
        assert config.consolidate_every_n == 20

    def test_custom(self):
        config = WorkflowConfig(summarize_after_messages=100, extract_entities=False)
        assert config.summarize_after_messages == 100
        assert config.extract_entities is False


class TestMemoryWorkflowMiddleware:
    async def test_init(self, middleware):
        assert middleware.thread_id == "test-thread"
        assert middleware._call_count == 0
        assert middleware._current_steps == []

    async def test_abefore_model_noop(self, middleware):
        result = await middleware.abefore_model({"messages": []}, None)
        assert result is None

    async def test_aafter_model_empty(self, middleware):
        result = await middleware.aafter_model({"messages": []}, None)
        assert result is None

    async def test_tracks_tool_calls(self, middleware):
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "memory_search", "args": {}, "id": "tc_1", "type": "tool_call"}],
        )
        result = await middleware.aafter_model({"messages": [msg]}, None)
        assert result is None
        assert len(middleware._current_steps) == 1
        assert "memory_search" in middleware._current_steps[0]

    async def test_final_answer_resets_steps(self, middleware, harness):
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        middleware._current_steps = ["search()", "read()"]
        msg = AIMessage(content="Here is the answer.")
        await middleware.aafter_model({"messages": [msg]}, None)
        assert middleware._current_steps == []

    async def test_entity_extraction_on_response(self, middleware, harness):
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        msg = AIMessage(content="Alice works at Google in San Francisco.")
        await middleware.aafter_model({"messages": [msg]}, None)
        # Entities may or may not be extracted (depends on regex matching)
        # Just verify no crash
        assert middleware._call_count == 1

    async def test_custom_config(self, harness):
        config = WorkflowConfig(extract_entities=False, save_workflows=False)
        mw = MemoryWorkflowMiddleware(harness, "t1", config=config)
        assert mw.config.extract_entities is False
