# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tests for MemoryAwareAgent workflow implementation.

These tests cover the BEFORE → INSIDE → AFTER agent loop pattern.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memharness.agents.agent_workflow import (
    AGENT_SYSTEM_PROMPT,
    AgentResult,
    MemoryAwareAgent,
)
from memharness.agents.context_assembler import AssembledContext
from memharness.core.harness import MemoryHarness


class TestAgentResult:
    """Test AgentResult dataclass."""

    def test_agent_result_creation(self):
        """Test creating an AgentResult."""
        result = AgentResult(
            answer="Test answer",
            steps=["step1", "step2"],
            thread_id="thread-123",
            tool_calls=2,
            context_usage_percent=0.5,
            workflow_saved=True,
            entities_extracted=3,
            iterations=5,
        )

        assert result.answer == "Test answer"
        assert result.steps == ["step1", "step2"]
        assert result.thread_id == "thread-123"
        assert result.tool_calls == 2
        assert result.context_usage_percent == 0.5
        assert result.workflow_saved is True
        assert result.entities_extracted == 3
        assert result.iterations == 5

    def test_agent_result_defaults(self):
        """Test AgentResult with minimal args (defaults)."""
        result = AgentResult(answer="Test")

        assert result.answer == "Test"
        assert result.steps == []
        assert result.thread_id == ""
        assert result.tool_calls == 0
        assert result.context_usage_percent == 0.0
        assert result.workflow_saved is False
        assert result.entities_extracted == 0
        assert result.iterations == 0


class TestMemoryAwareAgent:
    """Test MemoryAwareAgent class."""

    @pytest.fixture
    def mock_harness(self):
        """Create a mock MemoryHarness."""
        harness = MagicMock(spec=MemoryHarness)
        harness.add_conversational = AsyncMock()
        harness.add_entity = AsyncMock()
        harness.add_workflow = AsyncMock()
        harness.add_tool_log = AsyncMock()
        return harness

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm

    def test_agent_initialization(self, mock_harness, mock_llm):
        """Test MemoryAwareAgent initialization."""
        agent = MemoryAwareAgent(
            harness=mock_harness,
            llm=mock_llm,
            max_iterations=5,
            max_context_tokens=2000,
            summarize_threshold=0.7,
        )

        assert agent.harness == mock_harness
        assert agent.llm == mock_llm
        assert agent.max_iterations == 5
        assert agent.max_context_tokens == 2000
        assert agent.summarize_threshold == 0.7
        assert agent._context_agent is not None
        assert agent._summarizer is not None
        assert agent._entity_extractor is not None

    def test_agent_initialization_without_llm(self, mock_harness):
        """Test MemoryAwareAgent initialization without LLM."""
        agent = MemoryAwareAgent(harness=mock_harness, llm=None)

        assert agent.llm is None
        # Should still initialize sub-agents
        assert agent._context_agent is not None

    @pytest.mark.asyncio
    async def test_run_without_llm_raises_error(self, mock_harness):
        """Test that run() raises ValueError if LLM is not configured."""
        agent = MemoryAwareAgent(harness=mock_harness, llm=None)

        with pytest.raises(ValueError, match="LLM is required"):
            await agent.run(query="test", thread_id="thread-1")

    @pytest.mark.asyncio
    async def test_before_phase(self, mock_harness, mock_llm):
        """Test BEFORE phase: context assembly and entity extraction."""
        agent = MemoryAwareAgent(harness=mock_harness, llm=mock_llm)

        # Mock context assembly
        mock_ctx = AssembledContext(
            user_query="Test query",
            context_usage_percent=0.5,
        )
        agent._context_agent.assemble = AsyncMock(return_value=mock_ctx)

        # Mock entity extraction
        agent._entity_extractor.extract_entities = AsyncMock(
            return_value={"people": ["John"], "organizations": [], "locations": []}
        )

        # Mock LLM to return no tool calls (final answer immediately)
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Final answer"
        mock_llm.ainvoke.return_value = mock_response

        # Mock get_memory_tools
        with patch("memharness.tools.get_memory_tools", return_value=[]):
            result = await agent.run(query="Test query", thread_id="thread-1")

        # Verify BEFORE phase operations
        agent._context_agent.assemble.assert_called_once()
        agent._entity_extractor.extract_entities.assert_called()
        mock_harness.add_entity.assert_called()  # Entity added
        assert result.answer == "Final answer"

    @pytest.mark.asyncio
    async def test_before_phase_with_summarization(self, mock_harness, mock_llm):
        """Test BEFORE phase triggers summarization when context >80%."""
        agent = MemoryAwareAgent(
            harness=mock_harness,
            llm=mock_llm,
            summarize_threshold=0.8,
        )

        # Mock first context assembly with high usage
        mock_ctx_high = AssembledContext(
            user_query="Test query",
            context_usage_percent=0.9,  # >80%
        )
        # Mock second context assembly after summarization
        mock_ctx_low = AssembledContext(
            user_query="Test query",
            context_usage_percent=0.5,  # After summarization
        )
        agent._context_agent.assemble = AsyncMock(side_effect=[mock_ctx_high, mock_ctx_low])

        # Mock summarizer
        agent._summarizer.summarize_thread = AsyncMock()

        # Mock entity extraction
        agent._entity_extractor.extract_entities = AsyncMock(
            return_value={"people": [], "organizations": [], "locations": []}
        )

        # Mock LLM to return final answer
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Answer"
        mock_llm.ainvoke.return_value = mock_response

        with patch("memharness.tools.get_memory_tools", return_value=[]):
            await agent.run(query="Test", thread_id="thread-1")

        # Verify summarization was triggered
        agent._summarizer.summarize_thread.assert_called_once_with("thread-1")
        # Context assembly should be called twice (before and after summarization)
        assert agent._context_agent.assemble.call_count == 2

    @pytest.mark.asyncio
    async def test_inside_phase_with_tool_calls(self, mock_harness, mock_llm):
        """Test INSIDE phase: LLM loop with tool execution."""
        agent = MemoryAwareAgent(harness=mock_harness, llm=mock_llm)

        # Mock context assembly
        mock_ctx = AssembledContext(user_query="Test", context_usage_percent=0.5)
        agent._context_agent.assemble = AsyncMock(return_value=mock_ctx)

        # Mock entity extraction
        agent._entity_extractor.extract_entities = AsyncMock(
            return_value={"people": [], "organizations": [], "locations": []}
        )

        # Mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.ainvoke = AsyncMock(return_value="Tool result")

        # Mock LLM responses:
        # 1. First call: tool call
        # 2. Second call: final answer
        mock_response_with_tool = MagicMock()
        mock_response_with_tool.tool_calls = [
            {"name": "test_tool", "args": {"arg1": "value1"}, "id": "call-123"}
        ]
        mock_response_with_tool.content = None

        mock_response_final = MagicMock()
        mock_response_final.tool_calls = []
        mock_response_final.content = "Final answer"

        mock_llm.ainvoke.side_effect = [mock_response_with_tool, mock_response_final]

        with patch("memharness.tools.get_memory_tools", return_value=[mock_tool]):
            result = await agent.run(query="Test", thread_id="thread-1")

        # Verify tool was called
        mock_tool.ainvoke.assert_called_once()
        assert result.tool_calls == 1
        assert len(result.steps) == 1
        assert "test_tool" in result.steps[0]
        assert "success" in result.steps[0]

        # Verify tool log was written
        mock_harness.add_tool_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_after_phase(self, mock_harness, mock_llm):
        """Test AFTER phase: workflow save and entity extraction."""
        agent = MemoryAwareAgent(harness=mock_harness, llm=mock_llm)

        # Mock context assembly
        mock_ctx = AssembledContext(user_query="Test", context_usage_percent=0.5)
        agent._context_agent.assemble = AsyncMock(return_value=mock_ctx)

        # Mock entity extraction
        # First call (from query): return 1 entity
        # Second call (from response): return 2 entities
        agent._entity_extractor.extract_entities = AsyncMock(
            side_effect=[
                {"people": ["Alice"], "organizations": [], "locations": []},
                {
                    "people": ["Bob"],
                    "organizations": ["Acme"],
                    "locations": [],
                },
            ]
        )

        # Mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.ainvoke = AsyncMock(return_value="Tool result")

        # Mock LLM with tool call then answer
        mock_response_tool = MagicMock()
        mock_response_tool.tool_calls = [{"name": "test_tool", "args": {}, "id": "call-123"}]

        mock_response_final = MagicMock()
        mock_response_final.tool_calls = []
        mock_response_final.content = "Final answer"

        mock_llm.ainvoke.side_effect = [mock_response_tool, mock_response_final]

        with patch("memharness.tools.get_memory_tools", return_value=[mock_tool]):
            result = await agent.run(query="Test", thread_id="thread-1")

        # Verify AFTER phase operations

        # 1. Workflow saved (because steps were taken)
        mock_harness.add_workflow.assert_called_once()
        assert result.workflow_saved is True

        # 2. Entities extracted from both query and response
        assert agent._entity_extractor.extract_entities.call_count == 2
        # add_entity called for Alice, Bob, and Acme (3 total)
        assert mock_harness.add_entity.call_count == 3

        # 3. Assistant response saved
        calls = list(mock_harness.add_conversational.call_args_list)
        # The user message is saved by ContextAssemblyAgent (which is mocked),
        # so we only expect the assistant response from the AFTER phase
        assert len(calls) == 1
        # Last (and only) call should be assistant message
        assert calls[-1][0][1] == "assistant"
        assert calls[-1][0][2] == "Final answer"

    @pytest.mark.asyncio
    async def test_max_iterations(self, mock_harness, mock_llm):
        """Test that agent stops after max_iterations."""
        agent = MemoryAwareAgent(
            harness=mock_harness,
            llm=mock_llm,
            max_iterations=3,
        )

        # Mock context assembly
        mock_ctx = AssembledContext(user_query="Test", context_usage_percent=0.5)
        agent._context_agent.assemble = AsyncMock(return_value=mock_ctx)

        # Mock entity extraction
        agent._entity_extractor.extract_entities = AsyncMock(
            return_value={"people": [], "organizations": [], "locations": []}
        )

        # Mock LLM to always return tool calls (never final answer)
        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "fake_tool", "args": {}, "id": "call-1"}]
        mock_llm.ainvoke.return_value = mock_response

        with patch("memharness.tools.get_memory_tools", return_value=[]):
            result = await agent.run(query="Test", thread_id="thread-1")

        # Verify max iterations reached
        assert result.iterations == 3
        assert "unable to complete" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_harness, mock_llm):
        """Test using a custom system prompt."""
        agent = MemoryAwareAgent(harness=mock_harness, llm=mock_llm)

        # Mock context assembly
        mock_ctx = AssembledContext(user_query="Test", context_usage_percent=0.5)
        agent._context_agent.assemble = AsyncMock(return_value=mock_ctx)

        # Mock entity extraction
        agent._entity_extractor.extract_entities = AsyncMock(
            return_value={"people": [], "organizations": [], "locations": []}
        )

        # Mock LLM
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Answer"
        mock_llm.ainvoke.return_value = mock_response

        custom_prompt = "You are a custom agent."

        with patch("memharness.tools.get_memory_tools", return_value=[]):
            await agent.run(
                query="Test",
                thread_id="thread-1",
                system_prompt=custom_prompt,
            )

        # Verify custom system prompt was used
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert any(custom_prompt in str(msg) for msg in messages)


def test_system_prompt_constant():
    """Test that AGENT_SYSTEM_PROMPT constant is defined."""
    assert AGENT_SYSTEM_PROMPT is not None
    assert len(AGENT_SYSTEM_PROMPT) > 100
    assert "memory-aware" in AGENT_SYSTEM_PROMPT.lower()
    assert "Role" in AGENT_SYSTEM_PROMPT
    assert "Memory Store Semantics" in AGENT_SYSTEM_PROMPT
