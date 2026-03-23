# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Unit tests for the agent workflow (LangGraph integration).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memharness.agents.agent_workflow import (
    AgentState,
    assemble_context,
    call_agent_node,
    check_context,
    create_memory_agent,
    extract_entities,
    save_response,
    save_user_message,
    save_workflow,
    summarize_thread,
)
from memharness.core.harness import MemoryHarness


@pytest.fixture
def mock_harness():
    """Create a mock MemoryHarness."""
    harness = MagicMock(spec=MemoryHarness)
    harness.add_conversational = AsyncMock()
    harness.add_entity = AsyncMock()
    harness.add_workflow = AsyncMock()
    harness.log_tool_execution = AsyncMock(return_value="log-123")
    return harness


@pytest.fixture
def base_state() -> AgentState:
    """Create a base agent state for testing."""
    return {
        "messages": [],
        "thread_id": "test-thread",
        "query": "What is Python?",
        "context_usage": 0.5,
        "steps": [],
        "entities_extracted": 0,
        "workflow_saved": False,
        "final_answer": "",
        "max_context_tokens": 4000,
        "summarize_threshold": 0.8,
        "system_prompt": None,
    }


# ============================================================================
# NODE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_save_user_message(mock_harness: MagicMock, base_state: AgentState):
    """Test save_user_message node."""
    result = await save_user_message(base_state, mock_harness)

    # Should save to conversational memory
    mock_harness.add_conversational.assert_called_once_with(
        "test-thread", "user", "What is Python?"
    )

    # Should return empty dict (no state updates)
    assert result == {}


@pytest.mark.asyncio
async def test_assemble_context(mock_harness: MagicMock, base_state: AgentState):
    """Test assemble_context node."""
    # Mock the ContextAssemblyAgent
    with patch("memharness.agents.context_assembler.ContextAssemblyAgent") as mock_assembler:
        # Create mock context
        mock_ctx = MagicMock()
        mock_ctx.to_messages.return_value = [
            SystemMessage(content="Context here"),
            HumanMessage(content="What is Python?"),
        ]
        mock_ctx.context_usage_percent = 0.6

        # Setup mock
        mock_assembler.return_value.assemble = AsyncMock(return_value=mock_ctx)

        # Call node
        result = await assemble_context(base_state, mock_harness)

        # Should return messages and context usage
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["context_usage"] == 0.6


def test_check_context_summarize(base_state: AgentState):
    """Test check_context routing function when context is high."""
    base_state["context_usage"] = 0.85
    result = check_context(base_state)
    assert result == "summarize"


def test_check_context_proceed(base_state: AgentState):
    """Test check_context routing function when context is low."""
    base_state["context_usage"] = 0.5
    result = check_context(base_state)
    assert result == "proceed"


@pytest.mark.asyncio
async def test_summarize_thread(mock_harness: MagicMock, base_state: AgentState):
    """Test summarize_thread node."""
    with patch("memharness.agents.summarizer.SummarizerAgent") as mock_summarizer:
        mock_summarizer.return_value.summarize_thread = AsyncMock(
            return_value={
                "summarized": True,
                "summary_id": "sum-123",
                "messages_summarized": 20,
            }
        )

        result = await summarize_thread(base_state, mock_harness)

        # Should call summarizer
        mock_summarizer.return_value.summarize_thread.assert_called_once_with(
            "test-thread", max_messages=50
        )

        # Should return empty dict
        assert result == {}


@pytest.mark.asyncio
async def test_call_agent_node_no_tools(mock_harness: MagicMock, base_state: AgentState):
    """Test call_agent_node without tool calls."""
    # Mock LLM
    mock_llm = AsyncMock()
    mock_response = AIMessage(content="Python is a programming language.")
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    base_state["messages"] = [HumanMessage(content="What is Python?")]

    result = await call_agent_node(base_state, mock_harness, mock_llm, tools=None)

    # Should return final answer
    assert result["final_answer"] == "Python is a programming language."
    assert len(result["messages"]) == 2  # Original message + response


@pytest.mark.asyncio
async def test_call_agent_node_with_tools(mock_harness: MagicMock, base_state: AgentState):
    """Test call_agent_node with tool execution."""
    # Mock tool
    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.ainvoke = AsyncMock(return_value="Search results here")

    # Mock LLM with tool calls
    mock_llm = AsyncMock()
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-123",
                "name": "search",
                "args": {"query": "Python"},
            }
        ],
    )
    final_response = AIMessage(content="Based on search, Python is a language.")

    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

    base_state["messages"] = [HumanMessage(content="What is Python?")]

    result = await call_agent_node(
        base_state, mock_harness, mock_llm, tools=[mock_tool], max_iterations=10
    )

    # Should execute tool
    mock_tool.ainvoke.assert_called_once_with({"query": "Python"})

    # Should log tool execution
    mock_harness.log_tool_execution.assert_called_once()

    # Should have steps
    assert len(result["steps"]) == 1
    assert "search" in result["steps"][0]

    # Should have final answer
    assert result["final_answer"] == "Based on search, Python is a language."


@pytest.mark.asyncio
async def test_save_response(mock_harness: MagicMock, base_state: AgentState):
    """Test save_response node."""
    base_state["final_answer"] = "This is the answer."

    result = await save_response(base_state, mock_harness)

    # Should save to conversational memory
    mock_harness.add_conversational.assert_called_once_with(
        "test-thread", "assistant", "This is the answer."
    )

    # Should return empty dict
    assert result == {}


@pytest.mark.asyncio
async def test_extract_entities(mock_harness: MagicMock, base_state: AgentState):
    """Test extract_entities node."""
    base_state["final_answer"] = "John works at Google in California."

    with patch("memharness.agents.entity_extractor.EntityExtractorAgent") as mock_extractor:
        mock_extractor.return_value.extract_entities = AsyncMock(
            return_value={
                "people": ["John"],
                "organizations": ["Google"],
                "locations": ["California"],
            }
        )

        result = await extract_entities(base_state, mock_harness)

        # Should extract entities
        mock_extractor.return_value.extract_entities.assert_called_once_with(
            "John works at Google in California."
        )

        # Should save entities
        assert mock_harness.add_entity.call_count == 3

        # Should return count
        assert result["entities_extracted"] == 3


@pytest.mark.asyncio
async def test_save_workflow_with_steps(mock_harness: MagicMock, base_state: AgentState):
    """Test save_workflow node with steps."""
    base_state["steps"] = ["search(query) → success", "fetch(url) → success"]
    base_state["final_answer"] = "Done"

    result = await save_workflow(base_state, mock_harness)

    # Should save workflow
    mock_harness.add_workflow.assert_called_once()

    # Should return True
    assert result["workflow_saved"] is True


@pytest.mark.asyncio
async def test_save_workflow_no_steps(mock_harness: MagicMock, base_state: AgentState):
    """Test save_workflow node without steps."""
    base_state["steps"] = []

    result = await save_workflow(base_state, mock_harness)

    # Should not save workflow
    mock_harness.add_workflow.assert_not_called()

    # Should return False
    assert result["workflow_saved"] is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_create_memory_agent(mock_harness: MagicMock):
    """Test create_memory_agent factory function."""
    # Mock LLM
    mock_llm = MagicMock()

    # Create agent
    graph = create_memory_agent(
        harness=mock_harness,
        llm=mock_llm,
        tools=None,
        max_context_tokens=4000,
        summarize_threshold=0.8,
    )

    # Should return compiled graph
    assert graph is not None
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_agent_workflow_end_to_end(mock_harness: MagicMock):
    """Test the complete agent workflow (mocked)."""
    # This is a simplified end-to-end test
    # In reality, you'd want to use a real backend and LLM

    # Mock LLM
    mock_llm = AsyncMock()
    mock_response = AIMessage(content="Test response")
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    # Mock ContextAssemblyAgent
    with patch("memharness.agents.context_assembler.ContextAssemblyAgent") as mock_assembler:
        mock_ctx = MagicMock()
        mock_ctx.to_messages.return_value = [
            SystemMessage(content="Context"),
            HumanMessage(content="Test query"),
        ]
        mock_ctx.context_usage_percent = 0.5
        mock_assembler.return_value.assemble = AsyncMock(return_value=mock_ctx)

        # Mock EntityExtractorAgent
        with patch(
            "memharness.agents.entity_extractor.EntityExtractorAgent"
        ) as mock_entity_extractor:
            mock_entity_extractor.return_value.extract_entities = AsyncMock(
                return_value={"people": [], "organizations": [], "locations": []}
            )

            # Create graph
            graph = create_memory_agent(
                harness=mock_harness,
                llm=mock_llm,
                tools=None,
                max_context_tokens=4000,
                summarize_threshold=0.8,
            )

            # Invoke graph
            result = await graph.ainvoke(
                {
                    "messages": [],
                    "thread_id": "test-thread",
                    "query": "Test query",
                    "context_usage": 0.0,
                    "steps": [],
                    "entities_extracted": 0,
                    "workflow_saved": False,
                    "final_answer": "",
                    "max_context_tokens": 4000,
                    "summarize_threshold": 0.8,
                    "system_prompt": None,
                }
            )

            # Should have final answer
            assert result["final_answer"] == "Test response"

            # Should have called add_conversational at least once (for user message)
            assert mock_harness.add_conversational.call_count >= 1
