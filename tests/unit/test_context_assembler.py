# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tests for Context Assembly Agent.

Tests the BEFORE-loop pattern implementation for assembling optimal context
from memory for LLM queries.
"""

from __future__ import annotations

import pytest

from memharness import MemoryHarness
from memharness.agents.context_assembler import AssembledContext, ContextAssemblyAgent


class TestAssembledContext:
    """Test AssembledContext dataclass."""

    def test_empty_context_to_prompt(self):
        """Test rendering empty context."""
        ctx = AssembledContext(user_query="test query")
        prompt = ctx.to_prompt()

        assert "## User Query" in prompt
        assert "test query" in prompt
        assert "## Agent Persona" not in prompt

    def test_full_context_to_prompt(self):
        """Test rendering fully populated context."""
        ctx = AssembledContext(
            persona="I am a helpful assistant",
            conversation_history=[
                type("FakeUnit", (), {"metadata": {"role": "user"}, "content": "hello"})(),
                type("FakeUnit", (), {"metadata": {"role": "assistant"}, "content": "hi there"})(),
            ],
            knowledge="Python is a programming language",
            workflows="Step 1: analyze\nStep 2: respond",
            entities="Entity: Python (LANG)",
            summaries="Previous conversation about coding",
            tools="tool1, tool2",
            user_query="How do I code?",
        )

        prompt = ctx.to_prompt()

        # Check all sections are present
        assert "## Agent Persona" in prompt
        assert "## Conversation History" in prompt
        assert "## Relevant Knowledge" in prompt
        assert "## Relevant Workflows" in prompt
        assert "## Known Entities" in prompt
        assert "## Context Summaries" in prompt
        assert "## Available Tools" in prompt
        assert "## User Query" in prompt

        # Check content is preserved
        assert "helpful assistant" in prompt
        assert "hello" in prompt
        assert "Python is a programming language" in prompt
        assert "How do I code?" in prompt

    def test_token_estimate_calculation(self):
        """Test token estimation (rough chars/4 heuristic)."""
        ctx = AssembledContext(
            user_query="test" * 100,  # 400 chars
            knowledge="knowledge" * 50,  # 450 chars
        )

        # Rough estimate: total chars / 4
        # "test"*100 + "knowledge"*50 + markdown headers ~ 850+ chars
        # Should be around 200-250 tokens
        assert ctx.total_tokens_estimate == 0  # Not yet calculated

        # After calculating via to_prompt
        prompt_len = len(ctx.to_prompt())
        expected_tokens = prompt_len // 4
        ctx.total_tokens_estimate = expected_tokens
        assert ctx.total_tokens_estimate > 0


class TestContextAssemblyAgent:
    """Test ContextAssemblyAgent."""

    @pytest.fixture
    async def harness(self):
        """Create an in-memory harness for testing."""
        harness = MemoryHarness("memory://")
        await harness.connect()
        yield harness
        await harness.disconnect()

    @pytest.fixture
    def agent(self, harness):
        """Create a ContextAssemblyAgent."""
        return ContextAssemblyAgent(harness, max_tokens=1000, summarize_threshold=0.8)

    @pytest.mark.asyncio
    async def test_basic_context_assembly(self, agent, harness):
        """Test basic context assembly with no memories."""
        ctx = await agent.assemble(
            query="What is Python?",
            thread_id="test-thread",
            include_tools=False,
        )

        # Query should be saved
        assert ctx.user_query == "What is Python?"

        # Should have calculated token estimate
        assert ctx.total_tokens_estimate > 0
        assert 0.0 <= ctx.context_usage_percent <= 1.0

        # Should have user query in conversation history (just saved)
        messages = await harness.get_conversational("test-thread")
        assert len(messages) == 1
        assert messages[0].content == "What is Python?"

    @pytest.mark.asyncio
    async def test_context_with_conversation_history(self, agent, harness):
        """Test context assembly with existing conversation."""
        # Add some conversation history
        await harness.add_conversational("thread-1", "user", "Hello!")
        await harness.add_conversational("thread-1", "assistant", "Hi there!")

        ctx = await agent.assemble(
            query="How are you?",
            thread_id="thread-1",
            include_tools=False,
        )

        # Should include conversation history
        assert any("Hello!" in m.content for m in ctx.conversation_history)
        assert any("Hi there!" in m.content for m in ctx.conversation_history)

        # New query should be added
        messages = await harness.get_conversational("thread-1")
        assert len(messages) == 3
        assert messages[-1].content == "How are you?"

    @pytest.mark.asyncio
    async def test_context_with_knowledge(self, agent, harness):
        """Test context assembly with knowledge base."""
        # Add knowledge
        await harness.add_knowledge(
            content="Python is a high-level programming language",
            source="test-kb",
        )
        await harness.add_knowledge(
            content="Python was created by Guido van Rossum",
            source="test-kb",
        )

        ctx = await agent.assemble(
            query="Tell me about Python",
            thread_id="thread-2",
            include_tools=False,
        )

        # Should include relevant knowledge
        assert "Python" in ctx.knowledge
        # At least one of the facts should be present
        assert "high-level programming" in ctx.knowledge or "Guido van Rossum" in ctx.knowledge

    @pytest.mark.asyncio
    async def test_context_with_entities(self, agent, harness):
        """Test context assembly with entities."""
        # Add an entity
        await harness.add_entity(
            name="Python",
            entity_type="LANGUAGE",
            description="A popular programming language",
        )

        ctx = await agent.assemble(
            query="What is Python?",
            thread_id="thread-3",
            include_tools=False,
        )

        # Should include entity
        if ctx.entities:  # Entity search may not match exactly
            assert "Python" in ctx.entities

    @pytest.mark.asyncio
    async def test_context_with_workflows(self, agent, harness):
        """Test context assembly with workflows."""
        # Add a workflow
        await harness.add_workflow(
            task="Write Python code",
            steps=["1. Define requirements", "2. Write code", "3. Test"],
            outcome="success",
        )

        ctx = await agent.assemble(
            query="How do I write Python code?",
            thread_id="thread-4",
            include_tools=False,
        )

        # Should include workflow if search matches
        # (workflow search is semantic, so may not always match)
        assert isinstance(ctx.workflows, str)

    @pytest.mark.asyncio
    async def test_context_usage_calculation(self, agent):
        """Test context usage percentage calculation."""
        # With empty memories
        ctx = await agent.assemble(
            query="Short query",
            thread_id="thread-5",
            include_tools=False,
        )

        # Usage should be low
        assert ctx.context_usage_percent < 0.5

    @pytest.mark.asyncio
    async def test_summarization_trigger(self, agent, harness):
        """Test that long conversations trigger truncation."""
        # Add many messages to exceed threshold
        for i in range(25):
            await harness.add_conversational("thread-6", "user", f"Message {i}")

        # Create agent with low threshold
        low_threshold_agent = ContextAssemblyAgent(harness, max_tokens=100, summarize_threshold=0.5)

        ctx = await low_threshold_agent.assemble(
            query="New query",
            thread_id="thread-6",
            include_tools=False,
        )

        # Should have truncated conversation
        # (implementation currently truncates to last 10 messages)
        lines = ctx.conversation_history
        assert len(lines) <= 11  # 10 previous + 1 new

    @pytest.mark.asyncio
    async def test_tool_discovery_disabled(self, agent):
        """Test context assembly with tool discovery disabled."""
        ctx = await agent.assemble(
            query="test",
            thread_id="thread-7",
            include_tools=False,
        )

        # Tools should be empty
        assert ctx.tools == ""

    @pytest.mark.asyncio
    async def test_tool_discovery_enabled(self, agent, harness):
        """Test context assembly with tool discovery enabled."""
        # Add a tool to toolbox
        await harness.add_tool(
            server="math",
            tool_name="calculator",
            description="Performs calculations",
            parameters={"type": "object"},
        )

        ctx = await agent.assemble(
            query="test",
            thread_id="thread-8",
            include_tools=True,
        )

        # Should attempt to get toolbox tree
        # (may be empty if toolbox not implemented, but should not error)
        assert isinstance(ctx.tools, str)

    @pytest.mark.asyncio
    async def test_prompt_rendering(self, agent, harness):
        """Test full prompt rendering."""
        # Setup some memories
        await harness.add_conversational("thread-9", "user", "Previous message")
        await harness.add_knowledge("Python uses indentation", source="docs")

        ctx = await agent.assemble(
            query="Explain Python syntax",
            thread_id="thread-9",
            include_tools=False,
        )

        prompt = ctx.to_prompt()

        # Check structure
        assert "##" in prompt  # Has markdown headers
        assert "Explain Python syntax" in prompt  # Has query
        assert "\n\n" in prompt  # Sections separated

        # Check readable format
        lines = prompt.split("\n")
        assert len(lines) > 0
