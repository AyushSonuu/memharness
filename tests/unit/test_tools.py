# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Tests for LangChain-based memory tools.
"""

import pytest

from memharness import MemoryHarness
from memharness.tools import (
    LANGCHAIN_AVAILABLE,
    AssembleContextTool,
    ConversationHistoryTool,
    ExpandSummaryTool,
    MemoryReadTool,
    MemorySearchTool,
    MemoryStatsTool,
    MemoryWriteTool,
    ToolboxGrepTool,
    ToolboxTreeTool,
    get_memory_tools,
)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
class TestMemoryTools:
    """Test LangChain-based memory tools."""

    @pytest.fixture
    async def harness(self):
        """Create a memory harness for testing."""
        h = MemoryHarness("memory://")
        await h.connect()
        yield h
        await h.disconnect()

    async def test_get_memory_tools(self, harness):
        """Test get_memory_tools returns all tools."""
        tools = get_memory_tools(harness)

        assert len(tools) == 9
        assert all(hasattr(tool, "name") for tool in tools)
        assert all(hasattr(tool, "description") for tool in tools)
        assert all(hasattr(tool, "_arun") for tool in tools)

        tool_names = {tool.name for tool in tools}
        expected_names = {
            "memory_search",
            "memory_read",
            "memory_write",
            "memory_stats",
            "toolbox_tree",
            "toolbox_grep",
            "expand_summary",
            "get_conversation_history",
            "assemble_context",
        }
        assert tool_names == expected_names

    async def test_memory_search_tool(self, harness):
        """Test MemorySearchTool."""
        # Add some test data
        await harness.add_knowledge(
            content="Python is a programming language", metadata={"topic": "programming"}
        )
        await harness.add_knowledge(
            content="Machine learning is a subset of AI", metadata={"topic": "ai"}
        )

        tool = MemorySearchTool(harness=harness)

        # Test search
        result = await tool._arun(query="programming", k=1)
        assert isinstance(result, str)
        assert "Python" in result or "programming" in result.lower()

    async def test_memory_read_tool(self, harness):
        """Test MemoryReadTool."""
        # Add test data
        memory_id = await harness.add_knowledge(
            content="Test content for reading", metadata={"test": True}
        )

        tool = MemoryReadTool(harness=harness)

        # Test read
        result = await tool._arun(memory_id=memory_id)
        assert isinstance(result, str)
        assert "Test content for reading" in result

    async def test_memory_write_tool(self, harness):
        """Test MemoryWriteTool."""
        tool = MemoryWriteTool(harness=harness)

        # Test write
        result = await tool._arun(
            memory_type="knowledge_base",
            content="New knowledge to store",
            metadata={"source": "test"},
        )

        assert isinstance(result, str)
        assert "successfully" in result.lower()
        assert "ID:" in result

    async def test_memory_stats_tool(self, harness):
        """Test MemoryStatsTool."""
        # Add some data
        await harness.add_knowledge(content="Test 1")
        await harness.add_knowledge(content="Test 2")
        await harness.add_entity(name="Test Entity", entity_type="person", description="Info")

        tool = MemoryStatsTool(harness=harness)

        # Test stats
        result = await tool._arun()
        assert isinstance(result, str)
        assert "Statistics" in result or "Total" in result

    async def test_toolbox_tree_tool(self, harness):
        """Test ToolboxTreeTool."""
        # Add some toolbox entries
        await harness.add_tool(
            server="test_server",
            tool_name="test_tool",
            description="A test tool",
            parameters={"type": "function"},
        )

        tool = ToolboxTreeTool(harness=harness)

        # Test tree
        result = await tool._arun(path="/", depth=2)
        assert isinstance(result, str)

    async def test_toolbox_grep_tool(self, harness):
        """Test ToolboxGrepTool."""
        # Add toolbox entries
        await harness.add_tool(
            server="test_server",
            tool_name="search_tool",
            description="A tool for searching",
            parameters={"type": "function"},
        )
        await harness.add_tool(
            server="test_server",
            tool_name="write_tool",
            description="A tool for writing",
            parameters={"type": "function"},
        )

        tool = ToolboxGrepTool(harness=harness)

        # Test grep
        result = await tool._arun(pattern="search", case_sensitive=False)
        assert isinstance(result, str)

    async def test_tool_with_no_results(self, harness):
        """Test tool behavior when no results found."""
        tool = MemorySearchTool(harness=harness)

        # Search in empty memory
        result = await tool._arun(query="nonexistent content", k=5)
        assert isinstance(result, str)
        assert "No memories found" in result or "not found" in result.lower()

    async def test_tool_sync_raises_not_implemented(self, harness):
        """Test that sync methods raise NotImplementedError."""
        tool = MemorySearchTool(harness=harness)

        with pytest.raises(NotImplementedError):
            tool._run(query="test")

    async def test_tool_has_proper_schema(self, harness):
        """Test that tools have proper Pydantic schemas."""
        tool = MemorySearchTool(harness=harness)

        # Check that args_schema exists and is a Pydantic model
        assert hasattr(tool, "args_schema")
        assert tool.args_schema is not None

        # Check that the schema has the expected fields
        schema = tool.args_schema.model_json_schema()
        assert "properties" in schema
        assert "query" in schema["properties"]

    async def test_expand_summary_tool(self, harness):
        """Test ExpandSummaryTool."""
        # Create some messages
        msg1 = await harness.add_conversational("thread1", "user", "Hello, how are you?")
        msg2 = await harness.add_conversational("thread1", "assistant", "I'm doing well, thanks!")
        msg3 = await harness.add_conversational("thread1", "user", "What's the weather like?")

        # Create a summary
        summary_id = await harness.add_summary(
            summary="Conversation about greetings and weather query",
            source_ids=[msg1, msg2, msg3],
            thread_id="thread1",
        )

        tool = ExpandSummaryTool(harness=harness)

        # Test expand
        result = await tool._arun(summary_id=summary_id)
        assert isinstance(result, str)
        assert "Hello, how are you?" in result or "greetings" in result.lower()

    async def test_expand_summary_tool_not_found(self, harness):
        """Test ExpandSummaryTool with non-existent summary."""
        tool = ExpandSummaryTool(harness=harness)

        # Test with non-existent ID
        result = await tool._arun(summary_id="non-existent-id")
        assert isinstance(result, str)
        assert "Error:" in result or "not found" in result.lower()

    async def test_conversation_history_tool(self, harness):
        """Test ConversationHistoryTool."""
        # Add conversation messages
        await harness.add_conversational("thread1", "user", "Hello!")
        await harness.add_conversational("thread1", "assistant", "Hi there!")
        await harness.add_conversational("thread1", "user", "How can you help me?")

        tool = ConversationHistoryTool(harness=harness)

        # Test conversation history retrieval
        result = await tool._arun(thread_id="thread1", limit=10)
        assert isinstance(result, str)
        assert "Hello!" in result or "thread1" in result

    async def test_conversation_history_tool_empty(self, harness):
        """Test ConversationHistoryTool with empty thread."""
        tool = ConversationHistoryTool(harness=harness)

        # Test with non-existent thread
        result = await tool._arun(thread_id="non-existent-thread", limit=10)
        assert isinstance(result, str)
        assert "No conversation history" in result or "not found" in result.lower()

    async def test_assemble_context_tool(self, harness):
        """Test AssembleContextTool."""
        # Add some data for context assembly
        await harness.add_conversational("thread1", "user", "Tell me about Python")
        await harness.add_conversational("thread1", "assistant", "Python is a programming language")
        await harness.add_knowledge(content="Python was created by Guido van Rossum")
        await harness.add_entity(
            name="Python", entity_type="programming_language", description="High-level language"
        )

        tool = AssembleContextTool(harness=harness)

        # Test context assembly
        result = await tool._arun(query="Python programming", thread_id="thread1", max_tokens=4000)
        assert isinstance(result, str)
        # Should contain some assembled context
        assert len(result) > 0

    async def test_assemble_context_tool_empty(self, harness):
        """Test AssembleContextTool with no context."""
        tool = AssembleContextTool(harness=harness)

        # Test with empty memory
        result = await tool._arun(query="something", thread_id="empty-thread", max_tokens=4000)
        assert isinstance(result, str)
        # Should still return something (even if minimal)


@pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="Test behavior when langchain not available")
def test_import_error_without_langchain():
    """Test that ImportError is raised when langchain-core is not installed."""
    harness = MemoryHarness("memory://")

    with pytest.raises(ImportError, match="langchain-core"):
        get_memory_tools(harness)
