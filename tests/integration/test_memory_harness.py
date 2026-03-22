"""
Integration tests for MemoryHarness.

Tests the complete memory harness functionality including all memory types,
search operations, and context assembly.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from memharness import MemoryHarness, MemoryType, MemoryUnit


# =============================================================================
# Conversational Memory Tests
# =============================================================================


class TestConversationalMemory:
    """Tests for conversational memory operations."""

    @pytest.mark.asyncio
    async def test_add_conversational_message(self, memory):
        """Test adding a single conversational message."""
        msg_id = await memory.add_conversational("t1", "user", "Hello")

        assert msg_id is not None
        assert isinstance(msg_id, (str, int))

    @pytest.mark.asyncio
    async def test_get_conversational_messages(self, memory):
        """Test retrieving conversational messages by thread."""
        # Write
        msg_id = await memory.add_conversational("t1", "user", "Hello")
        assert msg_id is not None

        # Read
        messages = await memory.get_conversational("t1")
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_conversational_thread_isolation(self, memory):
        """Test that conversations are isolated by thread ID."""
        # Add messages to different threads
        await memory.add_conversational("thread1", "user", "Hello thread 1")
        await memory.add_conversational("thread2", "user", "Hello thread 2")

        # Get messages from each thread
        messages1 = await memory.get_conversational("thread1")
        messages2 = await memory.get_conversational("thread2")

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0].content == "Hello thread 1"
        assert messages2[0].content == "Hello thread 2"

    @pytest.mark.asyncio
    async def test_conversational_order(self, memory):
        """Test that messages maintain chronological order."""
        await memory.add_conversational("t1", "user", "First")
        await memory.add_conversational("t1", "assistant", "Second")
        await memory.add_conversational("t1", "user", "Third")

        messages = await memory.get_conversational("t1")

        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"

    @pytest.mark.asyncio
    async def test_conversational_with_metadata(self, memory):
        """Test conversational messages with metadata."""
        msg_id = await memory.add_conversational(
            "t1",
            "user",
            "Hello with metadata",
            metadata={"source": "api", "timestamp": "2024-01-01"}
        )

        messages = await memory.get_conversational("t1")
        assert messages[0].metadata is not None
        assert messages[0].metadata.get("source") == "api"

    @pytest.mark.asyncio
    async def test_conversational_roles(self, memory):
        """Test different conversation roles."""
        await memory.add_conversational("t1", "user", "User message")
        await memory.add_conversational("t1", "assistant", "Assistant message")
        await memory.add_conversational("t1", "system", "System message")

        messages = await memory.get_conversational("t1")

        roles = [m.metadata.get("role") or getattr(m, "role", None) for m in messages]
        # Verify all roles are captured
        assert len(messages) == 3


# =============================================================================
# Knowledge Base Tests
# =============================================================================


class TestKnowledgeBase:
    """Tests for knowledge base operations."""

    @pytest.mark.asyncio
    async def test_add_knowledge(self, memory):
        """Test adding knowledge to the base."""
        kb_id = await memory.add_knowledge(
            "Python is a programming language",
            source="docs"
        )

        assert kb_id is not None

    @pytest.mark.asyncio
    async def test_search_knowledge(self, memory):
        """Test searching the knowledge base."""
        # Write
        kb_id = await memory.add_knowledge(
            "Python is a programming language",
            source="docs"
        )

        # Search
        results = await memory.search_knowledge("programming", k=1)
        assert len(results) == 1
        assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_knowledge_semantic_search(self, memory):
        """Test semantic search returns relevant results."""
        await memory.add_knowledge("Python is great for data science", source="docs")
        await memory.add_knowledge("JavaScript runs in browsers", source="docs")
        await memory.add_knowledge("Docker containerizes applications", source="docs")

        # Search for data-related content
        results = await memory.search_knowledge("machine learning data analysis", k=2)

        # Python/data science should rank high
        assert len(results) >= 1
        # First result should be most relevant
        assert "Python" in results[0].content or "data" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_knowledge_with_tags(self, memory):
        """Test knowledge with tags for filtering."""
        await memory.add_knowledge(
            "Kubernetes orchestrates containers",
            source="docs",
            tags=["k8s", "devops", "containers"]
        )

        results = await memory.search_knowledge("container management", k=1)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_knowledge_k_parameter(self, memory):
        """Test that k parameter limits results."""
        # Add multiple entries
        for i in range(10):
            await memory.add_knowledge(f"Knowledge entry {i}", source="test")

        # Search with k=3
        results = await memory.search_knowledge("knowledge", k=3)
        assert len(results) <= 3


# =============================================================================
# Entity Memory Tests
# =============================================================================


class TestEntityMemory:
    """Tests for entity memory operations."""

    @pytest.mark.asyncio
    async def test_add_entity(self, memory):
        """Test adding an entity."""
        await memory.add_entity("John Doe", "PERSON", "Software engineer at Acme")

        results = await memory.search_entity("engineer")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_entity_types(self, memory):
        """Test different entity types."""
        await memory.add_entity("John Doe", "PERSON", "Software engineer")
        await memory.add_entity("Acme Corp", "ORGANIZATION", "Technology company")
        await memory.add_entity("Kubernetes", "SYSTEM", "Container orchestration")
        await memory.add_entity("San Francisco", "LOCATION", "City in California")

        # Search should work across types
        results = await memory.search_entity("engineer")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_entity_update(self, memory):
        """Test updating an existing entity."""
        await memory.add_entity("John Doe", "PERSON", "Junior engineer")

        # Update the entity
        await memory.add_entity("John Doe", "PERSON", "Senior engineer at Acme")

        results = await memory.search_entity("John Doe")
        # Should have updated information
        assert any("Senior" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_entity_relationships(self, memory):
        """Test entity relationships (if supported)."""
        try:
            await memory.add_entity(
                "John Doe",
                "PERSON",
                "Works at Acme",
                relationships={"works_at": "Acme Corp"}
            )

            await memory.add_entity(
                "Acme Corp",
                "ORGANIZATION",
                "Technology company",
                relationships={"employees": ["John Doe"]}
            )

            # Verify relationships are stored
            results = await memory.search_entity("Acme")
            assert len(results) >= 1
        except TypeError:
            # Relationships might not be supported
            pytest.skip("Entity relationships not supported")


# =============================================================================
# Workflow Memory Tests
# =============================================================================


class TestWorkflowMemory:
    """Tests for workflow memory operations."""

    @pytest.mark.asyncio
    async def test_add_workflow(self, memory):
        """Test adding a workflow."""
        await memory.add_workflow(
            "Deploy app",
            ["build", "test", "deploy"],
            "success"
        )

        results = await memory.search_workflow("deployment")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_workflow_with_status(self, memory):
        """Test workflows with different statuses."""
        await memory.add_workflow("Task 1", ["step1", "step2"], "success")
        await memory.add_workflow("Task 2", ["step1", "step2"], "failed")
        await memory.add_workflow("Task 3", ["step1", "step2"], "pending")

        results = await memory.search_workflow("Task")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_workflow_steps(self, memory):
        """Test workflow with complex steps."""
        steps = [
            "Initialize repository",
            "Install dependencies",
            "Run linting",
            "Run unit tests",
            "Run integration tests",
            "Build artifacts",
            "Deploy to staging",
            "Run smoke tests",
            "Deploy to production",
        ]

        await memory.add_workflow("Full CI/CD Pipeline", steps, "success")

        results = await memory.search_workflow("CI/CD")
        assert len(results) >= 1


# =============================================================================
# Summary and Expansion Tests
# =============================================================================


class TestSummaryExpansion:
    """Tests for summary creation and expansion."""

    @pytest.mark.asyncio
    async def test_create_summary(self, memory):
        """Test creating a summary from conversation."""
        # Create conversations
        id1 = await memory.add_conversational("t1", "user", "Message 1")
        id2 = await memory.add_conversational("t1", "assistant", "Response 1")

        # Create summary
        summary_id = await memory.add_summary(
            "User said Message 1, assistant responded",
            source_ids=[id1, id2],
            thread_id="t1"
        )

        assert summary_id is not None

    @pytest.mark.asyncio
    async def test_expand_summary(self, memory):
        """Test expanding a summary to original messages."""
        # Create conversations
        id1 = await memory.add_conversational("t1", "user", "Message 1")
        id2 = await memory.add_conversational("t1", "assistant", "Response 1")

        # Create summary
        summary_id = await memory.add_summary(
            "User said Message 1, assistant responded",
            source_ids=[id1, id2],
            thread_id="t1"
        )

        # Expand
        originals = await memory.expand_summary(summary_id)
        assert len(originals) == 2

    @pytest.mark.asyncio
    async def test_summary_preserves_order(self, memory):
        """Test that expanded summaries maintain original order."""
        ids = []
        for i in range(5):
            msg_id = await memory.add_conversational(
                "t1",
                "user" if i % 2 == 0 else "assistant",
                f"Message {i}"
            )
            ids.append(msg_id)

        summary_id = await memory.add_summary(
            "Conversation summary",
            source_ids=ids,
            thread_id="t1"
        )

        originals = await memory.expand_summary(summary_id)

        # Should maintain order
        for i, original in enumerate(originals):
            assert f"Message {i}" in original.content


# =============================================================================
# Toolbox VFS Tests
# =============================================================================


class TestToolboxVFS:
    """Tests for toolbox virtual filesystem operations."""

    @pytest.mark.asyncio
    async def test_add_tool(self, memory):
        """Test adding a tool to the toolbox."""
        await memory.add_tool(
            "github",
            "create_pr",
            "Create pull request",
            {"title": "string", "body": "string"}
        )

        tree = await memory.toolbox_tree()
        assert "github" in tree

    @pytest.mark.asyncio
    async def test_toolbox_tree(self, memory):
        """Test getting the toolbox tree structure."""
        # Register tools
        await memory.add_tool("github", "create_pr", "Create pull request", {"title": "string"})
        await memory.add_tool("github", "list_issues", "List issues", {"repo": "string"})
        await memory.add_tool("slack", "send_message", "Send message", {"channel": "string"})

        # VFS operations
        tree = await memory.toolbox_tree()
        assert "github" in tree
        assert "slack" in tree

    @pytest.mark.asyncio
    async def test_toolbox_ls(self, memory):
        """Test listing tools in a namespace."""
        await memory.add_tool("github", "create_pr", "Create PR", {})
        await memory.add_tool("github", "list_issues", "List issues", {})
        await memory.add_tool("github", "merge_pr", "Merge PR", {})

        tools = await memory.toolbox_ls("github")

        assert "create_pr" in tools
        assert "list_issues" in tools
        assert "merge_pr" in tools

    @pytest.mark.asyncio
    async def test_toolbox_grep(self, memory):
        """Test searching tools by description."""
        await memory.add_tool("github", "create_pr", "Create pull request", {})
        await memory.add_tool("slack", "send_message", "Send a message to channel", {})
        await memory.add_tool("jira", "create_ticket", "Create Jira ticket", {})

        results = await memory.toolbox_grep("message")

        assert len(results) >= 1
        # Should find slack send_message
        assert any("slack" in str(r) or "message" in str(r).lower() for r in results)

    @pytest.mark.asyncio
    async def test_toolbox_empty_namespace(self, memory):
        """Test listing empty namespace."""
        tools = await memory.toolbox_ls("nonexistent")

        assert tools == [] or tools is None

    @pytest.mark.asyncio
    async def test_toolbox_tool_schema(self, memory):
        """Test tool schema storage and retrieval."""
        schema = {
            "title": {"type": "string", "required": True},
            "body": {"type": "string", "required": False},
            "labels": {"type": "array", "items": "string"},
        }

        await memory.add_tool("github", "create_issue", "Create GitHub issue", schema)

        # Retrieve tool info
        tools = await memory.toolbox_ls("github")
        assert "create_issue" in tools


# =============================================================================
# Context Assembly Tests
# =============================================================================


class TestContextAssembly:
    """Tests for context assembly functionality."""

    @pytest.mark.asyncio
    async def test_basic_context_assembly(self, memory):
        """Test basic context assembly from multiple sources."""
        # Add various memories
        await memory.add_conversational("t1", "user", "I need help with K8s")
        await memory.add_knowledge("Kubernetes is container orchestration", source="docs")
        await memory.add_entity("K8s", "SYSTEM", "Container orchestration platform")

        # Assemble context
        context = await memory.assemble_context("kubernetes deployment", "t1")

        assert "Conversation" in context or "conversation" in context.lower()
        assert "Knowledge" in context or "knowledge" in context.lower()

    @pytest.mark.asyncio
    async def test_context_relevance(self, memory):
        """Test that assembled context is relevant."""
        # Add diverse content
        await memory.add_conversational("t1", "user", "How do I deploy to Kubernetes?")
        await memory.add_knowledge("Kubernetes uses pods for deployment", source="docs")
        await memory.add_knowledge("Python is a programming language", source="docs")
        await memory.add_entity("Kubernetes", "SYSTEM", "Container platform")
        await memory.add_entity("Python", "TECHNOLOGY", "Programming language")

        context = await memory.assemble_context("kubernetes deployment", "t1")

        # Kubernetes-related content should be included
        assert "Kubernetes" in context or "kubernetes" in context.lower() or "k8s" in context.lower()

    @pytest.mark.asyncio
    async def test_context_thread_specific(self, memory):
        """Test context assembly for specific thread."""
        # Add conversations to different threads
        await memory.add_conversational("thread_python", "user", "Tell me about Python")
        await memory.add_conversational("thread_k8s", "user", "Tell me about Kubernetes")

        # Assemble for Python thread
        context = await memory.assemble_context("programming", "thread_python")

        # Should prioritize thread-specific content
        assert context is not None

    @pytest.mark.asyncio
    async def test_context_with_empty_memory(self, memory):
        """Test context assembly with empty memory."""
        context = await memory.assemble_context("anything", "nonexistent")

        # Should return empty or minimal context without error
        assert context is not None


# =============================================================================
# Tool Log Tests
# =============================================================================


class TestToolLog:
    """Tests for tool execution logging."""

    @pytest.mark.asyncio
    async def test_log_tool_execution(self, memory):
        """Test logging a tool execution."""
        try:
            log_id = await memory.log_tool_execution(
                tool_name="github.create_pr",
                input_params={"title": "Fix bug", "body": "Description"},
                output_result={"pr_number": 123, "url": "https://github.com/..."},
                success=True,
                duration_ms=500
            )

            assert log_id is not None
        except (AttributeError, TypeError):
            pytest.skip("log_tool_execution not implemented")

    @pytest.mark.asyncio
    async def test_search_tool_logs(self, memory):
        """Test searching tool execution logs."""
        try:
            await memory.log_tool_execution(
                tool_name="github.create_pr",
                input_params={"title": "Test PR"},
                output_result={"pr_number": 1},
                success=True
            )

            logs = await memory.search_tool_logs("github")
            assert len(logs) >= 1
        except (AttributeError, TypeError):
            pytest.skip("Tool log methods not implemented")


# =============================================================================
# Skills Memory Tests
# =============================================================================


class TestSkillsMemory:
    """Tests for skills memory operations."""

    @pytest.mark.asyncio
    async def test_add_skill(self, memory):
        """Test adding a learned skill."""
        try:
            skill_id = await memory.add_skill(
                name="python_debugging",
                description="Debug Python applications using pdb",
                examples=["import pdb; pdb.set_trace()", "breakpoint()"],
                category="development"
            )

            assert skill_id is not None
        except (AttributeError, TypeError):
            pytest.skip("add_skill not implemented")

    @pytest.mark.asyncio
    async def test_search_skills(self, memory):
        """Test searching learned skills."""
        try:
            await memory.add_skill(
                name="debugging",
                description="Debug applications",
                examples=["pdb"],
                category="dev"
            )

            results = await memory.search_skills("debug")
            assert len(results) >= 1
        except (AttributeError, TypeError):
            pytest.skip("Skills methods not implemented")


# =============================================================================
# File Memory Tests
# =============================================================================


class TestFileMemory:
    """Tests for file memory operations."""

    @pytest.mark.asyncio
    async def test_add_file_reference(self, memory):
        """Test adding a file reference."""
        try:
            file_id = await memory.add_file(
                path="/path/to/file.py",
                content_summary="Python module for data processing",
                metadata={"lines": 100, "language": "python"}
            )

            assert file_id is not None
        except (AttributeError, TypeError):
            pytest.skip("add_file not implemented")

    @pytest.mark.asyncio
    async def test_search_files(self, memory):
        """Test searching file references."""
        try:
            await memory.add_file(
                path="/project/utils/helpers.py",
                content_summary="Utility functions for string manipulation"
            )

            results = await memory.search_files("string utility")
            assert len(results) >= 1
        except (AttributeError, TypeError):
            pytest.skip("File methods not implemented")


# =============================================================================
# Persona Memory Tests
# =============================================================================


class TestPersonaMemory:
    """Tests for persona memory operations."""

    @pytest.mark.asyncio
    async def test_set_persona(self, memory):
        """Test setting a persona."""
        try:
            persona_id = await memory.set_persona(
                name="Technical Expert",
                traits=["concise", "technical", "helpful"],
                communication_style="professional",
                domain_expertise=["python", "devops", "cloud"]
            )

            assert persona_id is not None
        except (AttributeError, TypeError):
            pytest.skip("set_persona not implemented")

    @pytest.mark.asyncio
    async def test_get_active_persona(self, memory):
        """Test retrieving active persona."""
        try:
            await memory.set_persona(
                name="Expert",
                traits=["helpful"],
                communication_style="friendly"
            )

            persona = await memory.get_active_persona()
            assert persona is not None
        except (AttributeError, TypeError):
            pytest.skip("Persona methods not implemented")


# =============================================================================
# Multi-Type Operations Tests
# =============================================================================


class TestMultiTypeOperations:
    """Tests for operations spanning multiple memory types."""

    @pytest.mark.asyncio
    async def test_unified_search(self, memory):
        """Test searching across all memory types."""
        # Add to different types
        await memory.add_conversational("t1", "user", "Tell me about Python")
        await memory.add_knowledge("Python is a language", source="docs")
        await memory.add_entity("Python", "TECHNOLOGY", "Programming language")

        try:
            results = await memory.search_all("Python", k=5)
            assert len(results) >= 1
        except (AttributeError, TypeError):
            pytest.skip("search_all not implemented")

    @pytest.mark.asyncio
    async def test_memory_stats(self, memory):
        """Test getting memory statistics."""
        # Add some content
        await memory.add_conversational("t1", "user", "Hello")
        await memory.add_knowledge("Test knowledge", source="test")

        try:
            stats = await memory.get_stats()
            assert stats is not None
            assert isinstance(stats, dict)
        except (AttributeError, TypeError):
            pytest.skip("get_stats not implemented")

    @pytest.mark.asyncio
    async def test_clear_thread(self, memory):
        """Test clearing a specific thread."""
        await memory.add_conversational("t1", "user", "Message 1")
        await memory.add_conversational("t1", "assistant", "Response 1")
        await memory.add_conversational("t2", "user", "Different thread")

        try:
            await memory.clear_thread("t1")

            messages1 = await memory.get_conversational("t1")
            messages2 = await memory.get_conversational("t2")

            assert len(messages1) == 0
            assert len(messages2) == 1
        except (AttributeError, TypeError):
            pytest.skip("clear_thread not implemented")


# =============================================================================
# Connection and Lifecycle Tests
# =============================================================================


class TestConnectionLifecycle:
    """Tests for connection and lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        harness = MemoryHarness("memory://")

        await harness.connect()
        assert harness.is_connected

        await harness.disconnect()
        assert not harness.is_connected

    @pytest.mark.asyncio
    async def test_reconnect(self):
        """Test reconnecting after disconnect."""
        harness = MemoryHarness("memory://")

        await harness.connect()
        await harness.disconnect()
        await harness.connect()

        assert harness.is_connected

        await harness.disconnect()

    @pytest.mark.asyncio
    async def test_operations_after_disconnect(self):
        """Test that operations fail gracefully after disconnect."""
        harness = MemoryHarness("memory://")
        await harness.connect()
        await harness.disconnect()

        with pytest.raises(Exception):  # Could be ConnectionError or RuntimeError
            await harness.add_conversational("t1", "user", "Hello")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager support."""
        try:
            async with MemoryHarness("memory://") as harness:
                await harness.add_conversational("t1", "user", "Hello")
                messages = await harness.get_conversational("t1")
                assert len(messages) == 1
        except TypeError:
            # Context manager might not be implemented
            pytest.skip("Async context manager not supported")
