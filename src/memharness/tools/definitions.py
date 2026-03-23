# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
LangChain-based tool definitions for memory exploration.

This module provides LangChain BaseTool subclasses that agents can use to
explore and manage their own memory. Each tool wraps operations from
MemoryHarness in a standard LangChain tool interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub for Field when pydantic is not available."""
        return None


if TYPE_CHECKING:
    from memharness import MemoryHarness


__all__ = [
    "MemorySearchTool",
    "MemoryReadTool",
    "MemoryWriteTool",
    "MemoryStatsTool",
    "ToolboxTreeTool",
    "ToolboxGrepTool",
    "ExpandSummaryTool",
    "ConversationHistoryTool",
    "AssembleContextTool",
    "SummarizeAndStoreTool",
    "WriteToolLogTool",
    "WriteWorkflowTool",
    "get_memory_tools",
    "LANGCHAIN_AVAILABLE",
]


# =============================================================================
# Pydantic Input Schemas
# =============================================================================


class MemorySearchInput(BaseModel):
    """Input schema for memory search."""

    query: str = Field(description="Natural language search query")
    memory_type: str | None = Field(
        default=None,
        description="Type of memory to search (conversational, knowledge_base, entity, etc.)",
    )
    k: int = Field(default=5, description="Number of results to return (1-20)")


class MemoryReadInput(BaseModel):
    """Input schema for memory read."""

    memory_id: str = Field(description="The unique identifier of the memory to read")


class MemoryWriteInput(BaseModel):
    """Input schema for memory write."""

    memory_type: str = Field(description="Type of memory (knowledge_base, entity, or workflow)")
    content: str = Field(description="The content to store in memory")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata to attach")


class MemoryStatsInput(BaseModel):
    """Input schema for memory stats (no parameters)."""

    pass


class ToolboxTreeInput(BaseModel):
    """Input schema for toolbox tree."""

    path: str = Field(default="/", description="Path to start the tree from")
    depth: int = Field(default=3, description="Maximum depth to display (1-5)")


class ToolboxGrepInput(BaseModel):
    """Input schema for toolbox grep."""

    pattern: str = Field(description="Regex pattern to search for in tool names and descriptions")
    case_sensitive: bool = Field(default=False, description="Whether the search is case-sensitive")


class ExpandSummaryInput(BaseModel):
    """Input schema for expand summary."""

    summary_id: str = Field(description="The ID of the summary to expand back to full content")


class ConversationHistoryInput(BaseModel):
    """Input schema for conversation history."""

    thread_id: str = Field(description="Conversation thread ID")
    limit: int = Field(default=20, description="Max messages to retrieve")


class AssembleContextInput(BaseModel):
    """Input schema for assemble context."""

    query: str = Field(description="Query to assemble context for")
    thread_id: str = Field(description="Conversation thread ID")
    max_tokens: int = Field(default=4000, description="Maximum tokens in assembled context")


class SummarizeAndStoreInput(BaseModel):
    """Input schema for summarize and store."""

    thread_id: str = Field(description="Thread to summarize")
    max_messages: int = Field(default=50, description="Max messages to include")


class WriteToolLogInput(BaseModel):
    """Input schema for write tool log."""

    tool_name: str = Field(description="Name of the tool that was executed")
    tool_input: str = Field(description="Input/arguments passed to the tool")
    tool_output: str = Field(description="Output/result from the tool")
    status: str = Field(default="success", description="Execution status: success, error, timeout")


class WriteWorkflowInput(BaseModel):
    """Input schema for write workflow."""

    task: str = Field(description="Description of the task completed")
    steps: list[str] = Field(description="List of steps taken to complete the task")
    outcome: str = Field(description="Result: success or failure with description")


# =============================================================================
# Tool Classes
# =============================================================================


class MemorySearchTool(BaseTool):
    """
    Search across memory types using semantic similarity.

    This tool allows agents to search their memory for relevant information
    using natural language queries.
    """

    name: str = "memory_search"
    description: str = (
        "Search your memory for relevant information using semantic similarity. "
        "Returns the most relevant memories matching your query. "
        "Use this when you need to recall something but don't know the exact ID."
    )
    args_schema: type[BaseModel] = MemorySearchInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(
        self,
        query: str,
        memory_type: str | None = None,
        k: int = 5,
    ) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(
        self,
        query: str,
        memory_type: str | None = None,
        k: int = 5,
    ) -> str:
        """
        Execute memory search.

        Args:
            query: Natural language search query.
            memory_type: Optional type to filter by.
            k: Number of results to return.

        Returns:
            Formatted search results.
        """
        from memharness.tools.executor import MemoryToolExecutor

        executor = MemoryToolExecutor(self.harness)
        return await executor.execute("memory_search", query=query, memory_type=memory_type, k=k)


class MemoryReadTool(BaseTool):
    """
    Read a specific memory by its ID.

    This tool allows agents to retrieve detailed information about a
    specific memory when they have its ID.
    """

    name: str = "memory_read"
    description: str = (
        "Read a specific memory by its ID. "
        "Use this when you have a memory ID from a previous search or reference."
    )
    args_schema: type[BaseModel] = MemoryReadInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, memory_id: str) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self, memory_id: str) -> str:
        """
        Execute memory read.

        Args:
            memory_id: The memory's unique identifier.

        Returns:
            Formatted memory content.
        """
        from memharness.tools.executor import MemoryToolExecutor

        executor = MemoryToolExecutor(self.harness)
        return await executor.execute("memory_read", memory_id=memory_id)


class MemoryWriteTool(BaseTool):
    """
    Write new information to memory.

    This tool allows agents to persist important facts, learnings,
    or observations to their memory.
    """

    name: str = "memory_write"
    description: str = (
        "Write new information to memory. "
        "Use this to persist important facts, learnings, or observations."
    )
    args_schema: type[BaseModel] = MemoryWriteInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(
        self,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(
        self,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Execute memory write.

        Args:
            memory_type: Type of memory to write.
            content: Content to store.
            metadata: Optional metadata.

        Returns:
            Confirmation with memory ID.
        """
        from memharness.tools.executor import MemoryToolExecutor

        executor = MemoryToolExecutor(self.harness)
        return await executor.execute(
            "memory_write", memory_type=memory_type, content=content, metadata=metadata
        )


class MemoryStatsTool(BaseTool):
    """
    Get memory statistics and usage information.

    This tool provides agents with insights into their memory usage,
    showing counts, sizes, and health metrics for each memory type.
    """

    name: str = "memory_stats"
    description: str = (
        "Get statistics about your memory usage. "
        "Shows counts, sizes, and health metrics for each memory type."
    )
    args_schema: type[BaseModel] = MemoryStatsInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self) -> str:
        """
        Execute memory stats.

        Returns:
            Formatted statistics overview.
        """
        from memharness.tools.executor import MemoryToolExecutor

        executor = MemoryToolExecutor(self.harness)
        return await executor.execute("memory_stats")


class ToolboxTreeTool(BaseTool):
    """
    Display a tree view of available tools.

    This tool shows agents a hierarchical view of all available
    tools in their toolbox, organized by server/category.
    """

    name: str = "toolbox_tree"
    description: str = (
        "Display a tree view of available tools in your toolbox. "
        "Shows tools organized by server/category in a hierarchical view."
    )
    args_schema: type[BaseModel] = ToolboxTreeInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, path: str = "/", depth: int = 3) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self, path: str = "/", depth: int = 3) -> str:
        """
        Execute toolbox tree.

        Args:
            path: Starting path.
            depth: Max depth.

        Returns:
            Formatted tree.
        """
        from memharness.tools.executor import MemoryToolExecutor

        executor = MemoryToolExecutor(self.harness)
        return await executor.execute("toolbox_tree", path=path, depth=depth)


class ToolboxGrepTool(BaseTool):
    """
    Search for tools by name or description pattern.

    This tool allows agents to search for specific tools when they
    know part of a tool name or what it does.
    """

    name: str = "toolbox_grep"
    description: str = (
        "Search for tools by name or description pattern. "
        "Useful when you know part of a tool name or what it does."
    )
    args_schema: type[BaseModel] = ToolboxGrepInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, pattern: str, case_sensitive: bool = False) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self, pattern: str, case_sensitive: bool = False) -> str:
        """
        Execute toolbox grep.

        Args:
            pattern: Regex pattern.
            case_sensitive: Match case.

        Returns:
            Matching tools.
        """
        from memharness.tools.executor import MemoryToolExecutor

        executor = MemoryToolExecutor(self.harness)
        return await executor.execute(
            "toolbox_grep", pattern=pattern, case_sensitive=case_sensitive
        )


class ExpandSummaryTool(BaseTool):
    """
    Expand a compacted summary back to its full original content.

    This tool allows agents to retrieve the full original content from a
    summary that was previously compacted. Use this when you need details
    from a conversation that was previously summarized.
    """

    name: str = "expand_summary"
    description: str = (
        "Expand a compacted summary back to its full original content. "
        "Use when you need details from a conversation that was previously summarized. "
        "The summary_id comes from the Context Summaries section."
    )
    args_schema: type[BaseModel] = ExpandSummaryInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, summary_id: str) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self, summary_id: str) -> str:
        """
        Execute expand summary.

        Args:
            summary_id: The ID of the summary to expand.

        Returns:
            Formatted original content.
        """
        try:
            originals = await self.harness.expand_summary(summary_id)
            if not originals:
                return f"No original content found for summary {summary_id}"

            # Format as conversation messages
            lines = []
            for m in originals:
                role = m.metadata.get("role", "user")
                lines.append(f"{role}: {m.content}")

            return "\n".join(lines)
        except KeyError as e:
            return f"Error: {e}"


class ConversationHistoryTool(BaseTool):
    """
    Get conversation history for a thread.

    This tool allows agents to retrieve messages from a specific conversation
    thread. Useful for reviewing what was discussed in a thread.
    """

    name: str = "get_conversation_history"
    description: str = (
        "Get conversation history for a thread as a list of messages. "
        "Returns recent messages from the specified conversation thread."
    )
    args_schema: type[BaseModel] = ConversationHistoryInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, thread_id: str, limit: int = 20) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self, thread_id: str, limit: int = 20) -> str:
        """
        Execute conversation history retrieval.

        Args:
            thread_id: Conversation thread ID.
            limit: Max messages to retrieve.

        Returns:
            Formatted conversation history.
        """
        messages = await self.harness.get_conversational(thread_id, limit=limit)
        if not messages:
            return f"No conversation history found for thread {thread_id}"

        # Format as conversation messages
        lines = [f"Conversation history for thread {thread_id} ({len(messages)} messages):\n"]
        for m in messages:
            role = m.metadata.get("role", "user")
            timestamp = m.metadata.get("timestamp", "")
            if timestamp:
                lines.append(f"[{timestamp}] {role}: {m.content}")
            else:
                lines.append(f"{role}: {m.content}")

        return "\n".join(lines)


class AssembleContextTool(BaseTool):
    """
    Assemble all relevant memory context for a query.

    This tool allows agents to explicitly request context assembly,
    gathering persona, conversation history, knowledge, workflows,
    entities, and tools relevant to their query.
    """

    name: str = "assemble_context"
    description: str = (
        "Assemble all relevant memory context for a query. "
        "Returns persona, conversation history, knowledge, workflows, entities, and tools. "
        "Use this when you need comprehensive context about a topic or task."
    )
    args_schema: type[BaseModel] = AssembleContextInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, query: str, thread_id: str, max_tokens: int = 4000) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self, query: str, thread_id: str, max_tokens: int = 4000) -> str:
        """
        Execute context assembly.

        Args:
            query: Query to assemble context for.
            thread_id: Conversation thread ID.
            max_tokens: Maximum tokens in assembled context.

        Returns:
            Formatted context string.
        """
        context = await self.harness.assemble_context(
            query=query, thread_id=thread_id, max_tokens=max_tokens
        )
        if not context:
            return "No relevant context found."

        return context


class SummarizeAndStoreTool(BaseTool):
    """
    Summarize a conversation thread and store the summary.

    This tool allows agents to compress long conversation histories by
    creating a summary and storing it. Original messages are preserved
    in the database (compaction pattern).
    """

    name: str = "summarize_and_store"
    description: str = (
        "Summarize a conversation thread and store the summary. "
        "Use when conversation history is getting long and you want to compress it. "
        "Original messages are preserved in the database (compaction pattern)."
    )
    args_schema: type[BaseModel] = SummarizeAndStoreInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, thread_id: str, max_messages: int = 50) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(self, thread_id: str, max_messages: int = 50) -> str:
        """
        Execute summarize and store.

        Args:
            thread_id: Thread to summarize.
            max_messages: Max messages to include.

        Returns:
            Summary text and summary ID.
        """
        try:
            from memharness.agents.summarizer import SummarizerAgent

            # Get messages to summarize
            messages = await self.harness.get_conversational(thread_id, limit=max_messages)
            if not messages:
                return f"No messages found in thread {thread_id}"

            # Create summarizer agent (heuristic mode, no LLM required)
            summarizer = SummarizerAgent(self.harness, llm=None)
            summary_text = await summarizer.summarize_thread(thread_id, max_messages)

            # Store summary with source message IDs
            source_ids = [msg.id for msg in messages]
            summary_id = await self.harness.add_summary(
                summary=summary_text, source_ids=source_ids, thread_id=thread_id
            )

            return (
                f"Summary created and stored.\n"
                f"Summary ID: {summary_id}\n"
                f"Summarized {len(messages)} message(s).\n\n"
                f"Summary: {summary_text}"
            )
        except Exception as e:
            return f"Error creating summary: {e}"


class WriteToolLogTool(BaseTool):
    """
    Log a tool execution for audit trail.

    This tool allows agents to record what tool was called, with what
    input, and what it returned. Use after every tool call to maintain
    execution history.
    """

    name: str = "write_tool_log"
    description: str = (
        "Log a tool execution for audit trail. "
        "Records what tool was called, with what input, and what it returned. "
        "Use after every tool call to maintain execution history."
    )
    args_schema: type[BaseModel] = WriteToolLogInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        status: str = "success",
    ) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        status: str = "success",
    ) -> str:
        """
        Execute tool log write.

        Args:
            tool_name: Name of the tool that was executed.
            tool_input: Input/arguments passed to the tool.
            tool_output: Output/result from the tool.
            status: Execution status.

        Returns:
            Confirmation with log ID.
        """
        try:
            import json

            # Parse tool_input if it's a JSON string
            try:
                args_dict = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            except (json.JSONDecodeError, TypeError):
                args_dict = {"input": tool_input}

            # Use a default thread_id if not available in context
            thread_id = "default"

            # Log the tool execution
            log_id = await self.harness.add_tool_log(
                thread_id=thread_id,
                tool_name=tool_name,
                args=args_dict,
                result=tool_output,
                status=status,
            )

            return f"Tool execution logged. Log ID: {log_id}\nTool: {tool_name}\nStatus: {status}"
        except Exception as e:
            return f"Error logging tool execution: {e}"


class WriteWorkflowTool(BaseTool):
    """
    Save a completed task as a reusable workflow pattern.

    This tool allows agents to record the steps taken and outcome so
    similar tasks can follow this recipe. Use after successfully
    completing a multi-step task.
    """

    name: str = "write_workflow"
    description: str = (
        "Save a completed task as a reusable workflow pattern. "
        "Records the steps taken and outcome so similar tasks can follow this recipe. "
        "Use after successfully completing a multi-step task."
    )
    args_schema: type[BaseModel] = WriteWorkflowInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(
        self,
        task: str,
        steps: list[str],
        outcome: str,
    ) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(
        self,
        task: str,
        steps: list[str],
        outcome: str,
    ) -> str:
        """
        Execute workflow write.

        Args:
            task: Description of the task completed.
            steps: List of steps taken to complete the task.
            outcome: Result (success or failure with description).

        Returns:
            Confirmation with workflow ID.
        """
        try:
            # Store the workflow
            workflow_id = await self.harness.add_workflow(task=task, steps=steps, outcome=outcome)

            steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))

            return (
                f"Workflow saved as reusable pattern.\n"
                f"Workflow ID: {workflow_id}\n\n"
                f"Task: {task}\n"
                f"Steps:\n{steps_text}\n"
                f"Outcome: {outcome}"
            )
        except Exception as e:
            return f"Error saving workflow: {e}"


# =============================================================================
# Helper Functions
# =============================================================================


def get_memory_tools(harness: MemoryHarness) -> list[BaseTool]:
    """
    Get all memory tools for a given harness instance.

    This function creates and returns a list of all available memory tools,
    each configured to use the provided MemoryHarness instance.

    Args:
        harness: The MemoryHarness instance to create tools for.

    Returns:
        List of BaseTool instances (12 tools total):
        1. MemorySearchTool - Search across memory types
        2. MemoryReadTool - Read memory by ID
        3. MemoryWriteTool - Write new memory
        4. MemoryStatsTool - Get memory statistics
        5. ToolboxTreeTool - Explore tools VFS
        6. ToolboxGrepTool - Search tools by pattern
        7. ExpandSummaryTool - Expand compacted summaries
        8. ConversationHistoryTool - Get thread messages
        9. AssembleContextTool - Full context assembly
        10. SummarizeAndStoreTool - Compress conversation history
        11. WriteToolLogTool - Log tool executions
        12. WriteWorkflowTool - Save task as reusable workflow

    Raises:
        ImportError: If langchain-core is not installed.

    Example:
        >>> from memharness import MemoryHarness
        >>> from memharness.tools import get_memory_tools
        >>>
        >>> harness = MemoryHarness("sqlite:///memory.db")
        >>> tools = get_memory_tools(harness)
        >>>
        >>> # Use with LangChain agent
        >>> from langchain.agents import AgentExecutor
        >>> agent = AgentExecutor(tools=tools, ...)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for memory tools. Install with: pip install langchain-core"
        )

    return [
        MemorySearchTool(harness=harness),
        MemoryReadTool(harness=harness),
        MemoryWriteTool(harness=harness),
        MemoryStatsTool(harness=harness),
        ToolboxTreeTool(harness=harness),
        ToolboxGrepTool(harness=harness),
        ExpandSummaryTool(harness=harness),
        ConversationHistoryTool(harness=harness),
        AssembleContextTool(harness=harness),
        SummarizeAndStoreTool(harness=harness),
        WriteToolLogTool(harness=harness),
        WriteWorkflowTool(harness=harness),
    ]
