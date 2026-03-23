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
    "ToolboxSearchTool",
    "ExpandSummaryTool",
    "AssembleContextTool",
    "SummarizeAndStoreTool",
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

    memory_type: str = Field(
        description="Type of memory: conversational, knowledge, entity, workflow, toolbox, summary, tool_log, persona, file"
    )
    content: str = Field(description="The content to store in memory")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata to attach")

    # Workflow-specific fields
    task: str | None = Field(default=None, description="For workflow: task description")
    steps: list[str] | None = Field(default=None, description="For workflow: list of steps")
    outcome: str | None = Field(default=None, description="For workflow: outcome description")

    # Tool log-specific fields
    tool_name: str | None = Field(default=None, description="For tool_log: name of the tool")
    tool_input: str | None = Field(default=None, description="For tool_log: tool input/arguments")
    tool_output: str | None = Field(default=None, description="For tool_log: tool output/result")
    status: str | None = Field(
        default=None, description="For tool_log: execution status (success/error/timeout)"
    )

    # Common fields
    thread_id: str | None = Field(
        default=None, description="Thread ID for conversational or tool_log"
    )
    role: str | None = Field(
        default=None, description="Role for conversational (user/assistant/system)"
    )


class ToolboxSearchInput(BaseModel):
    """Input schema for toolbox search."""

    pattern: str | None = Field(
        default=None,
        description="Optional regex pattern to search for (grep mode). If not provided, shows tree view.",
    )
    case_sensitive: bool = Field(
        default=False, description="Whether the search is case-sensitive (grep mode only)"
    )
    path: str = Field(default="/", description="Path to start from (tree mode only)")
    depth: int = Field(default=3, description="Maximum depth to display (tree mode only, 1-5)")


class ExpandSummaryInput(BaseModel):
    """Input schema for expand summary."""

    summary_id: str = Field(description="The ID of the summary to expand back to full content")


class AssembleContextInput(BaseModel):
    """Input schema for assemble context."""

    query: str = Field(description="Query to assemble context for")
    thread_id: str = Field(description="Conversation thread ID")
    max_tokens: int = Field(default=4000, description="Maximum tokens in assembled context")


class SummarizeAndStoreInput(BaseModel):
    """Input schema for summarize and store."""

    thread_id: str = Field(description="Thread to summarize")
    max_messages: int = Field(default=50, description="Max messages to include")


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
        "Write new information to memory. Supports ALL memory types: "
        "conversational (chat messages), knowledge (facts), entity (named entities), "
        "workflow (task steps), tool_log (tool execution logs), persona, file, summary, toolbox. "
        "Use the appropriate fields for your memory type."
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

    def _run(self, **kwargs: Any) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(
        self,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        task: str | None = None,
        steps: list[str] | None = None,
        outcome: str | None = None,
        tool_name: str | None = None,
        tool_input: str | None = None,
        tool_output: str | None = None,
        status: str | None = None,
        thread_id: str | None = None,
        role: str | None = None,
    ) -> str:
        """
        Execute memory write for ANY memory type.

        Args:
            memory_type: Type of memory to write.
            content: Content to store.
            metadata: Optional metadata.
            task: For workflow - task description.
            steps: For workflow - list of steps.
            outcome: For workflow - outcome description.
            tool_name: For tool_log - tool name.
            tool_input: For tool_log - tool input.
            tool_output: For tool_log - tool output.
            status: For tool_log - status (success/error/timeout).
            thread_id: For conversational or tool_log - thread ID.
            role: For conversational - role (user/assistant/system).

        Returns:
            Confirmation with memory ID.
        """
        try:
            import json

            # Normalize memory type names
            type_map = {
                "knowledge": "knowledge_base",
                "knowledge_base": "knowledge_base",
                "conversational": "conversational",
                "entity": "entity",
                "workflow": "workflow",
                "tool_log": "tool_log",
                "toolbox": "toolbox",
                "summary": "summary",
                "persona": "persona",
                "file": "file",
            }

            normalized_type = type_map.get(memory_type.lower())
            if not normalized_type:
                return f"Error: Unknown memory type '{memory_type}'. Supported types: {', '.join(type_map.keys())}"

            # Handle workflow type
            if normalized_type == "workflow":
                if not task or not steps or not outcome:
                    return "Error: workflow type requires task, steps, and outcome fields"
                memory_id = await self.harness.add_workflow(task=task, steps=steps, outcome=outcome)
                steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
                return (
                    f"Workflow saved successfully.\n"
                    f"Memory ID: {memory_id}\n\n"
                    f"Task: {task}\n"
                    f"Steps:\n{steps_text}\n"
                    f"Outcome: {outcome}"
                )

            # Handle tool_log type
            elif normalized_type == "tool_log":
                if not tool_name or not tool_input or not tool_output:
                    return "Error: tool_log type requires tool_name, tool_input, and tool_output fields"

                # Parse tool_input if it's a JSON string
                try:
                    args_dict = (
                        json.loads(tool_input) if isinstance(tool_input, str) else tool_input
                    )
                except (json.JSONDecodeError, TypeError):
                    args_dict = {"input": tool_input}

                tid = thread_id or "default"
                memory_id = await self.harness.add_tool_log(
                    thread_id=tid,
                    tool_name=tool_name,
                    args=args_dict,
                    result=tool_output,
                    status=status or "success",
                )
                return (
                    f"Tool execution logged successfully.\n"
                    f"Memory ID: {memory_id}\n"
                    f"Tool: {tool_name}\n"
                    f"Status: {status or 'success'}"
                )

            # Handle conversational type
            elif normalized_type == "conversational":
                if not thread_id or not role:
                    return "Error: conversational type requires thread_id and role fields"
                memory_id = await self.harness.add_conversational(
                    thread_id=thread_id, role=role, content=content
                )
                return (
                    f"Conversational memory saved successfully.\n"
                    f"Memory ID: {memory_id}\n"
                    f"Thread: {thread_id}\n"
                    f"Role: {role}"
                )

            # Handle knowledge_base type
            elif normalized_type == "knowledge_base":
                memory_id = await self.harness.add_knowledge(content=content, metadata=metadata)
                return f"Knowledge saved successfully. Memory ID: {memory_id}"

            # Handle entity type
            elif normalized_type == "entity":
                # Extract entity details from metadata or content
                name = (metadata or {}).get("name", "Unknown")
                entity_type = (metadata or {}).get("entity_type", "general")
                memory_id = await self.harness.add_entity(
                    name=name, entity_type=entity_type, description=content, metadata=metadata
                )
                return f"Entity '{name}' saved successfully. Memory ID: {memory_id}"

            # Handle other types generically
            else:
                return f"Error: Memory type '{normalized_type}' is supported by the harness but write logic not yet implemented in this tool. Use the harness directly."

        except Exception as e:
            return f"Error writing to memory: {e}"


class ToolboxSearchTool(BaseTool):
    """
    Search and explore available tools in your toolbox.

    This tool combines tree view and grep functionality. If a pattern is provided,
    it searches for tools matching that pattern (grep mode). If no pattern is given,
    it shows a tree view of all tools organized by server/category.
    """

    name: str = "toolbox_search"
    description: str = (
        "Search and explore available tools in your toolbox. "
        "Provide a 'pattern' to search for specific tools (grep mode), "
        "or leave pattern empty to see a tree view of all tools organized by server/category."
    )
    args_schema: type[BaseModel] = ToolboxSearchInput
    harness: Any  # MemoryHarness instance

    def __init__(self, harness: MemoryHarness, **kwargs: Any) -> None:
        """Initialize the tool with a memory harness."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for memory tools. "
                "Install with: pip install langchain-core"
            )
        super().__init__(harness=harness, **kwargs)

    def _run(self, **kwargs: Any) -> str:
        """Sync execution (not supported for async harness)."""
        raise NotImplementedError("Use async version (_arun) instead")

    async def _arun(
        self,
        pattern: str | None = None,
        case_sensitive: bool = False,
        path: str = "/",
        depth: int = 3,
    ) -> str:
        """
        Execute toolbox search.

        Args:
            pattern: Optional regex pattern to search for (grep mode).
            case_sensitive: Whether the search is case-sensitive.
            path: Starting path for tree view.
            depth: Maximum depth for tree view.

        Returns:
            Formatted tree or search results.
        """
        from memharness.tools.executor import MemoryToolExecutor

        executor = MemoryToolExecutor(self.harness)

        # Grep mode if pattern is provided
        if pattern:
            return await executor.execute(
                "toolbox_grep", pattern=pattern, case_sensitive=case_sensitive
            )
        # Tree mode otherwise
        else:
            return await executor.execute("toolbox_tree", path=path, depth=depth)


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

            # Create summarizer agent (heuristic mode, no LLM required)
            summarizer = SummarizerAgent(self.harness, llm=None)
            result = await summarizer.summarize_thread(thread_id, max_messages)

            # Handle the result dict (new API)
            if not result.get("summarized"):
                reason = result.get("reason", "unknown")
                if reason == "too_few_messages":
                    return f"Not enough messages to summarize in thread {thread_id} ({result.get('messages_summarized', 0)} messages found, need at least 10)"
                return f"Could not summarize: {reason}"

            # Success - return formatted message
            return (
                f"Summary created and stored.\n"
                f"Summary ID: {result['summary_id']}\n"
                f"Summarized {result['messages_summarized']} message(s).\n\n"
                f"Summary: {result['summary_text']}"
            )
        except Exception as e:
            return f"Error creating summary: {e}"


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
        List of BaseTool instances (7 tools total):
        1. memory_search - Search across memory types
        2. memory_read - Read memory by ID
        3. memory_write - Write to ANY memory type (knowledge, entity, workflow, tool_log, etc.)
        4. toolbox_search - Discover tools (tree + grep combined)
        5. expand_summary - Expand compacted summaries
        6. assemble_context - Full BEFORE-loop context assembly
        7. summarize_conversation - Compress conversation history

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
        >>> from langchain.agents import create_agent
        >>> agent = create_agent(model=..., tools=tools, ...)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for memory tools. Install with: pip install langchain-core"
        )

    return [
        MemorySearchTool(harness=harness),
        MemoryReadTool(harness=harness),
        MemoryWriteTool(harness=harness),
        ToolboxSearchTool(harness=harness),
        ExpandSummaryTool(harness=harness),
        AssembleContextTool(harness=harness),
        SummarizeAndStoreTool(harness=harness),
    ]
