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
        List of BaseTool instances.

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
    ]
