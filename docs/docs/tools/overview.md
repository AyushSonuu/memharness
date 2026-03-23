---
sidebar_position: 1
---

# Memory Tools Overview

memharness provides a set of LangChain-compatible tools that enable AI agents to explore and manage their own memory. These tools are implemented as `BaseTool` subclasses and can be used with any LangChain agent.

## Available Tools

The toolkit consists of 9 specialized tools organized into three categories:

### Memory Management Tools

These tools allow agents to search, read, write, and analyze their memory:

- **MemorySearchTool** - Search across memory types using semantic similarity
- **MemoryReadTool** - Read a specific memory by its ID
- **MemoryWriteTool** - Write new information to memory
- **MemoryStatsTool** - Get memory statistics and usage information

### Context Operations Tools

These tools enable context assembly and expansion for efficient memory operations:

- **ExpandSummaryTool** - Expand compacted summaries back to full original content
- **ConversationHistoryTool** - Get conversation history for a thread as a list of messages
- **AssembleContextTool** - Assemble all relevant memory context (persona, history, knowledge, workflows, entities)

### Toolbox Exploration Tools

These tools help agents discover and understand available tools (self-exploration):

- **ToolboxTreeTool** - Display a tree view of available tools
- **ToolboxGrepTool** - Search for tools by name or description pattern

## Quick Start

### Getting All Tools

The easiest way to get all memory tools is using the `get_memory_tools()` helper function:

```python
from memharness import MemoryHarness
from memharness.tools import get_memory_tools

# Initialize harness
harness = MemoryHarness("sqlite:///memory.db")
await harness.connect()

# Get all tools
tools = get_memory_tools(harness)

# Use with LangChain agent
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=tools,
    system_prompt="You are a helpful AI assistant with persistent memory."
)
```

### Using Individual Tools

You can also instantiate tools individually:

```python
from memharness.tools import MemorySearchTool, MemoryWriteTool

search_tool = MemorySearchTool(harness=harness)
write_tool = MemoryWriteTool(harness=harness)

# Use in agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_tool, write_tool],
)
```

## Tool Schemas

All tools use Pydantic models for input validation. Here are the input schemas:

### MemorySearchInput
```python
{
    "query": str,              # Natural language search query
    "memory_type": str | None, # Optional: filter by type
    "k": int                   # Number of results (default: 5)
}
```

### MemoryReadInput
```python
{
    "memory_id": str  # Unique identifier of the memory
}
```

### MemoryWriteInput
```python
{
    "memory_type": str,           # Type of memory (knowledge_base, entity, etc.)
    "content": str,               # Content to store
    "metadata": dict | None       # Optional metadata
}
```

### MemoryStatsInput
No parameters required.

### ToolboxTreeInput
```python
{
    "path": str,     # Starting path (default: "/")
    "depth": int     # Maximum depth to display (default: 3)
}
```

### ToolboxGrepInput
```python
{
    "pattern": str,           # Regex pattern to search for
    "case_sensitive": bool    # Whether search is case-sensitive (default: False)
}
```

### ExpandSummaryInput
```python
{
    "summary_id": str  # ID of the summary to expand
}
```

### ConversationHistoryInput
```python
{
    "thread_id": str,  # Conversation thread ID
    "limit": int       # Max messages to retrieve (default: 20)
}
```

### AssembleContextInput
```python
{
    "query": str,         # Query to assemble context for
    "thread_id": str,     # Conversation thread ID
    "max_tokens": int     # Maximum tokens in assembled context (default: 4000)
}
```

## Async-Only Tools

All memory tools are async-only and must be used with async agents. The `_run()` method will raise `NotImplementedError` - always use `_arun()` or let LangChain handle execution automatically.

```python
# Don't do this
result = tool._run(...)  # Raises NotImplementedError

# Do this
result = await tool._arun(...)  # Works correctly
```

## Requirements

Memory tools require the following packages:

```bash
pip install memharness[langchain]
```

This installs:
- `langchain-core` - For BaseTool interface
- `pydantic` - For input schema validation

## Next Steps

- Learn about [self-exploration tools](./self-exploration.md) for memory introspection
- Explore [toolbox VFS tools](./toolbox-vfs.md) for tool discovery
- See [LangChain integration](../integrations/langchain.md) for complete examples
