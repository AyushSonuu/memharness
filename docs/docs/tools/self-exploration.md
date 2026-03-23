---
sidebar_position: 2
---

# Self-Exploration Tools

Self-exploration tools enable AI agents to introspect and manage their own memory. These tools give agents the ability to search their past experiences, read specific memories, write new information, and understand their memory usage patterns.

## Overview

The four self-exploration tools provide complete CRUD operations for agent memory:

1. **MemorySearchTool** - Semantic search across memory types
2. **MemoryReadTool** - Retrieve specific memories by ID
3. **MemoryWriteTool** - Store new information
4. **MemoryStatsTool** - Analyze memory usage and health

## MemorySearchTool

Search across memory types using natural language queries and semantic similarity.

### Description
```
Search your memory for relevant information using semantic similarity.
Returns the most relevant memories matching your query.
Use this when you need to recall something but don't know the exact ID.
```

### Input Schema

```python
class MemorySearchInput(BaseModel):
    query: str = Field(description="Natural language search query")
    memory_type: str | None = Field(
        default=None,
        description="Type of memory to search (conversational, knowledge_base, entity, etc.)"
    )
    k: int = Field(default=5, description="Number of results to return (1-20)")
```

### Usage Example

```python
from memharness.tools import MemorySearchTool

search_tool = MemorySearchTool(harness=harness)

# Search across all memory types
result = await search_tool._arun(
    query="What did I learn about PostgreSQL yesterday?",
    k=3
)

# Search within a specific memory type
result = await search_tool._arun(
    query="database",
    memory_type="knowledge_base",
    k=10
)
```

### Agent Usage

When used with a LangChain agent, the tool is invoked naturally:

```python
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_tool],
)

# Agent will use the tool automatically
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "What do you remember about my work projects?"}]
})
```

## MemoryReadTool

Read a specific memory by its unique identifier.

### Description
```
Read a specific memory by its ID.
Use this when you have a memory ID from a previous search or reference.
```

### Input Schema

```python
class MemoryReadInput(BaseModel):
    memory_id: str = Field(description="The unique identifier of the memory to read")
```

### Usage Example

```python
from memharness.tools import MemoryReadTool

read_tool = MemoryReadTool(harness=harness)

# Read a specific memory
result = await read_tool._arun(
    memory_id="mem_abc123xyz"
)
```

### Typical Workflow

The read tool is often used after a search operation:

```python
# 1. Search for relevant memories
search_results = await search_tool._arun(
    query="Python best practices",
    k=5
)

# 2. Extract memory IDs from results
# (Results contain: ID, content snippet, relevance score)

# 3. Read full content of a specific memory
full_memory = await read_tool._arun(
    memory_id="mem_found_in_search"
)
```

## MemoryWriteTool

Write new information to memory for future retrieval.

### Description
```
Write new information to memory.
Use this to persist important facts, learnings, or observations.
```

### Input Schema

```python
class MemoryWriteInput(BaseModel):
    memory_type: str = Field(
        description="Type of memory (knowledge_base, entity, workflow, or skills)"
    )
    content: str = Field(description="The content to store in memory")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata to attach"
    )
```

### Usage Example

```python
from memharness.tools import MemoryWriteTool

write_tool = MemoryWriteTool(harness=harness)

# Write to knowledge base
result = await write_tool._arun(
    memory_type="knowledge_base",
    content="PostgreSQL uses MVCC for transaction isolation",
    metadata={"source": "documentation", "topic": "databases"}
)

# Write an entity
result = await write_tool._arun(
    memory_type="entity",
    content="Alice works at Anthropic in San Francisco",
    metadata={"entity_type": "person", "name": "Alice"}
)

# Write a learned skill
result = await write_tool._arun(
    memory_type="skills",
    content="Use async/await for non-blocking database operations in Python",
    metadata={"category": "programming", "language": "python"}
)
```

### Supported Memory Types

The write tool supports the following memory types:
- `knowledge_base` - General facts and information
- `entity` - People, organizations, locations
- `workflow` - Procedures and processes
- `skills` - Learned capabilities and techniques

## MemoryStatsTool

Get statistics and health metrics about memory usage.

### Description
```
Get statistics about your memory usage.
Shows counts, sizes, and health metrics for each memory type.
```

### Input Schema

No parameters required (uses `MemoryStatsInput` with no fields).

### Usage Example

```python
from memharness.tools import MemoryStatsTool

stats_tool = MemoryStatsTool(harness=harness)

# Get memory statistics
result = await stats_tool._arun()
```

### Example Output

```
Memory Statistics:
------------------
Conversational: 142 messages across 3 threads
Knowledge Base: 87 entries
Entity: 23 entities (15 people, 5 organizations, 3 locations)
Workflow: 12 procedures
Skills: 34 learned capabilities
Tool Log: 456 tool invocations
Toolbox: 8 registered tools
Summary: 6 conversation summaries
File: 19 file references
Persona: 1 agent profile

Total memory size: 2.4 MB
Health: Good (no issues detected)
```

## Integration with LangChain Agents

All self-exploration tools work seamlessly with LangChain agents:

```python
from memharness import MemoryHarness
from memharness.tools import get_memory_tools
from langchain.agents import create_agent

# Initialize harness
harness = MemoryHarness("sqlite:///memory.db")
await harness.connect()

# Get all tools (includes self-exploration tools)
tools = get_memory_tools(harness)

# Create agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=tools,
    system_prompt=(
        "You are a helpful AI assistant with persistent memory. "
        "Use your memory tools to remember important information "
        "and provide personalized responses."
    )
)

# Agent automatically uses tools when needed
response = await agent.ainvoke({
    "messages": [
        {"role": "user", "content": "Remember that I prefer Python over JavaScript"},
        {"role": "user", "content": "What programming language do I prefer?"}
    ]
})
```

## Best Practices

### 1. Use Search Before Read

Always search first to find relevant memory IDs, then read for full details:

```python
# Good: Search then read
search_results = await search_tool._arun(query="project deadlines", k=3)
# Extract IDs and read specific memories

# Avoid: Guessing memory IDs
result = await read_tool._arun(memory_id="some_random_id")
```

### 2. Add Metadata to Writes

Include relevant metadata to make memories easier to filter and retrieve:

```python
# Good: Rich metadata
await write_tool._arun(
    memory_type="knowledge_base",
    content="Meeting scheduled for next Tuesday at 2 PM",
    metadata={
        "type": "meeting",
        "date": "2026-03-30",
        "priority": "high"
    }
)

# Okay: Minimal metadata
await write_tool._arun(
    memory_type="knowledge_base",
    content="Meeting scheduled for next Tuesday at 2 PM"
)
```

### 3. Use Stats for Debugging

Check memory stats to understand what the agent has learned:

```python
# Useful for debugging and monitoring
stats = await stats_tool._arun()
print(stats)
```

### 4. Scope Search Appropriately

Use the `memory_type` parameter to narrow search results:

```python
# Search all memory types (slower but comprehensive)
await search_tool._arun(query="Python", k=10)

# Search specific type (faster and more precise)
await search_tool._arun(query="Python", memory_type="skills", k=10)
```

## Next Steps

- Learn about [toolbox VFS tools](./toolbox-vfs.md) for tool discovery
- See [middleware documentation](../middleware/overview.md) for automatic memory management
- Explore [complete examples](../integrations/langchain.md) with LangChain agents
