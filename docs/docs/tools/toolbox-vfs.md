---
sidebar_position: 3
---

# Toolbox VFS Tools

Toolbox VFS (Virtual File System) tools enable AI agents to explore and discover available tools in their environment. These tools provide a hierarchical view of the agent's capabilities, making it easier for agents to understand what actions they can perform.

## Overview

The toolbox VFS tools help agents answer questions like:
- "What tools do I have available?"
- "How do I search for files?"
- "What memory operations can I perform?"

Two tools provide different exploration methods:

1. **ToolboxTreeTool** - Browse tools hierarchically like a file system
2. **ToolboxGrepTool** - Search for tools by name or description

## ToolboxTreeTool

Display a tree view of available tools organized by server/category.

### Description
```
Display a tree view of available tools in your toolbox.
Shows tools organized by server/category in a hierarchical view.
```

### Input Schema

```python
class ToolboxTreeInput(BaseModel):
    path: str = Field(default="/", description="Path to start the tree from")
    depth: int = Field(default=3, description="Maximum depth to display (1-5)")
```

### Usage Example

```python
from memharness.tools import ToolboxTreeTool

tree_tool = ToolboxTreeTool(harness=harness)

# Show all tools (default: depth=3)
result = await tree_tool._arun()

# Show tools from a specific path
result = await tree_tool._arun(
    path="/memory",
    depth=2
)

# Show only top-level categories
result = await tree_tool._arun(
    path="/",
    depth=1
)
```

### Example Output

```
/
├── memory/
│   ├── search - Search memory using semantic similarity
│   ├── read - Read a specific memory by ID
│   ├── write - Write new information to memory
│   └── stats - Get memory statistics
├── toolbox/
│   ├── tree - Display tool hierarchy
│   └── grep - Search for tools by pattern
└── external/
    ├── web_search - Search the internet
    └── calculator - Perform calculations
```

### Use Cases

**1. Agent Self-Discovery**
```python
# Agent wants to know what it can do
agent_query = "What capabilities do I have?"
# Agent uses: await tree_tool._arun(path="/", depth=2)
```

**2. Exploring Categories**
```python
# Agent wants to know about memory operations
agent_query = "What memory operations can I perform?"
# Agent uses: await tree_tool._arun(path="/memory", depth=2)
```

**3. Understanding Tool Organization**
```python
# Agent wants a quick overview
agent_query = "Show me my main tool categories"
# Agent uses: await tree_tool._arun(path="/", depth=1)
```

## ToolboxGrepTool

Search for tools by name or description using regex patterns.

### Description
```
Search for tools by name or description pattern.
Useful when you know part of a tool name or what it does.
```

### Input Schema

```python
class ToolboxGrepInput(BaseModel):
    pattern: str = Field(
        description="Regex pattern to search for in tool names and descriptions"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the search is case-sensitive"
    )
```

### Usage Example

```python
from memharness.tools import ToolboxGrepTool

grep_tool = ToolboxGrepTool(harness=harness)

# Search by tool name (case-insensitive)
result = await grep_tool._arun(
    pattern="search",
    case_sensitive=False
)

# Search by description
result = await grep_tool._arun(
    pattern="memory.*write",
    case_sensitive=False
)

# Case-sensitive search
result = await grep_tool._arun(
    pattern="Search",
    case_sensitive=True
)
```

### Example Output

```
Matching tools:
--------------
/memory/search - Search memory using semantic similarity
/toolbox/grep - Search for tools by name or description
/external/web_search - Search the internet for information
```

### Use Cases

**1. Finding Related Tools**
```python
# Agent needs database-related tools
agent_query = "What database tools are available?"
# Agent uses: await grep_tool._arun(pattern="database|db|sql")
```

**2. Searching by Action**
```python
# Agent wants to know about write operations
agent_query = "How can I write or save data?"
# Agent uses: await grep_tool._arun(pattern="write|save|store")
```

**3. Exact Name Lookup**
```python
# Agent remembers part of a tool name
agent_query = "Is there a tool called 'memory_stats'?"
# Agent uses: await grep_tool._arun(pattern="memory_stats", case_sensitive=False)
```

## VFS Concept

The toolbox VFS organizes tools in a virtual file system structure:

```
/                          Root directory
├── memory/               Memory operations category
│   ├── search           Tool: MemorySearchTool
│   ├── read             Tool: MemoryReadTool
│   ├── write            Tool: MemoryWriteTool
│   └── stats            Tool: MemoryStatsTool
├── toolbox/              Toolbox exploration category
│   ├── tree             Tool: ToolboxTreeTool
│   └── grep             Tool: ToolboxGrepTool
└── custom/              User-defined tools category
    └── ...              Your custom tools
```

### Path Conventions

- **Paths start with `/`** - Root is always `/`
- **Categories use trailing `/`** - Example: `/memory/`
- **Tools are leaf nodes** - Example: `/memory/search`
- **Depth controls nesting** - `depth=1` shows only top-level

## Integration with LangChain Agents

Toolbox VFS tools work seamlessly with LangChain agents for self-discovery:

```python
from memharness import MemoryHarness
from memharness.tools import get_memory_tools
from langchain.agents import create_agent

# Initialize harness
harness = MemoryHarness("sqlite:///memory.db")
await harness.connect()

# Get all tools (includes VFS tools)
tools = get_memory_tools(harness)

# Create agent with self-exploration capabilities
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=tools,
    system_prompt=(
        "You are a helpful AI assistant with persistent memory. "
        "If a user asks what you can do, use your toolbox tools to explore "
        "and explain your capabilities."
    )
)

# Agent can now answer questions about its own capabilities
response = await agent.ainvoke({
    "messages": [
        {"role": "user", "content": "What can you do? What tools do you have?"}
    ]
})
# Agent will use ToolboxTreeTool to discover and list its capabilities
```

## Typical Agent Workflow

Here's how agents typically use VFS tools:

### Scenario 1: User Asks "What can you do?"

```python
# 1. Agent uses tree tool for overview
tree_result = await tree_tool._arun(path="/", depth=2)

# 2. Agent explains capabilities based on tree structure
agent_response = """
I have several capabilities:

**Memory Operations**: I can search, read, write, and analyze my memory
**Tool Discovery**: I can explore available tools
**External Tools**: I can search the web, perform calculations, etc.

Would you like details on any specific category?
"""
```

### Scenario 2: User Asks About Specific Capability

```python
# 1. Agent uses grep to find relevant tools
grep_result = await grep_tool._arun(pattern="database")

# 2. Agent uses tree to show category structure
tree_result = await tree_tool._arun(path="/database", depth=2)

# 3. Agent explains the tools found
agent_response = """
I have these database-related tools:
- /database/query - Execute SQL queries
- /database/schema - Get table schemas
- /database/backup - Create database backups
"""
```

### Scenario 3: Agent Self-Learning

```python
# When initialized, agent explores its own capabilities
initial_exploration = await tree_tool._arun(path="/", depth=1)

# Agent stores this information in knowledge base
from memharness.tools import MemoryWriteTool

write_tool = MemoryWriteTool(harness=harness)
await write_tool._arun(
    memory_type="knowledge_base",
    content=f"My available tool categories: {initial_exploration}",
    metadata={"type": "self_knowledge", "category": "capabilities"}
)
```

## Best Practices

### 1. Start Broad, Then Narrow

Use tree with increasing depth to drill down:

```python
# Step 1: Overview
await tree_tool._arun(depth=1)

# Step 2: Explore specific category
await tree_tool._arun(path="/memory", depth=2)

# Step 3: Deep dive if needed
await tree_tool._arun(path="/memory/advanced", depth=3)
```

### 2. Use Grep for Known Patterns

When you know what you're looking for:

```python
# Good: Specific search
await grep_tool._arun(pattern="search|find|query")

# Avoid: Too broad
await grep_tool._arun(pattern=".*")  # Returns everything
```

### 3. Combine Both Tools

Use grep to find, then tree to explore:

```python
# 1. Find tools related to "files"
grep_result = await grep_tool._arun(pattern="file")

# 2. Explore the file category
tree_result = await tree_tool._arun(path="/file", depth=2)
```

### 4. Cache Common Queries

Store frequently used tool lists in memory:

```python
# Get overview once
overview = await tree_tool._arun(depth=2)

# Store in knowledge base
await write_tool._arun(
    memory_type="knowledge_base",
    content=f"Tool overview: {overview}",
    metadata={"type": "tool_cache"}
)
```

## Regex Pattern Examples

For ToolboxGrepTool, here are useful regex patterns:

```python
# Find all search-related tools
pattern = "search|find|query|lookup"

# Find write/modification tools
pattern = "write|update|save|store|create"

# Find read/retrieval tools
pattern = "read|get|fetch|retrieve|load"

# Find memory-specific tools
pattern = "^memory"  # Tools starting with "memory"

# Find tools with specific keywords
pattern = "database.*connection"  # Database connection tools
```

## Next Steps

- Learn about [self-exploration tools](./self-exploration.md) for memory introspection
- See the [tools overview](./overview.md) for all available tools
- Explore [LangChain integration](../integrations/langchain.md) for complete examples
