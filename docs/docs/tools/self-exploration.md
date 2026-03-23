---
sidebar_position: 2
---

# Self-Exploration Tools

Self-exploration tools enable AI agents to introspect and manage their own memory. These tools give agents the ability to search their past experiences, read specific memories, write new information, and understand their memory usage patterns.

## Overview

The self-exploration tools provide complete CRUD operations and context management for agent memory:

1. **MemorySearchTool** - Semantic search across memory types
2. **MemoryReadTool** - Retrieve specific memories by ID
3. **MemoryWriteTool** - Store new information
4. **MemoryStatsTool** - Analyze memory usage and health
5. **ExpandSummaryTool** - Expand compacted summaries to full content
6. **ConversationHistoryTool** - Get conversation history for a thread
7. **AssembleContextTool** - Assemble comprehensive context for queries

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

## ExpandSummaryTool

Expand a compacted summary back to its full original content. This tool implements context expansion - the inverse of context compaction.

### Description
```
Expand a compacted summary back to its full original content.
Use when you need details from a conversation that was previously summarized.
The summary_id comes from the Context Summaries section.
```

### Input Schema

```python
class ExpandSummaryInput(BaseModel):
    summary_id: str = Field(description="The ID of the summary to expand back to full content")
```

### Usage Example

```python
from memharness.tools import ExpandSummaryTool

expand_tool = ExpandSummaryTool(harness=harness)

# Expand a summary to get original messages
result = await expand_tool._arun(
    summary_id="summary_abc123"
)
```

### Use Case: Context Compaction & Expansion

This tool implements the **context expansion** pattern from Lesson 05 of the agent memory course:

1. When context window fills up → **compact** conversation to a summary (stores summary_id + description)
2. When agent needs details → **expand** summary back to full original content

**Benefits:**
- **Lossless compression**: Original content preserved in DB
- **On-demand retrieval**: Agent gets full details when needed
- **Token efficiency**: Keep only summary_id in context, not full content

```python
# Typical workflow:
# 1. Context is compacted (elsewhere in your system)
summary_id = await harness.add_summary(
    summary="Discussion about Python async patterns",
    source_ids=["msg1", "msg2", "msg3"],
    thread_id="thread-123"
)

# 2. Later, agent uses ExpandSummaryTool to get full details
expanded = await expand_tool._arun(summary_id=summary_id)
# Returns: "user: How does async work?\nassistant: Async allows..."
```

## ConversationHistoryTool

Get conversation history for a specific thread as a list of messages.

### Description
```
Get conversation history for a thread as a list of messages.
Returns recent messages from the specified conversation thread.
```

### Input Schema

```python
class ConversationHistoryInput(BaseModel):
    thread_id: str = Field(description="Conversation thread ID")
    limit: int = Field(default=20, description="Max messages to retrieve")
```

### Usage Example

```python
from memharness.tools import ConversationHistoryTool

history_tool = ConversationHistoryTool(harness=harness)

# Get last 20 messages from a thread
result = await history_tool._arun(
    thread_id="thread-123",
    limit=20
)

# Get last 10 messages
result = await history_tool._arun(
    thread_id="thread-456",
    limit=10
)
```

### Example Output

```
Conversation history for thread thread-123 (15 messages):

[2026-03-23 10:30:00] user: Hello!
[2026-03-23 10:30:05] assistant: Hi there! How can I help you today?
[2026-03-23 10:31:00] user: Can you explain async programming?
[2026-03-23 10:31:30] assistant: Async programming allows...
...
```

### Use Case: Thread Review

This tool is useful when agents need to:
- Review what was discussed in a specific thread
- Get context for a conversation they weren't part of initially
- Check message history before continuing a conversation

## AssembleContextTool

Assemble all relevant memory context for a query. This is the most powerful tool for agent self-awareness.

### Description
```
Assemble all relevant memory context for a query.
Returns persona, conversation history, knowledge, workflows, entities, and tools.
Use this when you need comprehensive context about a topic or task.
```

### Input Schema

```python
class AssembleContextInput(BaseModel):
    query: str = Field(description="Query to assemble context for")
    thread_id: str = Field(description="Conversation thread ID")
    max_tokens: int = Field(default=4000, description="Maximum tokens in assembled context")
```

### Usage Example

```python
from memharness.tools import AssembleContextTool

context_tool = AssembleContextTool(harness=harness)

# Assemble full context for a query
result = await context_tool._arun(
    query="How do I deploy the application?",
    thread_id="thread-123",
    max_tokens=4000
)
```

### Example Output

```
## Persona
You are a helpful AI assistant specialized in software development...

## Recent Conversation
user: I'm working on the deployment pipeline
assistant: I can help you with that...

## Relevant Knowledge
- The application uses Docker for containerization
- Deployment is handled via GitHub Actions
- Production environment runs on AWS ECS

## Related Entities
- AWS: Cloud provider for production infrastructure
- Docker: Containerization platform used in deployment

## Relevant Workflows
**Deploy to Production**
1. Run tests locally
2. Push to main branch
3. GitHub Actions builds Docker image
4. Image pushed to ECR
5. ECS service updated
```

### Use Case: Full Context Assembly

This tool is ideal when agents need to:
- Get comprehensive context before starting a complex task
- Understand everything relevant to a specific query
- Make informed decisions based on all available information

The tool gathers:
1. **Persona** - Agent's identity and capabilities
2. **Conversation History** - Recent messages from the thread
3. **Knowledge Base** - Relevant facts and information
4. **Entities** - Related people, organizations, concepts
5. **Workflows** - Applicable procedures and processes

This implements the **Context Assembly Agent** pattern - agents explicitly requesting context assembly rather than having it happen implicitly.

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
