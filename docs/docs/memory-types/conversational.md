---
sidebar_position: 1
---

# Conversational Memory

Store and retrieve chat history and dialogue between users and AI agents.

## Overview

Conversational memory provides ordered, time-series storage for chat messages. It's the foundation for maintaining context across multi-turn conversations and enables agents to reference past interactions.

### When to Use

- Chat applications with ongoing conversations
- Multi-turn agent interactions
- Conversation summarization (paired with summary memory)
- Audit trails for agent interactions

### Storage Strategy

**Backend**: SQL (SQLite, PostgreSQL)
**Why**: Conversational memory requires:
- **Ordering** by timestamp (critical for chat context)
- **Fast retrieval** by thread_id
- **ACID compliance** for conversation integrity

Stored in SQL with B-tree indexes on `(thread_id, created_at)` for efficient ordered retrieval.

## API Methods

### add_conversational

Add a message to a conversation thread.

```python
async def add_conversational(
    thread_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None
) -> str
```

**Parameters**:
- `thread_id`: Unique identifier for the conversation
- `role`: Speaker role (`"user"`, `"assistant"`, `"system"`)
- `content`: The message text
- `metadata`: Optional metadata (timestamp, tool_calls, etc.)

**Returns**: Memory ID

**Example**:
```python
from memharness import MemoryHarness

async with MemoryHarness("sqlite:///memory.db") as harness:
    # User message
    await harness.add_conversational(
        thread_id="chat-123",
        role="user",
        content="What is the capital of France?"
    )

    # Assistant response
    await harness.add_conversational(
        thread_id="chat-123",
        role="assistant",
        content="The capital of France is Paris."
    )

    # System message (optional)
    await harness.add_conversational(
        thread_id="chat-123",
        role="system",
        content="User preferences: verbose explanations"
    )
```

### get_conversational

Retrieve conversation history for a thread.

```python
async def get_conversational(
    thread_id: str,
    limit: int = 50
) -> list[MemoryUnit]
```

**Parameters**:
- `thread_id`: The conversation thread
- `limit`: Maximum messages to retrieve (default: 50)

**Returns**: List of MemoryUnit objects, ordered oldest to newest

**Example**:
```python
messages = await harness.get_conversational("chat-123", limit=10)

for msg in messages:
    role = msg.metadata["role"]
    print(f"{role}: {msg.content}")

# Output:
# user: What is the capital of France?
# assistant: The capital of France is Paris.
```

### clear_thread

Clear all messages from a conversation thread.

```python
await harness.clear_thread("chat-123")
```

## Schema/Metadata Structure

Each conversational memory unit contains:

```python
{
    "id": "uuid",
    "content": "The message content",
    "memory_type": "conversational",
    "namespace": ("conversational", "chat-123"),
    "metadata": {
        "role": "user" | "assistant" | "system",
        "thread_id": "chat-123",
        # Optional fields:
        "timestamp": "2026-03-23T10:00:00Z",
        "tool_calls": [...],  # For assistant messages with tool usage
        "model": "claude-sonnet-4-6"
    },
    "created_at": "2026-03-23T10:00:00Z",
    "embedding": null  # Not embedded by default
}
```

## Best Practices

### 1. Thread ID Design

Use descriptive, hierarchical thread IDs:

```python
# Good
thread_id = f"user:{user_id}:session:{session_id}"
thread_id = "support-ticket:12345"

# Bad
thread_id = "123"  # Too generic
```

### 2. Limit History Length

Retrieve only what you need to avoid context overflow:

```python
# Recent context only
recent = await harness.get_conversational("chat-123", limit=10)

# Full history (careful!)
all_messages = await harness.get_conversational("chat-123", limit=1000)
```

### 3. Use with Summarization

For long conversations, pair with summary memory:

```python
messages = await harness.get_conversational("chat-123")

if len(messages) > 50:
    # Summarize old messages
    summary = await harness.add_summary(
        content="User discussed deployment issues...",
        source_ids=[m.id for m in messages[:40]]
    )

    # Keep recent + summary
    recent = messages[-10:]
```

### 4. Metadata for Rich Context

Store additional context in metadata:

```python
await harness.add_conversational(
    thread_id="chat-123",
    role="user",
    content="Create a GitHub issue",
    metadata={
        "intent": "tool_request",
        "entities": ["GitHub"],
        "timestamp": datetime.utcnow().isoformat()
    }
)
```

### 5. Use with LangChain

For automatic persistence with LangChain agents:

```python
from memharness.middleware import MemoryPersistenceMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    middleware=[
        MemoryPersistenceMiddleware(
            harness=harness,
            thread_id="chat-123"
        )
    ]
)

# Conversations are automatically stored
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

## SQL Storage Details

Conversational memory uses the following schema:

```sql
CREATE TABLE memory_store (
    id UUID PRIMARY KEY,
    namespace TEXT[] NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Critical index for conversational retrieval
CREATE INDEX idx_conversational_thread
ON memory_store (memory_type, namespace, created_at)
WHERE memory_type = 'conversational';
```

This allows:
- Fast lookups by thread_id (namespace)
- Efficient ordering by time
- Metadata filtering (e.g., by role)

## Related Memory Types

- [Summary Memory](./summary) — Compress long conversations
- [Tool Log Memory](./tool-log) — Track tool executions within conversations
- [Entity Memory](./entity) — Extract entities mentioned in conversations

## Next Steps

- [Knowledge Base Memory](./knowledge-base) — Store facts and documents
- [Summary Memory](./summary) — Compress conversation history
- [LangChain Integration](../integrations/langchain) — Automatic persistence
