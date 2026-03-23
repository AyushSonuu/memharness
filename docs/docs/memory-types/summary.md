---
sidebar_position: 1
---

# Summary Memory

Summary memory stores compressed summaries of larger memory collections with bidirectional links to source memories, enabling hierarchical compression and efficient context management.

## Overview

Summary memory implements a powerful compression strategy for managing long conversations, large document collections, or extensive knowledge bases. Each summary acts as a compressed representation of multiple source memories, with explicit links that allow "expansion" back to the original content when needed.

This memory type excels at:
- **Context compression**: Reduce token usage by replacing detailed histories with concise summaries
- **Hierarchical summarization**: Create multi-level summaries (summary of summaries)
- **Selective expansion**: Retrieve full details only when necessary
- **Thread management**: Organize summaries by conversation or session threads

The bidirectional linking (summary → sources, sources → summary) enables sophisticated memory management strategies where agents can work with compressed context but expand to details on demand.

## When to Use

Use summary memory to:
- **Compress long conversations**: Summarize older messages to stay within context limits
- **Manage document collections**: Create high-level overviews of large knowledge bases
- **Enable progressive detail**: Start with summaries, drill down to sources as needed
- **Implement memory consolidation**: Periodically compress older memories into summaries
- **Support multi-resolution context**: Maintain both high-level and detailed views

Summary memory is essential for long-running agents that accumulate extensive conversational or knowledge history.

## Storage Strategy

- **Backend**: VECTOR (semantic search with HNSW indexing)
- **Default k**: 3 results (summaries are typically broad, so fewer results)
- **Embeddings**: Yes (enables semantic similarity search across summaries)
- **Ordered**: No (accessed by relevance, though summaries may contain temporal info)
- **Expansion**: Supports bidirectional traversal to source memories

## Schema

Each summary memory includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `summary_type` | string | Yes | Type of summary: "conversation", "document", "session", "topic" |
| `source_ids` | array | Yes | List of memory IDs that this summary represents |
| `thread_id` | string | No | Thread/session ID if summarizing a conversation |
| `time_range_start` | datetime | No | Start timestamp of summarized content |
| `time_range_end` | datetime | No | End timestamp of summarized content |
| `source_count` | integer | No | Number of source memories summarized |

Additional fields are stored in the `metadata` dictionary.

## API Methods

### Adding Summaries

```python
async def add_summary(
    summary: str,
    source_ids: list[str],
    thread_id: str | None = None,
) -> str:
    """
    Add a summary that references source memories.

    Args:
        summary: The summary text
        source_ids: List of memory IDs that this summary is derived from
        thread_id: Optional thread ID if this summarizes a conversation

    Returns:
        The ID of the created summary memory
    """
```

### Expanding Summaries

```python
async def expand_summary(summary_id: str) -> list[MemoryUnit]:
    """
    Expand a summary to retrieve its source memories.

    Args:
        summary_id: The ID of the summary to expand

    Returns:
        List of source MemoryUnit objects that the summary was derived from

    Raises:
        KeyError: If the summary is not found
    """
```

## Examples

### Compressing Conversation History

```python
from memharness import MemoryHarness

harness = MemoryHarness(backend="sqlite:///memory.db")

# Long conversation has accumulated many messages
thread_id = "chat-123"
messages = await harness.get_conversational(thread_id, limit=50)

# Compress the first 30 messages into a summary
old_messages = messages[:30]
message_ids = [msg.id for msg in old_messages]

summary_text = """
User asked about Python async programming, specifically about the Global
Interpreter Lock (GIL) and how async/await works. Discussion covered:
- GIL limitations and when it matters
- asyncio event loop mechanics
- Differences between threading, multiprocessing, and async
- Best practices for I/O-bound vs CPU-bound tasks
User expressed preference for async patterns in web applications.
"""

summary_id = await harness.add_summary(
    summary=summary_text,
    source_ids=message_ids,
    thread_id=thread_id
)

print(f"Created summary {summary_id} covering {len(message_ids)} messages")
```

### Progressive Detail Retrieval

```python
# Start with summaries for efficient context loading
recent_messages = await harness.get_conversational("chat-123", limit=10)
summaries = await harness.search(
    query="Python async programming",
    memory_type=MemoryType.SUMMARY,
    k=2
)

# Build initial context with summaries
context = []
for summary in summaries:
    context.append(f"Previous discussion: {summary.content}")

# If agent needs more detail, expand specific summaries
if needs_more_detail:
    detailed_messages = await harness.expand_summary(summaries[0].id)
    for msg in detailed_messages:
        context.append(f"[Detail] {msg.content}")
```

### Hierarchical Summarization

```python
# Create first-level summaries (daily summaries)
day1_messages = await harness.get_conversational("chat-123", limit=100)
day1_msg_ids = [m.id for m in day1_messages]

day1_summary = await harness.add_summary(
    summary="Day 1: User onboarding, discussed Python basics, set up dev environment",
    source_ids=day1_msg_ids,
    thread_id="chat-123"
)

day2_summary = await harness.add_summary(
    summary="Day 2: Advanced Python topics, async programming, testing strategies",
    source_ids=[...],  # Day 2 message IDs
    thread_id="chat-123"
)

# Create second-level summary (weekly summary of daily summaries)
weekly_summary = await harness.add_summary(
    summary="Week 1: New user learned Python fundamentals through advanced topics",
    source_ids=[day1_summary, day2_summary],  # Summarizing summaries!
    thread_id="chat-123"
)
```

### Automatic Memory Consolidation

```python
async def consolidate_old_memories(thread_id: str, harness: MemoryHarness):
    """Automatically summarize older conversation segments."""

    messages = await harness.get_conversational(thread_id, limit=1000)

    # Summarize messages older than 1 hour in chunks of 20
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(hours=1)

    old_messages = [m for m in messages if m.created_at < cutoff]

    # Process in chunks of 20
    chunk_size = 20
    for i in range(0, len(old_messages), chunk_size):
        chunk = old_messages[i:i+chunk_size]
        chunk_ids = [m.id for m in chunk]

        # Generate summary (using LLM or rule-based)
        summary_text = await generate_summary(chunk)

        await harness.add_summary(
            summary=summary_text,
            source_ids=chunk_ids,
            thread_id=thread_id
        )

        # Optionally delete source messages to save space
        # for msg in chunk:
        #     await harness.delete(msg.id)

    print(f"Consolidated {len(old_messages)} messages into summaries")
```

## Best Practices

1. **Include metadata about coverage**: Store time ranges, message counts, and topics covered in metadata for quick assessment

2. **Balance compression vs. detail**: Don't over-summarize - keep enough detail that expansion isn't always necessary

3. **Use hierarchical summaries**: Create multiple levels (message → daily → weekly) for very long histories

4. **Preserve critical information**: Ensure important entities, decisions, and facts are retained in summaries

5. **Link related summaries**: Use metadata to connect related summaries (e.g., same topic across different threads)

6. **Clean up orphaned sources**: When deleting summarized memories, ensure summaries are updated or also removed

## Integration with Other Memory Types

Summary memory complements other memory types:

- **Conversational**: Primary use case - summarizing chat history to manage context length
- **Knowledge**: Summarize document collections for efficient retrieval
- **Entity**: Extract and preserve entity mentions when creating summaries
- **Workflow**: Summarize completed workflow sequences for learning patterns
- **Tool Log**: Compress extensive tool execution logs into high-level activity summaries

## Performance Notes

- **Token efficiency**: Summaries dramatically reduce token usage in LLM context windows
- **Semantic search**: Use embeddings to find relevant summaries without knowing exact content
- **Lazy expansion**: Only expand summaries when detail is actually needed
- **Small default k**: Returns 3 summaries by default since they typically cover broad topics
- **Async traversal**: Both summary creation and expansion are fully async for performance
- **Cached source lookups**: Frequently expanded summaries benefit from backend caching
