---
sidebar_position: 2
---

# Summarizer Agent

Compress conversation threads into concise summaries for efficient memory management.

## Overview

The Summarizer Agent generates summaries of conversation threads, enabling efficient memory usage by compressing long message histories into concise representations. It prevents context windows from overflowing while preserving semantic information.

### When to Use

- When conversation threads exceed 50+ messages
- Before passing context to LLM (context window management)
- When triggered by the 80% context threshold
- For archiving old conversations while preserving key information
- As a scheduled background task for maintenance

### Dual-Mode Operation

**Mode 1: Heuristic (No LLM)**
- Extracts first and last messages
- Counts total messages
- Fast and deterministic
- Zero LLM costs
- Output: `"Conversation with 47 message(s). Started with: '...' Latest: '...'"`

**Mode 2: LLM-Powered**
- Uses LangChain with ChatPromptTemplate
- Generates intelligent 2-3 sentence summaries
- Captures main topics and outcomes
- Higher quality, semantic understanding
- Output: `"User discussed Python async patterns. Agent explained event loops and provided code examples. Conversation ended with successful implementation."`

## API Methods

### summarize_thread

Summarize a conversation thread.

```python
async def summarize_thread(
    thread_id: str,
    max_messages: int = 50
) -> str
```

**Parameters**:
- `thread_id`: The conversation thread ID to summarize
- `max_messages`: Maximum number of recent messages to include (default: 50)

**Returns**: Summary string (format depends on mode)

**Example**:
```python
from memharness import MemoryHarness
from memharness.agents import SummarizerAgent
from langchain.chat_models import init_chat_model

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Heuristic mode
    agent_basic = SummarizerAgent(harness)
    summary = await agent_basic.summarize_thread("thread-1", max_messages=50)
    # Output: "Conversation with 47 message(s). Started with: 'Hello...' Latest: 'Thanks!...'"

    # LLM mode
    llm = init_chat_model("gpt-4o-mini")
    agent_smart = SummarizerAgent(harness, llm=llm)
    summary = await agent_smart.summarize_thread("thread-1", max_messages=50)
    # Output: "User requested help with async Python. Agent explained event loops..."
```

### run

Execute the summarizer agent (standard agent interface).

```python
async def run(
    thread_id: str,
    max_messages: int = 50,
    **kwargs
) -> dict[str, Any]
```

**Parameters**:
- `thread_id`: The thread ID to summarize
- `max_messages`: Maximum messages to process
- `**kwargs`: Additional arguments (ignored)

**Returns**: Dictionary with `summary` and `message_count` keys

**Example**:
```python
result = await agent.run(thread_id="thread-1")
# Returns: {"summary": "...", "message_count": 47}
```

## Implementation Details

### Heuristic Mode

The heuristic mode uses a simple template:

```python
def _heuristic_summary(self, messages: list[MemoryUnit]) -> str:
    total = len(messages)
    first_msg = messages[0].content[:100]
    last_msg = messages[-1].content[:100]

    return (
        f"Conversation with {total} message(s). "
        f"Started with: '{first_msg}...' "
        f"Latest: '{last_msg}...'"
    )
```

**Advantages**:
- Instant execution (no API calls)
- Zero cost
- Deterministic output
- Works offline

**Limitations**:
- No semantic understanding
- Limited context capture
- Fixed template format

### LLM Mode

The LLM mode builds a LangChain chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Build conversation text with roles
conversation = "\n".join(
    f"{m.metadata.get('role', 'user')}: {m.content}"
    for m in messages
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversation summarizer. Summarize the "
     "following conversation concisely in 2-3 sentences, "
     "capturing the main topics and outcomes."),
    ("user", "{conversation}")
])

# Build chain
chain = prompt | self.llm | StrOutputParser()

# Generate summary
summary = await chain.ainvoke({"conversation": conversation})
```

**Advantages**:
- Semantic understanding
- Captures key topics and outcomes
- Natural language output
- Contextually relevant

**Limitations**:
- Requires LLM API access
- Incurs API costs
- Slower than heuristic mode

## Triggering Strategies

### 1. Agent-Triggered (Automatic)

The Context Assembly Agent automatically triggers summarization when context exceeds 80%:

```python
from memharness.agents import ContextAssemblyAgent

agent = ContextAssemblyAgent(
    harness,
    max_tokens=4000,
    summarize_threshold=0.8  # Trigger at 80% capacity
)

ctx = await agent.assemble(query="...", thread_id="thread-1")
# If context > 80%, conversation is truncated to last 10 messages
```

### 2. Tool-Called (Inside Loop)

The Summarizer can be exposed as a LangChain tool:

```python
from memharness.tools import SummarizerTool

tool = SummarizerTool(harness, llm=llm)

# Agent decides when to summarize
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[tool, ...],
    system_prompt="When context exceeds 50 messages, use the summarizer tool."
)
```

### 3. Policy-Triggered (Scheduled)

Run summarization on a schedule:

```python
import asyncio
from datetime import datetime

async def nightly_summarization():
    """Summarize all active threads nightly."""
    agent = SummarizerAgent(harness, llm=llm)

    # Get all thread IDs (implementation-specific)
    threads = await harness.get_all_thread_ids()

    for thread_id in threads:
        summary = await agent.summarize_thread(thread_id, max_messages=100)

        # Store summary in summary memory type
        await harness.add_summary(
            thread_id=thread_id,
            summary=summary,
            message_count=len(await harness.get_conversational(thread_id))
        )

# Schedule nightly at 3 AM
# (use APScheduler, Celery, or similar in production)
```

### 4. On-Demand (Manual)

Call directly when needed:

```python
# User-triggered summarization
if user_input == "/summarize":
    agent = SummarizerAgent(harness, llm=llm)
    summary = await agent.summarize_thread(current_thread_id)
    print(f"Summary: {summary}")
```

## Configuration

### YAML Configuration

```yaml
agents:
  summarizer:
    enabled: true
    llm: gpt-4o-mini
    max_messages: 50

    # Trigger conditions
    triggers:
      - condition: "message_count > 50"
        action: summarize
      - condition: "age > 7d"
        action: summarize

    # Context threshold (used by ContextAssemblyAgent)
    context_threshold: 0.8  # 80%
```

### Python Configuration

```python
from memharness.agents import SummarizerAgent
from langchain.chat_models import init_chat_model

# Basic initialization
agent = SummarizerAgent(harness)

# With LLM
llm = init_chat_model("gpt-4o-mini")
agent = SummarizerAgent(harness, llm=llm)

# Usage
summary = await agent.summarize_thread("thread-1", max_messages=50)
```

## Best Practices

### 1. Choose the Right Mode

```python
# For production with budget: Use heuristic mode
agent = SummarizerAgent(harness)  # Free, instant

# For high-quality summaries: Use LLM mode
llm = init_chat_model("gpt-4o-mini")  # Cheap model
agent = SummarizerAgent(harness, llm=llm)
```

### 2. Limit Message Count

```python
# Don't summarize entire history (expensive)
summary = await agent.summarize_thread("thread-1", max_messages=50)

# For archival, summarize in chunks
for chunk_start in range(0, total_messages, 50):
    chunk_summary = await agent.summarize_thread(
        thread_id="thread-1",
        max_messages=50
    )
    await harness.add_summary(thread_id="thread-1", summary=chunk_summary)
```

### 3. Store Summaries

```python
from memharness import MemoryHarness

# Generate summary
agent = SummarizerAgent(harness, llm=llm)
summary = await agent.summarize_thread("thread-1")

# Store in summary memory type
await harness.add_summary(
    thread_id="thread-1",
    summary=summary,
    message_count=len(await harness.get_conversational("thread-1"))
)

# Later: Retrieve summary instead of full history
summaries = await harness.get_summaries("thread-1")
```

### 4. Progressive Summarization

```python
async def progressive_summarize(thread_id: str):
    """Summarize in stages: 50 msgs → 10 msgs → 1 summary."""

    # Stage 1: Recent 50 messages
    recent = await harness.get_conversational(thread_id, limit=50)
    if len(recent) < 50:
        return  # Not enough to summarize

    # Stage 2: Summarize to ~10 representative messages
    summary = await agent.summarize_thread(thread_id, max_messages=50)

    # Stage 3: Store summary and truncate history
    await harness.add_summary(thread_id, summary, message_count=len(recent))
    # Optionally: Delete old messages after summarization
```

## Related Components

- [Context Assembly Agent](./context-assembler) — Uses summarization threshold
- [Summary Memory Type](../memory-types/summary) — Stores generated summaries
- [Conversational Memory](../memory-types/conversational) — Source data for summarization
- [Garbage Collector](./gc) — Can trigger summarization before archival

## Next Steps

- [Entity Extractor](./entity-extractor) — Extract entities from conversations
- [Consolidator](./consolidator) — Merge duplicate memories
- [Context Assembler](./context-assembler) — Automatic context management
