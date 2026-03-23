---
sidebar_position: 1
---

# Memory Middleware

memharness provides [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware) that adds automatic memory capabilities to any agent built with `create_agent`.

## Overview

Middleware runs at specific points in the agent execution loop — before and after each model call. memharness uses this to:

- **Inject memory context** before each model call (so the model always has relevant context)
- **Persist conversations** after each model response (automatic memory storage)
- **Extract entities** from conversations (automatic knowledge building)

```
User Input
    │
    ▼
[MemoryContextMiddleware]  ← injects relevant memories as system message
    │
    ▼
Model Call (LLM)
    │
    ▼
[MemoryPersistenceMiddleware] ← stores user + AI messages
[EntityExtractionMiddleware]  ← extracts entities from response
    │
    ▼
Agent Response
```

## Installation

```bash
pip install memharness[langchain]
```

Requires: `langchain>=1.2.0`, `langchain-core>=1.2.0`

## Quick Start

```python
import asyncio
from memharness import MemoryHarness
from memharness.tools import get_memory_tools
from memharness.middleware import MemoryContextMiddleware, MemoryPersistenceMiddleware
from langchain.agents import create_agent

async def main():
    harness = MemoryHarness("sqlite:///memory.db")
    await harness.connect()

    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=get_memory_tools(harness),
        middleware=[
            MemoryContextMiddleware(harness=harness, thread_id="user-123"),
            MemoryPersistenceMiddleware(harness=harness, thread_id="user-123"),
        ],
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What did we discuss yesterday?"}]
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

See the full example in [`examples/langchain_agent.py`](https://github.com/AyushSonuu/memharness/blob/main/examples/langchain_agent.py).

---

## MemoryContextMiddleware

**Purpose:** Automatically injects relevant memory context before each model call.

Before the model processes each user message, this middleware:
1. Extracts the last user message as a query
2. Calls `harness.assemble_context()` to retrieve relevant memories
3. Prepends the context as a `SystemMessage`

This means the model always has access to relevant past knowledge, entities, skills, and persona — without the user having to repeat themselves.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `harness` | `MemoryHarness` | required | Harness instance |
| `thread_id` | `str` | required | Conversation thread ID |
| `max_tokens` | `int` | `2000` | Max tokens for injected context |

### Example

```python
from memharness.middleware import MemoryContextMiddleware

middleware = MemoryContextMiddleware(
    harness=harness,
    thread_id="session-42",
    max_tokens=3000,
)
```

### What Gets Injected

The context assembly draws from multiple memory types:
- **Persona** — agent identity and style guidelines
- **Knowledge** — relevant facts and documents
- **Entities** — known people, organizations, systems
- **Workflows** — relevant procedures
- **Skills** — learned capabilities
- **Summary** — compressed conversation history

---

## MemoryPersistenceMiddleware

**Purpose:** Automatically stores every conversation turn in conversational memory.

After each model response, this middleware:
1. Extracts the last user message
2. Extracts the AI response
3. Stores both in `conversational` memory with deduplication

This builds a complete, searchable conversation history without any manual memory management.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `harness` | `MemoryHarness` | required | Harness instance |
| `thread_id` | `str` | required | Conversation thread ID |

### Example

```python
from memharness.middleware import MemoryPersistenceMiddleware

middleware = MemoryPersistenceMiddleware(
    harness=harness,
    thread_id="session-42",
)
```

### Deduplication

The middleware tracks stored messages to prevent duplicates when the agent loop retries or when the same message appears multiple times.

---

## EntityExtractionMiddleware

**Purpose:** Automatically extracts entities from AI responses and stores them in entity memory.

After each model response, this middleware:
1. Gets the AI message content
2. Runs the `EntityExtractorAgent` to identify entities
3. Stores extracted entities in entity memory

This gradually builds up the agent's knowledge of people, organizations, concepts, and systems mentioned in conversations.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `harness` | `MemoryHarness` | required | Harness instance |
| `llm` | `BaseChatModel \| None` | `None` | Optional LLM for better extraction |
| `min_confidence` | `float` | `0.5` | Minimum confidence threshold |

### Example

```python
from memharness.middleware import EntityExtractionMiddleware

# Regex-only extraction (no LLM required)
middleware = EntityExtractionMiddleware(harness=harness)

# LLM-enhanced extraction (more accurate)
from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-4o-mini")

middleware = EntityExtractionMiddleware(
    harness=harness,
    llm=llm,
    min_confidence=0.7,
)
```

---

## Combining Middleware

Use multiple middleware together for full memory capabilities:

```python
from memharness.middleware import (
    MemoryContextMiddleware,
    MemoryPersistenceMiddleware,
    EntityExtractionMiddleware,
)
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware  # Built-in LangChain

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=get_memory_tools(harness),
    middleware=[
        # 1. Inject memory context before model call
        MemoryContextMiddleware(harness=harness, thread_id="user-1"),
        # 2. Persist conversations after model response
        MemoryPersistenceMiddleware(harness=harness, thread_id="user-1"),
        # 3. Extract entities from responses
        EntityExtractionMiddleware(harness=harness),
        # 4. LangChain built-in: summarize when context gets long
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 4000),
        ),
    ],
)
```

:::tip Using LangChain's built-in middleware
You don't need to replicate LangChain's summarization — just use `SummarizationMiddleware`.
memharness middleware focuses on **persistent cross-session memory**, while LangChain middleware
handles **in-context management**.
:::

---

## Middleware Execution Order

Middleware runs in the order specified:

**Before model call** (abefore_model): runs top → bottom
**After model response** (aafter_model): runs top → bottom

For memharness:
- Put `MemoryContextMiddleware` first (needs to inject before the model call)
- Put `MemoryPersistenceMiddleware` after (processes the response)
- `EntityExtractionMiddleware` can go last (asynchronous side effect)

---

## Writing Custom Middleware

You can write your own middleware using LangChain's `AgentMiddleware`:

```python
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from memharness import MemoryHarness

class CustomMemoryMiddleware(AgentMiddleware):
    """Example: Inject a specific memory category before model calls."""

    def __init__(self, harness: MemoryHarness, thread_id: str) -> None:
        super().__init__()
        self.harness = harness
        self.thread_id = thread_id

    async def abefore_model(self, state, runtime):
        """Called before each model invocation."""
        messages = state.get("messages", [])
        if not messages:
            return None

        # Custom logic: inject recent skills
        skills = await self.harness.search_skills("", k=5)
        if skills:
            skill_text = "\n".join(f"- {s.content}" for s in skills)
            system_msg = SystemMessage(content=f"Your learned skills:\n{skill_text}")
            return {"messages": [system_msg] + list(messages)}

        return None

    async def aafter_model(self, state, runtime):
        """Called after each model response."""
        # No-op in this example
        return None
```

---

## Requirements

```bash
pip install memharness[langchain]
# langchain >= 1.2.0 required for middleware support
```

---

## Next Steps

- See [complete example](https://github.com/AyushSonuu/memharness/blob/main/examples/langchain_agent.py)
- Learn about [memory tools](../tools/overview.md)
- Explore [LangChain integration](../integrations/langchain.md)
