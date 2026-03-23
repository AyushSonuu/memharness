# LangChain Integration

Complete working guide for using memharness with LangChain agents, including conversation persistence.

## Installation

```bash
pip install memharness langchain langchain-anthropic
```

## Basic Agent with Memory Tools

Here's a complete, runnable example of a LangChain agent with memory tools:

```python
import asyncio
from memharness import MemoryHarness
from memharness.tools import get_memory_tools
from langchain.agents import create_agent

async def main():
    # Initialize memory harness
    harness = MemoryHarness("sqlite:///memory.db")
    await harness.connect()

    # Get memory tools (7 tools)
    tools = get_memory_tools(harness)

    # Create agent with Anthropic model
    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=tools,
    )

    # Run the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Store a fact: Python was created by Guido van Rossum"}]
    })

    print(response)

    await harness.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Conversation Persistence Middleware

To persist conversation history across agent invocations, use the `MemharnessConversationMiddleware`. This middleware:

- **Before model call**: Loads past messages from `harness.get_conversational()` and injects them into `state[messages]`
- **After model call**: Saves new messages from `state[messages]` to `harness.add_conversational()`

### Complete Middleware Implementation

```python
"""
Conversation persistence middleware for LangChain agents.

This middleware automatically loads and saves conversation history
to/from memharness, enabling stateful multi-turn conversations.
"""
from typing import Any
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memharness import MemoryHarness


class MemharnessConversationMiddleware(AgentMiddleware):
    """
    Middleware for persisting conversation history with memharness.

    Usage:
        harness = MemoryHarness("sqlite:///memory.db")
        await harness.connect()

        middleware = MemharnessConversationMiddleware(harness, thread_id="user-123")

        agent = create_agent(
            model="anthropic:claude-sonnet-4-6",
            tools=get_memory_tools(harness),
            middleware=[middleware],
        )
    """

    def __init__(self, harness: MemoryHarness, thread_id: str):
        """
        Initialize the middleware.

        Args:
            harness: The MemoryHarness instance to use for persistence.
            thread_id: Conversation thread ID to load/save messages under.
        """
        super().__init__()
        self.harness = harness
        self.thread_id = thread_id

    async def abefore_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Load past conversation messages before the model runs.

        Args:
            state: The current agent state containing 'messages'.

        Returns:
            Updated state with historical messages prepended.
        """
        # Load past messages from memharness
        past_messages = await self.harness.get_conversational(self.thread_id, limit=50)

        if not past_messages:
            return state

        # Convert MemoryUnit objects to LangChain message objects
        langchain_messages = []
        for msg in past_messages:
            role = msg.metadata.get("role", "user")
            content = msg.content

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        # Prepend historical messages to current state
        state["messages"] = langchain_messages + state.get("messages", [])

        return state

    async def aafter_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Save new messages to memharness after the model runs.

        Args:
            state: The agent state containing new 'messages'.

        Returns:
            Unmodified state (persistence is a side effect).
        """
        messages = state.get("messages", [])

        # Save only new messages (skip already-persisted historical messages)
        # In practice, you'd track which messages are new vs. loaded from history
        # For simplicity, this example saves all messages in the current turn

        for msg in messages:
            # Determine role
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                continue  # Skip unknown message types

            # Save to memharness
            await self.harness.add_conversational(
                thread_id=self.thread_id,
                role=role,
                content=msg.content,
            )

        return state
```

### Using the Middleware

```python
import asyncio
from memharness import MemoryHarness
from memharness.tools import get_memory_tools
from langchain.agents import create_agent

# Include the MemharnessConversationMiddleware class from above

async def main():
    # Initialize memory harness
    harness = MemoryHarness("sqlite:///memory.db")
    await harness.connect()

    # Create middleware
    middleware = MemharnessConversationMiddleware(harness, thread_id="user-123")

    # Create agent with middleware
    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=get_memory_tools(harness),
        middleware=[middleware],
    )

    # First turn
    response1 = await agent.ainvoke({
        "messages": [{"role": "user", "content": "My name is Alice"}]
    })
    print("Turn 1:", response1)

    # Second turn - agent will remember "Alice" from previous turn
    response2 = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's my name?"}]
    })
    print("Turn 2:", response2)

    await harness.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Complete Working Example

Here's a full end-to-end script you can save as `example.py` and run:

```python
"""
Complete LangChain + memharness example with conversation persistence.

This script demonstrates:
1. Memory tools (7 tools: search, read, write, toolbox_search, expand_summary,
   assemble_context, summarize_conversation)
2. Conversation persistence middleware
3. Multi-turn conversation with memory

Run with: python example.py
"""
import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memharness import MemoryHarness
from memharness.tools import get_memory_tools


class MemharnessConversationMiddleware(AgentMiddleware):
    """Middleware for persisting conversation history with memharness."""

    def __init__(self, harness: MemoryHarness, thread_id: str):
        super().__init__()
        self.harness = harness
        self.thread_id = thread_id

    async def abefore_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """Load past conversation messages before the model runs."""
        past_messages = await self.harness.get_conversational(self.thread_id, limit=50)

        if not past_messages:
            return state

        # Convert MemoryUnit objects to LangChain message objects
        langchain_messages = []
        for msg in past_messages:
            role = msg.metadata.get("role", "user")
            content = msg.content

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        # Prepend historical messages to current state
        state["messages"] = langchain_messages + state.get("messages", [])

        return state

    async def aafter_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """Save new messages to memharness after the model runs."""
        messages = state.get("messages", [])

        for msg in messages:
            # Determine role
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                continue  # Skip unknown message types

            # Save to memharness
            await self.harness.add_conversational(
                thread_id=self.thread_id,
                role=role,
                content=msg.content,
            )

        return state


async def main():
    """Run the complete example."""
    print("=== memharness + LangChain Complete Example ===\n")

    # Initialize memory harness
    harness = MemoryHarness("sqlite:///memory.db")
    await harness.connect()
    print("✓ Connected to memory harness\n")

    # Create middleware for conversation persistence
    middleware = MemharnessConversationMiddleware(harness, thread_id="demo-user")

    # Get memory tools (7 tools)
    tools = get_memory_tools(harness)
    print(f"✓ Loaded {len(tools)} memory tools:")
    for tool in tools:
        print(f"  - {tool.name}")
    print()

    # Create agent with middleware
    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=tools,
        middleware=[middleware],
    )
    print("✓ Created agent with memory tools and conversation middleware\n")

    # Turn 1: Store knowledge
    print("--- Turn 1: Store Knowledge ---")
    response1 = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use memory_write to store this fact: Python was created by Guido van Rossum in 1991.",
                }
            ]
        }
    )
    print(f"Agent: {response1['messages'][-1].content}\n")

    # Turn 2: Recall knowledge
    print("--- Turn 2: Recall Knowledge ---")
    response2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Who created Python? Use memory_search to find out."}]}
    )
    print(f"Agent: {response2['messages'][-1].content}\n")

    # Turn 3: Check conversation history
    print("--- Turn 3: Check Conversation History ---")
    past_messages = await harness.get_conversational("demo-user", limit=10)
    print(f"✓ Found {len(past_messages)} messages in conversation history:")
    for msg in past_messages:
        role = msg.metadata.get("role", "unknown")
        content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"  [{role}] {content_preview}")
    print()

    # Cleanup
    await harness.disconnect()
    print("✓ Disconnected from memory harness")


if __name__ == "__main__":
    asyncio.run(main())
```

## 7 Memory Tools Reference

| Tool Name | Description |
|-----------|-------------|
| `memory_search` | Search across all memory types using semantic similarity |
| `memory_read` | Read a specific memory by its unique ID |
| `memory_write` | Write to ANY memory type: knowledge, entity, workflow, tool_log, conversational, etc. |
| `toolbox_search` | Discover available tools (combines tree view + grep search) |
| `expand_summary` | Expand a compacted summary back to its original content |
| `assemble_context` | Assemble full BEFORE-loop context (persona, history, knowledge, entities, workflows, tools) |
| `summarize_conversation` | Compress conversation history into a summary (preserves originals) |

### memory_write: Universal Write Tool

The `memory_write` tool supports ALL memory types through conditional fields:

```python
# Knowledge base
await tool.ainvoke({
    "memory_type": "knowledge",
    "content": "Python is a high-level programming language",
})

# Entity
await tool.ainvoke({
    "memory_type": "entity",
    "content": "Chief Technology Officer at Example Corp",
    "metadata": {"name": "Alice Smith", "entity_type": "person"},
})

# Workflow
await tool.ainvoke({
    "memory_type": "workflow",
    "task": "Deploy application to production",
    "steps": ["Run tests", "Build Docker image", "Push to registry", "Update k8s"],
    "outcome": "Deployed successfully",
})

# Tool log
await tool.ainvoke({
    "memory_type": "tool_log",
    "thread_id": "user-123",
    "tool_name": "github/create_issue",
    "tool_input": '{"title": "Bug fix", "body": "Fixed the bug"}',
    "tool_output": "Issue #42 created",
    "status": "success",
})

# Conversational
await tool.ainvoke({
    "memory_type": "conversational",
    "thread_id": "user-123",
    "role": "user",
    "content": "Hello, how are you?",
})
```

## Next Steps

- Explore [Memory Types](./concepts/memory-types) to understand what data each type stores
- Read [Context Assembly](./agents/context-assembler) to learn how `assemble_context` works
- Check [Embedded Agents](./agents/overview) for summarization and consolidation patterns

## Fast Path / Slow Path Architecture

memharness supports a **fast path / slow path** architecture for production agent systems:

- **Fast Path** (user-facing, low latency): Save message → assemble context → return. No extraction, no summarization in the hot path.
- **Slow Path** (background workers, async): Entity extraction, summarization, consolidation, garbage collection.

This separation ensures your user-facing agent is always fast, while background workers enrich memory asynchronously.

### Fast Path Example

```python
"""Fast path: user-facing agent interactions."""
import asyncio
from memharness import MemoryHarness
from memharness.core.fast_path import FastPath

async def main():
    # Initialize memory harness
    harness = MemoryHarness('sqlite:///memory.db')
    await harness.connect()

    # Fast path: user-facing
    fast = FastPath(harness)

    # User sends message
    ctx = await fast.process_user_message('thread-1', 'How do I deploy?')
    messages = ctx.to_messages()  # feed to LLM

    # Your LLM responds (using langchain, openai, anthropic, etc.)
    # Example with LangChain:
    # from langchain.chat_models import init_chat_model
    # llm = init_chat_model("anthropic:claude-sonnet-4-6")
    # response = await llm.ainvoke(messages)
    response = "Deploy using docker compose up -d..."

    # Save assistant response
    await fast.process_assistant_response('thread-1', response)

    await harness.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### Slow Path Example

```python
"""Slow path: run periodically (cron, background task, etc.)"""
import asyncio
from memharness import MemoryHarness
from memharness.core.slow_path import SlowPath

async def main():
    # Initialize memory harness
    harness = MemoryHarness('sqlite:///memory.db')
    await harness.connect()

    # Slow path: background workers
    slow = SlowPath(harness)

    # Run all background workers
    results = await slow.run_all()

    # Print results
    for r in results:
        print(f'{r.worker}: processed={r.processed}, errors={r.errors}, '
              f'duration={r.duration_ms:.1f}ms')

    await harness.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### What Each Path Does

**Fast Path** (deterministic, low latency):
1. `process_user_message()` → save user message to conv table + assemble context
2. `process_assistant_response()` → save assistant response to conv table
3. That's it. No extraction, no summarization in the hot path.

**Slow Path** (background workers, async):
1. **Entity Extraction**: Scan new conv messages → extract entities → upsert (update existing, don't duplicate)
2. **Summarization**: When thread gets long → compress older messages
3. **Consolidation**: Scan entity table → merge duplicates
4. **Garbage Collection**: Archive/delete old data

### With LLMs (optional)

You can optionally provide LLMs to the slow path for better entity extraction and summarization:

```python
from langchain.chat_models import init_chat_model
from memharness.core.slow_path import SlowPath

llm = init_chat_model("anthropic:claude-sonnet-4-6")

# Pass LLMs to slow path workers
slow = SlowPath(
    harness,
    entity_extractor_llm=llm,
    summarizer_llm=llm,
)

results = await slow.run_all()
```

Without LLMs, the slow path uses heuristic-based extraction and summarization (good for testing, lower cost).

### Context Assembly Prefers Recent Entities

The context assembly agent (used by fast path) automatically prefers recent entities based on `updated_at` timestamp. This solves the "staleness problem":

- If user says "I work at SAP" (today)
- And historical data says "I work at Google" (last year)
- The entity extractor (slow path) UPSERTs the entity, updating `updated_at`
- Context assembly (fast path) returns "works at SAP" (most recent)

This happens automatically — no extra code needed.
