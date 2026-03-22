---
sidebar_position: 2
---

# Getting Started

Get up and running with memharness in minutes.

## Installation

```bash
# Core package (includes SQLite backend)
pip install memharness

# With PostgreSQL support
pip install memharness[postgres]

# With embedding models
pip install memharness[embeddings]

# Everything
pip install memharness[all]
```

## Requirements

- Python 3.13+
- For PostgreSQL: PostgreSQL 15+ with pgvector extension

## Quick Start

### 1. Basic Usage (SQLite)

```python
import asyncio
from memharness import MemoryHarness

async def main():
    # Initialize with SQLite (development)
    async with MemoryHarness("sqlite:///memory.db") as memory:

        # Write conversational memory
        await memory.add_conversational(
            thread_id="chat_001",
            role="user",
            content="How do I deploy to Kubernetes?"
        )

        # Write knowledge base
        await memory.add_knowledge(
            content="Kubernetes is a container orchestration platform...",
            source="k8s-docs"
        )

        # Search knowledge
        results = await memory.search_knowledge("container orchestration")
        for r in results:
            print(f"[{r.score:.2f}] {r.content[:100]}...")

        # Get curated context for your LLM
        context = await memory.assemble_context(
            query="deploy my app",
            thread_id="chat_001"
        )
        print(context)

asyncio.run(main())
```

### 2. Production Setup (PostgreSQL)

```python
from memharness import MemoryHarness

# Initialize with PostgreSQL
memory = MemoryHarness(
    "postgresql://user:pass@localhost:5432/mydb"
)

# Use the same API
await memory.add_knowledge(...)
```

### 3. With Configuration File

```yaml
# memharness.yaml
backend: postgresql://localhost/memharness

summarization:
  enabled: true
  triggers:
    - condition: "age > 7d"
      memory_type: conversational
  keep_originals: true

consolidation:
  enabled: true
  schedule: "0 3 * * *"

gc:
  enabled: true
  schedule: "0 4 * * 0"
  archive_after: 90d
```

```python
memory = MemoryHarness.from_config("memharness.yaml")
```

## Basic Operations

### Writing Memory

```python
# Conversational (chat history)
await memory.add_conversational(thread_id, role, content)

# Knowledge Base (documents)
await memory.add_knowledge(content, source=None, metadata=None)

# Entity (people, systems)
await memory.add_entity(name, entity_type, description)

# Workflow (patterns)
await memory.add_workflow(task, steps, outcome)

# Tool (for toolbox)
await memory.add_tool(server, tool_name, description, parameters)
```

### Reading/Searching Memory

```python
# Get conversation history
messages = await memory.get_conversational(thread_id, limit=50)

# Search knowledge base
results = await memory.search_knowledge(query, k=5)

# Search entities
entities = await memory.search_entity(query, entity_type=None)

# Search workflows
workflows = await memory.search_workflow(query)
```

### Context Assembly

```python
# Get curated context for your LLM
context = await memory.assemble_context(
    query="user question",
    thread_id="chat_001",
    max_tokens=4000
)

# Returns formatted markdown:
# ## Conversation Memory
# ...
# ## Knowledge Base Memory
# ...
# ## Entity Memory
# ...
```

## Next Steps

- [Memory Types](./concepts/memory-types) — Deep dive into all 10 types
- [Configuration](./concepts/configuration) — Full configuration options
- [Backends](./backends/sqlite) — Backend setup guides
