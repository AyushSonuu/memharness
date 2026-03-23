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

## Docker Setup for PostgreSQL

For production deployments, we recommend PostgreSQL with pgvector:

### Using Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: memharness
      POSTGRES_PASSWORD: memharness_dev
      POSTGRES_DB: memharness
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Start the services:

```bash
docker compose up -d
```

Connect with memharness:

```python
from memharness import MemoryHarness

memory = MemoryHarness(
    "postgresql://memharness:memharness_dev@localhost:5432/memharness"
)
await memory.connect()
```

The PostgreSQL backend automatically creates all necessary tables and extensions (pgvector) on first connection.

## LangChain Integration with create_agent

memharness provides seamless integration with LangChain's `create_agent` API via middleware:

```python
from langchain.agents import create_agent
from memharness import MemoryHarness
from memharness.middleware import (
    MemoryContextMiddleware,
    MemoryPersistenceMiddleware,
    EntityExtractionMiddleware
)

# Initialize memharness
harness = MemoryHarness("sqlite:///memory.db")
await harness.connect()

# Create agent with memory middleware
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],  # your tools
    middleware=[
        # Automatically inject relevant context before each model call
        MemoryContextMiddleware(
            harness=harness,
            thread_id="conversation-1",
            max_tokens=2000
        ),
        # Automatically persist conversations after each turn
        MemoryPersistenceMiddleware(
            harness=harness,
            thread_id="conversation-1"
        ),
        # Extract entities from conversations (optional)
        EntityExtractionMiddleware(
            harness=harness,
            model="anthropic:claude-haiku-4"
        )
    ]
)

# Use the agent - memory is handled automatically
response = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

### What the Middleware Does

- **MemoryContextMiddleware**: Before each model call, retrieves relevant memories (conversation history, knowledge, entities) and injects them as context
- **MemoryPersistenceMiddleware**: After each model call, stores the user message and AI response in conversational memory
- **EntityExtractionMiddleware**: Extracts named entities (people, organizations, concepts) from conversations and stores them in entity memory

See [LangChain Integration](./integrations/langchain) for more details.

## Advanced Configuration Examples

### Configuration File

```yaml
# memharness.yaml
backend: postgresql://localhost/memharness

# Namespace isolation
namespace_prefix:
  - org:acme
  - project:chatbot

# Memory-specific settings
conversational:
  max_messages_per_thread: 1000
  auto_summarize_threshold: 50

knowledge:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  chunk_size: 512

entity:
  auto_extraction: true
  entity_types:
    - person
    - organization
    - location
    - concept

# Lifecycle policies
summarization:
  enabled: true
  triggers:
    - condition: "age > 7d"
      memory_type: conversational
  keep_originals: true

consolidation:
  enabled: true
  schedule: "0 3 * * *"  # Daily at 3 AM
  similarity_threshold: 0.9

gc:
  enabled: true
  schedule: "0 4 * * 0"  # Weekly on Sunday
  archive_after: 90d
  delete_archived_after: 365d
```

Load the configuration:

```python
memory = MemoryHarness.from_config("memharness.yaml")
await memory.connect()
```

### Environment Variables

```bash
# Backend
export MEMHARNESS_BACKEND="postgresql://localhost/memharness"

# Namespace
export MEMHARNESS_NAMESPACE="org:acme,project:chatbot"

# Optional: config file path
export MEMHARNESS_CONFIG_PATH="./config/memharness.yaml"

# Use in code
memory = MemoryHarness.from_env()
```

### Programmatic Configuration

```python
from memharness import MemoryHarness
from memharness.core.config import MemharnessConfig

config = MemharnessConfig(
    namespace_prefix=("org:acme", "user:alice"),
    auto_summarize=True,
    auto_extract_entities=True,
)

memory = MemoryHarness(
    backend="postgresql://localhost/memharness",
    config=config
)
```

## Troubleshooting

### Connection Issues

**Problem**: `Connection refused` when connecting to PostgreSQL

**Solution**:
```bash
# Check if PostgreSQL is running
docker compose ps

# Check logs
docker compose logs postgres

# Ensure port is not blocked
telnet localhost 5432
```

### pgvector Extension Not Found

**Problem**: `ERROR: could not open extension control file`

**Solution**: Use the official pgvector image:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16  # Use this image
```

### Import Errors

**Problem**: `ImportError: No module named 'langchain'`

**Solution**: Install the appropriate extras:
```bash
# For LangChain integration
pip install memharness[langchain]

# For all features
pip install memharness[all]
```

### Memory Not Persisting

**Problem**: Memories disappear after restarting

**Solution**: Ensure you're using a persistent backend:
```python
# Bad - in-memory only
memory = MemoryHarness("memory://")

# Good - persistent
memory = MemoryHarness("sqlite:///memory.db")
```

### Slow Search Performance

**Problem**: Searches are slow with many memories

**Solutions**:
1. **Use PostgreSQL with pgvector** for production (HNSW indexing)
2. **Add filters** to narrow search scope:
```python
results = await memory.search_knowledge(
    query="Python async",
    filters={"source": "docs"},  # Narrow by source
    k=5
)
```
3. **Increase k cautiously** - more results = more time

### Context Window Overflow

**Problem**: Too much context, model refuses or truncates

**Solution**: Adjust max_tokens in context assembly:
```python
context = await memory.assemble_context(
    query="user question",
    thread_id="chat-1",
    max_tokens=2000  # Reduce if needed
)
```

Or use MemoryContextMiddleware with a limit:
```python
MemoryContextMiddleware(harness=harness, thread_id="chat-1", max_tokens=2000)
```

## Next Steps

- [Memory Types](./memory-types/conversational) — Deep dive into all 10 types
- [LangChain Integration](./integrations/langchain) — Full middleware examples
- [Middleware Guide](./middleware/overview) — Understanding memory middleware
- [Configuration](./concepts/configuration) — Full configuration reference
- [Backends](./backends/sqlite) — Backend setup guides
