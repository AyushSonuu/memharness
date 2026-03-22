# memharness

> Framework-agnostic memory infrastructure for AI agents

[![PyPI version](https://badge.fury.io/py/memharness.svg)](https://badge.fury.io/py/memharness)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AyushSonuu/memharness/actions/workflows/test.yml/badge.svg)](https://github.com/AyushSonuu/memharness/actions/workflows/test.yml)

## Overview

**memharness** is a complete memory infrastructure layer for AI agents. It provides:

- **10 Memory Types**: Conversational, Knowledge Base, Entity, Workflow, Toolbox, Summary, Tool Log, Skills, File, Persona
- **Multiple Backends**: PostgreSQL + pgvector, SQLite + sqlite-vss, In-memory
- **Framework Agnostic**: Works with LangChain, LangGraph, CrewAI, or any custom agent
- **Deterministic + AI Operations**: Simple ops are deterministic, complex ops use embedded agents
- **Self-Exploration Tools**: Agents can explore and manage their own memory
- **Fully Configurable**: All thresholds, schedules, TTLs configurable via YAML or code

## Installation

```bash
# Core (SQLite backend)
pip install memharness

# With PostgreSQL support
pip install memharness[postgres]

# With embedding support
pip install memharness[embeddings]

# Everything
pip install memharness[all]
```

## Quick Start

```python
from memharness import MemoryHarness

# Initialize with SQLite (development)
memory = MemoryHarness("sqlite:///memory.db")

# Or PostgreSQL (production)
memory = MemoryHarness("postgresql://user:pass@localhost/db")

# Write conversational memory (deterministic)
await memory.add_conversational(
    thread_id="chat_001",
    role="user",
    content="How do I deploy to Kubernetes?"
)

# Write knowledge base (deterministic)
await memory.add_knowledge(
    content="Kubernetes deployment guide...",
    source="k8s-docs",
    metadata={"category": "devops"}
)

# Search knowledge base (semantic search)
results = await memory.search_knowledge(
    query="container orchestration",
    k=5
)

# Get curated context for LLM
context = await memory.assemble_context(
    query="deploy my app",
    thread_id="chat_001",
    max_tokens=4000
)
```

## Memory Types

| Type | Storage | Purpose |
|------|---------|---------|
| **Conversational** | SQL | Chat history per thread |
| **Knowledge Base** | Vector | Documents, facts, reference material |
| **Entity** | Vector | People, organizations, systems, concepts |
| **Workflow** | Vector | Reusable step-by-step patterns |
| **Toolbox** | Vector | Tool definitions with VFS discovery |
| **Summary** | Vector | Compressed conversations (expandable) |
| **Tool Log** | SQL | Tool execution audit trail |
| **Skills** | Vector | Learned agent capabilities |
| **File** | Hybrid | Document references |
| **Persona** | Vector | Agent identity blocks |

## Embedded Agents

memharness includes specialized agents for complex memory operations:

```python
from memharness import MemoryHarness
from memharness.agents import AgentConfig

memory = MemoryHarness(
    backend="postgresql://...",
    llm=your_llm,  # Optional: for AI-powered operations
    agents=AgentConfig(
        summarizer={"enabled": True, "triggers": [{"condition": "age > 7d"}]},
        entity_extractor={"enabled": True, "on_write": True},
        consolidator={"enabled": True, "schedule": "0 3 * * *"},
        gc={"enabled": True, "schedule": "0 4 * * 0"},
    )
)
```

## Memory Tools (Self-Exploration)

Agents can explore their own memory using built-in tools:

```python
# Get tools for your agent
tools = memory.get_memory_tools()

# Tools include:
# - memory_search: Search across memory types
# - memory_read: Read specific memory by ID
# - memory_write: Write new memory
# - memory_stats: Get memory statistics
# - toolbox_tree: Explore tool VFS
# - toolbox_grep: Search tools by pattern
```

## Configuration

```yaml
# memharness.yaml
backend: postgresql://localhost/memharness

summarization:
  enabled: true
  triggers:
    - condition: "age > 7d"
      memory_type: conversational
  keep_originals: true
  originals_ttl: 365d

consolidation:
  enabled: true
  schedule: "0 3 * * *"
  similarity_threshold: 0.9

gc:
  enabled: true
  schedule: "0 4 * * 0"
  archive_after: 90d
  delete_after: 365d
```

```python
memory = MemoryHarness.from_config("memharness.yaml")
```

## Framework Integrations

### LangChain

```python
from memharness.integrations.langchain import MemharnessMemory

memory = MemharnessMemory(backend="postgresql://...")
chain = ConversationChain(llm=llm, memory=memory)
```

### LangGraph

```python
from memharness.integrations.langgraph import MemharnessCheckpointer

checkpointer = MemharnessCheckpointer(backend="postgresql://...")
graph = builder.compile(checkpointer=checkpointer)
```

## Documentation

Full documentation: [https://ayushsonuu.github.io/memharness](https://ayushsonuu.github.io/memharness)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).
