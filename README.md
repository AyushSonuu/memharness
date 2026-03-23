# memharness

> Framework-agnostic memory infrastructure for AI agents

[![PyPI](https://img.shields.io/pypi/v/memharness)](https://pypi.org/project/memharness/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/AyushSonuu/memharness/actions/workflows/ci.yml/badge.svg)](https://github.com/AyushSonuu/memharness/actions/workflows/ci.yml)

**memharness** provides the memory layer for AI agents — persistent, searchable, typed memory with lifecycle management. Works with any agent framework.

📖 [Documentation](https://ayushsonuu.github.io/memharness/) · 📦 [PyPI](https://pypi.org/project/memharness/) · 🐛 [Issues](https://github.com/AyushSonuu/memharness/issues)

## Install

```bash
pip install memharness

# With PostgreSQL
pip install memharness[postgres]

# With HuggingFace embeddings
pip install memharness[embeddings]
```

## Quick Start

```python
from memharness import MemoryHarness

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Store memories
    await harness.add_conversational("thread-1", "user", "I work at SAP")
    await harness.add_knowledge("Python 3.13 has free-threading", source="docs")
    await harness.add_entity("Alice", "PERSON", "Engineer at Acme Corp")

    # Search semantically
    results = await harness.search_knowledge("concurrent programming")

    # Assemble context for any LLM
    from memharness.agents import ContextAssemblyAgent
    ctx = ContextAssemblyAgent(harness)
    context = await ctx.assemble("Tell me about Python", thread_id="thread-1")
    messages = context.to_messages()  # list[BaseMessage] for LangChain
```

## Memory Types

8 types covering the full agent memory taxonomy:

| Type | Storage | What it stores |
|------|---------|---------------|
| **Conversational** | SQL | Chat history per thread |
| **Knowledge Base** | Vector | Facts, documents, reference material |
| **Workflow** | Vector | Reusable multi-step task playbooks |
| **Toolbox** | Vector | Tool definitions (semantic retrieval) |
| **Entity** | Vector | People, organizations, systems |
| **Summary** | Vector | Compressed older conversations |
| **Tool Log** | SQL | Tool execution audit trail |
| **Persona** | Vector | Agent identity and style |

## 7 Self-Awareness Tools

Give any agent the ability to manage its own memory:

```python
from memharness.tools import get_memory_tools

tools = get_memory_tools(harness)  # Returns 7 LangChain BaseTool instances
```

| Tool | What the agent can do |
|------|-----------------------|
| `memory_search` | Search across all memory types |
| `memory_read` | Read a specific memory by ID |
| `memory_write` | Write to any memory type |
| `expand_summary` | Expand compacted summary to full content |
| `summarize_conversation` | Compress a long conversation thread |
| `assemble_context` | Full context assembly (BEFORE-loop) |
| `toolbox_search` | Discover available tools |

## Embedded Agents

4 meta-agents that manage memory autonomously:

| Agent | Purpose |
|-------|---------|
| **ContextAssemblyAgent** | Assembles optimal context before each LLM call |
| **SummarizerAgent** | Compresses long conversations (heuristic or LLM) |
| **EntityExtractorAgent** | Extracts entities from conversations (regex or LLM) |
| **ConsolidatorAgent** | Merges duplicate entities |

## Fast Path / Slow Path

```python
from memharness.core.fast_path import FastPath
from memharness.core.slow_path import SlowPath

# Fast path: user-facing (low latency)
fast = FastPath(harness)
ctx = await fast.process_user_message("thread-1", "How do I deploy?")
# → saves message + assembles context (no extraction, no summarization)

await fast.process_assistant_response("thread-1", response)

# Slow path: background workers (entity extraction, summarization, consolidation)
slow = SlowPath(harness)
results = await slow.run_all()
```

## Summarization (Course-Aligned)

After summarization, context loads **summary + recent messages only** — not all messages:

```
Before: [msg1, msg2, ... msg50]           ← 50 messages in context
After:  [Summary of msg1-40] + [msg41-50] ← 1 summary + 10 recent
```

Configurable via `max_tokens` (default 4000) and `summarize_threshold` (default 80%).

## Backends

| Backend | Best for | Setup |
|---------|----------|-------|
| **SQLite** | Development, testing | `MemoryHarness("sqlite:///memory.db")` |
| **PostgreSQL + pgvector** | Production | `MemoryHarness("postgresql://...")` |
| **In-memory** | Unit tests | `MemoryHarness("memory://")` |

## Docker

```bash
# PostgreSQL + pgvector
docker compose up -d

# With background workers (entity extraction, summarization, consolidation)
docker compose -f docker-compose.yml -f docker-compose.workers.yml up -d
```

Workers run `SlowPath.run_all()` every 5 minutes (configurable via `WORKER_INTERVAL`).

## Use with LangChain

```python
from langchain.agents import create_agent
from memharness.tools import get_memory_tools

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=get_memory_tools(harness),
)
```

See the full [LangChain usage guide](https://ayushsonuu.github.io/memharness/docs/usage-langchain) with middleware examples.

## Configuration

```python
harness = MemoryHarness(
    backend="sqlite:///memory.db",
    embedding_fn=my_embedding_function,  # default: hash-based
)

# Or with HuggingFace embeddings
from memharness.core.embedding import create_huggingface_embedding_fn
harness = MemoryHarness(
    backend="sqlite:///memory.db",
    embedding_fn=create_huggingface_embedding_fn("all-MiniLM-L6-v2"),
)
```

## Documentation

Full docs at **[ayushsonuu.github.io/memharness](https://ayushsonuu.github.io/memharness/)**

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
