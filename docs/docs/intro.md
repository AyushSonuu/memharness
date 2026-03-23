---
sidebar_position: 1
slug: /
---

# Introduction

**memharness** is a framework-agnostic Python package that provides complete memory infrastructure for AI agents. It enables any agent framework — LangChain, LangGraph, CrewAI, Deep Agents, or custom — to have persistent, searchable, typed memory with built-in lifecycle management.

## Why memharness?

Most agent frameworks treat memory as an afterthought. memharness makes it a first-class concern:

- **Typed memories** — 10 distinct memory types, each optimized for its use case
- **Any backend** — PostgreSQL, SQLite, or in-memory, swappable with one line
- **Lifecycle management** — Automated summarization, consolidation, and garbage collection
- **Agent tools** — Agents can explore and manage their own memory
- **Framework agnostic** — Works with any Python agent framework

## Install

```bash
pip install memharness
```

## Quick Example

```python
from memharness import MemoryHarness

async with MemoryHarness("sqlite:///memory.db") as memory:
    # Store conversational memory
    await memory.add_conversational("thread1", "user", "What is Python?")
    await memory.add_conversational("thread1", "assistant", "Python is a programming language.")

    # Store knowledge
    await memory.add_knowledge(
        "Python supports async/await since version 3.5",
        source="python-docs",
    )

    # Semantic search
    results = await memory.search_knowledge("async programming")

    # Assemble context for LLM
    context = await memory.assemble_context("Tell me about async", "thread1")
```

## Memory Types

| Type | Purpose | Storage |
|------|---------|---------|
| **Conversational** | Chat history per thread | SQL (ordered) |
| **Knowledge** | Facts, docs, reference material | Vector (semantic) |
| **Entity** | People, orgs, systems, concepts | Vector |
| **Workflow** | Step-by-step procedures | Vector |
| **Toolbox** | Tool definitions with VFS discovery | Vector |
| **Summary** | Compressed memories (expandable) | Vector |
| **Tool Log** | Tool execution audit trail | SQL (ordered) |
| **Skills** | Learned agent capabilities | Vector |
| **File** | Document references | Vector |
| **Persona** | Agent identity and style | Vector |

## Next Steps

- [Getting Started](./getting-started) — Full setup guide
- [Memory Types](./memory-types/conversational) — Deep dive into each type
- [API Reference](./api/harness) — Complete API docs
- [Backends](./backends/sqlite) — Backend configuration
