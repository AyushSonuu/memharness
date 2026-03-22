---
sidebar_position: 1
slug: /
---

# Introduction

**memharness** is a framework-agnostic memory infrastructure layer for AI agents. It provides everything you need to give your agents persistent, searchable, typed memory.

## Why memharness?

AI agents are **stateless by default** — brilliant in one conversation, blank slate the next. memharness treats memory as **infrastructure**: external, persistent, and structured.

```
┌─────────────────────────────────────┐
│         YOUR AGENT                   │
│  (LangChain | CrewAI | Custom)      │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│           memharness                 │  ← Memory Infrastructure
│  • 10 Memory Types                   │
│  • Deterministic + AI Operations     │
│  • Self-Exploration Tools            │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  PostgreSQL | SQLite | In-Memory    │
└─────────────────────────────────────┘
```

## Key Features

### 🧠 10 Memory Types

| Type | Purpose |
|------|---------|
| **Conversational** | Chat history per thread |
| **Knowledge Base** | Documents, facts, reference material |
| **Entity** | People, organizations, systems |
| **Workflow** | Reusable step-by-step patterns |
| **Toolbox** | Tool definitions with VFS discovery |
| **Summary** | Compressed conversations (expandable) |
| **Tool Log** | Tool execution audit trail |
| **Skills** | Learned agent capabilities |
| **File** | Document references |
| **Persona** | Agent identity blocks |

### ⚡ Deterministic + AI Operations

Simple operations are **deterministic** (no AI needed). Complex operations use **embedded agents**.

```python
# DETERMINISTIC — direct to storage
await memory.add_conversational(thread_id, role, content)
await memory.search_knowledge(query)

# AI-ASSISTED — when intelligence needed
await memory.extract_entities(content)  # Uses LLM
await memory.summarize(thread_id)       # Uses LLM
```

### 🔧 Self-Exploration Tools

Agents can explore and manage their own memory:

```python
tools = memory.get_memory_tools()
# memory_search, memory_stats, toolbox_tree, toolbox_grep, etc.
```

### 📦 Multiple Backends

- **PostgreSQL + pgvector** — Production-grade
- **SQLite** — Development and local use
- **In-memory** — Testing

### 🔌 Framework Integrations

Works with LangChain, LangGraph, CrewAI, or any custom agent.

## Quick Example

```python
from memharness import MemoryHarness

async with MemoryHarness("sqlite:///memory.db") as memory:
    # Write
    await memory.add_conversational("t1", "user", "Deploy to K8s")
    await memory.add_knowledge("K8s guide...", source="docs")

    # Search
    results = await memory.search_knowledge("kubernetes")

    # Assemble context for LLM
    context = await memory.assemble_context("deploy app", "t1")
```

## Next Steps

- [Getting Started](./getting-started) — Install and basic usage
- [Memory Types](./concepts/memory-types) — Understanding the 10 types
- [API Reference](./api/harness) — Full API documentation
