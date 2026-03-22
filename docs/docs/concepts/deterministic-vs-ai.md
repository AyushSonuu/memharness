---
sidebar_position: 2
---

# Deterministic vs AI Operations

memharness follows a **deterministic-first** design principle: simple operations work without AI, complex operations use embedded agents.

## The Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                 DETERMINISTIC (No AI needed)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXPLICIT WRITES — User specifies type, goes directly to store: │
│                                                                  │
│    memory.add_conversational(msg)  →  SQL Store (direct)        │
│    memory.add_entity(entity)       →  Vector Store (direct)     │
│    memory.add_knowledge(doc)       →  Vector Store (direct)     │
│                                                                  │
│  READS — Direct retrieval, no AI:                                │
│                                                                  │
│    memory.get_conversational(thread_id)                          │
│    memory.search_knowledge(query)                                │
│    memory.expand_summary(summary_id)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   AI-ASSISTED (Agents required)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COMPLEX OPERATIONS — Require LLM intelligence:                  │
│                                                                  │
│    Entity Extraction     →  Extract people/orgs from text        │
│    Summarization         →  Compress conversations               │
│    Consolidation         →  Merge similar memories               │
│    Context Assembly      →  Build optimal context for LLM        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Operation Classification

| Operation | Deterministic | AI Agent |
|-----------|:-------------:|:--------:|
| `add_conversational()` | ✅ | - |
| `add_entity()` | ✅ | - |
| `add_knowledge()` | ✅ | - |
| `add_workflow()` | ✅ | - |
| `get_*()` / `search_*()` | ✅ | - |
| `expand_summary()` | ✅ | - |
| Entity extraction | - | ✅ |
| Summarization | - | ✅ |
| Consolidation | - | ✅ |
| Context assembly | - | ✅ |

## Why Deterministic First?

### For Retrieval (reads)

1. **Context bootstrapping is non-negotiable** — agent needs prior context
2. **Chicken-and-egg problem** — agent can't decide to look up what it doesn't know exists
3. **Predictability** — consistent, debuggable behavior

### For Storage (writes)

1. **Reliability** — don't want agent to "forget to save"
2. **Completeness** — every interaction recorded
3. **Reduced cognitive load** — model focuses on task, not bookkeeping

### When AI is Needed

1. **Semantic understanding** — entity extraction requires understanding
2. **Creative compression** — summarization needs judgment
3. **Intelligent merging** — consolidation requires semantic comparison

## Code Examples

### Deterministic Operations

```python
# No LLM needed — direct storage
await memory.add_conversational("t1", "user", "Hello")
await memory.add_knowledge("Python is a language", source="docs")

# No LLM needed — direct retrieval
messages = await memory.get_conversational("t1", limit=50)
results = await memory.search_knowledge("programming")
```

### AI-Assisted Operations

```python
# Requires LLM — entity extraction
entities = await memory.extract_entities("Dr. Chen works at MIT")
# Returns: [Entity(name="Dr. Chen", type="PERSON"), Entity(name="MIT", type="ORG")]

# Requires LLM — summarization
summary_id = await memory.summarize(thread_id="t1")

# Requires LLM — consolidation
await memory.consolidate(memory_type="entity")
```

## Configuring AI Operations

```yaml
# memharness.yaml
agents:
  entity_extractor:
    enabled: true
    mode: on_write  # Extract entities automatically on write

  summarizer:
    enabled: true
    triggers:
      - condition: "age > 7d"
      - condition: "message_count > 50"

  consolidator:
    enabled: true
    schedule: "0 3 * * *"  # Daily at 3 AM
```

## Without LLM

If you don't provide an LLM, AI operations are disabled. The package works fully with deterministic operations only.

```python
# No LLM — deterministic only
memory = MemoryHarness("sqlite:///db.sqlite")

# This works
await memory.add_conversational(...)

# This is disabled (no LLM)
# await memory.extract_entities(...)  # Raises error
```
