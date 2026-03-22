---
sidebar_position: 3
---

# Memory Lifecycle

The memory lifecycle describes how data flows through memharness — from ingestion to eventual expiration.

## The Lifecycle Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                      MEMORY LIFECYCLE                           │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  INGEST  │───▶│  ENRICH  │───▶│  STORE   │───▶│ ORGANIZE │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲                                               │        │
│       │                                               ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ SERIALIZE│◀───│   LLM    │◀───│ ASSEMBLE │◀───│ RETRIEVE │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Stages

### 1. Ingest

Raw data enters the system from various sources:
- User messages
- Tool execution results
- Document uploads
- Agent responses

```python
# Example: ingesting a user message
await memory.add_conversational("t1", "user", "Deploy the app to K8s")
```

### 2. Enrich

Data is enriched with:
- **Embeddings** — Vector representation for semantic search
- **Metadata** — Timestamps, types, relationships
- **Extracted entities** — People, places, systems mentioned

```python
# Automatic enrichment on write
# - Embedding created
# - Timestamp added
# - Optionally: entity extraction
```

### 3. Store

Enriched data is persisted based on type:
- **SQL tables** — Conversational, Tool Log
- **Vector stores** — Knowledge Base, Entity, Workflow, etc.
- **Hot/Warm/Cold tiers** — Based on access frequency

### 4. Organize

Data is indexed and relationships are mapped:
- **B-tree indexes** — For exact lookups (thread_id)
- **HNSW indexes** — For vector similarity
- **Relationship mapping** — Entity connections

### 5. Retrieve

Data is recalled based on context:
- **Exact match** — Get conversation by thread_id
- **Semantic search** — Find similar content
- **Hybrid search** — Combine keyword and semantic

### 6. Assemble

Context is assembled for LLM consumption:
- **Prioritization** — Most relevant first
- **Token budgeting** — Fit within limits
- **Formatting** — Structured markdown

```python
context = await memory.assemble_context(
    query="deploy app",
    thread_id="t1",
    max_tokens=4000
)
```

### 7. LLM Processing

The LLM processes the assembled context and generates a response.

### 8. Serialize

LLM output is serialized back into memory:
- Assistant responses → Conversational memory
- Extracted facts → Entity memory
- Learned patterns → Workflow memory

**The cycle continues!**

## Lifecycle Operations

### Summarization

Old conversations are compressed:

```
┌─────────────────────────────────────┐
│  30 messages (old)                  │
│  ┌─────┐ ┌─────┐ ┌─────┐           │
│  │msg 1│ │msg 2│ │...  │ │msg 30│  │
│  └─────┘ └─────┘ └─────┘           │
└───────────────┬─────────────────────┘
                │ summarize()
                ▼
┌─────────────────────────────────────┐
│  1 summary + archived originals     │
│  ┌────────────────────────────────┐ │
│  │ "User discussed K8s deploy..." │ │
│  │ source_ids: [msg1...msg30]     │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Key**: Originals are **archived, not deleted**. You can always `expand_summary()`.

### Consolidation

Similar memories are merged:

```
┌─────────────────────────────────────┐
│  Duplicate entities                  │
│  • "Dr. Chen" (created day 1)       │
│  • "Dr. Sarah Chen" (created day 5) │
│  • "Sarah Chen" (created day 10)    │
└───────────────┬─────────────────────┘
                │ consolidate()
                ▼
┌─────────────────────────────────────┐
│  Single merged entity                │
│  • "Dr. Sarah Chen"                 │
│    - aliases: ["Dr. Chen", "Sarah"] │
│    - merged from 3 sources          │
└─────────────────────────────────────┘
```

### Garbage Collection

Expired and orphaned memories are cleaned:

```yaml
gc:
  schedule: "0 4 * * 0"  # Weekly Sunday 4 AM
  archive_after: 90d     # Move to cold storage
  delete_after: 365d     # Delete from cold storage
```

## Configuring Lifecycle

```yaml
# memharness.yaml
summarization:
  triggers:
    - condition: "age > 7d"
      memory_type: conversational
    - condition: "message_count > 50"
      memory_type: conversational
  keep_originals: true
  originals_ttl: 365d

consolidation:
  schedule: "0 3 * * *"  # Daily 3 AM
  similarity_threshold: 0.9

gc:
  schedule: "0 4 * * 0"  # Weekly
  archive_after: 90d
  delete_after: 365d
  protect_referenced: true  # Don't delete if referenced by summary
```
