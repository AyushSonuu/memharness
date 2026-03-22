---
sidebar_position: 1
---

# Memory Types Overview

memharness provides **10 memory types** organized into three categories based on cognitive alignment.

## The 10 Memory Types

```
memharness Memory Types
│
├── EPISODIC (experiences)
│   ├── 1. Conversational   → Chat history, thread-scoped
│   ├── 2. Summary          → Compressed conversations
│   └── 3. Tool Log         → Execution audit trail
│
├── SEMANTIC (facts)
│   ├── 4. Knowledge Base   → Documents, passages
│   ├── 5. Entity           → People, orgs, concepts
│   ├── 6. File             → Document references
│   └── 7. Persona          → Agent identity
│
└── PROCEDURAL (how-to)
    ├── 8. Workflow         → Reusable step patterns
    ├── 9. Toolbox          → Tool definitions
    └── 10. Skills          → Learned capabilities
```

## Storage Strategy

Different memory types use different storage strategies:

| Type | Storage | Index | Access Pattern |
|------|---------|-------|----------------|
| Conversational | SQL | B-tree | Exact match by thread_id + time ordering |
| Tool Log | SQL | B-tree | Exact match by thread_id + time ordering |
| Knowledge Base | Vector | HNSW | Semantic similarity search |
| Entity | Vector | HNSW | Semantic + exact lookup |
| Workflow | Vector | HNSW | Semantic search |
| Toolbox | Vector | HNSW | Semantic search + VFS navigation |
| Summary | Vector | HNSW | Semantic search |
| Skills | Vector | HNSW | Semantic search |
| File | Hybrid | Both | Path lookup + content search |
| Persona | Vector | HNSW | Block lookup |

### Why Two Storage Types?

| Storage | Used For | Why |
|---------|----------|-----|
| **SQL** | Conversational, Tool Log | Exact match by `thread_id`, chronological ordering |
| **Vector** | Everything else | Semantic similarity search — find by meaning |

## Quick Reference

| Memory Type | One-Liner | Example |
|-------------|-----------|---------|
| **Conversational** | Chat history per thread | `[user] "Book the first one"` |
| **Knowledge Base** | Domain knowledge & facts | arXiv papers, product docs |
| **Entity** | People, places, systems | `Dr. Chen (PERSON): ML researcher` |
| **Workflow** | "How did I do this before?" | `Query → Search → Filter → Summarize → ✅` |
| **Toolbox** | Available tools | `search_arxiv(query, k=5)` |
| **Summary** | Compressed conversations | 30 messages → 1 paragraph |
| **Tool Log** | Raw tool execution audit | `search_arxiv({query:"flash"}) → success` |
| **Skills** | Learned capabilities | `"Can analyze Python code"` |
| **File** | Document references | `/path/to/report.pdf` |
| **Persona** | Agent identity blocks | `"I am a helpful coding assistant"` |

## Detailed Documentation

See individual pages for each memory type:

- [Conversational](../memory-types/conversational)
- [Knowledge Base](../memory-types/knowledge-base)
- [Entity](../memory-types/entity)
- [Workflow](../memory-types/workflow)
- [Toolbox](../memory-types/toolbox)
- [Summary](../memory-types/summary)
- [Tool Log](../memory-types/tool-log)
- [Skills](../memory-types/skills)
- [File](../memory-types/file)
- [Persona](../memory-types/persona)
