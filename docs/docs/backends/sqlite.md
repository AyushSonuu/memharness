---
sidebar_position: 1
---

# SQLite Backend

The SQLite backend is the recommended choice for local development, testing, and lightweight production use. It requires no external services.

## When to Use

| Scenario | Recommendation |
|----------|---------------|
| Local development | ✅ Perfect |
| Testing and CI | ✅ Use in-memory SQLite |
| Single-user apps | ✅ Good fit |
| Multi-user production | ⚠️ Use PostgreSQL instead |
| Vector similarity search | ⚠️ Limited (cosine on floats) |

## Installation

```bash
pip install memharness
# SQLite support is built-in — no extra dependencies needed
```

## Basic Usage

```python
from memharness import MemoryHarness

# File-based (persists across runs)
harness = MemoryHarness("sqlite:///my_memory.db")
await harness.connect()

# Use it
await harness.add_conversational("thread1", "user", "Hello!")
results = await harness.search_knowledge("Python async")

await harness.disconnect()
```

## Connection String Format

```
sqlite:///relative/path/to/db.sqlite
sqlite:////absolute/path/to/db.sqlite
memory://  ← in-memory (for testing)
```

## Schema

The SQLite backend uses a single `memories` table:

```sql
CREATE TABLE IF NOT EXISTS memories (
    id        TEXT PRIMARY KEY,
    content   TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    namespace TEXT DEFAULT '',
    thread_id TEXT,
    parent_id TEXT,
    embedding BLOB,        -- JSON-serialized float list
    metadata  TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_type    ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_thread  ON memories(thread_id);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
```

## Vector Search

SQLite doesn't have native vector search. The backend implements cosine similarity in Python:

```python
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

This works well for small datasets. For large-scale vector search, use PostgreSQL + pgvector.

## Context Manager

```python
async with MemoryHarness("sqlite:///memory.db") as harness:
    await harness.add_knowledge("Python supports async/await", source="docs")
    results = await harness.search_knowledge("async programming")
```

## Performance Tips

- For testing: use `memory://` (in-memory, no disk I/O)
- For production: use a fast SSD path
- Keep embeddings small (384-dim works well)
- Index on frequently queried fields is handled automatically

## Limitations

- No concurrent writes from multiple processes (SQLite WAL mode helps)
- Vector search is O(n) — slow for thousands of vectors
- No full-text search (use PostgreSQL for that)

## Migration to PostgreSQL

When you're ready to scale:

```python
# Development
harness = MemoryHarness("sqlite:///memory.db")

# Production — same API, different backend
harness = MemoryHarness("postgresql://user:pass@localhost/memharness")
```
