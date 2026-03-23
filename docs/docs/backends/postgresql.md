---
sidebar_position: 2
---

# PostgreSQL Backend

The PostgreSQL backend with [pgvector](https://github.com/pgvector/pgvector) is the recommended production setup for memharness. It provides efficient vector similarity search via HNSW indexes.

## When to Use

| Scenario | Recommendation |
|----------|---------------|
| Production systems | ✅ Recommended |
| Multi-user applications | ✅ Best choice |
| Large-scale vector search | ✅ HNSW indexing |
| CI/CD with real DB | ✅ Use Docker |
| Quick local dev | Use SQLite instead |

## Installation

```bash
pip install memharness[postgres]
# Installs: asyncpg, pgvector
```

## Docker Setup (Recommended)

The easiest way to run PostgreSQL + pgvector locally:

```bash
# Start postgres with pgvector
docker compose up -d

# Or manually
docker run -d \
  --name memharness-postgres \
  -e POSTGRES_DB=memharness \
  -e POSTGRES_USER=memharness \
  -e POSTGRES_PASSWORD=memharness \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

A `docker-compose.yml` is included in the repository:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: memharness
      POSTGRES_USER: memharness
      POSTGRES_PASSWORD: memharness
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

## Basic Usage

```python
from memharness import MemoryHarness

harness = MemoryHarness("postgresql://memharness:memharness@localhost/memharness")
await harness.connect()

# Same API as SQLite
await harness.add_knowledge(
    "pgvector enables vector similarity search in PostgreSQL",
    source="docs"
)
results = await harness.search_knowledge("vector database", k=5)

await harness.disconnect()
```

## Schema Overview

PostgreSQL uses separate tables per memory type:

| Table | Type | Storage |
|-------|------|---------|
| `conversational_memory` | Conversational | SQL + timeline |
| `knowledge_base_memory` | Knowledge Base | Vector (HNSW) |
| `entity_memory` | Entity | Vector (HNSW) |
| `workflow_memory` | Workflow | Vector (HNSW) |
| `toolbox_memory` | Toolbox | Vector (HNSW) |
| `summary_memory` | Summary | Vector (HNSW) |
| `tool_log_memory` | Tool Log | SQL + timeline |
| `persona_memory` | Persona | Vector (HNSW) |
| `file_memory` | File | Vector (HNSW) |

The full schema is at `src/memharness/sql/postgres/schema.sql`.

## Vector Search

Uses pgvector's HNSW index for efficient approximate nearest-neighbor search:

```sql
CREATE INDEX ON knowledge_base_memory 
USING hnsw (embedding vector_cosine_ops);
```

This scales to millions of vectors while maintaining fast query times.

## Connection Pooling

The PostgreSQL backend uses asyncpg connection pooling automatically:

```python
harness = MemoryHarness(
    "postgresql://user:pass@host/db",
    # Connection pool size configurable via config
)
```

## Production Checklist

- Use a dedicated database user with limited permissions
- Enable connection pooling (PgBouncer or asyncpg pool)
- Set up regular backups
- Monitor pgvector index sizes
- Use environment variables for credentials, not hardcoded strings

```python
import os

DATABASE_URL = os.environ["DATABASE_URL"]
harness = MemoryHarness(DATABASE_URL)
```
