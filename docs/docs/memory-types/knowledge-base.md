---
sidebar_position: 2
---

# Knowledge Base Memory

Store facts, documents, and information for semantic search and retrieval-augmented generation (RAG).

## Overview

Knowledge Base memory is designed for storing factual information, documents, and reference material that agents can search semantically. It's the foundation for RAG applications and fact-based question answering.

### When to Use

- RAG (Retrieval-Augmented Generation) applications
- Storing documentation, articles, and reference material
- Fact-based question answering
- Grounding agent responses in source material
- Building searchable knowledge repositories

### Storage Strategy

**Backend**: Vector database (PostgreSQL with pgvector, SQLite with sqlite-vss)
**Why**: Knowledge base requires:
- **Semantic search** via embedding similarity
- **Efficient vector operations** (cosine similarity, HNSW indexing)
- **Metadata filtering** (by source, category, etc.)

Stored with vector embeddings for semantic search, indexed using HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search.

## API Methods

### add_knowledge

Add factual information or a document to the knowledge base.

```python
async def add_knowledge(
    content: str,
    source: str | None = None,
    metadata: dict[str, Any] | None = None
) -> str
```

**Parameters**:
- `content`: The knowledge content (fact, passage, document chunk)
- `source`: Optional source identifier (URL, document name, etc.)
- `metadata`: Optional metadata (category, tags, etc.)

**Returns**: Memory ID

**Example**:
```python
from memharness import MemoryHarness

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Add a fact
    await harness.add_knowledge(
        content="Python's Global Interpreter Lock (GIL) prevents true parallelism in CPU-bound threads.",
        source="Python Documentation",
        metadata={"category": "programming", "language": "python"}
    )

    # Add a document chunk
    await harness.add_knowledge(
        content="Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications.",
        source="https://kubernetes.io/docs",
        metadata={"category": "devops", "topic": "kubernetes"}
    )

    # Add research findings
    await harness.add_knowledge(
        content="The study found that regular exercise improves cognitive function and reduces the risk of dementia by 30%.",
        source="Journal of Neuroscience, 2025",
        metadata={"category": "health", "type": "research"}
    )
```

### search_knowledge

Search the knowledge base by semantic similarity.

```python
async def search_knowledge(
    query: str,
    k: int = 5,
    filters: dict[str, Any] | None = None
) -> list[MemoryUnit]
```

**Parameters**:
- `query`: The search query
- `k`: Number of results to return (default: 5)
- `filters`: Optional metadata filters (e.g., `{"category": "programming"}`)

**Returns**: List of MemoryUnit objects, ordered by relevance (similarity score)

**Example**:
```python
# Basic search
results = await harness.search_knowledge("Python concurrency", k=3)

for r in results:
    print(f"[{r.score:.2f}] {r.content[:100]}...")
    print(f"Source: {r.metadata.get('source')}\n")

# Filtered search
python_docs = await harness.search_knowledge(
    query="async programming",
    k=5,
    filters={"language": "python"}
)

# Multiple filters
kubernetes_docs = await harness.search_knowledge(
    query="deployment strategies",
    k=10,
    filters={
        "category": "devops",
        "topic": "kubernetes"
    }
)
```

## Schema/Metadata Structure

Each knowledge base memory unit contains:

```python
{
    "id": "uuid",
    "content": "The knowledge content",
    "memory_type": "knowledge",
    "namespace": ("knowledge",),
    "metadata": {
        "source": "Python Documentation",  # Optional
        # Custom metadata fields:
        "category": "programming",
        "language": "python",
        "tags": ["concurrency", "async"],
        "confidence": 0.95,
        "last_verified": "2026-03-23"
    },
    "embedding": [0.123, -0.456, ...],  # Vector embedding
    "created_at": "2026-03-23T10:00:00Z",
    "score": 0.87  # Similarity score (only in search results)
}
```

## Best Practices

### 1. Chunk Documents Appropriately

Break large documents into searchable chunks:

```python
def chunk_document(doc: str, chunk_size: int = 512) -> list[str]:
    """Split document into overlapping chunks."""
    chunks = []
    words = doc.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Add each chunk
document = "Long document text..."
chunks = chunk_document(document)

for i, chunk in enumerate(chunks):
    await harness.add_knowledge(
        content=chunk,
        source="long-document.pdf",
        metadata={"chunk_id": i, "total_chunks": len(chunks)}
    )
```

### 2. Use Source Tracking

Always include source information for attribution:

```python
await harness.add_knowledge(
    content="Fact from documentation",
    source="https://docs.example.com/page",
    metadata={
        "source_type": "documentation",
        "retrieved_at": datetime.utcnow().isoformat()
    }
)
```

### 3. Leverage Metadata Filters

Use metadata to organize and filter knowledge:

```python
# Add with rich metadata
await harness.add_knowledge(
    content="Docker containers provide isolated environments...",
    source="Docker Docs",
    metadata={
        "category": "devops",
        "technology": "docker",
        "difficulty": "beginner",
        "language": "en"
    }
)

# Search with filters
beginner_docker = await harness.search_knowledge(
    query="container basics",
    filters={
        "category": "devops",
        "technology": "docker",
        "difficulty": "beginner"
    }
)
```

### 4. Update vs. Replace

Knowledge can become stale. Choose whether to update or add new versions:

```python
# Option 1: Add new version (preserves history)
await harness.add_knowledge(
    content="Updated information about Python 3.14...",
    source="Python 3.14 Release Notes",
    metadata={"version": "3.14", "supersedes": old_id}
)

# Option 2: Update existing (if backend supports)
# await harness.update(old_id, new_content)
```

### 5. RAG Pattern with LangChain

Use knowledge base in a RAG pipeline:

```python
from langchain.agents import create_agent
from memharness.middleware import MemoryContextMiddleware

# Knowledge is automatically injected into context
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    middleware=[
        MemoryContextMiddleware(
            harness=harness,
            thread_id="chat-1",
            max_tokens=2000  # Includes knowledge search
        )
    ]
)

# Agent automatically searches knowledge base
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Explain Docker containers"}]
})
# Response will be grounded in stored knowledge
```

## Vector Storage Details

Knowledge base uses vector embeddings for semantic search:

```sql
CREATE TABLE memory_store (
    id UUID PRIMARY KEY,
    namespace TEXT[] NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(768) NOT NULL,  -- Dimension depends on model
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX idx_knowledge_embedding
ON memory_store USING hnsw(embedding vector_cosine_ops)
WHERE memory_type = 'knowledge';

-- GIN index for metadata filtering
CREATE INDEX idx_knowledge_metadata
ON memory_store USING gin(metadata)
WHERE memory_type = 'knowledge';
```

### Embedding Models

memharness uses HuggingFace embeddings by default:

```python
# Default: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
harness = MemoryHarness("sqlite:///memory.db")

# Custom embedding model
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
harness = MemoryHarness(
    "postgresql://localhost/memharness",
    embedding_fn=lambda text: embeddings.embed_query(text)
)
```

## Performance Considerations

### Search Speed

- **HNSW indexing**: Near-instant search even with millions of vectors
- **Metadata filtering**: Add GIN indexes on commonly filtered fields
- **Batch operations**: Add multiple knowledge items in parallel

```python
import asyncio

# Batch add for large datasets
async def add_knowledge_batch(items: list[dict]):
    tasks = [
        harness.add_knowledge(**item)
        for item in items
    ]
    return await asyncio.gather(*tasks)

await add_knowledge_batch([
    {"content": "fact 1", "source": "source1"},
    {"content": "fact 2", "source": "source2"},
    # ... thousands more
])
```

### Storage Size

- Each embedding: ~3KB (768 dimensions * 4 bytes)
- Monitor database size and consider archiving old knowledge

## Related Memory Types

- [Entity Memory](./entity) — Store structured entities mentioned in knowledge
- [File Memory](./file) — Track source documents and files
- [Conversational Memory](./conversational) — Ground conversations in knowledge

## Next Steps

- [Entity Memory](./entity) — Structured named entities
- [RAG Guide](../concepts/rag) — Building RAG applications
- [Embeddings](../concepts/embeddings) — Choosing embedding models
