---
sidebar_position: 3
---

# Entity Memory

Store named entities and their relationships for structured fact tracking.

## Overview

Entity memory stores structured information about people, organizations, places, concepts, and systems. It enables agents to build a knowledge graph of entities and their relationships, supporting entity-centric queries and relationship traversal.

### When to Use

- Tracking people, organizations, and systems mentioned in conversations
- Building knowledge graphs
- Customer relationship management (CRM) for agents
- Structured fact storage with relationships
- Entity extraction from unstructured text

### Storage Strategy

**Backend**: Vector database (PostgreSQL with pgvector)
**Why**: Entity memory requires:
- **Semantic search** to find entities by description
- **Exact lookup** by entity name
- **Metadata filtering** by entity type
- **Relationship storage** in metadata

Entities are embedded for semantic search while metadata enables exact matches and relationship queries.

## API Methods

### add_entity

Add an entity to memory.

```python
async def add_entity(
    name: str,
    entity_type: str,
    description: str,
    relationships: list[dict[str, str]] | None = None
) -> str
```

**Parameters**:
- `name`: Entity name (e.g., "John Smith", "OpenAI", "San Francisco")
- `entity_type`: Type (`"person"`, `"organization"`, `"location"`, `"concept"`, `"system"`)
- `description`: Description of the entity
- `relationships`: Optional list of relationships as dicts with `"target"` and `"type"`

**Returns**: Memory ID

**Example**:
```python
from memharness import MemoryHarness

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Add a person
    await harness.add_entity(
        name="Anthropic",
        entity_type="organization",
        description="AI safety company that created Claude",
        relationships=[
            {"target": "Claude", "type": "created"},
            {"target": "San Francisco", "type": "headquartered_in"}
        ]
    )

    # Add a concept
    await harness.add_entity(
        name="Python GIL",
        entity_type="concept",
        description="Global Interpreter Lock in Python that prevents true parallelism",
        relationships=[
            {"target": "Python", "type": "part_of"},
            {"target": "Threading", "type": "affects"}
        ]
    )

    # Add a system
    await harness.add_entity(
        name="PostgreSQL",
        entity_type="system",
        description="Open-source relational database system",
        relationships=[
            {"target": "SQL", "type": "uses_language"}
        ]
    )
```

### search_entity

Search for entities by semantic similarity or exact match.

```python
async def search_entity(
    query: str,
    entity_type: str | None = None,
    k: int = 5
) -> list[MemoryUnit]
```

**Parameters**:
- `query`: Search query (entity name or description)
- `entity_type`: Optional filter by entity type
- `k`: Number of results to return

**Returns**: List of MemoryUnit objects ordered by relevance

**Example**:
```python
# Semantic search
people = await harness.search_entity("AI researcher", entity_type="person")

# All organizations
orgs = await harness.search_entity("", entity_type="organization", k=10)

# By description
databases = await harness.search_entity("database management system")
```

## Schema/Metadata Structure

```python
{
    "id": "uuid",
    "content": "Anthropic: AI safety company that created Claude",
    "memory_type": "entity",
    "namespace": ("entity", "organization"),
    "metadata": {
        "name": "Anthropic",
        "entity_type": "organization",
        "relationships": [
            {"target": "Claude", "type": "created"},
            {"target": "San Francisco", "type": "headquartered_in"}
        ],
        # Optional:
        "aliases": ["Anthropic AI"],
        "properties": {
            "founded": "2021",
            "employees": "~150"
        }
    },
    "embedding": [0.123, ...],
    "created_at": "2026-03-23T10:00:00Z"
}
```

## Best Practices

### 1. Entity Extraction Middleware

Automatically extract entities from conversations:

```python
from memharness.middleware import EntityExtractionMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    middleware=[
        EntityExtractionMiddleware(
            harness=harness,
            model="anthropic:claude-haiku-4"  # Fast model for extraction
        )
    ]
)

# Entities are automatically extracted and stored
```

### 2. Relationship Management

Build a knowledge graph by tracking relationships:

```python
# Add entity with relationships
await harness.add_entity(
    name="Alice",
    entity_type="person",
    description="Software engineer specializing in Python",
    relationships=[
        {"target": "Python", "type": "specializes_in"},
        {"target": "TechCorp", "type": "works_at"},
        {"target": "Bob", "type": "colleague_of"}
    ]
)

# Query by relationship (search for it)
python_experts = await harness.search_entity("specializes in Python")
```

### 3. Entity Deduplication

Merge similar entities:

```python
# Search for potential duplicates
similar = await harness.search_entity("John Smith", entity_type="person")

if len(similar) > 1:
    # Merge logic: consolidate relationships, update description
    await consolidate_entities(similar)
```

### 4. Entity Types Taxonomy

Use consistent entity types:

```python
ENTITY_TYPES = {
    "person": ["individual", "user", "customer"],
    "organization": ["company", "institution", "team"],
    "location": ["place", "city", "country", "address"],
    "concept": ["idea", "theory", "methodology"],
    "product": ["service", "tool", "platform"],
    "system": ["software", "infrastructure", "technology"]
}
```

### 5. Rich Entity Properties

Store additional properties in metadata:

```python
await harness.add_entity(
    name="Claude",
    entity_type="product",
    description="AI assistant developed by Anthropic",
    relationships=[...],
    metadata={
        "properties": {
            "versions": ["claude-3-opus", "claude-sonnet-4-6"],
            "release_date": "2023-03-14",
            "capabilities": ["text", "vision", "code"]
        }
    }
)
```

## Automatic Entity Extraction

Using the extraction middleware:

```python
from memharness.middleware import EntityExtractionMiddleware

# Configure extraction
extractor = EntityExtractionMiddleware(
    harness=harness,
    model="anthropic:claude-haiku-4",
    entity_types=["person", "organization", "location", "concept"]
)

# Entities extracted from:
# - User messages
# - AI responses
# - Knowledge base additions
```

## Related Memory Types

- [Knowledge Base](./knowledge-base) — Unstructured facts that mention entities
- [Conversational](./conversational) — Conversations where entities are mentioned
- [Workflow](./workflow) — Workflows involving specific entities

## Next Steps

- [Entity Extraction Guide](../concepts/memory-lifecycle) — Automated extraction
- [Knowledge Graphs](../concepts/memory-types) — Building relationship graphs
- [Middleware Overview](../middleware/overview) — Entity extraction middleware
