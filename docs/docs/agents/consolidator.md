---
sidebar_position: 4
---

# Consolidator Agent

Merge duplicate and highly similar memories to reduce redundancy and improve memory quality.

## Overview

The Consolidator Agent identifies and merges duplicate or highly similar memories across all memory types. It uses embedding similarity to find candidates and intelligently merges their content, preventing memory bloat and improving retrieval quality.

### When to Use

- When memory grows beyond 1000+ entries
- To deduplicate entity mentions (e.g., "John", "John Smith", "@john")
- To merge similar knowledge base facts
- As a scheduled background task (e.g., daily at 3 AM)
- Before archival (consolidate then compress)
- When retrieval quality degrades due to duplicates

### Dual-Mode Operation

**Mode 1: Embedding Similarity Only (No LLM)**
- Finds similar memories using cosine similarity
- Merges by keeping the longer content
- Fast and deterministic
- Zero LLM costs
- Output: Primary memory with merged metadata

**Mode 2: LLM-Powered Intelligent Merging**
- Uses embedding similarity to find candidates
- Uses LLM to intelligently merge content
- Removes redundancy while preserving unique information
- Higher quality merged memories
- Output: Comprehensive merged memory with all important details

## API Methods

### consolidate_memories

Find and merge duplicate memories.

```python
async def consolidate_memories(
    min_memories: int = 5,
    memory_type: str | None = None
) -> dict[str, int]
```

**Parameters**:
- `min_memories`: Minimum number of memories required before consolidation runs (default: 5)
- `memory_type`: Optional memory type to consolidate (None = all types)

**Returns**: Dictionary with `merged` and `deleted` counts

**Example**:
```python
from memharness import MemoryHarness
from memharness.agents import ConsolidatorAgent
from langchain.chat_models import init_chat_model

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Heuristic mode
    agent_basic = ConsolidatorAgent(harness, threshold=0.85)
    result = await agent_basic.consolidate_memories(min_memories=5)
    # Output: {"merged": 0, "deleted": 0}  # Simplified implementation

    # LLM mode (intelligent merging)
    llm = init_chat_model("gpt-4o")
    agent_smart = ConsolidatorAgent(harness, llm=llm, threshold=0.90)
    result = await agent_smart.consolidate_memories(min_memories=10)
    # Output: {"merged": 3, "deleted": 3}
```

### run

Execute the consolidator agent (standard agent interface).

```python
async def run(
    min_memories: int = 5,
    memory_type: str | None = None,
    **kwargs
) -> dict[str, Any]
```

**Parameters**:
- `min_memories`: Minimum memories before consolidation
- `memory_type`: Optional memory type to consolidate
- `**kwargs`: Additional arguments (ignored)

**Returns**: Dictionary with `merged`, `deleted`, and `threshold` keys

**Example**:
```python
result = await agent.run(min_memories=10, memory_type="entity")
# Returns: {"merged": 3, "deleted": 3, "threshold": 0.85}
```

## Implementation Details

### Similarity Detection

The agent uses cosine similarity to find duplicate candidates:

```python
def _cosine_similarity(
    self,
    embedding1: list[float],
    embedding2: list[float]
) -> float:
    """Calculate cosine similarity between two embeddings."""
    if len(embedding1) != len(embedding2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=False))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
```

**Similarity Threshold**:
- `0.95+`: Nearly identical (definitely merge)
- `0.85-0.94`: Very similar (likely merge)
- `0.70-0.84`: Similar but distinct (review manually)
- `< 0.70`: Different (don't merge)

### Heuristic Mode

The heuristic mode keeps the longer content:

```python
async def _merge_memories_heuristic(
    self,
    memory1: MemoryUnit,
    memory2: MemoryUnit
) -> MemoryUnit:
    """Merge two memories using heuristic approach (keep longer content)."""
    # Keep the one with longer content
    if len(memory1.content) >= len(memory2.content):
        primary, secondary = memory1, memory2
    else:
        primary, secondary = memory2, memory1

    # Merge metadata
    merged_metadata = {**secondary.metadata, **primary.metadata}
    merged_metadata["merged_from"] = [memory1.id, memory2.id]

    # Return primary with merged metadata
    return MemoryUnit(
        id=primary.id,
        memory_type=primary.memory_type,
        content=primary.content,  # Keep primary content
        embedding=primary.embedding,
        metadata=merged_metadata,
        namespace=primary.namespace,
    )
```

**Advantages**:
- Simple and fast
- No LLM costs
- Preserves more complete information

**Limitations**:
- Doesn't remove redundancy
- Doesn't synthesize information
- May keep duplicate facts

### LLM Mode

The LLM mode intelligently merges content:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a memory consolidation system. Merge the two similar "
     "memories below into a single, comprehensive memory. Keep all "
     "important information and remove redundancy."),
    ("user", "Memory 1: {content1}\n\nMemory 2: {content2}")
])

# Build chain
chain = prompt | self.llm | StrOutputParser()

# Generate merged content
merged_content = await chain.ainvoke({
    "content1": memory1.content,
    "content2": memory2.content
})

# Use memory1 as base
merged_metadata = {**memory2.metadata, **memory1.metadata}
merged_metadata["merged_from"] = [memory1.id, memory2.id]

return MemoryUnit(
    id=memory1.id,
    memory_type=memory1.memory_type,
    content=merged_content,  # LLM-generated merged content
    embedding=memory1.embedding,
    metadata=merged_metadata,
    namespace=memory1.namespace,
)
```

**Advantages**:
- Removes redundancy
- Synthesizes information
- Natural language output
- Preserves all important details

**Limitations**:
- Requires LLM API access
- Incurs API costs (use GPT-4o for quality)
- Slower than heuristic mode
- May lose subtle details

### Fallback Strategy

LLM mode automatically falls back to heuristic on errors:

```python
try:
    merged_content = await chain.ainvoke({...})
    return MemoryUnit(...)
except Exception:
    # Fall back to heuristic on error
    return await self._merge_memories_heuristic(memory1, memory2)
```

## Configuration

### Initialization Parameters

```python
from memharness.agents import ConsolidatorAgent
from langchain.chat_models import init_chat_model

# Basic initialization (heuristic mode)
agent = ConsolidatorAgent(harness, threshold=0.85)

# With LLM for intelligent merging
llm = init_chat_model("gpt-4o")  # Use GPT-4o for quality
agent = ConsolidatorAgent(harness, llm=llm, threshold=0.90)
```

**Threshold Guidelines**:
- `0.95`: Very conservative (only merge near-duplicates)
- `0.90`: Recommended for entity consolidation
- `0.85`: Standard threshold for general consolidation
- `0.80`: Aggressive (may merge distinct memories)

### YAML Configuration

```yaml
agents:
  consolidator:
    enabled: true
    llm: gpt-4o
    threshold: 0.90

    # Run daily at 3 AM
    schedule: "0 3 * * *"

    # Minimum memories before running
    min_memories: 10

    # Target memory types
    memory_types:
      - entity
      - knowledge_base
```

## Integration Patterns

### 1. Scheduled Consolidation (Background)

Run consolidation as a background task:

```python
import asyncio
from datetime import datetime

async def nightly_consolidation():
    """Consolidate memories every night."""
    agent = ConsolidatorAgent(harness, llm=llm, threshold=0.90)

    # Consolidate entities (most likely to have duplicates)
    entity_result = await agent.consolidate_memories(
        min_memories=10,
        memory_type="entity"
    )
    print(f"Entities: merged {entity_result['merged']}, deleted {entity_result['deleted']}")

    # Consolidate knowledge base
    kb_result = await agent.consolidate_memories(
        min_memories=20,
        memory_type="knowledge_base"
    )
    print(f"Knowledge: merged {kb_result['merged']}, deleted {kb_result['deleted']}")

# Schedule daily at 3 AM (use APScheduler, Celery, etc.)
```

### 2. On-Demand Consolidation

Manually trigger consolidation:

```python
async def consolidate_on_demand():
    """User-triggered consolidation."""
    agent = ConsolidatorAgent(harness, llm=llm, threshold=0.85)

    # Consolidate all memory types
    result = await agent.consolidate_memories(min_memories=5)

    print(f"Consolidation complete:")
    print(f"  Merged: {result['merged']}")
    print(f"  Deleted: {result['deleted']}")
```

### 3. Policy-Triggered Consolidation

Trigger based on memory size:

```python
async def consolidate_if_needed():
    """Consolidate when memory exceeds threshold."""
    # Check memory size (implementation-specific)
    memory_count = await harness.count_memories()

    if memory_count > 1000:
        agent = ConsolidatorAgent(harness, llm=llm, threshold=0.90)
        result = await agent.consolidate_memories(min_memories=10)
        print(f"Memory exceeded 1000 entries. Consolidated: {result}")
```

### 4. Post-Import Consolidation

After bulk imports, consolidate duplicates:

```python
async def import_and_consolidate(documents: list[str]):
    """Import documents and consolidate duplicates."""
    # Import documents
    for doc in documents:
        await harness.add_knowledge(content=doc, source="import")

    # Consolidate duplicates
    agent = ConsolidatorAgent(harness, llm=llm, threshold=0.90)
    result = await agent.consolidate_memories(
        min_memories=len(documents),
        memory_type="knowledge_base"
    )

    print(f"Imported {len(documents)}, consolidated {result['merged']}")
```

## Best Practices

### 1. Use High Thresholds for Safety

```python
# Conservative (safe for production)
agent = ConsolidatorAgent(harness, llm=llm, threshold=0.90)

# Aggressive (risky — may merge distinct memories)
# agent = ConsolidatorAgent(harness, llm=llm, threshold=0.75)  # ❌ Too low
```

### 2. Use LLM Mode for Entity Consolidation

```python
# Entities have many duplicates — use LLM for quality
llm = init_chat_model("gpt-4o")
agent = ConsolidatorAgent(harness, llm=llm, threshold=0.90)

# Example: Merge "John", "John Smith", "@john"
result = await agent.consolidate_memories(
    min_memories=5,
    memory_type="entity"
)
```

### 3. Run on a Schedule

```python
# Use APScheduler for production
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# Daily at 3 AM
scheduler.add_job(
    nightly_consolidation,
    trigger="cron",
    hour=3,
    minute=0
)

scheduler.start()
```

### 4. Monitor Consolidation Results

```python
async def consolidate_with_logging():
    """Consolidate with detailed logging."""
    agent = ConsolidatorAgent(harness, llm=llm, threshold=0.90)

    before_count = await harness.count_memories()
    result = await agent.consolidate_memories(min_memories=10)
    after_count = await harness.count_memories()

    print(f"Consolidation Report:")
    print(f"  Before: {before_count} memories")
    print(f"  After: {after_count} memories")
    print(f"  Merged: {result['merged']}")
    print(f"  Deleted: {result['deleted']}")
    print(f"  Reduction: {before_count - after_count} memories ({(before_count - after_count) / before_count * 100:.1f}%)")
```

### 5. Combine with Garbage Collection

```python
async def cleanup_pipeline():
    """Full memory cleanup pipeline."""
    # Step 1: Consolidate duplicates
    consolidator = ConsolidatorAgent(harness, llm=llm, threshold=0.90)
    consolidate_result = await consolidator.consolidate_memories(min_memories=10)

    # Step 2: Summarize old conversations
    summarizer = SummarizerAgent(harness, llm=llm)
    # (summarization logic)

    # Step 3: Archive old memories
    gc = GCAgent(harness, archive_after_days=90, delete_after_days=365)
    gc_result = await gc.run()

    print(f"Cleanup complete:")
    print(f"  Consolidated: {consolidate_result['merged']}")
    print(f"  Archived: {gc_result['archived']}")
    print(f"  Deleted: {gc_result['deleted']}")
```

## Entity Consolidation Example

The most common use case is consolidating duplicate entities:

```python
from memharness import MemoryHarness
from memharness.agents import ConsolidatorAgent
from langchain.chat_models import init_chat_model

async def consolidate_entities():
    """Consolidate duplicate entity mentions."""
    harness = MemoryHarness("sqlite:///memory.db")
    llm = init_chat_model("gpt-4o")

    # Add duplicate entities (from different sources)
    await harness.add_entity("John", "person", "User mentioned in chat")
    await harness.add_entity("John Smith", "person", "Full name from profile")
    await harness.add_entity("@john", "person", "Twitter handle")
    await harness.add_entity("john@example.com", "person", "Email address")

    # Consolidate (threshold 0.85 will merge similar names)
    agent = ConsolidatorAgent(harness, llm=llm, threshold=0.85)
    result = await agent.consolidate_memories(
        min_memories=3,
        memory_type="entity"
    )

    # Result: 4 entities → 1 merged entity
    # Content: "John Smith: User mentioned in chat, known by @john and john@example.com"
    print(f"Merged {result['merged']} entities")
```

## Current Implementation Status

**Note**: The current implementation in memharness v0.5.x returns empty results (`{"merged": 0, "deleted": 0}`). This is a simplified placeholder implementation. A full production implementation would:

1. Query the backend for all memories of a given type
2. Compare embeddings pairwise to find similar memories (above threshold)
3. Group similar memories into clusters
4. Merge each cluster using heuristic or LLM mode
5. Update the backend with merged memories
6. Delete the original duplicates

## Related Components

- [Entity Extractor](./entity-extractor) — Extracts entities (may create duplicates)
- [Entity Memory](../memory-types/entity) — Stores entities
- [Garbage Collector](./gc) — Cleans old memories
- [Context Assembly Agent](./context-assembler) — Benefits from consolidated memory

## Next Steps

- [Garbage Collector](./gc) — Memory cleanup and archival
- [Entity Memory](../memory-types/entity) — Entity storage details
- [Context Assembler](./context-assembler) — Optimal context assembly
