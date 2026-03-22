---
sidebar_position: 1
---

# MemoryHarness API

The main entry point for memharness.

## Initialization

```python
from memharness import MemoryHarness

# SQLite (development)
memory = MemoryHarness("sqlite:///memory.db")

# PostgreSQL (production)
memory = MemoryHarness("postgresql://user:pass@localhost/db")

# In-memory (testing)
memory = MemoryHarness("memory://")

# With configuration
memory = MemoryHarness(
    backend="postgresql://...",
    config=config,
    embedding_fn=embedding_function,
    namespace_prefix=("org", "user"),
)

# From config file
memory = MemoryHarness.from_config("memharness.yaml")

# From environment
memory = MemoryHarness.from_env()
```

## Context Manager

```python
async with MemoryHarness("sqlite:///db.sqlite") as memory:
    await memory.add_conversational(...)
# Automatically disconnects
```

## Conversational Memory

```python
# Write
msg_id = await memory.add_conversational(
    thread_id: str,
    role: str,          # "user" | "assistant" | "system"
    content: str,
    metadata: dict = None,
) -> str

# Read
messages = await memory.get_conversational(
    thread_id: str,
    limit: int = 50,
) -> list[MemoryUnit]
```

## Knowledge Base

```python
# Write
kb_id = await memory.add_knowledge(
    content: str,
    source: str = None,
    metadata: dict = None,
) -> str

# Search
results = await memory.search_knowledge(
    query: str,
    k: int = 5,
    filters: dict = None,
) -> list[MemoryUnit]
```

## Entity

```python
# Write
entity_id = await memory.add_entity(
    name: str,
    entity_type: str,   # "PERSON" | "ORG" | "PLACE" | "CONCEPT" | "SYSTEM"
    description: str,
    relationships: list[dict] = None,
) -> str

# Search
entities = await memory.search_entity(
    query: str,
    entity_type: str = None,
    k: int = 5,
) -> list[MemoryUnit]
```

## Workflow

```python
# Write
workflow_id = await memory.add_workflow(
    task: str,
    steps: list[str],
    outcome: str,       # "success" | "failed"
    result: str = None,
) -> str

# Search
workflows = await memory.search_workflow(
    query: str,
    k: int = 3,
) -> list[MemoryUnit]
```

## Toolbox (VFS)

```python
# Register tool
tool_id = await memory.add_tool(
    server: str,        # e.g., "github", "slack"
    tool_name: str,
    description: str,
    parameters: dict,
) -> str

# VFS operations
tree = await memory.toolbox_tree(path: str = "/") -> str
tools = await memory.toolbox_ls(server: str) -> list[str]
results = await memory.toolbox_grep(pattern: str) -> list[dict]
schema = await memory.toolbox_cat(tool_path: str) -> dict
```

## Summary

```python
# Write
summary_id = await memory.add_summary(
    summary: str,
    source_ids: list[str],
    thread_id: str = None,
) -> str

# Expand
originals = await memory.expand_summary(
    summary_id: str,
) -> list[MemoryUnit]
```

## Tool Log

```python
# Write
log_id = await memory.add_tool_log(
    thread_id: str,
    tool_name: str,
    args: dict,
    result: str,
    status: str,        # "success" | "failed"
) -> str

# Read
logs = await memory.get_tool_log(
    thread_id: str,
    limit: int = 20,
) -> list[MemoryUnit]
```

## Skills

```python
# Write
skill_id = await memory.add_skill(
    name: str,
    description: str,
    examples: list[str] = None,
) -> str

# Search
skills = await memory.search_skills(
    query: str,
    k: int = 3,
) -> list[MemoryUnit]
```

## File

```python
# Write
file_id = await memory.add_file(
    path: str,
    content_summary: str = None,
    metadata: dict = None,
) -> str

# Search
files = await memory.search_files(
    query: str,
    k: int = 5,
) -> list[MemoryUnit]
```

## Persona

```python
# Write
persona_id = await memory.add_persona(
    block_name: str,    # e.g., "identity", "preferences", "rules"
    content: str,
) -> str

# Read
persona = await memory.get_persona(
    block_name: str = None,  # None = all blocks
) -> str
```

## Context Assembly

```python
context = await memory.assemble_context(
    query: str,
    thread_id: str,
    max_tokens: int = 4000,
    include_types: list[str] = None,
) -> str
```

## Memory Tools

```python
# Get tools for agent self-exploration
tools = memory.get_memory_tools() -> list[dict]
```

## Generic Operations

```python
# Add (with auto-routing if type not specified)
memory_id = await memory.add(
    content: str,
    memory_type: str = None,
    namespace: tuple = None,
    metadata: dict = None,
) -> str

# Search across types
results = await memory.search(
    query: str,
    memory_type: str = None,
    k: int = 10,
) -> list[MemoryUnit]

# Get by ID
memory = await memory.get(memory_id: str) -> MemoryUnit | None

# Update
success = await memory.update(
    memory_id: str,
    content: str = None,
    metadata: dict = None,
) -> bool

# Delete
success = await memory.delete(memory_id: str) -> bool
```

## Lifecycle

```python
await memory.connect() -> None
await memory.disconnect() -> None
```
