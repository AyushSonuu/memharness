---
sidebar_position: 1
---

# File Memory

File memory stores references to files with metadata and content summaries, enabling semantic search across codebases, document collections, and file systems.

## Overview

File memory provides a lightweight way to track and retrieve files without storing complete file contents in the memory system. Each file entry includes the path, optional content summary, and metadata (size, type, language, etc.), with semantic embeddings enabling content-based search.

This memory type excels at:
- **Codebase navigation**: Find relevant files based on functionality descriptions
- **Document tracking**: Maintain searchable references to document collections
- **Content-based retrieval**: Locate files by what they contain, not just filenames
- **Workspace awareness**: Help agents understand project structure
- **Change tracking**: Monitor which files are relevant to current work

File memory bridges the gap between filesystem operations and semantic understanding, making file discovery intuitive and content-aware.

## When to Use

Use file memory to:
- **Index codebases**: Create searchable catalogs of source files
- **Track documentation**: Maintain references to markdown, PDF, and text files
- **Support code navigation**: Help agents find relevant modules and functions
- **Build project context**: Give agents understanding of workspace structure
- **Enable semantic file search**: Find files by description rather than path patterns
- **Cache file summaries**: Store high-level understanding of file contents

File memory is essential for agents working with large codebases or document collections where manual file navigation is impractical.

## Storage Strategy

- **Backend**: VECTOR (semantic search with HNSW indexing)
- **Default k**: 5 results (balanced file retrieval)
- **Embeddings**: Yes (enables content-based semantic search)
- **Ordered**: No (accessed by relevance to search query)
- **Content storage**: Summaries only, not full file contents

## Schema

Each file memory includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Full path to the file |
| `file_path` | string | Yes | Full path (alias for consistency) |
| `content_summary` | string | No | Summary of file contents |
| `file_type` | string | No | File type/extension (e.g., "py", "md") |
| `file_size` | integer | No | File size in bytes |
| `language` | string | No | Programming language if applicable |
| `chunk_index` | integer | No | Chunk number for large files |
| `total_chunks` | integer | No | Total chunks for large files |
| `last_modified` | datetime | No | Last modification timestamp |

Additional fields are stored in the `metadata` dictionary.

## API Methods

### Adding File References

```python
async def add_file(
    path: str,
    content_summary: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Add a file reference to memory.

    Args:
        path: Path to the file
        content_summary: Optional summary of the file contents
        metadata: Optional additional metadata (size, type, etc.)

    Returns:
        The ID of the created file memory
    """
```

### Searching Files

```python
async def search_files(
    query: str,
    k: int = 5,
) -> list[MemoryUnit]:
    """
    Search for files by content or path.

    Args:
        query: The search query
        k: Number of results to return

    Returns:
        List of matching file MemoryUnit objects
    """
```

## Examples

### Indexing a Codebase

```python
from memharness import MemoryHarness
import os
from pathlib import Path

harness = MemoryHarness(backend="sqlite:///memory.db")

async def index_python_files(root_dir: str):
    """Index all Python files in a directory."""

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.py'):
                continue

            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, root_dir)

            # Read file to generate summary (simplified)
            with open(filepath, 'r') as f:
                content = f.read()

            # Generate simple summary (in practice, use LLM or AST analysis)
            lines = content.split('\n')
            docstring = None
            for line in lines:
                if '"""' in line or "'''" in line:
                    docstring = line.strip()
                    break

            summary = docstring or f"Python module: {file}"

            # Get file stats
            stats = os.stat(filepath)

            await harness.add_file(
                path=relative_path,
                content_summary=summary,
                metadata={
                    "size": stats.st_size,
                    "language": "python",
                    "last_modified": stats.st_mtime
                }
            )

    print(f"Indexed Python files in {root_dir}")

# Usage
await index_python_files("./src")
```

### Semantic File Search

```python
# Find files related to authentication
auth_files = await harness.search_files("authentication and user login", k=5)

print("Files related to authentication:")
for file in auth_files:
    path = file.metadata.get("path", "unknown")
    summary = file.metadata.get("content_summary", "No summary")
    print(f"  {path}")
    print(f"    {summary}")
    print()

# Find database-related files
db_files = await harness.search_files("database queries and SQL", k=5)

# Find configuration files
config_files = await harness.search_files("configuration and settings", k=3)
```

### Building Project Context

```python
async def build_context_for_task(task_description: str, harness: MemoryHarness):
    """Build file context for a specific task."""

    # Search for relevant files
    relevant_files = await harness.search_files(task_description, k=10)

    context = ["Relevant files for this task:\n"]

    for file in relevant_files:
        path = file.metadata.get("path", "unknown")
        summary = file.metadata.get("content_summary", "")
        size = file.metadata.get("size", 0)

        context.append(f"- {path} ({size} bytes)")
        if summary:
            context.append(f"  Purpose: {summary}")

    return "\n".join(context)

# Usage
task = "Fix the authentication bug in the login flow"
context = await build_context_for_task(task, harness)
print(context)
```

### Tracking Document Collections

```python
# Index markdown documentation
docs_dir = "./docs"

for doc_file in Path(docs_dir).rglob("*.md"):
    with open(doc_file, 'r') as f:
        content = f.read()

    # Extract title (first heading)
    title = "Untitled"
    for line in content.split('\n'):
        if line.startswith('# '):
            title = line[2:].strip()
            break

    # Generate summary (first paragraph)
    paragraphs = content.split('\n\n')
    summary = paragraphs[1] if len(paragraphs) > 1 else title

    await harness.add_file(
        path=str(doc_file.relative_to(docs_dir)),
        content_summary=f"{title}: {summary[:200]}",
        metadata={
            "type": "markdown",
            "category": "documentation",
            "title": title
        }
    )

print("Indexed documentation files")

# Later, search documentation
docs = await harness.search_files("API authentication guide", k=3)
```

## Best Practices

1. **Include meaningful summaries**: Content summaries drive search quality - invest in generating good ones

2. **Update on file changes**: Re-index files when they're modified to keep summaries current

3. **Use relative paths**: Store paths relative to project root for portability

4. **Chunk large files**: For very large files, create multiple entries with chunk indices

5. **Add rich metadata**: Include language, framework, purpose, and other contextual information

6. **Index incrementally**: Don't re-index unchanged files - track modification times

## Integration with Other Memory Types

File memory integrates with other memory types:

- **Knowledge**: File summaries can link to detailed knowledge entries about file contents
- **Workflow**: Workflows may reference specific files that were modified or reviewed
- **Tool Log**: File operations (read, write, edit) can be cross-referenced with file entries
- **Conversational**: Users may ask about specific files mentioned in conversation
- **Entity**: Files may be associated with entities (authors, projects, modules)

## Performance Notes

- **Semantic search**: File retrieval uses embeddings to match queries to content summaries
- **HNSW indexing**: Efficient nearest-neighbor search across large file collections
- **Lightweight storage**: Only summaries stored, not full file contents
- **Moderate default k**: Returns 5 files by default - enough context without overwhelming
- **Incremental indexing**: Only re-index modified files for efficiency
- **Path-based namespacing**: Optional namespace organization by directory or project
