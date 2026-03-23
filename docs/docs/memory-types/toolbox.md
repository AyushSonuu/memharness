---
sidebar_position: 1
---

# Toolbox Memory

Toolbox memory stores tool definitions and schemas in a virtual filesystem (VFS) interface, organizing tools by server/namespace for dynamic tool discovery and selection.

## Overview

Toolbox memory provides a structured way to catalog and organize all available tools in your agent system. Each tool is stored with its full definition, including parameters, description, and JSON schema, organized in a hierarchical namespace structure that mirrors a filesystem.

This memory type excels at:
- **Dynamic tool discovery**: Find relevant tools based on semantic search
- **Tool organization**: Hierarchical server/tool structure (e.g., `github/create_issue`)
- **Schema management**: Store complete parameter schemas for runtime validation
- **VFS interface**: Browse tools using familiar commands (ls, cat, grep, tree)

The VFS interface makes it intuitive to explore available tools, especially when working with dozens or hundreds of tool integrations across multiple services.

## When to Use

Use toolbox memory to:
- **Catalog MCP servers**: Register all tools from Model Context Protocol servers
- **Enable tool discovery**: Let agents find relevant tools for new tasks
- **Maintain tool inventory**: Track what capabilities are available
- **Organize by service**: Group tools by their source service (GitHub, Slack, etc.)
- **Support dynamic selection**: Allow agents to choose tools at runtime

Toolbox memory is essential for agents with large tool sets where manual tool selection is impractical.

## Storage Strategy

- **Backend**: VECTOR (semantic search with HNSW indexing)
- **Default k**: 5 results (focused tool retrieval)
- **Embeddings**: Yes (enables semantic tool discovery)
- **Ordered**: No (accessed by relevance and VFS navigation)
- **VFS Cache**: In-memory cache for fast filesystem operations

## Schema

Each toolbox memory includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `server` | string | Yes | Server/namespace (e.g., "github", "slack") |
| `tool_name` | string | Yes | Name of the tool within the server |
| `description` | string | Yes | What the tool does |
| `parameters` | object | Yes | JSON Schema for tool parameters |

Additional fields can be stored in metadata (tags, examples, etc.).

## API Methods

### Adding Tools

```python
async def add_tool(
    server: str,
    tool_name: str,
    description: str,
    parameters: dict[str, Any],
) -> str:
    """
    Add a tool definition to the toolbox.

    Args:
        server: The server/namespace (e.g., "github", "slack")
        tool_name: Name of the tool
        description: Description of what the tool does
        parameters: JSON Schema of the tool's parameters

    Returns:
        ID of the created memory unit
    """
```

### VFS Navigation

```python
async def toolbox_tree(path: str = "/") -> str:
    """
    Get a tree view of the toolbox virtual filesystem.

    Args:
        path: The path to start from (default "/" for root)

    Returns:
        A tree-formatted string showing the toolbox structure
    """

async def toolbox_ls(server: str) -> list[str]:
    """
    List all tools in a server/namespace.

    Args:
        server: The server name to list tools from

    Returns:
        List of tool names in the server
    """

async def toolbox_cat(tool_path: str) -> dict[str, Any]:
    """
    Get full details of a tool.

    Args:
        tool_path: Path to the tool in format "server/tool_name"

    Returns:
        Dict with full tool information including parameters

    Raises:
        ValueError: If tool_path format is invalid
        KeyError: If tool is not found
    """

async def toolbox_grep(pattern: str) -> list[dict[str, Any]]:
    """
    Search for tools matching a pattern.

    Args:
        pattern: Regex pattern to match against tool names and descriptions

    Returns:
        List of matching tool info dicts
    """
```

## Examples

### Registering MCP Server Tools

```python
from memharness import MemoryHarness

harness = MemoryHarness(backend="sqlite:///memory.db")

# Register GitHub tools
await harness.add_tool(
    server="github",
    tool_name="create_issue",
    description="Create a new GitHub issue",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "body": {"type": "string"},
            "labels": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title"]
    }
)

await harness.add_tool(
    server="github",
    tool_name="create_pr",
    description="Create a pull request",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "head": {"type": "string"},
            "base": {"type": "string"},
            "body": {"type": "string"}
        },
        "required": ["title", "head", "base"]
    }
)
```

### Browsing the Toolbox

```python
# Get a tree view of all tools
tree = await harness.toolbox_tree("/")
print(tree)
# /
# ├── github/
# │   ├── create_issue
# │   ├── create_pr
# │   └── list_repos
# └── slack/
#     ├── send_message
#     └── create_channel

# List tools in a specific server
github_tools = await harness.toolbox_ls("github")
print(github_tools)  # ["create_issue", "create_pr", "list_repos"]

# Get full details of a tool
tool_info = await harness.toolbox_cat("github/create_issue")
print(f"Description: {tool_info['description']}")
print(f"Parameters: {tool_info['parameters']}")
```

### Searching for Relevant Tools

```python
# Search for tools by pattern
create_tools = await harness.toolbox_grep("create.*")
for tool in create_tools:
    print(f"{tool['server']}/{tool['name']}: {tool['description']}")

# Example output:
# github/create_issue: Create a new GitHub issue
# github/create_pr: Create a pull request
# slack/create_channel: Create a new Slack channel
```

### Agent-Driven Tool Selection

```python
async def select_tools_for_task(task_description: str, harness: MemoryHarness):
    """Agent selects relevant tools based on task."""

    # Use grep to find potentially relevant tools
    # In practice, you'd use semantic search with embeddings
    tools = await harness.toolbox_grep(".*")

    relevant_tools = []
    for tool_info in tools:
        tool_path = f"{tool_info['server']}/{tool_info['name']}"
        details = await harness.toolbox_cat(tool_path)
        relevant_tools.append(details)

    return relevant_tools

# Usage
task = "Create a GitHub issue to track this bug"
tools = await select_tools_for_task(task, harness)
```

## Best Practices

1. **Use consistent server names**: Standardize namespace conventions (e.g., always use "github" not "GitHub" or "gh")

2. **Store complete schemas**: Include all parameter details, types, and requirements - this enables runtime validation

3. **Add descriptive tool names**: Use clear, action-oriented names (e.g., "create_issue" not "issue1")

4. **Include examples in metadata**: Store example calls in metadata for reference during tool selection

5. **Register tools at startup**: Load all available tools when initializing your agent system

6. **Cache the VFS structure**: The in-memory cache makes repeated navigation fast - leverage tree/ls operations freely

## Integration with Other Memory Types

Toolbox memory works closely with other memory types:

- **Tool Log**: After executing a tool from the toolbox, log the execution in tool log memory
- **Skills**: Tools become skills when the agent learns effective usage patterns
- **Workflow**: Complex workflows often involve sequences of tool executions
- **Conversational**: Tool selection can be influenced by conversation context

## Performance Notes

- **Vector search enabled**: Toolbox uses embeddings for semantic tool discovery beyond pattern matching
- **VFS cache**: Frequently accessed toolbox operations (ls, tree) are cached in memory for speed
- **HNSW indexing**: Efficient approximate nearest neighbor search for large tool catalogs
- **Moderate default k**: Returns 5 tools by default - adjust based on your selection strategy
- **Namespace isolation**: Server-based namespaces keep tool organization clean and queries focused
