---
sidebar_position: 1
---

# Tool Log Memory

Tool log memory provides an ordered audit trail of tool executions with status tracking, enabling debugging, analytics, and learning from tool usage patterns.

## Overview

Tool log memory records every tool invocation with complete details about inputs, outputs, execution status, and timing. Unlike the toolbox (which stores tool definitions), tool log captures the actual execution history, creating a chronological record of what tools were used, when, and with what results.

This memory type excels at:
- **Execution tracking**: Complete audit trail of all tool invocations
- **Error debugging**: Detailed records of failures with error messages
- **Performance monitoring**: Track execution duration and success rates
- **Usage analytics**: Understand which tools are used most frequently
- **Learning from history**: Discover successful tool usage patterns

The ordered nature of tool logs makes them ideal for temporal analysis, debugging sequences of actions, and understanding the chronological flow of agent behavior.

## When to Use

Use tool log memory to:
- **Debug tool failures**: Review what went wrong with specific tool executions
- **Track execution history**: Maintain a complete audit trail for compliance or analysis
- **Monitor performance**: Identify slow or unreliable tools
- **Learn patterns**: Discover which tool sequences lead to success
- **Provide user transparency**: Show users what actions the agent took
- **Analyze tool usage**: Generate statistics on tool popularity and reliability

Tool log is essential for production agents where observability and debugging are critical.

## Storage Strategy

- **Backend**: SQL (ordered, sequential access)
- **Default k**: 10 results (logs are numerous, so moderate retrieval)
- **Embeddings**: No (logs are accessed chronologically or by exact match)
- **Ordered**: Yes (temporal ordering is fundamental to logs)
- **Thread isolation**: Logs can be scoped to conversation threads

## Schema

Each tool log entry includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_name` | string | Yes | Name of the executed tool |
| `status` | string | Yes | Execution status: "success", "error", "timeout" |
| `input` | object | No | Input parameters passed to the tool |
| `output` | object | No | Output result from the tool |
| `duration_ms` | integer | No | Execution duration in milliseconds |
| `error` | string | No | Error message if status is "error" |
| `thread_id` | string | No | Conversation thread if applicable |

Additional fields are stored in the `metadata` dictionary.

## API Methods

### Logging Tool Executions

```python
async def add_tool_log(
    thread_id: str,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    status: str,
) -> str:
    """
    Log a tool execution.

    Args:
        thread_id: The conversation thread ID
        tool_name: Name of the executed tool
        args: Arguments passed to the tool
        result: Result or output from the tool
        status: Execution status ("success", "error", "timeout")

    Returns:
        The ID of the created log entry
    """

async def log_tool_execution(
    tool_name: str,
    input_params: dict[str, Any],
    output_result: dict[str, Any] | None = None,
    success: bool = True,
    duration_ms: int | None = None,
    error: str | None = None,
) -> str:
    """
    Log a tool execution with detailed parameters.

    Args:
        tool_name: Name of the tool executed
        input_params: Input parameters passed to the tool
        output_result: Output result from the tool
        success: Whether the execution was successful
        duration_ms: Duration in milliseconds
        error: Error message if execution failed

    Returns:
        The ID of the created log entry
    """
```

### Retrieving Logs

```python
async def get_tool_log(
    thread_id: str,
    limit: int = 20,
) -> list[MemoryUnit]:
    """
    Retrieve tool execution log for a thread.

    Args:
        thread_id: The conversation thread ID
        limit: Maximum number of log entries to retrieve

    Returns:
        List of tool log MemoryUnit objects, ordered from oldest to newest
    """

async def search_tool_logs(
    query: str,
    k: int = 10,
) -> list[MemoryUnit]:
    """
    Search tool execution logs.

    Args:
        query: Search query (tool name or partial match)
        k: Number of results to return

    Returns:
        List of matching tool log memory units
    """
```

## Examples

### Basic Tool Execution Logging

```python
from memharness import MemoryHarness
import time

harness = MemoryHarness(backend="sqlite:///memory.db")

# Execute a tool and log it
start_time = time.time()

try:
    result = await github_create_issue(
        title="Bug fix",
        body="Fixed the authentication issue"
    )

    duration_ms = int((time.time() - start_time) * 1000)

    await harness.log_tool_execution(
        tool_name="github.create_issue",
        input_params={
            "title": "Bug fix",
            "body": "Fixed the authentication issue"
        },
        output_result={"issue_number": result.number},
        success=True,
        duration_ms=duration_ms
    )

except Exception as e:
    duration_ms = int((time.time() - start_time) * 1000)

    await harness.log_tool_execution(
        tool_name="github.create_issue",
        input_params={
            "title": "Bug fix",
            "body": "Fixed the authentication issue"
        },
        success=False,
        duration_ms=duration_ms,
        error=str(e)
    )
```

### Debugging Tool Failures

```python
# Retrieve recent tool logs for a conversation thread
logs = await harness.get_tool_log("chat-123", limit=20)

# Find failed executions
failures = [log for log in logs if log.metadata.get("status") == "error"]

for failure in failures:
    print(f"Tool: {failure.metadata['tool_name']}")
    print(f"Error: {failure.metadata.get('error', 'Unknown')}")
    print(f"Input: {failure.metadata.get('input', {})}")
    print(f"Time: {failure.created_at}")
    print("---")
```

### Tool Usage Analytics

```python
async def analyze_tool_usage(harness: MemoryHarness):
    """Generate statistics on tool usage patterns."""

    # Search for all tool logs (adjust limit as needed)
    logs = await harness.search_tool_logs("", k=1000)

    # Compute statistics
    tool_counts = {}
    tool_failures = {}
    tool_durations = {}

    for log in logs:
        tool_name = log.metadata.get("tool_name", "unknown")
        status = log.metadata.get("status", "unknown")
        duration = log.metadata.get("duration_ms", 0)

        # Count executions
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Count failures
        if status == "error":
            tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1

        # Track durations
        if tool_name not in tool_durations:
            tool_durations[tool_name] = []
        if duration > 0:
            tool_durations[tool_name].append(duration)

    # Print report
    print("Tool Usage Report")
    print("=" * 50)

    for tool_name in sorted(tool_counts.keys()):
        total = tool_counts[tool_name]
        failures = tool_failures.get(tool_name, 0)
        success_rate = ((total - failures) / total * 100) if total > 0 else 0

        avg_duration = 0
        if tool_name in tool_durations and tool_durations[tool_name]:
            avg_duration = sum(tool_durations[tool_name]) / len(tool_durations[tool_name])

        print(f"{tool_name}:")
        print(f"  Executions: {total}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Avg duration: {avg_duration:.0f}ms")
        print()

# Usage
await analyze_tool_usage(harness)
```

### Automatic Tool Wrapper

```python
from functools import wraps
from typing import Callable, Any

def logged_tool(tool_name: str, harness: MemoryHarness) -> Callable:
    """Decorator that automatically logs tool executions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                await harness.log_tool_execution(
                    tool_name=tool_name,
                    input_params=kwargs,
                    output_result={"result": str(result)[:500]},  # Truncate long results
                    success=True,
                    duration_ms=duration_ms
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                await harness.log_tool_execution(
                    tool_name=tool_name,
                    input_params=kwargs,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e)
                )

                raise

        return wrapper
    return decorator

# Usage
@logged_tool("github.create_issue", harness)
async def create_github_issue(title: str, body: str):
    # Tool implementation
    return await github_api.create_issue(title=title, body=body)
```

## Best Practices

1. **Log every tool execution**: Don't selectively log - capture all invocations for complete observability

2. **Include timing information**: Always record duration_ms for performance analysis

3. **Truncate large outputs**: Store summaries of large results rather than full content to avoid bloat

4. **Use consistent tool names**: Match tool names with those in toolbox memory for cross-referencing

5. **Set reasonable limits**: When retrieving logs, use appropriate limits to avoid loading thousands of entries

6. **Implement log rotation**: For long-running agents, periodically archive or summarize old logs

## Integration with Other Memory Types

Tool log memory integrates with other memory types:

- **Toolbox**: Tool definitions in toolbox map to executions in tool log
- **Skills**: Successful tool patterns in logs inform skill development
- **Workflow**: Tool logs show actual execution vs. planned workflow steps
- **Conversational**: Thread-scoped logs provide context for conversation turns
- **Summary**: Compress extensive tool logs into high-level activity summaries

## Performance Notes

- **SQL storage**: Efficient for chronological queries and exact-match lookups
- **No embeddings**: Logs don't use vector search - accessed by time, thread, or tool name
- **Ordered retrieval**: Logs are returned in chronological order for temporal analysis
- **Moderate default k**: Returns 10 logs by default - adjust based on your analysis needs
- **Thread isolation**: Scoping logs to threads keeps queries focused and fast
- **Index on tool_name**: Backend indexes enable fast filtering by tool name
