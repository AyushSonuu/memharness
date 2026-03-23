---
sidebar_position: 4
---

# Workflow Memory

Workflow memory stores task procedures and execution outcomes, capturing step-by-step processes for completing complex tasks with expected and actual results.

## Overview

Workflows are procedural memories that help your agent learn and reuse successful patterns for complex, multi-step tasks. Each workflow captures:
- The task being performed
- A sequence of steps to complete it
- The expected outcome
- The actual result (success or failure)

This memory type uses **vector storage** with semantic search, making it easy to find relevant workflows based on task similarity rather than exact matching.

## When to Use

Use workflow memory to:
- **Record successful task patterns**: Store multi-step procedures that work well
- **Learn from failures**: Document what didn't work to avoid repeating mistakes
- **Discover similar tasks**: Find workflows that match new problems
- **Build procedural knowledge**: Create a library of reusable patterns

## Storage Strategy

- **Backend**: VECTOR (semantic search with HNSW indexing)
- **Default k**: 3 results (workflows are typically specific, so fewer results)
- **Embeddings**: Yes (enables semantic similarity search)
- **Ordering**: No (accessed by relevance, not chronologically)

## Schema

Each workflow memory includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow_name` | string | Yes | Name or identifier for this workflow |
| `steps` | array | No | List of step objects with `step_number`, `action`, `expected_output` |
| `preconditions` | array | No | Conditions that must be true before starting |
| `postconditions` | array | No | Expected state after completion |

Additional fields are stored in the `metadata` dictionary.

## API Methods

### Adding Workflows

```python
async def add_workflow(
    task: str,
    steps: list[str],
    outcome: str,
    result: str | None = None,
) -> str:
    """
    Add a workflow/procedure to memory.

    Args:
        task: Description of the task (e.g., "Deploy application to production")
        steps: List of steps to complete the task
        outcome: Expected outcome description
        result: Actual result (if completed)

    Returns:
        ID of the created workflow memory
    """
```

### Searching Workflows

```python
async def search_workflow(
    query: str,
    k: int = 3,
) -> list[MemoryUnit]:
    """
    Search for workflows by semantic similarity.

    Args:
        query: Search query (task description, keywords, etc.)
        k: Number of results to return (default: 3)

    Returns:
        List of matching workflow memories, sorted by relevance
    """
```

## Examples

### Recording a Successful Deployment

```python
from memharness import MemoryHarness

harness = MemoryHarness(backend="sqlite:///memory.db")

# Record a successful deployment workflow
wf_id = await harness.add_workflow(
    task="Deploy application to production",
    steps=[
        "Run unit and integration tests",
        "Build Docker image",
        "Push image to container registry",
        "Update Kubernetes deployment",
        "Verify health checks"
    ],
    outcome="Application deployed and healthy",
    result="Deployed v2.1.0 successfully"
)

print(f"Workflow recorded: {wf_id}")
```

### Finding Similar Workflows

```python
# Later, when faced with a similar task
workflows = await harness.search_workflow("deploy application", k=3)

for wf in workflows:
    print(f"Task: {wf.metadata['task']}")
    print(f"Steps: {wf.metadata.get('steps', [])}")
    print(f"Outcome: {wf.metadata['outcome']}")
    print(f"Result: {wf.metadata.get('result', 'Not recorded')}")
    print()
```

### Recording a Failed Attempt

```python
# Document what didn't work
wf_id = await harness.add_workflow(
    task="Optimize database query performance",
    steps=[
        "Added index on user_id column",
        "Attempted to denormalize user table",
        "Ran ANALYZE to update statistics"
    ],
    outcome="Query should execute in <100ms",
    result="FAILED: Denormalization caused data consistency issues. Rollback required."
)
```

### Using Workflows in Agent Logic

```python
async def handle_task(task_description: str):
    """Agent function that learns from past workflows."""

    # Search for similar past workflows
    similar_workflows = await harness.search_workflow(task_description, k=3)

    if similar_workflows:
        print("Found similar past workflows:")
        for wf in similar_workflows:
            if "FAILED" in wf.metadata.get("result", ""):
                print(f"⚠️  Past failure: {wf.metadata['result']}")
            elif "success" in wf.metadata.get("result", "").lower():
                print(f"✅ Past success: Use these steps:")
                for step in wf.metadata.get("steps", []):
                    print(f"   - {step}")
```

## Best Practices

1. **Be specific with task descriptions**: Use clear, descriptive task names that capture the essence of what you're doing

2. **Include failure workflows**: Failed attempts are valuable learning experiences - document what went wrong and why

3. **Standardize step format**: Keep step descriptions concise and action-oriented (e.g., "Run tests" not "The tests were run")

4. **Add context in metadata**: Store relevant details like environment, tool versions, or special conditions in the metadata

5. **Keep it reusable**: Write workflows that could apply to similar future tasks, not overly specific to one instance

6. **Update with new learnings**: As you discover better ways to complete tasks, add improved workflows

## Integration with Other Memory Types

Workflow memory complements other memory types:

- **Tool Log**: Workflow steps often involve tool executions - cross-reference with tool logs for detailed audit trails
- **Skills**: Workflows represent the application of skills - link successful workflows to skill development
- **Entity**: Workflows may involve specific people, systems, or organizations stored in entity memory
- **Knowledge**: Detailed technical documentation in knowledge base can supplement workflow steps

## Performance Notes

- **Semantic search**: Workflow retrieval uses embeddings for similarity matching, so you don't need exact keyword matches
- **Small default k**: Only 3 results by default since workflows are typically task-specific
- **Async operations**: All workflow operations are async for efficient backend interaction
