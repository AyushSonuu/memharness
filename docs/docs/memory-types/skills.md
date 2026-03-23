---
sidebar_position: 1
---

# Skills Memory

Skills memory tracks learned capabilities and patterns discovered through experience, enabling agents to recognize and reuse successful approaches for recurring tasks.

## Overview

Skills memory represents the agent's growing expertise - capabilities learned through interaction, practice, and reflection. Unlike tools (which are predefined functions) or workflows (which are task-specific procedures), skills are abstract patterns that the agent recognizes it can apply across different situations.

This memory type excels at:
- **Capability tracking**: Catalog what the agent has learned to do
- **Pattern recognition**: Identify when learned skills apply to new situations
- **Skill discovery**: Build expertise through experience and reflection
- **Semantic retrieval**: Find relevant skills based on task descriptions
- **Competency modeling**: Track which skills are well-developed vs. nascent

Skills represent meta-knowledge about the agent's own capabilities, enabling self-aware task planning and continuous learning.

## When to Use

Use skills memory to:
- **Build agent expertise**: Record capabilities as the agent learns them
- **Enable skill-based planning**: Let agents select approaches based on proven skills
- **Track learning progress**: Monitor which skills are developing over time
- **Support transfer learning**: Apply skills learned in one context to another
- **Provide capability introspection**: Let agents describe what they can do
- **Guide skill development**: Identify gaps and areas for improvement

Skills memory is essential for agents designed to learn and improve over time.

## Storage Strategy

- **Backend**: VECTOR (semantic search with HNSW indexing)
- **Default k**: 5 results (focused skill retrieval)
- **Embeddings**: Yes (enables semantic matching of skills to tasks)
- **Ordered**: No (accessed by relevance to current task)
- **Categories**: Optional organization by skill domain or type

## Schema

Each skill memory includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `skill_name` | string | Yes | Name/identifier for the skill |
| `capability` | string | Yes | What the skill enables the agent to do |
| `name` | string | Yes | Display name for the skill |
| `description` | string | Yes | Detailed description of the skill |
| `examples` | array | No | Example usage scenarios |
| `category` | string | No | Skill category (e.g., "analysis", "creation") |
| `learned_from` | string | No | How the skill was acquired |
| `success_rate` | number | No | Success rate (0-1) based on usage |
| `usage_count` | integer | No | Number of times skill has been applied |

Additional fields are stored in the `metadata` dictionary.

## API Methods

### Adding Skills

```python
async def add_skill(
    name: str,
    description: str,
    examples: list[str] | None = None,
    category: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Add a learned skill to memory.

    Args:
        name: Name of the skill
        description: Description of what the skill does
        examples: Optional list of example usages
        category: Optional category for the skill
        **kwargs: Additional skill attributes

    Returns:
        The ID of the created skill memory
    """
```

### Searching Skills

```python
async def search_skills(
    query: str,
    k: int = 3,
) -> list[MemoryUnit]:
    """
    Search for relevant skills.

    Args:
        query: The search query
        k: Number of results to return

    Returns:
        List of matching skill MemoryUnit objects
    """
```

## Examples

### Recording Learned Skills

```python
from memharness import MemoryHarness

harness = MemoryHarness(backend="sqlite:///memory.db")

# Agent learns a new skill through successful experience
skill_id = await harness.add_skill(
    name="code_review",
    description="Review code for bugs, style issues, performance problems, and security vulnerabilities",
    examples=[
        "Review this Python function for efficiency",
        "Check this code for security vulnerabilities",
        "Analyze this module for potential bugs"
    ],
    category="analysis",
    learned_from="multiple successful code review tasks",
    success_rate=0.85,
    usage_count=12
)

print(f"Recorded skill: {skill_id}")
```

### Skill-Based Task Planning

```python
async def plan_with_skills(task_description: str, harness: MemoryHarness):
    """Use available skills to plan task approach."""

    # Search for relevant skills
    relevant_skills = await harness.search_skills(task_description, k=3)

    if not relevant_skills:
        print("No matching skills found - this is a new type of task")
        return None

    # Select the most relevant skill
    best_skill = relevant_skills[0]
    skill_name = best_skill.metadata.get("name", "unknown")
    examples = best_skill.metadata.get("examples", [])

    print(f"Found applicable skill: {skill_name}")
    print(f"Description: {best_skill.metadata.get('description', '')}")

    if examples:
        print(f"Example applications:")
        for example in examples[:3]:
            print(f"  - {example}")

    return best_skill

# Usage
task = "Review this Python script for potential issues"
skill = await plan_with_skills(task, harness)
```

### Building a Skill Catalog

```python
# Define multiple skills across different categories

# Analysis skills
await harness.add_skill(
    name="data_analysis",
    description="Analyze datasets to extract insights, identify patterns, and generate visualizations",
    category="analysis",
    examples=[
        "Analyze sales data to find trends",
        "Identify correlations in customer behavior"
    ]
)

# Creation skills
await harness.add_skill(
    name="api_design",
    description="Design RESTful APIs with proper resource modeling, authentication, and documentation",
    category="creation",
    examples=[
        "Design API for user management system",
        "Create endpoint schema for e-commerce platform"
    ]
)

# Debugging skills
await harness.add_skill(
    name="error_diagnosis",
    description="Diagnose errors by analyzing stack traces, logs, and system behavior",
    category="debugging",
    examples=[
        "Debug authentication failure from error logs",
        "Trace source of memory leak in application"
    ]
)

# Optimization skills
await harness.add_skill(
    name="performance_tuning",
    description="Optimize code and system performance through profiling, caching, and algorithmic improvements",
    category="optimization",
    examples=[
        "Improve database query performance",
        "Reduce API response time"
    ]
)
```

### Tracking Skill Development

```python
async def update_skill_stats(skill_id: str, success: bool, harness: MemoryHarness):
    """Update skill statistics after usage."""

    # Retrieve the skill
    skill = await harness._backend.get(skill_id)
    if not skill:
        return

    # Update usage count and success rate
    usage_count = skill.metadata.get("usage_count", 0) + 1
    current_success_rate = skill.metadata.get("success_rate", 0.0)
    current_successes = current_success_rate * (usage_count - 1)

    new_successes = current_successes + (1 if success else 0)
    new_success_rate = new_successes / usage_count

    # Store updated skill
    await harness.add_skill(
        name=skill.metadata["name"],
        description=skill.metadata["description"],
        examples=skill.metadata.get("examples", []),
        category=skill.metadata.get("category"),
        usage_count=usage_count,
        success_rate=new_success_rate
    )

    print(f"Updated skill stats: {usage_count} uses, {new_success_rate:.1%} success rate")

# Usage after completing a task
await update_skill_stats(skill_id, success=True, harness=harness)
```

## Best Practices

1. **Be specific with skill descriptions**: Clearly describe what the skill enables, not just what it is called

2. **Include diverse examples**: Provide multiple example scenarios showing different applications of the skill

3. **Track usage metrics**: Update usage_count and success_rate to understand which skills are well-developed

4. **Organize with categories**: Group related skills for easier navigation and planning

5. **Link to workflows**: Reference successful workflows that demonstrate the skill in action

6. **Reflect on failures**: When a skill fails, analyze why and potentially create a refined version

## Integration with Other Memory Types

Skills memory integrates closely with other memory types:

- **Workflow**: Successful workflows demonstrate skill application - extract skills from proven workflows
- **Tool Log**: Tool usage patterns can inform skill development (e.g., expertise with specific tools)
- **Toolbox**: Skills often involve knowing which tools to use and how to combine them
- **Knowledge**: Domain knowledge supports skill development and application
- **Conversational**: User feedback helps identify which skills are valuable

## Performance Notes

- **Semantic search**: Skills use embeddings to match capabilities to task descriptions
- **HNSW indexing**: Efficient nearest-neighbor search for finding relevant skills
- **Moderate default k**: Returns 5 skills by default - few enough to evaluate, broad enough for options
- **Category filtering**: Optional categories can narrow search space for faster retrieval
- **Incremental learning**: Skills can be updated with new examples and statistics over time
- **Transfer potential**: Semantic search enables discovering skills that transfer to new domains
