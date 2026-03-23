---
sidebar_position: 1
---

# Persona Memory

Persona memory stores agent or user identity information in modular blocks, defining communication style, preferences, constraints, and behavioral traits that shape agent interactions.

## Overview

Persona memory enables agents to maintain consistent identity and behavior by storing personality traits, communication preferences, domain expertise, and operational constraints in structured blocks. Unlike static system prompts, persona memory is dynamic and composable, allowing different aspects of identity to be activated, updated, or combined based on context.

This memory type excels at:
- **Identity management**: Define and maintain consistent agent personality
- **Preference tracking**: Remember user preferences and communication styles
- **Behavioral constraints**: Store ethical guidelines and operational boundaries
- **Role adaptation**: Switch between different personas for different contexts
- **Personalization**: Tailor agent behavior to individual user needs

Persona memory enables agents to be both consistent (maintaining core identity) and adaptive (adjusting to context and user preferences).

## When to Use

Use persona memory to:
- **Define agent identity**: Establish personality traits, communication style, and expertise areas
- **Store user preferences**: Remember how each user prefers to interact
- **Implement role switching**: Enable agents to adopt different personas for different tasks
- **Track constraints**: Maintain ethical guidelines, content policies, and operational limits
- **Enable personalization**: Customize agent behavior for individual users or contexts
- **Support multi-agent systems**: Give each agent a distinct identity and role

Persona memory is essential for agents that need consistent personality, user-specific customization, or role-based behavior.

## Storage Strategy

- **Backend**: VECTOR (semantic search with HNSW indexing)
- **Default k**: 3 results (personas are high-level, so fewer results)
- **Embeddings**: Yes (enables semantic retrieval of relevant persona aspects)
- **Ordered**: No (accessed by relevance and block name)
- **Block-based**: Organized into named blocks (e.g., "communication_style", "preferences")

## Schema

Each persona memory includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `block_name` | string | Yes | Name of the persona block |
| `persona_name` | string | Yes | Name/identifier for the persona |
| `trait_type` | string | No | Type: "style", "constraint", "preference", "identity" |
| `priority` | integer | No | Priority level (1-10) for conflict resolution |
| `content` | string | Yes | The persona block content |

Additional persona attributes are stored in the `metadata` dictionary.

## API Methods

### Managing Persona Blocks

```python
async def add_persona(
    block_name: str,
    content: str,
) -> str:
    """
    Add or update a persona block.

    Args:
        block_name: Name of the persona block (e.g., "preferences", "background")
        content: The persona content

    Returns:
        The ID of the created/updated persona block
    """

async def get_persona(
    block_name: str | None = None,
) -> str:
    """
    Retrieve persona content.

    Args:
        block_name: Optional specific block name. If None, returns all blocks.

    Returns:
        The persona content as a string
    """
```

### Setting Complete Personas

```python
async def set_persona(
    name: str,
    traits: list[str] | None = None,
    communication_style: str | None = None,
    domain_expertise: list[str] | None = None,
    **kwargs: Any,
) -> str:
    """
    Set the active persona for the agent.

    Args:
        name: Name of the persona
        traits: List of personality traits
        communication_style: Communication style description
        domain_expertise: List of domain expertise areas
        **kwargs: Additional persona attributes

    Returns:
        The ID of the created persona
    """

async def get_active_persona() -> MemoryUnit | None:
    """
    Get the active persona.

    Returns:
        The active persona as a MemoryUnit, or None if no persona is set
    """
```

## Examples

### Defining Agent Identity

```python
from memharness import MemoryHarness

harness = MemoryHarness(backend="sqlite:///memory.db")

# Define core identity
await harness.set_persona(
    name="Technical Assistant",
    traits=["helpful", "precise", "thorough", "patient"],
    communication_style="professional and clear, with code examples",
    domain_expertise=["python", "devops", "system architecture", "API design"]
)

# Add specific behavioral blocks
await harness.add_persona(
    block_name="communication_style",
    content="""
    - Use clear, concise technical language
    - Provide code examples when explaining concepts
    - Break down complex topics into digestible steps
    - Ask clarifying questions when requirements are ambiguous
    """
)

await harness.add_persona(
    block_name="constraints",
    content="""
    - Never execute destructive operations without explicit confirmation
    - Prioritize security and data privacy in all recommendations
    - Acknowledge limitations rather than guessing
    - Escalate to human review for critical decisions
    """
)
```

### User-Specific Preferences

```python
# Store preferences for different users
user_id = "user-123"

await harness.add_persona(
    block_name=f"user_preferences_{user_id}",
    content="""
    User: Sarah (user-123)
    Preferences:
    - Prefers Python over JavaScript when both options work
    - Likes detailed explanations with examples
    - Working timezone: US Pacific
    - Current projects: E-commerce backend, API integration
    - Learning interests: GraphQL, microservices architecture
    """
)

# Later, retrieve user preferences
user_prefs = await harness.get_persona(f"user_preferences_{user_id}")
print(f"Preferences for user: {user_prefs}")
```

### Role-Based Persona Switching

```python
# Define different personas for different roles
code_reviewer_persona = await harness.set_persona(
    name="Code Reviewer",
    traits=["thorough", "constructive", "detail-oriented"],
    communication_style="direct feedback with specific suggestions",
    domain_expertise=["code quality", "security", "performance", "best practices"]
)

await harness.add_persona(
    block_name="code_reviewer_approach",
    content="""
    When reviewing code:
    1. Check for security vulnerabilities first
    2. Evaluate performance implications
    3. Assess readability and maintainability
    4. Verify test coverage
    5. Suggest specific improvements with examples
    """
)

# Define teaching persona
teacher_persona = await harness.set_persona(
    name="Programming Teacher",
    traits=["patient", "encouraging", "thorough", "adaptive"],
    communication_style="explanatory with analogies and progressive complexity",
    domain_expertise=["python basics", "web development", "data structures"]
)

await harness.add_persona(
    block_name="teaching_approach",
    content="""
    Teaching methodology:
    - Start with simple examples and build complexity gradually
    - Use real-world analogies to explain abstract concepts
    - Encourage experimentation and learning from mistakes
    - Provide exercises to reinforce understanding
    - Check comprehension regularly with questions
    """
)
```

### Building Context with Persona

```python
async def build_prompt_with_persona(user_query: str, harness: MemoryHarness):
    """Build LLM prompt including relevant persona information."""

    # Get all persona blocks
    full_persona = await harness.get_persona()

    # Build prompt
    prompt = f"""
{full_persona}

User Query: {user_query}

Respond according to the persona guidelines above.
"""

    return prompt

# Usage
query = "How do I implement authentication in FastAPI?"
prompt = await build_prompt_with_persona(query, harness)
```

### Dynamic Persona Composition

```python
# Combine multiple persona aspects for specific contexts
async def get_context_appropriate_persona(context: str, harness: MemoryHarness):
    """Retrieve persona blocks relevant to current context."""

    blocks_to_include = []

    if "code review" in context.lower():
        blocks_to_include.append(await harness.get_persona("code_reviewer_approach"))
        blocks_to_include.append(await harness.get_persona("constraints"))

    if "teaching" in context.lower() or "explain" in context.lower():
        blocks_to_include.append(await harness.get_persona("teaching_approach"))
        blocks_to_include.append(await harness.get_persona("communication_style"))

    if "user" in context:
        # Include user-specific preferences
        user_id = extract_user_id(context)  # Your logic here
        user_prefs = await harness.get_persona(f"user_preferences_{user_id}")
        if user_prefs:
            blocks_to_include.append(user_prefs)

    # Combine relevant blocks
    return "\n\n---\n\n".join(blocks_to_include)

# Usage
context = "code review for user-123"
persona = await get_context_appropriate_persona(context, harness)
```

## Best Practices

1. **Use modular blocks**: Separate different aspects of persona (style, constraints, expertise) into distinct blocks for flexible composition

2. **Keep blocks focused**: Each block should address one aspect of personality or behavior

3. **Update incrementally**: Modify specific blocks rather than rewriting entire persona

4. **Version persona changes**: Track how persona evolves over time by maintaining history

5. **Balance consistency and adaptation**: Maintain core identity while allowing context-specific adjustments

6. **Include priority levels**: Use priority metadata to resolve conflicts when blocks contradict

## Integration with Other Memory Types

Persona memory integrates with other memory types:

- **Conversational**: Persona shapes how the agent responds in conversations
- **Skills**: Domain expertise in persona relates to available skills
- **Knowledge**: Persona can reference preferred knowledge sources or domains
- **Workflow**: Persona influences how workflows are executed and communicated
- **Entity**: User preferences in persona link to entity information about users

## Performance Notes

- **Semantic retrieval**: Persona blocks can be retrieved semantically based on context
- **Block-level updates**: Update individual blocks without reloading entire persona
- **Small default k**: Returns 3 persona blocks by default since they're high-level
- **HNSW indexing**: Efficient similarity search for context-relevant persona aspects
- **Cache frequently used blocks**: Core identity blocks can be cached for fast access
- **Lightweight composition**: Combine relevant blocks on-demand rather than loading everything
