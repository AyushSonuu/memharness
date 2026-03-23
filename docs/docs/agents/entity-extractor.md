---
sidebar_position: 3
---

# Entity Extractor Agent

Extract named entities (people, places, organizations) from text for structured memory.

## Overview

The Entity Extractor Agent identifies and extracts named entities from text, populating the entity memory type with structured information. It enables agents to build knowledge graphs and track relationships between entities across conversations.

### When to Use

- Extract entities from conversation messages (ON_WRITE trigger)
- Extract entities from AI responses (POST_LLM trigger)
- Parse documents for entities (on-demand)
- Build CRM-style knowledge graphs
- Enable entity-centric queries and recommendations

### Dual-Mode Operation

**Mode 1: Regex-Based Heuristics (No LLM)**
- Extracts capitalized words (potential proper nouns)
- Finds @mentions
- Extracts email addresses
- Simple heuristics for classification
- Output: `{"people": ["@john"], "organizations": ["Acme Corp"], "locations": ["NYC"]}`

**Mode 2: LLM-Powered NER**
- Uses LangChain with structured JSON output
- Accurate named entity recognition
- Semantic classification
- Handles ambiguous cases
- Output: `{"people": ["John Smith"], "organizations": ["Acme Corporation"], "locations": ["New York City"]}`

## API Methods

### extract_entities

Extract entities from text.

```python
async def extract_entities(
    text: str
) -> dict[str, list[str]]
```

**Parameters**:
- `text`: The text to extract entities from

**Returns**: Dictionary with keys `people`, `organizations`, `locations` (each a list of strings, max 10 per type)

**Example**:
```python
from memharness import MemoryHarness
from memharness.agents import EntityExtractorAgent
from langchain.chat_models import init_chat_model

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Heuristic mode
    agent_basic = EntityExtractorAgent(harness)
    entities = await agent_basic.extract_entities(
        "Dr. Chen works at MIT in Cambridge"
    )
    # Output: {"people": [], "organizations": ["Mit"], "locations": ["Chen"]}

    # LLM mode (accurate)
    llm = init_chat_model("gpt-4o-mini")
    agent_smart = EntityExtractorAgent(harness, llm=llm)
    entities = await agent_smart.extract_entities(
        "Dr. Chen works at MIT in Cambridge"
    )
    # Output: {"people": ["Dr. Chen"], "organizations": ["MIT"], "locations": ["Cambridge"]}
```

### run

Execute the entity extractor agent (standard agent interface).

```python
async def run(
    text: str,
    **kwargs
) -> dict[str, Any]
```

**Parameters**:
- `text`: The text to extract entities from
- `**kwargs`: Additional arguments (ignored)

**Returns**: Dictionary with `entities` and `total_extracted` keys

**Example**:
```python
result = await agent.run(text="John works at Acme Corp in NYC")
# Returns: {
#     "entities": {
#         "people": ["John"],
#         "organizations": ["Acme Corp"],
#         "locations": ["NYC"]
#     },
#     "total_extracted": 3
# }
```

## Implementation Details

### Heuristic Mode

The heuristic mode uses regex patterns:

```python
import re

def _heuristic_extraction(self, text: str) -> dict[str, list[str]]:
    # Extract capitalized words (potential proper nouns)
    capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

    # Extract @mentions
    mentions = re.findall(r"@([a-zA-Z0-9_]+)", text)

    # Extract email addresses (potential people)
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    email_names = [email.split("@")[0].replace(".", " ").title() for email in emails]

    # Combine and deduplicate
    people = list(set(mentions + email_names))
    potential_entities = list(set(capitalized))

    # Simple heuristic: single words likely locations, multi-word likely orgs
    locations = [e for e in potential_entities if len(e.split()) == 1 and len(e) <= 5]
    organizations = [e for e in potential_entities if len(e.split()) > 1]

    return {
        "people": people[:10],  # Limit to 10
        "organizations": organizations[:10],
        "locations": locations[:10],
    }
```

**Advantages**:
- Instant execution
- No LLM costs
- Works offline
- Good for structured text (@mentions, emails)

**Limitations**:
- Low accuracy on ambiguous cases
- Cannot distinguish entity types reliably
- Misses non-capitalized entities
- No semantic understanding

### LLM Mode

The LLM mode uses structured output:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a named entity recognition system. Extract people, "
     "organizations, and locations from the text. Return JSON with "
     'keys: "people", "organizations", "locations" (each a list of strings).'),
    ("user", "{text}")
])

# Build chain with JSON output parser
parser = JsonOutputParser()
chain = prompt | self.llm | parser

# Extract entities
result = await chain.ainvoke({"text": text})
entities = {
    "people": result.get("people", [])[:10],
    "organizations": result.get("organizations", [])[:10],
    "locations": result.get("locations", [])[:10],
}
```

**Advantages**:
- High accuracy
- Semantic understanding
- Handles ambiguous cases
- Proper entity classification

**Limitations**:
- Requires LLM API access
- Incurs API costs
- Slower than heuristic mode
- May hallucinate entities

### Fallback Strategy

The LLM mode automatically falls back to heuristics on errors:

```python
try:
    result = await chain.ainvoke({"text": text})
    return {
        "people": result.get("people", [])[:10],
        "organizations": result.get("organizations", [])[:10],
        "locations": result.get("locations", [])[:10],
    }
except Exception:
    # Fall back to heuristic on error
    return self._heuristic_extraction(text)
```

## Integration Patterns

### 1. ON_WRITE Trigger (Automatic Extraction)

Extract entities from every message:

```python
from memharness import MemoryHarness
from memharness.agents import EntityExtractorAgent

async def add_message_with_entities(thread_id: str, role: str, content: str):
    """Add message and automatically extract entities."""
    # Add to conversational memory
    await harness.add_conversational(thread_id, role, content)

    # Extract entities
    agent = EntityExtractorAgent(harness, llm=llm)
    entities = await agent.extract_entities(content)

    # Store extracted entities
    for entity_type, entity_list in entities.items():
        for entity_name in entity_list:
            await harness.add_entity(
                name=entity_name,
                entity_type=entity_type.rstrip('s'),  # "people" → "person"
                description=f"Mentioned in conversation: {content[:100]}"
            )
```

### 2. POST_LLM Trigger (Extract from AI Responses)

Extract entities from agent responses:

```python
async def agent_loop():
    """Main agent loop with entity extraction."""
    while True:
        user_input = input("User: ")

        # Get AI response
        response = await llm.ainvoke(user_input)

        # Extract entities from response
        agent = EntityExtractorAgent(harness, llm=llm)
        entities = await agent.extract_entities(response.content)

        # Store entities
        for entity_type, entity_list in entities.items():
            for entity_name in entity_list:
                await harness.add_entity(
                    name=entity_name,
                    entity_type=entity_type.rstrip('s'),
                    description=f"Mentioned in agent response"
                )

        print(f"AI: {response.content}")
```

### 3. Batch Extraction (On-Demand)

Extract entities from multiple documents:

```python
async def batch_extract_entities(documents: list[str]):
    """Extract entities from a batch of documents."""
    agent = EntityExtractorAgent(harness, llm=llm)

    all_entities = {"people": set(), "organizations": set(), "locations": set()}

    for doc in documents:
        entities = await agent.extract_entities(doc)
        all_entities["people"].update(entities["people"])
        all_entities["organizations"].update(entities["organizations"])
        all_entities["locations"].update(entities["locations"])

    # Store unique entities
    for entity_type, entity_set in all_entities.items():
        for entity_name in entity_set:
            await harness.add_entity(
                name=entity_name,
                entity_type=entity_type.rstrip('s'),
                description="Extracted from document corpus"
            )

    return {k: list(v) for k, v in all_entities.items()}
```

### 4. Tool-Called (Inside Loop)

Expose as a LangChain tool:

```python
from langchain_core.tools import tool

@tool
async def extract_entities_tool(text: str) -> dict:
    """Extract named entities from text."""
    agent = EntityExtractorAgent(harness, llm=llm)
    entities = await agent.extract_entities(text)
    return entities

# Agent can call this tool
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[extract_entities_tool, ...],
)
```

## Configuration

### YAML Configuration

```yaml
agents:
  entity_extractor:
    enabled: true
    llm: gpt-4o-mini

    # Trigger on every write
    trigger: on_write

    # Entity types to extract
    entity_types:
      - person
      - organization
      - location
      - concept
      - system

    # Automatic storage
    auto_store: true
```

### Python Configuration

```python
from memharness.agents import EntityExtractorAgent
from langchain.chat_models import init_chat_model

# Basic initialization
agent = EntityExtractorAgent(harness)

# With LLM for accurate NER
llm = init_chat_model("gpt-4o-mini")
agent = EntityExtractorAgent(harness, llm=llm)

# Extract entities
entities = await agent.extract_entities("Dr. Chen works at MIT")
```

## Best Practices

### 1. Use LLM Mode for Production

```python
# Heuristic mode is inaccurate — use for prototyping only
agent_heuristic = EntityExtractorAgent(harness)  # Low accuracy

# LLM mode for production
llm = init_chat_model("gpt-4o-mini")  # Fast + cheap
agent_production = EntityExtractorAgent(harness, llm=llm)
```

### 2. Deduplicate Entities

```python
async def add_entity_safe(name: str, entity_type: str, description: str):
    """Add entity only if it doesn't already exist."""
    # Search for existing entity
    existing = await harness.search_entity(name, entity_type=entity_type, k=1)

    if existing and existing[0].metadata.get("entity_name") == name:
        # Update existing entity
        print(f"Entity '{name}' already exists")
        return existing[0].id
    else:
        # Add new entity
        return await harness.add_entity(name, entity_type, description)
```

### 3. Enrich Entities Over Time

```python
async def enrich_entity(entity_name: str, new_info: str):
    """Add information to existing entity."""
    # Find entity
    results = await harness.search_entity(entity_name, k=1)
    if not results:
        return

    entity = results[0]

    # Append new information
    updated_description = f"{entity.content}\n{new_info}"

    # Update (implementation-specific)
    await harness.update_entity(
        entity_id=entity.id,
        description=updated_description
    )
```

### 4. Track Entity Relationships

```python
async def extract_and_link_entities(text: str, thread_id: str):
    """Extract entities and track their relationships."""
    agent = EntityExtractorAgent(harness, llm=llm)
    entities = await agent.extract_entities(text)

    # Store entities with relationships
    for person in entities["people"]:
        for org in entities["organizations"]:
            await harness.add_entity(
                name=person,
                entity_type="person",
                description=f"Mentioned in thread {thread_id}",
                relationships=[{"target": org, "type": "associated_with"}]
            )

    for org in entities["organizations"]:
        for location in entities["locations"]:
            await harness.add_entity(
                name=org,
                entity_type="organization",
                description=f"Mentioned in thread {thread_id}",
                relationships=[{"target": location, "type": "located_in"}]
            )
```

### 5. Use Cheap Models

```python
# Entity extraction is simple — use cheap models
llm = init_chat_model("gpt-4o-mini")  # Claude Haiku or GPT-4o-mini
agent = EntityExtractorAgent(harness, llm=llm)

# Don't use expensive models for NER
# llm = init_chat_model("gpt-4o")  # ❌ Overkill and expensive
```

## Extended Entity Types

Beyond the default three types, you can extract custom entity types:

```python
# Extend the system prompt for custom entities
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract entities from text. Return JSON with keys: "
     '"people", "organizations", "locations", "products", "technologies", "events".'),
    ("user", "{text}")
])

chain = prompt | llm | JsonOutputParser()
result = await chain.ainvoke({"text": "Apple released iPhone 15 at WWDC 2023"})
# Output: {
#     "people": [],
#     "organizations": ["Apple"],
#     "locations": [],
#     "products": ["iPhone 15"],
#     "technologies": [],
#     "events": ["WWDC 2023"]
# }
```

## Related Components

- [Entity Memory Type](../memory-types/entity) — Stores extracted entities
- [Consolidator Agent](./consolidator) — Merges duplicate entities
- [Context Assembly Agent](./context-assembler) — Uses entities for context

## Next Steps

- [Consolidator](./consolidator) — Merge duplicate entities
- [Entity Memory](../memory-types/entity) — Entity storage details
- [Knowledge Graphs](../concepts/memory-types) — Building relationship graphs
