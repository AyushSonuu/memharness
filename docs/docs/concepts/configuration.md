---
sidebar_position: 4
---

# Configuration

memharness is fully configurable via Python code, YAML files, or environment variables.

## Configuration Methods

### 1. Python Code

```python
from memharness import MemoryHarness
from memharness.config import (
    Config,
    SummarizationConfig,
    ConsolidationConfig,
    GCConfig,
    EntityExtractionConfig,
    ContextAssemblyConfig,
)

memory = MemoryHarness(
    backend="postgresql://localhost/memharness",
    config=Config(
        summarization=SummarizationConfig(
            enabled=True,
            triggers=[
                {"condition": "age > 7d", "memory_type": "conversational"},
            ],
            keep_originals=True,
            originals_ttl="365d",
        ),
        consolidation=ConsolidationConfig(
            enabled=True,
            schedule="0 3 * * *",
            similarity_threshold=0.9,
        ),
        gc=GCConfig(
            enabled=True,
            schedule="0 4 * * 0",
            archive_after="90d",
            delete_after="365d",
        ),
        entity_extraction=EntityExtractionConfig(
            enabled=True,
            mode="on_write",
        ),
        context_assembly=ContextAssemblyConfig(
            default_max_tokens=4000,
            priorities={
                "conversational": 0.40,
                "knowledge_base": 0.30,
                "entity": 0.15,
                "workflow": 0.10,
                "summary": 0.05,
            },
        ),
    ),
)
```

### 2. YAML File

```yaml
# memharness.yaml
backend: postgresql://localhost/memharness

conversational:
  max_messages_per_thread: 1000
  default_ttl: null

summarization:
  enabled: true
  triggers:
    - condition: "age > 7d"
      memory_type: conversational
    - condition: "message_count > 50"
      memory_type: conversational
  keep_originals: true
  originals_ttl: 365d

consolidation:
  enabled: true
  schedule: "0 3 * * *"
  similarity_threshold: 0.9

gc:
  enabled: true
  schedule: "0 4 * * 0"
  archive_after: 90d
  delete_after: 365d

entity_extraction:
  enabled: true
  mode: on_write
  types:
    - PERSON
    - ORG
    - PLACE
    - CONCEPT

context_assembly:
  default_max_tokens: 4000
  priorities:
    conversational: 0.40
    knowledge_base: 0.30
    entity: 0.15
    workflow: 0.10
    summary: 0.05
```

```python
memory = MemoryHarness.from_config("memharness.yaml")
```

### 3. Environment Variables

```bash
# Backend
export MEMHARNESS_BACKEND=postgresql://localhost/memharness

# Quick toggles
export MEMHARNESS_SUMMARIZATION_ENABLED=true
export MEMHARNESS_CONSOLIDATION_ENABLED=true
export MEMHARNESS_GC_ENABLED=true

# LLM for agents
export MEMHARNESS_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

```python
memory = MemoryHarness.from_env()
```

## Configuration Options

### Backend

```yaml
backend: postgresql://user:pass@host:5432/db
# or
backend: sqlite:///path/to/memory.db
# or
backend: memory://
```

### Summarization

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable summarization |
| `triggers` | list | [] | Conditions that trigger summarization |
| `keep_originals` | bool | true | Archive originals (don't delete) |
| `originals_ttl` | str | "365d" | How long to keep originals |

**Trigger conditions**:
- `age > 7d` — Messages older than 7 days
- `message_count > 50` — More than 50 messages in thread
- `context_usage > 80%` — Context window above 80%

### Consolidation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable consolidation |
| `schedule` | str | "0 3 * * *" | Cron schedule |
| `similarity_threshold` | float | 0.9 | Merge if > 90% similar |
| `memory_types` | list | ["entity"] | Types to consolidate |

### Garbage Collection

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable GC |
| `schedule` | str | "0 4 * * 0" | Cron schedule (weekly) |
| `archive_after` | str | "90d" | Move to cold storage |
| `delete_after` | str | "365d" | Delete from cold |
| `protect_referenced` | bool | true | Don't delete if referenced |

### Entity Extraction

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable entity extraction |
| `mode` | str | "on_write" | "on_write", "batch", or "disabled" |
| `types` | list | [...] | Entity types to extract |
| `min_confidence` | float | 0.7 | Minimum confidence |

### Context Assembly

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_max_tokens` | int | 4000 | Default token budget |
| `priorities` | dict | {...} | Token allocation per type |
| `expand_summaries` | bool | false | Auto-expand summaries |

## Duration Format

Durations can be specified as:
- `7d` — 7 days
- `24h` — 24 hours
- `30m` — 30 minutes
- `365d` — 1 year

## Cron Format

Standard 5-field cron: `minute hour day-of-month month day-of-week`

Examples:
- `0 3 * * *` — Daily at 3 AM
- `0 4 * * 0` — Weekly on Sunday at 4 AM
- `0 */6 * * *` — Every 6 hours

## Per-Namespace Overrides

```python
# Global config
memory = MemoryHarness(backend="...", config=global_config)

# Override for specific namespace
memory.configure_namespace(
    namespace=("org", "enterprise"),
    config=Config(
        summarization=SummarizationConfig(
            triggers=[{"condition": "age > 30d"}],  # Keep longer
        ),
        gc=GCConfig(
            delete_after="7y",  # 7 year retention
        ),
    ),
)
```

## Runtime Updates

```python
# Update config without restart
await memory.update_config(
    summarization={"triggers": [{"condition": "age > 14d"}]}
)

# Enable/disable agents
await memory.disable_agent("consolidation")
await memory.enable_agent("consolidation")
```
