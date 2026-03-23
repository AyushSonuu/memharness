---
sidebar_position: 5
---

# Garbage Collection Agent

The GC Agent handles cleanup of expired memories. Unlike the other agents, it operates **entirely deterministically** — no LLM required.

## What It Does

1. Identifies memories past their TTL (time-to-live)
2. Archives old memories (moves to archive tier)
3. Permanently deletes memories past the delete threshold

## When It Runs

The GC Agent is triggered on-demand or on a schedule you control:

```python
from memharness import MemoryHarness
from memharness.agents import GCAgent

harness = MemoryHarness("sqlite:///memory.db")
await harness.connect()

gc = GCAgent(harness=harness)

# Run GC manually
result = await gc.run()
print(f"Archived: {result['archived']}, Deleted: {result['deleted']}")
```

## Configuration

```python
gc = GCAgent(
    harness=harness,
    archive_after_days=90,   # Archive memories older than 90 days
    delete_after_days=365,   # Delete archived memories older than 365 days
)
```

## No LLM Required

GC is fully deterministic — it just checks timestamps and applies policies. This is important: **simple operations should never require an LLM**.

> From the course: "Deterministic ops are like an alarm clock — they go off no matter what."

## Cleanup Schedule

Run GC periodically using a cron job or task scheduler:

```python
import asyncio

async def run_gc_daily():
    async with MemoryHarness("sqlite:///memory.db") as harness:
        gc = GCAgent(harness=harness)
        result = await gc.run()
        print(f"GC complete: {result}")

# Schedule with your preferred task scheduler
asyncio.run(run_gc_daily())
```
