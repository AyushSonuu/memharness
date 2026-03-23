---
sidebar_position: 3
---

# In-Memory Backend

The in-memory backend stores everything in Python dicts — no persistence, no external dependencies. It's the fastest option and ideal for testing and prototyping.

## When to Use

| Scenario | Recommendation |
|----------|---------------|
| Unit tests | ✅ Best choice |
| Integration tests | ✅ Fast and isolated |
| Prototyping | ✅ No setup required |
| Production | ❌ Data lost on restart |

## Usage

```python
from memharness import MemoryHarness

# In-memory — no files, no database
harness = MemoryHarness("memory://")
await harness.connect()

await harness.add_conversational("thread1", "user", "Hello!")
messages = await harness.get_conversational("thread1")

await harness.disconnect()
```

## In Tests

```python
import pytest
from memharness import MemoryHarness

@pytest.fixture
async def harness():
    h = MemoryHarness("memory://")
    await h.connect()
    yield h
    await h.disconnect()

async def test_my_agent(harness):
    await harness.add_knowledge("Python is great", source="test")
    results = await harness.search_knowledge("Python")
    assert len(results) > 0
```

## Vector Search

The in-memory backend implements cosine similarity in pure Python. It's not optimized for large datasets but works correctly for testing.

## Limitations

- All data is lost when the process ends
- Not suitable for multi-process use
- No persistence between test runs (which is usually what you want!)
