# memharness — Full Implementation Plan

## Current State (2026-03-23)
- 116 tests pass, 42 fail (all in registry), 43 skipped
- GitHub repo: AyushSonuu/memharness (public, main branch)
- Package name available on PyPI

## Architecture Decision: Single Source of Truth

### The Problem
Two MemoryType enums exist:
- `types.py` (Pydantic): uses `KNOWLEDGE_BASE = "knowledge_base"` 
- `harness.py` (dataclass): uses `KNOWLEDGE = "knowledge"`

Registry imports from types.py. Harness has its own. Tests use harness version.

### The Decision
**harness.py is the canonical source.** It's what the API uses, what tests reference, what users see.

Action: 
1. Update `types.py` to match harness.py values (KNOWLEDGE not KNOWLEDGE_BASE)
2. Make registry import from harness.py (not types.py)
3. types.py becomes supplementary (Pydantic models for SearchResult, SearchFilter, enums like StorageType, TriggerType)

## Phase 1: Fix Registry (42 failing tests)

### Required changes to registry.py:
1. Add `MemoryTypeRegistry.get_instance()` classmethod (singleton pattern)
2. Change `register()` to allow replacement (tests call register for existing types)
3. Add handler interface support (tests check for `handler_for()`, handler has `store/search/format` methods)
4. Add metadata methods: `get_description()`, `get_schema()`, `get_all_handlers()`
5. Import MemoryType from harness (not types.py), or unify the enum values

### Test expectations (from test_registry.py):
```python
registry = MemoryTypeRegistry.get_instance()  # singleton
registry.get(MemoryType.KNOWLEDGE)  # not KNOWLEDGE_BASE
registry.register(config)  # should replace, not raise
registry.list_types()  # returns list of MemoryType enums (not strings)
registry.handler_for(MemoryType.KNOWLEDGE)  # returns handler with store/search/format
registry.unregister(MemoryType.CUSTOM)
```

## Phase 2: Implement Skipped Tests

### Missing harness methods:
- `add_tool_log()`, `get_tool_log()` — partially there
- `add_skill()`, `search_skills()` — partially there  
- `set_persona()`, `get_active_persona()` — need implementation
- `unified_search()` — search across all memory types
- `memory_stats()` — return counts per type
- `clear_thread()` — delete all memories in a thread

## Phase 3: CI/CD

### GitHub Actions:
1. `.github/workflows/test.yml` — run pytest on push/PR
2. `.github/workflows/lint.yml` — ruff check + mypy
3. `.github/workflows/publish.yml` — publish to PyPI on tag

### Branch strategy:
- `main` — stable, tagged releases
- Feature branches → PR → merge

## Phase 4: LangChain Integration (use existing primitives)

Use langchain-core primitives:
- `BaseTool` for memory tools
- `BaseMemory` for LangChain memory adapter
- `BaseCheckpointSaver` for LangGraph checkpointer
- Don't reinvent — extend

## Phase 5: Build & Publish
1. Build: `python -m build`
2. Publish: `twine upload dist/*`
3. Docs: GitHub Pages with Docusaurus (already scaffolded in docs/)

## Commit Convention
```
feat: add new feature
fix: bug fix
docs: documentation only
test: adding tests
ci: CI/CD changes
refactor: code refactoring
chore: maintenance
```
