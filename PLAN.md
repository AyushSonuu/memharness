# memharness v0.1.0 → v0.2.0 Implementation Plan

## Current State Analysis

### Codebase Health
- **168 tests passing**, 0 failing, 33 skipped
- CI: tests ✅, docs ✅, lint ❌ (ruff warnings)
- `harness.py` is **2564 lines** — needs splitting
- Dual `MemoryUnit` definitions (dataclass in harness.py, Pydantic in types.py)
- Agent modules are stubs (not using LangChain)
- Integration modules have undefined names (`asyncio` in langgraph.py)

### Architecture Problems
1. **God file**: harness.py contains MemoryType, MemoryUnit, MemharnessConfig, BackendProtocol, InMemoryBackend, backend factory, embedding function, AND the main MemoryHarness class
2. **Dead code**: types.py Pydantic MemoryUnit is never used by the core
3. **Agent stubs**: agents/ has skeleton code that doesn't use LangChain
4. **Integration bugs**: langgraph.py references undefined `asyncio`

---

## Target Architecture (v0.2.0)

```
src/memharness/
├── __init__.py              # Clean public API
├── _version.py              # Version string
├── types.py                 # MemoryType, MemoryUnit, SearchResult, SearchFilter (SINGLE SOURCE)
├── exceptions.py            # All custom exceptions
├── config/
│   ├── __init__.py
│   ├── models.py            # Pydantic config models
│   └── loader.py            # YAML/env loading
├── core/
│   ├── __init__.py
│   ├── harness.py           # MemoryHarness class (slim: delegates to managers)
│   ├── embedding.py         # Embedding function registry
│   ├── context.py           # Context assembly logic
│   └── namespace.py         # Namespace utilities
├── backends/
│   ├── __init__.py
│   ├── protocol.py          # BackendProtocol (abstract)
│   ├── memory.py            # InMemoryBackend
│   ├── sqlite.py            # SqliteBackend
│   └── postgres.py          # PostgresBackend
├── memory_types/            # One module per memory type
│   ├── __init__.py
│   ├── base.py              # BaseMemoryManager
│   ├── conversational.py    # ConversationalManager
│   ├── knowledge.py         # KnowledgeManager
│   ├── entity.py            # EntityManager
│   ├── workflow.py          # WorkflowManager
│   ├── toolbox.py           # ToolboxManager (VFS)
│   ├── summary.py           # SummaryManager (expandable)
│   ├── tool_log.py          # ToolLogManager
│   ├── skills.py            # SkillsManager
│   ├── file.py              # FileManager
│   └── persona.py           # PersonaManager
├── agents/                  # LangChain-based embedded agents
│   ├── __init__.py
│   ├── base.py              # BaseMemoryAgent (extends LangChain BaseTool)
│   ├── summarizer.py        # Uses LangChain primitives
│   ├── consolidator.py
│   ├── entity_extractor.py
│   ├── gc.py
│   └── scheduler.py
├── tools/                   # Agent self-exploration tools
│   ├── __init__.py
│   ├── definitions.py       # Tool definitions as LangChain BaseTool subclasses
│   └── executor.py
├── integrations/
│   ├── __init__.py
│   ├── langchain.py         # LangChain BaseMemory adapter
│   └── langgraph.py         # LangGraph BaseCheckpointSaver adapter
└── registry.py              # MemoryTypeRegistry
```

---

## Execution Plan (Feature Branches)

### Phase 1: Fix CI (branch: `fix/lint-ci`)
- Fix all ruff lint errors
- Fix pyproject.toml ruff config (use lint.select not select)
- Fix the old Deploy Docs workflow (rename/remove)
- PR → merge to main

### Phase 2: Split harness.py (branch: `refactor/split-harness`)
- Extract types → types.py (single MemoryUnit, delete old Pydantic one)
- Extract BackendProtocol → backends/protocol.py
- Extract InMemoryBackend → backends/memory.py (already exists, reconcile)
- Extract embedding logic → core/embedding.py
- Extract context assembly → core/context.py
- Create memory_types/ managers (one per type)
- Slim harness.py → core/harness.py (delegates to managers)
- Create exceptions.py
- ALL tests must still pass after refactor
- PR → merge to main

### Phase 3: LangChain agents (branch: `feat/langchain-agents`)
- Research latest LangChain/LangGraph memory patterns (MCP docs)
- Rewrite agents/ using langchain-core primitives
- Tools as BaseTool subclasses
- Integration tests
- PR → merge to main

### Phase 4: Polish & Publish (branch: `feat/publish-ready`)
- Implement remaining skipped tests
- Add py.typed marker
- Build with `python -m build`
- Test with `twine check`
- Tag v0.1.0 release
- Publish to PyPI

---

## Git Workflow
```
main ← PR ← feature-branch
         ↑
    Claude Code review (automated)
```

Each branch:
1. `git checkout -b <branch> main`
2. Implement + test
3. `git push origin <branch>`
4. Create PR via `gh pr create`
5. Review (automated or manual)
6. Merge via `gh pr merge`

## Key Principles
- **No wheel reinvention**: Use LangChain for agents/tools
- **One file, one concern**: No 2500-line god files
- **Tests first**: Run after every change
- **Clean commits**: Conventional commits, small + focused
- **Python best practices**: Type hints, docstrings, ruff clean
