# CLAUDE.md — memharness Development Guide

## Project Overview
memharness is a **framework-agnostic Python package** providing memory infrastructure for AI agents.
- **Author**: Ayush Sonuu <sonuayush55@gmail.com>
- **GitHub**: AyushSonuu/memharness (personal github.com account)
- **Python**: 3.13+ (use .venv in project root)
- **Build**: hatchling
- **License**: MIT

## Research & Design
Full research docs at ~/Desktop/my-memory/research/:
- 09-HLD-memharness.md — the main HLD/spec (START HERE for design questions)
- 00-project-brief.md through 12-ai-native-memory-agents.md — detailed research

## Commands
```bash
# Run all tests
.venv/bin/pytest tests/ -x --tb=short

# Run specific test file
.venv/bin/pytest tests/unit/test_types.py -v

# Lint
.venv/bin/ruff check src/ tests/ --fix

# Format
.venv/bin/ruff format src/ tests/

# Type check
.venv/bin/mypy src/memharness/

# Install in dev mode
.venv/bin/pip install -e ".[dev]"
```

## Architecture
```
src/memharness/
├── __init__.py          # Public API exports
├── harness.py           # MemoryHarness main class + InMemoryBackend + MemoryUnit (dataclass)
├── types.py             # Pydantic MemoryUnit (NOT used by harness — legacy)
├── registry.py          # MemoryTypeRegistry
├── config/
│   ├── models.py        # Pydantic config models
│   └── loader.py        # YAML/env config loading
├── backends/
│   ├── protocol.py      # BackendProtocol
│   ├── memory.py        # InMemoryBackend (separate module)
│   ├── sqlite.py        # SqliteBackend
│   └── postgres.py      # PostgresBackend
├── agents/              # Embedded AI agents (summarizer, consolidator, gc, etc.)
├── tools/               # Memory tools for agent self-exploration
└── integrations/        # LangChain, LangGraph adapters
```

## KEY ARCHITECTURAL NOTE
There are TWO MemoryUnit definitions:
1. `harness.py` — dataclass-based, used by the actual MemoryHarness class (THE CANONICAL ONE)
2. `types.py` — Pydantic-based, with different fields (namespace is str, has expires_at, score, source_ids)

Tests and the public API import from harness.py. The types.py Pydantic version is NOT actively used.

## Code Style
- Async-first API
- Type hints everywhere
- Docstrings on all public methods
- ruff for linting/formatting
- pytest-asyncio for async tests
