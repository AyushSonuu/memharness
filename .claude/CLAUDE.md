# CLAUDE.md — memharness Development Guide

## Project Overview
memharness is a **framework-agnostic Python package** providing memory infrastructure for AI agents.
- **Author**: Ayush Sonuu <sonuayush55@gmail.com>
- **GitHub**: AyushSonuu/memharness (personal github.com account)
- **PyPI**: https://pypi.org/project/memharness/
- **Docs**: https://ayushsonuu.github.io/memharness/
- **Python**: 3.13+ (use .venv in project root)
- **Build**: hatchling
- **License**: MIT

## Commands
```bash
.venv/bin/pytest tests/ -x --tb=short      # Run tests
.venv/bin/ruff check src/ tests/            # Lint
.venv/bin/ruff format src/ tests/           # Format
.venv/bin/python -m build                   # Build package
docker compose up -d                        # Start postgres+pgvector
```

## Architecture (v0.3.0)
```
src/memharness/
├── types.py                 # MemoryType enum, MemoryUnit dataclass
├── exceptions.py            # Custom exceptions
├── core/
│   ├── harness.py           # MemoryHarness (main class, delegates to memory_types/)
│   ├── backend_factory.py   # _parse_backend()
│   ├── config.py            # MemharnessConfig dataclass
│   ├── context.py           # Context assembly
│   └── embedding.py         # Default hash-based embedding
├── memory_types/            # One module per memory type (10 types)
├── backends/                # InMemory, SQLite, Postgres
├── tools/                   # LangChain BaseTool subclasses
├── integrations/            # LangChain, LangGraph adapters
├── agents/                  # Memory agents (summarizer, consolidator, etc.)
├── config/                  # Pydantic config models + YAML loader
└── registry.py              # Memory type registry
```

## Key Installed Packages (in .venv)
- langchain 1.2.13 — `from langchain.agents import create_agent`
- langchain-core 1.2.20 — `BaseTool`, `BaseChatMessageHistory`, `ChatPromptTemplate`
- pydantic 2.12.5
- aiosqlite 0.22.1
- numpy 2.4.3

## LangChain API Notes (IMPORTANT — explore before coding!)
- `from langchain_core.tools import BaseTool, tool` — for tool definitions
- `from langchain_core.chat_history import BaseChatMessageHistory` — NOT BaseMemory
- `from langchain_core.prompts import ChatPromptTemplate` — for prompts
- `from langchain_core.messages import HumanMessage, AIMessage, SystemMessage`
- `from langchain.agents import create_agent` — NOT create_react_agent
- BaseMemory is GONE from langchain-core (moved to langchain.memory or deprecated)
- Always explore .venv packages before writing code: `.venv/bin/python -c "from X import Y; help(Y)"`

## Code Quality Rules
- No file > 300 lines
- Functions < 50 lines
- Google-style docstrings
- Type hints on everything
- `from __future__ import annotations` in all files
- ruff clean (check + format)
- Test after every change
