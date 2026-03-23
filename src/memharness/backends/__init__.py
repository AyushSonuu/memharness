# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Storage backends for memharness.

This module provides backend implementations for persisting memory data:

- BackendProtocol: The interface all backends must implement
- SQLiteBackend: Production-ready async SQLite backend with vector search
- InMemoryBackend: Simple dict-based backend for testing
- PostgresBackend: PostgreSQL backend (optional)

Example:
    from memharness.backends import SQLiteBackend, InMemoryBackend

    # For production use with SQLite
    backend = SQLiteBackend("./memory.db")
    await backend.connect()

    # For testing
    test_backend = InMemoryBackend()
    await test_backend.connect()

    # Using the factory function
    backend = get_backend("sqlite:///memory.db")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from memharness.backends.memory import InMemoryBackend
from memharness.backends.protocol import BackendProtocol
from memharness.backends.sqlite import SQLiteBackend, SQLiteBackendError

if TYPE_CHECKING:
    pass

__all__ = [
    "BackendProtocol",
    "SQLiteBackend",
    "SQLiteBackendError",
    "InMemoryBackend",
    "get_backend",
]


def get_backend(connection_string: str) -> SQLiteBackend | InMemoryBackend:
    """Get the appropriate backend based on connection string.

    Args:
        connection_string: Database connection URL.

    Returns:
        Backend instance matching the connection string.

    Raises:
        ValueError: If the connection string format is not supported.

    Examples:
        >>> backend = get_backend("sqlite:///memory.db")  # SQLite file
        >>> backend = get_backend("sqlite:///:memory:")   # SQLite in-memory
        >>> backend = get_backend(":memory:")             # InMemoryBackend
        >>> backend = get_backend("postgresql://...")     # PostgreSQL
    """
    if connection_string.startswith("postgresql://") or connection_string.startswith("postgres://"):
        from memharness.backends.postgres import PostgresBackend
        return PostgresBackend(connection_string)

    elif connection_string.startswith("sqlite:///"):
        # Extract path from sqlite:///path format
        db_path = connection_string[10:]  # Remove "sqlite:///"
        if db_path == ":memory:":
            return SQLiteBackend(":memory:")
        return SQLiteBackend(db_path)

    elif connection_string.startswith("sqlite://"):
        # Handle sqlite://:memory: format
        db_path = connection_string[9:]  # Remove "sqlite://"
        return SQLiteBackend(db_path if db_path else ":memory:")

    elif connection_string == ":memory:":
        # Simple in-memory backend for testing
        return InMemoryBackend()

    else:
        raise ValueError(f"Unsupported connection string format: {connection_string}")


# Lazy imports for optional backends
def __getattr__(name: str):
    if name == "PostgresBackend":
        from memharness.backends.postgres import PostgresBackend
        return PostgresBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
