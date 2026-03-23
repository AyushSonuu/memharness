# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Backend factory for memharness.

This module provides utilities for parsing backend URIs and instantiating
the appropriate backend implementation.

Example:
    from memharness.core.backend_factory import parse_backend

    # In-memory backend
    backend = parse_backend("memory://")

    # SQLite backend
    backend = parse_backend("sqlite:///memory.db")

    # PostgreSQL backend
    backend = parse_backend("postgresql://localhost/memdb")
"""

from __future__ import annotations

from memharness.backends.memory import InMemoryBackend
from memharness.backends.protocol import BackendProtocol

__all__ = ["parse_backend"]


def parse_backend(backend_uri: str) -> BackendProtocol:
    """
    Parse a backend URI and return the appropriate backend instance.

    Supported URIs:
        - "memory://" -> InMemoryBackend
        - "sqlite:///path/to/db.sqlite" -> SqliteBackend
        - "postgresql://user:pass@host:port/db" -> PostgresBackend

    Args:
        backend_uri: The backend connection string.

    Returns:
        An instance of the appropriate backend.

    Raises:
        ValueError: If the backend URI format is not recognized.
        ImportError: If required backend dependencies are not installed.
    """
    if backend_uri == "memory://" or backend_uri.startswith("memory://"):
        return InMemoryBackend()

    if backend_uri.startswith("sqlite:///"):
        # Extract path from sqlite:///path/to/db.sqlite
        db_path = backend_uri[10:]  # Remove "sqlite:///"
        try:
            from memharness.backends.sqlite import SqliteBackend

            return SqliteBackend(db_path)
        except ImportError as exc:
            msg = "SqliteBackend is not available. Ensure the sqlite backend module is installed."
            raise ImportError(msg) from exc

    if backend_uri.startswith("postgresql://") or backend_uri.startswith("postgres://"):
        try:
            from memharness.backends.postgres import PostgresBackend

            return PostgresBackend(backend_uri)
        except ImportError as exc:
            msg = "PostgresBackend is not available. Install with: pip install memharness[postgres]"
            raise ImportError(msg) from exc

    msg = (
        f"Unrecognized backend URI: {backend_uri}. "
        "Supported formats: memory://, sqlite:///path, postgresql://..."
    )
    raise ValueError(msg)
