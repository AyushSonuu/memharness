# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""SQL schema loader utility for memharness backends.

This module provides utilities for loading SQL schema files from the package
resources. It supports both SQLite and PostgreSQL backends.

Example:
    from memharness.sql.loader import load_schema

    # Load PostgreSQL schema
    schema_sql = load_schema('postgres')

    # Load a migration
    migration_sql = load_migration('postgres', '001_add_indexes')
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files


@cache
def load_schema(backend: str) -> str:
    """Load SQL schema for a backend.

    Args:
        backend: Backend name ('sqlite' or 'postgres')

    Returns:
        SQL schema content as a string

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If backend is not supported

    Example:
        >>> schema = load_schema('postgres')
        >>> print(schema[:50])
        -- memharness PostgreSQL + pgvector schema
    """
    if backend not in {"sqlite", "postgres"}:
        raise ValueError(f"Unsupported backend: {backend}. Must be 'sqlite' or 'postgres'")

    try:
        schema_file = files("memharness.sql").joinpath(f"{backend}/schema.sql")
        return schema_file.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Schema file not found for backend '{backend}'. "
            f"Expected: memharness/sql/{backend}/schema.sql"
        ) from e


@cache
def load_migration(backend: str, version: str) -> str:
    """Load a migration SQL file.

    Args:
        backend: Backend name ('sqlite' or 'postgres')
        version: Migration version identifier (e.g., '001_add_indexes')

    Returns:
        Migration SQL content as a string

    Raises:
        FileNotFoundError: If migration file doesn't exist
        ValueError: If backend is not supported

    Example:
        >>> migration = load_migration('postgres', '001_add_indexes')
    """
    if backend not in {"sqlite", "postgres"}:
        raise ValueError(f"Unsupported backend: {backend}. Must be 'sqlite' or 'postgres'")

    try:
        migration_file = files("memharness.sql").joinpath(f"{backend}/migrations/{version}.sql")
        return migration_file.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Migration file not found: {backend}/migrations/{version}.sql"
        ) from e


def clear_cache() -> None:
    """Clear the SQL loader cache.

    This is useful in testing scenarios where schema files might be modified.

    Example:
        >>> clear_cache()
    """
    load_schema.cache_clear()
    load_migration.cache_clear()
