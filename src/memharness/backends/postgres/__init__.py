# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""PostgreSQL backend with pgvector support for memharness.

This module provides a production-ready PostgreSQL backend with:
- Async operations via asyncpg with connection pooling
- Vector similarity search using pgvector extension
- Separate table schemas for SQL and Vector memory types
- HNSW indexes for efficient vector search
- Full-text search capabilities via pg_trgm
- Hybrid search combining semantic and keyword matching

Example:
    backend = PostgresBackend("postgresql://user:pass@localhost/db")
    await backend.connect()

    # Write to knowledge base (vector search enabled)
    await backend.write(
        namespace=("kb",),
        key="doc_001",
        value={"content": "PostgreSQL is powerful", "source": "docs"},
        embedding=[0.1, 0.2, 0.3, ...]
    )

    # Search by embedding similarity
    results = await backend.search(
        namespace=("kb",),
        query="database",
        embedding=[0.1, 0.2, 0.3, ...],
        k=5
    )

    await backend.disconnect()
"""

from __future__ import annotations

from typing import Any

from memharness.backends.postgres.connection import (
    DEFAULT_VECTOR_DIM,
    PostgresBackendError,
    PostgresConnectionManager,
)
from memharness.backends.postgres.queries import PostgresQueryExecutor
from memharness.backends.postgres.schema import PostgresSchemaManager

__all__ = ["PostgresBackend", "PostgresBackendError", "DEFAULT_VECTOR_DIM"]


class PostgresBackend:
    """PostgreSQL storage backend with pgvector support.

    Features:
    - Async operations via asyncpg with connection pooling
    - SQL tables for conversational and tool_log (with indexes on thread_id, timestamp)
    - Vector tables for other memory types with HNSW indexes
    - Hybrid search combining vector similarity and keyword matching
    - JSONB metadata storage for flexible filtering
    - Automatic schema creation and migration

    The backend creates separate tables for each memory type:
    - conversational_memory: SQL-based with thread_id/timestamp indexes
    - tool_log_memory: SQL-based audit trail
    - knowledge_base_memory: Vector-based with HNSW index
    - entity_memory: Vector-based for named entities
    - workflow_memory: Vector-based for procedures
    - toolbox_memory: Vector-based with VFS path support
    - summary_memory: Vector-based compressed representations
    - file_memory: Hybrid vector + file metadata
    - persona_memory: Vector-based agent identity
    """

    def __init__(
        self,
        connection_string: str,
        *,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
        vector_dim: int = DEFAULT_VECTOR_DIM,
    ) -> None:
        """Initialize the PostgreSQL backend.

        Args:
            connection_string: PostgreSQL connection URL
                (e.g., "postgresql://user:pass@localhost:5432/dbname")
            min_pool_size: Minimum connections in pool (default: 5)
            max_pool_size: Maximum connections in pool (default: 20)
            vector_dim: Dimension of embedding vectors (default: 768)
        """
        # Composition: use separate components for different concerns
        self._conn_manager = PostgresConnectionManager(
            connection_string,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            vector_dim=vector_dim,
        )
        self._schema_manager = PostgresSchemaManager(self._conn_manager)
        self._query_executor = PostgresQueryExecutor(self._conn_manager)

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._conn_manager.is_connected

    async def connect(self) -> None:
        """Initialize connection pool and create database schema.

        Creates:
        - pgvector extension for vector operations
        - pg_trgm extension for text similarity
        - All memory type tables with appropriate schemas
        - Indexes for efficient querying (B-tree, GIN, HNSW)

        Raises:
            PostgresBackendError: If connection fails
        """
        await self._conn_manager.connect()
        await self._schema_manager.initialize_schema()

    async def disconnect(self) -> None:
        """Close the connection pool.

        Raises:
            PostgresBackendError: If disconnect fails
        """
        await self._conn_manager.disconnect()
        self._schema_manager.clear_initialized_tables()

    # =========================================================================
    # Delegate CRUD Operations to QueryExecutor
    # =========================================================================

    async def write(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: list[float] | None = None,
    ) -> str:
        """Write a memory to storage.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace
            value: Memory data as a dictionary
            embedding: Optional embedding vector for semantic search

        Returns:
            The key of the written memory

        Raises:
            PostgresBackendError: If write operation fails
        """
        return await self._query_executor.write(namespace, key, value, embedding)

    async def read(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> dict | None:
        """Read a single memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace

        Returns:
            Memory data dictionary if found, None otherwise

        Raises:
            PostgresBackendError: If read operation fails
        """
        return await self._query_executor.read(namespace, key)

    async def update(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: list[float] | None = None,
    ) -> bool:
        """Update an existing memory.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace
            value: New memory data (merged with existing)
            embedding: Optional new embedding vector

        Returns:
            True if memory was updated, False if not found

        Raises:
            PostgresBackendError: If update operation fails
        """
        return await self._query_executor.update(namespace, key, value, embedding)

    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        """Delete a memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace

        Returns:
            True if memory was deleted, False if not found

        Raises:
            PostgresBackendError: If delete operation fails
        """
        return await self._query_executor.delete(namespace, key)

    # =========================================================================
    # Delegate Search Operations to QueryExecutor
    # =========================================================================

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        embedding: list[float] | None = None,
        k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search memories using text query or embedding similarity.

        Args:
            namespace: Hierarchical namespace tuple
            query: Text query for search
            embedding: Optional embedding vector for semantic similarity search
            k: Maximum number of results to return (default: 10)
            filters: Optional filters

        Returns:
            List of matching memory dictionaries, ordered by relevance

        Raises:
            PostgresBackendError: If search operation fails
        """
        return await self._query_executor.search(namespace, query, embedding, k, filters)

    async def hybrid_search(
        self,
        namespace: tuple[str, ...],
        query: str,
        embedding: list[float],
        k: int = 10,
        filters: dict | None = None,
        vector_weight: float = 0.7,
    ) -> list[dict]:
        """Perform hybrid search combining vector similarity and keyword matching.

        Args:
            namespace: Hierarchical namespace tuple
            query: Text query for keyword search
            embedding: Embedding vector for semantic search
            k: Maximum number of results (default: 10)
            filters: Optional filters to apply
            vector_weight: Weight for vector similarity (0-1)

        Returns:
            List of matching memory dictionaries, ordered by combined score

        Raises:
            PostgresBackendError: If memory type doesn't support vector search
        """
        return await self._query_executor.hybrid_search(
            namespace, query, embedding, k, filters, vector_weight
        )

    async def list(
        self,
        namespace: tuple[str, ...],
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """List memories in a namespace with optional filtering.

        Args:
            namespace: Hierarchical namespace tuple
            filters: Optional filters
            order_by: Field to order by (prefix with - for descending)
            limit: Maximum number of results

        Returns:
            List of memory dictionaries

        Raises:
            PostgresBackendError: If list operation fails
        """
        return await self._query_executor.list(namespace, filters, order_by, limit)

    # =========================================================================
    # Delegate Utility Methods to QueryExecutor
    # =========================================================================

    async def count(
        self,
        namespace: tuple[str, ...],
        filters: dict | None = None,
    ) -> int:
        """Count memories in a namespace.

        Args:
            namespace: Hierarchical namespace tuple
            filters: Optional filters to apply

        Returns:
            Number of matching memories
        """
        return await self._query_executor.count(namespace, filters)

    async def delete_by_filter(
        self,
        namespace: tuple[str, ...],
        filters: dict,
    ) -> int:
        """Delete multiple memories matching filters.

        Args:
            namespace: Hierarchical namespace tuple
            filters: Filters to match records for deletion

        Returns:
            Number of deleted records

        Raises:
            PostgresBackendError: If no filters provided or delete fails
        """
        return await self._query_executor.delete_by_filter(namespace, filters)

    async def truncate(
        self,
        namespace: tuple[str, ...],
    ) -> None:
        """Truncate a memory table (delete all records).

        WARNING: This permanently deletes all data in the table.

        Args:
            namespace: Hierarchical namespace tuple
        """
        return await self._query_executor.truncate(namespace)

    async def health_check(self) -> dict[str, Any]:
        """Check backend health and return status information.

        Returns:
            Dictionary with health status and pool statistics
        """
        return await self._conn_manager.health_check()

    async def get_table_stats(self) -> dict[str, int]:
        """Get row counts for all memory tables.

        Returns:
            Dictionary mapping table names to row counts
        """
        return await self._query_executor.get_table_stats()
