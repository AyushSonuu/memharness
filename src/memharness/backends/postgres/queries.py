# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""PostgreSQL query execution for memharness.

This module handles all database query operations including:
- CRUD operations (write, read, update, delete)
- Search operations (vector, text, hybrid)
- Filtering and listing
- Helper utilities
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from memharness.types import MemoryType

if TYPE_CHECKING:
    import asyncpg

    from memharness.backends.postgres.connection import (
        PostgresConnectionManager,
    )

logger = logging.getLogger(__name__)


class PostgresQueryExecutor:
    """Handles all query execution for PostgreSQL backend.

    This class provides:
    - CRUD operations
    - Vector and text search
    - Hybrid search combining semantic + keyword matching
    - Bulk operations and utilities
    """

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        """Initialize query executor.

        Args:
            conn_manager: Connection manager instance
        """
        self._conn_manager = conn_manager

    # =========================================================================
    # Namespace/Table Resolution
    # =========================================================================

    def _namespace_to_memory_type(self, namespace: tuple[str, ...]) -> MemoryType:
        """Resolve namespace tuple to MemoryType.

        The first element of the namespace determines the memory type.

        Args:
            namespace: Hierarchical namespace tuple (e.g., ("kb",), ("conv", "thread_1"))

        Returns:
            MemoryType enum value

        Raises:
            PostgresBackendError: If namespace cannot be resolved to a memory type
        """
        from memharness.backends.postgres.connection import PostgresBackendError

        if not namespace:
            raise PostgresBackendError("Namespace cannot be empty")

        type_mapping = {
            "conversational": MemoryType.CONVERSATIONAL,
            "conv": MemoryType.CONVERSATIONAL,
            "knowledge_base": MemoryType.KNOWLEDGE_BASE,
            "kb": MemoryType.KNOWLEDGE_BASE,
            "entity": MemoryType.ENTITY,
            "workflow": MemoryType.WORKFLOW,
            "toolbox": MemoryType.TOOLBOX,
            "summary": MemoryType.SUMMARY,
            "tool_log": MemoryType.TOOL_LOG,
            "skills": MemoryType.SKILLS,
            "file": MemoryType.FILE,
            "persona": MemoryType.PERSONA,
        }

        type_key = namespace[0].lower()
        if type_key not in type_mapping:
            raise PostgresBackendError(
                f"Unknown memory type in namespace: {type_key}. "
                f"Valid types: {list(type_mapping.keys())}"
            )

        return type_mapping[type_key]

    def _get_table_name(self, namespace: tuple[str, ...]) -> str:
        """Get table name for a namespace.

        Args:
            namespace: Hierarchical namespace tuple

        Returns:
            Table name string
        """
        memory_type = self._namespace_to_memory_type(namespace)
        return memory_type.table_name

    # =========================================================================
    # CRUD Operations
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
        from memharness.backends.postgres.connection import PostgresBackendError

        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        try:
            columns, values, placeholders = self._build_insert_params(
                memory_type, namespace, key, value, embedding
            )

            query = f"""
                INSERT INTO {table_name} ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
                ON CONFLICT (key) DO UPDATE SET
                    {", ".join(f"{col} = EXCLUDED.{col}" for col in columns if col != "key")},
                    updated_at = NOW()
                RETURNING id
            """

            async with self._conn_manager.pool.acquire() as conn:
                row = await conn.fetchrow(query, *values)
                logger.debug(f"Wrote memory {key} to {table_name} with id {row['id']}")
                return key

        except Exception as e:
            raise PostgresBackendError(f"Failed to write memory: {e}") from e

    def _build_insert_params(
        self,
        memory_type: MemoryType,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: list[float] | None = None,
    ) -> tuple[list[str], list[Any], list[str]]:
        """Build INSERT parameters based on memory type.

        Args:
            memory_type: The type of memory
            namespace: Namespace tuple
            key: Memory key
            value: Memory value dict
            embedding: Optional embedding vector

        Returns:
            Tuple of (columns, values, placeholders)
        """
        # Base columns present in all tables
        columns = ["key", "namespace", "content", "metadata"]
        content = value.get("content", "")
        metadata = {
            k: v for k, v in value.items() if k not in self._get_reserved_fields(memory_type)
        }

        values: list[Any] = [
            key,
            list(namespace),
            content,
            json.dumps(metadata),
        ]
        placeholders = ["$1", "$2", "$3", "$4"]
        param_idx = 5

        # Add type-specific fields
        type_fields = self._extract_type_fields(memory_type, value)
        for col, val in type_fields.items():
            columns.append(col)
            values.append(val)
            placeholders.append(f"${param_idx}")
            param_idx += 1

        # Add embedding for vector types
        if memory_type.uses_vector and embedding:
            columns.append("embedding")
            values.append(embedding)
            placeholders.append(f"${param_idx}")

        return columns, values, placeholders

    def _get_reserved_fields(self, memory_type: MemoryType) -> set[str]:
        """Get fields that are stored in dedicated columns, not in metadata."""
        base_fields = {"content", "id", "key", "namespace", "created_at", "updated_at", "embedding"}

        type_specific = {
            MemoryType.CONVERSATIONAL: {"thread_id", "role", "summary_id"},
            MemoryType.TOOL_LOG: {"thread_id", "tool_name"},
            MemoryType.SUMMARY: {"thread_id", "start_time", "end_time", "message_count"},
            MemoryType.KNOWLEDGE_BASE: {"source"},
            MemoryType.ENTITY: {"entity_type"},
            MemoryType.TOOLBOX: {"tool_name", "vfs_path"},
            MemoryType.SKILLS: {"skill_name"},
            MemoryType.FILE: {"source", "file_path", "file_hash"},
            MemoryType.PERSONA: {"persona_name"},
            MemoryType.WORKFLOW: set(),
        }

        return base_fields | type_specific.get(memory_type, set())

    def _extract_type_fields(
        self,
        memory_type: MemoryType,
        value: dict,
    ) -> dict[str, Any]:
        """Extract type-specific fields from value dict."""
        fields: dict[str, Any] = {}

        match memory_type:
            case MemoryType.CONVERSATIONAL:
                if "thread_id" in value:
                    fields["thread_id"] = value["thread_id"]
                if "role" in value:
                    fields["role"] = value["role"]
                if "summary_id" in value:
                    fields["summary_id"] = value["summary_id"]
            case MemoryType.TOOL_LOG:
                if "thread_id" in value:
                    fields["thread_id"] = value["thread_id"]
                if "tool_name" in value:
                    fields["tool_name"] = value["tool_name"]
            case MemoryType.SUMMARY:
                if "thread_id" in value:
                    fields["thread_id"] = value["thread_id"]
                if "start_time" in value:
                    fields["start_time"] = value["start_time"]
                if "end_time" in value:
                    fields["end_time"] = value["end_time"]
                if "message_count" in value:
                    fields["message_count"] = value["message_count"]
            case MemoryType.KNOWLEDGE_BASE:
                if "source" in value:
                    fields["source"] = value["source"]
            case MemoryType.ENTITY:
                if "entity_type" in value:
                    fields["entity_type"] = value["entity_type"]
            case MemoryType.TOOLBOX:
                if "tool_name" in value:
                    fields["tool_name"] = value["tool_name"]
                if "vfs_path" in value:
                    fields["vfs_path"] = value["vfs_path"]
            case MemoryType.SKILLS:
                if "skill_name" in value:
                    fields["skill_name"] = value["skill_name"]
            case MemoryType.FILE:
                if "source" in value:
                    fields["source"] = value["source"]
                if "file_path" in value:
                    fields["file_path"] = value["file_path"]
                if "file_hash" in value:
                    fields["file_hash"] = value["file_hash"]
            case MemoryType.PERSONA:
                if "persona_name" in value:
                    fields["persona_name"] = value["persona_name"]

        return fields

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
        from memharness.backends.postgres.connection import PostgresBackendError

        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        try:
            async with self._conn_manager.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT * FROM {table_name} WHERE key = $1",
                    key,
                )

                if row is None:
                    return None

                return self._row_to_dict(memory_type, row)

        except Exception as e:
            raise PostgresBackendError(f"Failed to read memory: {e}") from e

    def _row_to_dict(
        self,
        memory_type: MemoryType,
        row: asyncpg.Record,
    ) -> dict:
        """Convert a database row to a memory dictionary."""
        # Start with metadata
        result = dict(row.get("metadata", {})) if row.get("metadata") else {}

        # Add standard fields
        result["id"] = str(row["id"])
        result["key"] = row["key"]
        result["namespace"] = list(row["namespace"]) if row["namespace"] else []
        result["content"] = row["content"]
        result["created_at"] = row["created_at"].isoformat() if row.get("created_at") else None
        result["updated_at"] = row["updated_at"].isoformat() if row.get("updated_at") else None

        # Add embedding if present
        if "embedding" in row.keys() and row["embedding"] is not None:
            result["embedding"] = list(row["embedding"])

        # Add type-specific fields
        match memory_type:
            case MemoryType.CONVERSATIONAL:
                if row.get("thread_id"):
                    result["thread_id"] = row["thread_id"]
                if row.get("role"):
                    result["role"] = row["role"]
                if row.get("summary_id"):
                    result["summary_id"] = str(row["summary_id"])
            case MemoryType.TOOL_LOG:
                if row.get("thread_id"):
                    result["thread_id"] = row["thread_id"]
                if row.get("tool_name"):
                    result["tool_name"] = row["tool_name"]
            case MemoryType.SUMMARY:
                if row.get("thread_id"):
                    result["thread_id"] = row["thread_id"]
                if row.get("start_time"):
                    result["start_time"] = row["start_time"].isoformat()
                if row.get("end_time"):
                    result["end_time"] = row["end_time"].isoformat()
                if row.get("message_count"):
                    result["message_count"] = row["message_count"]
            case MemoryType.KNOWLEDGE_BASE:
                if row.get("source"):
                    result["source"] = row["source"]
            case MemoryType.ENTITY:
                if row.get("entity_type"):
                    result["entity_type"] = row["entity_type"]
            case MemoryType.TOOLBOX:
                if row.get("tool_name"):
                    result["tool_name"] = row["tool_name"]
                if row.get("vfs_path"):
                    result["vfs_path"] = row["vfs_path"]
            case MemoryType.SKILLS:
                if row.get("skill_name"):
                    result["skill_name"] = row["skill_name"]
            case MemoryType.FILE:
                if row.get("source"):
                    result["source"] = row["source"]
                if row.get("file_path"):
                    result["file_path"] = row["file_path"]
                if row.get("file_hash"):
                    result["file_hash"] = row["file_hash"]
            case MemoryType.PERSONA:
                if row.get("persona_name"):
                    result["persona_name"] = row["persona_name"]

        return result

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
        from memharness.backends.postgres.connection import PostgresBackendError

        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        try:
            # First, read existing value to merge
            existing = await self.read(namespace, key)
            if existing is None:
                return False

            # Merge values (new values override existing)
            merged = {**existing, **value}

            # Build SET clause
            set_parts = []
            params = []
            param_idx = 1

            # Update content if provided
            if "content" in value:
                set_parts.append(f"content = ${param_idx}")
                params.append(value["content"])
                param_idx += 1

            # Update metadata (everything not in reserved fields)
            reserved = self._get_reserved_fields(memory_type)
            metadata = {k: v for k, v in merged.items() if k not in reserved}
            set_parts.append(f"metadata = ${param_idx}")
            params.append(json.dumps(metadata))
            param_idx += 1

            # Update type-specific fields
            type_fields = self._extract_type_fields(memory_type, value)
            for field, val in type_fields.items():
                set_parts.append(f"{field} = ${param_idx}")
                params.append(val)
                param_idx += 1

            # Update embedding if provided
            if embedding:
                set_parts.append(f"embedding = ${param_idx}")
                params.append(embedding)
                param_idx += 1

            # Always update timestamp
            set_parts.append(f"updated_at = ${param_idx}")
            params.append(datetime.utcnow())
            param_idx += 1

            # Add key for WHERE clause
            params.append(key)

            sql = f"""
                UPDATE {table_name}
                SET {", ".join(set_parts)}
                WHERE key = ${param_idx}
            """

            async with self._conn_manager.pool.acquire() as conn:
                result = await conn.execute(sql, *params)
                return result == "UPDATE 1"

        except Exception as e:
            raise PostgresBackendError(f"Failed to update memory: {e}") from e

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
        from memharness.backends.postgres.connection import PostgresBackendError

        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        try:
            async with self._conn_manager.pool.acquire() as conn:
                result = await conn.execute(
                    f"DELETE FROM {table_name} WHERE key = $1",
                    key,
                )
                return result == "DELETE 1"

        except Exception as e:
            raise PostgresBackendError(f"Failed to delete memory: {e}") from e

    # =========================================================================
    # Search Operations
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

        For vector-enabled memory types:
        - If embedding is provided, uses pgvector's <=> operator for cosine distance
        - Falls back to text similarity using pg_trgm if no embedding

        For SQL-only types:
        - Uses LIKE-based text search on content

        Args:
            namespace: Hierarchical namespace tuple
            query: Text query for search
            embedding: Optional embedding vector for semantic similarity search
            k: Maximum number of results to return (default: 10)
            filters: Optional filters (e.g., {"thread_id": "t1", "source": "docs"})

        Returns:
            List of matching memory dictionaries, ordered by relevance

        Raises:
            PostgresBackendError: If search operation fails
        """
        from memharness.backends.postgres.connection import PostgresBackendError

        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        try:
            # Build filter conditions
            where_clauses, params = self._build_filter_clauses(memory_type, namespace, filters)

            # Vector search for vector-enabled types
            if memory_type.uses_vector and embedding:
                return await self._vector_search(
                    table_name, memory_type, embedding, k, where_clauses, params
                )

            # Text-based search fallback
            return await self._text_search(table_name, memory_type, query, k, where_clauses, params)

        except Exception as e:
            raise PostgresBackendError(f"Failed to search memories: {e}") from e

    async def _vector_search(
        self,
        table_name: str,
        memory_type: MemoryType,
        embedding: list[float],
        k: int,
        where_clauses: list[str],
        params: list[Any],
    ) -> list[dict]:
        """Perform vector similarity search using pgvector."""
        # Embedding is the first parameter
        params.insert(0, embedding)

        # Adjust parameter indices in where clauses
        adjusted_clauses = []
        for clause in where_clauses:
            adjusted = clause
            for i in range(len(params), 0, -1):
                adjusted = adjusted.replace(f"${i}", f"${i + 1}")
            adjusted_clauses.append(adjusted)

        query = f"""
            SELECT *, (embedding <=> $1) AS distance
            FROM {table_name}
            WHERE embedding IS NOT NULL
            {f"AND {' AND '.join(adjusted_clauses)}" if adjusted_clauses else ""}
            ORDER BY distance
            LIMIT {k}
        """

        async with self._conn_manager.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                result = self._row_to_dict(memory_type, row)
                result["_score"] = 1 - float(row["distance"])  # Convert distance to similarity
                result["_distance"] = float(row["distance"])
                results.append(result)

            return results

    async def _text_search(
        self,
        table_name: str,
        memory_type: MemoryType,
        query: str,
        k: int,
        where_clauses: list[str],
        params: list[Any],
    ) -> list[dict]:
        """Perform text-based search using LIKE or pg_trgm similarity."""
        if query:
            param_idx = len(params) + 1

            if memory_type.uses_vector:
                # Use pg_trgm similarity for vector-enabled tables
                where_clauses.append(f"similarity(content, ${param_idx}) > 0.1")
                params.append(query)
                order_clause = f"ORDER BY similarity(content, ${param_idx}) DESC"
            else:
                # Use LIKE for SQL-only tables
                where_clauses.append(f"content ILIKE ${param_idx}")
                params.append(f"%{query}%")
                order_clause = "ORDER BY created_at DESC"
        else:
            order_clause = "ORDER BY created_at DESC"

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT * FROM {table_name}
            {where_sql}
            {order_clause}
            LIMIT {k}
        """

        async with self._conn_manager.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_dict(memory_type, row) for row in rows]

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

        Uses pgvector for semantic similarity and pg_trgm for keyword matching,
        then combines scores with configurable weighting.

        Args:
            namespace: Hierarchical namespace tuple
            query: Text query for keyword search
            embedding: Embedding vector for semantic search
            k: Maximum number of results (default: 10)
            filters: Optional filters to apply
            vector_weight: Weight for vector similarity (0-1), keyword = 1 - vector_weight

        Returns:
            List of matching memory dictionaries, ordered by combined score

        Raises:
            PostgresBackendError: If memory type doesn't support vector search
        """
        from memharness.backends.postgres.connection import PostgresBackendError

        memory_type = self._namespace_to_memory_type(namespace)

        if not memory_type.uses_vector:
            raise PostgresBackendError(
                f"Memory type {memory_type.value} does not support hybrid search"
            )

        table_name = memory_type.table_name
        keyword_weight = 1 - vector_weight

        try:
            # Build filter conditions
            where_clauses, filter_params = self._build_filter_clauses(
                memory_type, namespace, filters
            )

            # Parameters: $1 = embedding, $2 = keyword_query
            params = [embedding, query] + filter_params

            # Adjust filter parameter indices
            adjusted_clauses = []
            for clause in where_clauses:
                adjusted = clause
                for i in range(len(filter_params), 0, -1):
                    adjusted = adjusted.replace(f"${i}", f"${i + 2}")
                adjusted_clauses.append(adjusted)

            filter_sql = f"AND {' AND '.join(adjusted_clauses)}" if adjusted_clauses else ""

            sql = f"""
                WITH scored AS (
                    SELECT *,
                        (1 - (embedding <=> $1)) AS vector_score,
                        COALESCE(similarity(content, $2), 0) AS keyword_score
                    FROM {table_name}
                    WHERE embedding IS NOT NULL
                    {filter_sql}
                )
                SELECT *,
                    (vector_score * {vector_weight} + keyword_score * {keyword_weight}) AS combined_score
                FROM scored
                WHERE keyword_score > 0.1 OR vector_score > 0.5
                ORDER BY combined_score DESC
                LIMIT {k}
            """

            async with self._conn_manager.pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)

                results = []
                for row in rows:
                    result = self._row_to_dict(memory_type, row)
                    result["_score"] = float(row["combined_score"])
                    result["_vector_score"] = float(row["vector_score"])
                    result["_keyword_score"] = float(row["keyword_score"])
                    results.append(result)

                return results

        except Exception as e:
            raise PostgresBackendError(f"Failed to perform hybrid search: {e}") from e

    # =========================================================================
    # List and Filter Operations
    # =========================================================================

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
            filters: Optional filters (e.g., {"thread_id": "t1"})
            order_by: Field to order by (prefix with - for descending, e.g., "-created_at")
            limit: Maximum number of results

        Returns:
            List of memory dictionaries

        Raises:
            PostgresBackendError: If list operation fails
        """
        from memharness.backends.postgres.connection import PostgresBackendError

        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        try:
            where_clauses, params = self._build_filter_clauses(memory_type, namespace, filters)
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            # Build ORDER BY clause
            if order_by:
                descending = order_by.startswith("-")
                field = order_by[1:] if descending else order_by
                direction = "DESC" if descending else "ASC"
                order_sql = f"ORDER BY {field} {direction}"
            else:
                order_sql = "ORDER BY created_at DESC"

            # Build LIMIT clause
            limit_sql = f"LIMIT {limit}" if limit else ""

            sql = f"""
                SELECT * FROM {table_name}
                {where_sql}
                {order_sql}
                {limit_sql}
            """

            async with self._conn_manager.pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
                return [self._row_to_dict(memory_type, row) for row in rows]

        except Exception as e:
            raise PostgresBackendError(f"Failed to list memories: {e}") from e

    def _build_filter_clauses(
        self,
        memory_type: MemoryType,
        namespace: tuple[str, ...],
        filters: dict | None,
    ) -> tuple[list[str], list[Any]]:
        """Build WHERE clauses from filters.

        Args:
            memory_type: The memory type
            namespace: Namespace tuple
            filters: Optional filter dict

        Returns:
            Tuple of (where_clauses, params)
        """
        clauses = []
        params: list[Any] = []
        param_idx = 1

        # Always filter by namespace (using array containment)
        clauses.append(f"namespace @> ${param_idx}")
        params.append(list(namespace))
        param_idx += 1

        if not filters:
            return clauses, params

        # Handle common filters
        if "thread_id" in filters and memory_type in {
            MemoryType.CONVERSATIONAL,
            MemoryType.TOOL_LOG,
            MemoryType.SUMMARY,
        }:
            clauses.append(f"thread_id = ${param_idx}")
            params.append(filters["thread_id"])
            param_idx += 1

        if "source" in filters and memory_type in {
            MemoryType.KNOWLEDGE_BASE,
            MemoryType.FILE,
        }:
            clauses.append(f"source = ${param_idx}")
            params.append(filters["source"])
            param_idx += 1

        if "entity_type" in filters and memory_type == MemoryType.ENTITY:
            clauses.append(f"entity_type = ${param_idx}")
            params.append(filters["entity_type"])
            param_idx += 1

        if "tool_name" in filters and memory_type in {
            MemoryType.TOOLBOX,
            MemoryType.TOOL_LOG,
        }:
            clauses.append(f"tool_name = ${param_idx}")
            params.append(filters["tool_name"])
            param_idx += 1

        if "skill_name" in filters and memory_type == MemoryType.SKILLS:
            clauses.append(f"skill_name = ${param_idx}")
            params.append(filters["skill_name"])
            param_idx += 1

        if "persona_name" in filters and memory_type == MemoryType.PERSONA:
            clauses.append(f"persona_name = ${param_idx}")
            params.append(filters["persona_name"])
            param_idx += 1

        if "file_path" in filters and memory_type == MemoryType.FILE:
            clauses.append(f"file_path = ${param_idx}")
            params.append(filters["file_path"])
            param_idx += 1

        if "vfs_path" in filters and memory_type == MemoryType.TOOLBOX:
            clauses.append(f"vfs_path = ${param_idx}")
            params.append(filters["vfs_path"])
            param_idx += 1

        # Date range filters
        if "created_after" in filters:
            clauses.append(f"created_at >= ${param_idx}")
            params.append(filters["created_after"])
            param_idx += 1

        if "created_before" in filters:
            clauses.append(f"created_at <= ${param_idx}")
            params.append(filters["created_before"])
            param_idx += 1

        # JSONB metadata filters
        if "metadata" in filters and isinstance(filters["metadata"], dict):
            for meta_key, meta_value in filters["metadata"].items():
                clauses.append(f"metadata->>'{meta_key}' = ${param_idx}")
                params.append(str(meta_value))
                param_idx += 1

        return clauses, params

    # =========================================================================
    # Utility Methods
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
        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        where_clauses, params = self._build_filter_clauses(memory_type, namespace, filters)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        async with self._conn_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM {table_name} {where_sql}",
                *params,
            )
            return row["count"]

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
        from memharness.backends.postgres.connection import PostgresBackendError

        if not filters:
            raise PostgresBackendError("Cannot delete without filters")

        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        where_clauses, params = self._build_filter_clauses(memory_type, namespace, filters)
        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        async with self._conn_manager.pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {table_name} {where_sql}",
                *params,
            )
            # Parse "DELETE N" response
            count = int(result.split()[1])
            return count

    async def truncate(
        self,
        namespace: tuple[str, ...],
    ) -> None:
        """Truncate a memory table (delete all records).

        WARNING: This permanently deletes all data in the table.

        Args:
            namespace: Hierarchical namespace tuple
        """
        memory_type = self._namespace_to_memory_type(namespace)
        table_name = memory_type.table_name

        async with self._conn_manager.pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {table_name} CASCADE")

    async def get_table_stats(self) -> dict[str, int]:
        """Get row counts for all memory tables.

        Returns:
            Dictionary mapping table names to row counts
        """
        stats = {}

        for memory_type in MemoryType:
            table_name = memory_type.table_name
            try:
                async with self._conn_manager.pool.acquire() as conn:
                    row = await conn.fetchrow(f"SELECT COUNT(*) FROM {table_name}")
                    stats[table_name] = row["count"]
            except Exception:
                stats[table_name] = -1  # Table might not exist

        return stats
