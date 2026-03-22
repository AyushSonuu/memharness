# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""SQLite backend with async support and vector similarity search."""

from typing import Optional, Any
import json
import math
import logging
from pathlib import Path

import aiosqlite


logger = logging.getLogger(__name__)


class SQLiteBackendError(Exception):
    """Exception raised for SQLite backend errors."""
    pass


class SQLiteBackend:
    """SQLite storage backend with async support.

    Features:
    - Async operations via aiosqlite
    - Vector similarity search using cosine similarity
    - Automatic table creation per namespace
    - B-tree indexing for efficient queries
    - JSON storage for flexible memory schemas

    Example:
        backend = SQLiteBackend("./memory.db")
        await backend.connect()
        await backend.write(("kb",), "key1", {"content": "hello"}, embedding=[0.1, 0.2])
        results = await backend.search(("kb",), "hello", embedding=[0.1, 0.2], k=5)
        await backend.disconnect()
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self._db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized_tables: set[str] = set()

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._connection is not None

    async def connect(self) -> None:
        """Initialize database connection and create base schema.

        Creates the database file if it doesn't exist (for file-based DBs).
        """
        if self._connection is not None:
            return

        try:
            # Ensure parent directory exists for file-based databases
            if self._db_path != ":memory:":
                db_file = Path(self._db_path)
                db_file.parent.mkdir(parents=True, exist_ok=True)

            self._connection = await aiosqlite.connect(self._db_path)

            # Enable WAL mode for better concurrent performance
            await self._connection.execute("PRAGMA journal_mode=WAL")

            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys=ON")

            await self._connection.commit()

            logger.info(f"Connected to SQLite database: {self._db_path}")

        except Exception as e:
            raise SQLiteBackendError(f"Failed to connect to database: {e}") from e

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
            finally:
                self._connection = None
                self._initialized_tables.clear()

    def _namespace_to_table(self, namespace: tuple[str, ...]) -> str:
        """Convert namespace tuple to valid table name.

        Args:
            namespace: Hierarchical namespace tuple

        Returns:
            Valid SQLite table name
        """
        # Join with double underscore and sanitize
        name = "__".join(namespace)
        # Replace any non-alphanumeric chars (except underscore) with underscore
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        return f"mem_{safe_name}"

    async def _ensure_table(self, namespace: tuple[str, ...]) -> str:
        """Ensure table exists for namespace.

        Args:
            namespace: Hierarchical namespace tuple

        Returns:
            Table name
        """
        if self._connection is None:
            raise SQLiteBackendError("Not connected to database")

        table_name = self._namespace_to_table(namespace)

        if table_name in self._initialized_tables:
            return table_name

        try:
            # Create main table with JSON value storage
            await self._connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    embedding TEXT,
                    thread_id TEXT,
                    timestamp TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Create indexes for common query patterns
            # B-tree index on thread_id for conversation filtering
            await self._connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_thread_id
                ON {table_name}(thread_id)
            """)

            # B-tree index on timestamp for ordering
            await self._connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp
                ON {table_name}(timestamp)
            """)

            # Index on created_at for chronological queries
            await self._connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at
                ON {table_name}(created_at)
            """)

            await self._connection.commit()
            self._initialized_tables.add(table_name)

            logger.debug(f"Initialized table: {table_name}")

        except Exception as e:
            raise SQLiteBackendError(f"Failed to create table {table_name}: {e}") from e

        return table_name

    async def write(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: Optional[list[float]] = None
    ) -> str:
        """Write a memory to storage.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within namespace
            value: Memory data
            embedding: Optional embedding vector

        Returns:
            The key of the written memory
        """
        table_name = await self._ensure_table(namespace)

        try:
            # Extract indexed fields from value
            thread_id = value.get("thread_id")
            timestamp = value.get("timestamp")

            # Serialize value and embedding to JSON
            value_json = json.dumps(value)
            embedding_json = json.dumps(embedding) if embedding else None

            await self._connection.execute(f"""
                INSERT OR REPLACE INTO {table_name}
                (key, value, embedding, thread_id, timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (key, value_json, embedding_json, thread_id, timestamp))

            await self._connection.commit()

            return key

        except Exception as e:
            raise SQLiteBackendError(f"Failed to write memory: {e}") from e

    async def read(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> Optional[dict]:
        """Read a single memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier

        Returns:
            Memory data if found, None otherwise
        """
        table_name = self._namespace_to_table(namespace)

        # Check if table exists
        if table_name not in self._initialized_tables:
            # Try to check if table exists in database
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not await cursor.fetchone():
                return None
            self._initialized_tables.add(table_name)

        try:
            cursor = await self._connection.execute(
                f"SELECT value FROM {table_name} WHERE key = ?",
                (key,)
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return json.loads(row[0])

        except Exception as e:
            raise SQLiteBackendError(f"Failed to read memory: {e}") from e

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        embedding: Optional[list[float]] = None,
        k: int = 10,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """Search memories using text or embedding similarity.

        If embedding is provided, uses cosine similarity.
        Otherwise, uses LIKE-based text search on the JSON value.

        Args:
            namespace: Hierarchical namespace tuple
            query: Text query
            embedding: Optional embedding vector
            k: Maximum results
            filters: Optional filters

        Returns:
            List of matching memories ordered by relevance
        """
        table_name = self._namespace_to_table(namespace)

        # Check if table exists
        if table_name not in self._initialized_tables:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not await cursor.fetchone():
                return []
            self._initialized_tables.add(table_name)

        try:
            # Build filter conditions
            conditions = []
            params: list[Any] = []

            if filters:
                for filter_key, filter_value in filters.items():
                    # Use JSON extraction for filtering
                    conditions.append(f"json_extract(value, '$.{filter_key}') = ?")
                    params.append(filter_value)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # If embedding provided, fetch all and calculate similarity in Python
            if embedding is not None:
                cursor = await self._connection.execute(
                    f"SELECT key, value, embedding FROM {table_name} WHERE {where_clause}",
                    params
                )
                rows = await cursor.fetchall()

                # Calculate similarity scores
                scored = []
                for row in rows:
                    key, value_json, embedding_json = row
                    value = json.loads(value_json)

                    if embedding_json:
                        stored_embedding = json.loads(embedding_json)
                        score = self._cosine_similarity(embedding, stored_embedding)
                    else:
                        score = 0.0

                    scored.append((score, value))

                # Sort by score descending
                scored.sort(key=lambda x: x[0], reverse=True)
                return [item[1] for item in scored[:k]]

            # Text-based search using LIKE
            if query:
                conditions.append("value LIKE ?")
                params.append(f"%{query}%")
                where_clause = " AND ".join(conditions) if conditions else "1=1"

            cursor = await self._connection.execute(
                f"SELECT value FROM {table_name} WHERE {where_clause} LIMIT ?",
                params + [k]
            )
            rows = await cursor.fetchall()

            return [json.loads(row[0]) for row in rows]

        except Exception as e:
            raise SQLiteBackendError(f"Failed to search memories: {e}") from e

    async def list(
        self,
        namespace: tuple[str, ...],
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[dict]:
        """List memories with optional filtering and ordering.

        Args:
            namespace: Hierarchical namespace tuple
            filters: Optional filters
            order_by: Field to order by (prefix with - for descending)
            limit: Maximum results

        Returns:
            List of memory dictionaries
        """
        table_name = self._namespace_to_table(namespace)

        # Check if table exists
        if table_name not in self._initialized_tables:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not await cursor.fetchone():
                return []
            self._initialized_tables.add(table_name)

        try:
            # Build query
            conditions = []
            params: list[Any] = []

            if filters:
                for filter_key, filter_value in filters.items():
                    # Check if it's a top-level indexed column
                    if filter_key in ("thread_id", "timestamp"):
                        conditions.append(f"{filter_key} = ?")
                    else:
                        conditions.append(f"json_extract(value, '$.{filter_key}') = ?")
                    params.append(filter_value)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Build ORDER BY clause
            order_clause = ""
            if order_by:
                descending = order_by.startswith("-")
                field = order_by[1:] if descending else order_by
                direction = "DESC" if descending else "ASC"

                # Check if it's a top-level indexed column
                if field in ("thread_id", "timestamp", "created_at", "updated_at"):
                    order_clause = f"ORDER BY {field} {direction}"
                else:
                    order_clause = f"ORDER BY json_extract(value, '$.{field}') {direction}"

            # Build LIMIT clause
            limit_clause = f"LIMIT {limit}" if limit else ""

            query = f"""
                SELECT value FROM {table_name}
                WHERE {where_clause}
                {order_clause}
                {limit_clause}
            """

            cursor = await self._connection.execute(query, params)
            rows = await cursor.fetchall()

            return [json.loads(row[0]) for row in rows]

        except Exception as e:
            raise SQLiteBackendError(f"Failed to list memories: {e}") from e

    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> bool:
        """Delete a memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        table_name = self._namespace_to_table(namespace)

        # Check if table exists
        if table_name not in self._initialized_tables:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not await cursor.fetchone():
                return False
            self._initialized_tables.add(table_name)

        try:
            cursor = await self._connection.execute(
                f"DELETE FROM {table_name} WHERE key = ?",
                (key,)
            )
            await self._connection.commit()

            return cursor.rowcount > 0

        except Exception as e:
            raise SQLiteBackendError(f"Failed to delete memory: {e}") from e

    async def update(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: Optional[list[float]] = None
    ) -> bool:
        """Update an existing memory.

        Merges new value with existing data.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier
            value: New data to merge
            embedding: Optional new embedding

        Returns:
            True if updated, False if not found
        """
        table_name = self._namespace_to_table(namespace)

        # Check if table exists
        if table_name not in self._initialized_tables:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not await cursor.fetchone():
                return False
            self._initialized_tables.add(table_name)

        try:
            # Read existing value
            cursor = await self._connection.execute(
                f"SELECT value FROM {table_name} WHERE key = ?",
                (key,)
            )
            row = await cursor.fetchone()

            if row is None:
                return False

            # Merge values
            existing = json.loads(row[0])
            existing.update(value)

            # Extract indexed fields
            thread_id = existing.get("thread_id")
            timestamp = existing.get("timestamp")

            # Serialize
            value_json = json.dumps(existing)

            # Update with or without new embedding
            if embedding is not None:
                embedding_json = json.dumps(embedding)
                await self._connection.execute(f"""
                    UPDATE {table_name}
                    SET value = ?, embedding = ?, thread_id = ?, timestamp = ?,
                        updated_at = datetime('now')
                    WHERE key = ?
                """, (value_json, embedding_json, thread_id, timestamp, key))
            else:
                await self._connection.execute(f"""
                    UPDATE {table_name}
                    SET value = ?, thread_id = ?, timestamp = ?,
                        updated_at = datetime('now')
                    WHERE key = ?
                """, (value_json, thread_id, timestamp, key))

            await self._connection.commit()

            return True

        except Exception as e:
            raise SQLiteBackendError(f"Failed to update memory: {e}") from e

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    # Utility methods

    async def drop_namespace(self, namespace: tuple[str, ...]) -> bool:
        """Drop a namespace table entirely.

        WARNING: This permanently deletes all data in the namespace.

        Args:
            namespace: Hierarchical namespace tuple

        Returns:
            True if dropped, False if table didn't exist
        """
        if self._connection is None:
            raise SQLiteBackendError("Not connected to database")

        table_name = self._namespace_to_table(namespace)

        try:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not await cursor.fetchone():
                return False

            await self._connection.execute(f"DROP TABLE {table_name}")
            await self._connection.commit()

            self._initialized_tables.discard(table_name)

            return True

        except Exception as e:
            raise SQLiteBackendError(f"Failed to drop namespace: {e}") from e

    async def get_namespace_count(self, namespace: tuple[str, ...]) -> int:
        """Get count of memories in a namespace.

        Args:
            namespace: Hierarchical namespace tuple

        Returns:
            Number of memories in namespace
        """
        table_name = self._namespace_to_table(namespace)

        try:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not await cursor.fetchone():
                return 0

            cursor = await self._connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

        except Exception as e:
            raise SQLiteBackendError(f"Failed to get namespace count: {e}") from e

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim space and optimize database.

        Should be called periodically for databases with many deletions.
        """
        if self._connection is None:
            raise SQLiteBackendError("Not connected to database")

        try:
            await self._connection.execute("VACUUM")
        except Exception as e:
            raise SQLiteBackendError(f"Failed to vacuum database: {e}") from e
