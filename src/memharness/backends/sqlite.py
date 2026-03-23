# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""SQLite backend implementing BackendProtocol from harness.py"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

# Import MemoryType and MemoryUnit from parent module
from memharness.types import MemoryType, MemoryUnit

logger = logging.getLogger(__name__)


class SqliteBackend:
    """
    SQLite storage backend implementing BackendProtocol.

    Stores MemoryUnit objects in a single table with vector embeddings support.
    Uses cosine similarity for semantic search.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
        """
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Establish connection to the backend storage."""
        if self._connection is not None:
            return

        # Ensure parent directory exists for file-based databases
        if self._db_path != ":memory:":
            db_file = Path(self._db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self._db_path)

        # Enable WAL mode for better concurrent performance
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA foreign_keys=ON")

        # Create main memories table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                namespace TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                thread_id TEXT,
                parent_id TEXT
            )
        """)

        # Create indexes
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memories(memory_type)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_namespace
            ON memories(namespace)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON memories(created_at)
        """)

        await self._connection.commit()

        logger.info(f"Connected to SQLite database: {self._db_path}")

    async def disconnect(self) -> None:
        """Close connection to the backend storage."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def store(self, unit: MemoryUnit) -> str:
        """
        Store a memory unit and return its ID.

        Args:
            unit: The MemoryUnit to store.

        Returns:
            The ID of the stored unit.
        """
        if self._connection is None:
            raise RuntimeError("Not connected to database")

        # Serialize namespace and embedding
        namespace_str = json.dumps(list(unit.namespace))
        embedding_str = json.dumps(unit.embedding) if unit.embedding else None
        metadata_str = json.dumps(unit.metadata)

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, content, memory_type, namespace, embedding, metadata,
             created_at, updated_at, thread_id, parent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                unit.id,
                unit.content,
                unit.memory_type.value,
                namespace_str,
                embedding_str,
                metadata_str,
                unit.created_at.isoformat(),
                unit.updated_at.isoformat(),
                unit.thread_id,
                unit.parent_id,
            ),
        )

        await self._connection.commit()
        return unit.id

    async def get(self, memory_id: str) -> MemoryUnit | None:
        """
        Retrieve a memory unit by ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            The MemoryUnit if found, None otherwise.
        """
        if self._connection is None:
            raise RuntimeError("Not connected to database")

        cursor = await self._connection.execute(
            """
            SELECT id, content, memory_type, namespace, embedding, metadata,
                   created_at, updated_at, thread_id, parent_id
            FROM memories
            WHERE id = ?
        """,
            (memory_id,),
        )

        row = await cursor.fetchone()
        if row is None:
            return None

        return self._row_to_unit(row)

    async def search(
        self,
        query_embedding: list[float],
        memory_type: MemoryType | None = None,
        namespace: tuple[str, ...] | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[MemoryUnit]:
        """
        Search for memory units by similarity.

        Args:
            query_embedding: The embedding vector to search with.
            memory_type: Optional filter by memory type.
            namespace: Optional filter by namespace prefix.
            filters: Optional metadata filters.
            k: Number of results to return.

        Returns:
            List of matching MemoryUnit objects.
        """
        if self._connection is None:
            raise RuntimeError("Not connected to database")

        # Build query conditions
        conditions = []
        params = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)

        if namespace:
            # For namespace prefix matching, we check if stored namespace starts with prefix
            namespace_str = json.dumps(list(namespace))
            conditions.append("(namespace = ? OR namespace LIKE ?)")
            params.append(namespace_str)
            params.append(namespace_str[:-1] + ",%]")  # Match prefix

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = await self._connection.execute(
            f"""
            SELECT id, content, memory_type, namespace, embedding, metadata,
                   created_at, updated_at, thread_id, parent_id
            FROM memories
            WHERE {where_clause}
        """,
            params,
        )

        rows = await cursor.fetchall()

        # Calculate similarity scores
        scored = []
        for row in rows:
            unit = self._row_to_unit(row)

            # Apply metadata filters
            if filters:
                if not all(unit.metadata.get(k) == v for k, v in filters.items()):
                    continue

            # Calculate similarity
            if unit.embedding and query_embedding:
                similarity = self._cosine_similarity(query_embedding, unit.embedding)
            else:
                similarity = 0.0

            scored.append((similarity, unit))

        # Sort by similarity descending and return top k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [unit for _, unit in scored[:k]]

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a memory unit.

        Args:
            memory_id: The ID of the memory to update.
            updates: Dictionary of fields to update.

        Returns:
            True if updated successfully, False if not found.
        """
        if self._connection is None:
            raise RuntimeError("Not connected to database")

        # Get existing unit
        unit = await self.get(memory_id)
        if unit is None:
            return False

        # Apply updates
        if "content" in updates:
            unit.content = updates["content"]
        if "metadata" in updates:
            unit.metadata.update(updates["metadata"])
        if "embedding" in updates:
            unit.embedding = updates["embedding"]

        unit.updated_at = datetime.now(UTC)

        # Store updated unit
        await self.store(unit)
        return True

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory unit.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted successfully, False if not found.
        """
        if self._connection is None:
            raise RuntimeError("Not connected to database")

        cursor = await self._connection.execute(
            """
            DELETE FROM memories WHERE id = ?
        """,
            (memory_id,),
        )

        await self._connection.commit()
        return cursor.rowcount > 0

    async def list_by_namespace(
        self,
        namespace: tuple[str, ...],
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[MemoryUnit]:
        """
        List memory units by namespace prefix.

        Args:
            namespace: The namespace prefix to match.
            memory_type: Optional filter by memory type.
            limit: Maximum number of results.

        Returns:
            List of MemoryUnit objects.
        """
        if self._connection is None:
            raise RuntimeError("Not connected to database")

        conditions = []
        params = []

        # Namespace prefix matching
        namespace_str = json.dumps(list(namespace))
        conditions.append("(namespace = ? OR namespace LIKE ?)")
        params.append(namespace_str)
        params.append(namespace_str[:-1] + ",%]")  # Match prefix

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        cursor = await self._connection.execute(
            f"""
            SELECT id, content, memory_type, namespace, embedding, metadata,
                   created_at, updated_at, thread_id, parent_id
            FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """,
            params,
        )

        rows = await cursor.fetchall()
        return [self._row_to_unit(row) for row in rows]

    def _row_to_unit(self, row: tuple) -> MemoryUnit:
        """
        Convert a database row to a MemoryUnit.

        Args:
            row: Database row tuple.

        Returns:
            MemoryUnit object.
        """
        (
            id_val,
            content,
            memory_type_str,
            namespace_str,
            embedding_str,
            metadata_str,
            created_at_str,
            updated_at_str,
            thread_id,
            parent_id,
        ) = row

        return MemoryUnit(
            id=id_val,
            content=content,
            memory_type=MemoryType(memory_type_str),
            namespace=tuple(json.loads(namespace_str)),
            embedding=json.loads(embedding_str) if embedding_str else None,
            metadata=json.loads(metadata_str),
            created_at=datetime.fromisoformat(created_at_str),
            updated_at=datetime.fromisoformat(updated_at_str),
            thread_id=thread_id,
            parent_id=parent_id,
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity score.
        """
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
