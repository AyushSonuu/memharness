# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""PostgreSQL schema initialization for memharness.

This module handles database schema creation including:
- Extension installation (pgvector, pg_trgm)
- Table creation for all 10 memory types
- Index creation for efficient querying
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

    from memharness.backends.postgres.connection import PostgresConnectionManager

logger = logging.getLogger(__name__)


class PostgresSchemaManager:
    """Manages PostgreSQL schema creation and initialization.

    This class handles:
    - Extension installation
    - Table creation with dynamic vector dimensions
    - Index creation
    - Foreign key constraints
    """

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        """Initialize schema manager.

        Args:
            conn_manager: Connection manager instance
        """
        self._conn_manager = conn_manager
        self._initialized_tables: set[str] = set()

    @property
    def initialized_tables(self) -> set[str]:
        """Get set of initialized table names."""
        return self._initialized_tables

    def clear_initialized_tables(self) -> None:
        """Clear the set of initialized tables."""
        self._initialized_tables.clear()

    async def initialize_schema(self) -> None:
        """Initialize complete database schema.

        Creates:
        - pgvector and pg_trgm extensions
        - All 10 memory type tables
        - Indexes for efficient querying
        - Foreign key constraints
        """
        await self._create_extensions()
        await self._create_all_tables()

    async def _create_extensions(self) -> None:
        """Create required PostgreSQL extensions."""
        async with self._conn_manager.pool.acquire() as conn:
            # pgvector for vector similarity search
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # pg_trgm for text similarity (used in hybrid search)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            logger.debug("PostgreSQL extensions created")

    async def _create_all_tables(self) -> None:
        """Create all memory type tables with indexes."""
        async with self._conn_manager.pool.acquire() as conn:
            # SQL-based tables (conversational, tool_log)
            await self._create_conversational_table(conn)
            await self._create_tool_log_table(conn)

            # Vector-based tables
            await self._create_summary_table(conn)
            await self._create_knowledge_base_table(conn)
            await self._create_entity_table(conn)
            await self._create_workflow_table(conn)
            await self._create_toolbox_table(conn)
            await self._create_file_table(conn)
            await self._create_persona_table(conn)

            # Add foreign key constraint after both tables exist
            await self._add_summary_foreign_key(conn)

    async def _create_conversational_table(self, conn: asyncpg.Connection) -> None:
        """Create conversational memory table (SQL-based)."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversational_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{}',
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                summary_id UUID
            )
        """)

        # Composite index on thread_id and created_at for efficient conversation retrieval
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_thread_time
            ON conversational_memory(thread_id, created_at DESC)
        """)

        # GIN index on namespace array for containment queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_namespace
            ON conversational_memory USING GIN(namespace)
        """)

        # Index on summary_id for finding original messages from summaries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_summary
            ON conversational_memory(summary_id) WHERE summary_id IS NOT NULL
        """)

        # Index on key for fast lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_key
            ON conversational_memory(key)
        """)

        self._initialized_tables.add("conversational_memory")

    async def _create_tool_log_table(self, conn: asyncpg.Connection) -> None:
        """Create tool log memory table (SQL-based audit trail)."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_log_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{}',
                thread_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Composite index for retrieving tool logs by thread
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_log_thread_time
            ON tool_log_memory(thread_id, created_at DESC)
        """)

        # Index on tool_name for filtering by specific tools
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_log_tool
            ON tool_log_memory(tool_name)
        """)

        # GIN index on namespace
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_log_namespace
            ON tool_log_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_log_key
            ON tool_log_memory(key)
        """)

        self._initialized_tables.add("tool_log_memory")

    async def _create_summary_table(self, conn: asyncpg.Connection) -> None:
        """Create summary memory table (Vector-based)."""
        vector_dim = self._conn_manager.vector_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS summary_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{{}}'::TEXT[],
                content TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                thread_id TEXT,
                start_time TIMESTAMPTZ,
                end_time TIMESTAMPTZ,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # HNSW index for fast approximate nearest neighbor search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_embedding
            ON summary_memory USING hnsw(embedding vector_cosine_ops)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_thread
            ON summary_memory(thread_id)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_namespace
            ON summary_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_key
            ON summary_memory(key)
        """)

        # Trigram index for text similarity in hybrid search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_content_trgm
            ON summary_memory USING GIN(content gin_trgm_ops)
        """)

        self._initialized_tables.add("summary_memory")

    async def _add_summary_foreign_key(self, conn: asyncpg.Connection) -> None:
        """Add foreign key from conversational to summary after both tables exist."""
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'fk_conv_summary'
                ) THEN
                    ALTER TABLE conversational_memory
                    ADD CONSTRAINT fk_conv_summary
                    FOREIGN KEY (summary_id) REFERENCES summary_memory(id)
                    ON DELETE SET NULL;
                END IF;
            END $$;
        """)

    async def _create_knowledge_base_table(self, conn: asyncpg.Connection) -> None:
        """Create knowledge base memory table (Vector-based)."""
        vector_dim = self._conn_manager.vector_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS knowledge_base_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{{}}'::TEXT[],
                content TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                source TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kb_embedding
            ON knowledge_base_memory USING hnsw(embedding vector_cosine_ops)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kb_source
            ON knowledge_base_memory(source)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kb_namespace
            ON knowledge_base_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kb_key
            ON knowledge_base_memory(key)
        """)

        # Trigram index for text similarity
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kb_content_trgm
            ON knowledge_base_memory USING GIN(content gin_trgm_ops)
        """)

        self._initialized_tables.add("knowledge_base_memory")

    async def _create_entity_table(self, conn: asyncpg.Connection) -> None:
        """Create entity memory table (Vector-based)."""
        vector_dim = self._conn_manager.vector_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS entity_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{{}}'::TEXT[],
                content TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                entity_type TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_embedding
            ON entity_memory USING hnsw(embedding vector_cosine_ops)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_type
            ON entity_memory(entity_type)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_namespace
            ON entity_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_key
            ON entity_memory(key)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_content_trgm
            ON entity_memory USING GIN(content gin_trgm_ops)
        """)

        self._initialized_tables.add("entity_memory")

    async def _create_workflow_table(self, conn: asyncpg.Connection) -> None:
        """Create workflow memory table (Vector-based)."""
        vector_dim = self._conn_manager.vector_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS workflow_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{{}}'::TEXT[],
                content TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_embedding
            ON workflow_memory USING hnsw(embedding vector_cosine_ops)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_namespace
            ON workflow_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_key
            ON workflow_memory(key)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_content_trgm
            ON workflow_memory USING GIN(content gin_trgm_ops)
        """)

        self._initialized_tables.add("workflow_memory")

    async def _create_toolbox_table(self, conn: asyncpg.Connection) -> None:
        """Create toolbox memory table (Vector-based with VFS path)."""
        vector_dim = self._conn_manager.vector_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS toolbox_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{{}}'::TEXT[],
                content TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                tool_name TEXT NOT NULL,
                vfs_path TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_toolbox_embedding
            ON toolbox_memory USING hnsw(embedding vector_cosine_ops)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_toolbox_tool_name
            ON toolbox_memory(tool_name)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_toolbox_vfs_path
            ON toolbox_memory(vfs_path)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_toolbox_namespace
            ON toolbox_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_toolbox_key
            ON toolbox_memory(key)
        """)

        self._initialized_tables.add("toolbox_memory")

    async def _create_file_table(self, conn: asyncpg.Connection) -> None:
        """Create file memory table (Hybrid - Vector + metadata)."""
        vector_dim = self._conn_manager.vector_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS file_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{{}}'::TEXT[],
                content TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                source TEXT,
                file_path TEXT NOT NULL,
                file_hash TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_embedding
            ON file_memory USING hnsw(embedding vector_cosine_ops)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path
            ON file_memory(file_path)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_hash
            ON file_memory(file_hash)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_namespace
            ON file_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_key
            ON file_memory(key)
        """)

        self._initialized_tables.add("file_memory")

    async def _create_persona_table(self, conn: asyncpg.Connection) -> None:
        """Create persona memory table (Vector-based)."""
        vector_dim = self._conn_manager.vector_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS persona_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                key TEXT NOT NULL UNIQUE,
                namespace TEXT[] NOT NULL DEFAULT '{{}}'::TEXT[],
                content TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                persona_name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_persona_embedding
            ON persona_memory USING hnsw(embedding vector_cosine_ops)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_persona_name
            ON persona_memory(persona_name)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_persona_namespace
            ON persona_memory USING GIN(namespace)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_persona_key
            ON persona_memory(key)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_persona_content_trgm
            ON persona_memory USING GIN(content gin_trgm_ops)
        """)

        self._initialized_tables.add("persona_memory")
