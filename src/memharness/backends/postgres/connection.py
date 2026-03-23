# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""PostgreSQL connection pool management for memharness.

This module handles connection pooling and lifecycle management for the
PostgreSQL backend using asyncpg.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import asyncpg
from pgvector.asyncpg import register_vector

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default vector dimension for embeddings
DEFAULT_VECTOR_DIM = 768


class PostgresBackendError(Exception):
    """Exception raised for PostgreSQL backend errors."""

    pass


class PostgresConnectionManager:
    """Manages PostgreSQL connection pool lifecycle.

    This class handles:
    - Connection pool creation and teardown
    - Vector type registration for pgvector support
    - Connection health checks
    """

    def __init__(
        self,
        connection_string: str,
        *,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
        vector_dim: int = DEFAULT_VECTOR_DIM,
    ) -> None:
        """Initialize connection manager.

        Args:
            connection_string: PostgreSQL connection URL
            min_pool_size: Minimum connections in pool (default: 5)
            max_pool_size: Maximum connections in pool (default: 20)
            vector_dim: Dimension of embedding vectors (default: 768)
        """
        self._connection_string = connection_string
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._vector_dim = vector_dim
        self._pool: asyncpg.Pool | None = None

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._pool is not None

    @property
    def pool(self) -> asyncpg.Pool:
        """Get the connection pool, raising if not connected."""
        if self._pool is None:
            raise PostgresBackendError("Backend not connected. Call connect() first.")
        return self._pool

    @property
    def vector_dim(self) -> int:
        """Get configured vector dimension."""
        return self._vector_dim

    async def connect(self) -> None:
        """Initialize connection pool with pgvector support.

        Raises:
            PostgresBackendError: If connection fails
        """
        if self._pool is not None:
            logger.debug("Connection pool already exists")
            return

        try:
            logger.info("Connecting to PostgreSQL...")

            # Create connection pool with vector type registration
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
                init=self._init_connection,
            )

            logger.info("PostgreSQL connection pool created successfully")

        except Exception as e:
            raise PostgresBackendError(f"Failed to connect to database: {e}") from e

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize each connection with pgvector support.

        This callback is invoked for each new connection in the pool.
        """
        await register_vector(conn)

    async def disconnect(self) -> None:
        """Close the connection pool.

        Raises:
            PostgresBackendError: If disconnect fails
        """
        if self._pool is not None:
            try:
                await self._pool.close()
                logger.info("PostgreSQL backend disconnected")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
            finally:
                self._pool = None

    async def health_check(self) -> dict:
        """Check connection pool health.

        Returns:
            Dictionary with health status and pool statistics
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            return {
                "status": "healthy",
                "backend": "postgres",
                "pool_size": self._pool.get_size() if self._pool else 0,
                "pool_min_size": self._min_pool_size,
                "pool_max_size": self._max_pool_size,
                "vector_dim": self._vector_dim,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "postgres",
                "error": str(e),
            }
