# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""Backend protocol definition for storage backends."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol for storage backends.

    All storage backends must implement this protocol to be compatible
    with the memharness system. Backends handle the actual persistence
    of memory units.

    Namespaces are hierarchical tuples that organize memories. For example:
    - ("conversational", "thread_123") for conversation messages
    - ("kb",) for knowledge base entries
    - ("entity", "user_456") for entity memories

    Keys uniquely identify memories within a namespace.

    Values are dictionaries containing the memory data.

    Embeddings are optional float vectors for semantic search.
    """

    async def write(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: list[float] | None = None
    ) -> str:
        """Write a memory to storage.

        Args:
            namespace: Hierarchical namespace tuple (e.g., ("kb",), ("entity", "user_id"))
            key: Unique identifier within the namespace
            value: Memory data as a dictionary
            embedding: Optional embedding vector for semantic search

        Returns:
            The key of the written memory

        Raises:
            BackendError: If write operation fails
        """
        ...

    async def read(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> dict | None:
        """Read a single memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace

        Returns:
            Memory data dictionary if found, None otherwise

        Raises:
            BackendError: If read operation fails
        """
        ...

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        embedding: list[float] | None = None,
        k: int = 10,
        filters: dict | None = None
    ) -> list[dict]:
        """Search memories using text query or embedding similarity.

        Args:
            namespace: Hierarchical namespace tuple
            query: Text query for search
            embedding: Optional embedding vector for semantic similarity search
            k: Maximum number of results to return
            filters: Optional filters to apply (e.g., {"thread_id": "t1"})

        Returns:
            List of matching memory dictionaries, ordered by relevance

        Raises:
            BackendError: If search operation fails
        """
        ...

    async def list(
        self,
        namespace: tuple[str, ...],
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int | None = None
    ) -> list[dict]:
        """List memories in a namespace with optional filtering.

        Args:
            namespace: Hierarchical namespace tuple
            filters: Optional filters to apply
            order_by: Optional field to order by (prefix with - for descending)
            limit: Optional maximum number of results

        Returns:
            List of memory dictionaries

        Raises:
            BackendError: If list operation fails
        """
        ...

    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> bool:
        """Delete a memory by key.

        Args:
            namespace: Hierarchical namespace tuple
            key: Unique identifier within the namespace

        Returns:
            True if memory was deleted, False if not found

        Raises:
            BackendError: If delete operation fails
        """
        ...

    async def update(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        embedding: list[float] | None = None
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
            BackendError: If update operation fails
        """
        ...

    async def connect(self) -> None:
        """Initialize connection to the backend.

        Called before any operations. Implementations should create
        tables, establish connections, etc.

        Raises:
            BackendError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """Close connection to the backend.

        Called during cleanup. Implementations should close connections,
        flush buffers, etc.

        Raises:
            BackendError: If disconnect fails
        """
        ...
