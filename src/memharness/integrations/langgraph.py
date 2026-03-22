# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
LangGraph integration for memharness.

This module provides a LangGraph-compatible checkpointer that uses memharness
as the backend storage. This enables persistent state management for LangGraph
workflows and agents.

Example:
    from langgraph.graph import StateGraph
    from memharness import MemoryHarness
    from memharness.integrations import MemharnessCheckpointer

    # Initialize memharness
    harness = MemoryHarness("postgresql://...")

    # Create checkpointer
    checkpointer = MemharnessCheckpointer(harness=harness)

    # Use with LangGraph
    graph = StateGraph(State)
    # ... define graph nodes and edges ...
    app = graph.compile(checkpointer=checkpointer)

Note: Requires langgraph to be installed.
Install with: pip install memharness[langgraph]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

# Optional dependency handling for LangGraph
try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Provide stub classes for when langgraph is not installed
    BaseCheckpointSaver = object  # type: ignore[misc, assignment]
    Checkpoint = Dict[str, Any]  # type: ignore[misc, assignment]
    CheckpointMetadata = Dict[str, Any]  # type: ignore[misc, assignment]
    CheckpointTuple = Tuple  # type: ignore[misc, assignment]
    ChannelVersions = Dict[str, Any]  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from langgraph.checkpoint.base import RunnableConfig
    from memharness import MemoryHarness


__all__ = [
    "MemharnessCheckpointer",
    "LANGGRAPH_AVAILABLE",
]


class MemharnessCheckpointer(BaseCheckpointSaver):
    """
    LangGraph checkpointer using memharness as the storage backend.

    This class implements LangGraph's BaseCheckpointSaver interface,
    allowing you to persist graph state using memharness's workflow
    memory type.

    Attributes:
        harness: The MemoryHarness instance to use for storage.
        serde: Serializer/deserializer for checkpoint data (optional).

    Example:
        >>> harness = MemoryHarness("sqlite:///memory.db")
        >>> checkpointer = MemharnessCheckpointer(harness=harness)
        >>>
        >>> # Use with a compiled graph
        >>> app = graph.compile(checkpointer=checkpointer)
        >>>
        >>> # Run with thread ID for persistence
        >>> config = {"configurable": {"thread_id": "my-thread"}}
        >>> result = await app.ainvoke({"input": "hello"}, config)
    """

    harness: Any  # MemoryHarness instance

    def __init__(
        self,
        harness: "MemoryHarness",
        *,
        serde: Optional[Any] = None,
    ) -> None:
        """
        Initialize MemharnessCheckpointer.

        Args:
            harness: The MemoryHarness instance to use for storage.
            serde: Optional serializer/deserializer. If None, uses JSON.

        Raises:
            ImportError: If langgraph is not installed.
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is not installed. "
                "Install with: pip install memharness[langgraph]"
            )

        super().__init__(serde=serde)
        self.harness = harness

    def get_tuple(self, config: "RunnableConfig") -> Optional[CheckpointTuple]:
        """
        Get a checkpoint tuple from storage.

        Args:
            config: The runnable config containing thread_id and optional checkpoint_id.

        Returns:
            CheckpointTuple if found, None otherwise.
        """
        import asyncio
        return self._run_async(self.aget_tuple(config))

    async def aget_tuple(self, config: "RunnableConfig") -> Optional[CheckpointTuple]:
        """
        Async version of get_tuple.

        Args:
            config: The runnable config.

        Returns:
            CheckpointTuple if found, None otherwise.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        # Build the workflow ID for lookup
        workflow_id = self._build_workflow_id(thread_id, checkpoint_id)

        # Try to get the checkpoint from workflow memory
        memories = await self.harness.get_workflow(workflow_id)

        if not memories:
            return None

        # Get the most recent checkpoint (or specific one if checkpoint_id provided)
        checkpoint_data = None
        for mem in memories:
            if checkpoint_id:
                if mem.metadata.get("checkpoint_id") == checkpoint_id:
                    checkpoint_data = mem
                    break
            else:
                # Get the latest
                if checkpoint_data is None or (
                    mem.metadata.get("timestamp", "") >
                    checkpoint_data.metadata.get("timestamp", "")
                ):
                    checkpoint_data = mem

        if checkpoint_data is None:
            return None

        # Deserialize the checkpoint
        try:
            checkpoint = self._deserialize_checkpoint(checkpoint_data.content)
            metadata = self._deserialize_metadata(
                checkpoint_data.metadata.get("checkpoint_metadata", "{}")
            )
        except Exception:
            return None

        # Get parent config if there's a parent checkpoint
        parent_checkpoint_id = checkpoint_data.metadata.get("parent_checkpoint_id")
        parent_config = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_data.metadata.get("checkpoint_id"),
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )

    def list(
        self,
        config: Optional["RunnableConfig"],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional["RunnableConfig"] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints matching the given criteria.

        Args:
            config: Config with thread_id to filter by.
            filter: Additional metadata filters.
            before: Only return checkpoints before this config.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple objects.
        """
        import asyncio

        async def _list():
            results = []
            async for item in self.alist(config, filter=filter, before=before, limit=limit):
                results.append(item)
            return results

        for item in self._run_async(_list()):
            yield item

    async def alist(
        self,
        config: Optional["RunnableConfig"],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional["RunnableConfig"] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """
        Async version of list.

        Args:
            config: Config with thread_id to filter by.
            filter: Additional metadata filters.
            before: Only return checkpoints before this config.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple objects.
        """
        if config is None:
            return

        thread_id = config["configurable"]["thread_id"]
        workflow_id = self._build_workflow_id(thread_id)

        # Get all checkpoints for this thread
        memories = await self.harness.get_workflow(workflow_id)

        # Sort by timestamp descending
        memories = sorted(
            memories,
            key=lambda m: m.metadata.get("timestamp", ""),
            reverse=True,
        )

        # Apply before filter
        before_ts = None
        if before:
            before_id = before["configurable"].get("checkpoint_id")
            if before_id:
                for mem in memories:
                    if mem.metadata.get("checkpoint_id") == before_id:
                        before_ts = mem.metadata.get("timestamp")
                        break

        count = 0
        for mem in memories:
            # Apply before filter
            if before_ts and mem.metadata.get("timestamp", "") >= before_ts:
                continue

            # Apply custom filter
            if filter:
                metadata = self._deserialize_metadata(
                    mem.metadata.get("checkpoint_metadata", "{}")
                )
                skip = False
                for key, value in filter.items():
                    if metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            # Deserialize and yield
            try:
                checkpoint = self._deserialize_checkpoint(mem.content)
                metadata = self._deserialize_metadata(
                    mem.metadata.get("checkpoint_metadata", "{}")
                )
            except Exception:
                continue

            parent_checkpoint_id = mem.metadata.get("parent_checkpoint_id")
            parent_config = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": mem.metadata.get("checkpoint_id"),
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )

            count += 1
            if limit and count >= limit:
                break

    def put(
        self,
        config: "RunnableConfig",
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> "RunnableConfig":
        """
        Store a checkpoint.

        Args:
            config: The runnable config.
            checkpoint: The checkpoint data to store.
            metadata: Metadata for the checkpoint.
            new_versions: Channel version information.

        Returns:
            Updated config with the new checkpoint_id.
        """
        import asyncio
        return self._run_async(
            self.aput(config, checkpoint, metadata, new_versions)
        )

    async def aput(
        self,
        config: "RunnableConfig",
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> "RunnableConfig":
        """
        Async version of put.

        Args:
            config: The runnable config.
            checkpoint: The checkpoint data to store.
            metadata: Metadata for the checkpoint.
            new_versions: Channel version information.

        Returns:
            Updated config with the new checkpoint_id.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Generate new checkpoint ID from the checkpoint itself
        checkpoint_id = checkpoint["id"]

        # Build workflow ID
        workflow_id = self._build_workflow_id(thread_id)

        # Serialize checkpoint and metadata
        checkpoint_str = self._serialize_checkpoint(checkpoint)
        metadata_str = self._serialize_metadata(metadata)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Store in workflow memory
        await self.harness.add_workflow(
            workflow_id=workflow_id,
            step_name=f"checkpoint_{checkpoint_id}",
            data=checkpoint_str,
            metadata={
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "checkpoint_metadata": metadata_str,
                "timestamp": timestamp,
                "channel_versions": json.dumps(new_versions),
            },
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: "RunnableConfig",
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store intermediate writes for a task.

        Args:
            config: The runnable config.
            writes: Sequence of (channel, value) tuples.
            task_id: The task identifier.
        """
        import asyncio
        self._run_async(self.aput_writes(config, writes, task_id))

    async def aput_writes(
        self,
        config: "RunnableConfig",
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Async version of put_writes.

        Args:
            config: The runnable config.
            writes: Sequence of (channel, value) tuples.
            task_id: The task identifier.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id", "")

        workflow_id = self._build_workflow_id(thread_id)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Serialize writes
        writes_data = []
        for channel, value in writes:
            try:
                writes_data.append({
                    "channel": channel,
                    "value": self.serde.dumps(value) if self.serde else json.dumps(value, default=str),
                })
            except Exception:
                writes_data.append({
                    "channel": channel,
                    "value": str(value),
                })

        await self.harness.add_workflow(
            workflow_id=workflow_id,
            step_name=f"writes_{checkpoint_id}_{task_id}",
            data=json.dumps(writes_data),
            metadata={
                "type": "writes",
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "timestamp": timestamp,
            },
        )

    def _build_workflow_id(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> str:
        """
        Build a workflow ID for memharness storage.

        Args:
            thread_id: The thread identifier.
            checkpoint_id: Optional checkpoint identifier.

        Returns:
            The workflow ID string.
        """
        # Use thread_id as the workflow ID to group all checkpoints together
        return f"langgraph:{thread_id}"

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """
        Serialize a checkpoint for storage.

        Args:
            checkpoint: The checkpoint data.

        Returns:
            Serialized checkpoint string.
        """
        if self.serde:
            return self.serde.dumps(checkpoint)
        return json.dumps(checkpoint, default=str)

    def _deserialize_checkpoint(self, data: str) -> Checkpoint:
        """
        Deserialize a checkpoint from storage.

        Args:
            data: The serialized checkpoint string.

        Returns:
            The checkpoint data.
        """
        if self.serde:
            return self.serde.loads(data)
        return json.loads(data)

    def _serialize_metadata(self, metadata: CheckpointMetadata) -> str:
        """
        Serialize checkpoint metadata for storage.

        Args:
            metadata: The metadata dict.

        Returns:
            Serialized metadata string.
        """
        return json.dumps(metadata, default=str)

    def _deserialize_metadata(self, data: str) -> CheckpointMetadata:
        """
        Deserialize checkpoint metadata from storage.

        Args:
            data: The serialized metadata string.

        Returns:
            The metadata dict.
        """
        return json.loads(data)

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """
        Run an async coroutine in a sync context.

        Args:
            coro: The coroutine to run.

        Returns:
            The result of the coroutine.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(coro)

        # If we're in a running loop, use thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
