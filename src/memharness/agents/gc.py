# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Garbage collection agent for memory maintenance.

Handles TTL expiration, archival of old memories, and cleanup
of orphaned references.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from memharness.agents.base import AgentConfig, AgentResult, EmbeddedAgent, TriggerType

if TYPE_CHECKING:
    from memharness import MemoryHarness


class GCAction(Enum):
    """Types of garbage collection actions."""

    EXPIRE = "expire"  # Remove expired (TTL) memories
    ARCHIVE = "archive"  # Move old memories to archive
    CLEAN_ORPHANS = "clean_orphans"  # Remove orphaned references
    VACUUM = "vacuum"  # Optimize storage


@dataclass
class GCTarget:
    """Represents a memory targeted for GC."""

    memory_id: str
    action: GCAction
    reason: str
    age_days: Optional[int] = None
    metadata: dict[str, Any] | None = None


class GCAgent(EmbeddedAgent):
    """
    Garbage collection agent for memory maintenance.

    Features:
    - Removes expired memories (TTL-based)
    - Archives old memories (configurable age)
    - Cleans up orphaned references
    - Runs on schedule (daily by default)

    All operations are deterministic (no LLM needed).
    """

    trigger = TriggerType.SCHEDULED
    schedule = "0 0 * * *"  # Daily at midnight

    def __init__(
        self,
        memory: "MemoryHarness",
        llm: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(memory, llm, config)

    @property
    def name(self) -> str:
        return "gc"

    async def run(
        self,
        actions: Optional[list[GCAction]] = None,
        namespace: Optional[tuple[str, ...]] = None,
        dry_run: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run garbage collection.

        Args:
            actions: Specific actions to run (None = all)
            namespace: Namespace to operate in
            dry_run: If True, report targets without acting

        Returns:
            Dictionary with GC results
        """
        started_at = datetime.now()
        errors: list[str] = []

        # Stats
        expired_count = 0
        archived_count = 0
        orphans_cleaned = 0
        total_processed = 0

        # Default to all actions
        if actions is None:
            actions = [GCAction.EXPIRE, GCAction.ARCHIVE, GCAction.CLEAN_ORPHANS]

        targets: list[GCTarget] = []

        try:
            # Collect targets for each action
            if GCAction.EXPIRE in actions:
                expired = await self._find_expired(namespace)
                targets.extend(expired)

            if GCAction.ARCHIVE in actions:
                to_archive = await self._find_archivable(namespace)
                targets.extend(to_archive)

            if GCAction.CLEAN_ORPHANS in actions:
                orphans = await self._find_orphans(namespace)
                targets.extend(orphans)

            # Process targets
            if not dry_run:
                for target in targets:
                    try:
                        await self._process_target(target)
                        total_processed += 1

                        if target.action == GCAction.EXPIRE:
                            expired_count += 1
                        elif target.action == GCAction.ARCHIVE:
                            archived_count += 1
                        elif target.action == GCAction.CLEAN_ORPHANS:
                            orphans_cleaned += 1

                    except Exception as e:
                        errors.append(
                            f"{target.action.value} {target.memory_id}: {str(e)}"
                        )

            result = await self._create_result(
                success=len(errors) == 0,
                started_at=started_at,
                items_processed=total_processed if not dry_run else len(targets),
                items_deleted=expired_count + orphans_cleaned,
                items_updated=archived_count,
                errors=errors,
                metadata={
                    "dry_run": dry_run,
                    "expired": expired_count,
                    "archived": archived_count,
                    "orphans_cleaned": orphans_cleaned,
                    "targets": [
                        {
                            "id": t.memory_id,
                            "action": t.action.value,
                            "reason": t.reason,
                            "age_days": t.age_days,
                        }
                        for t in targets[:20]  # Limit for reporting
                    ],
                },
            )
            await self._log_run(result)

            return result.to_dict()

        except Exception as e:
            result = await self._create_result(
                success=False,
                started_at=started_at,
                errors=[str(e)],
            )
            return result.to_dict()

    async def expire_ttl(
        self,
        namespace: Optional[tuple[str, ...]] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Expire memories that have exceeded their TTL.

        Args:
            namespace: Namespace to operate in
            dry_run: If True, report without expiring

        Returns:
            Expiration results
        """
        return await self.run(
            actions=[GCAction.EXPIRE],
            namespace=namespace,
            dry_run=dry_run,
        )

    async def archive_old(
        self,
        namespace: Optional[tuple[str, ...]] = None,
        age_days: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Archive old memories.

        Args:
            namespace: Namespace to operate in
            age_days: Archive memories older than this (default from config)
            dry_run: If True, report without archiving

        Returns:
            Archive results
        """
        # Override age if provided
        if age_days is not None:
            old_value = self.config.gc_archive_after_days
            self.config.gc_archive_after_days = age_days
            result = await self.run(
                actions=[GCAction.ARCHIVE],
                namespace=namespace,
                dry_run=dry_run,
            )
            self.config.gc_archive_after_days = old_value
            return result

        return await self.run(
            actions=[GCAction.ARCHIVE],
            namespace=namespace,
            dry_run=dry_run,
        )

    async def clean_orphans(
        self,
        namespace: Optional[tuple[str, ...]] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Clean orphaned references.

        Args:
            namespace: Namespace to operate in
            dry_run: If True, report without cleaning

        Returns:
            Cleanup results
        """
        return await self.run(
            actions=[GCAction.CLEAN_ORPHANS],
            namespace=namespace,
            dry_run=dry_run,
        )

    async def _find_expired(
        self,
        namespace: Optional[tuple[str, ...]],
    ) -> list[GCTarget]:
        """Find memories that have exceeded their TTL."""
        targets: list[GCTarget] = []
        now = datetime.now()

        # Would query memories with TTL metadata
        # memories = await self.memory.search(
        #     filters={"metadata.ttl": {"$exists": True}},
        #     namespace=namespace,
        # )

        memories: list[dict[str, Any]] = []

        for memory in memories:
            ttl = memory.get("metadata", {}).get("ttl")
            created_at = memory.get("created_at")

            if ttl and created_at:
                # Parse created_at if string
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)

                expiry = created_at + timedelta(seconds=ttl)

                if now > expiry:
                    age_days = (now - created_at).days
                    targets.append(
                        GCTarget(
                            memory_id=memory["id"],
                            action=GCAction.EXPIRE,
                            reason=f"TTL expired (set: {ttl}s)",
                            age_days=age_days,
                        )
                    )

        return targets

    async def _find_archivable(
        self,
        namespace: Optional[tuple[str, ...]],
    ) -> list[GCTarget]:
        """Find memories old enough to archive."""
        targets: list[GCTarget] = []
        now = datetime.now()
        archive_threshold = now - timedelta(days=self.config.gc_archive_after_days)

        # Would query old, non-archived memories
        # memories = await self.memory.search(
        #     filters={
        #         "created_at": {"$lt": archive_threshold.isoformat()},
        #         "metadata.archived": {"$ne": True},
        #     },
        #     namespace=namespace,
        # )

        memories: list[dict[str, Any]] = []

        for memory in memories:
            created_at = memory.get("created_at")

            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)

                age_days = (now - created_at).days

                # Skip recently accessed memories
                last_accessed = memory.get("metadata", {}).get("last_accessed")
                if last_accessed:
                    if isinstance(last_accessed, str):
                        last_accessed = datetime.fromisoformat(last_accessed)
                    # Don't archive if accessed recently
                    if (now - last_accessed).days < self.config.gc_archive_after_days:
                        continue

                targets.append(
                    GCTarget(
                        memory_id=memory["id"],
                        action=GCAction.ARCHIVE,
                        reason=f"Age exceeds {self.config.gc_archive_after_days} days",
                        age_days=age_days,
                    )
                )

        return targets

    async def _find_orphans(
        self,
        namespace: Optional[tuple[str, ...]],
    ) -> list[GCTarget]:
        """Find orphaned references (memories pointing to deleted items)."""
        targets: list[GCTarget] = []

        # Would query memories with source_ids or references
        # memories_with_refs = await self.memory.search(
        #     filters={"metadata.source_ids": {"$exists": True}},
        #     namespace=namespace,
        # )

        memories_with_refs: list[dict[str, Any]] = []

        for memory in memories_with_refs:
            source_ids = memory.get("metadata", {}).get("source_ids", [])

            if source_ids:
                # Check if referenced memories exist
                # existing = await self.memory.get_many(source_ids)
                existing_ids: set[str] = set()  # {m["id"] for m in existing if m}

                orphaned_refs = set(source_ids) - existing_ids

                if orphaned_refs:
                    targets.append(
                        GCTarget(
                            memory_id=memory["id"],
                            action=GCAction.CLEAN_ORPHANS,
                            reason=f"Orphaned refs: {list(orphaned_refs)[:3]}...",
                            metadata={"orphaned_refs": list(orphaned_refs)},
                        )
                    )

        return targets

    async def _process_target(self, target: GCTarget) -> None:
        """Process a GC target."""
        if target.action == GCAction.EXPIRE:
            await self._expire_memory(target.memory_id)

        elif target.action == GCAction.ARCHIVE:
            await self._archive_memory(target.memory_id)

        elif target.action == GCAction.CLEAN_ORPHANS:
            await self._clean_orphan_refs(
                target.memory_id,
                target.metadata.get("orphaned_refs", []) if target.metadata else [],
            )

    async def _expire_memory(self, memory_id: str) -> None:
        """Expire (delete) a memory."""
        # Would call memory.delete()
        # await self.memory.delete(memory_id)
        pass

    async def _archive_memory(self, memory_id: str) -> None:
        """Archive a memory (mark as archived, optionally move to cold storage)."""
        # Would call memory.update() to mark as archived
        # await self.memory.update(
        #     memory_id,
        #     metadata={
        #         "archived": True,
        #         "archived_at": datetime.now().isoformat(),
        #     }
        # )
        pass

    async def _clean_orphan_refs(
        self,
        memory_id: str,
        orphaned_refs: list[str],
    ) -> None:
        """Remove orphaned references from a memory."""
        # Would fetch memory and update source_ids
        # memory = await self.memory.get(memory_id)
        # current_refs = memory.get("metadata", {}).get("source_ids", [])
        # cleaned_refs = [r for r in current_refs if r not in orphaned_refs]
        #
        # await self.memory.update(
        #     memory_id,
        #     metadata={
        #         "source_ids": cleaned_refs,
        #         "refs_cleaned_at": datetime.now().isoformat(),
        #     }
        # )
        pass

    async def get_stats(
        self,
        namespace: Optional[tuple[str, ...]] = None,
    ) -> dict[str, Any]:
        """
        Get GC statistics without performing any actions.

        Args:
            namespace: Namespace to analyze

        Returns:
            Statistics about GC candidates
        """
        expired = await self._find_expired(namespace)
        archivable = await self._find_archivable(namespace)
        orphans = await self._find_orphans(namespace)

        return {
            "expired_count": len(expired),
            "archivable_count": len(archivable),
            "orphan_count": len(orphans),
            "total_gc_candidates": len(expired) + len(archivable) + len(orphans),
            "config": {
                "archive_after_days": self.config.gc_archive_after_days,
                "delete_after_days": self.config.gc_delete_after_days,
            },
        }
