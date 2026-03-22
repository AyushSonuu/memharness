# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Base classes for embedded memory management agents.

Provides the abstract base class and common infrastructure for all agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from memharness import MemoryHarness


class TriggerType(Enum):
    """When an agent should be triggered."""

    ON_WRITE = "on_write"  # After each write operation
    PRE_LLM = "pre_llm"  # Before LLM call (context preparation)
    SCHEDULED = "scheduled"  # On a cron schedule
    POLICY = "policy"  # Based on policy rules
    ON_DEMAND = "on_demand"  # Manual invocation only


@dataclass
class AgentConfig:
    """Configuration for embedded agents."""

    # General settings
    enabled: bool = True
    log_runs: bool = True

    # Summarizer settings
    summarizer_enabled: bool = True
    summarizer_threshold_messages: int = 20  # Trigger after N messages
    summarizer_max_age_hours: int = 24  # Summarize messages older than this

    # Entity extractor settings
    entity_extractor_enabled: bool = True
    entity_extractor_batch_size: int = 10  # Batch size for batch mode
    entity_extractor_on_write: bool = True  # Extract on each write

    # Consolidator settings
    consolidator_enabled: bool = True
    consolidator_schedule: str = "0 * * * *"  # Every hour
    consolidator_similarity_threshold: float = 0.85  # Merge above this
    consolidator_min_memories: int = 5  # Min memories to consider

    # GC settings
    gc_enabled: bool = True
    gc_schedule: str = "0 0 * * *"  # Daily at midnight
    gc_archive_after_days: int = 90  # Archive after N days
    gc_delete_after_days: int = 365  # Delete after N days

    # Rate limiting
    max_concurrent_agents: int = 2
    agent_timeout_seconds: int = 300


@dataclass
class AgentResult:
    """Result of an agent run."""

    agent_name: str
    success: bool
    started_at: datetime
    completed_at: datetime
    items_processed: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_deleted: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Calculate duration of the run."""
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "items_processed": self.items_processed,
            "items_created": self.items_created,
            "items_updated": self.items_updated,
            "items_deleted": self.items_deleted,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class EmbeddedAgent(ABC):
    """
    Base class for embedded memory management agents.

    Embedded agents perform automated maintenance and enhancement of the
    memory system. They can run on various triggers (writes, schedules, etc.)
    and work both with and without LLM support.

    Without LLM: Uses deterministic heuristics and rule-based logic
    With LLM: Leverages AI for more intelligent operations

    Example:
        class MyAgent(EmbeddedAgent):
            trigger = TriggerType.SCHEDULED
            schedule = "0 * * * *"  # Every hour

            @property
            def name(self) -> str:
                return "my_agent"

            async def run(self, **kwargs) -> dict:
                # Do work
                return {"processed": 10}
    """

    trigger: TriggerType
    schedule: Optional[str] = None

    def __init__(
        self,
        memory: "MemoryHarness",
        llm: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the agent.

        Args:
            memory: The MemoryHarness instance to operate on
            llm: Optional LLM for intelligent operations (None = deterministic mode)
            config: Optional configuration (uses defaults if not provided)
        """
        self.memory = memory
        self.llm = llm
        self.config = config or AgentConfig()
        self.enabled = True
        self._last_run: Optional[datetime] = None
        self._run_count: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this agent."""
        ...

    @abstractmethod
    async def run(self, **kwargs) -> dict[str, Any]:
        """
        Execute the agent's main logic.

        Args:
            **kwargs: Agent-specific arguments

        Returns:
            Dictionary with results (keys depend on agent type)
        """
        ...

    @property
    def has_llm(self) -> bool:
        """Check if LLM is available for intelligent operations."""
        return self.llm is not None

    async def _create_result(
        self,
        success: bool,
        started_at: datetime,
        items_processed: int = 0,
        items_created: int = 0,
        items_updated: int = 0,
        items_deleted: int = 0,
        errors: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Create an AgentResult for this run."""
        return AgentResult(
            agent_name=self.name,
            success=success,
            started_at=started_at,
            completed_at=datetime.now(),
            items_processed=items_processed,
            items_created=items_created,
            items_updated=items_updated,
            items_deleted=items_deleted,
            errors=errors or [],
            metadata=metadata or {},
        )

    async def _log_run(self, result: AgentResult) -> None:
        """Log the agent run if logging is enabled."""
        if self.config.log_runs:
            self._last_run = result.completed_at
            self._run_count += 1
            # Could persist to memory system for audit trail
            # await self.memory.add_tool_log(...)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(name={self.name!r}, "
            f"trigger={self.trigger.value!r}, enabled={self.enabled})>"
        )
