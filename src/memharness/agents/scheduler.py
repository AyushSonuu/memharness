# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Agent scheduler for coordinating all embedded agents.

Provides centralized management of agent execution, scheduling,
and coordination.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from memharness.agents.base import AgentConfig, AgentResult, EmbeddedAgent, TriggerType
from memharness.agents.consolidator import ConsolidatorAgent
from memharness.agents.entity_extractor import EntityExtractorAgent
from memharness.agents.gc import GCAgent
from memharness.agents.summarizer import SummarizerAgent

if TYPE_CHECKING:
    from memharness import MemoryHarness


class AgentScheduler:
    """
    Coordinates all embedded agents.

    Features:
    - Manages agent lifecycle
    - Triggers agents on writes
    - Runs scheduled agents
    - Provides on-demand execution
    - Handles concurrency limits
    - Tracks agent run history

    Example:
        scheduler = AgentScheduler(memory, llm=my_llm)

        # On write trigger
        await scheduler.on_write(content="Hello", memory_type="conversational", namespace=("user", "123"))

        # Run specific agent
        result = await scheduler.run_agent("summarizer", thread_id="t1")

        # Run all scheduled agents
        await scheduler.run_scheduled()
    """

    def __init__(
        self,
        memory: "MemoryHarness",
        llm: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the agent scheduler.

        Args:
            memory: The MemoryHarness instance
            llm: Optional LLM for intelligent operations (None = deterministic mode)
            config: Optional agent configuration
        """
        self.memory = memory
        self.llm = llm
        self.config = config or AgentConfig()

        # Initialize agents
        self.agents: dict[str, EmbeddedAgent] = {
            "summarizer": SummarizerAgent(memory, llm, self.config),
            "entity_extractor": EntityExtractorAgent(memory, llm, self.config),
            "consolidator": ConsolidatorAgent(memory, llm, self.config),
            "gc": GCAgent(memory, llm, self.config),
        }

        # Track running agents
        self._running: set[str] = set()
        self._lock = asyncio.Lock()

        # Run history
        self._run_history: list[AgentResult] = []
        self._max_history = 100

        # Background task for scheduled runs
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_running = False

    @property
    def available_agents(self) -> list[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())

    @property
    def enabled_agents(self) -> list[str]:
        """Get list of enabled agent names."""
        return [name for name, agent in self.agents.items() if agent.enabled]

    def get_agent(self, name: str) -> Optional[EmbeddedAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def register_agent(self, agent: EmbeddedAgent) -> None:
        """
        Register a custom agent.

        Args:
            agent: The agent to register
        """
        self.agents[agent.name] = agent

    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.

        Args:
            name: Name of the agent to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.agents:
            del self.agents[name]
            return True
        return False

    async def on_write(
        self,
        content: str,
        memory_type: str,
        namespace: Optional[tuple[str, ...]] = None,
        memory_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Handle a memory write event.

        Triggers all ON_WRITE agents with the written content.

        Args:
            content: The content that was written
            memory_type: Type of memory written
            namespace: Namespace of the memory
            memory_id: ID of the created memory

        Returns:
            Dictionary with results from triggered agents
        """
        if not self.config.enabled:
            return {"skipped": True, "reason": "agents_disabled"}

        results: dict[str, Any] = {}

        # Find ON_WRITE agents
        for name, agent in self.agents.items():
            if agent.trigger == TriggerType.ON_WRITE and agent.enabled:
                # Check specific agent configs
                if name == "entity_extractor" and not self.config.entity_extractor_on_write:
                    continue

                try:
                    result = await self._run_with_limits(
                        name,
                        content=content,
                        memory_type=memory_type,
                        namespace=namespace,
                        memory_id=memory_id,
                    )
                    results[name] = result
                except Exception as e:
                    results[name] = {"error": str(e), "success": False}

        return results

    async def run_scheduled(self) -> dict[str, Any]:
        """
        Run all scheduled agents that are due.

        Returns:
            Dictionary with results from all run agents
        """
        if not self.config.enabled:
            return {"skipped": True, "reason": "agents_disabled"}

        results: dict[str, Any] = {}

        # Find SCHEDULED agents
        for name, agent in self.agents.items():
            if agent.trigger == TriggerType.SCHEDULED and agent.enabled:
                # Check if agent should run (based on schedule)
                if self._should_run_scheduled(agent):
                    try:
                        result = await self._run_with_limits(name)
                        results[name] = result
                    except Exception as e:
                        results[name] = {"error": str(e), "success": False}

        return results

    async def run_agent(
        self,
        name: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run a specific agent on demand.

        Args:
            name: Name of the agent to run
            **kwargs: Arguments to pass to the agent

        Returns:
            Agent execution result
        """
        if name not in self.agents:
            return {"error": f"Unknown agent: {name}", "success": False}

        agent = self.agents[name]

        if not agent.enabled:
            return {"error": f"Agent {name} is disabled", "success": False}

        return await self._run_with_limits(name, **kwargs)

    async def run_all(
        self,
        parallel: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run all enabled agents.

        Args:
            parallel: If True, run agents concurrently (respecting limits)
            **kwargs: Arguments to pass to all agents

        Returns:
            Dictionary with results from all agents
        """
        results: dict[str, Any] = {}

        if parallel:
            # Run concurrently
            tasks = []
            for name in self.enabled_agents:
                tasks.append(self._run_with_limits(name, **kwargs))

            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(self.enabled_agents, task_results):
                if isinstance(result, Exception):
                    results[name] = {"error": str(result), "success": False}
                else:
                    results[name] = result
        else:
            # Run sequentially
            for name in self.enabled_agents:
                try:
                    results[name] = await self._run_with_limits(name, **kwargs)
                except Exception as e:
                    results[name] = {"error": str(e), "success": False}

        return results

    async def _run_with_limits(
        self,
        name: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Run an agent with concurrency limits and timeout."""
        async with self._lock:
            # Check concurrency limit
            if len(self._running) >= self.config.max_concurrent_agents:
                return {
                    "error": "Concurrency limit reached",
                    "success": False,
                    "running": list(self._running),
                }

            self._running.add(name)

        try:
            agent = self.agents[name]

            # Run with timeout
            result = await asyncio.wait_for(
                agent.run(**kwargs),
                timeout=self.config.agent_timeout_seconds,
            )

            # Track result
            if self.config.log_runs:
                self._add_to_history(result)

            return result

        except asyncio.TimeoutError:
            return {
                "error": f"Agent {name} timed out after {self.config.agent_timeout_seconds}s",
                "success": False,
            }

        finally:
            async with self._lock:
                self._running.discard(name)

    def _should_run_scheduled(self, agent: EmbeddedAgent) -> bool:
        """
        Check if a scheduled agent should run.

        Uses simple interval-based logic. For production, would use
        a proper cron parser.
        """
        if not agent.schedule:
            return False

        # Simple interval parsing for common patterns
        # Format: "minute hour day month weekday" (cron-like)
        # Special patterns: "0 * * * *" = hourly, "0 0 * * *" = daily

        schedule = agent.schedule
        now = datetime.now()

        # Check if agent ran recently
        if agent._last_run:
            elapsed = (now - agent._last_run).total_seconds()

            # Parse schedule for minimum interval
            parts = schedule.split()
            if len(parts) >= 2:
                minute, hour = parts[0], parts[1]

                if minute == "0" and hour == "*":
                    # Hourly: require at least 50 minutes since last run
                    return elapsed >= 3000  # 50 minutes

                if minute == "0" and hour == "0":
                    # Daily: require at least 23 hours since last run
                    return elapsed >= 82800  # 23 hours

                if hour.startswith("*/"):
                    # Every N hours
                    try:
                        hours = int(hour[2:])
                        return elapsed >= (hours * 3600 - 600)
                    except ValueError:
                        pass

        # If never run, run now
        return agent._last_run is None

    def _add_to_history(self, result: dict[str, Any]) -> None:
        """Add result to run history."""
        # Convert dict to AgentResult if needed
        if isinstance(result, dict):
            agent_result = AgentResult(
                agent_name=result.get("agent_name", "unknown"),
                success=result.get("success", False),
                started_at=datetime.fromisoformat(result["started_at"])
                if isinstance(result.get("started_at"), str)
                else result.get("started_at", datetime.now()),
                completed_at=datetime.fromisoformat(result["completed_at"])
                if isinstance(result.get("completed_at"), str)
                else result.get("completed_at", datetime.now()),
                items_processed=result.get("items_processed", 0),
                items_created=result.get("items_created", 0),
                items_updated=result.get("items_updated", 0),
                items_deleted=result.get("items_deleted", 0),
                errors=result.get("errors", []),
                metadata=result.get("metadata", {}),
            )
        else:
            agent_result = result

        self._run_history.append(agent_result)

        # Trim history if needed
        if len(self._run_history) > self._max_history:
            self._run_history = self._run_history[-self._max_history :]

    def get_history(
        self,
        agent_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get agent run history.

        Args:
            agent_name: Filter by agent name (None = all)
            limit: Maximum results to return

        Returns:
            List of run results
        """
        history = self._run_history

        if agent_name:
            history = [r for r in history if r.agent_name == agent_name]

        # Return most recent first
        return [
            r.to_dict() if isinstance(r, AgentResult) else r
            for r in reversed(history[-limit:])
        ]

    def get_status(self) -> dict[str, Any]:
        """
        Get current scheduler status.

        Returns:
            Status information
        """
        return {
            "enabled": self.config.enabled,
            "has_llm": self.llm is not None,
            "running_agents": list(self._running),
            "agents": {
                name: {
                    "enabled": agent.enabled,
                    "trigger": agent.trigger.value,
                    "schedule": agent.schedule,
                    "last_run": agent._last_run.isoformat() if agent._last_run else None,
                    "run_count": agent._run_count,
                }
                for name, agent in self.agents.items()
            },
            "config": {
                "max_concurrent_agents": self.config.max_concurrent_agents,
                "agent_timeout_seconds": self.config.agent_timeout_seconds,
            },
        }

    async def start_background_scheduler(
        self,
        interval_seconds: int = 60,
    ) -> None:
        """
        Start a background task that runs scheduled agents.

        Args:
            interval_seconds: How often to check for due agents
        """
        if self._scheduler_running:
            return

        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(
            self._background_scheduler_loop(interval_seconds)
        )

    async def stop_background_scheduler(self) -> None:
        """Stop the background scheduler."""
        self._scheduler_running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

    async def _background_scheduler_loop(self, interval: int) -> None:
        """Background loop for running scheduled agents."""
        while self._scheduler_running:
            try:
                await self.run_scheduled()
            except Exception:
                # Log error but continue
                pass

            await asyncio.sleep(interval)

    def enable_agent(self, name: str) -> bool:
        """Enable an agent."""
        if name in self.agents:
            self.agents[name].enabled = True
            return True
        return False

    def disable_agent(self, name: str) -> bool:
        """Disable an agent."""
        if name in self.agents:
            self.agents[name].enabled = False
            return True
        return False

    def configure_agent(
        self,
        name: str,
        **settings,
    ) -> bool:
        """
        Configure a specific agent.

        Args:
            name: Agent name
            **settings: Settings to update

        Returns:
            True if configured, False if agent not found
        """
        agent = self.agents.get(name)
        if not agent:
            return False

        # Update agent-specific settings
        for key, value in settings.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
            elif hasattr(agent.config, key):
                setattr(agent.config, key, value)

        return True

    def __repr__(self) -> str:
        return (
            f"<AgentScheduler(agents={len(self.agents)}, "
            f"enabled={len(self.enabled_agents)}, "
            f"running={len(self._running)})>"
        )
