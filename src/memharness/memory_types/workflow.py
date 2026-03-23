# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Workflow memory type mixin.

This module provides methods for managing workflow memories
(task procedures and outcomes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["WorkflowMixin"]


class WorkflowMixin(BaseMixin):
    """Mixin for workflow memory operations."""

    async def add_workflow(
        self,
        task: str,
        steps: list[str],
        outcome: str,
        result: str | None = None,
    ) -> str:
        """
        Add a workflow/procedure to memory.

        Args:
            task: Description of the task this workflow accomplishes.
            steps: List of steps to complete the task.
            outcome: Expected outcome of the workflow.
            result: Optional actual result after execution.

        Returns:
            The ID of the created memory unit.

        Example:
            ```python
            wf_id = await harness.add_workflow(
                task="Deploy application to production",
                steps=["Run tests", "Build Docker image", "Push to registry", "Update k8s"],
                outcome="Application deployed and healthy",
                result="Deployed v2.1.0 successfully"
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.WORKFLOW)

        # Create searchable content
        steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
        content = f"Task: {task}\nSteps:\n{steps_text}\nOutcome: {outcome}"
        if result:
            content += f"\nResult: {result}"

        embedding = await self._embed(content)

        meta = {
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "result": result,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.WORKFLOW,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_workflow(
        self,
        query: str,
        k: int = 3,
    ) -> list[MemoryUnit]:
        """
        Search for workflows by semantic similarity.

        Args:
            query: The search query (task description, keywords, etc.).
            k: Number of results to return.

        Returns:
            List of matching workflow MemoryUnit objects.

        Example:
            ```python
            workflows = await harness.search_workflow("deploy application", k=3)
            for wf in workflows:
                print(f"Task: {wf.metadata['task']}")
            ```
        """
        from memharness.types import MemoryType

        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.WORKFLOW,
            k=k,
        )
