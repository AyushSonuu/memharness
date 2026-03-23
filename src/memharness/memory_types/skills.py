# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Skills memory type mixin.

This module provides methods for managing skills memories
(learned capabilities).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["SkillsMixin"]


class SkillsMixin(BaseMixin):
    """Mixin for skills memory operations."""

    async def add_skill(
        self,
        name: str,
        description: str,
        examples: list[str] | None = None,
        category: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Add a learned skill to memory.

        Args:
            name: Name of the skill.
            description: Description of what the skill does.
            examples: Optional list of example usages.
            category: Optional category for the skill.
            **kwargs: Additional skill attributes.

        Returns:
            The ID of the created skill memory.

        Example:
            ```python
            skill_id = await harness.add_skill(
                name="code_review",
                description="Review code for bugs, style issues, and improvements",
                examples=[
                    "Review this Python function for efficiency",
                    "Check this code for security vulnerabilities"
                ]
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.SKILLS)

        content = f"Skill: {name}\n{description}"
        if category:
            content += f"\nCategory: {category}"
        if examples:
            content += "\nExamples:\n" + "\n".join(f"- {ex}" for ex in examples)

        embedding = await self._embed(content)

        meta = {
            "name": name,
            "description": description,
            "examples": examples or [],
            "category": category,
            **kwargs,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.SKILLS,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def search_skills(
        self,
        query: str,
        k: int = 3,
    ) -> list[MemoryUnit]:
        """
        Search for relevant skills.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of matching skill MemoryUnit objects.

        Example:
            ```python
            skills = await harness.search_skills("review code for bugs")
            for skill in skills:
                print(f"Skill: {skill.metadata['name']}")
            ```
        """
        from memharness.types import MemoryType

        query_embedding = await self._embed(query)
        return await self._backend.search(
            query_embedding=query_embedding,
            memory_type=MemoryType.SKILLS,
            k=k,
        )
