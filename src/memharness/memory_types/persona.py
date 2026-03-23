# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Persona memory type mixin.

This module provides methods for managing persona memories
(user/agent persona blocks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memharness.memory_types.base import BaseMixin

if TYPE_CHECKING:
    from memharness.types import MemoryUnit

__all__ = ["PersonaMixin"]


class PersonaMixin(BaseMixin):
    """Mixin for persona memory operations."""

    async def add_persona(
        self,
        block_name: str,
        content: str,
    ) -> str:
        """
        Add or update a persona block.

        Args:
            block_name: Name of the persona block (e.g., "preferences", "background").
            content: The persona content.

        Returns:
            The ID of the created/updated persona block.

        Example:
            ```python
            await harness.add_persona(
                block_name="communication_style",
                content="Prefers concise, technical explanations with code examples"
            )
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.PERSONA, block_name)

        # Check if block already exists and delete it
        existing = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.PERSONA,
            limit=1,
        )
        for unit in existing:
            await self._backend.delete(unit.id)

        embedding = await self._embed(content)

        meta = {
            "block_name": block_name,
        }

        unit = self._create_unit(
            content=content,
            memory_type=MemoryType.PERSONA,
            namespace=namespace,
            metadata=meta,
            embedding=embedding,
        )

        return await self._backend.store(unit)

    async def get_persona(
        self,
        block_name: str | None = None,
    ) -> str:
        """
        Retrieve persona content.

        Args:
            block_name: Optional specific block name. If None, returns all blocks.

        Returns:
            The persona content as a string.

        Example:
            ```python
            # Get specific block
            style = await harness.get_persona("communication_style")

            # Get all persona blocks
            full_persona = await harness.get_persona()
            ```
        """
        from memharness.types import MemoryType

        if block_name:
            namespace = self._build_namespace(MemoryType.PERSONA, block_name)
            results = await self._backend.list_by_namespace(
                namespace=namespace,
                memory_type=MemoryType.PERSONA,
                limit=1,
            )
            return results[0].content if results else ""

        # Get all persona blocks
        namespace = self._build_namespace(MemoryType.PERSONA)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.PERSONA,
            limit=self._config.persona_max_blocks,
        )

        blocks = []
        for unit in results:
            block_name = unit.metadata.get("block_name", "unknown")
            blocks.append(f"## {block_name}\n{unit.content}")

        return "\n\n".join(blocks)

    async def set_persona(
        self,
        name: str,
        traits: list[str] | None = None,
        communication_style: str | None = None,
        domain_expertise: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Set the active persona for the agent.

        Args:
            name: Name of the persona.
            traits: List of personality traits.
            communication_style: Communication style description.
            domain_expertise: List of domain expertise areas.
            **kwargs: Additional persona attributes.

        Returns:
            The ID of the created persona.

        Example:
            ```python
            persona_id = await harness.set_persona(
                name="Technical Expert",
                traits=["concise", "technical", "helpful"],
                communication_style="professional",
                domain_expertise=["python", "devops"]
            )
            ```
        """
        # Construct persona content
        content_parts = [f"Persona: {name}"]

        if traits:
            content_parts.append(f"Traits: {', '.join(traits)}")

        if communication_style:
            content_parts.append(f"Communication Style: {communication_style}")

        if domain_expertise:
            content_parts.append(f"Domain Expertise: {', '.join(domain_expertise)}")

        for key, value in kwargs.items():
            content_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        content = "\n".join(content_parts)

        # Store as persona block
        return await self.add_persona(block_name=name, content=content)

    async def get_active_persona(self) -> MemoryUnit | None:
        """
        Get the active persona.

        Returns:
            The active persona as a MemoryUnit, or None if no persona is set.

        Example:
            ```python
            persona = await harness.get_active_persona()
            if persona:
                print(persona.content)
            ```
        """
        from memharness.types import MemoryType

        namespace = self._build_namespace(MemoryType.PERSONA)
        results = await self._backend.list_by_namespace(
            namespace=namespace,
            memory_type=MemoryType.PERSONA,
            limit=1,
        )
        return results[0] if results else None
