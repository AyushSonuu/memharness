# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Context assembly for memharness.

This module provides context assembly functionality for gathering
relevant memories to use in agent prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = ["ContextMixin"]


class ContextMixin:
    """Mixin for context assembly operations."""

    async def assemble_context(
        self,
        query: str,
        thread_id: str,
        max_tokens: int = 4000,
    ) -> str:
        """
        Assemble relevant context for an agent query.

        This method gathers relevant memories from multiple sources:
        - Recent conversation history
        - Relevant knowledge base entries
        - Matching entities
        - Applicable workflows
        - Persona information

        Args:
            query: The query to assemble context for.
            thread_id: The conversation thread ID.
            max_tokens: Maximum tokens in the assembled context.

        Returns:
            A formatted context string ready to be used in a prompt.

        Example:
            ```python
            context = await harness.assemble_context(
                query="How do I deploy the application?",
                thread_id="chat-123",
                max_tokens=4000
            )
            # Use context in your prompt
            prompt = f"{context}\\n\\nUser: {query}"
            ```
        """
        # Type checking - self should have these methods from mixins
        self_mixin = self  # type: BaseMixin  # type: ignore[assignment]

        sections = []
        estimated_tokens = 0
        chars_per_token = 4  # Rough estimate

        # 1. Persona (always include, typically small)
        persona = await self_mixin.get_persona()  # type: ignore[attr-defined]
        if persona:
            sections.append(f"## Persona\n{persona}")
            estimated_tokens += len(persona) // chars_per_token

        # 2. Recent conversation history
        if estimated_tokens < max_tokens:
            messages = await self_mixin.get_conversational(thread_id, limit=10)  # type: ignore[attr-defined]
            if messages:
                conv_text = "\n".join(
                    f"{m.metadata.get('role', 'unknown')}: {m.content}"
                    for m in messages[-5:]  # Last 5 messages
                )
                sections.append(f"## Recent Conversation\n{conv_text}")
                estimated_tokens += len(conv_text) // chars_per_token

        # 3. Relevant knowledge
        if estimated_tokens < max_tokens:
            knowledge = await self_mixin.search_knowledge(query, k=3)  # type: ignore[attr-defined]
            if knowledge:
                kb_text = "\n\n".join(f"- {k.content}" for k in knowledge)
                sections.append(f"## Relevant Knowledge\n{kb_text}")
                estimated_tokens += len(kb_text) // chars_per_token

        # 4. Relevant entities
        if estimated_tokens < max_tokens:
            entities = await self_mixin.search_entity(query, k=3)  # type: ignore[attr-defined]
            if entities:
                ent_text = "\n".join(
                    f"- {e.metadata.get('name', 'Unknown')}: {e.content}" for e in entities
                )
                sections.append(f"## Related Entities\n{ent_text}")
                estimated_tokens += len(ent_text) // chars_per_token

        # 5. Relevant workflows
        if estimated_tokens < max_tokens:
            workflows = await self_mixin.search_workflow(query, k=2)  # type: ignore[attr-defined]
            if workflows:
                wf_text = "\n\n".join(
                    f"**{w.metadata.get('task', 'Task')}**\n{w.content}" for w in workflows
                )
                sections.append(f"## Relevant Workflows\n{wf_text}")
                estimated_tokens += len(wf_text) // chars_per_token

        # 6. Relevant skills
        if estimated_tokens < max_tokens:
            skills = await self_mixin.search_skills(query, k=2)  # type: ignore[attr-defined]
            if skills:
                skills_text = "\n".join(
                    f"- {s.metadata.get('name', 'Skill')}: {s.metadata.get('description', '')}"
                    for s in skills
                )
                sections.append(f"## Available Skills\n{skills_text}")

        return "\n\n".join(sections)
