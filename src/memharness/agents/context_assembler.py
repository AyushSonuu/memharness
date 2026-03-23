# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Context Assembly Agent for memharness.

This module implements the BEFORE-loop pattern from the agent memory course:
the Context Assembly Agent takes conversation history + user query and returns
optimal context from memory. This is the KEY feature for memory-aware agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memharness.core.harness import MemoryHarness

__all__ = ["AssembledContext", "ContextAssemblyAgent"]


@dataclass
class AssembledContext:
    """Structured context assembled from memory for LLM consumption.

    Provides both string rendering (to_prompt) and proper LangChain message
    list (to_messages) for direct use with any LLM provider.

    Attributes:
        persona: Agent persona/identity information.
        conversation_history: Raw MemoryUnit list from conversational memory.
        knowledge: Relevant knowledge base entries.
        workflows: Relevant workflow patterns.
        entities: Known entities related to the query.
        summaries: Compressed summaries if any.
        tools: Available tools from toolbox.
        user_query: The current user query.
        total_tokens_estimate: Rough token count (chars / 4).
        context_usage_percent: How full the context is (0.0 to 1.0).
    """

    persona: str = ""
    conversation_history: list[Any] = field(default_factory=list)  # list[MemoryUnit]
    knowledge: str = ""
    workflows: str = ""
    entities: str = ""
    summaries: str = ""
    tools: str = ""
    user_query: str = ""
    total_tokens_estimate: int = 0
    context_usage_percent: float = 0.0

    def to_messages(self) -> list[Any]:
        """Convert to a list of LangChain BaseMessage objects.

        Returns a proper message list that can be passed directly to any LLM:
        - SystemMessage with memory context (persona, knowledge, entities, etc.)
        - HumanMessage/AIMessage from conversation history (proper role mapping)
        - Final HumanMessage with the current query

        Returns:
            List of LangChain BaseMessage objects. If langchain-core is not
            installed, returns a list of dicts with 'role' and 'content' keys.
        """
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
            )

            use_langchain = True
        except ImportError:
            use_langchain = False

        messages: list[Any] = []

        # 1. System message with memory context
        context_sections = []
        if self.persona:
            context_sections.append(f"## Agent Persona\n{self.persona}")
        if self.knowledge:
            context_sections.append(f"## Relevant Knowledge\n{self.knowledge}")
        if self.workflows:
            context_sections.append(f"## Relevant Workflows\n{self.workflows}")
        if self.entities:
            context_sections.append(f"## Known Entities\n{self.entities}")
        if self.summaries:
            context_sections.append(f"## Context Summaries\n{self.summaries}")
        if self.tools:
            context_sections.append(f"## Available Tools\n{self.tools}")

        if context_sections:
            system_content = "\n\n".join(context_sections)
            if use_langchain:
                messages.append(SystemMessage(content=system_content))
            else:
                messages.append({"role": "system", "content": system_content})

        # 2. Conversation history as proper message objects
        for mem in self.conversation_history:
            role = mem.metadata.get("role", "user") if hasattr(mem, "metadata") else "user"
            content = mem.content if hasattr(mem, "content") else str(mem)

            if use_langchain:
                if role in ("user", "human"):
                    messages.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
            else:
                messages.append({"role": role, "content": content})

        return messages

    def to_prompt(self) -> str:
        """Render as markdown-sectioned prompt string for LLM.

        Returns:
            Formatted markdown string ready for LLM consumption.
        """
        sections = []
        if self.persona:
            sections.append(f"## Agent Persona\n{self.persona}")

        if self.conversation_history:
            conv_lines = []
            for m in self.conversation_history:
                role = m.metadata.get("role", "user") if hasattr(m, "metadata") else "user"
                content = m.content if hasattr(m, "content") else str(m)
                conv_lines.append(f"{role}: {content}")
            sections.append("## Conversation History\n" + "\n".join(conv_lines))

        if self.knowledge:
            sections.append(f"## Relevant Knowledge\n{self.knowledge}")

        if self.workflows:
            sections.append(f"## Relevant Workflows\n{self.workflows}")

        if self.entities:
            sections.append(f"## Known Entities\n{self.entities}")

        if self.summaries:
            sections.append(f"## Context Summaries\n{self.summaries}")

        if self.tools:
            sections.append(f"## Available Tools\n{self.tools}")

        sections.append(f"## User Query\n{self.user_query}")

        return "\n\n".join(sections)


class ContextAssemblyAgent:
    """
    Assembles optimal context from memory for the main agent.

    Implements the BEFORE-loop pattern from the agent memory course (L06):
    1. Save incoming query to conversational memory
    2. Deterministic reads across all memory types
    3. Context size check (summarize if needed)
    4. Tool discovery
    5. Return structured AssembledContext

    This agent operates deterministically (runs on every query) and provides
    the foundation for memory-aware agent execution.

    Attributes:
        harness: The MemoryHarness instance to read from.
        max_tokens: Maximum context size in tokens (default: 4000).
        summarize_threshold: Trigger summarization when context reaches this
                           percentage of max_tokens (default: 0.8 = 80%).

    Example:
        ```python
        agent = ContextAssemblyAgent(harness, max_tokens=4000)
        ctx = await agent.assemble(
            query="How do I deploy?",
            thread_id="chat-123"
        )
        prompt = ctx.to_prompt()
        # Send prompt to LLM
        ```
    """

    def __init__(
        self,
        harness: MemoryHarness,
        max_tokens: int = 4000,
        summarize_threshold: float = 0.8,
    ) -> None:
        """
        Initialize the Context Assembly Agent.

        Args:
            harness: The MemoryHarness instance.
            max_tokens: Maximum tokens for assembled context (default: 4000).
            summarize_threshold: Percentage threshold to trigger summarization
                               (default: 0.8 = 80%).
        """
        self.harness = harness
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold

    async def assemble(
        self,
        query: str,
        thread_id: str,
        include_tools: bool = True,
    ) -> AssembledContext:
        """
        Assemble context for a user query.

        This method implements the BEFORE-loop pattern:
        1. Write query to conversational memory (deterministic)
        2. Read all relevant memory types (deterministic)
        3. Check context size and summarize if needed
        4. Optionally discover tools
        5. Return structured context

        Args:
            query: The user's query.
            thread_id: The conversation thread ID.
            include_tools: Whether to include toolbox tree (default: True).

        Returns:
            AssembledContext with all relevant memories.

        Example:
            ```python
            ctx = await agent.assemble(
                query="Tell me about Python async",
                thread_id="thread-1"
            )
            print(f"Context usage: {ctx.context_usage_percent:.1%}")
            ```
        """
        ctx = AssembledContext(user_query=query)

        # 1. Save query to conversational memory (deterministic write)
        await self.harness.add_conversational(thread_id, "user", query)

        # 2. Deterministic reads across memory types

        # Persona (agent identity)
        persona = await self.harness.get_active_persona()
        if persona:
            ctx.persona = persona.content

        # Conversation history (raw MemoryUnits — converted to messages by to_messages())
        messages = await self.harness.get_conversational(thread_id, limit=20)
        if messages:
            ctx.conversation_history = list(messages)

        # Knowledge base (semantic search)
        kb_results = await self.harness.search_knowledge(query, k=5)
        if kb_results:
            ctx.knowledge = "\n".join(f"- {r.content}" for r in kb_results)

        # Workflows (learned patterns)
        wf_results = await self.harness.search_workflow(query, k=3)
        if wf_results:
            ctx.workflows = "\n".join(f"- {r.content}" for r in wf_results)

        # Entities (people, places, concepts) - prefer recent (updated_at DESC)
        # The backend search methods should sort by updated_at DESC to handle staleness
        # (e.g., prefer "works at SAP" over "works at Google" for the same person)
        entity_results = await self.harness.search_entity(query, k=5)
        if entity_results:
            ctx.entities = "\n".join(
                f"- {r.metadata.get('entity_name', '')}: {r.content}" for r in entity_results
            )

        # 3. Context size check
        ctx.total_tokens_estimate = len(ctx.to_prompt()) // 4
        ctx.context_usage_percent = ctx.total_tokens_estimate / self.max_tokens

        # If context too full, trigger summarization (agent-triggered from L06)
        if ctx.context_usage_percent >= self.summarize_threshold:
            # Truncate conversation to last 10 messages
            if messages and len(messages) > 10:
                ctx.conversation_history = list(messages[-10:])

        # 4. Tool discovery (if requested)
        if include_tools:
            try:
                tree = await self.harness.toolbox_tree("/")
                if tree:
                    ctx.tools = tree
            except Exception:
                # Toolbox may be empty or unavailable
                pass

        # Recalculate token estimate after potential truncation
        ctx.total_tokens_estimate = len(ctx.to_prompt()) // 4
        ctx.context_usage_percent = ctx.total_tokens_estimate / self.max_tokens

        return ctx
