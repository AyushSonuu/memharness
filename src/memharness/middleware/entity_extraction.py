# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Entity extraction middleware for LangChain agents.

This middleware automatically extracts named entities (people, organizations,
locations) from agent interactions and stores them in entity memory for
future reference and context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError as e:
    raise ImportError(
        "LangChain is required for middleware. Install with: pip install memharness[langchain]"
    ) from e

if TYPE_CHECKING:
    from langchain.agents.middleware.types import Runtime
    from langchain_core.language_models import BaseChatModel

    from memharness import MemoryHarness

__all__ = ["EntityExtractionMiddleware"]


class EntityExtractionMiddleware(AgentMiddleware):
    """
    Middleware that extracts and stores entities from agent interactions.

    This middleware uses the EntityExtractorAgent to identify named entities
    (people, organizations, locations) from both user messages and AI responses,
    then stores them in entity memory for future retrieval.

    Example:
        ```python
        from memharness import MemoryHarness
        from memharness.middleware import EntityExtractionMiddleware
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model

        harness = MemoryHarness('sqlite:///memory.db')
        await harness.connect()

        # Optional: provide an LLM for better entity extraction
        llm = init_chat_model('gpt-4o')

        agent = create_agent(
            model='anthropic:claude-sonnet-4-6',
            tools=[...],
            middleware=[
                EntityExtractionMiddleware(
                    harness=harness,
                    thread_id='main',
                    llm=llm  # Optional
                )
            ]
        )
        ```
    """

    def __init__(
        self,
        harness: MemoryHarness,
        thread_id: str,
        llm: BaseChatModel | None = None,
        extract_from_user: bool = True,
        extract_from_ai: bool = True,
    ) -> None:
        """
        Initialize the entity extraction middleware.

        Args:
            harness: The MemoryHarness instance to store entities in.
            thread_id: The conversation thread ID for this agent session.
            llm: Optional LLM for intelligent entity extraction. If not provided,
                uses regex-based heuristics.
            extract_from_user: Whether to extract entities from user messages.
            extract_from_ai: Whether to extract entities from AI responses.
        """
        super().__init__()
        self.harness = harness
        self.thread_id = thread_id
        self.llm = llm
        self.extract_from_user = extract_from_user
        self.extract_from_ai = extract_from_ai
        self._processed_message_ids: set[str] = set()

        # Lazy-load the entity extractor
        self._extractor = None

    def _get_extractor(self):
        """Lazy-load the EntityExtractorAgent."""
        if self._extractor is None:
            from memharness.agents import EntityExtractorAgent

            self._extractor = EntityExtractorAgent(harness=self.harness, llm=self.llm)
        return self._extractor

    async def aafter_model(
        self, state: dict[str, Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """
        Extract entities after the model responds.

        Analyzes both user messages and AI responses for named entities,
        then stores them in the entity memory store.

        Args:
            state: The current agent state containing messages.
            runtime: The runtime context (unused).

        Returns:
            None (no state updates needed).
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        try:
            extractor = self._get_extractor()

            # Process messages from the end (most recent)
            for msg in reversed(messages):
                msg_id = getattr(msg, "id", None)
                if msg_id and msg_id in self._processed_message_ids:
                    # Already processed this message
                    continue

                # Determine if we should process this message
                should_process = False
                if isinstance(msg, HumanMessage) and self.extract_from_user:
                    should_process = True
                elif isinstance(msg, AIMessage) and self.extract_from_ai:
                    should_process = True

                if not should_process:
                    continue

                # Extract content
                content = msg.content if hasattr(msg, "content") else str(msg)
                if not content or not isinstance(content, str):
                    continue

                # Extract entities
                entities = await extractor.extract_entities(content)

                # Store entities in memory
                for entity_type, entity_list in entities.items():
                    for entity_name in entity_list:
                        if not entity_name or not entity_name.strip():
                            continue

                        # Create entity description with context
                        description = f"Mentioned in conversation (thread: {self.thread_id})"

                        # Store in entity memory
                        await self.harness.add_entity(
                            name=entity_name,
                            entity_type=entity_type,
                            description=description,
                        )

                # Mark this message as processed
                if msg_id:
                    self._processed_message_ids.add(msg_id)

        except Exception:
            # If extraction fails, continue without raising
            # (don't break the agent flow)
            pass

        return None
