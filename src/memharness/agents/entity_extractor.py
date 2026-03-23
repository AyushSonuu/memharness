# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Entity extractor agent for named entity recognition.

Extracts entities (people, places, organizations) from text using LLM or regex.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from memharness.core.harness import MemoryHarness


class EntityExtractorAgent:
    """
    Agent that extracts named entities from text.

    Works in two modes:
    1. Without LLM: Uses regex-based heuristics (capitalized words, @mentions)
    2. With LLM: Uses LangChain with structured output for NER

    Example:
        ```python
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("gpt-4o")
        agent = EntityExtractorAgent(harness, llm=llm)
        entities = await agent.extract_entities("John works at Acme Corp in NYC")
        # Returns: {"people": ["John"], "organizations": ["Acme Corp"], "locations": ["NYC"]}
        ```
    """

    def __init__(self, harness: MemoryHarness, llm: BaseChatModel | None = None) -> None:
        """
        Initialize the entity extractor agent.

        Args:
            harness: The MemoryHarness instance to operate on.
            llm: Optional LLM for intelligent entity extraction.
        """
        self.harness = harness
        self.llm = llm

    async def extract_entities(self, text: str) -> dict[str, list[str]]:
        """
        Extract entities from text.

        Args:
            text: The text to extract entities from.

        Returns:
            Dictionary with entity types as keys and lists of entities as values.
        """
        if not text.strip():
            return {"people": [], "organizations": [], "locations": []}

        if not self.llm:
            return self._heuristic_extraction(text)

        return await self._llm_extraction(text)

    def _heuristic_extraction(self, text: str) -> dict[str, list[str]]:
        """
        Extract entities using simple regex heuristics.

        Args:
            text: The text to extract entities from.

        Returns:
            Dictionary with entity types and extracted entities.
        """
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

        # Extract @mentions
        mentions = re.findall(r"@([a-zA-Z0-9_]+)", text)

        # Extract email addresses (potential people)
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        email_names = [email.split("@")[0].replace(".", " ").title() for email in emails]

        # Combine and deduplicate
        people = list(set(mentions + email_names))
        potential_entities = list(set(capitalized))

        # Simple heuristic: single words likely people, multi-word likely orgs/places
        locations = [e for e in potential_entities if len(e.split()) == 1 and len(e) <= 5]
        organizations = [e for e in potential_entities if len(e.split()) > 1]

        return {
            "people": people[:10],  # Limit to 10
            "organizations": organizations[:10],
            "locations": locations[:10],
        }

    async def _llm_extraction(self, text: str) -> dict[str, list[str]]:
        """
        Extract entities using LangChain with structured output.

        Args:
            text: The text to extract entities from.

        Returns:
            Dictionary with entity types and extracted entities.
        """
        try:
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError as e:
            raise ImportError("Install langchain-core: pip install memharness[langchain]") from e

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a named entity recognition system. Extract people, "
                    "organizations, and locations from the text. Return JSON with "
                    'keys: "people", "organizations", "locations" (each a list of strings).',
                ),
                ("user", "{text}"),
            ]
        )

        # Build chain with JSON output parser
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser

        # Extract entities
        try:
            result = await chain.ainvoke({"text": text})
            # Ensure all keys exist
            return {
                "people": result.get("people", [])[:10],
                "organizations": result.get("organizations", [])[:10],
                "locations": result.get("locations", [])[:10],
            }
        except Exception:
            # Fall back to heuristic on error
            return self._heuristic_extraction(text)

    async def run(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """
        Run the entity extractor agent.

        Args:
            text: The text to extract entities from.
            **kwargs: Additional arguments (ignored).

        Returns:
            Dictionary with 'entities' key containing extracted entities.
        """
        entities = await self.extract_entities(text)
        total = sum(len(v) for v in entities.values())

        return {"entities": entities, "total_extracted": total}

    async def extract_from_recent(
        self, since: datetime | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """
        Extract entities from recent conversation messages.

        Scans recent conversational messages (since the given timestamp) and
        extracts entities. For each entity found:
        - If entity exists by name: UPDATE it (refreshes updated_at)
        - If entity doesn't exist: INSERT it

        This ensures no duplicates and keeps entity freshness tracking.

        Args:
            since: Optional timestamp - only process messages after this time.
                  If None, processes all messages.
            limit: Maximum number of messages to process (default: 100).

        Returns:
            Dictionary with 'extracted' count and 'entities' list.

        Example:
            ```python
            # Process all new messages since last run
            result = await agent.extract_from_recent(since=last_run_time)
            print(f"Extracted {result['extracted']} entities")
            ```
        """
        extracted_count = 0
        all_entities: list[dict[str, str]] = []

        try:
            # Get recent conversational messages
            # Since we don't have a direct timestamp filter, we'll get recent messages
            # and filter them manually if needed
            from memharness.types import MemoryType

            # Use search_all to get all conversational memories
            # (In production, you'd add timestamp filtering at backend level)
            messages = await self.harness.search("", memory_type=MemoryType.CONVERSATIONAL, k=limit)

            # Filter by timestamp if provided
            if since:
                messages = [
                    m for m in messages if m.metadata.get("created_at", datetime.min) > since
                ]

            # Process each message
            for message in messages:
                # Skip messages that have already been processed for entities
                if message.metadata.get("entities_extracted"):
                    continue

                # Extract entities from message content
                entities = await self.extract_entities(message.content)

                # Process each entity type
                for category, names in entities.items():
                    for name in names:
                        if not name.strip():
                            continue

                        # Search for existing entity by name
                        existing = await self.harness.search_entity(name, k=1)

                        entity_id = None
                        if existing and existing[0].metadata.get("entity_name") == name:
                            # Update existing entity (refresh timestamp)
                            entity_id = existing[0].id
                            await self.harness.update(
                                entity_id,
                                content=f"{category}: {name}",
                                metadata={
                                    **existing[0].metadata,
                                    "entity_type": category,
                                    "updated_at": datetime.utcnow().isoformat(),
                                },
                            )
                        else:
                            # Insert new entity
                            entity_id = await self.harness.add_entity(
                                name=name, entity_type=category, content=f"{category}: {name}"
                            )

                        extracted_count += 1
                        all_entities.append({"id": entity_id, "name": name, "type": category})

                # Mark message as processed
                await self.harness.update(
                    message.id,
                    metadata={**message.metadata, "entities_extracted": True},
                )

        except Exception as e:
            import logging

            logging.error(f"extract_from_recent failed: {e}", exc_info=True)

        return {"extracted": extracted_count, "entities": all_entities}
