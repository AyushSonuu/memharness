# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Entity extractor agent for named entity recognition.

Extracts entities (people, places, organizations) from text using LLM or regex.
"""

from __future__ import annotations

import re
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
