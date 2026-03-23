# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Entity extraction agent for identifying and tracking entities in memory.

Extracts PERSON, ORG, PLACE, CONCEPT entities from text and creates/updates
entity memories with relationships and context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from memharness.agents.base import AgentConfig, EmbeddedAgent, TriggerType

if TYPE_CHECKING:
    from memharness import MemoryHarness


class EntityType(Enum):
    """Types of entities that can be extracted."""

    PERSON = "person"
    ORG = "organization"
    PLACE = "place"
    CONCEPT = "concept"
    PRODUCT = "product"
    EVENT = "event"
    DATE = "date"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""

    name: str
    entity_type: EntityType
    confidence: float  # 0.0 to 1.0
    context: str  # Surrounding text
    mentions: int = 1
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "mentions": self.mentions,
            "aliases": self.aliases,
            "metadata": self.metadata,
        }


class EntityExtractorAgent(EmbeddedAgent):
    """
    Agent that extracts entities from text and manages entity memories.

    Features:
    - Extracts PERSON, ORG, PLACE, CONCEPT entities
    - Creates and updates entity memories
    - Tracks entity relationships and mentions
    - Supports both on_write (real-time) and batch mode

    Without LLM:
    - Uses regex patterns and heuristics
    - Identifies capitalized sequences, common patterns
    - Detects known entity indicators

    With LLM:
    - Uses AI for accurate NER
    - Understands context for disambiguation
    - Extracts implicit relationships
    """

    trigger = TriggerType.ON_WRITE
    schedule = None

    # Patterns for deterministic extraction
    _PERSON_INDICATORS = [
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sir",
        "madam",
        "ceo",
        "cto",
        "cfo",
        "founder",
        "director",
        "manager",
        "said",
        "asked",
        "replied",
        "mentioned",
        "told",
    ]

    _ORG_INDICATORS = [
        "inc.",
        "corp.",
        "llc",
        "ltd.",
        "company",
        "corporation",
        "organization",
        "institute",
        "university",
        "foundation",
        "team",
        "department",
        "group",
        "agency",
    ]

    _PLACE_INDICATORS = [
        "city",
        "country",
        "state",
        "region",
        "district",
        "street",
        "avenue",
        "road",
        "building",
        "headquarters",
        "located in",
        "based in",
        "from",
        "at",
    ]

    _CONCEPT_INDICATORS = [
        "concept",
        "idea",
        "theory",
        "method",
        "approach",
        "strategy",
        "technique",
        "framework",
        "model",
        "paradigm",
        "algorithm",
        "process",
        "system",
        "architecture",
    ]

    def __init__(
        self,
        memory: MemoryHarness,
        llm: Any | None = None,
        config: AgentConfig | None = None,
    ):
        super().__init__(memory, llm, config)
        self._entity_cache: dict[str, ExtractedEntity] = {}

    @property
    def name(self) -> str:
        return "entity_extractor"

    async def run(
        self,
        content: str | None = None,
        memory_ids: list[str] | None = None,
        namespace: tuple[str, ...] | None = None,
        batch_mode: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run entity extraction.

        Args:
            content: Text to extract entities from (for on_write)
            memory_ids: Specific memory IDs to process (for batch)
            namespace: Namespace to operate in
            batch_mode: If True, process all unprocessed memories

        Returns:
            Dictionary with extraction results
        """
        started_at = datetime.now()
        errors: list[str] = []
        entities_extracted = 0
        entities_created = 0
        entities_updated = 0
        items_processed = 0

        try:
            if content:
                # On-write mode: extract from single content
                entities = await self._extract_entities(content)
                for entity in entities:
                    created = await self._upsert_entity(entity, namespace)
                    if created:
                        entities_created += 1
                    else:
                        entities_updated += 1
                    entities_extracted += 1
                items_processed = 1

            elif batch_mode or memory_ids:
                # Batch mode: process multiple memories
                memories = await self._get_memories_to_process(memory_ids, namespace)

                for memory in memories:
                    try:
                        memory_content = memory.get("content", "")
                        if not memory_content:
                            continue

                        entities = await self._extract_entities(memory_content)
                        for entity in entities:
                            # Link entity to source memory
                            entity.metadata["source_memory_id"] = memory.get("id")
                            created = await self._upsert_entity(entity, namespace)
                            if created:
                                entities_created += 1
                            else:
                                entities_updated += 1
                            entities_extracted += 1

                        # Mark memory as processed
                        await self._mark_as_processed(memory.get("id"))
                        items_processed += 1

                    except Exception as e:
                        errors.append(f"Memory {memory.get('id')}: {str(e)}")

            result = await self._create_result(
                success=len(errors) == 0,
                started_at=started_at,
                items_processed=items_processed,
                items_created=entities_created,
                items_updated=entities_updated,
                errors=errors,
                metadata={
                    "mode": "llm" if self.has_llm else "heuristic",
                    "entities_extracted": entities_extracted,
                },
            )
            await self._log_run(result)

            return result.to_dict()

        except Exception as e:
            result = await self._create_result(
                success=False,
                started_at=started_at,
                errors=[str(e)],
            )
            return result.to_dict()

    async def extract_from_text(
        self,
        text: str,
        store: bool = True,
        namespace: tuple[str, ...] | None = None,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from text.

        Args:
            text: Text to extract from
            store: Whether to store extracted entities
            namespace: Namespace for storage

        Returns:
            List of extracted entities
        """
        entities = await self._extract_entities(text)

        if store:
            for entity in entities:
                await self._upsert_entity(entity, namespace)

        return entities

    async def _extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text."""
        if self.has_llm:
            return await self._extract_with_llm(text)
        return self._extract_with_heuristics(text)

    async def _extract_with_llm(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using LLM."""
        prompt = f"""Extract all named entities from the following text. For each entity, identify:
1. The entity name (canonical form)
2. The type: PERSON, ORG (organization), PLACE, CONCEPT, PRODUCT, EVENT, or DATE
3. Any aliases or alternate names mentioned
4. Brief context about the entity from the text

Format your response as a list, one entity per line:
NAME | TYPE | ALIASES (comma-separated) | CONTEXT

Text:
{text}

Entities:"""

        try:
            if hasattr(self.llm, "generate"):
                response = await self.llm.generate(prompt)
            elif hasattr(self.llm, "complete"):
                response = await self.llm.complete(prompt)
            elif callable(self.llm):
                response = await self.llm(prompt)
                response = response if isinstance(response, str) else str(response)
            else:
                return self._extract_with_heuristics(text)

            return self._parse_llm_response(response, text)

        except Exception:
            return self._extract_with_heuristics(text)

    def _parse_llm_response(self, response: str, original_text: str) -> list[ExtractedEntity]:
        """Parse LLM response into entities."""
        entities: list[ExtractedEntity] = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue

            name = parts[0]
            type_str = parts[1].upper()

            # Map type string to enum
            type_map = {
                "PERSON": EntityType.PERSON,
                "ORG": EntityType.ORG,
                "ORGANIZATION": EntityType.ORG,
                "PLACE": EntityType.PLACE,
                "LOCATION": EntityType.PLACE,
                "CONCEPT": EntityType.CONCEPT,
                "PRODUCT": EntityType.PRODUCT,
                "EVENT": EntityType.EVENT,
                "DATE": EntityType.DATE,
            }
            entity_type = type_map.get(type_str, EntityType.UNKNOWN)

            aliases = []
            if len(parts) > 2 and parts[2]:
                aliases = [a.strip() for a in parts[2].split(",") if a.strip()]

            context = parts[3] if len(parts) > 3 else ""

            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                    confidence=0.9,  # High confidence for LLM
                    context=context,
                    aliases=aliases,
                )
            )

        return entities

    def _extract_with_heuristics(self, text: str) -> list[ExtractedEntity]:
        """
        Extract entities using rule-based heuristics.

        Strategies:
        1. Capitalized word sequences (proper nouns)
        2. Pattern matching for common entity formats
        3. Context clues from surrounding words
        """
        entities: list[ExtractedEntity] = []
        seen_names: set[str] = set()

        # Strategy 1: Find capitalized sequences
        # Matches sequences like "John Smith", "New York City", "OpenAI Inc."
        cap_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        for match in re.finditer(cap_pattern, text):
            name = match.group(1)

            # Skip common words that are often capitalized
            skip_words = {
                "The",
                "This",
                "That",
                "There",
                "Then",
                "They",
                "What",
                "When",
                "Where",
                "How",
                "Why",
            }
            if name in skip_words:
                continue

            if name.lower() not in seen_names:
                seen_names.add(name.lower())

                # Determine type from context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].lower()

                entity_type = self._infer_type_from_context(name, context)
                confidence = self._calculate_confidence(name, context, entity_type)

                if confidence > 0.3:  # Minimum threshold
                    entities.append(
                        ExtractedEntity(
                            name=name,
                            entity_type=entity_type,
                            confidence=confidence,
                            context=text[start:end],
                        )
                    )

        # Strategy 2: Email patterns (can indicate person/org)
        email_pattern = r"\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
        for match in re.finditer(email_pattern, text):
            match.group(1)
            domain = match.group(2)

            # Extract org from domain
            org_name = domain.split(".")[0].title()
            if org_name.lower() not in seen_names:
                seen_names.add(org_name.lower())
                entities.append(
                    ExtractedEntity(
                        name=org_name,
                        entity_type=EntityType.ORG,
                        confidence=0.6,
                        context=f"Email domain: {domain}",
                    )
                )

        # Strategy 3: URL patterns
        url_pattern = r"https?://(?:www\.)?([a-zA-Z0-9.-]+)"
        for match in re.finditer(url_pattern, text):
            domain = match.group(1)
            org_name = domain.split(".")[0].title()
            if org_name.lower() not in seen_names and len(org_name) > 2:
                seen_names.add(org_name.lower())
                entities.append(
                    ExtractedEntity(
                        name=org_name,
                        entity_type=EntityType.ORG,
                        confidence=0.5,
                        context=f"URL: {match.group(0)}",
                    )
                )

        # Strategy 4: Quoted terms (often concepts or specific names)
        quote_pattern = r'"([^"]+)"'
        for match in re.finditer(quote_pattern, text):
            term = match.group(1)
            if len(term) > 2 and term.lower() not in seen_names:
                seen_names.add(term.lower())

                # Quoted terms are often concepts
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)

                entities.append(
                    ExtractedEntity(
                        name=term,
                        entity_type=EntityType.CONCEPT,
                        confidence=0.5,
                        context=text[start:end],
                    )
                )

        return entities

    def _infer_type_from_context(self, name: str, context: str) -> EntityType:
        """Infer entity type from surrounding context."""
        context_lower = context.lower()
        name_lower = name.lower()

        # Check for person indicators
        for indicator in self._PERSON_INDICATORS:
            if indicator in context_lower:
                return EntityType.PERSON

        # Check for org indicators
        for indicator in self._ORG_INDICATORS:
            if indicator in context_lower or indicator in name_lower:
                return EntityType.ORG

        # Check for place indicators
        for indicator in self._PLACE_INDICATORS:
            if indicator in context_lower:
                return EntityType.PLACE

        # Check for concept indicators
        for indicator in self._CONCEPT_INDICATORS:
            if indicator in context_lower:
                return EntityType.CONCEPT

        # Default heuristics
        words = name.split()
        if len(words) == 2 and all(w[0].isupper() for w in words):
            # Two capitalized words often = person name
            return EntityType.PERSON

        return EntityType.UNKNOWN

    def _calculate_confidence(self, name: str, context: str, entity_type: EntityType) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.5  # Base confidence

        # Longer names are more likely to be entities
        if len(name) > 10:
            confidence += 0.1

        # Multiple words increase confidence
        words = name.split()
        if len(words) >= 2:
            confidence += 0.1

        # Known type increases confidence
        if entity_type != EntityType.UNKNOWN:
            confidence += 0.2

        # Cap at 0.8 for heuristic extraction
        return min(confidence, 0.8)

    async def _get_memories_to_process(
        self,
        memory_ids: list[str] | None,
        namespace: tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        """Get memories that need entity extraction."""
        if memory_ids:
            # Fetch specific memories
            # return await self.memory.get_many(memory_ids)
            return []

        # Get unprocessed memories
        # Would query for memories without "entities_extracted" flag
        return []

    async def _upsert_entity(
        self,
        entity: ExtractedEntity,
        namespace: tuple[str, ...] | None,
    ) -> bool:
        """
        Create or update an entity memory.

        Returns True if created, False if updated.
        """
        # Generate deterministic ID from entity name
        entity_id = hashlib.sha256(
            f"entity:{entity.name.lower()}:{entity.entity_type.value}".encode()
        ).hexdigest()[:16]

        # Check cache first
        cache_key = entity_id
        if cache_key in self._entity_cache:
            # Update existing
            existing = self._entity_cache[cache_key]
            existing.mentions += entity.mentions
            existing.aliases = list(set(existing.aliases + entity.aliases))
            # Would call memory.update()
            return False

        # Create new
        self._entity_cache[cache_key] = entity
        # Would call memory.add_entity()
        # await self.memory.add_entity(
        #     name=entity.name,
        #     entity_type=entity.entity_type.value,
        #     content=entity.context,
        #     metadata=entity.to_dict(),
        #     namespace=namespace,
        # )
        return True

    async def _mark_as_processed(self, memory_id: str | None) -> None:
        """Mark a memory as having had entities extracted."""
        if memory_id:
            # Would call memory.update()
            # await self.memory.update(
            #     memory_id,
            #     metadata={"entities_extracted": True}
            # )
            pass
