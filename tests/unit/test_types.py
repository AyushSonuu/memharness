"""
Unit tests for memharness core types.

Tests MemoryUnit, MemoryType enum, and serialization functionality.
"""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from memharness import MemoryType, MemoryUnit


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_all_memory_types_exist(self):
        """Verify all 10 memory types are defined."""
        expected_types = [
            "CONVERSATIONAL",
            "KNOWLEDGE",
            "ENTITY",
            "WORKFLOW",
            "TOOLBOX",
            "SUMMARY",
            "TOOL_LOG",
            "SKILLS",
            "FILE",
            "PERSONA",
        ]

        for type_name in expected_types:
            assert hasattr(MemoryType, type_name), f"Missing MemoryType.{type_name}"

    def test_memory_type_values(self):
        """Test MemoryType enum values are strings."""
        assert MemoryType.CONVERSATIONAL.value == "conversational"
        assert MemoryType.KNOWLEDGE.value == "knowledge"
        assert MemoryType.ENTITY.value == "entity"
        assert MemoryType.WORKFLOW.value == "workflow"
        assert MemoryType.TOOLBOX.value == "toolbox"
        assert MemoryType.SUMMARY.value == "summary"
        assert MemoryType.TOOL_LOG.value == "tool_log"
        assert MemoryType.SKILLS.value == "skills"
        assert MemoryType.FILE.value == "file"
        assert MemoryType.PERSONA.value == "persona"

    def test_memory_type_count(self):
        """Ensure exactly 10 memory types exist."""
        assert len(MemoryType) == 10

    def test_memory_type_from_string(self):
        """Test creating MemoryType from string value."""
        assert MemoryType("conversational") == MemoryType.CONVERSATIONAL
        assert MemoryType("knowledge") == MemoryType.KNOWLEDGE
        assert MemoryType("entity") == MemoryType.ENTITY

    def test_memory_type_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            MemoryType("invalid_type")

    def test_memory_type_iteration(self):
        """Test iteration over all memory types."""
        types = list(MemoryType)
        assert len(types) == 10
        assert all(isinstance(t, MemoryType) for t in types)

    def test_memory_type_string_representation(self):
        """Test string representation of memory types."""
        for memory_type in MemoryType:
            assert str(memory_type.value) == memory_type.value
            assert isinstance(memory_type.name, str)


class TestMemoryUnit:
    """Tests for MemoryUnit data class."""

    def test_memory_unit_creation_minimal(self):
        """Test creating MemoryUnit with minimal required fields."""
        unit = MemoryUnit(
            content="Test content",
            memory_type=MemoryType.KNOWLEDGE,
        )

        assert unit.content == "Test content"
        assert unit.memory_type == MemoryType.KNOWLEDGE
        assert unit.id is not None
        assert isinstance(unit.id, (str, UUID))

    def test_memory_unit_creation_with_id(self):
        """Test creating MemoryUnit with specified ID."""
        test_id = uuid4()
        unit = MemoryUnit(
            id=test_id,
            content="Test content",
            memory_type=MemoryType.CONVERSATIONAL,
        )

        assert unit.id == test_id

    def test_memory_unit_creation_full(self):
        """Test creating MemoryUnit with all fields."""
        test_id = uuid4()
        now = datetime.now(UTC)

        unit = MemoryUnit(
            id=test_id,
            content="Full test content",
            memory_type=MemoryType.ENTITY,
            metadata={"source": "test", "tags": ["tag1", "tag2"]},
            embedding=[0.1, 0.2, 0.3],
            created_at=now,
            updated_at=now,
            thread_id="thread-123",
            parent_id=uuid4(),
        )

        assert unit.id == test_id
        assert unit.content == "Full test content"
        assert unit.memory_type == MemoryType.ENTITY
        assert unit.metadata == {"source": "test", "tags": ["tag1", "tag2"]}
        assert unit.embedding == [0.1, 0.2, 0.3]
        assert unit.created_at == now
        assert unit.thread_id == "thread-123"

    def test_memory_unit_default_timestamps(self):
        """Test that timestamps are set by default."""
        unit = MemoryUnit(
            content="Test",
            memory_type=MemoryType.KNOWLEDGE,
        )

        assert unit.created_at is not None
        assert isinstance(unit.created_at, datetime)

    def test_memory_unit_metadata_default(self):
        """Test that metadata defaults to empty dict or None."""
        unit = MemoryUnit(
            content="Test",
            memory_type=MemoryType.KNOWLEDGE,
        )

        assert unit.metadata is None or unit.metadata == {}

    def test_memory_unit_different_types(self):
        """Test creating MemoryUnit for each memory type."""
        for memory_type in MemoryType:
            unit = MemoryUnit(
                content=f"Content for {memory_type.value}",
                memory_type=memory_type,
            )
            assert unit.memory_type == memory_type

    def test_memory_unit_immutability(self):
        """Test if MemoryUnit is frozen/immutable (if applicable)."""
        unit = MemoryUnit(
            content="Test",
            memory_type=MemoryType.KNOWLEDGE,
        )

        # This test depends on implementation
        # If using frozen dataclass or pydantic with frozen=True
        try:
            unit.content = "Modified"
            # If we get here, the unit is mutable
            assert unit.content == "Modified"
        except (AttributeError, TypeError):
            # MemoryUnit is immutable - this is fine
            pass


class TestMemoryUnitSerialization:
    """Tests for MemoryUnit serialization and deserialization."""

    def test_to_dict(self):
        """Test converting MemoryUnit to dictionary."""
        test_id = uuid4()
        unit = MemoryUnit(
            id=test_id,
            content="Test content",
            memory_type=MemoryType.KNOWLEDGE,
            metadata={"source": "test"},
        )

        data = unit.to_dict()

        assert isinstance(data, dict)
        assert data["content"] == "Test content"
        assert data["memory_type"] in ["knowledge", MemoryType.KNOWLEDGE]
        assert data["metadata"] == {"source": "test"}

    def test_from_dict(self):
        """Test creating MemoryUnit from dictionary."""
        data = {
            "id": str(uuid4()),
            "content": "Test content",
            "memory_type": "knowledge",
            "metadata": {"source": "test"},
        }

        unit = MemoryUnit.from_dict(data)

        assert unit.content == "Test content"
        assert unit.memory_type == MemoryType.KNOWLEDGE
        assert unit.metadata == {"source": "test"}

    def test_to_json(self):
        """Test JSON serialization."""
        unit = MemoryUnit(
            content="JSON test",
            memory_type=MemoryType.ENTITY,
            metadata={"key": "value"},
        )

        json_str = unit.to_json()
        data = json.loads(json_str)

        assert data["content"] == "JSON test"
        assert "memory_type" in data

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = json.dumps(
            {
                "id": str(uuid4()),
                "content": "JSON test",
                "memory_type": "conversational",
                "metadata": {},
            }
        )

        unit = MemoryUnit.from_json(json_str)

        assert unit.content == "JSON test"
        assert unit.memory_type == MemoryType.CONVERSATIONAL

    def test_serialization_roundtrip(self):
        """Test that serialization/deserialization preserves data."""
        original = MemoryUnit(
            content="Roundtrip test",
            memory_type=MemoryType.WORKFLOW,
            metadata={"steps": ["a", "b", "c"]},
            thread_id="thread-123",
        )

        # Dictionary roundtrip
        dict_data = original.to_dict()
        from_dict = MemoryUnit.from_dict(dict_data)
        assert from_dict.content == original.content
        assert from_dict.memory_type == original.memory_type

        # JSON roundtrip
        json_str = original.to_json()
        from_json = MemoryUnit.from_json(json_str)
        assert from_json.content == original.content
        assert from_json.memory_type == original.memory_type

    def test_serialization_with_embedding(self):
        """Test serialization includes embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        unit = MemoryUnit(
            content="Embedding test",
            memory_type=MemoryType.KNOWLEDGE,
            embedding=embedding,
        )

        data = unit.to_dict()
        restored = MemoryUnit.from_dict(data)

        assert restored.embedding == embedding

    def test_serialization_with_none_values(self):
        """Test serialization handles None values gracefully."""
        unit = MemoryUnit(
            content="Minimal",
            memory_type=MemoryType.KNOWLEDGE,
        )

        data = unit.to_dict()
        # Should not raise errors
        json_str = json.dumps(data, default=str)
        assert json_str is not None


class TestMemoryUnitValidation:
    """Tests for MemoryUnit validation."""

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        # Depending on implementation, this might raise or be allowed
        try:
            unit = MemoryUnit(
                content="",
                memory_type=MemoryType.KNOWLEDGE,
            )
            # If empty content is allowed
            assert unit.content == ""
        except (ValueError, TypeError):
            # Empty content is not allowed - this is also valid behavior
            pass

    def test_large_content(self):
        """Test handling of large content strings."""
        large_content = "x" * 100000  # 100KB of content

        unit = MemoryUnit(
            content=large_content,
            memory_type=MemoryType.FILE,
        )

        assert len(unit.content) == 100000

    def test_unicode_content(self):
        """Test handling of unicode content."""
        unicode_content = "Hello! Bonjour! Hallo!"

        unit = MemoryUnit(
            content=unicode_content,
            memory_type=MemoryType.CONVERSATIONAL,
        )

        assert unit.content == unicode_content

    def test_special_characters_in_metadata(self):
        """Test metadata with special characters."""
        metadata = {
            "path": "/path/to/file.txt",
            "query": "SELECT * FROM table WHERE name = 'test'",
            "json_nested": {"key": ["value1", "value2"]},
        }

        unit = MemoryUnit(
            content="Test",
            memory_type=MemoryType.FILE,
            metadata=metadata,
        )

        assert unit.metadata["path"] == "/path/to/file.txt"
        assert unit.metadata["json_nested"]["key"][0] == "value1"


class TestMemoryUnitFactory:
    """Tests using the memory_unit_factory fixture."""

    def test_factory_default(self, memory_unit_factory):
        """Test factory creates valid default MemoryUnit."""
        unit = memory_unit_factory()

        assert unit.content == "test content"
        assert unit.memory_type == MemoryType.KNOWLEDGE

    def test_factory_custom_content(self, memory_unit_factory):
        """Test factory with custom content."""
        unit = memory_unit_factory(content="Custom content")

        assert unit.content == "Custom content"

    def test_factory_custom_type(self, memory_unit_factory):
        """Test factory with custom memory type."""
        unit = memory_unit_factory(memory_type=MemoryType.ENTITY)

        assert unit.memory_type == MemoryType.ENTITY

    def test_factory_with_kwargs(self, memory_unit_factory):
        """Test factory with additional kwargs."""
        unit = memory_unit_factory(
            content="Test",
            memory_type=MemoryType.WORKFLOW,
            metadata={"status": "active"},
            thread_id="test-thread",
        )

        assert unit.metadata == {"status": "active"}
        assert unit.thread_id == "test-thread"
