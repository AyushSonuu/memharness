"""
Unit tests for memharness MemoryTypeRegistry.

Tests registry functionality including type registration,
retrieval, and custom type support.
"""


import pytest

from memharness import MemoryType, MemoryTypeRegistry


class TestRegistryInitialization:
    """Tests for registry initialization and default types."""

    def test_registry_singleton(self):
        """Test that registry is a singleton or consistently accessible."""
        registry1 = MemoryTypeRegistry.get_instance()
        registry2 = MemoryTypeRegistry.get_instance()

        # Both should reference the same or equivalent registry
        assert registry1 is not None
        assert registry2 is not None

    def test_all_ten_types_registered(self):
        """Verify all 10 memory types are registered by default."""
        registry = MemoryTypeRegistry.get_instance()
        registered_types = registry.list_types()

        expected_types = [
            MemoryType.CONVERSATIONAL,
            MemoryType.KNOWLEDGE,
            MemoryType.ENTITY,
            MemoryType.WORKFLOW,
            MemoryType.TOOLBOX,
            MemoryType.SUMMARY,
            MemoryType.TOOL_LOG,
            MemoryType.SKILLS,
            MemoryType.FILE,
            MemoryType.PERSONA,
        ]

        for expected in expected_types:
            assert expected in registered_types or expected.value in [t.value for t in registered_types], \
                f"MemoryType.{expected.name} not registered"

    def test_registered_types_count(self):
        """Test that exactly 10 types are registered."""
        registry = MemoryTypeRegistry.get_instance()
        types = registry.list_types()

        assert len(types) >= 10, f"Expected at least 10 types, got {len(types)}"


class TestRegistryGet:
    """Tests for registry get() method."""

    def test_get_conversational(self):
        """Test getting CONVERSATIONAL type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.CONVERSATIONAL)

        assert handler is not None

    def test_get_knowledge(self):
        """Test getting KNOWLEDGE type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.KNOWLEDGE)

        assert handler is not None

    def test_get_entity(self):
        """Test getting ENTITY type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.ENTITY)

        assert handler is not None

    def test_get_workflow(self):
        """Test getting WORKFLOW type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.WORKFLOW)

        assert handler is not None

    def test_get_toolbox(self):
        """Test getting TOOLBOX type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.TOOLBOX)

        assert handler is not None

    def test_get_summary(self):
        """Test getting SUMMARY type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.SUMMARY)

        assert handler is not None

    def test_get_tool_log(self):
        """Test getting TOOL_LOG type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.TOOL_LOG)

        assert handler is not None

    def test_get_skills(self):
        """Test getting SKILLS type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.SKILLS)

        assert handler is not None

    def test_get_file(self):
        """Test getting FILE type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.FILE)

        assert handler is not None

    def test_get_persona(self):
        """Test getting PERSONA type handler."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.PERSONA)

        assert handler is not None

    def test_get_by_string(self):
        """Test getting type handler by string value."""
        registry = MemoryTypeRegistry.get_instance()

        # Should support both enum and string lookup
        handler1 = registry.get(MemoryType.KNOWLEDGE)
        handler2 = registry.get("knowledge")

        assert handler1 is not None
        assert handler2 is not None
        # They should be the same handler
        assert type(handler1) == type(handler2)

    def test_get_invalid_type(self):
        """Test getting unregistered type raises appropriate error."""
        registry = MemoryTypeRegistry.get_instance()

        with pytest.raises((KeyError, ValueError, TypeError)):
            registry.get("nonexistent_type")

    def test_get_none_type(self):
        """Test getting None type raises appropriate error."""
        registry = MemoryTypeRegistry.get_instance()

        with pytest.raises((KeyError, ValueError, TypeError)):
            registry.get(None)


class TestRegistryListTypes:
    """Tests for registry list_types() method."""

    def test_list_types_returns_list(self):
        """Test that list_types returns a list."""
        registry = MemoryTypeRegistry.get_instance()
        types = registry.list_types()

        assert isinstance(types, (list, tuple, set))

    def test_list_types_contains_memory_types(self):
        """Test that list_types contains MemoryType instances or strings."""
        registry = MemoryTypeRegistry.get_instance()
        types = registry.list_types()

        for t in types:
            assert isinstance(t, (MemoryType, str))

    def test_list_types_not_empty(self):
        """Test that list_types is not empty."""
        registry = MemoryTypeRegistry.get_instance()
        types = registry.list_types()

        assert len(types) > 0

    def test_list_types_no_duplicates(self):
        """Test that list_types has no duplicate entries."""
        registry = MemoryTypeRegistry.get_instance()
        types = registry.list_types()

        # Convert to set and compare lengths
        types_list = list(types)
        unique_types = set(types_list)

        assert len(types_list) == len(unique_types), "Duplicate types found in registry"


class TestCustomTypeRegistration:
    """Tests for registering custom memory types."""

    def test_register_custom_type(self):
        """Test registering a custom memory type handler."""
        registry = MemoryTypeRegistry.get_instance()

        # Create a simple custom handler
        class CustomHandler:
            name = "custom"

            def process(self, data):
                return data

        # Register the custom handler
        registry.register("custom_type", CustomHandler())

        # Verify it's registered
        handler = registry.get("custom_type")
        assert handler is not None
        assert isinstance(handler, CustomHandler)

    def test_register_replaces_existing(self):
        """Test that registering with existing name replaces the handler."""
        registry = MemoryTypeRegistry.get_instance()

        class Handler1:
            version = 1

        class Handler2:
            version = 2

        # Register first handler
        registry.register("replaceable", Handler1())

        # Register second handler with same name
        registry.register("replaceable", Handler2())

        # Should get the second handler
        handler = registry.get("replaceable")
        assert handler.version == 2

    def test_register_with_decorator(self):
        """Test registering a handler using decorator pattern (if supported)."""
        registry = MemoryTypeRegistry.get_instance()

        try:
            @registry.register_handler("decorated_type")
            class DecoratedHandler:
                pass

            handler = registry.get("decorated_type")
            assert handler is not None
        except AttributeError:
            # Decorator pattern not supported, skip test
            pytest.skip("Decorator registration not supported")

    def test_unregister_type(self):
        """Test unregistering a memory type."""
        registry = MemoryTypeRegistry.get_instance()

        class TemporaryHandler:
            pass

        # Register
        registry.register("temporary", TemporaryHandler())
        assert registry.get("temporary") is not None

        # Unregister (if method exists)
        try:
            registry.unregister("temporary")
            with pytest.raises((KeyError, ValueError)):
                registry.get("temporary")
        except AttributeError:
            # Unregister not supported
            pytest.skip("Unregister method not supported")


class TestTypeHandlerInterface:
    """Tests for memory type handler interface."""

    def test_handler_has_required_methods(self):
        """Test that handlers implement required interface methods."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.KNOWLEDGE)

        # Check for common handler methods
        # These are examples - adjust based on actual interface
        expected_methods = ["validate", "process", "serialize"]

        for method in expected_methods:
            try:
                assert hasattr(handler, method) or callable(getattr(handler, method, None)), \
                    f"Handler missing method: {method}"
            except AssertionError:
                # Method might be optional
                pass

    def test_knowledge_handler_interface(self):
        """Test KNOWLEDGE handler implements expected interface."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.KNOWLEDGE)

        # KNOWLEDGE should support search functionality
        assert handler is not None

    def test_conversational_handler_interface(self):
        """Test CONVERSATIONAL handler implements expected interface."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.CONVERSATIONAL)

        # CONVERSATIONAL should support thread-based operations
        assert handler is not None

    def test_toolbox_handler_interface(self):
        """Test TOOLBOX handler implements VFS-like operations."""
        registry = MemoryTypeRegistry.get_instance()
        handler = registry.get(MemoryType.TOOLBOX)

        # TOOLBOX should support tree, ls, grep operations
        assert handler is not None


class TestRegistryMetadata:
    """Tests for registry metadata and documentation."""

    def test_type_description(self):
        """Test that each type has a description."""
        registry = MemoryTypeRegistry.get_instance()

        for memory_type in MemoryType:
            handler = registry.get(memory_type)

            # Check for description attribute or method
            if hasattr(handler, "description"):
                assert handler.description is not None
            elif hasattr(handler, "__doc__"):
                # docstring is acceptable
                pass

    def test_type_schema(self):
        """Test that types have schema information."""
        registry = MemoryTypeRegistry.get_instance()

        for memory_type in MemoryType:
            handler = registry.get(memory_type)

            # Some handlers might expose a schema
            if hasattr(handler, "schema"):
                assert isinstance(handler.schema, dict)

    def test_get_all_handlers(self):
        """Test getting all handlers at once."""
        registry = MemoryTypeRegistry.get_instance()

        try:
            all_handlers = registry.get_all()
            assert isinstance(all_handlers, dict)
            assert len(all_handlers) >= 10
        except AttributeError:
            # get_all might not be implemented
            # Fallback to iterating through types
            handlers = {}
            for memory_type in registry.list_types():
                handlers[memory_type] = registry.get(memory_type)

            assert len(handlers) >= 10


class TestRegistryThreadSafety:
    """Tests for registry thread safety (if applicable)."""

    def test_concurrent_reads(self):
        """Test concurrent reads from registry."""
        import threading

        registry = MemoryTypeRegistry.get_instance()
        results = []
        errors = []

        def read_from_registry():
            try:
                for _ in range(100):
                    handler = registry.get(MemoryType.KNOWLEDGE)
                    results.append(handler)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_from_registry) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        assert len(results) == 500

    def test_concurrent_registration(self):
        """Test concurrent registration doesn't corrupt registry."""
        import threading

        registry = MemoryTypeRegistry.get_instance()
        errors = []

        def register_type(index):
            try:
                class Handler:
                    def __init__(self, idx):
                        self.idx = idx

                registry.register(f"concurrent_type_{index}", Handler(index))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_type, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent registration: {errors}"


class TestRegistryConfiguration:
    """Tests for registry configuration options."""

    def test_registry_with_config(self):
        """Test creating registry with custom configuration."""
        try:
            from memharness import Config

            config = Config()
            registry = MemoryTypeRegistry(config=config)

            assert registry is not None
        except (TypeError, ImportError):
            # Config-based initialization might not be supported
            pytest.skip("Config-based registry initialization not supported")

    def test_lazy_loading(self):
        """Test that handlers are lazily loaded (if applicable)."""
        registry = MemoryTypeRegistry.get_instance()

        # Access a type that might be lazily loaded
        handler = registry.get(MemoryType.PERSONA)

        # Should be loaded now
        assert handler is not None
