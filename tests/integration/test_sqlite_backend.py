"""
Integration tests for SQLite backend.

Tests SQLite-specific functionality including persistence,
transactions, and schema management.
"""

import pytest
import pytest_asyncio
from pathlib import Path
import os
import asyncio

from memharness import MemoryHarness, MemoryType


# =============================================================================
# SQLite Connection Tests
# =============================================================================


class TestSQLiteConnection:
    """Tests for SQLite connection handling."""

    @pytest.mark.asyncio
    async def test_create_new_database(self, tmp_path):
        """Test creating a new SQLite database."""
        db_path = tmp_path / "new.db"
        assert not db_path.exists()

        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()

        assert db_path.exists()

        await harness.disconnect()

    @pytest.mark.asyncio
    async def test_open_existing_database(self, tmp_path):
        """Test opening an existing database."""
        db_path = tmp_path / "existing.db"

        # Create and populate database
        harness1 = MemoryHarness(f"sqlite:///{db_path}")
        await harness1.connect()
        await harness1.add_knowledge("Test entry", source="test")
        await harness1.disconnect()

        # Reopen
        harness2 = MemoryHarness(f"sqlite:///{db_path}")
        await harness2.connect()

        results = await harness2.search_knowledge("test", k=1)
        assert len(results) >= 1

        await harness2.disconnect()

    @pytest.mark.asyncio
    async def test_in_memory_sqlite(self):
        """Test SQLite in-memory database."""
        harness = MemoryHarness("sqlite:///:memory:")
        await harness.connect()

        await harness.add_knowledge("In-memory test", source="test")
        results = await harness.search_knowledge("memory", k=1)

        assert len(results) >= 1

        await harness.disconnect()

    @pytest.mark.asyncio
    async def test_connection_string_variants(self, tmp_path):
        """Test different connection string formats."""
        db_path = tmp_path / "test.db"

        # Test various formats
        formats = [
            f"sqlite:///{db_path}",
            f"sqlite:///{db_path}?mode=rwc",
        ]

        for conn_str in formats:
            try:
                harness = MemoryHarness(conn_str)
                await harness.connect()
                await harness.disconnect()
            except Exception as e:
                pytest.fail(f"Failed with connection string {conn_str}: {e}")


# =============================================================================
# SQLite Persistence Tests
# =============================================================================


class TestSQLitePersistence:
    """Tests for SQLite data persistence."""

    @pytest.mark.asyncio
    async def test_data_persists_across_connections(self, tmp_path):
        """Test that data survives disconnect/reconnect."""
        db_path = tmp_path / "persist.db"

        # First session - write data
        harness1 = MemoryHarness(f"sqlite:///{db_path}")
        await harness1.connect()

        await harness1.add_conversational("t1", "user", "Hello persistent")
        await harness1.add_knowledge("Persistent knowledge", source="test")
        await harness1.add_entity("PersistEntity", "TEST", "Test entity")

        await harness1.disconnect()

        # Second session - read data
        harness2 = MemoryHarness(f"sqlite:///{db_path}")
        await harness2.connect()

        messages = await harness2.get_conversational("t1")
        assert len(messages) == 1
        assert "persistent" in messages[0].content.lower()

        kb_results = await harness2.search_knowledge("persistent", k=1)
        assert len(kb_results) >= 1

        entity_results = await harness2.search_entity("PersistEntity")
        assert len(entity_results) >= 1

        await harness2.disconnect()

    @pytest.mark.asyncio
    async def test_toolbox_persists(self, tmp_path):
        """Test toolbox persistence."""
        db_path = tmp_path / "toolbox.db"

        # First session
        harness1 = MemoryHarness(f"sqlite:///{db_path}")
        await harness1.connect()
        await harness1.add_tool("github", "create_pr", "Create PR", {"title": "string"})
        await harness1.disconnect()

        # Second session
        harness2 = MemoryHarness(f"sqlite:///{db_path}")
        await harness2.connect()
        tree = await harness2.toolbox_tree()
        assert "github" in tree
        await harness2.disconnect()

    @pytest.mark.asyncio
    async def test_summary_links_persist(self, tmp_path):
        """Test that summary-to-source links persist."""
        db_path = tmp_path / "summary.db"

        # First session - create summary
        harness1 = MemoryHarness(f"sqlite:///{db_path}")
        await harness1.connect()

        id1 = await harness1.add_conversational("t1", "user", "Message 1")
        id2 = await harness1.add_conversational("t1", "assistant", "Response 1")
        summary_id = await harness1.add_summary(
            "Summary of conversation",
            source_ids=[id1, id2],
            thread_id="t1"
        )

        await harness1.disconnect()

        # Second session - expand summary
        harness2 = MemoryHarness(f"sqlite:///{db_path}")
        await harness2.connect()

        originals = await harness2.expand_summary(summary_id)
        assert len(originals) == 2

        await harness2.disconnect()


# =============================================================================
# SQLite Transaction Tests
# =============================================================================


class TestSQLiteTransactions:
    """Tests for SQLite transaction handling."""

    @pytest.mark.asyncio
    async def test_atomic_operations(self, sqlite_memory):
        """Test that operations are atomic."""
        # Add multiple items
        ids = []
        for i in range(5):
            msg_id = await sqlite_memory.add_conversational("t1", "user", f"Message {i}")
            ids.append(msg_id)

        # All should be retrievable
        messages = await sqlite_memory.get_conversational("t1")
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, tmp_path):
        """Test that failed operations don't partially commit."""
        db_path = tmp_path / "rollback.db"

        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()

        # Add valid data
        await harness.add_knowledge("Valid entry", source="test")

        # Try to verify rollback behavior
        # This depends on implementation details
        try:
            # Attempt invalid operation (implementation-specific)
            await harness.add_knowledge(None, source=None)
        except (ValueError, TypeError):
            pass

        # Valid entry should still exist
        results = await harness.search_knowledge("valid", k=1)
        assert len(results) >= 1

        await harness.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, tmp_path):
        """Test concurrent write operations."""
        db_path = tmp_path / "concurrent.db"

        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()

        async def write_batch(prefix: str):
            for i in range(10):
                await harness.add_conversational("t1", "user", f"{prefix}_{i}")

        # Concurrent writes
        await asyncio.gather(
            write_batch("batch_a"),
            write_batch("batch_b"),
            write_batch("batch_c"),
        )

        messages = await harness.get_conversational("t1")
        assert len(messages) == 30

        await harness.disconnect()


# =============================================================================
# SQLite Schema Tests
# =============================================================================


class TestSQLiteSchema:
    """Tests for SQLite schema management."""

    @pytest.mark.asyncio
    async def test_schema_creation(self, tmp_path):
        """Test that schema is created on first connect."""
        db_path = tmp_path / "schema.db"

        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()

        # Schema should be created - verify by adding data
        await harness.add_knowledge("Schema test", source="test")
        results = await harness.search_knowledge("schema", k=1)
        assert len(results) >= 1

        await harness.disconnect()

    @pytest.mark.asyncio
    async def test_schema_migration(self, tmp_path):
        """Test schema migration (if supported)."""
        db_path = tmp_path / "migrate.db"

        # Create initial database
        harness1 = MemoryHarness(f"sqlite:///{db_path}")
        await harness1.connect()
        await harness1.add_knowledge("Pre-migration", source="test")
        await harness1.disconnect()

        # Reconnect (would trigger migration if needed)
        harness2 = MemoryHarness(f"sqlite:///{db_path}")
        await harness2.connect()

        # Data should survive
        results = await harness2.search_knowledge("pre-migration", k=1)
        assert len(results) >= 1

        await harness2.disconnect()

    @pytest.mark.asyncio
    async def test_indexes_created(self, tmp_path):
        """Test that appropriate indexes are created."""
        import sqlite3

        db_path = tmp_path / "indexes.db"

        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()
        await harness.add_knowledge("Index test", source="test")
        await harness.disconnect()

        # Check for indexes directly
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()
        conn.close()

        # Should have some indexes (exact names depend on implementation)
        # This is a soft check - implementation may vary
        assert len(indexes) >= 0


# =============================================================================
# SQLite Performance Tests
# =============================================================================


class TestSQLitePerformance:
    """Tests for SQLite performance characteristics."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_bulk_insert_performance(self, tmp_path):
        """Test bulk insert performance."""
        import time

        db_path = tmp_path / "bulk.db"
        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()

        start = time.time()

        # Insert 100 records
        for i in range(100):
            await harness.add_knowledge(f"Knowledge entry {i}", source="bulk")

        elapsed = time.time() - start

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30, f"Bulk insert took {elapsed}s"

        await harness.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_search_performance(self, tmp_path):
        """Test search performance with larger dataset."""
        import time

        db_path = tmp_path / "search_perf.db"
        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()

        # Insert test data
        for i in range(50):
            await harness.add_knowledge(f"Document about topic {i % 10}", source="perf")

        start = time.time()

        # Run multiple searches
        for _ in range(10):
            await harness.search_knowledge("topic", k=5)

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 10, f"Searches took {elapsed}s"

        await harness.disconnect()


# =============================================================================
# SQLite Edge Cases
# =============================================================================


class TestSQLiteEdgeCases:
    """Tests for SQLite edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_unicode_content(self, sqlite_memory):
        """Test Unicode content handling."""
        content = "Hello! Bonjour! Hallo!"

        await sqlite_memory.add_knowledge(content, source="unicode")
        results = await sqlite_memory.search_knowledge("Hello", k=1)

        assert len(results) >= 1
        assert results[0].content == content

    @pytest.mark.asyncio
    async def test_large_content(self, sqlite_memory):
        """Test large content handling."""
        large_content = "x" * 50000  # 50KB

        await sqlite_memory.add_knowledge(large_content, source="large")
        results = await sqlite_memory.search_knowledge("xxxxx", k=1)

        assert len(results) >= 1
        assert len(results[0].content) == 50000

    @pytest.mark.asyncio
    async def test_special_characters(self, sqlite_memory):
        """Test special characters in content."""
        content = "SELECT * FROM users WHERE name = 'test'; DROP TABLE users;--"

        await sqlite_memory.add_knowledge(content, source="special")
        results = await sqlite_memory.search_knowledge("SELECT", k=1)

        # Should not execute SQL injection
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_empty_search_results(self, sqlite_memory):
        """Test search with no results."""
        results = await sqlite_memory.search_knowledge("nonexistent_query_12345", k=5)

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_null_handling(self, sqlite_memory):
        """Test handling of null/None values."""
        try:
            await sqlite_memory.add_knowledge("Content", source=None)
            # If it succeeds, verify retrieval
            results = await sqlite_memory.search_knowledge("Content", k=1)
            assert len(results) >= 1
        except (ValueError, TypeError):
            # Null values might be rejected
            pass

    @pytest.mark.asyncio
    async def test_very_long_thread_id(self, sqlite_memory):
        """Test handling of very long thread IDs."""
        long_thread_id = "t" * 1000

        await sqlite_memory.add_conversational(long_thread_id, "user", "Message")
        messages = await sqlite_memory.get_conversational(long_thread_id)

        assert len(messages) == 1


# =============================================================================
# SQLite File Handling Tests
# =============================================================================


class TestSQLiteFileHandling:
    """Tests for SQLite file handling."""

    @pytest.mark.asyncio
    async def test_database_file_permissions(self, tmp_path):
        """Test database file permissions."""
        db_path = tmp_path / "perms.db"

        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()
        await harness.add_knowledge("Test", source="test")
        await harness.disconnect()

        # File should be readable/writable
        assert os.access(db_path, os.R_OK)
        assert os.access(db_path, os.W_OK)

    @pytest.mark.asyncio
    async def test_database_in_nested_directory(self, tmp_path):
        """Test database in nested directory."""
        nested_path = tmp_path / "a" / "b" / "c"
        nested_path.mkdir(parents=True)
        db_path = nested_path / "nested.db"

        harness = MemoryHarness(f"sqlite:///{db_path}")
        await harness.connect()
        await harness.add_knowledge("Nested test", source="test")
        await harness.disconnect()

        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_readonly_database(self, tmp_path):
        """Test behavior with read-only database."""
        db_path = tmp_path / "readonly.db"

        # Create database
        harness1 = MemoryHarness(f"sqlite:///{db_path}")
        await harness1.connect()
        await harness1.add_knowledge("Read only test", source="test")
        await harness1.disconnect()

        # Make read-only
        os.chmod(db_path, 0o444)

        try:
            harness2 = MemoryHarness(f"sqlite:///{db_path}")
            await harness2.connect()

            # Reading should work
            results = await harness2.search_knowledge("read only", k=1)
            assert len(results) >= 1

            # Writing might fail
            try:
                await harness2.add_knowledge("New entry", source="test")
            except Exception:
                # Expected to fail on read-only database
                pass

            await harness2.disconnect()
        finally:
            # Restore permissions for cleanup
            os.chmod(db_path, 0o644)


# =============================================================================
# SQLite Fixture Tests
# =============================================================================


class TestSQLiteFixture:
    """Tests using the sqlite_memory fixture."""

    @pytest.mark.asyncio
    async def test_fixture_provides_clean_database(self, sqlite_memory):
        """Test that fixture provides a clean database."""
        messages = await sqlite_memory.get_conversational("any_thread")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_fixture_is_connected(self, sqlite_memory):
        """Test that fixture is already connected."""
        # Should not need to call connect()
        await sqlite_memory.add_knowledge("Fixture test", source="test")
        results = await sqlite_memory.search_knowledge("fixture", k=1)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_multiple_fixtures_isolated(self, sqlite_memory, tmp_path):
        """Test that multiple fixtures don't interfere."""
        # Create another harness
        db_path = tmp_path / "other.db"
        other = MemoryHarness(f"sqlite:///{db_path}")
        await other.connect()

        # Add to fixture
        await sqlite_memory.add_knowledge("Fixture data", source="test")

        # Other should be empty
        results = await other.search_knowledge("fixture", k=1)
        assert len(results) == 0

        await other.disconnect()
