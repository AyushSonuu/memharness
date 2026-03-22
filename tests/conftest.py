"""
Pytest fixtures for memharness tests.

Provides reusable fixtures for testing memory harness with various backends.
"""

import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator

from memharness import MemoryHarness, MemoryType, MemoryUnit, Config


# =============================================================================
# Core Memory Harness Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def memory() -> AsyncGenerator[MemoryHarness, None]:
    """
    Create in-memory harness for fast testing.

    This uses the in-memory backend which is fastest for unit tests
    but doesn't persist data between tests.

    Usage:
        async def test_something(memory):
            await memory.add_conversational("t1", "user", "Hello")
    """
    harness = MemoryHarness("memory://")
    await harness.connect()
    yield harness
    await harness.disconnect()


@pytest_asyncio.fixture
async def sqlite_memory(tmp_path: Path) -> AsyncGenerator[MemoryHarness, None]:
    """
    Create SQLite harness for testing persistence.

    Uses a temporary directory for the database file, which is
    automatically cleaned up after the test.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Usage:
        async def test_persistence(sqlite_memory):
            await sqlite_memory.add_knowledge("test", source="test")
    """
    db_path = tmp_path / "test.db"
    harness = MemoryHarness(f"sqlite:///{db_path}")
    await harness.connect()
    yield harness
    await harness.disconnect()


@pytest_asyncio.fixture
async def postgres_memory() -> AsyncGenerator[MemoryHarness, None]:
    """
    Create PostgreSQL harness for integration testing.

    Requires POSTGRES_TEST_URL environment variable to be set.
    Skips test if not available.

    Usage:
        @pytest.mark.postgres
        async def test_postgres_feature(postgres_memory):
            ...
    """
    import os

    postgres_url = os.environ.get("POSTGRES_TEST_URL")
    if not postgres_url:
        pytest.skip("POSTGRES_TEST_URL not set")

    harness = MemoryHarness(postgres_url)
    await harness.connect()
    yield harness
    # Clean up test data
    await harness.clear_all()
    await harness.disconnect()


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> Config:
    """Return default configuration object."""
    return Config()


@pytest.fixture
def custom_config(tmp_path: Path) -> Config:
    """
    Create a custom configuration from YAML.

    Writes a test YAML config file and loads it.
    """
    config_content = """
memory:
  default_backend: sqlite
  retention:
    conversational: 7d
    summary: 30d
    knowledge: 365d

search:
  default_k: 5
  similarity_threshold: 0.7

agents:
  enabled: true
  model: gpt-4o-mini

lifecycle:
  auto_summarize: true
  summarize_threshold: 100
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return Config.from_yaml(config_path)


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_conversation():
    """Return sample conversation data for testing."""
    return [
        {"thread_id": "t1", "role": "user", "content": "Hello, how can you help me?"},
        {"thread_id": "t1", "role": "assistant", "content": "I can help with many tasks!"},
        {"thread_id": "t1", "role": "user", "content": "Tell me about Python"},
        {"thread_id": "t1", "role": "assistant", "content": "Python is a programming language."},
    ]


@pytest.fixture
def sample_knowledge():
    """Return sample knowledge base entries."""
    return [
        {"content": "Python is a high-level programming language", "source": "docs", "tags": ["python", "programming"]},
        {"content": "Kubernetes orchestrates containers", "source": "docs", "tags": ["k8s", "containers"]},
        {"content": "Docker packages applications in containers", "source": "wiki", "tags": ["docker", "containers"]},
    ]


@pytest.fixture
def sample_entities():
    """Return sample entity data."""
    return [
        {"name": "John Doe", "type": "PERSON", "description": "Software engineer at Acme Corp"},
        {"name": "Acme Corp", "type": "ORGANIZATION", "description": "Technology company"},
        {"name": "Kubernetes", "type": "SYSTEM", "description": "Container orchestration platform"},
    ]


@pytest.fixture
def sample_tools():
    """Return sample tool definitions."""
    return [
        {"namespace": "github", "name": "create_pr", "description": "Create pull request", "schema": {"title": "string", "body": "string"}},
        {"namespace": "github", "name": "list_issues", "description": "List repository issues", "schema": {"repo": "string", "state": "string"}},
        {"namespace": "slack", "name": "send_message", "description": "Send Slack message", "schema": {"channel": "string", "text": "string"}},
        {"namespace": "jira", "name": "create_ticket", "description": "Create Jira ticket", "schema": {"project": "string", "summary": "string"}},
    ]


@pytest.fixture
def sample_workflows():
    """Return sample workflow data."""
    return [
        {"name": "Deploy Application", "steps": ["build", "test", "deploy", "notify"], "status": "success"},
        {"name": "Run Tests", "steps": ["lint", "unit-test", "integration-test"], "status": "success"},
        {"name": "Release Process", "steps": ["tag", "build", "publish", "announce"], "status": "pending"},
    ]


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def populated_memory(memory, sample_conversation, sample_knowledge, sample_entities):
    """
    Create a memory harness pre-populated with sample data.

    Useful for testing search and retrieval operations.
    """
    # Add conversations
    for msg in sample_conversation:
        await memory.add_conversational(msg["thread_id"], msg["role"], msg["content"])

    # Add knowledge
    for kb in sample_knowledge:
        await memory.add_knowledge(kb["content"], source=kb["source"], tags=kb.get("tags"))

    # Add entities
    for entity in sample_entities:
        await memory.add_entity(entity["name"], entity["type"], entity["description"])

    return memory


@pytest.fixture
def memory_unit_factory():
    """Factory fixture for creating MemoryUnit instances."""
    def create(
        content: str = "test content",
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        **kwargs
    ) -> MemoryUnit:
        return MemoryUnit(
            content=content,
            memory_type=memory_type,
            **kwargs
        )
    return create


# =============================================================================
# Markers and Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "postgres: marks tests requiring PostgreSQL"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "agents: marks tests for AI agent functionality"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location.

    - tests/integration/* -> @pytest.mark.integration
    - tests/unit/* -> no additional marks
    """
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
