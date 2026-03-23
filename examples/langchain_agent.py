#!/usr/bin/env python3
# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Example: Using memharness middleware with LangChain agents.

This example demonstrates how to create a LangChain agent with memharness
middleware for automatic memory context injection, persistence, and entity
extraction.

Requirements:
    pip install memharness[langchain]
    export ANTHROPIC_API_KEY=your_api_key
"""

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from memharness import MemoryHarness
from memharness.middleware import (
    EntityExtractionMiddleware,
    MemoryContextMiddleware,
    MemoryPersistenceMiddleware,
)
from memharness.tools import get_memory_tools


async def main():
    """Run the example agent with memharness middleware."""
    # Initialize memory harness with SQLite backend
    harness = MemoryHarness("sqlite:///memory.db")
    await harness.connect()

    print("✓ Connected to memory backend")

    # Optional: Initialize an LLM for entity extraction
    # (Falls back to regex-based extraction if not provided)
    try:
        extraction_llm = init_chat_model("openai:gpt-4o-mini")
        print("✓ Using GPT-4o-mini for entity extraction")
    except Exception:
        extraction_llm = None
        print("ℹ Using regex-based entity extraction (no LLM configured)")

    # Get memory tools for the agent
    memory_tools = get_memory_tools(harness)

    # Create the agent with memharness middleware
    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=memory_tools,
        system_prompt=(
            "You are a helpful AI assistant with persistent memory. "
            "You can remember conversations, facts, and entities across sessions. "
            "Use the memory tools to search and store information."
        ),
        middleware=[
            # 1. Inject memory context before model calls
            MemoryContextMiddleware(
                harness=harness,
                thread_id="example-thread",
                max_tokens=2000,
            ),
            # 2. Extract entities from conversations
            EntityExtractionMiddleware(
                harness=harness,
                thread_id="example-thread",
                llm=extraction_llm,
            ),
            # 3. Persist conversations to memory
            MemoryPersistenceMiddleware(
                harness=harness,
                thread_id="example-thread",
            ),
        ],
    )

    print("✓ Agent created with memory middleware\n")

    # Example interactions
    conversations = [
        "Hi! My name is Alice and I work at Anthropic in San Francisco.",
        "What do you remember about me?",
        "Can you search for information about where I work?",
    ]

    for user_message in conversations:
        print(f"User: {user_message}")

        # Stream the agent's response
        response_chunks = []
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="updates",
        ):
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    if hasattr(msg, "content") and msg.content:
                        response_chunks.append(msg.content)

        response = "".join(response_chunks)
        print(f"Assistant: {response}\n")

    # Show what was stored in memory
    print("\n--- Memory Statistics ---")
    entities = await harness.search_entity("", k=10)
    print(f"Entities stored: {len(entities)}")
    for entity in entities:
        name = entity.metadata.get("name", "Unknown")
        entity_type = entity.metadata.get("entity_type", "unknown")
        print(f"  - {name} ({entity_type})")

    conv_messages = await harness.get_conversational("example-thread", limit=100)
    print(f"\nConversation messages stored: {len(conv_messages)}")

    # Cleanup
    await harness.disconnect()
    print("\n✓ Disconnected from memory backend")


if __name__ == "__main__":
    asyncio.run(main())
