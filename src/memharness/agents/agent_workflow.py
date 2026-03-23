# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Memory-aware agent workflow implementation.

This module implements the complete BEFORE → INSIDE → AFTER agent loop
from the agent memory course L06. It orchestrates memory operations at each
step: reading context, managing token limits, executing LLM+tools, and
persisting results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from memharness.core.harness import MemoryHarness

__all__ = ["AgentResult", "MemoryAwareAgent"]


# System prompt from agent memory course L06
AGENT_SYSTEM_PROMPT = """
# Role
You are a memory-aware agentic research assistant with access to tools.

# Context Window Structure (Partitioned Segments)
The user input is a partitioned context window. It contains a `# Question` section followed by memory segments.
Treat each segment as a distinct memory store with a specific purpose:
- `## Conversation Memory`
- `## Knowledge Base Memory`
- `## Workflow Memory`
- `## Entity Memory`
- `## Summary Memory`

# Memory Store Semantics
- Conversation Memory: Recent thread-level dialogue and instructions. Use it for continuity, user preferences, and unresolved requests.
- Knowledge Base Memory: Retrieved documents/passages. Use it to ground factual and technical claims.
- Workflow Memory: Prior execution patterns and step sequences. Use it to plan tool usage; adapt patterns, do not copy blindly.
- Entity Memory: Named people/orgs/systems and descriptors. Use it to disambiguate references and keep naming consistent.
- Summary Memory: Compressed older context represented by summary IDs. When thread-scoped summaries exist, prefer summaries for the active thread_id.

# Summary Expansion Policy
If critical detail is only present in Summary Memory or appears ambiguous, call `expand_summary(summary_id)` before relying on it.

# Operating Rules
1. Start with the provided memory segments before using tools.
2. If segments conflict, prioritize: current `# Question` > latest Conversation Memory > Knowledge Base evidence > older summaries/workflows.
3. Use only the tools provided in this turn and choose the minimum necessary tool calls.
4. If memory is insufficient, state what is missing and then use an appropriate tool.
5. For conversation compaction, use `summarize_and_store` with `thread_id` so source conversation units are marked as summarized.
"""


@dataclass
class AgentResult:
    """Result from a complete agent execution cycle.

    Attributes:
        answer: Final answer from the agent.
        steps: List of tool calls made during execution (e.g., "tool_name(args) → success").
        thread_id: Thread ID for this conversation.
        tool_calls: Total number of tool calls made.
        context_usage_percent: Context usage as a fraction (0.0 to 1.0).
        workflow_saved: Whether a workflow was saved after execution.
        entities_extracted: Total number of entities extracted from query and response.
        iterations: Number of LLM iterations performed.
    """

    answer: str
    steps: list[str] = field(default_factory=list)
    thread_id: str = ""
    tool_calls: int = 0
    context_usage_percent: float = 0.0
    workflow_saved: bool = False
    entities_extracted: int = 0
    iterations: int = 0


class MemoryAwareAgent:
    """
    Complete agent workflow aligned with agent memory course L06.

    Implements the full BEFORE → INSIDE → AFTER pattern:

    **BEFORE loop:**
    1. Assemble context from all memory types (conversational, knowledge, workflow, entity, summary)
    2. Check context size — if >80%, trigger summarization
    3. Extract entities from user query
    4. Save user message to conversational memory

    **INSIDE loop:**
    5. Call LLM with system prompt + context + tools
    6. Execute tool calls, log results, track steps
    7. Loop until final_answer or max_iterations

    **AFTER loop:**
    8. Save workflow if steps were taken
    9. Extract entities from final answer
    10. Save assistant response to conversational memory

    Example:
        ```python
        from langchain.chat_models import init_chat_model
        from memharness import MemoryHarness
        from memharness.agents import MemoryAwareAgent

        llm = init_chat_model("anthropic:claude-sonnet-4-6")
        harness = MemoryHarness("sqlite:///memory.db")
        agent = MemoryAwareAgent(harness, llm=llm)

        result = await agent.run(
            query="What papers did I read about MemGPT?",
            thread_id="chat-123"
        )
        print(f"Answer: {result.answer}")
        print(f"Tool calls: {result.tool_calls}")
        print(f"Context usage: {result.context_usage_percent:.1%}")
        ```

    Attributes:
        harness: MemoryHarness instance for memory operations.
        llm: LangChain chat model for reasoning.
        max_iterations: Maximum LLM iterations before stopping (default: 10).
        max_context_tokens: Maximum context window size (default: 4000).
        summarize_threshold: Trigger summarization at this percentage (default: 0.8).
    """

    def __init__(
        self,
        harness: MemoryHarness,
        llm: BaseChatModel | None = None,
        max_iterations: int = 10,
        max_context_tokens: int = 4000,
        summarize_threshold: float = 0.8,
    ) -> None:
        """
        Initialize the memory-aware agent.

        Args:
            harness: MemoryHarness instance for memory operations.
            llm: Optional LangChain chat model. If None, agent will fail on run.
            max_iterations: Maximum number of LLM reasoning loops (default: 10).
            max_context_tokens: Maximum tokens in assembled context (default: 4000).
            summarize_threshold: Context usage % to trigger summarization (default: 0.8).
        """
        self.harness = harness
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.summarize_threshold = summarize_threshold

        # Initialize sub-agents
        from memharness.agents.context_assembler import ContextAssemblyAgent
        from memharness.agents.entity_extractor import EntityExtractorAgent
        from memharness.agents.summarizer import SummarizerAgent

        self._context_agent = ContextAssemblyAgent(
            harness=harness,
            max_tokens=max_context_tokens,
            summarize_threshold=summarize_threshold,
        )
        self._summarizer = SummarizerAgent(harness=harness, llm=llm)
        self._entity_extractor = EntityExtractorAgent(harness=harness, llm=llm)

    async def run(
        self,
        query: str,
        thread_id: str,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
    ) -> AgentResult:
        """
        Run the full agent cycle: BEFORE → INSIDE → AFTER.

        Args:
            query: User query to process.
            thread_id: Conversation thread ID.
            tools: Optional additional tools to provide (added to memory tools).
            system_prompt: Optional custom system prompt (defaults to AGENT_SYSTEM_PROMPT).

        Returns:
            AgentResult with answer, steps, metrics, and metadata.

        Raises:
            ValueError: If LLM is not configured.
            ImportError: If langchain-core is not installed.
        """
        if not self.llm:
            raise ValueError(
                "LLM is required for MemoryAwareAgent. Pass a LangChain chat model to __init__."
            )

        try:
            from langchain_core.messages import (
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-core is required. Install with: pip install memharness[langchain]"
            ) from e

        # Use provided system prompt or default
        sys_prompt = system_prompt or AGENT_SYSTEM_PROMPT

        # =======================================================================
        # BEFORE: Assemble context, check size, extract entities, save user msg
        # =======================================================================

        # 1. Assemble context (reads all memory types)
        ctx = await self._context_agent.assemble(query, thread_id, include_tools=False)

        # 2. Context check — summarize if >threshold
        if ctx.context_usage_percent >= self.summarize_threshold:
            await self._summarizer.summarize_thread(thread_id)
            # Re-assemble after summarization
            ctx = await self._context_agent.assemble(query, thread_id, include_tools=False)

        # 3. Extract entities from user query
        entities_from_query = await self._entity_extractor.extract_entities(query)
        extracted_count = 0
        for category, names in entities_from_query.items():
            for name in names:
                try:
                    await self.harness.add_entity(
                        name=name,
                        entity_type=category,
                        description=f"{category}: {name}",
                    )
                    extracted_count += 1
                except Exception:
                    # Entity may already exist, continue
                    pass

        # 4. Get memory tools
        from memharness.tools import get_memory_tools

        memory_tools = get_memory_tools(self.harness)
        all_tools = memory_tools + (tools or [])

        # =======================================================================
        # INSIDE: LLM loop with tool execution
        # =======================================================================

        # Build messages: system prompt + context as user message
        context_str = ctx.to_prompt()
        messages: list[Any] = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=context_str),
        ]

        steps: list[str] = []
        tool_call_count = 0
        final_answer = ""
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Call LLM with tools
            response = await self.llm.ainvoke(messages, tools=all_tools)

            # Check for tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Add assistant message with tool calls
                messages.append(response)

                # Execute each tool call
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})
                    tool_call_id = tc.get("id", "")

                    # Format args for display (truncate long values)
                    args_display = {
                        k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                        for k, v in tool_args.items()
                    }

                    try:
                        # Find and execute tool
                        tool = next((t for t in all_tools if t.name == tool_name), None)
                        if not tool:
                            result = f"Error: Tool '{tool_name}' not found"
                            status = "failed"
                            error_message = f"Tool not found: {tool_name}"
                        else:
                            # Execute tool
                            result = await tool.ainvoke(tool_args)
                            status = "success"
                            error_message = None
                            steps.append(f"{tool_name}({args_display}) → success")
                            tool_call_count += 1
                    except Exception as e:
                        result = f"Error: {e}"
                        status = "failed"
                        error_message = str(e)
                        steps.append(f"{tool_name}({args_display}) → failed")

                    # Write tool log
                    try:
                        await self.harness.add_tool_log(
                            tool_name=tool_name,
                            tool_input=str(tool_args),
                            tool_output=str(result),
                            status=status,
                            thread_id=thread_id,
                            metadata={"iteration": iteration + 1, "error": error_message},
                        )
                    except Exception:
                        # Logging failure shouldn't stop execution
                        pass

                    # Truncate result if too long (context control)
                    result_str = str(result)
                    if len(result_str) > 3000:
                        result_for_llm = (
                            result_str[:3000]
                            + "\n\n[Truncated for context. Full output saved in tool log.]"
                        )
                    else:
                        result_for_llm = result_str

                    # Add tool result message
                    messages.append(ToolMessage(content=result_for_llm, tool_call_id=tool_call_id))
            else:
                # No tool calls — this is the final answer
                final_answer = response.content if hasattr(response, "content") else str(response)
                break
        else:
            # Max iterations reached without final answer
            final_answer = "I was unable to complete the request within the allowed iterations."

        # =======================================================================
        # AFTER: Save workflow, extract entities, save response
        # =======================================================================

        # 5. Save workflow if steps taken
        workflow_saved = False
        if steps:
            try:
                await self.harness.add_workflow(
                    task=query,
                    steps=steps,
                    outcome=final_answer[:200],  # Truncate outcome
                )
                workflow_saved = True
            except Exception:
                # Workflow save failure shouldn't stop execution
                pass

        # 6. Extract entities from response
        entities_from_response = await self._entity_extractor.extract_entities(final_answer)
        for category, names in entities_from_response.items():
            for name in names:
                try:
                    await self.harness.add_entity(
                        name=name,
                        entity_type=category,
                        description=f"{category}: {name}",
                    )
                    extracted_count += 1
                except Exception:
                    # Entity may already exist, continue
                    pass

        # 7. Save assistant response
        await self.harness.add_conversational(thread_id, "assistant", final_answer)

        # Return result
        return AgentResult(
            answer=final_answer,
            steps=steps,
            thread_id=thread_id,
            tool_calls=tool_call_count,
            context_usage_percent=ctx.context_usage_percent,
            workflow_saved=workflow_saved,
            entities_extracted=extracted_count,
            iterations=iterations,
        )
