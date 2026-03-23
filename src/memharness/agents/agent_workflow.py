# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
LangGraph workflow for memory-aware agent execution.

This module implements the complete agent harness from L06 as a LangGraph
workflow with nodes and edges, following the same logic as the call_agent
function from the course.

Key Design Principle:
- The main LLM agent node ONLY gets conversation messages
- Everything else (KB, entity, workflow, summary, toolbox) is prepared by
  PRIOR nodes in the graph and injected as context
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from memharness.core.harness import MemoryHarness

__all__ = ["AgentState", "create_memory_agent"]


class AgentState(TypedDict):
    """
    State for the memory-aware agent workflow.

    Attributes:
        messages: Conversation messages (includes context + history + query).
        thread_id: The conversation thread ID.
        query: The current user query.
        context_usage: Context usage percentage (0.0 to 1.0).
        steps: List of tool execution steps.
        entities_extracted: Number of entities extracted.
        workflow_saved: Whether workflow was saved.
        final_answer: The agent's final response.
        max_context_tokens: Maximum context tokens allowed.
        summarize_threshold: Threshold to trigger summarization.
        system_prompt: System prompt for the agent.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    query: str
    context_usage: float
    steps: list[str]
    entities_extracted: int
    workflow_saved: bool
    final_answer: str
    max_context_tokens: int
    summarize_threshold: float
    system_prompt: str | None


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


async def save_user_message(state: AgentState, harness: MemoryHarness) -> dict[str, Any]:
    """
    Node 1: Save the user query to conversational memory.

    Args:
        state: Current agent state.
        harness: MemoryHarness instance.

    Returns:
        Empty dict (no state updates).
    """
    await harness.add_conversational(state["thread_id"], "user", state["query"])
    return {}


async def assemble_context(state: AgentState, harness: MemoryHarness) -> dict[str, Any]:
    """
    Node 2: Assemble context from all memory types and inject as SystemMessage.

    This node reads all memory types and builds a SystemMessage with the context.
    The main agent node will receive this context along with the conversation history.

    Args:
        state: Current agent state.
        harness: MemoryHarness instance.

    Returns:
        Dict with 'messages' containing SystemMessage with context.
    """
    from memharness.agents.context_assembler import ContextAssemblyAgent

    # Use the ContextAssemblyAgent to assemble context
    assembler = ContextAssemblyAgent(
        harness=harness,
        max_tokens=state["max_context_tokens"],
        summarize_threshold=state["summarize_threshold"],
    )

    # Assemble context (this also saves the query to conversational memory)
    ctx = await assembler.assemble(
        query=state["query"],
        thread_id=state["thread_id"],
        include_tools=True,
    )

    # Convert to messages (SystemMessage + conversation history)
    context_messages = ctx.to_messages()

    # Add the user query as the final HumanMessage (if not already present)
    if not context_messages or not isinstance(context_messages[-1], HumanMessage):
        context_messages.append(HumanMessage(content=state["query"]))

    return {
        "messages": context_messages,
        "context_usage": ctx.context_usage_percent,
    }


def check_context(state: AgentState) -> Literal["summarize", "proceed"]:
    """
    Routing function: Check if context usage exceeds threshold.

    Args:
        state: Current agent state.

    Returns:
        'summarize' if context usage > threshold, else 'proceed'.
    """
    if state["context_usage"] > state["summarize_threshold"]:
        return "summarize"
    return "proceed"


async def summarize_thread(state: AgentState, harness: MemoryHarness) -> dict[str, Any]:
    """
    Node 4: Summarize the thread to reduce context usage.

    Args:
        state: Current agent state.
        harness: MemoryHarness instance.

    Returns:
        Empty dict (context will be re-assembled in next step).
    """
    from memharness.agents.summarizer import SummarizerAgent

    # Get LLM if available (from state or harness)
    # For now, we'll use heuristic summarization
    summarizer = SummarizerAgent(harness=harness, llm=None)

    # Summarize the thread
    await summarizer.summarize_thread(state["thread_id"], max_messages=50)

    # Note: After summarization, we'll go back to assemble_context
    # to rebuild the context with summaries
    return {}


async def call_agent_node(
    state: AgentState,
    harness: MemoryHarness,
    llm: BaseChatModel,
    tools: list[Any] | None = None,
    max_iterations: int = 10,
) -> dict[str, Any]:
    """
    Node 5: Call the LLM agent with tools.

    The LLM receives:
    - SystemMessage with context (assembled in prior node)
    - Conversation history
    - HumanMessage with the query

    Args:
        state: Current agent state.
        harness: MemoryHarness instance.
        llm: The LLM to use.
        tools: List of tools to bind to the LLM.
        max_iterations: Maximum tool execution iterations.

    Returns:
        Dict with 'messages', 'final_answer', and 'steps'.
    """
    from langchain_core.messages import ToolMessage

    messages = list(state["messages"])
    steps = list(state.get("steps", []))
    final_answer = ""

    # Bind tools to LLM if provided
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    # Agent loop with tool execution
    for iteration in range(max_iterations):
        # Call LLM
        response = await llm_with_tools.ainvoke(messages)

        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Add assistant message with tool calls
            messages.append(response)

            # Execute each tool
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                try:
                    # Find and execute the tool
                    tool_result = None
                    if tools:
                        for tool in tools:
                            if tool.name == tool_name:
                                tool_result = await tool.ainvoke(tool_args)
                                break

                    if tool_result is None:
                        tool_result = f"Error: Tool '{tool_name}' not found"
                        status = "failed"
                    else:
                        status = "success"

                    # Log tool execution
                    log_id = await harness.log_tool_execution(
                        thread_id=state["thread_id"],
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result=str(tool_result),
                        status=status,
                        metadata={"iteration": iteration + 1},
                    )

                    # Track step
                    steps.append(f"{tool_name}({tool_args}) → {status}")

                    # Truncate result if too long
                    result_str = str(tool_result)
                    if len(result_str) > 3000:
                        result_for_llm = (
                            result_str[:3000]
                            + f"\n\n[Truncated. Full output in tool log: {log_id}]"
                        )
                    else:
                        result_for_llm = result_str

                    # Add tool message
                    messages.append(ToolMessage(content=result_for_llm, tool_call_id=tool_call_id))

                except Exception as e:
                    # Log error
                    await harness.log_tool_execution(
                        thread_id=state["thread_id"],
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result="",
                        status="failed",
                        error_message=str(e),
                        metadata={"iteration": iteration + 1},
                    )

                    steps.append(f"{tool_name}({tool_args}) → failed: {e}")

                    # Add error message
                    messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call_id))

        else:
            # No tool calls - we have the final answer
            final_answer = response.content if hasattr(response, "content") else str(response)
            messages.append(response)
            break
    else:
        # Max iterations reached
        final_answer = "Unable to complete request within allowed iterations."
        messages.append(AIMessage(content=final_answer))

    return {
        "messages": messages,
        "final_answer": final_answer,
        "steps": steps,
    }


async def save_response(state: AgentState, harness: MemoryHarness) -> dict[str, Any]:
    """
    Node 6: Save the assistant's response to conversational memory.

    Args:
        state: Current agent state.
        harness: MemoryHarness instance.

    Returns:
        Empty dict (no state updates).
    """
    await harness.add_conversational(state["thread_id"], "assistant", state["final_answer"])
    return {}


async def extract_entities(state: AgentState, harness: MemoryHarness) -> dict[str, Any]:
    """
    Node 7: Extract entities from the response.

    Args:
        state: Current agent state.
        harness: MemoryHarness instance.

    Returns:
        Dict with 'entities_extracted' count.
    """
    from memharness.agents.entity_extractor import EntityExtractorAgent

    # Create entity extractor (using regex mode, no LLM)
    extractor = EntityExtractorAgent(harness=harness, llm=None)

    # Extract entities from the final answer
    entities = await extractor.extract_entities(state["final_answer"])

    # Save entities
    count = 0
    for category, names in entities.items():
        for name in names:
            if name.strip():
                await harness.add_entity(
                    name=name,
                    entity_type=category,
                    content=f"{category}: {name}",
                )
                count += 1

    return {"entities_extracted": count}


async def save_workflow(state: AgentState, harness: MemoryHarness) -> dict[str, Any]:
    """
    Node 8: Save the workflow if any tool calls happened.

    Args:
        state: Current agent state.
        harness: MemoryHarness instance.

    Returns:
        Dict with 'workflow_saved' boolean.
    """
    if state["steps"]:
        # Save workflow
        workflow_text = "\n".join(state["steps"])
        await harness.add_workflow(
            task=state["query"],
            steps=workflow_text,
            result=state["final_answer"],
        )
        return {"workflow_saved": True}

    return {"workflow_saved": False}


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_memory_agent(
    harness: MemoryHarness,
    llm: BaseChatModel | str = "anthropic:claude-sonnet-4-6",
    tools: list | None = None,
    max_context_tokens: int = 4000,
    summarize_threshold: float = 0.8,
    system_prompt: str | None = None,
    max_iterations: int = 10,
) -> Any:  # Returns CompiledGraph
    """
    Create a LangGraph workflow with full memory management.

    This factory function creates a complete memory-aware agent workflow
    that implements the same logic as the call_agent function from L06.

    The workflow includes:
    - Context assembly from all memory types
    - Automatic summarization when context is full
    - Tool execution with logging
    - Entity extraction and workflow persistence

    Args:
        harness: MemoryHarness instance for memory operations.
        llm: LLM to use (BaseChatModel or string like "anthropic:claude-sonnet-4-6").
        tools: List of tools to make available to the agent.
        max_context_tokens: Maximum context size in tokens (default: 4000).
        summarize_threshold: Trigger summarization at this percentage (default: 0.8).
        system_prompt: Optional system prompt override.
        max_iterations: Maximum tool execution iterations (default: 10).

    Returns:
        Compiled LangGraph graph ready to invoke.

    Example:
        ```python
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("anthropic:claude-sonnet-4-6")
        graph = create_memory_agent(harness, llm=llm)

        result = await graph.ainvoke({
            "messages": [],
            "thread_id": "user-123",
            "query": "How do I deploy?",
            "context_usage": 0.0,
            "steps": [],
            "entities_extracted": 0,
            "workflow_saved": False,
            "final_answer": "",
            "max_context_tokens": 4000,
            "summarize_threshold": 0.8,
            "system_prompt": None,
        })

        print(result["final_answer"])
        ```
    """
    # Initialize LLM if string provided
    if isinstance(llm, str):
        from langchain.chat_models import init_chat_model

        llm = init_chat_model(llm)

    # Create the graph builder
    builder = StateGraph(AgentState)

    # Create wrapper functions that properly handle the async calls
    # LangGraph will handle async functions automatically
    async def _save_user_message(state: AgentState) -> dict[str, Any]:
        return await save_user_message(state, harness)

    async def _assemble_context(state: AgentState) -> dict[str, Any]:
        return await assemble_context(state, harness)

    async def _summarize_thread(state: AgentState) -> dict[str, Any]:
        return await summarize_thread(state, harness)

    async def _call_agent(state: AgentState) -> dict[str, Any]:
        return await call_agent_node(state, harness, llm, tools, max_iterations)

    async def _save_response(state: AgentState) -> dict[str, Any]:
        return await save_response(state, harness)

    async def _extract_entities(state: AgentState) -> dict[str, Any]:
        return await extract_entities(state, harness)

    async def _save_workflow(state: AgentState) -> dict[str, Any]:
        return await save_workflow(state, harness)

    # Add nodes with bound harness/llm
    builder.add_node("save_user_message", _save_user_message)
    builder.add_node("assemble_context", _assemble_context)
    builder.add_node("summarize_thread", _summarize_thread)
    builder.add_node("call_agent", _call_agent)
    builder.add_node("save_response", _save_response)
    builder.add_node("extract_entities", _extract_entities)
    builder.add_node("save_workflow", _save_workflow)

    # Add edges
    builder.add_edge(START, "save_user_message")
    builder.add_edge("save_user_message", "assemble_context")

    # Conditional edge: check context usage
    builder.add_conditional_edges(
        "assemble_context",
        check_context,
        {
            "summarize": "summarize_thread",
            "proceed": "call_agent",
        },
    )

    # After summarization, re-assemble context
    builder.add_edge("summarize_thread", "assemble_context")

    # Linear flow after agent execution
    builder.add_edge("call_agent", "save_response")
    builder.add_edge("save_response", "extract_entities")
    builder.add_edge("extract_entities", "save_workflow")
    builder.add_edge("save_workflow", END)

    # Compile and return
    return builder.compile()
