"""
Memanto Tools — LangGraph-compatible wrappers for Memanto's three primitives.

These tools wrap the Memanto `remember`, `recall`, and `answer` primitives
as LangGraph tools that can be bound to any LLM node.

Usage:
    tools = [memanto_remember, memanto_recall, memanto_answer]
    llm_with_tools = llm.bind_tools(tools)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.tools import tool

# Import SdkClient at module level for type annotations
from memanto.cli.client.sdk_client import SdkClient

logger = logging.getLogger(__name__)

# Default demo agent used across sessions
DEMO_AGENT_ID = "langgraph-demo-agent"


def _ensure_demo_agent(client: SdkClient) -> str | None:
    """Ensure the demo agent exists and has an active session.

    Creates the agent if it doesn't exist, then activates a session.
    Returns the agent_id on success, None on failure.
    """
    try:
        try:
            client.get_agent(DEMO_AGENT_ID)
        except Exception:
            # Agent doesn't exist yet — create it
            client.create_agent(
                agent_id=DEMO_AGENT_ID,
                pattern="tool",
                description="LangGraph cross-session memory demo agent",
            )

        # Activate session so remember/recall/answer can run
        client.activate_agent(DEMO_AGENT_ID)
        return DEMO_AGENT_ID
    except Exception as e:
        logger.warning(f"Failed to set up demo agent '{DEMO_AGENT_ID}': {e}")
        return None


_client_instance = None
_client_agent_id = None


def _get_memanto_client():
    """Lazy-import and return a Memanto client with a configured demo agent.

    Returns a tuple of (client, agent_id) on success, or (None, None)
    if the required environment variable is missing.
    """
    global _client_instance, _client_agent_id

    if _client_instance is not None:
        return _client_instance, _client_agent_id

    api_key = os.environ.get("MOORCHEH_API_KEY")
    if not api_key:
        return None, None
    try:
        from memanto.cli.client.sdk_client import SdkClient

        client = SdkClient(api_key=api_key)

        # Set up the demo agent with an active session
        agent_id = _ensure_demo_agent(client)
        if agent_id is None:
            return None, None

        _client_instance = client
        _client_agent_id = agent_id
        return client, agent_id
    except ImportError:
        logger.warning("memanto package not installed")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to initialize Memanto client: {e}")
        return None, None


# ─── Tool: remember ───────────────────────────────────────────────────────────


@tool
def memanto_remember(
    content: str,
    memory_type: str = "observation",
    title: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Store a memory into Memanto's long-term memory store.

    Call this whenever the agent learns something important about the user
    or the environment that should be remembered across sessions.

    Args:
        content: The factual content to remember (e.g., "User prefers dark mode")
        memory_type: One of: fact, decision, instruction, commitment, event,
                     observation, preference, goal, relationship, context,
                     learning, artifact, error
        title: Optional short title for the memory (max 100 chars).
               Defaults to a summary prefix of content.
        agent_id: Optional agent namespace. Defaults to the demo agent.

    Returns:
        A JSON string with the stored memory details, or an error message.
    """
    client, default_agent = _get_memanto_client()
    if client is None:
        return (
            "ERROR: Memanto client not available. "
            "Set MOORCHEH_API_KEY environment variable."
        )

    try:
        aid = agent_id or default_agent
        if not aid:
            return "ERROR: No agent_id configured."

        # Default title from content if not provided
        mem_title = title or content[:80]

        result = client.remember(
            agent_id=aid,
            memory_type=memory_type,
            title=mem_title,
            content=content[:500],
        )
        return json.dumps(
            {
                "status": "stored",
                "memory_type": memory_type,
                "content_preview": content[:100],
                "memory_id": result.get("memory_id", ""),
            },
            indent=2,
        )
    except Exception as e:
        return f"ERROR storing memory: {e}"


# ─── Tool: recall ─────────────────────────────────────────────────────────────


@tool
def memanto_recall(
    query: str,
    memory_type: str | None = None,
    limit: int = 5,
    agent_id: str | None = None,
) -> str:
    """Search Memanto's long-term memory for relevant past information.

    Call this at the start of a conversation or when the user asks
    about past interactions. This is the key tool for cross-session recall.

    Args:
        query: Natural language search query (e.g., "What does the user like?")
        memory_type: Optional filter — restrict search to one memory type
        limit: Maximum number of results to return (default: 5)
        agent_id: Optional agent namespace. Defaults to the demo agent.

    Returns:
        A list of matching memories with their content, type, and timestamps.
    """
    client, default_agent = _get_memanto_client()
    if client is None:
        return (
            "ERROR: Memanto client not available. "
            "Set MOORCHEH_API_KEY environment variable."
        )

    try:
        aid = agent_id or default_agent
        if not aid:
            return "ERROR: No agent_id configured."

        kwargs: dict[str, Any] = {"limit": limit}
        if memory_type:
            kwargs["type"] = [memory_type]

        result = client.recall(agent_id=aid, query=query, **kwargs)
        memories = result.get("memories", [])

        if not memories:
            return "No relevant memories found."

        formatted = []
        for i, mem in enumerate(memories, 1):
            formatted.append(
                f"{i}. [{mem.get('type', 'unknown')}] "
                f"{mem.get('content', '')[:200]} "
                f"(confidence: {mem.get('confidence', 'N/A')})"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"ERROR recalling memories: {e}"


# ─── Tool: answer (RAG) ──────────────────────────────────────────────────────


@tool
def memanto_answer(
    query: str,
    agent_id: str | None = None,
) -> str:
    """Get a grounded AI answer generated directly from Memanto's memory store.

    Unlike `recall` which returns raw memory entries, `answer` runs
    retrieval-augmented generation (RAG) over the entire memory store
    to produce a natural-language answer.

    Use this when the user asks a question that requires synthesis
    across multiple memories.

    Args:
        query: The question to answer using stored memories
        agent_id: Optional agent namespace. Defaults to the demo agent.

    Returns:
        A natural language answer grounded in Memanto's memory store.
    """
    client, default_agent = _get_memanto_client()
    if client is None:
        return (
            "ERROR: Memanto client not available. "
            "Set MOORCHEH_API_KEY environment variable."
        )

    try:
        aid = agent_id or default_agent
        if not aid:
            return "ERROR: No agent_id configured."

        result = client.answer(agent_id=aid, question=query)
        answer_text = result.get("answer", str(result))
        return str(answer_text)
    except Exception as e:
        return f"ERROR generating answer from memories: {e}"


# ─── Tool Listing ─────────────────────────────────────────────────────────────


MEMANTO_TOOLS = [memanto_remember, memanto_recall, memanto_answer]
"""List of all Memanto tools for easy binding to LangGraph nodes."""


def format_memory_context(memories: list[dict]) -> str:
    """Format a list of memory dicts into a context string for the LLM.

    Args:
        memories: List of memory dicts from recall()

    Returns:
        Formatted context string.
    """
    if not memories:
        return "No past memories retrieved."

    lines = ["\n📚 **Past Memories (from Memanto):**\n"]
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "").strip()
        mtype = mem.get("type", "unknown")
        timestamp = mem.get("timestamp", "")
        lines.append(f'  {i}. [{mtype}] "{content}" ({timestamp})')
    lines.append("")
    return "\n".join(lines)
