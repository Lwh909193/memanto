"""
LangGraph node definitions for the Research + Writer pipeline.

Each node is a function (state_in, state_out) that gets compiled into
the LangGraph pipeline via langgraph_memanto.graph.build_graph().
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langgraph_memanto.memory_tools import (
    memanto_remember,
    memanto_recall,
    memanto_answer,
)

load_dotenv()

MOORCHEH_API_KEY = os.getenv("MOORCHEH_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm():
    """Build an OpenRouter-backed chat model."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-haiku",
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# Research Agent nodes
# ---------------------------------------------------------------------------


def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Research Agent analyzes a topic and stores key findings as Memanto memories.

    It reads the research topic from state, calls the LLM to produce findings,
    then calls memanto_remember for each one.
    """
    topic = state.get("research_topic", "")
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    llm = _get_llm()

    prompt = (
        f"You are a Senior Market Research Analyst.\n"
        f"Your task: Research '{topic}' thoroughly and store every key finding "
        f"as a structured memory using the memanto_remember tool.\n\n"
        f"For each major finding:\n"
        f"  - Pick the correct memory type (fact, observation, decision, etc.)\n"
        f"  - Write a concise title (under 100 chars)\n"
        f"  - Write atomic content (under 500 chars)\n"
        f"  - Set an appropriate confidence score (1.0 for facts, 0.7-0.9 for observations)\n"
        f"  - Add relevant tags\n\n"
        f"Store at least 3 distinct findings using the memanto_remember tool.\n"
        f"Then respond with a brief summary of what you stored."
    )

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    findings = []

    return {
        "messages": [{"role": "assistant", "content": content}],
        "findings": findings,
    }


def writer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Writer Agent retrieves stored memories and writes an executive briefing.

    It calls memanto_recall to fetch all research memories, then uses the LLM
    to synthesize them into a written briefing.
    """
    topic = state.get("research_topic", "")
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    llm = _get_llm()

    # First, recall memories
    recall_result = memanto_recall(
        state={**state, "memanto_agent_id": agent_id},
        api_key=MOORCHEH_API_KEY,
        query=f"research findings about {topic}",
        limit=10,
    )

    # Then answer a synthesis question
    answer_result = memanto_answer(
        state={**state, "memanto_agent_id": agent_id},
        api_key=MOORCHEH_API_KEY,
        question=f"Provide a detailed summary of all research findings about {topic}",
    )

    recall_content = recall_result.get("messages", [{}])[0].get("content", "")
    answer_content = answer_result.get("messages", [{}])[0].get("content", "")

    synthesis_prompt = (
        f"You are a Technical Briefing Writer.\n"
        f"Topic: {topic}\n\n"
        f"Retrieved memories:\n{recall_content}\n\n"
        f"RAG Answer:\n{answer_content}\n\n"
        f"Write a clear, data-driven executive briefing on '{topic}' "
        f"using ONLY the information from the retrieved memories above. "
        f"Do not fabricate data. Cite sources."
    )

    response = llm.invoke(synthesis_prompt)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [
            {"role": "assistant", "content": content},
        ],
    }


def should_continue(state: Dict[str, Any]) -> Literal["research", "writer", "end"]:
    """
    Routing logic: research → writer → end.
    """
    messages = state.get("messages", [])
    if not messages:
        return "research"
    last = messages[-1]
    role = last.get("role", "")
    if role == "assistant":
        # First pass: researcher spoke → go to writer
        return "writer"
    return "end"
