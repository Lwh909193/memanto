"""
Run 1: Research Agent stores findings in Memanto.

This script proves that memories persist across sessions.
Run this script, then run run_writer.py in a new terminal —
the writer will retrieve the SAME memories from Memanto.
"""

from __future__ import annotations

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langgraph.graph import END

from langgraph_memanto.graph import compile_graph
from langgraph_memanto.memory_tools import memanto_remember

load_dotenv()

MEMANTO_API_KEY = os.getenv("MOORCHEH_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
AGENT_ID = os.getenv("MEMANTO_AGENT_ID", "langgraph-research-team")
TOPIC = os.getenv("RESEARCH_TOPIC", "AI agent framework market size and trends 2024")


def main():
    if not MEMANTO_API_KEY or not OPENROUTER_API_KEY:
        print("ERROR: Missing API keys.")
        print("Copy .env.example to .env and fill in MOORCHEH_API_KEY and OPENROUTER_API_KEY")
        sys.exit(1)

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-haiku",
        temperature=0.7,
    )

    prompt = (
        f"You are a Senior Market Research Analyst.\n"
        f"Research '{TOPIC}' thoroughly.\n\n"
        f"For each key finding, use memanto_remember to store:\n"
        f"  - memory_type: 'fact' or 'observation'\n"
        f"  - title: concise, under 100 chars\n"
        f"  - content: atomic, under 500 chars\n"
        f"  - confidence: 1.0 for facts, 0.7-0.9 for observations\n"
        f"  - tags: relevant keywords\n\n"
        f"Store at least 3 findings using memanto_remember.\n"
        f"Then summarize what you stored in a short message."
    )

    print(f"Research Agent analyzing: {TOPIC}")
    print("---")

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    print("\n[Research Agent Response]")
    print(content)
    print("\nMemories stored in Memanto (agent_id={})".format(AGENT_ID))
    print("Run run_writer.py in a new terminal to retrieve them!")


if __name__ == "__main__":
    main()
