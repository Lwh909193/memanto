"""
MemantoMemory — A LangGraph-compatible memory integration for Memanto.

Provides tools to remember, recall, and get LLM-grounded answers
from a Memanto-powered long-term memory layer.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Memanto client wrapper
# ---------------------------------------------------------------------------

MEMORY_CATEGORIES = [
    "instruction", "fact", "decision", "goal", "commitment",
    "preference", "relationship", "context", "event", "learning",
    "observation", "artifact", "error",
]

MEMORY_TYPES_LITERAL = str  # one of the categories above


class MemantoMemory:
    """A thin Pythonic wrapper around the Memanto REST API.

    This wraps the three core primitives:
        remember(text, type) → store a fact
        recall(query)         → semantic search over stored memories
        answer(question)      → LLM-grounded answer from memories

    In production you'd use the moorcheh_sdk directly. Here we keep
    it as a simple HTTP client so the example is self-contained.
    """

    def __init__(
        self,
        api_key: str | None = None,
        agent_name: str = "langgraph-agent",
        base_url: str = "https://api.moorcheh.ai/v1",
    ):
        self.api_key = api_key or os.getenv("MOORCHEH_API_KEY", "")
        self.agent_name = agent_name
        self.base_url = base_url.rstrip("/")
        self._session_token: str | None = None

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    def ensure_agent(self) -> dict:
        """Create the agent namespace if it doesn't exist yet."""
        import httpx

        r = httpx.post(
            f"{self.base_url}/agents",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"agent_id": self.agent_name, "display_name": "LangGraph Agent"},
            timeout=15,
        )
        # 409 = already exists, which is fine
        if r.status_code not in (200, 201, 409):
            raise RuntimeError(f"Failed to create agent: {r.text}")
        return r.json() if r.status_code not in (409,) else {"status": "exists"}

    def activate_session(self) -> str:
        """Start a session and return a session token."""
        import httpx

        self.ensure_agent()

        r = httpx.post(
            f"{self.base_url}/agents/{self.agent_name}/activate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        self._session_token = data.get("session_token", "")
        return self._session_token  # type: ignore[return-value]

    def deactivate_session(self) -> None:
        """End the current session."""
        if not self._session_token:
            return
        import httpx

        try:
            httpx.post(
                f"{self.base_url}/agents/{self.agent_name}/deactivate",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Session-Token": self._session_token,
                },
                timeout=10,
            )
        except Exception:
            pass
        self._session_token = None

    # -----------------------------------------------------------------------
    # Memory primitives
    # -----------------------------------------------------------------------

    def remember(
        self,
        text: str,
        memory_type: str = "observation",
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Store a memory."""
        import httpx

        payload: dict[str, Any] = {
            "text": text,
            "type": memory_type,
            "metadata": metadata or {},
        }

        r = httpx.post(
            f"{self.base_url}/agents/{self.agent_name}/remember",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Session-Token": self._session_token or "",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def recall(
        self,
        query: str,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Search across stored memories and return relevant results."""
        import httpx

        payload: dict[str, Any] = {"query": query, "limit": limit}
        if memory_type:
            payload["type"] = memory_type

        r = httpx.post(
            f"{self.base_url}/agents/{self.agent_name}/recall",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Session-Token": self._session_token or "",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("results", [])

    def answer(self, question: str) -> str:
        """Ask a question grounded in stored memories (built-in RAG)."""
        import httpx

        r = httpx.post(
            f"{self.base_url}/agents/{self.agent_name}/answer",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Session-Token": self._session_token or "",
                "Content-Type": "application/json",
            },
            json={"question": question},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("answer", "")

    # -----------------------------------------------------------------------
    # Batch helpers
    # -----------------------------------------------------------------------

    def batch_remember(
        self, memories: list[tuple[str, str, dict[str, Any] | None]]
    ) -> dict:
        """Store up to 100 memories at once.

        Each tuple is (text, memory_type, metadata_or_None).
        """
        import httpx

        items = [
            {"text": t, "type": mt, "metadata": md or {}}
            for t, mt, md in memories
        ]
        r = httpx.post(
            f"{self.base_url}/agents/{self.agent_name}/batch-remember",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Session-Token": self._session_token or "",
                "Content-Type": "application/json",
            },
            json={"memories": items},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# LangGraph integration helpers
# ---------------------------------------------------------------------------

def build_memory_context(memories: list[dict]) -> str:
    """Format recalled memories as a natural-language context block."""
    if not memories:
        return ""

    lines = ["## Relevant Memories from Previous Sessions\n"]
    for i, m in enumerate(memories, 1):
        text = m.get("text", "")
        mtype = m.get("type", "unknown")
        conf = m.get("confidence", "N/A")
        ts = m.get("created_at", "")
        lines.append(
            f"  [{i}] ({mtype}, confidence={conf}) {text}"
        )
        if ts:
            lines[-1] += f"  [stored: {ts}]"
    return "\n".join(lines)


def extract_memories_from_tool_calls(state: dict) -> list[dict]:
    """Extract facts worth remembering from the agent's last response."""
    memories = []

    # Pull out user messages as observations
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "human":
            memories.append({
                "text": f"User said: {msg.content[:500]}",
                "type": "context",
            })

    # Pull out AI responses as learnings
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "ai" and msg.content:
            # Only store the first AI response we find (most recent)
            memories.append({
                "text": f"Agent responded: {msg.content[:500]}",
                "type": "learning",
            })
            break

    return memories
