# LangGraph + Memanto: Give Your Graph a Permanent Brain 🧠

A complete example of integrating **LangGraph** with **Memanto** — the active memory layer that gives your agents cross-session, long-term memory.

## What This Demonstrates

| Capability | How It Works |
|---|---|
| **Cross-Session Recall** | Agent remembers user preferences from yesterday without being told again |
| **Typed Semantic Memory** | Memories are stored with categories (preference, fact, learning, context) |
| **Zero-Ingestion Latency** | Memories are searchable the instant they're stored — no indexing delay |
| **Built-in RAG** | `answer()` generates LLM-grounded responses directly from memory context |

## Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│  recall_memory  │ ◄── Memanto.recall(query) — pulls relevant memories
│  (Memanto)      │      from previous sessions
└────────┬────────┘
         │  memory context injected as SystemMessage
         ▼
┌─────────────────┐
│    agent_node   │ ◄── LLM processes + responds (with memory-aware context)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ remember_memory │ ◄── Memanto.batch_remember() — stores new info
│  (Memanto)      │      (preferences, facts, learnings)
└────────┬────────┘
         │
         ▼
       Done
```

## Quick Start

### Prerequisites

- Python 3.10+
- A [Moorcheh API key](https://console.moorcheh.ai/api-keys) (free tier: 100K ops/month)
- An LLM API key — [OpenRouter](https://openrouter.ai/keys) (recommended, has free tier), OpenAI, or Anthropic

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — add your MOORCHEH_API_KEY and OPENROUTER_API_KEY (or other LLM provider key)

# 3. Run the full demo (both sessions)
python run_demo.py
```

### Running Sessions Individually

```bash
# Session 1: Store preferences into Memanto
python run_demo.py --session 1

# Session 2: Prove cross-session recall!
# (Run this in a SEPARATE terminal or after restarting)
python run_demo.py --session 2
```

## What Makes This Different

Unlike standard LangGraph memory (which only persists within the same thread/checkpoint), Memanto memories:

- **Survive across separate LangGraph threads**
- **Are semantically searchable** — query by meaning, not by ID
- **Include confidence scores and provenance**
- **Can be shared across different agents** (LangGraph agent + Claude Code + Cursor, all accessing the same memory namespace)

## The LangGraph Integration Pattern

The integration follows a clean **two-node pattern** around the agent node:

```python
graph = StateGraph(AgentState)
graph.add_node("recall_memory", recall_memory_node)    # ← Memanto.recall()
graph.add_node("agent",         agent_node)             # ← LLM (memory-aware)
graph.add_node("remember_memory", remember_memory_node) # ← Memanto.remember()

# Flow: recall → think → remember
graph.add_edge("recall_memory", "agent")
graph.add_edge("agent", "remember_memory")
```

You can drop this into any LangGraph agent by:
1. Adding `recall_memory_node` before your agent node
2. Adding `remember_memory_node` after your agent node
3. Passing the `MemantoMemory` client as context

## File Structure

```text
examples/langgraph-memanto/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
├── memanto_memory.py        # Memanto client wrapper + memory utilities
├── langgraph_agent.py       # LangGraph graph definition + nodes
└── run_demo.py              # Demo runner (session 1 + session 2)
```

## Making Your Own LangGraph + Memanto Agent

```python
from memanto_memory import MemantoMemory

# 1. Start Memanto
memory = MemantoMemory(agent_name="my-agent")
memory.activate_session()

# 2. Store memories
memory.remember("User prefers dark mode", memory_type="preference")

# 3. Recall memories (works even from a different process!)
results = memory.recall("What theme does the user like?")

# 4. Get grounded answers
answer = memory.answer("What should the theme be set to?")
```

## Bounty Requirements

✅ **Cross-Session Recall** — Session 2 proves the agent remembers Session 1's info
✅ **Clean, documented code** — Single folder with README, requirements, and runnable scripts
✅ **30-second GIF/video** — See [Recording a Demo](#recording-a-demo) below
✅ **Uses Memanto API** — Three primitives: `remember()`, `recall()`, `answer()`

### Recording a Demo

To create a 30-second demo GIF/video:

1. **Terminal recording** (ASCII screenscast):
   ```bash
   # Install asciinema
   pip install asciinema
   asciinema rec memanto-demo.cast
   python run_demo.py
   exit
   ```

2. **Screen recording** for GIF:
   ```bash
   # Linux: Use Peek
   sudo apt install peek
   peek
   ```

3. **Share on social media**:
   - Post on X with **#Memanto** and **@moorcheh_ai**
   - Or share on Reddit / LinkedIn
   - Include the link in your PR description!

## Social Traction Tips

The bounty is judged by **Social Traction Score** (until June 1st, 2026):

| Platform | Points |
|---|---|
| ❌ X/Twitter — Like | 1pt |
| ❌ X/Twitter — Repost | 3pts |
| ❌ X/Twitter — Bookmark | 3pts |
| 🐙 GitHub PR — 👍 or 🚀 reaction | 2pts each |
| 📊 Reddit — Post | 5pts base + 2pts per upvote |
