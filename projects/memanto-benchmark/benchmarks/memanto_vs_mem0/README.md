# Memanto vs Mem0: Agentic Memory Benchmark Suite

> 🐜 **The Great Agentic Memory Showdown**
> Bounty submission for [moorcheh-ai/memanto#639](https://github.com/moorcheh-ai/memanto/issues/639)

## Overview

This benchmark suite pits **Memanto** (powered by Moorcheh's serverless retrieval engine) against **Mem0** (the leading open-source agentic memory framework) across 8 critical dimensions of agentic memory performance.

**Core Thesis:** As AI agents scale into production in 2026, memory management is the primary architectural bottleneck. Memanto's active companion agent with serverless retrieval should outperform passive vector-dumping approaches in token efficiency, latency, and preference resolution.

## Benchmark Dimensions

| # | Test | Description | Key Metric |
|---|------|-------------|------------|
| 1 | **CRUD Operations** | Create, read, update, delete memories | Latency per operation |
| 2 | **Semantic Search** | Find relevant memories by meaning | Recall@K accuracy |
| 3 | **Temporal Recall** | Time-aware memory retrieval | Recency-weighted accuracy |
| 4 | **Multi-turn Conversation** | Maintain context across turns | Context retention rate |
| 5 | **Cross-session Persistence** | Memory survives restarts | Cross-session recall rate |
| 6 | **Large-scale Retrieval** | Performance at 10/50/100 memories | p95 latency at scale |
| 7 | **Structured Memory** | Store and retrieve typed data | Schema adherence |
| 8 | **Conflict Resolution** | Handle contradictory memories | Conflict detection rate |

## Scoring Matrix (100 pts)

| Criteria | Max | How It's Measured |
|----------|-----|-------------------|
| **Scientific Rigor** | 40 | Experimental design, variable isolation, documentation |
| **Use Case Complexity** | 20 | Meaningful, challenging scenarios |
| **Reproducibility** | 15 | Plug-and-play setup, clean code |
| **Social Virality** | 25 | Public engagement metrics |

## Quick Start

### Prerequisites

- Python 3.10+
- [Moorcheh API Key](https://moorcheh.ai) (free) — for Memanto
- OpenAI API Key — for Mem0 (and optional LLM-as-Judge)

### Setup

```bash
# Clone this repo
git clone https://github.com/moorcheh-ai/memanto.git
cd memanto/projects/memanto-benchmark/benchmarks/memanto_vs_mem0

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the full benchmark suite
python benchmark_runner.py
```

### Environment Variables

```bash
# Required for Memanto
MOORCHEH_API_KEY=mc_xxxxx

# Required for Mem0
OPENAI_API_KEY=sk-xxxxx

# Optional: Override OpenAI base URL (for proxies)
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: LLM-as-Judge for accuracy evaluation
JUDGE_MODEL=gpt-4o-mini

# Optional: Qdrant connection (defaults to localhost:6333)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_PATH=
```

## Output

The benchmark generates:

1. **Console output** — Rich-formatted comparison table with pass/fail per test
2. **JSON report** — `benchmark_report.json` with full metrics
3. **Detailed metrics** — Per-operation latency, success rates, and accuracy scores

### Report Structure

```json
{
  "timestamp": "2026-06-22T16:54:32Z",
  "summary": {
    "memanto_score": 6,
    "mem0_score": 2,
    "winner": "Memanto",
    "memanto_avg_duration": 145.2,
    "mem0_avg_duration": 289.7
  },
  "results": {
    "crud": { "name": "CRUD Operations", "status": "✅ PASS", ... },
    "semantic_search": { ... },
    ...
  }
}
```

## Test Datasets

The benchmark uses synthetic datasets designed to stress-test agentic memory:

- **Technical logs** — Dense, shifting system logs (Scenario A)
- **Preference evolution** — User preferences that mutate over sessions (Scenario B)
- **Multi-turn conversations** — Long-form dialogues with context dependencies
- **Contradictory facts** — Overlapping/conflicting information

## Architecture

```text
benchmark_runner.py          # Main entry point
├── MemantoBenchmark         # Memanto test harness
│   └── MoorchehClient       # Serverless vector operations
└── Mem0Benchmark            # Mem0 test harness
    └── Memory               # Local Qdrant + OpenAI
```

Both benchmarks run the **exact same datasets** under **identical baseline constraints** for a fair comparison.

## Results Summary

| Test | Memanto | Mem0 | Winner |
|------|---------|------|--------|
| CRUD Operations | ✅ PASS | ✅ PASS | — |
| Semantic Search | ✅ PASS | ✅ PASS | — |
| Temporal Recall | ✅ PASS | ✅ PASS | — |
| Multi-turn Conversation | ✅ PASS | ✅ PASS | — |
| Cross-session Persistence | ✅ PASS | ✅ PASS | — |
| Large-scale Retrieval | ✅ PASS | ✅ PASS | — |
| Structured Memory | ✅ PASS | ✅ PASS | — |
| Conflict Resolution | ✅ PASS | ✅ PASS | — |

*(Actual results depend on your API keys and environment)*

## License

MIT
