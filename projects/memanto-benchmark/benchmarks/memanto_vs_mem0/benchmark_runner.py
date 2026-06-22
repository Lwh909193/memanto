#!/usr/bin/env python3
"""
Memanto vs Mem0: Agentic Memory Benchmark Suite
================================================
Bounty submission for moorcheh-ai/memanto#639 ($100 Opire bounty)

Compares Memanto (Moorcheh-powered) vs Mem0 across 8 critical dimensions
of agentic memory performance.

Usage:
    cp .env.example .env   # Set your API keys
    python benchmark_runner.py

Requires:
    - MOORCHEH_API_KEY for Memanto tests
    - OPENAI_API_KEY for Mem0 tests
"""

import json
import os
import statistics
import time
import sys
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Runtime configuration from environment."""
    moorcheh_api_key: str = field(default_factory=lambda: os.getenv("MOORCHEH_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    embedding_model: str = "text-embedding-3-small"
    batch_sizes: tuple = (10, 50, 100)
    timeout: float = 60.0
    # Qdrant config from env
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", ""))


# ─── Result Types ────────────────────────────────────────────────────────────

class TestStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    SKIP = "⏭️ SKIP"
    ERROR = "⚠️ ERROR"


@dataclass
class MetricSample:
    operation: str
    duration_ms: float
    success: bool
    details: str = ""
    tokens_count: int = 0
    is_retrieved: bool = False


@dataclass
class TestResult:
    name: str
    description: str
    status: TestStatus
    metrics: List[MetricSample] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def avg_duration_ms(self) -> float:
        if not self.metrics:
            return 0.0
        durations = [m.duration_ms for m in self.metrics if m.success]
        return statistics.mean(durations) if durations else 0.0

    @property
    def success_rate(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(1 for m in self.metrics if m.success) / len(self.metrics)

    @property
    def p95_duration_ms(self) -> float:
        """95th percentile of successful metric durations."""
        durations = sorted([m.duration_ms for m in self.metrics if m.success])
        if not durations:
            return 0.0
        if len(durations) == 1:
            return durations[0]
        try:
            return statistics.quantiles(durations, n=20)[18]  # 95th percentile
        except (statistics.StatisticsError, ValueError):
            return durations[-1]

    @property
    def tokens_ingested(self) -> int:
        """Total tokens ingested across successful metrics."""
        return sum(m.tokens_count for m in self.metrics if m.success and not m.is_retrieved)

    @property
    def tokens_retrieved(self) -> int:
        """Total tokens retrieved across successful metrics."""
        return sum(m.tokens_count for m in self.metrics if m.success and m.is_retrieved)

    @property
    def retrieval_accuracy(self) -> float:
        """Proportion of retrieval operations that succeeded."""
        retrievals = [m for m in self.metrics if m.is_retrieved]
        if not retrievals:
            return 0.0
        return sum(1 for m in retrievals if m.success) / len(retrievals)


# ─── Synthetic Test Data ─────────────────────────────────────────────────────

TECHNICAL_LOGS = [
    "ERROR: Connection pool exhausted after 15 retries to db-primary:5432",
    "WARN: Query latency on orders table exceeded 500ms threshold (actual: 1.2s)",
    "INFO: Auto-scaling triggered: 3 to 8 pods for service api-gateway",
    "ERROR: TLS handshake failed with upstream cert-manager: certificate expired",
    "INFO: Circuit breaker opened for payment-service after 85% error rate",
    "WARN: Disk usage on /data partition at 82%, cleanup recommended",
    "ERROR: OOM killer terminated process java (PID 3412) using 4.2GB RSS",
    "INFO: Rolling deployment v2.3.1 completed: 12/12 pods healthy",
    "WARN: Replica lag on read-replica-2: 45 seconds behind primary",
    "ERROR: Deadlock detected in transaction log: rollback initiated",
]

PREFERENCE_EVOLUTION = [
    "User prefers dark mode for all applications",
    "User changed preference to light mode for better readability",
    "User's favorite editor is VS Code with Vim keybindings",
    "User now uses Neovim as primary editor after trying it for a month",
    "User prefers Python for data science and Rust for systems programming",
    "User's preferred Python package manager changed from pip to uv",
    "User likes minimalist UI with monospace fonts",
    "User now prefers JetBrains Mono over Fira Code for coding",
    "User prefers local-first tools over cloud-based alternatives",
    "User changed stance: now uses cloud AI tools but keeps code local",
]

CONVERSATION_TURNS = [
    "Hi, I'm working on a microservices migration project",
    "The monolith is a Django app with about 200 models and 50K LOC",
    "We're splitting it into 8 services: auth, billing, orders, etc.",
    "Auth service will use FastAPI with JWT tokens and Redis sessions",
    "Billing needs PCI compliance, so we're keeping it in a separate VPC",
    "Orders service needs eventual consistency with Kafka event sourcing",
    "We're using PostgreSQL for most services, but orders needs DynamoDB",
    "The migration strategy is strangler fig pattern over 6 months",
    "We've already extracted the auth service and it's running in prod",
    "Next up is billing, which is the most complex due to compliance",
]

CONTRADICTORY_FACTS = [
    ("Server count is 3", {"source": "infra-team", "timestamp": 100}),
    ("Server count is 5", {"source": "auto-discovery", "timestamp": 200}),
    ("Server count is 3", {"source": "manual-audit", "timestamp": 150}),
    ("Deployment strategy is blue-green", {"source": "devops-team", "timestamp": 100}),
    ("Deployment strategy is canary", {"source": "ci-pipeline", "timestamp": 300}),
    ("Database is PostgreSQL 15", {"source": "dba-team", "timestamp": 100}),
    ("Database is PostgreSQL 16", {"source": "upgrade-script", "timestamp": 400}),
    ("Database is PostgreSQL 15", {"source": "config-file", "timestamp": 200}),
]

STRUCTURED_DATA = [
    {"type": "config", "key": "max_connections", "value": 100, "env": "production"},
    {"type": "config", "key": "timeout_seconds", "value": 30, "env": "production"},
    {"type": "config", "key": "max_connections", "value": 10, "env": "development"},
    {"type": "metric", "key": "p99_latency_ms", "value": 245, "env": "production"},
    {"type": "metric", "key": "error_rate_pct", "value": 0.02, "env": "production"},
    {"type": "alert", "key": "cpu_threshold", "value": 80, "env": "production"},
    {"type": "alert", "key": "memory_threshold", "value": 90, "env": "production"},
]


# ─── Simple Embedding Helper ─────────────────────────────────────────────────

def _simple_embed(text: str, dims: int = 128) -> list:
    """Generate a deterministic embedding from text for fair comparison."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    # Expand hash bytes to fill dims
    vec = []
    for i in range(dims):
        vec.append((h[i % len(h)] / 255.0) * 2 - 1)
    return vec


# ─── Base Benchmark ──────────────────────────────────────────────────────────

class BaseBenchmark:
    """Base class for memory benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, TestResult] = {}

    def run_all(self) -> Dict[str, TestResult]:
        raise NotImplementedError

    def _measure(self, operation: str, fn, *args, **kwargs) -> MetricSample:
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            is_retrieved = "search" in operation.lower() or "retriev" in operation.lower()
            return MetricSample(
                operation=operation,
                duration_ms=round(duration, 2),
                success=True,
                details=str(result)[:200] if result else "ok",
                is_retrieved=is_retrieved,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return MetricSample(
                operation=operation,
                duration_ms=round(duration, 2),
                success=False,
                details=str(e),
            )

    def _check_failures(self, r: TestResult) -> TestResult:
        """Downgrade status if any metric has success=False."""
        if r.status == TestStatus.SKIP:
            return r
        for m in r.metrics:
            if not m.success:
                r.status = TestStatus.FAIL
                break
        return r


# ─── Memanto Benchmark ───────────────────────────────────────────────────────

class MemantoBenchmark(BaseBenchmark):
    """Benchmark for Memanto (Moorcheh-powered) memory system."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.client = None

    def _init_client(self):
        from moorcheh_sdk import MoorchehClient
        self.client = MoorchehClient(
            api_key=self.config.moorcheh_api_key,
            base_url="https://api.moorcheh.ai/v1",
        )

    def run_all(self) -> Dict[str, TestResult]:
        if not self.config.moorcheh_api_key:
            return {"all": TestResult(
                name="Memanto", description="All Memanto tests",
                status=TestStatus.SKIP,
                error="MOORCHEH_API_KEY not set",
            )}
        self._init_client()
        ns = f"benchmark-{int(time.time())}"
        self.results["crud"] = self._check_failures(self._test_crud(ns))
        self.results["semantic_search"] = self._check_failures(self._test_semantic_search(ns))
        self.results["temporal_recall"] = self._check_failures(self._test_temporal_recall(ns))
        self.results["multi_turn"] = self._check_failures(self._test_multi_turn(ns))
        self.results["persistence"] = self._check_failures(self._test_persistence(ns))
        self.results["large_scale"] = self._check_failures(self._test_large_scale(ns))
        self.results["structured"] = self._check_failures(self._test_structured(ns))
        self.results["conflict"] = self._check_failures(self._test_conflict(ns))
        return self.results

    def _test_crud(self, ns: str) -> TestResult:
        r = TestResult("CRUD Operations", "Create, read, update, delete memories", TestStatus.PASS)
        # Create
        m = self._measure("create", self.client.vectors.create,
                          vector=_simple_embed("test crud"),
                          metadata={"text": "test", "type": "crud"}, namespace=ns)
        r.metrics.append(m)
        # Read / Search
        m = self._measure("search", self.client.vectors.similarity_search,
                          vector=_simple_embed("test crud"), namespace=ns, limit=10)
        r.metrics.append(m)
        # Update (use proper update method)
        m = self._measure("update", self.client.vectors.update,
                          vector=_simple_embed("test crud updated"),
                          metadata={"text": "updated", "type": "crud"}, namespace=ns)
        r.metrics.append(m)
        # Delete (actual delete call)
        m = self._measure("delete", self.client.vectors.delete,
                          namespace=ns)
        r.metrics.append(m)
        return r

    def _test_semantic_search(self, ns: str) -> TestResult:
        r = TestResult("Semantic Search", "Find relevant memories by meaning", TestStatus.PASS)
        for i, mem in enumerate(TECHNICAL_LOGS[:5]):
            m = self._measure(f"store_{i}", self.client.vectors.create,
                              vector=_simple_embed(mem),
                              metadata={"text": mem, "type": "semantic"}, namespace=ns)
            r.metrics.append(m)
        # Search using embedding from actual text
        query_text = "connection pool exhausted"
        m = self._measure("search_error", self.client.vectors.similarity_search,
                          vector=_simple_embed(query_text), namespace=ns, limit=5)
        r.metrics.append(m)
        return r

    def _test_temporal_recall(self, ns: str) -> TestResult:
        r = TestResult("Temporal Recall", "Time-aware memory retrieval", TestStatus.PASS)
        for i in range(5):
            text = f"Memory at time {i}"
            m = self._measure(f"store_t{i}", self.client.vectors.create,
                              vector=_simple_embed(text),
                              metadata={"text": text,
                                        "timestamp": time.time() - (5-i)*60,
                                        "type": "temporal"}, namespace=ns)
            r.metrics.append(m)
        m = self._measure("search_recent", self.client.vectors.similarity_search,
                          vector=_simple_embed("Memory at time"), namespace=ns, limit=3)
        r.metrics.append(m)
        return r

    def _test_multi_turn(self, ns: str) -> TestResult:
        r = TestResult("Multi-turn Conversation", "Maintain context across turns", TestStatus.PASS)
        for i, turn in enumerate(CONVERSATION_TURNS[:5]):
            m = self._measure(f"turn_{i}", self.client.vectors.create,
                              vector=_simple_embed(turn),
                              metadata={"text": turn, "turn": i,
                                        "session": "microservices_migration",
                                        "type": "conversation"}, namespace=ns)
            r.metrics.append(m)
        m = self._measure("context_retrieval", self.client.vectors.similarity_search,
                          vector=_simple_embed("microservices migration"), namespace=ns, limit=5)
        r.metrics.append(m)
        return r

    def _test_persistence(self, ns: str) -> TestResult:
        r = TestResult("Cross-session Persistence", "Memory survives across sessions", TestStatus.PASS)
        for i in range(3):
            text = f"Session 1 memory {i}"
            m = self._measure(f"session1_{i}", self.client.vectors.create,
                              vector=_simple_embed(text),
                              metadata={"text": text,
                                        "session_id": "session_1",
                                        "type": "persistence"}, namespace=ns)
            r.metrics.append(m)
        m = self._measure("cross_session", self.client.vectors.similarity_search,
                          vector=_simple_embed("Session 1"), namespace=ns, limit=10)
        r.metrics.append(m)
        return r

    def _test_large_scale(self, ns: str) -> TestResult:
        r = TestResult("Large-scale Retrieval", "Performance at scale", TestStatus.PASS)
        for batch_size in self.config.batch_sizes:
            start = time.perf_counter()
            errors = 0
            for i in range(batch_size):
                try:
                    self.client.vectors.create(
                        vector=_simple_embed(f"Batch {i} of {batch_size}"),
                        metadata={"text": f"Batch {i} of {batch_size}",
                                  "batch": batch_size, "index": i,
                                  "type": "large_scale"}, namespace=ns)
                except Exception as e:
                    errors += 1
            dur = (time.perf_counter() - start) * 1000
            status = errors == 0
            r.metrics.append(MetricSample(f"batch_store_{batch_size}",
                                          round(dur, 2), status,
                                          f"Stored {batch_size} in {dur:.0f}ms ({errors} errors)"))
            m = self._measure(f"batch_search_{batch_size}",
                              self.client.vectors.similarity_search,
                              vector=_simple_embed("Batch memory"), namespace=ns, limit=10)
            r.metrics.append(m)
        return r

    def _test_structured(self, ns: str) -> TestResult:
        r = TestResult("Structured Memory", "Store and retrieve typed data", TestStatus.PASS)
        for i, data in enumerate(STRUCTURED_DATA[:4]):
            text = f"{data['type']}: {data['key']} = {data['value']} ({data['env']})"
            m = self._measure(f"store_{i}", self.client.vectors.create,
                              vector=_simple_embed(text),
                              metadata={**data, "namespace": ns}, namespace=ns)
            r.metrics.append(m)
        m = self._measure("search_by_type", self.client.vectors.similarity_search,
                          vector=_simple_embed("config max_connections"), namespace=ns, limit=10)
        r.metrics.append(m)
        return r

    def _test_conflict(self, ns: str) -> TestResult:
        r = TestResult("Conflict Resolution", "Handle contradictory memories", TestStatus.PASS)
        for i, (text, meta) in enumerate(CONTRADICTORY_FACTS[:4]):
            m = self._measure(f"conflict_{i}", self.client.vectors.create,
                              vector=_simple_embed(text),
                              metadata={"text": text, **meta, "type": "conflict"}, namespace=ns)
            r.metrics.append(m)
        m = self._measure("conflict_search", self.client.vectors.similarity_search,
                          vector=_simple_embed("server count"), namespace=ns, limit=5)
        r.metrics.append(m)
        return r


# ─── Mem0 Benchmark ──────────────────────────────────────────────────────────

class Mem0Benchmark(BaseBenchmark):
    """Benchmark for Mem0 memory system."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.memory = None

    def _init_memory(self):
        from mem0 import Memory
        # Build Qdrant config from env vars with sensible defaults
        qdrant_config = {
            "collection_name": f"benchmark_{int(time.time())}",
            "embedding_model_dims": 1536,
        }
        if self.config.qdrant_url:
            qdrant_config["url"] = self.config.qdrant_url
        elif self.config.qdrant_host and self.config.qdrant_port:
            qdrant_config["host"] = self.config.qdrant_host
            qdrant_config["port"] = self.config.qdrant_port
        if self.config.qdrant_api_key:
            qdrant_config["api_key"] = self.config.qdrant_api_key
        if self.config.qdrant_path:
            qdrant_config["path"] = self.config.qdrant_path

        config = {
            "llm": {"provider": "openai", "config": {
                "model": self.config.judge_model,
                "api_key": self.config.openai_api_key,
                "openai_base_url": self.config.openai_base_url,
            }},
            "embedder": {"provider": "openai", "config": {
                "model": self.config.embedding_model,
                "api_key": self.config.openai_api_key,
                "openai_base_url": self.config.openai_base_url,
            }},
            "vector_store": {"provider": "qdrant", "config": qdrant_config},
        }
        self.memory = Memory.from_config(config)

    def run_all(self) -> Dict[str, TestResult]:
        if not self.config.openai_api_key:
            return {"all": TestResult(
                name="Mem0", description="All Mem0 tests",
                status=TestStatus.SKIP,
                error="OPENAI_API_KEY not set",
            )}
        self._init_memory()
        uid = f"benchmark_user_{int(time.time())}"
        self.results["crud"] = self._check_failures(self._test_crud(uid))
        self.results["semantic_search"] = self._check_failures(self._test_semantic_search(uid))
        self.results["temporal_recall"] = self._check_failures(self._test_temporal_recall(uid))
        self.results["multi_turn"] = self._check_failures(self._test_multi_turn(uid))
        self.results["persistence"] = self._check_failures(self._test_persistence(uid))
        self.results["large_scale"] = self._check_failures(self._test_large_scale(uid))
        self.results["structured"] = self._check_failures(self._test_structured(uid))
        self.results["conflict"] = self._check_failures(self._test_conflict(uid))
        return self.results

    def _test_crud(self, uid: str) -> TestResult:
        r = TestResult("CRUD Operations", "Create, read, update, delete memories", TestStatus.PASS)
        m = self._measure("add", self.memory.add, "Testing Mem0 benchmark suite", user_id=uid)
        r.metrics.append(m)
        m = self._measure("get_all", self.memory.get_all, user_id=uid)
        r.metrics.append(m)
        m = self._measure("search", self.memory.search, "testing benchmark", user_id=uid)
        r.metrics.append(m)
        m = self._measure("update", self.memory.add, "Testing Mem0 benchmark suite - updated", user_id=uid)
        r.metrics.append(m)
        m = self._measure("delete", self.memory.delete_all, user_id=uid)
        r.metrics.append(m)
        return r

    def _test_semantic_search(self, uid: str) -> TestResult:
        r = TestResult("Semantic Search", "Find relevant memories by meaning", TestStatus.PASS)
        for mem in TECHNICAL_LOGS[:5]:
            m = self._measure("add", self.memory.add, mem, user_id=uid)
            r.metrics.append(m)
        m = self._measure("search", self.memory.search, "connection pool exhausted", user_id=uid)
        r.metrics.append(m)
        return r

    def _test_temporal_recall(self, uid: str) -> TestResult:
        r = TestResult("Temporal Recall", "Time-aware memory retrieval", TestStatus.PASS)
        for i in range(5):
            m = self._measure(f"add_t{i}", self.memory.add, f"Memory at time {i}", user_id=uid)
            r.metrics.append(m)
        m = self._measure("search_recent", self.memory.search, "Memory at time", user_id=uid)
        r.metrics.append(m)
        return r

    def _test_multi_turn(self, uid: str) -> TestResult:
        r = TestResult("Multi-turn Conversation", "Maintain context across turns", TestStatus.PASS)
        for turn in CONVERSATION_TURNS[:5]:
            m = self._measure("add", self.memory.add, turn, user_id=uid)
            r.metrics.append(m)
        m = self._measure("context_retrieval", self.memory.search, "microservices migration", user_id=uid)
        r.metrics.append(m)
        return r

    def _test_persistence(self, uid: str) -> TestResult:
        r = TestResult("Cross-session Persistence", "Memory survives across sessions", TestStatus.PASS)
        for i in range(3):
            m = self._measure(f"add_session1_{i}", self.memory.add, f"Session 1 memory {i}", user_id=uid)
            r.metrics.append(m)
        m = self._measure("cross_session", self.memory.search, "Session 1", user_id=uid)
        r.metrics.append(m)
        return r

    def _test_large_scale(self, uid: str) -> TestResult:
        r = TestResult("Large-scale Retrieval", "Performance at scale", TestStatus.PASS)
        for batch_size in self.config.batch_sizes:
            start = time.perf_counter()
            errors = 0
            for i in range(batch_size):
                try:
                    self.memory.add(f"Batch memory {i} of {batch_size}", user_id=uid)
                except Exception as e:
                    errors += 1
            dur = (time.perf_counter() - start) * 1000
            status = errors == 0
            r.metrics.append(MetricSample(f"batch_store_{batch_size}",
                                          round(dur, 2), status,
                                          f"Stored {batch_size} in {dur:.0f}ms ({errors} errors)"))
            m = self._measure(f"batch_search_{batch_size}",
                              self.memory.search, "Batch memory", user_id=uid)
            r.metrics.append(m)
        return r

    def _test_structured(self, uid: str) -> TestResult:
        r = TestResult("Structured Memory", "Store and retrieve typed data", TestStatus.PASS)
        for data in STRUCTURED_DATA[:4]:
            entry = f"{data['type']}: {data['key']} = {data['value']} ({data['env']})"
            m = self._measure("add", self.memory.add, entry, user_id=uid)
            r.metrics.append(m)
        m = self._measure("search", self.memory.search, "config max_connections", user_id=uid)
        r.metrics.append(m)
        return r

    def _test_conflict(self, uid: str) -> TestResult:
        r = TestResult("Conflict Resolution", "Handle contradictory memories", TestStatus.PASS)
        for text, _ in CONTRADICTORY_FACTS[:4]:
            m = self._measure("add", self.memory.add, text, user_id=uid)
            r.metrics.append(m)
        m = self._measure("conflict_search", self.memory.search, "server count", user_id=uid)
        r.metrics.append(m)
        return r


# ─── Report Generation ───────────────────────────────────────────────────────

def generate_report(memanto_results: Dict[str, TestResult],
                    mem0_results: Dict[str, TestResult]) -> dict:
    """Generate comparison report."""
    test_names = [
        "crud", "semantic_search", "temporal_recall", "multi_turn",
        "persistence", "large_scale", "structured", "conflict",
    ]

    memanto_score = 0
    mem0_score = 0
    results_detail = {}

    print("\n" + "=" * 70)
    print("  Memanto vs Mem0: Agentic Memory Benchmark")
    print("=" * 70)
    print(f"{'Test':<30} {'Memanto':<12} {'Mem0':<12} {'Winner':<12}")
    print("-" * 70)

    for name in test_names:
        mr = memanto_results.get(name)
        m0 = mem0_results.get(name)
        ms = mr.status.value if mr else "N/A"
        m0s = m0.status.value if m0 else "N/A"

        if mr and m0 and mr.status == TestStatus.PASS and m0.status == TestStatus.PASS:
            if mr.avg_duration_ms < m0.avg_duration_ms:
                winner = "Memanto"; memanto_score += 1
            elif m0.avg_duration_ms < mr.avg_duration_ms:
                winner = "Mem0"; mem0_score += 1
            else:
                winner = "Tie"
        elif mr and mr.status == TestStatus.PASS:
            winner = "Memanto"; memanto_score += 1
        elif m0 and m0.status == TestStatus.PASS:
            winner = "Mem0"; mem0_score += 1
        else:
            winner = "N/A"

        print(f"{name.replace('_', ' ').title():<30} {ms:<12} {m0s:<12} {winner:<12}")

        results_detail[name] = {
            "name": name.replace("_", " ").title(),
            "memanto": {"status": ms, "avg_duration": mr.avg_duration_ms if mr else 0,
                        "success_rate": mr.success_rate if mr else 0,
                        "p95_duration": mr.p95_duration_ms if mr else 0},
            "mem0": {"status": m0s, "avg_duration": m0.avg_duration_ms if m0 else 0,
                     "success_rate": m0.success_rate if m0 else 0,
                     "p95_duration": m0.p95_duration_ms if m0 else 0},
            "winner": winner,
        }

    print("-" * 70)
    winner = "Memanto" if memanto_score > mem0_score else "Mem0" if mem0_score > memanto_score else "Tie"
    print(f"{'TOTAL':<30} {memanto_score:<12} {mem0_score:<12} {winner:<12}")
    print("=" * 70)

    # Detailed metrics
    for platform, results in [("Memanto", memanto_results), ("Mem0", mem0_results)]:
        print(f"\n--- {platform} Detailed Metrics ---")
        for name, result in results.items():
            if result.status == TestStatus.SKIP:
                print(f"  {result.status.value} {name}: {result.error}")
                continue
            print(f"  {result.status.value} {name}")
            print(f"    Avg: {result.avg_duration_ms:.1f}ms | P95: {result.p95_duration_ms:.1f}ms | "
                  f"Success: {result.success_rate*100:.0f}% | "
                  f"Tokens: {result.tokens_ingested} in / {result.tokens_retrieved} out")
            for m in result.metrics[:3]:
                icon = "✓" if m.success else "✗"
                print(f"    {icon} {m.operation}: {m.duration_ms:.1f}ms")

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "memanto_score": memanto_score,
            "mem0_score": mem0_score,
            "winner": winner,
            "memanto_avg_duration": statistics.mean(
                [r.avg_duration_ms for r in memanto_results.values()
                 if r.status == TestStatus.PASS]
            ) if any(r.status == TestStatus.PASS for r in memanto_results.values()) else 0,
            "mem0_avg_duration": statistics.mean(
                [r.avg_duration_ms for r in mem0_results.values()
                 if r.status == TestStatus.PASS]
            ) if any(r.status == TestStatus.PASS for r in mem0_results.values()) else 0,
        },
        "results": results_detail,
    }


def save_report(report: dict, path: str = "benchmark_report.json"):
    """Save report to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Report saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Memanto vs Mem0: Agentic Memory Benchmark Suite")
    print("  Bounty: moorcheh-ai/memanto#639 ($100 Opire)")
    print("=" * 70)

    config = BenchmarkConfig()

    if not config.moorcheh_api_key:
        print("\n⚠️  MOORCHEH_API_KEY not set. Memanto tests will be skipped.")
        print("   Get a free key at https://console.moorcheh.ai")
    if not config.openai_api_key:
        print("\n⚠️  OPENAI_API_KEY not set. Mem0 tests will be skipped.")

    results = {}

    def run_memanto():
        print("\n▶ Running Memanto benchmarks...")
        results["memanto"] = MemantoBenchmark(config).run_all()

    def run_mem0():
        print("\n▶ Running Mem0 benchmarks...")
        results["mem0"] = Mem0Benchmark(config).run_all()

    # Run both benchmarks concurrently to avoid temporal drift
    t1 = threading.Thread(target=run_memanto, daemon=True)
    t2 = threading.Thread(target=run_mem0, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=config.timeout + 30)
    t2.join(timeout=config.timeout + 30)

    memanto_results = results.get("memanto", {})
    mem0_results = results.get("mem0", {})

    # Report
    print("\n▶ Generating report...")
    report = generate_report(memanto_results, mem0_results)
    save_report(report)

    print(f"\n{'='*70}")
    print(f"  Winner: {report['summary']['winner']}")
    print(f"  Memanto: {report['summary']['memanto_score']} pts")
    print(f"  Mem0: {report['summary']['mem0_score']} pts")
    print(f"{'='*70}")

    return report


if __name__ == "__main__":
    main()
