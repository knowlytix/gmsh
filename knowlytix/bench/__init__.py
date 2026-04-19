# SPDX-License-Identifier: Apache-2.0
"""
FinStructBench — Benchmark for Structured Retrieval from Financial Documents.

Auto-generates questions from document graph topology and evaluates
LLM accuracy against a provably correct graph traversal baseline.

Quick start::

    from knowlytix.bench import Benchmark
    from knowlytix.bench.instances import get_instance_path

    bench = Benchmark(get_instance_path("model_validation"))
    result = bench.run()            # graph-only (no LLM)
    bench.print_results(result)

With LLM evaluation::

    from knowlytix.bench.llm_caller import create_client

    client = create_client()        # requires ANTHROPIC_API_KEY
    result = bench.run(llm_client=client)
"""

__version__ = "0.1.0"

from knowlytix.bench.benchmark import Benchmark, BenchmarkResult
from knowlytix.bench.config import FinStructBenchSettings
from knowlytix.bench.generators import default_generators
from knowlytix.bench.generators.base import GeneratedQuestion
from knowlytix.bench.graph import DocumentGraph, ENMEntry, ENMKey, PhaseEncoder
from knowlytix.bench.ingest import ingest_markdown
from knowlytix.bench.instances import get_instance_path, list_instances
from knowlytix.bench.scorers import ScoreResult, score_answer

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "DocumentGraph",
    "ENMEntry",
    "ENMKey",
    "FinStructBenchSettings",
    "GeneratedQuestion",
    "PhaseEncoder",
    "ScoreResult",
    "default_generators",
    "get_instance_path",
    "ingest_markdown",
    "list_instances",
    "score_answer",
]
