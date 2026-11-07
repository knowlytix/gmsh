# SPDX-License-Identifier: Apache-2.0
"""Latency benchmarks for the multi-agent governance surface.

Three operations benchmarked:

  * ``can_execute`` -- the structural permission check (must be O(1)
    on hash sets).
  * ``handoff`` -- the orchestrator's handoff record creation.
  * ``test_boundary`` -- the boundary-test path used by the
    adversarial sweep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from knowlytix.harness.governance.agents.multi_agent import (
    MultiAgentOrchestrator,
)


@dataclass
class MultiAgentLatencyReport:
    operation: str
    n: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    slo_us: float
    slo_violations: int


def _quantiles_us(timings_s: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(timings_s, dtype=np.float64) * 1e6
    return (float(np.quantile(arr, 0.50)),
            float(np.quantile(arr, 0.95)),
            float(np.quantile(arr, 0.99)),
            float(arr.mean()))


def bench_can_execute(
    orch: MultiAgentOrchestrator,
    *,
    n: int = 1000,
    slo_us: float = 50.0,
    seed: int = 42,
) -> MultiAgentLatencyReport:
    """Microbenchmark agent.can_execute over n random (role, tool) pairs."""
    rng = np.random.default_rng(seed)
    role_names = list(orch.agents.keys())
    tools = ["load_dataset", "train_model", "diagnose_robustness",
             "diagnose_fairness", "predict", "leaderboard",
             "explain_pdp", "explain_shap", "data_summary",
             "preprocess_data"]
    pairs = [
        (role_names[rng.integers(0, len(role_names))],
         tools[rng.integers(0, len(tools))])
        for _ in range(n)
    ]
    # Warm-up
    for _ in range(20):
        orch.agents[pairs[0][0]].can_execute(pairs[0][1])
    timings: list[float] = []
    for role, tool in pairs:
        agent = orch.agents[role]
        t0 = time.perf_counter()
        agent.can_execute(tool)
        timings.append(time.perf_counter() - t0)
    p50, p95, p99, mean = _quantiles_us(timings)
    violations = sum(1 for t in timings if t * 1e6 > slo_us)
    return MultiAgentLatencyReport(
        operation="agent.can_execute",
        n=n,
        p50_us=p50, p95_us=p95, p99_us=p99,
        mean_us=mean,
        slo_us=slo_us,
        slo_violations=violations,
    )


def bench_handoff(
    orch: MultiAgentOrchestrator,
    *,
    n: int = 500,
    slo_us: float = 200.0,
) -> MultiAgentLatencyReport:
    """Microbenchmark orchestrator.handoff."""
    role_names = list(orch.agents.keys())
    if len(role_names) < 2:
        return MultiAgentLatencyReport(
            operation="orch.handoff", n=0, p50_us=0, p95_us=0,
            p99_us=0, mean_us=0, slo_us=slo_us, slo_violations=0,
        )
    timings: list[float] = []
    # Warm-up
    for _ in range(20):
        orch.handoff(role_names[0], role_names[1])
    for i in range(n):
        a = role_names[i % len(role_names)]
        b = role_names[(i + 1) % len(role_names)]
        t0 = time.perf_counter()
        orch.handoff(a, b)
        timings.append(time.perf_counter() - t0)
    p50, p95, p99, mean = _quantiles_us(timings)
    violations = sum(1 for t in timings if t * 1e6 > slo_us)
    return MultiAgentLatencyReport(
        operation="orch.handoff",
        n=n,
        p50_us=p50, p95_us=p95, p99_us=p99,
        mean_us=mean,
        slo_us=slo_us,
        slo_violations=violations,
    )


def bench_test_boundary(
    orch: MultiAgentOrchestrator,
    *,
    n: int = 500,
    slo_us: float = 200.0,
) -> MultiAgentLatencyReport:
    """Microbenchmark orch.test_boundary on forbidden calls."""
    role_names = list(orch.agents.keys())
    forbidden_pairs = []
    for rn in role_names:
        agent = orch.agents[rn]
        for tool in sorted(agent.role.forbidden_tools)[:3]:
            forbidden_pairs.append((rn, tool))
    if not forbidden_pairs:
        return MultiAgentLatencyReport(
            operation="orch.test_boundary", n=0, p50_us=0, p95_us=0,
            p99_us=0, mean_us=0, slo_us=slo_us, slo_violations=0,
        )
    timings: list[float] = []
    # Warm-up
    for _ in range(20):
        orch.test_boundary(*forbidden_pairs[0], {})
    for i in range(n):
        rn, tool = forbidden_pairs[i % len(forbidden_pairs)]
        t0 = time.perf_counter()
        orch.test_boundary(rn, tool, {})
        timings.append(time.perf_counter() - t0)
    p50, p95, p99, mean = _quantiles_us(timings)
    violations = sum(1 for t in timings if t * 1e6 > slo_us)
    return MultiAgentLatencyReport(
        operation="orch.test_boundary",
        n=n,
        p50_us=p50, p95_us=p95, p99_us=p99,
        mean_us=mean,
        slo_us=slo_us,
        slo_violations=violations,
    )
