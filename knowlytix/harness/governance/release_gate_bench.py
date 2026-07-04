# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for the tiered release gate.

The gate is pure scalar math; per-call latency should be far below
1 ms. This module exercises ``check_all_tiers`` over a 1000-call
loop and reports p50/p95/p99 against a configurable SLO ceiling.

The runtime monitor's per-run cost includes (a) the gate call,
(b) the SPC slope evaluation per metric, and (c) the audit-record
materialization. We measure the gate alone; the audit-record cost
is reported separately because it varies with bundle metadata size.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from knowlytix.harness.governance.release_gate_audit import to_audit_record
from knowlytix.harness.governance.release_gate_co_attestor import (
    SPCSlopeCoAttestor, SPCSlopeConfig,
)
from knowlytix.harness.governance.release_gates import (
    RunMetrics, RunTrend, TieredReleaseGate,
)


@dataclass
class ReleaseGateLatencyReport:
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


def bench_check_all_tiers(
    gate: TieredReleaseGate,
    *,
    n: int = 1000,
    slo_us: float = 1000.0,
    seed: int = 42,
) -> ReleaseGateLatencyReport:
    """Microbenchmark ``gate.check_all_tiers`` over ``n`` random RunMetrics.

    SLO target defaults to 1 ms (1000 µs); release-gate decisions
    sit on the orchestrator hot path so 1 ms is generous.
    """
    rng = np.random.default_rng(seed)
    samples = [
        RunMetrics(
            accuracy=float(rng.uniform(0.5, 1.0)),
            confidence=float(rng.uniform(0.0, 1.0)),
            robustness_drop=float(rng.uniform(0.0, 0.5)),
            fairness_air=float(rng.uniform(0.0, 1.0)),
            tool_calls=int(rng.integers(0, 20)),
            policy_blocks=int(rng.integers(0, 5)),
        )
        for _ in range(n)
    ]
    timings: list[float] = []
    # Warm-up
    for _ in range(20):
        gate.check_all_tiers(samples[0])
    for m in samples:
        t0 = time.perf_counter()
        gate.check_all_tiers(m)
        timings.append(time.perf_counter() - t0)
    p50, p95, p99, mean = _quantiles_us(timings)
    violations = sum(1 for t in timings if t * 1e6 > slo_us)
    return ReleaseGateLatencyReport(
        operation="gate.check_all_tiers",
        n=n,
        p50_us=p50, p95_us=p95, p99_us=p99,
        mean_us=mean,
        slo_us=slo_us,
        slo_violations=violations,
    )


def bench_co_attestor(
    gate: TieredReleaseGate,
    *,
    n: int = 1000,
    slo_us: float = 2000.0,
    seed: int = 42,
) -> ReleaseGateLatencyReport:
    """Microbenchmark the three-state co-attestor over ``n`` runs."""
    rng = np.random.default_rng(seed)
    attestor = SPCSlopeCoAttestor(gate, config=SPCSlopeConfig())
    samples = [
        RunMetrics(
            accuracy=float(rng.uniform(0.7, 1.0)),
            confidence=float(rng.uniform(0.4, 1.0)),
            robustness_drop=float(rng.uniform(0.0, 0.1)),
            fairness_air=float(rng.uniform(0.5, 1.0)),
            tool_calls=int(rng.integers(0, 20)),
            policy_blocks=int(rng.integers(0, 5)),
        )
        for _ in range(n)
    ]
    trend = RunTrend(metrics=samples[:10])
    timings: list[float] = []
    for _ in range(20):
        attestor.evaluate(samples[0], "action_taking", trend)
    for m in samples:
        t0 = time.perf_counter()
        attestor.evaluate(m, "action_taking", trend)
        timings.append(time.perf_counter() - t0)
    p50, p95, p99, mean = _quantiles_us(timings)
    violations = sum(1 for t in timings if t * 1e6 > slo_us)
    return ReleaseGateLatencyReport(
        operation="co_attestor.evaluate",
        n=n,
        p50_us=p50, p95_us=p95, p99_us=p99,
        mean_us=mean,
        slo_us=slo_us,
        slo_violations=violations,
    )


def bench_audit_record(
    gate: TieredReleaseGate,
    *,
    n: int = 1000,
    slo_us: float = 5000.0,
    seed: int = 42,
) -> ReleaseGateLatencyReport:
    """Microbenchmark the audit-record materialization."""
    rng = np.random.default_rng(seed)
    samples = [
        RunMetrics(
            accuracy=float(rng.uniform(0.7, 1.0)),
            confidence=float(rng.uniform(0.4, 1.0)),
            robustness_drop=float(rng.uniform(0.0, 0.1)),
            fairness_air=float(rng.uniform(0.5, 1.0)),
            tool_calls=int(rng.integers(0, 20)),
            policy_blocks=int(rng.integers(0, 5)),
            run_id=f"bench_{i:05d}",
        )
        for i in range(n)
    ]
    timings: list[float] = []
    for _ in range(20):
        to_audit_record(samples[0], "action_taking", gate)
    for m in samples:
        t0 = time.perf_counter()
        rec = to_audit_record(m, "action_taking", gate)
        _ = json.dumps(rec.to_dict(), sort_keys=True)
        timings.append(time.perf_counter() - t0)
    p50, p95, p99, mean = _quantiles_us(timings)
    violations = sum(1 for t in timings if t * 1e6 > slo_us)
    return ReleaseGateLatencyReport(
        operation="to_audit_record + json.dumps",
        n=n,
        p50_us=p50, p95_us=p95, p99_us=p99,
        mean_us=mean,
        slo_us=slo_us,
        slo_violations=violations,
    )
