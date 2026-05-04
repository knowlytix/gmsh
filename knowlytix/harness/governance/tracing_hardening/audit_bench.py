# SPDX-License-Identifier: Apache-2.0
"""Latency benchmarks for the audit-trail hardening pipeline.

A high-stakes deployment must answer: how much latency does
sealing and verifying add per audit packet?

Banks expect sealing to be effectively free at the per-packet
level (a SHA-256 over a small JSON payload) and verification to
be linear in trail length with a tight constant. This module
gives the numbers a deployment review will ask for.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from knowlytix.harness.governance.tracing import AuditPacket
from knowlytix.harness.governance.tracing_hardening.hash_chain import (
    seal_audit_trail, verify_audit_chain,
)
from knowlytix.harness.governance.tracing_hardening.audit_replay import (
    replay_audit_metrics,
)


@dataclass
class AuditLatencyReport:
    operation: str
    n_packets: int
    n_calls: int
    p50_us: float
    p95_us: float
    p99_us: float
    p_max_us: float
    p_mean_us: float

    def summary(self) -> str:
        return (
            f"AuditLatencyReport({self.operation}, n_packets="
            f"{self.n_packets}, n_calls={self.n_calls}, "
            f"p50={self.p50_us:.1f}μs, "
            f"p95={self.p95_us:.1f}μs, "
            f"p99={self.p99_us:.1f}μs, "
            f"max={self.p_max_us:.1f}μs, "
            f"mean={self.p_mean_us:.1f}μs)"
        )


def _synthetic_trail(n: int) -> list[AuditPacket]:
    return [
        AuditPacket(
            step_num=i, tool_name=f"tool_{i}",
            policy_decision="allowed", blocking_rules=[],
            gms_scores={"plausibility": 0.5},
            timestamp=1000.0 + i,
            state_before=f"s{i}", state_after=f"s{i+1}",
        )
        for i in range(n)
    ]


def _percentiles(values_us: list[float]) -> tuple[float, float, float, float, float]:
    s = sorted(values_us)
    n = len(s)
    return (
        s[int(0.50 * n)],
        s[int(0.95 * n)],
        s[int(0.99 * n)],
        s[-1],
        statistics.mean(values_us),
    )


def bench_seal_audit_trail(
    *, n_packets: int = 100, n_calls: int = 1000, warmup: int = 50,
) -> AuditLatencyReport:
    """Microbenchmark of :func:`seal_audit_trail` on a fresh trail.

    Each call works on its own deep-copied trail (so the timing
    measures the seal step, not in-place idempotence). Warmup
    amortises any one-shot effects.
    """
    import copy
    base = _synthetic_trail(n_packets)
    for _ in range(warmup):
        seal_audit_trail(copy.deepcopy(base))

    latencies_us: list[float] = []
    for _ in range(n_calls):
        trail = copy.deepcopy(base)
        t0 = time.perf_counter_ns()
        seal_audit_trail(trail)
        dt = time.perf_counter_ns() - t0
        latencies_us.append(dt / 1000.0)

    p50, p95, p99, pmax, pmean = _percentiles(latencies_us)
    return AuditLatencyReport(
        operation="seal_audit_trail",
        n_packets=n_packets, n_calls=n_calls,
        p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )


def bench_verify_audit_chain(
    *, n_packets: int = 100, n_calls: int = 1000, warmup: int = 50,
) -> AuditLatencyReport:
    """Microbenchmark of :func:`verify_audit_chain` on a sealed trail."""
    trail = _synthetic_trail(n_packets)
    seal_audit_trail(trail)
    for _ in range(warmup):
        verify_audit_chain(trail)

    latencies_us: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        verify_audit_chain(trail)
        dt = time.perf_counter_ns() - t0
        latencies_us.append(dt / 1000.0)

    p50, p95, p99, pmax, pmean = _percentiles(latencies_us)
    return AuditLatencyReport(
        operation="verify_audit_chain",
        n_packets=n_packets, n_calls=n_calls,
        p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )


def bench_replay_audit_metrics(
    *, n_packets: int = 100, n_calls: int = 1000, warmup: int = 50,
) -> AuditLatencyReport:
    """Microbenchmark of metric replay on a sealed trail."""
    trail = _synthetic_trail(n_packets)
    seal_audit_trail(trail)
    for _ in range(warmup):
        replay_audit_metrics("bench", trail)

    latencies_us: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        replay_audit_metrics("bench", trail)
        dt = time.perf_counter_ns() - t0
        latencies_us.append(dt / 1000.0)

    p50, p95, p99, pmax, pmean = _percentiles(latencies_us)
    return AuditLatencyReport(
        operation="replay_audit_metrics",
        n_packets=n_packets, n_calls=n_calls,
        p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )
