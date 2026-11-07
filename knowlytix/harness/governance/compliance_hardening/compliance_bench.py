# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks for the compliance regression hot path."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from knowlytix.harness.governance.compliance_hardening.compliance_alt_validator import (
    alt_overall_status,
)
from knowlytix.harness.governance.compliance_hardening.compliance_snapshot import (
    ComplianceSnapshot,
)


@dataclass
class ComplianceLatencyReport:
    operation: str
    n_calls: int
    p50_us: float
    p95_us: float
    p99_us: float
    p_max_us: float
    p_mean_us: float

    def summary(self) -> str:
        return (
            f"ComplianceLatencyReport({self.operation}, "
            f"n={self.n_calls}, "
            f"p50={self.p50_us:.1f}μs, p95={self.p95_us:.1f}μs, "
            f"p99={self.p99_us:.1f}μs, max={self.p_max_us:.1f}μs, "
            f"mean={self.p_mean_us:.1f}μs)"
        )


def _percentiles(values_us):
    s = sorted(values_us)
    n = len(s)
    return (
        s[int(0.50 * n)], s[int(0.95 * n)], s[int(0.99 * n)],
        s[-1], statistics.mean(values_us),
    )


def bench_compliance_fingerprint(
    snapshot: ComplianceSnapshot,
    *,
    n_calls: int = 2000, warmup: int = 100,
) -> ComplianceLatencyReport:
    """Microbenchmark `ComplianceSnapshot.fingerprint()`."""
    for _ in range(warmup):
        snapshot.fingerprint()
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        snapshot.fingerprint()
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return ComplianceLatencyReport(
        operation="snapshot.fingerprint",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )


def bench_alt_overall_status(
    statuses: list[str],
    *,
    n_calls: int = 10000, warmup: int = 100,
) -> ComplianceLatencyReport:
    """Microbenchmark `alt_overall_status`."""
    for _ in range(warmup):
        alt_overall_status(statuses)
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        alt_overall_status(statuses)
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return ComplianceLatencyReport(
        operation="alt_overall_status",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )
