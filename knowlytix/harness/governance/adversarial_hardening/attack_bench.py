# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks for the adversarial-hardening hot path.

Running the full `AdversarialSuite.run_all()` end-to-end takes
seconds (it instantiates a real `GovernedToolGateway`); that is
not a useful microbenchmark. The hot operations the regression
suite repeats are the snapshot fingerprint and the alt-validator
re-derivation. This module benchmarks those.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from knowlytix.harness.governance.adversarial_hardening.attack_snapshot import AttackSnapshot
from knowlytix.harness.governance.adversarial_hardening.attack_alt_validator import alt_blocked


@dataclass
class AttackLatencyReport:
    operation: str
    n_calls: int
    p50_us: float
    p95_us: float
    p99_us: float
    p_max_us: float
    p_mean_us: float

    def summary(self) -> str:
        return (
            f"AttackLatencyReport({self.operation}, n={self.n_calls}, "
            f"p50={self.p50_us:.1f}μs, p95={self.p95_us:.1f}μs, "
            f"p99={self.p99_us:.1f}μs, max={self.p_max_us:.1f}μs, "
            f"mean={self.p_mean_us:.1f}μs)"
        )


def _percentiles(values_us: list[float]):
    s = sorted(values_us)
    n = len(s)
    return (
        s[int(0.50 * n)],
        s[int(0.95 * n)],
        s[int(0.99 * n)],
        s[-1],
        statistics.mean(values_us),
    )


def bench_attack_fingerprint(
    snapshot: AttackSnapshot, *, n_calls: int = 2000, warmup: int = 100,
) -> AttackLatencyReport:
    """Microbenchmark `AttackSnapshot.fingerprint()`."""
    for _ in range(warmup):
        snapshot.fingerprint()
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        snapshot.fingerprint()
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return AttackLatencyReport(
        operation="snapshot.fingerprint",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )


def bench_alt_blocked(
    *, n_calls: int = 10000, warmup: int = 100,
) -> AttackLatencyReport:
    """Microbenchmark `alt_blocked` re-derivation."""
    for _ in range(warmup):
        alt_blocked("policy_engine")
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        alt_blocked("policy_engine")
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return AttackLatencyReport(
        operation="alt_blocked",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )
