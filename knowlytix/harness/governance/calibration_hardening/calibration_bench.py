# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks for the calibration hot path.

The full DOE sweep is expensive (each config runs the
ClaimVerifier on a labeled cohort). The microbenchmarks here
target the operations the regression suite repeats:

  * `snapshot.fingerprint()` — used on every CI check
  * `alt_best_config` — the alt-validator's path

Banks expect these to be sub-millisecond; the full DOE sweep
is reserved for less-frequent regression cadence.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from knowlytix.harness.governance.calibration_hardening.calibration_snapshot import (
    CalibrationSnapshot,
)


@dataclass
class CalibrationLatencyReport:
    operation: str
    n_calls: int
    p50_us: float
    p95_us: float
    p99_us: float
    p_max_us: float
    p_mean_us: float

    def summary(self) -> str:
        return (
            f"CalibrationLatencyReport({self.operation}, "
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


def bench_calibration_fingerprint(
    snapshot: CalibrationSnapshot,
    *,
    n_calls: int = 2000,
    warmup: int = 100,
) -> CalibrationLatencyReport:
    """Microbenchmark `CalibrationSnapshot.fingerprint()`."""
    for _ in range(warmup):
        snapshot.fingerprint()
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        snapshot.fingerprint()
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return CalibrationLatencyReport(
        operation="snapshot.fingerprint",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )


def bench_alt_best_config(
    sweep_results: list,
    *,
    n_calls: int = 5000,
    warmup: int = 100,
) -> CalibrationLatencyReport:
    """Microbenchmark `alt_best_config` over a sweep."""
    from knowlytix.harness.governance.calibration_hardening.calibration_alt_validator import (
        alt_best_config,
    )
    for _ in range(warmup):
        alt_best_config(sweep_results)
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        alt_best_config(sweep_results)
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return CalibrationLatencyReport(
        operation="alt_best_config",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )
