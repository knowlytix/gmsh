# SPDX-License-Identifier: Apache-2.0
"""Per-phase latency benchmark for the Adult Income workflow.

The full workflow is too heavy to run thousands of times for a
microbenchmark (training 5 sklearn models on 30k rows takes
seconds). Instead, this module reports per-phase wall-clock
latency from a single canonical run, plus a microbenchmark of
the snapshot + replay operations which *are* called repeatedly
in the regression suite.

Banks expect each phase to fit under an SLO that scales with the
data size. The default SLOs in `WorkflowOperatingPoint` reflect
typical Adult Income (30-50k rows) values; deployments scale
them per-dataset.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from knowlytix.harness.governance.workflow_hardening.workflow_snapshot import (
    WorkflowSnapshot,
)


@dataclass
class PhaseLatencyRecord:
    phase_name: str
    elapsed_s: float
    n_tools: int
    n_claims: int


@dataclass
class WorkflowLatencyReport:
    per_phase: list[PhaseLatencyRecord]
    total_s: float

    def summary(self) -> str:
        rows = [f"WorkflowLatencyReport(total={self.total_s:.2f}s,"]
        for r in self.per_phase:
            rows.append(
                f"  {r.phase_name}: {r.elapsed_s:.2f}s, "
                f"{r.n_tools} tools, {r.n_claims} claims"
            )
        rows.append(")")
        return "\n".join(rows)

    def per_phase_dict(self) -> dict[str, float]:
        return {r.phase_name: round(r.elapsed_s, 6) for r in self.per_phase}


def measure_phase_latencies(phase_results) -> WorkflowLatencyReport:
    """Pack `PhaseResult.elapsed_s` into a latency report."""
    records = [
        PhaseLatencyRecord(
            phase_name=p.phase_name,
            elapsed_s=p.elapsed_s,
            n_tools=len(p.tools_called),
            n_claims=len(p.claims),
        )
        for p in phase_results
    ]
    total = sum(r.elapsed_s for r in records)
    return WorkflowLatencyReport(per_phase=records, total_s=total)


# ---------------------------------------------------------------------------
# Microbenchmark: snapshot + replay operations
# ---------------------------------------------------------------------------


@dataclass
class SnapshotLatencyReport:
    operation: str
    n_calls: int
    p50_us: float
    p95_us: float
    p99_us: float
    p_max_us: float
    p_mean_us: float

    def summary(self) -> str:
        return (
            f"SnapshotLatencyReport({self.operation}, n={self.n_calls}, "
            f"p50={self.p50_us:.1f}μs, p95={self.p95_us:.1f}μs, "
            f"p99={self.p99_us:.1f}μs, max={self.p_max_us:.1f}μs, "
            f"mean={self.p_mean_us:.1f}μs)"
        )


def _percentiles(values: list[float]) -> tuple[float, float, float, float, float]:
    s = sorted(values)
    n = len(s)
    return (
        s[int(0.50 * n)], s[int(0.95 * n)], s[int(0.99 * n)],
        s[-1], statistics.mean(values),
    )


def bench_snapshot_fingerprint(
    snapshot: WorkflowSnapshot,
    *,
    n_calls: int = 1000, warmup: int = 50,
) -> SnapshotLatencyReport:
    """Microbenchmark of `WorkflowSnapshot.fingerprint()`."""
    for _ in range(warmup):
        snapshot.fingerprint()
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        snapshot.fingerprint()
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return SnapshotLatencyReport(
        operation="snapshot.fingerprint",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )


def bench_alt_release_gate(
    snapshots: list[WorkflowSnapshot],
    *,
    n_calls: int = 1000, warmup: int = 50,
) -> SnapshotLatencyReport:
    """Microbenchmark of the alt-validator's release-gate function."""
    from knowlytix.harness.governance.workflow_hardening.workflow_alt_validator import (
        alt_release_gate,
    )
    if not snapshots:
        return SnapshotLatencyReport(
            operation="alt_release_gate",
            n_calls=0, p50_us=0, p95_us=0, p99_us=0,
            p_max_us=0, p_mean_us=0,
        )
    snap = snapshots[0]
    for _ in range(warmup):
        alt_release_gate(
            cumulative_blocks=snap.cumulative_blocks,
            pass_rate=snap.pass_rate,
        )
    latencies: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        alt_release_gate(
            cumulative_blocks=snap.cumulative_blocks,
            pass_rate=snap.pass_rate,
        )
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    p50, p95, p99, pmax, pmean = _percentiles(latencies)
    return SnapshotLatencyReport(
        operation="alt_release_gate",
        n_calls=n_calls, p50_us=p50, p95_us=p95, p99_us=p99,
        p_max_us=pmax, p_mean_us=pmean,
    )
