# SPDX-License-Identifier: Apache-2.0
"""GMSH Performance Profiling — GMS-native latency and SLA analysis.

Encodes observed latencies as phase angles on S^1 via the GMS
``PhaseEncoder``.  SLA compliance is measured as ``tension_energy``
between the observed latency embedding and the SLA target embedding:

- Tension near 0 → within SLA (agreement)
- Tension near √2 → borderline (irrelevance)
- Tension near 2 → severe SLA violation (contradiction)

This geometric framing makes latency analysis composable with other
GMS-based scores (hallucination geodesic, consistency tension, etc.).

Usage::

    profiler = PerformanceProfiler(store, sla_latency_ms=5000.0)
    profiler.record(tool_name="gms_query", latency_ms=1200.0)
    profiler.record(tool_name="gms_lookup", latency_ms=8500.0)
    report = profiler.report()
    print(report.sla_pass_rate, report.p95_latency_ms)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Phase encoding helpers (S^1)
# ---------------------------------------------------------------------------

def _latency_to_phase(latency_ms: float, sla_ms: float) -> float:
    """Map latency to an angle in [0, pi].

    0 ms → 0 (perfect), sla_ms → pi/2 (borderline), 2*sla_ms → pi (severe).
    Values beyond 2*sla_ms are clamped to pi.
    """
    if sla_ms <= 0:
        return 0.0
    ratio = latency_ms / sla_ms
    # Map [0, 2] → [0, pi], clamp at pi
    angle = min(ratio, 2.0) * (math.pi / 2.0)
    return angle


def _tension_from_phases(phase_a: float, phase_b: float) -> float:
    """Compute tension energy between two S^1 phase angles.

    Uses the unit-vector formula: E = 2 sin(|θ_a - θ_b| / 2).
    Result is in [0, 2]: 0 = agreement, 2 = contradiction.
    """
    delta = abs(phase_a - phase_b)
    return 2.0 * math.sin(delta / 2.0)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PerformanceRecord:
    """A single latency measurement with GMS phase encoding.

    Attributes
    ----------
    tool_name : str
        Tool or operation that was timed.
    latency_ms : float
        Observed latency in milliseconds.
    latency_phase : float
        Latency encoded as an angle on S^1 in [0, pi].
    sla_phase : float
        SLA target encoded as an angle (always pi/2 for the boundary).
    sla_tension : float
        Tension energy between observed and SLA target phases.
        0 = within SLA, approaching 2 = severe violation.
    within_sla : bool
        Whether latency_ms <= sla_latency_ms.
    metadata : dict
        Arbitrary metadata (e.g., question ID, run ID).
    """

    tool_name: str = ""
    latency_ms: float = 0.0
    latency_phase: float = 0.0
    sla_phase: float = 0.0
    sla_tension: float = 0.0
    within_sla: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Aggregate performance report with GMS geometric SLA scores.

    Attributes
    ----------
    n_records : int
        Total number of latency records.
    sla_latency_ms : float
        SLA target in milliseconds.
    sla_pass_rate : float
        Fraction of records within SLA.
    mean_latency_ms : float
        Mean observed latency.
    p50_latency_ms : float
        Median (50th percentile) latency.
    p95_latency_ms : float
        95th percentile latency.
    p99_latency_ms : float
        99th percentile latency.
    mean_sla_tension : float
        Mean tension energy vs SLA target across all records.
    max_sla_tension : float
        Worst-case tension energy.
    gms_sla_geodesic : float or None
        GMS ``score_triple("sla_compliance", "measured_at", "run")``
        geodesic — overall run-level plausibility of SLA adherence.
    per_tool : dict[str, dict]
        Per-tool breakdown with mean/p95 latency and tension.
    """

    n_records: int = 0
    sla_latency_ms: float = 0.0
    sla_pass_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_sla_tension: float = 0.0
    max_sla_tension: float = 0.0
    gms_sla_geodesic: float | None = None
    per_tool: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_records": self.n_records,
            "sla_latency_ms": self.sla_latency_ms,
            "sla_pass_rate": round(self.sla_pass_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "mean_sla_tension": round(self.mean_sla_tension, 4),
            "max_sla_tension": round(self.max_sla_tension, 4),
            "gms_sla_geodesic": (
                round(self.gms_sla_geodesic, 4)
                if self.gms_sla_geodesic is not None else None
            ),
            "per_tool": self.per_tool,
        }


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class PerformanceProfiler:
    """GMS-native performance profiler using S^1 phase encoding.

    Records latency measurements, encodes them as phase angles, and
    computes tension energy against the SLA target for geometric SLA
    analysis.

    Parameters
    ----------
    store : GMSExpertStore or None
        GMS store for run-level SLA plausibility scoring via
        ``score_triple``.  If None, ``gms_sla_geodesic`` is omitted.
    sla_latency_ms : float
        SLA latency target in milliseconds.  The SLA boundary maps to
        phase = pi/2 on S^1.
    """

    def __init__(
        self,
        store: "GMSExpertStore | None" = None,
        sla_latency_ms: float = 5000.0,
    ):
        self.store = store
        self.sla_latency_ms = sla_latency_ms
        self._sla_phase = math.pi / 2.0  # SLA boundary on S^1
        self._records: list[PerformanceRecord] = []

    def record(
        self,
        tool_name: str,
        latency_ms: float,
        metadata: dict | None = None,
    ) -> PerformanceRecord:
        """Record a latency measurement with phase encoding.

        Parameters
        ----------
        tool_name : str
            Tool or operation name.
        latency_ms : float
            Observed latency in milliseconds.
        metadata : dict or None
            Optional metadata to attach.

        Returns
        -------
        PerformanceRecord
            The encoded record.
        """
        phase = _latency_to_phase(latency_ms, self.sla_latency_ms)
        tension = _tension_from_phases(phase, self._sla_phase)

        rec = PerformanceRecord(
            tool_name=tool_name,
            latency_ms=latency_ms,
            latency_phase=phase,
            sla_phase=self._sla_phase,
            sla_tension=tension,
            within_sla=latency_ms <= self.sla_latency_ms,
            metadata=metadata or {},
        )
        self._records.append(rec)
        return rec

    def report(self, run_id: str = "") -> PerformanceReport:
        """Generate an aggregate performance report.

        Parameters
        ----------
        run_id : str
            Run identifier for GMS-level SLA scoring.

        Returns
        -------
        PerformanceReport
            Aggregate statistics with geometric SLA scores.
        """
        if not self._records:
            return PerformanceReport(sla_latency_ms=self.sla_latency_ms)

        latencies = [r.latency_ms for r in self._records]
        tensions = [r.sla_tension for r in self._records]
        n = len(latencies)
        sorted_lat = sorted(latencies)

        def _percentile(data: list[float], p: float) -> float:
            k = (len(data) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[f] * (c - k) + data[c] * (k - f)

        # Per-tool breakdown
        per_tool: dict[str, dict] = {}
        tool_groups: dict[str, list[PerformanceRecord]] = {}
        for r in self._records:
            tool_groups.setdefault(r.tool_name, []).append(r)

        for tname, recs in tool_groups.items():
            t_lats = sorted(r.latency_ms for r in recs)
            t_tensions = [r.sla_tension for r in recs]
            per_tool[tname] = {
                "n": len(recs),
                "mean_latency_ms": round(statistics.mean(t_lats), 2),
                "p95_latency_ms": round(
                    _percentile(t_lats, 95.0), 2,
                ),
                "mean_sla_tension": round(statistics.mean(t_tensions), 4),
                "sla_pass_rate": round(
                    sum(1 for r in recs if r.within_sla) / len(recs), 4,
                ),
            }

        # GMS-level SLA plausibility
        gms_geodesic = None
        if self.store is not None and run_id:
            try:
                score = self.store.score_triple(
                    "sla_compliance", "measured_at", run_id,
                )
                gms_geodesic = float(score) if score is not None else None
            except Exception:
                pass

        return PerformanceReport(
            n_records=n,
            sla_latency_ms=self.sla_latency_ms,
            sla_pass_rate=sum(1 for r in self._records if r.within_sla) / n,
            mean_latency_ms=statistics.mean(latencies),
            p50_latency_ms=_percentile(sorted_lat, 50.0),
            p95_latency_ms=_percentile(sorted_lat, 95.0),
            p99_latency_ms=_percentile(sorted_lat, 99.0),
            mean_sla_tension=statistics.mean(tensions),
            max_sla_tension=max(tensions),
            gms_sla_geodesic=gms_geodesic,
            per_tool=per_tool,
        )

    @property
    def records(self) -> list[PerformanceRecord]:
        """Return all recorded measurements."""
        return list(self._records)

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._records.clear()
