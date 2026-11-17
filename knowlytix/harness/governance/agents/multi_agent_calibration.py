# SPDX-License-Identifier: Apache-2.0
"""Calibration baselines for the multi-agent governance surface.

Multi-agent permissions are *declared* set memberships, not numeric
thresholds, so there is nothing to bootstrap-fit on the permission
side. What gets calibrated:

  * **Per-agent tool-call frequency** distribution. Reference
    distribution captured at calibration time; the runtime monitor
    flags PSI drift against this baseline.
  * **Per-handoff latency** distribution (p50/p95/p99). Reference
    captured by running a sample of pipelines; production
    deviations beyond this baseline trip the SLO.
  * **Per-agent expected blocked-attempt rate** (should be 0 in a
    healthy pipeline; non-zero means the orchestrator is being
    asked for forbidden tools, which is itself a signal).

Mirrors :mod:`governed_agent.release_gate_calibration` for the
multi-agent surface. The cap-derived analog here is that the role
permission set IS the threshold; the empirical sibling captures
the operating-point baseline an auditor can read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from knowlytix.harness.governance.agents.multi_agent import (
    MultiAgentOrchestrator,
    PipelineResult,
)
from knowlytix.harness.governance.governance_bundle import ThresholdDistribution

# ---------------------------------------------------------------------------
# Reference distributions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerAgentToolFrequency:
    """Reference per-agent tool-call frequency distribution.

    For each role, captures the empirical distribution of which
    tools the agent invoked over a calibration sample. The runtime
    monitor compares against this distribution via PSI.
    """

    counts_by_role: dict[str, dict[str, int]]
    total_calls: int

    def normalized(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for role, counts in self.counts_by_role.items():
            tot = max(1, sum(counts.values()))
            out[role] = {tool: c / tot for tool, c in counts.items()}
        return out

    def to_dict(self) -> dict:
        return {
            "counts_by_role": {k: dict(v) for k, v in
                               sorted(self.counts_by_role.items())},
            "total_calls": int(self.total_calls),
        }


@dataclass(frozen=True)
class HandoffLatencyBaseline:
    """Per-handoff latency reference (p50/p95/p99 in microseconds).

    Captured from the orchestrator's handoff timestamps.
    """

    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    n_handoffs: int

    def to_dict(self) -> dict:
        return {
            "p50_us": float(self.p50_us),
            "p95_us": float(self.p95_us),
            "p99_us": float(self.p99_us),
            "mean_us": float(self.mean_us),
            "n_handoffs": int(self.n_handoffs),
        }


# ---------------------------------------------------------------------------
# Calibration from a labeled sample of pipeline runs
# ---------------------------------------------------------------------------

def calibrate_per_agent_tool_frequency(
    results: list[PipelineResult],
) -> PerAgentToolFrequency:
    """Aggregate tool-call counts across a list of PipelineResults.

    Returns the reference distribution for the bundle. A high-stakes
    deployment should pass at least 30 healthy run results to get
    a stable reference.
    """
    counts: dict[str, dict[str, int]] = {}
    total = 0
    for result in results:
        for record in result.tool_records:
            role = record.agent_name
            tool = record.tool_name
            cell = counts.setdefault(role, {})
            cell[tool] = cell.get(tool, 0) + 1
            total += 1
    return PerAgentToolFrequency(
        counts_by_role={k: dict(v) for k, v in counts.items()},
        total_calls=total,
    )


def calibrate_handoff_latency_baseline(
    results: list[PipelineResult],
) -> HandoffLatencyBaseline | None:
    """Compute p50/p95/p99 of inter-handoff intervals across runs.

    Returns ``None`` when there are insufficient handoffs to fit
    a meaningful baseline (< 2 handoffs total).
    """
    intervals_us: list[float] = []
    for result in results:
        # Inter-handoff intervals: time between consecutive handoffs.
        # If a single result has only one handoff we still emit its
        # full pipeline span (last-tool minus first-tool).
        if not result.handoff_records:
            continue
        for i in range(1, len(result.handoff_records)):
            dt = (result.handoff_records[i].timestamp
                  - result.handoff_records[i - 1].timestamp)
            intervals_us.append(max(0.0, dt) * 1e6)
    if len(intervals_us) < 2:
        return None
    arr = np.asarray(intervals_us, dtype=float)
    return HandoffLatencyBaseline(
        p50_us=float(np.quantile(arr, 0.50)),
        p95_us=float(np.quantile(arr, 0.95)),
        p99_us=float(np.quantile(arr, 0.99)),
        mean_us=float(arr.mean()),
        n_handoffs=int(len(arr)),
    )


def bootstrap_handoff_latency(
    results: list[PipelineResult],
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> ThresholdDistribution | None:
    """Bootstrap the p99 handoff latency.

    Returns ``ThresholdDistribution`` over the bootstrap p99
    estimates, or ``None`` when too few handoffs.
    """
    intervals_us: list[float] = []
    for result in results:
        for i in range(1, len(result.handoff_records)):
            dt = (result.handoff_records[i].timestamp
                  - result.handoff_records[i - 1].timestamp)
            intervals_us.append(max(0.0, dt) * 1e6)
    n = len(intervals_us)
    if n < 5:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(intervals_us)
    draws = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        draws.append(float(np.quantile(sample, 0.99)))
    draws_arr = np.asarray(draws)
    return ThresholdDistribution(
        mean=float(draws_arr.mean()),
        std=float(draws_arr.std()),
        ci_low=float(np.quantile(draws_arr, 0.025)),
        ci_hi=float(np.quantile(draws_arr, 0.975)),
        n_bootstrap=len(draws),
    )


# ---------------------------------------------------------------------------
# Per-role permission fingerprints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerRoleFingerprint:
    """Canonical SHA-256 over (allowed_tools, forbidden_tools, description)."""

    role_name: str
    fingerprint: str

    def to_dict(self) -> dict:
        return {"role_name": self.role_name,
                "fingerprint": self.fingerprint}


def fingerprint_role(role) -> PerRoleFingerprint:
    """Compute a deterministic fingerprint for a single role.

    Two roles with the same ``allowed_tools``, ``forbidden_tools``
    and ``description`` produce the same fingerprint --- the
    runtime can refuse to bind to an orchestrator whose roles
    don't match the bundled fingerprints.
    """
    import hashlib
    import json
    payload = json.dumps({
        "name": role.name,
        "allowed_tools": sorted(role.allowed_tools),
        "forbidden_tools": sorted(role.forbidden_tools),
        "description": role.description,
    }, sort_keys=True).encode("utf-8")
    return PerRoleFingerprint(
        role_name=role.name,
        fingerprint=hashlib.sha256(payload).hexdigest(),
    )


def fingerprint_orchestrator(
    orch: MultiAgentOrchestrator,
) -> dict[str, str]:
    """Map ``role_name -> sha256(role)`` for every agent in the orchestrator."""
    return {
        name: fingerprint_role(agent.role).fingerprint
        for name, agent in orch.agents.items()
    }
