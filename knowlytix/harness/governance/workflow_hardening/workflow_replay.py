# SPDX-License-Identifier: Apache-2.0
"""Workflow snapshot diff and assertion helpers.

Mirrors :mod:`governed_agent.policy_replay.compare_policy_replays`
for the workflow surface. Two snapshots can come from the same
workflow definition (rerun) or from two different builds; the
diff reports which fields drifted and which floating metrics
changed beyond their tolerance.

Floating-metric tolerance is set conservatively (1% absolute on
accuracies / AIRs) because CUDA non-determinism on a 5-model
training pass on Adult Income produces drift in the third-fourth
decimal. A bank with deterministic-CUDA infrastructure can
tighten this; the chapter ships the conservative default.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowlytix.harness.governance.workflow_hardening.workflow_snapshot import (
    WorkflowSnapshot,
)


@dataclass
class WorkflowDelta:
    n_phase_diffs: int
    cumulative_diffs: list[tuple[str, int, int]]   # (field, old, new)
    pass_rate_drift: float                          # |new - old|
    release_gate_changed: bool
    floating_drifts: list[tuple[str, float, float]]  # (key, old, new)
    fingerprint_match: bool

    @property
    def has_regressions(self) -> bool:
        return (
            not self.fingerprint_match
            or self.release_gate_changed
            or self.n_phase_diffs > 0
            or len(self.cumulative_diffs) > 0
        )

    def summary(self) -> str:
        rows = [
            "WorkflowDelta(",
            f"  fingerprint_match={self.fingerprint_match}",
            f"  release_gate_changed={self.release_gate_changed}",
            f"  phase_diffs={self.n_phase_diffs}",
            f"  cumulative_diffs={len(self.cumulative_diffs)}",
            f"  pass_rate_drift={self.pass_rate_drift:.6f}",
            f"  floating_drifts={len(self.floating_drifts)}",
        ]
        for k, o, n in self.cumulative_diffs[:5]:
            rows.append(f"    {k}: {o} -> {n}")
        for k, o, n in self.floating_drifts[:5]:
            rows.append(f"    {k}: {o:.4f} -> {n:.4f}")
        rows.append(")")
        return "\n".join(rows)


def compare_workflow_snapshots(
    old: WorkflowSnapshot,
    new: WorkflowSnapshot,
    *,
    floating_tolerance: float = 0.01,
) -> WorkflowDelta:
    """Diff two workflow snapshots; report drift on every dimension."""
    fp_match = old.fingerprint() == new.fingerprint()

    # Per-phase counts (compare matching phase names)
    phase_diffs = 0
    old_phases = {p.phase_name: p for p in old.phase_snapshots}
    new_phases = {p.phase_name: p for p in new.phase_snapshots}
    for name in sorted(set(old_phases) | set(new_phases)):
        op = old_phases.get(name)
        np_ = new_phases.get(name)
        if op is None or np_ is None:
            phase_diffs += 1
            continue
        if (op.tools_called != np_.tools_called
                or op.n_triples != np_.n_triples
                or op.n_enm != np_.n_enm
                or op.n_policy_blocks != np_.n_policy_blocks
                or op.n_claims != np_.n_claims
                or op.claim_type_counts != np_.claim_type_counts):
            phase_diffs += 1

    # Cumulative counts
    cum_diffs: list[tuple[str, int, int]] = []
    for k in (
        "cumulative_triples", "cumulative_enm",
        "cumulative_blocks", "cumulative_claims",
        "n_passed", "n_total",
    ):
        ov = getattr(old, k)
        nv = getattr(new, k)
        if ov != nv:
            cum_diffs.append((k, ov, nv))

    # Floating-metric drift
    floating_drifts: list[tuple[str, float, float]] = []
    for k in sorted(
        set(old.floating_metrics) | set(new.floating_metrics)
    ):
        ov = old.floating_metrics.get(k)
        nv = new.floating_metrics.get(k)
        if ov is None or nv is None:
            continue
        if abs(float(ov) - float(nv)) > floating_tolerance:
            floating_drifts.append((k, float(ov), float(nv)))

    return WorkflowDelta(
        n_phase_diffs=phase_diffs,
        cumulative_diffs=cum_diffs,
        pass_rate_drift=abs(old.pass_rate - new.pass_rate),
        release_gate_changed=(old.release_gate != new.release_gate),
        floating_drifts=floating_drifts,
        fingerprint_match=fp_match,
    )


class WorkflowReplayRegressionError(RuntimeError):
    """Workflow snapshot comparison detected a regression."""


def assert_no_workflow_regression(
    delta: WorkflowDelta,
) -> None:
    """Strict gate: raise if the fingerprint or release gate drifted.

    Floating-metric drift within tolerance does not raise --- those
    are reported separately and the audit log carries them for
    operator inspection.
    """
    problems: list[str] = []
    if not delta.fingerprint_match:
        problems.append("workflow fingerprint mismatch")
    if delta.release_gate_changed:
        problems.append("release gate decision changed")
    if delta.n_phase_diffs > 0:
        problems.append(
            f"{delta.n_phase_diffs} phase(s) drifted in counts/claims"
        )
    if delta.cumulative_diffs:
        problems.append(
            f"{len(delta.cumulative_diffs)} cumulative count(s) drifted"
        )
    if problems:
        raise WorkflowReplayRegressionError(
            "; ".join(problems) + "\n" + delta.summary()
        )
