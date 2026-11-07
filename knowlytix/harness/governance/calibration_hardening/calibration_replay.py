# SPDX-License-Identifier: Apache-2.0
"""Snapshot diff and assertion helpers for calibration regression.

Mirrors :mod:`workflow_replay` for the calibration surface. Two
calibration snapshots can be compared to detect drift in:

  * the calibrated optimum
  * any per-config sweep result
  * the grid keys (would indicate a different parameter set was
    swept)

Floating-numeric drift within `accuracy_tolerance` does not raise;
that is reported separately as informational drift.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowlytix.harness.governance.calibration_hardening.calibration_snapshot import (
    CalibrationSnapshot,
)


@dataclass
class CalibrationDelta:
    fingerprint_match: bool
    optimum_changed: bool
    n_sweep_diffs: int
    accuracy_drifts: list[tuple[str, float, float]]
    grid_keys_changed: bool
    detail: str = ""

    @property
    def has_regressions(self) -> bool:
        return (
            not self.fingerprint_match
            or self.optimum_changed
            or self.grid_keys_changed
        )

    def summary(self) -> str:
        rows = [
            "CalibrationDelta(",
            f"  fingerprint_match={self.fingerprint_match}",
            f"  optimum_changed={self.optimum_changed}",
            f"  n_sweep_diffs={self.n_sweep_diffs}",
            f"  accuracy_drifts={len(self.accuracy_drifts)}",
            f"  grid_keys_changed={self.grid_keys_changed}",
        ]
        for k, o, n in self.accuracy_drifts[:5]:
            rows.append(f"    {k}: {o:.6f} -> {n:.6f}")
        rows.append(")")
        return "\n".join(rows)


def compare_calibration_snapshots(
    old: CalibrationSnapshot,
    new: CalibrationSnapshot,
    *,
    accuracy_tolerance: float = 0.005,
) -> CalibrationDelta:
    """Diff two calibration snapshots; report drift on every dimension."""
    fp_match = old.fingerprint() == new.fingerprint()
    grid_changed = sorted(old.grid_keys) != sorted(new.grid_keys)

    # Optimum comparison: differ on any threshold field
    opt_changed = (
        old.optimum.theta_sim != new.optimum.theta_sim
        or old.optimum.theta_margin != new.optimum.theta_margin
        or old.optimum.theta_esc != new.optimum.theta_esc
        or old.optimum.d_max != new.optimum.d_max
        or old.optimum.budget != new.optimum.budget
    )

    # Per-config drift: index by (theta_sim, theta_margin, theta_esc,
    # d_max, budget)
    def _key(c):
        return (c.theta_sim, c.theta_margin, c.theta_esc,
                c.d_max, c.budget)

    old_by = {_key(c): c for c in old.sweep}
    new_by = {_key(c): c for c in new.sweep}
    drifts: list[tuple[str, float, float]] = []
    n_diffs = 0
    for k in sorted(set(old_by) | set(new_by)):
        oc = old_by.get(k)
        nc = new_by.get(k)
        if oc is None or nc is None:
            n_diffs += 1
            continue
        if oc.accuracy != nc.accuracy:
            n_diffs += 1
            if abs(oc.accuracy - nc.accuracy) > accuracy_tolerance:
                drifts.append((str(k), oc.accuracy, nc.accuracy))

    return CalibrationDelta(
        fingerprint_match=fp_match,
        optimum_changed=opt_changed,
        n_sweep_diffs=n_diffs,
        accuracy_drifts=drifts,
        grid_keys_changed=grid_changed,
    )


class CalibrationRegressionError(RuntimeError):
    """Calibration replay detected a regression."""


def assert_no_calibration_regression(
    delta: CalibrationDelta,
) -> None:
    """Strict gate: raise if the optimum changed or fingerprint drifted."""
    problems: list[str] = []
    if not delta.fingerprint_match:
        problems.append("calibration fingerprint mismatch")
    if delta.optimum_changed:
        problems.append("calibrated optimum changed")
    if delta.grid_keys_changed:
        problems.append("parameter grid keys changed")
    if problems:
        raise CalibrationRegressionError(
            "; ".join(problems) + "\n" + delta.summary()
        )
