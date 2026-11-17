# SPDX-License-Identifier: Apache-2.0
"""Drift monitor for the tiered release gate.

Distinct from the chapter's :class:`RunTrend` slope alert (which is
the *runtime* SPC signal). This monitor runs at the **bundle**
surface: PSI on the metric distribution against the bundle's
calibration cohort, plus SLO on the gate's own deferral rates
(action-taking pass rate, advisory pass rate). Severity:

  * ``stable``     -- both PSI and SLO healthy
  * ``slow_drift`` -- PSI breach but no SLO violation yet
  * ``slo_breach`` -- pass-rate moved outside operating-point band

Mirrors :mod:`governed_agent.policy_monitor` for the release-gate
surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from knowlytix.harness.governance.release_gate_bundle import ReleaseGateBundle
from knowlytix.harness.governance.release_gate_validation import LabeledRun
from knowlytix.harness.governance.release_gates import (
    RunMetrics,
    TieredReleaseGate,
)

Severity = Literal["stable", "slow_drift", "slo_breach"]


# ---------------------------------------------------------------------------
# PSI helper
# ---------------------------------------------------------------------------

def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, n_bins: int = 10,
) -> float:
    """Standard 10-bin PSI on continuous data."""
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    psi = 0.0
    eps = 1e-9
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            in_ref = (ref >= edges[i]) & (ref <= edges[i + 1])
            in_cur = (cur >= edges[i]) & (cur <= edges[i + 1])
        else:
            in_ref = (ref >= edges[i]) & (ref < edges[i + 1])
            in_cur = (cur >= edges[i]) & (cur < edges[i + 1])
        p_ref = max(in_ref.sum() / len(ref), eps)
        p_cur = max(in_cur.sum() / len(cur), eps)
        psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
    return float(psi)


# ---------------------------------------------------------------------------
# Monitor state + class
# ---------------------------------------------------------------------------

@dataclass
class ReleaseGateMonitorState:
    """Severity verdict + supporting numerics."""

    severity: Severity
    psi_per_metric: dict[str, float]
    pass_rates: dict[str, float]
    slo_violations: dict[str, str] = field(default_factory=dict)
    n_runs: int = 0
    alert: bool = False
    recalibrate: bool = False


class ReleaseGateMonitor:
    """Rolling-window drift monitor for the release gate.

    Construction:

      monitor = ReleaseGateMonitor(
          gate=gate, bundle=bundle, cohort=calibration_cohort,
          window_size=20, min_runs_for_alert=10,
      )

    Per-run:

      monitor.record(run_metrics, target_tier="action_taking")
      state = monitor.evaluate()
    """

    def __init__(
        self,
        gate: TieredReleaseGate,
        bundle: ReleaseGateBundle,
        cohort: list[LabeledRun],
        *,
        window_size: int = 20,
        min_runs_for_alert: int = 10,
        psi_warn: float = 0.10,
        psi_alert: float = 0.25,
        max_action_pass_rate_swing: float = 0.20,
    ) -> None:
        self.gate = gate
        self.bundle = bundle
        self.window_size = int(window_size)
        self.min_runs_for_alert = int(min_runs_for_alert)
        self.psi_warn = float(psi_warn)
        self.psi_alert = float(psi_alert)
        self.max_swing = float(max_action_pass_rate_swing)

        # Reference distributions per metric, drawn from the
        # calibration cohort.
        self._reference: dict[str, np.ndarray] = {}
        for name in ("accuracy", "confidence",
                     "robustness_drop", "fairness_air"):
            self._reference[name] = np.asarray(
                [getattr(lr.metrics, name) for lr in cohort],
                dtype=np.float64,
            )

        # Reference action-taking pass rate.
        n_at_pass = sum(
            1 for lr in cohort
            if gate.check(lr.metrics, "action_taking").passed
        )
        self._reference_action_pass_rate = (
            n_at_pass / len(cohort) if cohort else 0.0
        )

        self._observed: list[tuple[RunMetrics, str]] = []

    def record(self, metrics: RunMetrics, target_tier: str) -> None:
        self._observed.append((metrics, target_tier))
        if len(self._observed) > self.window_size:
            self._observed = self._observed[-self.window_size:]

    def evaluate(self) -> ReleaseGateMonitorState:
        n = len(self._observed)
        if n < self.min_runs_for_alert:
            return ReleaseGateMonitorState(
                severity="stable",
                psi_per_metric={},
                pass_rates={},
                n_runs=n,
                alert=False,
                recalibrate=False,
            )

        # PSI per metric
        psi: dict[str, float] = {}
        for name in ("accuracy", "confidence",
                     "robustness_drop", "fairness_air"):
            current = np.asarray(
                [getattr(m, name) for (m, _) in self._observed],
                dtype=np.float64,
            )
            psi[name] = population_stability_index(
                self._reference[name], current,
            )

        # Pass-rate per tier
        pass_counts = {t: 0 for t in self.gate.TIER_ORDER}
        per_tier_n = {t: 0 for t in self.gate.TIER_ORDER}
        for (m, _) in self._observed:
            for tier in self.gate.TIER_ORDER:
                if tier not in self.gate.TIERS_RUNTIME:
                    continue
                per_tier_n[tier] += 1
                if self.gate.check(m, tier).passed:
                    pass_counts[tier] += 1
        pass_rates = {
            t: (pass_counts[t] / per_tier_n[t]
                if per_tier_n[t] > 0 else 0.0)
            for t in self.gate.TIER_ORDER
            if t in self.gate.TIERS_RUNTIME
        }

        slo_violations: dict[str, str] = {}
        action_pass = pass_rates.get("action_taking", 0.0)
        swing = abs(action_pass - self._reference_action_pass_rate)
        if swing > self.max_swing:
            slo_violations["action_taking_swing"] = (
                f"{swing:.4f} > {self.max_swing:.2f}; ref="
                f"{self._reference_action_pass_rate:.4f}, "
                f"obs={action_pass:.4f}")

        # Severity decision
        any_alert_psi = any(v > self.psi_alert for v in psi.values())
        any_warn_psi = any(v > self.psi_warn for v in psi.values())
        if slo_violations:
            severity: Severity = "slo_breach"
            alert = True
            recalibrate = True
        elif any_alert_psi:
            severity = "slow_drift"
            alert = True
            recalibrate = False
        elif any_warn_psi:
            severity = "slow_drift"
            alert = False
            recalibrate = False
        else:
            severity = "stable"
            alert = False
            recalibrate = False

        return ReleaseGateMonitorState(
            severity=severity,
            psi_per_metric=psi,
            pass_rates=pass_rates,
            slo_violations=slo_violations,
            n_runs=n,
            alert=alert,
            recalibrate=recalibrate,
        )
