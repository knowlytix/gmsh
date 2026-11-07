# SPDX-License-Identifier: Apache-2.0
"""Online drift + operating-point monitor for a deployed PolicyEngine.

Mirrors :mod:`governed_agent.drift_monitor` and
:mod:`governed_agent.orchestrator_monitor` for the policy surface.
The monitor consumes runtime decisions, tracks block / review /
allow rates against the bundle's :class:`PolicyOperatingPoint`, and
PSI-tracks the Layer 3 plausibility distribution against a baseline
captured at calibration time.

Two distinct alert paths:

  * **SLO breach** — block or review rate moves outside the bundle's
    operating point. Often actionable without recalibration (e.g.
    a deployment-config issue).
  * **PSI drift** — Layer 3 plausibility distribution differs from
    the calibration distribution. Usually requires recalibration on
    a refreshed corpus.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.harness.governance.policy_bundle import PolicyBundle


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class PolicyMonitorState:
    n_observed: int
    window_full: bool
    block_rate: float
    review_rate: float
    allow_rate: float
    psi_plausibility: float | None
    severity: str            # "stable" | "drift" | "slo_breach"
    alert: bool
    recalibrate: bool
    slo_violation: str | None = None

    def summary(self) -> str:
        psi = (f"{self.psi_plausibility:.3f}"
               if self.psi_plausibility is not None else "N/A")
        slo = (f", slo_violation={self.slo_violation}"
               if self.slo_violation else "")
        return (
            f"PolicyMonitorState(n={self.n_observed}, "
            f"window_full={self.window_full}, "
            f"block={self.block_rate:.3f}, "
            f"review={self.review_rate:.3f}, "
            f"allow={self.allow_rate:.3f}, "
            f"psi_plausibility={psi}, "
            f"severity={self.severity}, "
            f"alert={self.alert}, "
            f"recalibrate={self.recalibrate}{slo})"
        )


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class PolicyMonitor:
    """Stateful rolling-window monitor for a deployed PolicyEngine.

    Usage::

        monitor = PolicyMonitor(bundle, window_size=500)
        for decision, verdict in stream:
            monitor.record(decision, verdict)
            if monitor.should_recalibrate():
                trigger_offline_recalibration()
            if monitor.should_alert():
                page_oncall()

    The monitor records the attestor verdict (BLOCK/ALLOW/REVIEW)
    and the L3 plausibility score (when available) for PSI drift.
    """

    def __init__(
        self,
        bundle: "PolicyBundle",
        *,
        window_size: int = 500,
        min_runs_for_alert: int = 50,
        psi_alert: float = 0.10,
        psi_recalibrate: float = 0.20,
    ):
        self.bundle = bundle
        self.window_size = window_size
        self.min_runs_for_alert = min_runs_for_alert
        self.psi_alert = psi_alert
        self.psi_recalibrate = psi_recalibrate

        self._verdicts: deque[str] = deque(maxlen=window_size)
        self._plausibility: deque[float] = deque(maxlen=window_size)
        # Baseline plausibility samples — populate at calibration time.
        self._baseline: np.ndarray | None = None

    def set_baseline_plausibility(self, samples: np.ndarray) -> None:
        """Stamp the baseline L3 plausibility distribution.

        Call once at calibration time with the cohort's plausibility
        scores. The monitor PSIs runtime against this.
        """
        self._baseline = np.asarray(samples, dtype=float)

    def record(self, decision, verdict) -> None:
        """Record one runtime (decision, attestor_verdict) pair."""
        self._verdicts.append(verdict.decision)
        plaus = decision.gms_scores.get("plausibility")
        if plaus is not None:
            self._plausibility.append(float(plaus))

    def state(self) -> PolicyMonitorState:
        n = len(self._verdicts)
        if n == 0:
            return PolicyMonitorState(
                n_observed=0,
                window_full=False,
                block_rate=0.0,
                review_rate=0.0,
                allow_rate=0.0,
                psi_plausibility=None,
                severity="stable",
                alert=False,
                recalibrate=False,
            )

        n_block = sum(1 for v in self._verdicts if v == "BLOCK")
        n_review = sum(1 for v in self._verdicts if v == "REVIEW")
        n_allow = sum(1 for v in self._verdicts if v == "ALLOW")

        block_rate = n_block / n
        review_rate = n_review / n
        allow_rate = n_allow / n

        # PSI on plausibility against baseline
        from knowlytix.harness.governance.drift_monitor import (
            population_stability_index,
        )
        psi: float | None = None
        if (self._baseline is not None
                and len(self._plausibility) >= self.min_runs_for_alert):
            psi = population_stability_index(
                self._baseline,
                np.asarray(self._plausibility, dtype=float),
            )

        # SLO check
        op = self.bundle.operating_point
        slo_violation: str | None = None
        if op.violates(block_rate=block_rate, review_rate=review_rate):
            if review_rate > op.max_review_rate:
                slo_violation = (
                    f"review_rate {review_rate:.3f} > "
                    f"max {op.max_review_rate:.3f}"
                )
            elif block_rate > op.max_block_rate:
                slo_violation = (
                    f"block_rate {block_rate:.3f} > "
                    f"max {op.max_block_rate:.3f}"
                )
            elif block_rate < op.min_block_rate:
                slo_violation = (
                    f"block_rate {block_rate:.3f} < "
                    f"min {op.min_block_rate:.3f}"
                )

        # Severity
        if psi is not None and psi >= self.psi_recalibrate:
            severity = "drift"
            alert = True
            recalibrate = True
        elif slo_violation is not None and n >= self.min_runs_for_alert:
            severity = "slo_breach"
            alert = True
            recalibrate = False
        elif psi is not None and psi >= self.psi_alert:
            severity = "drift"
            alert = True
            recalibrate = False
        else:
            severity = "stable"
            alert = False
            recalibrate = False

        return PolicyMonitorState(
            n_observed=n,
            window_full=(n >= self.window_size),
            block_rate=block_rate,
            review_rate=review_rate,
            allow_rate=allow_rate,
            psi_plausibility=psi,
            severity=severity,
            alert=alert,
            recalibrate=recalibrate,
            slo_violation=slo_violation,
        )

    def should_alert(self) -> bool:
        return self.state().alert

    def should_recalibrate(self) -> bool:
        return self.state().recalibrate
