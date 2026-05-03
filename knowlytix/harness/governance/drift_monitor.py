# SPDX-License-Identifier: Apache-2.0
"""Online drift detection for the runtime gate.

The bundle carries ``drift_baseline_te`` — the tension-energy
distribution from the calibration cohort. The monitor consumes
runtime telemetry and reports how far the runtime distribution has
drifted from the baseline. When drift exceeds a configured
threshold, the monitor signals that the bundle needs recalibration.

Drift metric: **Population Stability Index (PSI)**.

  PSI = Σ_i (p_i - q_i) * ln(p_i / q_i)

where ``p_i`` is the current bin proportion and ``q_i`` the baseline.
Standard interpretation in model-risk practice:

  PSI < 0.10   stable
  0.10–0.20    minor shift (advisory)
  > 0.20       significant shift (require recalibration)

The monitor also tracks abstain rate vs baseline expectation, since
a runtime that suddenly abstains far more (or less) than the
calibration cohort suggests is usually drifting even when the
tension distribution looks stable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.harness.governance.governance_bundle import GovernanceBundle
    from knowlytix.harness.governance.runtime_gate import RuntimeVerdict


@dataclass
class DriftState:
    n_observed: int
    window_full: bool
    psi_te: float | None
    abstain_rate: float
    review_rate: float
    abstain_rate_delta: float | None
    alert: bool
    recalibrate: bool
    severity: str
    slo_violation: str | None = None

    def summary(self) -> str:
        psi = f"{self.psi_te:.3f}" if self.psi_te is not None else "N/A"
        delta = (f"{self.abstain_rate_delta:+.3f}"
                 if self.abstain_rate_delta is not None else "N/A")
        slo = f", slo_violation={self.slo_violation}" if self.slo_violation else ""
        return (
            f"DriftState(n={self.n_observed}, "
            f"window_full={self.window_full}, "
            f"psi_te={psi}, "
            f"abstain_rate={self.abstain_rate:.3f}, "
            f"review_rate={self.review_rate:.3f}, "
            f"abstain_rate_delta={delta}, "
            f"severity={self.severity}, "
            f"alert={self.alert}, recalibrate={self.recalibrate}{slo})"
        )


def _bin_edges_from_baseline(
    baseline: np.ndarray, n_bins: int = 10,
) -> np.ndarray:
    """Quantile-based bin edges so each baseline bin has ~equal mass.

    Avoids the degenerate-bin problem of equal-width binning when the
    baseline is concentrated on part of the range.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(baseline, qs))
    if len(edges) < 3:
        return np.array(
            [baseline.min(), (baseline.min() + baseline.max()) / 2,
             baseline.max()]
        )
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def population_stability_index(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
    smoothing: float = 1e-4,
) -> float:
    """Compute PSI between two empirical distributions."""
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
    edges = _bin_edges_from_baseline(baseline, n_bins=n_bins)
    b_counts, _ = np.histogram(baseline, bins=edges)
    c_counts, _ = np.histogram(current, bins=edges)
    b_p = b_counts / max(b_counts.sum(), 1)
    c_p = c_counts / max(c_counts.sum(), 1)
    b_p = np.clip(b_p, smoothing, None)
    c_p = np.clip(c_p, smoothing, None)
    return float(np.sum((c_p - b_p) * np.log(c_p / b_p)))


class DriftMonitor:
    """Stateful, ring-buffered drift detector for a RuntimeGate.

    Usage:

        monitor = DriftMonitor(bundle, window_size=500)
        for verdict in stream:
            monitor.record(verdict)
            if monitor.should_recalibrate():
                trigger_offline_recalibration()
    """

    def __init__(
        self,
        bundle: "GovernanceBundle",
        *,
        window_size: int = 500,
        alert_threshold: float = 0.10,
        recalibrate_threshold: float = 0.20,
        abstain_drift_threshold: float = 0.10,
        baseline_abstain_rate: float = 0.0,
        n_bins: int = 10,
        min_te_samples: int = 50,
        recalibrate_requires_full_window: bool = True,
    ) -> None:
        if alert_threshold > recalibrate_threshold:
            raise ValueError(
                "alert_threshold must be <= recalibrate_threshold"
            )
        self.bundle = bundle
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.recalibrate_threshold = recalibrate_threshold
        self.abstain_drift_threshold = abstain_drift_threshold
        self.baseline_abstain_rate = baseline_abstain_rate
        self.n_bins = n_bins
        self.min_te_samples = min_te_samples
        self.recalibrate_requires_full_window = recalibrate_requires_full_window
        self._te_buf: deque[float] = deque(maxlen=window_size)
        self._verdict_buf: deque[str] = deque(maxlen=window_size)
        self._n_observed = 0

    def record(self, verdict: "RuntimeVerdict") -> None:
        self._n_observed += 1
        self._verdict_buf.append(verdict.verdict)
        if verdict.tension_energy is not None:
            self._te_buf.append(float(verdict.tension_energy))

    def record_te(self, te: float, verdict: str = "geometry") -> None:
        """Lightweight ingest path for tests / callers without RuntimeVerdict."""
        self._n_observed += 1
        self._verdict_buf.append(verdict)
        self._te_buf.append(float(te))

    def state(self) -> DriftState:
        n = len(self._verdict_buf)
        window_full = n >= self.window_size

        te_current = np.asarray(self._te_buf, dtype=float)
        if len(te_current) >= self.min_te_samples:
            psi = population_stability_index(
                self.bundle.drift_baseline_te,
                te_current,
                n_bins=self.n_bins,
            )
        else:
            psi = None

        n_abstain = sum(1 for v in self._verdict_buf if v == "abstain")
        n_review = sum(1 for v in self._verdict_buf if v == "review")
        abstain_rate = n_abstain / n if n else 0.0
        review_rate = n_review / n if n else 0.0
        abstain_delta = (
            abstain_rate - self.baseline_abstain_rate if n else None
        )

        alert = False
        recalibrate = False
        severity = "stable"
        if psi is not None:
            if psi >= self.recalibrate_threshold:
                alert = True
                severity = "significant"
                if (not self.recalibrate_requires_full_window
                        or window_full):
                    recalibrate = True
            elif psi >= self.alert_threshold:
                alert = True
                severity = "minor"
        elif n < self.min_te_samples:
            severity = "warming_up"

        # SLO check against the bundle's operating point.
        slo_violation = None
        op = getattr(self.bundle, "operating_point", None)
        if op is not None and n > 0:
            uncertain_rate = abstain_rate + review_rate
            violations = []
            if abstain_rate > op.max_abstain_rate:
                violations.append(
                    f"abstain_rate={abstain_rate:.3f} > "
                    f"max={op.max_abstain_rate:.3f}"
                )
            if review_rate > op.max_review_rate:
                violations.append(
                    f"review_rate={review_rate:.3f} > "
                    f"max={op.max_review_rate:.3f}"
                )
            if uncertain_rate > op.max_uncertain_rate:
                violations.append(
                    f"abstain+review={uncertain_rate:.3f} > "
                    f"max={op.max_uncertain_rate:.3f}"
                )
            if violations:
                slo_violation = "; ".join(violations)
                alert = True

        if (abstain_delta is not None
                and abs(abstain_delta) >= self.abstain_drift_threshold):
            alert = True
            if severity == "stable":
                severity = "abstain_shift"

        return DriftState(
            n_observed=self._n_observed,
            window_full=window_full,
            psi_te=psi,
            abstain_rate=abstain_rate,
            review_rate=review_rate,
            abstain_rate_delta=abstain_delta,
            alert=alert,
            recalibrate=recalibrate,
            severity=severity,
            slo_violation=slo_violation,
        )

    def should_alert(self) -> bool:
        return self.state().alert

    def should_recalibrate(self) -> bool:
        return self.state().recalibrate

    def reset(self) -> None:
        self._te_buf.clear()
        self._verdict_buf.clear()
        self._n_observed = 0
