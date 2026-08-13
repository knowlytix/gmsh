# SPDX-License-Identifier: Apache-2.0
"""Bootstrap-calibrated thresholds for the tiered release gate.

The single-fit ``GateThresholds`` shipped on
:class:`~governed_agent.release_gates.TieredReleaseGate` are
*policy-declared* defaults --- author intent, not data. In high-stakes
deployment the auditor's first question is "what is the threshold's
CI under shift?". This module bootstraps a logistic regression on
``(metric_value -> tier_pass)`` labels per tier per directional
metric, refits on K resamples, and returns a
:class:`~governed_agent.governance_bundle.ThresholdDistribution`
(mean, std, 95% CI) the bundle can ship.

The cohort is a list of
:class:`~governed_agent.release_gate_validation.LabeledRun` --- a
``RunMetrics`` plus a ``tier_truth`` label naming the highest tier
the run should pass. Cohort generation lives in
``release_gate_validation`` so this module can be tested with
small fixtures.

Pattern matches :mod:`governed_agent.policy_calibration` for
consistency with the Ch 12 policy surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from knowlytix.harness.governance.governance_bundle import ThresholdDistribution
from knowlytix.harness.governance.release_gates import (
    GateThresholds,
    TieredReleaseGate,
)

if TYPE_CHECKING:
    from knowlytix.harness.governance.release_gate_validation import LabeledRun


# Tier ordering used to derive per-tier label assignment.
_TIER_ORDER = TieredReleaseGate.TIER_ORDER

# Per-metric direction map: True = min-direction, False = max-direction.
_METRIC_IS_MIN = {
    "accuracy":        True,
    "confidence":      True,
    "fairness_air":    True,
    "robustness_drop": False,
}


def _label_passes_tier(tier_truth: str, target_tier: str) -> int:
    """1 iff the ground-truth tier reaches *target_tier* or stricter."""
    try:
        truth_idx = _TIER_ORDER.index(tier_truth)
    except ValueError:
        return 0
    target_idx = _TIER_ORDER.index(target_tier)
    return 1 if truth_idx >= target_idx else 0


def _derive_threshold(
    clf,
    X: np.ndarray,
    *,
    direction_is_min: bool,
) -> float | None:
    """Find the value where ``P(class=1 | value=x) = 0.5`` on the LR fit.

    For a min-direction metric (accuracy/confidence/fairness_air),
    we search the value range for the lowest x that flips the
    classifier's prediction from 0 (fail) to 1 (pass). For
    max-direction (robustness_drop), the highest x that still passes.
    """
    if X.size == 0:
        return None
    lo, hi = float(np.min(X)), float(np.max(X))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 801).reshape(-1, 1)
    pred = clf.predict(grid).astype(int)
    changes = np.where(np.diff(pred) != 0)[0]
    if len(changes) == 0:
        return None
    if direction_is_min:
        # First transition from 0 -> 1 (or 1 -> 0); use first change
        # index + 1 as the threshold.
        idx = int(changes[0])
        return float(grid[idx + 1, 0])
    # Max-direction: last transition.
    idx = int(changes[-1])
    return float(grid[idx, 0])


def _build_metric_training_set(
    cohort: list["LabeledRun"],
    *,
    metric_name: str,
    target_tier: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for a single (metric, tier) pair.

    Drops runs with non-finite metric values (they fail-closed at the
    gate; including them would smear the bootstrap fit).
    """
    X: list[float] = []
    y: list[int] = []
    for lr in cohort:
        v = getattr(lr.metrics, metric_name)
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(vf):
            continue
        X.append(vf)
        y.append(_label_passes_tier(lr.tier_truth, target_tier))
    return (np.asarray(X, dtype=float).reshape(-1, 1),
            np.asarray(y, dtype=int))


def bootstrap_metric_threshold(
    cohort: list["LabeledRun"],
    *,
    metric_name: str,
    target_tier: str,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> ThresholdDistribution | None:
    """Bootstrap a single per-metric threshold for a target tier.

    Returns ``None`` when the labeled cohort is too small or
    single-class on this (metric, tier) cell --- the bundle then
    surfaces ``signal_unavailable`` for that cell rather than
    fail-open at the policy-declared default.
    """
    direction_is_min = _METRIC_IS_MIN[metric_name]
    from sklearn.linear_model import LogisticRegression

    X, y = _build_metric_training_set(
        cohort, metric_name=metric_name, target_tier=target_tier,
    )
    n = len(y)
    if n < 30:
        return None
    if len(np.unique(y)) < 2:
        return None

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xs, ys = X[idx], y[idx]
        if len(np.unique(ys)) < 2:
            continue
        clf = LogisticRegression(random_state=seed + k, max_iter=2000)
        try:
            clf.fit(Xs, ys)
        except Exception:
            continue
        tau = _derive_threshold(
            clf, Xs, direction_is_min=direction_is_min,
        )
        if tau is None or not math.isfinite(tau):
            continue
        draws.append(tau)

    if len(draws) < max(10, n_bootstrap // 4):
        return None

    arr = np.asarray(draws)
    return ThresholdDistribution(
        mean=float(arr.mean()),
        std=float(arr.std()),
        ci_low=float(np.quantile(arr, 0.025)),
        ci_hi=float(np.quantile(arr, 0.975)),
        n_bootstrap=len(draws),
    )


@dataclass(frozen=True)
class CalibratedThresholdSet:
    """Bootstrap distributions for one tier's directional metrics.

    Cells set to ``None`` mean the cohort had insufficient signal; the
    bundle's :meth:`calibrated_tiers` falls back to the
    policy-declared default for that cell and marks it as
    ``not calibrated`` in the audit trail.
    """

    tier: str
    min_accuracy: ThresholdDistribution | None
    min_confidence: ThresholdDistribution | None
    max_robustness_drop: ThresholdDistribution | None
    min_fairness_air: ThresholdDistribution | None

    def to_gate_thresholds(
        self,
        fallback: GateThresholds,
        *,
        source: str,
    ) -> GateThresholds:
        """Materialize per-cell calibrated values, falling back to *fallback*.

        Sets ``is_calibrated=True`` on the returned thresholds when at
        least one cell was successfully calibrated.
        """
        any_calibrated = any(
            x is not None for x in (
                self.min_accuracy, self.min_confidence,
                self.max_robustness_drop, self.min_fairness_air,
            )
        )
        return GateThresholds(
            min_accuracy=(self.min_accuracy.mean
                          if self.min_accuracy is not None
                          else fallback.min_accuracy),
            min_confidence=(self.min_confidence.mean
                            if self.min_confidence is not None
                            else fallback.min_confidence),
            max_robustness_drop=(self.max_robustness_drop.mean
                                 if self.max_robustness_drop is not None
                                 else fallback.max_robustness_drop),
            min_fairness_air=(self.min_fairness_air.mean
                              if self.min_fairness_air is not None
                              else fallback.min_fairness_air),
            is_calibrated=any_calibrated,
            source=source,
        )


def calibrate_release_gate_thresholds(
    cohort: list["LabeledRun"],
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, CalibratedThresholdSet]:
    """Bootstrap-calibrate every (tier, metric) cell from a labeled cohort.

    Returns ``{tier_name: CalibratedThresholdSet}``. Cells with
    insufficient cohort signal (single-class, < 30 samples, or
    unstable fits) are returned as ``None`` so the bundle can decide
    fallback policy.

    The four metrics are calibrated independently per tier --- the
    bundle assembler enforces monotonic ordering across tiers
    afterwards (see :func:`enforce_monotonic_ordering`).
    """
    out: dict[str, CalibratedThresholdSet] = {}
    for tier in _TIER_ORDER:
        out[tier] = CalibratedThresholdSet(
            tier=tier,
            min_accuracy=bootstrap_metric_threshold(
                cohort, metric_name="accuracy",
                target_tier=tier,
                n_bootstrap=n_bootstrap, seed=seed,
            ),
            min_confidence=bootstrap_metric_threshold(
                cohort, metric_name="confidence",
                target_tier=tier,
                n_bootstrap=n_bootstrap, seed=seed,
            ),
            max_robustness_drop=bootstrap_metric_threshold(
                cohort, metric_name="robustness_drop",
                target_tier=tier,
                n_bootstrap=n_bootstrap, seed=seed,
            ),
            min_fairness_air=bootstrap_metric_threshold(
                cohort, metric_name="fairness_air",
                target_tier=tier,
                n_bootstrap=n_bootstrap, seed=seed,
            ),
        )
    return out


def enforce_monotonic_ordering(
    tiers: dict[str, GateThresholds],
) -> dict[str, GateThresholds]:
    """Repair calibrated thresholds so monotonicity holds.

    The per-cell bootstrap fits are independent --- nothing forces
    the action_taking ``min_accuracy`` to exceed the recommendation
    one. When the cohort produces a non-monotonic ordering, we
    project to the nearest monotonic configuration: each stricter
    tier's min-direction threshold is raised to the looser tier's
    if it would otherwise be lower; each stricter tier's
    max-direction threshold is lowered to the looser tier's if it
    would otherwise be higher.

    The audit log records when projection occurred so a reviewer can
    see that the operating threshold was not exactly the bootstrap
    mean.
    """
    repaired: dict[str, GateThresholds] = {}
    prev: GateThresholds | None = None
    for tier in _TIER_ORDER:
        if tier not in tiers:
            continue
        t = tiers[tier]
        if prev is None:
            repaired[tier] = t
            prev = t
            continue
        new_t = GateThresholds(
            min_accuracy=max(t.min_accuracy, prev.min_accuracy),
            min_confidence=max(t.min_confidence, prev.min_confidence),
            max_robustness_drop=min(
                t.max_robustness_drop, prev.max_robustness_drop),
            min_fairness_air=max(
                t.min_fairness_air, prev.min_fairness_air),
            is_calibrated=t.is_calibrated,
            source=t.source,
        )
        repaired[tier] = new_t
        prev = new_t
    return repaired


def policy_declared_baseline() -> dict[str, GateThresholds]:
    """The shipped class defaults, returned for side-by-side reporting.

    The bundle ships *both* the calibrated thresholds (operating) and
    the policy-declared baseline (provenance) so the auditor can see
    how far the calibration moved each cell.
    """
    return {
        tier: GateThresholds(
            min_accuracy=t.min_accuracy,
            min_confidence=t.min_confidence,
            max_robustness_drop=t.max_robustness_drop,
            min_fairness_air=t.min_fairness_air,
            is_calibrated=False,
            source="policy_declared",
        )
        for tier, t in TieredReleaseGate.TIERS.items()
    }
