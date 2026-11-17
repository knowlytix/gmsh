# SPDX-License-Identifier: Apache-2.0
"""Naive baselines for the tiered release gate.

The audit-defensible bar requires that the calibrated/tiered gate
demonstrably outperforms a fixed-threshold or random baseline. Two
baselines are provided:

  * :class:`FixedThresholdBaseline` -- single non-tiered cutoff on
    accuracy. The naive "what your competitor does" approach.
  * :class:`RandomBaseline` -- coin-flip per run. The
    "stupid-but-honest" floor.

:func:`compare_to_gate` reports per-tier uplift (correct-pair count
of the gate vs the baseline on a labeled cohort).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knowlytix.harness.governance.release_gate_validation import (
    LabeledRun,
    _expected_pass_at,
)
from knowlytix.harness.governance.release_gates import (
    RunMetrics,
    TieredReleaseGate,
)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class _Baseline:
    name: str = "baseline"

    def decide(self, m: RunMetrics, tier: str) -> bool:
        raise NotImplementedError


class FixedThresholdBaseline(_Baseline):
    """Single-cutoff baseline (accuracy only).

    The default ``min_accuracy=0.80`` matches the recommendation
    tier's bar; the baseline ignores tier and always uses this
    threshold on accuracy.
    """

    name = "fixed_threshold"

    def __init__(
        self,
        *,
        min_accuracy: float = 0.80,
    ) -> None:
        self.min_accuracy = float(min_accuracy)

    def decide(self, m: RunMetrics, tier: str) -> bool:
        try:
            return float(m.accuracy) >= self.min_accuracy
        except (TypeError, ValueError):
            return False


class RandomBaseline(_Baseline):
    """Coin-flip baseline. Deterministic per (probe, tier) given seed."""

    name = "random"

    def __init__(self, *, seed: int = 42, p_pass: float = 0.5) -> None:
        self.seed = int(seed)
        self.p_pass = float(p_pass)
        self._rng = np.random.default_rng(self.seed)

    def decide(self, m: RunMetrics, tier: str) -> bool:
        # Hash the probe key so the same (run_id, tier) flips
        # consistently across runs.
        key = f"{getattr(m, 'run_id', '')}|{tier}|{self.seed}"
        local = np.random.default_rng(abs(hash(key)) % (2**32))
        return bool(local.random() < self.p_pass)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class BaselineContrastReport:
    """Pair-uplift contrast between the gate and a baseline.

    ``n_correct_gate`` counts cohort items where the gate's verdict
    matches the truth at *every* tier; same for the baseline.
    ``uplift`` = gate - baseline. Positive uplift => the gate wins.
    """

    baseline_name: str
    n_probes: int
    per_tier: dict[str, dict[str, int]] = field(default_factory=dict)
    n_correct_gate: int = 0
    n_correct_baseline: int = 0
    uplift: int = 0


def compare_to_gate(
    gate: TieredReleaseGate,
    cohort: list[LabeledRun],
    baseline: _Baseline,
) -> BaselineContrastReport:
    """Pair-wise contrast: how many probes the gate gets right vs the baseline.

    For each (probe, tier) cell we record whether the verdict
    matches the expected truth. The gate "wins" when its number of
    correct cells exceeds the baseline's.
    """
    per_tier: dict[str, dict[str, int]] = {
        t: {"gate_correct": 0, "baseline_correct": 0, "n": 0}
        for t in gate.TIER_ORDER
    }
    n_correct_gate = 0
    n_correct_baseline = 0

    for lr in cohort:
        gate_correct_for_probe = 0
        baseline_correct_for_probe = 0
        for tier in gate.TIER_ORDER:
            if tier not in gate.TIERS_RUNTIME:
                continue
            expected = _expected_pass_at(lr.tier_truth, tier)
            gate_pass = gate.check(lr.metrics, tier).passed
            baseline_pass = baseline.decide(lr.metrics, tier)
            cell = per_tier[tier]
            cell["n"] += 1
            if gate_pass == expected:
                cell["gate_correct"] += 1
                gate_correct_for_probe += 1
            if baseline_pass == expected:
                cell["baseline_correct"] += 1
                baseline_correct_for_probe += 1
        # Per-probe correctness: matches truth on ALL tiers
        if gate_correct_for_probe == sum(
                1 for t in gate.TIER_ORDER if t in gate.TIERS_RUNTIME):
            n_correct_gate += 1
        if baseline_correct_for_probe == sum(
                1 for t in gate.TIER_ORDER if t in gate.TIERS_RUNTIME):
            n_correct_baseline += 1

    return BaselineContrastReport(
        baseline_name=baseline.name,
        n_probes=len(cohort),
        per_tier=per_tier,
        n_correct_gate=n_correct_gate,
        n_correct_baseline=n_correct_baseline,
        uplift=n_correct_gate - n_correct_baseline,
    )
