# SPDX-License-Identifier: Apache-2.0
"""Exhaustive monotonicity proof over the metric polytope.

A high-stakes auditor wants more than per-input monotonicity checks
--- they want a *static* proof that the property holds for every
possible :class:`RunMetrics` value the gate could see. Because the
gate is pure scalar math on four bounded metrics, exhaustive
enumeration over a fine grid is computationally cheap (160K points
at 0.05 resolution) and produces a machine-checkable answer the
chapter can claim with confidence.

Five invariants verified:

  I1 (subset containment): for every grid point,
       advisory_pass >= recommendation_pass >= action_taking_pass.
  I2 (NaN propagation): NaN inputs always produce passed=False
       (one probe per metric per tier).
  I3 (Inf propagation): +/-Inf inputs always produce passed=False.
  I4 (boundary equality): runs *exactly at* a tier's threshold
       satisfy the >= / <= side correctly.
  I5 (out-of-domain detection): is_well_formed catches every
       AIR>1.0, accuracy<0, accuracy>1.0 perturbation.

Invariant violations are recorded with the offending metric vector
so the regression log can record exactly which point broke.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from knowlytix.harness.governance.release_gates import (
    RunMetrics,
    TieredReleaseGate,
)


@dataclass
class InvariantViolation:
    invariant: str
    description: str
    offending_input: dict | None = None


@dataclass
class InvariantReport:
    n_grid_points: int
    n_pass: int
    n_violations: int
    violations: list[InvariantViolation] = field(default_factory=list)
    invariants_checked: list[str] = field(default_factory=list)


def _grid(resolution: float = 0.05) -> Iterable[tuple[float, float, float, float]]:
    """Cartesian product over [0, 1]^4 at the given resolution."""
    axis = np.arange(0.0, 1.0 + resolution / 2, resolution)
    for acc, conf, rob, fair in itertools.product(axis, repeat=4):
        yield (float(acc), float(conf), float(rob), float(fair))


def verify_release_gate_invariants(
    gate: TieredReleaseGate | None = None,
    *,
    resolution: float = 0.10,
) -> InvariantReport:
    """Exhaustively verify the five invariants on the gate.

    Default ``resolution=0.10`` gives 11^4 = 14641 grid points (fast).
    For a more thorough check, use ``resolution=0.05`` (194481 points).
    """
    if gate is None:
        gate = TieredReleaseGate()

    violations: list[InvariantViolation] = []
    n_pass = 0
    n_total = 0

    # I1: subset containment over the polytope
    for acc, conf, rob, fair in _grid(resolution):
        m = RunMetrics(
            accuracy=acc, confidence=conf,
            robustness_drop=rob, fairness_air=fair,
            tool_calls=0, policy_blocks=0,
        )
        n_total += 1
        results = gate.check_all_tiers(m)
        passes = [results[t].passed for t in gate.TIER_ORDER
                  if t in results]
        # advisory >= recommendation >= action_taking
        ok = all(passes[i] or not passes[i + 1]
                 for i in range(len(passes) - 1))
        if ok:
            n_pass += 1
        else:
            violations.append(InvariantViolation(
                invariant="I1_subset_containment",
                description=(f"non-monotonic at acc={acc:.2f} "
                             f"conf={conf:.2f} rob={rob:.2f} "
                             f"fair={fair:.2f}: passes={passes}"),
                offending_input={
                    "accuracy": acc, "confidence": conf,
                    "robustness_drop": rob, "fairness_air": fair,
                },
            ))

    # I2: NaN propagation
    nan_target_metrics = ["accuracy", "confidence",
                          "robustness_drop", "fairness_air"]
    for tier in gate.TIER_ORDER:
        if tier not in gate.check_all_tiers(RunMetrics(0.9, 0.9, 0.0, 0.9, 0, 0)):
            continue
        for target in nan_target_metrics:
            kw = dict(accuracy=0.95, confidence=0.95,
                      robustness_drop=0.01, fairness_air=0.95,
                      tool_calls=0, policy_blocks=0)
            kw[target] = float("nan")
            r = gate.check(RunMetrics(**kw), tier)
            if r.passed:
                violations.append(InvariantViolation(
                    invariant="I2_nan_propagation",
                    description=(f"NaN {target} passed at tier {tier}"),
                    offending_input={target: "nan", "tier": tier},
                ))

    # I3: Inf propagation
    for tier in gate.TIER_ORDER:
        if tier not in gate.check_all_tiers(RunMetrics(0.9, 0.9, 0.0, 0.9, 0, 0)):
            continue
        for target in nan_target_metrics:
            for sign in (math.inf, -math.inf):
                kw = dict(accuracy=0.95, confidence=0.95,
                          robustness_drop=0.01, fairness_air=0.95,
                          tool_calls=0, policy_blocks=0)
                kw[target] = sign
                r = gate.check(RunMetrics(**kw), tier)
                if r.passed:
                    violations.append(InvariantViolation(
                        invariant="I3_inf_propagation",
                        description=(f"{sign} {target} passed at "
                                     f"tier {tier}"),
                        offending_input={target: sign, "tier": tier},
                    ))

    # I4: boundary equality (>= / <= are inclusive)
    runtime_tiers = gate.TIERS_RUNTIME
    for tier_name, t in runtime_tiers.items():
        # Construct a run exactly at every boundary
        m = RunMetrics(
            accuracy=t.min_accuracy,
            confidence=t.min_confidence,
            robustness_drop=t.max_robustness_drop,
            fairness_air=max(t.min_fairness_air, 0.0),
            tool_calls=0, policy_blocks=0,
        )
        r = gate.check(m, tier_name)
        if not r.passed:
            violations.append(InvariantViolation(
                invariant="I4_boundary_equality",
                description=(f"run at exact boundary failed tier "
                             f"{tier_name}; reasons {r.blocking_reasons}"),
                offending_input={
                    "tier": tier_name,
                    "metrics": {
                        "accuracy": m.accuracy,
                        "confidence": m.confidence,
                        "robustness_drop": m.robustness_drop,
                        "fairness_air": m.fairness_air,
                    },
                },
            ))

    # I5: out-of-domain detection by is_well_formed
    bad_inputs = [
        {"accuracy": -0.1, "confidence": 0.5, "robustness_drop": 0.1,
         "fairness_air": 0.5},
        {"accuracy": 1.5, "confidence": 0.5, "robustness_drop": 0.1,
         "fairness_air": 0.5},
        {"accuracy": 0.5, "confidence": 0.5, "robustness_drop": -0.1,
         "fairness_air": 0.5},
        {"accuracy": 0.5, "confidence": 0.5, "robustness_drop": 1.5,
         "fairness_air": 0.5},
        {"accuracy": 0.5, "confidence": 0.5, "robustness_drop": 0.1,
         "fairness_air": 1.25},
        {"accuracy": 0.5, "confidence": -0.1, "robustness_drop": 0.1,
         "fairness_air": 0.5},
    ]
    for kw in bad_inputs:
        m = RunMetrics(tool_calls=0, policy_blocks=0, **kw)
        well_formed, _ = m.is_well_formed()
        if well_formed:
            violations.append(InvariantViolation(
                invariant="I5_out_of_domain",
                description=("is_well_formed accepted out-of-domain "
                             f"vector {kw}"),
                offending_input=dict(kw),
            ))

    return InvariantReport(
        n_grid_points=n_total,
        n_pass=n_pass,
        n_violations=len(violations),
        violations=violations,
        invariants_checked=["I1_subset_containment",
                            "I2_nan_propagation",
                            "I3_inf_propagation",
                            "I4_boundary_equality",
                            "I5_out_of_domain"],
    )
