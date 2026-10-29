# SPDX-License-Identifier: Apache-2.0
"""Independent boolean truth-table re-implementation of the release gate.

Used as a second-opinion attestor: if the production
:class:`TieredReleaseGate` and this alt-validator disagree on any
probe in the OOD cohort, the bundle build refuses to ship. Mirrors
:mod:`governed_agent.policy_validator_alt` for the release-gate
surface.

The alt-validator does NOT call into ``release_gates.TieredReleaseGate``;
it re-derives the verdict from raw GateThresholds via direct numeric
comparison. This is the *separate-implementation* contract: a bug
in the production gate would have to also exist in this module to
escape detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from knowlytix.harness.governance.release_gate_validation import (
    LabeledRun,
)
from knowlytix.harness.governance.release_gates import (
    GateThresholds,
    RunMetrics,
    TieredReleaseGate,
)

# ---------------------------------------------------------------------------
# Alt-validator core: pure boolean truth table
# ---------------------------------------------------------------------------

def _alt_finite(x) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _alt_min(value, threshold) -> bool:
    if not _alt_finite(value):
        return False
    return float(value) >= float(threshold)


def _alt_max(value, threshold) -> bool:
    if not _alt_finite(value):
        return False
    return float(value) <= float(threshold)


def alt_check(metrics: RunMetrics, t: GateThresholds) -> bool:
    """Independent boolean truth-table check.

    Replicates ``TieredReleaseGate.check`` semantics from first
    principles --- four independent comparisons; conjunction. NaN/Inf
    fail closed in both directions (Risk 4 fix).
    """
    return (_alt_min(metrics.accuracy, t.min_accuracy)
            and _alt_min(metrics.confidence, t.min_confidence)
            and _alt_max(metrics.robustness_drop, t.max_robustness_drop)
            and _alt_min(metrics.fairness_air, t.min_fairness_air))


# ---------------------------------------------------------------------------
# Agreement report
# ---------------------------------------------------------------------------

@dataclass
class AltAgreementReport:
    n: int
    n_agree: int
    agreement_rate: float
    disagreements: list[dict] = field(default_factory=list)


def assert_gate_alt_agreement(
    gate: TieredReleaseGate,
    cohort: list[LabeledRun],
    *,
    raise_on_disagreement: bool = True,
) -> AltAgreementReport:
    """Compare gate verdicts to the alt-validator on every (probe, tier).

    Reports per-probe-per-tier disagreements and returns the agreement
    rate. ``raise_on_disagreement=True`` raises ``RuntimeError`` if
    any disagreement is detected --- the production-bundle build
    refuses to ship without 100% agreement.
    """
    n_pairs = 0
    n_agree = 0
    disagreements: list[dict] = []

    for lr in cohort:
        for tier in gate.TIER_ORDER:
            if tier not in gate.TIERS_RUNTIME:
                continue
            n_pairs += 1
            t = gate.TIERS_RUNTIME[tier]
            gate_pass = gate.check(lr.metrics, tier).passed
            alt_pass = alt_check(lr.metrics, t)
            if gate_pass == alt_pass:
                n_agree += 1
            else:
                disagreements.append({
                    "run_id": lr.run_id,
                    "tier": tier,
                    "gate_pass": bool(gate_pass),
                    "alt_pass": bool(alt_pass),
                    "metrics": {
                        "accuracy": lr.metrics.accuracy,
                        "confidence": lr.metrics.confidence,
                        "robustness_drop": lr.metrics.robustness_drop,
                        "fairness_air": lr.metrics.fairness_air,
                    },
                })

    rate = (n_agree / n_pairs) if n_pairs > 0 else 0.0
    report = AltAgreementReport(
        n=n_pairs, n_agree=n_agree,
        agreement_rate=rate,
        disagreements=disagreements,
    )
    if raise_on_disagreement and disagreements:
        first = disagreements[0]
        raise RuntimeError(
            f"alt-validator disagreement on {len(disagreements)}/"
            f"{n_pairs} pairs; first: run_id={first['run_id']} "
            f"tier={first['tier']} gate={first['gate_pass']} "
            f"alt={first['alt_pass']}"
        )
    return report
