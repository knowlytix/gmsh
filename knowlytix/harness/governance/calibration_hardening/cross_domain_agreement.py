# SPDX-License-Identifier: Apache-2.0
"""Cross-domain calibration comparison.

Ch.19's core claim is that the governance architecture is
domain-portable: switching from Iris (classification) to
California Housing (regression) requires only threshold
recalibration, not architecture changes. This module formalises
the comparison: given two `CalibrationSnapshot`s (one per domain),
report which thresholds drifted and by how much.

A real deployment uses this to decide whether two domains can
share a calibration (drift small) or require independent
recalibration (drift large).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.harness.governance.calibration_hardening.calibration_snapshot import (
    CalibrationSnapshot,
)


@dataclass
class ThresholdDrift:
    field_name: str
    domain_a_value: float
    domain_b_value: float
    abs_drift: float
    relative_drift: float


@dataclass
class CrossDomainReport:
    domain_a: str
    domain_b: str
    drifts: list[ThresholdDrift] = field(default_factory=list)
    accuracy_a: float = 0.0
    accuracy_b: float = 0.0

    @property
    def max_abs_drift(self) -> float:
        if not self.drifts:
            return 0.0
        return max(d.abs_drift for d in self.drifts)

    @property
    def max_relative_drift(self) -> float:
        if not self.drifts:
            return 0.0
        return max(d.relative_drift for d in self.drifts)

    def summary(self) -> str:
        rows = [
            "CrossDomainReport(",
            f"  domain_a={self.domain_a} (acc={self.accuracy_a:.4f})",
            f"  domain_b={self.domain_b} (acc={self.accuracy_b:.4f})",
            f"  max_abs_drift={self.max_abs_drift:.4f}",
            f"  max_relative_drift={self.max_relative_drift:.4f}",
        ]
        for d in self.drifts:
            rows.append(
                f"    {d.field_name}: "
                f"{d.domain_a_value:.4f} -> {d.domain_b_value:.4f} "
                f"(abs={d.abs_drift:.4f}, rel={d.relative_drift:.4%})"
            )
        rows.append(")")
        return "\n".join(rows)


def compare_calibrations_across_domains(
    snap_a: CalibrationSnapshot,
    snap_b: CalibrationSnapshot,
) -> CrossDomainReport:
    """Compare two calibration snapshots from different domains."""
    drifts: list[ThresholdDrift] = []
    a, b = snap_a.optimum, snap_b.optimum
    for fname in ("theta_sim", "theta_margin", "theta_esc",
                  "d_max", "budget"):
        av = float(getattr(a, fname))
        bv = float(getattr(b, fname))
        abs_drift = abs(av - bv)
        denom = max(abs(av), abs(bv), 1e-9)
        rel = abs_drift / denom
        drifts.append(ThresholdDrift(
            field_name=fname,
            domain_a_value=av, domain_b_value=bv,
            abs_drift=abs_drift, relative_drift=rel,
        ))

    return CrossDomainReport(
        domain_a=snap_a.domain,
        domain_b=snap_b.domain,
        drifts=drifts,
        accuracy_a=a.accuracy,
        accuracy_b=b.accuracy,
    )
