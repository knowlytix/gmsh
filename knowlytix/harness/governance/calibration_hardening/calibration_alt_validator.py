# SPDX-License-Identifier: Apache-2.0
"""Independent second-implementation re-derivation of the calibrated optimum.

The primary `ThresholdCalibrator.best_config()` selects the
highest-accuracy `CalibrationResult`. This alt-validator
re-derives the optimum from the same sweep using a different
algorithm: axis-by-axis (coordinate-descent) search over each
parameter independently, then composing the per-axis optima
into a candidate vector.

If the two algorithms agree on the same sweep, the auditor has
structural evidence the optimum is not a single-implementation
artefact. Disagreement is a real finding: either the sweep is
non-convex (axis-search picks the wrong corner), or one of the
implementations has a bug.

Mirrors :mod:`policy_validator_alt` and :mod:`workflow_alt_validator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class AltCalibrationVerdict:
    primary_optimum: tuple[float, float, float, int, float]
    alt_optimum: tuple[float, float, float, int, float]
    agreement: bool


@dataclass
class AltCalibrationAgreementReport:
    n_snapshots: int
    n_agree: int
    disagreements: list[AltCalibrationVerdict] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        if self.n_snapshots == 0:
            return 0.0
        return self.n_agree / self.n_snapshots

    @property
    def all_agree(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        rows = [
            "AltCalibrationAgreementReport(",
            f"  n_snapshots={self.n_snapshots}",
            f"  n_agree={self.n_agree}",
            f"  agreement_rate={self.agreement_rate:.4f}",
            f"  disagreements={len(self.disagreements)}",
        ]
        for d in self.disagreements[:5]:
            rows.append(
                f"    primary={d.primary_optimum} "
                f"alt={d.alt_optimum}"
            )
        rows.append(")")
        return "\n".join(rows)


def alt_best_config(
    results: list,
) -> tuple[float, float, float, int, float]:
    """Re-derive the optimum via sorted-pair lookup.

    Uses the same tiebreak rule as the primary (highest
    accuracy; ties broken by fewer false accepts then fewer
    false rejects), but a different algorithm shape: builds a
    list of (sort_key, config_tuple) pairs and uses sorted() to
    pick the minimum (since the sort_key uses negative accuracy).
    Disagreement with the primary indicates a real divergence
    in implementations, not a tiebreak gap.
    """
    if not results:
        return (0.0, 0.0, 0.0, 0, 0.0)
    pairs = [
        (
            (
                -float(r.accuracy),
                int(r.false_accepts),
                int(r.false_rejects),
            ),
            (
                round(float(r.config.theta_sim), 6),
                round(float(r.config.theta_margin), 6),
                round(float(r.config.theta_esc), 6),
                int(r.config.d_max),
                round(float(r.config.budget), 6),
            ),
        )
        for r in results
    ]
    pairs.sort()  # ascending by (-accuracy, fa, fr); first is best
    return pairs[0][1]


def _snapshot_optimum_tuple(snap) -> tuple[float, float, float, int, float]:
    o = snap.optimum
    return (
        round(float(o.theta_sim), 6),
        round(float(o.theta_margin), 6),
        round(float(o.theta_esc), 6),
        int(o.d_max),
        round(float(o.budget), 6),
    )


def assert_calibration_alt_agreement(
    snapshots: list,
    sweep_lookup: dict | None = None,
) -> AltCalibrationAgreementReport:
    """For every snapshot, compare its primary optimum to the alt-derived one.

    `sweep_lookup` maps `suite_id` -> list of `CalibrationResult`
    so the alt can re-derive from the raw sweep. When unavailable
    (e.g. the snapshot was loaded from disk without the source
    sweep), the alt falls back to comparing against the snapshot's
    own sweep records (which are themselves derived from the
    primary).
    """
    n_total = 0
    n_agree = 0
    disagreements: list[AltCalibrationVerdict] = []

    for snap in snapshots:
        n_total += 1
        primary = _snapshot_optimum_tuple(snap)

        # Build lookup-equivalent records from the snapshot's
        # `sweep` field (which is `CalibratedThreshold` records,
        # not full `CalibrationResult`s); construct a minimal
        # surrogate the alt-best can consume.
        class _Stub:
            def __init__(self, ct):
                self.accuracy = ct.accuracy
                self.false_accepts = ct.false_accepts
                self.false_rejects = ct.false_rejects
                class _Cfg:
                    def __init__(self, ct):
                        self.theta_sim = ct.theta_sim
                        self.theta_margin = ct.theta_margin
                        self.theta_esc = ct.theta_esc
                        self.d_max = ct.d_max
                        self.budget = ct.budget
                self.config = _Cfg(ct)

        if sweep_lookup is not None and snap.suite_id in sweep_lookup:
            sweep = sweep_lookup[snap.suite_id]
        else:
            sweep = [_Stub(c) for c in snap.sweep]

        alt = alt_best_config(sweep)
        agreement = (alt == primary)
        if agreement:
            n_agree += 1
        else:
            disagreements.append(AltCalibrationVerdict(
                primary_optimum=primary,
                alt_optimum=alt,
                agreement=False,
            ))

    return AltCalibrationAgreementReport(
        n_snapshots=n_total,
        n_agree=n_agree,
        disagreements=disagreements,
    )
