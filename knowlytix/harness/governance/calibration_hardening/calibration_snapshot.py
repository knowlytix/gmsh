# SPDX-License-Identifier: Apache-2.0
"""Deterministic snapshot of a threshold-calibration result.

The Ch.19 `ThresholdCalibrator` produces a list of
`CalibrationResult` records --- one per threshold configuration
--- plus a single best-configuration extracted via
`best_config()`. The snapshot fingerprints the fields that must
reproduce byte-identically across runs:

  * the optimal threshold vector (rounded to 6 decimals)
  * the best accuracy (rounded)
  * the per-config (config, accuracy, false_accepts, false_rejects)
    tuples (rounded), sorted canonically
  * the parameter grid keys (sorted)

Floating numerics (CUDA-trained verifier accuracies) drift at
the 4th-5th decimal across runs; rounding to 6 decimals captures
the deterministic governance content while absorbing CUDA
non-determinism.

Mirrors :mod:`workflow_hardening.workflow_snapshot` and
:mod:`adversarial_hardening.attack_snapshot` for the
calibration surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.calibration import (
        CalibrationResult,
    )


@dataclass
class CalibratedThreshold:
    """Deterministic record of one (config, accuracy) pair."""

    theta_sim: float
    theta_margin: float
    theta_esc: float
    d_max: int
    budget: float
    accuracy: float
    false_accepts: int
    false_rejects: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_calibration_result(
        cls, r: "CalibrationResult", *, ndigits: int = 6,
    ) -> "CalibratedThreshold":
        return cls(
            theta_sim=round(float(r.config.theta_sim), ndigits),
            theta_margin=round(float(r.config.theta_margin), ndigits),
            theta_esc=round(float(r.config.theta_esc), ndigits),
            d_max=int(r.config.d_max),
            budget=round(float(r.config.budget), ndigits),
            accuracy=round(float(r.accuracy), ndigits),
            false_accepts=int(r.false_accepts),
            false_rejects=int(r.false_rejects),
        )


@dataclass
class CalibrationSnapshot:
    """Audit-grade summary of a calibration run.

    The fingerprint covers the calibrated optimum + the full sweep
    grid; two consecutive runs of the same calibrator on the same
    store + cohort must produce the same fingerprint.
    """

    suite_id: str
    domain: str                                  # "iris" / "california_housing" / etc.
    grid_keys: list[str]                         # sorted parameter names
    n_configs: int
    optimum: CalibratedThreshold
    sweep: list[CalibratedThreshold]             # sorted canonically

    def canonical_payload(self) -> bytes:
        d = {
            "suite_id": self.suite_id,
            "domain": self.domain,
            "grid_keys": sorted(self.grid_keys),
            "n_configs": self.n_configs,
            "optimum": self.optimum.to_dict(),
            "sweep": [s.to_dict() for s in self.sweep],
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()

    def summary(self) -> str:
        return (
            f"CalibrationSnapshot(id={self.suite_id}, "
            f"domain={self.domain}, "
            f"n_configs={self.n_configs}, "
            f"opt_acc={self.optimum.accuracy}, "
            f"opt_theta_sim={self.optimum.theta_sim}, "
            f"opt_theta_margin={self.optimum.theta_margin}, "
            f"fingerprint={self.fingerprint()[:16]}...)"
        )


def build_calibration_snapshot(
    *,
    suite_id: str,
    domain: str,
    grid_keys: list[str],
    sweep_results: list,
    optimum: "CalibrationResult",
    ndigits: int = 6,
) -> CalibrationSnapshot:
    """Pack a calibration sweep into a deterministic snapshot.

    Sort the sweep canonically by (theta_sim, theta_margin,
    theta_esc, d_max, budget) so two runs that visit configs in
    different order still produce the same fingerprint.
    """
    sweep = sorted(
        (CalibratedThreshold.from_calibration_result(r, ndigits=ndigits)
         for r in sweep_results),
        key=lambda c: (
            c.theta_sim, c.theta_margin, c.theta_esc,
            c.d_max, c.budget,
        ),
    )
    opt = CalibratedThreshold.from_calibration_result(
        optimum, ndigits=ndigits,
    )
    return CalibrationSnapshot(
        suite_id=suite_id,
        domain=domain,
        grid_keys=sorted(grid_keys),
        n_configs=len(sweep),
        optimum=opt,
        sweep=sweep,
    )
