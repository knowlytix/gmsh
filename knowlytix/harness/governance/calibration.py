# SPDX-License-Identifier: Apache-2.0
"""Threshold calibration utilities for governed agent verification.

Provides systematic threshold tuning via parameter sweeps and grid
search.  The calibrator runs the ClaimVerifier at multiple threshold
configurations, records accuracy/false-accept/false-reject counts,
and computes sensitivity scores for each parameter.

Built in Chapter 19 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.governance.session import SessionState

from knowlytix.harness.governance.verifier import Claim, ClaimVerifier


# ---------------------------------------------------------------------------
# Configuration and result data models
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """Threshold vector for governed agent verification.

    Attributes
    ----------
    theta_sim : float
        Plausibility threshold — maximum geodesic distance for a
        plausible triple via ``score_triple``.
    theta_margin : float
        Verification confidence threshold — minimum margin for a
        confident pass/fail decision.
    theta_esc : float
        Escalation threshold — holonomy defect above which a
        reasoning chain is flagged for human review.
    d_max : int
        Maximum tool calls allowed per session (budget cap).
    budget : float
        Normalized budget in [0, 1] controlling how aggressively
        the agent pursues verification.
    """

    theta_sim: float = 1.5
    theta_margin: float = 0.3
    theta_esc: float = 0.5
    d_max: int = 30
    budget: float = 1.0


@dataclass
class CalibrationResult:
    """Result from running the verifier at one threshold configuration.

    Attributes
    ----------
    config : ThresholdConfig
        The threshold configuration used.
    accuracy : float
        Fraction of claims whose verdict matches ground truth.
    false_accepts : int
        Number of incorrect claims that the verifier accepted.
    false_rejects : int
        Number of correct claims that the verifier rejected.
    tool_calls : int
        Number of GMS primitive calls made during verification.
    """

    config: ThresholdConfig
    accuracy: float = 0.0
    false_accepts: int = 0
    false_rejects: int = 0
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# Labeled claim wrapper
# ---------------------------------------------------------------------------

@dataclass
class LabeledClaim:
    """A claim paired with its ground-truth label.

    Attributes
    ----------
    claim : Claim
        The typed claim to verify.
    expected_pass : bool
        Whether the claim *should* pass verification.
    """

    claim: Claim
    expected_pass: bool


# ---------------------------------------------------------------------------
# ThresholdCalibrator
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """Systematic threshold tuning via parameter sweep.

    Given a GMS store and a set of labeled claims, the calibrator
    runs the ``ClaimVerifier`` at each threshold configuration and
    records accuracy, false accepts and false rejects.

    Parameters
    ----------
    store : GMSExpertStore
        The GMS store providing geometric verification primitives.
    session : SessionState | None
        Optional session state (unused by verifier but kept for API
        compatibility with future extensions).
    device : str
        PyTorch device string (default ``"cpu"``).
    """

    def __init__(self, store: "GMSExpertStore",
                 session: "SessionState | None" = None,
                 device: str = "cpu"):
        self.store = store
        self.session = session
        self.device = device

    # ---- Internal helpers -------------------------------------------------

    def _build_verifier(self, config: ThresholdConfig) -> ClaimVerifier:
        """Create a ClaimVerifier with the given threshold configuration."""
        return ClaimVerifier(
            store=self.store,
            device=self.device,
            plausibility_threshold=config.theta_sim,
            numeric_tolerance=config.theta_margin,
            holonomy_threshold=config.theta_esc,
        )

    def _evaluate(self, config: ThresholdConfig,
                  test_claims: list[LabeledClaim]) -> CalibrationResult:
        """Run the verifier on labeled claims and compute metrics."""
        verifier = self._build_verifier(config)

        false_accepts = 0
        false_rejects = 0
        correct = 0
        tool_calls = 0

        for lc in test_claims:
            verdict = verifier.verify_claim(lc.claim)
            tool_calls += 1

            if verdict.passed == lc.expected_pass:
                correct += 1
            elif verdict.passed and not lc.expected_pass:
                false_accepts += 1
            else:
                false_rejects += 1

        n = len(test_claims)
        accuracy = correct / n if n > 0 else 0.0

        return CalibrationResult(
            config=config,
            accuracy=round(accuracy, 4),
            false_accepts=false_accepts,
            false_rejects=false_rejects,
            tool_calls=tool_calls,
        )

    # ---- Public API -------------------------------------------------------

    def sweep(self, param_name: str, values: list[float],
              test_claims: list[LabeledClaim],
              base_config: ThresholdConfig | None = None,
              ) -> list[CalibrationResult]:
        """Sweep a single threshold parameter over a range of values.

        Parameters
        ----------
        param_name : str
            Name of the ``ThresholdConfig`` attribute to vary
            (e.g. ``"theta_sim"``).
        values : list[float]
            Values to sweep over.
        test_claims : list[LabeledClaim]
            Claims with ground-truth labels.
        base_config : ThresholdConfig | None
            Starting configuration (defaults to ``ThresholdConfig()``).

        Returns
        -------
        list[CalibrationResult]
            One result per swept value, in order.
        """
        if base_config is None:
            base_config = ThresholdConfig()

        results: list[CalibrationResult] = []
        for val in values:
            cfg = ThresholdConfig(
                theta_sim=base_config.theta_sim,
                theta_margin=base_config.theta_margin,
                theta_esc=base_config.theta_esc,
                d_max=base_config.d_max,
                budget=base_config.budget,
            )
            if param_name == "theta_sim":
                cfg.theta_sim = val
            elif param_name == "theta_margin":
                cfg.theta_margin = val
            elif param_name == "theta_esc":
                cfg.theta_esc = val
            elif param_name == "d_max":
                cfg.d_max = int(val)
            elif param_name == "budget":
                cfg.budget = val
            else:
                raise ValueError(f"Unknown parameter: {param_name}")

            results.append(self._evaluate(cfg, test_claims))

        return results

    def grid_search(self, param_grid: dict[str, list[float]],
                    test_claims: list[LabeledClaim],
                    ) -> list[CalibrationResult]:
        """Full factorial grid search over multiple parameters.

        Parameters
        ----------
        param_grid : dict[str, list[float]]
            Mapping from parameter names to lists of values.
            Example: ``{"theta_sim": [1.0, 1.5], "theta_margin": [0.1, 0.3]}``
        test_claims : list[LabeledClaim]
            Claims with ground-truth labels.

        Returns
        -------
        list[CalibrationResult]
            One result per grid point.
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        results: list[CalibrationResult] = []

        for combo in itertools.product(*param_values):
            cfg = ThresholdConfig()
            for name, val in zip(param_names, combo):
                if name == "theta_sim":
                    cfg.theta_sim = val
                elif name == "theta_margin":
                    cfg.theta_margin = val
                elif name == "theta_esc":
                    cfg.theta_esc = val
                elif name == "d_max":
                    cfg.d_max = int(val)
                elif name == "budget":
                    cfg.budget = val
            results.append(self._evaluate(cfg, test_claims))

        return results

    @staticmethod
    def sensitivity_analysis(
        results: list[CalibrationResult],
    ) -> dict[str, float]:
        """Compute per-parameter sensitivity scores.

        Sensitivity is the range of accuracy observed when varying
        a single parameter across the grid — i.e., the maximum
        accuracy minus the minimum accuracy for each parameter.
        A high score means the parameter has a large effect on
        verification accuracy.

        Parameters
        ----------
        results : list[CalibrationResult]
            Results from ``sweep`` or ``grid_search``.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to sensitivity score (range
            of accuracy values).
        """
        param_names = ["theta_sim", "theta_margin", "theta_esc",
                       "d_max", "budget"]
        sensitivity: dict[str, float] = {}

        for pname in param_names:
            # Group results by the value of this parameter
            buckets: dict[float, list[float]] = {}
            for r in results:
                val = getattr(r.config, pname)
                fval = float(val)
                if fval not in buckets:
                    buckets[fval] = []
                buckets[fval].append(r.accuracy)

            if len(buckets) <= 1:
                sensitivity[pname] = 0.0
                continue

            # Sensitivity = range of mean accuracy across values
            means = [np.mean(accs) for accs in buckets.values()]
            sensitivity[pname] = round(float(max(means) - min(means)), 4)

        return sensitivity

    @staticmethod
    def best_config(results: list[CalibrationResult]) -> CalibrationResult:
        """Return the configuration with the highest accuracy.

        Ties are broken by fewer false accepts, then fewer false
        rejects.
        """
        return min(
            results,
            key=lambda r: (-r.accuracy, r.false_accepts, r.false_rejects),
        )

    @staticmethod
    def results_table(results: list[CalibrationResult]) -> str:
        """Format results as a human-readable table."""
        header = (
            f"{'theta_sim':>10s} {'theta_margin':>13s} {'theta_esc':>10s} "
            f"{'accuracy':>9s} {'FA':>4s} {'FR':>4s} {'calls':>6s}"
        )
        lines = [header, "-" * len(header)]
        for r in results:
            lines.append(
                f"{r.config.theta_sim:10.2f} {r.config.theta_margin:13.3f} "
                f"{r.config.theta_esc:10.2f} {r.accuracy:9.4f} "
                f"{r.false_accepts:4d} {r.false_rejects:4d} "
                f"{r.tool_calls:6d}"
            )
        return "\n".join(lines)
