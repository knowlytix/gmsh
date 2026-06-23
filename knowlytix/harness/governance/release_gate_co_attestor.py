# SPDX-License-Identifier: Apache-2.0
"""SPC-slope co-attestor for the tiered release gate.

A point-threshold gate decides on a single run; it cannot tell that
the run is the third in a sequence whose accuracy is collapsing 1%
per run. The chapter's :class:`~governed_agent.release_gates.RunTrend`
SPC alert detects that, but conflating it with the threshold check
either:

  * destroys the binary structural guarantee --- a slope-only block
    can flicker on noisy series and refuse a perfectly good run, or
  * hides the deviation signal --- treating slope as advisory leaves
    the audit trail with no record of the SPC firing.

Following the Ch 13 ``HolonomyCoAttestor`` pattern, this module
returns a *three-state* verdict: ``pass`` / ``flag`` / ``block``. The
structural threshold check is binary (block iff fail), the SPC
slope check is continuous (flag iff slope below per-metric
threshold), and the combined verdict is the lattice ``block ⊃ flag
⊃ pass``.

This is **distinct** from the Ch 11 ``LLMCoAttestor`` OR-semantics
because the structural check is already binary --- adding another
binary attestor would be redundant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from knowlytix.harness.governance.release_gates import (
    GateResult, RunMetrics, RunTrend, TieredReleaseGate,
)


Verdict = Literal["pass", "flag", "block"]


@dataclass
class CoAttestorVerdict:
    """Three-state verdict on a release decision.

    Attributes
    ----------
    verdict : str
        One of ``"pass"``, ``"flag"``, ``"block"``.
    structural_passed : bool
        Whether the point-threshold check passed.
    slope_alerted : bool
        Whether any SPC slope alert fired (per-metric).
    blocking_reasons : list[str]
        Reasons from the structural attestor.
    flag_reasons : list[str]
        Reasons from the SPC slope attestor.
    structural_result : GateResult
        The underlying structural :class:`GateResult`, retained for
        the audit log.
    spc_slopes : dict[str, float]
        Observed slope per metric.
    """

    verdict: Verdict
    structural_passed: bool
    slope_alerted: bool
    blocking_reasons: list[str]
    flag_reasons: list[str]
    structural_result: GateResult
    spc_slopes: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SPCSlopeConfig:
    """Per-metric slope alert thresholds.

    Slope semantics: a slope below the threshold fires the alert. For
    min-direction metrics (accuracy, confidence, fairness_air) the
    threshold is negative (e.g., -0.005 means "alert if metric
    drops 0.005 per run on average"). For
    ``robustness_drop`` (max-direction) the slope-of-concern is
    *positive* (drop growing); the threshold is positive and the
    alert fires when slope **exceeds** it.
    """

    min_acc_slope: float = -0.005
    min_conf_slope: float = -0.005
    min_fair_slope: float = -0.005
    max_rob_slope: float = +0.005
    min_runs_for_alert: int = 5

    def per_metric(self) -> dict[str, tuple[str, float]]:
        """Return ``{metric_name: (direction, threshold)}``.

        Direction is "below" or "above" describing when the alert fires.
        """
        return {
            "accuracy":        ("below", self.min_acc_slope),
            "confidence":      ("below", self.min_conf_slope),
            "fairness_air":    ("below", self.min_fair_slope),
            "robustness_drop": ("above", self.max_rob_slope),
        }


class SPCSlopeCoAttestor:
    """Three-state co-attestor combining gate point-threshold + SPC slopes.

    Construction:

      attestor = SPCSlopeCoAttestor(
          gate=gate,
          config=SPCSlopeConfig(min_acc_slope=-0.005, ...),
      )
      verdict = attestor.evaluate(metrics, tier, trend)

    The ``trend`` is a :class:`RunTrend` of *prior* runs (the current
    ``metrics`` are not appended automatically; the caller is
    responsible for chronological ordering).
    """

    def __init__(
        self,
        gate: TieredReleaseGate,
        *,
        config: SPCSlopeConfig | None = None,
    ) -> None:
        self.gate = gate
        self.config = config or SPCSlopeConfig()

    def evaluate(
        self,
        metrics: RunMetrics,
        tier: str,
        trend: RunTrend | None = None,
    ) -> CoAttestorVerdict:
        """Return the three-state verdict on this run.

        Decision lattice:

          * structural_passed=False         -> ``"block"``
          * structural_passed=True, slope_alerted=False -> ``"pass"``
          * structural_passed=True, slope_alerted=True  -> ``"flag"``
        """
        structural = self.gate.check(metrics, tier)
        spc_slopes: dict[str, float] = {}
        flag_reasons: list[str] = []
        slope_alerted = False

        if (trend is not None
                and trend.n_runs() >= self.config.min_runs_for_alert):
            for name, (direction, threshold) in (
                    self.config.per_metric().items()):
                slope = trend.slope(name)
                spc_slopes[name] = slope
                fired = (
                    (direction == "below" and slope < threshold)
                    or (direction == "above" and slope > threshold)
                )
                if fired:
                    slope_alerted = True
                    flag_reasons.append(
                        f"{name} slope {slope:+.5f} "
                        f"{direction} {threshold:+.5f}"
                    )

        if not structural.passed:
            verdict: Verdict = "block"
        elif slope_alerted:
            verdict = "flag"
        else:
            verdict = "pass"

        return CoAttestorVerdict(
            verdict=verdict,
            structural_passed=structural.passed,
            slope_alerted=slope_alerted,
            blocking_reasons=list(structural.blocking_reasons),
            flag_reasons=flag_reasons,
            structural_result=structural,
            spc_slopes=spc_slopes,
        )

    def calibrate_from_canonical(
        self,
        canonical_runs: list[RunMetrics],
        *,
        target_quantile: float = 0.05,
    ) -> SPCSlopeConfig:
        """Fit slope thresholds against a canonical-compliant trend.

        Given a sequence of metrics from a healthy production trace,
        compute the per-metric slope distribution over rolling
        windows and set the alert threshold to the
        ``target_quantile`` of that distribution. This guards against
        over-alerting on natural variation: only slopes worse than
        95% of the healthy-baseline (at quantile=0.05) trip the
        flag.

        The fitted ``SPCSlopeConfig`` is returned but *not* applied;
        the caller decides whether to swap it in.
        """
        import numpy as np

        if len(canonical_runs) < 5:
            return self.config

        per_metric_slopes: dict[str, list[float]] = {
            "accuracy": [], "confidence": [],
            "fairness_air": [], "robustness_drop": [],
        }
        window = 5
        for i in range(len(canonical_runs) - window + 1):
            slice_ = RunTrend(metrics=canonical_runs[i:i + window])
            for name in per_metric_slopes:
                per_metric_slopes[name].append(slice_.slope(name))

        cfg = SPCSlopeConfig(
            min_acc_slope=float(
                np.quantile(per_metric_slopes["accuracy"], target_quantile)),
            min_conf_slope=float(
                np.quantile(per_metric_slopes["confidence"], target_quantile)),
            min_fair_slope=float(
                np.quantile(per_metric_slopes["fairness_air"],
                            target_quantile)),
            max_rob_slope=float(
                np.quantile(per_metric_slopes["robustness_drop"],
                            1.0 - target_quantile)),
            min_runs_for_alert=self.config.min_runs_for_alert,
        )
        return cfg
