# SPDX-License-Identifier: Apache-2.0
"""Tiered release gates and statistical process control for governed agents.

Provides three deployment tiers with monotonically increasing thresholds:

    advisory       - accuracy >= 0.70, confidence >= 0.30
    recommendation - accuracy >= 0.80, confidence >= 0.50, robustness_drop < 0.10
    action_taking  - accuracy >= 0.85, confidence >= 0.70, robustness_drop < 0.05,
                     fairness_air >= 0.80

The thresholds shipped on the class are *policy-declared* defaults
suitable for pedagogy. High-stakes deployment must instantiate
:class:`TieredReleaseGate` from a calibrated
:class:`~governed_agent.release_gate_bundle.ReleaseGateBundle` so the
operating thresholds are bootstrap-fit on a labeled run cohort and
shipped under a signed, fingerprinted artefact.

NaN/Inf handling: every comparison is wrapped so that a non-finite
metric value is treated as a hard fail-closed (BLOCK), regardless of
direction. A NaN ``accuracy`` does not silently pass; a NaN
``robustness_drop`` does not silently pass; a NaN ``fairness_air``
does not silently pass.

RunTrend tracks metrics across multiple runs and computes slopes to
detect drift via simple linear regression on the metric time series.

Built in Chapter 15 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from knowlytix.harness.governance.orchestrator import OrchestratorResult


# ---------------------------------------------------------------------------
# Helpers: metric-direction-aware finite-value guards
# ---------------------------------------------------------------------------

def _is_finite(x: float) -> bool:
    """Return True iff x is a finite real number (NaN/Inf -> False)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _check_min(value: float, threshold: float) -> bool:
    """Pass iff value is finite and value >= threshold.

    Used for ``min_*`` thresholds (accuracy, confidence, fairness_air).
    NaN/Inf fail closed.
    """
    if not _is_finite(value):
        return False
    return float(value) >= float(threshold)


def _check_max(value: float, threshold: float) -> bool:
    """Pass iff value is finite and value <= threshold.

    Used for ``max_*`` thresholds (robustness_drop). NaN/Inf fail
    closed --- a NaN robustness_drop must never be silently accepted
    just because the comparison happens to evaluate to False against
    the inverse-direction Python operator.
    """
    if not _is_finite(value):
        return False
    return float(value) <= float(threshold)


# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

@dataclass
class GateThresholds:
    """Per-tier thresholds for the release gate.

    Attributes
    ----------
    min_accuracy : float
        Minimum accuracy required.
    min_confidence : float
        Minimum geometric confidence required.
    max_robustness_drop : float
        Maximum allowed accuracy drop under noise (1.0 = no constraint).
    min_fairness_air : float
        Minimum adverse impact ratio. Direction convention: AIR is the
        ratio of the disadvantaged-group selection rate to the
        advantaged-group selection rate, in [0, 1] with 1.0 = perfect
        parity. A run where the disadvantaged group is *favored*
        (rate > advantaged) must compute AIR as min(rate_a/rate_b,
        rate_b/rate_a) so the ratio remains in [0, 1] and the gate
        cannot be fooled by direction-flipped numerators. The
        adversarial cohort exercises this case.
    is_calibrated : bool
        ``True`` iff these thresholds were derived from a bootstrap
        fit on a labeled cohort (carried in a
        :class:`~governed_agent.release_gate_bundle.ReleaseGateBundle`),
        ``False`` for the policy-declared class defaults shipped here.
        The audit log surfaces this flag so a reviewer can tell
        whether the operating thresholds came from data or from
        author intent.
    source : str
        Free-form provenance string (e.g.,
        ``"policy_declared"``, ``"bundle:7359d07153302322"``).
    """

    min_accuracy: float
    min_confidence: float
    max_robustness_drop: float = 1.0   # no constraint by default
    min_fairness_air: float = 0.0      # no constraint by default
    is_calibrated: bool = False
    source: str = "policy_declared"


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Outcome of checking a single tier.

    Attributes
    ----------
    tier : str
        Name of the tier that was checked.
    passed : bool
        Whether all checks in this tier passed.
    checks : dict[str, dict]
        Per-check details: {check_name: {value, threshold, passed,
        finite}}.
    blocking_reasons : list[str]
        Human-readable reasons for any failed check.
    is_calibrated : bool
        Whether the thresholds applied were calibrated.
    threshold_source : str
        Provenance of the thresholds applied.
    """

    tier: str
    passed: bool
    checks: dict[str, dict]
    blocking_reasons: list[str]
    is_calibrated: bool = False
    threshold_source: str = "policy_declared"


# ---------------------------------------------------------------------------
# Run metrics
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Metrics collected from a single agent run.

    Attributes
    ----------
    accuracy : float
        Classification or regression accuracy in [0, 1].
    confidence : float
        Mean geometric confidence from claim verification in [0, 1].
    robustness_drop : float
        Accuracy degradation under noise (0 = no drop, 1 = total
        failure).
    fairness_air : float
        Adverse impact ratio (1.0 = perfect parity); see
        :class:`GateThresholds.min_fairness_air` for direction
        convention.
    tool_calls : int
        Number of tool calls executed in the run.
    policy_blocks : int
        Number of tool calls blocked by the policy engine.
    run_id : str
        Optional identifier for tracing (audit log, replay battery).
    """

    accuracy: float
    confidence: float
    robustness_drop: float
    fairness_air: float
    tool_calls: int
    policy_blocks: int
    run_id: str = ""

    @classmethod
    def from_orchestrator_result(
        cls,
        result: "OrchestratorResult",
        *,
        accuracy: float,
        robustness_drop: float,
        fairness_air: float,
        run_id: str = "",
    ) -> "RunMetrics":
        """Adapter: build RunMetrics from an OrchestratorResult.

        ``confidence`` is computed from the orchestrator's claim
        verifications (mean geodesic-distance plausibility); ``tool_calls``
        and ``policy_blocks`` are read off the result. ``accuracy``,
        ``robustness_drop`` and ``fairness_air`` are external metrics
        the caller computes from a held-out evaluation set --- the
        orchestrator does not know its own ground-truth accuracy.

        This adapter exists so production wiring is one method
        call instead of an open question. A demo that fabricates
        synthetic ``RunMetrics`` is fine for pedagogy; a high-stakes
        deployment must call this method on real
        :class:`~governed_agent.orchestrator.OrchestratorResult`.
        """
        # Orchestrator surfaces:
        #   result.tool_calls       -> int
        #   result.policy_blocks    -> int
        #   result.claim_verifications  -> list of dicts with 'plausibility' key
        # Confidence is the mean plausibility distance, normalized to [0,1]
        # via 1 / (1 + d) so smaller distance => higher confidence.
        try:
            verifs = list(getattr(result, "claim_verifications", []) or [])
        except Exception:
            verifs = []
        if verifs:
            distances = []
            for v in verifs:
                d = (v.get("plausibility") if isinstance(v, dict)
                     else getattr(v, "plausibility", None))
                if d is not None and _is_finite(d):
                    distances.append(float(d))
            if distances:
                mean_d = float(np.mean(distances))
                confidence = 1.0 / (1.0 + mean_d)
            else:
                confidence = float("nan")
        else:
            confidence = float("nan")

        tool_calls = int(getattr(result, "tool_calls", 0) or 0)
        policy_blocks = int(getattr(result, "policy_blocks", 0) or 0)

        return cls(
            accuracy=float(accuracy),
            confidence=confidence,
            robustness_drop=float(robustness_drop),
            fairness_air=float(fairness_air),
            tool_calls=tool_calls,
            policy_blocks=policy_blocks,
            run_id=run_id or getattr(result, "run_id", "") or "",
        )

    def is_finite(self) -> bool:
        """Return True iff every numeric metric is finite.

        A reviewer can call this before passing metrics into the gate
        to surface NaN/Inf early; the gate itself fails-closed on
        non-finite values, but an explicit check produces a clearer
        audit trail.
        """
        return all(_is_finite(getattr(self, f)) for f in
                   ("accuracy", "confidence",
                    "robustness_drop", "fairness_air"))

    def is_well_formed(self) -> tuple[bool, list[str]]:
        """Return (ok, violations) for domain-range validation.

        Domain conventions:

          * ``accuracy``        in [0, 1]
          * ``confidence``      in [0, 1]
          * ``robustness_drop`` in [0, 1]
          * ``fairness_air``    in [0, 1] (1.0 = parity, lower = worse;
            producer must pre-normalize using
            ``min(rate_a/rate_b, rate_b/rate_a)`` so a flipped-direction
            ratio cannot present as super-parity)

        This is separate from :meth:`is_finite`: a finite metric can
        still be domain-out-of-range. A high-stakes deployment
        validates ``is_well_formed`` *before* feeding metrics to the
        gate, so a malformed RunMetrics is rejected with a clear
        domain reason rather than silently accepted by an
        AIR-fooled gate.
        """
        violations: list[str] = []
        for name, lo, hi in (
            ("accuracy", 0.0, 1.0),
            ("confidence", 0.0, 1.0),
            ("robustness_drop", 0.0, 1.0),
            ("fairness_air", 0.0, 1.0),
        ):
            v = getattr(self, name)
            if not _is_finite(v):
                violations.append(f"{name}={v!r} non-finite")
                continue
            vf = float(v)
            if vf < lo:
                violations.append(
                    f"{name}={vf:.4f} below domain floor {lo:.2f}")
            elif vf > hi:
                violations.append(
                    f"{name}={vf:.4f} above domain ceiling {hi:.2f} "
                    f"(producer must pre-normalize)")
        if self.tool_calls < 0:
            violations.append(
                f"tool_calls={self.tool_calls} negative")
        if self.policy_blocks < 0:
            violations.append(
                f"policy_blocks={self.policy_blocks} negative")
        return (len(violations) == 0, violations)


# ---------------------------------------------------------------------------
# Run trend (SPC)
# ---------------------------------------------------------------------------

@dataclass
class RunTrend:
    """Track metrics across multiple runs for statistical process control.

    Stores a sequence of RunMetrics and provides slope, mean and
    alert methods for drift detection.

    Attributes
    ----------
    metrics : list[RunMetrics]
        Chronologically ordered list of run metrics.
    """

    metrics: list[RunMetrics] = field(default_factory=list)

    def _values(self, field_name: str) -> np.ndarray:
        """Extract a 1-D array of finite values for the given field.

        Non-finite entries are dropped --- they would corrupt OLS slope
        estimates with NaN propagation, so SPC computes the slope on
        the finite subset and the audit log records how many runs
        were excluded for non-finiteness via :meth:`n_finite`.
        """
        raw = np.array([getattr(m, field_name) for m in self.metrics],
                       dtype=np.float64)
        return raw[np.isfinite(raw)]

    def n_finite(self, field_name: str) -> int:
        """Number of runs with a finite value for the metric."""
        return int(self._values(field_name).size)

    def n_runs(self) -> int:
        """Total number of runs recorded (including non-finite)."""
        return len(self.metrics)

    def slope(self, field_name: str) -> float:
        """Compute the OLS slope of the metric over run index.

        Returns 0.0 if fewer than 2 finite data points. NaN/Inf values
        in the metric are excluded from the fit.
        """
        vals = self._values(field_name)
        n = len(vals)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        # OLS: slope = cov(x, y) / var(x)
        x_mean = x.mean()
        y_mean = vals.mean()
        numerator = np.sum((x - x_mean) * (vals - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def mean(self, field_name: str) -> float:
        """Compute the arithmetic mean of the metric across all finite
        runs."""
        vals = self._values(field_name)
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals))

    def alert(self, field_name: str, threshold: float) -> bool:
        """Return True if the slope of the metric exceeds the threshold.

        For metrics where a negative slope is bad (accuracy, confidence),
        use a negative threshold (e.g., -0.01) and check whether
        slope < threshold.

        Parameters
        ----------
        field_name : str
            Name of the RunMetrics field.
        threshold : float
            If the slope is *below* this value, an alert fires.

        Returns
        -------
        bool
            True if the slope is below the threshold (drift detected).
        """
        return self.slope(field_name) < threshold


# ---------------------------------------------------------------------------
# Tiered release gate
# ---------------------------------------------------------------------------

class TieredReleaseGate:
    """Three-tier release gate with monotonically increasing stringency.

    Tiers (in increasing order of stringency):
        advisory       - minimal bar for informational use
        recommendation - safe for human-in-the-loop recommendations
        action_taking  - safe for autonomous decisions

    The check is fully deterministic: given the same RunMetrics and
    tier, the GateResult is always identical.

    Two instantiation paths:

      * Default (policy-declared): ``TieredReleaseGate()`` uses the
        class ``TIERS`` constants. Suitable for pedagogy and
        development.
      * Calibrated (bundle-bound): ``TieredReleaseGate.from_bundle(bundle)``
        binds the gate to thresholds carried in a signed, fingerprinted
        :class:`~governed_agent.release_gate_bundle.ReleaseGateBundle`.
        Required for high-stakes deployment.
    """

    TIERS: dict[str, GateThresholds] = {
        "advisory": GateThresholds(
            min_accuracy=0.70,
            min_confidence=0.30,
        ),
        "recommendation": GateThresholds(
            min_accuracy=0.80,
            min_confidence=0.50,
            max_robustness_drop=0.10,
        ),
        "action_taking": GateThresholds(
            min_accuracy=0.85,
            min_confidence=0.70,
            max_robustness_drop=0.05,
            min_fairness_air=0.80,
        ),
    }

    # Canonical ordering from least to most stringent
    TIER_ORDER = ["advisory", "recommendation", "action_taking"]

    def __init__(
        self,
        tiers: dict[str, GateThresholds] | None = None,
        *,
        threshold_source: str = "policy_declared",
    ) -> None:
        """Construct the gate.

        Parameters
        ----------
        tiers : dict[str, GateThresholds] | None
            Per-tier thresholds. Defaults to the class
            ``TIERS`` constants.
        threshold_source : str
            Free-form provenance string surfaced in the GateResult so
            an auditor can tell whether thresholds came from the
            policy-declared defaults or a calibrated bundle.

        Raises
        ------
        ValueError
            If the tier configuration violates the monotonicity
            invariant (advisory_pass >= recommendation_pass >=
            action_taking_pass for every metric vector). This catches
            mis-configured threshold sets at construction time.
        """
        if tiers is None:
            tiers = dict(self.TIERS)
        self._tiers = tiers
        self.threshold_source = threshold_source
        self._validate_monotonic_thresholds()

    def _validate_monotonic_thresholds(self) -> None:
        """Refuse to accept tier sets that violate monotonicity.

        Stricter tiers must require *at least as much* on every
        directional metric as the looser tiers. We enforce this on
        the threshold constants directly --- not just per-input ---
        so adversary-supplied tier configs are rejected at the gate
        constructor, not silently used until a divergent input
        surfaces the bug.
        """
        if list(self._tiers.keys()) and not all(
                t in self._tiers for t in self.TIER_ORDER):
            # Allow custom tiers; only enforce monotonicity within
            # the canonical advisory/recommendation/action_taking
            # ordering when all three are present.
            return
        prev = None
        for tier_name in self.TIER_ORDER:
            t = self._tiers.get(tier_name)
            if t is None:
                continue
            if prev is not None:
                # min-direction: stricter must be >= prev
                if t.min_accuracy < prev.min_accuracy:
                    raise ValueError(
                        f"Tier '{tier_name}' min_accuracy "
                        f"{t.min_accuracy} < previous tier "
                        f"{prev.min_accuracy}; monotonicity violated."
                    )
                if t.min_confidence < prev.min_confidence:
                    raise ValueError(
                        f"Tier '{tier_name}' min_confidence "
                        f"{t.min_confidence} < previous tier "
                        f"{prev.min_confidence}; monotonicity violated."
                    )
                if t.min_fairness_air < prev.min_fairness_air:
                    raise ValueError(
                        f"Tier '{tier_name}' min_fairness_air "
                        f"{t.min_fairness_air} < previous tier "
                        f"{prev.min_fairness_air}; monotonicity violated."
                    )
                # max-direction: stricter must be <= prev
                if t.max_robustness_drop > prev.max_robustness_drop:
                    raise ValueError(
                        f"Tier '{tier_name}' max_robustness_drop "
                        f"{t.max_robustness_drop} > previous tier "
                        f"{prev.max_robustness_drop}; monotonicity "
                        f"violated."
                    )
            prev = t

    @property
    def TIERS_RUNTIME(self) -> dict[str, GateThresholds]:
        """The tier set actually in effect on this instance.

        Distinct from the class-level ``TIERS`` constant when the
        gate was constructed with custom thresholds (e.g. from a
        calibrated bundle).
        """
        return self._tiers

    @classmethod
    def from_bundle(cls, bundle: Any) -> "TieredReleaseGate":
        """Construct a gate from a signed ReleaseGateBundle.

        Refuses to construct on a non-verified bundle (load-time
        verification is the bundle's responsibility; this method
        just trusts that the caller obtained the bundle from
        :meth:`ReleaseGateBundle.load`).

        Parameters
        ----------
        bundle : ReleaseGateBundle
            A verified, calibrated bundle.
        """
        # Late import to avoid cycle.
        from knowlytix.harness.governance.release_gate_bundle import ReleaseGateBundle
        if not isinstance(bundle, ReleaseGateBundle):
            raise TypeError(
                f"from_bundle requires a ReleaseGateBundle, "
                f"got {type(bundle).__name__}."
            )
        tiers = bundle.calibrated_tiers()
        gate = cls(
            tiers=tiers,
            threshold_source=f"bundle:{bundle.bundle_id}",
        )
        gate._bundle = bundle  # retain for audit
        return gate

    def check(self, metrics: RunMetrics, tier: str) -> GateResult:
        """Check whether *metrics* pass the specified *tier*.

        Parameters
        ----------
        metrics : RunMetrics
            The metrics from a single agent run.
        tier : str
            One of "advisory", "recommendation", "action_taking".

        Returns
        -------
        GateResult
            Deterministic gate result with per-check details.

        Raises
        ------
        ValueError
            If *tier* is not a recognized tier name.
        """
        if tier not in self._tiers:
            raise ValueError(
                f"Unknown tier '{tier}'. Choose from "
                f"{list(self._tiers)}")

        thresholds = self._tiers[tier]
        checks: dict[str, dict] = {}
        blocking: list[str] = []

        # --- accuracy (min) ---
        acc_finite = _is_finite(metrics.accuracy)
        acc_ok = _check_min(metrics.accuracy, thresholds.min_accuracy)
        checks["accuracy"] = {
            "value": metrics.accuracy,
            "threshold": thresholds.min_accuracy,
            "passed": acc_ok,
            "finite": acc_finite,
        }
        if not acc_ok:
            if not acc_finite:
                blocking.append(
                    f"accuracy {metrics.accuracy!r} is not finite "
                    f"(NaN/Inf fail-closed)")
            else:
                blocking.append(
                    f"accuracy {metrics.accuracy:.4f} "
                    f"< {thresholds.min_accuracy:.2f}")

        # --- confidence (min) ---
        conf_finite = _is_finite(metrics.confidence)
        conf_ok = _check_min(metrics.confidence,
                             thresholds.min_confidence)
        checks["confidence"] = {
            "value": metrics.confidence,
            "threshold": thresholds.min_confidence,
            "passed": conf_ok,
            "finite": conf_finite,
        }
        if not conf_ok:
            if not conf_finite:
                blocking.append(
                    f"confidence {metrics.confidence!r} is not finite "
                    f"(NaN/Inf fail-closed)")
            else:
                blocking.append(
                    f"confidence {metrics.confidence:.4f} "
                    f"< {thresholds.min_confidence:.2f}")

        # --- robustness_drop (max) ---
        rob_finite = _is_finite(metrics.robustness_drop)
        rob_ok = _check_max(metrics.robustness_drop,
                            thresholds.max_robustness_drop)
        checks["robustness_drop"] = {
            "value": metrics.robustness_drop,
            "threshold": thresholds.max_robustness_drop,
            "passed": rob_ok,
            "finite": rob_finite,
        }
        if not rob_ok:
            if not rob_finite:
                blocking.append(
                    f"robustness_drop {metrics.robustness_drop!r} "
                    f"is not finite (NaN/Inf fail-closed)")
            else:
                blocking.append(
                    f"robustness_drop {metrics.robustness_drop:.4f} "
                    f"> {thresholds.max_robustness_drop:.2f}")

        # --- fairness_air (min) ---
        fair_finite = _is_finite(metrics.fairness_air)
        fair_ok = _check_min(metrics.fairness_air,
                             thresholds.min_fairness_air)
        checks["fairness_air"] = {
            "value": metrics.fairness_air,
            "threshold": thresholds.min_fairness_air,
            "passed": fair_ok,
            "finite": fair_finite,
        }
        if not fair_ok:
            if not fair_finite:
                blocking.append(
                    f"fairness_air {metrics.fairness_air!r} "
                    f"is not finite (NaN/Inf fail-closed)")
            else:
                blocking.append(
                    f"fairness_air {metrics.fairness_air:.4f} "
                    f"< {thresholds.min_fairness_air:.2f}")

        passed = len(blocking) == 0

        return GateResult(
            tier=tier,
            passed=passed,
            checks=checks,
            blocking_reasons=blocking,
            is_calibrated=thresholds.is_calibrated,
            threshold_source=thresholds.source or self.threshold_source,
        )

    def check_all_tiers(self, metrics: RunMetrics) -> dict[str, GateResult]:
        """Check *metrics* against all three tiers.

        Returns
        -------
        dict[str, GateResult]
            Mapping from tier name to GateResult.
        """
        return {tier: self.check(metrics, tier)
                for tier in self.TIER_ORDER if tier in self._tiers}

    def verify_monotonicity(self, metrics: RunMetrics) -> bool:
        """Verify that gate outcomes are monotonically ordered.

        If action_taking passes, recommendation must also pass.
        If recommendation passes, advisory must also pass.

        This property must hold for any RunMetrics because the
        thresholds are monotonically ordered (the gate constructor
        verifies the threshold-set monotonicity invariant up front).

        Returns
        -------
        bool
            True if the monotonicity property holds.
        """
        results = self.check_all_tiers(metrics)
        passes = [results[t].passed for t in self.TIER_ORDER
                  if t in results]
        # advisory >= recommendation >= action_taking
        # Encoded as: if a stricter tier passes, all looser tiers must pass.
        for i in range(len(passes) - 1):
            if not passes[i] and passes[i + 1]:
                return False
        return True

    @staticmethod
    def format_gate_result(result: GateResult) -> str:
        """Format a GateResult for human-readable display."""
        status = "PASS" if result.passed else "FAIL"
        lines = [f"  [{status}] {result.tier}"]
        for name, info in result.checks.items():
            icon = "+" if info["passed"] else "X"
            value = info["value"]
            value_s = (f"{value:.4f}" if _is_finite(value) else f"{value!r}")
            lines.append(
                f"    [{icon}] {name}: {value_s} "
                f"(threshold: {info['threshold']:.2f})")
        if result.blocking_reasons:
            lines.append(
                f"    Blocking: {'; '.join(result.blocking_reasons)}")
        return "\n".join(lines)
