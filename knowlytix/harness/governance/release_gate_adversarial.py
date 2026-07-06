# SPDX-License-Identifier: Apache-2.0
"""Adversarial cohort and sweep for the tiered release gate.

Each :class:`AdversarialRunProbe` carries:

  * a (possibly malformed) :class:`RunMetrics`,
  * the target tier the gate is being asked to evaluate,
  * a ``safe_states`` contract: the *acceptable* gate outcomes for
    this attack class. A safe outcome is one of:

      - the gate refuses (``passed=False``) at the target tier, OR
      - the run is caught upstream by ``RunMetrics.is_well_formed()``
        and the upstream layer rejects it before reaching the gate.

The sweep evaluates each probe against the gate and reports per-class
detection rates plus a single safety_rate. A safety_rate below the
operating-point floor refuses to bind the gate to a bundle.

Attack classes (10 total, exceeds the Ch 12 ≥10-class bar):

  1. ``epsilon_below_threshold`` -- value just under the policy-declared
     floor; tests fail-closed on borderline.
  2. ``epsilon_above_threshold`` -- value just over the policy-declared
     floor; tests pass-monotonic on borderline (must pass).
  3. ``nan_metric`` -- metric set to NaN; tests NaN fail-closed
     (Risk 4 fix).
  4. ``inf_metric`` -- metric set to ±Inf; tests Inf fail-closed.
  5. ``negative_metric`` -- metric < 0 (out of domain).
  6. ``metric_above_one`` -- metric > 1.0; tests AIR-flipped attack
     (Risk 3 fix). Must fail upstream is_well_formed even if the
     gate's threshold check passes the inflated value.
  7. ``monotonicity_violating_thresholds`` -- adversary-supplied
     non-monotonic tier set; tests that gate constructor rejects.
  8. ``fairness_flipped`` -- AIR computed with disadvantaged in
     denominator (so ratio > 1) without normalization; gate must be
     fooled at the threshold check, but is_well_formed catches it.
  9. ``drift_spoofing_trend`` -- RunTrend with stationary mean but
     oscillating values; tests the SPC slope-only alert is not
     fooled.
 10. ``slope_zero_with_step_drop`` -- flat-then-step trend; tests
     that drift detection should not be slope-only.

This is the runtime equivalent of the Ch 10 OrchestratorAdversarial
sweep, applied to the SPC release gate surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knowlytix.harness.governance.release_gates import (
    GateThresholds, RunMetrics, RunTrend, TieredReleaseGate,
)


# ---------------------------------------------------------------------------
# Probe + report
# ---------------------------------------------------------------------------

@dataclass
class AdversarialRunProbe:
    """A single adversarial probe.

    Attributes
    ----------
    probe_id : str
        Stable identifier (used in audit log + replay battery).
    attack_class : str
        Name of the attack family (one of the 10 above).
    target_tier : str
        Tier the gate is asked to evaluate.
    metrics : RunMetrics | None
        The run-metric vector. ``None`` for class
        ``monotonicity_violating_thresholds`` where the attack is on
        the gate construction itself.
    custom_tiers : dict[str, GateThresholds] | None
        For class ``monotonicity_violating_thresholds``, the
        adversary-supplied tier set the gate must reject at
        construction.
    trend : RunTrend | None
        For drift-spoofing classes, the trend-time-series the SPC
        alert must (or must not) flag.
    safe_states : tuple[str, ...]
        Acceptable outcomes. Each element is one of:

          ``"gate_block"`` -- gate.check returns passed=False
          ``"gate_pass"`` -- gate.check returns passed=True (used for
                              ``epsilon_above_threshold`` positive
                              control)
          ``"upstream_block"`` -- is_well_formed returns False
          ``"constructor_reject"`` -- TieredReleaseGate(tiers=...)
                                       raises ValueError
          ``"trend_alert"`` -- RunTrend.alert returns True
          ``"trend_quiet"`` -- RunTrend.alert returns False
    description : str
        Human-readable explanation for the audit log.
    """

    probe_id: str
    attack_class: str
    target_tier: str
    metrics: RunMetrics | None = None
    custom_tiers: dict[str, GateThresholds] | None = None
    trend: RunTrend | None = None
    safe_states: tuple[str, ...] = ("gate_block",)
    description: str = ""


@dataclass
class AdversarialReleaseGateReport:
    n_probes: int
    n_safe: int
    safety_rate: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    per_probe: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cohort builder
# ---------------------------------------------------------------------------

def _action_taking_floor(
    gate: TieredReleaseGate | None = None,
) -> GateThresholds:
    """Return the action_taking floor from *gate* if provided.

    When a calibrated gate is supplied, the epsilon probes are built
    against the gate's *runtime* thresholds so positive controls
    (just-above) and negative controls (just-below) actually
    bracket the operating point. Without this, a calibrated gate
    would reject the just-above-policy-declared positive controls
    --- the probe failure would be a labeling artifact, not a real
    safety regression.
    """
    if gate is not None and "action_taking" in gate.TIERS_RUNTIME:
        return gate.TIERS_RUNTIME["action_taking"]
    return TieredReleaseGate.TIERS["action_taking"]


def build_release_gate_adversarial_cohort(
    *,
    n_per_class: int = 20,
    seed: int = 42,
    gate: TieredReleaseGate | None = None,
) -> list[AdversarialRunProbe]:
    """Generate ``n_per_class`` probes per attack class (default 200 total).

    The probes are deterministic given ``seed`` so the sweep
    reproduces byte-identically across runs --- a requirement of
    the audit-defensible bar.

    When ``gate`` is provided, the ``epsilon_below_threshold`` and
    ``epsilon_above_threshold`` boundary probes are built against
    the *gate's* operating thresholds (calibrated bundle path);
    otherwise they use the policy-declared class defaults.
    """
    rng = np.random.default_rng(seed)
    floor = _action_taking_floor(gate)
    out: list[AdversarialRunProbe] = []

    # 1. epsilon_below_threshold -- gate must refuse
    for i in range(n_per_class):
        # Pick the metric to push under
        metric_idx = i % 4
        eps = 1e-4 + float(rng.uniform(0, 1e-3))
        if metric_idx == 0:
            m = RunMetrics(
                accuracy=floor.min_accuracy - eps,
                confidence=0.80, robustness_drop=0.03,
                fairness_air=0.85, tool_calls=12, policy_blocks=1,
            )
            field_pushed = "accuracy"
        elif metric_idx == 1:
            m = RunMetrics(
                accuracy=0.90,
                confidence=floor.min_confidence - eps,
                robustness_drop=0.03, fairness_air=0.85,
                tool_calls=12, policy_blocks=1,
            )
            field_pushed = "confidence"
        elif metric_idx == 2:
            m = RunMetrics(
                accuracy=0.90, confidence=0.80,
                robustness_drop=floor.max_robustness_drop + eps,
                fairness_air=0.85, tool_calls=12, policy_blocks=1,
            )
            field_pushed = "robustness_drop"
        else:
            m = RunMetrics(
                accuracy=0.90, confidence=0.80,
                robustness_drop=0.03,
                fairness_air=floor.min_fairness_air - eps,
                tool_calls=12, policy_blocks=1,
            )
            field_pushed = "fairness_air"
        out.append(AdversarialRunProbe(
            probe_id=f"eps_below_{i:03d}",
            attack_class="epsilon_below_threshold",
            target_tier="action_taking",
            metrics=m,
            safe_states=("gate_block",),
            description=(f"action_taking floor breached on "
                         f"{field_pushed} by {eps:.4g}"),
        ))

    # 2. epsilon_above_threshold -- gate must pass (positive control)
    for i in range(n_per_class):
        eps = 1e-4 + float(rng.uniform(0, 1e-3))
        m = RunMetrics(
            accuracy=floor.min_accuracy + eps,
            confidence=floor.min_confidence + eps,
            robustness_drop=floor.max_robustness_drop - eps,
            fairness_air=floor.min_fairness_air + eps,
            tool_calls=12, policy_blocks=1,
        )
        out.append(AdversarialRunProbe(
            probe_id=f"eps_above_{i:03d}",
            attack_class="epsilon_above_threshold",
            target_tier="action_taking",
            metrics=m,
            safe_states=("gate_pass",),
            description=("all four metrics just above action_taking "
                         "floor; positive control"),
        ))

    # 3. nan_metric -- gate must refuse on every NaN field
    nan_targets = ["accuracy", "confidence",
                   "robustness_drop", "fairness_air"]
    for i in range(n_per_class):
        target = nan_targets[i % 4]
        kwargs = dict(accuracy=0.90, confidence=0.80,
                      robustness_drop=0.03, fairness_air=0.85,
                      tool_calls=12, policy_blocks=1)
        kwargs[target] = float("nan")
        out.append(AdversarialRunProbe(
            probe_id=f"nan_{i:03d}",
            attack_class="nan_metric",
            target_tier="action_taking",
            metrics=RunMetrics(**kwargs),
            safe_states=("gate_block", "upstream_block"),
            description=f"NaN injected into {target}",
        ))

    # 4. inf_metric
    for i in range(n_per_class):
        target = nan_targets[i % 4]
        kwargs = dict(accuracy=0.90, confidence=0.80,
                      robustness_drop=0.03, fairness_air=0.85,
                      tool_calls=12, policy_blocks=1)
        kwargs[target] = (float("inf") if i % 2 == 0
                          else float("-inf"))
        out.append(AdversarialRunProbe(
            probe_id=f"inf_{i:03d}",
            attack_class="inf_metric",
            target_tier="action_taking",
            metrics=RunMetrics(**kwargs),
            safe_states=("gate_block", "upstream_block"),
            description=f"+/-inf injected into {target}",
        ))

    # 5. negative_metric
    for i in range(n_per_class):
        target = nan_targets[i % 4]
        kwargs = dict(accuracy=0.90, confidence=0.80,
                      robustness_drop=0.03, fairness_air=0.85,
                      tool_calls=12, policy_blocks=1)
        kwargs[target] = -0.1 - float(rng.uniform(0, 0.5))
        out.append(AdversarialRunProbe(
            probe_id=f"neg_{i:03d}",
            attack_class="negative_metric",
            target_tier="action_taking",
            metrics=RunMetrics(**kwargs),
            safe_states=("gate_block", "upstream_block"),
            description=f"negative value in {target}",
        ))

    # 6. metric_above_one (out-of-domain; including AIR-flip)
    for i in range(n_per_class):
        target = nan_targets[i % 4]
        kwargs = dict(accuracy=0.90, confidence=0.80,
                      robustness_drop=0.03, fairness_air=0.85,
                      tool_calls=12, policy_blocks=1)
        kwargs[target] = 1.05 + float(rng.uniform(0, 0.5))
        # metric_above_one: gate's threshold passes but is_well_formed
        # MUST catch the domain violation upstream.
        out.append(AdversarialRunProbe(
            probe_id=f"above_one_{i:03d}",
            attack_class="metric_above_one",
            target_tier="action_taking",
            metrics=RunMetrics(**kwargs),
            safe_states=("upstream_block",),
            description=(f"{target}>1.0 (domain violation); only "
                         f"is_well_formed catches this"),
        ))

    # 7. monotonicity_violating_thresholds -- constructor must reject
    for i in range(n_per_class):
        # Make recommendation min_accuracy lower than advisory.
        bad = {
            "advisory": GateThresholds(
                min_accuracy=0.85, min_confidence=0.30,
            ),
            "recommendation": GateThresholds(
                min_accuracy=0.80, min_confidence=0.50,
                max_robustness_drop=0.10,
            ),
            "action_taking": GateThresholds(
                min_accuracy=0.85, min_confidence=0.70,
                max_robustness_drop=0.05,
                min_fairness_air=0.80,
            ),
        }
        out.append(AdversarialRunProbe(
            probe_id=f"mono_violating_{i:03d}",
            attack_class="monotonicity_violating_thresholds",
            target_tier="action_taking",
            metrics=None,
            custom_tiers=bad,
            safe_states=("constructor_reject",),
            description=("advisory min_accuracy >= recommendation "
                         "min_accuracy; gate constructor must "
                         "raise ValueError"),
        ))

    # 8. fairness_flipped (denom/numer reversed; mathematically AIR>1)
    # The fairness_flipped probes simulate a producer that
    # forgot to take min(rate_a/rate_b, rate_b/rate_a). Producer
    # ships AIR=1.5 ("disadvantaged group selected at 1.5x advantaged
    # rate"). The gate's >= 0.80 check passes; only is_well_formed
    # catches it.
    for i in range(n_per_class):
        flipped = 1.20 + float(rng.uniform(0, 0.40))
        m = RunMetrics(
            accuracy=0.88, confidence=0.78,
            robustness_drop=0.04,
            fairness_air=flipped,
            tool_calls=12, policy_blocks=1,
        )
        out.append(AdversarialRunProbe(
            probe_id=f"fair_flipped_{i:03d}",
            attack_class="fairness_flipped",
            target_tier="action_taking",
            metrics=m,
            safe_states=("upstream_block",),
            description=("AIR>1.0 from un-normalized "
                         "rate_disadvantaged/rate_advantaged"),
        ))

    # 9. drift_spoofing_trend: stationary mean, oscillating values.
    # Slope ≈ 0 so a slope-only alert never fires --- but the
    # safe behavior here is for the alert NOT to fire on this
    # legitimate-noise pattern (it's not actually drift). The
    # safe_state for this class is "trend_quiet".
    for i in range(n_per_class):
        # Generate sinusoidal series around 0.85
        t = np.arange(20)
        vals = 0.85 + 0.04 * np.sin(t * (0.3 + 0.05 * i))
        trend = RunTrend(metrics=[
            RunMetrics(
                accuracy=float(v), confidence=0.75,
                robustness_drop=0.04, fairness_air=0.85,
                tool_calls=12, policy_blocks=1,
            )
            for v in vals
        ])
        out.append(AdversarialRunProbe(
            probe_id=f"drift_spoof_{i:03d}",
            attack_class="drift_spoofing_trend",
            target_tier="action_taking",
            metrics=None,
            trend=trend,
            safe_states=("trend_quiet",),
            description=("oscillating ~0.85, slope ~0; SPC "
                         "must not fire"),
        ))

    # 10. slope_zero_with_step_drop: flat then sudden drop.
    # Slope-only alert may be fooled (single point pulls slope
    # weakly), so the *unsafe* outcome is "trend_quiet". The
    # safe outcome is "trend_alert" or any structural detection.
    # In production, drift detection layers in PSI on the
    # distribution; here we accept that slope-only may miss this
    # and surface it as a known limitation.
    for i in range(n_per_class):
        flat = [0.85] * 18
        drop = [0.65, 0.62]
        vals = flat + drop
        trend = RunTrend(metrics=[
            RunMetrics(
                accuracy=float(v), confidence=0.75,
                robustness_drop=0.04, fairness_air=0.85,
                tool_calls=12, policy_blocks=1,
            )
            for v in vals
        ])
        # Slope-only WILL detect this with a low threshold.
        out.append(AdversarialRunProbe(
            probe_id=f"step_drop_{i:03d}",
            attack_class="slope_zero_with_step_drop",
            target_tier="action_taking",
            metrics=None,
            trend=trend,
            safe_states=("trend_alert",),
            description=("flat 18 points + 2-point drop; SPC slope "
                         "must fire below alert_threshold=-0.005"),
        ))

    return out


# ---------------------------------------------------------------------------
# Sweep executor
# ---------------------------------------------------------------------------

def _evaluate_probe(
    gate: TieredReleaseGate,
    probe: AdversarialRunProbe,
    *,
    spc_alert_threshold: float = -0.005,
) -> tuple[str, str]:
    """Execute one probe; return ``(observed_state, evidence)``.

    Observed states map directly to the safe_states vocabulary in
    :class:`AdversarialRunProbe`.
    """
    if probe.attack_class == "monotonicity_violating_thresholds":
        try:
            TieredReleaseGate(tiers=probe.custom_tiers)
        except ValueError as exc:
            return "constructor_reject", f"raised ValueError: {exc!s:.80}"
        return "gate_pass", "constructor accepted non-monotonic config"

    if probe.attack_class in ("drift_spoofing_trend",
                              "slope_zero_with_step_drop"):
        if probe.trend is None:
            return "trend_quiet", "no trend supplied"
        slope = probe.trend.slope("accuracy")
        alerted = probe.trend.alert("accuracy", spc_alert_threshold)
        return (("trend_alert" if alerted else "trend_quiet"),
                f"slope={slope:.5f} threshold={spc_alert_threshold:.5f}")

    # Standard metric-based attacks
    if probe.metrics is None:
        return "gate_block", "no metrics; treated as block"
    well_formed, viol = probe.metrics.is_well_formed()
    if not well_formed:
        return ("upstream_block",
                f"is_well_formed=False: {viol[:1]}")
    result = gate.check(probe.metrics, probe.target_tier)
    if result.passed:
        return "gate_pass", "gate threshold check passed"
    return "gate_block", f"blocked: {result.blocking_reasons[:1]}"


def sweep_release_gate_adversarial(
    gate: TieredReleaseGate,
    cohort: list[AdversarialRunProbe],
    *,
    spc_alert_threshold: float = -0.005,
) -> AdversarialReleaseGateReport:
    """Run the adversarial cohort against the gate; report safety_rate.

    ``safety_rate = (# probes whose observed state is in safe_states)
    / total``. Per-class breakdown stratifies the rate by attack
    class so a reviewer can see exactly which class slipped.
    """
    n_safe = 0
    per_class: dict[str, dict[str, int]] = {}
    per_probe: list[dict] = []

    for probe in cohort:
        observed, evidence = _evaluate_probe(
            gate, probe, spc_alert_threshold=spc_alert_threshold,
        )
        is_safe = observed in probe.safe_states
        if is_safe:
            n_safe += 1
        cell = per_class.setdefault(
            probe.attack_class, {"n": 0, "n_safe": 0})
        cell["n"] += 1
        if is_safe:
            cell["n_safe"] += 1
        per_probe.append({
            "probe_id": probe.probe_id,
            "attack_class": probe.attack_class,
            "target_tier": probe.target_tier,
            "observed": observed,
            "is_safe": is_safe,
            "evidence": evidence,
            "safe_states": list(probe.safe_states),
        })

    n = len(cohort)
    safety_rate = (n_safe / n) if n > 0 else 0.0
    per_class_out: dict[str, dict[str, float]] = {
        cls: {
            "n": float(cell["n"]),
            "n_safe": float(cell["n_safe"]),
            "safety_rate": (cell["n_safe"] / cell["n"]
                            if cell["n"] > 0 else 0.0),
        }
        for cls, cell in per_class.items()
    }
    return AdversarialReleaseGateReport(
        n_probes=n,
        n_safe=n_safe,
        safety_rate=safety_rate,
        per_class=per_class_out,
        per_probe=per_probe,
    )
