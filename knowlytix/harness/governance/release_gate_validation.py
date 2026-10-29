# SPDX-License-Identifier: Apache-2.0
"""OOD validation for the tiered release gate (Adult Income surface).

A :class:`LabeledRun` carries a synthetic-but-grounded
``RunMetrics`` plus the ``tier_truth`` label naming the highest tier
the run should pass on a held-out evaluation set. The labelling rule
is *deterministic*: given the policy-declared thresholds, the truth
is just the strictest tier whose all-conditions-met predicate is
satisfied. This makes the cohort generator unit-testable and the
bootstrap calibration's per-tier confusion-matrix interpretable.

Adult Income context
--------------------
The cohort is grounded in the Adult-Income study used across
Ch 6/7/8/10/11/12: a tabular binary-classification task with
``sex`` and ``race`` protected attributes. ``fairness_air`` is the
adverse-impact ratio of the disadvantaged-group selection rate to
the advantaged-group rate, normalized to [0, 1]; ``accuracy`` is
test-set accuracy; ``robustness_drop`` is the accuracy delta
between the clean test set and a noise-perturbed copy;
``confidence`` is the mean geometric-distance plausibility of the
orchestrator's claim verifications.

A high-stakes deployment replaces ``adult_income_labeled_run_cohort``
with metrics computed on real validation runs --- this module's
generators provide a deterministic stand-in for calibration and
unit tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from knowlytix.harness.governance.release_gates import (
    RunMetrics,
    TieredReleaseGate,
)

_TIER_ORDER = TieredReleaseGate.TIER_ORDER


# ---------------------------------------------------------------------------
# Labeled run + report
# ---------------------------------------------------------------------------

@dataclass
class LabeledRun:
    """A RunMetrics paired with its ground-truth highest-passing tier.

    Attributes
    ----------
    run_id : str
        Stable identifier for traceability.
    metrics : RunMetrics
        The metrics observed on this run.
    tier_truth : str
        Highest tier this run *should* pass: one of "advisory",
        "recommendation", "action_taking", or "none" if the run
        should fail every tier.
    source_class : str
        Free-form provenance bucket so the OOD report can stratify
        FAR/FRR by source category (e.g.
        "adult_clean_strong_model").
    """

    run_id: str
    metrics: RunMetrics
    tier_truth: str
    source_class: str = "unspecified"


def _truth_index(tier_truth: str) -> int:
    if tier_truth == "none":
        return -1
    return _TIER_ORDER.index(tier_truth)


def _expected_pass_at(tier_truth: str, target_tier: str) -> bool:
    """True iff the labeled tier reaches *target_tier* or stricter."""
    truth_idx = _truth_index(tier_truth)
    target_idx = _TIER_ORDER.index(target_tier)
    return truth_idx >= target_idx


@dataclass
class ReleaseGateOODReport:
    """Aggregated OOD validation outcome.

    For each tier we report:

      * ``far`` : false-acceptance rate (gate passes a run whose
        truth tier is below this tier).
      * ``frr`` : false-rejection rate (gate fails a run whose truth
        tier is at or above this tier).
      * ``ece`` : expected calibration error of the gate's pass
        probability against the empirical label rate, reliability-
        diagram style with 5 equal-width bins.

    ``per_class`` stratifies FAR/FRR by ``LabeledRun.source_class``
    so the reviewer can see which generator family the gate
    misclassifies.
    """

    n: int
    decision_matches: int
    per_tier: dict[str, dict[str, float]]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    cohort_hash: str = ""
    well_formedness: dict[str, int] = field(default_factory=dict)


def _ece_from_binary(y: np.ndarray, p: np.ndarray, *, n_bins: int = 5) -> float:
    """5-bin equal-width ECE on binary labels."""
    if len(y) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    n = len(y)
    for i in range(n_bins):
        in_bin = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1
                                   else p <= bins[i + 1])
        if not np.any(in_bin):
            continue
        bin_acc = float(y[in_bin].mean())
        bin_conf = float(p[in_bin].mean())
        out += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return out


def relabel_cohort_against_gate(
    gate: TieredReleaseGate,
    cohort: list[LabeledRun],
) -> list[LabeledRun]:
    """Relabel each probe's ``tier_truth`` to match what *gate* would
    decide.

    Useful for reporting OOD against the *current operating point*
    rather than the policy-declared cohort labels --- the auditor
    sees both truths reported side by side. Truth-by-calibrated-gate
    is the gate's self-consistency check: a deterministic gate
    must produce the same verdict on the same input on every run.
    """
    out: list[LabeledRun] = []
    for lr in cohort:
        results = gate.check_all_tiers(lr.metrics)
        highest = "none"
        for t in _TIER_ORDER:
            if t in results and results[t].passed:
                highest = t
        out.append(LabeledRun(
            run_id=lr.run_id,
            metrics=lr.metrics,
            tier_truth=highest,
            source_class=f"calibrated_truth/{lr.source_class}",
        ))
    return out


def validate_release_gate(
    gate: TieredReleaseGate,
    cohort: list[LabeledRun],
) -> ReleaseGateOODReport:
    """Run the gate on every probe; report FAR/FRR/ECE per tier.

    Decision-match rule: the gate's *highest passing tier* on each
    run must equal ``tier_truth``. A run that passes recommendation
    but fails action_taking matches a truth of ``"recommendation"``.
    """
    n = len(cohort)
    if n == 0:
        return ReleaseGateOODReport(
            n=0, decision_matches=0, per_tier={}, per_class={})

    decision_matches = 0
    per_tier_counts = {
        t: {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "y": [], "p": []}
        for t in _TIER_ORDER
    }
    per_class_counts: dict[str, dict[str, int]] = {}
    well_formedness = {"finite": 0, "well_formed": 0}

    for lr in cohort:
        if lr.metrics.is_finite():
            well_formedness["finite"] += 1
        if lr.metrics.is_well_formed()[0]:
            well_formedness["well_formed"] += 1

        results = gate.check_all_tiers(lr.metrics)
        # Highest passing tier (or "none")
        highest = "none"
        for t in _TIER_ORDER:
            if t in results and results[t].passed:
                highest = t
        if highest == lr.tier_truth:
            decision_matches += 1

        for t in _TIER_ORDER:
            if t not in results:
                continue
            expected = _expected_pass_at(lr.tier_truth, t)
            actual = results[t].passed
            cell = per_tier_counts[t]
            if expected and actual:
                cell["tp"] += 1
            elif expected and not actual:
                cell["fn"] += 1
            elif (not expected) and actual:
                cell["fp"] += 1
            else:
                cell["tn"] += 1
            cell["y"].append(int(expected))
            cell["p"].append(1.0 if actual else 0.0)

        sc = lr.source_class
        sc_cell = per_class_counts.setdefault(
            sc, {"n": 0, "matches": 0})
        sc_cell["n"] += 1
        if highest == lr.tier_truth:
            sc_cell["matches"] += 1

    per_tier_out: dict[str, dict[str, float]] = {}
    for t, cell in per_tier_counts.items():
        tp, fp, tn, fn = cell["tp"], cell["fp"], cell["tn"], cell["fn"]
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        y = np.asarray(cell["y"], dtype=int)
        p = np.asarray(cell["p"], dtype=float)
        ece = _ece_from_binary(y, p)
        per_tier_out[t] = {
            "far": far, "frr": frr, "ece": ece,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    per_class_out: dict[str, dict[str, float]] = {
        sc: {"n": cell["n"],
             "match_rate": (cell["matches"] / cell["n"]
                            if cell["n"] > 0 else 0.0)}
        for sc, cell in per_class_counts.items()
    }

    return ReleaseGateOODReport(
        n=n,
        decision_matches=decision_matches,
        per_tier=per_tier_out,
        per_class=per_class_out,
        cohort_hash=cohort_fingerprint(cohort),
        well_formedness=well_formedness,
    )


# ---------------------------------------------------------------------------
# Adult Income labeled cohort
# ---------------------------------------------------------------------------

_ADULT_BASE_RUN_DESCRIPTORS = [
    # (source_class, tier_truth,
    #  accuracy, confidence, robustness_drop, fairness_air,
    #  tool_calls, policy_blocks)
    # ---- action_taking probes (clean strong model) ----
    ("adult_clean_strong_model", "action_taking",
     0.870, 0.730, 0.040, 0.870, 14, 1),
    ("adult_clean_strong_model", "action_taking",
     0.890, 0.760, 0.030, 0.880, 12, 0),
    ("adult_clean_strong_model", "action_taking",
     0.860, 0.710, 0.045, 0.840, 13, 1),
    ("adult_clean_strong_model", "action_taking",
     0.880, 0.745, 0.038, 0.860, 11, 0),
    # ---- recommendation probes (mid-tier model: passes recommendation
    # but fails action_taking on robustness_drop or fairness)
    ("adult_recommendation_robust_fail", "recommendation",
     0.835, 0.620, 0.080, 0.820, 15, 2),
    ("adult_recommendation_fairness_fail", "recommendation",
     0.840, 0.640, 0.045, 0.760, 14, 2),
    ("adult_recommendation_borderline", "recommendation",
     0.815, 0.580, 0.090, 0.810, 16, 2),
    ("adult_recommendation_high_acc_low_conf", "recommendation",
     0.860, 0.550, 0.060, 0.820, 13, 1),
    # ---- advisory probes (acceptable but limited model)
    ("adult_advisory_modest_accuracy", "advisory",
     0.750, 0.420, 0.150, 0.700, 18, 3),
    ("adult_advisory_low_robustness", "advisory",
     0.770, 0.480, 0.180, 0.720, 17, 2),
    ("adult_advisory_low_confidence", "advisory",
     0.760, 0.350, 0.120, 0.710, 16, 3),
    ("adult_advisory_low_fairness", "advisory",
     0.780, 0.480, 0.130, 0.500, 15, 2),
    # ---- "none" probes (everything below advisory)
    ("adult_below_floor_low_accuracy", "none",
     0.620, 0.350, 0.180, 0.700, 22, 4),
    ("adult_below_floor_low_confidence", "none",
     0.730, 0.250, 0.100, 0.700, 19, 3),
    ("adult_below_floor_protected_violation", "none",
     0.770, 0.500, 0.100, 0.300, 16, 5),
    ("adult_below_floor_extreme_drift", "none",
     0.720, 0.420, 0.450, 0.700, 23, 4),
]


def adult_income_labeled_run_cohort() -> list[LabeledRun]:
    """Hand-curated, deterministic 30-run Adult-Income cohort.

    Mirrors :func:`policy_validation.adult_income_labeled_cohort` for
    the gateway surface. 30 probes spanning all four truth labels
    (action_taking / recommendation / advisory / none), grounded in
    the Adult-Income study used in Ch 6 onwards. Each probe carries
    a stable ``run_id`` so the cohort hashes deterministically.

    Used to bootstrap-calibrate the gate thresholds in Lab 15.3 and
    to OOD-validate the calibrated bundle in Lab 15.4.
    """
    out: list[LabeledRun] = []
    for i, (sc, truth, acc, conf, rob, fair, tc, pb) in enumerate(
            _ADULT_BASE_RUN_DESCRIPTORS):
        out.append(LabeledRun(
            run_id=f"adult_run_{i:03d}",
            metrics=RunMetrics(
                accuracy=acc, confidence=conf,
                robustness_drop=rob, fairness_air=fair,
                tool_calls=tc, policy_blocks=pb,
                run_id=f"adult_run_{i:03d}",
            ),
            tier_truth=truth,
            source_class=sc,
        ))
    # Inject a few near-boundary probes to test the bootstrap CI
    # response. These are tier-truth deterministic given the
    # policy-declared thresholds.
    boundary = [
        # Right at action_taking floor on every metric (PASS)
        ("adult_action_taking_at_floor", "action_taking",
         0.851, 0.701, 0.049, 0.801, 12, 1),
        # Just below action_taking on accuracy only (recommendation)
        ("adult_just_below_action_acc", "recommendation",
         0.849, 0.720, 0.045, 0.820, 12, 1),
        # Just below action_taking on robustness only (recommendation)
        ("adult_just_below_action_rob", "recommendation",
         0.870, 0.720, 0.051, 0.820, 12, 1),
        # Just below recommendation on confidence (advisory)
        ("adult_just_below_rec_conf", "advisory",
         0.810, 0.499, 0.090, 0.810, 14, 2),
        # Just below advisory on accuracy (none)
        ("adult_just_below_advisory_acc", "none",
         0.699, 0.320, 0.150, 0.710, 19, 3),
        # Just below advisory on confidence (none)
        ("adult_just_below_advisory_conf", "none",
         0.720, 0.299, 0.140, 0.710, 18, 3),
        # Borderline + perfect-tier alignment for cross-checks
        ("adult_action_high_margin", "action_taking",
         0.910, 0.820, 0.020, 0.910, 10, 0),
        ("adult_recommendation_clean", "recommendation",
         0.825, 0.610, 0.075, 0.830, 14, 2),
        ("adult_advisory_clean", "advisory",
         0.745, 0.450, 0.140, 0.715, 17, 2),
        ("adult_well_below_floor", "none",
         0.500, 0.200, 0.350, 0.500, 25, 5),
        ("adult_recommendation_air_at_floor", "recommendation",
         0.860, 0.720, 0.040, 0.700, 12, 1),
        ("adult_action_robust_at_floor", "action_taking",
         0.880, 0.750, 0.050, 0.850, 11, 0),
        ("adult_advisory_high_robust_drop", "advisory",
         0.760, 0.420, 0.250, 0.700, 20, 3),
        ("adult_action_air_at_floor", "action_taking",
         0.870, 0.730, 0.040, 0.800, 12, 1),
    ]
    for j, (sc, truth, acc, conf, rob, fair, tc, pb) in enumerate(boundary):
        out.append(LabeledRun(
            run_id=f"adult_run_{len(_ADULT_BASE_RUN_DESCRIPTORS) + j:03d}",
            metrics=RunMetrics(
                accuracy=acc, confidence=conf,
                robustness_drop=rob, fairness_air=fair,
                tool_calls=tc, policy_blocks=pb,
                run_id=(f"adult_run_"
                        f"{len(_ADULT_BASE_RUN_DESCRIPTORS) + j:03d}"),
            ),
            tier_truth=truth,
            source_class=sc,
        ))
    return out


# ---------------------------------------------------------------------------
# Programmatic large cohort (Adult-Income parameterized)
# ---------------------------------------------------------------------------

def large_adult_income_run_cohort(
    *,
    n_per_tier: int = 300,
    seed: int = 42,
) -> list[LabeledRun]:
    """Programmatic ≥1000-probe Adult-Income cohort for serious audit.

    Sample n_per_tier runs from each of the four truth strata
    (action_taking / recommendation / advisory / none) by drawing
    metric values from per-tier-conditioned distributions. Tier
    truth is enforced by *clipping* the sample into the polytope
    that the policy-declared thresholds carve out for that tier ---
    the cohort is therefore deterministic given a seed and survives
    threshold recalibration without re-labeling (the calibrated
    thresholds are within bootstrap CIs of the declared ones, so a
    truth-by-construction labeling stays close to truth-by-policy).

    Total cohort size: 4 * n_per_tier (default 1200).
    """
    rng = np.random.default_rng(seed)
    out: list[LabeledRun] = []

    def _sample_action_taking() -> RunMetrics:
        # action_taking polytope: acc ≥ 0.85, conf ≥ 0.70, rob ≤ 0.05,
        # fair ≥ 0.80
        acc = float(np.clip(rng.normal(0.90, 0.025), 0.851, 1.0))
        conf = float(np.clip(rng.normal(0.78, 0.030), 0.701, 1.0))
        rob = float(np.clip(rng.exponential(0.025), 0.0, 0.049))
        fair = float(np.clip(rng.normal(0.88, 0.030), 0.801, 1.0))
        return RunMetrics(
            acc, conf, rob, fair,
            tool_calls=int(rng.integers(8, 16)),
            policy_blocks=int(rng.integers(0, 3)),
        )

    def _sample_recommendation() -> RunMetrics:
        # recommendation polytope but NOT action_taking: at least one of
        # (acc < 0.85, conf < 0.70, rob > 0.05, fair < 0.80) must hold,
        # while all of (acc ≥ 0.80, conf ≥ 0.50, rob ≤ 0.10) must hold.
        acc = float(np.clip(rng.normal(0.83, 0.025), 0.801, 0.9))
        conf = float(np.clip(rng.normal(0.62, 0.040), 0.501, 0.78))
        rob = float(np.clip(rng.uniform(0.04, 0.099), 0.04, 0.099))
        fair = float(np.clip(rng.normal(0.83, 0.05), 0.5, 1.0))
        # Ensure NOT action_taking by forcing at least one violator.
        # Pick one of the four to push under the action_taking floor.
        violator = int(rng.integers(0, 4))
        if violator == 0:
            acc = float(np.clip(acc, 0.801, 0.849))
        elif violator == 1:
            conf = float(np.clip(conf, 0.501, 0.699))
        elif violator == 2:
            rob = float(np.clip(rob, 0.051, 0.099))
        else:
            fair = float(np.clip(fair, 0.5, 0.799))
        return RunMetrics(
            acc, conf, rob, fair,
            tool_calls=int(rng.integers(10, 18)),
            policy_blocks=int(rng.integers(0, 4)),
        )

    def _sample_advisory() -> RunMetrics:
        # advisory polytope but NOT recommendation: passes (acc ≥ 0.70,
        # conf ≥ 0.30); at least one of (acc < 0.80, conf < 0.50,
        # rob > 0.10) holds.
        acc = float(np.clip(rng.normal(0.75, 0.025), 0.701, 0.85))
        conf = float(np.clip(rng.normal(0.42, 0.06), 0.301, 0.65))
        rob = float(np.clip(rng.uniform(0.09, 0.30), 0.09, 0.30))
        fair = float(np.clip(rng.normal(0.7, 0.10), 0.3, 1.0))
        # Force at least one to violate recommendation.
        violator = int(rng.integers(0, 3))
        if violator == 0:
            acc = float(np.clip(acc, 0.701, 0.799))
        elif violator == 1:
            conf = float(np.clip(conf, 0.301, 0.499))
        else:
            rob = float(np.clip(rob, 0.101, 0.30))
        return RunMetrics(
            acc, conf, rob, fair,
            tool_calls=int(rng.integers(12, 20)),
            policy_blocks=int(rng.integers(1, 5)),
        )

    def _sample_none() -> RunMetrics:
        # none = below advisory: at least one of (acc < 0.70, conf < 0.30)
        acc = float(np.clip(rng.normal(0.62, 0.07), 0.20, 0.80))
        conf = float(np.clip(rng.normal(0.25, 0.08), 0.05, 0.55))
        rob = float(np.clip(rng.uniform(0.05, 0.50), 0.05, 0.50))
        fair = float(np.clip(rng.normal(0.65, 0.15), 0.10, 1.0))
        violator = int(rng.integers(0, 2))
        if violator == 0:
            acc = float(np.clip(acc, 0.20, 0.699))
        else:
            conf = float(np.clip(conf, 0.05, 0.299))
        return RunMetrics(
            acc, conf, rob, fair,
            tool_calls=int(rng.integers(15, 25)),
            policy_blocks=int(rng.integers(2, 7)),
        )

    samplers = {
        "action_taking": _sample_action_taking,
        "recommendation": _sample_recommendation,
        "advisory": _sample_advisory,
        "none": _sample_none,
    }
    source_classes = {
        "action_taking": "adult_strong_model_synth",
        "recommendation": "adult_mid_model_synth",
        "advisory": "adult_modest_model_synth",
        "none": "adult_below_floor_synth",
    }

    idx = 0
    for tier in ("action_taking", "recommendation", "advisory", "none"):
        for _ in range(n_per_tier):
            m = samplers[tier]()
            m.run_id = f"adult_synth_{idx:05d}"
            out.append(LabeledRun(
                run_id=f"adult_synth_{idx:05d}",
                metrics=m,
                tier_truth=tier,
                source_class=source_classes[tier],
            ))
            idx += 1
    return out


# ---------------------------------------------------------------------------
# Cohort fingerprint + persistence
# ---------------------------------------------------------------------------

def cohort_fingerprint(cohort: list[LabeledRun]) -> str:
    """Deterministic SHA-256 over the canonical cohort representation.

    Includes every probe's (run_id, tier_truth, source_class,
    metrics tuple). Two cohorts that produce the same hash bind the
    bundle to the same calibration evidence base.
    """
    payload = []
    for lr in cohort:
        m = lr.metrics
        payload.append({
            "run_id": lr.run_id,
            "tier_truth": lr.tier_truth,
            "source_class": lr.source_class,
            "accuracy": m.accuracy,
            "confidence": m.confidence,
            "robustness_drop": m.robustness_drop,
            "fairness_air": m.fairness_air,
            "tool_calls": m.tool_calls,
            "policy_blocks": m.policy_blocks,
        })
    canon = json.dumps(
        sorted(payload, key=lambda d: d["run_id"]),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def save_cohort(cohort: list[LabeledRun], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for lr in cohort:
        d = {
            "run_id": lr.run_id,
            "tier_truth": lr.tier_truth,
            "source_class": lr.source_class,
            "metrics": asdict(lr.metrics),
        }
        payload.append(d)
    p.write_text(json.dumps(payload, indent=2))


def load_cohort(path: str | Path) -> list[LabeledRun]:
    p = Path(path)
    payload = json.loads(p.read_text())
    out: list[LabeledRun] = []
    for d in payload:
        m_d = d["metrics"]
        m = RunMetrics(**m_d)
        out.append(LabeledRun(
            run_id=d["run_id"],
            metrics=m,
            tier_truth=d["tier_truth"],
            source_class=d.get("source_class", "unspecified"),
        ))
    return out
