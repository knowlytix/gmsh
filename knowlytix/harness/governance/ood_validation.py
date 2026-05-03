# SPDX-License-Identifier: Apache-2.0
"""Held-out / out-of-distribution validation of a GovernanceBundle.

Bootstrap CIs are valid only on the calibration distribution. Before
deploying a bundle into a high-stakes setting, the runtime gate's
behavior must be characterized on labeled data drawn from a
*different* distribution: paraphrased entities, alternate domain,
adversarial perturbations, or a held-out store.

Two validation modes:

  * ``validate_calibration(bundle, samples)`` — checks the bundle's
    isotonic ``te → P(contra)`` mapping against labeled
    ``CalibrationSample`` values. No store / gate needed. Tests
    *calibration alone* under shift.
  * ``validate_gate(gate, labeled_pairs)`` — runs the full
    ``RuntimeGate`` (including policy and support gating) over
    labeled entity pairs and reports per-band confusion. Tests
    *end-to-end deployment behavior*.

Both modes return an ``OODValidationReport`` with FAR / FRR / ECE
and a per-class confusion matrix.  A minimum-acceptable report
under shift is the load-bearing artifact of high-stakes sign-off:
it is what an auditor reads.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.harness.testing.primitive_calibration import CalibrationSample
    from knowlytix.harness.governance.governance_bundle import GovernanceBundle
    from knowlytix.harness.governance.runtime_gate import RuntimeGate


GroundTruth = Literal["agree", "unrelated", "contradict"]


@dataclass
class LabeledPair:
    """One ground-truth pair for end-to-end gate validation."""

    a: str
    b: str
    label: GroundTruth
    provenance: str = ""


@dataclass
class OODValidationReport:
    """Validation outcome on a held-out cohort.

    Fields:

      n_total        : pairs evaluated
      per_class_n    : {label: count}
      confusion      : {(true_label, gate_verdict): count}
      far_per_class  : false-accept rate per non-contradict label
                       (we computed contradiction when truth was not)
      frr_per_class  : false-reject rate per contradict label
                       (we did NOT compute contradiction when truth was)
      ece            : expected calibration error on P(contra)
      abstain_rate   : fraction of pairs the gate refused to verdict
      review_rate    : fraction routed to review
      tau_used       : (tau_ent, tau_contra) the bundle had at validate time
    """

    n_total: int
    per_class_n: dict[str, int]
    confusion: dict[tuple[str, str], int]
    far_per_class: dict[str, float]
    frr_per_class: dict[str, float]
    ece: float
    abstain_rate: float
    review_rate: float
    tau_used: tuple[float, float]
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        rows = [f"n_total={self.n_total}",
                f"per_class_n={self.per_class_n}",
                f"abstain_rate={self.abstain_rate:.3f}",
                f"review_rate={self.review_rate:.3f}",
                f"ece={self.ece:.3f}",
                f"tau=({self.tau_used[0]:.3f}, {self.tau_used[1]:.3f})"]
        if self.far_per_class:
            rows.append("far_per_class=" + ", ".join(
                f"{k}:{v:.3f}" for k, v in self.far_per_class.items()))
        if self.frr_per_class:
            rows.append("frr_per_class=" + ", ".join(
                f"{k}:{v:.3f}" for k, v in self.frr_per_class.items()))
        return "OODValidationReport(\n  " + "\n  ".join(rows) + "\n)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LABEL_NAMES = {0: "agree", 1: "unrelated", 2: "contradict"}


def _isotonic_apply(
    iso_x: np.ndarray, iso_y: np.ndarray, te: float,
) -> float:
    if te <= iso_x[0]:
        return float(iso_y[0])
    if te >= iso_x[-1]:
        return float(iso_y[-1])
    idx = int(np.searchsorted(iso_x, te))
    x0, x1 = iso_x[idx - 1], iso_x[idx]
    y0, y1 = iso_y[idx - 1], iso_y[idx]
    if x1 == x0:
        return float(y0)
    w = (te - x0) / (x1 - x0)
    return float(y0 + w * (y1 - y0))


def _expected_calibration_error(
    p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10,
) -> float:
    """Standard ECE on a binary outcome."""
    if len(p_pred) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_pred)
    for i in range(n_bins):
        mask = (p_pred >= edges[i]) & (p_pred < edges[i + 1])
        if i == n_bins - 1:
            mask = (p_pred >= edges[i]) & (p_pred <= edges[i + 1])
        if not mask.any():
            continue
        bucket_p = p_pred[mask].mean()
        bucket_y = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bucket_p - bucket_y)
    return float(ece)


# ---------------------------------------------------------------------------
# Mode 1: calibration-only
# ---------------------------------------------------------------------------


def validate_calibration(
    bundle: "GovernanceBundle",
    samples: list["CalibrationSample"],
) -> OODValidationReport:
    """Run the bundle's isotonic over labeled samples.

    Tests *calibration only* — no store, no opposition table, no
    policy. Reports how well the bundle's frozen isotonic mapping
    discriminates contradict from non-contradict on this cohort.
    """
    if not samples:
        raise ValueError("Empty cohort")

    iso_x = bundle.isotonic_x
    iso_y = bundle.isotonic_y
    tau_ent = bundle.tau_ent.mean
    tau_contra = bundle.tau_contra.mean

    p_arr = np.array([
        _isotonic_apply(iso_x, iso_y, float(s.value)) for s in samples
    ])
    y_arr = np.array([1 if s.label == 2 else 0 for s in samples])
    te_arr = np.array([float(s.value) for s in samples])

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    per_class_n: dict[str, int] = Counter()
    for s, te in zip(samples, te_arr):
        true = _LABEL_NAMES[s.label]
        per_class_n[true] += 1
        if te <= tau_ent:
            verdict = "agreement"
        elif te >= tau_contra:
            verdict = "contradiction"
        else:
            verdict = "irrelevance"
        confusion[(true, verdict)] += 1

    # FAR per non-contradict class: how often was it labeled contradiction?
    far = {}
    for label in ("agree", "unrelated"):
        n = per_class_n.get(label, 0)
        if n == 0:
            continue
        false_accepts = confusion.get((label, "contradiction"), 0)
        far[label] = false_accepts / n

    # FRR for contradict class: how often did we miss it?
    frr = {}
    n_contra = per_class_n.get("contradict", 0)
    if n_contra > 0:
        misses = (
            confusion.get(("contradict", "agreement"), 0)
            + confusion.get(("contradict", "irrelevance"), 0)
        )
        frr["contradict"] = misses / n_contra

    ece = _expected_calibration_error(p_arr, y_arr)

    return OODValidationReport(
        n_total=len(samples),
        per_class_n=dict(per_class_n),
        confusion=dict(confusion),
        far_per_class=far,
        frr_per_class=frr,
        ece=ece,
        abstain_rate=0.0,
        review_rate=0.0,
        tau_used=(tau_ent, tau_contra),
        extra={
            "mean_te_per_class_agree": float(
                te_arr[y_arr == 0].mean()) if (y_arr == 0).any() else 0.0,
            "mean_te_per_class_contra": float(
                te_arr[y_arr == 1].mean()) if (y_arr == 1).any() else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Mode 2: end-to-end gate
# ---------------------------------------------------------------------------


def validate_gate(
    gate: "RuntimeGate",
    pairs: Iterable[LabeledPair],
) -> OODValidationReport:
    """Run the full RuntimeGate over labeled entity pairs.

    Tests *end-to-end deployment behavior*: opposition shortcut,
    support gating, geometry, isotonic, margin abstain, two-attestor
    concurrence, all included.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Empty pair cohort")

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    per_class_n: dict[str, int] = Counter()
    p_collect: list[float] = []
    y_collect: list[int] = []
    n_abstain = 0
    n_review = 0

    for lp in pairs:
        per_class_n[lp.label] += 1
        v = gate.verdict(lp.a, lp.b)
        confusion[(lp.label, v.verdict)] += 1
        if v.verdict == "abstain":
            n_abstain += 1
        elif v.verdict == "review":
            n_review += 1
        if v.p_contradiction is not None:
            p_collect.append(v.p_contradiction)
            y_collect.append(1 if lp.label == "contradict" else 0)

    far = {}
    for label in ("agree", "unrelated"):
        n = per_class_n.get(label, 0)
        if n == 0:
            continue
        false_accepts = confusion.get((label, "contradiction"), 0)
        far[label] = false_accepts / n

    frr = {}
    n_contra = per_class_n.get("contradict", 0)
    if n_contra > 0:
        misses = (
            confusion.get(("contradict", "agreement"), 0)
            + confusion.get(("contradict", "irrelevance"), 0)
        )
        frr["contradict"] = misses / n_contra

    ece = (_expected_calibration_error(np.array(p_collect), np.array(y_collect))
           if p_collect else 0.0)

    return OODValidationReport(
        n_total=len(pairs),
        per_class_n=dict(per_class_n),
        confusion=dict(confusion),
        far_per_class=far,
        frr_per_class=frr,
        ece=ece,
        abstain_rate=n_abstain / len(pairs),
        review_rate=n_review / len(pairs),
        tau_used=(gate.bundle.tau_ent.mean, gate.bundle.tau_contra.mean),
    )


def shift_samples(
    samples: list["CalibrationSample"],
    *,
    delta_per_class: dict[int, float] | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> list["CalibrationSample"]:
    """Apply a synthetic distribution shift to a labeled cohort.

    Use to simulate an OOD condition for testing: shift the agree
    class up by 0.3 (training collapse), shift the contradict class
    down by 0.3 (degraded discrimination), etc. Useful for
    regression-testing the validator and for illustrating what
    drift looks like in metrics.
    """
    from knowlytix.harness.testing.primitive_calibration import CalibrationSample

    rng = np.random.default_rng(seed)
    delta_per_class = delta_per_class or {}
    out = []
    for s in samples:
        delta = delta_per_class.get(s.label, 0.0)
        eps = rng.normal(0.0, noise) if noise > 0 else 0.0
        out.append(CalibrationSample(
            value=float(s.value + delta + eps),
            label=s.label,
            label_name=s.label_name,
            provenance=f"{s.provenance}|shift_d{delta:+.2f}_n{noise:.2f}",
        ))
    return out
