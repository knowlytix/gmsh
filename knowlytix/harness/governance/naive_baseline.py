# SPDX-License-Identifier: Apache-2.0
"""Naive baselines for contrast with the geometric verdict gate.

For high-stakes deployment, "the model said so" is not a defensible
answer to "why is this verdict trustworthy?"  Part of the answer is:
*because a simpler approach would have been wrong here, and the
geometry got it right.*  This module provides two simple baselines
and a contrast report so that uplift over them is auditable.

Baselines:

  ``NaiveOppositionBaseline``   — uses only the bundle's
                                  declared-opposition table; every
                                  non-declared pair abstains.
  ``JaccardBaseline``           — pure structural similarity over
                                  incident triples.  Above
                                  ``theta_agree`` → agreement; below
                                  ``theta_unrelated`` → unrelated;
                                  in between → abstain.

Neither uses learned embeddings.  Both are fast (graph lookup only)
and deterministic.  The ``compare_to_gate`` driver runs both the
gate and a baseline over a labeled cohort and reports per-class
accuracy and the uplift the geometry provides.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from knowlytix.harness.governance.governance_bundle import GovernanceBundle
    from knowlytix.harness.governance.ood_validation import LabeledPair
    from knowlytix.harness.governance.runtime_gate import RuntimeGate


BaselineVerdict = Literal[
    "agreement", "irrelevance", "contradiction", "abstain",
]

_LABEL_TO_VERDICT = {
    "agree": "agreement",
    "unrelated": "irrelevance",
    "contradict": "contradiction",
}


class Baseline(Protocol):
    """Minimal interface for a contrast baseline."""

    name: str

    def predict(self, a: str, b: str) -> BaselineVerdict: ...


@dataclass
class NaiveOppositionBaseline:
    """Verdict from the opposition table only.

    Declared opposition → contradiction; everything else abstains.
    Useful as a *floor*: the gate must do at least this well.
    """

    bundle: "GovernanceBundle"
    name: str = "naive_opposition"

    def predict(self, a: str, b: str) -> BaselineVerdict:
        if self.bundle.is_declared_opposition(a, b):
            return "contradiction"
        return "abstain"


@dataclass
class JaccardBaseline:
    """Jaccard similarity over incident triples.

    For each entity, build the set of incident ``(relation, target)``
    pairs (heads only — symmetric handling could be added).  Jaccard
    similarity is ``|A ∩ B| / |A ∪ B|``.  Bands:

      sim >= theta_agree     → agreement
      sim <= theta_unrelated → irrelevance
      otherwise              → abstain
    """

    triples: list
    theta_agree: float = 0.4
    theta_unrelated: float = 0.05
    name: str = "jaccard"

    def __post_init__(self):
        idx: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for h, r, t in self.triples:
            if r in {"opposite_of", "is_functional"}:
                continue
            idx[h].add((r, t))
        self._idx = dict(idx)

    @classmethod
    def from_store(cls, store, **kwargs) -> "JaccardBaseline":
        triples = (list(store.doc_graph.triples)
                   if store.doc_graph is not None else [])
        return cls(triples=triples, **kwargs)

    def predict(self, a: str, b: str) -> BaselineVerdict:
        a_set = self._idx.get(a, set())
        b_set = self._idx.get(b, set())
        if not a_set or not b_set:
            return "abstain"
        inter = a_set & b_set
        union = a_set | b_set
        sim = len(inter) / len(union) if union else 0.0
        if sim >= self.theta_agree:
            return "agreement"
        if sim <= self.theta_unrelated:
            return "irrelevance"
        return "abstain"


@dataclass
class BaselineContrastReport:
    """Side-by-side accuracy and uplift comparison."""

    cohort_size: int
    baseline_name: str
    gate_correct: int
    baseline_correct: int
    gate_abstain: int
    baseline_abstain: int
    gate_accuracy_on_decided: float
    baseline_accuracy_on_decided: float
    uplift_correct_pairs: int
    examples_gate_correct_baseline_wrong: list[tuple[str, str, str, str]] = (
        field(default_factory=list))
    examples_baseline_correct_gate_wrong: list[tuple[str, str, str, str]] = (
        field(default_factory=list))

    @property
    def uplift_accuracy(self) -> float:
        return self.gate_accuracy_on_decided - self.baseline_accuracy_on_decided

    def summary(self) -> str:
        return (
            f"BaselineContrast({self.baseline_name}, n={self.cohort_size})\n"
            f"  gate     : {self.gate_correct}/"
            f"{self.cohort_size - self.gate_abstain} decided correct, "
            f"acc={self.gate_accuracy_on_decided:.3f}, "
            f"abstain={self.gate_abstain}\n"
            f"  baseline : {self.baseline_correct}/"
            f"{self.cohort_size - self.baseline_abstain} decided correct, "
            f"acc={self.baseline_accuracy_on_decided:.3f}, "
            f"abstain={self.baseline_abstain}\n"
            f"  uplift   : +{self.uplift_correct_pairs} correct pairs, "
            f"+{self.uplift_accuracy:.3f} accuracy"
        )


def _correct(verdict: BaselineVerdict, label: str) -> bool:
    expected = _LABEL_TO_VERDICT.get(label)
    return expected is not None and verdict == expected


def compare_to_gate(
    gate: "RuntimeGate",
    baseline: Baseline,
    cohort: "list[LabeledPair]",
    *,
    keep_examples: int = 5,
) -> BaselineContrastReport:
    """Run gate and baseline over the same cohort; report uplift."""
    gate_correct = baseline_correct = 0
    gate_abstain = baseline_abstain = 0
    gate_correct_baseline_wrong: list = []
    baseline_correct_gate_wrong: list = []

    for lp in cohort:
        gv = gate.verdict(lp.a, lp.b)
        bv = baseline.predict(lp.a, lp.b)
        g_ok = _correct(gv.verdict, lp.label)
        b_ok = _correct(bv, lp.label)
        if gv.verdict == "abstain":
            gate_abstain += 1
        elif g_ok:
            gate_correct += 1
        if bv == "abstain":
            baseline_abstain += 1
        elif b_ok:
            baseline_correct += 1
        if g_ok and not b_ok and len(gate_correct_baseline_wrong) < keep_examples:
            gate_correct_baseline_wrong.append(
                (lp.a, lp.b, gv.verdict, bv))
        elif b_ok and not g_ok and len(baseline_correct_gate_wrong) < keep_examples:
            baseline_correct_gate_wrong.append(
                (lp.a, lp.b, gv.verdict, bv))

    g_decided = len(cohort) - gate_abstain
    b_decided = len(cohort) - baseline_abstain
    g_acc = gate_correct / g_decided if g_decided else 0.0
    b_acc = baseline_correct / b_decided if b_decided else 0.0

    return BaselineContrastReport(
        cohort_size=len(cohort),
        baseline_name=baseline.name,
        gate_correct=gate_correct,
        baseline_correct=baseline_correct,
        gate_abstain=gate_abstain,
        baseline_abstain=baseline_abstain,
        gate_accuracy_on_decided=g_acc,
        baseline_accuracy_on_decided=b_acc,
        uplift_correct_pairs=gate_correct - baseline_correct,
        examples_gate_correct_baseline_wrong=gate_correct_baseline_wrong,
        examples_baseline_correct_gate_wrong=baseline_correct_gate_wrong,
    )
