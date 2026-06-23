# SPDX-License-Identifier: Apache-2.0
"""Naive baseline verifiers for orchestrator uplift comparison.

The calibrated ``ClaimVerifier`` produces 44/44 PASS on the
canonical Ch 10 plan.  An auditor's reasonable next question is
"is that better than something simpler?"  This module provides two
baseline verifiers that the orchestrator can run against the same
labeled cohort the calibrated verifier sees, and a comparison
helper that reports the uplift the calibrated path provides.

Baselines:

  ``JaccardClaimVerifier`` — fact claims pass iff Jaccard similarity
                             of token sets against any corpus triple
                             exceeds a threshold; numeric claims
                             pass iff the value is finite and
                             non-negative.  Pure structural lookup,
                             no embeddings.
  ``RandomClaimVerifier``  — coin flip per claim with a fixed seed.
                             Establishes the floor.

Both implement the duck-typed subset of ``ClaimVerifier`` that the
orchestrator uses (``verify_all``), so they drop into the same
interface without modifying the orchestrator.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowlytix.harness.governance.verifier import (
    Claim, ClaimType, ClaimVerdict, OverallVerdict,
)

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.governance.orchestrator_validation import LabeledPlan
    from knowlytix.harness.governance.orchestrator import GovernedOrchestrator


# ---------------------------------------------------------------------------
# Jaccard-on-triples baseline
# ---------------------------------------------------------------------------


def _tokenize(s: str) -> set[str]:
    return set(t.lower() for t in s.replace("_", " ").split() if t)


@dataclass
class JaccardClaimVerifier:
    """Surface-level baseline: Jaccard token similarity, no embeddings.

    A fact claim ``(h, r, t)`` passes iff some corpus triple
    ``(h', r', t')`` has Jaccard similarity above ``theta`` between
    the concatenated token sets.  Numeric claims pass iff the value
    is finite and non-negative (a deliberately weak surrogate for
    the calibrated ENM check).
    """

    store: "GMSExpertStore"
    theta: float = 0.6
    name: str = "jaccard_baseline"

    def __post_init__(self):
        triples = (list(self.store.doc_graph.triples)
                   if self.store.doc_graph is not None else [])
        self._triple_token_sets = [
            (h, r, t, _tokenize(f"{h} {r} {t}"))
            for h, r, t in triples
        ]

    def verify_claim(self, claim: Claim) -> ClaimVerdict:
        if claim.type == ClaimType.FACT:
            h = claim.metadata.get("head", "")
            r = claim.metadata.get("relation", "")
            t = claim.metadata.get("tail", "")
            target = _tokenize(f"{h} {r} {t}")
            if not target:
                return ClaimVerdict(
                    claim=claim, passed=False, score=0.0,
                    severity="warning",
                    evidence="empty token set",
                    gms_method="jaccard",
                )
            best_sim = 0.0
            for _, _, _, ts in self._triple_token_sets:
                if not ts:
                    continue
                inter = target & ts
                union = target | ts
                sim = len(inter) / len(union) if union else 0.0
                if sim > best_sim:
                    best_sim = sim
            passed = best_sim >= self.theta
            return ClaimVerdict(
                claim=claim, passed=passed,
                score=best_sim,
                severity="info" if passed else "error",
                evidence=(
                    f"jaccard_max={best_sim:.3f} "
                    f"{'>=' if passed else '<'} theta={self.theta}"
                ),
                gms_method="jaccard",
            )
        if claim.type == ClaimType.NUMERIC:
            v = claim.metadata.get("claimed_value")
            try:
                f = float(v) if v is not None else None
            except (TypeError, ValueError):
                f = None
            passed = f is not None and f >= 0.0
            return ClaimVerdict(
                claim=claim, passed=passed,
                score=1.0 if passed else 0.0,
                severity="info" if passed else "warning",
                evidence=(
                    f"value={v} {'finite, >= 0' if passed else 'not finite or < 0'}"
                ),
                gms_method="value_finite",
            )
        # Other claim types: default PASS (deliberately weak baseline)
        return ClaimVerdict(
            claim=claim, passed=True, score=1.0, severity="info",
            evidence=f"baseline default-PASS for {claim.type}",
            gms_method="default_pass",
        )

    def verify_all(self, claims: list[Claim]) -> OverallVerdict:
        verdicts = [self.verify_claim(c) for c in claims]
        n_passed = sum(1 for v in verdicts if v.passed)
        n_total = len(verdicts)
        return OverallVerdict(
            claim_verdicts=verdicts,
            n_passed=n_passed,
            n_total=n_total,
            overall_status="PASS" if n_passed == n_total else "FAIL",
            pass_rate=n_passed / n_total if n_total else 1.0,
            severity_distribution=dict(
                Counter(v.severity for v in verdicts)
            ),
        )


# ---------------------------------------------------------------------------
# Random coin-flip baseline
# ---------------------------------------------------------------------------


@dataclass
class RandomClaimVerifier:
    """Establishes the floor: each claim PASSes with ``p_pass``."""

    p_pass: float = 0.5
    seed: int = 42
    name: str = "random_baseline"

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def verify_claim(self, claim: Claim) -> ClaimVerdict:
        passed = self._rng.random() < self.p_pass
        return ClaimVerdict(
            claim=claim, passed=passed,
            score=1.0 if passed else 0.0,
            severity="info" if passed else "warning",
            evidence=f"random PASS with p={self.p_pass}",
            gms_method="random",
        )

    def verify_all(self, claims: list[Claim]) -> OverallVerdict:
        verdicts = [self.verify_claim(c) for c in claims]
        n_passed = sum(1 for v in verdicts if v.passed)
        n_total = len(verdicts)
        return OverallVerdict(
            claim_verdicts=verdicts,
            n_passed=n_passed,
            n_total=n_total,
            overall_status="PASS" if n_passed == n_total else "FAIL",
            pass_rate=n_passed / n_total if n_total else 1.0,
            severity_distribution=dict(
                Counter(v.severity for v in verdicts)
            ),
        )


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------


@dataclass
class BaselineComparisonReport:
    """Side-by-side per-cohort outcome of calibrated vs baseline."""

    cohort_size: int
    baseline_name: str
    calibrated_correct: int
    baseline_correct: int
    calibrated_state_match: int  # actual_state == expected_state
    baseline_state_match: int
    calibrated_pass_rate_avg: float
    baseline_pass_rate_avg: float
    examples_calibrated_correct_baseline_wrong: list = field(default_factory=list)
    examples_baseline_correct_calibrated_wrong: list = field(default_factory=list)

    @property
    def uplift_correct(self) -> int:
        return self.calibrated_correct - self.baseline_correct

    @property
    def uplift_state(self) -> int:
        return self.calibrated_state_match - self.baseline_state_match

    def summary(self) -> str:
        rows = [
            f"BaselineComparisonReport({self.baseline_name}, "
            f"n={self.cohort_size})",
            f"  calibrated  : {self.calibrated_correct}/{self.cohort_size} "
            f"correct, state_match={self.calibrated_state_match}/{self.cohort_size}, "
            f"avg_pass_rate={self.calibrated_pass_rate_avg:.3f}",
            f"  {self.baseline_name:11s} : {self.baseline_correct}/{self.cohort_size} "
            f"correct, state_match={self.baseline_state_match}/{self.cohort_size}, "
            f"avg_pass_rate={self.baseline_pass_rate_avg:.3f}",
            f"  uplift      : +{self.uplift_correct} correct pairs, "
            f"+{self.uplift_state} state matches",
        ]
        return "\n".join(rows)


def compare_to_baseline(
    orch_calibrated: "GovernedOrchestrator",
    orch_baseline: "GovernedOrchestrator",
    cohort: list["LabeledPlan"],
    *,
    keep_examples: int = 5,
    fresh_memory_factory=None,
) -> BaselineComparisonReport:
    """Run the same labeled cohort through two orchestrators and compare.

    ``orch_baseline`` is constructed exactly like ``orch_calibrated``
    except its verifier is one of the baseline classes above.  We
    measure two quantities:

      * **correctness** — the run reaches its expected terminal state
        AND, when COMPLETED, the verdict is PASS.
      * **state match** — the run's terminal state equals the labeled
        ``expected_state`` regardless of pass-rate.
    """
    calibrated_correct = 0
    baseline_correct = 0
    calibrated_state_match = 0
    baseline_state_match = 0
    calibrated_pass_rates: list[float] = []
    baseline_pass_rates: list[float] = []
    cal_correct_base_wrong: list = []
    base_correct_cal_wrong: list = []

    for lp in cohort:
        if fresh_memory_factory is not None:
            orch_calibrated.memory = fresh_memory_factory("cal")
            orch_baseline.memory = fresh_memory_factory("base")
        cr = orch_calibrated.run(lp.plan, lp.scenario)
        br = orch_baseline.run(lp.plan, lp.scenario)

        c_state_ok = cr.state.value == lp.expected_state
        b_state_ok = br.state.value == lp.expected_state
        c_pass = cr.verdict.pass_rate if cr.verdict else 0.0
        b_pass = br.verdict.pass_rate if br.verdict else 0.0
        calibrated_pass_rates.append(c_pass)
        baseline_pass_rates.append(b_pass)

        if c_state_ok:
            calibrated_state_match += 1
        if b_state_ok:
            baseline_state_match += 1

        c_correct = c_state_ok and (
            lp.expected_state != "completed"
            or (cr.verdict and cr.verdict.overall_status == "PASS")
        )
        b_correct = b_state_ok and (
            lp.expected_state != "completed"
            or (br.verdict and br.verdict.overall_status == "PASS")
        )
        if c_correct:
            calibrated_correct += 1
        if b_correct:
            baseline_correct += 1
        if c_correct and not b_correct and len(cal_correct_base_wrong) < keep_examples:
            cal_correct_base_wrong.append(
                (lp.scenario.name, lp.expected_state,
                 cr.state.value, br.state.value),
            )
        elif b_correct and not c_correct and len(base_correct_cal_wrong) < keep_examples:
            base_correct_cal_wrong.append(
                (lp.scenario.name, lp.expected_state,
                 cr.state.value, br.state.value),
            )

    n = len(cohort)
    return BaselineComparisonReport(
        cohort_size=n,
        baseline_name=getattr(orch_baseline.verifier, "name", "baseline"),
        calibrated_correct=calibrated_correct,
        baseline_correct=baseline_correct,
        calibrated_state_match=calibrated_state_match,
        baseline_state_match=baseline_state_match,
        calibrated_pass_rate_avg=(
            sum(calibrated_pass_rates) / n if n else 0.0),
        baseline_pass_rate_avg=(
            sum(baseline_pass_rates) / n if n else 0.0),
        examples_calibrated_correct_baseline_wrong=cal_correct_base_wrong,
        examples_baseline_correct_calibrated_wrong=base_correct_cal_wrong,
    )
