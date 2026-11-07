# SPDX-License-Identifier: Apache-2.0
"""Naive baselines for policy enforcement.

A high-stakes deployment claim of the form "the four-layer engine
catches X% more violations than prompt guardrails" requires a
side-by-side comparison on the same labeled cohort. This module
formalises the two natural baselines:

  * :class:`PromptGuardrailBaseline` -- regex/string matching over
    a written-out instruction list, the simplified analogue of an
    LLM-system-prompt guardrail.  This is the *best case* for
    prompt-style enforcement (a real LLM would make probabilistic
    misreads on top of the same structural limits).
  * :class:`JaccardPolicyBaseline` -- scores how many YAML rules'
    action substrings overlap the proposed action; treats any
    overlap as "rule applies, treat as BLOCK".  Models a deployment
    that derives policy from documentation alone, without semantic
    understanding.

Both produce :class:`PolicyDecision`-shaped results so they can be
fed through the same OOD validation report as the calibrated engine.
The contrast report (``compare_to_engine``) reports per-class uplift.

Lab 12.2 used the prompt baseline informally; this module promotes
it to an audit-defensible artefact that the CI ship-gate consumes.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.policy_validation import LabeledDecision
    from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule


# ---------------------------------------------------------------------------
# Baseline interface
# ---------------------------------------------------------------------------


@dataclass
class BaselineDecision:
    """Lightweight stand-in for PolicyDecision used by baselines."""

    action: str
    allowed: bool
    blocking_rules: list[str] = field(default_factory=list)
    reason: str = ""


class _Baseline:
    """Common API: ``check(action, args, context)`` ->
    :class:`BaselineDecision`."""

    name: str = "baseline"

    def check(self, action: str, args: dict, context: str
              ) -> BaselineDecision:  # pragma: no cover -- abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Prompt guardrail baseline
# ---------------------------------------------------------------------------


class PromptGuardrailBaseline(_Baseline):
    """Regex/keyword guardrails as written into an LLM system prompt.

    Models the best case for advisory text instructions. Cannot
    handle:

      * state-tracking rules (RML-004 model count, RML-007 budget)
      * system-config flags (RML-009 tracing)
      * wildcard coverage of unknown tools (RML-007/009 patterns)
    """

    name = "prompt_guardrail"

    # Exact-match rules the prompt can express.
    PROMPT_PAIRS: tuple[tuple[str, str], ...] = (
        ("train_model",      "Should preprocess before training"),
        ("deploy_model",     "Should check fairness before deploy"),
        ("delete_raw_data",  "Never delete raw data"),
    )

    def check(self, action, args, context):
        violations: list[str] = []
        # Rule 1: preprocess before training
        if action == "train_model":
            violations.append(
                "Should preprocess before training (advisory)"
            )
        # Rule 2: fairness before deploy
        if action == "deploy_model":
            violations.append(
                "Should check fairness before deploying (advisory)"
            )
            if "high_risk" in (context or ""):
                violations.append(
                    "Should check robustness in high-risk context"
                )
        # Rule 3: substring on context for test data
        if action == "access_test_data" and "training" in (context or ""):
            violations.append("No test data during training (advisory)")
        # Rule 8: never delete raw data
        if action == "delete_raw_data":
            violations.append("Never delete raw data (advisory)")
        # Rule 10: human approval for deploy
        if action == "deploy_model":
            violations.append("Should get human approval (advisory)")
        # Rule for ADULT-005 (drop_feature on adult_income)
        if (action == "drop_feature"
                and "adult_income" in (context or "")):
            violations.append(
                "Should not drop protected attributes (advisory)"
            )

        allowed = not violations
        return BaselineDecision(
            action=action,
            allowed=allowed,
            blocking_rules=(
                ["prompt"] if violations else []
            ),
            reason="; ".join(violations) if violations else "",
        )


# ---------------------------------------------------------------------------
# Jaccard rule-ID baseline
# ---------------------------------------------------------------------------


class JaccardPolicyBaseline(_Baseline):
    """Token-overlap heuristic over the YAML rules.

    For each rule, computes Jaccard similarity between the proposed
    action's tokens and the rule's action tokens (split on
    underscore). If Jaccard > ``threshold``, the rule "applies"
    and the action is BLOCK. Models a deployment that infers
    enforcement from documented rule names without semantic
    understanding.
    """

    name = "jaccard"

    def __init__(
        self, rules: list["PolicyRule"], *, threshold: float = 0.5,
    ):
        self._rules = rules
        self._threshold = threshold

    @staticmethod
    def _tokens(s: str) -> set[str]:
        return set(s.split("_"))

    def check(self, action, args, context):
        action_tokens = self._tokens(action)
        if not action_tokens:
            return BaselineDecision(
                action=action, allowed=True, blocking_rules=[],
                reason="empty action",
            )

        matched: list[str] = []
        for r in self._rules:
            # Wildcard rules should always be considered (RML-007/009).
            if r.action == "*":
                matched.append(r.rule_id)
                continue
            rule_tokens = self._tokens(r.action)
            if not rule_tokens:
                continue
            overlap = len(action_tokens & rule_tokens)
            denom = len(action_tokens | rule_tokens)
            j = overlap / denom if denom else 0.0
            if j >= self._threshold:
                matched.append(r.rule_id)

        # Context fnmatch (also done by the real engine, but Jaccard
        # baseline keeps it strict).
        applicable = []
        for rule_id in matched:
            r = next((rr for rr in self._rules if rr.rule_id == rule_id),
                     None)
            if r is None:
                continue
            ctx = context or ""
            if r.context == "*" or fnmatch.fnmatch(ctx, r.context):
                applicable.append(rule_id)

        allowed = not applicable
        return BaselineDecision(
            action=action, allowed=allowed,
            blocking_rules=applicable,
            reason=(
                f"jaccard match: {applicable}" if applicable else
                "no rule matched by token overlap"
            ),
        )


# ---------------------------------------------------------------------------
# Contrast report
# ---------------------------------------------------------------------------


@dataclass
class BaselineContrastReport:
    """Per-class engine-vs-baseline contrast on a labeled cohort."""

    n_total: int
    engine_correct: int
    baseline_correct: int
    uplift: int                          # engine_correct - baseline_correct
    per_class_engine: dict[str, int] = field(default_factory=dict)
    per_class_baseline: dict[str, int] = field(default_factory=dict)
    per_class_n: dict[str, int] = field(default_factory=dict)

    @property
    def engine_accuracy(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.engine_correct / self.n_total

    @property
    def baseline_accuracy(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.baseline_correct / self.n_total

    def summary(self) -> str:
        rows = [
            f"n_total={self.n_total}",
            f"engine_correct={self.engine_correct} "
            f"({self.engine_accuracy:.1%})",
            f"baseline_correct={self.baseline_correct} "
            f"({self.baseline_accuracy:.1%})",
            f"uplift={self.uplift:+d} pairs",
        ]
        for cls, n in sorted(self.per_class_n.items()):
            e = self.per_class_engine.get(cls, 0)
            b = self.per_class_baseline.get(cls, 0)
            rows.append(
                f"  {cls}: engine {e}/{n} | baseline {b}/{n}"
            )
        return ("BaselineContrastReport(\n  " + "\n  ".join(rows) + "\n)")


def _maps_to_block(allowed: bool) -> str:
    return "ALLOW" if allowed else "BLOCK"


def compare_to_engine(
    engine: "PolicyEngine",
    baseline: _Baseline,
    cohort: list["LabeledDecision"],
    *,
    quorum_required: int = 2,
) -> BaselineContrastReport:
    """Run engine and baseline on the same cohort; report uplift.

    Both produce ALLOW/BLOCK; engine quorum produces REVIEW which we
    score as ``BLOCK`` for accuracy purposes since a deployment that
    routes to review is *not* allowing the call.
    """
    from collections import Counter

    from knowlytix.harness.governance.policy_calibration import check_with_quorum

    per_n: Counter[str] = Counter()
    per_engine: Counter[str] = Counter()
    per_baseline: Counter[str] = Counter()
    e_correct = 0
    b_correct = 0

    for ld in cohort:
        per_n[ld.source_class] += 1
        # Engine
        _, verdict = check_with_quorum(
            engine, ld.action, ld.args, ld.context,
            quorum_required=quorum_required,
        )
        if verdict.decision == ld.expected_decision:
            # Exact match (BLOCK/ALLOW/REVIEW)
            e_correct += 1
            per_engine[ld.source_class] += 1
        elif (verdict.decision == "REVIEW"
              and ld.expected_decision == "BLOCK"):
            # REVIEW is conservative when expected was BLOCK — the
            # engine refuses-pending-operator, which still does not
            # auto-allow. Count as correct.
            e_correct += 1
            per_engine[ld.source_class] += 1

        # Baseline (BLOCK or ALLOW only). Maps:
        #   expected REVIEW -> baseline correct iff baseline blocks
        #     (REVIEW is closer to BLOCK than ALLOW — both refuse to
        #     auto-allow, but baseline cannot express the REVIEW
        #     semantics so this is the most charitable scoring).
        bd = baseline.check(ld.action, ld.args, ld.context)
        baseline_decision = _maps_to_block(bd.allowed)
        if baseline_decision == ld.expected_decision:
            b_correct += 1
            per_baseline[ld.source_class] += 1
        elif (baseline_decision == "BLOCK"
              and ld.expected_decision == "REVIEW"):
            b_correct += 1
            per_baseline[ld.source_class] += 1

    return BaselineContrastReport(
        n_total=len(cohort),
        engine_correct=e_correct,
        baseline_correct=b_correct,
        uplift=e_correct - b_correct,
        per_class_engine=dict(per_engine),
        per_class_baseline=dict(per_baseline),
        per_class_n=dict(per_n),
    )
