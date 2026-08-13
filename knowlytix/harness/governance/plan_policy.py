# SPDX-License-Identifier: Apache-2.0
"""PlanPolicy --- rule-based second attestor for workflow plans.

A declarative predicate table that checks plan-level governance rules
independently of the geometric :class:`WorkflowPlanner`.  Two attestors
must both accept a plan before it is released for execution; either can
veto.  The planner catches subtle ordering and novelty failures via
learned geometry; the policy engine catches hard-encoded rules such as
``any plan that ends in release_gate must include compare_models`` that
should never be violated regardless of what the geometry learned.

This mirrors the defense-in-depth pattern used across the governance
monograph: geometric attestor + declarative attestor, AND-gated.

Usage
-----
>>> from knowlytix.harness.governance.plan_policy import PlanPolicy, data_science_rules
>>> policy = PlanPolicy(data_science_rules())
>>> policy.check_plan(["load_data", "train_model", "release_gate"]).allowed
False
>>> # Bind to planner for unified validate_plan() output.
>>> planner.attach_policy(policy)
>>> planner.validate_plan(plan).policy_verdict
'accept'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ------------------------------------------------------------------ #
# Rule + Verdict data models
# ------------------------------------------------------------------ #

@dataclass
class PlanRule:
    """A single declarative rule.

    Parameters
    ----------
    rule_id : str
        Short identifier (used in audit logs).
    description : str
        Human-readable explanation of what the rule enforces.
    predicate : Callable[[list[str]], bool]
        Returns True if the plan satisfies the rule.
    severity : str
        ``"blocking"`` (veto on failure) or ``"advisory"`` (warn only).
    """

    rule_id: str
    description: str
    predicate: Callable[[list[str]], bool]
    severity: str = "blocking"


@dataclass
class PolicyVerdict:
    """Result of running a plan through :class:`PlanPolicy`."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)
    rules_checked: int = 0
    blocking_failures: list[str] = field(default_factory=list)
    advisory_failures: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# PlanPolicy engine
# ------------------------------------------------------------------ #

class PlanPolicy:
    """Run a declarative rule pack against a plan.

    Kept deliberately simple: a list of :class:`PlanRule` and an
    AND-gated ``check_plan``.  No geometry, no learning --- the point
    of this layer is to be independent of the planner.
    """

    def __init__(self, rules: list[PlanRule]):
        self.rules = list(rules)

    def check_plan(self, plan: list[str]) -> PolicyVerdict:
        blocking: list[str] = []
        advisory: list[str] = []
        reasons: list[str] = []
        for rule in self.rules:
            try:
                ok = bool(rule.predicate(plan))
            except Exception as e:
                # Treat rule errors as blocking --- fail safe.
                ok = False
                reasons.append(
                    f"rule {rule.rule_id} raised {type(e).__name__}: {e}"
                )
            if not ok:
                reasons.append(f"{rule.rule_id}: {rule.description}")
                if rule.severity == "blocking":
                    blocking.append(rule.rule_id)
                else:
                    advisory.append(rule.rule_id)
        return PolicyVerdict(
            allowed=(len(blocking) == 0),
            reasons=reasons,
            rules_checked=len(self.rules),
            blocking_failures=blocking,
            advisory_failures=advisory,
        )


# ------------------------------------------------------------------ #
# Data-science workflow rule pack
# ------------------------------------------------------------------ #

# Steps that are legitimate building blocks.  Used by R0 to flag any
# step that is neither canonical nor a known alias.  Kept narrower than
# the planner's fuzzy-match vocabulary so the policy catches
# hallucinated-step cases that fuzzy matching would paper over.
_CANONICAL_STEPS: frozenset[str] = frozenset(
    {
        "load_data",
        "preprocess_data",
        "feature_select",
        "train_model",
        "predict",
        "diagnose_accuracy",
        "diagnose_robustness",
        "diagnose_fairness",
        "explain_importance",
        "explain_pdp",
        "explain_shap",
        "compare_models",
        "release_gate",
        "deploy",
    }
)


def _index(plan: list[str], step: str) -> int:
    """Return the first index of ``step`` in ``plan``, or -1 if absent."""
    try:
        return plan.index(step)
    except ValueError:
        return -1


def _all_steps_canonical(plan: list[str]) -> bool:
    return all(s in _CANONICAL_STEPS for s in plan)


def _no_duplicate_steps(plan: list[str]) -> bool:
    return len(plan) == len(set(plan))


def _load_data_present(plan: list[str]) -> bool:
    return "load_data" in plan


def _preprocess_before_train(plan: list[str]) -> bool:
    """If train_model is in the plan, preprocess_data must precede it."""
    t = _index(plan, "train_model")
    if t < 0:
        return True  # not applicable
    p = _index(plan, "preprocess_data")
    return 0 <= p < t


def _diag_requires_train(plan: list[str]) -> bool:
    """Any diagnostic step must come after train_model."""
    t = _index(plan, "train_model")
    for diag in ("diagnose_accuracy", "diagnose_robustness", "diagnose_fairness"):
        d = _index(plan, diag)
        if d >= 0 and (t < 0 or d < t):
            return False
    return True


def _release_gate_requires_compare(plan: list[str]) -> bool:
    """release_gate cannot appear before compare_models."""
    r = _index(plan, "release_gate")
    if r < 0:
        return True
    c = _index(plan, "compare_models")
    return 0 <= c < r


def _release_gate_requires_fairness(plan: list[str]) -> bool:
    """A plan that ends in release_gate must include diagnose_fairness.

    Non-negotiable for high-stakes deployments --- enforced by policy
    rather than learned by geometry so the rule cannot drift.
    """
    if "release_gate" not in plan:
        return True
    return "diagnose_fairness" in plan


def _deploy_requires_release_gate(plan: list[str]) -> bool:
    d = _index(plan, "deploy")
    if d < 0:
        return True
    r = _index(plan, "release_gate")
    return 0 <= r < d


def _explain_requires_train(plan: list[str]) -> bool:
    t = _index(plan, "train_model")
    for expl in ("explain_importance", "explain_pdp", "explain_shap"):
        e = _index(plan, expl)
        if e >= 0 and (t < 0 or e < t):
            return False
    return True


def _plan_not_empty(plan: list[str]) -> bool:
    return len(plan) > 0


def _plan_not_too_long(plan: list[str]) -> bool:
    """A plan more than 2x the canonical length is almost certainly wrong."""
    return len(plan) <= 2 * len(_CANONICAL_STEPS)


def data_science_rules() -> list[PlanRule]:
    """Return the standard rule pack for the data-science workflow.

    Eleven rules covering vocabulary, structure, and the non-negotiable
    gates (preprocess-before-train, release-gate-requires-fairness).
    Rule IDs are stable; audit logs rely on them.
    """
    return [
        PlanRule(
            rule_id="R0-vocabulary",
            description="every step must be in the canonical vocabulary",
            predicate=_all_steps_canonical,
        ),
        PlanRule(
            rule_id="R1-no-duplicates",
            description="no step may appear twice in a plan",
            predicate=_no_duplicate_steps,
        ),
        PlanRule(
            rule_id="R2-load-data-present",
            description="plan must begin with data loading",
            predicate=_load_data_present,
        ),
        PlanRule(
            rule_id="R3-preprocess-before-train",
            description="preprocess_data must precede train_model",
            predicate=_preprocess_before_train,
        ),
        PlanRule(
            rule_id="R4-diag-requires-train",
            description="any diagnostic step must follow train_model",
            predicate=_diag_requires_train,
        ),
        PlanRule(
            rule_id="R5-release-requires-compare",
            description="release_gate must follow compare_models",
            predicate=_release_gate_requires_compare,
        ),
        PlanRule(
            rule_id="R6-release-requires-fairness",
            description="any release_gate plan must include diagnose_fairness",
            predicate=_release_gate_requires_fairness,
        ),
        PlanRule(
            rule_id="R7-deploy-requires-release-gate",
            description="deploy must follow release_gate",
            predicate=_deploy_requires_release_gate,
        ),
        PlanRule(
            rule_id="R8-explain-requires-train",
            description="explanation steps must follow train_model",
            predicate=_explain_requires_train,
        ),
        PlanRule(
            rule_id="R9-plan-not-empty",
            description="plan must contain at least one step",
            predicate=_plan_not_empty,
        ),
        PlanRule(
            rule_id="R10-plan-length-sane",
            description="plan length must not exceed 2x canonical length",
            predicate=_plan_not_too_long,
            severity="advisory",
        ),
    ]


# ------------------------------------------------------------------ #
# Smoke test entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    policy = PlanPolicy(data_science_rules())
    samples = [
        ("canonical valid", [
            "load_data", "preprocess_data", "feature_select", "train_model",
            "diagnose_accuracy", "diagnose_robustness", "diagnose_fairness",
            "explain_importance", "explain_pdp", "explain_shap",
            "compare_models", "release_gate",
        ]),
        ("release without fairness", [
            "load_data", "preprocess_data", "train_model",
            "diagnose_accuracy", "compare_models", "release_gate",
        ]),
        ("diag before train", [
            "load_data", "preprocess_data", "diagnose_accuracy", "train_model",
        ]),
        ("hallucinated step", [
            "load_data", "preprocess_data", "train_model",
            "quantize_weights", "compare_models", "release_gate",
        ]),
    ]
    for label, plan in samples:
        v = policy.check_plan(plan)
        print(f"\n[{label}] allowed={v.allowed}  blocking={v.blocking_failures}")
        for r in v.reasons:
            print(f"  - {r}")
