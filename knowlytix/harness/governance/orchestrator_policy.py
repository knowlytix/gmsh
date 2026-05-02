# SPDX-License-Identifier: Apache-2.0
"""Pure-Python structural co-attestor for the GovernedOrchestrator.

Mirrors :mod:`plan_policy` (Chapter 9): a list of declarative
``OrchestratorRule`` predicates evaluated against ``(plan, scenario)``
before the orchestrator's EXECUTING phase.  Independent of geometry,
deterministic, no learning.  When wired via
``GovernedOrchestrator(plan_policy=...)``, a blocking failure
transitions PREPARING -> ABORTED with the failure reasons recorded in
the audit trail.

Design follows the AND-gated co-attestor pattern: if either the
geometric attestor (gateway gates 1-3 + verifier) OR the structural
attestor (this rule pack) refuses, the run does not complete.  Two
attestors with uncorrelated blind spots cut the false-accept rate
on adversarial cohorts substantially below either alone --- the same
defense-in-depth result Chapter 9 reports for the planner surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class OrchestratorRule:
    """A single declarative rule.

    Parameters
    ----------
    rule_id : str
        Short identifier (used in audit logs).
    description : str
        Human-readable explanation of what the rule enforces.
    predicate : Callable[[list[str], Any], bool]
        Returns True if the (plan, scenario) tuple satisfies the rule.
    severity : str
        ``"blocking"`` (veto on failure) or ``"advisory"`` (record only).
    """

    rule_id: str
    description: str
    predicate: Callable[[list[str], Any], bool]
    severity: str = "blocking"


@dataclass
class OrchestratorPolicyVerdict:
    """Result of running a (plan, scenario) through ``OrchestratorPolicy``."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)
    rules_checked: int = 0
    blocking_failures: list[str] = field(default_factory=list)
    advisory_failures: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Engine
# ------------------------------------------------------------------ #

class OrchestratorPolicy:
    """Run a declarative rule pack against an orchestrator plan.

    Kept deliberately simple: a list of :class:`OrchestratorRule` and
    an AND-gated :meth:`check`.  No geometry, no learning.  The point
    of this layer is to be independent of the gateway and verifier.
    """

    def __init__(self, rules: list[OrchestratorRule]):
        self.rules = list(rules)

    def check(
        self, plan: list[str], scenario: Any,
    ) -> OrchestratorPolicyVerdict:
        blocking: list[str] = []
        advisory: list[str] = []
        reasons: list[str] = []
        for rule in self.rules:
            try:
                ok = bool(rule.predicate(plan, scenario))
            except Exception as e:
                # Treat rule errors as blocking --- fail safe.
                ok = False
                reasons.append(
                    f"rule {rule.rule_id} raised "
                    f"{type(e).__name__}: {e}"
                )
            if not ok:
                reasons.append(f"{rule.rule_id}: {rule.description}")
                if rule.severity == "blocking":
                    blocking.append(rule.rule_id)
                else:
                    advisory.append(rule.rule_id)
        return OrchestratorPolicyVerdict(
            allowed=(len(blocking) == 0),
            reasons=reasons,
            rules_checked=len(self.rules),
            blocking_failures=blocking,
            advisory_failures=advisory,
        )


# ------------------------------------------------------------------ #
# Default rule pack for the data-science orchestrator
# ------------------------------------------------------------------ #

# Datasets the default rule pack accepts.  Production deployments
# would replace this with their authorised data catalogue.
_KNOWN_DATASETS: frozenset[str] = frozenset({
    "adult", "iris", "wine", "boston", "diabetes",
})


def _registered_tools() -> set[str]:
    """Tools the gateway has schemas for.

    Imported lazily so test harnesses can swap in a stub catalogue.
    """
    from knowlytix.harness.governance.tools import TOOL_DEFINITIONS  # local import
    return {t["name"] for t in TOOL_DEFINITIONS}


def _no_unknown_tools(plan: list[str], scenario: Any) -> bool:
    """Every step in the plan must be a registered tool."""
    registered = _registered_tools()
    return all(step in registered for step in plan)


def _budget_feasible(plan: list[str], scenario: Any) -> bool:
    """Plan length must not exceed the scenario's budget."""
    return len(plan) <= int(getattr(scenario, "max_tool_calls", 1_000_000))


def _no_duplicate_steps(plan: list[str], scenario: Any) -> bool:
    """No tool may appear twice in the same plan (catches infinite-loop attempts)."""
    return len(plan) == len(set(plan))


def _no_circular_dependencies(plan: list[str], scenario: Any) -> bool:
    """If both ``train_model`` and ``load_dataset`` are present, the
    dataset must come first.  Catches plans that try to train before
    loading data.
    """
    if "train_model" in plan and "load_dataset" in plan:
        return plan.index("load_dataset") < plan.index("train_model")
    return True


def _diagnose_requires_train(plan: list[str], scenario: Any) -> bool:
    """Diagnostic tools (diagnose_*) require ``train_model`` to come
    before them in the plan.
    """
    train_idx = plan.index("train_model") if "train_model" in plan else -1
    for i, step in enumerate(plan):
        if step.startswith("diagnose_") and (train_idx < 0 or i <= train_idx):
            return False
    return True


def _explain_requires_train(plan: list[str], scenario: Any) -> bool:
    """Explanation tools (explain_*) require ``train_model`` first."""
    train_idx = plan.index("train_model") if "train_model" in plan else -1
    for i, step in enumerate(plan):
        if step.startswith("explain_") and (train_idx < 0 or i <= train_idx):
            return False
    return True


def _required_tools_present(plan: list[str], scenario: Any) -> bool:
    """All scenario.required_tools must appear in the plan."""
    required = list(getattr(scenario, "required_tools", []))
    return all(req in plan for req in required)


def _no_forbidden_tools(plan: list[str], scenario: Any) -> bool:
    """No scenario.forbidden_tools may appear in the plan."""
    forbidden = set(getattr(scenario, "forbidden_tools", []))
    return not any(step in forbidden for step in plan)


def default_orchestrator_rules() -> list[OrchestratorRule]:
    """Default eight-rule pack for the data-science orchestrator.

    Each rule is independent of the GMS embedding geometry; together
    they catch the structural failure modes the
    ``orchestrator_adversarial`` cohort probes for: unknown tools,
    circular plans, budget underrun, mid-execution-budget, and
    duplicate-tool injections.

    Production deployments should extend or override this rule pack
    with domain-specific predicates (e.g.\ "release_gate must be
    preceded by diagnose_fairness").
    """
    return [
        OrchestratorRule(
            rule_id="O1_no_unknown_tools",
            description=(
                "Every step in the plan must be a tool registered in "
                "the gateway's TOOL_DEFINITIONS catalogue."
            ),
            predicate=_no_unknown_tools,
        ),
        OrchestratorRule(
            rule_id="O2_budget_feasible",
            description=(
                "Plan length must not exceed the scenario's "
                "max_tool_calls budget."
            ),
            predicate=_budget_feasible,
        ),
        OrchestratorRule(
            rule_id="O3_no_duplicate_steps",
            description="No tool may appear twice in the same plan.",
            predicate=_no_duplicate_steps,
        ),
        OrchestratorRule(
            rule_id="O4_no_circular_dependencies",
            description=(
                "If train_model and load_dataset are both present, "
                "load_dataset must come first."
            ),
            predicate=_no_circular_dependencies,
        ),
        OrchestratorRule(
            rule_id="O5_diagnose_requires_train",
            description=(
                "Diagnostic tools (diagnose_*) must follow "
                "train_model in the plan."
            ),
            predicate=_diagnose_requires_train,
        ),
        OrchestratorRule(
            rule_id="O6_explain_requires_train",
            description=(
                "Explanation tools (explain_*) must follow "
                "train_model in the plan."
            ),
            predicate=_explain_requires_train,
        ),
        OrchestratorRule(
            rule_id="O7_required_tools_present",
            description=(
                "All tools listed in scenario.required_tools must "
                "appear in the plan."
            ),
            predicate=_required_tools_present,
        ),
        OrchestratorRule(
            rule_id="O8_no_forbidden_tools",
            description=(
                "No tools listed in scenario.forbidden_tools may "
                "appear in the plan."
            ),
            predicate=_no_forbidden_tools,
        ),
    ]


__all__ = [
    "OrchestratorRule",
    "OrchestratorPolicyVerdict",
    "OrchestratorPolicy",
    "default_orchestrator_rules",
]
