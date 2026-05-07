# SPDX-License-Identifier: Apache-2.0
"""Rule-based second attestor for entity-comparison verdicts.

The geometry produces one signal; the policy produces a second,
*independent* signal derived from graph structure / ENM lookups
(no embeddings, no learned models). The runtime gate combines the
two: a ``contradiction`` verdict survives only when both signals
concur. Disagreement or a single attestor demotes to ``review``.

Built-in rules are minimal scaffolding. High-stakes deployment is
expected to supply domain-specific rules. The architecture is the
contract; the rule library is configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.governance.governance_bundle import GovernanceBundle


PolicySignal = Literal["contradiction", "agreement", "abstain"]


@dataclass
class RuleResult:
    """One rule's verdict on an entity pair.

    ``signal`` is what the rule says when it fires; ``None`` when the
    rule has nothing to say.  ``reason`` is a short audit string.
    """

    signal: PolicySignal | None
    reason: str = ""


@dataclass
class ComparisonRule:
    """Predicate + metadata for one rule.

    ``predicate(a, b, store, bundle) -> RuleResult`` runs at verdict
    time. Keep predicates pure and fast (microseconds): they are on
    the hot path.
    """

    rule_id: str
    description: str
    severity: Literal["blocking", "advisory"]
    predicate: Callable[
        [str, str, "GMSExpertStore", "GovernanceBundle"],
        RuleResult,
    ]

    def __call__(
        self, a: str, b: str,
        store: "GMSExpertStore",
        bundle: "GovernanceBundle",
    ) -> RuleResult:
        return self.predicate(a, b, store, bundle)


@dataclass
class ComparisonPolicyVerdict:
    """Aggregated verdict from running a policy against an entity pair."""

    verdict: PolicySignal
    fired_rules: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    contradicting_rules: list[str] = field(default_factory=list)
    agreeing_rules: list[str] = field(default_factory=list)


class ComparisonPolicy:
    """A set of rules that produce a second-attestor verdict.

    Combining logic: ``contradiction`` dominates ``agreement``
    dominates ``abstain``. Mirrors the contradict-dominates-agreement
    rule used inside ``label_tension``, so the policy and the
    labeling pipeline classify edge cases the same way.
    """

    def __init__(self, rules: list[ComparisonRule]) -> None:
        if not rules:
            raise ValueError("ComparisonPolicy needs at least one rule")
        self.rules = list(rules)

    def check(
        self, a: str, b: str,
        store: "GMSExpertStore",
        bundle: "GovernanceBundle",
    ) -> ComparisonPolicyVerdict:
        contra: list[tuple[str, str]] = []
        agree: list[tuple[str, str]] = []
        for rule in self.rules:
            r = rule(a, b, store, bundle)
            if r.signal == "contradiction":
                contra.append((rule.rule_id, r.reason))
            elif r.signal == "agreement":
                agree.append((rule.rule_id, r.reason))

        if contra:
            return ComparisonPolicyVerdict(
                verdict="contradiction",
                fired_rules=[r for r, _ in contra],
                reasons=[reason for _, reason in contra],
                contradicting_rules=[r for r, _ in contra],
                agreeing_rules=[r for r, _ in agree],
            )
        if agree:
            return ComparisonPolicyVerdict(
                verdict="agreement",
                fired_rules=[r for r, _ in agree],
                reasons=[reason for _, reason in agree],
                contradicting_rules=[],
                agreeing_rules=[r for r, _ in agree],
            )
        return ComparisonPolicyVerdict(
            verdict="abstain",
            fired_rules=[],
            reasons=[],
        )


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


def _declared_opposition_predicate(
    a: str, b: str, store, bundle,
) -> RuleResult:
    """Confirms a declared (passes, opposite_of, fails)-style pair.

    Independent signal: this comes from the static opposition table
    in the bundle, not from embeddings. Useful as a redundant
    confirmation when geometry says contradiction — geometry could
    in principle have learned a different opposition than the one
    declared in the corpus.
    """
    if bundle.is_declared_opposition(a, b):
        reason = (bundle.opposition_table.get((a, b))
                  or bundle.opposition_table.get((b, a))
                  or "declared")
        return RuleResult(signal="contradiction", reason=reason)
    return RuleResult(signal=None)


def _functional_relation_disagree_predicate(
    a: str, b: str, store, bundle,
) -> RuleResult:
    """Two entities disagree on a functional relation.

    A relation R is *functional* when ``(R, is_functional, true)`` is
    declared in the corpus.  For such an R, ``(a, R, X)`` and
    ``(b, R, Y)`` with ``X != Y`` is structurally a contradiction.

    No embeddings used — pure graph lookup.
    """
    if store.doc_graph is None:
        return RuleResult(signal=None)
    triples = list(store.doc_graph.triples)
    functional: set[str] = set()
    for h, r, t in triples:
        if r == "is_functional" and t in {"true", "True", "1"}:
            functional.add(h)
    if not functional:
        return RuleResult(signal=None)

    a_facts: dict[str, set[str]] = {}
    b_facts: dict[str, set[str]] = {}
    for h, r, t in triples:
        if r not in functional:
            continue
        if h == a:
            a_facts.setdefault(r, set()).add(t)
        elif h == b:
            b_facts.setdefault(r, set()).add(t)

    for r, a_targets in a_facts.items():
        b_targets = b_facts.get(r)
        if b_targets is None:
            continue
        if a_targets and b_targets and a_targets.isdisjoint(b_targets):
            return RuleResult(
                signal="contradiction",
                reason=(
                    f"functional relation '{r}': "
                    f"{a}->{sorted(a_targets)} vs {b}->{sorted(b_targets)}"
                ),
            )
    return RuleResult(signal=None)


def _shared_target_agreement_predicate(min_shared: int = 2):
    """Many shared (relation, target) pairs and no opposing relations.

    Pure graph lookup. Mirrors the agreement-mining rule in
    ``label_tension`` so that this rule and the labeler classify
    the same pairs as agreement.
    """

    def _predicate(a: str, b: str, store, bundle) -> RuleResult:
        if store.doc_graph is None:
            return RuleResult(signal=None)
        triples = list(store.doc_graph.triples)
        a_rt: set[tuple[str, str]] = set()
        b_rt: set[tuple[str, str]] = set()
        for h, r, t in triples:
            if r in {"is_functional", "opposite_of"}:
                continue
            if h == a:
                a_rt.add((r, t))
            elif h == b:
                b_rt.add((r, t))
        shared = a_rt & b_rt
        if len(shared) < min_shared:
            return RuleResult(signal=None)
        if bundle.is_declared_opposition(a, b):
            return RuleResult(signal=None)
        return RuleResult(
            signal="agreement",
            reason=f"{len(shared)} shared (relation, target) pairs",
        )

    return _predicate


def declared_opposition_rule() -> ComparisonRule:
    return ComparisonRule(
        rule_id="R1_declared_opposition",
        description=(
            "Pair is in the bundle's declared opposition table "
            "(corpus-mined from (r_a, opposite_of, r_b) triples)."
        ),
        severity="blocking",
        predicate=_declared_opposition_predicate,
    )


def functional_relation_rule() -> ComparisonRule:
    return ComparisonRule(
        rule_id="R2_functional_disagree",
        description=(
            "Entities disagree on a functional relation "
            "(declared via (r, is_functional, true))."
        ),
        severity="blocking",
        predicate=_functional_relation_disagree_predicate,
    )


def shared_target_rule(min_shared: int = 2) -> ComparisonRule:
    return ComparisonRule(
        rule_id=f"R3_shared_target_min{min_shared}",
        description=(
            "Entities share at least N (relation, target) pairs and "
            "have no declared opposition."
        ),
        severity="advisory",
        predicate=_shared_target_agreement_predicate(min_shared=min_shared),
    )


def default_comparison_policy(
    *, min_shared_for_agreement: int = 2,
) -> ComparisonPolicy:
    """Minimal policy demonstrating the two-attestor pattern.

    Suitable as a starting point. High-stakes deployment will add
    domain-specific rules.
    """
    return ComparisonPolicy(rules=[
        declared_opposition_rule(),
        functional_relation_rule(),
        shared_target_rule(min_shared=min_shared_for_agreement),
    ])
