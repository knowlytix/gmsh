# SPDX-License-Identifier: Apache-2.0
"""STR008: values a class restates although a superclass already gives them.

The inheritance model is explicit and narrow: a relation declared
``inherited`` in the audit schema sidecar states a value for a class that holds
for every subclass. Under that model a subclass asserting the same value as an
ancestor adds nothing. Without such a declaration no value is inherited and the
rule is unavailable; child facts are never redundant by default, and the rule
only suggests a review, never a deletion.
"""

from __future__ import annotations

from knowlytix.knowledge.kgcheck.datatypes import value_key
from knowlytix.knowledge.kgcheck.models import DeclKey, Severity
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin
from knowlytix.knowledge.kgcheck.rules._common import label, rows_by_assertion
from knowlytix.knowledge.kgcheck.taxonomy import (
    HIERARCHY_VIEWS,
    SUBCLASS_PREDICATES,
    hierarchy,
)
from knowlytix.knowledge.kgcheck.typed import Typing

_WITNESS = 50


@builtin(RuleSpec(
    "STR008", "1", "structure", "Restated inherited values", Severity.SUGGESTION, phase="C",
    prerequisites=("hierarchy",), views=HIERARCHY_VIEWS,
    basis="for each relation declared inherited, an assertion on a class whose superclass, "
          "through declared subclass edges, asserts the same value (numbers compared by value); "
          "the witness is the shortest path to the nearest such ancestor"))
def str008(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    inherited = {d.subject for d in snap.semantics.accepted().of(DeclKey.INHERITED)
                 if d.value == "true"}
    if not inherited:
        return Outcome.unavailable("inherited_relations")
    h = hierarchy(snap, ctx.view)
    if not h.predicates:
        return Outcome.unavailable(SUBCLASS_PREDICATES)
    values: dict[tuple[str, str], dict[str, str]] = {}  # (class, relation) -> value key -> aid
    rows = rows_by_assertion(snap, ctx.view)
    for aid in rows:
        a = snap.assertion_index[aid]
        if a.predicate in inherited:
            values.setdefault((a.head, a.predicate), {})[value_key(label(snap, a.object))] = aid
    above = Typing({}, h.successors(), closed=False)
    findings, eligible = [], 0
    for (cls, relation), stated in sorted(values.items()):
        ancestors = {c: p for c, p in above.superclasses(cls).items() if c != cls}
        if not ancestors:
            continue
        for key, aid in sorted(stated.items()):
            eligible += 1
            path = _nearest(ancestors, relation, key, values)
            if path is not None:
                findings.append(_restated(ctx, aid, path, values[(path[-1], relation)][key],
                                          rows[aid]))
    return Outcome(eligible=eligible, findings=tuple(findings),
                   metrics={"inherited_relations": len(inherited), "restated": len(findings)})


def _nearest(ancestors: dict[str, list[str]], relation: str, key: str,
             values: dict[tuple[str, str], dict[str, str]]) -> list[str] | None:
    hits = [p for c, p in ancestors.items() if key in values.get((c, relation), {})]
    return min(hits, key=lambda p: (len(p), p)) if hits else None


def _restated(ctx: RuleContext, aid: str, path: list[str], ancestor_aid: str, refs) -> object:
    snap = ctx.snapshot
    a = snap.assertion_index[aid]
    return ctx.finding(
        f"{label(snap, a.head)!r} restates {label(snap, a.predicate)!r} = "
        f"{label(snap, a.object)!r}, which it inherits from {label(snap, path[-1])!r}",
        affected=(aid,), locations=tuple(refs[:1]), discriminator="restated",
        witness={"class": label(snap, a.head), "relation": label(snap, a.predicate),
                 "value": label(snap, a.object), "ancestor": label(snap, path[-1]),
                 "path": [label(snap, n) for n in path[:_WITNESS]],
                 "ancestor_assertion": ancestor_aid},
        suggestion="review: the subclass assertion is implied by the inheritance declaration; "
                   "keep it only if it records intent")
