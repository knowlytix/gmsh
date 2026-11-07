# SPDX-License-Identifier: Apache-2.0
"""Draft an audit schema sidecar from a snapshot, for a person to review.

Every declaration in a draft has the source ``inferred``, so a draft takes no
effect even if it is used as an audit's sidecar: a reviewer accepts an entry by
changing its source to ``user``. The guesses are deliberately few and each rests
on stated evidence in the graph:

- subclass and type predicates: relations whose label is a common name for
  subclass or instance-of (``is_a``, ``subclass_of``, ``instance_of`` and the like);
- classes: the objects of a guessed type predicate;
- symmetric relations: every edge of the relation also appears reversed;
- inverse relations: two relations whose edges are each other's reversals.

Names are guesses about intent, not facts about the graph, which is why none of
this is applied automatically. Descriptions, roots, modules and conventions are
never drafted.
"""

from __future__ import annotations

from collections.abc import Iterable

from knowlytix.knowledge.kgcheck.audit_schema import (
    SIDECAR,
    contradictions,
    declarations,
    sidecar_violations,
)
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    GraphView,
    JsonObject,
    NodeKind,
)
from knowlytix.knowledge.kgcheck.normalize import label_key

SCHEMA_VERSION = "kgcheck-audit-schema/1"
SUBCLASS_NAMES = frozenset({"is a", "isa", "is an", "subclass of", "is subclass of",
                            "is a subclass of", "sub class of", "kind of", "is a kind of",
                            "is kind of", "broader", "has broader", "has parent"})
TYPE_NAMES = frozenset({"type", "rdf type", "has type", "instance of", "is instance of",
                        "is an instance of", "is a instance of"})
MIN_EDGES = 2  # a symmetry or inverse guess needs at least this many reversed edges

Edges = dict[str, set[tuple[str, str]]]  # relation token -> {(head, object)}


def draft_audit_schema(snapshot: AuditSnapshot, view: GraphView = GraphView.STORED) -> JsonObject:
    """A sidecar document of inferred declarations drawn from ``view``.

    The result conforms to the sidecar contract and contradicts itself nowhere;
    both are checked before it is returned.
    """
    edges = _edges(snapshot, view)
    subclass = sorted(r for r in edges if label_key(r) in SUBCLASS_NAMES)
    typing = sorted(r for r in edges if label_key(r) in TYPE_NAMES)
    doc: JsonObject = {"version": SCHEMA_VERSION, "default_source": "inferred"}
    if subclass or typing:
        doc["hierarchy"] = {k: v for k, v in (("subclass_predicates", subclass),
                                              ("type_predicates", typing)) if v}
    classes = sorted({o for r in typing for _, o in edges[r]})
    if classes:
        doc["terms"] = {c: {"kind": "class"} for c in classes}
    relations = _relation_guesses(edges, exclude={*subclass, *typing})
    if relations:
        doc["relations"] = relations
    _check(doc)
    return doc


def _edges(snapshot: AuditSnapshot, view: GraphView) -> Edges:
    """Entity-to-entity edges of the view by relation token; literal objects and blank tokens
    (which the sidecar contract cannot name) are skipped."""
    out: Edges = {}
    for row in snapshot.view(view):
        a = snapshot.assertion_index[row.assertion_id]
        head, relation, obj = (snapshot.node_index[n] for n in (a.head, a.predicate, a.object))
        blank = not all(n.label.strip() for n in (head, relation, obj))
        if obj.kind is not NodeKind.LITERAL and not blank:
            out.setdefault(relation.label, set()).add((head.label, obj.label))
    return out


def _relation_guesses(edges: Edges, exclude: set[str]) -> dict[str, JsonObject]:
    out: dict[str, JsonObject] = {}
    candidates = sorted(r for r in edges if r not in exclude)
    for r in candidates:
        pairs = {(h, o) for h, o in edges[r] if h != o}
        if len(pairs) >= MIN_EDGES and all((o, h) in pairs for h, o in pairs):
            out[r] = {"symmetric": True}
    paired: set[str] = set(out)
    for r in candidates:
        for s in candidates:
            if r < s and not {r, s} & paired and _reversed(edges[r], edges[s]):
                out[r] = {"inverse_of": s}
                paired |= {r, s}
    return dict(sorted(out.items()))


def _reversed(a: Iterable[tuple[str, str]], b: Iterable[tuple[str, str]]) -> bool:
    forward, backward = set(a), {(o, h) for h, o in b}
    return len(forward) >= MIN_EDGES and forward == backward


def _check(doc: JsonObject) -> None:
    problems = [v.message for v in sidecar_violations(doc)]
    if problems:  # a drafting bug, not a property of the store
        raise AssertionError(f"draft breaks the {SIDECAR} contract: {problems[:3]}")
    conflicts = contradictions(declarations(doc))
    if conflicts:
        raise AssertionError(f"draft contradicts itself: {[c.message for c in conflicts[:3]]}")
