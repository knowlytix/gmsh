# SPDX-License-Identifier: Apache-2.0
"""Hierarchy rules STR003-STR007: cycles, redundant edges, multiple inheritance,
reverse edges and classes outside the taxonomy.

Subclass edges exist only under predicates the audit schema sidecar declares
(``kgcheck.taxonomy``); without them the hierarchy rules report unavailable.
Multiple inheritance and mutual subclass edges are legitimate in OWL (the
latter states equivalence), so neither fails by default: a cycle is an error
only when the sidecar declares the taxonomy acyclic, and multiple parents are
information. STR006 needs no sidecar to find reverse edges, but classifies them
only by declared relation properties, never by predicate names.
"""

from __future__ import annotations

from collections.abc import Iterator

from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    Finding,
    GraphView,
    NodeKind,
    RawRef,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin
from knowlytix.knowledge.kgcheck.rules._common import label
from knowlytix.knowledge.kgcheck.taxonomy import (
    HIERARCHY_VIEWS,
    SUBCLASS_PREDICATES,
    Budget,
    Condensation,
    Hierarchy,
    condense,
    cycle_through,
    cyclic_components,
    hierarchy,
    reachable,
    shortest_path,
)

_WITNESS = 50
_STEPS = {"max_steps": 2_000_000}
_STEP_BOUNDS = {"max_steps": (1, None)}


def _hierarchy(ctx: RuleContext) -> Hierarchy | None:
    """The view's hierarchy, or None when no subclass predicate is declared."""
    h = hierarchy(ctx.snapshot, ctx.view)
    return h if h.predicates else None


def _labels(snapshot: AuditSnapshot, ids: list[str]) -> list[str]:
    return [label(snapshot, i) for i in ids[:_WITNESS]]


def _path_refs(h: Hierarchy, path: list[str]) -> tuple[RawRef, ...]:
    return tuple(r for c, p in zip(path, path[1:]) for r in h.refs(c, p)[:1])


def _partial(budget: Budget) -> str | None:
    return f"stopped after {budget.limit} traversal steps (max_steps)" if budget.exhausted else None


# --- STR003 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "STR003", "2", "structure", "Subclass cycles", Severity.WARNING, phase="B",
    prerequisites=("hierarchy",), views=HIERARCHY_VIEWS, parameters=_STEPS,
    parameter_bounds=_STEP_BOUNDS,
    basis="strongly connected components (iterative Tarjan) over the edges of the declared "
          "subclass predicates; the witness is a shortest cycle through the component's first "
          "member, or the shortest found before max_steps when the search stopped"))
def str003(ctx: RuleContext) -> Outcome:
    h = _hierarchy(ctx)
    if h is None:
        return Outcome.unavailable(SUBCLASS_PREDICATES)
    acyclic = ctx.snapshot.semantics.accepted().find(DeclKey.ACYCLIC, "")
    contract = acyclic is not None and acyclic.value == "true"
    budget, succ, findings = Budget(int(ctx.params["max_steps"])), h.successors(), []
    for members in cyclic_components(h):
        cycle = cycle_through(members[0], succ, set(members), budget)
        findings.append(_cycle(ctx, h, members, cycle, contract, shortest=not budget.exhausted))
    return Outcome(eligible=len(h.nodes), findings=tuple(findings),
                   metrics={"classes": len(h.nodes), "subclass_edges": len(h.edges),
                            "cyclic_components": len(findings), "acyclic_contract": contract,
                            "witness_truncated": budget.exhausted})


def _cycle(ctx: RuleContext, h: Hierarchy, members: list[str], cycle: list[str] | None,
           contract: bool, *, shortest: bool) -> Finding:
    shown = " -> ".join(_labels(ctx.snapshot, cycle)) if cycle else "(witness not computed)"
    return ctx.finding(
        f"{len(members)} classes form a subclass cycle: {shown}",
        severity=Severity.ERROR if contract else Severity.WARNING,
        affected=tuple(members), locations=_path_refs(h, cycle or [])[:_WITNESS],
        witness={"members": _labels(ctx.snapshot, members), "size": len(members),
                 "cycle": _labels(ctx.snapshot, cycle) if cycle else None,
                 "cycle_length": len(cycle) - 1 if cycle else None,
                 "shortest": shortest and cycle is not None, "acyclic_contract": contract},
        suggestion="if the classes are equivalent, merge them or declare one an alias of the "
                   "other; otherwise remove the edge that closes the cycle",
        discriminator="cycle")


# --- STR004 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "STR004", "2", "structure", "Redundant subclass edges", Severity.SUGGESTION, phase="B",
    prerequisites=("hierarchy",), views=HIERARCHY_VIEWS, parameters=_STEPS,
    parameter_bounds=_STEP_BOUNDS,
    basis="on the hierarchy with each cycle collapsed to one node, a breadth-first search for "
          "another path between the two components of each edge with that edge excluded; edges "
          "inside a cycle are left to STR003"))
def str004(ctx: RuleContext) -> Outcome:
    h = _hierarchy(ctx)
    if h is None:
        return Outcome.unavailable(SUBCLASS_PREDICATES)
    cond = condense(h)
    succ = cond.successors()
    pairs = sorted((c, p) for c, ps in cond.parents.items() for p in ps)
    budget, findings, done = Budget(int(ctx.params["max_steps"])), [], 0
    for child, parent in pairs:
        path = shortest_path(child, parent, succ, budget, skip=(child, parent))
        if budget.exhausted:
            break
        done += 1
        if path is not None:
            findings.append(_redundant(ctx, h, cond, cond.parents[child][parent], path))
    inside = len(h.edges) - sum(len(e) for ps in cond.parents.values() for e in ps.values())
    return Outcome(eligible=len(pairs), tested=done, findings=tuple(findings),
                   partial=_partial(budget),
                   metrics={"subclass_edges": len(h.edges), "skipped_in_cycles": inside,
                            "redundant": len(findings), "steps": budget.spent})


def _redundant(ctx: RuleContext, h: Hierarchy, cond: Condensation, edges: list[tuple[str, str]],
               path: list[int]) -> Finding:
    snap = ctx.snapshot
    child, parent = edges[0]
    route = [cond.components[c][0] for c in path]
    return ctx.finding(
        f"{label(snap, child)!r} -> {label(snap, parent)!r} is implied by "
        f"{' -> '.join(_labels(snap, route))}",
        affected=(child, parent), locations=tuple(r for c, p in edges for r in h.refs(c, p)[:1])[:_WITNESS],
        witness={"edge": [label(snap, child), label(snap, parent)],
                 "edges": [[label(snap, c), label(snap, p)] for c, p in edges[:_WITNESS]],
                 "alternative_path": _labels(snap, route)},
        suggestion="review: the direct edge adds nothing to the hierarchy unless it records "
                   "intent; remove it or keep it deliberately",
        discriminator=f"redundant:{child}")


# --- STR005 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "STR005", "2", "structure", "Multiple inheritance", Severity.INFO, phase="B",
    prerequisites=("hierarchy",), views=HIERARCHY_VIEWS, parameters=_STEPS,
    parameter_bounds=_STEP_BOUNDS,
    basis="direct parents of each class, or of each set of mutually subclassed classes, counted "
          "raw and after removing parents reachable from another parent; mutually reachable "
          "parents count once"))
def str005(ctx: RuleContext) -> Outcome:
    h = _hierarchy(ctx)
    if h is None:
        return Outcome.unavailable(SUBCLASS_PREDICATES)
    cond = condense(h)
    succ = cond.successors()
    budget, findings, done = Budget(int(ctx.params["max_steps"])), [], 0
    for comp, parents in sorted(succ.items()):
        reduced = _reduced(parents, succ, budget)
        if budget.exhausted:
            break
        done += 1
        if len(reduced) > 1:
            findings.append(_multiple(ctx, h, cond, comp, reduced))
    raw = sum(len(ps) > 1 for ps in h.parents.values())
    return Outcome(eligible=len(succ), tested=done, findings=tuple(findings),
                   partial=_partial(budget),
                   metrics={"classes_with_parents": len(h.parents), "multiple_parents_raw": raw,
                            "multiple_parents_reduced": len(findings), "steps": budget.spent})


def _reduced(parents: list[int], succ: dict[int, list[int]], budget: Budget) -> list[int]:
    """Parent components minus those reachable from another parent component."""
    if len(parents) < 2:
        return parents
    implied: set[int] = set()
    for p in parents:
        implied |= reachable(p, succ, budget) - {p}
    return [p for p in parents if p not in implied]


def _multiple(ctx: RuleContext, h: Hierarchy, cond: Condensation, comp: int,
              reduced: list[int]) -> Finding:
    snap = ctx.snapshot
    members = cond.components[comp]
    parents = [cond.components[p][0] for p in reduced]
    raw = sorted({p for m in members for p in h.parents.get(m, {})} - set(members))
    edges = [e for p in reduced for e in cond.parents[comp][p][:1]]
    subject = (repr(label(snap, members[0])) if len(members) == 1 else
               "equivalent classes " + ", ".join(repr(m) for m in _labels(snap, members)))
    return ctx.finding(
        f"{subject} has {len(reduced)} independent parents: "
        f"{', '.join(repr(label(snap, p)) for p in parents)}",
        affected=(*members, *parents), locations=tuple(r for c, p in edges for r in h.refs(c, p)[:1]),
        witness={"classes": _labels(snap, members), "raw_parents": _labels(snap, raw),
                 "reduced_parents": _labels(snap, parents), "raw_count": len(raw),
                 "reduced_count": len(reduced)},
        suggestion="multiple inheritance is valid; confirm the class belongs under each parent",
        discriminator=f"multiple_parents:{members[0]}")


# --- STR006 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "STR006", "2", "structure", "Reverse edges", Severity.INFO, phase="B",
    views=(GraphView.STORED, GraphView.CONFORMANT, GraphView.QUARANTINED, GraphView.PROPOSAL),
    basis="entity pairs linked in both directions, classified by the sidecar's declared "
          "symmetric, asymmetric and inverse relations; an asymmetric relation is checked "
          "together with the edges its declared inverse entails; subclass edges are left to "
          "STR003"))
def str006(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    sem = snap.semantics.accepted()
    subclass = {d.subject for d in sem.of(DeclKey.SUBCLASS_PREDICATE) if d.value == "true"}
    edges = _entity_edges(ctx, subclass)
    props = _RelationProps.of(snap)
    findings, counts = [], {"symmetric": 0, "inverse": 0, "bidirectional": 0, "asymmetric": 0}
    for (a, b), (forward, backward) in sorted(_pairs(edges).items()):
        for kind, relations in _classify(props, forward, backward, loop=a == b):
            counts[kind] += 1
            if kind == "bidirectional":
                findings.append(_reverse(ctx, a, b, relations, edges))
    for r, x, y, sources in _asymmetry(props, edges):
        counts["asymmetric"] += 1
        findings.append(_asymmetric(ctx, r, x, y, sources, edges))
    return Outcome(eligible=len(edges), findings=tuple(findings),
                   metrics={"entity_edges": len(edges), "relation_properties_declared":
                            snap.capabilities.relation_properties, **counts})


class _RelationProps:
    def __init__(self, symmetric: set[str], asymmetric: set[str], inverse: set[tuple[str, str]]):
        self.symmetric, self.asymmetric, self.inverse = symmetric, asymmetric, inverse

    @classmethod
    def of(cls, snapshot: AuditSnapshot) -> _RelationProps:
        sem = snapshot.semantics.accepted()
        flag = lambda key: {d.subject for d in sem.of(key) if d.value == "true"}  # noqa: E731
        pairs = set(sem.inverse_pairs())
        return cls(flag(DeclKey.SYMMETRIC) | {a for a, b in pairs if a == b},
                   flag(DeclKey.ASYMMETRIC), pairs)

    def inverse_of(self, r: str, s: str) -> bool:
        return tuple(sorted((r, s))) in self.inverse


Edge = tuple[str, str, str]  # head, predicate, object node ids


def _entity_edges(ctx: RuleContext, excluded: set[str]) -> dict[Edge, list[RawRef]]:
    snap, out = ctx.snapshot, {}
    for row in snap.view(ctx.view):
        a = snap.assertion_index[row.assertion_id]
        node = snap.node_index.get(a.object)
        if a.predicate not in excluded and not (node and node.kind is NodeKind.LITERAL):
            out.setdefault((a.head, a.predicate, a.object), []).append(row.raw)
    return out


def _pairs(edges: dict[Edge, list[RawRef]]) -> dict[tuple[str, str], tuple[set[str], set[str]]]:
    """Unordered entity pairs (self loops included) with the relations in each direction."""
    out: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    for h, r, o in edges:
        a, b = sorted((h, o))
        forward, backward = out.setdefault((a, b), (set(), set()))
        (forward if (h, o) == (a, b) else backward).add(r)
    return {k: v for k, v in out.items() if (v[0] and v[1]) or k[0] == k[1]}


def _classify(props: _RelationProps, forward: set[str], backward: set[str], *, loop: bool
              ) -> Iterator[tuple[str, list[str]]]:
    """Reverse-edge kinds; asymmetric relations are left to ``_asymmetry``."""
    if loop:  # a self loop is its own reverse; only an asymmetric relation forbids it
        return
    for r in sorted((forward & backward) - props.asymmetric):
        yield ("symmetric" if r in props.symmetric else "bidirectional"), [r]
    for r in sorted(forward - backward):
        for s in sorted(backward - forward):
            yield ("inverse" if props.inverse_of(r, s) else "bidirectional"), [r, s]


def _asymmetry(props: _RelationProps, edges: dict[Edge, list[RawRef]]
              ) -> Iterator[tuple[str, str, str, list[Edge]]]:
    """(relation, x, y, stated edges) where an asymmetric relation holds both ways or on itself.

    ``a s b`` with ``s`` the declared inverse of ``r`` entails ``b r a``.
    """
    for r in sorted(props.asymmetric):
        holds: dict[tuple[str, str], list[Edge]] = {}
        for h, rel, o in edges:
            if rel == r:
                holds.setdefault((h, o), []).append((h, rel, o))
            elif rel != r and props.inverse_of(r, rel):
                holds.setdefault((o, h), []).append((h, rel, o))
        for (x, y), stated in sorted(holds.items()):
            if x == y:
                yield r, x, y, stated
            elif x < y and (y, x) in holds:
                yield r, x, y, stated + holds[(y, x)]


def _asymmetric(ctx: RuleContext, r: str, x: str, y: str, stated: list[Edge],
                edges: dict[Edge, list[RawRef]]) -> Finding:
    snap = ctx.snapshot
    where = (f"relates {label(snap, x)!r} to itself" if x == y else
             f"holds between {label(snap, x)!r} and {label(snap, y)!r} in both directions")
    return ctx.finding(
        f"{label(snap, r)!r} is declared asymmetric but {where}", severity=Severity.ERROR,
        affected=(x, y, r), locations=tuple(edges[e][0] for e in stated),
        witness={"entities": [label(snap, x), label(snap, y)], "relation": label(snap, r),
                 "stated": [[label(snap, n) for n in e] for e in stated]},
        suggestion="remove the assertion that contradicts the asymmetry, or the declaration",
        discriminator=f"asymmetric:{r}")


def _reverse(ctx: RuleContext, a: str, b: str, relations: list[str],
             edges: dict[Edge, list[RawRef]]) -> Finding:
    snap = ctx.snapshot
    names = [label(snap, r) for r in relations]
    refs = tuple(ref for (h, r, o), rs in edges.items() if {h, o} == {a, b} and r in relations
                 for ref in rs[:1])
    via = f"{names[0]!r}" if len(names) == 1 else f"{names[0]!r} and {names[1]!r}"
    return ctx.finding(
        f"{label(snap, a)!r} and {label(snap, b)!r} are linked in both directions by {via}",
        affected=(a, b, *relations), locations=refs,
        witness={"entities": [label(snap, a), label(snap, b)], "relations": names},
        suggestion="declare the relation symmetric, or the two relations inverses, if the "
                   "reverse edge is intended", discriminator=f"bidirectional:{'|'.join(relations)}")


# --- STR007 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "STR007", "2", "structure", "Classes outside the taxonomy", Severity.WARNING, phase="B",
    prerequisites=("node_kinds", "hierarchy"), views=HIERARCHY_VIEWS,
    parameters={"require_participation": False, "exempt": ()},
    basis="declared classes with no subclass edge in the view; when roots are declared, "
          "top-level classes, or top-level cycles, with no declared root among them"))
def str007(ctx: RuleContext) -> Outcome:
    h = _hierarchy(ctx)
    if h is None:
        return Outcome.unavailable(SUBCLASS_PREDICATES)
    snap, sem = ctx.snapshot, ctx.snapshot.semantics.accepted()
    standalone = {d.subject for d in sem.of(DeclKey.STANDALONE) if d.value == "true"}
    roots = set(sem.subjects(DeclKey.ROOT))
    exempt = standalone | roots | {f"entity:{t}" for t in ctx.params["exempt"]}
    classes = sorted(n.node_id for n in snap.nodes if n.kind is NodeKind.CLASS)
    incident = set(h.nodes)
    severity = Severity.WARNING if ctx.params["require_participation"] else Severity.INFO
    findings = [_outside(ctx, c, severity) for c in classes if c not in incident and c not in exempt]
    cond = condense(h)
    tops = [c for c in cond.components if cond.of[c[0]] not in cond.parents] if roots else []
    findings += [_unexpected_root(ctx, h, members) for members in tops if not set(members) & roots]
    return Outcome(eligible=len(set(classes) | incident), findings=tuple(findings),
                   metrics={"classes": len(classes), "in_hierarchy": len(incident & set(classes)),
                            "declared_roots": len(roots), "top_level": len(tops)})


def _outside(ctx: RuleContext, node: str, severity: Severity) -> Finding:
    return ctx.finding(f"class {label(ctx.snapshot, node)!r} has no subclass edge in the "
                       f"{ctx.view.value} view", severity=severity, affected=(node,),
                       suggestion="place the class under a parent, declare it a root, or mark it "
                                  "standalone", discriminator="outside")


def _unexpected_root(ctx: RuleContext, h: Hierarchy, members: list[str]) -> Finding:
    inside = set(members)
    below = sorted((c, p) for c, p in h.edges if p in inside and c not in inside)
    names = ", ".join(repr(m) for m in _labels(ctx.snapshot, members))
    return ctx.finding(f"{names} {'is' if len(members) == 1 else 'are'} at the top of the "
                       "hierarchy but not a declared root", affected=tuple(members),
                       locations=tuple(r for c, p in below for r in h.refs(c, p)[:1])[:_WITNESS],
                       witness={"classes": _labels(ctx.snapshot, members),
                                "children": _labels(ctx.snapshot, [c for c, _ in below])},
                       suggestion="give the class a parent, or add it to hierarchy.roots",
                       discriminator="unexpected_root")
