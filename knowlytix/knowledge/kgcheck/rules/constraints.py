# SPDX-License-Identifier: Apache-2.0
"""Typed constraint rules CON001-CON006: types, literal values, cardinality,
disjointness, irreflexivity and materialized entailments.

These are a documented application policy, evaluated natively on the
declarations of the audit schema sidecar; they are not SHACL or OWL
conformance (CON007 delegates to a SHACL engine). Two choices follow from that.
Domain and range restrict the data instead of inferring types, as OWL would.
And absence proves nothing unless declared: an untyped node is an instance of
no class only under ``closed_world.typing``, a missing value violates a minimum
cardinality only when the relation is declared ``complete``, and an entailed
edge must be stated only when the relation declares ``materialize``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from decimal import Decimal

from knowlytix.knowledge.kgcheck.datatypes import (
    effective_datatype,
    parse_typed,
    same_value,
    value_key,
    value_relations,
)
from knowlytix.knowledge.kgcheck.identity import assertion_id
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    Finding,
    GraphView,
    NodeKind,
    RawRef,
    SemanticsDecl,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin
from knowlytix.knowledge.kgcheck.rules._common import (
    is_literal,
    label,
    rows_by_assertion,
)
from knowlytix.knowledge.kgcheck.taxonomy import HIERARCHY_VIEWS, Budget
from knowlytix.knowledge.kgcheck.typed import Typing, typing

VIEWS = (GraphView.STORED, GraphView.CONFORMANT, GraphView.QUARANTINED)
_WITNESS = 50
_PRE = ("typed_constraints",)
_REFERENCES = ("https://www.w3.org/TR/shacl/", "https://www.w3.org/TR/owl-primer/")


def _labels(snapshot: AuditSnapshot, ids: list[str]) -> list[str]:
    return [label(snapshot, i) for i in ids[:_WITNESS]]


def _edges(ctx: RuleContext) -> dict[tuple[str, str, str], list[RawRef]]:
    """(head, predicate, object) node ids of the view, with their raw rows."""
    snap, out = ctx.snapshot, {}
    for aid, refs in rows_by_assertion(snap, ctx.view).items():
        a = snap.assertion_index[aid]
        out[(a.head, a.predicate, a.object)] = refs
    return out


def _sem(ctx: RuleContext) -> SemanticsDecl:
    return ctx.snapshot.semantics.accepted()


# --- CON001 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "CON001", "1", "constraint", "Subject and object types", Severity.ERROR, phase="C",
    prerequisites=_PRE, views=VIEWS, references=_REFERENCES,
    basis="for each assertion of a relation with a declared domain or range, the subject's or "
          "object's declared and asserted types closed under declared subclass edges must "
          "include an allowed class; domain and range never add types, a literal is an "
          "instance of no class, and an untyped node is unknown unless closed_world.typing"))
def con001(ctx: RuleContext) -> Outcome:
    sem = _sem(ctx)
    allowed = {r: (sem.values(DeclKey.DOMAIN, r), sem.values(DeclKey.RANGE, r))
               for r in {*sem.subjects(DeclKey.DOMAIN), *sem.subjects(DeclKey.RANGE)}}
    if not allowed:
        return Outcome.unavailable("domain_or_range")
    types, snap = typing(ctx.snapshot, ctx.view), ctx.snapshot
    findings: list[Finding] = []
    unknown: dict[tuple[str, str], list[str]] = {}
    eligible = 0
    for (head, predicate, obj), refs in sorted(_edges(ctx).items()):
        domain, range_ = allowed.get(predicate, ((), ()))
        for side, node, classes in (("domain", head, domain), ("range", obj, range_)):
            if not classes:
                continue
            eligible += 1
            state = _type_state(snap, types, node, classes)
            if state == "unknown":
                unknown.setdefault((predicate, side), []).append(node)
            elif state == "violation":
                findings.append(_type_violation(ctx, types, (head, predicate, obj), refs, side,
                                                node, classes))
    findings += [_untyped(ctx, predicate, side, nodes) for (predicate, side), nodes in
                 sorted(unknown.items())]
    return Outcome(eligible=eligible, findings=tuple(findings), metrics={
        "checked": eligible, "violations": sum(f.severity is Severity.ERROR for f in findings),
        "unknown": sum(len(n) for n in unknown.values()), "closed_typing": types.closed})


def _type_state(snap: AuditSnapshot, types: Typing, node: str, classes: tuple[str, ...]) -> str:
    if is_literal(snap, node):
        return "violation"
    held = types.types(node)
    if not held:
        return "violation" if types.closed else "unknown"
    return "ok" if any(c in held for c in classes) else "violation"


def _type_violation(ctx: RuleContext, types: Typing, edge: tuple[str, str, str],
                    refs: list[RawRef], side: str, node: str, classes: tuple[str, ...]) -> Finding:
    snap = ctx.snapshot
    held = types.types(node)
    what = ("is a literal" if is_literal(snap, node) else
            "has no type" if not held else f"has types {', '.join(map(repr, _labels(snap, list(held))))}")
    aid = assertion_id(*(label(snap, n) for n in edge))
    return ctx.finding(
        f"{side} of {label(snap, edge[1])!r}: {label(snap, node)!r} {what}, not "
        f"{' or '.join(repr(c) for c in _labels(snap, list(classes)))}",
        affected=(aid, node), locations=tuple(refs[:1]), discriminator=side,
        witness={"triple": [label(snap, n) for n in edge], "side": side, "node": label(snap, node),
                 "allowed": _labels(snap, list(classes)),
                 "types": {label(snap, c): _labels(snap, p) for c, p in list(held.items())[:_WITNESS]},
                 "closed_typing": types.closed},
        suggestion="correct the assertion, declare the node's type, or widen the "
                   f"{side} of the relation")


def _untyped(ctx: RuleContext, predicate: str, side: str, nodes: list[str]) -> Finding:
    snap, distinct = ctx.snapshot, sorted(set(nodes))
    return ctx.finding(
        f"{len(distinct)} {'subjects' if side == 'domain' else 'objects'} of "
        f"{label(snap, predicate)!r} have no type, so its {side} is not checked for them",
        severity=Severity.INFO, affected=(predicate,), discriminator=f"untyped:{side}",
        witness={"relation": label(snap, predicate), "side": side, "untyped": len(distinct),
                 "nodes": _labels(snap, distinct)},
        suggestion="declare their types, or declare closed_world.typing when the declared "
                   "types are complete")


# --- CON002 -------------------------------------------------------------------------

@builtin(RuleSpec(
    "CON002", "2", "constraint", "Literal values", Severity.ERROR, phase="C",
    prerequisites=_PRE, views=VIEWS, references=_REFERENCES,
    basis="each object of a relation with a declared value contract parsed as its datatype "
          "(decimal when only bounds are declared, else string) in its XSD lexical form, exactly "
          "as stored and without coercion; then checked against the declared inclusive bounds and allowed values, "
          "compared numerically for decimal and integer"))
def con002(ctx: RuleContext) -> Outcome:
    sem = _sem(ctx)
    relations = value_relations(sem)
    if not relations:
        return Outcome.unavailable("value_contract")
    contracts = {r: _contract(sem, r) for r in relations}
    findings, eligible, problems = [], 0, {}
    for (head, predicate, obj), refs in sorted(_edges(ctx).items()):
        contract = contracts.get(predicate)
        if contract is None:
            continue
        eligible += 1
        problem = _value_problem(ctx.snapshot, obj, contract)
        if problem is not None:
            problems[problem[0]] = problems.get(problem[0], 0) + 1
            findings.append(_bad_value(ctx, (head, predicate, obj), refs, contract, *problem))
    return Outcome(eligible=eligible, findings=tuple(findings),
                   metrics={"checked": eligible, "problems": dict(sorted(problems.items()))})


def _contract(sem: SemanticsDecl, relation: str) -> dict[str, object]:
    low, high = sem.find(DeclKey.MINIMUM, relation), sem.find(DeclKey.MAXIMUM, relation)
    return {"datatype": effective_datatype(sem, relation),
            "minimum": Decimal(low.value) if low else None,
            "maximum": Decimal(high.value) if high else None,
            "allowed": sem.values(DeclKey.ALLOWED_VALUE, relation)}


def is_declared_entity(snap: AuditSnapshot, node_id: str) -> bool:
    """A declared class or instance, or an ontology entity: never a literal value."""
    node = snap.node_index.get(node_id)
    return (node is not None and node.kind in (NodeKind.CLASS, NodeKind.INSTANCE)) \
        or label(snap, node_id) in snap.ontology.entities


def _value_problem(snap: AuditSnapshot, obj: str, contract: Mapping[str, object]
                   ) -> tuple[str, str] | None:
    token = label(snap, obj)
    if is_declared_entity(snap, obj):
        return "not_literal", "is a declared entity, not a literal value"
    datatype = str(contract["datatype"])
    parsed = parse_typed(token, datatype)
    if parsed.problem == "nonfinite":
        return "nonfinite", f"is not a finite {datatype}"
    if parsed.problem:
        return "lexical", f"is not a {datatype}"
    low, high = contract["minimum"], contract["maximum"]
    if isinstance(parsed.value, Decimal):
        if low is not None and parsed.value < low:  # type: ignore[operator]
            return "below_minimum", f"is below the minimum {low}"
        if high is not None and parsed.value > high:  # type: ignore[operator]
            return "above_maximum", f"is above the maximum {high}"
    allowed = contract["allowed"]
    if allowed and not any(_equal(token, a, datatype) for a in allowed):  # type: ignore[union-attr]
        return "not_allowed", "is not an allowed value"
    return None


def _equal(token: str, allowed: str, datatype: str) -> bool:
    return same_value(token, allowed) if datatype in ("decimal", "integer") else token == allowed


def _bad_value(ctx: RuleContext, edge: tuple[str, str, str], refs: list[RawRef],
               contract: Mapping[str, object], code: str, why: str) -> Finding:
    snap = ctx.snapshot
    t = [label(snap, n) for n in edge]
    return ctx.finding(
        f"{t[2]!r} as the object of {t[1]!r} {why}", affected=(assertion_id(*t),),
        locations=tuple(refs[:1]), discriminator=code,
        witness={"triple": t, "lexical": t[2], "problem": code,
                 "datatype": contract["datatype"],
                 "minimum": None if contract["minimum"] is None else str(contract["minimum"]),
                 "maximum": None if contract["maximum"] is None else str(contract["maximum"]),
                 "allowed": list(contract["allowed"])[:_WITNESS]},  # type: ignore[call-overload]
        suggestion="correct the value or the relation's value contract")


# --- CON003 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "CON003", "1", "constraint", "Cardinality", Severity.ERROR, phase="C",
    prerequisites=_PRE, views=VIEWS, references=_REFERENCES,
    basis="distinct objects per subject of each relation with a declared cardinality, equal "
          "when lexically identical (or numerically equal under equality: value); a maximum is "
          "checked for every subject; a minimum for the subjects that state the relation and "
          "the typed instances of its domain, and violated only when the relation is declared "
          "complete. OntoDisc assertions carry no context, so counts are per subject"))
def con003(ctx: RuleContext) -> Outcome:
    sem = _sem(ctx)
    relations = sorted({*sem.subjects(DeclKey.MIN_COUNT), *sem.subjects(DeclKey.MAX_COUNT)})
    if not relations:
        return Outcome.unavailable("cardinality")
    values = _values_by_subject(ctx, set(relations))
    types = typing(ctx.snapshot, ctx.view)
    findings, eligible, unverified = [], 0, 0
    for r in relations:
        scope = _scope(ctx.snapshot, types, sem, r, values.get(r, {}))
        eligible += len(scope)
        new, open_ = _cardinality(ctx, sem, r, scope, values.get(r, {}))
        findings += new
        unverified += open_
    return Outcome(eligible=eligible, findings=tuple(findings),
                   metrics={"subjects_checked": eligible, "minimum_unverified": unverified})


def _values_by_subject(ctx: RuleContext, relations: set[str]
                       ) -> dict[str, dict[str, dict[str, list[RawRef]]]]:
    """relation -> subject -> object -> rows."""
    out: dict[str, dict[str, dict[str, list[RawRef]]]] = {}
    for (head, predicate, obj), refs in sorted(_edges(ctx).items()):
        if predicate in relations:
            out.setdefault(predicate, {}).setdefault(head, {})[obj] = refs
    return out


def _scope(snap: AuditSnapshot, types: Typing, sem: SemanticsDecl, relation: str,
           stated: Mapping[str, object]) -> list[str]:
    domain = set(sem.values(DeclKey.DOMAIN, relation))
    instances = {n.node_id for n in snap.nodes
                 if n.namespace == "entity" and domain & set(types.types(n.node_id))} \
        if domain and sem.find(DeclKey.MIN_COUNT, relation) else set()
    return sorted(set(stated) | instances)


def _cardinality(ctx: RuleContext, sem: SemanticsDecl, relation: str, scope: list[str],
                 stated: Mapping[str, Mapping[str, list[RawRef]]]) -> tuple[list[Finding], int]:
    snap = ctx.snapshot
    equality = sem.find(DeclKey.EQUALITY, relation)
    by_value = equality is not None and equality.value == "value"
    low, high = sem.find(DeclKey.MIN_COUNT, relation), sem.find(DeclKey.MAX_COUNT, relation)
    complete = sem.flag(DeclKey.COMPLETE, relation)
    findings, below = [], []
    for subject in scope:
        objects = stated.get(subject, {})
        distinct = {value_key(label(snap, o)) if by_value else o for o in objects}
        if high is not None and len(distinct) > int(high.value):
            findings.append(_count(ctx, relation, subject, objects, len(distinct),
                                   f"more than {high.value}", "max"))
        if low is not None and len(distinct) < int(low.value):
            below.append(subject)
            if complete:
                findings.append(_count(ctx, relation, subject, objects, len(distinct),
                                       f"fewer than {low.value}", "min"))
    if below and not complete:
        findings.append(ctx.finding(
            f"{len(below)} subjects state fewer than {low.value} values of "  # type: ignore[union-attr]
            f"{label(snap, relation)!r}; the relation is not declared complete, so absence is "
            "not a violation", severity=Severity.INFO, affected=(relation,),
            discriminator="min_unverified",
            witness={"relation": label(snap, relation), "subjects": _labels(snap, below),
                     "count": len(below)},
            suggestion="declare cardinality.complete when the view states every value"))
    return findings, 0 if complete else len(below)


def _count(ctx: RuleContext, relation: str, subject: str, objects: Mapping[str, list[RawRef]],
           distinct: int, bound: str, code: str) -> Finding:
    snap = ctx.snapshot
    return ctx.finding(
        f"{label(snap, subject)!r} has {distinct} distinct values of {label(snap, relation)!r}, "
        f"{bound}", affected=(subject, relation), discriminator=code,
        locations=tuple(refs[0] for refs in list(objects.values())[:_WITNESS]),
        witness={"subject": label(snap, subject), "relation": label(snap, relation),
                 "values": _labels(snap, sorted(objects))},
        suggestion="these values may differ by a context the triples do not record; qualify "
                   "them, or correct the cardinality")


# --- CON004 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "CON004", "1", "constraint", "Disjoint classes", Severity.ERROR, phase="C",
    prerequisites=_PRE, views=HIERARCHY_VIEWS, references=_REFERENCES,
    basis="each node's types closed under declared subclass edges, and each class's "
          "superclasses, intersected with every declared group of disjoint classes; two members "
          "of one group make an instance inconsistent and a class unable to have instances"))
def con004(ctx: RuleContext) -> Outcome:
    sem = _sem(ctx)
    groups: dict[str, list[str]] = {}
    for d in sem.of(DeclKey.DISJOINT):
        if d.value not in groups.setdefault(d.subject, []):
            groups[d.subject].append(d.value)
    if not groups:
        return Outcome.unavailable("disjoint")
    types, snap = typing(ctx.snapshot, ctx.view), ctx.snapshot
    classes = sorted({c for g in groups.values() for c in g}
                     | {n.node_id for n in snap.nodes if n.kind is NodeKind.CLASS}
                     | set(types.parents) | {p for ps in types.parents.values() for p in ps})
    findings = [f for node in types.typed() for f in _clash(ctx, groups, types.types(node), node, False)]
    findings += [f for cls in classes for f in _clash(ctx, groups, types.superclasses(cls), cls, True)]
    return Outcome(eligible=len(types.typed()) + len(classes), findings=tuple(findings),
                   metrics={"groups": len(groups), "typed_nodes": len(types.typed()),
                            "classes": len(classes)})


def _clash(ctx: RuleContext, groups: Mapping[str, list[str]], held: Mapping[str, list[str]],
           node: str, is_class: bool) -> Iterator[Finding]:
    snap = ctx.snapshot
    for group, members in sorted(groups.items()):
        hits = [m for m in members if m in held]
        if len(hits) < 2:
            continue
        name = label(snap, node)
        message = (f"class {name!r} is below disjoint classes {', '.join(map(repr, _labels(snap, hits)))}, "
                   "so it can have no instance" if is_class else
                   f"{name!r} is an instance of disjoint classes "
                   f"{', '.join(map(repr, _labels(snap, hits)))}")
        yield ctx.finding(
            message, severity=Severity.WARNING if is_class else Severity.ERROR,
            affected=(node, *hits), discriminator=f"{'class' if is_class else 'instance'}:{group}",
            witness={"node": name, "disjoint": _labels(snap, members),
                     "derivations": {label(snap, h): _labels(snap, held[h]) for h in hits}},
            suggestion="remove one of the type or subclass statements, or the disjointness")


# --- CON005 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "CON005", "1", "constraint", "Irreflexive relations", Severity.ERROR, phase="C",
    prerequisites=_PRE, views=VIEWS, references=_REFERENCES,
    basis="assertions of relations declared irreflexive whose subject and object are the same "
          "node; asymmetric relations, which are irreflexive too, are checked by STR006, "
          "including their self loops"))
def con005(ctx: RuleContext) -> Outcome:
    sem = _sem(ctx)
    relations = {d.subject for d in sem.of(DeclKey.IRREFLEXIVE) if d.value == "true"}
    if not relations:
        return Outcome.unavailable("irreflexive")
    snap, findings, eligible = ctx.snapshot, [], 0
    for (head, predicate, obj), refs in sorted(_edges(ctx).items()):
        if predicate not in relations:
            continue
        eligible += 1
        if head == obj:
            findings.append(ctx.finding(
                f"{label(snap, predicate)!r} is declared irreflexive but relates "
                f"{label(snap, head)!r} to itself", affected=(head, predicate),
                locations=tuple(refs[:1]), discriminator="self_loop",
                witness={"triple": [label(snap, n) for n in (head, predicate, obj)]},
                suggestion="remove the assertion, or the irreflexive declaration"))
    return Outcome(eligible=eligible, findings=tuple(findings))


# --- CON006 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "CON006", "1", "constraint", "Materialized entailments", Severity.ERROR, phase="C",
    prerequisites=_PRE, views=VIEWS, parameters={"max_steps": 2_000_000},
    parameter_bounds={"max_steps": (1, None)}, references=_REFERENCES,
    basis="for relations declared to materialize inverse, symmetric or transitive edges: each "
          "edge an assertion entails (the declared inverse, the reverse edge, or an edge to every "
          "node reachable through the relation, cycles included) must be stated in the view; "
          "inverse and reverse edges to literals are not required"))
def con006(ctx: RuleContext) -> Outcome:
    sem = _sem(ctx)
    contracts = sorted((d.subject, d.value) for d in sem.of(DeclKey.MATERIALIZE))
    if not contracts:
        return Outcome.unavailable("materialize")
    edges = _edges(ctx)
    stated = set(edges)
    budget = Budget(int(ctx.params["max_steps"]))
    required: list[tuple[tuple[str, str, str], list[tuple[str, str, str]], str]] = []
    for relation, kind in contracts:
        required += _entailed(ctx.snapshot, sem, relation, kind, stated, budget)
        if budget.exhausted:
            break
    findings = [_missing(ctx, edge, why, kind, edges) for edge, why, kind in required
                if edge not in stated]
    partial = f"stopped after {budget.limit} steps (max_steps)" if budget.exhausted else None
    return Outcome(eligible=len(required), findings=tuple(findings), partial=partial,
                   metrics={"required_edges": len(required), "missing": len(findings),
                            "steps": budget.spent})


def _entailed(snap: AuditSnapshot, sem: SemanticsDecl, relation: str, kind: str,
              stated: set[tuple[str, str, str]], budget: Budget
              ) -> list[tuple[tuple[str, str, str], list[tuple[str, str, str]], str]]:
    own = sorted(e for e in stated if e[1] == relation)
    if kind == "symmetric":
        return [((o, relation, h), [(h, r, o)], kind) for h, r, o in own
                if h != o and not is_literal(snap, o)]
    if kind == "inverse":
        partner = _inverse(sem, relation)
        return [((o, partner, h), [(h, r, o)], kind) for h, r, o in own
                if partner is not None and not is_literal(snap, o)]
    return _transitive(relation, own, budget)


def _inverse(sem: SemanticsDecl, relation: str) -> str | None:
    for a, b in sem.inverse_pairs():
        if relation in (a, b):
            return b if a == relation else a
    return None


def _transitive(relation: str, own: list[tuple[str, str, str]], budget: Budget
                ) -> list[tuple[tuple[str, str, str], list[tuple[str, str, str]], str]]:
    succ: dict[str, list[str]] = {}
    for h, _, o in own:
        succ.setdefault(h, []).append(o)
    out = []
    for start in sorted(succ):
        previous: dict[str, str] = {}
        frontier = list(succ[start])
        for n in frontier:
            previous.setdefault(n, start)
        while frontier and budget.spend(len(frontier)):
            nxt = []
            for node in frontier:
                for o in succ.get(node, ()):
                    if o not in previous:
                        previous[o] = node
                        nxt.append(o)
            frontier = nxt
        for target in sorted(previous):
            if previous[target] != start:
                out.append(((start, relation, target), _chain(start, target, previous, relation),
                            "transitive"))
        if budget.exhausted:
            break
    return out


def _chain(start: str, target: str, previous: Mapping[str, str], relation: str
           ) -> list[tuple[str, str, str]]:
    nodes, current = [target], target
    while (current := previous[current]) != start:
        nodes.append(current)
    nodes.append(start)
    nodes.reverse()
    return [(a, relation, b) for a, b in zip(nodes, nodes[1:])]


def _missing(ctx: RuleContext, edge: tuple[str, str, str], why: list[tuple[str, str, str]],
             kind: str, edges: Mapping[tuple[str, str, str], list[RawRef]]) -> Finding:
    snap = ctx.snapshot
    shown = [label(snap, n) for n in edge]
    return ctx.finding(
        f"{shown} is entailed ({kind}) but not stated", affected=edge,
        discriminator=f"{kind}:{edge[0]}->{edge[2]}",  # identity sorts ids; keep the direction
        locations=tuple(edges[e][0] for e in why if e in edges)[:_WITNESS],
        witness={"required": shown, "kind": kind,
                 "entailed_by": [[label(snap, n) for n in e] for e in why[:_WITNESS]]},
        suggestion="state the entailed edge, or drop the materialize declaration")
