# SPDX-License-Identifier: Apache-2.0
"""Structure rules STR001-STR002: unused declared terms and disconnected components.

Topology is computed over entity nodes only. An edge to a literal (a number, a
date, rule text) is a value edge: it counts towards a term's use but never
connects two entities, so shared values cannot join unrelated parts of the graph.
Modules declared in the audit schema sidecar are expected to be separate: a
component whose entities are all members of one declared module does not count
towards ``expected_components``, and a module whose members fall into several
components is reported as split. A component with any undeclared member is not
a module, so declaring one entity cannot exempt the rest.
"""

from __future__ import annotations

from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    Finding,
    GraphView,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import (
    ONTOLOGY,
    Outcome,
    RuleContext,
    RuleSpec,
    builtin,
)
from knowlytix.knowledge.kgcheck.rules._common import (
    entity_resolver,
    is_literal,
    rows_by_assertion,
    term_ref,
    triple,
)

_VIEWS = (GraphView.STORED, GraphView.CONFORMANT, GraphView.QUARANTINED, GraphView.PROPOSAL)
_WITNESS_MEMBERS = 50


def _edges(snapshot: AuditSnapshot, view: GraphView) -> tuple[list[tuple[str, str]], list[str]]:
    """Entity-entity edges and value-edge heads of ``view``, by raw token."""
    entity_edges, value_heads = [], []
    for aid in rows_by_assertion(snapshot, view):
        h, _, o = triple(snapshot, aid)
        if is_literal(snapshot, snapshot.assertion_index[aid].object):
            value_heads.append(h)
        else:
            entity_edges.append((h, o))
    return entity_edges, value_heads


# --- STR001 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "STR001", "1", "structure", "Declared entities unused in the view", Severity.WARNING,
    prerequisites=(ONTOLOGY,), views=_VIEWS, parameters={"standalone": ()},
    basis="degree of each declared entity over the view's triples, matching tokens by name or "
          "alias; entity and value edges are counted separately"))
def str001(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    resolver, (entity_edges, value_heads) = entity_resolver(snap), _edges(snap, ctx.view)
    degree: dict[str, list[int]] = {t: [0, 0] for t in snap.ontology.entities}
    for h, o in entity_edges:
        for term in {*resolver.targets(h), *resolver.targets(o)}:
            degree[term][0] += 1
    for h in value_heads:
        for term in resolver.targets(h):
            degree[term][1] += 1
    exempt = set(ctx.params["standalone"])
    findings = [_unused(ctx, t) for t, (e, v) in degree.items() if e + v == 0 and t not in exempt]
    return Outcome(eligible=len(degree), findings=tuple(findings), metrics={
        "declared_entities": len(degree), "unused": len(findings),
        "value_edges_only": sum(1 for e, v in degree.values() if e == 0 and v > 0),
        "with_entity_edges": sum(1 for e, _ in degree.values() if e > 0)})


def _unused(ctx: RuleContext, term: str) -> Finding:
    return ctx.finding(f"declared entity {term!r} occurs in no {ctx.view.value} triple",
                       affected=(f"entity:{term}",), locations=(term_ref("entity", term),),
                       suggestion="remove the term, list it as standalone, or check extraction "
                                  "for the section that should mention it")


# --- STR002 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "STR002", "3", "structure", "Disconnected components", Severity.WARNING,
    views=_VIEWS, parameters={"expected_components": 1},
    parameter_bounds={"expected_components": (1, None)},
    basis="undirected connected components over entity-entity edges (iterative union-find); "
          "nodes without entity edges are counted as isolated, not as components; components "
          "whose members all belong to one declared module are expected"))
def str002(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    entity_edges, value_heads = _edges(snap, ctx.view)
    nodes = sorted({t for e in entity_edges for t in e} | set(value_heads))
    components = _components(entity_edges)
    isolated = sorted(set(nodes) - {t for c in components for t in c})
    modules = _modules(snap)
    found = [_modules_of(c, modules) for c in components]
    expected = [len(m) == 1 and all(x in modules for x in c) for m, c in zip(found, components)]
    unexplained = [i for i, ok in enumerate(expected) if not ok]
    findings: list[Finding] = []
    if len(unexplained) > int(ctx.params["expected_components"]):  # module components are expected
        findings = [_component(ctx, i, components[i], components) for i in unexplained]
    findings += [_split(ctx, name, [i for i, f in enumerate(found) if name in f], components, modules)
                 for name in sorted({m for f in found for m in f})
                 if sum(name in f for f in found) > 1]
    sizes = [len(c) for c in components]
    return Outcome(eligible=len(nodes), findings=tuple(findings), metrics={
        "nodes": len(nodes), "components": len(components), "component_sizes": sizes,
        "largest_component_size": max(sizes, default=0),
        "largest_component_ties": sizes.count(max(sizes)) if sizes else 0,
        "declared_modules": len(set(modules.values())), "module_components": sum(expected),
        "isolated": len(isolated), "isolated_sample": isolated[:_WITNESS_MEMBERS]})


def _modules(snapshot: AuditSnapshot) -> dict[str, str]:
    """Declared (accepted) module of each entity token."""
    sem = snapshot.semantics.accepted()
    return {d.subject.partition(":")[2]: d.value for d in sem.of(DeclKey.MODULE)
            if d.subject.startswith("entity:")}


def _modules_of(members: list[str], modules: dict[str, str]) -> frozenset[str]:
    return frozenset(modules[m] for m in members if m in modules)


def _split(ctx: RuleContext, name: str, parts: list[int], all_: list[list[str]],
           modules: dict[str, str]) -> Finding:
    return ctx.finding(f"declared module {name!r} is split across {len(parts)} components",
                       affected=tuple(f"entity:{m}" for i in parts for m in all_[i]
                                      if modules.get(m) == name),
                       witness={"module": name, "component_sizes": [len(all_[i]) for i in parts]},
                       suggestion="link the module's parts, or declare them as separate modules",
                       discriminator=f"module_split:{name}")


def _components(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Components with at least one edge, largest first, members sorted."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g))


def _component(ctx: RuleContext, i: int, members: list[str], all_: list[list[str]]) -> Finding:
    return ctx.finding(f"component {i + 1} of {len(all_)} has {len(members)} entities "
                       f"(largest has {len(all_[0])})",
                       affected=tuple(f"entity:{m}" for m in members),
                       witness={"size": len(members), "members": members[:_WITNESS_MEMBERS],
                                "component_sizes": [len(c) for c in all_]},
                       suggestion="link the modules, or set expected_components if they are "
                                  "intentionally separate", discriminator="component")
