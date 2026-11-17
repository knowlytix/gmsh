# SPDX-License-Identifier: Apache-2.0
"""Taxonomy semantics: the class hierarchy a view states under declared predicates.

OntoDisc stores record no hierarchy. A view's triples become subclass edges only
when the audit schema sidecar declares their predicate a subclass predicate
(``hierarchy.subclass_predicates``), and instance-of edges only under a declared
type predicate or a term's declared ``types``. Inferred declarations take no
effect. Nothing is guessed from predicate names.

Traversals are iterative, so deep hierarchies cannot exhaust the recursion
limit, and the expensive ones spend a shared ``Budget``: a rule that runs out
reports a partial result instead of a count it did not finish.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    GraphView,
    NodeKind,
    RawRef,
)

# The views a hierarchy can be read from: the three store views and an ontology proposal.
HIERARCHY_VIEWS = (GraphView.STORED, GraphView.CONFORMANT, GraphView.QUARANTINED,
                   GraphView.PROPOSAL)
SUBCLASS_PREDICATES = "subclass_predicates"  # missing prerequisite reported by hierarchy rules
N = TypeVar("N", bound=Hashable)


@dataclass
class Budget:
    """A work limit shared by the traversals of one rule execution."""

    limit: int
    spent: int = 0

    def spend(self, n: int = 1) -> bool:
        """Charge ``n`` steps; False once the limit is exceeded."""
        self.spent += n
        return self.spent <= self.limit

    @property
    def exhausted(self) -> bool:
        return self.spent > self.limit


@dataclass(frozen=True)
class Hierarchy:
    """Subclass and instance-of edges of one view, by node id.

    ``parents[c]`` maps each direct parent of ``c`` to the raw rows stating the
    edge, in artifact order. ``types[i]`` lists the classes ``i`` is declared or
    asserted an instance of. ``predicates`` are the accepted subclass predicates.
    """

    predicates: tuple[str, ...]
    parents: Mapping[str, Mapping[str, tuple[RawRef, ...]]]
    types: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def nodes(self) -> list[str]:
        """Every node incident to a subclass edge, sorted."""
        return sorted({*self.parents, *(p for ps in self.parents.values() for p in ps)})

    @property
    def edges(self) -> list[tuple[str, str]]:
        """(child, parent) pairs, sorted."""
        return sorted((c, p) for c, ps in self.parents.items() for p in ps)

    def children(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for c, p in self.edges:
            out.setdefault(p, []).append(c)
        return out

    def successors(self) -> dict[str, list[str]]:
        """Child to sorted parents, for every node with a parent."""
        return {c: sorted(ps) for c, ps in self.parents.items()}

    def refs(self, child: str, parent: str) -> tuple[RawRef, ...]:
        return self.parents.get(child, {}).get(parent, ())


def accepted_predicates(snapshot: AuditSnapshot, key: DeclKey) -> tuple[str, ...]:
    """Relation node ids declared (and accepted) under ``key``."""
    sem = snapshot.semantics.accepted()
    return tuple(d.subject for d in sem.of(key) if d.value == "true")


def hierarchy(snapshot: AuditSnapshot, view: GraphView) -> Hierarchy:
    """The view's subclass edges under the declared predicates, and its instance types.

    Edges to literals are ignored: a number or a rule text is not a class.
    """
    subclass = set(accepted_predicates(snapshot, DeclKey.SUBCLASS_PREDICATE))
    typing = set(accepted_predicates(snapshot, DeclKey.TYPE_PREDICATE))
    parents: dict[str, dict[str, list[RawRef]]] = {}
    types: dict[str, dict[str, None]] = {}
    for node in snapshot.nodes:
        for cls in node.types:
            types.setdefault(node.node_id, {})[cls] = None
    for row in snapshot.view(view):
        a = snapshot.assertion_index[row.assertion_id]
        if _is_literal(snapshot, a.object):
            continue
        if a.predicate in subclass:
            parents.setdefault(a.head, {}).setdefault(a.object, []).append(row.raw)
        elif a.predicate in typing:
            types.setdefault(a.head, {})[a.object] = None
    return Hierarchy(tuple(sorted(subclass)),
                     {c: {p: tuple(r) for p, r in sorted(ps.items())} for c, ps in sorted(parents.items())},
                     {i: tuple(cs) for i, cs in sorted(types.items())})


def _is_literal(snapshot: AuditSnapshot, node_id: str) -> bool:
    node = snapshot.node_index.get(node_id)
    return node is not None and node.kind is NodeKind.LITERAL


# --- graph algorithms -------------------------------------------------------------------
def strongly_connected(nodes: Iterable[str], successors: Mapping[str, list[str]]) -> list[list[str]]:
    """Strongly connected components (iterative Tarjan), members sorted, in sorted order."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    for root in sorted(nodes):
        if root in index:
            continue
        work = [(root, iter(successors.get(root, ())))]
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            nxt = next(it, None)
            if nxt is None:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    out.append(_pop_component(stack, on_stack, node))
            elif nxt not in index:
                index[nxt] = low[nxt] = len(index)
                stack.append(nxt)
                on_stack.add(nxt)
                work.append((nxt, iter(successors.get(nxt, ()))))
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
    return sorted(out)


def _pop_component(stack: list[str], on_stack: set[str], node: str) -> list[str]:
    members = []
    while True:
        member = stack.pop()
        on_stack.discard(member)
        members.append(member)
        if member == node:
            return sorted(members)


def cyclic_components(h: Hierarchy) -> list[list[str]]:
    """Components holding a cycle: two or more members, or one member that is its own parent."""
    succ = h.successors()
    return [c for c in strongly_connected(h.nodes, succ)
            if len(c) > 1 or c[0] in succ.get(c[0], ())]


@dataclass(frozen=True)
class Condensation:
    """The hierarchy with each strongly connected component as one node.

    ``of`` maps a class to its component index; ``parents[c]`` maps each parent
    component of component ``c`` to the original edges between them. Mutually
    reachable classes are equivalent under subclass semantics, so redundancy,
    inheritance and top-level questions are answered per component.
    """

    components: list[list[str]]
    of: dict[str, int]
    parents: dict[int, dict[int, list[tuple[str, str]]]]

    def successors(self) -> dict[int, list[int]]:
        return {c: sorted(ps) for c, ps in self.parents.items()}


def condense(h: Hierarchy) -> Condensation:
    components = strongly_connected(h.nodes, h.successors())
    of = {m: i for i, comp in enumerate(components) for m in comp}
    parents: dict[int, dict[int, list[tuple[str, str]]]] = {}
    for c, p in h.edges:
        if of[c] != of[p]:
            parents.setdefault(of[c], {}).setdefault(of[p], []).append((c, p))
    return Condensation(components, of, parents)


def reachable(start: N, successors: Mapping[N, list[N]], budget: Budget) -> set[N]:
    """Nodes reachable from ``start`` (itself included); one budget step per edge scanned."""
    seen, stack = {start}, [start]
    while stack:
        for nxt in successors.get(stack.pop(), ()):
            if not budget.spend():
                return seen
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def shortest_path(start: N, goal: N, successors: Mapping[N, list[N]], budget: Budget, *,
                  skip: tuple[N, N] | None = None,
                  within: set[N] | None = None) -> list[N] | None:
    """A shortest path ``start`` -> ``goal`` (breadth first), or None.

    ``skip`` is one edge not to use; ``within`` confines the search to a node
    set. Returns None when no path exists or the budget runs out; check
    ``budget.exhausted`` to tell them apart.
    """
    previous: dict[N, N | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in successors.get(node, ()):
            if not budget.spend():
                return None
            if (node, nxt) == skip or nxt in previous or (within is not None and nxt not in within):
                continue
            previous[nxt] = node
            if nxt == goal:
                return _walk_back(previous, goal)
            queue.append(nxt)
    return None


def _walk_back(previous: Mapping[N, N | None], goal: N) -> list[N]:
    path: list[N] = [goal]
    while (step := previous[path[-1]]) is not None:
        path.append(step)
    return path[::-1]


def cycle_through(node: str, successors: Mapping[str, list[str]], members: set[str],
                  budget: Budget) -> list[str] | None:
    """A shortest cycle starting and ending at ``node`` inside one component.

    If the budget runs out, the shortest cycle found so far, or None.
    """
    if node in successors.get(node, ()):
        return [node, node]
    best: list[str] | None = None
    for first in successors.get(node, ()):
        if first not in members:
            continue
        back = shortest_path(first, node, successors, budget, within=members)
        if back is not None and (best is None or len(back) + 1 < len(best)):
            best = [node, *back]
    return best
