# SPDX-License-Identifier: Apache-2.0
"""Reason-chain construction for entity-comparison verdicts.

When the gate emits a geometry-path verdict, an auditor needs the
*evidence* for that verdict, not just ``te=X``. The reason chain
exposes:

  * Top-K incident triples for each entity — the surrounding graph
    context the geometry was trained on.
  * Shared ``(relation, target)`` pairs — structural agreement
    signals; agreement-class triples in the labeling pipeline.
  * Opposing-relation patterns — pairs of relations that the corpus
    declares opposite_of, where ``a`` is on one side and ``b`` on
    the other for some shared target. These are the contradiction
    signals the geometry was trained to push apart.
  * Counts of agreement vs contradiction signals.

Construction is pure graph lookup — no embeddings, deterministic,
fast (typically < 1 ms for stores up to ~10K triples).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


_OPPOSITE_OF_RELATION = "opposite_of"
_SCHEMA_RELATIONS = frozenset({"opposite_of", "is_functional"})


@dataclass
class ReasonChain:
    """Audit-grade evidence for one entity-comparison verdict."""

    entity_a: str
    entity_b: str
    a_top_triples: list[tuple[str, str, str]] = field(default_factory=list)
    b_top_triples: list[tuple[str, str, str]] = field(default_factory=list)
    shared_targets: list[tuple[str, str]] = field(default_factory=list)
    opposing_signals: list[tuple[str, str, str]] = field(default_factory=list)
    n_shared_targets: int = 0
    n_opposing_signals: int = 0

    def summary(self) -> str:
        return (
            f"ReasonChain({self.entity_a} vs {self.entity_b}: "
            f"shared={self.n_shared_targets}, "
            f"opposing={self.n_opposing_signals}, "
            f"a_triples={len(self.a_top_triples)}, "
            f"b_triples={len(self.b_top_triples)})"
        )

    def to_dict(self) -> dict:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "a_top_triples": [list(t) for t in self.a_top_triples],
            "b_top_triples": [list(t) for t in self.b_top_triples],
            "shared_targets": [list(t) for t in self.shared_targets],
            "opposing_signals": [list(t) for t in self.opposing_signals],
            "n_shared_targets": self.n_shared_targets,
            "n_opposing_signals": self.n_opposing_signals,
        }


def _opposing_relation_pairs(triples) -> set[tuple[str, str]]:
    """Symmetric (r_a, r_b) pairs declared via ``opposite_of``."""
    pairs: set[tuple[str, str]] = set()
    for h, r, t in triples:
        if r == _OPPOSITE_OF_RELATION and h != t:
            pairs.add((h, t))
            pairs.add((t, h))
    return pairs


def build_reason_chain(
    a: str, b: str,
    store: "GMSExpertStore",
    *,
    top_k_triples: int = 5,
) -> ReasonChain:
    """Compute the reason chain for an A vs B comparison.

    All work is graph lookup — no model forward pass. Safe to call
    on the runtime hot path (typical: < 1 ms for ~10K triples).
    """
    doc_graph = getattr(store, "doc_graph", None)
    if doc_graph is None:
        return ReasonChain(entity_a=a, entity_b=b)
    triples = list(doc_graph.triples)

    a_incident: list[tuple[str, str, str]] = []
    b_incident: list[tuple[str, str, str]] = []
    a_rt: dict[str, set[str]] = defaultdict(set)
    b_rt: dict[str, set[str]] = defaultdict(set)

    for h, r, t in triples:
        if r in _SCHEMA_RELATIONS:
            continue
        if h == a:
            a_incident.append((h, r, t))
            a_rt[r].add(t)
        elif t == a:
            a_incident.append((h, r, t))
        if h == b:
            b_incident.append((h, r, t))
            b_rt[r].add(t)
        elif t == b:
            b_incident.append((h, r, t))

    # Shared (relation, target) pairs: agreement evidence.
    shared: list[tuple[str, str]] = []
    for r, a_targets in a_rt.items():
        common = a_targets & b_rt.get(r, set())
        for t in sorted(common):
            shared.append((r, t))

    # Opposing-relation evidence: relations r1, r2 declared
    # opposite_of with (a, r1, X) and (b, r2, X) on a shared target.
    opposing_rel_pairs = _opposing_relation_pairs(triples)
    opposing: list[tuple[str, str, str]] = []
    for r1, a_targets in a_rt.items():
        for r2 in {r2 for (s, r2) in opposing_rel_pairs if s == r1}:
            common = a_targets & b_rt.get(r2, set())
            for t in sorted(common):
                opposing.append((r1, r2, t))

    return ReasonChain(
        entity_a=a,
        entity_b=b,
        a_top_triples=a_incident[:top_k_triples],
        b_top_triples=b_incident[:top_k_triples],
        shared_targets=shared,
        opposing_signals=opposing,
        n_shared_targets=len(shared),
        n_opposing_signals=len(opposing),
    )
