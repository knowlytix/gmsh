# SPDX-License-Identifier: Apache-2.0
"""Tier-3 co-reference by link-prediction equivalence (geometry-native).

Two entities are the same when the trained GMS cannot tell them apart by their
relational behavior: over the union of relations either participates in, the
model predicts the *same object* for both. This is a sharper test than v-space
cosine — which measures semantic proximity and so conflates "similar" with
"same" (``GradientBoosting`` sits near ``XGBoost``) — because it keys on the
predicted tails, not the embedding distance: two surfaces of one entity complete
the same facts, while two merely-similar entities disagree on at least one
relation and are kept apart.

The criterion (``e1``, ``e2`` co-referent) is:

* **no conflict** — for every relation both assert, their observed tail sets
  intersect (they never name different objects for the same relation);
* **link-prediction profile agreement** — for every relation in the union, the
  model's predicted tail for ``e1`` and for ``e2`` coincide (top-k / admissible);
* **enough evidence** — at least ``min_shared`` shared relations ground the
  match; a pair too sparse to fingerprint abstains (flagged, never guessed).

Grounding the agreement in the shared relations (not the geometry's own
placement) guards against the circularity of merging two nodes the model put near
merely because their names look alike: the shared relations must actually agree,
and a disagreement anywhere vetoes.

This proposer only *proposes*. It emits an
:class:`~knowlytix.knowledge.geode.ontology.map.OntologyMap`; the symbolic vetoes,
the surface-origin provenance bridge and the Phase-0 harness invariants gate it
exactly as for the deterministic tier. It needs a trained model, so unlike tier 1
it is not a pure function.
"""

from __future__ import annotations

import torch

from knowlytix.core.geometry.sphere import geodesic_distance
from knowlytix.knowledge.geode.ontology.map import OntologyMap

__all__ = ["linkpred_entity_merges"]

Triple = tuple[str, str, str]

# Relations that carry a surface/structural role, not a fact — excluded from the
# behavioral fingerprint (same exclusions the rest of the ontology layer uses).
_NON_FACT_RELATIONS = frozenset({
    "has_alias", "has_name", "has_policy_name", "in_section", "is_functional",
    "opposite_of", "passes", "fails", "is_weak", "is_strong",
})

_NUM_RE = None  # lazy; a value entity is never a merge candidate


def _is_value(name: str) -> bool:
    import re
    global _NUM_RE
    if _NUM_RE is None:
        _NUM_RE = re.compile(r"^-?[\d,]*\.?\d+%?$")
    return bool(_NUM_RE.match(name.strip()))


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self):
        out: dict = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return [c for c in out.values() if len(c) > 1]


def _fact_profile(triples, adapter):
    """``entity -> {relation -> {tail entities}}`` over fact relations, indices."""
    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    prof: dict[str, dict[str, set[str]]] = {}
    for h, r, t in triples:
        if r in _NON_FACT_RELATIONS or h not in e2i or t not in e2i:
            continue
        prof.setdefault(h, {}).setdefault(r, set()).add(t)
    return prof


def _predicted_tails(model, adapter, v_all, e_idx, r_idx, dev, k):
    """Top-k link-predicted tail entities for ``(e_idx, r_idx)`` by geodesic."""
    v_h = model.dual_emb.project_v(torch.tensor([e_idx], device=dev))
    p = model.apply_relation(v_h, torch.tensor([r_idx], device=dev))
    d = geodesic_distance(p.expand(v_all.shape[0], -1), v_all).cpu()
    d[e_idx] = float("inf")            # never predict itself
    return set(int(i) for i in torch.argsort(d)[:k].tolist())


def _profiles_agree(model, adapter, v_all, e1, e2, rels, dev, k):
    """Do the link-prediction profiles of ``e1`` and ``e2`` coincide over
    ``rels``? Each relation's predicted tail must match (top-k overlap)."""
    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    i1, i2 = e2i[e1], e2i[e2]
    for r in rels:
        ri = r2i[r]
        p1 = _predicted_tails(model, adapter, v_all, i1, ri, dev, k)
        p2 = _predicted_tails(model, adapter, v_all, i2, ri, dev, k)
        if not (p1 & p2):
            return False
    return True


def linkpred_entity_merges(triples, model, adapter, *, min_shared: int = 1,
                           topk: int = 1, require_profile: bool = True):
    """Propose entity merges by link-prediction equivalence.

    Args:
        triples: current ``(head, relation, tail)`` list.
        model: a trained ``GeometricKnowledgeGraph`` over ``adapter``'s vocab.
        adapter: the ``GraphToGMS`` the model was built on.
        min_shared: minimum shared fact-relations to consider a pair (sparser
            pairs abstain).
        topk: predicted-tail neighborhood size (1 = strict argmin; >1 tolerates
            one-to-many relations).
        require_profile: also require the full link-prediction profile over the
            union of relations to agree (the geometric check); when False, only
            the asserted shared-relation agreement is used.

    Returns:
        An :class:`OntologyMap` with the entity merges, per-merge evidence
        (shared relations, mechanism), and the pairs flagged as too sparse. The
        map's :meth:`~OntologyMap.rewrite` applies it while preserving provenance.
    """
    dev = next(model.parameters()).device
    prof = _fact_profile(triples, adapter)
    cands = sorted(e for e in prof if not _is_value(e))
    omap = OntologyMap()
    if len(cands) < 2:
        return omap

    with torch.no_grad():
        v_all = torch.nn.functional.normalize(
            model.dual_emb.project_v(
                torch.arange(adapter.num_entities, device=dev)), dim=-1)

    uf = _UnionFind(cands)
    pair_ev: dict[tuple[str, str], dict] = {}
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            e1, e2 = cands[a], cands[b]
            ra, rb = prof[e1], prof[e2]
            shared = ra.keys() & rb.keys()
            # (1) conflict: a shared relation whose tail sets are disjoint.
            if any(not (ra[r] & rb[r]) for r in shared):
                continue
            # (3) evidence: enough shared relations to fingerprint, else abstain.
            if len(shared) < min_shared:
                omap.flagged.append({
                    "kind": "entity", "pair": [e1, e2],
                    "reason": "insufficient_shared_relations",
                    "shared": sorted(shared)})
                continue
            # (2) link-prediction profile agreement over the union of relations.
            union_rels = ra.keys() | rb.keys()
            if require_profile and not _profiles_agree(
                    model, adapter, v_all, e1, e2, union_rels, dev, topk):
                continue
            uf.union(e1, e2)
            pair_ev[(e1, e2)] = {"shared": sorted(shared),
                                 "union": sorted(union_rels)}

    counts: dict[str, int] = {}
    for h, _r, t in triples:
        counts[h] = counts.get(h, 0) + 1
        counts[t] = counts.get(t, 0) + 1
    for cluster in uf.clusters():
        canonical = sorted(cluster, key=lambda e: (-counts.get(e, 0), len(e), e))[0]
        members = sorted(c for c in cluster if c != canonical)
        for m in members:
            omap.entity[m] = canonical
        omap.evidence.append({
            "kind": "entity", "tier": "linkpred", "canonical": canonical,
            "members": members,
            "shared_relations": sorted({r for e in cluster
                                        for r in prof.get(e, {})})})
    return omap
