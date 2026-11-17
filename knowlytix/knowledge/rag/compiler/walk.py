# SPDX-License-Identifier: Apache-2.0
"""Store-grounded resolution of an emitted hop chain to a query-triple chain.

:class:`StoreChainWalker` holds the store index (the asserted ``(head, relation)``
facts, their tails and the unambiguous name-canonicalization maps) and walks an
emitted ``[(head, relation), ...]`` chain against it by EXACT lookup, hop by hop.
It is the resolution half of the query compiler, factored out of
:class:`~knowlytix.knowledge.rag.compiler.model.CompiledQueryParser` so the same
walk backs the frozen few-shot binder (Arm C) and the alias-table hybrid binder
(Arm D) of the bake-off without duplicating the abstention logic.

A chain that breaks at any hop -- a relation the current entity does not hold --
yields the empty list, so the pipeline abstains through its no-binding path rather
than drifting to a different head. A valid chain is returned as ``?x``-chained
:class:`QueryTriple`s (variables for the intermediate and asked slots).
"""

from __future__ import annotations

import re

from knowlytix.knowledge.rag.query_triples import ASKED, QueryTriple

__all__ = ["StoreChainWalker", "STRUCTURAL_RELATIONS"]

# Relation ids treated as structural graph scaffolding, excluded from walkable
# content facts. The store-builder's conventional structural edges.
#
# `has_feature` links a compound row entity (dti/prime) back to its base
# feature, and a table with a literal `Feature` column emits the same relation
# for its cells. Walking either composes a hop that restates a one-hop fact,
# and a row with a blank key cell makes it a self-loop (dti -has_feature-> dti).
# Kept in step with the same list in
# `knowlytix.benchmark.generators.relation_recall`.
STRUCTURAL_RELATIONS = {"in_section", "has_alias", "has_product",
                        "has_regulatory_section", "has_feature"}

_NUM = re.compile(r"^-?[\d,]*\.?\d+%?$")


def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


class StoreChainWalker:
    """Exact-lookup resolver of a hop chain against a GMS store.

    Args:
        store: a loaded :class:`GMSExpertStore`.
        structural_relations: relation ids excluded from walkable content facts;
            defaults to :data:`STRUCTURAL_RELATIONS`.
    """

    def __init__(self, store, *, structural_relations: "set[str] | None" = None):
        # Normalized on both sides: a relation derived from a column name
        # carries the source header's casing (`has_Feature`), so an exact-string
        # check lets it through while excluding the hardcoded `has_feature`.
        self._structural = {
            _norm(r) for r in (STRUCTURAL_RELATIONS if structural_relations is None
                               else structural_relations)
        }
        self._build_index(store)

    # -- store index -------------------------------------------------------

    def _build_index(self, store) -> None:
        triples = list(store.doc_graph.triples) if store.doc_graph else []
        self._facts: set[tuple[str, str]] = set()
        self._fact_tails: dict[tuple[str, str], list[str]] = {}
        ent_norm: dict[str, set] = {}
        rel_norm: dict[str, set] = {}
        for h, r, t in triples:
            if _norm(r) in self._structural:
                continue
            self._facts.add((h, r))
            self._fact_tails.setdefault((h, r), []).append(str(t))
            ent_norm.setdefault(_norm(h), set()).add(h)
            rel_norm.setdefault(_norm(r), set()).add(r)
        # Canonicalization maps only where a normalized key is unambiguous.
        self._ent_norm = ent_norm
        self._ent_canon = {k: next(iter(v)) for k, v in ent_norm.items() if len(v) == 1}
        self._rel_canon = {k: next(iter(v)) for k, v in rel_norm.items() if len(v) == 1}

    @property
    def entities(self) -> list[str]:
        """Canonical head entities that appear in a walkable content fact."""
        return sorted({h for h, _ in self._facts})

    def relations_of(self, head: str) -> list[str]:
        """Canonical relations asserted on ``head`` in a walkable content fact."""
        h = self.canon_entity(head)
        return sorted({r for hh, r in self._facts if hh == h})

    def holds(self, head: str, relation: str) -> bool:
        """Whether ``(head, relation)`` is an asserted content fact (canonicalized)."""
        return (self.canon_entity(head), self.canon_relation(relation)) in self._facts

    def canon_entity(self, e: str) -> str:
        return e if any(e == h for h, _ in self._facts) else self._ent_canon.get(_norm(e), e)

    def canon_relation(self, r: str) -> str:
        return self._rel_canon.get(_norm(r), r)

    # -- walk --------------------------------------------------------------

    def walk(self, hops: list[tuple[str, str]]) -> list[QueryTriple]:
        """Walk an emitted hop chain against the store by EXACT lookup. Returns the
        ``?x``-chained query triples if every hop is an asserted fact, else ``[]``."""
        if not hops:
            return []
        cur = self.canon_entity(hops[0][0])
        out: list[QueryTriple] = []
        for i, (_hd, rel) in enumerate(hops):
            r = self.canon_relation(rel)
            tails = self._fact_tails.get((cur, r))
            if not tails:                       # hop is not an asserted fact -> abstain
                return []
            is_final = i == len(hops) - 1
            head_slot = cur if i == 0 else f"?x{i - 1}"
            tail_slot = ASKED if is_final else f"?x{i}"
            out.append(QueryTriple(head_slot, r, tail_slot))
            if not is_final:
                nxt_rel = self.canon_relation(hops[i + 1][1])
                cont = [t for t in tails if (t, nxt_rel) in self._facts]
                nxt = cont[0] if cont else tails[0]
                if _NUM.match(str(nxt).strip()):  # cannot continue through a value
                    return []
                cur = nxt
        return out
