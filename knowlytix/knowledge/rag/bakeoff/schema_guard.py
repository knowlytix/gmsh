# SPDX-License-Identifier: Apache-2.0
"""Schema-validity guard for bound queries (bake-off invariant I3).

I3 requires that every arm emit only schema-valid relations and entity references
-- and that this hold *by construction*, not by a post-hoc repair that a wrong
binding could slip past. The enforcement mechanism the arms already share is
closed-vocabulary resolution: the compiler and the embedding/fuzzy binder resolve
each slot against the store's actual relations and entities (exact lookup or
nearest match), so a relation the model invents resolves to nothing and the query
does not bind -- an off-schema slot never becomes a bound query, it becomes an
abstention. This is the decode-time closed-vocabulary constraint; token-level
grammar-constrained generation is a stricter alternative but unnecessary when the
bound output is confined to the store vocabulary.

:class:`SchemaGuard` turns that into a checkable guarantee and the metric the G2
gate reads: :meth:`grammar_violation_rate` over a batch of bound queries must be
zero, and :meth:`enforce` returns ``None`` (abstain) for any query carrying an
off-schema slot rather than emitting it. Variable slots (``?``-prefixed unknowns)
are always admissible -- they are the asked/intermediate positions, not bindings.
"""

from __future__ import annotations

from dataclasses import dataclass


def _is_var(slot) -> bool:
    return isinstance(slot, str) and slot.startswith("?")


@dataclass
class SchemaGuard:
    """Closed-vocabulary validity check over the store's relations and entities."""

    relations: set
    entities: set

    @classmethod
    def from_store(cls, store, *, max_entities: int = 100000):
        """Build the guard from the store's schema. ``max_entities`` is raised well
        above the default sample so the entity set is complete for validity
        checking (the sampling in ``schema_from_store`` is for prompting, not
        gating)."""
        from knowlytix.knowledge.rag.query_triples import schema_from_store
        sch = schema_from_store(store, max_entities=max_entities)
        return cls(relations=set(sch["relations"]), entities=set(sch["entities"]))

    def valid_slot_head(self, head) -> bool:
        return _is_var(head) or head in self.entities

    def valid_hop(self, hop) -> bool:
        head, rel = hop[0], hop[1]
        return (rel in self.relations) and self.valid_slot_head(head)

    def is_valid(self, hops) -> bool:
        """A bound query is schema-valid when it is non-empty and every hop uses a
        schema relation and a schema entity (or a variable) in the head slot."""
        return bool(hops) and all(self.valid_hop(h) for h in hops)

    def enforce(self, hops):
        """Return ``hops`` if schema-valid, else ``None`` -- the arm abstains rather
        than emit an off-schema binding (I3: not repaired after)."""
        return hops if self.is_valid(hops) else None

    def grammar_violation_rate(self, hops_list) -> float:
        """Fraction of bound (non-abstained) queries carrying any off-schema slot.
        The G2 gate requires this to be zero over the emitted bindings."""
        emitted = [h for h in hops_list if h]
        if not emitted:
            return 0.0
        bad = sum(not self.is_valid(h) for h in emitted)
        return round(bad / len(emitted), 4)
