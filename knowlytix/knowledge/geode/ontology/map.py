# SPDX-License-Identifier: Apache-2.0
"""OntologyMap — the surface<->canonical identity artifact for GEODE.

The normalization / co-reference layer rewrites the extracted graph into a
canonical vocabulary for the geometry, but the downstream consumers that key on
surface identity must not break:

* **provenance** resolves a triple by matching its surface against the *source
  text*; if the graph now says ``fee`` but the document spells ``fees``, a naive
  lookup returns ``unaligned``;
* **ENM** keys on ``(entity, relation)`` — merging keys is wanted (the duplicate
  anchor finally sees a ``fees``/``fee`` conflict), but exact numeric *values*
  must pass through untouched.

:class:`OntologyMap` carries both directions so that translation is lossless: the
canonical id for the geometry, and — via :attr:`triple_origins` — the original
surface triples each canonical triple came from, so provenance can resolve
against a real source span. It is a plain data record (JSON-serializable via
:meth:`to_records`); it holds no model and does no I/O.

Only *non-identity* mappings are stored: a surface absent from :attr:`entity` /
:attr:`relation` is its own canonical. That keeps the map small and makes
:meth:`translate` a safe ``dict.get(x, x)`` — a surface the layer never merged is
returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["OntologyMap"]

Triple = tuple[str, str, str]


@dataclass
class OntologyMap:
    """A surface -> canonical map for entities and relations, plus provenance.

    Attributes:
        entity: ``surface -> canonical`` for merged entity surfaces only
            (identity mappings are omitted).
        relation: ``surface -> canonical`` for merged relation surfaces only.
        evidence: one audit record per applied merge group — ``{kind, canonical,
            members, tier, reason}`` — for review and reproducibility.
        flagged: surfaces the layer declined to merge (e.g. a lexical group split
            by a conflicting attribute), kept for review, never applied.
        triple_origins: ``canonical_triple -> [original surface triples]``. The
            provenance bridge: a canonical triple resolves to its source span
            through any of its origin surface triples.
    """

    entity: dict[str, str] = field(default_factory=dict)
    relation: dict[str, str] = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    flagged: list[dict] = field(default_factory=list)
    triple_origins: dict[Triple, list[Triple]] = field(default_factory=dict)

    # --- translation -------------------------------------------------------

    def canon_entity(self, surface: str) -> str:
        """Canonical id for an entity surface (identity if never merged)."""
        return self.entity.get(surface, surface)

    def canon_relation(self, relation: str) -> str:
        """Canonical id for a relation surface (identity if never merged)."""
        return self.relation.get(relation, relation)

    def translate(self, triple: Triple) -> Triple:
        """Map a surface triple to its canonical form.

        Head and tail go through the entity map, the relation through the
        relation map. Because the entity map's keys are only ever merged entity
        surfaces, a value tail that was never an entity is returned unchanged —
        so numeric / categorical answer values are never rewritten.
        """
        h, r, t = triple
        return (self.entity.get(h, h), self.relation.get(r, r),
                self.entity.get(t, t))

    # --- reverse lookups ---------------------------------------------------

    def surfaces_of(self, canonical: str, *, kind: str = "entity") -> list[str]:
        """All surfaces that map to ``canonical`` (including itself), sorted."""
        table = self.entity if kind == "entity" else self.relation
        out = {s for s, c in table.items() if c == canonical}
        out.add(canonical)
        return sorted(out)

    def origins(self, canonical_triple: Triple) -> list[Triple]:
        """Original surface triples that collapsed into ``canonical_triple``.

        The provenance bridge: resolve any of these against the source text to
        locate the fact's span. Falls back to the canonical triple itself when no
        distinct origin was recorded (an untouched triple is its own origin).
        """
        return self.triple_origins.get(canonical_triple, [canonical_triple])

    # --- status ------------------------------------------------------------

    @property
    def n_entity_merges(self) -> int:
        """Number of entity surfaces remapped to a different canonical."""
        return len(self.entity)

    @property
    def n_relation_merges(self) -> int:
        """Number of relation surfaces remapped to a different canonical."""
        return len(self.relation)

    def is_noop(self) -> bool:
        """True when the map merges nothing (the common clean-corpus case)."""
        return not self.entity and not self.relation

    # --- application -------------------------------------------------------

    def rewrite(self, triples: list[Triple]) -> list[Triple]:
        """Apply the map to a triple list: translate every triple, collapse the
        exact duplicates the merge creates (order preserved), and record each
        canonical triple's surface-triple origins so provenance stays resolvable.

        A no-op returns a copy unchanged. Shared by the deterministic and the
        geometric proposers so the surface-origin bookkeeping lives in one place.
        """
        if self.is_noop():
            return list(triples)
        seen: set[Triple] = set()
        out: list[Triple] = []
        for tr in triples:
            ct = self.translate(tr)
            if ct != tr:
                origins = self.triple_origins.setdefault(ct, [])
                if tr not in origins:
                    origins.append(tr)
            if ct in seen:
                continue
            seen.add(ct)
            out.append(ct)
        return out

    # --- persistence -------------------------------------------------------

    def to_records(self) -> dict:
        """Serializable form: the maps, the audit evidence and the flagged set.

        ``triple_origins`` is emitted as a list of ``{canonical, origins}`` rows
        (tuples are not JSON keys); read it back with :meth:`from_records`.
        """
        return {
            "entity": dict(sorted(self.entity.items())),
            "relation": dict(sorted(self.relation.items())),
            "evidence": self.evidence,
            "flagged": self.flagged,
            "triple_origins": [
                {"canonical": list(ct), "origins": [list(o) for o in origs]}
                for ct, origs in self.triple_origins.items()
            ],
        }

    @classmethod
    def from_records(cls, rec: dict) -> "OntologyMap":
        """Reconstruct an :class:`OntologyMap` from :meth:`to_records` output."""
        origins = {
            tuple(row["canonical"]): [tuple(o) for o in row["origins"]]
            for row in rec.get("triple_origins", [])
        }
        return cls(
            entity=dict(rec.get("entity", {})),
            relation=dict(rec.get("relation", {})),
            evidence=list(rec.get("evidence", [])),
            flagged=list(rec.get("flagged", [])),
            triple_origins=origins,
        )
