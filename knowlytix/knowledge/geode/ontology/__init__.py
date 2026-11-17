# SPDX-License-Identifier: Apache-2.0
"""GEODE ontology layer — normalization and co-reference (additive).

This subpackage builds the identity layer the rest of GEODE assumes but never
constructs: a normalized, co-reference-resolved *canonical* vocabulary for the
extracted graph, produced BEFORE the geometry trains so the composition critic,
dedup and the anchors — all of which key on exact surface identity — can see the
redundancy they rely on.

It is strictly additive. Nothing in the existing GEODE modules
(:mod:`knowlytix.knowledge.geode.loop`, ``critic``, ``dedup``, ``anchor``,
``provenance``, ``canonicalize``, ``rag``) is modified; this layer composes them
through their public seams and is reached only by its own entry points.

The design separates two things the base pipeline conflates:

* the **canonical** id the geometry/critic/dedup/anchors operate on, and
* the **surface** strings the source document actually spells,

so normalization can merge ``fees`` -> ``fee`` for the geometry while provenance
still resolves the fact back to the ``fees`` cell in the source. That mapping is
carried by :class:`~knowlytix.knowledge.geode.ontology.map.OntologyMap`.

Tiers (built incrementally):

* tier 1 — deterministic (this phase): case/separator/whitespace unification and
  conservative inflectional singularization of *observed* surface pairs;
* tier 2 — declared-alias merging (in-document ``has_alias`` / name edges);
* tier 3 — fuzzy co-reference (wraps the existing ``canonicalize_graph``),
  review-gated.

Numeric tails are never normalized (the exact-number guarantee is preserved), and
a surface is only rewritten when normalization collapses two or more *observed*
surfaces to a shared form — a singleton is never renamed, so a mis-normalized lone
value cannot corrupt the graph.
"""

from knowlytix.knowledge.geode.ontology.map import OntologyMap
from knowlytix.knowledge.geode.ontology.normalize import (
    normalize_graph,
    singularize,
)

__all__ = ["OntologyMap", "normalize_graph", "singularize"]
