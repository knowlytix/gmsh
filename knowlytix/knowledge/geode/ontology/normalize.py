# SPDX-License-Identifier: Apache-2.0
"""Tier-1 deterministic normalization for the GEODE ontology layer.

No model, no embeddings: this merges *observed* surface forms that differ only by
case, separator, whitespace or inflection (``fees`` / ``fee``, ``od_fee`` /
``od fee``). It is the high-precision floor of the normalization layer — the part
that makes the composition critic's exact-identity redundancy visible without any
learned judgement.

Safety is structural, not statistical:

* **Merge only observed pairs.** A surface is remapped only when normalization
  collapses it with at least one *other* surface present in the graph. A lone
  surface is never renamed, so a mis-singularized value (``status`` -> ``statu``)
  is inert — it changes a grouping key, never the graph.
* **Canonical is an observed surface** (most frequent, then shortest), never a
  synthesized string, so provenance always has a real span to resolve to.
* **Conflict veto.** Two entity surfaces that share an attribute with *different*
  values are not merged even if lexically identical after normalization (the
  analog for relations is a functional conflict: one head asserting both with
  different tails). This stops a lexical coincidence from collapsing two distinct
  things and manufacturing wrong logic.
* **Values are never touched.** Only surfaces that appear as a *head* somewhere
  are entity-merge candidates; a tail that is not itself an entity (a number, a
  categorical answer) is left exactly as extracted.

Returns the canonical triples plus an :class:`~knowlytix.knowledge.geode.ontology.map.OntologyMap`
recording every merge, every declined group, and the surface-triple origins that
keep provenance resolvable.
"""

from __future__ import annotations

import re

from knowlytix.knowledge.geode.ontology.map import OntologyMap

__all__ = ["normalize_graph", "singularize", "entity_key", "relation_key"]

Triple = tuple[str, str, str]

# Relations that carry a surface form / structural role rather than a fact, kept
# out of entity attribute signatures (mirrors the exclusions the existing
# canonicalize/rag layers already use). Used only by the conflict veto.
_NON_ATTR_RELATIONS = frozenset({
    "has_alias", "has_name", "has_policy_name", "in_section", "is_functional",
    "opposite_of", "passes", "fails", "is_weak", "is_strong",
})

# Tokens whose trailing 's' is not a plural inflection — never singularize these.
# Conservative and extensible; the harness prints every applied merge so a bad
# singularization on a real corpus surfaces for review and lands here.
_SINGULAR_STOPWORDS = frozenset({
    "series", "species", "analysis", "basis", "axis", "status", "bias",
    "access", "business", "process", "address", "class", "loss", "cross",
    "gross", "mass", "pass", "press", "miss", "boss", "glass", "grass",
    "news", "gas", "plus", "versus", "bonus", "focus", "virus", "census",
    "corpus", "lens", "bias", "canvas", "atlas", "always", "aliases",
})

_NUM_RE = re.compile(r"^-?[\d,]*\.?\d+%?$")
_SEP_RE = re.compile(r"[_\-]+")
_WS_RE = re.compile(r"\s+")


def _is_value(surface: str) -> bool:
    """A bare number/percent is an answer value, never an entity."""
    return bool(_NUM_RE.match(surface.strip()))


def singularize(token: str, *, stopwords: frozenset[str] = _SINGULAR_STOPWORDS
                ) -> str:
    """Conservative inflectional singularization of a single lowercase token.

    Handles the regular English plural endings and leaves everything uncertain
    alone (``-ss``, ``-us``, ``-is``, ``-os``, short tokens, and an explicit
    stopword list). This is a grouping heuristic, not a linguistic lemmatizer:
    because :func:`normalize_graph` only merges surfaces that actually co-occur,
    an over- or under-singularized token is harmless unless a real pair exists.
    """
    t = token
    if len(t) <= 3 or t in stopwords:
        return t
    if t.endswith("ss") or t.endswith(("us", "is", "os", "as")):
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("ses", "xes", "zes", "ches", "shes")):
        return t[:-2]
    if t.endswith("s"):
        return t[:-1]
    return t


def _normal_form(text: str, *, singular: bool,
                 stopwords: frozenset[str]) -> str:
    """Aggressive grouping KEY: lowercase, unify separators/whitespace, and
    singularize the head (last) token. Never used as a stored name — only to
    decide which observed surfaces belong to the same group."""
    s = _WS_RE.sub(" ", _SEP_RE.sub(" ", text.strip().lower())).strip()
    if not s:
        return s
    if not singular:
        return s
    toks = s.split(" ")
    toks[-1] = singularize(toks[-1], stopwords=stopwords)
    return " ".join(toks)


def entity_key(surface: str, *, singular: bool = True,
               stopwords: frozenset[str] = _SINGULAR_STOPWORDS) -> str:
    """Grouping key for an entity surface."""
    return _normal_form(surface, singular=singular, stopwords=stopwords)


def relation_key(relation: str, *, singular: bool = True,
                 stopwords: frozenset[str] = _SINGULAR_STOPWORDS) -> str:
    """Grouping key for a relation: the ``has_`` prefix is preserved so a
    plural column (``has_fees`` / ``has_fee``) groups but content and structural
    relations never collide by phrase alone."""
    prefix = "has_" if relation.startswith("has_") else ""
    body = relation[len(prefix):]
    return prefix + _normal_form(body, singular=singular, stopwords=stopwords)


def _counts(triples: list[Triple]) -> dict[str, int]:
    """Surface-frequency over all head/tail positions (canonical tie-break)."""
    c: dict[str, int] = {}
    for h, _r, t in triples:
        c[h] = c.get(h, 0) + 1
        c[t] = c.get(t, 0) + 1
    return c


def _pick_canonical(members: list[str], counts: dict[str, int]) -> str:
    """Most frequent surface, ties broken by shortest then lexical."""
    return sorted(members, key=lambda s: (-counts.get(s, 0), len(s), s))[0]


def _entity_attrs(triples: list[Triple]) -> dict[str, dict[str, set[str]]]:
    """``entity -> {relation -> {tails}}`` over fact relations, for the veto."""
    attr: dict[str, dict[str, set[str]]] = {}
    for h, r, t in triples:
        if r.startswith("has_") and r not in _NON_ATTR_RELATIONS:
            attr.setdefault(h, {}).setdefault(r, set()).add(t)
    return attr


def _entity_conflict(a: str, b: str,
                     attr: dict[str, dict[str, set[str]]]) -> bool:
    """True if ``a`` and ``b`` share a fact attribute with different values."""
    ra, rb = attr.get(a, {}), attr.get(b, {})
    return any(ra[r] != rb[r] for r in (ra.keys() & rb.keys()))


def _assign_group(members: list[str], counts: dict[str, int],
                  conflict) -> tuple[str, list[str], list[str]]:
    """Greedily map a lexical group to one canonical, holding out conflicts.

    The canonical is the group's representative; a member that conflicts with the
    canonical is not merged (returned as flagged), never silently collapsed.
    """
    canonical = _pick_canonical(members, counts)
    merged, held = [], []
    for m in members:
        if m == canonical:
            continue
        (held if conflict(m, canonical) else merged).append(m)
    return canonical, merged, held


def _build_groups(surfaces: set[str], key_fn) -> dict[str, list[str]]:
    """Group surfaces by their normalization key; keep groups of size >= 2."""
    groups: dict[str, list[str]] = {}
    for s in surfaces:
        groups.setdefault(key_fn(s), []).append(s)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def normalize_graph(
    triples: list[Triple],
    *,
    singularize_entities: bool = True,
    singularize_relations: bool = True,
    conflict_veto: bool = True,
    stopwords: frozenset[str] = _SINGULAR_STOPWORDS,
) -> tuple[list[Triple], OntologyMap]:
    """Deterministically normalize entity and relation surfaces (tier 1).

    Args:
        triples: extracted ``(head, relation, tail)`` list.
        singularize_entities: fold entity plural/singular surface pairs.
        singularize_relations: fold relation plural/singular surface pairs.
        conflict_veto: hold out a lexical-group member that shares an attribute
            with the canonical but disagrees on its value (entities) or a
            functional conflict (relations).
        stopwords: tokens whose trailing ``s`` is never treated as a plural.

    Returns:
        ``(canonical_triples, OntologyMap)``. ``canonical_triples`` has every
        surface remapped to its canonical and exact duplicates collapsed, order
        preserved. The map records the merges, the flagged (declined) surfaces
        and the surface-triple origins that keep provenance resolvable. A no-op
        (map merges nothing) when there is no observed fragmentation.
    """
    counts = _counts(triples)
    attr = _entity_attrs(triples) if conflict_veto else {}

    # Entity candidates = surfaces that head a fact (a subject is an entity);
    # numeric/value tails that never head anything are not candidates, so values
    # are untouched.
    heads = {h for h, _r, _t in triples if not _is_value(h)}
    ent_groups = _build_groups(
        heads, lambda s: entity_key(s, singular=singularize_entities,
                                    stopwords=stopwords))
    rel_surfaces = {r for _h, r, _t in triples}
    rel_groups = _build_groups(
        rel_surfaces, lambda r: relation_key(r, singular=singularize_relations,
                                             stopwords=stopwords))

    omap = OntologyMap()

    def ent_conflict(a: str, b: str) -> bool:
        return conflict_veto and _entity_conflict(a, b, attr)

    for _k, members in sorted(ent_groups.items()):
        canonical, merged, held = _assign_group(members, counts, ent_conflict)
        for m in merged:
            omap.entity[m] = canonical
        if merged:
            omap.evidence.append({
                "kind": "entity", "tier": "deterministic", "reason": "inflection",
                "canonical": canonical, "members": merged})
        for h in held:
            omap.flagged.append({"kind": "entity", "surface": h,
                                 "canonical": canonical,
                                 "reason": "attribute_conflict"})

    rel_conflict = _relation_conflict_fn(triples) if conflict_veto else (
        lambda a, b: False)
    for _k, members in sorted(rel_groups.items()):
        canonical, merged, held = _assign_group(members, counts, rel_conflict)
        for m in merged:
            omap.relation[m] = canonical
        if merged:
            omap.evidence.append({
                "kind": "relation", "tier": "deterministic",
                "reason": "inflection", "canonical": canonical,
                "members": merged})
        for r in held:
            omap.flagged.append({"kind": "relation", "surface": r,
                                 "canonical": canonical,
                                 "reason": "functional_conflict"})

    rewritten = _rewrite(triples, omap)
    return rewritten, omap


def _relation_conflict_fn(triples: list[Triple]):
    """Functional-conflict predicate: some head asserts both relations with
    different tails -> they are distinct attributes, never merge."""
    head_rel_tails: dict[str, dict[str, set[str]]] = {}
    for h, r, t in triples:
        head_rel_tails.setdefault(h, {}).setdefault(r, set()).add(t)

    def conflict(a: str, b: str) -> bool:
        for rt in head_rel_tails.values():
            if a in rt and b in rt and rt[a] != rt[b]:
                return True
        return False

    return conflict


def _rewrite(triples: list[Triple], omap: OntologyMap) -> list[Triple]:
    """Apply the map: translate every triple, collapse exact duplicates (order
    preserved), and record each canonical triple's surface-triple origins."""
    if omap.is_noop():
        return list(triples)
    seen: set[Triple] = set()
    out: list[Triple] = []
    for tr in triples:
        ct = omap.translate(tr)
        # Record the surface origin of this canonical triple only when it was
        # actually rewritten; an untouched triple is its own origin (see
        # OntologyMap.origins), so identity triples stay out of the map.
        if ct != tr:
            origins = omap.triple_origins.setdefault(ct, [])
            if tr not in origins:
                origins.append(tr)
        if ct in seen:
            continue
        seen.add(ct)
        out.append(ct)
    return out
