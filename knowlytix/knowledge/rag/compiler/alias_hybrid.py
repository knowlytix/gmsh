# SPDX-License-Identifier: Apache-2.0
"""The alias-table hybrid binder: compiler relations + a patchable entity resolver.

This is the *product* form of the bake-off's Arm D (``query_parse_mode="alias_hybrid"``).
The trained query compiler supplies the *relations* (the learned part), but the
*head entity* is resolved through a patchable alias table (surface form ->
canonical id) with a calibrated encoder-nearest fallback. A novel or mis-resolved
entity is fixed by adding one alias-table row -- an ``alias_edit`` -- rather than
retraining the compiler, which is the only difference from the pure compiler
front-end (``query_parse_mode="compiler"``, Arm B): both bind the same relations
with the same downstream, so they bind equally well, but the alias-table hybrid is
patched by editing a table instead of a full retrain.

The parser implements the extractor contract (``extract(question) -> list[QueryTriple]``
and a batched ``extract_batch``) that :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline`
selects as ``self.extractor``. It emits a ``(head, relation)`` hop chain and resolves
it with the compiler's :class:`~knowlytix.knowledge.rag.compiler.walk.StoreChainWalker`,
so its abstention semantics match the compiler exactly (a hop that is not an asserted
fact yields the empty chain and the pipeline abstains through its no-binding path).

The encoder-fallback margin ``tau`` is a *calibrated, persisted* operating point, not
a hand-picked default: :func:`calibrate_alias_resolver` fits it under a false-accept
ceiling and :func:`build_alias_hybrid_parser` loads it from (or writes it to) the
store. With no encoder or too few entities the fallback stays disabled and the
resolver recuses (abstains) rather than guessing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from knowlytix.knowledge.rag.compiler.walk import StoreChainWalker, _norm

__all__ = [
    "AliasTableResolver",
    "RelationResolver",
    "AliasHybridParser",
    "calibrate_alias_resolver",
    "calibrate_relation_resolver",
    "build_alias_hybrid_parser",
    "ALIAS_CALIBRATION_FILENAME",
    "RELATION_CALIBRATION_FILENAME",
]

# Persisted next to the store; read by build_alias_hybrid_parser, written by
# calibrate_alias_resolver when it fits a fresh margin.
ALIAS_CALIBRATION_FILENAME = "alias_resolver_calibration.json"
RELATION_CALIBRATION_FILENAME = "relation_resolver_calibration.json"


def _collapse(s: str) -> str:
    """Alphanumeric-collapse key: lower-case, drop every non-alphanumeric character.

    Digits are preserved, so distinct numeric identifiers stay distinct
    (``5318(g)(3)`` -> ``5318g3`` is not ``5318(g)(2)`` -> ``5318g2``); only
    separators and casing are removed. Used as a resolver key that tolerates
    punctuation/spacing variants of an identifier (``31 U.S.C. 5318(g)(3)`` ->
    ``31usc5318g3`` matches ``31_usc_5318_g_3``). Exact numeric *values*
    (thresholds, amounts) are the ENM's domain, not this key's; the numeric-order
    route resolves them before the alias resolver runs.
    """
    import re
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _normalize_rows(mat):
    import numpy as np

    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@dataclass
class AliasTableResolver:
    """A patchable surface-form -> canonical-entity map with an encoder fallback.

    The table is seeded from the store's entities (each entity keys itself and its
    normalized form) and is edited, not trained: :meth:`add_alias` binds a new
    surface form to a canonical entity in one step (``alias_edit``). When a surface
    misses the table, :meth:`resolve` falls back to the nearest store entity in the
    encoder's space, accepted only if its cosine clears the calibrated margin
    ``tau`` -- otherwise it returns ``None`` (the resolver recuses and the query
    abstains). With no calibrated ``tau`` the fallback is disabled (recuse), never a
    bare default.
    """

    table: dict[str, str]                       # normalized surface -> canonical id
    entities: list[str]
    tau: float | None = None                    # calibrated encoder-fallback margin
    collapse: dict[str, str] = field(default_factory=dict)  # collapse key -> canonical
    _ent_emb: object = field(default=None, repr=False)
    _encoder: object = field(default=None, repr=False)
    n_alias_edits: int = 0

    @classmethod
    def from_store(cls, store, encoder=None, *, tau: float | None = None,
                   structural_relations=None) -> "AliasTableResolver":
        walker = StoreChainWalker(store, structural_relations=structural_relations)
        ents = walker.entities
        table = {}
        for e in ents:
            table[_norm(e)] = e
            table[e] = e
        # Alphanumeric-collapse index, kept only for keys that are UNAMBIGUOUS
        # across entities so a collapse hit never silently merges two entities
        # (e.g. two ids differing only by a separator). A colliding key is dropped.
        seen: dict[str, str] = {}
        collide: set[str] = set()
        for e in ents:
            k = _collapse(e)
            if not k:
                continue
            if k in seen and seen[k] != e:
                collide.add(k)
            seen.setdefault(k, e)
        collapse = {k: v for k, v in seen.items() if k not in collide}
        ent_emb = _normalize_rows(encoder(ents)) if (encoder and ents) else None
        return cls(table=table, entities=ents, tau=tau, collapse=collapse,
                   _ent_emb=ent_emb, _encoder=encoder)

    def add_alias(self, surface: str, canonical: str) -> None:
        """Bind a surface form to a canonical entity -- the unit patch (I6)."""
        self.table[_norm(surface)] = canonical
        self.table[surface] = canonical
        k = _collapse(surface)
        if k:                                    # an explicit edit wins any collision
            self.collapse[k] = canonical
        self.n_alias_edits += 1

    def resolve(self, surface: str) -> str | None:
        """Resolve a proposed head surface to a canonical store entity, or ``None``
        to recuse (abstain)."""
        if surface in self.table:
            return self.table[surface]
        n = _norm(surface)
        if n in self.table:
            return self.table[n]
        c = self.collapse.get(_collapse(surface))  # punctuation/spacing-tolerant
        if c is not None:
            return c
        if self._encoder is None or self._ent_emb is None or self.tau is None:
            return None                          # no calibrated fallback -> recuse
        import numpy as np

        qv = _normalize_rows(self._encoder([surface]))[0]
        sims = self._ent_emb @ qv
        j = int(np.argmax(sims))
        return self.entities[j] if float(sims[j]) >= self.tau else None


def _rel_phrase(r: str) -> str:
    """Human phrasing of a relation id for encoding (mirrors the follow-on route)."""
    return (r[4:] if r.startswith("has_") else r).replace("_", " ").strip()


@dataclass
class RelationResolver:
    """Map a compiler-emitted relation string to a canonical store relation.

    The store walk matches ``(head, relation)`` exactly, so a relation the compiler
    paraphrases (``protects_from``) misses the asserted relation (``provides_
    protection_from``) and the chain abstains even though the head resolved. This
    resolver recovers the canonical relation by, in order: an exact / normalized /
    collapse hit, then the nearest relation in the encoder's space -- but only among
    the relations the resolved HEAD actually asserts (the candidate set passed to
    :meth:`resolve`), and only if the cosine clears the calibrated margin ``tau``.
    Scoping to the head's own relations keeps the candidate set tiny, so the
    cross-relation false-accept margin fit over all store relations is conservative.
    With no encoder or no calibrated ``tau`` the fallback is disabled and only the
    exact/normalized/collapse hits apply (never a bare guess).
    """

    tau: float | None = None
    _rel_emb: dict = field(default_factory=dict, repr=False)   # relation -> unit vec
    _encoder: object = field(default=None, repr=False)

    @classmethod
    def from_store(cls, store, encoder=None, *, tau: float | None = None,
                   structural_relations=None) -> "RelationResolver":
        walker = StoreChainWalker(store, structural_relations=structural_relations)
        rels = sorted({r for _, r in walker._facts})
        emb = {}
        if encoder and rels:
            mat = _normalize_rows(encoder([_rel_phrase(r) for r in rels]))
            emb = {r: mat[i] for i, r in enumerate(rels)}
        return cls(tau=tau, _rel_emb=emb, _encoder=encoder)

    def resolve(self, surface: str, candidates: list[str]) -> str | None:
        """Resolve ``surface`` to one of ``candidates`` (a head's relations), or
        ``None`` to leave it unresolved (the chain then abstains downstream)."""
        if not candidates:
            return None
        if surface in candidates:
            return surface
        ns = _norm(surface)
        for c in candidates:
            if _norm(c) == ns:
                return c
        cs = _collapse(surface)
        for c in candidates:
            if _collapse(c) == cs:
                return c
        if self._encoder is None or self.tau is None or not self._rel_emb:
            return None
        import numpy as np

        qv = _normalize_rows(self._encoder([_rel_phrase(surface)]))[0]
        best, best_sim = None, -1.0
        for c in candidates:
            v = self._rel_emb.get(c)
            if v is None:
                continue
            s = float(np.dot(v, qv))
            if s > best_sim:
                best, best_sim = c, s
        return best if best_sim >= self.tau else None


class AliasHybridParser:
    """Compiler relations + alias-table head resolution (Arm D).

    The trained compiler emits the raw hop chain (the *relations* are the learned
    part). Before the store walk, the first hop's head is re-resolved through the
    :class:`AliasTableResolver`; if the resolver recuses, the chain abstains. The
    remaining hops walk exactly as in the pure compiler front-end.

    When the compiler's emitted relation is not itself an asserted fact of the
    resolved head, the first hop's relation is repaired: first by a deterministic
    surface match (exact / normalized / collapse) against the head's relations, then
    -- if a ``relation_gate`` is supplied -- by the calibrated question->relation
    step of :class:`~knowlytix.knowledge.rag.relevance.GeometricRelevanceGate`, which
    picks the head relation the QUESTION is closest to in v-space and accepts it only
    above the store's persisted ``tau_accept``. Repairing off the question rather
    than off the emitted relation string avoids binding a lexically-near sibling
    relation that answers a different question (``protects`` for ``protects_from``);
    an unresolved relation leaves the chain to abstain through the walk.

    Args:
        store: a loaded GMS store.
        compiler: a :class:`~knowlytix.knowledge.rag.compiler.model.CompiledQueryParser`.
        resolver: an :class:`AliasTableResolver`.
        relation_resolver: an :class:`RelationResolver` for the deterministic
            surface match of a mis-emitted relation to one the head holds.
        relation_gate: a :class:`GeometricRelevanceGate` (or any object exposing
            ``relevant_relation(question, head) -> (relation | None, detail)``) for
            the calibrated question->relation repair; ``None`` disables it.
        structural_relations: passed through to the walker.
    """

    def __init__(self, store, compiler, resolver: AliasTableResolver, *,
                 relation_resolver: "RelationResolver | None" = None,
                 relation_gate=None,
                 structural_relations=None):
        self.store = store
        self.compiler = compiler
        self.resolver = resolver
        self.relation_resolver = relation_resolver
        self.relation_gate = relation_gate
        self._walker = StoreChainWalker(store, structural_relations=structural_relations)

    def _resolve_head(self, chain: list[tuple[str, str]], question: str | None = None):
        if not chain:
            return []
        head = self.resolver.resolve(chain[0][0])
        if head is None:                          # resolver recuses -> abstain
            return []
        rel = chain[0][1]
        # Repair a mis-emitted first-hop relation to one the head actually holds:
        # deterministic surface match first, then the calibrated question->relation
        # gate. A failure leaves the original relation, so the walk abstains as before.
        if not self._walker.holds(head, rel):
            cands = self._walker.relations_of(head)
            alt = (self.relation_resolver.resolve(rel, cands)
                   if self.relation_resolver is not None else None)
            if alt is None and self.relation_gate is not None and question:
                matched, _detail = self.relation_gate.relevant_relation(question, head)
                if matched in cands:
                    alt = matched
            if alt is not None:
                rel = alt
        return [(head, rel)] + list(chain[1:])

    def compile_raw(self, questions: list[str]) -> list[list[tuple[str, str]]]:
        qs = list(questions)
        return [self._resolve_head(c, q)
                for c, q in zip(self.compiler.compile_raw(qs), qs)]

    def extract(self, question: str, *, hint: str | None = None):
        return self._walker.walk(self.compile_raw([question])[0])

    def extract_batch(self, questions: list[str]):
        return [self._walker.walk(chain) for chain in self.compile_raw(list(questions))]


def calibrate_alias_resolver(entities: list[str], encoder, *, far_ceiling: float = 0.05,
                             min_entities: int = 4, persist_path: str | None = None) -> dict:
    """Fit the encoder-fallback acceptance margin ``tau`` for :class:`AliasTableResolver`.

    The fallback must accept a genuine surface form (nearest store entity) while
    rejecting a surface that belongs to no store entity. With only the store's own
    entities as a labeled set, ``tau`` is set to the largest cosine below which the
    *cross-entity* nearest-neighbor false-accept rate stays at or under
    ``far_ceiling``: each entity's nearest OTHER entity is a negative (a surface that
    must not resolve to it), and ``tau`` is the ``(1 - far_ceiling)`` quantile of
    those cross similarities, so at most a ``far_ceiling`` fraction of confusable
    pairs clear the bar. Fewer than ``min_entities`` entities cannot calibrate: the
    fallback is left disabled (``tau=None``) and the resolver recuses.

    Returns ``{"tau", "far_ceiling", "n_entities", "calibrated"}`` and, if
    ``persist_path`` is given, writes it there.
    """
    import numpy as np

    result = {"tau": None, "far_ceiling": far_ceiling,
              "n_entities": len(entities), "calibrated": False}
    if len(entities) >= min_entities and encoder is not None:
        emb = _normalize_rows(encoder(entities))
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1.0)
        nearest_other = sims.max(axis=1)          # each entity's most confusable other
        tau = float(np.quantile(nearest_other, 1.0 - far_ceiling))
        result.update(tau=tau, calibrated=True)
    if persist_path:
        with open(persist_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


def calibrate_relation_resolver(relations: list[str], encoder, *, far_ceiling: float = 0.05,
                                min_relations: int = 4, persist_path: str | None = None) -> dict:
    """Fit the acceptance margin ``tau`` for :class:`RelationResolver`.

    Mirrors :func:`calibrate_alias_resolver` over relation phrasings: each store
    relation's nearest OTHER relation is a cross-relation negative, and ``tau`` is
    the ``(1 - far_ceiling)`` quantile of those cross similarities. The margin is fit
    over ALL store relations, while inference scopes candidates to a single head's
    relations, so the operating point is conservative. Fewer than ``min_relations``
    relations leaves the fallback disabled (``tau=None``).

    Returns ``{"tau", "far_ceiling", "n_relations", "calibrated"}`` and, if
    ``persist_path`` is given, writes it there.
    """
    import numpy as np

    result = {"tau": None, "far_ceiling": far_ceiling,
              "n_relations": len(relations), "calibrated": False}
    if len(relations) >= min_relations and encoder is not None:
        emb = _normalize_rows(encoder([_rel_phrase(r) for r in relations]))
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1.0)
        nearest_other = sims.max(axis=1)
        tau = float(np.quantile(nearest_other, 1.0 - far_ceiling))
        result.update(tau=tau, calibrated=True)
    if persist_path:
        with open(persist_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


def build_alias_hybrid_parser(store, compiler, encoder, *, far_ceiling: float = 0.05,
                              store_path: str | None = None, relation_gate=None,
                              structural_relations=None) -> AliasHybridParser:
    """Build an alias-hybrid parser with a calibrated, persisted encoder-fallback margin.

    Wires the pieces the ``query_parse_mode="alias_hybrid"`` route needs, reusing an
    ALREADY-LOADED ``compiler`` so no second adapter is loaded (the pipeline passes
    the compiler it built; a multi-binder harness can pass the one it shares with the
    pure-compiler arm). The head encoder-fallback margin is a calibrated operating
    point: it is read from ``store_path/alias_resolver_calibration.json`` when
    present, and otherwise fit once with :func:`calibrate_alias_resolver` over the
    store entities under ``far_ceiling`` and persisted there. With no ``encoder`` or
    too few entities the fallback stays disabled and the resolver recuses -- never a
    bare default.

    The relation of a mis-emitted first hop is repaired deterministically (exact /
    normalized / collapse surface match to one the head holds) and, when
    ``relation_gate`` is supplied, by its calibrated question->relation step. The gate
    carries the store's persisted question-cohort calibration, so no second margin is
    fit here.

    Args:
        store: a loaded GMS store.
        compiler: a :class:`~knowlytix.knowledge.rag.compiler.model.CompiledQueryParser`.
        encoder: the tuned v-encoder (``list[str] -> array``) for the head fallback;
            None leaves the fallback disabled (exact alias-table hits still resolve).
        far_ceiling: cross-entity false-accept ceiling for the head fallback margin.
        store_path: where the head calibration is read from / written to; defaults to
            ``store.store_path``.
        relation_gate: a calibrated question->relation gate (see
            :class:`AliasHybridParser`); ``None`` leaves relation repair
            deterministic-only.
        structural_relations: passed through to the resolver and walker.
    """
    resolver = AliasTableResolver.from_store(
        store, encoder=encoder, structural_relations=structural_relations)
    sp = store_path or getattr(store, "store_path", None)
    cal_path = os.path.join(sp, ALIAS_CALIBRATION_FILENAME) if sp else None
    cal = None
    if cal_path and os.path.isfile(cal_path):
        try:
            cal = json.load(open(cal_path))
        except (ValueError, OSError):
            cal = None
    if cal is None:
        cal = calibrate_alias_resolver(resolver.entities, encoder,
                                       far_ceiling=far_ceiling, persist_path=cal_path)
    resolver.tau = cal.get("tau")

    # Deterministic relation surface-matcher (exact / normalized / collapse to a
    # relation the head holds). The semantic step is the calibrated question->relation
    # gate, not an emitted-relation-string nearest neighbor, so no margin is fit here.
    rel_resolver = RelationResolver.from_store(
        store, encoder=None, structural_relations=structural_relations)
    return AliasHybridParser(store, compiler, resolver, relation_resolver=rel_resolver,
                             relation_gate=relation_gate,
                             structural_relations=structural_relations)
