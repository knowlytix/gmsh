# SPDX-License-Identifier: Apache-2.0
"""GeometricReasoner: agent-friendly multi-hop inference on the GMS manifold.

Wraps GeometricSearch, RelationalTransport and GMSExpertStore primitives
into a high-level reasoning API for governed agents.  All geometry lives
in the existing GMS library; this module provides the agent-level verbs:
plausibility checking, path finding, chain validation, contradiction
detection, next-step recommendation and explanation.

Built in Chapter 6 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from knowlytix.core.graph.search import GeometricSearch, ScoredPath, SearchResult
from knowlytix.core.graph.transport import RelationalTransport
from knowlytix.knowledge.store import GMSExpertStore

# ------------------------------------------------------------------ #
# Result data classes
# ------------------------------------------------------------------ #

@dataclass
class ChainLink:
    """One hop in a validated reasoning chain."""
    head: str
    relation: str
    tail: str
    score: float | None = None       # geodesic distance (lower = better)
    plausible: bool | None = None     # score <= threshold


@dataclass
class ChainValidation:
    """Result of validating a multi-hop reasoning chain."""
    links: list[ChainLink] = field(default_factory=list)
    all_plausible: bool = False
    mean_score: float | None = None
    holonomy_defect: float | None = None
    consistent: bool | None = None    # holonomy <= tau_path


@dataclass
class Contradiction:
    """A pair of entities judged to contradict each other.

    ``provenances`` tells callers which mechanism(s) flagged the pair
    -- a :class:`frozenset` drawn from:

      * ``"schema"``   -- deterministic, from corpus-declared
                          ``(r_a, opposite_of, r_b)`` triples; the two
                          heads share a tail via opposing relations.
      * ``"geometric"`` -- learned, from pairwise tension energy
                           above the calibrated ``tau_contra``.

    Both may fire for the same pair (strongest evidence: declared
    *and* learned separately). Keeping them as a set preserves that
    audit information, whereas a single-string provenance would
    force dedup to drop one.

    ``threshold`` records the calibrated cut that the geometric arm
    used; it is ``None`` when only the schema arm fired.
    """
    entity_a: str
    entity_b: str
    tension: float
    provenances: frozenset = field(default_factory=lambda: frozenset(["geometric"]))
    threshold: float | None = None
    interpretation: str = ""
    # Schema provenance carries the opposing relations that fired.
    schema_relations: tuple[str, str] | None = None
    shared_tail: str | None = None

    # Convenience accessor: legacy callers expected a single string.
    @property
    def provenance(self) -> str:
        if "schema" in self.provenances and "geometric" in self.provenances:
            return "schema+geometric"
        if "schema" in self.provenances:
            return "schema"
        if "geometric" in self.provenances:
            return "geometric"
        return "none"


@dataclass(frozen=True)
class CalibratedThresholds:
    """Thresholds produced by :meth:`GMSJudge.calibrate`.

    Every value here is derived from a fitted logistic regression on
    labeled graph samples, not hardcoded. ``tension_status`` reports
    whether the 3-class tension fit was usable:

      * ``"ok"``                -- both ``tau_ent`` and ``tau_contra``
                                    are set and strictly ordered.
      * ``"degenerate"``         -- the 3-class classifier's arg-max
                                    for the contradict class never
                                    appears in the observed tension
                                    range. Geometric contradiction
                                    detection is not safe.
      * ``"insufficient_data"``  -- fewer samples than the calibrator
                                    minimum.
      * ``"insufficient_classes"`` -- <3 distinct labels present.

    When the status is not ``"ok"`` the geometric arm of
    :meth:`GeometricReasoner.detect_contradictions` refuses to run
    and the caller must rely on schema-declared contradictions only
    (or retrain with more/cleaner opposition signal).
    """
    tau_plausibility: float | None
    tau_ent: float | None
    tau_contra: float | None
    tension_status: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def tension_ok(self) -> bool:
        return (
            self.tension_status == "ok"
            and self.tau_ent is not None
            and self.tau_contra is not None
        )

    @classmethod
    def from_judge(cls, judge) -> "CalibratedThresholds":
        """Build a CalibratedThresholds from a fitted GMSJudge."""
        ths = getattr(judge, "_thresholds", None) or {}
        return cls(
            tau_plausibility=_coerce_float(ths.get("geodesic")),
            tau_ent=_coerce_float(ths.get("tau_ent")),
            tau_contra=_coerce_float(ths.get("tau_contra")),
            tension_status=str(ths.get("tension_status", "unknown")),
            diagnostics={
                k: v for k, v in ths.items()
                if k not in ("tau_ent", "tau_contra", "tension",
                             "tension_status")
            },
        )


def _coerce_float(v):
    """Accept a scalar or a length-1/2 tuple (geodesic may be 2-class)."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, tuple):
        return float(v[0]) if v else None
    return None


class CalibrationNotUsable(RuntimeError):
    """Raised when a reasoning method requires calibrated thresholds
    that the judge could not produce. Carries the status string so
    callers can explain the refusal to end users."""

    def __init__(self, status: str, detail: str = ""):
        self.status = status
        super().__init__(
            f"calibration status='{status}': {detail}"
            if detail else f"calibration status='{status}'"
        )


@dataclass
class Explanation:
    """Supporting and contradicting evidence for a claim."""
    head: str
    relation: str
    tail: str
    direct_score: float | None = None
    supporting_paths: list[ScoredPath] = field(default_factory=list)
    contradicting_entities: list[Contradiction] = field(default_factory=list)
    verdict: str = ""                 # "supported" / "unsupported" / "contradicted"


# ------------------------------------------------------------------ #
# GeometricReasoner
# ------------------------------------------------------------------ #

class GeometricReasoner:
    """Agent-friendly multi-hop reasoning over a GMS expert store.

    Wraps :class:`GeometricSearch` and :class:`RelationalTransport`
    with convenience methods that return plain Python objects suitable
    for agent consumption.  All tensor computation is delegated to the
    existing GMS library.

    Parameters
    ----------
    store : GMSExpertStore
        A loaded expert store with a trained GKG model.
    device : str or torch.device
        Compute device (``"cuda"`` or ``"cpu"``).
    plausibility_threshold : float
        Geodesic distance below which a triple is considered plausible.
    tension_contradiction : float
        Tension energy above which two entities are flagged as
        contradictory.
    """

    def __init__(
        self,
        store: GMSExpertStore,
        thresholds: CalibratedThresholds,
        *,
        device: str | torch.device = "cuda",
    ):
        if not isinstance(thresholds, CalibratedThresholds):
            raise TypeError(
                "GeometricReasoner now requires a CalibratedThresholds "
                "object produced by CalibratedThresholds.from_judge("
                "GMSJudge). Hardcoded thresholds are not supported; "
                "every cut must be data-driven and auditable."
            )
        self.store = store
        self.thresholds = thresholds
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )

        # Build GeometricSearch if the store has a trained model
        self.search: GeometricSearch | None = None
        self._build_search()

    # Legacy accessors for callers that previously read the hardcoded
    # floats directly.
    @property
    def plausibility_threshold(self) -> float | None:
        return self.thresholds.tau_plausibility

    @property
    def tension_contradiction(self) -> float | None:
        return self.thresholds.tau_contra

    # -------------------------------------------------------------- #
    # Internal helpers
    # -------------------------------------------------------------- #

    def _build_search(self) -> None:
        """Construct GeometricSearch from the store's model and adapter."""
        if self.store.model is None or self.store.adapter is None:
            return
        if self.store.transport is None:
            self.store.transport = RelationalTransport(self.store.model)

        self.search = GeometricSearch(
            model=self.store.model,
            transport=self.store.transport,
            entity_to_idx=self.store.adapter.entity_to_idx,
            relation_to_idx=self.store.adapter.relation_to_idx,
            idx_to_entity=self.store.adapter.idx_to_entity,
            idx_to_relation=self.store.adapter.idx_to_relation,
            device=self.device,
        )
        # Build adjacency from the store's triple set
        if self.store.doc_graph is not None:
            self.search.build_adjacency(self.store.doc_graph.triples)

    def _resolve_entity(self, name: str) -> str | None:
        """Resolve an entity name via exact match or fuzzy fallback."""
        if self.store.entity_exists(name):
            return name
        return self.store.fuzzy_match_entity(name)

    def _interpret_tension(self, t: float) -> str:
        """Label a tension value using the CALIBRATED cuts, not
        hardcoded sqrt(2) bands. Falls back to ``"unlabelled"`` when
        calibration was not usable so the caller cannot silently
        read a default band."""
        ths = self.thresholds
        if ths.tau_ent is None or ths.tau_contra is None:
            return "unlabelled"
        if t < ths.tau_ent:
            return "agreement"
        if t < ths.tau_contra:
            return "irrelevant"
        return "contradiction"

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def is_plausible(
        self, head: str, relation: str, tail: str
    ) -> tuple[bool, float] | None:
        """Score a single triple and check plausibility.

        Parameters
        ----------
        head, relation, tail : str
            The triple to evaluate.

        Returns
        -------
        (plausible, geodesic_distance) or None if entities are unknown.
        """
        score = self.store.score_triple(head, relation, tail)
        if score is None:
            # Try fuzzy resolution
            h = self._resolve_entity(head)
            t = self._resolve_entity(tail)
            if h is None or t is None:
                return None
            score = self.store.score_triple(h, relation, t)
            if score is None:
                return None
        tau = self.thresholds.tau_plausibility
        if tau is None:
            raise CalibrationNotUsable(
                "no_plausibility_threshold",
                "is_plausible requires a calibrated geodesic cut "
                "(tau_plausibility). Run GMSJudge.calibrate() first.",
            )
        return (score <= tau, score)

    def find_path(
        self,
        source_entities: set[str] | list[str],
        target_entities: set[str] | list[str],
        max_hops: int = 4,
    ) -> SearchResult | None:
        """Find geometrically scored paths between entity groups.

        Parameters
        ----------
        source_entities : set or list of str
            Start entities for BFS.
        target_entities : set or list of str
            Goal entities.
        max_hops : int
            Maximum BFS depth.

        Returns
        -------
        SearchResult with scored paths, or None if search unavailable.
        """
        if self.search is None:
            return None

        # Resolve entity names (fuzzy fallback)
        sources: set[str] = set()
        for e in source_entities:
            resolved = self._resolve_entity(e)
            if resolved is not None:
                sources.add(resolved)
        targets: set[str] = set()
        for e in target_entities:
            resolved = self._resolve_entity(e)
            if resolved is not None:
                targets.add(resolved)

        if not sources or not targets:
            return None

        return self.search.connect_groups(sources, targets, max_hops)

    def validate_chain(
        self, chain: list[tuple[str, str, str]]
    ) -> ChainValidation:
        """Validate a multi-hop reasoning chain.

        Each link is scored individually via ``score_triple``.  If the
        chain has a common relation threading through it the holonomy
        defect is also computed.

        Parameters
        ----------
        chain : list of (head, relation, tail) tuples
            The reasoning chain to validate.

        Returns
        -------
        ChainValidation with per-link scores and overall assessment.
        """
        result = ChainValidation()
        scores: list[float] = []
        tau_plaus = self.thresholds.tau_plausibility

        for h, r, t in chain:
            score = self.store.score_triple(h, r, t)
            plausible = None
            if score is not None:
                # Without a calibrated plausibility cut we cannot make
                # the pass/fail judgement; leave ``plausible=None``
                # rather than silently defaulting.
                if tau_plaus is not None:
                    plausible = score <= tau_plaus
                scores.append(score)
            result.links.append(
                ChainLink(head=h, relation=r, tail=t,
                          score=score, plausible=plausible)
            )

        if scores:
            result.mean_score = sum(scores) / len(scores)
            result.all_plausible = all(
                lnk.plausible for lnk in result.links
                if lnk.plausible is not None
            )

        # Holonomy: if all relations are the same, compare path vs direct
        relations = [r for _, r, _ in chain]
        if len(set(relations)) == 1 and len(relations) >= 2:
            defect = self.store.check_holonomy(relations, relations[0])
            if defect is not None:
                result.holonomy_defect = defect
                result.consistent = defect <= 0.5

        # If chain has >=2 distinct relations, pick first relation as
        # "direct" for holonomy comparison (heuristic)
        elif len(relations) >= 2:
            defect = self.store.check_holonomy(
                relations, relations[0]
            )
            if defect is not None:
                result.holonomy_defect = defect
                result.consistent = defect <= 0.5

        return result

    def detect_contradictions(
        self,
        entities: list[str],
        *,
        require_geometric: bool = False,
    ) -> list[Contradiction]:
        """Detect contradictions among a set of entities.

        Two provenance streams are returned in a single list:

          * ``provenance="schema"`` -- deterministic. The corpus
            declared ``(r_a, opposite_of, r_b)`` and the graph
            contains two heads sharing the same tail under ``r_a``
            and ``r_b`` respectively. The tension reported is the
            learned value (for audit) but the judgement itself
            does not depend on any threshold.

          * ``provenance="geometric"`` -- learned. The pair's
            tension exceeds the calibrated ``tau_contra``. This
            arm is silent when ``tension_status != "ok"``; pass
            ``require_geometric=True`` to make that case raise
            :class:`CalibrationNotUsable` instead.

        Parameters
        ----------
        entities : list of str
            Entity names to check.
        require_geometric : bool
            Default ``False`` -- skip geometric detection when
            calibration is not usable, return only schema-declared
            contradictions. Set ``True`` to surface the calibration
            failure loudly.

        Returns
        -------
        list[Contradiction]
            Schema-declared contradictions first (deterministic),
            followed by geometric contradictions sorted by descending
            tension.
        """
        resolved: list[str] = []
        for e in entities:
            r = self._resolve_entity(e)
            if r is not None:
                resolved.append(r)
        resolved_set = set(resolved)

        schema_hits = self._detect_schema_contradictions(resolved_set)
        geometric_hits = self._detect_geometric_contradictions(
            resolved, require=require_geometric,
        )

        # Merge the two streams by pair, unioning provenances. A pair
        # flagged by both schema and geometric is stronger evidence
        # than either alone and must surface as such in the audit
        # record rather than being dropped by dedup.
        by_pair: dict[tuple[str, str], Contradiction] = {}
        for c in schema_hits + geometric_hits:
            key = tuple(sorted((c.entity_a, c.entity_b)))
            existing = by_pair.get(key)
            if existing is None:
                by_pair[key] = c
                continue
            merged_provs = existing.provenances | c.provenances
            # Prefer schema-side audit fields (opposing relations,
            # shared tail) when schema fired; otherwise keep the
            # geometric-side threshold.
            schema_rel = (
                existing.schema_relations
                if existing.schema_relations is not None
                else c.schema_relations
            )
            shared_tail = (
                existing.shared_tail
                if existing.shared_tail is not None
                else c.shared_tail
            )
            threshold = (
                existing.threshold
                if existing.threshold is not None
                else c.threshold
            )
            by_pair[key] = Contradiction(
                entity_a=existing.entity_a,
                entity_b=existing.entity_b,
                tension=existing.tension,
                provenances=merged_provs,
                threshold=threshold,
                interpretation=existing.interpretation,
                schema_relations=schema_rel,
                shared_tail=shared_tail,
            )

        # Output order: schema-only first, then schema+geometric,
        # then geometric-only (strongest deterministic evidence first,
        # then soft evidence).
        def _order_key(c: Contradiction) -> tuple[int, float]:
            is_schema = "schema" in c.provenances
            is_geom = "geometric" in c.provenances
            if is_schema and not is_geom:
                group = 0
            elif is_schema and is_geom:
                group = 1
            else:
                group = 2
            # Within a group, higher tension first.
            return (group, -c.tension)

        return sorted(by_pair.values(), key=_order_key)

    def _detect_schema_contradictions(
        self, resolved: set[str]
    ) -> list[Contradiction]:
        """Deterministic contradictions from graph-declared schema.

        Finds ``(r_a, opposite_of, r_b)`` in the graph, then pairs
        of heads ``(h_a, h_b)`` from ``resolved`` where the graph
        contains ``(h_a, r_a, t)`` and ``(h_b, r_b, t)``.
        """
        if self.store.doc_graph is None:
            return []

        triples = self.store.doc_graph.triples
        opposing: set[tuple[str, str]] = set()
        for h, r, t in triples:
            if r == "opposite_of" and h != t:
                opposing.add((h, t))
                opposing.add((t, h))
        if not opposing:
            return []

        tail_rel_heads: dict[str, dict[str, set[str]]] = {}
        for h, r, t in triples:
            if r in ("opposite_of", "is_functional"):
                continue
            tail_rel_heads.setdefault(t, {}).setdefault(r, set()).add(h)

        hits: list[Contradiction] = []
        for tail, rel_heads in tail_rel_heads.items():
            for (r_a, r_b) in opposing:
                heads_a = rel_heads.get(r_a, set()) & resolved
                heads_b = rel_heads.get(r_b, set()) & resolved
                if not heads_a or not heads_b:
                    continue
                for a in heads_a:
                    for b in heads_b:
                        if a == b:
                            continue
                        te = self.store.tension_energy(a, b)
                        hits.append(Contradiction(
                            entity_a=a, entity_b=b,
                            tension=float(te) if te is not None else 0.0,
                            provenances=frozenset(["schema"]),
                            threshold=None,
                            interpretation=self._interpret_tension(
                                float(te) if te is not None else 0.0
                            ),
                            schema_relations=(r_a, r_b),
                            shared_tail=tail,
                        ))
        return hits

    def _detect_geometric_contradictions(
        self, resolved: list[str], *, require: bool
    ) -> list[Contradiction]:
        """Learned contradictions from calibrated tension threshold."""
        if not self.thresholds.tension_ok:
            if require:
                raise CalibrationNotUsable(
                    self.thresholds.tension_status,
                    "geometric contradiction detection requires a "
                    "usable 3-class tension calibration. Declare "
                    "(r_a, opposite_of, r_b) triples in the corpus "
                    "so training receives contradiction signal, "
                    "then retrain and recalibrate.",
                )
            return []

        tau_contra = self.thresholds.tau_contra
        hits: list[Contradiction] = []
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                t = self.store.tension_energy(resolved[i], resolved[j])
                if t is None or t <= tau_contra:
                    continue
                hits.append(Contradiction(
                    entity_a=resolved[i],
                    entity_b=resolved[j],
                    tension=float(t),
                    provenances=frozenset(["geometric"]),
                    threshold=float(tau_contra),
                    interpretation=self._interpret_tension(float(t)),
                ))
        hits.sort(key=lambda c: c.tension, reverse=True)
        return hits

    def recommend_next(
        self, context: str, relation: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Recommend the next entity given a context and relation.

        Wraps ``link_predict`` to answer questions like "given that we
        have loaded data, what preprocessing step should come next?"

        Parameters
        ----------
        context : str
            The current entity (e.g., a workflow step or data condition).
        relation : str
            The relation to follow (e.g., ``"precedes"``).
        top_k : int
            Number of candidates to return.

        Returns
        -------
        List of (entity_name, geodesic_distance), best first.
        """
        resolved = self._resolve_entity(context)
        if resolved is None:
            return []
        return self.store.link_predict(resolved, relation, top_k=top_k)

    def explain_why(
        self, head: str, relation: str, tail: str, max_hops: int = 4
    ) -> Explanation:
        """Explain why a claim is or is not plausible.

        Combines direct scoring, path search and contradiction detection
        to build a multi-faceted explanation.

        Parameters
        ----------
        head, relation, tail : str
            The claim to explain.
        max_hops : int
            Maximum hops for supporting path search.

        Returns
        -------
        Explanation with direct score, supporting paths and
        contradicting entities.
        """
        expl = Explanation(head=head, relation=relation, tail=tail)

        # 1. Direct plausibility
        result = self.is_plausible(head, relation, tail)
        if result is not None:
            _, expl.direct_score = result

        # 2. Find supporting paths
        h_resolved = self._resolve_entity(head)
        t_resolved = self._resolve_entity(tail)
        if h_resolved and t_resolved and self.search is not None:
            search_result = self.search.connect_groups(
                {h_resolved}, {t_resolved}, max_hops
            )
            expl.supporting_paths = search_result.paths

        # 3. Contradictions with head and tail, routed through the
        #    two-provenance detector so the Explanation carries the
        #    same audit trail as standalone calls.
        if h_resolved and t_resolved:
            expl.contradicting_entities = self.detect_contradictions(
                [h_resolved, t_resolved]
            )

        # 4. Verdict — gated on calibrated thresholds. If calibration
        #    did not produce a plausibility cut the verdict cannot be
        #    rendered; say so rather than silently defaulting.
        tau_plaus = self.thresholds.tau_plausibility
        if expl.direct_score is None:
            expl.verdict = "unknown (entities not in store)"
        elif tau_plaus is None:
            expl.verdict = "uncalibrated (no plausibility threshold)"
        elif expl.direct_score <= tau_plaus:
            expl.verdict = (
                "supported" if expl.supporting_paths
                else "supported (direct only)"
            )
        else:
            contradicted = any(
                "schema" in c.provenances or
                ("geometric" in c.provenances
                 and c.threshold is not None
                 and c.tension > c.threshold)
                for c in expl.contradicting_entities
            )
            expl.verdict = "contradicted" if contradicted else "unsupported"

        return expl

    def get_all_tensions(
        self, entities: list[str]
    ) -> list[tuple[str, str, float, str]]:
        """Compute all pairwise tension energies (no threshold filter).

        Parameters
        ----------
        entities : list of str
            Entity names.

        Returns
        -------
        List of (entity_a, entity_b, tension, interpretation).
        """
        results = []
        resolved = []
        for e in entities:
            r = self._resolve_entity(e)
            if r is not None:
                resolved.append(r)

        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                t = self.store.tension_energy(resolved[i], resolved[j])
                if t is not None:
                    results.append((
                        resolved[i], resolved[j], t,
                        self._interpret_tension(t),
                    ))
        return results

    def score_triple_set(
        self, triples: Sequence[tuple[str, str, str]]
    ) -> float:
        """Geometric confidence for a set of triples.

        Returns conf = exp(-mean_geodesic) in [0, 1].
        Delegates to GeometricSearch.score_triple_set.
        """
        if self.search is None:
            return 0.5
        return self.search.score_triple_set(triples)
