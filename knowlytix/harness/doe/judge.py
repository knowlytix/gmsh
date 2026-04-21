# SPDX-License-Identifier: Apache-2.0
"""GMS Geometric Judge — scores LLM answers against the manifold.

Instead of only binary correct/incorrect matching, the judge uses the
trained GMS to produce rich geometric verdicts:

1. **Triple extraction**: Parse the LLM answer into (head, rel, tail) triples
2. **Geodesic scoring**: Score each extracted triple against the manifold
3. **Tension verification**: Check for contradictions with known facts
4. **Holonomy checking**: Verify multi-hop reasoning consistency
5. **Confidence aggregation**: Combine geometric signals into a single score

This produces a ``GeometricVerdict`` that goes beyond pass/fail:
- geodesic_confidence:  exp(-mean_geodesic) — how well the answer fits the manifold
- tension_score:  mean tension energy — agreement with existing knowledge
- holonomy_score: path consistency defect — reasoning coherence
- overall_confidence: weighted combination
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


from knowlytix.doc.config import VerifyConfig
from knowlytix.doc.extract import ExtractedTriple, ScoredTriple
from knowlytix.doc.store import GMSExpertStore
from knowlytix.doc.verify import (
    check_contradictions,
    check_holonomy,
    ContradictionResult,
    HolonomyResult,
)
from knowlytix.core.graph.search import GeometricSearch

from .claims import ClaimCompiler
from .verifiers import VerificationRouter
from .verdict import TypedVerdict


@dataclass
class GeometricVerdict:
    """Rich geometric judgment of an LLM answer.

    Goes beyond binary correct/incorrect to provide continuous
    geometric metrics from the GMS manifold.
    """
    # Core result (from traditional scorer)
    correct: bool = False
    detail: str = ""

    # Geometric metrics
    geodesic_confidence: float = 0.0   # exp(-mean_geodesic), [0, 1]
    tension_score: float = 0.0         # mean tension energy, [0, 2]
    holonomy_score: float = 0.0        # mean holonomy defect, [0, ∞)
    overall_confidence: float = 0.0    # weighted combination, [0, 1]

    # Extracted evidence
    extracted_triples: list[ExtractedTriple] = field(default_factory=list)
    scored_triples: list[ScoredTriple] = field(default_factory=list)
    contradictions: list[ContradictionResult] = field(default_factory=list)
    inconsistencies: list[HolonomyResult] = field(default_factory=list)

    # Number of geometric checks performed
    n_triples_scored: int = 0
    n_tension_checks: int = 0
    n_holonomy_checks: int = 0


class GMSJudge:
    """Geometric judge that scores LLM answers against the GMS manifold.

    Parameters
    ----------
    store : GMSExpertStore
        The trained GMS store (ground truth).
    verify_config : VerifyConfig or None
        Thresholds for contradiction and holonomy checks.
    weights : dict or None
        Weights for combining geometric scores into overall confidence.
        Keys: 'geodesic', 'tension', 'holonomy', 'exact'.
    """

    DEFAULT_WEIGHTS = {
        "geodesic": 0.3,   # manifold fit
        "tension": 0.2,    # consistency with known facts
        "holonomy": 0.2,   # reasoning path coherence
        "exact": 0.3,      # traditional exact-match score
    }

    def __init__(
        self,
        store: GMSExpertStore,
        verify_config: VerifyConfig | None = None,
        weights: dict | None = None,
        semantic_mode: str = "off",
        semantic_model: str = "",
        tracing=None,
    ):
        self.store = store
        self.config = verify_config or VerifyConfig()
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.tracing = tracing

        # Build geometric search if model is trained
        self._search: GeometricSearch | None = None
        if store.model is not None and store.transport is not None:
            self._search = GeometricSearch(
                model=store.model,
                transport=store.transport,
                entity_to_idx=store.adapter.entity_to_idx,
                relation_to_idx=store.adapter.relation_to_idx,
                device=store.device,
            )
            if store.doc_graph is not None:
                self._search.build_adjacency(store.doc_graph.triples)

        # GMSH Semantic Verifier (LLM-as-judge anchored by GMS).
        # When ``semantic_model`` is empty, SemanticVerifier defers to
        # ``gms.llm.LLMSettings.model_judge`` (or ``.model``); callers
        # who want to force a specific provider/model pass it in
        # LiteLLM format (e.g. ``"anthropic/claude-opus-4-6"``).
        semantic_verifier = None
        if semantic_mode != "off":
            from .semantic import SemanticVerifier
            semantic_verifier = SemanticVerifier(
                store=store,
                model=semantic_model or None,
                tracing=tracing,
            )

        # GMSH typed verification pipeline
        self._compiler = ClaimCompiler(
            store, semantic_mode=semantic_mode)
        self._router = VerificationRouter(
            store, self.config,
            semantic_verifier=semantic_verifier)

    def judge(
        self,
        llm_answer: str,
        ground_truth,
        exact_correct: bool,
        exact_detail: str,
        question_metadata: dict | None = None,
    ) -> GeometricVerdict:
        """Produce a geometric verdict for an LLM answer.

        Parameters
        ----------
        llm_answer : str
            Raw text answer from the LLM.
        ground_truth : Any
            Expected answer (from question generator).
        exact_correct : bool
            Whether the traditional scorer marked it correct.
        exact_detail : str
            Detail string from the traditional scorer.
        question_metadata : dict or None
            Metadata from the question (head, relation, tail, etc.)

        Returns
        -------
        GeometricVerdict
            Rich geometric judgment.
        """
        verdict = GeometricVerdict(
            correct=exact_correct,
            detail=exact_detail,
        )

        # Step 1: Extract triples from the LLM answer
        triples = self._extract_triples(llm_answer, question_metadata)
        verdict.extracted_triples = triples

        if not triples:
            # Fall back to metadata-based scoring
            verdict = self._score_from_metadata(
                verdict, llm_answer, ground_truth, question_metadata)
            return verdict

        # Step 2: Score triples against the manifold
        scored = self._score_triples(triples)
        verdict.scored_triples = scored
        verdict.n_triples_scored = len(scored)

        if scored:
            geodesics = [st.gms_score for st in scored
                         if st.gms_score is not None]
            if geodesics:
                mean_geo = sum(geodesics) / len(geodesics)
                verdict.geodesic_confidence = math.exp(-mean_geo)

        # Step 3: Tension energy — check for contradictions
        entity_pairs = self._gather_entity_pairs(scored)
        if entity_pairs:
            contradictions = check_contradictions(
                entity_pairs, self.store, self.config)
            verdict.contradictions = contradictions
            verdict.n_tension_checks = len(contradictions)
            if contradictions:
                tensions = [c.tension_energy for c in contradictions]
                verdict.tension_score = sum(tensions) / len(tensions)

        # Step 4: Holonomy — check path consistency
        if scored:
            inconsistencies = check_holonomy(scored, self.store, self.config)
            verdict.inconsistencies = inconsistencies
            verdict.n_holonomy_checks = len(inconsistencies)
            if inconsistencies:
                defects = [h.defect for h in inconsistencies]
                verdict.holonomy_score = sum(defects) / len(defects)

        # Step 5: Compute overall confidence
        verdict.overall_confidence = self._compute_overall(verdict)

        # Build detail string
        verdict.detail = self._build_detail(verdict, exact_detail)

        return verdict

    # ----- GMSH Typed Judging ---------------------------------------------

    def judge_typed(
        self,
        llm_answer: str,
        ground_truth,
        exact_correct: bool,
        exact_detail: str,
        question_metadata: dict | None = None,
        question_category: str = "",
        run_id: str = "",
        question_id: str = "",
        question_text: str = "",
    ) -> TypedVerdict:
        """Produce a TypedVerdict with claim-level GMS verification.

        This is the GMSH enhancement to ``judge()``.  Instead of scoring
        holistically, it:

        1. Compiles the LLM answer into typed claims via ClaimCompiler
        2. Routes each claim to the appropriate GMS sub-verifier
        3. Aggregates sub-verdicts into a TypedVerdict with full audit trail

        The existing ``judge()`` method is unchanged and still available.

        Parameters
        ----------
        llm_answer : str
            Raw text answer from the SUT.
        ground_truth : Any
            Expected answer.
        exact_correct : bool
            Whether the traditional scorer marked it correct.
        exact_detail : str
            Detail string from the traditional scorer.
        question_metadata : dict or None
            Metadata from the question generator.
        question_category : str
            Generator category (e.g. "plausibility").
        run_id : str
            Benchmark run identifier.
        question_id : str
            DOE question identifier.

        Returns
        -------
        TypedVerdict
            Rich verdict with claims, sub-verdicts, and failure attributions.
        """
        # Step 1: Compile claims
        claims = self._compiler.compile(
            llm_answer=llm_answer,
            question_metadata=question_metadata,
            question_category=question_category,
            run_id=run_id,
            question_id=question_id,
            ground_truth=str(ground_truth) if ground_truth else "",
            question_text=question_text,
        )

        # Step 2: Route each claim to the appropriate verifier
        sub_verdicts = self._router.route_all(claims)

        # Step 3: Aggregate into TypedVerdict
        return TypedVerdict.from_sub_verdicts(
            claims=claims,
            sub_verdicts=sub_verdicts,
            exact_correct=exact_correct,
            exact_detail=exact_detail,
            run_id=run_id,
            question_id=question_id,
            weights=self.weights,
        )

    # ----- Triple extraction -----------------------------------------------

    def _extract_triples(
        self,
        text: str,
        metadata: dict | None,
    ) -> list[ExtractedTriple]:
        """Extract triples from LLM answer text and question metadata.

        Uses both metadata-guided extraction (when we know what the question
        is about) and lightweight text pattern matching.
        """
        triples = []

        # From metadata: we know the expected structure
        if metadata:
            head = metadata.get("head")
            relation = metadata.get("relation") or metadata.get("r1")
            tail = metadata.get("tail") or metadata.get("true_tail")
            if head and relation and tail:
                triples.append(ExtractedTriple(
                    head=head, relation=relation, tail=tail))

            # For multi-hop questions
            if metadata.get("mid"):
                r1 = metadata.get("r1")
                r2 = metadata.get("r2")
                mid = metadata.get("mid")
                if r1 and mid:
                    triples.append(ExtractedTriple(
                        head=head, relation=r1, tail=mid))
                if r2 and mid and tail:
                    triples.append(ExtractedTriple(
                        head=mid, relation=r2, tail=tail))

        # From text: match entities mentioned in the answer against store
        if self.store.adapter:
            entities_in_text = []
            for entity in self.store.adapter.entity_to_idx:
                if len(entity) > 2 and entity.lower() in text.lower():
                    entities_in_text.append(entity)

            # For each pair of entities found in the answer, check if a
            # known relation exists
            for i, e1 in enumerate(entities_in_text):
                for e2 in entities_in_text[i + 1:]:
                    for h, r, t in (self.store.doc_graph.triples
                                     if self.store.doc_graph else []):
                        if (h == e1 and t == e2) or (h == e2 and t == e1):
                            triples.append(ExtractedTriple(
                                head=h, relation=r, tail=t))
                            break

        return triples

    def _score_triples(
        self, triples: list[ExtractedTriple]
    ) -> list[ScoredTriple]:
        """Score triples against the GMS manifold."""
        scored = []
        for et in triples:
            head_match = self.store.fuzzy_match_entity(et.head)
            tail_match = self.store.fuzzy_match_entity(et.tail)

            gms_score = None
            if head_match and tail_match:
                gms_score = self.store.score_triple(
                    head_match, et.relation, tail_match)

            scored.append(ScoredTriple(
                triple=et,
                gms_score=gms_score,
                head_match=head_match,
                tail_match=tail_match,
                is_novel=(head_match is None or tail_match is None),
            ))
        return scored

    def _gather_entity_pairs(
        self, scored: list[ScoredTriple]
    ) -> list[tuple[str, str]]:
        """Gather entity pairs for tension energy checks."""
        pairs = set()
        for st in scored:
            if st.head_match and st.tail_match:
                pairs.add((st.head_match, st.tail_match))
            # Also check against related entities in the store
            if st.head_match and self.store.doc_graph:
                related = self.store.query_triples(head=st.head_match)
                for _, _, t in list(related)[:5]:
                    if st.tail_match and t != st.tail_match:
                        pairs.add((st.tail_match, t))
        return list(pairs)

    # ----- Metadata-based scoring (when no triples extracted) ---------------

    def _score_from_metadata(
        self,
        verdict: GeometricVerdict,
        llm_answer: str,
        ground_truth,
        metadata: dict | None,
    ) -> GeometricVerdict:
        """Score using geometric signals from question metadata.

        When we can't extract triples from the answer text, use the
        question's known entities to compute geometric confidence.
        """
        if not metadata:
            verdict.overall_confidence = 1.0 if verdict.correct else 0.0
            return verdict

        # Score the ground truth triple if known
        head = metadata.get("head")
        relation = metadata.get("relation") or metadata.get("r1")
        tail = metadata.get("tail") or metadata.get("true_tail")

        if head and relation and tail:
            score = self.store.score_triple(head, relation, tail)
            if score is not None:
                verdict.geodesic_confidence = math.exp(-score)

        # Tension from metadata
        te_val = metadata.get("tension_energy")
        if te_val is not None:
            verdict.tension_score = te_val

        # Holonomy from metadata
        hol_val = metadata.get("holonomy_defect")
        if hol_val is not None:
            verdict.holonomy_score = hol_val

        # Use triple set confidence from search if available
        if self._search and head and tail:
            triple_set = [(head, relation, tail)] if relation else []
            if triple_set:
                conf = self._search.score_triple_set(triple_set)
                verdict.geodesic_confidence = max(
                    verdict.geodesic_confidence, conf)

        verdict.overall_confidence = self._compute_overall(verdict)
        return verdict

    # ----- Overall confidence computation -----------------------------------

    def _compute_overall(self, verdict: GeometricVerdict) -> float:
        """Compute weighted overall confidence from geometric signals.

        Formula:
            overall = w_exact * exact_score
                    + w_geodesic * geodesic_confidence
                    + w_tension * (1 - tension_score / 2)
                    + w_holonomy * exp(-holonomy_score)
        """
        w = self.weights
        exact_score = 1.0 if verdict.correct else 0.0

        # Tension: 0 = agreement (good) → 1.0, 2 = contradiction → 0.0
        tension_quality = 1.0 - (verdict.tension_score / 2.0)
        tension_quality = max(0.0, min(1.0, tension_quality))

        # Holonomy: 0 = consistent (good) → 1.0, high = inconsistent → 0.0
        holonomy_quality = math.exp(-verdict.holonomy_score)

        overall = (
            w.get("exact", 0.3) * exact_score
            + w.get("geodesic", 0.3) * verdict.geodesic_confidence
            + w.get("tension", 0.2) * tension_quality
            + w.get("holonomy", 0.2) * holonomy_quality
        )
        return max(0.0, min(1.0, overall))

    # ----- Reporting -------------------------------------------------------

    def _build_detail(self, v: GeometricVerdict, exact_detail: str) -> str:
        """Build a human-readable detail string."""
        parts = [exact_detail]
        parts.append(
            f"GMS: geo_conf={v.geodesic_confidence:.3f} "
            f"tension={v.tension_score:.3f} "
            f"holonomy={v.holonomy_score:.3f} "
            f"overall={v.overall_confidence:.3f}"
        )
        if v.contradictions:
            n_contra = sum(1 for c in v.contradictions if c.is_contradiction)
            if n_contra > 0:
                parts.append(f"CONTRADICTIONS: {n_contra} found")
        if v.inconsistencies:
            n_incon = sum(1 for h in v.inconsistencies if not h.is_consistent)
            if n_incon > 0:
                parts.append(f"INCONSISTENCIES: {n_incon} paths")
        return " | ".join(parts)


# =========================================================================
# Convenience
# =========================================================================

def create_judge(store: GMSExpertStore, **kwargs) -> GMSJudge:
    """Create a GMSJudge from a store with default config."""
    return GMSJudge(store, **kwargs)
