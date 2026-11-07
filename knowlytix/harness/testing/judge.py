# SPDX-License-Identifier: Apache-2.0
"""GMS Geometric Judge — scores LLM answers against the manifold.

Instead of only binary correct/incorrect matching, the judge uses the
trained GMS to produce rich geometric verdicts:

1. **Triple extraction**: Parse the LLM answer into (head, rel, tail) triples
2. **Geodesic scoring**: Score each extracted triple against the manifold
3. **Tension verification**: Check for contradictions with known facts
4. **Holonomy checking**: Verify multi-hop reasoning consistency
5. **Phase scoring**: Compare numeric values via phase encoding on S^1
6. **Calibrated classification**: Per-category logistic regression on the
   measure that matches each question type

This produces a ``GeometricVerdict`` that goes beyond pass/fail:
- geodesic_confidence:  exp(-mean_geodesic) — how well the answer fits the manifold
- tension_score:  mean tension energy — agreement with existing knowledge
- holonomy_score: path consistency defect — reasoning coherence
- phase_score:  angular distance on S^1 — numeric precision
- overall_confidence: weighted combination (or calibrated probability)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from knowlytix.core.graph.search import GeometricSearch
from knowlytix.knowledge.config import VerifyConfig
from knowlytix.knowledge.extract import ExtractedTriple, ScoredTriple
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.knowledge.verify import (
    ContradictionResult,
    HolonomyResult,
    check_contradictions,
    check_holonomy,
)

from .claims import ClaimCompiler
from .primitive_calibration import (
    label_holonomy,
    label_tension,
    samples_to_arrays,
)
from .verdict import TypedVerdict
from .verifiers import VerificationRouter


def _derive_thresholds(clf, X):
    """Scan the fitted classifier to extract interpretable thresholds.

    For a 2-class classifier returns a single float τ such that
    ``P(class=1 | value=τ) = 0.5``.  For a 3-class classifier returns
    the two value boundaries where the arg-max class changes.  Assumes
    ``X`` is a one-feature column; threshold is sought inside its
    observed range.
    """
    try:
        lo = float(np.min(X))
        hi = float(np.max(X))
    except (ValueError, TypeError):
        return None
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 801).reshape(-1, 1)
    pred = clf.predict(grid).astype(int)
    classes = getattr(clf, "classes_", np.unique(pred))
    if len(classes) <= 2:
        # find the first crossing where prediction changes
        changes = np.where(np.diff(pred) != 0)[0]
        if len(changes) == 0:
            return None
        return float(grid[changes[0], 0])
    # Multi-class: return tuple of crossings
    changes = np.where(np.diff(pred) != 0)[0]
    if len(changes) == 0:
        return None
    return tuple(float(grid[c, 0]) for c in changes)


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
    phase_score: float = 0.0           # angular distance on S^1, [0, π]
    overall_confidence: float = 0.0    # weighted combination, [0, 1]

    # Calibrated classification (set by calibrated_classify)
    calibrated_verdict: bool | None = None   # True/False from calibration
    calibrated_prob: float = 0.0             # P(correct) from logistic reg
    calibration_measure: str = ""            # which measure was used
    calibration_score: float | None = None   # raw feature value (d, E, δ, θ)
    calibration_detail: str = ""             # one-line decision explanation

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
        claim_extraction: str = "rule",
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

        # GMSH typed verification pipeline. ``claim_extraction="geometric"``
        # parses the agent's answer into store-bound triples (answer-driven),
        # scoring each against the GMS; "rule" (default) takes head/relation
        # from the generator schema and scrapes the tail.
        claim_extractor = None
        if claim_extraction == "geometric":
            from .claims import GeometricClaimExtractor
            claim_extractor = GeometricClaimExtractor(store)
        self._compiler = ClaimCompiler(
            store, semantic_mode=semantic_mode, extractor=claim_extractor)
        from .completeness import CompletenessEvaluator
        self._completeness = CompletenessEvaluator(store)
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

        # Step 3: Tension energy — check for contradictions.
        # Skipped (signal_unavailable) when tau_contra is None,
        # which is the canonical state for corpora with no
        # opposite_of / is_functional declarations. The verify-layer
        # gate raises on unconfigured tau_contra to prevent silent
        # miscalibration; the judge honours that contract by not
        # invoking the gate when the threshold is absent.
        entity_pairs = self._gather_entity_pairs(scored)
        if entity_pairs and self.config.tau_contra is not None:
            contradictions = check_contradictions(
                entity_pairs, self.store, self.config)
            verdict.contradictions = contradictions
            verdict.n_tension_checks = len(contradictions)
            if contradictions:
                tensions = [c.tension_energy for c in contradictions]
                verdict.tension_score = sum(tensions) / len(tensions)

        # Step 4: Holonomy — check path consistency. Same contract as
        # tension: skipped when tau_path is None.
        if scored and self.config.tau_path is not None:
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
        verdict = TypedVerdict.from_sub_verdicts(
            claims=claims,
            sub_verdicts=sub_verdicts,
            exact_correct=exact_correct,
            exact_detail=exact_detail,
            run_id=run_id,
            question_id=question_id,
            weights=self.weights,
        )
        # Completeness (recall): coverage of the question's expected ground-truth
        # atoms by the answer -- a separate axis from claim correctness above.
        verdict.completeness = self._completeness.evaluate(
            llm_answer, ground_truth, question_metadata).as_dict()
        return verdict

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
        head = metadata.get("head") or metadata.get("entity")
        relation = (metadata.get("relation") or metadata.get("r1")
                    or metadata.get("consequence_relation"))
        tail = (metadata.get("tail") or metadata.get("true_tail")
                or metadata.get("consequence_value"))

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

        # Phase score from metadata (boundary / phase_comparison)
        phase_encoders = getattr(self.store.doc_graph, "phase_encoders", {})
        enm_type = metadata.get("enm_type") or metadata.get("category")
        if enm_type and enm_type in phase_encoders:
            pe = phase_encoders[enm_type]
            test_val = metadata.get("test_value") or metadata.get("value_a")
            if test_val is not None:
                verdict.phase_score = pe.encode(float(test_val))

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

    # ----- Category-to-measure mapping ------------------------------------

    # Maps each question category to its primary GMS primitive (Ch 5).
    # Tension and holonomy are diagnostic overlays: they describe the
    # *type* of geometric evidence but don't separate correct from
    # incorrect answers.  All triple-based categories classify via
    # geodesic distance; only phase categories use phase angle.
    CATEGORY_MEASURE = {
        "plausibility":       "geodesic",
        "link_prediction":    "geodesic",
        "multi_hop_transport": "geodesic",
        "conditional":        "geodesic",
        "procedural":         "geodesic",
        "robustness":         "geodesic",
        "recovery_test":      "geodesic",
        "governance":         "geodesic",
        "tension":            "geodesic",
        "adversarial":        "geodesic",
        "holonomy":           "geodesic",
        "phase_comparison":   "phase",
        "boundary":           "enm",
    }

    # Maps category to the diagnostic signal it was designed to test.
    CATEGORY_DIAGNOSTIC = {
        "plausibility":       "geodesic",
        "link_prediction":    "geodesic",
        "multi_hop_transport": "geodesic",
        "conditional":        "geodesic",
        "procedural":         "geodesic",
        "robustness":         "geodesic",
        "recovery_test":      "geodesic",
        "governance":         "geodesic",
        "tension":            "tension",
        "adversarial":        "tension",
        "holonomy":           "holonomy",
        "phase_comparison":   "phase",
        "boundary":           "enm",
    }

    def calibrate(self, seed: int = 42) -> dict:
        """Fit per-measure logistic regressions from the store's training data.

        Produces five classifiers:

        * **geodesic** — ``score_triple(h, r, t)`` on single-hop triples
        * **path_transport** — ``transport.path_score(h, [r1, r2], t)``
          on 2-hop paths with composed rotors
        * **tension** — ``tension_energy(a, b)`` on entity pairs
        * **holonomy** — ``check_holonomy([r_path], r_direct)`` on
          relation triangles
        * **phase** — phase encoder angles on S^1

        Each classifier is trained on positive (real) vs negative
        (corrupted) examples using the measure for that category.

        Returns dict of measure_name → fitted sklearn LogisticRegression.
        """
        from sklearn.linear_model import LogisticRegression

        rng = random.Random(seed)
        triples = list(self.store.doc_graph.triples)
        entities = list(self.store.adapter.entity_to_idx.keys())

        # ── Geodesic: score_triple(h, r, t) ──
        X_geo, y_geo = [], []
        for h, r, t in triples:
            geo = self.store.score_triple(h, r, t)
            if geo is None:
                continue
            X_geo.append([geo])
            y_geo.append(1)
            fake_t = rng.choice(entities)
            while fake_t == t:
                fake_t = rng.choice(entities)
            geo_neg = self.store.score_triple(h, r, fake_t)
            X_geo.append([geo_neg if geo_neg is not None else math.pi])
            y_geo.append(0)

        # ── Path transport: path_score(h, [r1, r2], t) ──
        # Composed rotor R_{r2} R_{r1} applied to head; geodesic to tail.
        # Path distances are systematically larger than single-hop because
        # composed rotors accumulate curvature error.
        X_path, y_path = [], []
        if self.store.transport is not None:
            adj: dict[str, list[tuple[str, str]]] = {}
            for h, r, t in triples:
                adj.setdefault(h, []).append((r, t))

            path_count = 0
            for h, r1, mid in triples:
                if mid not in adj:
                    continue
                for r2, tail in adj[mid]:
                    if tail == h or tail == mid:
                        continue
                    h_idx = self.store.adapter.entity_to_idx.get(h)
                    r1_idx = self.store.adapter.relation_to_idx.get(r1)
                    r2_idx = self.store.adapter.relation_to_idx.get(r2)
                    t_idx = self.store.adapter.entity_to_idx.get(tail)
                    if any(x is None for x in [h_idx, r1_idx, r2_idx, t_idx]):
                        continue
                    with torch.no_grad():
                        geo = self.store.transport.path_score(
                            torch.tensor([h_idx], device=self.store.device),
                            [r1_idx, r2_idx],
                            torch.tensor([t_idx], device=self.store.device),
                        ).item()
                    X_path.append([geo])
                    y_path.append(1)

                    # Negative: corrupt the tail
                    fake_t = rng.choice(entities)
                    while fake_t == tail:
                        fake_t = rng.choice(entities)
                    ft_idx = self.store.adapter.entity_to_idx.get(fake_t)
                    if ft_idx is not None:
                        with torch.no_grad():
                            geo_neg = self.store.transport.path_score(
                                torch.tensor([h_idx],
                                             device=self.store.device),
                                [r1_idx, r2_idx],
                                torch.tensor([ft_idx],
                                             device=self.store.device),
                            ).item()
                        X_path.append([geo_neg])
                        y_path.append(0)

                    path_count += 1
                    if path_count >= 600:
                        break
                if path_count >= 600:
                    break

        # ── Tension: agree / unrelated / contradict from graph structure ──
        tension_samples = label_tension(self.store, seed=seed)
        X_ten_np, y_ten_np = samples_to_arrays(tension_samples)
        X_ten = X_ten_np.tolist()
        y_ten = y_ten_np.tolist()
        self._tension_samples = tension_samples

        # ── Holonomy: consistent / twisted paths ──
        holonomy_samples = label_holonomy(self.store, seed=seed)
        X_hol_np, y_hol_np = samples_to_arrays(holonomy_samples)
        X_hol = X_hol_np.tolist()
        y_hol = y_hol_np.tolist()
        self._holonomy_samples = holonomy_samples

        # ── Phase: angular distance on S^1 via phase encoders ──
        X_pha, y_pha = [], []
        phase_encoders = getattr(self.store.doc_graph, "phase_encoders", {})
        if phase_encoders and self.store.enm:
            enm_by_type = {}
            for key in self.store.enm.keys():
                val = self.store.enm.read_exact(key)
                if val is not None:
                    v = float(val.item() if hasattr(val, "item") else val)
                    enm_by_type.setdefault(key.type, []).append(
                        (key.id, v))

            for cat, entries in enm_by_type.items():
                pe = phase_encoders.get(cat)
                if pe is None or len(entries) < 2:
                    continue
                for eid, val in entries:
                    angle_real = pe.encode(val)
                    X_pha.append([angle_real])
                    y_pha.append(1)
                    # Negative: perturbed value outside range
                    fake_val = val + (pe.v_max - pe.v_min) * 0.5
                    angle_fake = pe.encode(fake_val)
                    X_pha.append([angle_fake])
                    y_pha.append(0)

        # Fit classifiers
        calibrations = {}
        thresholds: dict[str, float | tuple[float, ...] | str] = {}
        for name, X_raw, y_raw in [
            ("geodesic", X_geo, y_geo),
            ("path_transport", X_path, y_path),
            ("tension", X_ten, y_ten),
            ("holonomy", X_hol, y_hol),
            ("phase", X_pha, y_pha),
        ]:
            if len(X_raw) < 4:
                if name == "tension":
                    thresholds["tension_status"] = "insufficient_data"
                continue
            X_arr = np.array(X_raw)
            y_arr = np.array(y_raw)
            classes = np.unique(y_arr)
            if len(classes) < 2:
                print(f"  Skipped {name:14s}: only one class present "
                      f"({classes.tolist()}), n={len(y_arr)}")
                if name == "tension":
                    thresholds["tension_status"] = "insufficient_classes"
                continue
            clf = LogisticRegression(random_state=seed, max_iter=2000)
            clf.fit(X_arr, y_arr)
            y_pred = clf.predict(X_arr)
            acc = float((y_arr == y_pred).sum()) / len(y_arr)
            calibrations[name] = clf

            # Tension goes through the strict 3-class calibrator so a
            # degenerate single-crossing fit is refused rather than
            # silently collapsed into tau_ent == tau_contra.
            if name == "tension":
                from knowlytix.harness.testing.threshold_calibration import (
                    calibrate_logic_tension,
                )
                samples_for_fit = self._tension_samples
                strict_clf, cuts = calibrate_logic_tension(
                    samples_for_fit, seed=seed,
                )
                if strict_clf is None or cuts is None:
                    thresholds["tension_status"] = "degenerate"
                    print(
                        f"  Calibrated {name:14s}: {len(y_arr):5d} samples, "
                        f"{len(classes)}-class, train accuracy={acc:.1%}  "
                        f"τ=DEGENERATE (no usable two-cut separation)"
                    )
                    continue
                calibrations[name] = strict_clf
                thresholds["tension"] = (cuts.tau_ent, cuts.tau_contra)
                thresholds["tau_ent"] = float(cuts.tau_ent)
                thresholds["tau_contra"] = float(cuts.tau_contra)
                thresholds["tension_status"] = "ok"
                print(
                    f"  Calibrated {name:14s}: {len(y_arr):5d} samples, "
                    f"{len(classes)}-class, train accuracy={acc:.1%}  "
                    f"τ_ent={cuts.tau_ent:.4f} τ_contra={cuts.tau_contra:.4f}"
                )
                continue

            # Derive interpretable thresholds by scanning the value range
            tau = _derive_thresholds(clf, X_arr)
            if tau is not None:
                thresholds[name] = tau

            tau_str = (
                f"  τ={tau}"
                if tau is not None else ""
            )
            print(f"  Calibrated {name:14s}: {len(y_arr):5d} samples, "
                  f"{len(classes)}-class, train accuracy={acc:.1%}"
                  + tau_str)

        self._calibrations = calibrations
        self._thresholds = thresholds
        # Rebuild the typed-verdict router so the freshly calibrated thresholds
        # actually reach the verifiers. The router built in __init__ predates
        # calibration -- it captured empty thresholds, leaving
        # GraphVerifier.plausibility_threshold None so every fact claim came
        # back NOT_CALIBRATED. Passing judge=self wires the calibrated operating
        # points (geodesic / holonomy / tension / policy) and classifiers through.
        self._router = VerificationRouter(
            self.store, self.config,
            semantic_verifier=getattr(self._router, "semantic", None),
            judge=self)
        # Push the scalar thresholds onto the freshly built verifiers too
        # (idempotent): keeps the documented threshold->verifier mapping working
        # for judges that are loaded but not recalibrated (load() also calls it).
        self.apply_thresholds()
        return calibrations

    def apply_thresholds(self) -> None:
        """Push calibrated thresholds into the live verification router.

        ``calibrate()`` and :meth:`load` populate ``self._thresholds``, but the
        verifiers inside ``self._router`` are constructed in ``__init__`` —
        before any calibration exists — and do not re-read that dict on their
        own. Without this step a freshly calibrated (or loaded) judge still
        routes claims through the verifiers' *uncalibrated* decision
        boundaries, so typed verdicts come back ``NOT_CALIBRATED``.

        This method copies each calibrated threshold onto the corresponding
        verifier using the documented mapping:

        ===============  ======================================
        threshold key    verifier attribute
        ===============  ======================================
        ``geodesic``     ``graph.plausibility_threshold``
        ``holonomy``     ``logic.config.tau_path``
        ``tau_ent``      ``logic.config.tau_ent``
        ``tau_contra``   ``logic.config.tau_contra``
        ===============  ======================================

        It is idempotent and safe to call repeatedly. Keys that are absent (or
        verifiers that are not present on the router) are left untouched, so it
        is a no-op when nothing has been calibrated yet.
        """
        thr = getattr(self, "_thresholds", None) or {}
        router = getattr(self, "_router", None)
        if router is None:
            return
        graph = getattr(router, "graph", None)
        if graph is not None and thr.get("geodesic") is not None:
            graph.plausibility_threshold = float(thr["geodesic"])
        logic = getattr(router, "logic", None)
        if logic is not None and getattr(logic, "config", None) is not None:
            if thr.get("holonomy") is not None:
                logic.config.tau_path = float(thr["holonomy"])
            if thr.get("tau_ent") is not None:
                logic.config.tau_ent = float(thr["tau_ent"])
            if thr.get("tau_contra") is not None:
                logic.config.tau_contra = float(thr["tau_contra"])

    def calibrate_policy(
        self,
        relation: str = "uses_tool",
        *,
        positive_triples=None,
        negative_triples=None,
        seed: int = 42,
        max_negatives_per_positive: int = 3,
    ):
        """Calibrate the PolicyVerifier geodesic threshold.

        This is a separate calibration from :meth:`calibrate` because
        the tool-use corpus is disjoint from the content corpus:
        ``(agent, uses_tool, tool)`` triples live in a different region
        of the manifold and need their own labeled sample.

        On success, stores the derived cut under
        ``self._thresholds["policy_geodesic"]`` so it is picked up by
        :class:`VerificationRouter`.

        Parameters
        ----------
        relation : str
            Tool-use relation name in the store's adapter vocabulary.
        positive_triples, negative_triples : iterables, optional
            See :func:`threshold_calibration.label_policy_geodesic`.
        seed, max_negatives_per_positive : passthrough parameters.

        Returns
        -------
        (classifier, threshold) or ``(None, None)`` if calibration
        cannot proceed (e.g.\\ relation not in store, no scorable
        positives).
        """
        from .threshold_calibration import (
            calibrate_policy_geodesic,
            label_policy_geodesic,
        )

        # Cap-trained store: the admissibility radius is already learned
        # during training (gms_paper.tex Section 7).  Use it directly as the
        # policy threshold; no post-hoc Youden-J fit required.
        if getattr(self.store.model, "cap_enabled", False):
            rho = self.store.cap_radius(relation)
            if rho is not None:
                if not hasattr(self, "_thresholds") or self._thresholds is None:
                    self._thresholds = {}
                self._thresholds["policy_geodesic"] = float(rho)
                # Expose per-relation radii as a diagnostic artifact.
                self._thresholds["cap_radii"] = self.store.cap_radii_all() or {}
                print(f"  Policy radius (cap-trained): relation='{relation}', "
                      f"ρ={rho:.4f}")
                return None, float(rho)
            # Fall through to the classifier-based calibration if the
            # relation name is not in the adapter vocabulary.

        samples = label_policy_geodesic(
            self.store,
            relation=relation,
            positive_triples=positive_triples,
            negative_triples=negative_triples,
            seed=seed,
            max_negatives_per_positive=max_negatives_per_positive,
        )
        clf, tau = calibrate_policy_geodesic(samples, seed=seed)
        if not hasattr(self, "_thresholds") or self._thresholds is None:
            self._thresholds = {}
        if tau is not None:
            self._thresholds["policy_geodesic"] = float(tau)
            if not hasattr(self, "_calibrations") or self._calibrations is None:
                self._calibrations = {}
            self._calibrations["policy_geodesic"] = clf
            self._policy_samples = samples
            print(f"  Calibrated policy_geodesic: {len(samples):5d} samples, "
                  f"τ={tau:.4f}")
        else:
            # Fallback: reuse the general `geodesic` threshold so downstream
            # gates always have some value to consume, and flag it so the
            # caller knows policy-specific calibration did not fit.
            fallback = self._thresholds.get("geodesic")
            if fallback is not None:
                self._thresholds["policy_geodesic"] = float(fallback)
                self._thresholds["policy_geodesic_fallback"] = True
                print(f"  Skipped policy_geodesic: insufficient labeled "
                      f"samples for relation='{relation}'; "
                      f"using geodesic τ={fallback:.4f} as fallback")
            else:
                print(f"  Skipped policy_geodesic: insufficient labeled "
                      f"samples for relation='{relation}' "
                      f"(no fallback available -- run calibrate() first)")
        return clf, tau

    def calibrated_classify(
        self,
        verdict: GeometricVerdict,
        category: str,
        llm_answer: str = "",
        question_metadata: dict | None = None,
    ) -> GeometricVerdict:
        """Apply calibrated geometric classification to the verdict.

        Each category uses its principled GMS primitive:

        * **Geodesic** (single-hop): ``score_triple(h, r, answer_entity)``
          — one relation rotor applied to head, geodesic to answer.
        * **Geodesic** (multi-hop / procedural): ``transport.path_score(
          h, [r1, ..., rL], answer_entity)`` — composed rotor chain
          ``R_{r_L} \\cdots R_{r_1}`` applied to head, geodesic to answer.
        * **Tension**: ``tension_energy(a, b)`` with known thresholds.
        * **Holonomy**: ``check_holonomy([r_path], r_direct)`` —
          ``\\|R_{path} R_{direct}^{-1} - I\\|_F``.
        * **Phase**: phase encoder angle on S^1.

        No ad-hoc fallbacks.  Every question was generated from the
        graph so proper triples can always be constructed from the
        question metadata + LLM answer.
        """
        if not hasattr(self, "_calibrations") or not self._calibrations:
            raise RuntimeError("Call judge.calibrate() first")

        meta = question_metadata or {}
        ans_text = llm_answer.strip() if llm_answer else ""
        low = ans_text.lower()

        # ── Tension: direct geometric verification ──────────────
        if category == "tension":
            return self._classify_tension(verdict, low, meta)

        # ── Holonomy: direct geometric verification ─────────────
        if category == "holonomy":
            return self._classify_holonomy(verdict, low, meta)

        # ── ENM / Phase: direct verification ──────────────────────
        measure = self.CATEGORY_MEASURE.get(category, "geodesic")
        if measure in ("phase", "enm"):
            return self._classify_phase(verdict, category, low, meta)

        # ── Absolute plausibility (True/False): direct geodesic ──
        # Absolute plausibility asks "is this triple well-supported?"
        # Use the geodesic distance directly against the generator's
        # threshold (0.5), not the logistic regression.
        if (category == "plausibility"
                and low in ("true", "false")
                and "geodesic" in meta):
            verdict.calibration_measure = "geodesic"
            geo_in_meta = meta["geodesic"]
            geo_plausible = geo_in_meta < 0.5
            llm_says_plausible = low == "true"
            verdict.calibrated_verdict = (geo_plausible == llm_says_plausible)
            verdict.calibrated_prob = (
                1.0 if verdict.calibrated_verdict else 0.0)
            verdict.calibration_score = geo_in_meta
            verdict.calibration_detail = (
                f"geodesic={geo_in_meta:.4f} "
                f"({'<' if geo_plausible else '>'} 0.5 threshold) "
                f"→ {'plausible' if geo_plausible else 'not plausible'}, "
                f"LLM says {'plausible' if llm_says_plausible else 'not plausible'}"
            )
            return verdict

        # ── Geodesic: score proper answer triple(s) ─────────────
        # All triple-based categories use the geodesic calibration.
        # Multi-hop/procedural decompose into individual hop triples
        # (a path is valid iff every hop is valid).
        clf = self._calibrations.get("geodesic")
        if clf is None:
            verdict.calibrated_verdict = verdict.correct
            verdict.calibration_measure = "geodesic"
            return verdict

        feature = self._geodesic_feature(
            category, ans_text, low, meta, verdict)

        if feature is None:
            # No scoreable triple — geometric judge cannot classify
            verdict.calibrated_verdict = verdict.correct
            verdict.calibration_measure = "geodesic"
            return verdict

        # ── ENM cross-check for numeric answers ────────────────
        # When the answer entity is numeric and has an ENM match,
        # ENM provides an exact value lookup that overrides the
        # geodesic embedding (exact > approximate for numbers).
        enm_result = self._enm_verify_numeric(ans_text, low, meta)
        if enm_result is not None:
            enm_match, enm_val, enm_detail = enm_result
            verdict.calibration_measure = "enm"
            verdict.calibration_score = feature
            if low in ("true", "false"):
                llm_says_real = low == "true"
                verdict.calibrated_verdict = (enm_match == llm_says_real)
            else:
                verdict.calibrated_verdict = enm_match
            verdict.calibrated_prob = (
                1.0 if verdict.calibrated_verdict else 0.0)
            verdict.calibration_detail = (
                f"geodesic={feature:.4f}, {enm_detail}"
            )
            return verdict

        X = np.array([[feature]])

        # True/False about triple validity: flip interpretation
        if low in ("true", "false"):
            p_real = float(clf.predict_proba(X)[0, 1])
            verdict.calibrated_verdict = (
                (p_real > 0.5) if low == "true" else (p_real <= 0.5))
            verdict.calibrated_prob = (
                p_real if low == "true" else (1.0 - p_real))
            verdict.calibration_detail = (
                f"geodesic={feature:.4f}, p_real={p_real:.4f} "
                f"→ {'real' if p_real > 0.5 else 'not real'}, "
                f"LLM says {'real' if low == 'true' else 'not real'}"
            )
        else:
            verdict.calibrated_verdict = bool(clf.predict(X)[0])
            p_real = float(clf.predict_proba(X)[0, 1])
            verdict.calibrated_prob = p_real
            verdict.calibration_detail = (
                f"geodesic={feature:.4f}, p_real={p_real:.4f} "
                f"→ {'ACCEPT' if verdict.calibrated_verdict else 'REJECT'}"
            )

        verdict.calibration_measure = "geodesic"
        verdict.calibration_score = feature
        return verdict

    # ── Per-category feature extraction ──────────────────────────

    def _geodesic_feature(
        self,
        category: str,
        ans_text: str,
        low: str,
        meta: dict,
        verdict: GeometricVerdict,
    ) -> float | None:
        """Extract geodesic feature from proper answer triple(s).

        Multi-hop and procedural categories use composed rotor transport
        ``path_score(h, [r1,...,rL], t)`` on the full relation chain —
        not individual-hop averages.
        """
        # ── Multi-hop: composed rotor transport ─────────────────
        if category == "multi_hop_transport":
            return self._path_transport_feature(ans_text, meta)

        # ── Procedural: composed rotor transport over chain ─────
        if category == "procedural":
            return self._procedural_feature(ans_text, meta)

        # ── Single-hop: score (head, relation, answer_entity) ───
        ans_entity = self._resolve_answer_entity(
            ans_text, low, meta, verdict)

        head = meta.get("head") or meta.get("entity")
        relation = (meta.get("relation") or meta.get("r1")
                    or meta.get("consequence_relation"))

        if head and relation and ans_entity:
            geo = self.store.score_triple(head, relation, ans_entity)
            if geo is not None:
                return geo

        return None

    def _path_transport_feature(
        self, ans_text: str, meta: dict
    ) -> float | None:
        """Score the answer hop of a multi-hop question as a single triple.

        For ``h --r1--> mid --r2--> ?``, the intermediate ``mid`` is
        known from the question.  The LLM answer is the final entity.
        Score the answer hop ``(mid, r2, answer_entity)`` via the
        single-hop geodesic — a proper triple at full calibration
        accuracy.

        Principle: a path is valid iff every hop is valid.  The known
        hops from the question are real triples in the graph (always
        valid).  The only unknown hop involves the LLM answer.
        """
        mid = meta.get("mid")
        r2 = meta.get("r2")
        ans_entity = self.store.fuzzy_match_entity(ans_text) if ans_text else None

        if mid and r2 and ans_entity:
            geo = self.store.score_triple(mid, r2, ans_entity)
            if geo is not None:
                return geo

        return None

    def _procedural_feature(
        self, ans_text: str, meta: dict
    ) -> float | None:
        """Score the LLM's answer entities against the expected chain.

        Parses the LLM's comma-separated answer into entities and scores
        each one against the corresponding hop in the chain.  Returns the
        **max** geodesic distance (weakest link).  If the LLM provides
        fewer entities than the chain, the missing hops are scored with
        the max geodesic from the present hops (conservative).

        This is the procedural analog of ``_path_transport_feature``:
        the chain structure is known from the generator; the LLM's
        contribution is the entity at each step.
        """
        chain = meta.get("chain", [])
        steps = meta.get("steps")        # mixed: [(r1, mid1), ...]
        rel = meta.get("relation")        # same-relation chains

        if len(chain) < 2:
            return None

        # Parse LLM answer into entity candidates
        # Match each answer token against known store entities
        llm_entities: list[str | None] = []
        if ans_text:
            # Split on commas, but match whole tokens against store
            # to avoid splitting entity names containing commas
            known_chain = set(chain[1:])  # expected entities
            remaining = ans_text
            for expected in chain[1:]:
                matched = self.store.fuzzy_match_entity(expected)
                if matched and matched.lower() in remaining.lower():
                    llm_entities.append(matched)
                    # Remove from remaining to avoid double-matching
                    idx = remaining.lower().find(matched.lower())
                    if idx >= 0:
                        remaining = (remaining[:idx]
                                     + remaining[idx + len(matched):])
                else:
                    # Try fuzzy match on comma-split tokens
                    found = False
                    for token in remaining.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        ent = self.store.fuzzy_match_entity(token)
                        if ent:
                            llm_entities.append(ent)
                            found = True
                            break
                    if not found:
                        llm_entities.append(None)

        # If no entities parsed, try the whole answer as a single entity
        if not llm_entities and ans_text:
            ent = self.store.fuzzy_match_entity(ans_text)
            if ent:
                llm_entities = [ent]

        if not llm_entities:
            return None

        # Score each hop: (chain[i-1], relation, llm_entity[i-1])
        geodesics = []
        for i in range(1, len(chain)):
            prev = chain[i - 1]
            if steps and (i - 1) < len(steps):
                hop_rel = steps[i - 1][0]
            elif rel:
                hop_rel = rel
            else:
                continue

            # Use LLM entity if available, else skip
            llm_idx = i - 1
            if llm_idx < len(llm_entities) and llm_entities[llm_idx]:
                ent = llm_entities[llm_idx]
            else:
                continue

            geo = self.store.score_triple(prev, hop_rel, ent)
            if geo is not None:
                geodesics.append(geo)

        return max(geodesics) if geodesics else None

    def _resolve_answer_entity(
        self,
        ans_text: str,
        low: str,
        meta: dict,
        verdict: GeometricVerdict,
    ) -> str | None:
        """Map LLM answer text to a store entity for single-hop scoring.

        * **A/B labels**: resolve to the entity the label represents
          (true_tail if correct, wrong_tail if incorrect).
        * **True/False**: score the question's known tail triple.
        * **Entity name**: fuzzy-match directly.
        * **Prefixed answers** (e.g. ``ESCALATION REQUIRED: High``):
          extract entity after the colon.
        """
        # A/B label: resolve via metadata
        if low in ("a", "b"):
            true_tail = meta.get("true_tail") or meta.get("tail")
            wrong_tail = meta.get("wrong_tail")
            if verdict.correct and true_tail:
                return self.store.fuzzy_match_entity(true_tail)
            elif not verdict.correct and wrong_tail:
                return self.store.fuzzy_match_entity(wrong_tail)
            return None

        # True/False: score the question's known triple
        if low in ("true", "false"):
            tail = meta.get("tail") or meta.get("true_tail")
            return self.store.fuzzy_match_entity(tail) if tail else None

        # Direct entity name
        if ans_text:
            ent = self.store.fuzzy_match_entity(ans_text)
            if ent:
                return ent
            # Try extracting entity after colon (governance prefix)
            if ":" in ans_text:
                after = ans_text.split(":", 1)[1].strip()
                if after:
                    return self.store.fuzzy_match_entity(after)

        return None

    # ── ENM: numeric value verification ────────────────────────────

    def _enm_verify_numeric(
        self,
        ans_text: str,
        low: str,
        meta: dict,
    ) -> tuple[bool, float, str] | None:
        """Check if a numeric answer entity matches an ENM value.

        For geodesic categories with numeric tails, ENM provides exact
        value lookups that are more reliable than geodesic distance
        (which treats numbers as opaque entity tokens).

        **One-sided confirmation only**: if ENM finds the exact value
        for this head entity, it confirms the answer.  If ENM has no
        match, it falls through to geodesic (not a rejection), because
        the head entity may appear in many unrelated ENM entries and
        a value mismatch doesn't prove the answer is wrong.

        Returns ``(True, enm_value, detail_string)`` if ENM confirms,
        or ``None`` if non-numeric, no ENM entry, or no match.
        """
        if not self.store.enm:
            return None

        head = meta.get("head") or meta.get("entity")
        if not head:
            return None

        # Resolve the answer to a tail value
        if low in ("true", "false"):
            tail = meta.get("tail") or meta.get("true_tail")
        elif low in ("a", "b"):
            # Don't attempt ENM for A/B — need to know which option
            # maps to which value, and the search is ambiguous
            return None
        else:
            tail = ans_text

        if not tail:
            return None

        # Parse tail as numeric
        try:
            t_val = float(
                str(tail).replace(",", "").replace("%", "").replace("$", ""))
        except (ValueError, AttributeError):
            return None

        # Search ENM for entries containing head entity with exact match
        for k in self.store.enm.keys():
            if head.lower() not in k.id.lower():
                continue
            v = self.store.enm.read_exact(k)
            if v is None:
                continue
            ev = float(v.item())
            if abs(ev - t_val) < 1e-6:  # exact match (float precision)
                detail = (
                    f"ENM({k.type}): {k.id[:40]}={ev} "
                    f"confirms {t_val}"
                )
                return (True, ev, detail)

        # No ENM confirmation — fall back to geodesic (not a rejection)
        return None

    # ── Calibrated threshold accessors ───────────────────────────

    # Geometric anchors — used as fallback when calibrate() has not
    # been called or the labelers found too few samples.  The agree/
    # unrelated split sits near E = sqrt(2); the unrelated/contradict
    # split is placed symmetrically above it.
    _TENSION_AGREE_ANCHOR = 1.0
    _TENSION_CONTRADICT_ANCHOR = 1.6
    _HOLONOMY_ANCHOR = 2.0

    def _tension_label_from_value(self, te: float) -> str:
        """Return "agree" / "unrelated" / "contradict" for a tension
        value.  Uses the fitted classifier for the classes it saw and
        the geometric anchor for any class absent from training."""
        clf = getattr(self, "_calibrations", {}).get("tension")
        if clf is not None:
            classes = {int(c) for c in clf.classes_.tolist()}
            pred = int(clf.predict(np.array([[te]]))[0])
            name_of = {0: "agree", 1: "unrelated", 2: "contradict"}
            label = name_of.get(pred, "unrelated")
            # If the labeler found no explicit contradict class, the
            # classifier can only emit agree/unrelated — overlay the
            # geometric anchor so extreme values still flag contradict.
            if 2 not in classes and label == "unrelated" and \
                    te > self._TENSION_CONTRADICT_ANCHOR:
                return "contradict"
            return label
        tau_lo = self._TENSION_AGREE_ANCHOR
        tau_hi = self._TENSION_CONTRADICT_ANCHOR
        if te < tau_lo:
            return "agree"
        if te > tau_hi:
            return "contradict"
        return "unrelated"

    def _holonomy_threshold(self) -> float:
        """Return the fitted holonomy threshold τ or the geometric
        anchor when calibration has not been run."""
        tau = getattr(self, "_thresholds", {}).get("holonomy")
        if isinstance(tau, float):
            return tau
        return float(self._HOLONOMY_ANCHOR)

    # ── Tension: direct geometric verification ───────────────────

    def _classify_tension(
        self,
        verdict: GeometricVerdict,
        low: str,
        meta: dict,
    ) -> GeometricVerdict:
        """Tension classification using ``tension_energy()`` directly.

        Tension energy gives the geometric ground truth:
        E < 0.5 → agree, 0.5 ≤ E ≤ 1.5 → unrelated, E > 1.5 → contradict.
        """
        verdict.calibration_measure = "tension"

        # Type 1: classify (agree / contradict / unrelated)
        entity_a = meta.get("entity_a") or meta.get("head")
        entity_b = meta.get("entity_b") or meta.get("tail")

        if entity_a and entity_b and low in (
                "agree", "contradict", "unrelated"):
            te = self.store.tension_energy(entity_a, entity_b)
            if te is not None:
                geo_label = self._tension_label_from_value(te)
                verdict.calibrated_verdict = (low == geo_label)
                verdict.calibrated_prob = (
                    1.0 if verdict.calibrated_verdict else 0.0)
                verdict.calibration_score = te
                verdict.calibration_detail = (
                    f"E({entity_a}, {entity_b})={te:.4f} "
                    f"→ {geo_label}, LLM says {low}"
                )
                return verdict

        # Type 2: comparative (A/B — which pair is more contradictory)
        pair_a = meta.get("pair_a")
        pair_b = meta.get("pair_b")
        if pair_a and pair_b and low in ("a", "b"):
            te_a = self.store.tension_energy(pair_a[0], pair_a[1])
            te_b = self.store.tension_energy(pair_b[0], pair_b[1])
            if te_a is not None and te_b is not None:
                geo_answer = "b" if te_b > te_a else "a"
                verdict.calibrated_verdict = (low == geo_answer)
                verdict.calibrated_prob = (
                    1.0 if verdict.calibrated_verdict else 0.0)
                verdict.calibration_score = max(te_a, te_b)
                verdict.calibration_detail = (
                    f"E_A={te_a:.4f}, E_B={te_b:.4f} "
                    f"→ {'B' if te_b > te_a else 'A'} more contradictory, "
                    f"LLM says {low.upper()}"
                )
                return verdict

        verdict.calibrated_verdict = verdict.correct
        return verdict

    # ── Holonomy: direct geometric verification ──────────────────

    def _classify_holonomy(
        self,
        verdict: GeometricVerdict,
        low: str,
        meta: dict,
    ) -> GeometricVerdict:
        """Holonomy classification using ``check_holonomy()`` directly.

        The defect ``||R_path R_direct^{-1} - I||_F`` is compared against
        the *calibrated* threshold ``τ_hol`` produced by ``calibrate()``.
        That τ is fit on a structural-truth set built by ``label_holonomy``
        (real triangle vs r2-twisted twin), so the threshold tracks the
        actual consistent/twisted separation on this corpus rather than
        any hard-coded value.  Cap-on cap-trained stores typically place
        τ_hol in the 8--12 range — the absolute number depends on the
        manifold spread the rotors learned, not on a universal constant.

        If calibration has not run, ``_holonomy_threshold`` falls back to
        ``_HOLONOMY_ANCHOR``, which is a coarse safety value that will
        misclassify on cap-on stores.  Always call ``calibrate()`` first
        for audit-grade verdicts.
        """
        verdict.calibration_measure = "holonomy"

        r1 = meta.get("r1")
        r2 = meta.get("r2")
        r_direct = meta.get("r_direct")

        if r1 and r2 and r_direct:
            defect = self.store.check_holonomy([r1, r2], r_direct)
            if defect is not None:
                tau = self._holonomy_threshold()
                geo_consistent = defect <= tau
                llm_says_consistent = low in ("true", "yes", "consistent")
                verdict.calibrated_verdict = (
                    geo_consistent == llm_says_consistent)
                verdict.calibrated_prob = (
                    1.0 if verdict.calibrated_verdict else 0.0)
                verdict.calibration_score = defect
                tau_val = self._holonomy_threshold()
                verdict.calibration_detail = (
                    f"δ({r1},{r2} vs {r_direct})={defect:.4f} "
                    f"({'≤' if geo_consistent else '>'} τ={tau_val:.3f}) "
                    f"→ {'consistent' if geo_consistent else 'inconsistent'}, "
                    f"LLM says {'consistent' if llm_says_consistent else 'inconsistent'}"
                )
                return verdict

        verdict.calibrated_verdict = verdict.correct
        return verdict

    # ── Phase: direct phase-encoder verification ─────────────────

    def _classify_phase(
        self,
        verdict: GeometricVerdict,
        category: str,
        low: str,
        meta: dict,
    ) -> GeometricVerdict:
        """ENM / Phase classification.

        * **boundary**: ENM range check — test whether value falls within
          the phase encoder's ``[v_min, v_max]`` range.  Answer must be
          an explicit boolean; non-boolean answers are rejected.
        * **phase_comparison**: compare two ENM values via phase angle.
        """
        verdict.calibration_measure = "enm" if category == "boundary" else "phase"

        if category == "boundary":
            # Boundary asks "is value X within the valid range?"
            # Answer must be an explicit boolean; anything else is wrong.
            if low not in ("true", "false", "yes", "no"):
                verdict.calibrated_verdict = False
                verdict.calibrated_prob = 0.0
                return verdict

            test_val = meta.get("test_value")
            v_min = meta.get("v_min")
            v_max = meta.get("v_max")

            # Prefer ENM lookup: ground truth from the manifold
            boundary_key = meta.get("boundary_key")
            if boundary_key and self.store.enm:
                pe = getattr(self.store.doc_graph, "phase_encoders", {})
                enc = pe.get(boundary_key)
                if enc is not None:
                    v_min = enc.v_min
                    v_max = enc.v_max

            if test_val is not None and v_min is not None and v_max is not None:
                in_range = (v_min <= test_val <= v_max)
                llm_says_in = low in ("true", "yes")
                verdict.calibrated_verdict = (in_range == llm_says_in)
                verdict.calibrated_prob = (
                    1.0 if verdict.calibrated_verdict else 0.0)
                verdict.calibration_score = test_val
                verdict.calibration_detail = (
                    f"ENM: {test_val:.4f} "
                    f"{'∈' if in_range else '∉'} [{v_min}, {v_max}] "
                    f"→ {'in range' if in_range else 'out of range'}, "
                    f"LLM says {'in range' if llm_says_in else 'out of range'}"
                )
                return verdict

        elif category == "phase_comparison":
            val_a = meta.get("value_a")
            val_b = meta.get("value_b")

            if val_a is not None and val_b is not None:
                geo_answer = "a" if val_a > val_b else "b"
                verdict.calibrated_verdict = (low == geo_answer)
                verdict.calibrated_prob = (
                    1.0 if verdict.calibrated_verdict else 0.0)
                verdict.calibration_score = abs(val_a - val_b)
                verdict.calibration_detail = (
                    f"ENM: A={val_a:.4f}, B={val_b:.4f} "
                    f"→ {'A' if val_a > val_b else 'B'} larger, "
                    f"LLM says {low.upper()}"
                )
                return verdict

            # Look up from ENM if values not in metadata
            enm_type = meta.get("category")
            entity_a = meta.get("entity_a")
            entity_b = meta.get("entity_b")
            if enm_type and entity_a and entity_b:
                va = self.store.lookup_enm(enm_type, entity_a)
                vb = self.store.lookup_enm(enm_type, entity_b)
                if va is not None and vb is not None:
                    geo_answer = "a" if va > vb else "b"
                    verdict.calibrated_verdict = (low == geo_answer)
                    verdict.calibrated_prob = (
                        1.0 if verdict.calibrated_verdict else 0.0)
                    verdict.calibration_score = abs(va - vb)
                    verdict.calibration_detail = (
                        f"ENM({enm_type}): A={va:.4f}, B={vb:.4f} "
                        f"→ {'A' if va > vb else 'B'} larger, "
                        f"LLM says {low.upper()}"
                    )
                    return verdict

        verdict.calibrated_verdict = verdict.correct
        return verdict

    # ── Calibration persistence ──────────────────────────────────

    CALIBRATION_JSON = "calibration.json"
    CALIBRATION_PKL = "calibration.pkl"

    def save(self, path: str | None = None) -> str:
        """Persist calibration state so later runs can reuse it.

        Writes two files under ``path`` (defaults to ``store.store_path``):

        * ``calibration.json`` — flat, human-diffable record of
          ``_thresholds`` plus ``weights``.  Tuples are serialized as
          lists.  This is the only file the oracles and verifiers
          need for threshold-driven pass/fail.
        * ``calibration.pkl`` — pickled ``_calibrations`` (the fitted
          sklearn classifiers).  Required for ``calibrated_classify``
          and ``_tension_label_from_value``; oracles do not need it.

        Missing state is saved as empty containers so a partial
        calibration (e.g. content only, no policy) round-trips
        correctly.

        Returns the directory the files were written to so callers
        can log it.
        """
        import json
        import os
        import pickle

        target = path or self.store.store_path
        os.makedirs(target, exist_ok=True)

        thresholds = getattr(self, "_thresholds", None) or {}
        serializable: dict[str, object] = {}
        for k, v in thresholds.items():
            if isinstance(v, tuple):
                serializable[k] = [float(x) for x in v]
            elif isinstance(v, (int, float)):
                serializable[k] = float(v)
            else:
                serializable[k] = v

        payload = {
            "schema_version": 1,
            "thresholds": serializable,
            "weights": dict(self.weights),
        }
        with open(os.path.join(target, self.CALIBRATION_JSON), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        calibrations = getattr(self, "_calibrations", None) or {}
        with open(os.path.join(target, self.CALIBRATION_PKL), "wb") as f:
            pickle.dump(calibrations, f)

        return target

    def load(self, path: str | None = None) -> bool:
        """Load calibration state written by :meth:`save`.

        Returns ``True`` if thresholds were restored, ``False`` if no
        calibration file exists at ``path``.  A missing pickle is
        tolerated: thresholds alone are enough for oracle pass/fail;
        classifier-dependent code paths (``calibrated_classify``) will
        raise until ``calibrate()`` runs.
        """
        import json
        import os
        import pickle

        target = path or self.store.store_path
        json_path = os.path.join(target, self.CALIBRATION_JSON)
        if not os.path.isfile(json_path):
            return False

        with open(json_path) as f:
            payload = json.load(f)

        raw = payload.get("thresholds", {}) or {}
        restored: dict[str, float | tuple[float, ...]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                restored[k] = tuple(float(x) for x in v)
            elif isinstance(v, (int, float)):
                restored[k] = float(v)
        self._thresholds = restored

        weights = payload.get("weights")
        if isinstance(weights, dict):
            self.weights = {k: float(v) for k, v in weights.items()}

        pkl_path = os.path.join(target, self.CALIBRATION_PKL)
        if os.path.isfile(pkl_path):
            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                self._calibrations = loaded
        else:
            self._calibrations = getattr(self, "_calibrations", {}) or {}

        self.apply_thresholds()
        return True


# =========================================================================
# Convenience
# =========================================================================

def create_judge(store: GMSExpertStore, **kwargs) -> GMSJudge:
    """Create a GMSJudge from a store with default config."""
    return GMSJudge(store, **kwargs)
