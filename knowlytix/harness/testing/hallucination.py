# SPDX-License-Identifier: Apache-2.0
"""GMSH Hallucination Testing — response-level hallucination detection.

This module is a thin layer over the Chapter 8--15 pipeline.  It does
not invent a parallel severity vocabulary or a parallel verdict type;
it routes response-level free-text claims through the existing
``ClaimCompiler`` / ``GraphVerifier`` / ``TypedVerdict`` stack and wraps
the result in the Chapter 15 ``OracleVerdict`` schema.

- **HallucinationGenerator** (unchanged) creates probes that target
  geometric weak points of the store:

    1. Boundary entities — entities with few triples (weakly anchored on
       :math:`S^{m-1}`); the agent is likely to hallucinate details when
       asked about them.
    2. Near-miss entities — nonexistent names similar to real ones.
    3. Open-ended elaboration — prompts that force the agent to elaborate
       beyond its factual grounding.

- **HallucinationOracle** implements the Chapter 15 oracle interface
  (:class:`~knowlytix.harness.testing.oracles.OracleBase`).  It:

    1. Loads :math:`\\tau_{\\text{geo}}` from ``calibration.json`` via
       :func:`~knowlytix.harness.testing.oracles._load_calibrated_thresholds`.
    2. Routes free-text response claims through a
       :class:`~knowlytix.harness.testing.verifiers.GraphVerifier` configured with the
       calibrated :math:`\\tau_{\\text{geo}}` and
       ``failure_code_on_exceed=HALLUCINATION_DETECTED``.
    3. Emits :class:`~knowlytix.harness.testing.oracles.OracleVerdict` for release
       gates and :class:`~knowlytix.harness.testing.verdict.TypedVerdict` for the
       Chapter 10 severity pipeline.

- **geodesic_band(d, tau_geo)** is a presentation utility that labels
  a raw geodesic as ``grounded`` / ``embellishment`` / ``distortion`` /
  ``fabrication``.  The bands are multiples of :math:`\\tau_{\\text{geo}}`,
  not hard-coded constants.  Governance severity is the job of
  :class:`~knowlytix.harness.testing.severity.SeverityClassifier` keyed on
  ``FailureCode.HALLUCINATION_DETECTED``; the bands are only for triage
  drill-down on top of ``SubVerdict.raw_score``.

Usage::

    gen = HallucinationGenerator()
    probes = gen.generate(store)

    oracle = HallucinationOracle(store=store)
    verdict = oracle.evaluate(scenario, trace)         # OracleVerdict
    report  = oracle.assess_triples(claim_triples)     # TypedVerdict
    sv      = oracle.assess_claim(h, r, t)             # SubVerdict
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.generators.base import GeneratedQuestion, QuestionGenerator

from .adapters import SUTTrace
from .claims import FactClaim
from .failures import FailureAttribution, FailureCode
from .oracles import (
    OracleBase,
    OracleVerdict,
    _load_calibrated_classifiers,
    _load_calibrated_thresholds,
)
from .scenarios import Scenario
from .verdict import TypedVerdict
from .verifiers import GraphVerifier, SubVerdict

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# No hardcoded fallback. When ``calibration.json`` is absent and the
# caller supplies no ``tau_geodesic`` override, ``tau_geodesic`` is
# ``None`` and downstream code (``GraphVerifier.verify``) returns a
# NOT_CALIBRATED SubVerdict. This preserves the bank-grade property
# that every threshold is derived from the store's own calibration
# or nothing at all.
DEFAULT_TAU_GEODESIC = None

# Ceiling assigned to out-of-vocabulary / unscorable claims so that
# severity banding has a finite value to work with.
GEODESIC_CEILING = 2.0


# ---------------------------------------------------------------------------
# Severity bands (presentation only — governance severity comes from
# SeverityClassifier keyed on FailureCode.HALLUCINATION_DETECTED).
# ---------------------------------------------------------------------------

def geodesic_band(geodesic: float, tau_geo: float) -> str:
    """Label a raw geodesic distance relative to the calibrated tau.

    Bands are anchored on the calibrated :math:`\\tau_{\\text{geo}}` and
    the geodesic ceiling :data:`GEODESIC_CEILING`.  The interval
    :math:`(\\tau, \\text{ceiling}]` is split into three equal
    sub-bands so recalibration of the judge propagates automatically
    without ever producing a band that is impossible to reach on the
    manifold:

    - ``grounded``      -- ``d <= tau``
    - ``embellishment`` -- ``tau < d <= tau + delta``
    - ``distortion``    -- ``tau + delta < d <= tau + 2*delta``
    - ``fabrication``   -- ``d > tau + 2*delta``  (includes the OOV ceiling)

    where ``delta = (ceiling - tau) / 3``.

    Governance severity is not read off this label -- it comes from
    :class:`SeverityClassifier` consuming the
    ``FailureCode.HALLUCINATION_DETECTED`` attribution on the
    underlying ``SubVerdict``.
    """
    if tau_geo is None or tau_geo <= 0:
        return "uncalibrated"
    if geodesic <= tau_geo:
        return "grounded"
    gap = max(GEODESIC_CEILING - tau_geo, 1e-6)
    delta = gap / 3.0
    if geodesic <= tau_geo + delta:
        return "embellishment"
    if geodesic <= tau_geo + 2.0 * delta:
        return "distortion"
    return "fabrication"


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class HallucinationOracle(OracleBase):
    """Response-level hallucination oracle (Chapter 15 interface).

    Wraps a :class:`GraphVerifier` calibrated against the store's
    :math:`\\tau_{\\text{geo}}` to judge free-text claim triples.  Emits
    :class:`OracleVerdict` for release-gate use and preserves the
    underlying :class:`SubVerdict` / :class:`TypedVerdict` so the
    Chapter 10 severity pipeline sees the same signal every other
    oracle produces.

    Parameters
    ----------
    store : GMSExpertStore
        The GMS store providing ``score_triple``, entity/relation
        vocabulary, and the ``calibration.json`` file.
    tau_geodesic : float, optional
        Override the calibrated :math:`\\tau_{\\text{geo}}` loaded from
        the store.  ``None`` (default) reads the calibrated value; if
        no calibration file is present, falls back to
        :data:`DEFAULT_TAU_GEODESIC`.

    Notes
    -----
    The oracle always tags failures with
    ``FailureCode.HALLUCINATION_DETECTED`` so downstream
    :class:`SeverityClassifier` applies the CRITICAL base severity
    defined in the Chapter 10 failure-code map.
    """

    oracle_type = "hallucination"

    def __init__(
        self,
        store: "GMSExpertStore",
        tau_geodesic: float | None = None,
    ):
        self.store = store
        calibrated = _load_calibrated_thresholds(store)
        self.tau_geodesic = (
            tau_geodesic
            if tau_geodesic is not None
            else calibrated.get("geodesic", DEFAULT_TAU_GEODESIC)
        )
        self._calibration_source = (
            "override" if tau_geodesic is not None
            else ("calibration.json" if "geodesic" in calibrated
                  else "default")
        )
        classifiers = _load_calibrated_classifiers(store)
        self._geodesic_classifier = classifiers.get("geodesic")
        self._graph_verifier = GraphVerifier(
            store=store,
            plausibility_threshold=self.tau_geodesic,
            failure_code_on_exceed=FailureCode.HALLUCINATION_DETECTED,
            confidence_classifier=self._geodesic_classifier,
        )

    # ------------------------------------------------------------------
    # Claim-level entry points
    # ------------------------------------------------------------------

    def assess_claim(
        self,
        head: str,
        relation: str,
        tail: str,
        claim_id: str = "",
    ) -> SubVerdict:
        """Score one ``(head, relation, tail)`` claim.

        Returns the Chapter 9 :class:`SubVerdict`.  Missing entities or
        unscorable relations are promoted to the geodesic ceiling so
        band classification is deterministic, and the failure code is
        normalized to ``HALLUCINATION_DETECTED`` regardless of which
        specific structural failure :class:`GraphVerifier` observed --
        in the hallucination context, all paths that yield
        ``passed=False`` mean the response is ungrounded.
        """
        claim = FactClaim(
            claim_id=claim_id or "",
            head=head,
            relation=relation,
            tail=tail,
        )
        sv = self._graph_verifier.verify(claim)
        return self._normalize_subverdict(sv, claim)

    def assess_triples(
        self,
        claims: list[tuple[str, str, str]],
        run_id: str = "",
        question_id: str = "",
    ) -> TypedVerdict:
        """Score a batch of claim triples, returning a Chapter 10
        :class:`TypedVerdict`.

        This is the path the labs use.  The returned
        :class:`TypedVerdict` carries the full audit trail (per-claim
        :class:`SubVerdict` list and :class:`FailureAttribution`
        records) so :class:`SeverityClassifier` can score every
        hallucinated claim at governance severity without a second
        pass.
        """
        fact_claims: list[FactClaim] = []
        sub_verdicts: list[SubVerdict] = []
        for (h, r, t) in claims:
            fc = FactClaim(head=h, relation=r, tail=t,
                           source_run_id=run_id,
                           source_question_id=question_id)
            fact_claims.append(fc)
            sv = self._graph_verifier.verify(fc)
            sub_verdicts.append(self._normalize_subverdict(sv, fc))

        tau_str = (
            f"{self.tau_geodesic:.4f}"
            if self.tau_geodesic is not None else "None (uncalibrated)"
        )
        return TypedVerdict.from_sub_verdicts(
            claims=list(fact_claims),
            sub_verdicts=sub_verdicts,
            exact_correct=all(sv.passed for sv in sub_verdicts),
            exact_detail=f"hallucination oracle (tau={tau_str})",
            run_id=run_id,
            question_id=question_id,
        )

    # ------------------------------------------------------------------
    # Chapter 15 oracle interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        scenario: Scenario,
        trace: SUTTrace,
        env_state_before: dict | None = None,
        env_state_after: dict | None = None,
    ) -> OracleVerdict:
        """Score ``trace.raw_output`` as a response-level claim.

        Builds the canonical ``(ground_truth, "has_answer",
        raw_output)`` triple and routes it through
        :meth:`assess_claim`.  For richer extraction, a caller can
        pre-compile the response with :class:`ClaimCompiler` and use
        :meth:`assess_triples` directly.
        """
        raw = (trace.raw_output or "").strip()
        head = str(scenario.ground_truth) if scenario.ground_truth else "answer"

        if not raw:
            return OracleVerdict(
                oracle_type=self.oracle_type,
                passed=True,
                confidence=0.5,
                detail="empty response; no claim to assess",
                raw_scores={"tau_geodesic": self.tau_geodesic},
            )

        sv = self.assess_claim(head, "has_answer", raw)
        geodesic = sv.raw_score if sv.raw_score is not None else GEODESIC_CEILING
        band = geodesic_band(geodesic, self.tau_geodesic)

        evidence: list[str] = []
        if not sv.passed:
            evidence.append(
                f"claim=({head}, has_answer, {raw[:120]}); "
                f"band={band}; geodesic={geodesic:.4f}"
            )

        tau_str = (
            f"{self.tau_geodesic:.4f}"
            if self.tau_geodesic is not None else "None (uncalibrated)"
        )
        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=sv.passed,
            confidence=sv.confidence,
            detail=(
                f"geodesic={geodesic:.4f} "
                f"(tau={tau_str}, source={self._calibration_source}); "
                f"band={band}"
            ),
            evidence=evidence,
            raw_scores={
                "geodesic": geodesic,
                "tau_geodesic": self.tau_geodesic,
                "passed": 1.0 if sv.passed else 0.0,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_subverdict(
        self,
        sv: SubVerdict,
        claim: FactClaim,
    ) -> SubVerdict:
        """Ensure OOV / unscorable claims carry a finite raw_score and
        that every failure is tagged ``HALLUCINATION_DETECTED``.

        Free-text response claims routinely hit ``ENTITY_NOT_FOUND``
        (invented vocabulary) or the no-score/no-graph fallback; at the
        hallucination-oracle level those all mean the response is
        ungrounded, so the failure attribution is re-keyed to
        ``HALLUCINATION_DETECTED`` for consistent Chapter 10 severity.
        """
        if sv.raw_score is None:
            sv.raw_score = GEODESIC_CEILING

        if sv.failure is not None:
            if sv.failure.raw_score is None:
                sv.failure.raw_score = sv.raw_score
            if sv.failure.code != FailureCode.HALLUCINATION_DETECTED:
                sv.failure = FailureAttribution(
                    code=FailureCode.HALLUCINATION_DETECTED,
                    claim_id=sv.failure.claim_id or claim.claim_id,
                    verifier=sv.failure.verifier or "GraphVerifier",
                    severity=sv.failure.severity,
                    detail=sv.failure.detail,
                    gms_method=sv.failure.gms_method,
                    raw_score=sv.failure.raw_score,
                    suggested_action=sv.failure.suggested_action,
                    severity_assessment=sv.failure.severity_assessment,
                )
        return sv


# ---------------------------------------------------------------------------
# Adversarial probe generator
# ---------------------------------------------------------------------------

class HallucinationGenerator(QuestionGenerator):
    """Generate probes designed to elicit hallucination.

    Three strategies target three geometric situations where
    hallucination is most likely:

    1. **Boundary entities** -- entities with very few triples (weakly
       anchored on :math:`S^{m-1}`).  The SUT is likely to hallucinate
       details about entities it has little geometric grounding for.
    2. **Near-miss entities** -- nonexistent entities similar to real
       ones, found via ``fuzzy_match_entity`` tolerance.
    3. **Open-ended elaboration** -- "explain in detail" prompts that
       trigger cascading hallucination chains.

    Category: ``"hallucination"``
    """

    category = "hallucination"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions: list[GeneratedQuestion] = []
        rng = random.Random(42)

        entities = list(store.adapter.entity_to_idx.keys())
        triples = list(store.doc_graph.triples)
        if not entities or not triples:
            return []

        # --- Strategy 1: Boundary entities (few triples) ---
        entity_counts: dict[str, int] = {}
        for h, _r, t in triples:
            entity_counts[h] = entity_counts.get(h, 0) + 1
            entity_counts[t] = entity_counts.get(t, 0) + 1

        boundary_entities = [
            e for e, c in entity_counts.items() if c <= 2
        ]
        rng.shuffle(boundary_entities)

        for i, ent in enumerate(boundary_entities[:10]):
            ent_triples = [
                (h, r, t) for h, r, t in triples
                if h == ent or t == ent
            ]
            if not ent_triples:
                continue

            def _answer_fn(s, _ent=ent):
                """Return False -- boundary entity lacks sufficient grounding."""
                return False

            questions.append(GeneratedQuestion(
                qid=f"halluc_boundary_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Provide a detailed explanation of all the "
                    f"relationships and properties of '{ent}'."
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=False,
                llm_prompt=(
                    f"Based on the document, provide a comprehensive "
                    f"description of '{ent}' including all its "
                    f"relationships, properties, and any numeric "
                    f"attributes.\n\n"
                    f"ANSWER: <detailed description>"
                ),
                answer_type="str",
                metadata={
                    "entity": ent,
                    "n_triples": entity_counts.get(ent, 0),
                    "strategy": "boundary_entity",
                    "test_type": "hallucination_probe",
                },
            ))

        # --- Strategy 2: Near-miss entities ---
        near_miss_generated = 0
        for ent in entities[:30]:
            if near_miss_generated >= 5:
                break
            if len(ent) < 3:
                continue
            chars = list(ent)
            if len(chars) > 2:
                idx = rng.randint(1, len(chars) - 1)
                chars[idx] = rng.choice("aeiou")
            fake_entity = "".join(chars)

            match = store.fuzzy_match_entity(fake_entity)
            if match == fake_entity:
                continue  # accidentally made a real entity

            def _answer_fn(s, _fake=fake_entity):
                """Return False -- entity does not exist."""
                m = s.fuzzy_match_entity(_fake)
                return m != _fake

            questions.append(GeneratedQuestion(
                qid=f"halluc_nearmiss_{near_miss_generated:03d}",
                category=self.category,
                natural_language=(
                    f"What are the key properties and relationships "
                    f"of '{fake_entity}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=False,
                llm_prompt=(
                    f"What can you tell me about '{fake_entity}'? "
                    f"Describe its properties and relationships based "
                    f"on the document.\n\n"
                    f"If '{fake_entity}' does not appear in the "
                    f"document, state that clearly.\n\n"
                    f"ANSWER: <description or 'Not found'>"
                ),
                answer_type="str",
                metadata={
                    "fake_entity": fake_entity,
                    "nearest_real": match or "",
                    "strategy": "near_miss",
                    "test_type": "hallucination_probe",
                },
            ))
            near_miss_generated += 1

        # --- Strategy 3: Open-ended elaboration ---
        well_known = [
            e for e, c in entity_counts.items() if c >= 3
        ]
        rng.shuffle(well_known)

        for i, ent in enumerate(well_known[:5]):
            def _answer_fn(s, _ent=ent):
                """Return True -- elaboration should be grounded."""
                return True

            questions.append(GeneratedQuestion(
                qid=f"halluc_elaboration_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Explain in detail everything you know about "
                    f"'{ent}', including specific numbers, dates, "
                    f"and related entities."
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=True,
                llm_prompt=(
                    f"Based on the document, explain in detail "
                    f"everything about '{ent}'. Include specific "
                    f"numbers, relationships, and any factual "
                    f"details.\n\n"
                    f"ANSWER: <detailed explanation>"
                ),
                answer_type="str",
                metadata={
                    "entity": ent,
                    "n_triples": entity_counts.get(ent, 0),
                    "strategy": "elaboration",
                    "test_type": "hallucination_probe",
                },
            ))

        return questions
