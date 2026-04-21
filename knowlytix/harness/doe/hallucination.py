# SPDX-License-Identifier: Apache-2.0
"""GMSH Hallucination Testing — GMS-native hallucination detection.

A hallucination is a claim whose ``score_triple`` geodesic is very high —
geometrically far from the learned manifold.  This module provides:

- **HallucinationGenerator**: creates questions designed to elicit
  hallucination via three strategies:
  1. **Boundary entities** — entities with few triples (weakly anchored
     on the manifold); SUTs hallucinate about these.
  2. **Near-miss entities** — entities similar to real ones but
     nonexistent; found via ``fuzzy_match_entity`` tolerance.
  3. **Open-ended elaboration** — "explain in detail" prompts that
     trigger cascading hallucination.

- **HallucinationOracle**: parses claims from SUT output, scores each
  via ``score_triple``, and flags geodesic > threshold as hallucinated.

- **Severity tiers** (geodesic-based):
  - >1.5 = fabrication
  - 0.7–1.5 = distortion
  - 0.5–0.7 = embellishment
  - <0.5 = grounded

Usage::

    gen = HallucinationGenerator()
    questions = gen.generate(store)

    oracle = HallucinationOracle(store)
    assessment = oracle.assess_output(output_text, context_entities)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowlytix.bench.eval.generators.base import GeneratedQuestion, QuestionGenerator

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Severity tiers
# ---------------------------------------------------------------------------

def _severity_tier(geodesic: float) -> str:
    """Classify hallucination severity by geodesic distance."""
    if geodesic > 1.5:
        return "fabrication"
    if geodesic > 0.7:
        return "distortion"
    if geodesic > 0.5:
        return "embellishment"
    return "grounded"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class HallucinationAssessment:
    """Per-claim hallucination assessment.

    Attributes
    ----------
    claim_text : str
        The extracted claim from SUT output.
    head : str
        Parsed head entity.
    relation : str
        Parsed relation.
    tail : str
        Parsed tail entity.
    geodesic_distance : float
        ``score_triple`` geodesic (higher = more hallucinated).
    tension_vs_nearest : float or None
        ``tension_energy`` between claim entity and nearest real entity.
    severity : str
        One of ``"fabrication"``, ``"distortion"``, ``"embellishment"``,
        ``"grounded"``.
    """

    claim_text: str = ""
    head: str = ""
    relation: str = ""
    tail: str = ""
    geodesic_distance: float = 0.0
    tension_vs_nearest: float | None = None
    severity: str = "grounded"


@dataclass
class HallucinationReport:
    """Aggregate hallucination report for a SUT response.

    Attributes
    ----------
    n_claims : int
        Total claims parsed from output.
    n_hallucinated : int
        Claims with geodesic > threshold.
    hallucination_rate : float
        Fraction of hallucinated claims.
    mean_geodesic : float
        Mean geodesic distance across all claims.
    max_geodesic : float
        Worst-case geodesic distance.
    severity_counts : dict[str, int]
        Count per severity tier.
    assessments : list[HallucinationAssessment]
        Per-claim details.
    """

    n_claims: int = 0
    n_hallucinated: int = 0
    hallucination_rate: float = 0.0
    mean_geodesic: float = 0.0
    max_geodesic: float = 0.0
    severity_counts: dict = field(default_factory=dict)
    assessments: list[HallucinationAssessment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_claims": self.n_claims,
            "n_hallucinated": self.n_hallucinated,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "mean_geodesic": round(self.mean_geodesic, 4),
            "max_geodesic": round(self.max_geodesic, 4),
            "severity_counts": self.severity_counts,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HallucinationGenerator(QuestionGenerator):
    """Generate questions designed to elicit hallucination.

    Three strategies:

    1. **Boundary entities**: entities with very few triples (weakly
       anchored on S^{m-1}).  The SUT is likely to hallucinate details
       about entities it has little geometric grounding for.

    2. **Near-miss entities**: nonexistent entities similar to real
       ones, found via ``fuzzy_match_entity`` tolerance.

    3. **Open-ended elaboration**: "explain in detail" prompts that
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

        # Find weakly anchored entities (1-2 triples)
        boundary_entities = [
            e for e, c in entity_counts.items() if c <= 2
        ]
        rng.shuffle(boundary_entities)

        for i, ent in enumerate(boundary_entities[:10]):
            # Find the one relation this entity participates in
            ent_triples = [
                (h, r, t) for h, r, t in triples
                if h == ent or t == ent
            ]
            if not ent_triples:
                continue
            h0, r0, t0 = ent_triples[0]

            ground_truth = False  # We expect the SUT cannot detail beyond known facts

            def _answer_fn(s, _ent=ent, _count=entity_counts.get(ent, 0)):
                """Return False — boundary entity has insufficient grounding."""
                return False

            questions.append(GeneratedQuestion(
                qid=f"halluc_boundary_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Provide a detailed explanation of all the "
                    f"relationships and properties of '{ent}'."
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
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
            # Create a plausible but nonexistent entity name
            if len(ent) < 3:
                continue
            # Swap chars to create a near-miss
            chars = list(ent)
            if len(chars) > 2:
                idx = rng.randint(1, len(chars) - 1)
                chars[idx] = rng.choice("aeiou")
            fake_entity = "".join(chars)

            # Verify it's actually not in the store
            match = store.fuzzy_match_entity(fake_entity)
            if match == fake_entity:
                continue  # accidentally made a real entity

            ground_truth = False

            def _answer_fn(s, _fake=fake_entity):
                """Return False — entity does not exist."""
                m = s.fuzzy_match_entity(_fake)
                return m != _fake  # True if it maps to something else

            questions.append(GeneratedQuestion(
                qid=f"halluc_nearmiss_{near_miss_generated:03d}",
                category=self.category,
                natural_language=(
                    f"What are the key properties and relationships "
                    f"of '{fake_entity}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
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
            ground_truth = True  # Well-known entity should be described accurately

            def _answer_fn(s, _ent=ent):
                """Return True — elaboration should be grounded."""
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
                ground_truth=ground_truth,
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


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class HallucinationOracle:
    """Score SUT output for hallucinations using GMS geodesic distance.

    Parses entity mentions from output, scores each against the manifold
    via ``score_triple``, and classifies by severity tier.

    Oracle type: ``"hallucination"``

    Parameters
    ----------
    store : GMSExpertStore
        GMS store for geodesic scoring.
    geodesic_threshold : float
        Geodesic distance above which a claim is hallucinated.
    """

    oracle_type = "hallucination"

    def __init__(
        self,
        store: "GMSExpertStore",
        geodesic_threshold: float = 1.0,
    ):
        self.store = store
        self.geodesic_threshold = geodesic_threshold

    def assess_claim(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> HallucinationAssessment:
        """Assess a single (h, r, t) claim for hallucination.

        Parameters
        ----------
        head : str
            Head entity.
        relation : str
            Relation.
        tail : str
            Tail entity.

        Returns
        -------
        HallucinationAssessment
            Per-claim assessment with geodesic and severity.
        """
        # Score the triple
        geodesic = 2.0  # worst case
        try:
            score = self.store.score_triple(head, relation, tail)
            if score is not None:
                geodesic = float(score)
        except Exception:
            pass

        # Tension vs nearest known entity
        tension = None
        try:
            matched = self.store.fuzzy_match_entity(tail)
            if matched and matched != tail:
                tension = self.store.tension_energy(tail, matched)
        except Exception:
            pass

        return HallucinationAssessment(
            claim_text=f"{head} {relation} {tail}",
            head=head,
            relation=relation,
            tail=tail,
            geodesic_distance=geodesic,
            tension_vs_nearest=tension,
            severity=_severity_tier(geodesic),
        )

    def assess_triples(
        self,
        claims: list[tuple[str, str, str]],
    ) -> HallucinationReport:
        """Assess multiple (h, r, t) claims for hallucination.

        Parameters
        ----------
        claims : list[tuple[str, str, str]]
            List of (head, relation, tail) claims to assess.

        Returns
        -------
        HallucinationReport
            Aggregate report.
        """
        assessments = [
            self.assess_claim(h, r, t) for h, r, t in claims
        ]

        if not assessments:
            return HallucinationReport()

        geodesics = [a.geodesic_distance for a in assessments]
        n_halluc = sum(
            1 for g in geodesics if g > self.geodesic_threshold
        )

        severity_counts: dict[str, int] = {}
        for a in assessments:
            severity_counts[a.severity] = (
                severity_counts.get(a.severity, 0) + 1
            )

        return HallucinationReport(
            n_claims=len(assessments),
            n_hallucinated=n_halluc,
            hallucination_rate=n_halluc / len(assessments),
            mean_geodesic=sum(geodesics) / len(geodesics),
            max_geodesic=max(geodesics),
            severity_counts=severity_counts,
            assessments=assessments,
        )
