# SPDX-License-Identifier: Apache-2.0
"""GMSH Entity Disambiguation Testing — GMS-native ambiguity probes.

Uses ``fuzzy_match_entity`` with decreasing tolerance to find ambiguous
surface forms.  Scores disambiguation quality via ``score_triple`` with
both candidate entities — the correct interpretation has a lower
geodesic distance in the relational context.

Usage::

    gen = DisambiguationGenerator()
    questions = gen.generate(store)

    oracle = DisambiguationOracle(store)
    verdict = oracle.evaluate("entity_surface_form", "rel", expected="A", claimed="B")
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.generators.base import (
    GeneratedQuestion,
    QuestionGenerator,
)

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AmbiguousEntity:
    """An entity surface form that could refer to multiple candidates.

    Attributes
    ----------
    surface_form : str
        The ambiguous name/surface form.
    candidate_a : str
        First candidate entity.
    candidate_b : str
        Second candidate entity.
    geodesic_a : float or None
        Geodesic distance when interpreting as candidate_a in context.
    geodesic_b : float or None
        Geodesic distance when interpreting as candidate_b in context.
    geodesic_separation : float
        |geodesic_a - geodesic_b| — larger means easier to disambiguate.
    correct_candidate : str
        The geometrically correct candidate (lower geodesic).
    """

    surface_form: str = ""
    candidate_a: str = ""
    candidate_b: str = ""
    geodesic_a: float | None = None
    geodesic_b: float | None = None
    geodesic_separation: float = 0.0
    correct_candidate: str = ""


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class DisambiguationOracle:
    """Score disambiguation quality using GMS geodesic distance.

    Scores ``score_triple(head, relation, candidate)`` for both
    candidate entities — the correct interpretation is the one with
    lower geodesic distance on the manifold.

    Parameters
    ----------
    store : GMSExpertStore
        GMS store for geodesic scoring.
    """

    oracle_type = "disambiguation"

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def evaluate(
        self,
        head: str,
        relation: str,
        candidate_a: str,
        candidate_b: str,
        claimed: str,
    ) -> dict:
        """Evaluate a disambiguation choice.

        Parameters
        ----------
        head : str
            Head entity providing context.
        relation : str
            Relation providing context.
        candidate_a : str
            First candidate.
        candidate_b : str
            Second candidate.
        claimed : str
            The candidate the SUT chose.

        Returns
        -------
        dict
            Verdict with geodesics, correct candidate, and pass/fail.
        """
        geo_a = None
        geo_b = None
        try:
            score = self.store.score_triple(head, relation, candidate_a)
            geo_a = float(score) if score is not None else None
        except Exception:
            pass
        try:
            score = self.store.score_triple(head, relation, candidate_b)
            geo_b = float(score) if score is not None else None
        except Exception:
            pass

        # Correct candidate = lower geodesic
        correct = candidate_a
        if geo_a is not None and geo_b is not None:
            correct = candidate_a if geo_a <= geo_b else candidate_b
        elif geo_b is not None:
            correct = candidate_b

        passed = (claimed == correct)

        return {
            "oracle_type": self.oracle_type,
            "passed": passed,
            "geodesic_a": round(geo_a, 4) if geo_a is not None else None,
            "geodesic_b": round(geo_b, 4) if geo_b is not None else None,
            "correct_candidate": correct,
            "claimed": claimed,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DisambiguationGenerator(QuestionGenerator):
    """Generate questions that require entity disambiguation.

    Finds near-match entity pairs using ``fuzzy_match_entity`` and
    crafts context-dependent questions where the SUT must identify
    the correct entity based on relational context.

    Category: ``"disambiguation"``
    """

    category = "disambiguation"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions: list[GeneratedQuestion] = []
        rng = random.Random(42)

        entities = sorted(store.adapter.entity_to_idx.keys())
        triples = list(store.doc_graph.triples)
        if len(entities) < 4 or not triples:
            return []

        # Find entity pairs that are similar (potential ambiguity)
        ambiguous_pairs: list[AmbiguousEntity] = []
        seen_pairs: set[tuple[str, str]] = set()

        for ent in entities:
            # Use fuzzy_match to find similar entities
            match = store.fuzzy_match_entity(ent)
            if match is not None and match != ent:
                pair = tuple(sorted((ent, match)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Find a relational context where they differ
                ent_triple = next(
                    ((h, r, t) for h, r, t in triples if h == ent), None,
                )
                match_triple = next(
                    ((h, r, t) for h, r, t in triples if h == match), None,
                )
                if ent_triple is None or match_triple is None:
                    continue

                # Score both as tails in a shared relational context
                geo_ent = None
                geo_match = None
                try:
                    h, r, t = ent_triple
                    geo_ent = store.score_triple(h, r, ent)
                    geo_match = store.score_triple(h, r, match)
                except Exception:
                    continue

                if geo_ent is None or geo_match is None:
                    continue

                separation = abs(float(geo_ent) - float(geo_match))
                correct = ent if float(geo_ent) <= float(geo_match) else match

                ambiguous_pairs.append(AmbiguousEntity(
                    surface_form=ent[:3] + "...",  # truncated ambiguous form
                    candidate_a=ent,
                    candidate_b=match,
                    geodesic_a=float(geo_ent),
                    geodesic_b=float(geo_match),
                    geodesic_separation=separation,
                    correct_candidate=correct,
                ))

        # Also generate questions from entities with shared substrings
        for i in range(min(len(entities) - 1, 30)):
            ent_a = entities[i]
            for ent_b in entities[i + 1:i + 5]:
                # Check for substring overlap (at least 3 chars)
                overlap = set(ent_a.lower().split()) & set(ent_b.lower().split())
                if not overlap:
                    continue
                pair = tuple(sorted((ent_a, ent_b)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Find a distinguishing relational context
                a_triples = [
                    (h, r, t) for h, r, t in triples if h == ent_a
                ]
                if not a_triples:
                    continue
                h, r, t = a_triples[0]

                geo_a = None
                geo_b = None
                try:
                    geo_a = store.score_triple(h, r, ent_a)
                    geo_b = store.score_triple(h, r, ent_b)
                except Exception:
                    continue

                if geo_a is None or geo_b is None:
                    continue

                separation = abs(float(geo_a) - float(geo_b))
                if separation < 0.1:
                    continue  # Too similar, not disambiguable

                correct = ent_a if float(geo_a) <= float(geo_b) else ent_b
                ambiguous_pairs.append(AmbiguousEntity(
                    surface_form=" ".join(overlap),
                    candidate_a=ent_a,
                    candidate_b=ent_b,
                    geodesic_a=float(geo_a),
                    geodesic_b=float(geo_b),
                    geodesic_separation=separation,
                    correct_candidate=correct,
                ))

        # Generate questions from ambiguous pairs
        rng.shuffle(ambiguous_pairs)
        for i, amb in enumerate(ambiguous_pairs[:15]):
            ground_truth = amb.correct_candidate

            # Randomize option order
            if rng.random() > 0.5:
                opt_a, opt_b = amb.candidate_a, amb.candidate_b
                label_map = {"A": amb.candidate_a, "B": amb.candidate_b}
            else:
                opt_a, opt_b = amb.candidate_b, amb.candidate_a
                label_map = {"A": amb.candidate_b, "B": amb.candidate_a}

            correct_label = "A" if label_map["A"] == ground_truth else "B"

            def _answer_fn(s, _gt=correct_label):
                return _gt

            questions.append(GeneratedQuestion(
                qid=f"disambig_{i:03d}",
                category=self.category,
                natural_language=(
                    f"Which entity is the correct '{amb.surface_form}' "
                    f"in the context of the document: "
                    f"(A) {opt_a} or (B) {opt_b}?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=correct_label,
                llm_prompt=(
                    f"These two entities might be confused:\n"
                    f"  (A) {opt_a}\n"
                    f"  (B) {opt_b}\n\n"
                    f"Based on the document context, which is the more "
                    f"relevant entity?\n\n"
                    f"ANSWER: A or B"
                ),
                answer_type="str",
                metadata={
                    "candidate_a": amb.candidate_a,
                    "candidate_b": amb.candidate_b,
                    "geodesic_a": round(amb.geodesic_a, 4) if amb.geodesic_a else None,
                    "geodesic_b": round(amb.geodesic_b, 4) if amb.geodesic_b else None,
                    "geodesic_separation": round(amb.geodesic_separation, 4),
                    "test_type": "disambiguation",
                },
            ))

        return questions
