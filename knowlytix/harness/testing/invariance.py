# SPDX-License-Identifier: Apache-2.0
"""GMSH Paraphrase Invariance Testing — GMS-native consistency analysis.

Tests whether an agent gives consistent answers to paraphrased versions
of the same question.  Uses ``tension_energy`` between answer entity
embeddings across paraphrases:

- Zero tension → perfectly consistent answers
- Rising tension → inconsistent (order-sensitive or phrasing-sensitive)

The ``score_triple`` geodesic further validates that each individual
answer is plausible on the manifold.

Usage::

    tester = InvarianceTester(store)
    tester.add_paraphrase_result("q1_v1", answer="entity_A")
    tester.add_paraphrase_result("q1_v2", answer="entity_A")
    tester.add_paraphrase_result("q1_v3", answer="entity_B")
    result = tester.evaluate("q1")
    print(result.mean_tension, result.invariant)

    gen = InvarianceGenerator()
    questions = gen.generate(store)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InvarianceResult:
    """Result of paraphrase invariance analysis for one question group.

    Attributes
    ----------
    question_group : str
        Identifier for the question group (base question ID).
    n_paraphrases : int
        Number of paraphrase variants tested.
    answers : list[str]
        Collected answers across paraphrases.
    pairwise_tensions : dict[str, float]
        Tension energy between each pair of answer entities.
        Keys are ``"v1:v2"`` strings.
    mean_tension : float
        Mean pairwise tension.
    max_tension : float
        Worst-case pairwise tension.
    invariant : bool
        Whether mean tension is below threshold (consistent answers).
    """

    question_group: str = ""
    n_paraphrases: int = 0
    answers: list[str] = field(default_factory=list)
    pairwise_tensions: dict = field(default_factory=dict)
    mean_tension: float = 0.0
    max_tension: float = 0.0
    invariant: bool = True

    def to_dict(self) -> dict:
        return {
            "question_group": self.question_group,
            "n_paraphrases": self.n_paraphrases,
            "n_unique_answers": len(set(self.answers)),
            "mean_tension": round(self.mean_tension, 4),
            "max_tension": round(self.max_tension, 4),
            "invariant": self.invariant,
        }


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

class InvarianceTester:
    """Test paraphrase invariance using GMS tension_energy.

    Collects answers from paraphrased question variants and computes
    pairwise ``tension_energy`` between answer entities.

    Parameters
    ----------
    store : GMSExpertStore
        GMS store for tension energy computation.
    tension_threshold : float
        Maximum mean tension for invariance (default 0.3).
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        tension_threshold: float = 0.3,
    ):
        self.store = store
        self.tension_threshold = tension_threshold
        self._groups: dict[str, list[tuple[str, str]]] = {}

    def add_paraphrase_result(
        self,
        variant_id: str,
        answer: str,
        group: str = "",
    ) -> None:
        """Record an answer for a paraphrase variant.

        Parameters
        ----------
        variant_id : str
            Identifier for this paraphrase variant.
        answer : str
            The SUT's answer.
        group : str
            Question group key.  If empty, derived from variant_id
            by stripping the last ``_vN`` suffix.
        """
        if not group:
            parts = variant_id.rsplit("_", 1)
            group = parts[0] if len(parts) > 1 else variant_id
        self._groups.setdefault(group, []).append((variant_id, answer))

    def evaluate(self, group: str) -> InvarianceResult:
        """Evaluate invariance for a question group.

        Parameters
        ----------
        group : str
            Question group to evaluate.

        Returns
        -------
        InvarianceResult
            Pairwise tensions and invariance flag.
        """
        entries = self._groups.get(group, [])
        if len(entries) < 2:
            return InvarianceResult(
                question_group=group,
                n_paraphrases=len(entries),
                answers=[a for _, a in entries],
                invariant=True,
            )

        answers = [a for _, a in entries]
        variant_ids = [v for v, _ in entries]
        tensions: dict[str, float] = {}

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                pair_key = f"{variant_ids[i]}:{variant_ids[j]}"
                tension = self._compute_tension(answers[i], answers[j])
                if tension is not None:
                    tensions[pair_key] = tension

        tension_values = list(tensions.values())
        mean_t = sum(tension_values) / len(tension_values) if tension_values else 0.0
        max_t = max(tension_values) if tension_values else 0.0

        return InvarianceResult(
            question_group=group,
            n_paraphrases=len(entries),
            answers=answers,
            pairwise_tensions=tensions,
            mean_tension=mean_t,
            max_tension=max_t,
            invariant=mean_t <= self.tension_threshold,
        )

    def evaluate_all(self) -> list[InvarianceResult]:
        """Evaluate invariance for all recorded groups.

        Returns
        -------
        list[InvarianceResult]
            Results for each question group.
        """
        return [self.evaluate(g) for g in self._groups]

    def _compute_tension(self, answer_a: str, answer_b: str) -> float | None:
        """Compute tension energy between two answer entities."""
        if answer_a == answer_b:
            return 0.0  # identical answers = zero tension
        try:
            tension = self.store.tension_energy(answer_a, answer_b)
            if tension is not None:
                return float(tension)
        except Exception:
            pass
        # Fallback: score_triple difference
        try:
            sa = self.store.score_triple(answer_a, "is_type", answer_a)
            sb = self.store.score_triple(answer_b, "is_type", answer_b)
            if sa is not None and sb is not None:
                return abs(float(sa) - float(sb))
        except Exception:
            pass
        return None

    def reset(self) -> None:
        """Clear all recorded groups."""
        self._groups.clear()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class InvarianceGenerator(QuestionGenerator):
    """Generate base questions suitable for paraphrase invariance testing.

    Produces questions about well-grounded facts (low geodesic) that
    should yield identical answers regardless of phrasing.  The DOE
    rephraser creates the paraphrase variants.

    Category: ``"paraphrase_invariance"``
    """

    category = "paraphrase_invariance"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions: list[GeneratedQuestion] = []
        rng = random.Random(42)

        triples = list(store.doc_graph.triples)
        if not triples:
            return []

        # Select well-grounded triples (low geodesic)
        scored_triples = []
        for h, r, t in triples:
            score = store.score_triple(h, r, t)
            if score is not None and score < 0.5:
                scored_triples.append((h, r, t, float(score)))

        scored_triples.sort(key=lambda x: x[3])
        selected = scored_triples[:20]
        rng.shuffle(selected)

        for i, (h, r, t, geo) in enumerate(selected[:10]):
            ground_truth = t

            def _answer_fn(s, _h=h, _r=r, _t=t):
                return _t

            questions.append(GeneratedQuestion(
                qid=f"invariance_{i:03d}",
                category=self.category,
                natural_language=(
                    f"What is the '{r}' of '{h}'?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the document, what is the '{r}' of "
                    f"'{h}'?\n\n"
                    f"{ANSWER_FORMATS['str']}"
                ),
                answer_type="str",
                metadata={
                    "head": h,
                    "relation": r,
                    "tail": t,
                    "geodesic": round(geo, 4),
                    "test_type": "paraphrase_invariance",
                },
            ))

        return questions
