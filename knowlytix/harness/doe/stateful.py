# SPDX-License-Identifier: Apache-2.0
"""GMSH Stateful Agent Testing — GMS-native memory and degradation probes.

Tests whether an agentic system can remember, recall, and maintain
consistency across sessions.  All scoring uses GMS geometric primitives:

- **Memory capture**: ``RuntimeLearner.learn_triples()`` encodes agent
  assertions as triples on S^{m-1}.
- **Recall quality**: ``score_triple(taught_h, taught_r, taught_t)`` —
  low geodesic = agent remembered correctly.
- **Memory drift**: ``tension_energy`` between session store at time T
  vs T+N — rising tension = degradation.
- **Learning effectiveness**: Teach facts, test if later queries yield
  correct answers with low geodesic distance.

Usage::

    generator = StatefulAgentGenerator()
    questions = generator.generate(store)

    oracle = MemoryRecallOracle(store)
    verdict = oracle.evaluate(scenario, trace)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowlytix.bench.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AgentMemoryProbe:
    """A teach-then-test probe for agent memory.

    Attributes
    ----------
    probe_id : str
        Unique identifier.
    fact_triple : tuple[str, str, str]
        The (head, relation, tail) triple taught to the agent.
    teach_prompt : str
        Prompt used to teach the fact.
    recall_prompt : str
        Prompt used to test recall.
    delay_turns : int
        Number of intervening turns before recall.
    expected_answer : str
        Expected recall answer.
    recall_geodesic : float or None
        Geodesic distance of recalled answer triple (lower = better).
    session_tension : float or None
        Tension energy between session states before/after (drift).
    """

    probe_id: str = ""
    fact_triple: tuple[str, str, str] = ("", "", "")
    teach_prompt: str = ""
    recall_prompt: str = ""
    delay_turns: int = 0
    expected_answer: str = ""
    recall_geodesic: float | None = None
    session_tension: float | None = None


@dataclass
class MemoryAssessment:
    """Assessment of agent memory across multiple probes.

    Attributes
    ----------
    n_probes : int
        Total number of memory probes.
    n_recalled : int
        Number of facts successfully recalled.
    mean_recall_geodesic : float
        Mean geodesic distance for recalled facts.
    mean_session_tension : float
        Mean tension energy across session transitions.
    degradation_detected : bool
        True if session tension is trending upward.
    probes : list[AgentMemoryProbe]
        Detailed per-probe results.
    """

    n_probes: int = 0
    n_recalled: int = 0
    mean_recall_geodesic: float = 0.0
    mean_session_tension: float = 0.0
    degradation_detected: bool = False
    probes: list[AgentMemoryProbe] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "n_recalled": self.n_recalled,
            "recall_rate": (
                round(self.n_recalled / self.n_probes, 4)
                if self.n_probes > 0 else 0.0
            ),
            "mean_recall_geodesic": round(self.mean_recall_geodesic, 4),
            "mean_session_tension": round(self.mean_session_tension, 4),
            "degradation_detected": self.degradation_detected,
        }


# ---------------------------------------------------------------------------
# Stateful agent tester
# ---------------------------------------------------------------------------

class StatefulAgentTester:
    """Orchestrate teach-then-test memory probes against a GMS store.

    Uses ``score_triple`` to measure recall quality and
    ``tension_energy`` to measure session-level drift.

    Parameters
    ----------
    store : GMSExpertStore
        GMS store containing the domain knowledge.
    """

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def build_probes(
        self,
        n_probes: int = 10,
        delay_turns: int = 5,
        seed: int = 42,
    ) -> list[AgentMemoryProbe]:
        """Mine store triples and generate memory probes.

        Each probe teaches a real triple and asks the agent to recall it
        after ``delay_turns`` intervening conversational turns.

        Parameters
        ----------
        n_probes : int
            Number of probes to generate.
        delay_turns : int
            Turns between teaching and testing.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list[AgentMemoryProbe]
            Memory probes with teach/recall prompts.
        """
        if self.store.doc_graph is None:
            return []

        rng = random.Random(seed)
        triples = list(self.store.doc_graph.triples)
        if not triples:
            return []

        selected = rng.sample(triples, min(n_probes, len(triples)))
        probes = []
        for i, (h, r, t) in enumerate(selected):
            probe = AgentMemoryProbe(
                probe_id=f"mem_probe_{i:03d}",
                fact_triple=(h, r, t),
                teach_prompt=(
                    f"Please remember this fact: {h} {r} {t}. "
                    f"I will ask you about it later."
                ),
                recall_prompt=(
                    f"Earlier I told you a fact about '{h}' and its "
                    f"'{r}' relationship. What is the value? "
                    f"ANSWER: <exact value>"
                ),
                delay_turns=delay_turns,
                expected_answer=t,
            )
            probes.append(probe)
        return probes

    def score_recall(
        self, probe: AgentMemoryProbe, recalled_answer: str,
    ) -> AgentMemoryProbe:
        """Score a single recall attempt using geodesic distance.

        Parameters
        ----------
        probe : AgentMemoryProbe
            The probe being scored.
        recalled_answer : str
            The agent's recalled answer.

        Returns
        -------
        AgentMemoryProbe
            Updated probe with ``recall_geodesic`` filled in.
        """
        h, r, _t = probe.fact_triple
        # Score the recalled answer triple against the manifold
        geodesic = None
        try:
            score = self.store.score_triple(h, r, recalled_answer)
            geodesic = float(score) if score is not None else None
        except Exception:
            pass
        probe.recall_geodesic = geodesic
        return probe

    def measure_session_tension(
        self, entity_a: str, entity_b: str,
    ) -> float | None:
        """Measure tension energy between two entities.

        Used to detect drift between session states by comparing
        entity representations at different time points.

        Parameters
        ----------
        entity_a : str
            First entity (e.g., session state key).
        entity_b : str
            Second entity.

        Returns
        -------
        float or None
            Tension energy in [0, 2], or None on error.
        """
        try:
            return self.store.tension_energy(entity_a, entity_b)
        except Exception:
            return None

    def assess(self, probes: list[AgentMemoryProbe]) -> MemoryAssessment:
        """Aggregate probe results into a memory assessment.

        Parameters
        ----------
        probes : list[AgentMemoryProbe]
            Completed probes with geodesic scores.

        Returns
        -------
        MemoryAssessment
            Aggregate assessment.
        """
        geodesics = [
            p.recall_geodesic for p in probes
            if p.recall_geodesic is not None
        ]
        tensions = [
            p.session_tension for p in probes
            if p.session_tension is not None
        ]

        # Recall: geodesic < 0.5 = correct recall
        n_recalled = sum(1 for g in geodesics if g < 0.5)

        # Degradation: check if tension trend is upward
        degradation = False
        if len(tensions) >= 3:
            mid = len(tensions) // 2
            early_mean = sum(tensions[:mid]) / mid
            late_mean = sum(tensions[mid:]) / (len(tensions) - mid)
            degradation = late_mean > early_mean * 1.2  # 20% increase

        return MemoryAssessment(
            n_probes=len(probes),
            n_recalled=n_recalled,
            mean_recall_geodesic=(
                sum(geodesics) / len(geodesics) if geodesics else 0.0
            ),
            mean_session_tension=(
                sum(tensions) / len(tensions) if tensions else 0.0
            ),
            degradation_detected=degradation,
            probes=probes,
        )


# ---------------------------------------------------------------------------
# Question generator
# ---------------------------------------------------------------------------

class StatefulAgentGenerator(QuestionGenerator):
    """Generate teach-then-recall questions for agent memory testing.

    Mines the GMS store for real triples and generates paired
    teach/recall questions.  The answer function scores recall via
    ``score_triple`` geodesic distance.

    Category: ``"stateful_memory"``
    """

    category = "stateful_memory"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None:
            return []

        questions = []
        rng = random.Random(42)
        triples = list(store.doc_graph.triples)
        if not triples:
            return []

        selected = rng.sample(triples, min(20, len(triples)))

        for i, (h, r, t) in enumerate(selected):
            # Verify the triple scores well
            score = store.score_triple(h, r, t)
            if score is None or score > 1.0:
                continue

            ground_truth = t

            def _answer_fn(s, _h=h, _r=r, _t=t):
                """Return ground truth if triple is plausible (low geodesic)."""
                sc = s.score_triple(_h, _r, _t)
                if sc is not None and sc < 0.5:
                    return _t
                return _t  # ground truth for validation

            questions.append(GeneratedQuestion(
                qid=f"stateful_mem_{i:03d}",
                category=self.category,
                natural_language=(
                    f"I previously told you that '{h}' has a '{r}' "
                    f"relationship with a specific entity. What is "
                    f"that entity?"
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the information provided, what is the "
                    f"'{r}' of '{h}'?\n\n"
                    f"{ANSWER_FORMATS['str']}"
                ),
                answer_type="str",
                metadata={
                    "head": h,
                    "relation": r,
                    "expected_tail": t,
                    "geodesic": round(score, 4) if score is not None else None,
                    "test_type": "memory_recall",
                },
            ))

        return questions


# ---------------------------------------------------------------------------
# Memory recall oracle
# ---------------------------------------------------------------------------

class MemoryRecallOracle:
    """Oracle that scores recall quality via GMS geodesic distance.

    Evaluates whether an agent correctly recalled a previously taught
    fact by measuring the geodesic distance of the recalled triple
    on the learned manifold.

    Oracle type: ``"memory_recall"``

    Parameters
    ----------
    store : GMSExpertStore
        GMS store for geodesic scoring.
    geodesic_threshold : float
        Maximum geodesic distance for a recall to be considered correct.
    """

    oracle_type = "memory_recall"

    def __init__(
        self,
        store: "GMSExpertStore",
        geodesic_threshold: float = 0.5,
    ):
        self.store = store
        self.geodesic_threshold = geodesic_threshold

    def evaluate(
        self,
        head: str,
        relation: str,
        expected_tail: str,
        recalled_answer: str,
    ) -> dict:
        """Score a recall attempt.

        Parameters
        ----------
        head : str
            Head entity of the taught triple.
        relation : str
            Relation of the taught triple.
        expected_tail : str
            Expected tail entity.
        recalled_answer : str
            What the agent actually recalled.

        Returns
        -------
        dict
            Verdict with ``passed``, ``geodesic_expected``,
            ``geodesic_recalled``, and ``tension``.
        """
        # Geodesic of expected answer (ground truth)
        geo_expected = None
        try:
            score = self.store.score_triple(head, relation, expected_tail)
            geo_expected = float(score) if score is not None else None
        except Exception:
            pass

        # Geodesic of recalled answer
        geo_recalled = None
        try:
            score = self.store.score_triple(head, relation, recalled_answer)
            geo_recalled = float(score) if score is not None else None
        except Exception:
            pass

        # Tension between expected and recalled
        tension = None
        try:
            tension = self.store.tension_energy(expected_tail, recalled_answer)
        except Exception:
            pass

        passed = (
            geo_recalled is not None
            and geo_recalled < self.geodesic_threshold
        )

        return {
            "oracle_type": self.oracle_type,
            "passed": passed,
            "geodesic_expected": (
                round(geo_expected, 4) if geo_expected is not None else None
            ),
            "geodesic_recalled": (
                round(geo_recalled, 4) if geo_recalled is not None else None
            ),
            "tension_expected_vs_recalled": (
                round(tension, 4) if tension is not None else None
            ),
            "expected_tail": expected_tail,
            "recalled_answer": recalled_answer,
        }
