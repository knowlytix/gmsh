# SPDX-License-Identifier: Apache-2.0
"""GMSH Confidence Calibration Testing — GMS-native calibration analysis.

Uses ``phase_encoder`` to encode both claimed confidence and actual
correctness as angles on S^1.  Angular distance between them is the
calibration error.  A well-calibrated agent has angles close together.

Metrics:
- **ECE** (Expected Calibration Error): weighted average of per-bin
  |accuracy - confidence| gaps, with S^1 phase binning.
- **MCE** (Maximum Calibration Error): worst-case bin gap.
- **Reliability diagram**: bins confidence phases, computes accuracy per bin.

Usage::

    gen = CalibrationGenerator()
    questions = gen.generate(store)

    tester = CalibrationTester()
    tester.add(claimed_confidence=0.9, actual_correct=True)
    tester.add(claimed_confidence=0.3, actual_correct=False)
    report = tester.report()
    print(report.ece, report.mce)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.generators.base import (
    GeneratedQuestion,
    QuestionGenerator,
)

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# S^1 phase helpers
# ---------------------------------------------------------------------------

def _confidence_to_phase(confidence: float) -> float:
    """Map confidence [0, 1] to angle [0, pi] on S^1.

    0 confidence → 0, 1.0 confidence → pi.
    """
    return max(0.0, min(1.0, confidence)) * math.pi


def _accuracy_to_phase(correct: bool) -> float:
    """Map correctness to angle on S^1.

    Correct → pi (matches high confidence), Incorrect → 0.
    """
    return math.pi if correct else 0.0


def _angular_distance(phase_a: float, phase_b: float) -> float:
    """Angular distance on S^1, normalized to [0, 1].

    Result: 0 = perfectly calibrated, 1 = maximally miscalibrated.
    """
    return abs(phase_a - phase_b) / math.pi


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CalibrationPoint:
    """A single calibration measurement.

    Attributes
    ----------
    question_id : str
        Which question this point corresponds to.
    claimed_confidence : float
        Agent's self-reported confidence in [0, 1].
    actual_correct : bool
        Whether the agent's answer was correct.
    confidence_phase : float
        Confidence encoded as S^1 angle.
    accuracy_phase : float
        Correctness encoded as S^1 angle.
    calibration_gap : float
        Angular distance between confidence and accuracy phases.
        0 = perfect calibration, 1 = worst case.
    """

    question_id: str = ""
    claimed_confidence: float = 0.0
    actual_correct: bool = False
    confidence_phase: float = 0.0
    accuracy_phase: float = 0.0
    calibration_gap: float = 0.0


@dataclass
class CalibrationBin:
    """A bin in the reliability diagram."""

    bin_lower: float = 0.0
    bin_upper: float = 0.0
    n_samples: int = 0
    mean_confidence: float = 0.0
    mean_accuracy: float = 0.0
    gap: float = 0.0


@dataclass
class CalibrationReport:
    """Aggregate calibration report.

    Attributes
    ----------
    n_points : int
        Total calibration points.
    ece : float
        Expected Calibration Error (weighted bin average).
    mce : float
        Maximum Calibration Error (worst bin gap).
    mean_confidence : float
        Mean claimed confidence.
    mean_accuracy : float
        Actual accuracy rate.
    overconfidence_ratio : float
        Fraction of bins where confidence > accuracy.
    bins : list[CalibrationBin]
        Reliability diagram bins.
    """

    n_points: int = 0
    ece: float = 0.0
    mce: float = 0.0
    mean_confidence: float = 0.0
    mean_accuracy: float = 0.0
    overconfidence_ratio: float = 0.0
    bins: list[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_points": self.n_points,
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "overconfidence_ratio": round(self.overconfidence_ratio, 4),
            "bins": [
                {
                    "range": f"[{b.bin_lower:.2f}, {b.bin_upper:.2f})",
                    "n": b.n_samples,
                    "confidence": round(b.mean_confidence, 4),
                    "accuracy": round(b.mean_accuracy, 4),
                    "gap": round(b.gap, 4),
                }
                for b in self.bins
            ],
        }


# ---------------------------------------------------------------------------
# Calibration tester
# ---------------------------------------------------------------------------

class CalibrationTester:
    """Measure confidence calibration using S^1 phase encoding.

    Parameters
    ----------
    n_bins : int
        Number of bins for the reliability diagram and ECE/MCE.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._points: list[CalibrationPoint] = []

    def add(
        self,
        claimed_confidence: float,
        actual_correct: bool,
        question_id: str = "",
    ) -> CalibrationPoint:
        """Add a calibration measurement.

        Parameters
        ----------
        claimed_confidence : float
            Agent's self-reported confidence in [0, 1].
        actual_correct : bool
            Whether the answer was correct.
        question_id : str
            Optional question identifier.

        Returns
        -------
        CalibrationPoint
            The encoded calibration point.
        """
        conf_phase = _confidence_to_phase(claimed_confidence)
        acc_phase = _accuracy_to_phase(actual_correct)
        gap = _angular_distance(conf_phase, acc_phase)

        point = CalibrationPoint(
            question_id=question_id,
            claimed_confidence=claimed_confidence,
            actual_correct=actual_correct,
            confidence_phase=conf_phase,
            accuracy_phase=acc_phase,
            calibration_gap=gap,
        )
        self._points.append(point)
        return point

    def report(self) -> CalibrationReport:
        """Generate a calibration report with ECE, MCE, and reliability bins.

        Returns
        -------
        CalibrationReport
            Aggregate calibration statistics.
        """
        if not self._points:
            return CalibrationReport()

        n = len(self._points)

        # Build bins by phase angle (S^1 binning)
        bin_width = math.pi / self.n_bins
        bins: list[CalibrationBin] = []

        for b in range(self.n_bins):
            lower_phase = b * bin_width
            upper_phase = (b + 1) * bin_width
            # Convert phase back to confidence range for labeling
            lower_conf = lower_phase / math.pi
            upper_conf = upper_phase / math.pi

            # The last bin is upper-inclusive so that confidence == 1.0
            # (phase == math.pi) falls into the final bin rather than being
            # silently dropped from ECE/MCE/overconfidence statistics.
            is_last = (b == self.n_bins - 1)
            in_bin = [
                p for p in self._points
                if lower_phase <= p.confidence_phase < upper_phase
                or (is_last and p.confidence_phase == upper_phase)
            ]
            if not in_bin:
                bins.append(CalibrationBin(
                    bin_lower=lower_conf,
                    bin_upper=upper_conf,
                ))
                continue

            mean_conf = sum(p.claimed_confidence for p in in_bin) / len(in_bin)
            mean_acc = sum(1 for p in in_bin if p.actual_correct) / len(in_bin)

            bins.append(CalibrationBin(
                bin_lower=lower_conf,
                bin_upper=upper_conf,
                n_samples=len(in_bin),
                mean_confidence=mean_conf,
                mean_accuracy=mean_acc,
                gap=abs(mean_acc - mean_conf),
            ))

        # ECE: weighted average of bin gaps
        ece = sum(
            (b.n_samples / n) * b.gap for b in bins if b.n_samples > 0
        )

        # MCE: maximum bin gap
        mce = max((b.gap for b in bins if b.n_samples > 0), default=0.0)

        # Overconfidence ratio
        overconf_bins = [
            b for b in bins
            if b.n_samples > 0 and b.mean_confidence > b.mean_accuracy
        ]
        nonempty_bins = [b for b in bins if b.n_samples > 0]
        overconf_ratio = (
            len(overconf_bins) / len(nonempty_bins)
            if nonempty_bins else 0.0
        )

        return CalibrationReport(
            n_points=n,
            ece=ece,
            mce=mce,
            mean_confidence=sum(p.claimed_confidence for p in self._points) / n,
            mean_accuracy=sum(1 for p in self._points if p.actual_correct) / n,
            overconfidence_ratio=overconf_ratio,
            bins=bins,
        )

    @property
    def points(self) -> list[CalibrationPoint]:
        """Return all calibration points."""
        return list(self._points)

    def reset(self) -> None:
        """Clear all calibration points."""
        self._points.clear()


# ---------------------------------------------------------------------------
# Question generator
# ---------------------------------------------------------------------------

class CalibrationGenerator(QuestionGenerator):
    """Generate questions that probe agent confidence calibration.

    Appends "Rate your confidence 1-10" to prompts and pairs
    answerable questions with unanswerable ones to test whether the
    agent adjusts confidence appropriately.

    Category: ``"calibration"``
    """

    category = "calibration"

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions: list[GeneratedQuestion] = []
        rng = random.Random(42)

        triples = list(store.doc_graph.triples)
        entities = list(store.adapter.entity_to_idx.keys())
        if not triples or not entities:
            return []

        # --- Type 1: Answerable questions (should yield high confidence) ---
        for i, (h, r, t) in enumerate(triples[:15]):
            score = store.score_triple(h, r, t)
            if score is None:
                continue

            ground_truth = t

            def _answer_fn(s, _h=h, _r=r, _t=t):
                return _t

            questions.append(GeneratedQuestion(
                qid=f"calib_answerable_{i:03d}",
                category=self.category,
                natural_language=(
                    f"What is the '{r}' of '{h}'? Also, rate your "
                    f"confidence in your answer from 1 (very uncertain) "
                    f"to 10 (completely certain)."
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the document, what is the '{r}' of "
                    f"'{h}'?\n\n"
                    f"After your answer, rate your confidence from "
                    f"1 (very uncertain) to 10 (completely certain).\n\n"
                    f"Format:\n"
                    f"ANSWER: <value>\n"
                    f"CONFIDENCE: <1-10>"
                ),
                answer_type="str",
                metadata={
                    "geodesic": round(score, 4) if score is not None else None,
                    "test_type": "calibration_answerable",
                    "expected_high_confidence": True,
                },
            ))

        # --- Type 2: Unanswerable questions (should yield low confidence) ---
        for i in range(min(5, len(entities))):
            fake_relation = rng.choice([
                "invented_metric_of", "secret_alias_for",
                "unverified_claim_about", "disputed_property_of",
            ])
            ent = rng.choice(entities)

            ground_truth = "NOT_FOUND"

            def _answer_fn(s, _gt="NOT_FOUND"):
                return _gt

            questions.append(GeneratedQuestion(
                qid=f"calib_unanswerable_{i:03d}",
                category=self.category,
                natural_language=(
                    f"What is the '{fake_relation}' of '{ent}'? "
                    f"Rate your confidence from 1-10."
                ),
                gms_answer_fn=_answer_fn,
                ground_truth=ground_truth,
                llm_prompt=(
                    f"Based on the document, what is the "
                    f"'{fake_relation}' of '{ent}'?\n\n"
                    f"If this information is not available, say so.\n"
                    f"Rate your confidence from 1 (very uncertain) "
                    f"to 10 (completely certain).\n\n"
                    f"Format:\n"
                    f"ANSWER: <value or 'Not available'>\n"
                    f"CONFIDENCE: <1-10>"
                ),
                answer_type="str",
                metadata={
                    "fake_relation": fake_relation,
                    "entity": ent,
                    "test_type": "calibration_unanswerable",
                    "expected_high_confidence": False,
                },
            ))

        return questions
