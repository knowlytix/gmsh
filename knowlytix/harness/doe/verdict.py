# SPDX-License-Identifier: Apache-2.0
"""GMSH TypedVerdict — extends GeometricVerdict with claim-level detail.

A TypedVerdict carries the full audit trail: which claims were extracted,
which GMS verifier checked each claim, what the raw geometric signal was,
and how failures were attributed.  It is backwards-compatible with the
existing GeometricVerdict via ``to_geometric_verdict()``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .claims import ClaimBase
from .failures import FailureAttribution
from .verifiers import SubVerdict

def _get_geometric_verdict_class():
    """Lazy import to avoid circular dependency with judge.py."""
    try:
        from .judge import GeometricVerdict
        return GeometricVerdict
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# TypedVerdict
# ---------------------------------------------------------------------------

@dataclass
class TypedVerdict:
    """Full GMSH verdict with claim-level granularity.

    Backwards-compatible with GeometricVerdict: the top-level fields
    (correct, geodesic_confidence, tension_score, holonomy_score,
    overall_confidence) are computed identically.

    New fields provide the audit trail:
    - claims: typed claims extracted from SUT output
    - sub_verdicts: per-claim verification results
    - failures: structured failure attributions
    """

    # -- Backwards-compatible fields from GeometricVerdict --
    correct: bool = False
    detail: str = ""
    geodesic_confidence: float = 0.0
    tension_score: float = 0.0
    holonomy_score: float = 0.0
    overall_confidence: float = 0.0

    # -- GMSH typed fields --
    claims: list[ClaimBase] = field(default_factory=list)
    sub_verdicts: list[SubVerdict] = field(default_factory=list)
    failures: list[FailureAttribution] = field(default_factory=list)

    # -- Provenance --
    run_id: str = ""
    question_id: str = ""
    timestamp: str = ""

    # -- Counts --
    n_claims: int = 0
    n_passed: int = 0
    n_failed: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def from_sub_verdicts(
        cls,
        claims: list[ClaimBase],
        sub_verdicts: list[SubVerdict],
        exact_correct: bool,
        exact_detail: str,
        run_id: str = "",
        question_id: str = "",
        weights: dict | None = None,
    ) -> "TypedVerdict":
        """Construct a TypedVerdict from claims and sub-verdicts.

        Aggregates per-claim geometric signals into the backwards-compatible
        top-level fields (geodesic_confidence, tension_score, etc.) and
        computes overall_confidence using the same weighted formula as
        GeometricVerdict.

        Parameters
        ----------
        claims : list[ClaimBase]
            Typed claims from ClaimCompiler.
        sub_verdicts : list[SubVerdict]
            Per-claim verification results from VerificationRouter.
        exact_correct : bool
            Whether the traditional scorer marked the answer correct.
        exact_detail : str
            Detail string from the traditional scorer.
        run_id : str
            Benchmark run identifier.
        question_id : str
            DOE question identifier.
        weights : dict or None
            Weights for combining signals (default: same as GMSJudge).
        """
        w = weights or {
            "geodesic": 0.3,
            "tension": 0.2,
            "holonomy": 0.2,
            "exact": 0.3,
        }

        # Collect failures
        failures = [
            sv.failure for sv in sub_verdicts
            if sv.failure is not None
        ]

        # Aggregate geometric signals by verifier type
        graph_scores = []
        tension_scores = []
        holonomy_scores = []

        for sv in sub_verdicts:
            if sv.raw_score is None:
                continue
            if sv.verifier_name == "GraphVerifier":
                graph_scores.append(sv.raw_score)
            elif sv.verifier_name == "LogicVerifier":
                if sv.gms_method_used == "tension_energy":
                    tension_scores.append(sv.raw_score)
                elif sv.gms_method_used == "check_holonomy":
                    holonomy_scores.append(sv.raw_score)

        # Compute backwards-compatible fields
        geo_conf = 0.0
        if graph_scores:
            mean_geo = sum(graph_scores) / len(graph_scores)
            geo_conf = math.exp(-mean_geo)

        tension = 0.0
        if tension_scores:
            tension = sum(tension_scores) / len(tension_scores)

        holonomy = 0.0
        if holonomy_scores:
            holonomy = sum(holonomy_scores) / len(holonomy_scores)

        # Overall confidence (same formula as GMSJudge._compute_overall)
        exact_score = 1.0 if exact_correct else 0.0
        tension_quality = max(0.0, min(1.0, 1.0 - tension / 2.0))
        holonomy_quality = math.exp(-holonomy)

        overall = (
            w.get("exact", 0.3) * exact_score
            + w.get("geodesic", 0.3) * geo_conf
            + w.get("tension", 0.2) * tension_quality
            + w.get("holonomy", 0.2) * holonomy_quality
        )
        overall = max(0.0, min(1.0, overall))

        n_passed = sum(1 for sv in sub_verdicts if sv.passed)
        n_failed = sum(1 for sv in sub_verdicts if not sv.passed)

        # Build detail string
        parts = [exact_detail]
        parts.append(
            f"GMSH: {len(claims)} claims, {n_passed} passed, "
            f"{n_failed} failed"
        )
        parts.append(
            f"geo_conf={geo_conf:.3f} tension={tension:.3f} "
            f"holonomy={holonomy:.3f} overall={overall:.3f}"
        )
        if failures:
            codes = set(f.code.value for f in failures)
            parts.append(f"failures: {', '.join(sorted(codes))}")

        return cls(
            correct=exact_correct,
            detail=" | ".join(parts),
            geodesic_confidence=geo_conf,
            tension_score=tension,
            holonomy_score=holonomy,
            overall_confidence=overall,
            claims=claims,
            sub_verdicts=sub_verdicts,
            failures=failures,
            run_id=run_id,
            question_id=question_id,
            n_claims=len(claims),
            n_passed=n_passed,
            n_failed=n_failed,
        )

    def to_geometric_verdict(self):
        """Downcast to GeometricVerdict for backwards compatibility.

        Returns a GeometricVerdict with the same top-level fields.
        If GeometricVerdict is not importable, returns a dict.
        """
        GeoVerdict = _get_geometric_verdict_class()
        if GeoVerdict is not None:
            return GeoVerdict(
                correct=self.correct,
                detail=self.detail,
                geodesic_confidence=self.geodesic_confidence,
                tension_score=self.tension_score,
                holonomy_score=self.holonomy_score,
                overall_confidence=self.overall_confidence,
            )
        return {
            "correct": self.correct,
            "detail": self.detail,
            "geodesic_confidence": self.geodesic_confidence,
            "tension_score": self.tension_score,
            "holonomy_score": self.holonomy_score,
            "overall_confidence": self.overall_confidence,
        }

    @property
    def failure_codes(self) -> list[str]:
        """Unique failure codes in this verdict."""
        return sorted(set(f.code.value for f in self.failures))

    @property
    def verifier_summary(self) -> dict[str, dict]:
        """Per-verifier pass/fail summary."""
        summary: dict[str, dict] = {}
        for sv in self.sub_verdicts:
            name = sv.verifier_name
            if name not in summary:
                summary[name] = {"passed": 0, "failed": 0, "total": 0}
            summary[name]["total"] += 1
            if sv.passed:
                summary[name]["passed"] += 1
            else:
                summary[name]["failed"] += 1
        return summary
