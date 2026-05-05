# SPDX-License-Identifier: Apache-2.0
"""GMSH Audit Reporter — audit packets, failure analysis, release gates.

Provides structured reporting over GMSH typed verdicts:

- **AuditPacket**: per-question full audit trail (question, answer, claims,
  sub-verdicts, failures, truth references)
- **ReleaseGate**: pass/fail determination against configurable thresholds
- **AuditReporter**: generates audit packets, failure reports, release gates,
  and executive summaries from a benchmark run
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.harness.graphdoe.results import DOEBenchmarkResult
    from .verdict import TypedVerdict
    from .truth import TruthStore


# ---------------------------------------------------------------------------
# Audit Packet
# ---------------------------------------------------------------------------

@dataclass
class AuditPacket:
    """Complete audit trail for a single DOE question.

    Bundles the question, ground truth, SUT answer, typed verdict (claims,
    sub-verdicts, failures), and truth-store references into one reviewable
    unit.
    """

    doe_qid: str = ""
    category: str = ""
    question: str = ""
    ground_truth: str = ""
    sut_answer: str = ""
    correct: bool = False
    detail: str = ""

    # Geometric scores
    geodesic_confidence: float = 0.0
    tension_score: float = 0.0
    holonomy_score: float = 0.0
    overall_confidence: float = 0.0

    # GMSH typed verdict detail
    n_claims: int = 0
    n_passed: int = 0
    n_failed: int = 0
    claim_types: list[str] = field(default_factory=list)
    failure_codes: list[str] = field(default_factory=list)
    verifier_summary: dict = field(default_factory=dict)

    # Factor assignment
    factor_assignment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doe_qid": self.doe_qid,
            "category": self.category,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "sut_answer": self.sut_answer,
            "correct": self.correct,
            "detail": self.detail,
            "geodesic_confidence": self.geodesic_confidence,
            "tension_score": self.tension_score,
            "holonomy_score": self.holonomy_score,
            "overall_confidence": self.overall_confidence,
            "n_claims": self.n_claims,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "claim_types": self.claim_types,
            "failure_codes": self.failure_codes,
            "verifier_summary": self.verifier_summary,
            "factor_assignment": self.factor_assignment,
        }


# ---------------------------------------------------------------------------
# Release Gate
# ---------------------------------------------------------------------------

@dataclass
class ReleaseGate:
    """Pass/fail determination against configurable quality thresholds.

    A release gate checks:
    - Overall accuracy >= threshold
    - Mean geometric confidence >= threshold
    - Failure counts per code <= budget
    """

    passed: bool = False
    accuracy_threshold: float = 0.0
    actual_accuracy: float = 0.0
    geo_confidence_threshold: float = 0.0
    actual_geo_confidence: float = 0.0
    failure_budget: dict[str, int] = field(default_factory=dict)
    actual_failures: dict[str, int] = field(default_factory=dict)
    blocking_failures: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "accuracy_threshold": self.accuracy_threshold,
            "actual_accuracy": self.actual_accuracy,
            "geo_confidence_threshold": self.geo_confidence_threshold,
            "actual_geo_confidence": self.actual_geo_confidence,
            "failure_budget": self.failure_budget,
            "actual_failures": self.actual_failures,
            "blocking_failures": self.blocking_failures,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Risk-Tiered Release Gates (WS7)
# ---------------------------------------------------------------------------


@dataclass
class RiskTierProfile:
    """Per-risk-tier quality thresholds.

    Three factory methods provide sensible defaults for common tiers:

    - ``advisory()`` — low-risk, advisory-only agents
    - ``recommendation()`` — agents that produce recommendations / drafts
    - ``action_taking()`` — high-risk agents that take real-world actions

    Parameters
    ----------
    tier_name : str
        Unique identifier for this tier.
    accuracy_threshold : float
        Minimum pass rate.
    geo_confidence_threshold : float
        Minimum mean geometric confidence.
    max_tension : float
        Maximum allowed mean tension energy.
    max_holonomy : float
        Maximum allowed mean holonomy defect.
    failure_budget : dict[str, int]
        Max allowed failures per failure code.
    required_evaluators : list[str]
        Evaluator names that must have run (e.g. ``["ProcessEvaluator"]``).
    max_regression_rate : float
        Maximum fraction of regressions vs baseline run.
    require_baseline_comparison : bool
        Whether a baseline run comparison is required for this tier.
    """

    tier_name: str = "standard"
    accuracy_threshold: float = 0.0
    geo_confidence_threshold: float = 0.0
    max_tension: float = 2.0
    max_holonomy: float = 1.0
    failure_budget: dict[str, int] = field(default_factory=dict)
    required_evaluators: list[str] = field(default_factory=list)
    max_regression_rate: float = 0.05
    require_baseline_comparison: bool = False

    @classmethod
    def advisory(cls) -> "RiskTierProfile":
        """Low-risk tier: correctness and groundedness focus."""
        return cls(
            tier_name="advisory",
            accuracy_threshold=0.70,
            geo_confidence_threshold=0.40,
        )

    @classmethod
    def recommendation(cls) -> "RiskTierProfile":
        """Medium-risk tier: adds process compliance and evidence sufficiency."""
        return cls(
            tier_name="recommendation",
            accuracy_threshold=0.80,
            geo_confidence_threshold=0.55,
            max_tension=1.5,
            failure_budget={"hallucination_detected": 0},
        )

    @classmethod
    def action_taking(cls) -> "RiskTierProfile":
        """High-risk tier: strict thresholds, requires extended evaluators."""
        return cls(
            tier_name="action_taking",
            accuracy_threshold=0.95,
            geo_confidence_threshold=0.70,
            max_tension=1.0,
            max_holonomy=0.5,
            required_evaluators=["ProcessEvaluator", "RecoveryEvaluator"],
            failure_budget={
                "hallucination_detected": 0,
                "policy_violation": 0,
                "policy_blocked_pre_execution": 0,
            },
            require_baseline_comparison=True,
            max_regression_rate=0.02,
        )

    def to_dict(self) -> dict:
        return {
            "tier_name": self.tier_name,
            "accuracy_threshold": self.accuracy_threshold,
            "geo_confidence_threshold": self.geo_confidence_threshold,
            "max_tension": self.max_tension,
            "max_holonomy": self.max_holonomy,
            "failure_budget": self.failure_budget,
            "required_evaluators": self.required_evaluators,
            "max_regression_rate": self.max_regression_rate,
            "require_baseline_comparison": self.require_baseline_comparison,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskTierProfile":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class TieredReleaseGate:
    """Release gate result for a specific risk tier.

    Parameters
    ----------
    tier : RiskTierProfile
        The profile this gate evaluated against.
    passed : bool
        Whether all checks passed.
    checks : dict[str, dict]
        Per-check detail, e.g. ``{"accuracy": {"threshold": 0.9, "actual": 0.85, "passed": False}}``.
    blocking_reasons : list[str]
        Human-readable reasons the gate was blocked.
    regression_comparison : RunComparison or None
        If a baseline comparison was performed, the comparison result.
    """

    tier: RiskTierProfile = field(default_factory=RiskTierProfile)
    passed: bool = False
    checks: dict[str, dict] = field(default_factory=dict)
    blocking_reasons: list[str] = field(default_factory=list)
    regression_comparison: object = None  # RunComparison, avoid circular import
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.to_dict(),
            "passed": self.passed,
            "checks": self.checks,
            "blocking_reasons": self.blocking_reasons,
            "timestamp": self.timestamp,
        }


@dataclass
class RunTrend:
    """Trend data for a metric across multiple runs.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g. "accuracy", "geo_confidence").
    run_ids : list[str]
        Run identifiers in chronological order.
    values : list[float]
        Metric values for each run.
    trend_direction : str
        "improving", "degrading", or "stable".
    slope : float
        Linear regression slope (positive = improving).
    """

    metric_name: str = ""
    run_ids: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    trend_direction: str = "stable"
    slope: float = 0.0

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "run_ids": self.run_ids,
            "values": self.values,
            "trend_direction": self.trend_direction,
            "slope": self.slope,
        }


# ---------------------------------------------------------------------------
# Audit Reporter
# ---------------------------------------------------------------------------

class AuditReporter:
    """Generate structured audit reports from GMSH benchmark results.

    Parameters
    ----------
    result : DOEBenchmarkResult
        The benchmark result (questions, accuracy, factors).
    typed_verdicts : dict[str, TypedVerdict]
        Map from doe_qid to TypedVerdict.
    truth_store : TruthStore or None
        If provided, enriches audit packets with truth references.
    """

    def __init__(
        self,
        result: "DOEBenchmarkResult",
        typed_verdicts: dict[str, "TypedVerdict"],
        truth_store: "TruthStore | None" = None,
    ):
        self.result = result
        self.typed_verdicts = typed_verdicts
        self.truth_store = truth_store

    # ----- Audit Packets --------------------------------------------------

    def generate_audit_packets(self) -> list[AuditPacket]:
        """Create one AuditPacket per question in the benchmark result."""
        packets = []

        for qr in self.result.questions:
            tv = self.typed_verdicts.get(qr.doe_qid)

            packet = AuditPacket(
                doe_qid=qr.doe_qid,
                category=qr.category,
                question=qr.question,
                ground_truth=qr.ground_truth,
                sut_answer=qr.answer,
                correct=qr.correct,
                detail=qr.detail,
                factor_assignment=qr.factor_assignment,
            )

            if tv:
                packet.geodesic_confidence = tv.geodesic_confidence
                packet.tension_score = tv.tension_score
                packet.holonomy_score = tv.holonomy_score
                packet.overall_confidence = tv.overall_confidence
                packet.n_claims = tv.n_claims
                packet.n_passed = tv.n_passed
                packet.n_failed = tv.n_failed
                packet.claim_types = [c.claim_type for c in tv.claims]
                packet.failure_codes = tv.failure_codes
                packet.verifier_summary = tv.verifier_summary

            packets.append(packet)

        return packets

    # ----- Failure Report -------------------------------------------------

    def generate_failure_report(self) -> dict:
        """Analyze failure patterns across the benchmark run.

        Returns
        -------
        dict
            Keys: by_code, by_category, by_severity, by_verifier,
            systematic_patterns, total_failures.
        """
        by_code: Counter = Counter()
        by_category: dict[str, Counter] = defaultdict(Counter)
        by_severity: Counter = Counter()
        by_verifier: Counter = Counter()

        # Heatmap: failure_code x question_category
        heatmap: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        for qr in self.result.questions:
            tv = self.typed_verdicts.get(qr.doe_qid)
            if not tv:
                continue

            for failure in tv.failures:
                code = failure.code.value
                by_code[code] += 1
                by_category[qr.category][code] += 1
                by_severity[failure.severity.value] += 1
                by_verifier[failure.verifier] += 1
                heatmap[code][qr.category] += 1

        # Detect systematic patterns (e.g. all questions in a category fail)
        patterns = []
        for category in sorted(set(qr.category for qr in self.result.questions)):
            cat_qs = [qr for qr in self.result.questions
                      if qr.category == category]
            cat_failures = sum(
                1 for qr in cat_qs
                if self.typed_verdicts.get(qr.doe_qid)
                and self.typed_verdicts[qr.doe_qid].n_failed > 0
            )
            if cat_qs and cat_failures / len(cat_qs) >= 0.8:
                top_codes = by_category.get(category, Counter()).most_common(3)
                patterns.append({
                    "category": category,
                    "failure_rate": cat_failures / len(cat_qs),
                    "total_questions": len(cat_qs),
                    "top_failure_codes": [
                        {"code": c, "count": n} for c, n in top_codes
                    ],
                })

        return {
            "by_code": dict(by_code.most_common()),
            "by_category": {
                cat: dict(codes.most_common())
                for cat, codes in sorted(by_category.items())
            },
            "by_severity": dict(by_severity),
            "by_verifier": dict(by_verifier.most_common()),
            "heatmap": {code: dict(cats) for code, cats in heatmap.items()},
            "systematic_patterns": patterns,
            "total_failures": sum(by_code.values()),
        }

    # ----- Release Gate ---------------------------------------------------

    def generate_release_gate(
        self,
        accuracy_threshold: float = 0.80,
        geo_confidence_threshold: float = 0.50,
        failure_budget: dict[str, int] | None = None,
    ) -> ReleaseGate:
        """Check benchmark results against quality thresholds.

        Parameters
        ----------
        accuracy_threshold : float
            Minimum pass rate (0-1).
        geo_confidence_threshold : float
            Minimum mean geometric confidence (0-1).
        failure_budget : dict[str, int] or None
            Max allowed failures per code. E.g. {"exact_mismatch": 5}.
            Codes exceeding their budget block the release.
        """
        budget = failure_budget or {}

        # Compute actual metrics
        actual_accuracy = self.result.accuracy

        geo_confidences = []
        actual_failures: Counter = Counter()
        for qr in self.result.questions:
            tv = self.typed_verdicts.get(qr.doe_qid)
            if tv:
                geo_confidences.append(tv.overall_confidence)
                for failure in tv.failures:
                    actual_failures[failure.code.value] += 1

        actual_geo = (
            sum(geo_confidences) / len(geo_confidences)
            if geo_confidences else 0.0
        )

        # Check thresholds
        blocking = []
        if actual_accuracy < accuracy_threshold:
            blocking.append(
                f"accuracy {actual_accuracy:.1%} < {accuracy_threshold:.1%}")
        if actual_geo < geo_confidence_threshold:
            blocking.append(
                f"geo_confidence {actual_geo:.3f} < {geo_confidence_threshold:.3f}")

        for code, max_count in budget.items():
            actual_count = actual_failures.get(code, 0)
            if actual_count > max_count:
                blocking.append(
                    f"{code}: {actual_count} > {max_count} (budget)")

        return ReleaseGate(
            passed=len(blocking) == 0,
            accuracy_threshold=accuracy_threshold,
            actual_accuracy=actual_accuracy,
            geo_confidence_threshold=geo_confidence_threshold,
            actual_geo_confidence=actual_geo,
            failure_budget=budget,
            actual_failures=dict(actual_failures),
            blocking_failures=blocking,
        )

    # ----- Executive Summary ----------------------------------------------

    def generate_executive_summary(self) -> str:
        """One-page text summary: pass rate, top failures, recommendation.

        Returns
        -------
        str
            Formatted text summary suitable for reports or CLI output.
        """
        r = self.result
        n = r.total_questions
        acc = r.accuracy

        lines = [
            "GMSH Benchmark Executive Summary",
            f"{'=' * 50}",
            f"Instance:   {r.instance_name}",
            f"Model:      {r.model}",
            f"Timestamp:  {r.timestamp}",
            f"Questions:  {n} ({r.n_base_questions} base x "
            f"{r.n_design_runs} DOE runs)",
            f"Accuracy:   {r.total_correct}/{n} ({acc:.1%})",
            "",
        ]

        # Geometric confidence summary
        geo_vals = []
        tension_vals = []
        holonomy_vals = []
        total_claims = 0
        total_passed = 0
        total_failed = 0

        for tv in self.typed_verdicts.values():
            geo_vals.append(tv.overall_confidence)
            tension_vals.append(tv.tension_score)
            holonomy_vals.append(tv.holonomy_score)
            total_claims += tv.n_claims
            total_passed += tv.n_passed
            total_failed += tv.n_failed

        if geo_vals:
            mean_geo = sum(geo_vals) / len(geo_vals)
            mean_tension = sum(tension_vals) / len(tension_vals)
            mean_holonomy = sum(holonomy_vals) / len(holonomy_vals)
            lines.extend([
                "Geometric Confidence:",
                f"  Mean overall:   {mean_geo:.3f}",
                f"  Mean tension:   {mean_tension:.3f}",
                f"  Mean holonomy:  {mean_holonomy:.3f}",
                f"  Total claims:   {total_claims} "
                f"({total_passed} passed, {total_failed} failed)",
                "",
            ])

        # Top failures
        failure_report = self.generate_failure_report()
        if failure_report["total_failures"] > 0:
            lines.append(f"Top Failure Codes ({failure_report['total_failures']} total):")
            for code, count in sorted(
                failure_report["by_code"].items(),
                key=lambda x: -x[1],
            )[:5]:
                lines.append(f"  {code}: {count}")
            lines.append("")

        # Systematic patterns
        patterns = failure_report.get("systematic_patterns", [])
        if patterns:
            lines.append("Systematic Failure Patterns:")
            for p in patterns:
                top_codes = ", ".join(
                    c["code"] for c in p["top_failure_codes"][:2])
                lines.append(
                    f"  {p['category']}: {p['failure_rate']:.0%} failure rate "
                    f"({p['total_questions']} questions) — {top_codes}")
            lines.append("")

        # Category breakdown
        lines.append("Category Breakdown:")
        categories = sorted(set(qr.category for qr in r.questions))
        for cat in categories:
            cat_qs = [qr for qr in r.questions if qr.category == cat]
            cat_correct = sum(1 for qr in cat_qs if qr.correct)
            cat_rate = cat_correct / len(cat_qs) if cat_qs else 0
            lines.append(
                f"  {cat:<25} {cat_correct:>3}/{len(cat_qs):<3} ({cat_rate:.0%})")
        lines.append("")

        # Recommendation
        if acc >= 0.9:
            recommendation = "PASS — Strong performance across all categories."
        elif acc >= 0.7:
            recommendation = (
                "CONDITIONAL — Adequate overall, but review categories "
                "with low accuracy before release.")
        else:
            recommendation = (
                "FAIL — Significant accuracy gaps. Address systematic "
                "failure patterns before release.")

        if patterns:
            recommendation += (
                f" Note: {len(patterns)} categories show systematic "
                f"failure patterns requiring investigation.")

        lines.extend([
            f"Recommendation: {recommendation}",
        ])

        return "\n".join(lines)

    # ----- Risk-Tiered Release Gates (WS7) --------------------------------

    def generate_tiered_release_gate(
        self,
        profiles: list[RiskTierProfile],
        baseline_result: "DOEBenchmarkResult | None" = None,
    ) -> list[TieredReleaseGate]:
        """Evaluate against multiple risk tier profiles.

        Parameters
        ----------
        profiles : list[RiskTierProfile]
            Tier profiles to evaluate against.
        baseline_result : DOEBenchmarkResult or None
            If provided and a tier requires baseline comparison,
            compute run comparison for regression checking.

        Returns
        -------
        list[TieredReleaseGate]
            One gate result per tier profile.
        """
        # Precompute shared metrics
        actual_accuracy = self.result.accuracy

        geo_vals, tension_vals, holonomy_vals = [], [], []
        actual_failures: Counter = Counter()
        verifiers_seen: set = set()

        for tv in self.typed_verdicts.values():
            geo_vals.append(tv.overall_confidence)
            tension_vals.append(tv.tension_score)
            holonomy_vals.append(tv.holonomy_score)
            for failure in tv.failures:
                actual_failures[failure.code.value] += 1
            for sv in tv.sub_verdicts:
                verifiers_seen.add(sv.verifier_name)

        mean_geo = sum(geo_vals) / len(geo_vals) if geo_vals else 0.0
        mean_tension = sum(tension_vals) / len(tension_vals) if tension_vals else 0.0
        mean_holonomy = sum(holonomy_vals) / len(holonomy_vals) if holonomy_vals else 0.0

        # Compute baseline comparison if needed
        comparison = None
        if baseline_result is not None:
            try:
                from .replay import compare_runs
                comparison = compare_runs(baseline_result, self.result)
            except ImportError:
                pass

        gates = []
        for profile in profiles:
            checks: dict[str, dict] = {}
            blocking: list[str] = []

            # Accuracy
            passed_acc = actual_accuracy >= profile.accuracy_threshold
            checks["accuracy"] = {
                "threshold": profile.accuracy_threshold,
                "actual": actual_accuracy,
                "passed": passed_acc,
            }
            if not passed_acc:
                blocking.append(
                    f"accuracy {actual_accuracy:.1%} < "
                    f"{profile.accuracy_threshold:.1%}")

            # Geometric confidence
            passed_geo = mean_geo >= profile.geo_confidence_threshold
            checks["geo_confidence"] = {
                "threshold": profile.geo_confidence_threshold,
                "actual": mean_geo,
                "passed": passed_geo,
            }
            if not passed_geo:
                blocking.append(
                    f"geo_confidence {mean_geo:.3f} < "
                    f"{profile.geo_confidence_threshold:.3f}")

            # Tension
            passed_tension = mean_tension <= profile.max_tension
            checks["tension"] = {
                "threshold": profile.max_tension,
                "actual": mean_tension,
                "passed": passed_tension,
            }
            if not passed_tension:
                blocking.append(
                    f"tension {mean_tension:.3f} > {profile.max_tension:.3f}")

            # Holonomy
            passed_holonomy = mean_holonomy <= profile.max_holonomy
            checks["holonomy"] = {
                "threshold": profile.max_holonomy,
                "actual": mean_holonomy,
                "passed": passed_holonomy,
            }
            if not passed_holonomy:
                blocking.append(
                    f"holonomy {mean_holonomy:.3f} > "
                    f"{profile.max_holonomy:.3f}")

            # Failure budget
            for code, max_count in profile.failure_budget.items():
                actual_count = actual_failures.get(code, 0)
                code_passed = actual_count <= max_count
                checks[f"budget_{code}"] = {
                    "threshold": max_count,
                    "actual": actual_count,
                    "passed": code_passed,
                }
                if not code_passed:
                    blocking.append(
                        f"{code}: {actual_count} > {max_count} (budget)")

            # Required evaluators
            for eval_name in profile.required_evaluators:
                eval_present = eval_name in verifiers_seen
                checks[f"evaluator_{eval_name}"] = {
                    "threshold": "required",
                    "actual": "present" if eval_present else "missing",
                    "passed": eval_present,
                }
                if not eval_present:
                    blocking.append(f"required evaluator {eval_name} missing")

            # Regression comparison
            tier_comparison = None
            if profile.require_baseline_comparison:
                if comparison is not None:
                    tier_comparison = comparison
                    reg_rate = comparison.regression_rate
                    reg_passed = reg_rate <= profile.max_regression_rate
                    checks["regression_rate"] = {
                        "threshold": profile.max_regression_rate,
                        "actual": reg_rate,
                        "passed": reg_passed,
                    }
                    if not reg_passed:
                        blocking.append(
                            f"regression rate {reg_rate:.1%} > "
                            f"{profile.max_regression_rate:.1%}")
                else:
                    checks["baseline_comparison"] = {
                        "threshold": "required",
                        "actual": "not provided",
                        "passed": False,
                    }
                    blocking.append(
                        "baseline comparison required but not provided")

            gates.append(TieredReleaseGate(
                tier=profile,
                passed=len(blocking) == 0,
                checks=checks,
                blocking_reasons=blocking,
                regression_comparison=tier_comparison,
            ))

        return gates

    def compute_trends(
        self,
        run_results: list[tuple[str, "DOEBenchmarkResult"]],
        metrics: list[str] | None = None,
    ) -> list[RunTrend]:
        """Compute metric trends across historical runs.

        Parameters
        ----------
        run_results : list[tuple[str, DOEBenchmarkResult]]
            Pairs of (run_id, result) in chronological order.
        metrics : list[str] or None
            Which metrics to trend. Default: accuracy.

        Returns
        -------
        list[RunTrend]
            One trend per metric.
        """
        if metrics is None:
            metrics = ["accuracy"]

        trends = []
        for metric in metrics:
            run_ids = []
            values = []

            for run_id, result in run_results:
                run_ids.append(run_id)
                if metric == "accuracy":
                    values.append(result.accuracy)
                else:
                    values.append(0.0)

            # Compute linear regression slope
            slope = 0.0
            direction = "stable"
            if len(values) >= 2:
                x = np.arange(len(values), dtype=float)
                y = np.array(values, dtype=float)
                if np.std(y) > 1e-10:
                    slope = float(np.polyfit(x, y, 1)[0])
                    if slope > 0.01:
                        direction = "improving"
                    elif slope < -0.01:
                        direction = "degrading"

            trends.append(RunTrend(
                metric_name=metric,
                run_ids=run_ids,
                values=values,
                trend_direction=direction,
                slope=slope,
            ))

        return trends

    def generate_tiered_executive_summary(
        self,
        tiered_gates: list[TieredReleaseGate],
        trends: list[RunTrend] | None = None,
    ) -> str:
        """Extended executive summary with per-tier verdicts and trends.

        Parameters
        ----------
        tiered_gates : list[TieredReleaseGate]
            Gate results from ``generate_tiered_release_gate()``.
        trends : list[RunTrend] or None
            Optional metric trends from ``compute_trends()``.

        Returns
        -------
        str
            Formatted text summary.
        """
        lines = [
            "GMSH Risk-Tiered Release Summary",
            "=" * 50,
            "",
        ]

        for gate in tiered_gates:
            status = "PASS" if gate.passed else "FAIL"
            lines.append(
                f"Tier: {gate.tier.tier_name.upper()} — {status}")

            for check_name, detail in gate.checks.items():
                check_status = "ok" if detail.get("passed") else "BLOCKED"
                lines.append(
                    f"  {check_name}: {detail.get('actual', '?')} "
                    f"(threshold: {detail.get('threshold', '?')}) "
                    f"[{check_status}]")

            if gate.blocking_reasons:
                lines.append("  Blocking:")
                for reason in gate.blocking_reasons:
                    lines.append(f"    - {reason}")
            lines.append("")

        if trends:
            lines.append("Metric Trends:")
            for trend in trends:
                arrow = {"improving": "^", "degrading": "v",
                         "stable": "-"}.get(trend.trend_direction, "?")
                lines.append(
                    f"  {trend.metric_name}: {arrow} "
                    f"(slope={trend.slope:+.4f}, "
                    f"n={len(trend.values)} runs)")
            lines.append("")

        # Overall disposition
        all_pass = all(g.passed for g in tiered_gates)
        if all_pass:
            disposition = "ALL TIERS PASS — release approved."
        else:
            failing = [g.tier.tier_name for g in tiered_gates if not g.passed]
            disposition = (
                f"BLOCKED — failing tiers: {', '.join(failing)}. "
                f"Address blocking issues before release.")
        lines.append(f"Disposition: {disposition}")

        return "\n".join(lines)
