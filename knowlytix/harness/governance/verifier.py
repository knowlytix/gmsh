# SPDX-License-Identifier: Apache-2.0
"""ClaimVerifier — typed claim routing for governed agent evaluation.

Each claim type maps to the appropriate GMS primitive:

    ClaimType.FACT       → store.score_triple(head, relation, tail)
    ClaimType.NUMERIC    → store.lookup_enm(category, entity_id)
    ClaimType.THRESHOLD  → store.lookup_enm() + inequality check
    ClaimType.COMPARISON → store.lookup_enm() for both entities, compare
    ClaimType.REASONING  → store.check_holonomy() on the reasoning chain

The verifier reuses existing GMSH infrastructure:
    - GMSExpertStore from knowlytix.knowledge.store for all GMS primitives
    - SubVerdict, FailureAttribution, FailureCode from knowlytix.harness.testing
    - SeverityLevel, SeverityClassifier from knowlytix.harness.testing.severity

Built in Chapter 8 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Claim types
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Five claim types covering data science agent outputs."""

    FACT = "fact"             # e.g., "RF handles categorical features"
    NUMERIC = "numeric"       # e.g., "Test AUC is 0.847"
    THRESHOLD = "threshold"   # e.g., "AUC meets 0.85 minimum"
    COMPARISON = "comparison" # e.g., "GB outperforms RF on accuracy"
    REASONING = "reasoning"   # e.g., "Low accuracy caused by overfitting"


# ---------------------------------------------------------------------------
# Claim data model
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A typed claim extracted from agent output.

    Attributes
    ----------
    type : ClaimType
        Which verification route to use.
    text : str
        Human-readable claim text.
    metadata : dict
        Type-specific fields for verification routing:

        FACT:       head, relation, tail
        NUMERIC:    category, entity_id, claimed_value
        THRESHOLD:  category, entity_id, operator, limit
        COMPARISON: category, entity_a, entity_b, metric, direction
        REASONING:  relation_path (list[str]), direct_relation
    """

    type: ClaimType
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Verdict data models
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerdict:
    """Result from verifying a single claim.

    Attributes
    ----------
    claim : Claim
        The claim that was verified.
    passed : bool
        Whether the claim passed verification.
    score : float
        Continuous confidence in [0, 1].
    severity : str
        One of "info", "warning", "error", "critical".
    evidence : str
        Which GMS primitive was used and what it returned.
    raw_signal : float | None
        The raw geometric signal (geodesic, tension, holonomy defect).
    gms_method : str
        The GMS method that was called.
    """

    claim: Claim
    passed: bool
    score: float
    severity: str
    evidence: str
    raw_signal: float | None = None
    gms_method: str = ""


@dataclass
class OverallVerdict:
    """Aggregate verdict across all claims.

    Attributes
    ----------
    claim_verdicts : list[ClaimVerdict]
        Per-claim results.
    n_passed : int
        Number of claims that passed.
    n_total : int
        Total number of claims verified.
    overall_status : str
        "PASS" if all claims passed, "FAIL" otherwise.
    pass_rate : float
        Fraction of claims that passed.
    severity_distribution : dict[str, int]
        Count of verdicts by severity level.
    """

    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    n_passed: int = 0
    n_total: int = 0
    overall_status: str = "PASS"
    pass_rate: float = 1.0
    severity_distribution: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

# Map (claim_type, passed) to default severity
_SEVERITY_MAP = {
    # Failed claims
    (ClaimType.FACT, False): "error",
    (ClaimType.NUMERIC, False): "error",
    (ClaimType.THRESHOLD, False): "critical",
    (ClaimType.COMPARISON, False): "warning",
    (ClaimType.REASONING, False): "error",
    # Passed claims
    (ClaimType.FACT, True): "info",
    (ClaimType.NUMERIC, True): "info",
    (ClaimType.THRESHOLD, True): "info",
    (ClaimType.COMPARISON, True): "info",
    (ClaimType.REASONING, True): "info",
}


def _classify_severity(claim_type: ClaimType, passed: bool,
                       raw_signal: float | None) -> str:
    """Assign severity using claim type and raw geometric signal."""
    base = _SEVERITY_MAP.get((claim_type, passed), "warning")
    if passed:
        return "info"
    # Escalate based on geometric signal magnitude
    if raw_signal is not None:
        if claim_type == ClaimType.FACT and raw_signal > 2.0:
            return "critical"
        if claim_type == ClaimType.REASONING and raw_signal > 1.0:
            return "critical"
    return base


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """Route typed claims to GMS primitives and produce verdicts.

    Parameters
    ----------
    store : GMSExpertStore
        The GMS store providing geometric verification primitives.
    device : str
        PyTorch device (default "cuda").
    plausibility_threshold : float or None
        Maximum geodesic distance for a plausible triple. No hardcoded
        default -- supply a value from GMSJudge.calibrate() or the
        verifier will refuse to evaluate FACT / REASONING claims.
    numeric_tolerance : float
        Relative tolerance for numeric comparisons (default 0.01).
        Unit-free, not a geometric threshold -- kept as a concrete
        default because it reflects a user-facing precision policy
        rather than a learned cut.
    holonomy_threshold : float or None
        Maximum holonomy defect for consistent reasoning. Same
        rule as ``plausibility_threshold``: no hardcoded
        default; supply from calibration or the verifier refuses the
        REASONING path.
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        device: str = "cuda",
        plausibility_threshold: float | None = None,
        numeric_tolerance: float = 0.01,
        holonomy_threshold: float | None = None,
        numeric_range_index: dict | None = None,
    ):
        self.store = store
        self.device = device
        self.plausibility_threshold = plausibility_threshold
        self.numeric_tolerance = numeric_tolerance
        self.holonomy_threshold = holonomy_threshold
        # Per-category (min, max, n_observations) learned offline from
        # corpus ENM values.  When supplied, numeric verification runs
        # a structural second attestor (range plausibility) alongside
        # the ENM exact-match check.  ENM-PASS + range-FAIL routes to
        # ``review`` so a value that is technically present in ENM but
        # outside the calibration distribution is flagged for human
        # review rather than silently accepted.
        self.numeric_range_index = numeric_range_index or {}

        # Dispatch table: ClaimType -> verification method
        self._dispatch = {
            ClaimType.FACT: self._verify_fact,
            ClaimType.NUMERIC: self._verify_numeric,
            ClaimType.THRESHOLD: self._verify_threshold,
            ClaimType.COMPARISON: self._verify_comparison,
            ClaimType.REASONING: self._verify_reasoning,
        }

    # ----- Public API -------------------------------------------------------

    def verify_claim(self, claim: Claim) -> ClaimVerdict:
        """Verify a single claim by routing to the appropriate GMS primitive."""
        handler = self._dispatch.get(claim.type)
        if handler is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity="error",
                evidence=f"Unknown claim type: {claim.type}",
            )
        return handler(claim)

    def verify_all(self, claims: list[Claim]) -> OverallVerdict:
        """Verify a list of claims and aggregate into an OverallVerdict."""
        verdicts = [self.verify_claim(c) for c in claims]
        n_passed = sum(1 for v in verdicts if v.passed)
        n_total = len(verdicts)

        # Severity distribution
        dist: dict[str, int] = {}
        for v in verdicts:
            dist[v.severity] = dist.get(v.severity, 0) + 1

        return OverallVerdict(
            claim_verdicts=verdicts,
            n_passed=n_passed,
            n_total=n_total,
            overall_status="PASS" if n_passed == n_total else "FAIL",
            pass_rate=n_passed / n_total if n_total > 0 else 1.0,
            severity_distribution=dist,
        )

    # ----- FACT verification ------------------------------------------------

    def _verify_fact(self, claim: Claim) -> ClaimVerdict:
        """Verify a factual claim via score_triple.

        metadata: {head, relation, tail}
        """
        if self.plausibility_threshold is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity="error",
                evidence=(
                    "ClaimVerifier has no calibrated "
                    "plausibility_threshold; run GMSJudge.calibrate() "
                    "and pass the geodesic key at construction."
                ),
            )
        head = claim.metadata.get("head", "")
        relation = claim.metadata.get("relation", "")
        tail = claim.metadata.get("tail", "")

        # Resolve entities
        h_match = self.store.fuzzy_match_entity(head)
        t_match = self.store.fuzzy_match_entity(tail)

        if h_match is None or t_match is None:
            missing = []
            if h_match is None:
                missing.append(f"head '{head}'")
            if t_match is None:
                missing.append(f"tail '{tail}'")
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity=_classify_severity(ClaimType.FACT, False, None),
                evidence=f"Entity not found: {', '.join(missing)}",
                gms_method="fuzzy_match_entity",
            )

        # Two-attestor verification (matches the Lab 10.3§2 pattern
        # for fact verification): the structural attestor
        # (query_triples graph match) is checked first; the
        # geometric attestor (score_triple) is consulted second.
        #
        # Why graph first?  Runtime-extracted facts (e.g. a freshly
        # trained model's metrics) are written to the graph by
        # mem.store_findings during EXECUTING but the embedding has
        # not retrained on them in time for EVALUATING.  Their
        # geodesic distance is therefore high (the embedding is
        # stale), yet the fact is genuinely in the corpus.  Graph
        # match is exact, fast and not affected by training latency.
        # When graph match succeeds, the fact is verified.  When
        # graph match fails, geodesic falls back as the geometric
        # attestor for unseen-but-plausible triples.
        graph_hits = self.store.query_triples(
            head=h_match, relation=relation, tail=t_match,
        )
        if graph_hits:
            geodesic = self.store.score_triple(h_match, relation, t_match)
            geo_str = (
                f", score_triple={geodesic:.4f}"
                if geodesic is not None else ""
            )
            return ClaimVerdict(
                claim=claim,
                passed=True,
                score=0.95,
                severity="info",
                evidence=(
                    f"({h_match}, {relation}, {t_match}) "
                    f"verified via query_triples (exact graph match){geo_str}"
                ),
                raw_signal=geodesic,
                gms_method="query_triples+score_triple",
            )

        # No graph match.  Fall through to geodesic plausibility.
        geodesic = self.store.score_triple(h_match, relation, t_match)
        if geodesic is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity=_classify_severity(ClaimType.FACT, False, None),
                evidence=(
                    f"({h_match}, {relation}, {t_match}) "
                    f"relation not trained and not in graph"
                ),
                gms_method="query_triples",
            )

        plausible = geodesic < self.plausibility_threshold
        confidence = math.exp(-geodesic)

        return ClaimVerdict(
            claim=claim,
            passed=plausible,
            score=confidence,
            severity=_classify_severity(ClaimType.FACT, plausible, geodesic),
            evidence=(
                f"score_triple({h_match}, {relation}, {t_match}) "
                f"= {geodesic:.4f} "
                f"{'< ' if plausible else '>= '}{self.plausibility_threshold} "
                f"(no graph match)"
            ),
            raw_signal=geodesic,
            gms_method="score_triple",
        )

    # ----- NUMERIC verification ---------------------------------------------

    def _verify_numeric(self, claim: Claim) -> ClaimVerdict:
        """Verify a numeric claim via lookup_enm.

        metadata: {category, entity_id, claimed_value}
        """
        category = claim.metadata.get("category", "")
        entity_id = claim.metadata.get("entity_id", "")
        claimed = claim.metadata.get("claimed_value")

        if claimed is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity="warning",
                evidence="No claimed_value in metadata",
                gms_method="lookup_enm",
            )

        stored = self.store.lookup_enm(category, entity_id)
        if stored is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity=_classify_severity(ClaimType.NUMERIC, False, None),
                evidence=(
                    f"lookup_enm({category}, {entity_id}) "
                    f"returned None — entry not in ENM"
                ),
                gms_method="lookup_enm",
            )

        # Compare with tolerance
        diff = abs(claimed - stored)
        rel_diff = diff / (1.0 + abs(stored))
        matched = rel_diff <= self.numeric_tolerance
        confidence = max(0.0, 1.0 - rel_diff * 10)

        # Second attestor: range plausibility against the calibrated
        # per-category bounds (when supplied).  A claim that matches
        # ENM but lies outside the calibration range is flagged for
        # review rather than silently accepted as PASS.
        range_entry = self.numeric_range_index.get(category)
        if range_entry is not None:
            r_min, r_max, _ = range_entry
            in_range = (r_min <= float(claimed) <= r_max)
            range_clause = (
                f", range[{category}]=[{r_min:.4f}, {r_max:.4f}]"
                f"{' (in)' if in_range else ' (out)'}"
            )
            two_attestor_passed = matched and in_range
            if matched and not in_range:
                # ENM agrees but the value is outside the calibration
                # range — demote to review.  We surface this as a
                # severity bump rather than a separate verdict slot
                # so existing audit consumers continue to work.
                severity = "warning"
                evidence_extra = (
                    "; second-attestor (range) DISAGREES "
                    "-- verdict downgraded to review"
                )
            else:
                severity = _classify_severity(
                    ClaimType.NUMERIC, two_attestor_passed, rel_diff,
                )
                evidence_extra = (
                    "; both attestors concur"
                    if two_attestor_passed
                    else "; both attestors reject"
                )
            return ClaimVerdict(
                claim=claim,
                passed=two_attestor_passed,
                score=confidence if two_attestor_passed else max(
                    0.0, confidence * 0.5,
                ),
                severity=severity,
                evidence=(
                    f"lookup_enm({category}, {entity_id}) = {stored:.6f}, "
                    f"claimed = {claimed:.6f}, "
                    f"rel_diff = {rel_diff:.6f} "
                    f"{'<= ' if matched else '> '}{self.numeric_tolerance}"
                    f"{range_clause}{evidence_extra}"
                ),
                raw_signal=rel_diff,
                gms_method="lookup_enm + range_attestor",
            )

        return ClaimVerdict(
            claim=claim,
            passed=matched,
            score=confidence,
            severity=_classify_severity(ClaimType.NUMERIC, matched, rel_diff),
            evidence=(
                f"lookup_enm({category}, {entity_id}) = {stored:.6f}, "
                f"claimed = {claimed:.6f}, "
                f"rel_diff = {rel_diff:.6f} "
                f"{'<= ' if matched else '> '}{self.numeric_tolerance}"
            ),
            raw_signal=rel_diff,
            gms_method="lookup_enm",
        )

    # ----- THRESHOLD verification -------------------------------------------

    def _verify_threshold(self, claim: Claim) -> ClaimVerdict:
        """Verify a threshold claim via lookup_enm + inequality.

        metadata: {category, entity_id, operator, limit}
        operator: "ge", "gt", "le", "lt"
        """
        category = claim.metadata.get("category", "")
        entity_id = claim.metadata.get("entity_id", "")
        operator = claim.metadata.get("operator", "ge")
        limit = claim.metadata.get("limit")

        if limit is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity="warning",
                evidence="No limit in metadata",
                gms_method="lookup_enm",
            )

        stored = self.store.lookup_enm(category, entity_id)
        if stored is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity=_classify_severity(
                    ClaimType.THRESHOLD, False, None),
                evidence=(
                    f"lookup_enm({category}, {entity_id}) "
                    f"returned None — entry not in ENM"
                ),
                gms_method="lookup_enm",
            )

        # Apply inequality
        ops = {
            "ge": stored >= limit,
            "gt": stored > limit,
            "le": stored <= limit,
            "lt": stored < limit,
        }
        passed = ops.get(operator, False)
        # Confidence: how far above/below the threshold
        margin = stored - limit
        if operator in ("le", "lt"):
            margin = -margin
        confidence = min(1.0, max(0.0, 0.5 + margin * 5))

        return ClaimVerdict(
            claim=claim,
            passed=passed,
            score=confidence,
            severity=_classify_severity(
                ClaimType.THRESHOLD, passed, abs(margin)),
            evidence=(
                f"lookup_enm({category}, {entity_id}) = {stored:.6f} "
                f"{operator} {limit:.6f} -> {'PASS' if passed else 'FAIL'} "
                f"(margin = {margin:+.6f})"
            ),
            raw_signal=margin,
            gms_method="lookup_enm",
        )

    # ----- COMPARISON verification ------------------------------------------

    def _verify_comparison(self, claim: Claim) -> ClaimVerdict:
        """Verify a comparison claim via lookup_enm for both entities.

        metadata: {category, entity_a, entity_b, metric, direction}
        direction: "greater" means entity_a > entity_b
        """
        category = claim.metadata.get("category", "")
        entity_a = claim.metadata.get("entity_a", "")
        entity_b = claim.metadata.get("entity_b", "")
        metric = claim.metadata.get("metric", category)
        direction = claim.metadata.get("direction", "greater")

        val_a = self.store.lookup_enm(metric, entity_a)
        val_b = self.store.lookup_enm(metric, entity_b)

        if val_a is None or val_b is None:
            missing = []
            if val_a is None:
                missing.append(f"({metric}, {entity_a})")
            if val_b is None:
                missing.append(f"({metric}, {entity_b})")
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity=_classify_severity(
                    ClaimType.COMPARISON, False, None),
                evidence=(
                    f"ENM lookup failed: {', '.join(missing)} not found"
                ),
                gms_method="lookup_enm",
            )

        # Compare
        if direction == "greater":
            passed = val_a > val_b
        elif direction == "less":
            passed = val_a < val_b
        else:
            passed = val_a > val_b  # default: "greater"

        diff = val_a - val_b
        confidence = min(1.0, max(0.0, 0.5 + abs(diff) * 5))

        return ClaimVerdict(
            claim=claim,
            passed=passed,
            score=confidence,
            severity=_classify_severity(
                ClaimType.COMPARISON, passed, abs(diff)),
            evidence=(
                f"lookup_enm({metric}, {entity_a}) = {val_a:.6f}, "
                f"lookup_enm({metric}, {entity_b}) = {val_b:.6f}, "
                f"diff = {diff:+.6f}, direction = {direction} "
                f"-> {'PASS' if passed else 'FAIL'}"
            ),
            raw_signal=diff,
            gms_method="lookup_enm",
        )

    # ----- REASONING verification -------------------------------------------

    def _verify_reasoning(self, claim: Claim) -> ClaimVerdict:
        """Verify a reasoning chain via check_holonomy.

        metadata: {relation_path (list[str]), direct_relation}
        """
        if self.holonomy_threshold is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity="error",
                evidence=(
                    "ClaimVerifier has no calibrated "
                    "holonomy_threshold; run GMSJudge.calibrate() and "
                    "pass the holonomy key at construction."
                ),
            )
        relation_path = claim.metadata.get("relation_path", [])
        direct_relation = claim.metadata.get("direct_relation", "")

        if not relation_path or not direct_relation:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity="warning",
                evidence="Missing relation_path or direct_relation",
                gms_method="check_holonomy",
            )

        defect = self.store.check_holonomy(relation_path, direct_relation)
        if defect is None:
            return ClaimVerdict(
                claim=claim,
                passed=False,
                score=0.0,
                severity=_classify_severity(
                    ClaimType.REASONING, False, None),
                evidence=(
                    f"check_holonomy({relation_path}, {direct_relation}) "
                    f"returned None — relations not in model"
                ),
                gms_method="check_holonomy",
            )

        consistent = defect <= self.holonomy_threshold
        confidence = math.exp(-defect)

        return ClaimVerdict(
            claim=claim,
            passed=consistent,
            score=confidence,
            severity=_classify_severity(
                ClaimType.REASONING, consistent, defect),
            evidence=(
                f"check_holonomy({relation_path}, {direct_relation}) "
                f"= {defect:.4f} "
                f"{'<= ' if consistent else '> '}{self.holonomy_threshold}"
            ),
            raw_signal=defect,
            gms_method="check_holonomy",
        )

    # ----- Tension energy (used by exercises) -------------------------------

    def check_tension(self, entity_a: str, entity_b: str) -> float | None:
        """Expose tension_energy for self-consistency exercises.

        Returns tension energy (0 = agree, sqrt(2) = irrelevant, 2 = contradict).
        """
        return self.store.tension_energy(entity_a, entity_b)

    # ----- Pretty printing --------------------------------------------------

    @staticmethod
    def format_verdict(verdict: ClaimVerdict) -> str:
        """Format a single claim verdict for display."""
        status = "PASS" if verdict.passed else "FAIL"
        icon = "+" if verdict.passed else "X"
        return (
            f"  [{icon}] {verdict.claim.type.value:12s} | "
            f"{status:4s} | score={verdict.score:.3f} | "
            f"severity={verdict.severity:8s} | "
            f"{verdict.claim.text}\n"
            f"      evidence: {verdict.evidence}"
        )

    @staticmethod
    def format_overall(overall: OverallVerdict) -> str:
        """Format the overall verdict summary."""
        lines = [
            f"=== Overall Verdict: {overall.overall_status} ===",
            f"  Passed: {overall.n_passed}/{overall.n_total} "
            f"({overall.pass_rate:.1%})",
            f"  Severity distribution: {overall.severity_distribution}",
        ]
        return "\n".join(lines)
