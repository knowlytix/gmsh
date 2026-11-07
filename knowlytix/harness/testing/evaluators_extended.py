# SPDX-License-Identifier: Apache-2.0
"""GMSH Extended Evaluators — process, recovery, and interaction verification.

Extends the GMSH verification pipeline with three new claim types and
their corresponding evaluators for agentic testing:

    ProcessClaim      → ProcessEvaluator      → check_holonomy() + query_triples()
    RecoveryClaim     → RecoveryEvaluator      → tension_energy() + fuzzy_match_entity()
    InteractionClaim  → HumanInteractionEvaluator → score_triple() + query_triples()

These evaluators handle the agentic dimension of LLM testing: whether the
agent follows prescribed tool-call processes, recovers gracefully from
injected faults, and produces appropriate human interactions (clarifications,
escalations, confirmations).

GMS primitives are the primary signal throughout.  When entities cannot
be resolved or GMS methods return None, evaluators degrade gracefully
(pass with low confidence) rather than crashing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .claims import ClaimBase
from .failures import FailureAttribution, FailureCode
from .verifiers import SubVerdict

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Helper — extract known GMS entities from free text
# ---------------------------------------------------------------------------

def _extract_entities(
    text: str,
    store: "GMSExpertStore",
    max_entities: int = 10,
) -> list[str]:
    """Extract known GMS entities from free text via fuzzy matching.

    Splits text into candidate tokens and n-grams, attempts
    ``store.fuzzy_match_entity()`` on each, and returns unique matches
    (up to *max_entities*).

    Parameters
    ----------
    text : str
        Free-form text to scan for entity mentions.
    store : GMSExpertStore
        GMS store providing ``fuzzy_match_entity()``.
    max_entities : int
        Maximum number of unique entities to return.

    Returns
    -------
    list[str]
        Resolved GMS entity names found in *text*.
    """
    if not text or not text.strip():
        return []

    # Clean and split into words
    cleaned = re.sub(r"[^\w\s\-]", " ", text)
    words = cleaned.split()

    seen: set[str] = set()
    results: list[str] = []

    # Try bigrams and trigrams first (longer phrases are more specific)
    candidates: list[str] = []
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            candidates.append(" ".join(words[i : i + n]))
    # Then individual words (skip very short tokens)
    candidates.extend(w for w in words if len(w) > 2)

    for candidate in candidates:
        if len(results) >= max_entities:
            break
        match = store.fuzzy_match_entity(candidate)
        if match is not None and match not in seen:
            seen.add(match)
            results.append(match)

    return results


# ---------------------------------------------------------------------------
# New claim types
# ---------------------------------------------------------------------------

@dataclass
class ProcessClaim(ClaimBase):
    """Claim about the agent's tool call process.

    Captures the actual sequence of tool calls made by the SUT, the
    expected sequence, required/forbidden tools, and budget constraints.
    Verified via ProcessEvaluator using holonomy-based process consistency.
    """

    actual_tool_sequence: list[str] = field(default_factory=list)
    expected_tool_sequence: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    max_tool_calls: int = 0
    claim_type: str = "process"


@dataclass
class RecoveryClaim(ClaimBase):
    """Claim about agent recovery from an injected fault.

    Captures the fault scenario (type, which tool was faulted) and
    the agent's output before and after the fault.  Verified via
    RecoveryEvaluator using tension_energy() to measure semantic
    coherence between pre- and post-fault output.
    """

    fault_type: str = ""                # "latency", "error", "stale_data", "mock"
    fault_tool: str = ""                # which tool was faulted
    pre_fault_output: str = ""          # output before fault (or context)
    post_fault_output: str = ""         # output after fault
    recovery_detected: bool = False     # whether agent acknowledged/handled the fault
    claim_type: str = "recovery"


@dataclass
class InteractionClaim(ClaimBase):
    """Claim about human interaction quality.

    Captures whether the agent's clarification, escalation, or
    confirmation was appropriate given the triggering context.
    Verified via HumanInteractionEvaluator using score_triple().
    """

    interaction_type: str = ""          # "clarification", "escalation", "confirmation"
    trigger_context: str = ""           # what triggered the interaction
    response_text: str = ""             # the agent's interaction response
    claim_type: str = "interaction"


# ---------------------------------------------------------------------------
# ProcessEvaluator — tool call ordering and completeness
# ---------------------------------------------------------------------------

class ProcessEvaluator:
    """Evaluate tool call ordering and completeness.

    Uses check_holonomy() to verify that the actual tool call sequence
    is geometrically consistent with the expected process.  Also enforces
    required/forbidden tool constraints and budget limits.

    Verification cascade:
        1. Required tools check — set difference
        2. Forbidden tools check — set intersection
        3. Budget check — len(actual) vs max_tool_calls
        4. Process consistency — check_holonomy() on resolved tool paths
        5. Ordering constraints — query_triples(tool, "must_precede", *)

    GMS primitives used:
        store.fuzzy_match_entity() — resolve tool names to graph entities
        store.check_holonomy()     — process path deviation
        store.query_triples()      — ordering constraint lookup
    """

    HOLONOMY_THRESHOLD = 0.5  # max acceptable process deviation

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def verify(self, claim: ProcessClaim) -> SubVerdict:
        """Verify a ProcessClaim against tool ordering and policy constraints."""
        actual_set = set(claim.actual_tool_sequence)

        # ---- 1. Required tools check ----
        if claim.required_tools:
            missing = set(claim.required_tools) - actual_set
            if missing:
                return SubVerdict(
                    verifier_name="ProcessEvaluator",
                    claim_id=claim.claim_id,
                    passed=False,
                    confidence=0.0,
                    detail=(
                        f"Required tools missing from actual sequence: "
                        f"{sorted(missing)}"
                    ),
                    gms_method_used="set_check",
                    failure=FailureAttribution(
                        code=FailureCode.REQUIRED_TOOL_MISSING,
                        claim_id=claim.claim_id,
                        verifier="ProcessEvaluator",
                        detail=(
                            f"Missing required tools: {sorted(missing)}; "
                            f"actual: {claim.actual_tool_sequence}"
                        ),
                        gms_method="set_check",
                        suggested_action=(
                            "Ensure the agent calls all required tools "
                            "before producing a final answer"
                        ),
                    ),
                )

        # ---- 2. Forbidden tools check ----
        if claim.forbidden_tools:
            violations = set(claim.forbidden_tools) & actual_set
            if violations:
                return SubVerdict(
                    verifier_name="ProcessEvaluator",
                    claim_id=claim.claim_id,
                    passed=False,
                    confidence=0.0,
                    detail=(
                        f"Forbidden tools called: {sorted(violations)}"
                    ),
                    gms_method_used="set_check",
                    failure=FailureAttribution(
                        code=FailureCode.FORBIDDEN_TOOL_CALLED,
                        claim_id=claim.claim_id,
                        verifier="ProcessEvaluator",
                        detail=(
                            f"Forbidden tools used: {sorted(violations)}; "
                            f"actual: {claim.actual_tool_sequence}"
                        ),
                        gms_method="set_check",
                        suggested_action=(
                            "Configure tool access policies to prevent "
                            "the agent from invoking forbidden tools"
                        ),
                    ),
                )

        # ---- 3. Budget check ----
        if claim.max_tool_calls > 0:
            if len(claim.actual_tool_sequence) > claim.max_tool_calls:
                return SubVerdict(
                    verifier_name="ProcessEvaluator",
                    claim_id=claim.claim_id,
                    passed=False,
                    confidence=0.0,
                    detail=(
                        f"Tool budget exceeded: "
                        f"{len(claim.actual_tool_sequence)} calls > "
                        f"max {claim.max_tool_calls}"
                    ),
                    gms_method_used="budget_check",
                    failure=FailureAttribution(
                        code=FailureCode.TOOL_BUDGET_EXCEEDED,
                        claim_id=claim.claim_id,
                        verifier="ProcessEvaluator",
                        detail=(
                            f"Agent made {len(claim.actual_tool_sequence)} "
                            f"tool calls; budget is {claim.max_tool_calls}"
                        ),
                        gms_method="budget_check",
                        raw_score=float(len(claim.actual_tool_sequence)),
                        suggested_action=(
                            "Reduce tool call count or increase budget "
                            "if the task genuinely requires more calls"
                        ),
                    ),
                )

        # ---- 4. Process consistency via holonomy ----
        if claim.expected_tool_sequence:
            return self._check_process_holonomy(claim)

        # No expected sequence provided — structural checks passed
        return SubVerdict(
            verifier_name="ProcessEvaluator",
            claim_id=claim.claim_id,
            passed=True,
            confidence=0.8,
            detail=(
                "Structural checks passed (required/forbidden/budget); "
                "no expected sequence to verify holonomy"
            ),
            gms_method_used="set_check",
        )

    def _check_process_holonomy(self, claim: ProcessClaim) -> SubVerdict:
        """Check process consistency via GMS holonomy and ordering triples."""
        # Resolve actual tool names to GMS entities
        actual_relations: list[str] = []
        unresolved_count = 0
        for tool in claim.actual_tool_sequence:
            resolved = self.store.fuzzy_match_entity(tool)
            if resolved is not None:
                actual_relations.append(resolved)
            else:
                actual_relations.append(tool)
                unresolved_count += 1

        # Resolve expected sequence endpoint as the direct relation
        expected_endpoint = claim.expected_tool_sequence[-1]
        direct_resolved = self.store.fuzzy_match_entity(expected_endpoint)
        direct_relation = direct_resolved if direct_resolved else expected_endpoint

        # Check holonomy: actual path vs expected direct relation
        defect = self.store.check_holonomy(actual_relations, direct_relation)

        # Also check ordering constraints via query_triples
        ordering_violations: list[str] = []
        for i, tool in enumerate(actual_relations):
            hits = self.store.query_triples(head=tool, relation="must_precede")
            if hits:
                required_successors = {t for _, _, t in hits}
                # Check that required successors appear after this tool
                remaining = set(actual_relations[i + 1 :])
                missing_successors = required_successors - remaining
                if missing_successors:
                    ordering_violations.append(
                        f"'{tool}' must_precede {sorted(missing_successors)} "
                        f"but they do not follow"
                    )

        if defect is not None:
            # Holonomy signal available
            deviated = defect > self.HOLONOMY_THRESHOLD

            # Combine holonomy with ordering violations
            if deviated or ordering_violations:
                violation_detail = (
                    f"Holonomy defect={defect:.4f} > "
                    f"threshold={self.HOLONOMY_THRESHOLD}"
                )
                if ordering_violations:
                    violation_detail += (
                        f"; ordering violations: {ordering_violations}"
                    )
                return SubVerdict(
                    verifier_name="ProcessEvaluator",
                    claim_id=claim.claim_id,
                    passed=False,
                    confidence=max(0.0, 1.0 - defect),
                    detail=violation_detail,
                    gms_method_used="check_holonomy",
                    raw_score=defect,
                    failure=FailureAttribution(
                        code=FailureCode.PROCESS_DEVIATION,
                        claim_id=claim.claim_id,
                        verifier="ProcessEvaluator",
                        detail=violation_detail,
                        gms_method="check_holonomy",
                        raw_score=defect,
                        suggested_action=(
                            "Review agent tool call ordering against "
                            "the expected process specification"
                        ),
                    ),
                )

            # Holonomy is within threshold and no ordering violations
            confidence = max(0.0, 1.0 - defect)
            if unresolved_count > 0:
                # Reduce confidence for unresolved entities
                confidence *= max(
                    0.3,
                    1.0 - unresolved_count / max(len(actual_relations), 1),
                )

            return SubVerdict(
                verifier_name="ProcessEvaluator",
                claim_id=claim.claim_id,
                passed=True,
                confidence=confidence,
                detail=(
                    f"Process holonomy defect={defect:.4f} <= "
                    f"threshold={self.HOLONOMY_THRESHOLD}; "
                    f"path={actual_relations}"
                ),
                gms_method_used="check_holonomy",
                raw_score=defect,
            )

        # Holonomy returned None — relations not in trained model
        # Fall back to ordering violations only
        if ordering_violations:
            return SubVerdict(
                verifier_name="ProcessEvaluator",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.3,
                detail=(
                    f"Holonomy not computable; "
                    f"ordering violations: {ordering_violations}"
                ),
                gms_method_used="query_triples",
                failure=FailureAttribution(
                    code=FailureCode.PROCESS_DEVIATION,
                    claim_id=claim.claim_id,
                    verifier="ProcessEvaluator",
                    detail=f"Ordering violations: {ordering_violations}",
                    gms_method="query_triples",
                    suggested_action=(
                        "Fix tool call ordering to satisfy "
                        "must_precede constraints"
                    ),
                ),
            )

        # No holonomy and no ordering violations — degrade gracefully
        confidence = 0.4
        if unresolved_count > 0:
            confidence *= max(
                0.3,
                1.0 - unresolved_count / max(len(actual_relations), 1),
            )

        return SubVerdict(
            verifier_name="ProcessEvaluator",
            claim_id=claim.claim_id,
            passed=True,
            confidence=confidence,
            detail=(
                f"Holonomy not computable (relations not trained); "
                f"no ordering violations detected; "
                f"{unresolved_count}/{len(actual_relations)} tools unresolved"
            ),
            gms_method_used="check_holonomy",
        )


# ---------------------------------------------------------------------------
# RecoveryEvaluator — agent recovery from injected faults
# ---------------------------------------------------------------------------

class RecoveryEvaluator:
    """Evaluate agent recovery from injected faults.

    Uses tension_energy() to measure semantic coherence between
    pre-fault and post-fault agent output.  High tension indicates
    the agent's reasoning diverged after the fault, suggesting
    poor recovery.

    Verification logic:
        1. Extract entity mentions from pre- and post-fault output
        2. Compute pairwise tension_energy() across entity pairs
        3. Mean tension > TENSION_THRESHOLD → RECOVERY_FAILURE
        4. Missing recovery acknowledgement → RECOVERY_FAILURE

    GMS primitives used:
        store.fuzzy_match_entity() — entity extraction from text
        store.tension_energy()     — semantic coherence measurement
    """

    TENSION_THRESHOLD = 1.5  # above this = poor recovery

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def verify(self, claim: RecoveryClaim) -> SubVerdict:
        """Verify a RecoveryClaim via tension energy between outputs."""
        # ---- Check recovery acknowledgement ----
        if claim.fault_type and not claim.recovery_detected:
            return SubVerdict(
                verifier_name="RecoveryEvaluator",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=(
                    f"Fault type '{claim.fault_type}' injected on "
                    f"'{claim.fault_tool}' but agent did not "
                    f"acknowledge or handle the fault"
                ),
                gms_method_used="recovery_check",
                failure=FailureAttribution(
                    code=FailureCode.RECOVERY_FAILURE,
                    claim_id=claim.claim_id,
                    verifier="RecoveryEvaluator",
                    detail=(
                        f"Agent failed to detect/acknowledge "
                        f"'{claim.fault_type}' fault on "
                        f"'{claim.fault_tool}'"
                    ),
                    gms_method="recovery_check",
                    suggested_action=(
                        "Ensure the agent has error handling logic "
                        "that detects and reports tool failures"
                    ),
                ),
            )

        # ---- Extract entities from pre- and post-fault output ----
        pre_entities = _extract_entities(
            claim.pre_fault_output, self.store, max_entities=10
        )
        post_entities = _extract_entities(
            claim.post_fault_output, self.store, max_entities=10
        )

        if not pre_entities and not post_entities:
            # No entities found in either output — degrade gracefully
            return SubVerdict(
                verifier_name="RecoveryEvaluator",
                claim_id=claim.claim_id,
                passed=True,
                confidence=0.2,
                detail=(
                    "No GMS entities found in pre- or post-fault output; "
                    "cannot measure recovery coherence via tension_energy"
                ),
                gms_method_used="fuzzy_match_entity",
            )

        if not pre_entities or not post_entities:
            # Only one side has entities — limited signal
            return SubVerdict(
                verifier_name="RecoveryEvaluator",
                claim_id=claim.claim_id,
                passed=True,
                confidence=0.3,
                detail=(
                    f"Entities found in only one output "
                    f"(pre={len(pre_entities)}, post={len(post_entities)}); "
                    f"insufficient for pairwise tension measurement"
                ),
                gms_method_used="fuzzy_match_entity",
            )

        # ---- Compute pairwise tension energy ----
        tensions: list[float] = []
        for pre_ent in pre_entities:
            for post_ent in post_entities:
                te = self.store.tension_energy(pre_ent, post_ent)
                if te is not None:
                    tensions.append(te)

        if not tensions:
            # tension_energy returned None for all pairs
            return SubVerdict(
                verifier_name="RecoveryEvaluator",
                claim_id=claim.claim_id,
                passed=True,
                confidence=0.2,
                detail=(
                    f"tension_energy() returned None for all "
                    f"{len(pre_entities)}x{len(post_entities)} entity pairs; "
                    f"cannot measure recovery coherence"
                ),
                gms_method_used="tension_energy",
            )

        mean_tension = sum(tensions) / len(tensions)
        confidence = max(0.0, min(1.0, 1.0 - mean_tension / 2.0))

        if mean_tension > self.TENSION_THRESHOLD:
            return SubVerdict(
                verifier_name="RecoveryEvaluator",
                claim_id=claim.claim_id,
                passed=False,
                confidence=confidence,
                detail=(
                    f"Mean tension={mean_tension:.4f} > "
                    f"threshold={self.TENSION_THRESHOLD}; "
                    f"agent reasoning diverged after "
                    f"'{claim.fault_type}' fault on '{claim.fault_tool}' "
                    f"({len(tensions)} entity pairs measured)"
                ),
                gms_method_used="tension_energy",
                raw_score=mean_tension,
                failure=FailureAttribution(
                    code=FailureCode.RECOVERY_FAILURE,
                    claim_id=claim.claim_id,
                    verifier="RecoveryEvaluator",
                    detail=(
                        f"Post-fault output diverged: mean tension "
                        f"{mean_tension:.4f} exceeds {self.TENSION_THRESHOLD}"
                    ),
                    gms_method="tension_energy",
                    raw_score=mean_tension,
                    suggested_action=(
                        "Improve agent recovery logic to maintain "
                        "semantic coherence after tool failures"
                    ),
                ),
            )

        return SubVerdict(
            verifier_name="RecoveryEvaluator",
            claim_id=claim.claim_id,
            passed=True,
            confidence=confidence,
            detail=(
                f"Recovery coherent: mean tension={mean_tension:.4f} <= "
                f"threshold={self.TENSION_THRESHOLD}; "
                f"fault_type='{claim.fault_type}', "
                f"fault_tool='{claim.fault_tool}' "
                f"({len(tensions)} entity pairs measured)"
            ),
            gms_method_used="tension_energy",
            raw_score=mean_tension,
        )


# ---------------------------------------------------------------------------
# HumanInteractionEvaluator — clarification and escalation quality
# ---------------------------------------------------------------------------

class HumanInteractionEvaluator:
    """Evaluate clarification and escalation quality.

    Uses score_triple() to check if the interaction pattern is plausible
    given the triggering context.  For escalations, also verifies the
    context is linked to high-risk concepts via query_triples().

    Verification logic:
        1. Resolve interaction_type and trigger_context to GMS entities
        2. score_triple(interaction, "appropriate_for", context)
        3. For escalations, check high-risk linkage via query_triples()
        4. Fail with ESCALATION_INAPPROPRIATE or CLARIFICATION_INSUFFICIENT

    GMS primitives used:
        store.fuzzy_match_entity() — resolve interaction/context to entities
        store.score_triple()       — plausibility of interaction pattern
        store.query_triples()      — high-risk concept linkage
    """

    PLAUSIBILITY_THRESHOLD = 1.5

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def verify(self, claim: InteractionClaim) -> SubVerdict:
        """Verify an InteractionClaim via geodesic plausibility."""
        # ---- Resolve entities ----
        interaction_entity = self.store.fuzzy_match_entity(
            claim.interaction_type
        )
        context_entity = self.store.fuzzy_match_entity(claim.trigger_context)

        # Also try resolving from the response text for additional signal
        response_entities = _extract_entities(
            claim.response_text, self.store, max_entities=5
        )

        if interaction_entity is None and context_entity is None:
            # Neither side resolved — degrade gracefully
            return SubVerdict(
                verifier_name="HumanInteractionEvaluator",
                claim_id=claim.claim_id,
                passed=True,
                confidence=0.2,
                detail=(
                    f"Cannot resolve interaction_type "
                    f"'{claim.interaction_type}' or trigger_context "
                    f"'{claim.trigger_context}' to GMS entities; "
                    f"skipping geometric verification"
                ),
                gms_method_used="fuzzy_match_entity",
            )

        # ---- Score plausibility ----
        if interaction_entity and context_entity:
            score = self.store.score_triple(
                interaction_entity, "appropriate_for", context_entity
            )

            if score is not None:
                plausible = score < self.PLAUSIBILITY_THRESHOLD

                if not plausible:
                    # Determine failure code based on interaction type
                    failure_code = self._failure_code_for(claim.interaction_type)
                    return SubVerdict(
                        verifier_name="HumanInteractionEvaluator",
                        claim_id=claim.claim_id,
                        passed=False,
                        confidence=max(0.0, min(1.0, 1.0 - score / 3.0)),
                        detail=(
                            f"Interaction '{interaction_entity}' "
                            f"inappropriate for context "
                            f"'{context_entity}': "
                            f"geodesic={score:.4f} > "
                            f"threshold={self.PLAUSIBILITY_THRESHOLD}"
                        ),
                        gms_method_used="score_triple",
                        raw_score=score,
                        failure=FailureAttribution(
                            code=failure_code,
                            claim_id=claim.claim_id,
                            verifier="HumanInteractionEvaluator",
                            detail=(
                                f"({interaction_entity}, appropriate_for, "
                                f"{context_entity}) "
                                f"geodesic={score:.4f} > "
                                f"{self.PLAUSIBILITY_THRESHOLD}"
                            ),
                            gms_method="score_triple",
                            raw_score=score,
                            suggested_action=(
                                f"Review {claim.interaction_type} logic; "
                                f"the interaction does not match the "
                                f"triggering context"
                            ),
                        ),
                    )

                # Plausible — also check escalation-specific constraints
                if claim.interaction_type == "escalation":
                    escalation_verdict = self._check_escalation_risk(
                        claim, context_entity, score
                    )
                    if escalation_verdict is not None:
                        return escalation_verdict

                confidence = max(0.0, min(1.0, 1.0 - score / 3.0))
                return SubVerdict(
                    verifier_name="HumanInteractionEvaluator",
                    claim_id=claim.claim_id,
                    passed=True,
                    confidence=confidence,
                    detail=(
                        f"Interaction '{interaction_entity}' plausible "
                        f"for context '{context_entity}': "
                        f"geodesic={score:.4f}"
                    ),
                    gms_method_used="score_triple",
                    raw_score=score,
                )

            # score_triple returned None — fall back to graph lookup
            graph_hits = self.store.query_triples(
                head=interaction_entity, tail=context_entity
            )
            if graph_hits:
                return SubVerdict(
                    verifier_name="HumanInteractionEvaluator",
                    claim_id=claim.claim_id,
                    passed=True,
                    confidence=0.7,
                    detail=(
                        f"score_triple unavailable; graph lookup found "
                        f"({interaction_entity}, *, {context_entity}) "
                        f"in {len(graph_hits)} triples"
                    ),
                    gms_method_used="query_triples",
                )

        # ---- Partial resolution: one entity found ----
        known = interaction_entity or context_entity
        unknown_label = (
            "trigger_context" if interaction_entity else "interaction_type"
        )
        unknown_value = (
            claim.trigger_context if interaction_entity
            else claim.interaction_type
        )

        # Try graph lookup with the known entity
        graph_hits = self.store.query_triples(head=known)
        if not graph_hits:
            graph_hits = self.store.query_triples(tail=known)

        if graph_hits:
            return SubVerdict(
                verifier_name="HumanInteractionEvaluator",
                claim_id=claim.claim_id,
                passed=True,
                confidence=0.4,
                detail=(
                    f"Partial match: '{known}' found in GMS with "
                    f"{len(graph_hits)} triples; "
                    f"{unknown_label}='{unknown_value}' not resolved"
                ),
                gms_method_used="query_triples",
            )

        # Known entity has no graph triples either
        return SubVerdict(
            verifier_name="HumanInteractionEvaluator",
            claim_id=claim.claim_id,
            passed=True,
            confidence=0.2,
            detail=(
                f"Limited GMS signal: '{known}' resolved but has no "
                f"graph triples; '{unknown_value}' not resolved"
            ),
            gms_method_used="fuzzy_match_entity",
        )

    def _check_escalation_risk(
        self,
        claim: InteractionClaim,
        context_entity: str,
        base_score: float,
    ) -> SubVerdict | None:
        """Additional check for escalations: verify high-risk linkage.

        If the context entity is NOT linked to high-risk concepts in the
        graph, the escalation may be inappropriate (over-escalation).

        Returns a SubVerdict if the escalation is inappropriate, or
        None to let the caller return the normal plausible verdict.
        """
        # Check if context is linked to risk-related concepts
        risk_relations = ("has_risk", "risk_level", "requires_review",
                          "high_risk", "critical")
        has_risk_link = False
        for rel in risk_relations:
            hits = self.store.query_triples(
                head=context_entity, relation=rel
            )
            if hits:
                has_risk_link = True
                break
        if not has_risk_link:
            # Also check reverse direction
            for rel in risk_relations:
                hits = self.store.query_triples(
                    tail=context_entity, relation=rel
                )
                if hits:
                    has_risk_link = True
                    break

        if not has_risk_link:
            # No risk linkage found — escalation may be inappropriate
            # But don't hard-fail; the base score was plausible
            # Only fail if base_score is also borderline
            if base_score > self.PLAUSIBILITY_THRESHOLD * 0.8:
                return SubVerdict(
                    verifier_name="HumanInteractionEvaluator",
                    claim_id=claim.claim_id,
                    passed=False,
                    confidence=max(0.0, min(1.0, 1.0 - base_score / 3.0)),
                    detail=(
                        f"Escalation on '{context_entity}' has no "
                        f"high-risk linkage in GMS and borderline "
                        f"plausibility (geodesic={base_score:.4f})"
                    ),
                    gms_method_used="query_triples",
                    raw_score=base_score,
                    failure=FailureAttribution(
                        code=FailureCode.ESCALATION_INAPPROPRIATE,
                        claim_id=claim.claim_id,
                        verifier="HumanInteractionEvaluator",
                        detail=(
                            f"Context '{context_entity}' not linked to "
                            f"risk concepts; escalation may be unnecessary"
                        ),
                        gms_method="query_triples",
                        raw_score=base_score,
                        suggested_action=(
                            "Review escalation triggers; only escalate "
                            "when the context involves high-risk concepts"
                        ),
                    ),
                )

        # Risk link found or score clearly plausible — no override needed
        return None

    @staticmethod
    def _failure_code_for(interaction_type: str) -> FailureCode:
        """Map interaction type to the appropriate failure code."""
        if interaction_type == "escalation":
            return FailureCode.ESCALATION_INAPPROPRIATE
        elif interaction_type == "clarification":
            return FailureCode.CLARIFICATION_INSUFFICIENT
        else:
            # Default for "confirmation" and other types
            return FailureCode.CLARIFICATION_INSUFFICIENT
