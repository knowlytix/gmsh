# SPDX-License-Identifier: Apache-2.0
"""GMSH Failure Taxonomy — structured failure codes and attribution.

Every failure code names which GMS verification route failed, enabling
systematic root-cause analysis across benchmark runs.

Failure categories map to GMS primitives:

- Graph failures → score_triple(), fuzzy_match_entity()
- Logic failures → tension_energy(), check_holonomy()
- Numeric failures → phase_encoders, check_inequality()
- Artifact failures → ENM read_exact(), SHA-256 integrity
- Policy failures → score_triple() on tool-use relations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCode(str, Enum):
    """Controlled vocabulary for failure classification.

    Each code identifies a specific GMS verification route that failed.
    """

    # -- Content / semantic failures --
    SEMANTIC_EXTRACTION_FAILURE = "semantic_extraction_failure"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    HALLUCINATION_DETECTED = "hallucination_detected"
    INCOMPLETE_ANSWER = "incomplete_answer"
    NO_CLAIMS_EXTRACTED = "no_claims_extracted"

    # -- Graph / structural failures --
    ENTITY_NOT_FOUND = "entity_not_found"
    RELATION_NOT_FOUND = "relation_not_found"
    FACT_INCORRECT = "fact_incorrect"
    PATH_INCONSISTENCY = "path_inconsistency"

    # -- Logic failures (tension + holonomy) --
    CONTRADICTION_DETECTED = "contradiction_detected"
    TENSION_EXCEEDED = "tension_exceeded"
    HOLONOMY_DEFECT = "holonomy_defect"

    # -- Numeric failures (phase encoding) --
    THRESHOLD_VIOLATION = "threshold_violation"
    BOUNDARY_FAILURE = "boundary_failure"

    # -- Exactness failures (ENM) --
    EXACT_MISMATCH = "exact_mismatch"
    ARTIFACT_INTEGRITY_FAILURE = "artifact_integrity_failure"

    # -- Agentic / policy failures --
    TOOL_CALL_INVALID = "tool_call_invalid"
    POLICY_VIOLATION = "policy_violation"
    POLICY_BLOCKED_PRE_EXECUTION = "policy_blocked_pre_execution"
    APPROVAL_REQUIRED = "approval_required"

    # -- Process failures (tool call ordering / completeness) --
    REQUIRED_TOOL_MISSING = "required_tool_missing"
    FORBIDDEN_TOOL_CALLED = "forbidden_tool_called"
    PROCESS_DEVIATION = "process_deviation"
    TOOL_BUDGET_EXCEEDED = "tool_budget_exceeded"

    # -- Recovery failures --
    RECOVERY_FAILURE = "recovery_failure"

    # -- Human interaction failures --
    ESCALATION_INAPPROPRIATE = "escalation_inappropriate"
    CLARIFICATION_INSUFFICIENT = "clarification_insufficient"

    # -- System failures --
    VERIFICATION_ERROR = "verification_error"


class FailureSeverity(str, Enum):
    """Severity level for failure attribution."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class FailureAttribution:
    """Structured failure record linking a claim to a verification outcome.

    Attributes
    ----------
    code : FailureCode
        Which type of failure occurred.
    claim_id : str
        The claim that failed verification.
    verifier : str
        Which verifier produced this failure (e.g. "GraphVerifier").
    severity : FailureSeverity
        Error, warning, or informational.
    detail : str
        Human-readable description of the failure.
    gms_method : str
        The GMS method that produced the failing signal
        (e.g. "score_triple", "tension_energy").
    raw_score : float or None
        The raw geometric score that triggered the failure.
    suggested_action : str
        Recommended remediation step.
    """

    code: FailureCode
    claim_id: str
    verifier: str
    severity: FailureSeverity = FailureSeverity.ERROR
    detail: str = ""
    gms_method: str = ""
    raw_score: float | None = None
    suggested_action: str = ""
    # Structured severity assessment (populated by SeverityClassifier)
    severity_assessment: object | None = None  # SeverityAssessment from severity.py
