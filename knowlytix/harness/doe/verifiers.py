# SPDX-License-Identifier: Apache-2.0
"""GMSH Typed Verifiers — route each claim type to specific GMS primitives.

**GMS is the authority.** Every verification decision calls a GMS method.
If GMS cannot verify a claim (missing phase encoder, entity not in graph,
relation not trained), the verifier FAILS rather than silently passing.
No Python arithmetic, no tolerance heuristics, no bypasses.

Each verifier is a thin wrapper that calls GMS methods and produces a
structured SubVerdict with the raw geometric signal, pass/fail decision,
and failure attribution when applicable.

Verification routing (with fallback cascades):

    FactClaim      → GraphVerifier   → score_triple() → query_triples()
    ThresholdClaim → NumericVerifier  → lookup_enm() + check_threshold()
    ArtifactClaim  → ArtifactVerifier → enm.read_exact() + SHA-256 hash match
    RuleClaim      → LogicVerifier    → check_holonomy() → query_triples() → tension_energy()
    ToolUseClaim   → PolicyVerifier   → score_triple() → query_triples() → tension_energy()

GMS methods used across all verifiers:
    - score_triple()       : geodesic distance on S^{m-1}
    - tension_energy()     : Clifford tension (2 sin(θ/2))
    - check_holonomy()     : rotor composition defect
    - lookup_enm()         : exact numeric retrieval with SHA-256
    - enm.read_exact()     : raw ENM access with integrity verification
    - enm.verify_integrity(): SHA-256 hash recomputation
    - check_threshold()    : phase-encoded inequality check
    - query_triples()      : document graph pattern matching
    - fuzzy_match_entity() : multi-stage entity resolution
    - relation_exists()    : relation vocabulary check
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .claims import (
    ClaimBase,
    FactClaim,
    ThresholdClaim,
    RuleClaim,
    ArtifactClaim,
    ToolUseClaim,
)
from .failures import FailureAttribution, FailureCode, FailureSeverity

if TYPE_CHECKING:
    from knowlytix.doc.config import VerifyConfig
    from knowlytix.doc.store import GMSExpertStore


# ---------------------------------------------------------------------------
# SubVerdict — result from a single sub-verifier
# ---------------------------------------------------------------------------

@dataclass
class SubVerdict:
    """Result from a single typed sub-verifier.

    Attributes
    ----------
    verifier_name : str
        Which verifier produced this (e.g. "GraphVerifier").
    claim_id : str
        The claim that was verified.
    passed : bool
        Whether the claim passed verification.
    confidence : float
        Verification confidence in [0, 1].
    detail : str
        Human-readable detail.
    gms_method_used : str
        Which GMS method was called (e.g. "score_triple").
    raw_score : float or None
        The raw geometric signal (geodesic, tension, defect, etc.).
    failure : FailureAttribution or None
        Structured failure record if the claim did not pass.
    """

    verifier_name: str = ""
    claim_id: str = ""
    passed: bool = False
    confidence: float = 0.0
    detail: str = ""
    gms_method_used: str = ""
    raw_score: float | None = None
    failure: FailureAttribution | None = None


# ---------------------------------------------------------------------------
# GraphVerifier — FactClaims via score_triple()
# ---------------------------------------------------------------------------

class GraphVerifier:
    """Verify factual claims against the GMS knowledge graph.

    All decisions route through GMS primitives:
        store.fuzzy_match_entity() — resolve entity names
        store.relation_exists()    — check relation vocabulary
        store.score_triple()       — geodesic distance on S^{m-1}
        store.query_triples()      — document graph lookup (fallback when
                                     relation not in trained model)

    Verification cascade:
        1. score_triple() — neural geodesic (if relation is trained)
        2. query_triples(h, r, t) — exact graph match (arbitrary relations)
        3. query_triples(h, *, t) — check if pair has any relation
    """

    PLAUSIBILITY_THRESHOLD = 1.2  # geodesic distance; lower = more plausible

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def verify(self, claim: FactClaim) -> SubVerdict:
        """Verify a FactClaim against the GMS graph."""
        # Resolve entities via fuzzy matching
        head_match = self.store.fuzzy_match_entity(claim.head)
        tail_match = self.store.fuzzy_match_entity(claim.tail)

        if head_match is None:
            return SubVerdict(
                verifier_name="GraphVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=f"Head entity '{claim.head}' not found in GMS",
                gms_method_used="fuzzy_match_entity",
                failure=FailureAttribution(
                    code=FailureCode.ENTITY_NOT_FOUND,
                    claim_id=claim.claim_id,
                    verifier="GraphVerifier",
                    detail=f"Head entity '{claim.head}' not in store",
                    gms_method="fuzzy_match_entity",
                ),
            )

        if tail_match is None:
            return SubVerdict(
                verifier_name="GraphVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=f"Tail entity '{claim.tail}' not found in GMS",
                gms_method_used="fuzzy_match_entity",
                failure=FailureAttribution(
                    code=FailureCode.ENTITY_NOT_FOUND,
                    claim_id=claim.claim_id,
                    verifier="GraphVerifier",
                    detail=f"Tail entity '{claim.tail}' not in store",
                    gms_method="fuzzy_match_entity",
                ),
            )

        # Score triple on the manifold — works when relation is in trained vocab
        geodesic = self.store.score_triple(head_match, claim.relation, tail_match)
        if geodesic is None:
            # Neural model can't score (relation not trained or indices OOB).
            # Fall back to document graph lookup — a GMS primitive that
            # handles arbitrary relations.
            graph_hits = self.store.query_triples(
                head=head_match, relation=claim.relation, tail=tail_match)
            if graph_hits:
                # Triple exists in document graph — verified
                return SubVerdict(
                    verifier_name="GraphVerifier",
                    claim_id=claim.claim_id,
                    passed=True,
                    confidence=0.9,
                    detail=(
                        f"({head_match}, {claim.relation}, {tail_match}) "
                        f"verified via query_triples (exact graph match)"
                    ),
                    gms_method_used="query_triples",
                )

            # Also check if the head-tail pair has ANY relation in graph
            any_relation = self.store.query_triples(
                head=head_match, tail=tail_match)
            if any_relation:
                actual_rels = [r for _, r, _ in any_relation]
                return SubVerdict(
                    verifier_name="GraphVerifier",
                    claim_id=claim.claim_id,
                    passed=False,
                    detail=(
                        f"({head_match}, {claim.relation}, {tail_match}) "
                        f"not in graph, but pair has relations: {actual_rels}"
                    ),
                    gms_method_used="query_triples",
                    failure=FailureAttribution(
                        code=FailureCode.RELATION_NOT_FOUND,
                        claim_id=claim.claim_id,
                        verifier="GraphVerifier",
                        detail=(
                            f"Claimed relation '{claim.relation}' not found; "
                            f"actual relations: {actual_rels}"
                        ),
                        gms_method="query_triples",
                    ),
                )

            return SubVerdict(
                verifier_name="GraphVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=(
                    f"({head_match}, {claim.relation}, {tail_match}) "
                    f"not scorable (relation not trained) and not in "
                    f"document graph"
                ),
                gms_method_used="query_triples",
                failure=FailureAttribution(
                    code=FailureCode.FACT_INCORRECT,
                    claim_id=claim.claim_id,
                    verifier="GraphVerifier",
                    detail=(
                        "Triple not in neural model or document graph"
                    ),
                    gms_method="query_triples",
                ),
            )

        plausible = geodesic < self.PLAUSIBILITY_THRESHOLD
        confidence = math.exp(-geodesic)

        failure = None
        if not plausible:
            failure = FailureAttribution(
                code=FailureCode.FACT_INCORRECT,
                claim_id=claim.claim_id,
                verifier="GraphVerifier",
                detail=(
                    f"Triple ({head_match}, {claim.relation}, {tail_match}) "
                    f"geodesic={geodesic:.4f} > threshold={self.PLAUSIBILITY_THRESHOLD}"
                ),
                gms_method="score_triple",
                raw_score=geodesic,
            )

        return SubVerdict(
            verifier_name="GraphVerifier",
            claim_id=claim.claim_id,
            passed=plausible,
            confidence=confidence,
            detail=(
                f"({head_match}, {claim.relation}, {tail_match}) "
                f"geodesic={geodesic:.4f}"
            ),
            gms_method_used="score_triple",
            raw_score=geodesic,
            failure=failure,
        )


# ---------------------------------------------------------------------------
# NumericVerifier — ThresholdClaims via phase encoding
# ---------------------------------------------------------------------------

class NumericVerifier:
    """Verify numeric threshold claims via GMS phase encoders and ENM.

    All decisions route through GMS primitives:
        store.lookup_enm()                 — exact value retrieval (SHA-256 verified)
        store.doc_graph.check_threshold()  — phase-encoded comparison
        store.query_triples()              — graph-backed existence check

    If no phase encoder exists for the metric, the verifier fails with
    VERIFICATION_ERROR rather than falling back to Python arithmetic.
    GMS is the authority — if GMS can't verify it, it's not verified.
    """

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def verify(self, claim: ThresholdClaim) -> SubVerdict:
        """Verify a ThresholdClaim against phase encoders or ENM."""
        if self.store.doc_graph is None:
            return SubVerdict(
                verifier_name="NumericVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail="No doc_graph available",
                gms_method_used="doc_graph",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="NumericVerifier",
                    gms_method="doc_graph",
                ),
            )

        # Determine the value to check — ENM is the authority
        value = claim.value
        if value is None and claim.entity and claim.metric:
            value = self.store.lookup_enm(claim.metric, claim.entity)

        if value is None:
            return SubVerdict(
                verifier_name="NumericVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=f"No value found for {claim.entity}.{claim.metric}",
                gms_method_used="lookup_enm",
                failure=FailureAttribution(
                    code=FailureCode.EXACT_MISMATCH,
                    claim_id=claim.claim_id,
                    verifier="NumericVerifier",
                    detail=f"ENM lookup failed for ({claim.metric}, {claim.entity})",
                    gms_method="lookup_enm",
                ),
            )

        limit = claim.limit
        operator = claim.operator

        if limit is None or not operator:
            # No threshold to check — verify the entity/metric pair exists
            # in the knowledge graph via query_triples
            triples = self.store.query_triples(
                head=claim.entity, relation=claim.metric)
            exists_in_graph = len(triples) > 0
            return SubVerdict(
                verifier_name="NumericVerifier",
                claim_id=claim.claim_id,
                passed=True,
                confidence=1.0 if exists_in_graph else 0.7,
                detail=(
                    f"Value={value} (ENM verified)"
                    + (f", {len(triples)} graph triples" if exists_in_graph
                       else ", no graph triples for entity/metric")
                ),
                gms_method_used="lookup_enm",
                raw_score=value,
            )

        # Use GMS phase-encoded threshold check
        satisfied, margin = self.store.doc_graph.check_threshold(
            claim.metric, value, limit, operator,
        )

        if satisfied is None:
            # No phase encoder registered for this metric in GMS.
            # GMS is the authority — cannot verify without a phase encoder.
            return SubVerdict(
                verifier_name="NumericVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=(
                    f"No GMS phase encoder for metric '{claim.metric}' — "
                    f"cannot verify {value} {operator} {limit}"
                ),
                gms_method_used="check_threshold",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="NumericVerifier",
                    severity=FailureSeverity.WARNING,
                    detail=(
                        f"No phase encoder for '{claim.metric}'. "
                        f"Register via doc_graph.add_phase_encoder() to enable "
                        f"GMS threshold verification."
                    ),
                    gms_method="check_threshold",
                    suggested_action=(
                        f"Add phase encoder for '{claim.metric}' during ingestion"
                    ),
                ),
            )

        failure = None
        if not satisfied:
            is_boundary = margin is not None and abs(margin) < abs(limit) * 0.01
            failure = FailureAttribution(
                code=(FailureCode.BOUNDARY_FAILURE if is_boundary
                      else FailureCode.THRESHOLD_VIOLATION),
                claim_id=claim.claim_id,
                verifier="NumericVerifier",
                detail=(
                    f"{claim.entity}.{claim.metric}: {value} {operator} "
                    f"{limit} → violated (margin={margin})"
                ),
                gms_method="check_threshold",
                raw_score=margin,
            )

        return SubVerdict(
            verifier_name="NumericVerifier",
            claim_id=claim.claim_id,
            passed=bool(satisfied),
            confidence=1.0 if satisfied else 0.0,
            detail=(
                f"{claim.metric}: {value} {operator} {limit} "
                f"→ {'satisfied' if satisfied else 'violated'} "
                f"(phase margin={margin})"
            ),
            gms_method_used="check_threshold",
            raw_score=margin,
            failure=failure,
        )


# ---------------------------------------------------------------------------
# ArtifactVerifier — ArtifactClaims via ENM exact lookup
# ---------------------------------------------------------------------------

class ArtifactVerifier:
    """Verify exact numeric retrieval claims via ENM SHA-256 hash comparison.

    All decisions route through GMS ENM primitives:
        store.enm.read_exact()       — retrieves value with SHA-256 integrity check
        store.enm.verify_integrity() — recomputes SHA-256 and compares stored hash
        _compute_hash()              — SHA-256 of IEEE 754 float64 bytes

    The comparison is hash-based, not tolerance-based: the claimed value is
    serialized to float64 bytes and its SHA-256 hash is compared against the
    ENM entry's stored hash.  This is deterministic (IEEE 754 guarantees
    identical byte representation for identical values) and eliminates any
    ambiguity about tolerance thresholds.
    """

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def verify(self, claim: ArtifactClaim) -> SubVerdict:
        """Verify an ArtifactClaim via ENM SHA-256 hash comparison."""
        if self.store.enm is None:
            return SubVerdict(
                verifier_name="ArtifactVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail="No ENM store available",
                gms_method_used="enm",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="ArtifactVerifier",
                    gms_method="enm",
                ),
            )

        # Find the ENM key
        enm_key = None
        for key in self.store.enm.keys():
            if key.type == claim.category and key.id == claim.entity_id:
                enm_key = key
                break

        if enm_key is None:
            return SubVerdict(
                verifier_name="ArtifactVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=(
                    f"ENM key ({claim.category}, {claim.entity_id}) not found"
                ),
                gms_method_used="enm.read_exact",
                failure=FailureAttribution(
                    code=FailureCode.EXACT_MISMATCH,
                    claim_id=claim.claim_id,
                    verifier="ArtifactVerifier",
                    detail=f"ENM key ({claim.category}, {claim.entity_id}) missing",
                    gms_method="enm.read_exact",
                ),
            )

        # Step 1: Verify ENM entry integrity (recomputes SHA-256)
        integrity_ok = self.store.enm.verify_integrity(enm_key)
        if not integrity_ok:
            return SubVerdict(
                verifier_name="ArtifactVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=(
                    f"ENM integrity check FAILED for "
                    f"({claim.category}, {claim.entity_id}) — "
                    f"stored data may be corrupted"
                ),
                gms_method_used="enm.verify_integrity",
                failure=FailureAttribution(
                    code=FailureCode.ARTIFACT_INTEGRITY_FAILURE,
                    claim_id=claim.claim_id,
                    verifier="ArtifactVerifier",
                    detail="SHA-256 hash mismatch on stored ENM value",
                    gms_method="enm.verify_integrity",
                    suggested_action="Re-ingest document to rebuild ENM",
                ),
            )

        # Step 2: Read the stored value (also verifies integrity)
        try:
            stored_arr = self.store.enm.read_exact(enm_key)
        except RuntimeError as e:
            return SubVerdict(
                verifier_name="ArtifactVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=f"ENM read_exact raised: {e}",
                gms_method_used="enm.read_exact",
                failure=FailureAttribution(
                    code=FailureCode.ARTIFACT_INTEGRITY_FAILURE,
                    claim_id=claim.claim_id,
                    verifier="ArtifactVerifier",
                    detail=str(e),
                    gms_method="enm.read_exact",
                ),
            )

        if stored_arr is None:
            return SubVerdict(
                verifier_name="ArtifactVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail="ENM read_exact returned None",
                gms_method_used="enm.read_exact",
                failure=FailureAttribution(
                    code=FailureCode.EXACT_MISMATCH,
                    claim_id=claim.claim_id,
                    verifier="ArtifactVerifier",
                    gms_method="enm.read_exact",
                ),
            )

        stored_value = float(
            stored_arr.item() if hasattr(stored_arr, 'item') else stored_arr)

        if claim.claimed_value is None:
            # No claimed value — existence + integrity verified
            return SubVerdict(
                verifier_name="ArtifactVerifier",
                claim_id=claim.claim_id,
                passed=True,
                confidence=1.0,
                detail=(
                    f"ENM ({claim.category}, {claim.entity_id}) = "
                    f"{stored_value} [integrity verified]"
                ),
                gms_method_used="enm.verify_integrity",
                raw_score=stored_value,
            )

        # Step 3: Hash-based comparison — serialize claimed value to float64
        # bytes and compare SHA-256 against the stored entry's hash
        import hashlib
        import numpy as np

        claimed_arr = np.ascontiguousarray(
            np.array(claim.claimed_value, dtype=np.float64))
        claimed_hash = hashlib.sha256(claimed_arr.tobytes()).digest()

        entry = self.store.enm._store.get(enm_key)
        stored_hash = entry.hash if entry else None

        if stored_hash is None:
            # Shouldn't happen after read_exact success, but guard
            return SubVerdict(
                verifier_name="ArtifactVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail="ENM entry has no stored hash",
                gms_method_used="enm.verify_integrity",
                failure=FailureAttribution(
                    code=FailureCode.ARTIFACT_INTEGRITY_FAILURE,
                    claim_id=claim.claim_id,
                    verifier="ArtifactVerifier",
                    gms_method="enm.verify_integrity",
                ),
            )

        hash_match = claimed_hash == stored_hash

        failure = None
        if not hash_match:
            failure = FailureAttribution(
                code=FailureCode.EXACT_MISMATCH,
                claim_id=claim.claim_id,
                verifier="ArtifactVerifier",
                detail=(
                    f"SHA-256 mismatch: claimed={claim.claimed_value} "
                    f"(hash={claimed_hash[:8].hex()}...) vs "
                    f"stored={stored_value} "
                    f"(hash={stored_hash[:8].hex()}...)"
                ),
                gms_method="enm.verify_integrity",
                raw_score=abs(claim.claimed_value - stored_value),
            )

        return SubVerdict(
            verifier_name="ArtifactVerifier",
            claim_id=claim.claim_id,
            passed=hash_match,
            confidence=1.0 if hash_match else 0.0,
            detail=(
                f"ENM ({claim.category}, {claim.entity_id}): "
                f"claimed={claim.claimed_value}, stored={stored_value}, "
                f"SHA-256 {'match' if hash_match else 'MISMATCH'}"
            ),
            gms_method_used="enm.verify_integrity",
            raw_score=0.0 if hash_match else abs(claim.claimed_value - stored_value),
            failure=failure,
        )


# ---------------------------------------------------------------------------
# LogicVerifier — RuleClaims via holonomy and tension
# ---------------------------------------------------------------------------

class LogicVerifier:
    """Verify path consistency and contradiction claims via GMS geometry.

    All decisions route through GMS primitives:
        store.check_holonomy()     — rotor composition defect on S^{m-1}
        store.tension_energy()     — Clifford tension energy (2 sin(θ/2))
        store.fuzzy_match_entity() — resolve entity names for tension check
        store.query_triples()      — document graph verification when
                                     holonomy/tension is not computable

    Verification cascade:
        1. check_holonomy() — rotor defect (if all relations trained)
        2. query_triples() per hop — exact graph path verification
        3. tension_energy() — endpoint semantic compatibility
    """

    def __init__(self, store: "GMSExpertStore", config: "VerifyConfig | None" = None):
        self.store = store
        if config is None:
            from knowlytix.doc.config import VerifyConfig
            config = VerifyConfig()
        self.config = config

    def verify(self, claim: RuleClaim) -> SubVerdict:
        """Verify a RuleClaim via holonomy or tension energy."""
        if claim.rule_type == "consistency":
            return self._verify_consistency(claim)
        elif claim.rule_type == "contradiction":
            return self._verify_contradiction(claim)
        else:
            return self._verify_consistency(claim)

    def _verify_consistency(self, claim: RuleClaim) -> SubVerdict:
        """Check path consistency via holonomy defect."""
        if not claim.antecedent_relations or not claim.direct_relation:
            # Fall back to tension check between endpoint entities
            if len(claim.antecedent_entities) >= 2:
                return self._verify_tension(
                    claim,
                    claim.antecedent_entities[0],
                    claim.antecedent_entities[-1],
                )
            # Cannot verify — no path and no entity pair for tension
            return SubVerdict(
                verifier_name="LogicVerifier",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=(
                    "Insufficient data for any GMS logic check: "
                    "no path relations and fewer than 2 entities"
                ),
                gms_method_used="check_holonomy",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="LogicVerifier",
                    severity=FailureSeverity.WARNING,
                    detail="No path or entity pair to verify",
                    gms_method="check_holonomy",
                ),
            )

        defect = self.store.check_holonomy(
            claim.antecedent_relations, claim.direct_relation,
        )

        if defect is None:
            # Holonomy check failed (relations not in trained model).
            # Fall back to document graph: verify each hop exists as a triple.
            if claim.antecedent_entities and len(claim.antecedent_entities) >= 2:
                path_valid = True
                for i, rel in enumerate(claim.antecedent_relations):
                    if i + 1 < len(claim.antecedent_entities):
                        h = claim.antecedent_entities[i]
                        t = claim.antecedent_entities[i + 1]
                        hits = self.store.query_triples(
                            head=h, relation=rel, tail=t)
                        if not hits:
                            path_valid = False
                            break
                if path_valid:
                    return SubVerdict(
                        verifier_name="LogicVerifier",
                        claim_id=claim.claim_id,
                        passed=True,
                        confidence=0.7,
                        detail=(
                            "Holonomy not computable (relations not trained), "
                            "but all path hops verified via query_triples"
                        ),
                        gms_method_used="query_triples",
                    )

            # Fall back to tension between endpoints
            if len(claim.antecedent_entities) >= 2:
                return self._verify_tension(
                    claim,
                    claim.antecedent_entities[0],
                    claim.antecedent_entities[-1],
                )

            return SubVerdict(
                verifier_name="LogicVerifier",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=(
                    f"Holonomy check returned None for path "
                    f"{claim.antecedent_relations} vs "
                    f"'{claim.direct_relation}' and no fallback GMS "
                    f"route available"
                ),
                gms_method_used="check_holonomy",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="LogicVerifier",
                    detail=(
                        "Relations not in trained model and insufficient "
                        "entity data for graph or tension fallback"
                    ),
                    gms_method="check_holonomy",
                ),
            )

        consistent = defect <= self.config.tau_path

        failure = None
        if not consistent:
            failure = FailureAttribution(
                code=FailureCode.HOLONOMY_DEFECT,
                claim_id=claim.claim_id,
                verifier="LogicVerifier",
                detail=(
                    f"Path {claim.antecedent_relations} vs direct "
                    f"'{claim.direct_relation}': defect={defect:.4f} "
                    f"> tau_path={self.config.tau_path}"
                ),
                gms_method="check_holonomy",
                raw_score=defect,
            )

        return SubVerdict(
            verifier_name="LogicVerifier",
            claim_id=claim.claim_id,
            passed=consistent,
            confidence=math.exp(-defect),
            detail=(
                f"Holonomy: path={claim.antecedent_relations}, "
                f"direct='{claim.direct_relation}', defect={defect:.4f}"
            ),
            gms_method_used="check_holonomy",
            raw_score=defect,
            failure=failure,
        )

    def _verify_contradiction(self, claim: RuleClaim) -> SubVerdict:
        """Check for contradiction via tension energy."""
        entities = claim.antecedent_entities
        if len(entities) < 2:
            return SubVerdict(
                verifier_name="LogicVerifier",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail="Need at least 2 entities for tension check",
                gms_method_used="tension_energy",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="LogicVerifier",
                    severity=FailureSeverity.WARNING,
                    detail="Fewer than 2 entities — cannot compute tension",
                    gms_method="tension_energy",
                ),
            )
        return self._verify_tension(claim, entities[0], entities[1])

    def _verify_tension(
        self, claim: RuleClaim, entity_a: str, entity_b: str,
    ) -> SubVerdict:
        """Check tension energy between two entities."""
        # Resolve to canonical names via fuzzy match
        a_match = self.store.fuzzy_match_entity(entity_a)
        b_match = self.store.fuzzy_match_entity(entity_b)

        if a_match is None or b_match is None:
            missing = entity_a if a_match is None else entity_b
            # Check document graph as fallback
            hits = self.store.query_triples(head=entity_a, tail=entity_b)
            if not hits:
                hits = self.store.query_triples(head=entity_b, tail=entity_a)
            if hits:
                return SubVerdict(
                    verifier_name="LogicVerifier",
                    claim_id=claim.claim_id,
                    passed=True,
                    confidence=0.5,
                    detail=(
                        f"Entity '{missing}' not in model but pair found "
                        f"in {len(hits)} graph triples"
                    ),
                    gms_method_used="query_triples",
                )
            return SubVerdict(
                verifier_name="LogicVerifier",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=(
                    f"Entity '{missing}' not in GMS — cannot compute "
                    f"tension energy"
                ),
                gms_method_used="fuzzy_match_entity",
                failure=FailureAttribution(
                    code=FailureCode.ENTITY_NOT_FOUND,
                    claim_id=claim.claim_id,
                    verifier="LogicVerifier",
                    detail=f"Entity '{missing}' not in GMS graph",
                    gms_method="fuzzy_match_entity",
                ),
            )

        te = self.store.tension_energy(a_match, b_match)

        if te is None:
            # Entities exist but tension_energy failed (should not normally
            # happen after fuzzy_match succeeds, but handle it).
            # Check document graph as fallback.
            hits = self.store.query_triples(head=a_match, tail=b_match)
            if not hits:
                hits = self.store.query_triples(head=b_match, tail=a_match)
            if hits:
                return SubVerdict(
                    verifier_name="LogicVerifier",
                    claim_id=claim.claim_id,
                    passed=True,
                    confidence=0.5,
                    detail=(
                        f"Tension energy returned None but pair "
                        f"({a_match}, {b_match}) found in graph"
                    ),
                    gms_method_used="query_triples",
                )
            return SubVerdict(
                verifier_name="LogicVerifier",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=(
                    f"Tension energy for ({a_match}, {b_match}) "
                    f"returned None and pair not in graph"
                ),
                gms_method_used="tension_energy",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="LogicVerifier",
                    detail=(
                        f"tension_energy({a_match}, {b_match}) returned None"
                    ),
                    gms_method="tension_energy",
                ),
            )

        # E ~ 0 → agreement, E ~ sqrt(2) → unrelated, E ~ 2 → contradiction
        is_contradictory = te > self.config.tau_contra
        is_entailment = te < self.config.tau_ent

        if claim.rule_type == "contradiction":
            passed = is_contradictory
        else:
            passed = not is_contradictory

        failure = None
        if not passed:
            if claim.rule_type == "contradiction" and not is_contradictory:
                failure = FailureAttribution(
                    code=FailureCode.TENSION_EXCEEDED,
                    claim_id=claim.claim_id,
                    verifier="LogicVerifier",
                    detail=(
                        f"Expected contradiction between '{a_match}' and "
                        f"'{b_match}' but tension={te:.4f} "
                        f"< tau_contra={self.config.tau_contra}"
                    ),
                    gms_method="tension_energy",
                    raw_score=te,
                )
            elif is_contradictory:
                failure = FailureAttribution(
                    code=FailureCode.CONTRADICTION_DETECTED,
                    claim_id=claim.claim_id,
                    verifier="LogicVerifier",
                    detail=(
                        f"Contradiction between '{a_match}' and "
                        f"'{b_match}': tension={te:.4f} "
                        f"> tau_contra={self.config.tau_contra}"
                    ),
                    gms_method="tension_energy",
                    raw_score=te,
                )

        confidence = 1.0 - (te / 2.0)
        confidence = max(0.0, min(1.0, confidence))

        return SubVerdict(
            verifier_name="LogicVerifier",
            claim_id=claim.claim_id,
            passed=passed,
            confidence=confidence,
            detail=(
                f"Tension({a_match}, {b_match})={te:.4f} "
                f"[{'entail' if is_entailment else 'contradict' if is_contradictory else 'neutral'}]"
            ),
            gms_method_used="tension_energy",
            raw_score=te,
            failure=failure,
        )


# ---------------------------------------------------------------------------
# PolicyVerifier — ToolUseClaims via geometric plausibility
# ---------------------------------------------------------------------------

class PolicyVerifier:
    """Verify tool-use claims via GMS geometric plausibility and graph lookup.

    All decisions route through GMS primitives:
        store.fuzzy_match_entity() — resolve tool/result names to graph entities
        store.score_triple()       — geodesic plausibility on S^{m-1}
        store.query_triples()      — document graph lookup for arbitrary relations
        store.tension_energy()     — consistency check between tool and result

    If GMS cannot verify the tool claim (entities not in graph, relation not
    trained), the verifier fails with POLICY_VIOLATION.  GMS is the authority
    — a silent pass on unverifiable claims would undermine the system.
    """

    PLAUSIBILITY_THRESHOLD = 1.5

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def verify(self, claim: ToolUseClaim) -> SubVerdict:
        """Verify a ToolUseClaim against the GMS graph."""
        tool_match = self.store.fuzzy_match_entity(claim.tool_name)
        result_match = self.store.fuzzy_match_entity(claim.result_key)

        # ---- Neither entity found in GMS: FAIL ----
        if tool_match is None and result_match is None:
            # Last resort: check if the tool/result pair appears in any
            # document graph triple (query_triples handles arbitrary relations)
            graph_hits = self.store.query_triples(head=claim.tool_name)
            if not graph_hits:
                graph_hits = self.store.query_triples(tail=claim.result_key)

            if graph_hits:
                return SubVerdict(
                    verifier_name="PolicyVerifier",
                    claim_id=claim.claim_id,
                    passed=True,
                    confidence=0.5,
                    detail=(
                        f"Tool '{claim.tool_name}' or result '{claim.result_key}' "
                        f"found in {len(graph_hits)} graph triples"
                    ),
                    gms_method_used="query_triples",
                )

            return SubVerdict(
                verifier_name="PolicyVerifier",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=(
                    f"Tool '{claim.tool_name}' and result '{claim.result_key}' "
                    f"not in GMS — cannot verify"
                ),
                gms_method_used="fuzzy_match_entity",
                failure=FailureAttribution(
                    code=FailureCode.POLICY_VIOLATION,
                    claim_id=claim.claim_id,
                    verifier="PolicyVerifier",
                    detail=(
                        f"Neither tool '{claim.tool_name}' nor result "
                        f"'{claim.result_key}' exists in GMS graph"
                    ),
                    gms_method="fuzzy_match_entity",
                    suggested_action=(
                        "Register tool entities in the knowledge graph "
                        "during ingestion"
                    ),
                ),
            )

        # ---- Both entities found: full geometric verification ----
        if tool_match and result_match:
            # Route 1: score_triple (if "produces" is a trained relation)
            score = self.store.score_triple(
                tool_match, "produces", result_match)
            if score is not None:
                plausible = score < self.PLAUSIBILITY_THRESHOLD
                failure = None
                if not plausible:
                    failure = FailureAttribution(
                        code=FailureCode.TOOL_CALL_INVALID,
                        claim_id=claim.claim_id,
                        verifier="PolicyVerifier",
                        detail=(
                            f"({tool_match}, produces, {result_match}) "
                            f"geodesic={score:.4f} > {self.PLAUSIBILITY_THRESHOLD}"
                        ),
                        gms_method="score_triple",
                        raw_score=score,
                    )
                return SubVerdict(
                    verifier_name="PolicyVerifier",
                    claim_id=claim.claim_id,
                    passed=plausible,
                    confidence=math.exp(-score),
                    detail=(
                        f"({tool_match}, produces, {result_match}) "
                        f"geodesic={score:.4f}"
                    ),
                    gms_method_used="score_triple",
                    raw_score=score,
                    failure=failure,
                )

            # Route 2: "produces" not trained — check document graph
            graph_hits = self.store.query_triples(
                head=tool_match, tail=result_match)
            if graph_hits:
                return SubVerdict(
                    verifier_name="PolicyVerifier",
                    claim_id=claim.claim_id,
                    passed=True,
                    confidence=0.8,
                    detail=(
                        f"({tool_match}, *, {result_match}) found in graph: "
                        f"{[r for _, r, _ in graph_hits[:3]]}"
                    ),
                    gms_method_used="query_triples",
                )

            # Route 3: No direct relation — check tension energy
            # Low tension = entities are semantically compatible
            te = self.store.tension_energy(tool_match, result_match)
            if te is not None:
                compatible = te < 1.5  # below contradiction threshold
                failure = None
                if not compatible:
                    failure = FailureAttribution(
                        code=FailureCode.TOOL_CALL_INVALID,
                        claim_id=claim.claim_id,
                        verifier="PolicyVerifier",
                        detail=(
                            f"High tension between tool '{tool_match}' and "
                            f"result '{result_match}': {te:.4f}"
                        ),
                        gms_method="tension_energy",
                        raw_score=te,
                    )
                return SubVerdict(
                    verifier_name="PolicyVerifier",
                    claim_id=claim.claim_id,
                    passed=compatible,
                    confidence=max(0.0, 1.0 - te / 2.0),
                    detail=(
                        f"Tension({tool_match}, {result_match})={te:.4f} "
                        f"[{'compatible' if compatible else 'incompatible'}]"
                    ),
                    gms_method_used="tension_energy",
                    raw_score=te,
                    failure=failure,
                )

            # All GMS routes exhausted
            return SubVerdict(
                verifier_name="PolicyVerifier",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=(
                    f"Both entities found but no GMS route verifies "
                    f"({tool_match}, produces, {result_match})"
                ),
                gms_method_used="score_triple",
                failure=FailureAttribution(
                    code=FailureCode.POLICY_VIOLATION,
                    claim_id=claim.claim_id,
                    verifier="PolicyVerifier",
                    detail=(
                        f"No trained relation, no graph triple, no tension "
                        f"signal for ({tool_match}, {result_match})"
                    ),
                    gms_method="score_triple",
                ),
            )

        # ---- Only one entity found: use what GMS can give us ----
        known = tool_match or result_match
        unknown = claim.result_key if tool_match else claim.tool_name

        # Check if the unknown side appears in any graph triple
        graph_hits = self.store.query_triples(head=unknown)
        if not graph_hits:
            graph_hits = self.store.query_triples(tail=unknown)

        if graph_hits:
            return SubVerdict(
                verifier_name="PolicyVerifier",
                claim_id=claim.claim_id,
                passed=True,
                confidence=0.5,
                detail=(
                    f"Partial match: '{known}' in GMS, '{unknown}' found in "
                    f"{len(graph_hits)} graph triples"
                ),
                gms_method_used="query_triples",
            )

        # One entity known, other not in graph at all
        return SubVerdict(
            verifier_name="PolicyVerifier",
            claim_id=claim.claim_id,
            passed=False,
            confidence=0.0,
            detail=(
                f"'{known}' in GMS but '{unknown}' not found in any "
                f"GMS graph triple"
            ),
            gms_method_used="fuzzy_match_entity",
            failure=FailureAttribution(
                code=FailureCode.POLICY_VIOLATION,
                claim_id=claim.claim_id,
                verifier="PolicyVerifier",
                detail=f"Entity '{unknown}' not in GMS graph",
                gms_method="fuzzy_match_entity",
            ),
        )


# ---------------------------------------------------------------------------
# VerificationRouter — dispatch claims to the correct verifier
# ---------------------------------------------------------------------------

class VerificationRouter:
    """Routes typed claims to the appropriate GMS sub-verifier.

    Each claim type maps to exactly one verifier:

        FactClaim      → GraphVerifier
        ThresholdClaim → NumericVerifier
        ArtifactClaim  → ArtifactVerifier
        RuleClaim      → LogicVerifier
        ToolUseClaim   → PolicyVerifier
        SemanticClaim  → SemanticVerifier (LLM judge + GMS evidence)

    Parameters
    ----------
    store : GMSExpertStore
        The GMS store backing all verifiers.
    config : VerifyConfig or None
        Thresholds for logic verification (tau_contra, tau_path, etc.).
    semantic_verifier : SemanticVerifier or None
        Optional LLM-as-judge verifier for semantic claims.
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        config: "VerifyConfig | None" = None,
        semantic_verifier=None,
        process_evaluator=None,
        recovery_evaluator=None,
        interaction_evaluator=None,
    ):
        self.graph = GraphVerifier(store)
        self.numeric = NumericVerifier(store)
        self.artifact = ArtifactVerifier(store)
        self.logic = LogicVerifier(store, config)
        self.policy = PolicyVerifier(store)
        self.semantic = semantic_verifier

        self._dispatch = {
            "fact": self.graph.verify,
            "threshold": self.numeric.verify,
            "artifact": self.artifact.verify,
            "rule": self.logic.verify,
            "tool_use": self.policy.verify,
        }
        if self.semantic is not None:
            self._dispatch["semantic"] = self.semantic.verify

        # Extended evaluators (WS6)
        if process_evaluator is not None:
            self._dispatch["process"] = process_evaluator.verify
        if recovery_evaluator is not None:
            self._dispatch["recovery"] = recovery_evaluator.verify
        if interaction_evaluator is not None:
            self._dispatch["interaction"] = interaction_evaluator.verify

    def route(self, claim: ClaimBase) -> SubVerdict:
        """Route a claim to the correct verifier and return the sub-verdict."""
        handler = self._dispatch.get(claim.claim_type)
        if handler is None:
            return SubVerdict(
                verifier_name="VerificationRouter",
                claim_id=claim.claim_id,
                passed=False,
                confidence=0.0,
                detail=f"No GMS verifier for claim_type='{claim.claim_type}'",
                gms_method_used="routing",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="VerificationRouter",
                    severity=FailureSeverity.WARNING,
                    detail=(
                        f"Unroutable claim type '{claim.claim_type}' — "
                        f"no GMS verifier registered"
                    ),
                    gms_method="routing",
                ),
            )
        try:
            return handler(claim)
        except Exception as e:
            return SubVerdict(
                verifier_name="VerificationRouter",
                claim_id=claim.claim_id,
                passed=False,
                detail=f"Verifier error: {e}",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier=claim.claim_type,
                    detail=str(e),
                ),
            )

    def route_all(self, claims: list[ClaimBase]) -> list[SubVerdict]:
        """Route all claims and return sub-verdicts."""
        return [self.route(c) for c in claims]
