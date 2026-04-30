# SPDX-License-Identifier: Apache-2.0
"""GMSH Claim Compiler — decompose SUT output into typed, verifiable claims.

The Claim Compiler is the bridge between free-form model output and
deterministic GMS verification.  Each claim type maps to a specific
GMS verification route:

- FactClaim      → GraphVerifier   → store.score_triple()
- ThresholdClaim → NumericVerifier  → doc_graph.check_threshold()
- ArtifactClaim  → ArtifactVerifier → store.lookup_enm() + SHA-256
- RuleClaim      → LogicVerifier    → store.check_holonomy(), tension_energy()
- ToolUseClaim   → PolicyVerifier   → store.score_triple() on tool relations
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Claim data models
# ---------------------------------------------------------------------------

@dataclass
class ClaimProvenance:
    """Links a claim back to its source document and section."""

    doc_id: str = ""
    section_tag: str = ""
    span_start: int = -1
    span_end: int = -1
    extraction_method: str = ""  # "metadata", "text_match", "numeric_match"


@dataclass
class ClaimBase:
    """Base class for all typed claims.

    Every claim carries a unique ID, links to the run and question that
    produced it, and records the raw text fragment it was extracted from.
    """

    claim_id: str = ""
    source_run_id: str = ""
    source_question_id: str = ""
    raw_text: str = ""
    confidence: float = 0.0
    provenance: ClaimProvenance | None = None
    claim_type: str = "base"

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = uuid.uuid4().hex[:12]


@dataclass
class FactClaim(ClaimBase):
    """A triple-shaped factual claim: (head, relation, tail).

    Verified via GraphVerifier → store.score_triple().
    """

    head: str = ""
    relation: str = ""
    tail: str = ""
    claim_type: str = "fact"


@dataclass
class ThresholdClaim(ClaimBase):
    """A numeric comparison claim: entity.metric op limit.

    Verified via NumericVerifier → doc_graph.check_threshold().
    """

    entity: str = ""
    metric: str = ""
    value: float | None = None
    limit: float | None = None
    operator: str = ""  # "ge", "gt", "le", "lt"
    claim_type: str = "threshold"


@dataclass
class RuleClaim(ClaimBase):
    """A logical or path-consistency claim.

    Verified via LogicVerifier → store.check_holonomy(), tension_energy().
    """

    antecedent_entities: list[str] = field(default_factory=list)
    antecedent_relations: list[str] = field(default_factory=list)
    consequent_entity: str = ""
    direct_relation: str = ""
    rule_type: str = "consistency"  # "consistency", "contradiction", "implication"
    claim_type: str = "rule"


@dataclass
class ArtifactClaim(ClaimBase):
    """A claim about an exact numeric value from ENM.

    Verified via ArtifactVerifier → store.lookup_enm() + SHA-256.
    """

    category: str = ""
    entity_id: str = ""
    claimed_value: float | None = None
    claim_type: str = "artifact"


@dataclass
class ToolUseClaim(ClaimBase):
    """A claim derived from an agent's tool call.

    Verified via PolicyVerifier → store.score_triple() on tool relations.
    """

    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    result_key: str = ""
    claimed_value: str = ""
    claim_type: str = "tool_use"


@dataclass
class SemanticClaim(ClaimBase):
    """A claim requiring semantic judgment beyond structural verification.

    Verified via SemanticVerifier → LLM-as-judge anchored by GMS evidence.

    Structural verifiers (Graph, Numeric, Artifact, Logic, Policy) handle
    deterministic checks against GMS primitives.  SemanticClaim covers the
    gap: semantic equivalence, explanation quality, completeness, and
    groundedness — evaluated by an LLM judge whose confidence is calibrated
    by GMS geometric signals (geodesic proximity, tension energy, entity
    coverage).
    """

    ground_truth: str = ""
    question_text: str = ""
    context_section: str = ""
    semantic_type: str = ""  # "equivalence", "explanation", "completeness"
    claim_type: str = "semantic"


# ---------------------------------------------------------------------------
# Claim Compiler
# ---------------------------------------------------------------------------

# Regex for extracting numbers from text
_NUMBER_RE = re.compile(
    r"(?:^|[\s:=])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
)


class ClaimCompiler:
    """Decompose SUT output into typed claims routable to GMS verifiers.

    Extraction strategy (in priority order):

    1. **Metadata-guided**: question_metadata contains head/relation/tail,
       encoder/limit/op, enm_type/entity_id, or r1/mid/r2 — emit the
       corresponding typed claim directly.

    2. **Text-match**: scan the answer for entities known to the GMS store
       and emit FactClaims for any matching (head, relation, tail) triples.

    3. **Numeric-match**: parse numbers from the answer and match against
       ENM entries, emitting ArtifactClaims.

    Parameters
    ----------
    store : GMSExpertStore
        The GMS store that provides entity/relation vocabularies and ENM.
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        truth_store=None,
        semantic_mode: str = "off",
    ):
        self.store = store
        self.truth_store = truth_store  # Phase 2: for provenance enrichment
        self.semantic_mode = semantic_mode  # "off", "fallback", "complement"

    def compile(
        self,
        llm_answer: str,
        question_metadata: dict | None = None,
        question_category: str = "",
        run_id: str = "",
        question_id: str = "",
        ground_truth: str = "",
        question_text: str = "",
    ) -> list[ClaimBase]:
        """Extract typed claims from an LLM/agent answer.

        Parameters
        ----------
        llm_answer : str
            Raw text output from the system under test.
        question_metadata : dict or None
            Metadata from the question generator (head, relation, tail, etc.).
        question_category : str
            Generator category (e.g. "plausibility", "tension").
        run_id : str
            Benchmark run identifier.
        question_id : str
            DOE question identifier.
        ground_truth : str
            Expected answer (passed to SemanticClaim for LLM judge).
        question_text : str
            Original question text (passed to SemanticClaim for LLM judge).

        Returns
        -------
        list[ClaimBase]
            Typed claims ready for verification routing.
        """
        claims: list[ClaimBase] = []
        meta = question_metadata or {}

        common = dict(
            source_run_id=run_id,
            source_question_id=question_id,
            raw_text=llm_answer[:500] if llm_answer else "",
        )

        # 1. Metadata-guided extraction
        claims.extend(self._from_metadata(meta, common))

        # 2. Text-match extraction
        claims.extend(self._from_text_match(llm_answer, meta, common))

        # 3. Numeric-match extraction
        claims.extend(self._from_numeric_match(llm_answer, meta, common))

        # 4. Enrich with provenance from truth store (Phase 2)
        if self.truth_store:
            for claim in claims:
                if claim.provenance and claim.provenance.doc_id:
                    continue  # already has provenance
                truth = self.truth_store.get_truth_for_claim(claim)
                if truth:
                    if claim.provenance is None:
                        claim.provenance = ClaimProvenance()
                    claim.provenance.doc_id = truth.doc_id
                    claim.provenance.section_tag = truth.section_tag

        # 5. Emit SemanticClaim for LLM-as-judge verification
        #    - "complement" mode: always (alongside structural claims)
        #    - "fallback" mode: only when no structural claims extracted
        #    Controlled by self.semantic_mode (set via config)
        gt_str = str(ground_truth) if ground_truth else ""
        if gt_str and llm_answer:
            structural_count = len(claims)
            emit_semantic = (
                self.semantic_mode == "complement"
                or (self.semantic_mode == "fallback"
                    and structural_count == 0)
            )
            if emit_semantic:
                claims.append(SemanticClaim(
                    **common,
                    ground_truth=gt_str,
                    question_text=question_text,
                    semantic_type=(
                        "equivalence" if structural_count > 0
                        else "completeness"
                    ),
                    confidence=0.8,
                    provenance=ClaimProvenance(
                        extraction_method="semantic"),
                ))

        # 6. Extended evaluator claims (WS6)
        #    Emitted when scenario metadata includes behavioral requirements
        claims.extend(self._from_process_metadata(meta, common))
        claims.extend(self._from_recovery_metadata(meta, common))
        claims.extend(self._from_interaction_metadata(meta, common, llm_answer))

        return claims

    # ----- Metadata-guided extraction -------------------------------------

    def _from_metadata(
        self, meta: dict, common: dict
    ) -> list[ClaimBase]:
        """Extract claims guided by question metadata."""
        claims: list[ClaimBase] = []

        head = meta.get("head")
        relation = meta.get("relation") or meta.get("r1")
        tail = meta.get("tail") or meta.get("true_tail")

        # FactClaim from (head, relation, tail)
        # Guard: skip when mid1 is present (workflow_compliance uses
        # r1 for the first hop, not for head→tail directly)
        if head and relation and tail and not meta.get("mid1"):
            claims.append(FactClaim(
                **common,
                head=head,
                relation=relation,
                tail=tail,
                confidence=1.0,
                provenance=ClaimProvenance(extraction_method="metadata"),
            ))

        # RuleClaim from multi-hop metadata (r1, mid, r2, r_direct)
        mid = meta.get("mid")
        r1 = meta.get("r1")
        r2 = meta.get("r2")
        r_direct = meta.get("r_direct")
        if mid and r1 and r2:
            # Emit individual hop FactClaims
            if head and mid:
                claims.append(FactClaim(
                    **common,
                    head=head,
                    relation=r1,
                    tail=mid,
                    confidence=1.0,
                    provenance=ClaimProvenance(extraction_method="metadata"),
                ))
            if mid and tail:
                claims.append(FactClaim(
                    **common,
                    head=mid,
                    relation=r2,
                    tail=tail,
                    confidence=1.0,
                    provenance=ClaimProvenance(extraction_method="metadata"),
                ))
            # Emit RuleClaim for path consistency
            if head and r_direct:
                claims.append(RuleClaim(
                    **common,
                    antecedent_entities=[head, mid, tail] if tail else [head, mid],
                    antecedent_relations=[r1, r2],
                    consequent_entity=tail or "",
                    direct_relation=r_direct,
                    rule_type="consistency",
                    confidence=1.0,
                    provenance=ClaimProvenance(extraction_method="metadata"),
                ))

        # ThresholdClaim from encoder metadata
        encoder_name = meta.get("encoder") or meta.get("category")
        operator = meta.get("operator") or meta.get("op")
        limit = meta.get("limit") or meta.get("threshold")
        value = meta.get("value")
        if encoder_name and operator:
            claims.append(ThresholdClaim(
                **common,
                entity=meta.get("entity", head or ""),
                metric=encoder_name,
                value=float(value) if value is not None else None,
                limit=float(limit) if limit is not None else None,
                operator=operator,
                confidence=1.0,
                provenance=ClaimProvenance(extraction_method="metadata"),
            ))

        # ArtifactClaim from ENM metadata
        enm_type = meta.get("enm_type") or meta.get("category")
        entity_id = meta.get("entity_id") or meta.get("id")
        if enm_type and entity_id and not operator:
            # Only emit if not already captured as threshold
            claimed_val = meta.get("value")
            claims.append(ArtifactClaim(
                **common,
                category=enm_type,
                entity_id=entity_id,
                claimed_value=float(claimed_val) if claimed_val is not None else None,
                confidence=1.0,
                provenance=ClaimProvenance(extraction_method="metadata"),
            ))

        # Tension metadata (entity pairs)
        entity_a = meta.get("entity_a")
        entity_b = meta.get("entity_b")
        tension = meta.get("tension_energy")
        if entity_a and entity_b and tension is not None:
            claims.append(RuleClaim(
                **common,
                antecedent_entities=[entity_a, entity_b],
                antecedent_relations=[],
                rule_type="contradiction" if tension > 1.5 else "consistency",
                confidence=1.0,
                provenance=ClaimProvenance(extraction_method="metadata"),
            ))

        # =================================================================
        # Category-specific metadata extraction
        #
        # Generators that produce metadata keys not captured by the generic
        # handlers above.  Each block maps generator-specific keys to the
        # correct claim type so that ALL questions route through GMS.
        # =================================================================

        # -- ContradictionGenerator: features/feature → RuleClaims --------
        # Verified via LogicVerifier → tension_energy() between feature pairs
        features = meta.get("features")
        feature = meta.get("feature")
        if features and isinstance(features, list):
            # Pairwise RuleClaims: each feature pair checked for tension
            for i, f1 in enumerate(features):
                for f2 in features[i + 1:]:
                    claims.append(RuleClaim(
                        **common,
                        antecedent_entities=[f1, f2],
                        antecedent_relations=[],
                        rule_type="contradiction",
                        confidence=1.0,
                        provenance=ClaimProvenance(
                            extraction_method="metadata"),
                    ))
        elif feature and not entity_a:
            # Single-feature contradiction check
            claims.append(RuleClaim(
                **common,
                antecedent_entities=[feature],
                antecedent_relations=[],
                rule_type="contradiction",
                confidence=1.0,
                provenance=ClaimProvenance(extraction_method="metadata"),
            ))

        # -- CountingGenerator: relation + expected_count / unique_heads --
        # Verified via GraphVerifier → query_triples() on sample triple
        expected_count = meta.get("expected_count")
        unique_heads = meta.get("unique_heads")
        count_val = meta.get("count")

        if (relation and not head
                and (expected_count is not None or unique_heads is not None)):
            # Verify the relation exists via a sample triple from the graph
            if self.store.doc_graph:
                for h, r, t in self.store.doc_graph.triples:
                    if r == relation:
                        claims.append(FactClaim(
                            **common,
                            head=h, relation=r, tail=t,
                            confidence=0.8,
                            provenance=ClaimProvenance(
                                extraction_method="metadata"),
                        ))
                        break

        # CountingGenerator: relation + tail filter (no head)
        if (relation and tail and not head
                and expected_count is None and unique_heads is None
                and not meta.get("test_type")):
            if self.store.doc_graph:
                for h, r, t in self.store.doc_graph.triples:
                    if r == relation and t == tail:
                        claims.append(FactClaim(
                            **common,
                            head=h, relation=r, tail=t,
                            confidence=0.8,
                            provenance=ClaimProvenance(
                                extraction_method="metadata"),
                        ))
                        break

        # CountingGenerator: enm_type + count (no entity_id)
        if enm_type and count_val is not None and not entity_id:
            if self.store.enm:
                for key in self.store.enm.keys():
                    if key.type == enm_type:
                        claims.append(ArtifactClaim(
                            **common,
                            category=enm_type,
                            entity_id=key.id,
                            claimed_value=None,
                            confidence=0.8,
                            provenance=ClaimProvenance(
                                extraction_method="metadata"),
                        ))
                        break

        # -- MultiHopGenerator: enm_type + which + relation ---------------
        # Verified via ArtifactVerifier (ENM) + GraphVerifier (relation)
        which = meta.get("which")
        if enm_type and which and relation and not entity_id:
            # ENM aggregate (argmin/argmax) → verify ENM type exists
            if self.store.enm:
                for key in self.store.enm.keys():
                    if key.type == enm_type:
                        claims.append(ArtifactClaim(
                            **common,
                            category=enm_type,
                            entity_id=key.id,
                            claimed_value=None,
                            confidence=0.8,
                            provenance=ClaimProvenance(
                                extraction_method="metadata"),
                        ))
                        break
            # Relation follow → verify relation exists in graph
            if self.store.doc_graph:
                for h, r, t in self.store.doc_graph.triples:
                    if r == relation:
                        claims.append(FactClaim(
                            **common,
                            head=h, relation=r, tail=t,
                            confidence=0.8,
                            provenance=ClaimProvenance(
                                extraction_method="metadata"),
                        ))
                        break

        # MultiHopGenerator: type1 + type2 + which (two ENM types)
        type1 = meta.get("type1")
        type2 = meta.get("type2")
        if type1 and type2 and which:
            for enm_t in (type1, type2):
                if self.store.enm:
                    for key in self.store.enm.keys():
                        if key.type == enm_t:
                            claims.append(ArtifactClaim(
                                **common,
                                category=enm_t,
                                entity_id=key.id,
                                claimed_value=None,
                                confidence=0.8,
                                provenance=ClaimProvenance(
                                    extraction_method="metadata"),
                            ))
                            break

        # -- CrossReferenceGenerator: relation_1 + relation_2 -------------
        # Verified via GraphVerifier → score_triple() per relation
        relation_1 = meta.get("relation_1")
        relation_2 = meta.get("relation_2")
        if relation_1 and relation_2:
            for rel in (relation_1, relation_2):
                if self.store.doc_graph:
                    for h, r, t in self.store.doc_graph.triples:
                        if r == rel:
                            claims.append(FactClaim(
                                **common,
                                head=h, relation=r, tail=t,
                                confidence=0.8,
                                provenance=ClaimProvenance(
                                    extraction_method="metadata"),
                            ))
                            break

        # CrossReferenceGenerator: original_type + resolved_type + entity_id
        original_type = meta.get("original_type")
        resolved_type = meta.get("resolved_type")
        if original_type and resolved_type and entity_id:
            for enm_t in (original_type, resolved_type):
                claims.append(ArtifactClaim(
                    **common,
                    category=enm_t,
                    entity_id=entity_id,
                    claimed_value=None,
                    confidence=0.8,
                    provenance=ClaimProvenance(
                        extraction_method="metadata"),
                ))

        # -- WorkflowComplianceGenerator: mid1/mid2/r3 path --------------
        # Verified via GraphVerifier (per hop) + LogicVerifier (path)
        mid1 = meta.get("mid1")
        mid2 = meta.get("mid2")
        r3 = meta.get("r3")
        if mid1 and mid2 and r1 and r2 and r3:
            # Individual hop FactClaims
            if head and r1 and mid1:
                claims.append(FactClaim(
                    **common,
                    head=head, relation=r1, tail=mid1,
                    confidence=1.0,
                    provenance=ClaimProvenance(
                        extraction_method="metadata"),
                ))
            if mid1 and r2 and mid2:
                claims.append(FactClaim(
                    **common,
                    head=mid1, relation=r2, tail=mid2,
                    confidence=1.0,
                    provenance=ClaimProvenance(
                        extraction_method="metadata"),
                ))
            if mid2 and r3 and tail:
                claims.append(FactClaim(
                    **common,
                    head=mid2, relation=r3, tail=tail,
                    confidence=1.0,
                    provenance=ClaimProvenance(
                        extraction_method="metadata"),
                ))
            # Path consistency RuleClaim
            entities = [e for e in [head, mid1, mid2, tail] if e]
            relations = [r for r in [r1, r2, r3] if r]
            if len(entities) >= 3:
                claims.append(RuleClaim(
                    **common,
                    antecedent_entities=entities,
                    antecedent_relations=relations,
                    consequent_entity=tail or "",
                    rule_type="consistency",
                    confidence=1.0,
                    provenance=ClaimProvenance(
                        extraction_method="metadata"),
                ))

        # WorkflowComplianceGenerator: entities + relations list
        entities_list = meta.get("entities")
        relations_list = meta.get("relations")
        if (entities_list and relations_list
                and isinstance(entities_list, list)
                and isinstance(relations_list, list)):
            # Emit hop FactClaims for consecutive entity pairs
            for i, rel in enumerate(relations_list):
                if i < len(entities_list) - 1:
                    claims.append(FactClaim(
                        **common,
                        head=entities_list[i],
                        relation=rel,
                        tail=entities_list[i + 1],
                        confidence=1.0,
                        provenance=ClaimProvenance(
                            extraction_method="metadata"),
                    ))
            # Path consistency RuleClaim
            if len(entities_list) >= 3:
                claims.append(RuleClaim(
                    **common,
                    antecedent_entities=list(entities_list),
                    antecedent_relations=list(relations_list),
                    consequent_entity=entities_list[-1],
                    rule_type="consistency",
                    confidence=1.0,
                    provenance=ClaimProvenance(
                        extraction_method="metadata"),
                ))

        # -- NegativeGenerator: fake_entity / wrong_relation --------------
        # Verified via GraphVerifier → should correctly report entity/rel
        # not found, confirming the negative test
        test_type = meta.get("test_type")
        fake_entity = meta.get("fake_entity")
        wrong_relation = meta.get("wrong_relation")
        if test_type == "non_existent_entity" and fake_entity:
            claims.append(FactClaim(
                **common,
                head=fake_entity,
                relation=relation or "",
                tail=tail or "",
                confidence=1.0,
                provenance=ClaimProvenance(extraction_method="metadata"),
            ))
        elif test_type == "wrong_relation" and wrong_relation:
            claims.append(FactClaim(
                **common,
                head=head or "",
                relation=wrong_relation,
                tail=tail or "",
                confidence=1.0,
                provenance=ClaimProvenance(extraction_method="metadata"),
            ))

        return claims

    # ----- Text-match extraction ------------------------------------------

    def _from_text_match(
        self, text: str, meta: dict, common: dict
    ) -> list[ClaimBase]:
        """Scan answer text for entities known to the store.

        For each pair of entities found in the answer, check if a known
        relation exists in the store and emit a FactClaim.

        Migrated from GMSJudge._extract_triples() (judge.py:232-248).
        """
        if not self.store.adapter or not self.store.doc_graph:
            return []

        claims: list[ClaimBase] = []
        text_lower = text.lower()

        # Find entities mentioned in the answer
        entities_in_text = []
        for entity in self.store.adapter.entity_to_idx:
            if len(entity) > 2 and entity.lower() in text_lower:
                entities_in_text.append(entity)

        # Skip entities already captured via metadata
        meta_entities = set()
        for key in ("head", "tail", "true_tail", "mid", "entity_a", "entity_b"):
            val = meta.get(key)
            if val:
                meta_entities.add(val.lower())

        # For each pair, check if a known relation exists
        seen = set()
        for i, e1 in enumerate(entities_in_text):
            for e2 in entities_in_text[i + 1:]:
                if (e1, e2) in seen:
                    continue
                # Skip pairs already in metadata
                if (e1.lower() in meta_entities and e2.lower() in meta_entities):
                    continue
                for h, r, t in self.store.doc_graph.triples:
                    if (h == e1 and t == e2) or (h == e2 and t == e1):
                        seen.add((h, t))
                        claims.append(FactClaim(
                            **common,
                            head=h,
                            relation=r,
                            tail=t,
                            confidence=0.6,
                            provenance=ClaimProvenance(
                                extraction_method="text_match"),
                        ))
                        break  # one relation per pair

        return claims

    # ----- Numeric-match extraction ---------------------------------------

    def _from_numeric_match(
        self, text: str, meta: dict, common: dict
    ) -> list[ClaimBase]:
        """Parse numbers from the answer and match against ENM entries."""
        if not self.store.enm:
            return []

        # Skip if metadata already provided artifact info
        if meta.get("enm_type") or meta.get("entity_id"):
            return []

        claims: list[ClaimBase] = []
        numbers = [float(m.group(1)) for m in _NUMBER_RE.finditer(text)]
        if not numbers:
            return []

        # Match against ENM entries
        for key in self.store.enm.keys():
            val = self.store.enm.read_exact(key)
            if val is None:
                continue
            stored = float(val.item() if hasattr(val, "item") else val)
            for num in numbers:
                if abs(num - stored) < 1e-6 * (1 + abs(stored)):
                    claims.append(ArtifactClaim(
                        **common,
                        category=key.type,
                        entity_id=key.id,
                        claimed_value=num,
                        confidence=0.5,
                        provenance=ClaimProvenance(
                            extraction_method="numeric_match"),
                    ))
                    break  # one match per ENM key

        return claims

    # ----- Extended evaluator claim extraction (WS6) ----------------------

    def _from_process_metadata(
        self, meta: dict, common: dict
    ) -> list[ClaimBase]:
        """Emit ProcessClaim when scenario metadata includes tool requirements."""
        if not any(k in meta for k in (
            "required_tools", "forbidden_tools",
            "expected_tool_sequence", "max_tool_calls",
            "actual_tool_sequence",
        )):
            return []

        try:
            from .evaluators_extended import ProcessClaim
        except ImportError:
            return []

        return [ProcessClaim(
            **common,
            actual_tool_sequence=meta.get("actual_tool_sequence", []),
            expected_tool_sequence=meta.get("expected_tool_sequence", []),
            required_tools=meta.get("required_tools", []),
            forbidden_tools=meta.get("forbidden_tools", []),
            max_tool_calls=meta.get("max_tool_calls", 0),
            confidence=1.0,
            provenance=ClaimProvenance(extraction_method="process_metadata"),
        )]

    def _from_recovery_metadata(
        self, meta: dict, common: dict
    ) -> list[ClaimBase]:
        """Emit RecoveryClaim when scenario metadata includes fault info."""
        if "fault_type" not in meta:
            return []

        try:
            from .evaluators_extended import RecoveryClaim
        except ImportError:
            return []

        return [RecoveryClaim(
            **common,
            fault_type=meta.get("fault_type", ""),
            fault_tool=meta.get("fault_tool", ""),
            pre_fault_output=meta.get("pre_fault_output", ""),
            post_fault_output=meta.get("post_fault_output", ""),
            recovery_detected=meta.get("recovery_detected", False),
            confidence=1.0,
            provenance=ClaimProvenance(extraction_method="recovery_metadata"),
        )]

    def _from_interaction_metadata(
        self, meta: dict, common: dict, llm_answer: str
    ) -> list[ClaimBase]:
        """Emit InteractionClaim when scenario metadata includes interaction type."""
        if "interaction_type" not in meta:
            return []

        try:
            from .evaluators_extended import InteractionClaim
        except ImportError:
            return []

        return [InteractionClaim(
            **common,
            interaction_type=meta.get("interaction_type", ""),
            trigger_context=meta.get("trigger_context", ""),
            response_text=llm_answer[:500] if llm_answer else "",
            confidence=1.0,
            provenance=ClaimProvenance(extraction_method="interaction_metadata"),
        )]
