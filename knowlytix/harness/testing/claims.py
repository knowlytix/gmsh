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
    from knowlytix.knowledge.store import GMSExpertStore


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
    # Extraction method:
    #   "rule_based"   — RuleBasedClaimExtractor found the content in llm_answer
    #   "structured"   — StructuredClaimExtractor received typed claim from agent
    #   "llm"          — LLMClaimExtractor parsed content via LLM
    #   "missing"      — schema required, agent did not assert content
    #   "text_match"   — supplementary entity-pair scan of llm_answer
    #   "numeric_match"— supplementary number scan of llm_answer
    #   "metadata"     — legacy ground-truth synthesis (deprecated)
    extraction_method: str = ""


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

    Attributes:
        claim_id: unique claim identifier, auto-generated when empty.
        source_run_id: identifier of the benchmark run that produced the claim.
        source_question_id: identifier of the question the claim answers.
        raw_text: source text fragment the claim was extracted from.
        confidence: extractor confidence that the agent asserted this claim.
        provenance: link back to the source document and extraction method, or None.
        claim_type: claim discriminator, fixed to ``"fact"``.
        head: subject entity of the triple.
        relation: relation of the triple.
        tail: object value the agent asserted for ``(head, relation)``.
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
# Claim Schemas — describe what the agent is expected to assert
# ---------------------------------------------------------------------------

# A ClaimSchema carries the *structure* the benchmark generator expects
# the agent to address, plus optional ground-truth fields used for grading
# and analysis.  Extractors consume schemas together with the agent answer
# and produce typed Claims whose *content* is what the agent actually
# asserted — never the ground truth.

@dataclass
class ClaimSchema:
    """Base schema describing an expected claim slot."""

    schema_type: str = "base"
    expected_claim_type: str = ""


@dataclass
class FactClaimSchema(ClaimSchema):
    """Expected-claim slot for a ``(head, relation, tail)`` fact.

    Attributes:
        head: subject entity the agent is expected to address.
        relation: relation the agent is expected to assert.
        expected_tail: ground-truth tail, used for grading not verification.
        schema_type: schema discriminator, fixed to ``"fact_schema"``.
        expected_claim_type: claim type this schema produces, fixed to ``"fact"``.
    """
    head: str = ""
    relation: str = ""
    expected_tail: str = ""
    schema_type: str = "fact_schema"
    expected_claim_type: str = "fact"


@dataclass
class ThresholdClaimSchema(ClaimSchema):
    entity: str = ""
    metric: str = ""
    operator: str = ""
    limit: float | None = None
    expected_value: float | None = None
    schema_type: str = "threshold_schema"
    expected_claim_type: str = "threshold"


@dataclass
class ArtifactClaimSchema(ClaimSchema):
    category: str = ""
    entity_id: str = ""
    expected_value: float | None = None
    schema_type: str = "artifact_schema"
    expected_claim_type: str = "artifact"


@dataclass
class RuleClaimSchema(ClaimSchema):
    antecedent_entities: list[str] = field(default_factory=list)
    antecedent_relations: list[str] = field(default_factory=list)
    direct_relation: str = ""
    expected_consequent: str = ""
    rule_type: str = "consistency"
    schema_type: str = "rule_schema"
    expected_claim_type: str = "rule"


# ---------------------------------------------------------------------------
# Claim Extractors — pluggable bridge from agent output to typed claims
# ---------------------------------------------------------------------------

# Regex for extracting numbers from text
_NUMBER_RE = re.compile(
    r"(?:^|[\s:=])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
)


class ClaimExtractor:
    """Abstract bridge from agent output to typed Claims.

    An extractor consumes the agent's output (free text, structured dict,
    etc.) together with a list of ClaimSchemas and emits typed Claims.
    The claim's *content* (tail, value, consequent) must reflect what the
    agent asserted; if the agent did not assert it, emit the claim with
    empty content so verifiers report INCOMPLETE_ANSWER.

    Concrete implementations:
      - RuleBasedClaimExtractor: parses free text using store vocabulary
      - StructuredClaimExtractor: expects typed claims from the agent
      - LLMClaimExtractor: delegates parsing to an LLM (stub)
    """

    def extract(
        self,
        agent_output,
        schemas: list[ClaimSchema],
        common: dict,
    ) -> list[ClaimBase]:
        raise NotImplementedError


class RuleBasedClaimExtractor(ClaimExtractor):
    """Default extractor: match store vocabulary + regex against free text.

    For each schema, look in ``agent_output`` (treated as a string) for the
    content field: tail entity for FactClaimSchema, numeric value for
    ThresholdClaimSchema / ArtifactClaimSchema, consequent entity for
    RuleClaimSchema.  If not found, emit the claim with empty content.
    """

    def __init__(self, store: "GMSExpertStore"):
        self.store = store

    def extract(self, agent_output, schemas, common):
        text = agent_output if isinstance(agent_output, str) else ""
        claims: list[ClaimBase] = []
        for schema in schemas:
            if isinstance(schema, FactClaimSchema):
                claims.append(self._extract_fact(text, schema, common))
            elif isinstance(schema, ThresholdClaimSchema):
                claims.append(self._extract_threshold(text, schema, common))
            elif isinstance(schema, ArtifactClaimSchema):
                claims.append(self._extract_artifact(text, schema, common))
            elif isinstance(schema, RuleClaimSchema):
                claims.append(self._extract_rule(text, schema, common))
        return claims

    def _extract_fact(self, text, schema: FactClaimSchema, common):
        tail = self._find_tail(text, schema.head, schema.relation,
                               schema.expected_tail)
        return FactClaim(
            **common,
            head=schema.head,
            relation=schema.relation,
            tail=tail or "",
            confidence=1.0 if tail else 0.0,
            provenance=ClaimProvenance(
                extraction_method="rule_based" if tail else "missing"),
        )

    def _extract_threshold(self, text, schema: ThresholdClaimSchema, common):
        value = self._find_number_for(
            text, anchors=[schema.metric, schema.entity])
        return ThresholdClaim(
            **common,
            entity=schema.entity,
            metric=schema.metric,
            value=value,
            limit=schema.limit,
            operator=schema.operator,
            confidence=1.0 if value is not None else 0.0,
            provenance=ClaimProvenance(
                extraction_method="rule_based"
                if value is not None else "missing"),
        )

    def _extract_artifact(self, text, schema: ArtifactClaimSchema, common):
        value = self._find_number_for(
            text, anchors=[schema.entity_id, schema.category])
        return ArtifactClaim(
            **common,
            category=schema.category,
            entity_id=schema.entity_id,
            claimed_value=value,
            confidence=1.0 if value is not None else 0.0,
            provenance=ClaimProvenance(
                extraction_method="rule_based"
                if value is not None else "missing"),
        )

    def _extract_rule(self, text, schema: RuleClaimSchema, common):
        """Emit a RuleClaim if the agent mentioned every antecedent entity
        (and the consequent, when the schema expects one).

        Contradiction/tension rules have no consequent — we check that
        the agent named both entities in the pair.  Path-consistency
        rules require the consequent plus every hop entity.
        """
        text_lower = text.lower() if text else ""
        antecedents_ok = bool(schema.antecedent_entities) and all(
            self._word_match(e, text_lower)
            for e in schema.antecedent_entities if e
        )
        if schema.expected_consequent:
            consequent = (
                schema.expected_consequent
                if self._word_match(schema.expected_consequent, text_lower)
                else ""
            )
            asserted = antecedents_ok and bool(consequent)
        else:
            consequent = ""
            asserted = antecedents_ok
        return RuleClaim(
            **common,
            antecedent_entities=(list(schema.antecedent_entities)
                                 if asserted else []),
            antecedent_relations=(list(schema.antecedent_relations)
                                  if asserted else []),
            consequent_entity=consequent,
            direct_relation=schema.direct_relation,
            rule_type=schema.rule_type,
            confidence=1.0 if asserted else 0.0,
            provenance=ClaimProvenance(
                extraction_method="rule_based" if asserted else "missing"),
        )

    # ---- helpers ---------------------------------------------------------

    def _find_tail(self, text, head, relation, hint):
        """Find the asserted tail for (head, relation, ?) in text.

        Candidates are drawn from the *tail vocabulary* of ``relation``:
        tails known to sit on ``(head, relation, ?)`` first, then the
        hint (ground truth), then any tail seen for ``relation`` across
        all heads.  The head entity is excluded so a verbose agent that
        echoes "MDL-001 has risk rating High" does not match the head.
        """
        if not text:
            return None
        text_lower = text.lower()
        candidates: list[str] = []
        if self.store.doc_graph:
            for h, r, t in self.store.doc_graph.triples:
                if h == head and r == relation and t not in candidates:
                    candidates.append(t)
        if hint and hint not in candidates:
            candidates.append(hint)
        if self.store.doc_graph:
            for h, r, t in self.store.doc_graph.triples:
                if r == relation and t not in candidates:
                    candidates.append(t)
        candidates = [c for c in candidates if c and c != head]
        # Word-boundary match, longest first (avoid "Low" inside "Lower")
        for cand in sorted(candidates, key=len, reverse=True):
            if len(cand) > 1 and self._word_match(cand, text_lower):
                return cand
        return None

    @staticmethod
    def _word_match(needle: str, haystack_lower: str) -> bool:
        return bool(
            re.search(r"\b" + re.escape(needle.lower()) + r"\b",
                      haystack_lower)
        )

    def _find_number(self, text):
        if not text:
            return None
        match = _NUMBER_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _find_number_for(self, text, anchors):
        """Return the number nearest to any of ``anchors`` in ``text``.

        If no anchor appears in the text, fall back to the first number.
        This disambiguates answers like "ran 5 experiments; VaR is 0.03"
        when the schema asks about VaR.
        """
        if not text:
            return None
        hits = [
            (m.start(), float(m.group(1)))
            for m in _NUMBER_RE.finditer(text)
            if self._is_float(m.group(1))
        ]
        if not hits:
            return None
        text_lower = text.lower()
        anchor_positions = []
        for a in anchors:
            if not a:
                continue
            for m in re.finditer(
                r"\b" + re.escape(a.lower()) + r"\b", text_lower
            ):
                anchor_positions.append(m.start())
        if not anchor_positions:
            return hits[0][1]
        best = min(
            hits,
            key=lambda hv: min(abs(hv[0] - ap) for ap in anchor_positions),
        )
        return best[1]

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False


class GeometricClaimExtractor(ClaimExtractor):
    """Answer-driven extraction: parse the agent's answer into store-bound
    triples with the same geometric machinery as query parsing.

    :class:`RuleBasedClaimExtractor` takes ``head``/``relation`` from the
    generator's schema and only scrapes the tail -- it can only check the facts
    the question was built from. This extractor instead derives the asserted
    triples from the *answer itself*: it reuses
    :class:`~knowlytix.knowledge.rag.query_triples.GeometricQueryParser`'s
    bound vocabulary (entity names + aliases, and the graph adjacency), finds
    every entity the answer names, and for each asserted ``(head, relation,
    tail)`` whose tail the answer also states, emits a :class:`FactClaim`. Every
    claim is real store vocabulary, so ``GraphVerifier.score_triple`` gives a
    meaningful geodesic (an asserted fact scores ~0; a cross-bound tail scores
    far away). Schema-driven numeric/threshold checks still run via the
    compiler's supplementary passes, so this composes rather than replaces.
    """

    def __init__(self, store: "GMSExpertStore", encoder=None):
        from knowlytix.knowledge.rag.query_triples import GeometricQueryParser
        self.store = store
        self._parser = GeometricQueryParser(store, encoder=encoder)

    @staticmethod
    def _mentions_entity(term: str, text_lower: str) -> bool:
        """Word-boundary match, tolerant of snake_case vs spaced surface forms."""
        term = str(term).strip()
        if len(term) <= 1:
            return False
        forms = {term.lower(), term.replace("_", " ").lower()}
        return any(re.search(r"\b" + re.escape(f) + r"\b", text_lower)
                   for f in forms)

    def _mentions_tail(self, tail: str, text_lower: str, text: str) -> bool:
        """A numeric tail matches on value (35.0 == "35"); else word-boundary."""
        t = str(tail).strip()
        try:
            val = float(t)
        except ValueError:
            return self._mentions_entity(t, text_lower)
        for m in re.findall(r"-?\d[\d,]*\.?\d*", text):
            try:
                if abs(float(m.replace(",", "")) - val) < 1e-9:
                    return True
            except ValueError:
                continue
        return False

    def extract(self, agent_output, schemas, common):
        """Emit a :class:`FactClaim` for every store triple the answer names on both ends.

        Args:
            agent_output: the agent's answer; non-string input yields no claims.
            schemas: expected-claim schemas (unused; extraction is answer-driven).
            common: shared claim fields merged into each emitted claim.

        Returns:
            One :class:`FactClaim` per distinct ``(head, relation, tail)`` whose head
            (or an alias) and tail both appear in the answer.
        """
        text = agent_output if isinstance(agent_output, str) else ""
        if not text:
            return []
        low = text.lower()
        claims: list[ClaimBase] = []
        seen: set[tuple[str, str, str]] = set()
        for head, edges in self._parser._adj.items():
            names = [head] + self._parser._aliases.get(head, [])
            if not any(self._mentions_entity(n, low) for n in names):
                continue
            for rel, tail in edges:
                key = (head, rel, str(tail))
                if key in seen or not self._mentions_tail(tail, low, text):
                    continue
                seen.add(key)
                claims.append(FactClaim(
                    **common, head=head, relation=rel, tail=str(tail),
                    confidence=1.0,
                    provenance=ClaimProvenance(extraction_method="geometric"),
                ))
        return claims


class StructuredClaimExtractor(ClaimExtractor):
    """Agent returns typed claim dicts directly (tool-call / JSON contract).

    Expected ``agent_output`` is a list of dicts, each shaped like::

        {"type": "fact", "head": "MDL-001",
         "relation": "has_risk_rating", "tail": "High"}
        {"type": "threshold", "entity": "MDL-001",
         "metric": "var", "value": 0.03, "operator": "lt", "limit": 0.05}
        {"type": "artifact", "category": "shap_sum",
         "entity_id": "MDL-001", "value": 1.23}
        {"type": "rule", "antecedent_entities": [...],
         "antecedent_relations": [...], "consequent": "..."}

    Schemas are used to cross-check that the agent addressed every
    expected slot; missing entries become empty-content claims.
    """

    def extract(self, agent_output, schemas, common):
        claim_dicts = agent_output if isinstance(agent_output, list) else []
        by_type: dict[str, list[dict]] = {}
        for d in claim_dicts:
            if isinstance(d, dict) and "type" in d:
                by_type.setdefault(d["type"], []).append(d)

        claims: list[ClaimBase] = []
        for schema in schemas:
            pool = by_type.get(schema.expected_claim_type, [])
            match = self._match(schema, pool)
            claims.append(self._build(schema, match, common))
        return claims

    def _match(self, schema, pool):
        for d in pool:
            if isinstance(schema, FactClaimSchema):
                if d.get("head") == schema.head and \
                   d.get("relation") == schema.relation:
                    return d
            elif isinstance(schema, ThresholdClaimSchema):
                if d.get("entity") == schema.entity and \
                   d.get("metric") == schema.metric:
                    return d
            elif isinstance(schema, ArtifactClaimSchema):
                if d.get("category") == schema.category and \
                   d.get("entity_id") == schema.entity_id:
                    return d
            elif isinstance(schema, RuleClaimSchema):
                if d.get("direct_relation") == schema.direct_relation:
                    return d
        return None

    def _build(self, schema, match, common):
        asserted = match is not None
        prov = ClaimProvenance(
            extraction_method="structured" if asserted else "missing")
        if isinstance(schema, FactClaimSchema):
            return FactClaim(
                **common, head=schema.head, relation=schema.relation,
                tail=(match or {}).get("tail", "") or "",
                confidence=1.0 if asserted else 0.0, provenance=prov)
        if isinstance(schema, ThresholdClaimSchema):
            val = (match or {}).get("value") if asserted else None
            return ThresholdClaim(
                **common, entity=schema.entity, metric=schema.metric,
                value=float(val) if val is not None else None,
                limit=schema.limit, operator=schema.operator,
                confidence=1.0 if asserted else 0.0, provenance=prov)
        if isinstance(schema, ArtifactClaimSchema):
            val = (match or {}).get("value") if asserted else None
            return ArtifactClaim(
                **common, category=schema.category,
                entity_id=schema.entity_id,
                claimed_value=float(val) if val is not None else None,
                confidence=1.0 if asserted else 0.0, provenance=prov)
        if isinstance(schema, RuleClaimSchema):
            cons = (match or {}).get("consequent", "") if asserted else ""
            return RuleClaim(
                **common,
                antecedent_entities=(
                    list(schema.antecedent_entities) if asserted else []),
                antecedent_relations=(
                    list(schema.antecedent_relations) if asserted else []),
                consequent_entity=cons or "",
                direct_relation=schema.direct_relation,
                rule_type=schema.rule_type,
                confidence=1.0 if asserted else 0.0, provenance=prov)
        return None  # unknown schema


class LLMClaimExtractor(ClaimExtractor):
    """Extract typed claims by prompting an LLM.

    Stub: expects a callable ``llm_callable(prompt) -> str`` that returns
    a JSON-serializable list of claim dicts in the same shape as
    StructuredClaimExtractor.  The default prompt is minimal — replace or
    wrap for production use.
    """

    def __init__(self, llm_callable):
        self.llm = llm_callable
        self._structured = StructuredClaimExtractor()

    def extract(self, agent_output, schemas, common):
        import json
        text = agent_output if isinstance(agent_output, str) else ""
        prompt = self._build_prompt(text, schemas)
        try:
            response = self.llm(prompt)
            parsed = json.loads(response)
        except Exception:
            parsed = []
        return self._structured.extract(parsed, schemas, common)

    def _build_prompt(self, text, schemas):
        slots = [
            {"type": s.expected_claim_type, "schema_type": s.schema_type,
             **{k: v for k, v in s.__dict__.items()
                if k not in ("schema_type", "expected_claim_type")}}
            for s in schemas
        ]
        return (
            "Extract what the agent asserted for each slot below. "
            "Return JSON list of claim dicts (type + content fields). "
            "If the agent did not assert a slot, omit it.\n\n"
            f"SLOTS:\n{slots}\n\nAGENT ANSWER:\n{text}"
        )


class ClaimCompiler:
    """Decompose SUT output into typed claims routable to GMS verifiers.

    The compiler separates three concerns:

    1. **Schema** — what the agent is *expected* to address, derived from
       question metadata (``_schema_from_metadata``).  Schemas carry
       ground-truth fields for grading/analysis but never for verification.

    2. **Content extraction** — what the agent *asserted*, produced by a
       pluggable :class:`ClaimExtractor`.  Swap extractors to adapt to any
       SUT: free-text (default :class:`RuleBasedClaimExtractor`), JSON /
       tool-call (:class:`StructuredClaimExtractor`), or LLM-as-extractor
       (:class:`LLMClaimExtractor`).

    3. **Supplementary extraction** — ``_from_text_match`` and
       ``_from_numeric_match`` scan the agent's free text for additional
       entity pairs or numbers beyond the metadata schema.

    Parameters
    ----------
    store : GMSExpertStore
        GMS store that provides entity/relation vocabularies and ENM.
    truth_store : optional
        Truth store for provenance enrichment.
    semantic_mode : str
        SemanticClaim emission mode: ``"off"``, ``"fallback"``,
        ``"complement"``.
    extractor : ClaimExtractor, optional
        Bridge from agent output to typed claims.  Defaults to
        :class:`RuleBasedClaimExtractor` over the same store.
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        truth_store=None,
        semantic_mode: str = "off",
        extractor: ClaimExtractor | None = None,
    ):
        self.store = store
        self.truth_store = truth_store
        self.semantic_mode = semantic_mode
        self.extractor = extractor or RuleBasedClaimExtractor(store)

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

        Contract: a claim represents what the agent *asserted*, not what
        the ground truth is.  Metadata supplies only the claim *schema*
        (head/relation/metric/entity_id); the claim *content* (tail,
        numeric value, artifact value) must be parsed from ``llm_answer``.
        When the agent did not assert the content, the claim is emitted
        with empty content so verifiers can report INCOMPLETE_ANSWER.

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

        # 1. Build expected-claim schemas from metadata
        schemas = self._schema_from_metadata(meta)

        # 2. Schema-driven extraction: agent output → typed claims
        claims.extend(self.extractor.extract(llm_answer, schemas, common))

        # 3. Supplementary text-match (entity pairs the agent mentioned
        #    beyond the schema)
        claims.extend(self._from_text_match(llm_answer, meta, common))

        # 4. Supplementary numeric-match (ENM values the agent mentioned)
        claims.extend(self._from_numeric_match(llm_answer, meta, common))

        # 4. Enrich with provenance from truth store
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

    # ----- Schema derivation ---------------------------------------------

    def _schema_from_metadata(self, meta: dict) -> list[ClaimSchema]:
        """Translate question metadata into expected-claim schemas.

        Schemas describe what the agent is expected to address.  They
        carry ground-truth fields only for grading / analysis; the
        :class:`ClaimExtractor` is responsible for producing claim
        *content* from the agent's output.
        """
        schemas: list[ClaimSchema] = []
        if not meta:
            return schemas

        head = meta.get("head")
        # Single-hop uses ``relation`` directly; the r1 fallback is reserved
        # for multi-hop where r1 is only the first hop (not head→tail).
        relation = meta.get("relation")
        tail = meta.get("tail") or meta.get("true_tail")

        # Single-hop fact: (head, relation, ?tail).  Skip when the question
        # is multi-hop (presence of ``mid`` or ``mid1``), where r1/r2/r3
        # describe hops rather than a head→tail relation.
        is_multihop = bool(meta.get("mid") or meta.get("mid1"))
        if head and relation and tail and not is_multihop:
            schemas.append(FactClaimSchema(
                head=head, relation=relation, expected_tail=tail,
            ))

        # Multi-hop: (head, r1, mid) + (mid, r2, tail) + rule(head, r_direct, tail)
        mid = meta.get("mid")
        r1 = meta.get("r1")
        r2 = meta.get("r2")
        r_direct = meta.get("r_direct")
        if mid and r1 and r2:
            if head:
                schemas.append(FactClaimSchema(
                    head=head, relation=r1, expected_tail=mid))
            if tail:
                schemas.append(FactClaimSchema(
                    head=mid, relation=r2, expected_tail=tail))
            if head and r_direct:
                schemas.append(RuleClaimSchema(
                    antecedent_entities=[e for e in [head, mid, tail] if e],
                    antecedent_relations=[r1, r2],
                    direct_relation=r_direct,
                    expected_consequent=tail or "",
                    rule_type="consistency",
                ))

        # Workflow-compliance: (head, r1, mid1) + (mid1, r2, mid2) +
        #                      (mid2, r3, tail) + rule over path
        mid1 = meta.get("mid1")
        mid2 = meta.get("mid2")
        r3 = meta.get("r3")
        if mid1 and mid2 and r1 and r2 and r3:
            if head:
                schemas.append(FactClaimSchema(
                    head=head, relation=r1, expected_tail=mid1))
            schemas.append(FactClaimSchema(
                head=mid1, relation=r2, expected_tail=mid2))
            if tail:
                schemas.append(FactClaimSchema(
                    head=mid2, relation=r3, expected_tail=tail))
            entities = [e for e in [head, mid1, mid2, tail] if e]
            if len(entities) >= 3:
                schemas.append(RuleClaimSchema(
                    antecedent_entities=entities,
                    antecedent_relations=[r1, r2, r3],
                    expected_consequent=tail or "",
                    rule_type="consistency",
                ))

        # Generic path given as entities + relations lists
        entities_list = meta.get("entities")
        relations_list = meta.get("relations")
        if (entities_list and relations_list
                and isinstance(entities_list, list)
                and isinstance(relations_list, list)):
            for i, rel in enumerate(relations_list):
                if i < len(entities_list) - 1:
                    schemas.append(FactClaimSchema(
                        head=entities_list[i], relation=rel,
                        expected_tail=entities_list[i + 1]))
            if len(entities_list) >= 3:
                schemas.append(RuleClaimSchema(
                    antecedent_entities=list(entities_list),
                    antecedent_relations=list(relations_list),
                    expected_consequent=entities_list[-1],
                    rule_type="consistency",
                ))

        # Threshold: encoder + operator + limit
        encoder_name = meta.get("encoder") or meta.get("category")
        operator = meta.get("operator") or meta.get("op")
        limit = meta.get("limit") or meta.get("threshold")
        value_gt = meta.get("value")
        if encoder_name and operator:
            schemas.append(ThresholdClaimSchema(
                entity=meta.get("entity", head or ""),
                metric=encoder_name,
                operator=operator,
                limit=float(limit) if limit is not None else None,
                expected_value=(
                    float(value_gt) if value_gt is not None else None),
            ))

        # Artifact (ENM): category + entity_id (no operator, to avoid
        # double-counting with threshold).  Only fall back to category if it
        # actually names an ENM key type in the store — otherwise we synthesize
        # spurious artifact schemas for plausibility/tension/etc. categories.
        enm_type = meta.get("enm_type")
        if not enm_type:
            cat = meta.get("category")
            if cat and self.store is not None and self.store.enm is not None:
                if any(k.type == cat for k in self.store.enm.keys()):
                    enm_type = cat
        entity_id = meta.get("entity_id") or meta.get("id")
        if enm_type and entity_id and not operator:
            schemas.append(ArtifactClaimSchema(
                category=enm_type,
                entity_id=entity_id,
                expected_value=(
                    float(value_gt) if value_gt is not None else None),
            ))

        # Tension pair: agent is expected to identify contradiction
        entity_a = meta.get("entity_a")
        entity_b = meta.get("entity_b")
        tension = meta.get("tension_energy")
        if entity_a and entity_b and tension is not None:
            schemas.append(RuleClaimSchema(
                antecedent_entities=[entity_a, entity_b],
                rule_type="contradiction" if tension > 1.5 else "consistency",
            ))

        # Contradiction features (list of feature names).  A contradiction
        # rule needs a pair for tension_energy; skip single-feature schemas
        # that cannot be verified.
        features = meta.get("features")
        if features and isinstance(features, list) and len(features) >= 2:
            for i, f1 in enumerate(features):
                for f2 in features[i + 1:]:
                    schemas.append(RuleClaimSchema(
                        antecedent_entities=[f1, f2],
                        rule_type="contradiction",
                    ))

        # Procedural workflow: ``start`` + ``steps=[(rel, entity), ...]``.
        # Emit one FactClaim per (cursor, rel, entity) hop and a RuleClaim
        # over the chain.  Without this branch the procedural generator
        # produces zero schemas and its claims silently disappear.
        steps = meta.get("steps")
        start = meta.get("start")
        if start and steps and isinstance(steps, list):
            cursor = start
            chain_entities = [start]
            chain_relations = []
            for step in steps:
                if not (isinstance(step, (list, tuple)) and len(step) == 2):
                    continue
                rel, nxt = step
                if not (rel and nxt):
                    continue
                schemas.append(FactClaimSchema(
                    head=str(cursor), relation=str(rel),
                    expected_tail=str(nxt),
                ))
                chain_relations.append(str(rel))
                chain_entities.append(str(nxt))
                cursor = nxt
            if len(chain_entities) >= 3 and chain_relations:
                schemas.append(RuleClaimSchema(
                    antecedent_entities=chain_entities,
                    antecedent_relations=chain_relations,
                    expected_consequent=chain_entities[-1],
                    rule_type="consistency",
                ))

        # Negative tests (fake_entity / wrong_relation): schema describes
        # the probe; extractor emits an empty-tail claim if the agent does
        # not confirm it (correct refusal → INCOMPLETE_ANSWER).  Downstream
        # grading classifies INCOMPLETE_ANSWER on a negative probe as the
        # desired outcome.
        test_type = meta.get("test_type")
        fake_entity = meta.get("fake_entity")
        wrong_relation = meta.get("wrong_relation")
        if test_type == "non_existent_entity" and fake_entity:
            schemas.append(FactClaimSchema(
                head=fake_entity, relation=relation or "",
                expected_tail=tail or "",
            ))
        elif test_type == "wrong_relation" and wrong_relation:
            schemas.append(FactClaimSchema(
                head=head or "", relation=wrong_relation,
                expected_tail=tail or "",
            ))

        return schemas

    # ----- Text-match extraction ------------------------------------------

    def _from_text_match(
        self, text, meta: dict, common: dict
    ) -> list[ClaimBase]:
        """Scan answer text for entities known to the store.

        For each pair of entities found in the answer, check if a known
        relation exists in the store and emit a FactClaim.

        Migrated from GMSJudge._extract_triples() (judge.py:232-248).
        """
        if not isinstance(text, str) or not text:
            return []
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
        self, text, meta: dict, common: dict
    ) -> list[ClaimBase]:
        """Parse numbers from the answer and match against ENM entries."""
        if not isinstance(text, str) or not text:
            return []
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
