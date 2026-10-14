# SPDX-License-Identifier: Apache-2.0
"""GMSH Semantic Verifier — LLM-as-judge anchored by GMS geometric evidence.

The SemanticVerifier handles verification that structural verifiers cannot:
semantic equivalence, explanation quality, completeness, and groundedness.

Design principle: **GMS anchors every judgment.**  Before the LLM judge
evaluates a claim, we gather all available GMS structural evidence
(related triples, geodesic scores, tension energy).  This evidence is
injected into the judge prompt so the LLM makes informed decisions.
After the LLM returns its verdict, we calibrate its confidence using
GMS geometric scores.

GMS methods used:
    store.fuzzy_match_entity()   — resolve answer entities to graph
    store.query_triples()        — gather related facts for context
    store.score_triple()         — geodesic proximity between entities
    store.tension_energy()       — contradiction signal between claim/truth
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .claims import SemanticClaim
from .failures import FailureAttribution, FailureCode

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# GMS Evidence bundle
# ---------------------------------------------------------------------------

@dataclass
class GMSEvidence:
    """Structural evidence gathered from GMS for semantic judging."""

    # Entities found in the answer text: raw_text → matched GMS entity
    answer_entities: dict[str, str] = field(default_factory=dict)
    # Entities found in the ground truth: raw_text → matched GMS entity
    truth_entities: dict[str, str] = field(default_factory=dict)
    # Related triples per matched entity: entity → list[(h, r, t)]
    entity_triples: dict[str, list[tuple]] = field(default_factory=dict)
    # Geodesic scores between answer/truth entity pairs: "a↔b" → score
    geodesic_scores: dict[str, float] = field(default_factory=dict)
    # Tension energy between answer/truth entity pairs: "a↔b" → tension
    tension_scores: dict[str, float] = field(default_factory=dict)
    # Fraction of answer entities found in GMS
    entity_coverage: float = 0.0


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

@dataclass
class RubricDimension:
    """A single evaluation dimension for the LLM judge."""
    name: str
    description: str
    weight: float = 1.0


DEFAULT_RUBRIC: list[RubricDimension] = [
    RubricDimension(
        name="factual_accuracy",
        description=(
            "Is the answer factually correct based on the source document? "
            "Use the GMS evidence triples to verify specific facts. "
            "Score lower if claims contradict GMS evidence."
        ),
        weight=2.0,
    ),
    RubricDimension(
        name="completeness",
        description=(
            "Does the answer address all key aspects of the question? "
            "Check whether relevant entities from the GMS evidence "
            "are covered in the answer."
        ),
        weight=1.5,
    ),
    RubricDimension(
        name="groundedness",
        description=(
            "Is the answer grounded in the source material, or does it "
            "contain hallucinated information?  Flag any claims that are "
            "NOT supported by the GMS evidence triples."
        ),
        weight=2.0,
    ),
    RubricDimension(
        name="coherence",
        description=(
            "Is the answer internally consistent and logically structured? "
            "High tension scores between entities suggest potential "
            "contradictions."
        ),
        weight=1.0,
    ),
]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert evaluation judge. Your task is to evaluate a candidate \
answer against the ground truth, using structural evidence from a \
knowledge graph to anchor your judgment.

You will be given:
1. The original question
2. The expected ground truth answer
3. The candidate answer to evaluate
4. Structural evidence from a Geometric Memory System (GMS) knowledge graph

Evaluate the candidate answer on these dimensions, scoring each 1-5:
{rubric_section}

IMPORTANT: Base your judgment on the evidence provided.  If the GMS \
evidence shows specific facts, use those as ground truth anchors.  If \
the candidate answer contradicts GMS evidence, score factual_accuracy \
lower.

Respond in EXACTLY this format:
<scores>
factual_accuracy: [1-5]
completeness: [1-5]
groundedness: [1-5]
coherence: [1-5]
</scores>
<reasoning>
[Brief explanation of your scores, referencing specific GMS evidence \
where relevant]
</reasoning>"""

_JUDGE_USER = """\
<question>
{question}
</question>

<ground_truth>
{ground_truth}
</ground_truth>

<candidate_answer>
{candidate_answer}
</candidate_answer>

<gms_evidence>
{evidence_text}
</gms_evidence>

Evaluate the candidate answer."""


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r"(\w+):\s*(\d)")
_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)


@dataclass
class SemanticScore:
    """Parsed scores from the LLM judge."""
    scores: dict[str, int] = field(default_factory=dict)
    reasoning: str = ""
    raw_response: str = ""

    @property
    def weighted_score(self) -> float:
        """Compute weighted average using default rubric weights."""
        weights = {d.name: d.weight for d in DEFAULT_RUBRIC}
        total_weight = 0.0
        total_score = 0.0
        for dim, score in self.scores.items():
            w = weights.get(dim, 1.0)
            total_score += score * w
            total_weight += w
        if total_weight == 0:
            return 0.0
        return total_score / total_weight


# ---------------------------------------------------------------------------
# SemanticVerifier
# ---------------------------------------------------------------------------

class SemanticVerifier:
    """LLM-as-judge verifier anchored by GMS geometric evidence.

    For each :class:`SemanticClaim`, this verifier:

    1. **Gathers GMS evidence** — related triples, geodesic scores,
       tension energy for entities mentioned in answer and ground truth.
    2. **Calls the LLM judge** — sends claim + evidence with a structured
       rubric prompt, parses dimension scores (1-5).
    3. **Calibrates confidence with GMS** — adjusts the LLM's confidence
       using entity coverage, geodesic proximity, and tension signals.

    GMS methods used at evidence-gathering time:

    =========================  ==========================================
    GMS method                 Purpose
    =========================  ==========================================
    ``fuzzy_match_entity()``   Resolve raw text entities to GMS graph
    ``query_triples()``        Gather related facts per entity
    ``score_triple()``         Geodesic proximity between entity pairs
    ``tension_energy()``       Contradiction signal between answer/truth
    =========================  ==========================================

    Parameters
    ----------
    store : GMSExpertStore
        GMS store for evidence gathering.
    client : gms.llm.LLMClient or None
        Pre-constructed LLM client. If ``None``, one is created lazily
        via :func:`gms.llm.get_llm` with :data:`ModelPurpose.JUDGE`.
    model : str or None
        Optional per-instance model override. When ``None`` (the
        default), the model is resolved from
        :class:`gms.llm.LLMSettings` (``GMS_LLM_MODEL_JUDGE`` → ``GMS_LLM_MODEL``).
        When set, it overrides the client's bound model on every call.
    rubric : list[RubricDimension] or None
        Custom rubric.  Uses :data:`DEFAULT_RUBRIC` if *None*.
    pass_threshold : float
        Weighted score (1-5 scale) at or above which the claim passes.
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        client=None,
        model: str | None = None,
        rubric: list[RubricDimension] | None = None,
        pass_threshold: float = 3.0,
        tracing=None,
    ):
        self.store = store
        self._client = client
        # Empty string is equivalent to None (callers sometimes pass "").
        self.model = model or None
        self.rubric = rubric or list(DEFAULT_RUBRIC)
        self.pass_threshold = pass_threshold
        self.tracing = tracing  # DOETracingGMS for recording judge calls

    @property
    def client(self):
        """Lazy-initialize the gms.llm LLMClient bound to ModelPurpose.JUDGE."""

        if self._client is None:
            from knowlytix.core.llm import ModelPurpose, get_llm

            self._client = get_llm(ModelPurpose.JUDGE)
        return self._client

    # ----- Public API -------------------------------------------------------

    def verify(self, claim: SemanticClaim):
        """Verify a SemanticClaim via LLM judge anchored by GMS evidence.

        Returns
        -------
        SubVerdict
            Verdict with GMS-calibrated confidence and rubric scores.
        """
        from .verifiers import SubVerdict

        # Step 1: Gather GMS evidence
        evidence = self._gather_gms_evidence(claim)

        # Step 2: Call LLM judge
        try:
            semantic_score = self._call_judge(claim, evidence)
        except Exception as e:
            return SubVerdict(
                verifier_name="SemanticVerifier",
                claim_id=claim.claim_id,
                passed=False,
                detail=f"LLM judge call failed: {e}",
                gms_method_used="semantic_judge",
                failure=FailureAttribution(
                    code=FailureCode.VERIFICATION_ERROR,
                    claim_id=claim.claim_id,
                    verifier="SemanticVerifier",
                    detail=str(e),
                    gms_method="semantic_judge",
                ),
            )

        # Step 3: Calibrate with GMS geometric scores
        calibrated = self._calibrate_confidence(semantic_score, evidence)

        passed = semantic_score.weighted_score >= self.pass_threshold

        failure = None
        if not passed:
            failure = self._determine_failure(claim, semantic_score, evidence)

        # Step 4: Trace the semantic judge call
        if self.tracing and hasattr(self.tracing, "trace_semantic_verdict"):
            self.tracing.trace_semantic_verdict(
                doe_qid=claim.source_question_id,
                run_id=claim.source_run_id,
                judge_model=self.model,
                rubric_scores=dict(semantic_score.scores),
                weighted_score=semantic_score.weighted_score,
                entity_coverage=evidence.entity_coverage,
                n_geodesic_pairs=len(evidence.geodesic_scores),
                n_tension_pairs=len(evidence.tension_scores),
                calibrated_confidence=calibrated,
                passed=passed,
                reasoning=semantic_score.reasoning,
            )

        return SubVerdict(
            verifier_name="SemanticVerifier",
            claim_id=claim.claim_id,
            passed=passed,
            confidence=calibrated,
            detail=(
                f"Semantic scores: {semantic_score.scores} "
                f"(weighted={semantic_score.weighted_score:.2f}, "
                f"threshold={self.pass_threshold}). "
                f"GMS coverage={evidence.entity_coverage:.0%}. "
                f"{semantic_score.reasoning}"
            ),
            gms_method_used=(
                "fuzzy_match_entity,query_triples,"
                "score_triple,tension_energy"
            ),
            raw_score=semantic_score.weighted_score / 5.0,
            failure=failure,
        )

    # ----- Evidence gathering -----------------------------------------------

    def _gather_gms_evidence(self, claim: SemanticClaim) -> GMSEvidence:
        """Query GMS for structural evidence relevant to the claim."""
        evidence = GMSEvidence()

        if not self.store.adapter or not self.store.doc_graph:
            return evidence

        # Find entities in answer and ground truth
        answer_ents = self._find_entities(claim.raw_text)
        truth_ents = self._find_entities(claim.ground_truth)

        # Resolve to GMS entities via fuzzy_match_entity
        for raw in answer_ents:
            matched = self.store.fuzzy_match_entity(raw)
            if matched:
                evidence.answer_entities[raw] = matched

        for raw in truth_ents:
            matched = self.store.fuzzy_match_entity(raw)
            if matched:
                evidence.truth_entities[raw] = matched

        # Entity coverage
        if answer_ents:
            evidence.entity_coverage = (
                len(evidence.answer_entities) / len(answer_ents)
            )

        # Related triples for context (via query_triples / doc_graph)
        all_matched = (
            set(evidence.answer_entities.values())
            | set(evidence.truth_entities.values())
        )
        for entity in all_matched:
            triples = self.store.query_triples(head=entity)
            # Also triples where entity is tail
            triples += self.store.query_triples(tail=entity)
            evidence.entity_triples[entity] = triples[:10]

        # Geodesic proximity between answer and truth entities
        for a_match in evidence.answer_entities.values():
            for t_match in evidence.truth_entities.values():
                if a_match == t_match:
                    continue
                key = f"{a_match}\u2194{t_match}"
                if key in evidence.geodesic_scores:
                    continue
                score = self.store.score_triple(
                    a_match, "related_to", t_match)
                if score is not None:
                    evidence.geodesic_scores[key] = score

        # Tension energy between answer and truth entities
        for a_match in evidence.answer_entities.values():
            for t_match in evidence.truth_entities.values():
                if a_match == t_match:
                    continue
                key = f"{a_match}\u2194{t_match}"
                if key in evidence.tension_scores:
                    continue
                try:
                    tension = self.store.tension_energy(a_match, t_match)
                    if tension is not None:
                        evidence.tension_scores[key] = tension
                except Exception:
                    pass

        return evidence

    def _find_entities(self, text: str) -> list[str]:
        """Find GMS store entities mentioned in text."""
        if not text or not self.store.adapter:
            return []
        text_lower = text.lower()
        found = []
        for entity in self.store.adapter.entity_to_idx:
            if len(entity) > 2 and entity.lower() in text_lower:
                found.append(entity)
        return found

    # ----- LLM judge --------------------------------------------------------

    def _call_judge(
        self, claim: SemanticClaim, evidence: GMSEvidence
    ) -> SemanticScore:
        """Build prompt, call LLM, parse structured verdict."""
        rubric_lines = []
        for dim in self.rubric:
            rubric_lines.append(
                f"- **{dim.name}** (weight {dim.weight}): {dim.description}"
            )
        rubric_section = "\n".join(rubric_lines)

        system = _JUDGE_SYSTEM.format(rubric_section=rubric_section)

        evidence_text = self._format_evidence(evidence)

        user = _JUDGE_USER.format(
            question=claim.question_text or "(not provided)",
            ground_truth=claim.ground_truth or "(not provided)",
            candidate_answer=claim.raw_text or "(empty)",
            evidence_text=evidence_text,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {"max_tokens": 512}
        if self.model:
            # Per-instance override wins over the client's bound model.
            kwargs["model"] = self.model
        raw = self.client.complete(messages, **kwargs)

        return self._parse_response(raw)

    def _format_evidence(self, evidence: GMSEvidence) -> str:
        """Format GMS evidence as readable text for the LLM prompt."""
        lines: list[str] = []

        if evidence.answer_entities:
            lines.append(
                "## Entities in candidate answer found in knowledge graph")
            for raw, matched in evidence.answer_entities.items():
                label = f'"{raw}"' if raw != matched else f'"{matched}"'
                lines.append(f"- {label} \u2192 matched to \"{matched}\"")
                for h, r, t in evidence.entity_triples.get(matched, [])[:5]:
                    lines.append(f"  - ({h}, {r}, {t})")

        if evidence.truth_entities:
            missing = (
                set(evidence.truth_entities.keys())
                - set(evidence.answer_entities.keys())
            )
            if missing:
                lines.append(
                    "\n## Ground truth entities NOT mentioned in answer")
                for raw in missing:
                    lines.append(
                        f"- \"{raw}\" (exists in knowledge graph)")

        if evidence.geodesic_scores:
            lines.append(
                "\n## Geodesic proximity scores (higher = closer)")
            for pair, score in evidence.geodesic_scores.items():
                lines.append(f"- {pair}: {score:.3f}")

        if evidence.tension_scores:
            lines.append(
                "\n## Tension energy (>1.5 suggests contradiction)")
            for pair, tension in evidence.tension_scores.items():
                lines.append(f"- {pair}: {tension:.3f}")

        lines.append(
            f"\nEntity coverage: {evidence.entity_coverage:.0%} "
            f"of answer entities found in knowledge graph"
        )

        return "\n".join(lines) if lines else "No GMS evidence available."

    def _parse_response(self, raw: str) -> SemanticScore:
        """Parse the LLM judge's structured response."""
        score = SemanticScore(raw_response=raw)

        scores_match = re.search(
            r"<scores>(.*?)</scores>", raw, re.DOTALL)
        if scores_match:
            for m in _SCORE_RE.finditer(scores_match.group(1)):
                dim_name = m.group(1)
                dim_score = int(m.group(2))
                score.scores[dim_name] = min(5, max(1, dim_score))

        reasoning_match = _REASONING_RE.search(raw)
        if reasoning_match:
            score.reasoning = reasoning_match.group(1).strip()

        # Fill defaults for any missing rubric dimensions
        for dim in self.rubric:
            if dim.name not in score.scores:
                score.scores[dim.name] = 3  # neutral default

        return score

    # ----- Confidence calibration -------------------------------------------

    def _calibrate_confidence(
        self, score: SemanticScore, evidence: GMSEvidence
    ) -> float:
        """Calibrate LLM confidence using GMS geometric signals.

        Base confidence comes from the LLM's weighted score (normalized
        to [0, 1]).  GMS signals adjust it:

        - **Entity coverage**: high coverage → boost (GMS confirms the
          entities the answer references actually exist).
        - **Geodesic proximity**: close proximity between answer and
          truth entities → boost.
        - **Tension energy**: high tension between answer and truth
          entities → penalize (contradiction signal).
        """
        base = score.weighted_score / 5.0  # [0, 1]

        # Coverage: [0.8, 1.0]
        coverage_factor = 0.8 + 0.2 * evidence.entity_coverage

        # Proximity: [0.9, 1.0]
        if evidence.geodesic_scores:
            avg_proximity = (
                sum(evidence.geodesic_scores.values())
                / len(evidence.geodesic_scores)
            )
            proximity_factor = 0.9 + 0.1 * min(1.0, avg_proximity)
        else:
            proximity_factor = 0.95  # neutral

        # Tension penalty: [0.8, 1.0]
        if evidence.tension_scores:
            max_tension = max(evidence.tension_scores.values())
            if max_tension > 1.5:
                tension_penalty = 0.8
            elif max_tension > 1.0:
                tension_penalty = 0.9
            else:
                tension_penalty = 1.0
        else:
            tension_penalty = 1.0

        calibrated = (
            base * coverage_factor * proximity_factor * tension_penalty
        )
        return max(0.0, min(1.0, calibrated))

    # ----- Failure determination --------------------------------------------

    def _determine_failure(
        self,
        claim: SemanticClaim,
        score: SemanticScore,
        evidence: GMSEvidence,
    ) -> FailureAttribution:
        """Map worst semantic dimension to the appropriate failure code."""
        scores = score.scores

        worst_dim = min(scores, key=scores.get, default="factual_accuracy")
        worst_val = scores.get(worst_dim, 3)

        if worst_dim == "factual_accuracy" and worst_val <= 2:
            code = FailureCode.SEMANTIC_MISMATCH
            detail = "LLM judge found significant factual inaccuracies"
        elif worst_dim == "groundedness" and worst_val <= 2:
            code = FailureCode.HALLUCINATION_DETECTED
            detail = (
                "LLM judge detected hallucinated content not grounded "
                "in source material"
            )
        elif worst_dim == "completeness" and worst_val <= 2:
            code = FailureCode.INCOMPLETE_ANSWER
            detail = "LLM judge found the answer incomplete"
        elif worst_dim == "coherence" and worst_val <= 2:
            code = FailureCode.CONTRADICTION_DETECTED
            detail = "LLM judge found internal inconsistencies"
        else:
            code = FailureCode.SEMANTIC_MISMATCH
            detail = (
                f"Semantic score below threshold "
                f"({score.weighted_score:.2f} < {self.pass_threshold})"
            )

        return FailureAttribution(
            code=code,
            claim_id=claim.claim_id,
            verifier="SemanticVerifier",
            detail=f"{detail}. {score.reasoning}",
            gms_method="semantic_judge+gms_evidence",
            raw_score=score.weighted_score / 5.0,
        )
