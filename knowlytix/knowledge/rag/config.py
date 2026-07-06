# SPDX-License-Identifier: Apache-2.0
"""Configuration for the GEODE graph-RAG pipeline.

See ``GEODE_RAG_DESIGN.md``. The pipeline is triple-mediated and bank-grade by
default: dense retrieval is off, the final answer is verifiable, and per-role
LLMs are user-selectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from knowlytix.knowledge.llm_backend import LLMBackend

__all__ = ["RagConfig"]


@dataclass
class RagConfig:
    """Tuning knobs for :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline`.

    Args:
        llm: synthesis backend (facts + spans -> answer). Required.
        llm_extract: NL question -> query triples. Defaults to ``llm``.
        llm_verify: answer -> claim triples for self-verification. Defaults to
            ``llm_extract``.
        binding: query-triple binding strategy — ``"fuzzy"`` (string, P1) or
            ``"embedding"`` (P2).
        top_k: candidate tails per ``?``-slot retrieval.
        max_hops: bound on multi-hop chain resolution.
        verify_llm_output: extract triples from the answer and check them against
            the GMS (P2). Off until wired.
        on_verify_fail: action when a claim triple fails GMS verification.
        accept_threshold: min confidence to ``accept`` (else ``abstain``).
        dense_fallback: opt-in, distrusted dense retrieval when nothing binds
            (P5). Results are flagged ``verified=False``.
        strict_mode: when ``True``, a dense-fallback route is downgraded to
            ``abstain`` rather than returned.
    """

    llm: LLMBackend
    llm_extract: LLMBackend | None = None
    llm_verify: LLMBackend | None = None

    binding: Literal["fuzzy", "embedding"] = "fuzzy"
    # Embedding-binding knobs (used only when binding="embedding"):
    encoder: Callable[[list[str]], object] | None = None  # default: MiniLM
    bind_threshold: float = 0.5   # min cosine to accept a match
    bind_margin: float = 0.05     # min top1-top2 gap (else refuse as ambiguous)
    top_k: int = 10
    max_hops: int = 5
    # Re-ask the extractor (with the graph vocab + failed terms) when binding
    # fails — the main extraction-reliability lever for small models.
    max_extract_retries: int = 1
    ground_extraction: bool = True   # inject graph schema into the extract prompt
    # Query parsing: "geometric" (GeometricQueryParser -- tuned-encoder +
    # graph-structure, no parse-time LLM) or "llm" (QueryTripleExtractor).
    # Geometric is the default: it is robust where a small model emits
    # attribute-as-head / generic relations, and is GEODE-native. It prefers
    # RagConfig.encoder (the document-tuned v-encoder) and otherwise falls back
    # to the default MiniLM encoder. Use "llm" only when no doc graph is
    # available to parse against (GeometricQueryParser needs store.doc_graph).
    query_parse_mode: Literal["llm", "geometric"] = "geometric"

    # Retrieval mode. "query_triple" (default) parses one (head, relation, tail)
    # pattern and retrieves that attribute -- relevance is decided upstream by the
    # relevance gate. "head_facts" is the operator-native path: it binds the
    # question's head entity (embedding), then retrieves ALL of that head's
    # admissible facts through the relation operators (the top ``n_heads`` head
    # candidates, expanded ``head_hops`` hops for multi-hop), and lets
    # select-and-answer synthesis pick the attribute or abstain. In "head_facts"
    # mode relations are never matched by phrase and the relevance gate is unused;
    # abstention comes from the head-bind floor and from synthesis. See
    # GEODE_RAG_DESIGN.md and assemble._SELECT_SYSTEM.
    retrieve_mode: Literal["query_triple", "head_facts"] = "query_triple"
    # head_facts knobs:
    head_bind_floor: float = 0.0   # min head-name cosine to proceed; below -> abstain
    n_heads: int = 3               # head candidates whose facts are pooled
    head_hops: int = 2             # graph hops expanded from each head (>=2 = multi-hop)
    # Provenance: when True, a fact's source span is the SENTENCE that states it
    # (ProvenanceLedger prefer_prose) rather than the table cell it was extracted
    # from -- readable context for the synthesizer, and it lets the model restate a
    # policy stance ("... is forbidden") instead of decoding a relation name from a
    # bare value. Off by default, so a table-only corpus (no prose) is unchanged.
    provenance_prefer_prose: bool = False

    # Numeric ordering route (superlatives / comparisons over the ENM). A
    # question like "which fee is highest?" names no anchor entity, so the
    # triple route cannot bind it. When on, such a question is answered by
    # ranking the relevant ENM type through u-space directed entailment
    # (numeric_order.NumericUSpaceOrder) -- exact values, GMS-native logic.
    # The values are exact, so the winner is unambiguous unless there is no
    # *distinct* runner-up (all entries equal): abstain when the winner's
    # u-space energy gap to the nearest distinct value is <= this (default 0,
    # i.e. only the degenerate all-equal case abstains). A near-but-distinct
    # runner-up is NOT a tie -- the exact values already separate them.
    numeric_order: bool = True
    numeric_order_tie_margin: float = 0.0
    # Min cosine of the question to an ENM-type phrase to route there (needs
    # ``encoder``; without one, the type name must appear in the question).
    numeric_order_floor: float = 0.25

    verify_llm_output: bool = False
    on_verify_fail: Literal["annotate", "abstain", "regenerate"] = "abstain"
    # Claim-extraction mode for self-verification. A synthesized answer is
    # decomposed into one claim per asserted fact and EACH claim is checked
    # against the GMS. "geometric" walks the graph (no LLM) but keys on the
    # STORED tail appearing in the text, so it cannot see a value REVERSAL
    # (999 for a stored 340) -- it just omits that claim. "llm" decomposes with
    # the model so the asserted value is carried and a reversal surfaces as a
    # contradicted claim. "hybrid" runs both (deduped): geometric gives grounding
    # coverage, the LLM pass adds reversal detection -- at the cost of regressing
    # recall on terse numeric values (the small model mis-decomposes them), which
    # is why it is opt-in, not the default. None -> "geometric"; value-reversal and
    # polarity contradictions are recovered geometrically by the calibrated u-space
    # polarity check instead of the LLM pass. Override per-call here or process-wide
    # with the KNOWLYTIX_VERIFY_MODE env var. See verify.AnswerVerifier.
    verify_mode: Literal["llm", "geometric", "hybrid"] | None = None
    # u-space POLARITY check in the verifier (optional). When verify_u_encoder is
    # set (the GEODE contradiction encoder), the verifier flags an answer that
    # reverses a grounding fact's polarity ("can be sent" against a stored
    # forbidden) -- a contradiction the lexical value match misses because the
    # prose never quotes the stored value. verify_tau_polarity is the u-tension cut
    # above which the answer contradicts the fact (calibrated, persisted).
    verify_u_encoder: Callable[[list[str]], object] | None = None
    verify_tau_polarity: float = 0.9
    # Fused categorical-value polarity check (ValuePolarityChecker). When the
    # polarity-tuned u-encoder and its calibrated 3-class cuts are set, the verifier
    # decides a CATEGORICAL claim's value by cap plausibility + v-space resolution
    # (synonym tolerance) + 3-class u-space tension (polarity), abstaining on the
    # unrelated middle band ("uncertain" -> the answer is not accepted). Numeric
    # values keep the exact path. Off by default; cuts are calibrated and persisted
    # (PolarityCuts). See value_polarity.ValuePolarityChecker.
    verify_value_polarity_encoder: Callable[[list[str]], object] | None = None
    verify_polarity_cuts: tuple[float, float] | None = None
    verify_polarity_res_margin: float = 0.08

    # Question<->attribute relevance gate (P-relevance): after binding, an LLM
    # confirms the bound relation is the attribute the question actually asked
    # about (over the head's real relations), abstaining on "none". Catches the
    # true-but-irrelevant leak (interest rate -> has_fee_amount) that cap/tension
    # and embedding cosine cannot. Needs an LLM; uses llm_extract. On by default
    # (bank-grade); the prefilter keeps it cheap.
    relevance_gate: bool = True
    # Relevance decision: "llm" (a 3B judge over the head's relations) or
    # "geometric" (LLM-free, GEODE-native): v-space accepts the nearest relation,
    # u-space contradiction tension vetoes an absent-but-adjacent attribute.
    # Geometric needs a v-encoder (RagConfig.encoder) and a contradiction-tuned
    # u-encoder (relevance_u_encoder), both from the GEODE embed loop.
    relevance_mode: Literal["llm", "geometric"] = "llm"
    relevance_u_encoder: Callable[[list[str]], object] | None = None
    relevance_tau_accept: float = 0.30   # min v-cosine to accept a relation
    relevance_tau_contra: float = 0.75   # max u-tension before the veto fires
    relevance_tau_contra_per_relation: dict = field(default_factory=dict)
    # Hybrid prefilter: skip the LLM call and abstain immediately when the
    # question is clearly unrelated to every relation the head holds (max cosine
    # below the floor). Reject-only and conservative -- the floor sits well below
    # any observed relevant question (min ANSWER ~0.35), so it never short-circuits
    # a plausibly-relevant query; those still go to the LLM judge. A latency
    # optimization, not the decision boundary.
    relevance_prefilter: bool = True
    relevance_prefilter_floor: float = 0.25

    accept_threshold: float = 0.0  # P1: accept any grounded answer; P4 calibrates

    dense_fallback: bool = False
    vector_backend: object | None = None   # VectorBackend; default in-memory
    # When dense_fallback is enabled, strict_mode=True still refuses to return a
    # dense (unverified) answer and abstains instead. Default False so that
    # opting into dense_fallback actually yields answers (flagged unverified).
    strict_mode: bool = False

    # Audit: called with each answer's audit() dict (bank-grade logging hook).
    audit_sink: Callable[[dict], None] | None = None

    def extract_llm(self) -> LLMBackend:
        return self.llm_extract or self.llm

    def verify_llm(self) -> LLMBackend:
        return self.llm_verify or self.extract_llm()
