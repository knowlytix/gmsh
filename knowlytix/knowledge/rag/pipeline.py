# SPDX-License-Identifier: Apache-2.0
"""RagPipeline — the triple-mediated graph-RAG orchestrator.

Online flow (``GEODE_RAG_DESIGN.md`` §5): extract query triples -> bind to graph
vocab -> bind-check -> GMS retrieve (+ provenance) -> synthesize -> decide
(accept / abstain / escalate). Dense fallback and answer self-verification are
opt-in extensions wired in later phases; the default is the verifiable triple
route only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.knowledge.geode.provenance import ProvenanceLedger
from knowlytix.knowledge.rag.arith import (
    ARITH_SYSTEM,
    Op,
    asked_unit,
    compute,
    detect,
    listing,
    needs_question_value,
    parse_expression,
    quantities,
    sentence,
    worth_asking,
)
from knowlytix.knowledge.rag.assemble import Assembler
from knowlytix.knowledge.rag.binding import BoundTriple, TripleBinder
from knowlytix.knowledge.rag.config import RagConfig
from knowlytix.knowledge.rag.dense import (
    DenseSpan,
    InMemoryVectorBackend,
    build_spans,
)
from knowlytix.knowledge.rag.extraction import Extraction
from knowlytix.knowledge.rag.facts import (
    LINK_SOURCE,
    SCAFFOLD_RELATIONS,
    FactIndex,
    Link,
    cited,
    link_sentence,
    renumber,
)
from knowlytix.knowledge.rag.query_triples import (
    ASKED,
    QueryTriple,
    QueryTripleExtractor,
    is_var,
    schema_from_store,
)
from knowlytix.knowledge.rag.retrieve import RetrievedFact, Retriever
from knowlytix.knowledge.rag.verify import AnswerVerifier

__all__ = ["RagAnswer", "RagPipeline"]


_SCAFFOLD_RELATIONS = SCAFFOLD_RELATIONS

def _relation_phrase(relation: str) -> str:
    """A relation as words for an encoder: ``has_filing_deadline`` -> ``filing deadline``."""
    body = relation[4:] if relation.startswith("has_") else relation
    return body.replace("_", " ")


def _fact_record(f: RetrievedFact) -> dict:
    """One fact as it appears in the audit record."""
    return {"triple": (f.head, f.relation, f.tail), "source": f.source,
            "score": f.score, "location": f.location}


@dataclass
class RagAnswer:
    """A RAG answer with its gate decision, route, evidence and verification detail.

    Attributes:
        answer: the synthesized answer text.
        confidence: retrieval confidence in [0, 1].
        decision: gate outcome (``"accept"``, ``"abstain"`` or ``"escalate"``).
        route: retrieval route (``"triple"`` or ``"dense_fallback"``).
        verified: whether the answer passed GMS self-verification.
        query_triples: the query triples parsed from the question.
        bound_triples: those triples bound to store vocabulary.
        sources: the retrieved grounding facts.
        dense_sources: dense-fallback spans, when that route was used.
        verification: the self-verification report as a dict.
        notice: optional notice explaining the decision or route.
        nearest_facts: on an abstention, the document's facts closest to the
            question that did not answer it, at most ``top_k`` (empty when no
            entity in the question resolved, and on an accepted answer). Only
            facts the document asserts. Kept apart from ``sources``, which are the
            facts that ground an answer.
    """
    answer: str
    confidence: float
    decision: str                       # "accept" | "abstain" | "escalate"
    route: str                          # "triple" | "dense_fallback"
    verified: bool
    query_triples: list[QueryTriple] = field(default_factory=list)
    bound_triples: list[BoundTriple] = field(default_factory=list)
    sources: list[RetrievedFact] = field(default_factory=list)
    dense_sources: list[DenseSpan] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    notice: str | None = None
    nearest_facts: list[RetrievedFact] = field(default_factory=list)

    def audit(self) -> dict:
        """Structured audit record of this answer (route, evidence, verdicts)."""
        return {
            "decision": self.decision,
            "route": self.route,
            "verified": self.verified,
            "confidence": self.confidence,
            "query_triples": [t.as_tuple() for t in self.query_triples],
            "bound_triples": [
                (b.head, b.relation, b.tail, b.bound) for b in self.bound_triples
            ],
            "sources": [_fact_record(f) for f in self.sources],
            "verification": self.verification,
            "dense_sources": [s.location for s in self.dense_sources],
            "notice": self.notice,
            "nearest_facts": [_fact_record(f) for f in self.nearest_facts],
        }


_ABSTAIN_MSG = "I cannot answer this from the available grounded evidence."


class RagPipeline:
    """Triple-mediated RAG over a :class:`GMSExpertStore`."""

    def __init__(self, store, rag: RagConfig, *, ledger: ProvenanceLedger | None = None):
        self.store = store
        self.rag = rag
        self.ledger = ledger or self._default_ledger(store, rag.provenance_prefer_prose)
        self._vocab = schema_from_store(store) if rag.ground_extraction else None
        if rag.query_parse_mode in ("compiler", "alias_hybrid"):
            if not rag.compiler_dir:
                raise ValueError(
                    f"query_parse_mode={rag.query_parse_mode!r} requires rag.compiler_dir "
                    "(the trained adapter dir); build one with "
                    "knowlytix.knowledge.rag.compiler.build_query_compiler, or use "
                    "RagConfig.from_store_dir which sets it when query_compiler/ exists.")
            from knowlytix.knowledge.rag.compiler import CompiledQueryParser
            compiler = CompiledQueryParser(
                store, rag.compiler_dir, base_model=rag.compiler_base_model,
                device=rag.compiler_device)
            if rag.query_parse_mode == "compiler":
                self.extractor = compiler
            else:
                # Arm D: compiler relations + a patchable alias-table head resolver
                # with a calibrated, persisted encoder-fallback margin (recuses when
                # it cannot calibrate). Reuses the compiler just loaded. A mis-emitted
                # first-hop relation is repaired against the head's relations by the
                # calibrated question->relation accept step of the geometric relevance
                # gate (tau_accept from the store's persisted question-cohort
                # calibration), so the repair follows the QUESTION, not the emitted
                # relation string. Accept-only (no u-veto) -- the veto is the
                # admissibility gate's job, left off on this route by design.
                from knowlytix.knowledge.rag.compiler import build_alias_hybrid_parser
                relation_gate = None
                if rag.encoder is not None:
                    from knowlytix.knowledge.rag.relevance import GeometricRelevanceGate
                    relation_gate = GeometricRelevanceGate(
                        store, v_encoder=rag.encoder, u_encoder=None,
                        tau_accept=rag.relevance_tau_accept)
                self.extractor = build_alias_hybrid_parser(
                    store, compiler, rag.encoder, relation_gate=relation_gate,
                    store_path=getattr(store, "store_path", None))
        elif rag.query_parse_mode == "geometric":
            from knowlytix.knowledge.rag.query_triples import GeometricQueryParser
            self.extractor = GeometricQueryParser(store, encoder=rag.encoder,
                                                  max_hops=rag.max_hops)
        else:
            self.extractor = QueryTripleExtractor(rag.extract_llm(), vocab=self._vocab)
        self.binder = TripleBinder(store, mode=rag.binding, encoder=rag.encoder,
                                   bind_threshold=rag.bind_threshold,
                                   bind_margin=rag.bind_margin)
        self.retriever = Retriever(store, self.ledger, top_k=rag.top_k)
        # The operator-native head_facts path retrieves all of a head's facts, so
        # synthesis must select the answering fact (or abstain) rather than
        # verbalize a pre-selected one.
        self.assembler = Assembler(rag.llm, mode={"head_facts": "select", "facts": "cite"}
                                   .get(rag.retrieve_mode, "verbalize"))
        self.fact_index = (FactIndex(store, self.retriever, rag.encoder)
                           if rag.retrieve_mode == "facts" else None)
        # Default geometric (graph-walk grounding coverage, no LLM decomposition):
        # it does not regress recall on terse values the way the LLM-decomposition
        # pass does, and contradiction detection is recovered geometrically through
        # the calibrated u-space polarity check. "hybrid"/"llm" are opt-in (explicit
        # config, or the KNOWLYTIX_VERIFY_MODE env var) for adversarial value-reversal
        # auditing where the extra LLM pass is worth the recall cost.
        import os
        verify_mode = (rag.verify_mode
                       or os.getenv("KNOWLYTIX_VERIFY_MODE") or "geometric")
        # Optional fused categorical-value polarity checker (cap + v-resolution +
        # 3-class u-tension). Built only when its tuned u-encoder + calibrated cuts
        # are configured; otherwise the verifier's geometric value check is cap-only.
        value_checker = None
        if (rag.verify_value_polarity_encoder is not None
                and rag.verify_polarity_cuts is not None):
            from knowlytix.knowledge.rag.value_polarity import (
                PolarityCuts,
                ValuePolarityChecker,
            )
            value_checker = ValuePolarityChecker(
                store, rag.encoder, rag.verify_value_polarity_encoder,
                PolarityCuts(*rag.verify_polarity_cuts),
                res_margin=rag.verify_polarity_res_margin)
        self.verifier = (AnswerVerifier(
                            store, rag.verify_llm(), binder=self.binder,
                            mode=verify_mode,
                            u_encoder=rag.verify_u_encoder, v_encoder=rag.encoder,
                            tau_polarity=rag.verify_tau_polarity,
                            value_checker=value_checker)
                         if rag.verify_llm_output else None)
        self.relevance = None
        if rag.relevance_gate:
            if rag.relevance_mode == "geometric":
                from knowlytix.knowledge.rag.relevance import GeometricRelevanceGate
                self.relevance = GeometricRelevanceGate(
                    store, v_encoder=rag.encoder,
                    u_encoder=rag.relevance_u_encoder,
                    tau_accept=rag.relevance_tau_accept,
                    tau_contra=rag.relevance_tau_contra,
                    tau_contra_per_relation=rag.relevance_tau_contra_per_relation)
            else:
                from knowlytix.knowledge.rag.relevance import RelevanceGate
                self.relevance = RelevanceGate(
                    store, rag.extract_llm(), encoder=rag.encoder,
                    prefilter=rag.relevance_prefilter,
                    prefilter_floor=rag.relevance_prefilter_floor)
        self.dense = None
        if rag.dense_fallback:
            backend = rag.vector_backend or InMemoryVectorBackend(rag.encoder)
            backend.index(build_spans(store.markdown,
                                      source_path=store.store_path))
            self.dense = backend

        # Numeric-order route: cache the rankable ENM types (>= 2 scalar
        # entries) and, when an encoder is present, their phrase embeddings so a
        # superlative question can be matched to a type by cosine.
        self._enm_types: list[str] = []
        self._enm_type_emb = None
        if rag.numeric_order and store.enm is not None:
            from collections import Counter
            counts = Counter(k.type for k in store.enm.keys())
            self._enm_types = sorted(t for t, n in counts.items() if n >= 2)
            if self._enm_types and rag.encoder is not None:
                import torch
                emb = torch.as_tensor(
                    rag.encoder([t.replace("_", " ") for t in self._enm_types]),
                    dtype=torch.float32)
                self._enm_type_emb = torch.nn.functional.normalize(
                    emb, p=2, dim=-1)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_store(cls, store, rag: RagConfig) -> "RagPipeline":
        """Build a :class:`RagPipeline` over an already-built store and config."""
        return cls(store, rag)

    @classmethod
    def build(cls, md_path: str, config, rag: RagConfig, *, device=None,
              **build_kwargs) -> "RagPipeline":
        """Run GEODE self-correction + train a store, then wrap it.

        ``build_kwargs`` are forwarded to
        :func:`knowlytix.knowledge.geode.rag.build_rag_store` (e.g. ``llm`` for
        the Qwen actor, ``max_iters``).
        """
        from knowlytix.knowledge.geode.rag import build_rag_store

        result = build_rag_store(md_path, config, device=device, **build_kwargs)
        return cls.from_store(result.store, rag)

    @staticmethod
    def _default_ledger(store, prefer_prose: bool = False) -> ProvenanceLedger | None:
        if getattr(store, "markdown", ""):
            return ProvenanceLedger.from_text(store.markdown,
                                              source_path=store.store_path,
                                              prefer_prose=prefer_prose)
        return None

    # -- extract (reusable parse⇄bind) --------------------------------------

    def extract(self, question: str) -> Extraction:
        """Parse ``question`` into graph-bound query triples — the reusable
        parse⇄bind half of :meth:`query`, with no retrieval or generation.

        Returns an :class:`Extraction` (the query triples plus their binding to
        the store's vocabulary). Use :attr:`Extraction.is_bound` to tell a
        grounded message from one that binds to nothing, and :attr:`Extraction.
        bound_facts` for the grounded ``(head, relation, tail)`` triples. Hand
        the result back into :meth:`query` to answer it without re-parsing.
        """
        qts, bound = self._extract_and_bind(question)
        return Extraction(query_triples=qts, bound_triples=bound)

    # -- query --------------------------------------------------------------

    def query(self, question: str, *, generate: bool = True,
              extraction: Extraction | None = None) -> RagAnswer:
        """Answer ``question`` through the triple-mediated pipeline and gates.

        Runs extract, bind, relevance, retrieve and the cap/accept gate, then
        synthesizes and self-verifies the answer.

        Args:
            question: the natural-language question.
            generate: when False, run retrieval and the gates but skip the LLM
                assembler and self-verifier, returning the gated sources with no
                prose answer.
            extraction: an optional precomputed :class:`Extraction` to reuse instead
                of re-parsing the question.

        Returns:
            A :class:`RagAnswer` with the decision, evidence and verification detail.
        """
        # ``generate=False`` is a retrieve-only path: run extract -> bind ->
        # relevance -> retrieve -> cap/accept gate, but SKIP the LLM assembler and
        # self-verifier. It returns the gated ``sources`` (the grounded graph
        # facts) with no prose answer -- for callers that route a query to its
        # governing graph entities (claim-decomposition extraction) rather than
        # needing a generated sentence. Same gates, no generation cost.
        # Numeric-order route first: a superlative/comparison question names no
        # anchor entity, so the triple route below cannot bind it. Resolve it by
        # ranking the relevant ENM type through u-space directed entailment.
        # A supplied ``extraction`` is reused verbatim: skip the numeric-order
        # route and the parse⇄bind loop so the message is parsed exactly once
        # across an extract()-then-query() flow. (Passing an extraction commits
        # to the triple route; a superlative resolves via numeric_order only on
        # the pipeline's own parse pass.)
        if extraction is None and self._enm_types:
            numeric = self._try_numeric_order(question)
            if numeric is not None:
                return self._finalize(numeric, question)

        # Operator-native route: bind the head, retrieve ALL of its facts through
        # the relation operators, and let select-and-answer synthesis pick the
        # attribute or abstain. No relation-phrase matching, no relevance gate.
        if self.rag.retrieve_mode == "head_facts" and extraction is None:
            return self._query_head_facts(question)

        # Fact-search route: every fact, by meaning; the answer cites what it uses.
        if self.fact_index is not None and extraction is None:
            return self._query_facts(question, generate)

        if extraction is None:
            extraction = self.extract(question)
        qts, bound = extraction.query_triples, extraction.bound_triples

        # Bind-check: if nothing resolved to the graph, we have no grounded
        # path. Try the (distrusted, opt-in) dense fallback, else abstain.
        if not qts or not any(b.bound for b in bound):
            return self._finalize(
                self._fallback_or_abstain(question, qts, bound, "no_binding"), question)

        # Relevance gate: a bound relation can be graph-valid yet not the attribute
        # the question asked about (interest rate -> has_fee_amount). An LLM checks
        # the asked attribute against the head's real relations; "none" abstains.
        if self.relevance is not None:
            for b in bound:
                # Skip hops the gate cannot judge against the question: a
                # variable head (resolved only at retrieval), or an intermediate
                # hop whose tail feeds the next hop (its relation is not the
                # asked attribute, so judging it against the final question
                # wrongly rejects valid chains, e.g. has_division for a question
                # about region). Only the hop bearing the asked slot is gated.
                if (not b.bound or b.head is None or b.relation is None
                        or is_var(b.original.head)
                        or (is_var(b.original.tail) and b.original.tail != ASKED)):
                    continue
                matched, _ = self.relevance.relevant_relation(question, b.head)
                if matched is None:
                    return self._finalize(self._abstain(qts, bound, reason="irrelevant"),
                                          question, heads=[b.head])

        retrieved = self.retriever.retrieve(bound)
        if not retrieved.matched:
            return self._finalize(
                self._fallback_or_abstain(question, qts, bound, "no_match"), question)

        if not generate:
            confidence = self._confidence(retrieved.facts)
            decision = ("accept" if confidence >= self.rag.accept_threshold
                        else "abstain")
            return self._finalize(RagAnswer(
                answer="", confidence=confidence, decision=decision,
                route="triple", verified=False, query_triples=qts,
                bound_triples=bound, sources=retrieved.facts,
                verification={}, notice=None), question, candidates=retrieved.facts)

        answer = self.assembler.assemble(question, retrieved.facts)
        confidence = self._confidence(retrieved.facts)

        verification: dict = {}
        notice: str | None = None
        if self.verifier is not None:
            answer, confidence, notice, verification = self._self_verify(
                question, answer, retrieved.facts, confidence)
            if verification.get("ok") is False and self.rag.on_verify_fail == "abstain":
                return self._finalize(self._abstain(
                    qts, bound, reason="verify_fail", verification=verification),
                    question, candidates=retrieved.facts)

        decision = ("accept" if confidence >= self.rag.accept_threshold
                    else "abstain")
        return self._finalize(RagAnswer(
            answer=answer if decision == "accept" else _ABSTAIN_MSG,
            confidence=confidence,
            decision=decision,
            route="triple",
            verified=True,
            query_triples=qts,
            bound_triples=bound,
            sources=retrieved.facts,
            verification=verification,
            notice=notice,
        ), question, candidates=retrieved.facts)

    # -- operator-native head-facts route -----------------------------------

    def _query_head_facts(self, question: str) -> RagAnswer:
        """Bind the head entity, retrieve ALL of its admissible facts through the
        relation operators (top ``n_heads`` heads, ``head_hops`` expanded), and
        let select-and-answer synthesis pick the attribute or abstain. Relations
        are never matched by phrase; abstention comes from the head-bind floor
        (no entity binds) and from synthesis (no retrieved fact answers)."""
        from knowlytix.knowledge.rag.assemble import ABSTAIN_SENTINEL
        parser = self.extractor
        if not hasattr(parser, "head_candidates"):
            raise ValueError("retrieve_mode='head_facts' needs "
                             "query_parse_mode='geometric'")

        def _abstain(notice: str, sources=None) -> RagAnswer:
            return self._finalize(RagAnswer(
                answer=_ABSTAIN_MSG, confidence=0.0, decision="abstain",
                route="head_facts", verified=True, sources=sources or [],
                notice=notice), question, candidates=sources)

        cands = parser.head_candidates(question, top_k=self.rag.n_heads)
        if not cands or cands[0][1] < self.rag.head_bind_floor:
            return _abstain("No entity in the question binds to the graph.")
        heads = [h for h, s in cands if s >= self.rag.head_bind_floor]
        qts = parser.head_facts(heads, hops=self.rag.head_hops)
        facts = [self.retriever.fact(t.head, t.relation, t.tail) for t in qts]
        if not facts:
            return _abstain("The bound entity holds no facts.")

        answer = self.assembler.assemble(question, facts)
        if ABSTAIN_SENTINEL[:24].lower() in (answer or "").lower():
            return _abstain("No retrieved fact answers the question.", facts)

        confidence = self._confidence(facts)
        verification: dict = {}
        notice: str | None = None
        if self.verifier is not None:
            answer, confidence, notice, verification = self._self_verify(
                question, answer, facts, confidence)
            if verification.get("ok") is False and self.rag.on_verify_fail == "abstain":
                return self._finalize(RagAnswer(
                    answer=_ABSTAIN_MSG, confidence=0.0, decision="abstain",
                    route="head_facts", verified=True, sources=facts,
                    verification=verification,
                    notice="Answer failed GMS self-verification."),
                    question, candidates=facts)
        decision = ("accept" if confidence >= self.rag.accept_threshold
                    else "abstain")
        return self._finalize(RagAnswer(
            answer=answer if decision == "accept" else _ABSTAIN_MSG,
            confidence=confidence, decision=decision, route="head_facts",
            verified=True, query_triples=qts, sources=facts,
            verification=verification, notice=notice), question, candidates=facts)

    # -- fact-search route ---------------------------------------------------

    def _query_facts(self, question: str, generate: bool) -> RagAnswer:
        """Search every fact by meaning; answer from the ones the model cites, or
        abstain. Unrelated questions (below ``facts_floor``) never reach the model."""
        from knowlytix.knowledge.rag.assemble import ABSTAIN_SENTINEL
        found = self.fact_index.search(question, self.rag.top_k)
        if not found or 1.0 - found[0].score < self.rag.facts_floor:
            return self._facts_abstain(question, found, "No fact in the document is "
                                       "close to the question.")
        if not generate:
            return self._finalize(RagAnswer(
                answer="", confidence=self._confidence(found), decision="accept",
                route="facts", verified=False, sources=found), question)
        if self.rag.facts_arithmetic and (asked := detect(question)) is not None:
            derived = self._facts_arithmetic(question, found, asked)
            if derived is not None:
                return derived
        links: dict[int, list[Link]] = {}
        searched = found               # what search found: the nearest facts on an abstention
        if self.rag.facts_link_definitions:
            found, links = self.fact_index.link_definitions(question, found)
        notes = {i: f'defines the term "{ls[0].term}" used in '
                    f'{", ".join(f"[{ln.parent + 1}]" for ln in ls)}; cite both'
                 for i, ls in links.items()}
        answer = self.assembler.assemble(question, found, notes=notes) or ""
        if ABSTAIN_SENTINEL[:24].lower() in answer.lower():
            return self._facts_abstain(question, searched,
                                       "No retrieved fact answers the question.")
        used = cited(answer, len(found))
        if not used:
            return self._facts_abstain(question, searched,
                                       "The answer did not cite a retrieved fact.")
        answer, used = self._add_definitions(renumber(answer, used), used, found, links)
        sources = [found[i] for i in used]
        return self._facts_answer(question, answer, sources, self._ranked(sources))

    @staticmethod
    def _ranked(sources: list[RetrievedFact]) -> list[RetrievedFact]:
        """The sources an answer's confidence comes from: those search found, not the
        definitions added through the facts that use their terms (all of them when
        the answer cites only added ones)."""
        return [f for f in sources if f.source != LINK_SOURCE] or sources

    @staticmethod
    def _add_definitions(answer: str, used: list[int], found: list[RetrievedFact],
                         links: dict[int, list[Link]]) -> tuple[str, list[int]]:
        """Add the definition of a term a cited fact uses when the answer left it out:
        the defining fact's source sentence, cited by its number among the sources
        (``facts_link_definitions``). Runs after renumbering, so brackets in the
        sentence itself are left alone."""
        for child, parents in links.items():
            cited_parents = [found[ln.parent] for ln in parents if ln.parent in used]
            if cited_parents and child not in used:
                used = [*used, child]
                text = answer.rstrip()
                end = "" if not text or text[-1] in ".!?" else "."
                answer = (f"{text}{end} {link_sentence(found[child], cited_parents)} "
                          f"[{len(used)}].")
        return answer, used

    def _facts_arithmetic(self, question: str, found: list[RetrievedFact],
                          asked: Op) -> RagAnswer | None:
        """A difference or total of values the facts give (``facts_arithmetic``): the
        model picks the quantities, code checks their units and computes, and the
        answer cites every fact a value came from. None (the usual route answers)
        when the model picks nothing usable."""
        qs = quantities(question, found)
        from_question = needs_question_value(question)
        unit = asked_unit(question)         # "how many days" asks for days
        fact_units = {q.unit for q in qs if q.fact is not None}
        if not worth_asking(qs) or (from_question and not any(
                q.fact is None and q.unit in fact_units and unit in (None, q.unit)
                for q in qs)):
            return None        # nothing the model could combine: no call
        reply = self.assembler.llm.call(
            system=ARITH_SYSTEM, max_tokens=64,
            user=f"<quantities>\n{listing(qs)}\n</quantities>\n\nQuestion: {question}")
        parsed = parse_expression(reply or "", len(qs))
        result = compute(*parsed, qs, asked) if parsed else None
        if result is None or (unit is not None and result.unit != unit) or (
                from_question and all(qs[k].fact is not None for k in result.operands)):
            return None
        used = list(dict.fromkeys(qs[k].fact for k in result.operands if qs[k].fact is not None))
        if not used:
            return None        # every value from the question: nothing from the document
        cite = {f: n for n, f in enumerate(used, start=1)}
        sources = [found[i] for i in used]
        derived = {"op": result.op, "value": str(result.value), "unit": result.unit,
                   "operands": [{"text": qs[k].text, "value": str(qs[k].value),
                                 "fact": cite.get(qs[k].fact) if qs[k].fact is not None else None,
                                 "in_source": qs[k].in_source}
                                for k in result.operands]}
        return self._facts_answer(question, sentence(result, qs, cite), sources,
                                  self._ranked(sources), derived=derived)

    def _facts_answer(self, question: str, answer: str, sources: list[RetrievedFact],
                      ranked: list[RetrievedFact] | None = None,
                      derived: dict | None = None) -> RagAnswer:
        """An answer from cited facts, through self-verification when configured.
        Its confidence comes from ``ranked`` (the cited facts search found; default
        every source). ``derived`` records a computed answer in ``verification``."""
        confidence, notice, verification = self._confidence(ranked or sources), None, {}
        if derived is not None:
            # Written by code from values the cited facts state, computed in code: the
            # free-text verifier would read "30 days" as a claim against a fact saying
            # 60, so a computed answer is checked by construction instead.
            verification = {"method": "computed", "claims": [], "failed": [],
                            "derived": derived,
                            **({"ok": True} if self.verifier is not None else {})}
        elif self.verifier is not None:
            answer, confidence, notice, verification = self._self_verify(
                question, answer, sources, confidence)
            if verification.get("ok") is False and self.rag.on_verify_fail == "abstain":
                return self._facts_abstain(question, sources, "Answer failed GMS "
                                           "self-verification.", verification)
        decision = "accept" if confidence >= self.rag.accept_threshold else "abstain"
        return self._finalize(RagAnswer(
            answer=answer if decision == "accept" else _ABSTAIN_MSG,
            confidence=confidence, decision=decision, route="facts", verified=True,
            sources=sources, verification=verification, notice=notice),
            question, candidates=sources)

    def _facts_abstain(self, question: str, found: list[RetrievedFact], notice: str,
                       verification: dict | None = None) -> RagAnswer:
        return self._finalize(RagAnswer(
            answer=_ABSTAIN_MSG, confidence=0.0, decision="abstain", route="facts",
            verified=True, notice=notice, verification=verification or {}),
            question, candidates=found)

    # -- numeric-order route ------------------------------------------------

    _HIGHEST_CUES = ("highest", "largest", "greatest", "maximum", "max ", "most ",
                     "biggest", "steepest", "top ", "dearest", "priciest")
    _LOWEST_CUES = ("lowest", "smallest", "least", "minimum", "min ", "fewest",
                    "cheapest", "lowest-cost", "bottom ")

    def _which_extreme(self, q: str) -> str | None:
        ql = f" {q.lower()} "
        hi = any(c in ql for c in self._HIGHEST_CUES)
        lo = any(c in ql for c in self._LOWEST_CUES)
        if hi == lo:            # neither, or ambiguous both -> not a superlative
            return None
        return "highest" if hi else "lowest"

    def _match_enm_type(self, question: str) -> str | None:
        """Pick the ENM type the question is about (encoder cosine, or substring
        fallback when no encoder is configured)."""
        if self._enm_type_emb is not None:
            import torch
            qv = torch.nn.functional.normalize(
                torch.as_tensor(self.rag.encoder([question]), dtype=torch.float32),
                p=2, dim=-1)[0]
            sims = self._enm_type_emb @ qv
            best = int(sims.argmax())
            if float(sims[best]) >= self.rag.numeric_order_floor:
                return self._enm_types[best]
            return None
        ql = question.lower()
        hits = [t for t in self._enm_types
                if t.replace("_", " ") in ql or t in ql]
        return hits[0] if len(hits) == 1 else None

    def _followon_relation(self, question: str, base_entity: str):
        """The asserted (head, rel, tail) triples of ``base_entity`` the question
        asks to list. If a relation matches the question, restrict to it; else
        return all of the entity's asserted triples (generic "its relationships").
        """
        triples = [(h, r, t) for h, r, t in self.store.query_triples(head=base_entity)
                   if r not in _SCAFFOLD_RELATIONS]
        if not triples or self.rag.encoder is None:
            return triples
        import torch
        rels = sorted({r for _, r, _ in triples})
        emb = torch.nn.functional.normalize(
            torch.as_tensor(self.rag.encoder([_relation_phrase(r) for r in rels]),
                            dtype=torch.float32), p=2, dim=-1)
        qv = torch.nn.functional.normalize(
            torch.as_tensor(self.rag.encoder([question]), dtype=torch.float32),
            p=2, dim=-1)[0]
        sims = emb @ qv
        best = int(sims.argmax())
        if float(sims[best]) >= self.rag.numeric_order_floor:
            return [(h, r, t) for h, r, t in triples if r == rels[best]]
        return triples

    def _try_numeric_order(self, question: str) -> RagAnswer | None:
        """Answer a superlative question by ranking an ENM type in u-space.

        Returns ``None`` (fall through to the triple route) unless the question
        is a superlative whose target ENM type is confidently identified.
        """
        import math
        which = self._which_extreme(question)
        if which is None:
            return None
        enm_type = self._match_enm_type(question)
        if enm_type is None:
            return None
        res = self.store.enm_extreme(enm_type, which)
        if res is None:
            return None

        # The values are exact, so the winner is ambiguous only when there is no
        # distinct runner-up (all entries equal -> margin 0). A near-but-distinct
        # runner-up is NOT a tie; the exact values already separate them.
        if res.margin <= self.rag.numeric_order_tie_margin:
            tied = RetrievedFact(head=res.entity.split("/")[0], relation=enm_type,
                                 tail=f"{res.value:g}", score=0.0, confidence=1.0,
                                 source="enm")
            return RagAnswer(
                answer=_ABSTAIN_MSG, confidence=0.0, decision="abstain",
                route="numeric_order", verified=True, nearest_facts=[tied],
                notice=(f"No distinct {which} '{enm_type}' value "
                        "(entries are equal); cannot single one out."))

        base = res.entity.split("/")[0]
        related = self._followon_relation(question, base)
        sources = [RetrievedFact(head=base, relation=enm_type,
                                 tail=f"{res.value:g}", score=0.0,
                                 confidence=1.0, source="enm")]
        sources += [RetrievedFact(head=h, relation=r, tail=t, score=0.0,
                                  confidence=1.0, source="triple")
                    for h, r, t in related]

        answer = (f"The {which} '{enm_type}' value is {base} "
                  f"at {res.value:g}.")
        if related:
            rels = sorted({r for _, r, _ in related})
            tails = ", ".join(sorted({t for _, _, t in related}))
            rel_label = "/".join(rels)
            answer += f" Its {rel_label} relationships: {tails}."

        confidence = 1.0 - math.exp(-res.margin)
        return RagAnswer(
            answer=answer, confidence=confidence, decision="accept",
            route="numeric_order", verified=True, sources=sources,
            notice=None)

    def _extract_and_bind(self, question: str):
        """Extract query triples and bind them; on a bind failure re-ask the
        extractor once (with the graph vocab + the terms that didn't match).

        This repair loop is the main extraction-reliability lever for small
        models: the first pass often guesses a relation/entity that doesn't
        exist; the retry, grounded in the actual schema, usually fixes it.
        """
        qts = self.extractor.extract(question)
        bound = [self.binder.bind(qt) for qt in qts]
        attempts = 0
        while (qts and not all(b.bound for b in bound)
               and attempts < self.rag.max_extract_retries):
            attempts += 1
            qts = self.extractor.extract(question, hint=self._repair_hint(bound))
            bound = [self.binder.bind(qt) for qt in qts]
        return qts, bound

    def _repair_hint(self, bound: list[BoundTriple]) -> str:
        failed: list[str] = []
        for b in bound:
            o = b.original
            if not is_var(o.head) and b.head is None:
                failed.append(o.head)
            if not is_var(o.relation) and b.relation is None:
                failed.append(o.relation)
            if not is_var(o.tail) and b.tail is None:
                failed.append(o.tail)
        rels = ", ".join((self._vocab or {}).get("relations", [])[:60])
        return (f"These terms did not match the knowledge graph: {failed}. "
                f"Re-express the query using ONLY these relations: {rels}.")

    def _fallback_or_abstain(self, question, qts, bound, reason: str) -> RagAnswer:
        """Dense fallback when enabled and not strict; otherwise abstain.

        Dense answers are quarantined: ``verified=False``, ``route=
        "dense_fallback"``, and a notice that they are not GMS-verified.
        """
        if self.dense is None or self.rag.strict_mode:
            return self._abstain(qts, bound, reason=reason)
        hits = self.dense.search(question, top_k=self.rag.top_k)
        if not hits:
            return self._abstain(qts, bound, reason=reason)
        spans = [s for s, _ in hits]
        answer = self.assembler.assemble_passages(question, [s.text for s in spans])
        return RagAnswer(
            answer=answer, confidence=float(hits[0][1]), decision="accept",
            route="dense_fallback", verified=False, query_triples=qts,
            bound_triples=bound, dense_sources=spans,
            notice="Answer from dense fallback; NOT GMS-verified.",
        )

    def _finalize(self, ans: RagAnswer, question: str | None = None, *,
                  candidates: list[RetrievedFact] | None = None,
                  heads: list[str] | None = None) -> RagAnswer:
        """Fill ``nearest_facts`` on an abstention, emit the audit record (if a sink
        is configured), and return. The single place nearest facts are computed, so
        every abstain path gets them: callers only pass what they retrieved
        (``candidates``) or the entity they rejected (``heads``)."""
        if ans.decision == "abstain" and not ans.nearest_facts:
            ans.nearest_facts = self._nearest(question, ans.bound_triples, candidates, heads)
        if self.rag.audit_sink is not None:
            self.rag.audit_sink(ans.audit())
        return ans

    def coverage(self):
        """Coverage report for this store — measurable triple-mediation blind spots."""
        from knowlytix.knowledge.rag.coverage import coverage_report
        return coverage_report(self.store)

    def _self_verify(self, question: str, answer: str,
                     facts: list[RetrievedFact], confidence: float):
        """Run GMS self-verification; apply ``on_verify_fail``.

        Returns ``(answer, confidence, notice, verification_dict)``. The
        ``abstain`` action is handled by the caller (it changes control flow).
        """
        report = self.verifier.verify(answer, sources=facts)
        if report.ok:
            return answer, confidence, None, report.as_dict()

        action = self.rag.on_verify_fail
        if action == "regenerate":
            answer = self.assembler.assemble(question, facts)
            report = self.verifier.verify(answer, sources=facts)
            if report.ok:
                return answer, confidence, None, report.as_dict()
        # annotate (and regenerate-still-failing): surface the contradiction,
        # downgrade confidence so the accept threshold can catch it.
        failed = ", ".join(f"{t[0]}.{t[1]}={t[2]}" for t in
                           report.as_dict()["failed"])
        notice = (f"GMS self-verification flagged unsupported claims: {failed}. "
                  "Answer not fully grounded.")
        return answer, 0.0, notice, report.as_dict()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _confidence(facts: list[RetrievedFact]) -> float:
        if not facts:
            return 0.0
        return min(f.confidence for f in facts)

    def _nearest(self, question: str | None, bound: list[BoundTriple],
                 candidates: list[RetrievedFact] | None = None,
                 heads: list[str] | None = None) -> list[RetrievedFact]:
        """The document's facts closest to a question the pipeline could not answer.

        Retrieved ``candidates`` count only if the document asserts them: the score
        route echoes the question's own triple and link prediction proposes tails,
        neither of which the document says. Otherwise the facts of the question's
        entities (``heads``, else every head that resolved, even when its relation
        did not), ranked toward the question. At most ``top_k``."""
        asserted = [f for f in candidates or [] if self._asserted(f)]
        if asserted:
            return sorted(asserted, key=lambda f: f.score)[: self.rag.top_k]
        if heads is None:
            heads = [b.head for b in bound
                     if b.head is not None and not is_var(b.original.head)]
        triples = [(h, r, t) for head in dict.fromkeys(heads)
                   for h, r, t in self.store.query_triples(head=head)
                   if r not in _SCAFFOLD_RELATIONS]
        hints = {b.relation for b in bound if b.relation}
        ranked = self._rank_toward(question, triples, hints)[: self.rag.top_k]
        return [self.retriever.fact(h, r, t) for h, r, t in ranked]

    def _asserted(self, fact: RetrievedFact) -> bool:
        """Whether the document states ``fact`` (an asserted edge or an ENM value)."""
        return fact.source == "enm" or bool(self.store.query_triples(
            head=fact.head, relation=fact.relation, tail=fact.tail))

    def _rank_toward(self, question: str | None, triples: list[tuple[str, str, str]],
                     hints: set[str]) -> list[tuple[str, str, str]]:
        """Order triples by how close their relation is to the question: encoder
        similarity when one is configured, else the question's bound relations first.
        Stable, so ties keep the store's order."""
        rels = sorted({r for _, r, _ in triples})
        if not rels:
            return triples
        if self.rag.encoder is not None and question:
            import torch
            emb = torch.nn.functional.normalize(torch.as_tensor(
                self.rag.encoder([_relation_phrase(r) for r in rels]), dtype=torch.float32),
                p=2, dim=-1)
            qv = torch.nn.functional.normalize(torch.as_tensor(
                self.rag.encoder([question]), dtype=torch.float32), p=2, dim=-1)[0]
            closeness = dict(zip(rels, (emb @ qv).tolist()))
        else:
            closeness = {r: float(r in hints) for r in rels}
        return sorted(triples, key=lambda tr: -closeness[tr[1]])

    def _abstain(self, qts, bound, *, reason: str,
                 verification: dict | None = None) -> RagAnswer:
        notice = {
            "no_binding": "Question did not bind to known entities/relations.",
            "no_match": "No grounded facts matched the query.",
            "verify_fail": "Answer failed GMS self-verification (unsupported claims).",
            "irrelevant": "The entity does not hold the attribute the question asked about.",
        }[reason]
        return RagAnswer(
            answer=_ABSTAIN_MSG, confidence=0.0, decision="abstain",
            route="triple", verified=True, query_triples=qts,
            bound_triples=bound, notice=notice,
            verification=verification or {},
        )
