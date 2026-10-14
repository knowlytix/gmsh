# SPDX-License-Identifier: Apache-2.0
"""GMS self-verification of LLM output — a hallucination detector.

Decompose the synthesized answer into claim triples and check each against the
GMS (``GEODE_RAG_DESIGN.md`` §7). Two calibration-free signals:

* **Contradiction** — the answer asserts ``(h, r, t)`` but the graph asserts
  ``(h, r, t')`` with ``t' != t``. Exact and threshold-free; catches an LLM that
  changed a value.
* **Admissibility** — for cap-trained stores, an unasserted claim is implausible
  when its geodesic distance exceeds the learned per-relation cap radius
  ``rho_r`` (a threshold the model learned, not a hand-set constant).

Claims that touch nothing in the graph are ``unverifiable`` (advisory, not a
failure). This turns the GMS into a faithfulness check on generated text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from knowlytix.knowledge.geode.provenance import value_canon
from knowlytix.knowledge.rag.binding import TripleBinder
from knowlytix.knowledge.rag.query_triples import QueryTriple, QueryTripleExtractor

__all__ = ["ClaimVerdict", "VerifyReport", "AnswerVerifier"]

_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")

_SYSTEM = (
    "Extract the factual claims in the text as knowledge-graph triples. Output "
    "ONLY a JSON array of objects with keys head, relation, tail, using concrete "
    "values (no variables). Relations are lowercase snake_case. Include only "
    "claims the text actually asserts."
)

# Phrase-form decomposition prompt. With an embedding (tuned) binder, the relation
# is resolved geometrically, so the LLM should name the relation in the words the
# text uses ("reported revenue") rather than guess a canonical snake_case token --
# guessing snaps incidental tokens ("in FY2025" -> has_fy2025) onto real relations
# the head does not hold, which the (head, relation) lookup then cannot contradict.
# The tuned binder maps the phrase to the canonical relation; the fuzzy binder
# cannot, so the canonical-token hint below is kept for the fuzzy path.
_SYSTEM_PHRASE = (
    "Extract the factual claims in the text as knowledge-graph triples. Output "
    "ONLY a JSON array of objects with keys head, relation, tail, using concrete "
    "values (no variables). Phrase the relation in the words the text uses for the "
    "attribute (for example 'reported revenue', 'headcount'); do not rename it to a "
    "code or guess a different attribute. Include only claims the text actually "
    "asserts."
)


def _value_grounded(claim_tail: str, fact_tail: str) -> bool:
    """Value match tolerant of currency/separator formatting: ``$330.0`` grounds
    ``330.0`` numerically, falling back to :func:`value_canon` for strings."""
    if _values_match(claim_tail, fact_tail):
        return True
    from knowlytix.benchmark.ingest import parse_financial_number
    a, b = parse_financial_number(str(claim_tail)), parse_financial_number(str(fact_tail))
    return a is not None and b is not None and abs(a - b) < 1e-6


def _values_match(a: str, b: str) -> bool:
    """Numeric-aware equality: compare as floats when both parse, else compare
    with :func:`value_canon` (separators unified, so a prose answer value matches
    the graph's stored slug)."""
    ma, mb = _NUM_RE.search(a.replace(",", "")), _NUM_RE.search(b.replace(",", ""))
    if ma and mb:
        try:
            return abs(float(ma.group()) - float(mb.group())) < 1e-6
        except ValueError:
            pass
    return value_canon(a) == value_canon(b)


def _surface_match(term: str, text_lower: str) -> bool:
    """Word-boundary match of an entity name/alias, snake/spaced tolerant."""
    t = str(term).strip().lower()
    if len(t) <= 1:
        return False
    forms = {t, t.replace("_", " ")}
    return any(re.search(r"\b" + re.escape(f) + r"\b", text_lower) for f in forms)


def _tail_in_text(tail: str, text: str, text_lower: str) -> bool:
    """A numeric tail matches on value (35.0 == '$35'); else word-boundary."""
    s = str(tail).strip()
    try:
        val = float(s)
    except ValueError:
        return _surface_match(s, text_lower)
    for m in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            if abs(float(m.replace(",", "")) - val) < 1e-9:
                return True
        except ValueError:
            continue
    return False


@dataclass
class ClaimVerdict:
    triple: QueryTriple
    status: str          # supported | contradicted | implausible | uncertain | unverifiable
    detail: str = ""

    @property
    def failed(self) -> bool:
        # "uncertain" is a deferral (the fused value check could not place the
        # value): bank-grade, an unconfirmable value claim must not ship, so it
        # blocks acceptance like a contradiction (the caller abstains / escalates).
        return self.status in ("contradicted", "implausible", "uncertain")


@dataclass
class VerifyReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(v.failed for v in self.verdicts)

    @property
    def failures(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.failed]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "claims": [{"triple": v.triple.as_tuple(), "status": v.status,
                        "detail": v.detail} for v in self.verdicts],
            "failed": [v.triple.as_tuple() for v in self.failures],
        }


class AnswerVerifier:
    """Check an answer's claim triples against the GMS."""

    def __init__(self, store, llm, *, binder: TripleBinder | None = None,
                 max_tokens: int = 512, mode: str = "llm",
                 u_encoder=None, v_encoder=None,
                 tau_polarity: float = 0.9, polarity_v_floor: float = 0.30,
                 value_checker=None):
        self.store = store
        self.llm = llm
        self.binder = binder or TripleBinder(store)
        self.max_tokens = max_tokens
        # Optional fused categorical-value checker (ValuePolarityChecker): cap
        # plausibility + v-space resolution + 3-class u-space polarity, returning
        # supported / contradicted / uncertain. When set it decides a categorical
        # claim's value (synonym-tolerant, polarity-aware, abstain-on-uncertain);
        # otherwise the geometric check falls back to cap admissibility alone.
        self._value_checker = value_checker
        # "llm": parse the answer into claim triples with an LLM (legacy default).
        # "geometric": extract them by walking the store graph (no LLM) so claims
        # bind to canonical vocabulary instead of LLM-invented relations.
        # "hybrid": BOTH -- geometric claims confirm the answer states real edges
        # (grounding coverage), and LLM-decomposed, VOCAB-CONSTRAINED claims carry
        # the answer's ASSERTED value, so a value/polarity REVERSAL the geometric
        # extractor misses (it keys on the STORED tail appearing in text, which a
        # flipped answer never contains) surfaces as a claim that binds to a real
        # (head, relation) with a mismatching tail -> _check returns "contradicted".
        # Invented relations bind to nothing -> "unverifiable" (advisory, never a
        # failure), so the LLM pass can only ADD a contradiction, not false-fail.
        self.mode = mode
        self._adj = None  # lazy adjacency for geometric mode
        self._vocab = None  # lazy graph vocabulary for the llm decomposition
        # u-space POLARITY check (optional). The lexical/cap checks above catch a
        # changed VALUE only when the answer quotes a value; they miss a polarity
        # REVERSAL stated in prose ("PII can be sent" against a stored
        # has_unencrypted_channel_pii=forbidden, where the answer never quotes
        # "forbidden"). The contradiction (u) encoder places opposite polarities at
        # high tension, so for each grounding fact the answer is ABOUT (v-space
        # near its relation), tension between the answer and the stored fact above
        # tau_polarity flags a contradiction the value match cannot see.
        self._u = u_encoder
        self._v = v_encoder
        self.tau_polarity = tau_polarity
        self.polarity_v_floor = polarity_v_floor

    def verify(self, answer: str, sources=None) -> VerifyReport:
        """Decompose an answer into claim triples and check each against the GMS.

        Args:
            answer: the synthesized answer to verify.
            sources: grounding facts for the optional u-space polarity check.

        Returns:
            A :class:`VerifyReport` with one :class:`ClaimVerdict` per claim plus any
            polarity contradictions.
        """
        if self.mode == "geometric":
            claims = self._geometric_claims(answer)
        elif self.mode == "hybrid":
            claims = self._dedup(self._geometric_claims(answer)
                                 + self._llm_claims(answer))
        else:
            claims = self._llm_claims(answer)
        verdicts = [self._check(c) for c in claims]
        if self._u is not None and sources:
            verdicts.extend(self._polarity_checks(answer, sources))
        return VerifyReport(verdicts)

    def verify_batch(self, answers: "list[str]",
                     sources_list=None) -> "list[VerifyReport]":
        """Verify many answers at once, decomposing them into claim triples in a
        single batched LLM sweep (:meth:`LLMBackend.call_batch`) rather than one
        :meth:`call` per answer. Equivalent, answer-for-answer, to mapping
        :meth:`verify` over ``answers``: only the LLM claim-decomposition stage is
        batched, while the geometric extraction, the per-claim checks and the
        optional polarity check are LLM-free and run per answer. ``sources_list``,
        if given, supplies the grounding facts for the per-answer polarity check
        (same length as ``answers``)."""
        answers = list(answers)
        srcs = sources_list if sources_list is not None else [None] * len(answers)
        llm_all = (self._llm_claims_batch(answers)
                   if self.mode in ("llm", "hybrid")
                   else [[] for _ in answers])
        reports = []
        for answer, sources, llm_claims in zip(answers, srcs, llm_all):
            if self.mode == "geometric":
                claims = self._geometric_claims(answer)
            elif self.mode == "hybrid":
                claims = self._dedup(self._geometric_claims(answer) + llm_claims)
            else:
                claims = llm_claims
            verdicts = [self._check(c) for c in claims]
            if self._u is not None and sources:
                verdicts.extend(self._polarity_checks(answer, sources))
            reports.append(VerifyReport(verdicts))
        return reports

    @staticmethod
    def _dedup(claims: "list[QueryTriple]") -> "list[QueryTriple]":
        out, seen = [], set()
        for c in claims:
            k = (c.head.lower(), c.relation.lower(), str(c.tail).lower())
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out

    def _claim_system(self) -> str:
        """Build the claim-decomposition system prompt from the store vocabulary.
        With an embedding (tuned) binder the relation is resolved geometrically, so
        the LLM names the relation in the text's own words and the binder maps it to
        the canonical relation; with the fuzzy binder, which cannot bridge a
        paraphrase, the LLM is steered toward the graph's exact relation names."""
        from knowlytix.knowledge.rag.query_triples import schema_from_store
        if self._vocab is None:
            self._vocab = schema_from_store(self.store)
        ents = ", ".join(self._vocab.get("entities", [])[:80])
        tuned = getattr(self.binder, "mode", "fuzzy") == "embedding" and \
            getattr(self.binder, "_encoder", None) is not None
        if tuned:
            # Geometric relation resolution: ask for the text's phrasing, let the
            # tuned binder canonicalize it. Forcing snake_case tokens here makes the
            # LLM snap incidental tokens onto wrong-but-real relations.
            system = _SYSTEM_PHRASE
        else:
            system = _SYSTEM
            rels = ", ".join(self._vocab.get("relations", [])[:60])
            if rels:
                system += ("\nPrefer these exact relation names when one matches the "
                           f"attribute the claim is about: {rels}.")
        if ents:
            system += f"\nKnown entities include: {ents}."
        return system

    def _llm_claims(self, answer: str) -> "list[QueryTriple]":
        """Decompose one answer into claim triples with the LLM. Each claim carries
        the value the answer ASSERTS (not the stored value), so a reversal is
        checkable."""
        raw = self.llm.call(system=self._claim_system(), user=answer,
                            max_tokens=self.max_tokens)
        return QueryTripleExtractor.parse(raw)

    def _llm_claims_batch(self, answers: "list[str]") -> "list[list[QueryTriple]]":
        """Decompose many answers into claim triples in ONE batched LLM sweep
        (:meth:`LLMBackend.call_batch`), rather than a per-answer :meth:`call`. The
        system prompt is shared across the batch; returns one claim list per
        answer, in order."""
        raws = self.llm.call_batch(list(answers), system=self._claim_system(),
                                   max_tokens=self.max_tokens)
        return [QueryTripleExtractor.parse(r) for r in raws]

    @staticmethod
    def _clauses(answer: str) -> "list[str]":
        """Split an answer into clauses, one assertion each. A synthesized answer
        states several facts in one sentence ("revenue is 120, headcount is 340,
        and it is in technology"); polarity must be judged per clause, because the
        u-space tension between the WHOLE answer and a SINGLE fact is inflated by
        the sibling clauses' content and reads a correct answer as a reversal."""
        parts = re.split(r"[.;\n]|,(?:\s+and\b)?|\s+and\b", answer)
        return [p.strip() for p in parts if p.strip()]

    def _polarity_checks(self, answer: str, sources) -> "list[ClaimVerdict]":
        """Per-clause u-space polarity check against the grounding facts. Each
        clause is matched to the single fact it is about (relation phrase nearest
        in v-space, clearing the v-floor); u-space tension between THAT CLAUSE and
        the stored fact statement above tau_polarity means the clause reverses the
        fact's polarity -- a contradiction the lexical/cap checks miss because the
        prose never quotes the stored value. Scoping to the clause (not the whole
        answer) is what keeps a correct multi-claim answer from false-failing."""
        import torch

        from knowlytix.knowledge.rag.relevance import _strip_has, _tension

        def _emb(enc, texts):
            return torch.nn.functional.normalize(
                torch.as_tensor(enc(texts), dtype=torch.float32), p=2, dim=-1)

        # Distinct grounding facts (drop structural relations).
        facts = []
        seen: set = set()
        for f in sources:
            h = getattr(f, "head", None)
            r = getattr(f, "relation", None)
            t = getattr(f, "tail", None)
            if not h or not r or t is None or r in ("in_section", "has_alias"):
                continue
            if (h, r, t) not in seen:
                seen.add((h, r, t))
                facts.append((h, r, t))
        if not facts:
            return []
        phrases = [_strip_has(r).replace("_", " ") for _h, r, _t in facts]
        clauses = self._clauses(answer) or [answer]
        # v-space: embed every clause and every relation phrase once.
        emb = _emb(self._v, clauses + phrases) if self._v is not None else None
        verdicts: list[ClaimVerdict] = []
        flagged: set = set()
        for ci, clause in enumerate(clauses):
            if emb is not None:
                sims = emb[len(clauses):] @ emb[ci]
                bi = int(sims.argmax())
                if float(sims[bi]) < self.polarity_v_floor:
                    continue  # this clause is about no grounding fact
            else:
                bi = 0
            h, r, t = facts[bi]
            if (h, r, t) in flagged:
                continue
            u = _emb(self._u, [clause, f"{phrases[bi]} {t}"])
            ten = _tension(float(u[0] @ u[1]))
            if ten > self.tau_polarity:
                flagged.add((h, r, t))
                verdicts.append(ClaimVerdict(
                    QueryTriple(h, r, str(t)), "contradicted",
                    f"u-space polarity tension {ten:.2f} > {self.tau_polarity:.2f} "
                    f"(clause reverses {h}.{r}={t})"))
        return verdicts

    def _check(self, claim: QueryTriple) -> ClaimVerdict:
        bt = self.binder.bind(claim)
        if bt.head is None or bt.relation is None:
            return ClaimVerdict(claim, "unverifiable",
                                "head/relation not in graph")
        # Geometric value check for a CATEGORICAL/polarity tail (route B): embed the
        # asserted value with the v-encoder and score it against the relation cap via
        # the operator geometry. Within the cap -> the value is admissible (supported),
        # even when it is a synonym ("prohibited" for stored "forbidden") a string
        # match would false-fail, or an out-of-vocabulary token the graph never stored
        # as an entity ("optional"). Outside the cap -> a polarity/value reversal
        # (contradicted). Numeric tails are left to the exact/numeric path below
        # (sentence encoders do not separate "120" from "999"); the check is skipped
        # when no v-encoder is set or the store is not cap-trained.
        geo = self._geometric_value_check(claim, bt)
        if geo is not None:
            return geo
        asserted = self.store.query_triples(head=bt.head, relation=bt.relation)
        if asserted:
            if any(_values_match(claim.tail, t) for _, _, t in asserted):
                return ClaimVerdict(claim, "supported")
            return ClaimVerdict(
                claim, "contradicted",
                f"graph asserts {sorted({t for _, _, t in asserted})}")
        # Relation-tolerant grounding. A small LLM often mislabels the relation
        # when it decomposes its own answer ("value" for "has_amount") while
        # quoting the right figure. If the head asserts the claimed tail value
        # under ANY relation, the answer's value is grounded in the graph. A
        # hallucinated value matches no asserted fact and still falls through to
        # the cap check; a lie on a known relation is already "contradicted" above.
        head_facts = self.store.query_triples(head=bt.head)
        if any(_value_grounded(claim.tail, t) for _, _, t in head_facts):
            return ClaimVerdict(claim, "supported", "value asserted for head")
        # Direction-tolerant grounding. Natural language states a real edge with
        # head and tail swapped relative to the graph -- "Sam Reyes heads
        # operations" for the stored (operations, has_head, sam reyes). If the
        # graph asserts an edge between the same two entities in the reverse
        # direction, the claim names a true relationship, just oriented as prose;
        # a fabricated pairing matches neither direction and falls to the cap.
        tail_ent = self.store.fuzzy_match_entity(claim.tail) if claim.tail else None
        if tail_ent is not None:
            if any(t == bt.head or _value_grounded(claim.head, t)
                   for _, _, t in self.store.query_triples(head=tail_ent)):
                return ClaimVerdict(claim, "supported",
                                    "asserted in reverse direction")
        return self._check_admissibility(claim, bt)

    def _geometric_value_check(self, claim: QueryTriple, bt):
        """Route B: score a categorical/polarity asserted value against the
        relation cap via the operator geometry (``store.score_triple_emb`` on the
        v-encoder embedding of the value). Returns a verdict, or ``None`` to defer
        to the exact/numeric path (numeric or empty tail, no v-encoder, store not
        cap-trained, or an unscorable embedding)."""
        tail = str(claim.tail).strip() if claim.tail is not None else ""
        if not tail or re.match(r"^-?[\d,]*\.?\d+%?$", tail):
            return None  # numeric/empty -> exact path (encoders don't separate numbers)
        if self._v is None or not hasattr(self.store, "score_triple_emb"):
            return None
        radius = self.store.cap_radius(bt.relation)
        if radius is None:
            return None
        # Fused path: cap + v-resolution + 3-class u-polarity vs the single stored
        # value (synonym-tolerant, polarity-aware, abstains as "uncertain").
        if self._value_checker is not None:
            stored = [t for _, _, t in
                      self.store.query_triples(head=bt.head, relation=bt.relation)]
            if len(stored) == 1:
                try:
                    v = self._value_checker.check(bt.head, bt.relation, tail, stored[0])
                except Exception:  # noqa: BLE001 - fall back to cap-only on any issue
                    v = None
                if v == "supported":
                    return ClaimVerdict(claim, "supported", "fused v+u: value consistent")
                if v == "contradicted":
                    return ClaimVerdict(claim, "contradicted",
                                        f"fused v+u: value contradicts stored {stored[0]!r}")
                if v == "uncertain":
                    return ClaimVerdict(claim, "uncertain",
                                        f"fused v+u: value vs stored {stored[0]!r} unresolved")
        # Cap-only fallback (no fused checker, or no single stored value).
        try:
            vec = self._v([tail])[0]
        except Exception:  # noqa: BLE001 - encoder issues defer to the exact path
            return None
        score = self.store.score_triple_emb(bt.head, bt.relation, vec)
        if score is None:
            return None  # dim mismatch / not scorable -> defer
        if score <= radius:
            return ClaimVerdict(claim, "supported",
                                f"value within cap (geodesic {score:.3f} <= {radius:.3f})")
        return ClaimVerdict(claim, "contradicted",
                            f"value off cap (geodesic {score:.3f} > radius {radius:.3f})")

    def _check_admissibility(self, claim: QueryTriple, bt) -> ClaimVerdict:
        radius = self.store.cap_radius(bt.relation)
        tail = bt.tail if bt.tail is not None else self.store.fuzzy_match_entity(
            claim.tail)
        if radius is None or tail is None:
            return ClaimVerdict(claim, "unverifiable",
                                "no asserted fact and not cap-verifiable")
        score = self.store.score_triple(bt.head, bt.relation, tail)
        if score is None:
            return ClaimVerdict(claim, "unverifiable", "not scorable")
        if score > radius:
            return ClaimVerdict(
                claim, "implausible",
                f"geodesic {score:.3f} > cap radius {radius:.3f}")
        return ClaimVerdict(claim, "supported", "within cap radius")

    # -- geometric (LLM-free) claim extraction --------------------------------
    def _adjacency(self):
        if self._adj is None:
            adj: dict = {}
            aliases: dict = {}
            for h, r, t in self.store.triples:
                if r == "has_alias":
                    aliases.setdefault(h, []).append(str(t))
                elif r != "in_section":
                    adj.setdefault(h, []).append((r, str(t)))
            self._adj = (adj, aliases)
        return self._adj

    def _geometric_claims(self, answer: str) -> "list[QueryTriple]":
        """Extract the answer's claims by walking the store graph -- no LLM. Emit a
        canonical ``(head, relation, tail)`` for every real edge whose head (or an
        alias) and tail the answer states, so a correct answer yields store-bound
        triples that verify, instead of the LLM-invented relations that false-fail.
        Multi-hop chains surface as their constituent edges (each a real triple)."""
        low = answer.lower()
        adj, aliases = self._adjacency()
        claims: list = []
        seen: set = set()
        for head, edges in adj.items():
            names = [str(head).replace("_", " ")] + list(aliases.get(head, []))
            if not any(_surface_match(n, low) for n in names):
                continue
            for rel, tail in edges:
                key = (head, rel, tail)
                if key in seen or not _tail_in_text(tail, answer, low):
                    continue
                seen.add(key)
                claims.append(QueryTriple(head=head, relation=rel, tail=tail))
        return claims
