# SPDX-License-Identifier: Apache-2.0
"""Triple completeness: how completely does an LLM-generated text restate a set
of ground-truth triples?

This is the recall counterpart to claim correctness, expressed in the GMS's own
vocabulary instead of string overlap. Given

  * a piece of generated text (an LLM answer, a draft, a summary), and
  * a set of ground-truth ``(head, relation, tail)`` triples,

it parses-and-binds the text to GMS triples with the SAME
:class:`~knowlytix.harness.testing.claims.GeometricClaimExtractor` the
correctness path uses (entity/alias binding + numeric value-equality), then
reports how many ground-truth triples the text actually asserts. A value only
counts when it is bound to the right head -- no context-free number matching, no
bag-of-words overlap.

The ground truth is supplied by the caller, so the API is reusable across
sources:

  * a RAG's RETRIEVED facts -- ``triples_from_rag_answer(rag_answer)`` -- to ask
    "did the generated answer faithfully restate everything the retriever
    grounded on?";
  * a golden triple set, to ask "did the agent cover the facts we expect?";
  * any ``[(head, relation, tail)]`` list.

Example::

    from knowlytix.harness.testing.triple_completeness import (
        TripleCompletenessScorer, triples_from_rag_answer)

    scorer = TripleCompletenessScorer(store)
    ans = retriever.pipe.query(question)               # a RagAnswer
    report = scorer.score(ans.answer, triples_from_rag_answer(ans))
    print(report.score, report.missing)
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TripleCompletenessReport",
    "TripleCompletenessScorer",
    "triples_from_rag_answer",
]

Triple = "tuple[str, str, str]"


def _is_number(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _tail_eq(a, b) -> bool:
    """Numeric value-equality (``35`` == ``35.0``); else case/space-insensitive."""
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _spans(text: str) -> "list[str]":
    """Unigram + bigram spans of the answer, for span-level value matching."""
    import re
    toks = re.findall(r"[A-Za-z0-9.$]+", (text or "").lower())
    return list(dict.fromkeys(
        toks + [f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)]))


def _tension(cos: float) -> float:
    """Tension energy ``2 sin(theta/2)`` from a cosine (unit vectors)."""
    import math
    return 2.0 * math.sin(math.acos(max(-1.0, min(1.0, cos))) / 2.0)


@dataclass
class TripleCompletenessReport:
    """Coverage of the ground-truth triples by an LLM-generated text."""

    expected: int = 0                       # ground-truth triples
    covered: int = 0                        # of those, how many the text asserts
    produced: int = 0                       # triples the text asserted (any)
    missing: list = field(default_factory=list)   # ground-truth triples absent

    @property
    def score(self) -> float:
        """Recall in [0, 1]; 1.0 when nothing was expected (vacuously complete)."""
        return 1.0 if self.expected == 0 else self.covered / self.expected

    def as_dict(self) -> dict:
        return {
            "expected": self.expected,
            "covered": self.covered,
            "produced": self.produced,
            "missing": [list(m) for m in self.missing],
            "score": round(self.score, 4),
        }


class TripleCompletenessScorer:
    """Score triple completeness of generated text against ground-truth triples.

    Parameters
    ----------
    store : GMSExpertStore
        The trained store whose vocabulary the text is bound against.
    encoder : optional
        A document-tuned encoder for the underlying
        :class:`GeometricClaimExtractor` (e.g. a GEODE embed-loop encoder). When
        omitted the extractor uses graph adjacency + the store's alias lexicon.
    """

    def __init__(self, store, encoder=None, *, u_encoder=None,
                 tau_contra_per_relation=None, default_tau_contra=None):
        from knowlytix.harness.testing.claims import GeometricClaimExtractor
        self.store = store
        self._extractor = GeometricClaimExtractor(store, encoder=encoder)
        # Dual-embedding value check (optional): v/string binds head+relation; the
        # GEODE-calibrated contradiction (u) encoder verifies the VALUE. A value
        # that ENTAILS the expected tail (u-tension within the relation's
        # tau_contra) counts as covered; a contradiction (optional vs required,
        # denied vs issued) does not. Thresholds are GEODE's, not ours.
        self._u = u_encoder
        self._tau_rel = tau_contra_per_relation or {}
        self._tau_def = default_tau_contra

    def _u_value_present(self, text, tail, relation) -> bool:
        """True if an answer span entails the expected tail in u-space (tension to
        it within the relation's GEODE-calibrated tau_contra). Numerics are left
        to exact string value-equality."""
        if self._u is None or _is_number(tail):
            return False
        cut = self._tau_rel.get(relation, self._tau_def)
        if cut is None:
            return False
        spans = _spans(text)
        if not spans:
            return False
        import torch
        import torch.nn.functional as F

        def vec(ts):
            return F.normalize(torch.as_tensor(self._u(ts), dtype=torch.float32),
                               dim=-1)
        tv = vec([str(tail).replace("_", " ")])[0]
        sims = (vec(spans) @ tv).tolist()
        return min(_tension(c) for c in sims) <= cut

    def parse(self, text: str) -> "set[Triple]":
        """Parse-and-bind text into the GMS triples it asserts (the produced set).

        Reuses the answer-driven :class:`GeometricClaimExtractor`: a triple is
        emitted only where the text names a real store edge's head entity and
        states its tail (value-equal or alias-bound)."""
        return {(c.head, c.relation, str(c.tail))
                for c in self._extractor.extract(text or "", [], {})}

    def score(self, text: str, ground_truth) -> TripleCompletenessReport:
        """Completeness = ground-truth triples present in the text's produced set.

        ``ground_truth`` is any iterable of ``(head, relation, tail)``."""
        produced = self.parse(text)
        gt: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for trip in ground_truth or []:
            h, r, t = str(trip[0]), str(trip[1]), str(trip[2])
            if (h, r, t) not in seen:
                seen.add((h, r, t))
                gt.append((h, r, t))
        low = (text or "").lower()
        covered, missing = 0, []
        for (h, r, t) in gt:
            string_hit = any(ph == h and pr == r and _tail_eq(t, pt)
                             for (ph, pr, pt) in produced)
            # u-space recovery: head bound (v/string) AND value entails the tail
            # within the relation's calibrated tau_contra.
            u_hit = (not string_hit
                     and self._extractor._mentions_entity(h, low)
                     and self._u_value_present(text, t, r))
            if string_hit or u_hit:
                covered += 1
            else:
                missing.append((h, r, t))
        return TripleCompletenessReport(expected=len(gt), covered=covered,
                                        produced=len(produced), missing=missing)


def triples_from_rag_answer(rag_answer) -> "list[Triple]":
    """Ground-truth triples = the RAG's RETRIEVED facts (``RagAnswer.sources``).

    Lets the RAG-faithfulness case be one call: did the generated ``answer``
    restate everything the retriever grounded on?"""
    return [(f.head, f.relation, str(f.tail))
            for f in (getattr(rag_answer, "sources", None) or [])]
