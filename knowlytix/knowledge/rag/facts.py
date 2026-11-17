# SPDX-License-Identifier: Apache-2.0
"""Fact search for the ``facts`` retrieval route (``RagConfig.retrieve_mode="facts"``).

Graphs extracted from prose carry hundreds of one-off relations, so binding a
question to a single (head, relation) pair picks the wrong one. This route searches
every asserted fact, written with its source sentence, by meaning; the answering
model then answers only from the facts it cites. Helpers here keep the citations
honest: only numbers of retrieved facts count, renumbered in the order of sources.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from knowlytix.knowledge.rag.retrieve import RetrievedFact, Retriever

__all__ = ["LINK_SOURCE", "SCAFFOLD_RELATIONS", "FactIndex", "Link", "cited", "link_sentence",
           "renumber"]

# Structural edges (section membership, alias links), not facts about an entity.
SCAFFOLD_RELATIONS = frozenset({"in_section", "has_alias"})

Encoder = Callable[[list[str]], object]
_CITATION = re.compile(r"\[(\d+)\]")
_QUOTED = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{3,80})["\u201c\u201d]')
# Definition links (``RagConfig.facts_link_definitions``): only the first few facts
# are followed, at most this many defining facts per term, and a term that heads more
# facts than this is a topic of the document, not a term defined in one place.
LINK_SOURCE = "facts_link"   # a definition added through the fact that uses its term
LINK_PARENTS = 3
LINKS_PER_TERM = 2
MAX_DEFINING_FACTS = 5


@dataclasses.dataclass(frozen=True)
class Link:
    """A fact defining ``term``, a quoted term in the tail of the fact at ``parent``
    (positions in the searched list)."""

    parent: int
    term: str


def _canon(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _phrase(relation: str) -> str:
    """A relation as words: ``has_filing_deadline`` -> ``filing deadline``."""
    body = relation[4:] if relation.startswith("has_") else relation
    return body.replace("_", " ")


class FactIndex:
    """Every asserted fact with its source sentence, searchable by meaning. Built on
    the first search: encoding all facts takes a moment and most pipelines that
    never use this route should not pay it."""

    def __init__(self, store, retriever: Retriever, encoder: Encoder | None = None):
        self._store = store
        self._retriever = retriever
        self._encoder = encoder
        self._facts: list[RetrievedFact] | None = None
        self._emb: torch.Tensor | None = None
        self._evidence = None   # sentence-level provenance, loaded with the index
        self._by_head: dict[str, list[int]] | None = None
        self._last: tuple[str, torch.Tensor] | None = None

    def search(self, question: str, top_k: int) -> list[RetrievedFact]:
        """The ``top_k`` facts closest to ``question``, closest first. A fact's
        ``score`` is ``1 - cosine`` (lower is closer, as for other routes)."""
        facts, emb = self._index()
        if not facts:
            return []
        sims = self._similarities(question, emb)
        top = torch.topk(sims, min(top_k, len(facts))).indices.tolist()
        return [dataclasses.replace(facts[i], score=1.0 - float(sims[i]), source="facts",
                                    confidence=float(sims[i].clamp(min=0.0)))
                for i in top]

    def link_definitions(self, question: str, found: list[RetrievedFact]
                         ) -> tuple[list[RetrievedFact], dict[int, list[Link]]]:
        """``found`` with the facts that define its quoted terms, each placed right
        after the fact that uses the term, and the links by position.

        A term the document puts in quotes (e.g. an "appropriate law enforcement
        authority") is often defined by another fact, whose head is that term. Its
        meaning is far from the question, so search alone ranks it low; linked here,
        the answer can use it. Only quoted terms are followed, only from the first
        :data:`LINK_PARENTS` facts, and never a term that heads more than
        :data:`MAX_DEFINING_FACTS` facts (a topic, not a defined term). A defining
        fact already in ``found`` stays where it is and is linked there. An added
        one keeps its own score and is marked ``source="facts_link"``: it was found
        through the fact it defines, not by closeness to the question. Each linked
        position lists every fact whose term it defines.
        """
        facts, emb = self._index()
        if not facts:
            return list(found), {}
        by_head = self._heads(facts)
        sims = self._similarities(question, emb)
        key = lambda f: (f.head, f.relation, f.tail)   # noqa: E731
        present = {key(f) for f in found}
        out: list[RetrievedFact] = []
        wanted: list[tuple[int, str, tuple[str, str, str]]] = []   # (parent, term, fact)
        for pos, fact in enumerate(found):
            out.append(fact)
            parent = len(out) - 1
            if pos >= LINK_PARENTS:
                continue
            for term in dict.fromkeys(m.group(1).strip() for m in _QUOTED.finditer(fact.tail)):
                heads = by_head.get(_canon(term), [])
                if not heads or len(heads) > MAX_DEFINING_FACTS or _canon(term) == _canon(fact.head):
                    continue
                for i in sorted(heads, key=lambda i: -float(sims[i]))[:LINKS_PER_TERM]:
                    wanted.append((parent, term, key(facts[i])))
                    if key(facts[i]) not in present:
                        present.add(key(facts[i]))
                        out.append(dataclasses.replace(
                            facts[i], score=1.0 - float(sims[i]), source=LINK_SOURCE,
                            confidence=float(sims[i].clamp(min=0.0))))
        position = {key(f): n for n, f in enumerate(out)}
        links: dict[int, list[Link]] = {}
        for parent, term, k in wanted:
            child = position[k]
            if child != parent and parent not in [ln.parent for ln in links.get(child, [])]:
                links.setdefault(child, []).append(Link(parent=parent, term=term))
        return out, links

    def _similarities(self, question: str, emb: torch.Tensor) -> torch.Tensor:
        """Cosine of every fact to ``question``; the last question's are reused (a
        query searches, then links definitions, for one question)."""
        last = self._last          # one read: another thread may replace it meanwhile
        if last is None or last[0] != question:
            last = (question, emb @ self._encode([question])[0])
            self._last = last
        return last[1]

    def _heads(self, facts: list[RetrievedFact]) -> dict[str, list[int]]:
        if self._by_head is None:
            by_head: dict[str, list[int]] = {}      # built whole, then published
            for i, f in enumerate(facts):
                by_head.setdefault(_canon(f.head), []).append(i)
            self._by_head = by_head
        return self._by_head

    def _index(self) -> tuple[list[RetrievedFact], torch.Tensor]:
        if self._facts is None:
            graph = getattr(self._store, "doc_graph", None)
            triples = dict.fromkeys((h, r, str(t)) for h, r, t in graph.triples
                                    if r not in SCAFFOLD_RELATIONS) if graph else {}
            self._evidence = self._load_evidence()
            facts = [self._with_sentence(self._retriever.fact(h, r, t))
                     for h, r, t in triples]
            texts = [f"{f.head} {_phrase(f.relation)} {f.tail}. {f.raw or ''}"
                     for f in facts]
            # Published embeddings first, facts last: a concurrent first query that
            # sees the facts must also see their embeddings (a double build is harmless).
            emb = self._encode(texts) if texts else torch.empty(0)
            self._emb = emb
            self._facts = facts
            return facts, emb
        assert self._emb is not None   # published before the facts
        return self._facts, self._emb

    def _load_evidence(self):
        """The sentence-level provenance a prose build writes
        (``provenance.json`` as ``{"triples": [...]}``), or None. The table-aligned
        ledger has no span for facts extracted from prose."""
        path = Path(getattr(self._store, "store_path", None) or "") / "provenance.json"
        if not path.is_file() or not isinstance(json.loads(path.read_text()), dict):
            return None
        from knowlytix.core.ontodisc.kg.evidence import EvidenceIndex
        return EvidenceIndex.load(path.parent)

    def _with_sentence(self, fact: RetrievedFact) -> RetrievedFact:
        """The fact with its source sentence as ``raw``: the ledger's span, else the
        build's provenance, else the sentence of the document that grounds it."""
        if fact.raw:
            return fact
        sentences = (self._evidence.passages(fact.head, fact.relation, fact.tail)
                     if self._evidence is not None else [])
        sentence = sentences[0] if sentences else self._grounding_sentence(fact)
        return dataclasses.replace(fact, raw=sentence) if sentence else fact

    def _grounding_sentence(self, fact: RetrievedFact) -> str | None:
        from knowlytix.core.ontodisc.kg.ontology_score import Ontology, triple_evidence
        text = getattr(self._store, "markdown", "") or ""
        found = triple_evidence(fact.head, fact.relation, fact.tail, text,
                                Ontology.parse({})) if text else []
        return found[0]["sentence"] if found else None

    def _encode(self, texts: list[str]) -> torch.Tensor:
        if self._encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            self._encoder = encode_texts
        emb = torch.as_tensor(self._encoder(texts), dtype=torch.float32)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)


def cited(answer: str, n_facts: int) -> list[int]:
    """Zero-based indices of the retrieved facts ``answer`` cites as ``[n]``, in
    order of first citation. Numbers outside ``1..n_facts`` are ignored."""
    order: list[int] = []
    for m in _CITATION.finditer(answer):
        i = int(m.group(1)) - 1
        if 0 <= i < n_facts and i not in order:
            order.append(i)
    return order


def link_sentence(fact: RetrievedFact, parents: Sequence[RetrievedFact] = ()) -> str:
    """What a defining fact adds to an answer that left it out: its source sentence,
    as the document words it, else the fact itself (also when that sentence is one
    the answer's fact came from, so it isn't repeated). Nothing beyond the fact."""
    raw = fact.raw if fact.raw and fact.raw not in {p.raw for p in parents} else None
    text = (raw or f"{fact.head} {_phrase(fact.relation)} {fact.tail}").strip()
    return text[:1].upper() + text[1:].rstrip(". ")


def renumber(answer: str, order: Sequence[int]) -> str:
    """Rewrite citations to the position of each cited fact in ``order`` (the
    answer's sources), so ``[1]`` is the first source; others are dropped."""
    position = {i + 1: k + 1 for k, i in enumerate(order)}

    def sub(m: re.Match[str]) -> str:
        k = position.get(int(m.group(1)))
        return f"[{k}]" if k else ""
    return _CITATION.sub(sub, answer)
