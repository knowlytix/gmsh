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

__all__ = ["SCAFFOLD_RELATIONS", "FactIndex", "cited", "renumber"]

# Structural edges (section membership, alias links), not facts about an entity.
SCAFFOLD_RELATIONS = frozenset({"in_section", "has_alias"})

Encoder = Callable[[list[str]], object]
_CITATION = re.compile(r"\[(\d+)\]")


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

    def search(self, question: str, top_k: int) -> list[RetrievedFact]:
        """The ``top_k`` facts closest to ``question``, closest first. A fact's
        ``score`` is ``1 - cosine`` (lower is closer, as for other routes)."""
        facts, emb = self._index()
        if not facts:
            return []
        sims = emb @ self._encode([question])[0]
        top = torch.topk(sims, min(top_k, len(facts))).indices.tolist()
        return [dataclasses.replace(facts[i], score=1.0 - float(sims[i]), source="facts",
                                    confidence=float(sims[i].clamp(min=0.0)))
                for i in top]

    def _index(self) -> tuple[list[RetrievedFact], torch.Tensor]:
        if self._facts is None:
            graph = getattr(self._store, "doc_graph", None)
            triples = dict.fromkeys((h, r, str(t)) for h, r, t in graph.triples
                                    if r not in SCAFFOLD_RELATIONS) if graph else {}
            self._evidence = self._load_evidence()
            self._facts = [self._with_sentence(self._retriever.fact(h, r, t))
                           for h, r, t in triples]
            texts = [f"{f.head} {_phrase(f.relation)} {f.tail}. {f.raw or ''}"
                     for f in self._facts]
            self._emb = self._encode(texts) if texts else torch.empty(0)
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


def renumber(answer: str, order: Sequence[int]) -> str:
    """Rewrite citations to the position of each cited fact in ``order`` (the
    answer's sources), so ``[1]`` is the first source; others are dropped."""
    position = {i + 1: k + 1 for k, i in enumerate(order)}

    def sub(m: re.Match[str]) -> str:
        k = position.get(int(m.group(1)))
        return f"[{k}]" if k else ""
    return _CITATION.sub(sub, answer)
