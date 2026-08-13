# SPDX-License-Identifier: Apache-2.0
"""Relation-path and negative-probe question generators over a DocumentGraph.

These complement the existing single-hop
:class:`~knowlytix.benchmark.generators.fact_recall.FactRecallGenerator` (reused for
the ``fact_recall`` family) and fill the ``path_compose`` and negative families of
``harness/suite/catalogs/base_catalog.yaml`` for relation-triple stores, where the
ENM-oriented multi-hop generator (argmin/argmax over numeric types) produces no
relation-chain target. Each generated question exposes the hop chain in
``metadata['hops']`` (``[[head, relation], ["?x0", relation], ...]``), so a consumer
can read a faithful compiler target while the natural-language surface is varied
downstream by :class:`~knowlytix.harness.graphdoe.QuestionRephraser`.

* :class:`RelationChainGenerator` — one question per 2-hop relation chain
  ``h -r0-> m -r1-> t`` (m an entity), the ``hop_count = 2-Hop`` case.
* :class:`RelationAbsentProbeGenerator` — a negative probe asking a head for a
  relation it does NOT assert (the correct behavior is to abstain). Its validity is
  that the pair is genuinely absent, so it does not use value-recovery validation.
"""

from __future__ import annotations

import re

from knowlytix.benchmark.generators.base import GeneratedQuestion, QuestionGenerator

__all__ = ["RelationChainGenerator", "RelationAbsentProbeGenerator",
           "STRUCTURAL_RELATIONS"]

STRUCTURAL_RELATIONS = ("in_section", "has_alias", "has_product",
                        "has_regulatory_section")
_NUM = re.compile(r"^-?[\d,]*\.?\d+%?$")


def _phrase(rel: str) -> str:
    for p in ("has_", "is_", "are_", "defines_", "excludes_", "applies_to_"):
        if rel.startswith(p):
            return rel[len(p):].replace("_", " ")
    return rel.replace("_", " ")


def _name(e: str) -> str:
    return e.replace("_", " ")


def _seed_single(h: str, r: str) -> str:
    return f"What is the {_phrase(r)} for {_name(h)}?"


def _seed_chain(h: str, r0: str, r1: str) -> str:
    return f"What is the {_phrase(r1)} of the {_phrase(r0)} of {_name(h)}?"


def _facts(graph):
    return [(h, r, t) for h, r, t in graph.triples if r not in STRUCTURAL_RELATIONS]


class RelationChainGenerator(QuestionGenerator):
    """One question per 2-hop relation chain ``h -r0-> m -r1-> t`` (m an entity)."""

    category = "path_compose"

    def __init__(self, max_chains: int = 0, seed: int = 42):
        self.max_chains = max_chains
        self.seed = seed

    def generate(self, graph) -> list[GeneratedQuestion]:
        adj: dict = {}
        for h, r, t in _facts(graph):
            adj.setdefault(h, []).append((r, str(t)))
        out, idx = [], 0
        for h, r0, m in _facts(graph):
            if _NUM.match(str(m).strip()):
                continue
            for r1, _t in adj.get(m, []):
                def fn(g, _h=h, _r0=r0, _r1=r1):
                    mids = g.query_triples(head=_h, relation=_r0)
                    if not mids:
                        return None
                    hits = g.query_triples(head=str(mids[0][2]), relation=_r1)
                    return str(hits[0][2]) if hits else None

                q = GeneratedQuestion(
                    qid=f"path_compose_{idx:04d}", category=self.category,
                    natural_language=_seed_chain(h, r0, r1),
                    graph_answer_fn=fn, ground_truth=fn(graph),
                    llm_prompt=f"What is the {_phrase(r1)} of the {_phrase(r0)} of "
                               f"{_name(h)}? Format: ANSWER: <exact string>",
                    answer_type="str",
                    metadata={"head": h, "r1": r0, "r2": r1,
                              "hops": [[h, r0], ["?x0", r1]]})
                if self._validate(graph, q):
                    out.append(q)
                    idx += 1
        if self.max_chains and len(out) > self.max_chains:
            import random
            out = random.Random(self.seed).sample(out, self.max_chains)
        return out


class RelationAbsentProbeGenerator(QuestionGenerator):
    """A negative probe: ask a head for a relation it does NOT assert.

    The correct behavior for the system under test is to abstain (no such fact).
    Validity is that the ``(head, relation)`` pair is genuinely absent, so this
    generator does not run value-recovery validation (there is no value to recover).
    """

    category = "relation_absent"

    def __init__(self, negatives_per_head: int = 1, seed: int = 42):
        self.negatives_per_head = negatives_per_head
        self.seed = seed

    def generate(self, graph) -> list[GeneratedQuestion]:
        import random
        rng = random.Random(self.seed)
        all_rels, rels_by_head = set(), {}
        for h, r, _t in _facts(graph):
            all_rels.add(r)
            rels_by_head.setdefault(h, set()).add(r)
        out, idx = [], 0
        for h in sorted(rels_by_head):
            absent = sorted(all_rels - rels_by_head[h])
            rng.shuffle(absent)
            for r in absent[:self.negatives_per_head]:
                q = GeneratedQuestion(
                    qid=f"relation_absent_{idx:04d}", category=self.category,
                    natural_language=_seed_single(h, r),
                    graph_answer_fn=lambda g: None, ground_truth=None,
                    llm_prompt=f"What is the {_phrase(r)} of {_name(h)}?",
                    answer_type="str",
                    metadata={"head": h, "relation": r, "hops": [[h, r]]})
                out.append(q)
                idx += 1
        return out
