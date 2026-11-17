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

# Naming/containment edges, not facts worth asking about. `has_feature` links
# a compound row entity (dti/prime) back to its base feature, and a table with
# a literal `Feature` column emits the same relation for its cells -- mining
# either produces questions that restate a one-hop fact through a structural
# hop ("what is the auc of the feature of dti?"), and a blank key cell makes it
# a self-loop (dti -has_feature-> dti).
STRUCTURAL_RELATIONS = ("in_section", "has_alias", "has_product",
                        "has_regulatory_section", "has_feature")
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


def _seed_chain_via(h: str, r0: str, m: str, r1: str) -> str:
    """Surface form for a head with several ``r0`` edges.

    Naming the intermediate keeps the question well-posed instead of dropping
    the chain, which matters because multi-valued relations are the norm in
    real doc graphs and ``path_compose`` has no other producer.
    """
    return (f"What is the {_phrase(r1)} of {_name(m)}, "
            f"the {_phrase(r0)} of {_name(h)}?")


_STRUCTURAL_LOWER = frozenset(r.lower() for r in STRUCTURAL_RELATIONS)


def _facts(graph):
    # Case-insensitive: relations derived from a column name carry the source
    # header's casing (`has_Feature`), so an exact-string check would let them
    # through while excluding the hardcoded `has_feature`.
    return [(h, r, t) for h, r, t in graph.triples
            if str(r).lower() not in _STRUCTURAL_LOWER]


class RelationChainGenerator(QuestionGenerator):
    """One question per distinct 2-hop relation chain ``h -r0-> m -r1-> t``.

    Every emitted question denotes exactly one answer. When ``h`` has a single
    ``r0`` edge the short form is used ("the ``r1`` of the ``r0`` of ``h``");
    when it branches, the intermediate is named instead ("the ``r1`` of ``m``,
    the ``r0`` of ``h``") rather than dropping the chain — multi-valued
    relations are common and this generator is the only producer of
    ``path_compose``. A chain is skipped only when the final hop is itself
    multi-valued, which no phrasing can disambiguate without naming the answer.

    Deduplication is on the rendered question, not the raw ids: ``_phrase``
    strips ``has_``/``is_`` prefixes, so distinct relations can collapse to the
    same surface form.
    """

    category = "path_compose"

    def __init__(self, max_chains: int = 0, seed: int = 42):
        self.max_chains = max_chains
        self.seed = seed

    def generate(self, graph) -> list[GeneratedQuestion]:
        adj: dict = {}
        for h, r, t in _facts(graph):
            adj.setdefault(h, []).append((r, str(t)))
        def _targets(node: str, rel: str) -> set:
            return {t for r, t in adj.get(node, []) if r == rel}

        # Pass 1 — collect the chains whose final hop is single-valued. Naming
        # the intermediate cannot rescue a multi-valued final hop, since "the
        # r1 of m" would still denote several things.
        chains = []
        for h, r0, m in _facts(graph):
            m = str(m)
            if _NUM.match(m.strip()):
                continue
            for r1, _t in adj.get(m, []):
                if len(_targets(m, r1)) == 1:
                    chains.append((h, r0, m, r1))

        # Pass 2 — pin (name the intermediate) whenever the short form would
        # not denote one thing: either the head branches on r0, or two chains
        # render identically because _phrase() collapsed distinct relations.
        short_counts: dict = {}
        for h, r0, _m, r1 in chains:
            key = _seed_chain(h, r0, r1)
            short_counts[key] = short_counts.get(key, 0) + 1

        questions: dict = {}
        for h, r0, m, r1 in chains:
            short = _seed_chain(h, r0, r1)
            pinned = short_counts[short] > 1 or len(_targets(h, r0)) != 1
            text = _seed_chain_via(h, r0, m, r1) if pinned else short
            questions.setdefault(text, []).append((h, r0, m, r1, pinned))

        def answer_from_index(h, r0, m, r1, pinned):
            """The chain's answer, read from the adjacency built above.

            Computing ground truth as ``fn(graph)`` cost a full
            ``query_triples`` scan per candidate, and ``_validate`` then ran
            ``fn`` a second time and compared it against itself -- quadratic,
            and a tautology. Reading the answer from the index instead halves
            the scans and makes the check real: the closure a consumer will run
            has to reproduce what the index says.
            """
            if pinned:
                if m not in _targets(h, r0):
                    return None
                head = m
            else:
                mids = _targets(h, r0)
                if len(mids) != 1:
                    return None
                head = next(iter(mids))
            vals = _targets(head, r1)
            return str(next(iter(vals))) if len(vals) == 1 else None

        out, idx = [], 0
        for text, group in questions.items():
            # A collision that survives pinning renders identically from
            # different relations; no phrasing separates those, so drop them.
            if len(group) != 1:
                continue
            h, r0, m, r1, pinned = group[0]
            question = text
            def fn(g, _h=h, _r0=r0, _m=m, _r1=r1, _pinned=pinned):
                mids = {str(t) for _, _, t in
                        g.query_triples(head=_h, relation=_r0)}
                if _pinned:
                    if _m not in mids:      # chain no longer in the graph
                        return None
                    head = _m
                elif len(mids) != 1:
                    return None
                else:
                    head = next(iter(mids))
                vals = {str(t) for _, _, t in
                        g.query_triples(head=head, relation=_r1)}
                return vals.pop() if len(vals) == 1 else None

            q = GeneratedQuestion(
                qid=f"path_compose_{idx:04d}", category=self.category,
                natural_language=question,
                graph_answer_fn=fn,
                ground_truth=answer_from_index(h, r0, m, r1, pinned),
                llm_prompt=f"{question} Format: ANSWER: <exact string>",
                answer_type="str",
                metadata={"head": h, "r1": r0, "r2": r1, "via": m,
                          "pinned": pinned,
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
