# SPDX-License-Identifier: Apache-2.0
"""Generic, domain-agnostic question generators (catalog bases without a
dedicated legacy class).

Each class here is a :class:`QuestionGenerator` that mines any
:class:`~knowlytix.benchmark.graph.DocumentGraph` — it hard-codes no domain
content and reads only the real graph API (``enm``/``lookup``,
``triples``/``query_triples``, ``phase_encoders``/``check_threshold``,
``find_contradictions``). Together with the nine named generators
(ExactRecall, FactRecall, Counting, CrossReference, MultiHop, RelationChain,
RelationAbsentProbe, Threshold, Contradiction) these complete the 21 base
categories of ``base_catalog.yaml``.

Every generator is capability-gated: if the substrate it needs is absent (no
ENM, no phase encoders, no ordered/policy relations, ...) its ``generate``
returns an empty list rather than raising. A store therefore decides which
bases it supports; applying all 21 to a store that lacks a substrate simply
yields nothing for the unsupported ones.

Each class's docstring records its base name, ``ground_truth`` primitive,
required capabilities, answer type and gate condition. The full procedure —
graph substrate, the 18-base mapping and the coverage report — is documented in
``applications/ace/docs/base_generator_methodology.md``.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque

from knowlytix.benchmark.generators.base import (
    ANSWER_FORMATS,
    GeneratedQuestion,
    QuestionGenerator,
)

# Relations that carry no useful content for the derived miners.
# `has_feature` is structural scaffolding: it links a compound row entity back
# to its base feature, so questions over it restate the entity id.
_SKIP_RELATIONS = {"in_section", "has_value", "has_effect", "has_feature"}
# Order / policy relations are detected, never assumed.
_ORDER_RELATIONS = {"next", "precedes", "followed_by", "then", "before",
                    "succeeded_by", "step_after"}
_POLICY_RELATIONS = {"policy", "requires_action", "must", "escalate",
                     "governed_by", "action", "requires"}

_FORMATS = dict(ANSWER_FORMATS)
_FORMATS.update({
    "mcq": "Answer with exactly one of the listed options. Format: ANSWER: <option>",
    "decision": "Format: ANSWER: answer or abstain",
    "list": "List the items in order, comma-separated. Format: ANSWER: a, b, c",
})


def _phrase(relation: str) -> str:
    r = relation[4:] if relation.startswith("has_") else relation
    return r.replace("_", " ").strip()


def _name(entity: str) -> str:
    return str(entity).replace("/", " / ").strip()


def _is_number(x) -> bool:
    try:
        float(str(x).replace(",", "").replace("$", "").strip())
        return True
    except (TypeError, ValueError):
        return False


class _GraphMiner(QuestionGenerator):
    """Shared scaffolding for the generic generators.

    Subclasses set ``name``, ``category`` and ``answer_type`` and implement
    ``_mine``. ``generate`` wraps ``_mine`` with a uniform try/except and the
    per-category cap, so a substrate-absent miner degrades to an empty list.
    """

    name = "base"
    category = "base"
    answer_type = "str"

    def __init__(self, *, max_questions: int = 200, seed: int = 42,
                 negatives_per_head: int = 2):
        self.max_questions = max_questions
        self.seed = seed
        self.negatives_per_head = negatives_per_head

    # -- API ----------------------------------------------------------------
    def generate(self, graph) -> list[GeneratedQuestion]:
        try:
            qs = self._mine(graph)
        except Exception:
            return []
        return qs[: self.max_questions]

    def _mine(self, graph) -> list[GeneratedQuestion]:  # pragma: no cover
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    def _fmt(self) -> str:
        return _FORMATS.get(self.answer_type, "Format: ANSWER: <value>")

    def _mk(self, idx, nl, fn, gt, prompt, meta) -> GeneratedQuestion:
        return GeneratedQuestion(
            qid=f"{self.name}_{idx:04d}", category=self.category,
            natural_language=nl, graph_answer_fn=fn, ground_truth=gt,
            llm_prompt=prompt, answer_type=self.answer_type, metadata=meta)

    def _content_triples(self, graph):
        # Case-insensitive: a column-derived relation carries the header's
        # casing (`has_Feature`).
        return [(h, r, t) for h, r, t in graph.triples
                if str(r).lower() not in _SKIP_RELATIONS]

    def _enm_by_type(self, graph):
        by_type: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for key, entry in graph.enm.items():
            by_type[key.type].append((key.id, entry.value))
        return by_type


# =========================================================================
# A. RETRIEVAL / RECALL
# =========================================================================
class FactCompletionGenerator(_GraphMiner):
    """Fact completion (base ``fact_completion``): fill a functional
    (head, relation) blank; distractors are other tails asserted for the same
    relation.

    GMS mapping: ground_truth=query_triples · requires=[triples, enm, negatives]
    · answer_type=mcq.
    Gates when: no functional (head, relation) has an alternative tail to serve
    as a distractor.
    """
    name = "fact_completion"
    category = "Fact completion"
    answer_type = "mcq"

    def _mine(self, graph):
        by_rel: dict[str, set] = defaultdict(set)
        for h, r, t in self._content_triples(graph):
            by_rel[r].add(t)
        rng = random.Random(self.seed)
        out, idx, seen = [], 0, set()
        for h, r, t in self._content_triples(graph):
            if (h, r) in seen or len(graph.query_triples(head=h, relation=r)) != 1:
                continue
            seen.add((h, r))
            distractors = [x for x in by_rel[r] if x != t]
            if not distractors:
                continue
            opts = rng.sample(distractors, min(3, len(distractors))) + [t]
            rng.shuffle(opts)

            def fn(g, _h=h, _r=r):
                res = g.query_triples(head=_h, relation=_r)
                return res[0][2] if res else None

            out.append(self._mk(
                idx, f"What is the {_phrase(r)} of {_name(h)}? "
                     f"Options: {', '.join(map(str, opts))}.",
                fn, t,
                f"Choose the {_phrase(r)} of {_name(h)} from: "
                f"{', '.join(map(str, opts))}. {self._fmt()}",
                {"head": h, "relation": r, "options": list(map(str, opts))}))
            idx += 1
        return out


class PlausibilityGenerator(_GraphMiner):
    """Plausibility (base ``plausibility``): is (head, relation, tail) asserted?
    One positive per edge plus one sampled negative (a tail of the same relation
    not asserted for head).

    GMS mapping: ground_truth=score_triple · requires=[triples, negatives] ·
    answer_type=bool.
    Gates when: there are no content triples.
    """
    name = "plausibility"
    category = "Plausibility"
    answer_type = "bool"

    def _mine(self, graph):
        by_rel: dict[str, list] = defaultdict(list)
        for h, r, t in self._content_triples(graph):
            by_rel[r].append(t)
        rng = random.Random(self.seed)
        out, idx = [], 0

        def present(g, h, r, t):
            return len(g.query_triples(head=h, relation=r, tail=t)) > 0

        for h, r, t in self._content_triples(graph):
            out.append(self._mk(
                idx, f"Does the report state that {_name(h)} has {_phrase(r)} {t}?",
                lambda g, _h=h, _r=r, _t=t: present(g, _h, _r, _t), True,
                f"Is it stated in the report that the {_phrase(r)} of {_name(h)} "
                f"is {t}? {self._fmt()}", {"head": h, "relation": r, "tail": t}))
            idx += 1
            pool = [x for x in by_rel[r] if not present(graph, h, r, x)]
            if not pool:
                continue
            neg = rng.choice(pool)
            out.append(self._mk(
                idx, f"Does the report state that {_name(h)} has {_phrase(r)} {neg}?",
                lambda g, _h=h, _r=r, _t=neg: present(g, _h, _r, _t), False,
                f"Is it stated in the report that the {_phrase(r)} of {_name(h)} "
                f"is {neg}? {self._fmt()}",
                {"head": h, "relation": r, "tail": neg, "negative": True}))
            idx += 1
        return out


class AbsenceDetectionGenerator(_GraphMiner):
    """Absence detection (base ``absence_detection``): is the (head, relation)
    fact absent from the report? Positives are relations the head does not
    assert; negatives are ones it does.

    GMS mapping: ground_truth=query_triples_empty · requires=[triples] ·
    answer_type=bool.
    Gates when: there are no content triples.
    """
    name = "absence_detection"
    category = "Absence detection"
    answer_type = "bool"

    def _mine(self, graph):
        rels = sorted({r for _, r, _ in self._content_triples(graph)})
        by_head: dict[str, set] = defaultdict(set)
        for h, r, _t in self._content_triples(graph):
            by_head[h].add(r)
        rng = random.Random(self.seed)
        out, idx = [], 0

        def absent(g, h, r):
            return len(g.query_triples(head=h, relation=r)) == 0

        for h in sorted(by_head):
            missing = [r for r in rels if r not in by_head[h]]
            present = sorted(by_head[h])
            rng.shuffle(missing)
            picks = [(r, True) for r in missing[: self.negatives_per_head]]
            if present:
                picks.append((rng.choice(present), False))
            for r, is_absent in picks:
                out.append(self._mk(
                    idx, f"Is the {_phrase(r)} of {_name(h)} absent from the report?",
                    lambda g, _h=h, _r=r: absent(g, _h, _r), is_absent,
                    f"Does the report state the {_phrase(r)} of {_name(h)}? Answer "
                    f"whether that fact is absent. {self._fmt()}",
                    {"head": h, "relation": r}))
                idx += 1
        return out


# =========================================================================
# B. COMPARISON / RANKING / AGGREGATION
# =========================================================================
class RankingGenerator(_GraphMiner):
    """Ranking (base ``ranking``): argmin / argmax entity within an ENM type.

    GMS mapping: ground_truth=enm_sort · requires=[enm] · answer_type=str.
    Gates when: no ENM type has at least three entries.
    """
    name = "ranking"
    category = "Ranking"
    answer_type = "str"

    def _mine(self, graph):
        out, idx = [], 0
        for etype, entries in self._enm_by_type(graph).items():
            if len(entries) < 3:
                continue
            order = sorted(entries, key=lambda x: x[1])
            for which, (eid, _val) in (("lowest", order[0]), ("highest", order[-1])):
                def fn(g, _t=etype, _w=which):
                    best = None
                    for k, e in g.enm.items():
                        if k.type != _t:
                            continue
                        if best is None or (_w == "lowest" and e.value < best[1]) \
                                or (_w == "highest" and e.value > best[1]):
                            best = (k.id, e.value)
                    return best[0] if best else None
                out.append(self._mk(
                    idx, f"Which entry has the {which} value in '{etype}'?",
                    fn, eid,
                    f"In the '{etype}' data, name the entry with the {which} "
                    f"value. {self._fmt()}", {"enm_type": etype, "which": which}))
                idx += 1
        return out


class ComparativeDiscriminationGenerator(_GraphMiner):
    """Comparative discrimination (base ``comparative_discrimination``): pick the
    candidate closest to a source entity by graph distance.

    GMS mapping: ground_truth=graph_distance · requires=[enm, triples] ·
    answer_type=str.
    Gates when: no source entity has at least three reachable nodes at distinct
    distances.
    """
    name = "comparative_discrimination"
    category = "Comparative discrimination"
    answer_type = "str"

    def _mine(self, graph):
        adj: dict[str, set] = defaultdict(set)
        for h, _r, t in self._content_triples(graph):
            if _is_number(t):
                continue
            adj[h].add(t)
            adj[t].add(h)
        if not adj:
            return []

        def bfs(src):
            dist, dq = {src: 0}, deque([src])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        dq.append(v)
            return dist

        rng = random.Random(self.seed)
        out, idx = [], 0
        for src in sorted(adj):
            dist = bfs(src)
            reach = sorted((d, n) for n, d in dist.items() if n != src and d > 0)
            if len(reach) < 3:
                continue
            near = reach[0][1]
            far = [n for d, n in reach if d > reach[0][0]]
            if len(far) < 2:
                continue
            cands = [near] + rng.sample(far, 2)
            rng.shuffle(cands)

            def fn(g, _src=src, _c=tuple(cands)):
                d = bfs(_src)
                return sorted(_c, key=lambda n: d.get(n, 1 << 30))[0]
            out.append(self._mk(
                idx, f"Which of these is most closely related to {_name(src)}: "
                     f"{', '.join(_name(c) for c in cands)}?",
                fn, near,
                f"Among {', '.join(_name(c) for c in cands)}, which is most "
                f"closely related to {_name(src)} in the report? {self._fmt()}",
                {"source": src, "candidates": cands}))
            idx += 1
        return out


class NumericComputationGenerator(_GraphMiner):
    """Numeric computation (base ``numeric_computation``): sum or difference of
    two stored numbers within an ENM type.

    GMS mapping: ground_truth=arithmetic · requires=[enm] · answer_type=float.
    Gates when: no ENM type has at least two entries.
    """
    name = "numeric_computation"
    category = "Numeric computation"
    answer_type = "float"

    def _mine(self, graph):
        rng = random.Random(self.seed)
        out, idx = [], 0
        for etype, entries in self._enm_by_type(graph).items():
            if len(entries) < 2:
                continue
            for _ in range(min(len(entries), 5)):
                (a_id, a_v), (b_id, b_v) = rng.sample(entries, 2)
                for op, sym, val in (("sum", "+", a_v + b_v),
                                     ("difference", "-", a_v - b_v)):
                    def fn(g, _t=etype, _a=a_id, _b=b_id, _op=op):
                        va, vb = g.lookup(_t, _a), g.lookup(_t, _b)
                        if va is None or vb is None:
                            return None
                        return va + vb if _op == "sum" else va - vb
                    out.append(self._mk(
                        idx, f"In '{etype}', what is the {op} of '{a_id}' and '{b_id}'?",
                        fn, float(val),
                        f"In the '{etype}' data, compute {a_id} {sym} {b_id} "
                        f"exactly. {self._fmt()}",
                        {"enm_type": etype, "a": a_id, "b": b_id, "op": op}))
                    idx += 1
        return out


class CrossTableAggregationGenerator(_GraphMiner):
    """Cross-table aggregation (base ``cross_table_aggregation``): aggregate one
    entity's values across ENM types.

    GMS mapping: ground_truth=aggregate · requires=[enm, provenance] ·
    answer_type=float.
    Gates when: no entity appears in at least two ENM types.
    """
    name = "cross_table_aggregation"
    category = "Cross-table aggregation"
    answer_type = "float"

    def _mine(self, graph):
        base_types: dict[str, dict[str, float]] = defaultdict(dict)
        for key, entry in graph.enm.items():
            base_types[key.id.split("/")[0]][key.type] = entry.value
        out, idx = [], 0
        for base, per_type in base_types.items():
            if len(per_type) < 2:
                continue
            total = float(sum(per_type.values()))

            def fn(g, _base=base, _types=tuple(per_type)):
                vals = [e.value for k, e in g.enm.items()
                        if k.id.split("/")[0] == _base and k.type in _types]
                return float(sum(vals)) if vals else None
            out.append(self._mk(
                idx, f"What is the total of {_name(base)}'s values across "
                     f"{', '.join(per_type)}?",
                fn, total,
                f"Sum the values recorded for {_name(base)} across the "
                f"{', '.join(per_type)} sections. {self._fmt()}",
                {"entity": base, "enm_types": list(per_type)}))
            idx += 1
        return out


# =========================================================================
# C. REASONING / COMPOSITION
# =========================================================================
class PathConsistencyGenerator(_GraphMiner):
    """Reasoning-path consistency (base ``reasoning_path_consistency``): a graph
    proxy for holonomy — do two distinct relations out of a head, each composed
    with a second hop, land on the same tail?

    GMS mapping: ground_truth=check_holonomy · requires=[path] · answer_type=bool.
    Gates when: no head has two second-hop paths to compare.
    """
    name = "reasoning_path_consistency"
    category = "Reasoning-path consistency"
    answer_type = "bool"

    def _mine(self, graph):
        adj: dict[str, list] = defaultdict(list)
        for h, r, t in self._content_triples(graph):
            if not _is_number(t):
                adj[h].append((r, t))
        out, idx = [], 0
        for h, outs in adj.items():
            if len(outs) < 2:
                continue
            for i in range(len(outs)):
                for j in range(i + 1, len(outs)):
                    (r0, m0), (r1, m1) = outs[i], outs[j]
                    n0, n1 = adj.get(m0, []), adj.get(m1, [])
                    if not n0 or not n1:
                        continue
                    consistent = (n0[0][1] == n1[0][1])

                    def fn(g, _h=h, _r0=r0, _r1=r1):
                        a = g.query_triples(head=_h, relation=_r0)
                        b = g.query_triples(head=_h, relation=_r1)
                        if not a or not b:
                            return None
                        ta = g.query_triples(head=str(a[0][2]))
                        tb = g.query_triples(head=str(b[0][2]))
                        if not ta or not tb:
                            return None
                        return ta[0][2] == tb[0][2]
                    out.append(self._mk(
                        idx, f"Do the {_phrase(r0)} and {_phrase(r1)} paths from "
                             f"{_name(h)} lead to the same fact?",
                        fn, consistent,
                        f"Trace {_name(h)} through its {_phrase(r0)} and through "
                        f"its {_phrase(r1)}, one more hop each. Do both paths reach "
                        f"the same value? {self._fmt()}",
                        {"head": h, "r0": r0, "r1": r1}))
                    idx += 1
                    if idx >= self.max_questions:
                        return out
        return out


class ProceduralOrderingGenerator(_GraphMiner):
    """Procedural ordering (base ``procedural_ordering``): order a sequence, only
    if the graph carries order relations.

    GMS mapping: ground_truth=sequence_check · requires=[ordered_sequence] ·
    answer_type=list.
    Gates when: the graph has no order relations (next / precedes / ...).
    """
    name = "procedural_ordering"
    category = "Procedural ordering"
    answer_type = "list"

    def _mine(self, graph):
        order_rels = {r for _, r, _ in graph.triples} & _ORDER_RELATIONS
        if not order_rels:
            return []
        succ: dict[str, str] = {}
        for h, r, t in graph.triples:
            if r in order_rels:
                succ[h] = t
        starts = set(succ) - set(succ.values())
        out, idx = [], 0
        for s in sorted(starts):
            chain, cur, guard = [s], s, 0
            while cur in succ and guard < 100:
                cur = succ[cur]
                chain.append(cur)
                guard += 1
            if len(chain) < 3:
                continue
            out.append(self._mk(
                idx, f"Put these steps in order starting from {_name(s)}: "
                     f"{', '.join(_name(c) for c in chain)}.",
                lambda g, _c=list(chain): _c, list(chain),
                f"Order the steps beginning at {_name(s)}. {self._fmt()}",
                {"start": s, "length": len(chain)}))
            idx += 1
        return out


# =========================================================================
# D. CONSISTENCY / INTEGRITY
# =========================================================================
class BoundaryGenerator(_GraphMiner):
    """Boundary (base ``boundary``): threshold at the exact stored value
    (margin 0) — the decision edge. Requires a phase encoder whose name matches
    the ENM type.

    GMS mapping: ground_truth=check_threshold · requires=[threshold, enm] ·
    answer_type=bool.
    Gates when: no phase-encoder name matches an ENM type.
    """
    name = "boundary"
    category = "Boundary"
    answer_type = "bool"

    def _mine(self, graph):
        out, idx = [], 0
        for key, entry in graph.enm.items():
            if graph.phase_encoders.get(key.type) is None:
                continue
            v = entry.value
            for op, word, gt in (("ge", "at least", True), ("gt", "greater than", False)):
                def fn(g, _t=key.type, _id=key.id, _lim=v, _op=op):
                    val = g.lookup(_t, _id)
                    if val is None:
                        return None
                    ok, _ = g.check_threshold(_t, val, _lim, _op)
                    return ok
                out.append(self._mk(
                    idx, f"Is '{key.id}' ({key.type}) {word} {v}?",
                    fn, gt,
                    f"Find the value of '{key.id}' in '{key.type}'. Is it {word} "
                    f"{v}? {self._fmt()}",
                    {"enm_type": key.type, "id": key.id, "threshold": v, "op": op}))
                idx += 1
        return out


# =========================================================================
# E. BEHAVIORAL / OPERATIONAL
# =========================================================================
class RecoveryGenerator(_GraphMiner):
    """Recovery (base ``recovery``): expected abstention — the report does not
    contain the asked fact, so the correct behavior is to abstain rather than
    answer.

    GMS mapping: ground_truth=expected_abstain · requires=[triples] ·
    answer_type=decision.
    Gates when: there are no content triples (nothing whose absence to probe).
    """
    name = "recovery"
    category = "Recovery"
    answer_type = "decision"

    def _mine(self, graph):
        rels = sorted({r for _, r, _ in self._content_triples(graph)})
        by_head: dict[str, set] = defaultdict(set)
        for h, r, _t in self._content_triples(graph):
            by_head[h].add(r)
        rng = random.Random(self.seed)
        out, idx = [], 0
        for h in sorted(by_head):
            missing = [r for r in rels if r not in by_head[h]]
            rng.shuffle(missing)
            for r in missing[: self.negatives_per_head]:
                def fn(g, _h=h, _r=r):
                    return "abstain" if not g.query_triples(head=_h, relation=_r) else "answer"
                out.append(self._mk(
                    idx, f"A user asks for the {_phrase(r)} of {_name(h)}, which the "
                         f"report does not record. What should the assistant do?",
                    fn, "abstain",
                    f"The report does not state the {_phrase(r)} of {_name(h)}. State "
                    f"the correct action (answer or abstain). {self._fmt()}",
                    {"head": h, "relation": r}))
                idx += 1
        return out


class GovernancePolicyGenerator(_GraphMiner):
    """Governance policy (base ``governance_policy``): apply a stored policy rule,
    only if the graph carries policy edges.

    GMS mapping: ground_truth=policy_rule · requires=[policy, triples] ·
    answer_type=decision.
    Gates when: the graph has no policy relations (policy / requires_action / ...).
    """
    name = "governance_policy"
    category = "Governance policy"
    answer_type = "decision"

    def _mine(self, graph):
        policy = [(h, r, t) for h, r, t in graph.triples if r in _POLICY_RELATIONS]
        if not policy:
            return []
        out, idx = [], 0
        for h, r, t in policy:
            def fn(g, _h=h, _r=r):
                res = g.query_triples(head=_h, relation=_r)
                return res[0][2] if res else None
            out.append(self._mk(
                idx, f"Per policy, what action does {_name(h)} require?",
                fn, t,
                f"Given the report's policy for {_name(h)}, state the required "
                f"action. {self._fmt()}", {"head": h, "relation": r}))
            idx += 1
        return out


# The twelve generic generators that, with the nine named ones, complete
# the 21 base categories.
GENERIC_GENERATORS = [
    FactCompletionGenerator, PlausibilityGenerator, AbsenceDetectionGenerator,
    RankingGenerator, ComparativeDiscriminationGenerator, NumericComputationGenerator,
    CrossTableAggregationGenerator, PathConsistencyGenerator,
    ProceduralOrderingGenerator, BoundaryGenerator, RecoveryGenerator,
    GovernancePolicyGenerator,
]
