# SPDX-License-Identifier: Apache-2.0
"""Separate holdout DOE with an external materializer (bake-off Phase-1).

The DOE that trains the binders covers the presentation factors the design
*named*; it cannot cover the phrasings no one thought to name, and every training
variant is written by the same local rewriter, so a model can learn that
rewriter's surface regularities rather than the task. A holdout meant to measure
real-language reliability must be disjoint on both axes.

This module emits a *materialization manifest*: for each answerable fact and each
relation-absent probe in the store (the same cohort structure the accept-gate uses),
one cell carrying the gold ``(head, relation)`` chain, the expected answer, and an
*intended* assignment over a factor set **disjoint from the training design**,
drawn straight from the graphDOE catalog. No question text is generated here and
no local model is called. An external, stronger materializer (Claude, in this
project) writes the natural-language question for each cell, realizing its intended
factors; :func:`load_materialized_holdout` reads the filled manifest back with the
gold preserved.

The result is disjoint from training on the factor axis (different factors) and on
the materializer axis (a different model writes it), so a binder's accuracy on it
is a genuine out-of-distribution reading, and it is the calibration set for the
conformal wrapper and the source of ``synthetic_vs_real_gap``. It is never used in
training or few-shot.
"""

from __future__ import annotations

import json

from .roundtrip import graph_execute

# Factors disjoint from the training substrate's design (which names
# clarity / style / length / expertise / paraphrase_depth). Names only; their
# levels are pulled from the bundled graphDOE catalog so the holdout and the
# training corpus share one factor vocabulary.
DISJOINT_HOLDOUT_FACTOR_NAMES = [
    "question_format", "entity_aliasing", "reasoning_cue", "role", "emotional_frame",
]


def _answerable_and_absent(store, *, negatives_per_head: int, seed: int):
    """Enumerate the accept-cohort base set from the store: one answerable
    ``(head, relation)`` per distinct fact plus ``negatives_per_head`` relations
    that exist elsewhere but not on the head. Mirrors
    :func:`knowlytix.knowledge.rag.eval.build_accept_cohort` so the holdout has the
    same fact/probe structure the gate is scored on."""
    import random

    rng = random.Random(seed)
    facts, rels_by_head, all_rels, seen = [], {}, set(), set()
    for h, r, _t in store.triples:
        if r == "in_section":
            continue
        all_rels.add(r)
        rels_by_head.setdefault(h, set()).add(r)
        if (h, r) not in seen:
            seen.add((h, r))
            facts.append((h, r))

    bases = []
    for h, r in facts:
        bases.append({"head": h, "relation": r, "hops": [[h, r]], "category": "fact_recall"})
    for h in sorted(rels_by_head):
        absent = sorted(all_rels - rels_by_head[h])
        rng.shuffle(absent)
        for r in absent[:negatives_per_head]:
            bases.append({"head": h, "relation": r, "hops": [[h, r]],
                          "category": "relation_absent"})
    return bases


def build_holdout_manifest(store, *, factor_names: list[str] | None = None,
                           n_cells: int = 40, negatives_per_head: int = 1,
                           seed: int = 123, method: str = "sobol") -> list[dict]:
    """Emit materialization cells for a disjoint-factor holdout DOE.

    Each cell is ``{cell_id, head, relation, hops, category, expected_answer,
    reference_question, factors, question: None}``, where ``factors`` is the
    *intended* disjoint-factor assignment the materializer should realize and
    ``question`` is left blank for the materializer to fill. ``n_cells`` bounds the
    number of cells (answerable facts first, then absent probes), so the set stays
    small enough to materialize by hand. No local model is invoked.
    """
    from knowlytix.harness.graphdoe import DesignMatrix

    names = factor_names or DISJOINT_HOLDOUT_FACTOR_NAMES
    graph = store.doc_graph
    bases = _answerable_and_absent(store, negatives_per_head=negatives_per_head, seed=seed)
    bases = bases[:n_cells]

    design = DesignMatrix.from_catalog(factor_names=names, n_runs=max(8, len(bases)),
                                       method=method, seed=seed).generate()
    cell_factors = design.to_dict("records")

    cells = []
    for i, b in enumerate(bases):
        expected = graph_execute(graph, b["hops"])
        if isinstance(expected, set):
            expected = sorted(expected)
        factors = {k: str(v) for k, v in cell_factors[i % len(cell_factors)].items()}
        cells.append({
            "cell_id": f"holdout_{i:03d}",
            "head": b["head"],
            "relation": b["relation"],
            "hops": b["hops"],
            "category": b["category"],
            "expected_answer": expected,
            "reference_question": f"What is the {b['relation']} of {b['head']}?",
            "factors": factors,
            "question": None,           # <- materializer fills this in
            "_holdout": True,
            "_materializer": "claude",
        })
    return cells


def write_manifest(cells: list[dict], path: str) -> None:
    """Write manifest cells to a jsonl file, one cell per line."""
    with open(path, "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")


def load_materialized_holdout(path: str, *, require_question: bool = True) -> list[dict]:
    """Read a filled manifest back as holdout eval rows. Each row keeps the gold
    ``hops``/``head``/``relation``/``category``/``expected_answer`` and the
    materialized ``question`` under ``_factors`` (the realized-factor scorer's key).
    Raises if any cell is unmaterialized and ``require_question`` is set."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        q = c.get("question")
        if require_question and not (q and str(q).strip()):
            raise ValueError(f"cell {c.get('cell_id')} has no materialized question")
        rows.append({
            "question": q,
            "hops": c["hops"],
            "head": c["head"],
            "relation": c["relation"],
            "category": c["category"],
            "expected_answer": c.get("expected_answer"),
            "_factors": c.get("factors", {}),
            "_holdout": True,
            "cell_id": c.get("cell_id"),
        })
    return rows
