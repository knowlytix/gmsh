# SPDX-License-Identifier: Apache-2.0
"""Evaluate a query compiler against the geometric-binding baseline.

The compiler is scored on the SAME accept cohort as geometric binding
(:func:`~knowlytix.knowledge.rag.eval.build_accept_cohort`): one answerable gold
question per ``(head, relation)`` plus a relation-absent (out-of-scope) question
per head. Running the compiler-mode pipeline over this cohort and scoring at its
operating point yields answerable recall, the out-of-scope false-accept rate (the
peripheral-head harm), and operating F1 — directly comparable to geometric binding.

The held-out metric follows graphDOE's rule that **the graph is the oracle**: a
compiled query is correct iff EXECUTING it against the graph recovers the same
answer the gold graph path recovers (:func:`_graph_answer`), scored with
:func:`~knowlytix.benchmark.scorers.score_answer`. Truth is recomputed from the
graph every time, never read from a stored label, so a different-but-equivalent
query counts as correct and abstention is correct exactly when the graph holds no
such fact.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["evaluate_compiler", "GEOMETRIC_BASELINE_CIP"]

# Geometric parse+bind on the CIP store (a named reference point from the compiler
# experiment write-up). Available for callers that want the published CIP numbers;
# it is NOT a default -- evaluate_compiler measures the geometric baseline on the
# store under test (see _geometric_baseline) unless an explicit baseline is passed.
GEOMETRIC_BASELINE_CIP = {"recall": 0.979, "oos_false_accept": 0.657,
                          "operating_f1": 0.876}


def _load_rows(data) -> list[dict]:
    if isinstance(data, (str, Path)):
        return [json.loads(line) for line in open(data)]
    return list(data)


def _geometric_baseline(store, llm, *, negatives_per_head: int, seed: int) -> dict:
    """Score geometric parse+bind on the SAME accept cohort as the compiler, so the
    comparison point is measured on this store rather than read from a frozen
    constant. Geometric mode is store-only (no LLM synthesis, relevance gate off), so
    this is a cheap extra cohort pass. Returns ``{recall, oos_false_accept,
    operating_f1}``."""
    from knowlytix.knowledge.rag import RagConfig, RagPipeline
    from knowlytix.knowledge.rag.eval import (
        build_accept_cohort,
        score_accept_at_threshold,
    )

    rag = RagConfig(llm=llm, query_parse_mode="geometric", relevance_gate=False)
    pipe = RagPipeline.from_store(store, rag)
    records = build_accept_cohort(store, pipe, negatives_per_head=negatives_per_head,
                                  seed=seed)
    m = score_accept_at_threshold(records, 0.0)
    return {"recall": m["recall"], "oos_false_accept": m["false_accept"],
            "operating_f1": m["f1"]}


def _graph_answer(graph, hops):
    """Execute a ``[(head, relation), ...]`` chain against the graph and return the
    recovered tail (a str, or a set of tails at the final hop), or ``None`` when the
    graph holds no such fact. Intermediate heads may be ``?x`` placeholders — the walk
    follows the graph, resolving each intermediate to a tail that continues the chain.
    This is the graph oracle: truth for any query is what traversing the graph yields."""
    if not hops:
        return None
    cur = hops[0][0]
    for i, (_hd, rel) in enumerate(hops):
        tails = [t for (h, r, t) in graph.triples if h == cur and r == rel]
        if not tails:
            return None
        if i == len(hops) - 1:
            return tails[0] if len(tails) == 1 else set(tails)
        # continue through a tail that has the next relation, else the first
        nxt_rel = hops[i + 1][1]
        cont = [t for t in tails if any(h == t and r == nxt_rel for h, r, _ in graph.triples)]
        cur = cont[0] if cont else tails[0]
    return None


def evaluate_compiler(
    store,
    llm,
    *,
    parser=None,
    compiler_dir: str | None = None,
    base_model: str | None = None,
    device: str = "cuda",
    heldout=None,
    negatives_per_head: int = 1,
    seed: int = 42,
    baseline: dict | None = None,
    compare_geometric: bool = True,
) -> dict:
    """Score a compiler on ``store``'s accept cohort; compare to geometric binding.

    Args:
        store: a loaded :class:`GMSExpertStore`.
        llm: synthesis backend for the pipeline (never called — cohort uses
            ``generate=False`` and the relevance gate is off).
        parser: an existing :class:`CompiledQueryParser` to reuse (avoids reloading
            the model). If ``None``, one is built from ``compiler_dir``.
        compiler_dir: the trained adapter dir (required if ``parser`` is None).
        heldout: optional rows or jsonl path with ``question/head/relation`` to
            measure raw ``(head, relation)`` generation accuracy on unseen forms.
        baseline: comparison numbers to attach. If ``None`` and
            ``compare_geometric`` is True, the geometric parse+bind baseline is
            MEASURED on this store's cohort (:func:`_geometric_baseline`); if
            ``None`` with ``compare_geometric`` False, no baseline is attached.
            :data:`GEOMETRIC_BASELINE_CIP` remains available as a named reference
            for the CIP store but is no longer a silent default.
        compare_geometric: measure the geometric baseline in-run when no explicit
            ``baseline`` is given (default True).

    Returns:
        ``{recall, oos_false_accept, operating_f1, precision, n_pos, n_neg,
        baseline[, heldout_graph_recovery, n_heldout]}``. ``baseline`` is the
        measured geometric result, an explicit override, or ``None``.
    """
    from knowlytix.knowledge.rag import RagConfig, RagPipeline
    from knowlytix.knowledge.rag.eval import (
        build_accept_cohort,
        score_accept_at_threshold,
    )

    if parser is None and compiler_dir is None:
        raise ValueError("pass parser= or compiler_dir=")

    # When a parser is supplied, build the pipeline in a mode that needs no adapter
    # dir (geometric is cheap and store-only) and swap the provided compiler in —
    # this reuses the already-loaded model instead of reloading it from disk.
    if parser is not None:
        rag = RagConfig(llm=llm, query_parse_mode="geometric", relevance_gate=False)
        pipe = RagPipeline.from_store(store, rag)
        pipe.extractor = parser
    else:
        rag = RagConfig(llm=llm, query_parse_mode="compiler", compiler_dir=compiler_dir,
                        compiler_base_model=base_model, compiler_device=device,
                        relevance_gate=False)
        pipe = RagPipeline.from_store(store, rag)

    records = build_accept_cohort(store, pipe, negatives_per_head=negatives_per_head,
                                  seed=seed)
    m = score_accept_at_threshold(records, 0.0)
    if baseline is None and compare_geometric:
        baseline = _geometric_baseline(store, llm, negatives_per_head=negatives_per_head,
                                       seed=seed)
    result = {
        "recall": m["recall"],
        "oos_false_accept": m["false_accept"],
        "operating_f1": m["f1"],
        "precision": m["precision"],
        "n_pos": m["n_pos"],
        "n_neg": m["n_neg"],
        "baseline": baseline,
    }

    if heldout is not None:
        from knowlytix.benchmark.scorers import score_answer

        rows = _load_rows(heldout)
        graph = store.doc_graph
        preds = pipe.extractor.compile_raw([r["question"] for r in rows])

        def _gold_hops(r):
            return ([tuple(x) for x in r["hops"]] if "hops" in r
                    else [(r["head"], r["relation"])])

        n_correct = 0
        for r, pred in zip(rows, preds):
            gold = _graph_answer(graph, _gold_hops(r))      # truth from the graph
            got = _graph_answer(graph, pred)                # execute the prediction
            if gold is None:                                # relation-absent: abstain is correct
                ok = got is None
            else:
                ok = got is not None and score_answer(got, gold).correct
            n_correct += ok
        result["heldout_graph_recovery"] = round(n_correct / len(rows), 4) if rows else 0.0
        result["n_heldout"] = len(rows)

    return result
