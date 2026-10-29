# SPDX-License-Identifier: Apache-2.0
"""Round-trip filter with a vanished-cell ledger (bake-off Phase-1 data QA).

A generated ``(question, hops)`` pair is only usable if it round-trips against
the graph oracle:

* **triples -> execute -> gold.** Executing the pair's gold ``(head, relation)``
  chain against the graph must recover a fact for an answerable pair, and must
  recover nothing for a relation-absent (abstain) pair. A pair that fails this is
  bad data -- the design named a fact the graph does not hold, or tagged an
  answerable probe as absent.
* **NL -> extract -> execute -> gold** (optional). With a reference extractor,
  the question itself must parse to a query that executes to the same answer;
  otherwise the surface, whatever the design intended, is not answerable as
  written.

Pairs that do not round-trip are dropped, each with a reason. The **vanished-cell
ledger** groups pairs by their base cell -- by default the ``(head, relation,
category)`` fact/probe, aggregated over its presentation variants -- and flags any
cell that loses *all* its variants. A whole cell vanishing is signal, not noise:
the graph cannot support that fact/probe under any phrasing, so downstream
per-cell analysis would otherwise silently rest on zero data.

This complements :func:`knowlytix.knowledge.rag.eval.grounding_preserved`, which
checks NL->retrieve head recovery on the pipeline; here the check is the graph
execution of the gold chain and, optionally, an extractor's chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def graph_execute(graph, hops):
    """Execute a ``[(head, relation), ...]`` chain against ``graph`` and return the
    recovered tail (a str, or a set of tails at the final hop), or ``None`` when the
    graph holds no such fact. Intermediate heads may be placeholders -- the walk
    follows the graph, resolving each intermediate to a tail that continues the
    chain. Mirrors the graph oracle used by the compiler evaluator so the filter
    and the evaluator agree on what "executes"."""
    if not hops:
        return None
    cur = hops[0][0]
    for i, hop in enumerate(hops):
        rel = hop[1]
        tails = [t for (h, r, t) in graph.triples if h == cur and r == rel]
        if not tails:
            return None
        if i == len(hops) - 1:
            return tails[0] if len(tails) == 1 else set(tails)
        nxt_rel = hops[i + 1][1]
        cont = [t for t in tails if any(h == t and r == nxt_rel for h, r, _ in graph.triples)]
        cur = cont[0] if cont else tails[0]
    return None


def _answers_match(a, b) -> bool:
    """Recovered-answer equality tolerant of set/scalar shape."""
    if a is None or b is None:
        return a is None and b is None
    sa = a if isinstance(a, set) else {a}
    sb = b if isinstance(b, set) else {b}
    return bool(sa & sb)


@dataclass
class RoundTripReport:
    n_in: int
    n_kept: int
    dropped: list[dict] = field(default_factory=list)       # {index, reason, cell}
    ledger: dict = field(default_factory=dict)              # cell(str) -> {n, kept, dropped}
    vanished_cells: list[str] = field(default_factory=list)  # cells losing all variants

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        for d in self.dropped:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        return {
            "n_in": self.n_in,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "keep_rate": round(self.n_kept / self.n_in, 4) if self.n_in else 0.0,
            "drop_reasons": reasons,
            "n_cells": len(self.ledger),
            "n_vanished_cells": len(self.vanished_cells),
            "vanished_cells": self.vanished_cells,
        }


def _cell_of(row: dict, cell_keys) -> str:
    parts = []
    for k in cell_keys:
        v = row.get(k)
        if v is None:
            v = (row.get("factors") or {}).get(k)
        parts.append(str(v))
    return " | ".join(parts)


def round_trip_filter(rows, graph, *, extractor=None, hops_key: str = "hops",
                      question_key: str = "question", category_key: str = "category",
                      absent_category: str = "relation_absent",
                      cell_keys=("head", "relation", "category")):
    """Filter ``rows`` (each a ``{question, hops, category, ...}`` dict) to those
    that round-trip against ``graph``; return ``(kept_rows, RoundTripReport)``.

    A row is dropped when its gold chain does not execute as its category implies
    (answerable -> a fact must exist; ``absent_category`` -> none must), or, when
    ``extractor`` is given, when the extracted chain does not execute to the gold
    answer. ``extractor`` is a callable mapping a list of questions to a list of
    ``hops`` (e.g. ``CompiledQueryParser.compile_raw``); it is called once on all
    questions. The vanished-cell ledger groups by ``cell_keys``.
    """
    rows = list(rows)
    preds = None
    if extractor is not None:
        preds = extractor([r.get(question_key, "") for r in rows])

    kept: list[dict] = []
    dropped: list[dict] = []
    ledger: dict[str, dict] = {}

    for i, row in enumerate(rows):
        cell = _cell_of(row, cell_keys)
        ledger.setdefault(cell, {"n": 0, "kept": 0, "dropped": 0})
        ledger[cell]["n"] += 1

        is_absent = row.get(category_key) == absent_category
        gold = graph_execute(graph, row.get(hops_key) or [])
        reason = None
        if is_absent and gold is not None:
            reason = "absent_but_gold_executes"
        elif not is_absent and gold is None:
            reason = "answerable_but_gold_missing"
        elif preds is not None:
            pred = graph_execute(graph, preds[i] or [])
            if is_absent:
                if pred is not None:
                    reason = "absent_but_extract_executes"
            elif not _answers_match(pred, gold):
                reason = "extract_mismatch"

        if reason is None:
            kept.append(row)
            ledger[cell]["kept"] += 1
        else:
            dropped.append({"index": i, "reason": reason, "cell": cell})
            ledger[cell]["dropped"] += 1

    vanished = sorted(c for c, v in ledger.items() if v["n"] > 0 and v["kept"] == 0)
    report = RoundTripReport(n_in=len(rows), n_kept=len(kept), dropped=dropped,
                             ledger=ledger, vanished_cells=vanished)
    return kept, report
