# SPDX-License-Identifier: Apache-2.0
"""N-arm factor-sliced bake-off harness (spec Section 6).

Runs several binder configurations on identical inputs, scores each with the
shared metric catalog, slices every metric by realized factor, and reports the
per-slice crossover -- which arm leads where -- rather than a pooled leaderboard.
Optionally runs each arm on the real-language holdout too, attaching the
conformal coverage and the synthetic-vs-real accuracy gap.

An **arm** is anything that maps a batch of questions to per-question predictions;
the existing Arm A extractor, the Arm B compiler, and the Arm C/D configurations
all wrap to the same :class:`Arm` interface, so the harness swaps them without
touching the scoring. Gold answers and predicted answers are both obtained by
executing against the graph oracle, so a different-but-equivalent bound query
counts as correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .conformal import ABSTAIN_LABEL, ConformalGate
from .metrics import (
    BinderResult,
    metric_report,
    novel_entity_report,
    patch_cost,
    slice_by_factor,
    synthetic_vs_real_gap,
)
from .roundtrip import graph_execute


def _is_var(slot) -> bool:
    return isinstance(slot, str) and slot.startswith("?")


@dataclass
class BinderPrediction:
    """One arm's prediction for one question."""
    hops: list | None                      # bound [[head, relation], ...] or None
    decision: str = "accept"               # "accept" | "abstain"
    scores: dict | None = None             # candidate label -> confidence (for conformal)


@dataclass
class Arm:
    """A named binder configuration under test.

    ``binder`` maps ``list[str]`` questions to ``list[BinderPrediction]`` (one per
    question, same order). ``patch_mechanism`` is how a discovered mis-binding is
    fixed (``alias_edit`` / ``adapter_retune`` / ``full_retrain``), used for
    ``patch_cost`` (I6)."""
    name: str
    binder: Callable[[list[str]], list[BinderPrediction]]
    patch_mechanism: str = "full_retrain"


def run_arm(arm: Arm, cohort: list[dict], graph, *,
            category_key: str = "category", absent_category: str = "relation_absent",
            hops_key: str = "hops", factors_key: str = "_factors",
            novel_key: str = "novel") -> list[BinderResult]:
    """Execute one arm over a cohort and return scored :class:`BinderResult` rows.
    Each cohort row carries the gold ``hops``, its ``category``, realized
    ``factors`` and an optional ``novel`` flag."""
    preds = arm.binder([r.get("question", "") for r in cohort])
    out = []
    for row, pred in zip(cohort, preds):
        gold = graph_execute(graph, row.get(hops_key) or [])
        phops = pred.hops or []
        pa = graph_execute(graph, phops)
        bound = bool(phops) and not _is_var(phops[0][0]) and all(
            not _is_var(h[1]) for h in phops)
        out.append(BinderResult(
            is_absent=row.get(category_key) == absent_category,
            decision=pred.decision, gold_answer=gold, pred_answer=pa, bound=bound,
            novel=bool(row.get(novel_key, False)),
            factors=row.get(factors_key) or row.get("factors") or {},
        ))
    return out


def _arm_scores_for_conformal(arm: Arm, cohort: list[dict], graph) -> list[tuple[dict, str]]:
    """Build (scores, true_label) records for conformal calibration from an arm's
    predictions on a labeled cohort. The true label is the gold answer's string,
    or ABSTAIN_LABEL for a relation-absent row; the arm's ``scores`` (or a
    one-hot on its own prediction) supply candidate confidences."""
    preds = arm.binder([r.get("question", "") for r in cohort])
    records = []
    for row, pred in zip(cohort, preds):
        is_absent = row.get("category") == "relation_absent"
        gold = graph_execute(graph, row.get("hops") or [])
        true = ABSTAIN_LABEL if is_absent or gold is None else str(gold)
        if pred.scores:
            scores = {str(k): float(v) for k, v in pred.scores.items()}
        else:
            pa = graph_execute(graph, pred.hops or [])
            key = ABSTAIN_LABEL if pred.decision == "abstain" or pa is None else str(pa)
            scores = {key: 1.0}
        records.append((scores, true))
    return records


def bakeoff(arms: list[Arm], cohort: list[dict], graph, *, factors=(),
            holdout: list[dict] | None = None, conformal_alpha: float = 0.1) -> dict:
    """Score every arm on ``cohort`` (and, if given, ``holdout``) and assemble the
    crossover report.

    Returns ``{"arms": {name: {...}}, "crossover": {...}}``. Each arm entry has the
    overall report, the novel-entity report, per-factor slices, ``patch_cost`` and
    -- when a holdout is supplied -- the holdout report, ``synthetic_vs_real_gap``,
    and conformal coverage (calibrated on the holdout at ``1 - conformal_alpha``).
    ``crossover`` gives, per factor level, the leading arm by accuracy and the
    minimum ``mis_bind_rate``.
    """
    per_arm: dict[str, dict] = {}
    results_by_arm: dict[str, list[BinderResult]] = {}

    for arm in arms:
        res = run_arm(arm, cohort, graph)
        results_by_arm[arm.name] = res
        entry = {
            "overall": metric_report(res),
            "novel_entity": novel_entity_report(res),
            "slices": {f: slice_by_factor(res, f) for f in factors},
            "patch_cost": patch_cost(arm.patch_mechanism),
            "patch_mechanism": arm.patch_mechanism,
        }
        if holdout is not None:
            hres = run_arm(arm, holdout, graph)
            entry["holdout"] = metric_report(hres)
            entry["synthetic_vs_real_gap"] = synthetic_vs_real_gap(
                entry["overall"], entry["holdout"], key="accuracy")
            gate = ConformalGate(alpha=conformal_alpha)
            gate.calibrate(_arm_scores_for_conformal(arm, holdout, graph))
            entry["conformal"] = gate.empirical_coverage(
                _arm_scores_for_conformal(arm, holdout, graph))
        per_arm[arm.name] = entry

    return {"arms": per_arm, "crossover": _crossover(per_arm, factors)}


def _crossover(per_arm: dict, factors) -> dict:
    """Per factor level, the arm leading on accuracy and the lowest mis_bind_rate,
    plus the overall accuracy leader. The measured line, not a pooled winner."""
    names = list(per_arm)
    overall_leader = max(names, key=lambda n: per_arm[n]["overall"]["accuracy"],
                         default=None)
    by_factor: dict[str, dict] = {}
    for f in factors:
        levels: dict[str, dict] = {}
        all_levels = set()
        for n in names:
            all_levels |= set(per_arm[n]["slices"].get(f, {}))
        for lv in sorted(all_levels):
            acc = {n: per_arm[n]["slices"][f].get(lv, {}).get("accuracy", 0.0)
                   for n in names}
            mis = {n: per_arm[n]["slices"][f].get(lv, {}).get("mis_bind_rate", 1.0)
                   for n in names}
            levels[lv] = {
                "accuracy_leader": max(acc, key=acc.get) if acc else None,
                "accuracy": acc,
                "min_mis_bind": min(mis, key=mis.get) if mis else None,
            }
        by_factor[f] = levels
    return {"overall_accuracy_leader": overall_leader, "by_factor": by_factor}
