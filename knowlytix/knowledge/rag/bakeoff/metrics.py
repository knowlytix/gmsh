# SPDX-License-Identifier: Apache-2.0
"""Bake-off metric catalog and factor-sliced scoring (spec Section 6).

Accuracy alone does not decide a binder; the *character* of its failures does. A
binder that refuses when it cannot bind is safe; one that binds to a schema-valid
but wrong relation is confidently wrong and expensive to catch. These metrics
separate those cases, over a common per-query record so every arm is scored the
same way, and can be sliced by realized factor to locate the reliability
crossover rather than a pooled leaderboard.

Per-query outcomes over the *answerable* queries partition into:

* **correct bind** — accepted and the bound query executes to the gold answer;
* **mis-bind** — accepted but executes to the wrong answer (the critical failure);
* **fail-bind** — abstained (a safe failure: it caught what a mis-bind would have
  taken silently, at the cost of recall).

Relation-absent queries are scored by whether the binder abstained (correct) or
answered (a false accept). ``patch_cost`` is a declared per-arm property (how a
discovered mis-binding is fixed), not computed here; ``synthetic_vs_real_gap`` is
the accuracy drop from the synthetic cohort to the real-language holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .roundtrip import _answers_match

# patch_cost: steps to correct a discovered mis-binding, by resolution mechanism.
# An alias-table edit is a config change; an adapter re-tune is a training run on
# a small delta; a full retrain rebuilds the binder. Declared per arm (I6).
PATCH_COST_STEPS = {"alias_edit": 1, "adapter_retune": 2, "full_retrain": 3}


@dataclass
class BinderResult:
    """One arm's outcome on one query, in the common scoring shape.

    ``pred_answer``/``gold_answer`` are the graph-executed answers (see
    :func:`knowlytix.knowledge.rag.bakeoff.graph_execute`); ``pred_answer`` should
    be filled even when ``decision == "abstain"`` (the binding the arm *would*
    have committed), so over-abstention can be told from a caught mis-bind.
    """

    is_absent: bool                       # gold: relation-absent (correct = abstain)
    decision: str                         # "accept" | "abstain"
    gold_answer: object = None
    pred_answer: object = None
    bound: bool = False                   # all non-variable slots bound (schema-valid, complete)
    novel: bool = False                   # head/relation unseen in training
    factors: dict = field(default_factory=dict)
    latency_ms: float | None = None

    def _would_be_correct(self) -> bool:
        return (not self.is_absent) and _answers_match(self.pred_answer, self.gold_answer)

    @property
    def outcome(self) -> str:
        """One of correct_bind / mis_bind / fail_bind (answerable) or
        correct_abstain / false_accept (absent)."""
        if self.is_absent:
            return "correct_abstain" if self.decision == "abstain" else "false_accept"
        if self.decision == "abstain":
            return "fail_bind"
        return "correct_bind" if _answers_match(self.pred_answer, self.gold_answer) else "mis_bind"


def metric_report(results: list[BinderResult]) -> dict:
    """Full catalog over a set of results. Rates over answerable queries unless
    noted; ``accuracy`` is over all queries (a correct bind or a correct
    abstention). ``mis_bind_rate`` is the load-bearing column."""
    n = len(results)
    ans = [r for r in results if not r.is_absent]
    absent = [r for r in results if r.is_absent]
    na = len(ans) or 1

    correct_bind = sum(r.outcome == "correct_bind" for r in ans)
    mis_bind = sum(r.outcome == "mis_bind" for r in ans)
    fail_bind = sum(r.outcome == "fail_bind" for r in ans)
    false_accept = sum(r.outcome == "false_accept" for r in absent)
    correct_abstain = sum(r.outcome == "correct_abstain" for r in absent)

    abstained = [r for r in results if r.decision == "abstain"]
    good_abstain = sum(not r._would_be_correct() for r in abstained)

    lat = [r.latency_ms for r in results if r.latency_ms is not None]
    correct_all = correct_bind + correct_abstain

    return {
        "n": n,
        "n_answerable": len(ans),
        "n_absent": len(absent),
        "bind_rate": round(sum(r.decision == "accept" and r.bound for r in ans) / na, 4),
        "correct_bind_rate": round(correct_bind / na, 4),
        "mis_bind_rate": round(mis_bind / na, 4),
        "fail_bind_rate": round(fail_bind / na, 4),
        "oos_false_accept": round(false_accept / (len(absent) or 1), 4),
        "abstention_quality": round(good_abstain / len(abstained), 4) if abstained else None,
        "accuracy": round(correct_all / n, 4) if n else 0.0,
        "latency_ms_mean": round(sum(lat) / len(lat), 2) if lat else None,
    }


def novel_entity_report(results: list[BinderResult]) -> dict:
    """The catalog restricted to novel (unseen head/relation) queries -- where a
    finetuned binder is expected to degrade and a patchable one to hold."""
    rep = metric_report([r for r in results if r.novel])
    rep["novel_entity_score"] = rep["correct_bind_rate"]   # named alias for the spec
    return rep


def slice_by_factor(results: list[BinderResult], factor: str) -> dict:
    """``{level: metric_report}`` grouped by a realized factor level, for the
    per-slice crossover analysis. Rows missing the factor are skipped."""
    groups: dict[str, list[BinderResult]] = {}
    for r in results:
        lv = (r.factors or {}).get(factor)
        if lv is None:
            continue
        groups.setdefault(str(lv), []).append(r)
    return {lv: metric_report(rs) for lv, rs in sorted(groups.items())}


def synthetic_vs_real_gap(synthetic: dict, real: dict, *, key: str = "accuracy") -> float:
    """``metric(synthetic) - metric(real)`` for a report key (default accuracy):
    the distribution-mismatch meter. A large positive gap means the arm looks
    good on the synthetic cohort but drops on real language."""
    a, b = synthetic.get(key), real.get(key)
    if a is None or b is None:
        return 0.0
    return round(float(a) - float(b), 4)


def patch_cost(mechanism: str) -> int:
    """Declared patch cost (steps) for an arm's entity-resolution mechanism; see
    :data:`PATCH_COST_STEPS`. Unknown mechanisms cost the maximum (a full retrain)."""
    return PATCH_COST_STEPS.get(mechanism, max(PATCH_COST_STEPS.values()))
