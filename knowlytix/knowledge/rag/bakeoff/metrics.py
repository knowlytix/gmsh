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


# ---------------------------------------------------------------------------
# End-to-end accept-threshold calibration.
#
# The accept gate only decides whether an already-bound triple is kept; the
# downstream outcome (a correct or mis-bind, or an absent query's false-accept)
# is fixed once the query is bound, and is independent of the threshold. So a
# single pass with the gate open (tau = -inf) records, per query, the accept
# score and the outcome-if-accepted, and any candidate threshold is scored by
# re-thresholding those records -- no second pipeline pass. Calibrating against
# these end-to-end outcomes, rather than the accept-stage false-accept rate,
# credits the protection downstream binding already provides: where binding
# rejects out-of-schema queries the false-accept constraint is slack and the
# objective drives the threshold to its floor; where a wrong bind executes to a
# plausible answer the mis-bind ceiling holds the threshold up.
# ---------------------------------------------------------------------------


def score_accept_end_to_end(records, threshold: float) -> dict:
    """End-to-end confusion and metrics at a fixed accept ``threshold``.

    Each record is a mapping with ``v_score`` (the relevance-gate accept score,
    or ``None`` if the query never bound so the gate never ran), ``is_absent``
    (relation-absent gold), ``would_correct`` (answerable and the bound query
    executes to gold -- threshold-independent), and ``bound`` (a schema-valid
    triple was bound). A query is accepted iff it bound and ``v_score >= threshold``.
    """
    cb = mb = fb = ca = fa = 0
    for r in records:
        vs = r.get("v_score")
        accepted = bool(r.get("bound")) and vs is not None and vs >= threshold
        if r.get("is_absent"):
            fa += accepted
            ca += not accepted
        elif not accepted:
            fb += 1
        elif r.get("would_correct"):
            cb += 1
        else:
            mb += 1
    na = cb + mb + fb
    nabs = ca + fa
    n = na + nabs
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round((cb + ca) / n, 4) if n else 0.0,
        "correct_bind_rate": round(cb / na, 4) if na else 0.0,
        "mis_bind_rate": round(mb / na, 4) if na else 0.0,
        "fail_bind_rate": round(fb / na, 4) if na else 0.0,
        "oos_false_accept": round(fa / nabs, 4) if nabs else 0.0,
        "n": n, "n_answerable": na, "n_absent": nabs,
        "correct_bind": cb, "mis_bind": mb, "fail_bind": fb,
        "correct_abstain": ca, "false_accept": fa,
    }


_E2E_OBJECTIVES = {
    "accuracy": lambda m: m["accuracy"],
    "recall": lambda m: m["correct_bind_rate"],
}


def _e2e_pick(records, grid, obj, mis_ceil, far_ceil):
    feasible = []
    for t in grid:
        m = score_accept_end_to_end(records, t)
        if m["mis_bind_rate"] <= mis_ceil and m["oos_false_accept"] <= far_ceil:
            feasible.append((t, obj(m)))
    if feasible:                      # maximize objective; tie-break lower tau (recall)
        return max(feasible, key=lambda s: (s[1], -s[0]))[0]
    return min(grid, key=lambda t: score_accept_end_to_end(records, t)["mis_bind_rate"])


def calibrate_accept_end_to_end(records, *, objective: str = "accuracy",
                                mis_bind_ceiling: float = 0.05,
                                oos_far_ceiling: float = 0.05,
                                n_folds: int = 1, seed: int = 42) -> dict:
    """Choose the accept threshold against end-to-end outcomes under hard mis-bind
    and out-of-schema false-accept ceilings, maximizing ``objective`` (``accuracy``
    or ``recall``). Records are those of :func:`score_accept_end_to_end`, captured
    once with the gate open. When ``n_folds > 1`` the threshold is selected
    out-of-fold (persisted as the median per-fold threshold) and the reported
    metrics are pooled across held folds. Returns the metric payload at the
    persisted threshold plus ``{objective, mis_bind_ceiling, oos_far_ceiling,
    n_folds, oof, fold_thresholds}``.
    """
    import statistics
    if objective not in _E2E_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_E2E_OBJECTIVES)}")
    obj = _E2E_OBJECTIVES[objective]
    grid = sorted({round(r["v_score"], 3) for r in records
                   if r.get("v_score") is not None} | {-1.0, 1.0001})

    if n_folds and n_folds > 1:
        import random
        idx = list(range(len(records)))
        random.Random(seed).shuffle(idx)
        folds = [idx[i::n_folds] for i in range(n_folds)]
        taus, pooled = [], []
        for held in folds:
            hs = set(held)
            train = [records[i] for i in idx if i not in hs]
            heldrecs = [records[i] for i in held]
            if not train or not heldrecs:
                continue
            t = _e2e_pick(train, grid, obj, mis_bind_ceiling, oos_far_ceiling)
            taus.append(t)
            pooled.extend((records[i], t) for i in held)
        tau = round(statistics.median(taus), 4) if taus else 0.0
        # pooled held-out confusion: each held record scored at its fold's tau.
        cb = mb = fb = ca = fa = 0
        for r, t in pooled:
            vs = r.get("v_score")
            accepted = bool(r.get("bound")) and vs is not None and vs >= t
            if r.get("is_absent"):
                fa += accepted
                ca += not accepted
            elif not accepted:
                fb += 1
            elif r.get("would_correct"):
                cb += 1
            else:
                mb += 1
        na, nabs = cb + mb + fb, ca + fa
        n = na + nabs
        m = {"threshold": tau,
             "accuracy": round((cb + ca) / n, 4) if n else 0.0,
             "correct_bind_rate": round(cb / na, 4) if na else 0.0,
             "mis_bind_rate": round(mb / na, 4) if na else 0.0,
             "fail_bind_rate": round(fb / na, 4) if na else 0.0,
             "oos_false_accept": round(fa / nabs, 4) if nabs else 0.0,
             "n": n, "n_answerable": na, "n_absent": nabs}
        m["fold_thresholds"] = [round(t, 4) for t in taus]
        m["oof"] = True
    else:
        tau = _e2e_pick(records, grid, obj, mis_bind_ceiling, oos_far_ceiling)
        m = score_accept_end_to_end(records, tau)
        m["oof"] = False
    m.update({"objective": objective, "mis_bind_ceiling": mis_bind_ceiling,
              "oos_far_ceiling": oos_far_ceiling, "n_folds": n_folds})
    return m


# ---------------------------------------------------------------------------
# Joint end-to-end calibration of BOTH relevance gates (accept + veto).
#
# The accept threshold and the veto threshold both only decide whether an
# already-bound triple is kept; the bound outcome is fixed. A single gates-open
# pass that records, per query, the accept score AND the veto tension AND the
# outcome-if-committed lets any (tau_accept, tau_contra) pair be scored offline.
# Calibrating both against end-to-end outcomes decides, per gate, whether it earns
# its recall cost: a gate that reduces neither mis-bind nor false-accept is driven
# to its permissive limit (accept threshold to its floor, veto tension to its
# ceiling), which is the calibrated way to conclude a gate is redundant on a store
# where binding already enforces scope.
# ---------------------------------------------------------------------------


def score_relevance_end_to_end(records, tau_accept: float, tau_contra: float) -> dict:
    """End-to-end confusion at a joint operating point ``(tau_accept, tau_contra)``.

    A query is committed iff it bound, ``v_score >= tau_accept``, and (when a veto
    tension was recorded) ``u_tension <= tau_contra``. Records extend
    :func:`score_accept_end_to_end` with an optional ``u_tension`` (``None`` skips
    the veto for that query)."""
    cb = mb = fb = ca = fa = 0
    for r in records:
        vs, ut = r.get("v_score"), r.get("u_tension")
        committed = (bool(r.get("bound")) and vs is not None and vs >= tau_accept
                     and (ut is None or ut <= tau_contra))
        if r.get("is_absent"):
            fa += committed
            ca += not committed
        elif not committed:
            fb += 1
        elif r.get("would_correct"):
            cb += 1
        else:
            mb += 1
    na, nabs = cb + mb + fb, ca + fa
    n = na + nabs
    return {
        "tau_accept": round(float(tau_accept), 4),
        "tau_contra": round(float(tau_contra), 4),
        "accuracy": round((cb + ca) / n, 4) if n else 0.0,
        "correct_bind_rate": round(cb / na, 4) if na else 0.0,
        "mis_bind_rate": round(mb / na, 4) if na else 0.0,
        "fail_bind_rate": round(fb / na, 4) if na else 0.0,
        "oos_false_accept": round(fa / nabs, 4) if nabs else 0.0,
        "n": n, "n_answerable": na, "n_absent": nabs,
        "correct_bind": cb, "mis_bind": mb, "fail_bind": fb,
        "correct_abstain": ca, "false_accept": fa,
    }


def _grid(values, cap=40):
    """A bounded ascending grid over observed scores plus permissive sentinels."""
    uniq = sorted({round(v, 3) for v in values if v is not None})
    if len(uniq) > cap:
        step = len(uniq) / cap
        uniq = [uniq[min(len(uniq) - 1, int(i * step))] for i in range(cap)]
    return uniq


def calibrate_relevance_end_to_end(records, *, objective: str = "accuracy",
                                   mis_bind_ceiling: float = 0.05,
                                   oos_far_ceiling: float = 0.05,
                                   n_folds: int = 1, seed: int = 42) -> dict:
    """Jointly select ``(tau_accept, tau_contra)`` against end-to-end outcomes under
    hard mis-bind and out-of-schema false-accept ceilings, maximizing ``objective``
    (``accuracy`` or ``recall``). Records are those of
    :func:`score_relevance_end_to_end`, captured once with both gates open. A gate
    that buys no reduction in mis-bind or false-accept is driven to its permissive
    limit -- ``tau_accept`` to a floor below every score, ``tau_contra`` to a ceiling
    above every tension. Out-of-fold when ``n_folds > 1`` (persisted thresholds are
    the per-fold medians; reported metrics are pooled held-out)."""
    import statistics
    if objective not in _E2E_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_E2E_OBJECTIVES)}")
    obj = _E2E_OBJECTIVES[objective]
    ta_grid = [-1.0] + _grid([r.get("v_score") for r in records])
    tc_grid = _grid([r.get("u_tension") for r in records]) + [9.0]

    def pick(recs):
        feasible = []
        for ta in ta_grid:
            for tc in tc_grid:
                m = score_relevance_end_to_end(recs, ta, tc)
                if (m["mis_bind_rate"] <= mis_bind_ceiling
                        and m["oos_false_accept"] <= oos_far_ceiling):
                    # maximize objective; tie-break toward permissive gates (recall)
                    feasible.append((obj(m), -ta, tc, ta))
        if feasible:
            best = max(feasible)
            return best[3], best[2]
        # ceilings unreachable: least mis-binding point
        allpts = [(score_relevance_end_to_end(recs, ta, tc)["mis_bind_rate"], ta, tc)
                  for ta in ta_grid for tc in tc_grid]
        _, ta, tc = min(allpts)
        return ta, tc

    if n_folds and n_folds > 1:
        import random
        idx = list(range(len(records)))
        random.Random(seed).shuffle(idx)
        folds = [idx[i::n_folds] for i in range(n_folds)]
        tas, tcs, pooled = [], [], []
        for held in folds:
            hs = set(held)
            train = [records[i] for i in idx if i not in hs]
            if not train or not held:
                continue
            ta, tc = pick(train)
            tas.append(ta)
            tcs.append(tc)
            pooled.append(([records[i] for i in held], ta, tc))
        ta = round(statistics.median(tas), 4) if tas else -1.0
        tc = round(statistics.median(tcs), 4) if tcs else 9.0
        cb = mb = fb = ca = fa = 0
        for recs, fta, ftc in pooled:
            s = score_relevance_end_to_end(recs, fta, ftc)
            cb += s["correct_bind"]
            mb += s["mis_bind"]
            fb += s["fail_bind"]
            ca += s["correct_abstain"]
            fa += s["false_accept"]
        na, nabs = cb + mb + fb, ca + fa
        n = na + nabs
        m = {"tau_accept": ta, "tau_contra": tc,
             "accuracy": round((cb + ca) / n, 4) if n else 0.0,
             "correct_bind_rate": round(cb / na, 4) if na else 0.0,
             "mis_bind_rate": round(mb / na, 4) if na else 0.0,
             "fail_bind_rate": round(fb / na, 4) if na else 0.0,
             "oos_false_accept": round(fa / nabs, 4) if nabs else 0.0,
             "n": n, "n_answerable": na, "n_absent": nabs}
        m["fold_thresholds"] = [(round(a, 4), round(c, 4)) for a, c in zip(tas, tcs)]
        m["oof"] = True
    else:
        ta, tc = pick(records)
        m = score_relevance_end_to_end(records, ta, tc)
        m["oof"] = False
    m.update({"objective": objective, "mis_bind_ceiling": mis_bind_ceiling,
              "oos_far_ceiling": oos_far_ceiling, "n_folds": n_folds})
    return m
