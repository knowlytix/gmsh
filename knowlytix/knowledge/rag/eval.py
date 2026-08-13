# SPDX-License-Identifier: Apache-2.0
"""Evaluation + calibration for the RAG pipeline (P4).

Two tools to make extraction reliability *measurable* and *tunable*:

* :func:`evaluate` — run a labeled question set through a pipeline and report
  bind-rate, decision correctness, answer accuracy, and abstention precision.
  Use it to quantify the lift from schema-grounded extraction / the repair loop.
* :func:`calibrate_bind_threshold` — fit the embedding-binding similarity
  threshold from labeled positive/negative term pairs (the literal calibration
  step), so synonyms bind and unrelated terms don't.
* :func:`calibrate_accept_threshold` — fit the retrieval-confidence accept gate
  (tau) on a labeled accept/abstain cohort by balanced accuracy under a
  false-accept ceiling, the operating point the pipeline reads.
* :func:`benchmark_query_parse` — isolate the parse→bind step and compare
  parsers (geometric vs LLM) on store-derived questions, scoring recovery of the
  asked ``(head, relation)`` and latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.knowledge.geode.provenance import value_canon
from knowlytix.knowledge.rag.query_triples import QueryTriple

__all__ = ["EvalCase", "EvalReport", "evaluate", "calibrate_bind_threshold",
           "calibrate_accept_threshold", "benchmark_query_parse",
           "score_accept_at_threshold", "select_accept_threshold",
           "evaluate_accept_gate", "evaluate_accept_gate_cv",
           "build_accept_cohort",
           "RetrievalCaseResult", "benchmark_retrieval",
           "GroundingCheck", "grounding_preserved",
           "GroundingFilter", "filter_grounding_preserving"]


@dataclass
class EvalCase:
    question: str
    expected_answer: str | None = None        # substring/value expected in answer
    expect_decision: str | None = None        # "accept" | "abstain" (if known)


@dataclass
class CaseResult:
    case: EvalCase
    decision: str
    bound: bool
    answer_correct: bool | None
    decision_correct: bool | None


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    def _rate(self, pred) -> float:
        vals = [pred(r) for r in self.results]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def bind_rate(self) -> float:
        return self._rate(lambda r: r.bound)

    @property
    def accept_rate(self) -> float:
        return self._rate(lambda r: r.decision == "accept")

    @property
    def answer_accuracy(self) -> float:
        """Accuracy over cases that carried an expected_answer."""
        return self._rate(lambda r: r.answer_correct)

    @property
    def decision_accuracy(self) -> float:
        return self._rate(lambda r: r.decision_correct)

    def as_dict(self) -> dict:
        return {
            "n": len(self.results),
            "bind_rate": self.bind_rate,
            "accept_rate": self.accept_rate,
            "answer_accuracy": self.answer_accuracy,
            "decision_accuracy": self.decision_accuracy,
        }


def _answer_correct(ans, expected: str) -> bool:
    if expected is None:
        return None
    exp = value_canon(expected)
    if exp in value_canon(ans.answer):
        return True
    return any(exp == value_canon(f.tail) for f in ans.sources)


def evaluate(pipeline, cases: list[EvalCase]) -> EvalReport:
    """Run cases through ``pipeline.query`` and score them."""
    results: list[CaseResult] = []
    for case in cases:
        ans = pipeline.query(case.question)
        bound = any(b.bound for b in ans.bound_triples)
        answer_correct = (None if case.expected_answer is None
                          else _answer_correct(ans, case.expected_answer))
        decision_correct = (None if case.expect_decision is None
                            else ans.decision == case.expect_decision)
        results.append(CaseResult(case, ans.decision, bound,
                                  answer_correct, decision_correct))
    return EvalReport(results)


def calibrate_bind_threshold(
    binder, positives: list[tuple[str, str]], negatives: list[str], *,
    grid: tuple[float, ...] = tuple(i / 20 for i in range(1, 20)),
) -> tuple[float, float]:
    """Fit ``binder.bind_threshold`` to best separate positives from negatives.

    Args:
        binder: an embedding-mode :class:`TripleBinder`.
        positives: ``(term, expected_entity)`` that *should* bind.
        negatives: terms that should *not* bind to any entity.
        grid: candidate thresholds to sweep.

    Returns ``(best_threshold, accuracy)`` and sets it on the binder.
    """
    best_th, best_acc = binder.bind_threshold, -1.0
    total = len(positives) + len(negatives)
    for th in grid:
        binder.bind_threshold = th
        tp = sum(binder.bind(QueryTriple(t, "_", "?")).head == exp
                 for t, exp in positives)
        tn = sum(binder.bind(QueryTriple(t, "_", "?")).head is None
                 for t in negatives)
        acc = (tp + tn) / total if total else 0.0
        if acc > best_acc:
            best_acc, best_th = acc, th
    binder.bind_threshold = best_th
    return best_th, best_acc


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (k of n)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return max(0.0, center - half), min(1.0, center + half)


def calibrate_accept_threshold(
    records: list[tuple[int, bool, float]], *, max_false_accept: float = 0.05,
) -> dict:
    """Fit the accept-gate threshold ``tau`` on a labeled accept/abstain cohort.

    Each record is ``(label, abstained, confidence)``: ``label`` is 1 for an
    answerable (ACCEPT) question and 0 for an out-of-scope / no-such-attribute
    (ABSTAIN) question; ``abstained`` is whether a pre-threshold gate (bind-check
    or verify) already abstained (independent of ``tau``); ``confidence`` is the
    pipeline's retrieval confidence. Collect them by running the cohort through a
    pipeline with ``accept_threshold=0`` so its own gate does not pre-filter.

    The objective is balanced accuracy ``0.5*(recall + (1 - false_accept))`` so the
    gate cannot win by abstaining on everything; the ``max_false_accept`` ceiling is
    a tie-breaker, not a hard constraint (a bimodal signal can carry an irreducible
    leak). For a clean bimodal split ``tau`` is set to the midpoint of the gap
    between the highest abstained and the lowest accepted confidence. Returns a
    payload ready to persist as ``rag_gate_calibration.json``.
    """
    def decide(abstained: bool, conf: float, tau: float) -> int:
        return 0 if abstained else (1 if conf >= tau else 0)

    n = len(records)
    n_pos = sum(1 for lbl, _, _ in records if lbl == 1)
    n_neg = n - n_pos
    grid = sorted({round(c, 3) for _, _, c in records} | {0.0, 1.0001})

    def score(tau: float):
        tp = sum(1 for lbl, ab, c in records if lbl == 1 and decide(ab, c, tau))
        fa = sum(1 for lbl, ab, c in records if lbl == 0 and decide(ab, c, tau))
        recall = tp / n_pos if n_pos else 0.0
        false_accept = fa / n_neg if n_neg else 0.0
        bal = 0.5 * (recall + (1.0 - false_accept))
        acc = (tp + (n_neg - fa)) / n if n else 0.0
        return bal, acc, recall, false_accept, false_accept <= max_false_accept

    ranked = sorted(grid, key=lambda t: (score(t)[0], score(t)[4], t), reverse=True)
    tau = ranked[0]
    bal, acc, recall, false_accept, meets = score(tau)
    accepted = [c for lbl, ab, c in records if not ab and decide(ab, c, tau)]
    abstained = [c for _, ab, c in records if ab or not decide(ab, c, tau)]
    if accepted and abstained and min(accepted) > max(abstained):
        tau = round(0.5 * (min(accepted) + max(abstained)), 4)
        bal, acc, recall, false_accept, meets = score(tau)

    ci_lo, ci_hi = _wilson_ci(int(round(acc * n)), n)
    return {
        "method": "accept_threshold_balanced_acc_v2",
        "max_false_accept": max_false_accept,
        "false_accept_ceiling_met": meets,
        "accept_threshold": tau,
        "balanced_accuracy": round(bal, 4),
        "accuracy": round(acc, 4),
        "recall": round(recall, 4),
        "false_accept": round(false_accept, 4),
        "accuracy_ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "cohort_n": n, "n_pos": n_pos, "n_neg": n_neg,
    }


def _gate_counts(records: list[tuple[int, bool, float]], tau: float):
    """``(tp, fp, fn, tn)`` for the accept gate at threshold ``tau``.

    Accept iff not pre-abstained and ``confidence >= tau``. Positive class is
    ACCEPT; a positive label (1) is an answerable question.
    """
    tp = fp = fn = tn = 0
    for lbl, ab, c in records:
        accept = (not ab) and c >= tau
        if lbl == 1:
            tp += accept
            fn += not accept
        else:
            fp += accept
            tn += not accept
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int, tn: int):
    """``(precision, recall, f1, false_accept, accuracy)`` from a confusion count."""
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    false_accept = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return precision, recall, f1, false_accept, accuracy


def score_accept_at_threshold(
    records: list[tuple[int, bool, float]], threshold: float,
) -> dict:
    """Score a *fixed* accept threshold on ``records`` — no re-selection.

    Records are ``(label, abstained, confidence)`` as in
    :func:`calibrate_accept_threshold`. Reports precision/recall/F1 with the
    achieved false-accept rate at ``threshold``, so the metric is not selected on
    the set it is measured on. Use it to score a threshold chosen on a disjoint
    split.
    """
    tp, fp, fn, tn = _gate_counts(records, threshold)
    precision, recall, f1, false_accept, accuracy = _prf(tp, fp, fn, tn)
    n = tp + fp + fn + tn
    ci_lo, ci_hi = _wilson_ci(tp + tn, n)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_accept": round(false_accept, 4),
        "accuracy": round(accuracy, 4),
        "accuracy_ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": n, "n_pos": tp + fn, "n_neg": fp + tn,
    }


_ACCEPT_OBJECTIVES = {
    "f1": lambda p, r, f, fa, a: f,
    "recall": lambda p, r, f, fa, a: r,
    "balanced_accuracy": lambda p, r, f, fa, a: 0.5 * (r + (1.0 - fa)),
    "accuracy": lambda p, r, f, fa, a: a,
}


def select_accept_threshold(
    records: list[tuple[int, bool, float]], *,
    max_false_accept: float = 0.05, objective: str = "f1",
) -> dict:
    """Choose ``tau`` maximizing ``objective`` subject to a **hard** FAR ceiling.

    Unlike :func:`calibrate_accept_threshold`, ``max_false_accept`` is a hard
    constraint: only thresholds whose false-accept rate is at or below the ceiling
    are eligible. When none is (an irreducible leak), the least-leaking threshold
    is returned with ``false_accept_ceiling_met=False``. ``objective`` is one of
    ``f1`` (default), ``recall``, ``balanced_accuracy`` or ``accuracy``. Selecting
    and scoring on the same ``records`` is in-sample; for an honest number pair the
    chosen threshold with a disjoint split via :func:`score_accept_at_threshold` or
    use :func:`evaluate_accept_gate`.
    """
    if objective not in _ACCEPT_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {sorted(_ACCEPT_OBJECTIVES)}")
    obj = _ACCEPT_OBJECTIVES[objective]
    grid = sorted({round(c, 3) for _, _, c in records} | {0.0, 1.0001})

    scored = []
    for tau in grid:
        p, r, f, fa, a = _prf(*_gate_counts(records, tau))
        scored.append((tau, obj(p, r, f, fa, a), fa))

    feasible = [s for s in scored if s[2] <= max_false_accept]
    if feasible:
        # maximize objective; tie-break: lower FAR, then higher (stricter) tau
        tau = max(feasible, key=lambda s: (s[1], -s[2], s[0]))[0]
    else:
        # ceiling unreachable: take the least-leaking threshold
        tau = min(scored, key=lambda s: (s[2], -s[0]))[0]

    # Center tau in the gap for a clean split — an edge threshold sits on the
    # nearest training point and generalizes worse than the gap midpoint (this
    # mirrors :func:`calibrate_accept_threshold`; it leaves the confusion counts
    # unchanged so the objective and FAR are preserved).
    accepted = [c for _, ab, c in records if (not ab) and c >= tau]
    abstained = [c for _, ab, c in records if ab or c < tau]
    if accepted and abstained and min(accepted) > max(abstained):
        tau = round(0.5 * (min(accepted) + max(abstained)), 4)

    out = score_accept_at_threshold(records, tau)
    out["objective"] = objective
    out["max_false_accept"] = max_false_accept
    # The hard ceiling is always satisfiable in-sample (accepting nothing gives
    # FAR 0), so a "ceiling met" flag here would be tautological. Read ``recall``
    # / ``f1`` to see whether the ceiling is met *non-trivially*; read the
    # held-out FAR from :func:`evaluate_accept_gate` to see whether it holds out
    # of sample.
    return out


def evaluate_accept_gate(
    train_records: list[tuple[int, bool, float]],
    heldout_records: list[tuple[int, bool, float]], *,
    max_false_accept: float = 0.05, objective: str = "f1",
) -> dict:
    """Held-out F1 at a fixed false-accept ceiling.

    Select ``tau`` on ``train_records`` under the hard FAR ceiling
    (:func:`select_accept_threshold`), then score it on the disjoint
    ``heldout_records``. The reported held-out F1 and false-accept rate are not
    selected on the set they are measured on — the acceptance signal an outer
    ingestion-tuning loop consumes, and not the in-sample fit of
    :func:`calibrate_accept_threshold`. ``heldout_ceiling_met`` reports whether the
    FAR ceiling still holds out of sample; a True training fit with a False held-out
    flag is an overfit gate.
    """
    train = select_accept_threshold(
        train_records, max_false_accept=max_false_accept, objective=objective)
    heldout = score_accept_at_threshold(heldout_records, train["threshold"])
    return {
        "method": "accept_gate_heldout_f1_v1",
        "objective": objective,
        "max_false_accept": max_false_accept,
        "threshold": train["threshold"],
        "heldout_ceiling_met": heldout["false_accept"] <= max_false_accept,
        "heldout_f1": heldout["f1"],
        "heldout_recall": heldout["recall"],
        "train_f1": train["f1"],
        "train": train,
        "heldout": heldout,
    }


def evaluate_accept_gate_cv(
    records: list[tuple[int, bool, float]], *,
    k: int = 5, max_false_accept: float = 0.05,
    objective: str = "f1", seed: int = 42,
) -> dict:
    """k-fold cross-validated held-out F1@FAR for small cohorts.

    Each fold selects ``tau`` on the other ``k-1`` folds and scores the held fold;
    counts are pooled across folds (micro-average) into a single held-out F1 and
    false-accept rate with a Wilson accuracy interval. ``fold_thresholds`` reports
    per-fold ``tau`` for stability. Preferred where a single train/held-out split is
    too small to trust, as with the per-topic FFIEC cohorts.
    """
    import random

    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]

    tp = fp = fn = tn = 0
    thresholds = []
    for held in folds:
        held_set = set(held)
        train_rec = [records[i] for i in idx if i not in held_set]
        held_rec = [records[i] for i in held]
        if not train_rec or not held_rec:
            continue
        sel = select_accept_threshold(
            train_rec, max_false_accept=max_false_accept, objective=objective)
        thresholds.append(sel["threshold"])
        a, b, c, d = _gate_counts(held_rec, sel["threshold"])
        tp += a
        fp += b
        fn += c
        tn += d

    precision, recall, f1, false_accept, accuracy = _prf(tp, fp, fn, tn)
    n = tp + fp + fn + tn
    ci_lo, ci_hi = _wilson_ci(tp + tn, n)
    return {
        "method": "accept_gate_cv_f1_v1",
        "k": k, "objective": objective, "max_false_accept": max_false_accept,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_accept": round(false_accept, 4),
        "accuracy": round(accuracy, 4),
        "accuracy_ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "heldout_ceiling_met": false_accept <= max_false_accept,
        "fold_thresholds": thresholds,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": n, "n_pos": tp + fn, "n_neg": fp + tn,
    }


def _accept_cohort_bases(store, *, negatives_per_head: int, seed: int):
    """``[(label, head, relation)]`` — the accept cohort's base set.

    One answerable base per distinct ``(head, relation)`` fact in the store
    (``in_section`` excluded, as in :func:`benchmark_query_parse`), plus
    ``negatives_per_head`` relations that exist elsewhere in the store but NOT on
    that head — the out-of-scope probes that expose peripheral-head drift. Seeded,
    so the same store yields the same cohort. Mirrored by
    :func:`knowlytix.knowledge.rag.bakeoff.holdout._answerable_and_absent`.
    """
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

    bases = [(1, h, r) for h, r in facts]
    for h in sorted(rels_by_head):
        absent = sorted(all_rels - rels_by_head[h])
        rng.shuffle(absent)
        bases.extend((0, h, r) for r in absent[:negatives_per_head])
    return bases


def build_accept_cohort(
    store, pipeline, *, negatives_per_head: int = 1, seed: int = 42,
) -> list[tuple[int, bool, float]]:
    """Run a store's answerable/out-of-scope cohort through ``pipeline``.

    Builds the labeled cohort the accept gate is scored on — one templated,
    gold-anchored question per distinct ``(head, relation)`` fact (label 1) plus
    ``negatives_per_head`` questions per head asking for a relation the head does
    not hold (label 0, answerable by no fact in the store) — and returns
    ``(label, abstained, confidence)`` records ready for
    :func:`calibrate_accept_threshold`, :func:`select_accept_threshold` or
    :func:`score_accept_at_threshold`.

    ``abstained`` is the pipeline's own decision (anything other than ``accept``),
    so the cohort measures the pipeline **at its operating point**: pass a pipeline
    configured with ``accept_threshold=0`` when calibrating a threshold, so its gate
    does not pre-filter the records the threshold is being fit on. Retrieval-only
    (``generate=False``) — no synthesis LLM is called. A parser or retrieval failure
    is recorded as an abstention at confidence 0.0 rather than aborting the cohort.
    """
    records: list[tuple[int, bool, float]] = []
    for label, head, relation in _accept_cohort_bases(
            store, negatives_per_head=negatives_per_head, seed=seed):
        try:
            ans = pipeline.query(_parse_question(head, relation), generate=False)
        except Exception:  # noqa: BLE001 - a pipeline failure is a data point
            records.append((label, True, 0.0))
            continue
        records.append((label, ans.decision != "accept", float(ans.confidence)))
    return records


def _parse_question(head: str, relation: str) -> str:
    """A deterministic, gold-anchored question for one fact triple."""
    base = relation[4:] if relation.startswith("has_") else relation
    rel = base.replace("_", " ").strip()
    return f"What is the {rel} for {head.replace('_', ' ').replace('/', ' ')}?"


def benchmark_query_parse(store, parsers: dict, *, binder=None, n: int = 0) -> dict:
    """Compare query parsers on the parse→bind step, holding the binder constant.

    Reverses one templated, gold-anchored question per distinct ``(head, relation)``
    fact in ``store`` (``in_section`` excluded), runs each parser in ``parsers``
    (``{name: parser}``, each with ``.extract(question) -> list[QueryTriple]``),
    binds the result with the shared ``binder`` (defaults to an embedding
    ``TripleBinder`` over the store) and scores whether the asked
    ``(head, relation)`` was recovered. Returns per-parser ``parse_rate``,
    ``bind_rate``, ``recover_rate`` and latency, so a geometric parser and an LLM
    parser are measured on the same questions and the same binder.
    """
    import statistics
    import time

    from knowlytix.knowledge.rag.binding import TripleBinder

    binder = binder or TripleBinder(store, mode="embedding")
    golds, seen = [], set()
    for h, r, _t in store.triples:
        if r == "in_section" or (h, r) in seen:
            continue
        seen.add((h, r))
        golds.append((h, r, _parse_question(h, r)))
    if n:
        golds = golds[:n]

    results: dict = {}
    for name, parser in parsers.items():
        per, lat = [], []
        n_parse = n_bind = n_recover = 0
        for head, rel, q in golds:
            t0 = time.perf_counter()
            try:
                qts = parser.extract(q)
            except Exception:  # noqa: BLE001 - a parser failure is a data point
                qts = []
            lat.append(time.perf_counter() - t0)
            bound = [binder.bind(qt) for qt in qts]
            parsed = len(qts) > 0
            any_bound = any(b.bound for b in bound)
            recovered = any(b.bound and b.head == head and b.relation == rel
                            for b in bound)
            n_parse += parsed
            n_bind += any_bound
            n_recover += recovered
            per.append({"q": q, "gold": [head, rel], "parsed": parsed,
                        "bound": any_bound, "recovered": recovered,
                        "triples": [[b.head, b.relation, b.tail] for b in bound]})
        m = len(golds)
        results[name] = {
            "n": m,
            "parse_rate": round(n_parse / m, 4) if m else 0.0,
            "bind_rate": round(n_bind / m, 4) if m else 0.0,
            "recover_rate": round(n_recover / m, 4) if m else 0.0,
            "latency_ms_mean": round(1000 * statistics.mean(lat), 1) if lat else 0.0,
            "latency_ms_p50": round(1000 * statistics.median(lat), 1) if lat else 0.0,
            "examples": sorted(per, key=lambda r: r["recovered"])[:10],
        }
    return {"n_questions": len(golds), "results": results}


# ---------------------------------------------------------------------------
# Triple-mediated (GMS) vs dense: answer-recall on the same questions/encoder.
# ---------------------------------------------------------------------------
@dataclass
class RetrievalCaseResult:
    question: str
    expected: str | None
    parsed: bool
    bound: bool
    gms_hit: bool | None        # None when no expected_answer to score
    dense_hit: bool | None      # expected value in ANY of the top-k chunks
    miss_stage: str | None      # "parse" | "bind" | "retrieve" | None
    gms_triples: list
    gms_answers: list
    gms_answered: bool = False   # GMS returned an answer (vs abstained / empty)
    dense_p1: bool | None = None  # the TOP-1 chunk carries the expected value
    dense_pk: float | None = None  # fraction of the top-k chunks carrying it


def _value_hit(expected: str, ans) -> bool:
    """True if a retrieval result grounds ``expected``: it appears in the answer
    value or equals an asserted source tail (numeric-aware via ``value_canon``)."""
    exp = value_canon(expected)
    if any(exp == value_canon(str(v)) or exp in value_canon(str(v))
           for v, _ in ans.answers):
        return True
    return any(exp == value_canon(str(f.tail)) for f in ans.facts)


def benchmark_retrieval(store, cases, *, encoder=None, parser=None, binder=None,
                        corpus: str | None = None, ledger=None, top_k: int = 5,
                        include_dense: bool = True) -> dict:
    """Compare triple-mediated (GMS) retrieval against dense top-k answer recall
    on the SAME questions and the SAME ``encoder`` -- so the gap measures the
    retrieval *mechanism*, not the embedding quality.

    The GMS path is the book's spine: ``parser.extract`` -> ``binder.bind`` ->
    :class:`Retriever` over the graph. A case is a GMS hit when a retrieved answer
    value (or an asserted source tail) matches ``case.expected_answer``. The dense
    path chunks the corpus into spans and takes the top-``top_k`` cosine neighbors;
    a hit is ``expected_answer`` appearing in a returned span -- the generous
    standard a chunk-and-pray system is usually judged by. Holding ``encoder``
    fixed across both isolates triple-mediation from similarity search.

    ``cases`` is a list of :class:`EvalCase` (or any objects with ``question`` and
    ``expected_answer``). ``parser`` defaults to a :class:`GeometricQueryParser`
    over ``store`` with ``encoder``; ``binder`` to an embedding
    :class:`TripleBinder` with the same ``encoder``; ``corpus`` to
    ``store.markdown``. Returns per-path ``recall`` AND ``precision`` plus
    ``parse_rate`` / ``bind_rate`` and a per-case :class:`RetrievalCaseResult`
    list whose ``miss_stage`` localizes every GMS miss to parse, bind, or
    retrieve. The two recalls use deliberately asymmetric bars -- a GMS hit is
    *the answer*, a dense hit is the value appearing in *any* of the k chunks --
    so the precision keys are what put them on the same footing:
    ``gms_precision`` (correct / answered), ``dense_precision_at_1`` (top chunk
    right) and ``dense_precision_at_k`` (fraction of the k chunks carrying the
    answer). ``gms_abstention`` scores the prose (expected-None) subset.
    """
    from knowlytix.knowledge.rag.binding import TripleBinder
    from knowlytix.knowledge.rag.query_triples import GeometricQueryParser
    from knowlytix.knowledge.rag.retrieve import Retriever

    parser = parser or GeometricQueryParser(store, encoder=encoder)
    binder = binder or TripleBinder(store, mode="embedding", encoder=encoder)
    retriever = Retriever(store, ledger, top_k=top_k)

    dense = None
    if include_dense:
        from knowlytix.knowledge.rag.dense import (
            InMemoryVectorBackend,
            build_spans,
        )
        md = corpus if corpus is not None else getattr(store, "markdown", "") or ""
        dense = InMemoryVectorBackend(encoder=encoder)
        dense.index(build_spans(md))

    results: list[RetrievalCaseResult] = []
    for case in cases:
        q = case.question
        expected = getattr(case, "expected_answer", None)
        qts = parser.extract(q)
        bound = [binder.bind(t) for t in qts]
        any_bound = any(b.bound for b in bound)
        res = retriever.retrieve(bound)

        gms_answered = len(res.answers) > 0
        gms_hit = dense_hit = None
        dense_p1 = dense_pk = None
        if expected is not None:
            gms_hit = _value_hit(expected, res)
            if dense is not None:
                exp = value_canon(expected)
                contains = [exp in value_canon(s.text)
                            for s, _ in dense.search(q, top_k)]
                dense_hit = any(contains)
                dense_p1 = contains[0] if contains else False
                dense_pk = sum(contains) / top_k if contains else 0.0
        stage = (None if gms_hit else
                 "parse" if not qts else
                 "bind" if not any_bound else "retrieve")
        results.append(RetrievalCaseResult(
            question=q, expected=expected, parsed=bool(qts), bound=any_bound,
            gms_hit=gms_hit, dense_hit=dense_hit, miss_stage=stage,
            gms_triples=[[b.head, b.relation, b.tail] for b in bound],
            gms_answers=[[v, round(c, 3)] for v, c in res.answers[:3]],
            gms_answered=gms_answered, dense_p1=dense_p1, dense_pk=dense_pk))

    def _recall(attr: str) -> float:
        vals = [getattr(r, attr) for r in results
                if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    n = len(results)
    # Precision is scored only over answerable cases (those with an expected
    # value). GMS precision = correct / answered: a wrong-but-answered case
    # counts against it, an abstention does not. Dense has no single answer, so
    # precision@1 (is the top chunk right?) and precision@k (signal/noise of the
    # k chunks) are the comparable measures.
    answerable = [r for r in results if r.expected is not None]
    n_ans = len(answerable)
    g_answered = [r for r in answerable if r.gms_answered]
    g_correct = sum(bool(r.gms_hit) for r in answerable)
    gms_precision = round(g_correct / len(g_answered), 4) if g_answered else 0.0
    # Abstention is scored on the prose subset (expected is None): the right
    # behavior there is to NOT answer.
    prose = [r for r in results if r.expected is None]
    abst_ok = sum(not r.gms_answered for r in prose)
    out = {
        "n": n,
        "n_answerable": n_ans,
        "parse_rate": round(sum(r.parsed for r in results) / n, 4) if n else 0.0,
        "bind_rate": round(sum(r.bound for r in results) / n, 4) if n else 0.0,
        "gms_recall": _recall("gms_hit"),
        "gms_precision": gms_precision,
        "gms_answered": len(g_answered),
        "gms_abstention": (round(abst_ok / len(prose), 4) if prose else None),
        "dense_recall": _recall("dense_hit") if include_dense else None,
        "dense_precision_at_1": _recall("dense_p1") if include_dense else None,
        "dense_precision_at_k": (
            round(sum(r.dense_pk for r in answerable if r.dense_pk is not None)
                  / n_ans, 4) if include_dense and n_ans else None),
        "top_k": top_k,
        "results": results,
    }
    return out


# ---------------------------------------------------------------------------
# Grounding invariance: did a rephrasing preserve what a message binds to?
# ---------------------------------------------------------------------------
@dataclass
class GroundingCheck:
    """Result of checking whether a phrasing grounds where it should.

    ``preserved`` is true when the variant retains every expected policy entity
    (``lost`` empty). ``gained`` lists spurious entities the variant added --- a
    softer drift signal the caller may also choose to reject on.
    """
    variant_heads: list[str] = field(default_factory=list)
    expected_heads: list[str] = field(default_factory=list)
    preserved: bool = False
    lost: list[str] = field(default_factory=list)
    gained: list[str] = field(default_factory=list)


def _recovered_heads(pipeline, message: str) -> set:
    """The policy entities a message recovers along the GMS retrieval *path*.

    Uses ``pipeline.query(generate=False)`` --- parse, bind, relevance, retrieve
    and the calibrated accept gate, but no generation --- so an answer reachable
    through the graph (via the relevance gate, a multi-candidate retrieval or an
    alias edge) counts even when it is not the parser's top bound head. This is
    the "valid in the GMS path" test: a rephrasing is preserved when the expected
    answer is still recoverable here, not merely when the parse head matched.
    """
    ans = pipeline.query(message, generate=False)
    srcs = getattr(ans, "sources", None) or []
    return {getattr(s, "head", None) for s in srcs} - {None}


def grounding_preserved(pipeline, variant: str, *,
                        reference: str | None = None,
                        expected: "str | set[str] | list[str] | None" = None,
                        match: str = "all",
                        normalize=None) -> GroundingCheck:
    """Check whether a rephrased message still grounds to the expected policy.

    Enrichment varies a message's surface under factors declared ground-truth
    invariant; invariance is a contract, not a guarantee --- an out-of-vocabulary
    alias or a corrupting typo can quietly move the message to a different policy.

    Recoverability is judged along the GMS *path*, not the parser's top bound head:
    ``variant`` is run through ``pipeline.query(generate=False)`` (retrieve + the
    calibrated accept gate, no generation) and the recovered policy entities are
    compared to the expected set --- so a rephrasing the graph can still resolve to
    the right policy (via the relevance gate, a multi-candidate retrieval or an
    alias edge) counts even when the top parse head differs.

    Args:
        expected: the target grounding --- a known entity or set (the answer-key
            form, preferred). Mutually exclusive with ``reference``.
        reference: a canonical phrasing whose own recovered path defines the target.
        match: ``"all"`` (default) preserves only when *every* expected entity is
            recovered (strict label-preservation); ``"any"`` preserves when the
            variant recovers *at least one* expected entity --- the **downstream-
            success** test against a set of *acceptable* answers (the frame BPP
            Chapter 15 grades ``extract_facts`` by: did the query retrieve a right
            policy, not did a head string match).
        normalize: optional ``str -> str`` applied to every head and expected term
            before comparison --- e.g. ``lambda h: h.split("/")[0]`` to compare on
            the policy stem (``overdraft/per_occurrence`` -> ``overdraft``).
    """
    norm = normalize or (lambda h: h)

    def recover(msg):
        return {norm(h) for h in _recovered_heads(pipeline, msg)}

    if expected is not None:
        exp = {norm(expected)} if isinstance(expected, str) else {norm(e) for e in expected}
    elif reference is not None:
        exp = recover(reference)
    else:
        raise ValueError("provide expected= (entity/set) or reference= (a phrasing)")
    var = recover(variant)
    lost = exp - var
    if match == "any":
        preserved = bool(exp) and bool(exp & var)
    elif match == "all":
        preserved = bool(exp) and not lost
    else:
        raise ValueError(f"match must be 'all' or 'any', got {match!r}")
    return GroundingCheck(
        variant_heads=sorted(var),
        expected_heads=sorted(exp),
        preserved=preserved,
        lost=sorted(lost),
        gained=sorted(var - exp),
    )


@dataclass
class GroundingFilter:
    """Outcome of filtering a set of (rephrased) items by grounding preservation.

    ``dropped`` keeps the offending item *with* its :class:`GroundingCheck`, so a
    filtered corpus reports what it removed and why --- the drop is never silent.
    """
    kept: list = field(default_factory=list)
    dropped: list = field(default_factory=list)        # list of (item, GroundingCheck)
    n_in: int = 0

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)


def filter_grounding_preserving(pipeline, items, *,
                                message=lambda x: x["message"],
                                expected=lambda x: x.get("expected"),
                                reference=None,
                                match: str = "all",
                                normalize=None,
                                drop_on_gain: bool = False) -> GroundingFilter:
    """Keep only items whose rephrasing preserved the grounded label; drop the rest.

    For each item, parse-and-bind its message (``message(item)``) and compare to
    its expected grounding (``expected(item)`` --- the answer-key entity/set) or,
    when ``reference`` is given, to ``reference(item)``'s own binding. An item is
    kept when :func:`grounding_preserved` reports ``preserved`` under ``match`` /
    ``normalize`` (use ``match="any"`` + a stem ``normalize`` for the downstream-
    success frame: keep an item whose query retrieves any *acceptable* policy).
    With ``drop_on_gain=True`` an item is also dropped when the variant added a
    spurious entity. The result carries the kept items and the dropped ones paired
    with their check, so a training corpus or a test suite is cleaned of rephrasings
    that moved the label, with an auditable record of every removal.
    """
    out = GroundingFilter()
    for it in items:
        out.n_in += 1
        ref = reference(it) if reference is not None else None
        gc = grounding_preserved(pipeline, message(it),
                                 expected=None if ref is not None else expected(it),
                                 reference=ref, match=match, normalize=normalize)
        keep = gc.preserved and not (drop_on_gain and gc.gained)
        if keep:
            out.kept.append(it)
        else:
            out.dropped.append((it, gc))
    return out
