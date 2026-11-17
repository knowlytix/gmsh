# SPDX-License-Identifier: Apache-2.0
"""Question -> attribute relevance gate for triple-mediated RAG.

The cap (v-space) gates *graph validity* and tension (u-space) gates *logical
consistency*. Neither, on its own, sees a third failure mode: a query that
retrieves a **true, admissible** fact for an attribute the question never asked
about -- "what is the overdraft interest rate?" binding to ``has_fee_amount``
and returning the (correct) fee. That is a question<->attribute *relevance*
problem, one layer up, at query->relation binding.

There are two gates here.

:class:`RelevanceGate` (legacy) -- an LLM discrimination judgment over the head's
relations, abstaining on "none". It was introduced because raw embedding *cosine*
cannot gate this: a colloquial synonym ("charge") and an absent-but-adjacent
attribute ("interest rate") embed almost equally close to ``has_fee_amount`` in
any single space, so no similarity *floor* separates them. On a small policy the
LLM judge is accurate, but on a richer store (many typed relations, snake-case
names) a 3B model over-abstains, rejecting valid attributes -- and an LLM in the
loop is exactly what GEODE's geometry is meant to replace.

:class:`GeometricRelevanceGate` (preferred) -- the dual-space decision GEODE was
designed for, with no LLM at decision time. Relevance is *not* a single distance;
it is two questions answered in two spaces:

  * **accept (v-space, semantic):** which relation is the question closest to?
    A fine-tuned v-encoder picks the candidate relation by cosine.
  * **veto (u-space, logical):** is the asked attribute the *opposite* of that
    relation -- a distinct attribute that merely sounds money-adjacent? This is a
    *contradiction*, not a distance, and it lives in u-space. Symmetric tension
    ``2 sin(theta/2)`` is monotone in cosine, so it needs a u-encoder *fine-tuned
    for contradiction* (``full`` mode -- a rotation is angle-preserving and cannot
    move tension). Trained on same-relation (consistent) vs cross-relation
    (contradictory) pairs derived from the graph, it learns a tension that
    *generalizes to unseen attributes*: on the banking schema, held-out
    "interest rate" / "annual percentage rate" move from raw-NLI tension ~1.0
    (indistinguishable from synonyms) to >0.9 (clearly contradictory) while
    genuine synonyms ("monthly charge", "amount billed") fall to ~0.55. A query
    is relevant when v-space accepts a relation AND u-space tension to it stays
    below a calibrated contradiction threshold.

v proposes, u vetoes: the same separation of "is it valid / is it consistent"
the cap and tension gates already make, now applied to question<->attribute
relevance instead of a stored triple. The encoders are produced by the GEODE
embed loop (:mod:`knowlytix.knowledge.geode.embed_loop`) and supplied via
``RagConfig.encoder`` (v) and ``RagConfig.relevance_u_encoder`` (u).
"""
from __future__ import annotations

import difflib

_SYSTEM = (
    "You map a question to the ONE entity attribute it asks about.\n"
    "You are given an entity, a list of its known attributes, and a question.\n"
    "Reply with EXACTLY ONE attribute copied VERBATIM from the list, or the single "
    "word NONE.\n"
    "Output only that token -- never the entity name, never extra words.\n"
    "Pick the attribute whose MEANING matches what the question asks for: an "
    "'escalate to' question matches an escalation attribute (not a reporting one); "
    "a colloquial 'charge' or 'cost' is the fee amount.\n"
    "Reply NONE only when NO attribute in the list is what the question asks about "
    "(an interest rate or APR is NOT a fee; a minimum balance is NOT a maximum "
    "reversal)."
)


def _strip_has(rel: str) -> str:
    """Drop a leading ``has_`` prefix from a relation name."""
    return rel[4:] if rel.startswith("has_") else rel


def _norm(s: str) -> str:
    """Lowercase, underscores/whitespace -> single spaces (for verbatim matching)."""
    return " ".join(s.lower().replace("_", " ").split())


class RelevanceGate:
    """LLM gate: does the question's asked attribute match a relation the bound
    head actually has? Returns the matched relation, or ``None`` to abstain."""

    def __init__(self, store, llm, *, encoder=None, prefilter: bool = True,
                 prefilter_floor: float = 0.25, max_tokens: int = 16,
                 exclude_relations: tuple[str, ...] = ("in_section",)):
        self.store = store
        self.llm = llm
        self._encoder = encoder
        self.prefilter = prefilter
        self.prefilter_floor = prefilter_floor
        self.max_tokens = max_tokens
        self.exclude = set(exclude_relations)

    def head_relations(self, head: str) -> list[str]:
        """Relations the graph actually asserts on ``head`` (content only)."""
        rels = {r for _h, r, _t in self.store.query_triples(head=head)
                if r not in self.exclude}
        return sorted(rels)

    def _max_cos(self, question: str, rels: list[str]) -> float:
        """Best cosine between the question and the head's relation phrases."""
        import torch

        if self._encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            self._encoder = encode_texts
        texts = [question] + [_strip_has(r).replace("_", " ") for r in rels]
        emb = torch.nn.functional.normalize(
            torch.as_tensor(self._encoder(texts), dtype=torch.float32), p=2, dim=-1)
        return float((emb[1:] @ emb[0]).max().item())

    def relevant_relation(self, question: str, head: str) -> tuple[str | None, str]:
        """Return ``(matched_relation, detail)``; ``matched_relation is None``
        means the question asks about an attribute the head does not hold -> abstain."""
        rels = self.head_relations(head)
        if not rels:
            return None, "head has no content relations"
        # Hybrid prefilter: reject (cheaply) only when nothing is even close. The
        # floor is conservative, so a plausibly-relevant question always reaches
        # the LLM judge -- the prefilter never decides a borderline case.
        if self.prefilter and self._max_cos(question, rels) < self.prefilter_floor:
            return None, "prefilter: unrelated to all head attributes"
        # Present attributes as readable phrases (snake_case -> words): the raw
        # relation name (e.g. has_exposure_limit_usd_mm) makes a small model both
        # over-abstain on suffixed names and mis-pick among adjacent ones.
        listing = ", ".join(_strip_has(r).replace("_", " ") for r in rels)
        user = (f"Entity: {head}\nKnown attributes: {listing}\n"
                f"Question: {question}\nAnswer:")
        ans = (self.llm.call(system=_SYSTEM, user=user,
                             max_tokens=self.max_tokens) or "").strip()
        # Map the reply back to a real relation by normalized (space-folded)
        # matching, tolerant of morphological variants the model emits ("escalate
        # to" for the relation "escalates to"): containment either way, or a high
        # sequence ratio. The LLM already made the hard relevance call; this only
        # canonicalizes its answer to an exact relation name. NONE / no match abstains.
        ans_n = _norm(ans)
        best_r, best_score = None, 0.0
        for r in rels:
            r_n = _norm(_strip_has(r))
            if not r_n:
                continue
            contained = r_n in ans_n or ans_n in r_n
            ratio = difflib.SequenceMatcher(None, ans_n, r_n).ratio()
            score = 1.0 if contained else ratio
            if score > best_score:
                best_r, best_score = r, score
        if best_r is not None and best_score >= 0.8:
            return best_r, f"matched {best_r} (llm={ans!r})"
        return None, f"none (llm={ans!r})"


def _tension(cos: float) -> float:
    """Tension energy ``2 sin(theta/2)`` from a cosine (unit vectors)."""
    import math

    return 2.0 * math.sin(math.acos(max(-1.0, min(1.0, cos))) / 2.0)


def calibrate_relevance_thresholds(relation_phrasings: dict[str, list[str]],
                                   v_encoder, u_encoder, *,
                                   accept_margin: float = 0.05,
                                   contra_margin: float = 0.02) -> dict:
    """Fit the geometric relevance gate's operating points from the store, the
    same discipline as the cap / tension / accept-threshold gates -- no constants.

    For each relation the build-time phrasings give a labelled cohort:
      * v-accept floor (global): the lowest v-cosine a genuine phrasing has to its
        own relation, minus a margin -- below it, nothing is close enough to accept.
      * u-veto cut (per relation): consistent tension = a phrasing of R vs R's own
        phrase (should pass); contradictory tension = a phrasing of a *different*
        relation vs R's phrase (should be vetoed). The cut is recall-first --
        ``max(consistent) + margin`` -- so a valid attribute is never vetoed; when
        the contradictory band overlaps (``min(contra) <= cut``) the residual leak
        is reported in ``overlap`` rather than sacrificing recall.

    Returns ``{tau_accept, default_tau_contra, tau_contra_per_relation, overlap}``.
    """
    import torch
    import torch.nn.functional as F

    def _vec(enc, texts):
        return F.normalize(torch.as_tensor(enc(texts), dtype=torch.float32), dim=-1)

    rels = [r for r, ws in relation_phrasings.items() if ws]
    phrase = {r: _strip_has(r).replace("_", " ") for r in rels}

    # v-accept floor: min cosine of any phrasing to its own relation phrase.
    accept_floor = 1.0
    for r in rels:
        zr = _vec(v_encoder, [phrase[r]])[0]
        zq = _vec(v_encoder, relation_phrasings[r])
        accept_floor = min(accept_floor, float((zq @ zr).min()))
    tau_accept = max(0.05, accept_floor - accept_margin)

    # u-veto per-relation cut. Precompute each relation's anchor and phrasing
    # encodings ONCE (O(n) encoder calls); the pairwise loop below is then just
    # dot products. Encoding inside the r x r2 loop is O(n^2) encoder forward
    # passes and makes calibration quadratic in the relation count (it stalls on
    # large vocabularies), for identical results since the encoder is per-text.
    u_anchor = {r: _vec(u_encoder, [phrase[r]])[0] for r in rels}
    u_phr = {r: _vec(u_encoder, relation_phrasings[r]) for r in rels}
    per_rel: dict[str, float] = {}
    overlap: dict[str, float] = {}
    cuts = []
    for r in rels:
        ur = u_anchor[r]
        cons = [_tension(float(z @ ur)) for z in u_phr[r]]
        contra = []
        for r2 in rels:
            if r2 == r:
                continue
            for z in u_phr[r2][:3]:
                contra.append(_tension(float(z @ ur)))
        cut = (max(cons) if cons else 0.0) + contra_margin
        per_rel[r] = round(cut, 4)
        cuts.append(cut)
        if contra and min(contra) <= cut:
            overlap[r] = round(min(contra), 4)
    default = round(sorted(cuts)[len(cuts) // 2], 4) if cuts else 1.0  # median
    return {"tau_accept": round(tau_accept, 4), "default_tau_contra": default,
            "tau_contra_per_relation": per_rel, "overlap": overlap}


def score_relevance_cohort(cohort, store, v_encoder, u_encoder=None, *,
                           exclude_relations: tuple[str, ...] = ("in_section",)):
    """Score a labeled question cohort exactly as :class:`GeometricRelevanceGate`
    does, in a few **batched** encoder passes (all questions once, all relation
    phrases once) rather than a per-question loop.

    Calibration itself needs no encoder -- it selects a threshold on scores. This
    function produces those scores so they can be cached and calibrated repeatedly
    (different objective / FAR / folds) without re-encoding. Returns one record per
    question: ``{label, in_scope, v_score, u_tension, best_relation, gold_relation}``,
    where ``v_score`` is the max question->relation cosine over the head's relations
    and ``u_tension`` the contradiction tension to the accepted relation (``None``
    when ``u_encoder`` is omitted). Questions whose head holds no content relation
    are skipped.
    """
    import torch
    import torch.nn.functional as F

    exclude = set(exclude_relations)
    cohort = list(cohort)

    heads = {rec["head"] for rec in cohort}
    head_rels = {h: sorted({r for _h, r, _t in store.query_triples(head=h)
                            if r not in exclude}) for h in heads}
    rels = sorted({r for rs in head_rels.values() for r in rs})
    ridx = {r: i for i, r in enumerate(rels)}
    phrases = [_strip_has(r).replace("_", " ") for r in rels]

    def _enc(enc, texts):
        return F.normalize(torch.as_tensor(enc(texts), dtype=torch.float32), dim=-1)

    if not rels:
        return []
    pv = _enc(v_encoder, phrases)
    qv = _enc(v_encoder, [rec["question"] for rec in cohort])
    pu = qu = None
    if u_encoder is not None:
        pu = _enc(u_encoder, phrases)
        qu = _enc(u_encoder, [rec["question"] for rec in cohort])

    scored = []
    for i, rec in enumerate(cohort):
        rs = head_rels.get(rec["head"], [])
        if not rs:
            continue
        cand = [ridx[r] for r in rs]
        sims = pv[cand] @ qv[i]
        bj = int(sims.argmax())
        best_r = rs[bj]
        u_ten = None if qu is None else _tension(float(qu[i] @ pu[ridx[best_r]]))
        scored.append({
            "label": 1 if bool(rec["in_scope"]) else 0,
            "in_scope": bool(rec["in_scope"]),
            "v_score": float(sims[bj]),
            "u_tension": u_ten,
            "best_relation": best_r,
            "gold_relation": rec.get("relation"),
        })
    return scored


def calibrate_relevance_from_scores(scored, *, objective: str = "recall",
                                    max_false_accept: float = 0.05,
                                    n_folds: int = 1, contra_margin: float = 0.02):
    """Fit the relevance gate's operating points from pre-scored records (the output
    of :func:`score_relevance_cohort`) -- pure threshold selection, **no encoding**.

    ``tau_accept`` is chosen by :func:`select_accept_threshold` under ``objective``
    and the ``max_false_accept`` ceiling, out-of-fold via
    :func:`evaluate_accept_gate_cv` when ``n_folds > 1`` (the persisted threshold is
    then the median of the per-fold thresholds and the reported ``accept`` recall/FAR
    is held-out). Per-relation ``tau_contra`` is fit recall-first on the question
    tension distribution. Returns the ``relevance_calibration.json`` payload plus
    ``{method, objective, max_false_accept, n_folds, accept}``.
    """
    import statistics

    from knowlytix.knowledge.rag.eval import (
        evaluate_accept_gate_cv,
        select_accept_threshold,
    )

    records = [(r["label"], False, r["v_score"]) for r in scored]
    if n_folds and n_folds > 1:
        cv = evaluate_accept_gate_cv(records, k=n_folds,
                                     max_false_accept=max_false_accept,
                                     objective=objective)
        folds = cv.get("fold_thresholds") or []
        tau_accept = round(statistics.median(folds), 4) if folds else 0.05
        accept = {k: cv[k] for k in
                  ("recall", "false_accept", "f1", "accuracy", "n_pos", "n_neg",
                   "k", "fold_thresholds", "heldout_ceiling_met") if k in cv}
        accept["oof"] = True
    else:
        sel = select_accept_threshold(records, max_false_accept=max_false_accept,
                                      objective=objective)
        tau_accept = float(sel["threshold"])
        accept = {k: sel[k] for k in
                  ("recall", "false_accept", "f1", "accuracy", "n_pos", "n_neg")
                  if k in sel}
        accept["oof"] = False

    cons_by_rel: dict[str, list[float]] = {}
    contra_by_rel: dict[str, list[float]] = {}
    for r in scored:
        if r["u_tension"] is None or not r["in_scope"]:
            continue
        bucket = (cons_by_rel if r["gold_relation"] == r["best_relation"]
                  else contra_by_rel)
        bucket.setdefault(r["best_relation"], []).append(r["u_tension"])
    per_rel: dict[str, float] = {}
    overlap: dict[str, float] = {}
    cuts: list[float] = []
    for r, cons in cons_by_rel.items():
        cut = max(cons) + contra_margin          # recall-first: never veto a match
        per_rel[r] = round(cut, 4)
        cuts.append(cut)
        contra = contra_by_rel.get(r, [])
        if contra and min(contra) <= cut:
            overlap[r] = round(min(contra), 4)
    default = round(sorted(cuts)[len(cuts) // 2], 4) if cuts else 1.0
    return {
        "tau_accept": round(tau_accept, 4),
        "default_tau_contra": default,
        "tau_contra_per_relation": per_rel,
        "overlap": overlap,
        "method": "relevance_from_questions_v1",
        "objective": objective,
        "max_false_accept": max_false_accept,
        "n_folds": n_folds,
        "n_scored": len(scored),
        "accept": accept,
    }


def calibrate_relevance_from_questions(
    cohort, store, v_encoder, u_encoder=None, *,
    objective: str = "recall", max_false_accept: float = 0.05,
    n_folds: int = 1, contra_margin: float = 0.02,
    exclude_relations: tuple[str, ...] = ("in_section",),
) -> dict:
    """Fit the geometric relevance gate's operating points from a labeled QUESTION
    cohort, scoring each question exactly as :class:`GeometricRelevanceGate` does
    at inference: the accept score is the max question->relation v-cosine over the
    head's real relations, and the veto score is the question->relation u-tension
    to the accepted relation.

    This differs from :func:`calibrate_relevance_thresholds`, which fits the accept
    floor from relation-phrasing self-similarity (a relation vs its own paraphrases).
    A question sits farther from a relation phrase than that relation's paraphrases
    do, so a floor fit on relation self-similarity is systematically too strict for
    questions and the gate over-abstains. Fitting on the question distribution under
    an explicit objective and a false-accept ceiling removes that mismatch; the
    accept threshold is chosen by :func:`select_accept_threshold`.

    Args:
        cohort: labeled question records, each a mapping with ``question`` (str),
            ``head`` (str), ``relation`` (str: the gold relation when in-scope, the
            absent relation when out-of-scope) and ``in_scope`` (bool -- True is an
            answerable ACCEPT, False an out-of-scope ABSTAIN).
        store: the GMS store; supplies each head's real relations.
        v_encoder / u_encoder: ``list[str] -> vectors`` for the accept (v) and veto
            (u) spaces. ``u_encoder=None`` fits the accept threshold only.
        objective: forwarded to :func:`select_accept_threshold`
            (``recall`` | ``accuracy`` | ``balanced_accuracy`` | ``f1``).
        max_false_accept: hard false-accept ceiling for the accept threshold.
        n_folds: when ``> 1``, choose ``tau_accept`` out-of-fold via
            :func:`evaluate_accept_gate_cv` -- each fold selects the threshold on
            the other folds and scores the held fold, the counts are pooled, and
            the persisted ``tau_accept`` is the median of the per-fold thresholds.
            The reported ``accept`` recall/FAR is then the held-out (unbiased)
            estimate, not the optimistic in-sample one. ``n_folds=1`` selects and
            reports in-sample.
        contra_margin: recall-first slack added to each per-relation veto cut.

    Returns the ``relevance_calibration.json`` payload
    ``{tau_accept, default_tau_contra, tau_contra_per_relation, overlap}`` plus
    ``{method, objective, max_false_accept, accept}`` where ``accept`` carries the
    recall/FAR at the chosen ``tau_accept`` (out-of-fold when ``n_folds > 1``).
    Drop-in for the gate.
    """
    scored = score_relevance_cohort(cohort, store, v_encoder, u_encoder,
                                    exclude_relations=exclude_relations)
    return calibrate_relevance_from_scores(
        scored, objective=objective, max_false_accept=max_false_accept,
        n_folds=n_folds, contra_margin=contra_margin)


class GeometricRelevanceGate:
    """LLM-free relevance: v-space accepts a relation, u-space tension vetoes a
    contradictory (absent-but-adjacent) attribute.

    Args:
        store: provides ``query_triples`` to list a head's asserted relations.
        v_encoder: ``list[str] -> vectors`` for the semantic accept step
            (the GEODE embed loop's v-encoder; falls back to MiniLM).
        u_encoder: ``list[str] -> vectors`` for the contradiction veto
            (the embed loop's ``full``-mode contradiction-tuned u-encoder).
            When ``None`` the veto is skipped (accept-only).
        tau_accept: min v-cosine to accept a candidate relation.
        tau_contra: max u-tension to the accepted relation; above it the asked
            attribute contradicts the relation -> abstain. Per-relation overrides
            may be supplied in ``tau_contra_per_relation``.
    """

    def __init__(self, store, v_encoder=None, u_encoder=None, *,
                 tau_accept: float = 0.30, tau_contra: float = 0.75,
                 tau_contra_per_relation: dict[str, float] | None = None,
                 exclude_relations: tuple[str, ...] = ("in_section",)):
        self.store = store
        self._v = v_encoder
        self._u = u_encoder
        self.tau_accept = tau_accept
        self.tau_contra = tau_contra
        self.tau_contra_per_relation = tau_contra_per_relation or {}
        self.exclude = set(exclude_relations)

    def head_relations(self, head: str) -> list[str]:
        """Return the content relations the graph asserts on ``head``, excluded ones dropped."""
        rels = {r for _h, r, _t in self.store.query_triples(head=head)
                if r not in self.exclude}
        return sorted(rels)

    def _enc(self, which, texts):
        import torch
        enc = self._v if which == "v" else self._u
        if enc is None:
            from knowlytix.core.graph.encoders import encode_texts
            enc = encode_texts
        return torch.nn.functional.normalize(
            torch.as_tensor(enc(texts), dtype=torch.float32), p=2, dim=-1)

    def relevant_relation(self, question: str, head: str) -> tuple[str | None, str]:
        """``(matched_relation, detail)``; ``None`` abstains."""
        rels = self.head_relations(head)
        if not rels:
            return None, "head has no content relations"
        phrases = [_strip_has(r).replace("_", " ") for r in rels]

        # accept (v-space): nearest relation by cosine.
        v = self._enc("v", [question] + phrases)
        sims = (v[1:] @ v[0])
        bi = int(sims.argmax())
        best_r, best_cos = rels[bi], float(sims[bi])
        if best_cos < self.tau_accept:
            return None, f"v-accept: nothing >= tau_accept (best {best_cos:.2f})"

        # veto (u-space): contradiction tension to the accepted relation.
        if self._u is not None:
            u = self._enc("u", [question, phrases[bi]])
            ten = _tension(float(u[0] @ u[1]))
            tau_c = self.tau_contra_per_relation.get(best_r, self.tau_contra)
            if ten > tau_c:
                return None, (f"u-veto: tension {ten:.2f} > {tau_c:.2f} "
                              f"(asked attribute contradicts {best_r})")
            return best_r, f"matched {best_r} (v={best_cos:.2f} u-tension={ten:.2f})"
        return best_r, f"matched {best_r} (v={best_cos:.2f}, no u-veto)"
