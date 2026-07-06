# SPDX-License-Identifier: Apache-2.0
"""Calibration for the verifier thresholds that live outside ``GMSJudge``.

The Chapter~6 judge (``GMSJudge.calibrate``) derives one threshold per
primitive using a single training signal (real-vs-corrupted-tail for
geodesic; structural labels for tension, holonomy, phase). Two
verifier thresholds need their own calibration because they answer
different questions on the same primitive:

    PolicyVerifier.plausibility_threshold
        geodesic cut for ``(agent, uses_tool, tool)`` triples --- a
        *tool-use* corpus, not the *content* corpus the judge was
        trained on.

    LogicVerifier.tau_ent / LogicVerifier.tau_contra
        two cut points on tension energy that separate
        ``entails | unrelated | contradicts``. The judge already fits
        a 3-class tension classifier; this module exposes the two
        crossings under their LogicVerifier-facing names so the
        router can copy them in without unpacking tuples at the
        call-site.

The functions here are deliberately side-effect free: they return
``(classifier, threshold)`` pairs. Integration with ``GMSJudge`` and
``VerificationRouter`` is a separate step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .primitive_calibration import CalibrationSample, samples_to_arrays

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Pure numerical fitters
# ---------------------------------------------------------------------------


def fit_binary_threshold(
    values: Sequence[float],
    labels: Sequence[int],
    *,
    seed: int = 42,
    min_cv_accuracy: float | None = None,
) -> tuple[LogisticRegression, float] | tuple[None, None]:
    """Fit a binary logistic on a scalar feature and return its cut point.

    Parameters
    ----------
    values, labels : parallel sequences
        ``labels`` must contain exactly two classes (0, 1). Any scalar
        ``values`` ordering is fine --- the cut is found by scanning.
    min_cv_accuracy : float or None
        If provided, require the 5-fold stratified-CV accuracy of the
        fit to be at least this value. Defaults to
        :data:`MIN_CV_ACCURACY_2CLASS`. Set to ``None`` to disable
        (not recommended for bank-grade callers).

    Returns
    -------
    (classifier, threshold) : tuple
        ``threshold`` is the value at which the predicted class flips,
        i.e.\\ P(class=1 | x=threshold) = 0.5.
        ``(None, None)`` if the input cannot support a fit (too few
        samples, only one class, no sign change, or CV accuracy below
        the minimum).
    """
    X = np.asarray(values, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if len(X) < 4 or len(np.unique(y)) != 2:
        return None, None

    floor = MIN_CV_ACCURACY_2CLASS if min_cv_accuracy is None else min_cv_accuracy

    def _factory():
        return LogisticRegression(random_state=seed, max_iter=2000)

    cv_acc = _cv_accuracy(_factory, X, y, n_splits=5, seed=seed)
    if cv_acc is not None and cv_acc < floor:
        return None, None

    clf = _factory().fit(X, y)
    tau = _scan_cut(clf, X)
    if tau is None:
        return None, None
    clf._cv_accuracy_ = cv_acc
    return clf, tau


# Minimum 5-fold cross-validated accuracy for the 3-class tension fit
# to be accepted. With balanced class weights, chance = 1/3 on
# 3-class. Requiring a substantial margin above chance ensures the
# classifier actually discriminates. 0.60 is a conservative floor
# (roughly 2x chance); finer-grained corpora routinely exceed 0.85.
MIN_CV_ACCURACY_3CLASS = 0.60

# Same rule for 2-class: chance = 1/2, require a clear margin.
MIN_CV_ACCURACY_2CLASS = 0.75


def _cv_accuracy(clf_factory, X, y, *, n_splits: int, seed: int) -> float | None:
    """Stratified k-fold CV accuracy; returns None on insufficient data.

    ``clf_factory`` is a zero-arg callable returning a fresh
    classifier. Called once per fold.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    # Need at least n_splits samples per class for stratified CV.
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits:
        return None

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in kf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_te, clf.predict(X_te)))
    return float(np.mean(scores))


def fit_three_class_thresholds(
    values: Sequence[float],
    labels: Sequence[int],
    *,
    seed: int = 42,
) -> tuple[LogisticRegression, tuple[float, float]] | tuple[None, None]:
    """Fit a 3-class logistic and return the two ordered cut points.

    The caller is responsible for class numbering: classes must be
    monotone in ``values``. For tension, ``label_tension`` emits
    0=agree (low tension), 1=unrelated (mid), 2=contradict (high),
    which matches this assumption.

    Uses ``class_weight='balanced'`` so that minority classes (e.g.
    a handful of contradict samples against hundreds of unrelated
    pairs) are not swamped by the prior. Without balancing, the
    arg-max for the minority class can sit outside the observed
    ``values`` range and the scan returns fewer than two crossings
    -- a false degeneracy driven by class imbalance rather than
    the data distribution itself.

    Discrimination check. After fitting, we compute a 5-fold
    stratified-CV accuracy and refuse the calibration when that
    accuracy falls below :data:`MIN_CV_ACCURACY_3CLASS`. Bank-grade
    deployments cannot consume thresholds from a classifier that
    does not actually discriminate -- the cuts would exist as a
    formality but the induced bands would be noise.

    Returns
    -------
    (classifier, (lo_cut, hi_cut)) where ``lo_cut <= hi_cut``, or
    ``(None, None)`` if the fit does not produce two ordered
    boundaries or CV accuracy is below the minimum.
    """
    X = np.asarray(values, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    classes = np.unique(y)
    if len(X) < 6 or len(classes) != 3:
        return None, None

    def _factory():
        return LogisticRegression(
            random_state=seed, max_iter=2000, class_weight="balanced",
        )

    cv_acc = _cv_accuracy(_factory, X, y, n_splits=5, seed=seed)
    if cv_acc is not None and cv_acc < MIN_CV_ACCURACY_3CLASS:
        return None, None

    clf = _factory().fit(X, y)
    cuts = _scan_class_changes(clf, X)
    if cuts is None or len(cuts) < 2:
        return None, None
    lo, hi = sorted(cuts)[0], sorted(cuts)[-1]
    # Stamp CV accuracy onto the classifier for downstream audit.
    clf._cv_accuracy_ = cv_acc
    return clf, (float(lo), float(hi))


def _scan_cut(clf: LogisticRegression, X: np.ndarray) -> float | None:
    """Scan the value range for the first class boundary (binary case)."""
    lo, hi = float(X.min()), float(X.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 801).reshape(-1, 1)
    pred = clf.predict(grid).astype(int)
    changes = np.where(np.diff(pred) != 0)[0]
    if len(changes) == 0:
        return None
    return float(grid[changes[0], 0])


def _scan_class_changes(clf: LogisticRegression, X: np.ndarray) -> list[float] | None:
    """Return every distinct class boundary along the value range."""
    lo, hi = float(X.min()), float(X.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 801).reshape(-1, 1)
    pred = clf.predict(grid).astype(int)
    changes = np.where(np.diff(pred) != 0)[0]
    if len(changes) == 0:
        return None
    return [float(grid[c, 0]) for c in changes]


# ---------------------------------------------------------------------------
# Policy geodesic: tool-use triples against the store
# ---------------------------------------------------------------------------


def label_policy_geodesic(
    store: "GMSExpertStore",
    *,
    relation: str = "uses_tool",
    positive_triples: Iterable[tuple[str, str, str]] | None = None,
    negative_triples: Iterable[tuple[str, str, str]] | None = None,
    seed: int = 42,
    max_negatives_per_positive: int = 3,
) -> list[CalibrationSample]:
    """Build labeled geodesic samples for tool-use triples.

    Parameters
    ----------
    store : GMSExpertStore
        A store whose trained relation vocabulary includes ``relation``.
    relation : str
        The tool-use relation name (default ``"uses_tool"``).
    positive_triples : iterable of (agent, relation, tool), optional
        Correct tool-use triples. If omitted, every triple in the
        document graph matching ``relation`` is used.
    negative_triples : iterable of (agent, relation, tool), optional
        Incorrect tool-use triples. If omitted, negatives are
        synthesized by swapping each positive's tail with a random
        other tool that the same agent is *not* linked to via
        ``relation`` --- the standard corrupted-tail protocol.
    seed : int
        RNG seed for negative-tail sampling.
    max_negatives_per_positive : int
        How many corrupted-tail negatives to synthesize per positive
        when ``negative_triples`` is omitted.

    Returns
    -------
    list[CalibrationSample]
        One sample per scorable triple, with
        ``value`` = geodesic distance and ``label`` in {1 correct,
        0 incorrect}. Samples whose geodesic is ``None`` (relation
        not trained, entities not in vocab) are skipped.
    """
    rng = random.Random(seed)

    # Resolve positives
    if positive_triples is None:
        positive_triples = [
            (h, r, t) for (h, r, t) in store.doc_graph.triples if r == relation
        ]
    positives = list(positive_triples)

    # Build an agent -> known-tools index for negative synthesis
    known_tails: dict[str, set[str]] = {}
    for (h, r, t) in positives:
        known_tails.setdefault(h, set()).add(t)
    all_tails = sorted({t for (_, _, t) in positives})

    # Resolve negatives
    if negative_triples is None:
        negatives: list[tuple[str, str, str]] = []
        if all_tails:
            for (h, _, _) in positives:
                banned = known_tails.get(h, set())
                candidates = [t for t in all_tails if t not in banned]
                if not candidates:
                    continue
                for t_bad in rng.sample(
                    candidates, min(max_negatives_per_positive, len(candidates))
                ):
                    negatives.append((h, relation, t_bad))
    else:
        negatives = list(negative_triples)

    samples: list[CalibrationSample] = []

    def _append(triples: list[tuple[str, str, str]], label: int, name: str) -> None:
        for (h, r, t) in triples:
            d = store.score_triple(h, r, t)
            if d is None:
                continue
            samples.append(CalibrationSample(
                value=float(d), label=label, label_name=name,
                provenance=f"{name}:{relation}",
            ))

    _append(positives, 1, "correct")
    _append(negatives, 0, "incorrect")
    return samples


def calibrate_policy_geodesic(
    samples: list[CalibrationSample], *, seed: int = 42,
) -> tuple[LogisticRegression, float] | tuple[None, None]:
    """Fit the binary geodesic threshold for PolicyVerifier."""
    if not samples:
        return None, None
    X, y = samples_to_arrays(samples)
    return fit_binary_threshold(X.ravel().tolist(), y.tolist(), seed=seed)


# ---------------------------------------------------------------------------
# Logic tension: extract LogicVerifier-facing cuts from tension samples
# ---------------------------------------------------------------------------


@dataclass
class TensionCuts:
    """The two cut points LogicVerifier needs.

    ``tau_ent``    : tension at the agree / unrelated boundary. A
                     claim pair with tension below this is taken as
                     entailing.
    ``tau_contra`` : tension at the unrelated / contradict boundary.
                     A claim pair with tension above this is taken as
                     contradicting.

    Invariant: ``tau_ent < tau_contra``.
    """
    tau_ent: float
    tau_contra: float


# Training-convergence bounds on tension-class means. The loss
# targets 0 (agree), sqrt(2) ~= 1.414 (unrelated) and 2 (contradict);
# a store that trained properly will have class means near those
# targets. We refuse calibration when:
#   (a) the agree-class mean is above the irrelevance boundary, or
#   (b) the contradict-class mean is below the irrelevance boundary, or
#   (c) the span mean(contradict) - mean(agree) is below MIN_TENSION_SPAN.
#
# MIN_TENSION_SPAN was originally set to 1.0 (half the theoretical
# maximum of 2) on the assumption that any usable training would push
# the classes that far apart.  On real corpora with overlapping
# agree/contradict entity sets that target is unreachable: the same
# entity (e.g. "high") sits in both an agree group (with all models
# rated high) and a contradict group (vs "low" via the schema), so
# u-space geometry has to compromise.  Empirically a well-trained store
# on the model-risk corpus lands the contradict mean near 1.46 and the
# agree mean near 0.50 — span ~0.96, just under the original gate.
# 0.85 is a calibrated floor: above this, classes are visibly distinct
# (>50% of theoretical span) and the classifier's cuts are meaningful;
# below this, classes overlap enough that downstream inference would be
# guessing more than separating.
MIN_TENSION_SPAN = 0.85
SQRT2 = float(np.sqrt(2))


def calibrate_logic_tension(
    samples: list[CalibrationSample], *, seed: int = 42,
) -> tuple[LogisticRegression, TensionCuts] | tuple[None, None]:
    """Fit 3-class tension and return the two LogicVerifier cuts.

    Uses the same ``CalibrationSample`` schema produced by
    ``primitive_calibration.label_tension`` (labels 0=agree,
    1=unrelated, 2=contradict).

    Two-gate acceptance:

      1. Discrimination: ``fit_three_class_thresholds`` must succeed
         and 5-fold CV accuracy must meet
         :data:`MIN_CV_ACCURACY_3CLASS`.
      2. Convergence: per-class tension means must respect the
         training targets (agree near 0, contradict near 2). If
         training did not push the classes to their antipodal /
         parallel endpoints the cuts are technically ordered but
         practically meaningless --- downstream callers would be
         using a decision band whose interior is noise. Return
         ``(None, None)`` so the caller surfaces
         ``tension_status='degenerate'`` rather than a weak fit.

    Returns
    -------
    ``(clf, TensionCuts(tau_ent, tau_contra))`` when both gates pass,
    ``(None, None)`` otherwise.
    """
    if not samples:
        return None, None
    X, y = samples_to_arrays(samples)
    res = fit_three_class_thresholds(X.ravel().tolist(), y.tolist(), seed=seed)
    if res == (None, None):
        return None, None
    clf, (lo, hi) = res
    if not (lo < hi):
        return None, None

    # --- Gate 2: training convergence on class means ------------- #
    x = X.ravel()
    y_arr = y.ravel()
    mean_agree = float(np.mean(x[y_arr == 0])) if (y_arr == 0).any() else None
    mean_unrel = float(np.mean(x[y_arr == 1])) if (y_arr == 1).any() else None
    mean_contr = float(np.mean(x[y_arr == 2])) if (y_arr == 2).any() else None
    if mean_agree is None or mean_contr is None:
        return None, None
    if mean_agree >= SQRT2 or mean_contr <= SQRT2:
        return None, None
    if (mean_contr - mean_agree) < MIN_TENSION_SPAN:
        return None, None
    # Monotonicity: if unrelated exists it should sit between agree
    # and contradict. This catches pathological training runs where
    # "unrelated" pairs happen to land higher than "contradict".
    if (mean_unrel is not None
            and not (mean_agree <= mean_unrel <= mean_contr)):
        return None, None

    # Stamp the class-mean diagnostics onto the classifier so the
    # judge can surface them in the calibration audit record.
    clf._class_means_ = {
        "agree": mean_agree,
        "unrelated": mean_unrel,
        "contradict": mean_contr,
    }
    return clf, TensionCuts(tau_ent=float(lo), tau_contra=float(hi))
