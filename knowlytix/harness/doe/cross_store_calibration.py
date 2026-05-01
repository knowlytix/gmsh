# SPDX-License-Identifier: Apache-2.0
"""Cross-store tension calibration for multi-agent / cross-document testing.

Per-claim tension (`GMSJudge.calibrate`) measures how often a single store
rejects a claim; its calibration regime is **intra-store**.  When two agents
hold different stores, or two source documents disagree, the signal of
interest is the disagreement between the two stores on the same
``(h, r, t)`` triple.  A scalar ``score_triple`` delta is not sufficient:
numeric revisions orphan the original tail from amended's vocabulary, so
half the MODIFIED training samples have ``amended.score_triple`` = ``None``.
Categorical flips keep the tail (because other rows still use it), so their
signal is pure delta magnitude.  One feature cannot cover both regimes.

This module fits a **3-feature** cross-store calibration:

    features:
        delta            = |score_base(h,r,t) - score_amended(h,r,t)|
                           (0.0 when either side returns None)
        tail_missing     = 1 if t not in amended.entity_vocab else 0
        has_replacement  = 1 if amended has (h, r, t') for some t' != t
                           else 0
    classes:   0 UNCHANGED       triple present verbatim in both stores
               1 MODIFIED        numeric value revised (tail often orphaned)
               2 CONTRADICTED    categorical value replaced (tail preserved)
    fit:       multinomial LogisticRegression  +  CalibratedClassifierCV
               (prefit, isotonic) for proper probability calibration
    metrics:   confusion matrix, per-class precision/recall,
               expected calibration error, Brier score, reliability bins
    outputs:   calibrated 3-class classifier -- inference uses
               ``predict_proba`` directly; scalar tau thresholds are
               deliberately not extracted because the 1D monotonic
               assumption (UNCHANGED < MODIFIED < CONTRADICTED on delta)
               does not hold -- MODIFIED has larger delta than
               CONTRADICTED when the tail survives.

The architecture mirrors `primitive_calibration.py` / `threshold_calibration.py`
so auditors can trace the bank-grade calibration pipeline through a single
consistent pattern.

Persisted artifacts (see `save_cross_store_calibration`):
    cross_store_calibration.pkl           calibrated classifier (3D input)
    cross_store_calibration.json          feature schema + metadata
    cross_store_calibration_metrics.json  full metrics bundle
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


SCHEMA_VERSION = 2
CLASS_NAMES = ("UNCHANGED", "MODIFIED", "CONTRADICTED")
FEATURE_NAMES = ("delta", "tail_missing", "has_replacement")


@dataclass
class CrossStoreSample:
    head: str
    relation: str
    tail: str
    score_base: float | None
    score_amended: float | None
    delta: float             # 0.0 when either score is None
    tail_missing: int        # 1 if tail not in amended's entity vocab
    has_replacement: int     # 1 if amended has (h, r, t') with t' != t
    label: int
    label_name: str
    provenance: str

    def features(self) -> list[float]:
        return [float(self.delta),
                float(self.tail_missing),
                float(self.has_replacement)]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


def label_cross_store_tension(
    base_store: "GMSExpertStore",
    amended_store: "GMSExpertStore",
    manifest: dict,
) -> list[CrossStoreSample]:
    """Produce one calibration sample per base-store triple.

    Granularity.  One sample per ``(h, r, t)`` triple in ``base_store``.
    Per-triple (not per-head) because severity belongs to the triple --
    a head may have many triples, only some perturbed, and aggregating
    to a single scalar collapses the signal.

    Feature vector (3-D, see module docstring):

    - ``delta``            : ``|score_base - score_amended|``, or 0 if
                             either side returns None (tail absent or
                             embedding lookup failure).
    - ``tail_missing``     : 1 if ``t`` is not in amended's entity
                             vocabulary.  Hits for most MODIFIED triples
                             because numeric revisions orphan the
                             original tail.
    - ``has_replacement``  : 1 if amended has ``(h, r, t')`` for any
                             ``t' != t``.  Fires on both MODIFIED and
                             CONTRADICTED; helps separate UNCHANGED
                             (where amended preserves ``(h, r, t)``)
                             from the disagreement classes.

    Labels.  A triple is confirmed MODIFIED/CONTRADICTED only when the
    manifest records a perturbation whose ``(relation_hint, original)``
    matches, *and* amended contains ``(h, r, replacement)``, *and*
    amended does not still contain the base triple.  All other triples
    are UNCHANGED.  This is strict by design: an unconfirmed candidate
    (e.g., the generator sampled this row out at p=0.85) is treated as
    UNCHANGED rather than creating a noisy MODIFIED label.
    """
    # (relation, original_tail) -> (label, label_name, replacement_tail)
    perturb_by_pair: dict[tuple[str, str], tuple[int, str, str]] = {}
    for p in manifest.get("perturbations", []):
        key = (p["relation_hint"], _normalize_value(p["original"]))
        perturb_by_pair[key] = (
            int(p["label"]),
            p["label_name"],
            _normalize_value(p["replacement"]),
        )

    amended_triples: set[tuple[str, str, str]] = {
        (h, r, _normalize_value(t))
        for h, r, t in amended_store.doc_graph.triples
    }
    amended_entities: set[str] = set(amended_store.adapter.entity_to_idx)

    # Per-(h, r) index of amended tails, for has_replacement lookup.
    amended_hr_tails: dict[tuple[str, str], set[str]] = {}
    for h, r, t in amended_store.doc_graph.triples:
        amended_hr_tails.setdefault((h, r), set()).add(_normalize_value(t))

    samples: list[CrossStoreSample] = []
    for h, r, t in base_store.doc_graph.triples:
        t_norm = _normalize_value(t)
        base_key = (h, r, t_norm)

        # --- label
        label, label_name, provenance = 0, "UNCHANGED", "unchanged"
        pair_key = (r, t_norm)
        if pair_key in perturb_by_pair:
            lbl, lname, expected_repl = perturb_by_pair[pair_key]
            repl_key = (h, r, expected_repl)
            # Confirm: amended has the replacement and no longer has
            # the original triple.
            if repl_key in amended_triples and base_key not in amended_triples:
                label, label_name = lbl, lname
                provenance = f"manifest:{lname}"

        # --- features
        tail_missing = 0 if t_norm in amended_entities else 1
        other_tails = amended_hr_tails.get((h, r), set())
        has_replacement = 1 if any(tt != t_norm for tt in other_tails) else 0

        s_base = base_store.score_triple(h, r, t_norm)
        s_amend = None
        if tail_missing == 0 and s_base is not None:
            s_amend = amended_store.score_triple(h, r, t_norm)

        if s_base is None:
            # Base itself can't score -- skip; no usable signal.
            continue
        delta = (abs(float(s_base) - float(s_amend))
                 if s_amend is not None else 0.0)

        samples.append(CrossStoreSample(
            head=h, relation=r, tail=t_norm,
            score_base=float(s_base),
            score_amended=float(s_amend) if s_amend is not None else None,
            delta=float(delta),
            tail_missing=int(tail_missing),
            has_replacement=int(has_replacement),
            label=int(label),
            label_name=label_name,
            provenance=provenance,
        ))
    return samples


def _normalize_value(value: str) -> str:
    """Normalize a tail value so manifest keys match store triple tails.

    Handles the two inconsistencies between hand-written manifest values
    and the ingested triple store:

    - whitespace and a trailing percent sign
    - numeric-precision drift (``"82.50"`` vs ``"82.5"``) -- we strip
      trailing zeros on the fractional part and the trailing ``.`` when
      the value parses as a number.
    """
    s = str(value).strip()
    if not s:
        return s
    # Try parsing as number (with optional trailing %)
    pct = s.endswith("%")
    body = s[:-1] if pct else s
    try:
        f = float(body)
    except ValueError:
        return s
    # Normalize numeric: preserve trailing % on integers, strip zeros.
    if "." in body:
        normalized = f"{f:.10f}".rstrip("0").rstrip(".")
        if normalized == "" or normalized == "-":
            normalized = "0"
    else:
        normalized = str(int(f)) if f == int(f) else str(f)
    return normalized + ("%" if pct else "")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    classifier: object                           # CalibratedClassifierCV
    raw_classifier: LogisticRegression
    feature_names: tuple[str, ...]
    projection_thresholds: dict                  # informational only
    metrics: dict
    sample_counts: dict
    splits: dict


def _project_delta_thresholds(
    classifier,
    *,
    tail_missing: int = 0,
    has_replacement: int = 1,
    delta_max: float = 0.8,
    n_pts: int = 401,
) -> dict:
    """Scan the delta axis at a fixed (tail_missing, has_replacement) cell
    and record where the classifier's argmax changes class.

    Returned thresholds are *informational*: they are the delta values
    at which class predictions cross in the most-common non-UNCHANGED
    cell of the feature grid.  Inference uses the full 3-D classifier;
    these scalars exist so operators can quickly sanity-check the
    classifier's decision surface and so pedagogical notebooks can show
    a 1-D sensitivity curve.

    Because MODIFIED delta exceeds CONTRADICTED delta when the tail
    survives in amended, the 1-D ordering at (tm=0, hr=1) is typically
    UNCHANGED < CONTRADICTED < MODIFIED -- not the legacy scalar
    assumption U < M < C.  Callers that interpret these values for
    classification must therefore consult ``crossings`` rather than
    assuming a fixed class order.
    """
    deltas = np.linspace(0.0, float(delta_max), int(n_pts))
    X = np.column_stack([
        deltas,
        np.full(n_pts, tail_missing, dtype=float),
        np.full(n_pts, has_replacement, dtype=float),
    ])
    preds = np.asarray(classifier.predict(X), dtype=int)
    crossings = []
    for i in range(1, n_pts):
        if preds[i] != preds[i - 1]:
            crossings.append({
                "delta": float(deltas[i]),
                "from_class": CLASS_NAMES[int(preds[i - 1])],
                "to_class": CLASS_NAMES[int(preds[i])],
            })
    # tau_minor_upper: smallest delta where prediction leaves UNCHANGED.
    #   - If the scan starts outside UNCHANGED, use 0.0 (the cell has no
    #     UNCHANGED region).
    #   - If the scan is UNCHANGED throughout, use delta_max (the cell
    #     never leaves UNCHANGED; the scalar projection cannot classify
    #     in this cell and operators should consult the 3-D model).
    u_idx = int(CLASS_NAMES.index("UNCHANGED"))
    c_idx = int(CLASS_NAMES.index("CONTRADICTED"))
    non_u = np.flatnonzero(preds != u_idx)
    tau_minor_upper = (
        float(deltas[int(non_u[0])]) if non_u.size > 0 else float(delta_max)
    )
    # tau_contra_lower: smallest delta where prediction first enters
    # CONTRADICTED.  Falls back to delta_max when CONTRADICTED is never
    # reached within the scan window.
    contra = np.flatnonzero(preds == c_idx)
    tau_contra_lower = (
        float(deltas[int(contra[0])]) if contra.size > 0 else float(delta_max)
    )
    # Keep the ordering tau_minor_upper <= tau_contra_lower even in
    # degenerate projections (monotonicity is a reader expectation).
    if tau_contra_lower < tau_minor_upper:
        tau_contra_lower = tau_minor_upper
    return {
        "projection_cell": {
            "tail_missing": int(tail_missing),
            "has_replacement": int(has_replacement),
        },
        "tau_minor_upper": float(tau_minor_upper),
        "tau_contra_lower": float(tau_contra_lower),
        "crossings": crossings,
    }


def calibrate_cross_store_tension(
    samples: Sequence[CrossStoreSample],
    *,
    seed: int = 42,
    test_size: float = 0.20,
    calibration_size: float = 0.25,        # fraction of non-test for isotonic fit
    max_unchanged_multiplier: int = 2,     # balance class sizes
    calibration_method: str = "isotonic",
) -> CalibrationResult:
    """Fit a probability-calibrated 3-class cross-store classifier.

    Input is the 3-feature vector ``(delta, tail_missing, has_replacement)``
    produced by :func:`label_cross_store_tension`.

    Pipeline:
      1. Balance: cap UNCHANGED to ``multiplier * max(other class counts)``.
      2. Stratified split: train / calibration / test (e.g. 60/20/20).
      3. Fit raw multinomial LR on train.
      4. Fit isotonic CalibratedClassifierCV(prefit) on calibration split.
      5. Report confusion / P/R / ECE / Brier on held-out test split.

    Inference uses ``classifier.predict_proba`` on the 3-D vector directly;
    the 1-D monotonic assumption of scalar tau thresholds does not hold
    (MODIFIED's delta is larger than CONTRADICTED's when the tail survives
    in amended), so no tau cuts are extracted.
    """
    if not samples:
        raise ValueError("no samples to calibrate")

    X_all = np.array([s.features() for s in samples], dtype=float)
    y_all = np.array([s.label for s in samples], dtype=int)

    # Class balancing (stratified subsample of UNCHANGED)
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_all, minlength=3)
    max_minor = max(int(counts[1]), int(counts[2]))
    keep_mask = np.ones_like(y_all, dtype=bool)
    if counts[0] > max_unchanged_multiplier * max_minor and max_minor > 0:
        idx0 = np.where(y_all == 0)[0]
        chosen = rng.choice(idx0, size=max_unchanged_multiplier * max_minor, replace=False)
        keep_mask = np.zeros_like(y_all, dtype=bool)
        keep_mask[chosen] = True
        keep_mask[y_all != 0] = True
    X = X_all[keep_mask]
    y = y_all[keep_mask]

    # Stratified train_test split: first carve off test
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    # Then carve calibration off the remainder
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_rest, y_rest, test_size=calibration_size,
        random_state=seed, stratify=y_rest,
    )

    # Raw classifier.  sklearn 1.5+ defaults to multinomial; we omit the
    # deprecated ``multi_class`` kwarg so future sklearn versions stay quiet.
    raw = LogisticRegression(
        max_iter=2000, random_state=seed,
        class_weight="balanced",
    ).fit(X_train, y_train)

    # Probability calibration on frozen classifier
    calibrator = CalibratedClassifierCV(
        estimator=raw, method=calibration_method, cv="prefit",
    ).fit(X_cal, y_cal)

    # Informational 1-D projection for notebooks / auditors
    projection_thresholds = _project_delta_thresholds(calibrator)

    # Metrics on held-out test split
    metrics = _compute_metrics(calibrator, X_test, y_test)
    metrics["test_size"] = int(len(y_test))
    metrics["feature_names"] = list(FEATURE_NAMES)
    metrics["raw_lr_coef"] = raw.coef_.tolist()
    metrics["raw_lr_intercept"] = raw.intercept_.tolist()

    sample_counts = {
        "total": int(len(samples)),
        "after_balance": int(len(X)),
        "per_class_raw": {CLASS_NAMES[k]: int(counts[k]) for k in range(3)},
        "per_class_balanced": {
            CLASS_NAMES[k]: int(np.sum(y == k)) for k in range(3)
        },
        "feature_cells": _feature_cell_counts(samples),
    }
    splits = {
        "train": int(len(y_train)),
        "calibration": int(len(y_cal)),
        "test": int(len(y_test)),
        "seed": int(seed),
        "test_size": float(test_size),
        "calibration_size": float(calibration_size),
    }

    return CalibrationResult(
        classifier=calibrator,
        raw_classifier=raw,
        feature_names=FEATURE_NAMES,
        projection_thresholds=projection_thresholds,
        metrics=metrics,
        sample_counts=sample_counts,
        splits=splits,
    )


def _feature_cell_counts(
    samples: Sequence[CrossStoreSample],
) -> dict[str, dict[str, int]]:
    """Count per-class samples in each (tail_missing, has_replacement) cell.

    Useful for auditors to verify the generator's perturbation regime
    actually populates all four cells as expected.
    """
    out: dict[str, dict[str, int]] = {c: {} for c in CLASS_NAMES}
    for s in samples:
        cell = f"tm={s.tail_missing},hr={s.has_replacement}"
        out[s.label_name][cell] = out[s.label_name].get(cell, 0) + 1
    return out


def _compute_metrics(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0,
    )

    # ECE across all classes (one-vs-rest on the predicted probability)
    ece = _expected_calibration_error(y_test, probs)

    # Brier loss per class (one-vs-rest)
    briers = {}
    for cls in (0, 1, 2):
        y_bin = (y_test == cls).astype(int)
        briers[CLASS_NAMES[cls]] = float(brier_score_loss(y_bin, probs[:, cls]))

    # Reliability bins on the top-class probability
    bins = _reliability_bins(y_test, probs, n_bins=10)

    return {
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(CLASS_NAMES),
        "per_class": {
            CLASS_NAMES[k]: {
                "precision": float(prec[k]),
                "recall": float(rec[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            } for k in range(3)
        },
        "expected_calibration_error": float(ece),
        "brier_score_ovr": briers,
        "reliability_bins": bins,
    }


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                                n_bins: int = 10) -> float:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _reliability_bins(y_true: np.ndarray, probs: np.ndarray,
                      n_bins: int = 10) -> list[dict]:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            out.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "count": 0, "mean_confidence": None, "empirical_accuracy": None,
            })
            continue
        out.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()),
            "empirical_accuracy": float(correct[mask].mean()),
        })
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_cross_store_calibration(
    result: CalibrationResult,
    target_dir: Path,
    *,
    source: dict | None = None,
) -> dict[str, Path]:
    """Write the three calibration artifacts next to the existing Ch 4 ones."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = target_dir / "cross_store_calibration.pkl"
    json_path = target_dir / "cross_store_calibration.json"
    metrics_path = target_dir / "cross_store_calibration_metrics.json"

    with pkl_path.open("wb") as f:
        pickle.dump(result.classifier, f)

    header = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(result.feature_names),
        "classes": list(CLASS_NAMES),
        "calibration_method": "isotonic",
        "classifier_type": "CalibratedClassifierCV(LogisticRegression, cv=prefit)",
        "projection_thresholds": result.projection_thresholds,
        "sample_counts": result.sample_counts,
        "splits": result.splits,
    }
    if source is not None:
        header["source"] = source
    json_path.write_text(json.dumps(header, indent=2))

    metrics_path.write_text(json.dumps(result.metrics, indent=2))

    return {"classifier": pkl_path, "header": json_path, "metrics": metrics_path}


def load_cross_store_calibration(target_dir: Path) -> dict:
    """Load the three artifacts; returns {"classifier", "header", "metrics"}.

    Raises FileNotFoundError if the calibration hasn't been trained yet.
    Callers that need the 3-D feature schema can read
    ``result["header"]["feature_names"]``.
    """
    target_dir = Path(target_dir)
    pkl_path = target_dir / "cross_store_calibration.pkl"
    json_path = target_dir / "cross_store_calibration.json"
    metrics_path = target_dir / "cross_store_calibration_metrics.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"no cross_store_calibration.json in {target_dir}")

    header = json.loads(json_path.read_text())
    clf = None
    if pkl_path.is_file():
        with pkl_path.open("rb") as f:
            clf = pickle.load(f)
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    return {"classifier": clf, "header": header, "metrics": metrics}
