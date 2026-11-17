# SPDX-License-Identifier: Apache-2.0
"""Offline validation pipeline that produces a ``GovernanceBundle``.

Heavy work runs once, here. The runtime gate consumes the bundle and
never recalibrates.

Pipeline:
  1. Pull the labeled tension cohort once via ``label_tension``.
  2. Bootstrap K resamples → distribution of (tau_ent, tau_contra).
  3. Fit a single isotonic regression ``tension_energy → P(contra)``
     on the full cohort.
  4. Build per-entity support index from store triples.
  5. Read declared opposition pairs from ``(r_a, opposite_of, r_b)``
     triples (mirrors ``label_tension``).
  6. Record drift baseline (tension-energy distribution on the cohort).
  7. Sign and return the bundle.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from knowlytix.harness.governance.governance_bundle import (
    BUNDLE_SCHEMA_VERSION,
    EntitySupport,
    GovernanceBundle,
    OperatingPoint,
    ThresholdDistribution,
    make_bundle_id,
    utcnow_iso,
)
from knowlytix.harness.governance.store_fingerprint import compute_store_fingerprint
from knowlytix.harness.governance.support_calibration import (
    calibrate_support_thresholds,
)

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


_OPPOSITE_OF_RELATION = "opposite_of"


def bootstrap_thresholds(
    samples: list,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[ThresholdDistribution, ThresholdDistribution]:
    """Resample the labeled cohort K times and refit the 3-class fit.

    Returns ``(tau_ent_dist, tau_contra_dist)``.
    """
    from knowlytix.harness.testing.threshold_calibration import calibrate_logic_tension

    rng = np.random.default_rng(seed)
    n = len(samples)
    if n < 30:
        raise ValueError(
            f"Need at least 30 labeled samples to bootstrap; got {n}. "
            f"Train the store with more triples or lower min_agree_support."
        )

    ent_draws: list[float] = []
    con_draws: list[float] = []
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resample = [samples[int(i)] for i in idx]
        clf, cuts = calibrate_logic_tension(resample, seed=seed + k)
        if cuts is None:
            continue
        ent_draws.append(float(cuts.tau_ent))
        con_draws.append(float(cuts.tau_contra))

    if len(ent_draws) < max(10, n_bootstrap // 4):
        raise RuntimeError(
            f"Bootstrap calibration produced too few valid fits: "
            f"{len(ent_draws)}/{n_bootstrap}. The store likely failed "
            f"the convergence gate (per-class means too close)."
        )

    def _summarize(values: list[float]) -> ThresholdDistribution:
        a = np.asarray(values)
        return ThresholdDistribution(
            mean=float(a.mean()),
            std=float(a.std()),
            ci_low=float(np.quantile(a, 0.025)),
            ci_hi=float(np.quantile(a, 0.975)),
            n_bootstrap=len(values),
        )

    return _summarize(ent_draws), _summarize(con_draws)


def fit_isotonic(samples: list) -> tuple[np.ndarray, np.ndarray]:
    """Fit isotonic regression ``tension_energy → P(contradict)``.

    Treat label==2 as positive (contradict), label in {0,1} as negative.
    Returns ``(x_grid, y_grid)`` arrays so the bundle can rebuild the
    estimator on load without persisting an sklearn object.
    """
    from sklearn.isotonic import IsotonicRegression

    X = np.array([s.value for s in samples], dtype=float)
    y = np.array([1 if s.label == 2 else 0 for s in samples], dtype=float)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(X, y)

    grid = np.linspace(X.min(), X.max(), 200)
    y_grid = iso.predict(grid)
    return grid, y_grid


def _build_support_index(
    store: "GMSExpertStore",
) -> dict[str, EntitySupport]:
    """Count incident triples and distinct relations per entity."""
    triples = list(store.doc_graph.triples)
    incident: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    relations: dict[str, set[str]] = defaultdict(set)
    for h, r, t in triples:
        incident[h].add((h, r, t))
        incident[t].add((h, r, t))
        relations[h].add(r)
        relations[t].add(r)
    return {
        e: EntitySupport(
            triple_count=len(incident[e]),
            relation_count=len(relations[e]),
        )
        for e in incident
    }


def _build_opposition_table(
    store: "GMSExpertStore",
) -> dict[tuple[str, str], str]:
    """Extract declared (r_a, opposite_of, r_b) triples → table.

    The runtime shortcut needs *entity* pairs that are in declared
    contradiction. We follow ``label_tension``: a contradiction holds
    between two entity heads when they share a tail under opposing
    relations. We materialize this as an explicit table at bundle
    build time.
    """
    triples = list(store.doc_graph.triples)
    relations = set(store.adapter.relation_to_idx.keys())
    schema = {_OPPOSITE_OF_RELATION, "is_functional"}

    opposing_relations: set[tuple[str, str]] = set()
    for h, r, t in triples:
        if r != _OPPOSITE_OF_RELATION:
            continue
        if h in relations and t in relations and h != t:
            opposing_relations.add((h, t))
            opposing_relations.add((t, h))

    target_rel_heads: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for h, r, t in triples:
        if r in schema:
            continue
        target_rel_heads[t][r].add(h)

    table: dict[tuple[str, str], str] = {}
    for pos, neg in opposing_relations:
        for tail, rel_map in target_rel_heads.items():
            for a in rel_map.get(pos, set()):
                for b in rel_map.get(neg, set()):
                    if a == b:
                        continue
                    reason = f"declared:{pos}/opposite_of/{neg}@{tail}"
                    table[(a, b)] = reason
                    table[(b, a)] = reason
    return table


def build_numeric_range_index(
    store: "GMSExpertStore",
    *,
    pad: float = 0.0,
) -> dict[str, tuple[float, float, int]]:
    """Compute (min, max, n_obs) per ENM category from corpus values.

    Used by ``ClaimVerifier`` as a second attestor: a numeric claim
    whose value matches ENM but lies outside the calibration range
    routes to ``review`` rather than ``PASS``.  ``pad`` widens the
    range by a fraction of its span (e.g. ``pad=0.05`` allows 5%
    expansion beyond observed values) for robustness against
    sampling noise; default 0 is strict.
    """
    if not getattr(store, "enm", None):
        return {}
    by_cat: dict[str, list[float]] = {}
    for key in store.enm.keys():
        val = store.enm.read_exact(key)
        if val is None:
            continue
        v = float(val.item() if hasattr(val, "item") else val)
        by_cat.setdefault(key.type, []).append(v)
    out: dict[str, tuple[float, float, int]] = {}
    for cat, values in by_cat.items():
        if not values:
            continue
        lo, hi = min(values), max(values)
        if pad > 0:
            span = hi - lo
            lo -= span * pad
            hi += span * pad
        out[cat] = (float(lo), float(hi), len(values))
    return out


def _validation_metrics(samples: list) -> dict[str, float]:
    """Per-class counts and basic distribution stats."""
    n = len(samples)
    by_label = defaultdict(int)
    for s in samples:
        by_label[s.label] += 1
    values = np.array([s.value for s in samples])
    return {
        "n_samples": float(n),
        "n_agree": float(by_label[0]),
        "n_unrelated": float(by_label[1]),
        "n_contradict": float(by_label[2]),
        "mean_tension": float(values.mean()),
        "std_tension": float(values.std()),
    }


def build_bundle(
    store: "GMSExpertStore",
    *,
    source_store_path: str,
    n_bootstrap: int = 200,
    seed: int = 42,
    operating_point: OperatingPoint | None = None,
) -> GovernanceBundle:
    """Assemble and sign a GovernanceBundle from a trained store.

    Tension calibration is attempted but gracefully skipped when the
    corpus declares no oppositions (the strict 3-class fit refuses
    rather than producing a degenerate threshold).  Plausibility
    calibration is always attempted: it only requires the
    real-vs-corrupted-tail signal already produced by training.

    A bundle without tension thresholds remains useful for the
    claim-verification surface (Ch 10's orchestrator).  A bundle
    without plausibility thresholds is fail-closed downstream and
    raises here.
    """
    from knowlytix.harness.governance.plausibility_calibration import (
        bootstrap_plausibility_threshold,
    )
    from knowlytix.harness.testing.primitive_calibration import label_tension

    # ---- Plausibility calibration (always required) -------------- #
    try:
        tau_plausibility = bootstrap_plausibility_threshold(
            store, n_bootstrap=n_bootstrap, seed=seed,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Plausibility calibration failed: {exc}. The store has "
            f"insufficient (real, corrupted-tail) signal to derive a "
            f"geodesic threshold."
        ) from exc

    # ---- Tension calibration (optional, skipped if no oppositions)
    samples = label_tension(store, seed=seed)
    has_contradict = any(s.label == 2 for s in samples)
    has_agree = any(s.label == 0 for s in samples)
    tau_ent: ThresholdDistribution | None = None
    tau_contra: ThresholdDistribution | None = None
    iso_x = iso_y = None
    if samples and has_contradict and has_agree:
        try:
            tau_ent, tau_contra = bootstrap_thresholds(
                samples, n_bootstrap=n_bootstrap, seed=seed,
            )
            iso_x, iso_y = fit_isotonic(samples)
        except (RuntimeError, ValueError):
            tau_ent = tau_contra = None
            iso_x = iso_y = None

    support = _build_support_index(store)
    opposition = _build_opposition_table(store)

    # ---- Drift baseline ----------------------------------------- #
    # Use tension energies when available; otherwise fall back to
    # the plausibility (geodesic) distribution as the baseline so
    # the drift monitor has something to compare runtime telemetry to.
    if samples:
        baseline_values = np.array([s.value for s in samples])
    else:
        from knowlytix.harness.governance.plausibility_calibration import (
            _build_geodesic_training_set,
        )
        X, _ = _build_geodesic_training_set(store, seed=seed)
        baseline_values = X.ravel()
    te_quantiles = tuple(
        float(np.quantile(baseline_values, q))
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    )

    numeric_range_index = build_numeric_range_index(store)
    metrics = _validation_metrics(samples) if samples else {
        "n_samples": 0.0,
        "n_agree": 0.0, "n_unrelated": 0.0, "n_contradict": 0.0,
        "mean_baseline": float(baseline_values.mean()) if baseline_values.size else 0.0,
        "std_baseline": float(baseline_values.std()) if baseline_values.size else 0.0,
    }
    fingerprint = compute_store_fingerprint(store)

    # Calibrated support thresholds. When tension cuts are absent, we
    # call the calibrator with the plausibility threshold so the
    # support floor is selected against the same labeled cohort.
    if tau_ent is not None and tau_contra is not None:
        support_thresholds = calibrate_support_thresholds(
            store, support, tau_ent.mean, tau_contra.mean, seed=seed,
        )
    else:
        support_thresholds = None

    created = utcnow_iso()
    bundle = GovernanceBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        bundle_id=make_bundle_id(source_store_path, created),
        created_at=created,
        source_store_path=source_store_path,
        tau_ent=tau_ent,
        tau_contra=tau_contra,
        isotonic_x=iso_x,
        isotonic_y=iso_y,
        support_index=support,
        opposition_table=opposition,
        drift_baseline_te=baseline_values,
        drift_baseline_te_quantiles=te_quantiles,
        validation_metrics=metrics,
        tau_plausibility=tau_plausibility,
        store_fingerprint=fingerprint,
        support_thresholds=support_thresholds,
        operating_point=operating_point or OperatingPoint(),
        numeric_range_index=numeric_range_index,
    )
    bundle.freeze()
    return bundle
