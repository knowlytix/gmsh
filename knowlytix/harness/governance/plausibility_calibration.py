# SPDX-License-Identifier: Apache-2.0
"""Bootstrap-calibrated plausibility threshold for the geodesic measure.

The single-fit ``GMSJudge.calibrate()`` produces one ``tau_plausibility``
value; in high-stakes deployment that hides which side of the threshold
borderline cases were on at the time of the fit.  This module bootstraps
the geodesic logistic regression: K resamples of the (real, corrupted-tail)
training pairs, refit the threshold on each, return the bootstrap
distribution as a :class:`ThresholdDistribution` (mean, std, 95% CI).

The runtime claim verifier consumes the mean (the operating threshold);
the auditor reads the CI when a verdict lands near the boundary.

Fact-claim attestor:

  * **Geodesic** — the existing learned signal.
  * **Graph presence** — does the (head, relation, tail) match an
    existing triple in the store's ``doc_graph``? Pure structural
    lookup with fuzzy entity matching, deterministic, no embeddings.

A claim accepted by both attestors PASSes; a single-attestor pass routes
to ``review`` (or fails, depending on the policy chosen at gate time).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from knowlytix.harness.governance.governance_bundle import ThresholdDistribution

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


def _derive_threshold(clf, X: np.ndarray) -> float | None:
    """Scan the fitted binary classifier to find the τ where
    ``P(class=1 | value=τ) = 0.5``. Mirrors ``GMSJudge._derive_thresholds``
    so the bootstrap distribution agrees with the single-fit point."""
    if X.size == 0:
        return None
    lo, hi = float(np.min(X)), float(np.max(X))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 801).reshape(-1, 1)
    pred = clf.predict(grid).astype(int)
    changes = np.where(np.diff(pred) != 0)[0]
    if len(changes) == 0:
        return None
    return float(grid[changes[0], 0])


def _build_geodesic_training_set(
    store: "GMSExpertStore", *, seed: int = 42,
):
    """Build (real, corrupted-tail) pairs and their geodesic scores.

    Mirrors the ``geodesic`` block of ``GMSJudge.calibrate()`` so the
    bootstrap sees the same population the original calibration did.
    """
    rng = random.Random(seed)
    triples = list(store.doc_graph.triples)
    entities = list(store.adapter.entity_to_idx.keys())
    if len(entities) < 2:
        return np.empty((0, 1)), np.empty((0,), dtype=int)

    X: list[float] = []
    y: list[int] = []
    for h, r, t in triples:
        geo = store.score_triple(h, r, t)
        if geo is None:
            continue
        X.append(float(geo))
        y.append(1)
        fake_t = rng.choice(entities)
        attempts = 0
        while fake_t == t and attempts < 10:
            fake_t = rng.choice(entities)
            attempts += 1
        if fake_t == t:
            continue
        geo_neg = store.score_triple(h, r, fake_t)
        if geo_neg is None:
            X.append(float(math.pi))
        else:
            X.append(float(geo_neg))
        y.append(0)

    return np.asarray(X, dtype=float).reshape(-1, 1), np.asarray(y, dtype=int)


def bootstrap_plausibility_threshold(
    store: "GMSExpertStore",
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> ThresholdDistribution:
    """Refit the geodesic threshold on K bootstrap resamples.

    Returns the bootstrap distribution of ``tau_plausibility``. Pass the
    ``mean`` to the runtime verifier; expose ``ci_low``/``ci_hi`` in the
    audit report.

    Raises ``RuntimeError`` if too few resamples produce a valid fit
    (the store likely is too thin for stable calibration).
    """
    from sklearn.linear_model import LogisticRegression

    X, y = _build_geodesic_training_set(store, seed=seed)
    n = len(y)
    if n < 30:
        raise RuntimeError(
            f"Geodesic training set has only {n} samples; need at "
            f"least 30 for bootstrap calibration. Train on a richer "
            f"corpus or increase the corpus's triple count."
        )
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Geodesic training set is single-class; need both real "
            "and corrupted samples."
        )

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xs, ys = X[idx], y[idx]
        if len(np.unique(ys)) < 2:
            continue
        clf = LogisticRegression(random_state=seed + k, max_iter=2000)
        try:
            clf.fit(Xs, ys)
        except Exception:
            continue
        tau = _derive_threshold(clf, Xs)
        if tau is None or not math.isfinite(tau):
            continue
        draws.append(tau)

    if len(draws) < max(10, n_bootstrap // 4):
        raise RuntimeError(
            f"Bootstrap calibration produced too few valid fits: "
            f"{len(draws)}/{n_bootstrap}. The geodesic separation is "
            f"unstable on this store."
        )

    arr = np.asarray(draws)
    return ThresholdDistribution(
        mean=float(arr.mean()),
        std=float(arr.std()),
        ci_low=float(np.quantile(arr, 0.025)),
        ci_hi=float(np.quantile(arr, 0.975)),
        n_bootstrap=len(draws),
    )


# ---------------------------------------------------------------------------
# Two-attestor structural check
# ---------------------------------------------------------------------------


@dataclass
class GraphPresenceResult:
    """Outcome of the structural attestor on a fact claim."""

    present: bool
    matched_head: str | None
    matched_tail: str | None
    matched_triple: tuple[str, str, str] | None
    reason: str

    @property
    def signal(self) -> str:
        return "agreement" if self.present else "abstain"


def check_graph_presence(
    store: "GMSExpertStore",
    head: str, relation: str, tail: str,
) -> GraphPresenceResult:
    """Independent structural attestor: does the corpus already assert
    this triple?

    Pure graph lookup with fuzzy entity matching, no embeddings.
    Returns ``present=True`` only when ``(matched_head, relation,
    matched_tail)`` exists in the store's document graph.
    """
    if store.doc_graph is None:
        return GraphPresenceResult(
            present=False, matched_head=None, matched_tail=None,
            matched_triple=None,
            reason="store has no doc_graph",
        )

    h_match = store.fuzzy_match_entity(head)
    t_match = store.fuzzy_match_entity(tail)
    if h_match is None or t_match is None:
        return GraphPresenceResult(
            present=False, matched_head=h_match, matched_tail=t_match,
            matched_triple=None,
            reason=(
                f"entity not in store: head={h_match}, tail={t_match}"
            ),
        )

    triples = list(store.doc_graph.triples)
    for h, r, t in triples:
        if h == h_match and r == relation and t == t_match:
            return GraphPresenceResult(
                present=True, matched_head=h_match, matched_tail=t_match,
                matched_triple=(h, r, t),
                reason=f"corpus asserts ({h}, {r}, {t})",
            )

    return GraphPresenceResult(
        present=False, matched_head=h_match, matched_tail=t_match,
        matched_triple=None,
        reason=f"({h_match}, {relation}, {t_match}) not in corpus",
    )


# ---------------------------------------------------------------------------
# Two-attestor verifier wrapper
# ---------------------------------------------------------------------------


@dataclass
class TwoAttestorVerdict:
    """Combined verdict from geodesic + graph-presence attestors."""

    passed: bool
    geodesic_pass: bool
    presence_pass: bool
    geodesic: float | None
    threshold: float
    threshold_ci: tuple[float, float]
    presence_reason: str
    decision: str  # "PASS" | "REVIEW" | "FAIL"
    evidence: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "geodesic_pass": self.geodesic_pass,
            "presence_pass": self.presence_pass,
            "geodesic": self.geodesic,
            "threshold": self.threshold,
            "threshold_ci": list(self.threshold_ci),
            "presence_reason": self.presence_reason,
            "decision": self.decision,
            "evidence": self.evidence,
        }


def verify_fact_two_attestor(
    store: "GMSExpertStore",
    head: str, relation: str, tail: str,
    threshold: ThresholdDistribution,
    *,
    require_concurrence: bool = True,
) -> TwoAttestorVerdict:
    """Verify a fact claim with geodesic + graph-presence concurrence.

    Decision matrix:

      geodesic PASS, presence PASS  -> PASS
      geodesic PASS, presence FAIL  -> REVIEW (single attestor)
      geodesic FAIL, presence PASS  -> REVIEW (geometry disagrees with
                                       the structural ground truth)
      geodesic FAIL, presence FAIL  -> FAIL

    Set ``require_concurrence=False`` to fall back to single-attestor
    geodesic, useful for backwards compatibility with the original
    Ch 10 verifier.
    """
    h_match = store.fuzzy_match_entity(head)
    t_match = store.fuzzy_match_entity(tail)
    if h_match is None or t_match is None:
        return TwoAttestorVerdict(
            passed=False, geodesic_pass=False, presence_pass=False,
            geodesic=None,
            threshold=threshold.mean,
            threshold_ci=(threshold.ci_low, threshold.ci_hi),
            presence_reason=(
                f"entity not in store: head={h_match}, tail={t_match}"
            ),
            decision="FAIL",
            evidence=(
                f"score_triple unavailable (head_match={h_match}, "
                f"tail_match={t_match})"
            ),
        )

    geo = store.score_triple(h_match, relation, t_match)
    geodesic_pass = geo is not None and geo < threshold.mean

    presence = check_graph_presence(store, head, relation, tail)
    presence_pass = presence.present

    if not require_concurrence:
        decision = "PASS" if geodesic_pass else "FAIL"
        return TwoAttestorVerdict(
            passed=geodesic_pass,
            geodesic_pass=geodesic_pass,
            presence_pass=presence_pass,
            geodesic=geo,
            threshold=threshold.mean,
            threshold_ci=(threshold.ci_low, threshold.ci_hi),
            presence_reason=presence.reason,
            decision=decision,
            evidence=(
                f"score_triple({h_match}, {relation}, {t_match}) "
                f"= {geo:.4f} {'<' if geodesic_pass else '>='} "
                f"{threshold.mean:.4f} (95% CI [{threshold.ci_low:.4f}, "
                f"{threshold.ci_hi:.4f}], single-attestor)"
            ) if geo is not None else "geodesic unavailable",
        )

    if geodesic_pass and presence_pass:
        decision = "PASS"
    elif (not geodesic_pass) and (not presence_pass):
        decision = "FAIL"
    else:
        decision = "REVIEW"

    geo_str = f"{geo:.4f}" if geo is not None else "N/A"
    cmp = "<" if geodesic_pass else ">=" if geo is not None else "??"
    evidence = (
        f"geo={geo_str} {cmp} tau={threshold.mean:.4f} "
        f"(95% CI [{threshold.ci_low:.4f}, {threshold.ci_hi:.4f}]); "
        f"presence: {presence.reason}"
    )
    return TwoAttestorVerdict(
        passed=(decision == "PASS"),
        geodesic_pass=geodesic_pass,
        presence_pass=presence_pass,
        geodesic=geo,
        threshold=threshold.mean,
        threshold_ci=(threshold.ci_low, threshold.ci_hi),
        presence_reason=presence.reason,
        decision=decision,
        evidence=evidence,
    )
