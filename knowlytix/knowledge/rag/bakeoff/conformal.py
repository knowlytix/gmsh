# SPDX-License-Identifier: Apache-2.0
"""Conformal prediction-set abstention (bake-off invariant I5).

A binder emits a confidence over candidate bindings, but a raw confidence
threshold gives no coverage guarantee: the operating point that looks calibrated
on synthetic data need not cover the truth on real language. Split conformal
prediction turns the scores into a set with a distribution-free marginal coverage
guarantee. Given a calibration set drawn from the target distribution (here the
Claude-materialized real-language holdout, disjoint from training), it fits a
single nonconformity quantile ``tau`` such that, on exchangeable future queries,
the true binding lies in the prediction set with probability at least ``1 - alpha``.

The set drives abstention: a **singleton** set is a committed binding, so the
pipeline proceeds; an **empty or multi-element** set means the binder cannot
single out one binding at the ratified coverage, so the pipeline abstains (I5).
This is the wrapper every arm carries, so arms are compared at a common coverage
level rather than at each one's private threshold.

Nonconformity is ``1 - confidence``: a candidate is admitted to the set when its
confidence is at least ``1 - tau``. The true label of a relation-absent query is
the reserved key :data:`ABSTAIN_LABEL`; a binder that scores abstention is covered
by the same guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ABSTAIN_LABEL = "<abstain>"


def _conformal_quantile(nonconf: list[float], alpha: float) -> float:
    """Split-conformal quantile: the ``ceil((n+1)(1-alpha))``-th smallest
    nonconformity score. Returns 1.0 (admit everything, the conservative default)
    when the level exceeds the sample, or when there is no calibration data."""
    n = len(nonconf)
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return 1.0
    k = max(k, 1)
    return sorted(nonconf)[k - 1]


@dataclass
class ConformalGate:
    """Split-conformal prediction-set gate over candidate binding confidences.

    Parameters
    ----------
    alpha:
        Miscoverage level; target marginal coverage is ``1 - alpha``.
    tau:
        Fitted nonconformity quantile. ``None`` until :meth:`calibrate` runs; a
        gate with no fitted ``tau`` recuses (admits every candidate) rather than
        silently picking a threshold.
    """

    alpha: float = 0.1
    tau: float | None = None

    def calibrate(self, records: list[tuple[dict, str]]) -> float:
        """Fit ``tau`` on ``records`` = ``(scores, true_label)`` pairs, where
        ``scores`` maps a candidate label to a confidence in ``[0, 1]`` and
        ``true_label`` is the correct binding (or :data:`ABSTAIN_LABEL` for a
        relation-absent query). Missing/true-absent candidates score 0.
        Returns the fitted ``tau``."""
        nonconf = [1.0 - float(scores.get(true, 0.0)) for scores, true in records]
        self.tau = _conformal_quantile(nonconf, self.alpha)
        return self.tau

    def prediction_set(self, scores: dict) -> set:
        """Candidates whose confidence is at least ``1 - tau`` (nonconformity
        ``<= tau``). With no fitted ``tau`` the gate recuses and returns every
        candidate."""
        tau = 1.0 if self.tau is None else self.tau
        return {k for k, c in scores.items() if (1.0 - float(c)) <= tau + 1e-12}

    def decide(self, scores: dict) -> tuple[str, object]:
        """Return ``("proceed", label)`` when the prediction set is a singleton,
        else ``("abstain", prediction_set)``. A singleton on :data:`ABSTAIN_LABEL`
        is a committed abstention (proceed to the abstain route)."""
        ps = self.prediction_set(scores)
        if len(ps) == 1:
            return "proceed", next(iter(ps))
        return "abstain", ps

    def empirical_coverage(self, records: list[tuple[dict, str]]) -> dict:
        """Coverage report on a labeled set: the share of items whose true label
        is in the prediction set (``conformal_coverage``, compared to the ``1 -
        alpha`` target), plus the mean prediction-set size and the singleton rate
        (efficiency). Coverage well below target means the calibration set was not
        exchangeable with this one -- the ``synthetic_vs_real`` failure the wrapper
        exists to expose."""
        n = len(records)
        if n == 0:
            return {"target": 1.0 - self.alpha, "coverage": 0.0, "n": 0,
                    "avg_set_size": 0.0, "singleton_rate": 0.0}
        covered = sizes = singl = 0
        for scores, true in records:
            ps = self.prediction_set(scores)
            covered += int(true in ps)
            sizes += len(ps)
            singl += int(len(ps) == 1)
        return {
            "target": round(1.0 - self.alpha, 4),
            "coverage": round(covered / n, 4),
            "avg_set_size": round(sizes / n, 4),
            "singleton_rate": round(singl / n, 4),
            "n": n,
        }
