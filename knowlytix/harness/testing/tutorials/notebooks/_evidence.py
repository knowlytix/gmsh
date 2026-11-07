# SPDX-License-Identifier: Apache-2.0
"""Audit-defensible helpers for the user-guide tutorial notebooks.

These helpers produce evidence that survives a high-stakes review:

  * Wilson and bootstrap confidence intervals on accuracy / pass-rate /
    ECE / MCE — no point estimates without uncertainty.
  * Type-aware parser pipeline that mirrors `DOEGMSBenchmark.run`
    so tutorials are scored on the same text that the production
    benchmark sees.
  * Clean (type-respecting, hidden-positive-filtered) negative
    sampling for hallucination and calibration training cohorts.
  * Deterministic seed including `torch.use_deterministic_algorithms`
    so reruns of the same tutorial produce byte-identical numbers.
  * `tutorial_doe_config` — single source of truth for the audit-grade
    DOE size every tutorial uses, so a single bump propagates.

Import everywhere with::

    from _evidence import (
        seed_all_deterministic, wilson_ci, bootstrap_ci_ece,
        clean_negative_pairs, parse_and_score, tutorial_doe_config,
    )
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Sequence

from _shared import LAB_STORES_PATH

# ---------------------------------------------------------------------------
# 1. Audit-grade tutorial-sized config
# ---------------------------------------------------------------------------

# Single source of truth for tutorial-sized cohorts.  Bumped from
# (n_runs=2, max_per_category=3) so Wilson CIs on accuracy are narrow
# enough for a tier comparison to be meaningful (~±8 pp at p=0.85).
TUTORIAL_N_RUNS: int = 5
TUTORIAL_MAX_PER_CATEGORY: int = 10
TUTORIAL_BOOTSTRAP_B: int = 1000


def tutorial_doe_config(
    *,
    store_name: str = "ch04_ground_truth",
    factor_group: str = "quick_screen",
    enable_typed_verdicts: bool = True,
    enable_tracing: bool = True,
    target_type: str = "llm",
    target_model: str | None = None,
    **overrides: Any,
):
    """Build a `DOEHarnessConfig` with audit-grade defaults.

    Yields ~`TUTORIAL_N_RUNS * TUTORIAL_MAX_PER_CATEGORY * n_generators`
    probes — enough for Wilson 95% CIs of ~±8 pp at p=0.85.  Tracing
    defaults ON so the tutorial leaves a replayable trail.
    """
    from knowlytix.harness.testing import DOEHarnessConfig

    defaults = dict(
        markdown_path=str(LAB_STORES_PATH / store_name / "documents" / "combined.md"),
        stores_dir=str(LAB_STORES_PATH),
        store_name=store_name,
        factor_group=factor_group,
        n_runs=TUTORIAL_N_RUNS,
        max_per_category=TUTORIAL_MAX_PER_CATEGORY,
        cap_enabled=True,
        enable_typed_verdicts=enable_typed_verdicts,
        enable_tracing=enable_tracing,
        target_type=target_type,
        target_model=target_model,
    )
    defaults.update(overrides)
    return DOEHarnessConfig(**defaults)


# ---------------------------------------------------------------------------
# 2. Deterministic seeding (stricter than _shared.seed_all)
# ---------------------------------------------------------------------------

def seed_all_deterministic(seed: int = 42) -> None:
    """Seed every RNG plus pin CUDA / cuDNN determinism.

    Produces byte-identical numbers across reruns on the same hardware.
    Sets CUBLAS workspace per CUDA's reproducibility contract; on CPU
    it is a no-op past the RNG seeding.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Some kernels do not have deterministic implementations;
            # warn_only=True is the audit-friendly stance — we still
            # observe non-determinism when it occurs.
            pass
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 3. Confidence intervals
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score 95% CI for a binomial proportion.

    Returns ``(point, lower, upper)``.  Wilson is the audit-grade
    default because it does not collapse at p=0 or p=1, and it has
    correct coverage at small n where normal-approx CIs blow up.
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    # Two-sided z for given alpha, no scipy dependency
    z = _z_for(1 - alpha / 2)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def _z_for(p: float) -> float:
    """Inverse normal CDF via Acklam's approximation.  No scipy."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def bootstrap_ci_ece(
    confidences: Sequence[float],
    corrects: Sequence[bool],
    *,
    n_bins: int = 10,
    B: int = TUTORIAL_BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile-bootstrap 95% CI on Expected Calibration Error.

    Mirrors the binning in `knowlytix.harness.testing.CalibrationTester` (equal-width
    bins on [0,1]).  Returns ``(ece, ci_low, ci_high)`` over B resamples.
    """
    import numpy as np

    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=bool)
    if len(conf) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = _ece(conf, corr, n_bins)
    rng = np.random.default_rng(seed)
    n = len(conf)
    samples = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        samples[b] = _ece(conf[idx], corr[idx], n_bins)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def _ece(conf, corr, n_bins: int) -> float:
    import numpy as np
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        m = in_bin.sum()
        if m == 0:
            continue
        bin_conf = conf[in_bin].mean()
        bin_acc = corr[in_bin].mean()
        total += (m / n) * abs(bin_conf - bin_acc)
    return float(total)


# ---------------------------------------------------------------------------
# 4. Type-aware parser pipeline
# ---------------------------------------------------------------------------

def parse_and_score(
    raw_answer: str,
    *,
    ground_truth: Any,
    answer_type: str,
):
    """Parse the LLM's raw output through the production parser and
    score it the way `DOEGMSBenchmark.run` does.

    Returns a `ScoreResult` with `correct: bool` plus the parsed value.
    Surfaces the same failure modes (parse miss, type coercion, etc.)
    that the production benchmark would record.
    """
    from knowlytix.benchmark.scorers import PARSERS, score_answer

    parser = PARSERS.get(answer_type, PARSERS["str"])
    parsed = parser(raw_answer)
    s = score_answer(parsed, ground_truth)
    return s, parsed


# ---------------------------------------------------------------------------
# 5. Clean negative sampling
# ---------------------------------------------------------------------------

def clean_negative_pairs(
    store,
    *,
    n: int = 200,
    seed: int = 42,
    type_constrained: bool = True,
) -> list[tuple[str, str, str]]:
    """Generate ``n`` ``(h, r, t_fake)`` corruptions that are guaranteed
    to be false in the store.

    Filters in two ways:
      1. ``(h, r, t_fake)`` is not already a triple (no hidden positives).
      2. If ``type_constrained``, ``t_fake`` is drawn from the same
         observed-tail set for ``r`` as the original tail — i.e. real
         models are confused with real models, not with relation names.

    Banking-relevant because random-tail corruptions trivially produce
    type-violating pairs that any classifier can rule out; the real
    discrimination test is type-consistent confusion.
    """
    rng = random.Random(seed)
    triples_list = list(store.doc_graph.triples)
    triples_set = set(triples_list)

    # Per-relation tail pool for type-constrained sampling (de-duped)
    tail_pool_by_rel: dict[str, list[str]] = {}
    for h, r, t in triples_list:
        s = tail_pool_by_rel.setdefault(r, [])
        if t not in s:
            s.append(t)

    all_entities = list(store.adapter.entity_to_idx.keys())
    out: list[tuple[str, str, str]] = []
    seen_out: set[tuple[str, str, str]] = set()

    # Loop the triples list as many times as needed until we hit n or
    # exhaust the search budget.  Some (h, r) combinations have a tiny
    # tail pool and will simply not yield a fresh corruption; that is
    # fine — we move on rather than fabricate one.
    max_passes = 20
    for _pass in range(max_passes):
        if len(out) >= n:
            break
        for h, r, t in triples_list:
            if len(out) >= n:
                break
            pool = tail_pool_by_rel.get(r, all_entities) if type_constrained else all_entities
            if not pool:
                continue
            # Up to 8 attempts per (h, r, t); cheap and stops quickly on
            # small-pool relations.
            for _ in range(8):
                t_fake = rng.choice(pool)
                if t_fake == t:
                    continue
                trip = (h, r, t_fake)
                if trip in triples_set or trip in seen_out:
                    continue
                out.append(trip)
                seen_out.add(trip)
                break
    return out


# ---------------------------------------------------------------------------
# 6. Pretty CI rendering for tutorial output
# ---------------------------------------------------------------------------

def fmt_ci(point: float, lo: float, hi: float, *, pct: bool = False) -> str:
    """Render a point estimate with CI for tutorial print statements."""
    if pct:
        return f"{point:.1%}  [95% CI {lo:.1%}, {hi:.1%}]"
    return f"{point:.4f}  [95% CI {lo:.4f}, {hi:.4f}]"


@dataclass
class CIRow:
    """A single labelled point estimate with its CI."""
    label: str
    k: int
    n: int
    point: float
    lo: float
    hi: float

    @classmethod
    def from_binomial(cls, label: str, k: int, n: int) -> "CIRow":
        p, lo, hi = wilson_ci(k, n)
        return cls(label=label, k=k, n=n, point=p, lo=lo, hi=hi)

    def render(self) -> str:
        return (f"{self.label:32s}  {self.k:>4d}/{self.n:<4d}  "
                f"{fmt_ci(self.point, self.lo, self.hi, pct=True)}")


# ---------------------------------------------------------------------------
# 7. Calibrated-threshold loader (single source of truth across NB 16/19/20/22)
# ---------------------------------------------------------------------------

def calibrated_thresholds_for(
    store_name: str = "ch04_ground_truth",
    *,
    seed: int = 42,
) -> dict[str, float]:
    """Calibrate a fresh `DOEGMSBenchmark` judge on ``store_name`` and
    return the trained τ thresholds.

    Used by NB 16 (τ_geo), NB 19/20 (gate τ), NB 22 (demo τ in
    failure-attribution detail strings).  Centralising the call ensures
    every notebook quotes the same numbers and a single recalibration
    propagates.
    """
    import contextlib
    import io

    from knowlytix.harness.testing import DOEGMSBenchmark

    cfg = tutorial_doe_config(
        store_name=store_name,
        n_runs=1,
        max_per_category=1,
        enable_typed_verdicts=False,
        enable_tracing=False,
        target_type="mock",  # judge.calibrate() only needs the store + judge
    )
    bench = DOEGMSBenchmark(cfg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bench.ingest()
        bench.judge.calibrate(seed=seed)
    return dict(bench.judge._thresholds)
