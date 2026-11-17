# SPDX-License-Identifier: Apache-2.0
"""Bootstrap-calibrated policy thresholds for the four-layer PolicyEngine.

The single-fit ``GMSJudge.calibrate_policy()`` produces one
``policy_geodesic`` value; in high-stakes deployment that hides which
side of the threshold borderline ``(agent, uses_tool, tool)`` decisions
were on at the time of the fit. This module bootstraps the policy
geodesic logistic regression on K resamples of the labeled
(positive, corrupted-tail-negative) tool-use pairs and refits the
threshold per resample. Returns the bootstrap distribution as a
:class:`ThresholdDistribution` (mean, std, 95% CI).

The runtime PolicyEngine consumes the mean (the operating threshold);
the auditor reads the CI when a Layer 3 verdict lands near the
boundary. The same pattern is applied to ``tau_contra`` for Layer 4
when the corpus carries declared-contradiction triples.

Built to mirror :mod:`governed_agent.plausibility_calibration` for
consistency with the Ch 10 orchestrator surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from knowlytix.harness.governance.governance_bundle import ThresholdDistribution

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


def _derive_threshold(clf, X: np.ndarray) -> float | None:
    """Find τ where ``P(class=1 | value=τ) = 0.5`` on the fitted LR.

    Mirrors :func:`knowlytix.harness.testing.threshold_calibration.fit_binary_threshold`
    so the bootstrap distribution agrees with the single-fit point.
    """
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


def _build_policy_geodesic_training_set(
    store: "GMSExpertStore",
    *,
    relation: str = "uses_tool",
    seed: int = 42,
    max_negatives_per_positive: int = 3,
):
    """Build (real, corrupted-tail) policy pairs and their geodesic scores.

    Mirrors :func:`knowlytix.harness.testing.threshold_calibration.label_policy_geodesic`
    so the bootstrap sees the same population the single-fit calibration
    did. Returns ``(X[N,1], y[N])`` ready for sklearn LR.
    """
    import random
    rng = random.Random(seed)

    positives = [
        (h, r, t) for (h, r, t) in store.doc_graph.triples
        if r == relation
    ]
    if not positives:
        return np.empty((0, 1)), np.empty((0,), dtype=int)

    known_tails: dict[str, set[str]] = {}
    for (h, _, t) in positives:
        known_tails.setdefault(h, set()).add(t)
    all_tails = sorted({t for (_, _, t) in positives})

    negatives: list[tuple[str, str, str]] = []
    if all_tails:
        for (h, _, _) in positives:
            banned = known_tails.get(h, set())
            candidates = [t for t in all_tails if t not in banned]
            if not candidates:
                continue
            for t_bad in rng.sample(
                candidates,
                min(max_negatives_per_positive, len(candidates)),
            ):
                negatives.append((h, relation, t_bad))

    X: list[float] = []
    y: list[int] = []

    def _append(triples: list[tuple[str, str, str]], label: int) -> None:
        for (h, r, t) in triples:
            d = store.score_triple(h, r, t)
            if d is None:
                continue
            X.append(float(d))
            y.append(label)

    _append(positives, 1)
    _append(negatives, 0)
    return (np.asarray(X, dtype=float).reshape(-1, 1),
            np.asarray(y, dtype=int))


def bootstrap_policy_geodesic_threshold(
    store: "GMSExpertStore",
    *,
    relation: str = "uses_tool",
    n_bootstrap: int = 200,
    seed: int = 42,
) -> ThresholdDistribution:
    """Refit the policy geodesic threshold on K bootstrap resamples.

    Cap-aware: if the store was cap-trained
    (``store.model.cap_enabled``) the per-relation cap radius is the
    learned admissibility floor; we report it as a degenerate
    distribution (mean=ρ, std=0, CI collapsed to ρ). Auditors who see
    ``std=0.0`` know the threshold came from the geometry, not from
    a post-hoc fit.

    Returns the bootstrap distribution. Pass the ``mean`` to the
    runtime engine; expose ``ci_low`` / ``ci_hi`` in the audit report.

    Raises ``RuntimeError`` if too few resamples produce a valid fit
    (the relation likely has too few positives or unstable separation).
    """
    # Cap-trained shortcut: the radius is the policy threshold; no fit.
    if getattr(store.model, "cap_enabled", False):
        rho = store.cap_radius(relation)
        if rho is not None:
            rho_f = float(rho)
            return ThresholdDistribution(
                mean=rho_f, std=0.0,
                ci_low=rho_f, ci_hi=rho_f,
                n_bootstrap=0,
            )

    from sklearn.linear_model import LogisticRegression

    X, y = _build_policy_geodesic_training_set(
        store, relation=relation, seed=seed,
    )
    n = len(y)
    if n < 30:
        raise RuntimeError(
            f"Policy geodesic training set for relation '{relation}' has "
            f"only {n} samples; need at least 30 for bootstrap calibration. "
            f"Train on a richer policy corpus or seed more "
            f"({relation}) triples."
        )
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Policy geodesic training set for relation '{relation}' is "
            f"single-class; need both authorized and unauthorized samples."
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
            f"Bootstrap calibration for relation '{relation}' produced "
            f"too few valid fits: {len(draws)}/{n_bootstrap}. The geodesic "
            f"separation is unstable on this store."
        )

    arr = np.asarray(draws)
    return ThresholdDistribution(
        mean=float(arr.mean()),
        std=float(arr.std()),
        ci_low=float(np.quantile(arr, 0.025)),
        ci_hi=float(np.quantile(arr, 0.975)),
        n_bootstrap=len(draws),
    )


def bootstrap_policy_geodesic_empirical(
    store: "GMSExpertStore",
    *,
    relation: str = "uses_tool",
    n_bootstrap: int = 200,
    seed: int = 42,
) -> ThresholdDistribution | None:
    """Always-empirical bootstrap on labeled (authorized, unauthorized) pairs.

    Unlike :func:`bootstrap_policy_geodesic_threshold`, this function
    skips the cap-trained shortcut and *always* runs the resampling
    fit on the labeled pairs. Returns ``None`` if the corpus has too
    few pairs for a stable bootstrap (>= 30 samples, both classes
    present, >= n_bootstrap/4 valid fits).

    The returned ``ThresholdDistribution`` is the auditor's answer to
    ``what is the threshold's CI under shift?'' --- a question the
    cap-derived threshold (``std=0, n_bootstrap=0``) does not
    address. A high-stakes deployment ships the bundle carrying
    *both* values: the cap-derived one as the runtime threshold (the
    geometry's learned admissibility floor) and the empirical CI as
    documentation of the empirical separation.
    """
    from sklearn.linear_model import LogisticRegression

    X, y = _build_policy_geodesic_training_set(
        store, relation=relation, seed=seed,
    )
    n = len(y)
    if n < 30:
        return None
    if len(np.unique(y)) < 2:
        return None

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
        return None

    arr = np.asarray(draws)
    return ThresholdDistribution(
        mean=float(arr.mean()),
        std=float(arr.std()),
        ci_low=float(np.quantile(arr, 0.025)),
        ci_hi=float(np.quantile(arr, 0.975)),
        n_bootstrap=len(draws),
    )


# ---------------------------------------------------------------------------
# Layer 4 (tau_contra) bootstrap, only when the corpus declares oppositions
# ---------------------------------------------------------------------------


def bootstrap_policy_tau_contra(
    store: "GMSExpertStore",
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> ThresholdDistribution | None:
    """Bootstrap ``tau_contra`` for Layer 4 contradiction detection.

    Uses the same labeled tension-energy samples that
    :func:`calibrate_logic_tension` would consume. Returns ``None`` when
    the corpus has no declared-contradiction structure (no
    ``opposite_of`` triples and no per-class span at calibration time).

    A bundle that ships ``tau_contra=None`` documents Layer 4 as
    ``signal_unavailable`` rather than fail-open at the class default.
    """
    try:
        from knowlytix.harness.testing.primitive_calibration import label_tension
        from knowlytix.harness.testing.threshold_calibration import (
            calibrate_logic_tension,
        )
    except Exception:
        return None

    samples = label_tension(store, seed=seed)
    if not samples:
        return None

    rng = np.random.default_rng(seed)
    # Bootstrap by index over the labeled tension samples.
    n = len(samples)
    if n < 30:
        return None

    draws: list[float] = []
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot = [samples[i] for i in idx]
        clf, cuts = calibrate_logic_tension(boot, seed=seed + k)
        if cuts is None:
            continue
        if (not math.isfinite(cuts.tau_contra)
                or cuts.tau_contra <= cuts.tau_ent):
            continue
        draws.append(float(cuts.tau_contra))

    if len(draws) < max(10, n_bootstrap // 4):
        return None

    arr = np.asarray(draws)
    return ThresholdDistribution(
        mean=float(arr.mean()),
        std=float(arr.std()),
        ci_low=float(np.quantile(arr, 0.025)),
        ci_hi=float(np.quantile(arr, 0.975)),
        n_bootstrap=len(draws),
    )


# ---------------------------------------------------------------------------
# Two-attestor verdict on a policy decision
# ---------------------------------------------------------------------------


@dataclass
class PolicyAttestorVerdict:
    """Quorum result on a four-layer policy check.

    A high-stakes decision PASSes only when at least two layers
    independently concur on BLOCK (or all-ALLOW). One-layer signals
    demote to REVIEW so an operator examines the call before the
    gateway lets it through. Layers that returned
    ``signal_unavailable`` (entity not on manifold; threshold not
    calibrated) are excluded from the quorum count, never silently
    treated as ALLOW.
    """

    decision: str  # "BLOCK" | "ALLOW" | "REVIEW"
    n_layers_blocking: int
    n_layers_signal_available: int
    layer1_block: bool
    layer2_block: bool
    layer3_block: bool | None       # None when signal unavailable
    layer4_block: bool | None       # None when signal unavailable
    blocking_rules: list[str]
    quorum_required: int
    evidence: str

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "n_layers_blocking": self.n_layers_blocking,
            "n_layers_signal_available":
                self.n_layers_signal_available,
            "layer1_block": self.layer1_block,
            "layer2_block": self.layer2_block,
            "layer3_block": self.layer3_block,
            "layer4_block": self.layer4_block,
            "blocking_rules": list(self.blocking_rules),
            "quorum_required": self.quorum_required,
            "evidence": self.evidence,
        }


def derive_attestor_verdict(
    decision,
    *,
    quorum_required: int = 2,
) -> PolicyAttestorVerdict:
    """Convert a :class:`PolicyDecision` into a quorum verdict.

    The four PolicyEngine layers produce rule IDs with distinct
    prefixes:

      * Layer 1 — explicit YAML ``rule_id`` (e.g. ``RML-008``).
      * Layer 2 — ``gms_graph:<head>-><rel>-><tail>``.
      * Layer 3 — ``gms_plausibility:<entity>``.
      * Layer 4 — recorded only as a tension entry under
        ``gms_scores`` since it never fires alone.

    Quorum policy:

      * ALL-ALLOW (no blocking layers)             → ``ALLOW``
      * ≥ ``quorum_required`` blocking layers      → ``BLOCK``
      * Exactly one blocking layer                 → ``REVIEW``
    """
    blocking = list(decision.blocking_rules)

    layer1 = any(
        not r.startswith("gms_graph:")
        and not r.startswith("gms_plausibility:")
        for r in blocking
    )
    layer2 = any(r.startswith("gms_graph:") for r in blocking)
    layer3 = any(r.startswith("gms_plausibility:") for r in blocking)

    # Layer 4 fires only as a reinforcement on already-blocking rules.
    # We surface it as a separate signal: if any tension score in
    # gms_scores exceeds the engine's contradiction threshold we count
    # it. We can't read the engine's TENSION_THRESHOLD here, so we use
    # the canonical 1.7 floor; downstream callers wanting the engine's
    # actual cut can call this with the engine's value via
    # `attestor_verdict_from_engine`.
    layer4_signals = [
        v for k, v in decision.gms_scores.items()
        if isinstance(k, str) and k.startswith("tension:")
    ]
    layer4: bool | None
    if not layer4_signals:
        layer4 = None
    else:
        layer4 = any(float(v) > 1.7 for v in layer4_signals)

    # Layer 3 signal availability: gms_scores carries 'plausibility'
    # only when the engine actually scored. If absent and no
    # gms_plausibility:* rule fired, signal is unavailable.
    has_plausibility_score = "plausibility" in decision.gms_scores
    if has_plausibility_score:
        layer3_avail: bool = True
    elif layer3:
        layer3_avail = True
    else:
        layer3_avail = False
    layer3_signal: bool | None = layer3 if layer3_avail else None

    # Layer 1, 2 always signal-available (they always either match or
    # don't, no calibration involved).
    n_avail = 2  # L1 + L2 always available
    if layer3_signal is not None:
        n_avail += 1
    if layer4 is not None:
        n_avail += 1

    n_blocking = (
        int(layer1) + int(layer2)
        + int(layer3_signal is True)
        + int(layer4 is True)
    )

    if n_blocking == 0:
        result = "ALLOW"
    elif n_blocking >= quorum_required:
        result = "BLOCK"
    else:
        result = "REVIEW"

    parts = [
        f"L1={'block' if layer1 else 'pass'}",
        f"L2={'block' if layer2 else 'pass'}",
        f"L3={'block' if layer3_signal else 'pass' if layer3_signal is False else 'unavailable'}",
        f"L4={'block' if layer4 else 'pass' if layer4 is False else 'unavailable'}",
    ]
    evidence = (
        f"quorum {n_blocking}/{n_avail} (need ≥{quorum_required}); "
        + "; ".join(parts)
    )

    return PolicyAttestorVerdict(
        decision=result,
        n_layers_blocking=n_blocking,
        n_layers_signal_available=n_avail,
        layer1_block=layer1,
        layer2_block=layer2,
        layer3_block=layer3_signal,
        layer4_block=layer4,
        blocking_rules=blocking,
        quorum_required=quorum_required,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Convenience: engine-aware quorum check (uses engine's calibrated
# tension threshold instead of the hardcoded 1.7 floor)
# ---------------------------------------------------------------------------


def check_with_quorum(
    engine,
    action: str,
    args: dict,
    context: str,
    *,
    quorum_required: int = 2,
):
    """Run the engine and return ``(PolicyDecision, PolicyAttestorVerdict)``.

    The contract: a high-stakes deployment binds the gateway to this
    function rather than to ``engine.check`` directly. Block decisions
    require concurrence by at least ``quorum_required`` layers; single-
    layer signals demote to REVIEW (the gateway routes those to a
    human reviewer); ``signal_unavailable`` layers (entity not on
    manifold; threshold not calibrated; tension cuts not derivable)
    are excluded from the quorum count -- never silently treated as
    ALLOW.

    The returned verdict carries the engine's calibrated tension
    threshold (not the class-default 1.7) so the contradiction
    signal matches the manifold the engine was trained on.
    """
    decision = engine.check(action, args, context)
    verdict = derive_attestor_verdict(
        decision, quorum_required=quorum_required,
    )
    # Re-evaluate Layer 4 with the engine's actual TENSION_THRESHOLD
    # so the quorum reflects the calibrated cut.
    tau_contra = float(getattr(
        engine, "TENSION_THRESHOLD",
        type(engine).TENSION_THRESHOLD,
    ))
    layer4_signals = [
        v for k, v in decision.gms_scores.items()
        if isinstance(k, str) and k.startswith("tension:")
    ]
    if layer4_signals:
        layer4_block = any(float(v) > tau_contra for v in layer4_signals)
        # Recompute quorum with refreshed Layer 4
        n_blocking = (
            int(verdict.layer1_block) + int(verdict.layer2_block)
            + int(verdict.layer3_block is True) + int(layer4_block)
        )
        if n_blocking == 0:
            new_decision = "ALLOW"
        elif n_blocking >= quorum_required:
            new_decision = "BLOCK"
        else:
            new_decision = "REVIEW"
        evidence = verdict.evidence + (
            f"; tau_contra={tau_contra:.4f} "
            f"(class default 1.7 was used initially)"
        )
        verdict = PolicyAttestorVerdict(
            decision=new_decision,
            n_layers_blocking=n_blocking,
            n_layers_signal_available=verdict.n_layers_signal_available,
            layer1_block=verdict.layer1_block,
            layer2_block=verdict.layer2_block,
            layer3_block=verdict.layer3_block,
            layer4_block=layer4_block,
            blocking_rules=verdict.blocking_rules,
            quorum_required=quorum_required,
            evidence=evidence,
        )
    return decision, verdict
