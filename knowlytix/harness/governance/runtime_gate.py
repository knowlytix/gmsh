# SPDX-License-Identifier: Apache-2.0
"""Bounded-latency runtime gate for entity-comparison verdicts.

Pipeline (in order; first hit wins):

  1. **Opposition shortcut** — declared (r_a, opposite_of, r_b) on a
     shared tail produces a deterministic ``contradiction`` verdict
     with ``p=1.0``. No geometry computed.
  2. **Support gate** — either entity below the support floor →
     ``abstain``. Routes to offline review queue.
  3. **Forward pass** — single ``store.tension_energy(a, b)`` call.
  4. **Isotonic apply** — bundle's frozen isotonic mapping converts
     tension → ``P(contradict)`` in O(log n).
  5. **Margin abstain** — ``P`` inside the review band → ``review``.
  6. **Final verdict** — agree / irrelevance / contradiction from
     bundle thresholds.

The gate is deterministic and reentrant. It does not cache state
between calls (callers wrap an LRU cache if they want one).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import numpy as np

from knowlytix.harness.governance.governance_bundle import GovernanceBundle
from knowlytix.harness.governance.reason_chain import ReasonChain, build_reason_chain
from knowlytix.harness.governance.store_fingerprint import (
    StoreFingerprintMismatch,
    compute_store_fingerprint,
)

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.governance.comparison_policy import ComparisonPolicy


Verdict = Literal[
    "agreement", "irrelevance", "contradiction", "review", "abstain"
]


@dataclass
class RuntimeVerdict:
    """One verdict from ``RuntimeGate.verdict``.

    Every field is suitable for emitting as telemetry. ``reason`` is
    the path the pipeline took (one of seven), so an auditor can
    reconstruct why this verdict was issued from the bundle alone.
    """

    entity_a: str
    entity_b: str
    verdict: Verdict
    p_contradiction: float | None
    tension_energy: float | None
    reason: str
    bundle_id: str
    bundle_signature: str
    latency_ms: float
    features: dict[str, float] = field(default_factory=dict)
    reason_chain: ReasonChain | None = None

    def to_dict(self) -> dict:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "verdict": self.verdict,
            "p_contradiction": self.p_contradiction,
            "tension_energy": self.tension_energy,
            "reason": self.reason,
            "bundle_id": self.bundle_id,
            "bundle_signature": self.bundle_signature[:16],
            "latency_ms": self.latency_ms,
            "features": dict(self.features),
        }


class RuntimeGate:
    """Read-only consumer of a ``GovernanceBundle``."""

    def __init__(
        self,
        bundle: GovernanceBundle,
        store: "GMSExpertStore",
        *,
        policy: "ComparisonPolicy | None" = None,
        require_two_attestors: bool = True,
        min_triples: int = 2,
        min_relations: int = 1,
        review_p_low: float = 0.35,
        review_p_high: float = 0.65,
        verify_store_fingerprint: bool = True,
        include_reason_chain: bool = True,
        reason_chain_top_k: int = 5,
    ) -> None:
        if not bundle.verify():
            raise ValueError(
                "Bundle signature does not verify. Refusing to serve."
            )
        if not bundle.is_tension_capable:
            raise ValueError(
                "Bundle has no tension thresholds (calibration likely "
                "found no declared oppositions in the corpus). "
                "RuntimeGate is for entity-comparison verdicts and "
                "requires tau_ent / tau_contra. Use the orchestrator "
                "claim-verification surface instead, or train against "
                "a corpus that declares opposite_of relations."
            )
        if verify_store_fingerprint and bundle.store_fingerprint is not None:
            current = compute_store_fingerprint(store)
            if not bundle.store_fingerprint.matches(current):
                diff = bundle.store_fingerprint.diff(current)
                raise StoreFingerprintMismatch(
                    "Runtime store does not match the bundle's "
                    "calibration-time store. Components differing: "
                    f"{sorted(diff.keys())}. The bundle's calibration "
                    "is not valid against this store; refusing to serve. "
                    f"Bundle store: n_entities="
                    f"{bundle.store_fingerprint.n_entities}, "
                    f"n_relations={bundle.store_fingerprint.n_relations}, "
                    f"n_triples={bundle.store_fingerprint.n_triples}. "
                    f"Runtime store: n_entities={current.n_entities}, "
                    f"n_relations={current.n_relations}, "
                    f"n_triples={current.n_triples}."
                )
        self.bundle = bundle
        self.store = store
        self.policy = policy
        self.require_two_attestors = require_two_attestors
        # Calibrated thresholds, when stamped on the bundle, take
        # precedence over the heuristic floor passed to __init__.
        if bundle.support_thresholds is not None:
            self.min_triples = bundle.support_thresholds.min_triples
            self.min_relations = bundle.support_thresholds.min_relations
        else:
            self.min_triples = min_triples
            self.min_relations = min_relations
        self.review_p_low = review_p_low
        self.review_p_high = review_p_high
        self.include_reason_chain = include_reason_chain
        self.reason_chain_top_k = reason_chain_top_k

        self._iso_x = bundle.isotonic_x
        self._iso_y = bundle.isotonic_y

        self._tau_ent = bundle.tau_ent.mean
        self._tau_contra = bundle.tau_contra.mean

    def _isotonic_apply(self, te: float) -> float:
        if te <= self._iso_x[0]:
            return float(self._iso_y[0])
        if te >= self._iso_x[-1]:
            return float(self._iso_y[-1])
        idx = int(np.searchsorted(self._iso_x, te))
        x0, x1 = self._iso_x[idx - 1], self._iso_x[idx]
        y0, y1 = self._iso_y[idx - 1], self._iso_y[idx]
        if x1 == x0:
            return float(y0)
        w = (te - x0) / (x1 - x0)
        return float(y0 + w * (y1 - y0))

    def _band_from_tension(self, te: float) -> Verdict:
        if te <= self._tau_ent:
            return "agreement"
        if te >= self._tau_contra:
            return "contradiction"
        return "irrelevance"

    def verdict(self, a: str, b: str) -> RuntimeVerdict:
        t0 = time.perf_counter()

        def _emit(
            v: Verdict,
            *,
            reason: str,
            te: float | None = None,
            p: float | None = None,
            extra: dict | None = None,
            attach_chain: bool = False,
        ) -> RuntimeVerdict:
            features = {}
            if te is not None:
                features["tension_energy"] = float(te)
            if p is not None:
                features["p_contradiction"] = float(p)
            features["tau_ent"] = self._tau_ent
            features["tau_contra"] = self._tau_contra
            if extra:
                features.update(extra)
            chain = None
            if attach_chain and self.include_reason_chain:
                chain = build_reason_chain(
                    a, b, self.store,
                    top_k_triples=self.reason_chain_top_k,
                )
                features["chain_n_shared"] = float(chain.n_shared_targets)
                features["chain_n_opposing"] = float(chain.n_opposing_signals)
            return RuntimeVerdict(
                entity_a=a,
                entity_b=b,
                verdict=v,
                p_contradiction=p,
                tension_energy=te,
                reason=reason,
                bundle_id=self.bundle.bundle_id,
                bundle_signature=self.bundle.signature,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                features=features,
                reason_chain=chain,
            )

        # 1. Opposition shortcut
        if self.bundle.is_declared_opposition(a, b):
            return _emit("contradiction", reason="opposition_shortcut", p=1.0)

        # 2. Support gate
        sup_a = self.bundle.support_for(a)
        sup_b = self.bundle.support_for(b)
        for entity, sup in [(a, sup_a), (b, sup_b)]:
            if sup is None:
                return _emit(
                    "abstain",
                    reason="unknown_entity",
                    extra={"missing_entity": 1.0, "which": float(entity == b)},
                )
            if not sup.is_supported(self.min_triples, self.min_relations):
                return _emit(
                    "abstain",
                    reason="thin_support",
                    extra={
                        "support_a_triples": float(sup_a.triple_count) if sup_a else 0.0,
                        "support_b_triples": float(sup_b.triple_count) if sup_b else 0.0,
                    },
                )

        # 3. Forward pass
        te = self.store.tension_energy(a, b)
        if te is None:
            return _emit("abstain", reason="tension_unavailable")

        # 4. Isotonic apply
        p = self._isotonic_apply(float(te))

        # 5. Margin abstain
        if self.review_p_low <= p <= self.review_p_high:
            return _emit("review", reason="margin_review",
                         te=float(te), p=p, attach_chain=True)

        # 6. Geometry band verdict (single-attestor result).
        v = self._band_from_tension(float(te))

        # 7. Two-attestor concurrence on the contradiction path.
        # A contradiction issued by geometry alone is demoted to
        # review unless the policy second-attestor concurs. The
        # opposition shortcut at step 1 is an explicit, documented
        # single-attestor exception (declared opposition is treated
        # as ground truth and emitted before this branch runs).
        if v == "contradiction" and self.policy is not None:
            pv = self.policy.check(a, b, self.store, self.bundle)
            extra = {
                "policy_verdict_contra": float(pv.verdict == "contradiction"),
                "policy_verdict_agree": float(pv.verdict == "agreement"),
                "policy_n_fired": float(len(pv.fired_rules)),
            }
            if pv.verdict == "contradiction":
                return _emit(
                    "contradiction",
                    reason="geometry+policy",
                    te=float(te), p=p, extra=extra, attach_chain=True,
                )
            if self.require_two_attestors:
                return _emit(
                    "review",
                    reason=("policy_disagrees" if pv.verdict == "agreement"
                            else "policy_no_concurrence"),
                    te=float(te), p=p, extra=extra, attach_chain=True,
                )
            return _emit(v, reason="geometry", te=float(te), p=p,
                         extra=extra, attach_chain=True)

        # Symmetric guard: geometry says agreement, policy says
        # contradiction → demote to review.
        if v == "agreement" and self.policy is not None:
            pv = self.policy.check(a, b, self.store, self.bundle)
            if pv.verdict == "contradiction":
                extra = {
                    "policy_verdict_contra": 1.0,
                    "policy_n_fired": float(len(pv.fired_rules)),
                }
                return _emit(
                    "review",
                    reason="policy_disagrees",
                    te=float(te), p=p, extra=extra, attach_chain=True,
                )

        return _emit(v, reason="geometry", te=float(te), p=p,
                     attach_chain=True)


def serve(
    store: "GMSExpertStore",
    bundle_path: str,
    *,
    policy: "ComparisonPolicy | None" = None,
    min_triples: int = 2,
    min_relations: int = 1,
) -> RuntimeGate:
    """Convenience: load a bundle from disk and bind it to a store."""
    bundle = GovernanceBundle.load(bundle_path)
    return RuntimeGate(
        bundle, store,
        policy=policy,
        min_triples=min_triples, min_relations=min_relations,
    )
