# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed artifact bundle produced by offline validation.

A ``GovernanceBundle`` carries everything ``RuntimeGate`` needs to
issue bounded-latency entity-comparison verdicts without recomputing
calibration. The bundle is the contract between the offline validation
pipeline (heavy: bootstrap, sensitivity, adversarial sweep) and the
runtime gate (light: O(1) lookups + one forward pass + one isotonic
apply).

The bundle is content-addressed: every load verifies the SHA-256
signature over the canonical JSON payload. A tampered bundle raises
``BundleSignatureError`` at load time.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from knowlytix.harness.governance.bundle_signing import (
    SigningMethod, resolve_signing_key, sign, verify,
)
from knowlytix.harness.governance.store_fingerprint import StoreFingerprint
from knowlytix.harness.governance.support_calibration import CalibratedSupportThresholds


BUNDLE_SCHEMA_VERSION = "1.4"


@dataclass(frozen=True)
class OperatingPoint:
    """SLO for the runtime gate's deferral behavior.

    Defines the maximum acceptable rate at which the gate may abstain
    or route to review. Violations indicate the gate is deferring
    more than the offline review queue can absorb (or, conversely,
    that the gate is deferring too little to be safe).
    """

    max_abstain_rate: float = 0.20
    max_review_rate: float = 0.20
    max_uncertain_rate: float = 0.30  # abstain + review combined


class BundleSignatureError(RuntimeError):
    """Bundle signature does not match contents."""


@dataclass(frozen=True)
class EntitySupport:
    """Per-entity coverage statistics computed at bundle build time."""

    triple_count: int
    relation_count: int

    def is_supported(self, min_triples: int, min_relations: int) -> bool:
        return (self.triple_count >= min_triples
                and self.relation_count >= min_relations)


@dataclass(frozen=True)
class ThresholdDistribution:
    """Bootstrap distribution of a calibrated cut.

    ``mean`` is the value the runtime gate uses; ``ci_low`` / ``ci_hi``
    are reported in the verdict for auditability.
    """

    mean: float
    std: float
    ci_low: float
    ci_hi: float
    n_bootstrap: int


@dataclass
class GovernanceBundle:
    """Frozen artifact bundle for entity-comparison governance.

    The runtime gate is read-only with respect to the bundle. Any
    field change requires producing and signing a new bundle.
    """

    schema_version: str
    bundle_id: str
    created_at: str
    source_store_path: str

    # Tension cuts: present only when the corpus declares oppositions
    # via (r_a, opposite_of, r_b) triples.  Stores without declared
    # oppositions (e.g., the orchestrator's data_science_practices
    # corpus in Ch 10) ship a bundle with tau_ent / tau_contra and the
    # isotonic mapping set to None — the bundle still provides the
    # plausibility threshold, support index, fingerprint, drift
    # baseline, and operating point for claim-verification surfaces.
    tau_ent: ThresholdDistribution | None
    tau_contra: ThresholdDistribution | None

    isotonic_x: np.ndarray | None
    isotonic_y: np.ndarray | None

    support_index: dict[str, EntitySupport]
    opposition_table: dict[tuple[str, str], str]

    drift_baseline_te: np.ndarray
    drift_baseline_te_quantiles: tuple[float, ...]

    validation_metrics: dict[str, float]

    # Plausibility threshold for the geodesic measure. Always present
    # when calibration was viable (the geodesic logistic regression
    # only needs real-vs-corrupted-tail labels, not oppositions).
    tau_plausibility: ThresholdDistribution | None = None

    store_fingerprint: StoreFingerprint | None = None
    support_thresholds: CalibratedSupportThresholds | None = None
    operating_point: OperatingPoint | None = None
    # Per-ENM-category numeric range learned at calibration time:
    # ``{category: (min, max, n_observations)}``.  Used by
    # ``ClaimVerifier`` as a second attestor on numeric claims --- a
    # claim whose value matches ENM but lies outside the calibrated
    # range routes to review rather than PASS.
    numeric_range_index: dict | None = None

    signature: str = ""
    signing_method: SigningMethod = "sha256"

    @property
    def is_tension_capable(self) -> bool:
        """True iff bundle carries calibrated tension thresholds."""
        return (self.tau_ent is not None and self.tau_contra is not None
                and self.isotonic_x is not None and self.isotonic_y is not None)

    @property
    def is_plausibility_capable(self) -> bool:
        """True iff bundle carries a calibrated plausibility threshold."""
        return self.tau_plausibility is not None

    @property
    def isotonic(self) -> IsotonicRegression:
        if not self.is_tension_capable:
            raise ValueError(
                "Bundle has no isotonic mapping — tension calibration "
                "was not viable on the source store. Use "
                "is_tension_capable to check before accessing."
            )
        iso = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        )
        iso.fit(self.isotonic_x, self.isotonic_y)
        return iso

    def support_for(self, entity: str) -> EntitySupport | None:
        return self.support_index.get(entity)

    def is_declared_opposition(self, a: str, b: str) -> bool:
        return ((a, b) in self.opposition_table
                or (b, a) in self.opposition_table)

    def canonical_payload(self) -> bytes:
        """Bytes used for signing. Excludes the signature itself."""
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_store_path": self.source_store_path,
            "tau_ent": (asdict(self.tau_ent)
                        if self.tau_ent is not None else None),
            "tau_contra": (asdict(self.tau_contra)
                           if self.tau_contra is not None else None),
            "tau_plausibility": (asdict(self.tau_plausibility)
                                 if self.tau_plausibility is not None
                                 else None),
            "isotonic_x": (self.isotonic_x.tolist()
                           if self.isotonic_x is not None else None),
            "isotonic_y": (self.isotonic_y.tolist()
                           if self.isotonic_y is not None else None),
            "support_index": {
                k: asdict(v) for k, v in sorted(self.support_index.items())
            },
            "opposition_table": sorted(
                [list(k) + [v] for k, v in self.opposition_table.items()]
            ),
            "drift_baseline_te": self.drift_baseline_te.tolist(),
            "drift_baseline_te_quantiles": list(
                self.drift_baseline_te_quantiles
            ),
            "validation_metrics": dict(sorted(self.validation_metrics.items())),
            "store_fingerprint": (
                self.store_fingerprint.to_dict()
                if self.store_fingerprint is not None else None
            ),
            "support_thresholds": (
                {
                    "min_triples": self.support_thresholds.min_triples,
                    "min_relations": self.support_thresholds.min_relations,
                    "target_accuracy": self.support_thresholds.target_accuracy,
                    "coverage_accuracy": self.support_thresholds.coverage_accuracy,
                    "n_pairs": self.support_thresholds.n_pairs,
                    "coverage_pairs": self.support_thresholds.coverage_pairs,
                }
                if self.support_thresholds is not None else None
            ),
            "operating_point": (
                {
                    "max_abstain_rate": self.operating_point.max_abstain_rate,
                    "max_review_rate": self.operating_point.max_review_rate,
                    "max_uncertain_rate": self.operating_point.max_uncertain_rate,
                }
                if self.operating_point is not None else None
            ),
            "numeric_range_index": (
                {k: list(v) for k, v in sorted(
                    self.numeric_range_index.items())}
                if self.numeric_range_index is not None else None
            ),
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def freeze(
        self, key: bytes | str | None = None,
    ) -> "GovernanceBundle":
        """Compute and stamp the signature. Call once after assembly.

        If a key is provided (or ``GOVERNED_AGENT_SIGNING_KEY`` is set
        in the environment) the bundle is signed with HMAC-SHA-256;
        otherwise it falls back to SHA-256 integrity-only signing.
        The chosen method is recorded on the bundle and is part of
        the signed payload, so downgrading the method invalidates
        the signature.
        """
        resolved = resolve_signing_key(key)
        # Decide the method *before* computing the canonical payload,
        # since the method is part of the payload.
        self.signing_method = "hmac-sha256" if resolved is not None else "sha256"
        self.signature, _ = sign(self.canonical_payload(), key=resolved)
        return self

    def compute_signature(
        self, key: bytes | str | None = None,
    ) -> str:
        """Recompute the signature for the current state. For tests."""
        resolved = resolve_signing_key(key)
        sig, _ = sign(self.canonical_payload(), key=resolved)
        return sig

    def verify(self, key: bytes | str | None = None) -> bool:
        resolved = resolve_signing_key(key)
        return verify(
            self.canonical_payload(),
            self.signature,
            self.signing_method,
            key=resolved,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, path: str | Path, *, key: bytes | str | None = None,
    ) -> "GovernanceBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: GovernanceBundle = pickle.load(f)
        if not isinstance(bundle, cls):
            raise TypeError(
                f"Loaded object is {type(bundle).__name__}, "
                f"expected {cls.__name__}"
            )
        try:
            ok = bundle.verify(key=key)
        except BundleSigningError as exc:
            # A bundle with an unverifiable signing scheme (e.g.
            # downgraded sha256 in a HMAC deployment) must be
            # rejected with the same error as a content tamper.
            raise BundleSignatureError(
                f"Refusing to load bundle at {path}: {exc}"
            ) from exc
        if not ok:
            raise BundleSignatureError(
                f"Signature mismatch on bundle at {path}: "
                f"signing_method={bundle.signing_method}. The bundle "
                f"has been tampered with, the signing key is wrong, "
                f"or the file is corrupted."
            )
        return bundle


def make_bundle_id(source_store_path: str, created_at: str) -> str:
    """Deterministic short id derived from store path + timestamp."""
    raw = f"{source_store_path}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
