# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed policy artefact bundle for high-stakes deployment.

A :class:`PolicyBundle` is the contract between offline policy
calibration (heavy: bootstrap, OOD validation, adversarial sweep,
conflict detection) and the runtime policy engine (light: O(1) rule
match + bounded geometric scoring + quorum derivation). The bundle
carries:

  * the YAML rule set (frozen by SHA-256 over canonical YAML)
  * bootstrap-calibrated thresholds for Layer 3 and (when viable)
    Layer 4 of the four-layer check
  * the policy store fingerprint so the runtime refuses to bind to
    a non-matching store
  * an operating-point SLO (max block / review rate)
  * an HMAC-SHA-256 signature over the canonical payload

A bundle that ships ``tau_contra=None`` documents Layer 4 as
``signal_unavailable`` rather than fail-open at the class default --
the runtime quorum then runs on Layers 1-3 with the calibrated L3
threshold.

Mirrors :class:`governed_agent.governance_bundle.GovernanceBundle`
for the policy surface; signing path reuses
:mod:`governed_agent.bundle_signing`.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from knowlytix.harness.governance.bundle_signing import (
    SigningMethod, resolve_signing_key, sign, verify,
)
from knowlytix.harness.governance.governance_bundle import (
    BundleSignatureError, ThresholdDistribution,
)
from knowlytix.harness.governance.store_fingerprint import StoreFingerprint

if TYPE_CHECKING:
    from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule


POLICY_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PolicyOperatingPoint:
    """SLO for the policy engine's deferral and refusal behaviour.

    A deployment that produces a higher review-rate than this
    quantity has either drifted (recalibrate) or has a corpus mismatch
    (review the agent identity rolls).
    """

    max_review_rate: float = 0.20
    max_block_rate: float = 0.80   # heavy block rate signals dataset shift
    min_block_rate: float = 0.10   # too few blocks suggests fail-open

    def violates(self, *, block_rate: float, review_rate: float) -> bool:
        return (review_rate > self.max_review_rate
                or block_rate > self.max_block_rate
                or block_rate < self.min_block_rate)


@dataclass
class PolicyBundle:
    """Frozen artefact bundle for high-stakes policy enforcement."""

    schema_version: str
    bundle_id: str
    created_at: str
    source_store_path: str

    # YAML rules: stored as dicts so the bundle is fully serialisable.
    rules: list[dict]
    rules_yaml_hash: str  # SHA-256 over canonical YAML payload

    # Layer 3: always present when the policy store has a viable
    # uses_tool relation. On a cap-trained store this is the
    # geometry's learned admissibility radius (std=0); on a non-cap
    # store this is the bootstrap-fit empirical threshold.
    policy_geodesic: ThresholdDistribution | None = None

    # Empirical bootstrap on labeled (authorized, unauthorized) pairs.
    # Always-empirical: skips the cap shortcut. Documents the
    # threshold's CI under shift, the auditor's primary follow-up
    # question to a cap-derived threshold reporting std=0.
    # ``None`` when the corpus has too few pairs for a stable fit.
    policy_geodesic_empirical: ThresholdDistribution | None = None

    # Layer 4: present only when the corpus carries declared
    # contradictions (opposite_of triples) with a per-class span
    # large enough for tension calibration.
    tau_contra: ThresholdDistribution | None = None

    # Plausibility relation used at Layer 3 (e.g. "uses_tool" for
    # the agent-centric data_science corpus).
    plausibility_relation: str = "uses_tool"

    # Numeric range index for state-tracking rules (RML-007 budget,
    # RML-004 model count, etc.). Maps category -> (min, max, n_obs).
    numeric_range_index: dict[str, tuple[float, float, int]] | None = None

    # Bound the runtime to the store the bundle was calibrated on.
    store_fingerprint: StoreFingerprint | None = None

    operating_point: PolicyOperatingPoint = field(
        default_factory=PolicyOperatingPoint,
    )

    # Quorum required for a BLOCK to ship (default 2-of-N attestors).
    quorum_required: int = 2

    # Validation evidence at calibration time (FAR/FRR/safety_rate
    # from the OOD + adversarial sweeps; a deployment that ships a
    # bundle with worse numbers than the previous bundle should
    # require a deployment-team sign-off).
    validation_metrics: dict[str, float] = field(default_factory=dict)

    signature: str = ""
    signing_method: SigningMethod = "sha256"

    # Multi-signer (k-of-n) signatures, written by
    # :func:`policy_multisign.attach_signatures`. Each entry is a
    # ``{"signer_id", "method", "signature"}`` dict so the bundle is
    # fully serialisable. The single-signer ``signature`` field
    # remains the integrity-only / single-HMAC fast path; the
    # multi-signer field is the high-stakes ship requirement.
    attestor_signatures: list[dict] = field(default_factory=list)

    # ----------------------------------------------------------------- #
    # Capabilities
    # ----------------------------------------------------------------- #
    @property
    def is_layer3_capable(self) -> bool:
        return self.policy_geodesic is not None

    @property
    def is_layer4_capable(self) -> bool:
        return self.tau_contra is not None

    # ----------------------------------------------------------------- #
    # Canonical payload + signing
    # ----------------------------------------------------------------- #
    def canonical_payload(self) -> bytes:
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_store_path": self.source_store_path,
            "rules": sorted(
                self.rules,
                key=lambda r: r.get("rule_id", "") if isinstance(r, dict)
                              else "",
            ),
            "rules_yaml_hash": self.rules_yaml_hash,
            "policy_geodesic": (asdict(self.policy_geodesic)
                                if self.policy_geodesic is not None
                                else None),
            "policy_geodesic_empirical": (
                asdict(self.policy_geodesic_empirical)
                if self.policy_geodesic_empirical is not None
                else None
            ),
            "tau_contra": (asdict(self.tau_contra)
                           if self.tau_contra is not None else None),
            "plausibility_relation": self.plausibility_relation,
            "numeric_range_index": (
                {k: list(v) for k, v in
                 sorted(self.numeric_range_index.items())}
                if self.numeric_range_index is not None else None
            ),
            "store_fingerprint": (
                self.store_fingerprint.to_dict()
                if self.store_fingerprint is not None else None
            ),
            "operating_point": {
                "max_review_rate": self.operating_point.max_review_rate,
                "max_block_rate": self.operating_point.max_block_rate,
                "min_block_rate": self.operating_point.min_block_rate,
            },
            "quorum_required": self.quorum_required,
            "validation_metrics": dict(
                sorted(self.validation_metrics.items())
            ),
            "signing_method": self.signing_method,
            # NOTE: attestor_signatures are intentionally NOT in the
            # canonical payload, since they are computed *over* the
            # canonical payload. Including them would create a
            # circular dependency.
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def freeze(
        self, key: bytes | str | None = None,
    ) -> "PolicyBundle":
        """Compute and stamp the signature. Call once after assembly.

        With a deployment key (or
        ``GOVERNED_AGENT_SIGNING_KEY`` env var) the bundle is
        HMAC-SHA-256 signed; otherwise SHA-256 integrity-only.
        Downgrade is detected on load because the method is part of
        the signed payload.
        """
        resolved = resolve_signing_key(key)
        self.signing_method = (
            "hmac-sha256" if resolved is not None else "sha256"
        )
        self.signature, _ = sign(
            self.canonical_payload(), key=resolved,
        )
        return self

    def verify(self, key: bytes | str | None = None) -> bool:
        resolved = resolve_signing_key(key)
        return verify(
            self.canonical_payload(),
            self.signature,
            self.signing_method,
            key=resolved,
        )

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, path: str | Path, *, key: bytes | str | None = None,
    ) -> "PolicyBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: PolicyBundle = pickle.load(f)
        if not isinstance(bundle, cls):
            raise TypeError(
                f"Loaded object is {type(bundle).__name__}, "
                f"expected {cls.__name__}"
            )
        try:
            ok = bundle.verify(key=key)
        except BundleSigningError as exc:
            raise BundleSignatureError(
                f"Refusing to load policy bundle at {path}: {exc}"
            ) from exc
        if not ok:
            raise BundleSignatureError(
                f"Signature mismatch on policy bundle at {path}: "
                f"signing_method={bundle.signing_method}. The bundle "
                f"has been tampered with, the signing key is wrong, "
                f"or the file is corrupted."
            )
        return bundle


# ---------------------------------------------------------------------------
# Bundle production
# ---------------------------------------------------------------------------


def _hash_rules(rules: list["PolicyRule"]) -> str:
    """Deterministic hash over the rule set."""
    payload = json.dumps(
        sorted([r.to_dict() for r in rules],
               key=lambda d: d.get("rule_id", "")),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_bundle_id(source_store_path: str, created_at: str) -> str:
    raw = f"{source_store_path}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_policy_bundle(
    *,
    engine: "PolicyEngine",
    rules: list["PolicyRule"],
    source_store_path: str,
    relation: str = "uses_tool",
    n_bootstrap: int = 200,
    seed: int = 42,
    operating_point: PolicyOperatingPoint | None = None,
    quorum_required: int = 2,
    validation_metrics: dict[str, float] | None = None,
    numeric_range_index: dict[str, tuple[float, float, int]] | None = None,
) -> PolicyBundle:
    """Produce a calibrated, signed policy bundle from an engine + rules.

    Steps:

      1. Bootstrap-calibrate ``policy_geodesic`` (Layer 3) on the
         engine's store using the labeled tool-use pairs.
      2. Optionally bootstrap ``tau_contra`` (Layer 4); falls back to
         ``None`` when the corpus has no declared oppositions.
      3. Stamp the store fingerprint so the runtime refuses to bind to
         a different store.
      4. Sign the canonical payload (HMAC-SHA-256 with deployment key,
         else SHA-256).
    """
    from knowlytix.harness.governance.policy_calibration import (
        bootstrap_policy_geodesic_empirical,
        bootstrap_policy_geodesic_threshold,
        bootstrap_policy_tau_contra,
    )
    from knowlytix.harness.governance.store_fingerprint import compute_store_fingerprint

    policy_geodesic = bootstrap_policy_geodesic_threshold(
        engine.store, relation=relation,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    # Always-empirical sibling: documents the threshold's CI under
    # shift even when the runtime threshold is cap-derived (std=0).
    try:
        policy_geodesic_empirical = bootstrap_policy_geodesic_empirical(
            engine.store, relation=relation,
            n_bootstrap=n_bootstrap, seed=seed,
        )
    except Exception:
        policy_geodesic_empirical = None

    try:
        tau_contra = bootstrap_policy_tau_contra(
            engine.store, n_bootstrap=n_bootstrap, seed=seed,
        )
    except Exception:
        tau_contra = None

    fingerprint = compute_store_fingerprint(engine.store)

    bundle = PolicyBundle(
        schema_version=POLICY_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_bundle_id(source_store_path, utcnow_iso()),
        created_at=utcnow_iso(),
        source_store_path=source_store_path,
        rules=[r.to_dict() for r in rules],
        rules_yaml_hash=_hash_rules(rules),
        policy_geodesic=policy_geodesic,
        policy_geodesic_empirical=policy_geodesic_empirical,
        tau_contra=tau_contra,
        plausibility_relation=relation,
        numeric_range_index=numeric_range_index,
        store_fingerprint=fingerprint,
        operating_point=operating_point or PolicyOperatingPoint(),
        quorum_required=quorum_required,
        validation_metrics=validation_metrics or {},
    )
    return bundle.freeze()


def bind_engine_from_bundle(
    bundle: PolicyBundle,
    store: Any,
    *,
    verify_store_fingerprint: bool = True,
) -> "PolicyEngine":
    """Construct a PolicyEngine from a verified bundle.

    The engine is wired with:

      * the bundle's rule set (deserialised back into PolicyRule)
      * the bundle's calibrated thresholds (set on the instance, so
        the engine reads them directly without a judge round-trip)
      * the bundle's plausibility relation

    Refuses to bind when the store fingerprint does not match the
    fingerprint stamped at calibration time.
    """
    from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule
    from knowlytix.harness.governance.store_fingerprint import (
        compute_store_fingerprint, StoreFingerprintMismatch,
    )

    if verify_store_fingerprint and bundle.store_fingerprint is not None:
        actual = compute_store_fingerprint(store)
        if actual != bundle.store_fingerprint:
            raise StoreFingerprintMismatch(
                f"Policy bundle was calibrated on a different store. "
                f"Bundle: {bundle.store_fingerprint.entity_hash[:16]}; "
                f"actual: {actual.entity_hash[:16]}. Refuse to bind."
            )

    rules = [PolicyRule.from_dict(d) for d in bundle.rules]
    engine = PolicyEngine(
        store=store,
        rules=rules,
        plausibility_relation=bundle.plausibility_relation,
    )
    if bundle.policy_geodesic is not None:
        engine.PLAUSIBILITY_THRESHOLD = bundle.policy_geodesic.mean
        engine._calibration_source["plausibility"] = (
            "policy_bundle.policy_geodesic"
        )
    if bundle.tau_contra is not None:
        engine.TENSION_THRESHOLD = bundle.tau_contra.mean
        engine._calibration_source["tension"] = (
            "policy_bundle.tau_contra"
        )
    # Stash the bundle on the instance for the audit log.
    engine._policy_bundle = bundle
    return engine
