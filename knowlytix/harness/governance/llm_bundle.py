# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed artifact bundle for the constrained-LLM gate.

A ``ConstrainedLLMBundle`` carries everything a runtime
``ConstrainedLLM`` needs to issue auditable verdicts without
recomputing calibration: the verifier thresholds, the store
fingerprint, the allowed-tool list, the co-attestor rule pack, and
an SLO ``adversarial_floor`` (the minimum safety rate the bundle
must achieve on the standard cohort to be considered viable).

Like :class:`GovernanceBundle` (\\ref{ch:runtime-gate}), the bundle
is content-addressed and HMAC-signed: any post-build mutation
invalidates the signature, and ``load`` raises
``BundleSignatureError``.  This is the contract between offline
calibration (bootstrap thresholds, run the adversarial sweep,
record the floor) and the runtime constrained-LLM (load the
bundle, verify the store fingerprint matches, refuse to start if
the SLO floor is below the bundle's own measurement).

Why a separate bundle from the entity-comparison
:class:`GovernanceBundle`: the constrained-LLM gate has different
operating parameters (allowed tools, co-attestor rules) and the
adversarial cohort exercises different paths (forbidden tool,
backend error, malformed response).  Mixing them into a single
bundle conflates two governance surfaces with different SLOs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from knowlytix.harness.governance.bundle_signing import (
    SigningMethod, resolve_signing_key, sign, verify,
)
from knowlytix.harness.governance.governance_bundle import BundleSignatureError
from knowlytix.harness.governance.store_fingerprint import StoreFingerprint


LLM_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LLMOperatingPoint:
    """SLO for the constrained-LLM gate.

    ``adversarial_floor`` is the minimum safety rate the bundle's
    cohort sweep must achieve at calibration time.  At runtime the
    operator can rerun the sweep against a candidate bundle; if the
    rate dips below the floor, the candidate is rejected.

    ``max_backend_error_rate`` caps the rate at which the runtime
    surface backend errors over a moving window.  An exceeded cap
    means the LLM provider is unhealthy enough that the gate is
    routinely rejecting on degraded-mode rather than on real
    governance signals --- a reason to fail over to a different
    backend.
    """

    adversarial_floor: float = 1.0
    max_backend_error_rate: float = 0.05


@dataclass
class ConstrainedLLMBundle:
    """Frozen artifact bundle for a constrained-LLM deployment."""

    schema_version: str
    bundle_id: str
    created_at: str
    source_store_path: str

    # Calibrated thresholds the runtime verifier will use.
    min_plausibility: float | None
    tau_contra: float | None
    tau_ent: float | None
    tau_path: float | None
    tension_status: str  # "calibrated" | "degenerate"

    # Store fingerprint --- the runtime refuses to bind if the
    # store has drifted off this fingerprint.
    store_fingerprint: StoreFingerprint

    # Allow-list and co-attestor rules.  Stored as plain types so
    # the bundle remains JSON-canonicalizable for signing.
    allowed_tools: tuple[str, ...]
    forbidden_relations: tuple[str, ...]
    numeric_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    forbidden_tool_args: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # SLO and the measurement that produced it.
    operating_point: LLMOperatingPoint = field(
        default_factory=LLMOperatingPoint
    )
    adversarial_safety_rate: float = 0.0
    adversarial_n_probes: int = 0

    # Provenance for audit.
    backend_model_name: str = ""
    calibration_seed: int = 42

    signature: str = ""
    signing_method: SigningMethod = "sha256"

    # ---- canonicalisation ------------------------------------------- #

    def canonical_payload(self) -> bytes:
        """Bytes used for signing.  Excludes the signature itself."""
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_store_path": self.source_store_path,
            "min_plausibility": self.min_plausibility,
            "tau_contra": self.tau_contra,
            "tau_ent": self.tau_ent,
            "tau_path": self.tau_path,
            "tension_status": self.tension_status,
            "store_fingerprint": asdict(self.store_fingerprint),
            "allowed_tools": list(self.allowed_tools),
            "forbidden_relations": list(self.forbidden_relations),
            "numeric_bounds": {
                k: list(v)
                for k, v in sorted(self.numeric_bounds.items())
            },
            "forbidden_tool_args": {
                k: sorted(v)
                for k, v in sorted(self.forbidden_tool_args.items())
            },
            "operating_point": asdict(self.operating_point),
            "adversarial_safety_rate": self.adversarial_safety_rate,
            "adversarial_n_probes": self.adversarial_n_probes,
            "backend_model_name": self.backend_model_name,
            "calibration_seed": self.calibration_seed,
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    # ---- sign / verify --------------------------------------------- #

    def freeze(self, key: bytes | str | None = None) -> "ConstrainedLLMBundle":
        """Compute and stamp the signature.  Call once after assembly."""
        resolved = resolve_signing_key(key)
        self.signing_method = (
            "hmac-sha256" if resolved is not None else "sha256"
        )
        self.signature, _ = sign(self.canonical_payload(), key=resolved)
        return self

    def verify_signature(self, key: bytes | str | None = None) -> bool:
        resolved = resolve_signing_key(key)
        return verify(
            self.canonical_payload(),
            self.signature,
            self.signing_method,
            key=resolved,
        )

    # ---- store-binding gate ---------------------------------------- #

    def assert_store_matches(self, store) -> None:
        """Refuse to bind if the live store has drifted from the bundle."""
        from knowlytix.harness.governance.store_fingerprint import compute_store_fingerprint
        live = compute_store_fingerprint(store)
        if not self.store_fingerprint.matches(live):
            diff = self.store_fingerprint.diff(live)
            raise BundleSignatureError(
                f"Store fingerprint mismatch: bundle was calibrated "
                f"against a different store. Differing components: "
                f"{sorted(diff.keys())}.  Recalibrate before binding."
            )

    def assert_meets_slo(self) -> None:
        """Refuse to bind if measured safety < SLO floor."""
        if self.adversarial_safety_rate < self.operating_point.adversarial_floor:
            raise BundleSignatureError(
                f"Bundle's adversarial_safety_rate "
                f"({self.adversarial_safety_rate:.3f}) is below the "
                f"declared SLO floor "
                f"({self.operating_point.adversarial_floor:.3f}). "
                f"Refuse to deploy."
            )

    # ---- persistence ----------------------------------------------- #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        key: bytes | str | None = None,
    ) -> "ConstrainedLLMBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: ConstrainedLLMBundle = pickle.load(f)
        if not isinstance(bundle, cls):
            raise TypeError(
                f"Loaded object is {type(bundle).__name__}, "
                f"expected {cls.__name__}"
            )
        try:
            ok = bundle.verify_signature(key=key)
        except BundleSigningError as exc:
            raise BundleSignatureError(
                f"Refusing to load bundle at {path}: {exc}"
            ) from exc
        if not ok:
            raise BundleSignatureError(
                f"Signature mismatch on bundle at {path}: "
                f"signing_method={bundle.signing_method}.  Tampered "
                f"or corrupted bundle, or wrong signing key."
            )
        return bundle


# -------------------------------------------------------------------- #
# Bundle builder                                                       #
# -------------------------------------------------------------------- #


def make_llm_bundle_id(source_store_path: str, created_at: str) -> str:
    raw = f"{source_store_path}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_constrained_llm_bundle(
    *,
    store,
    verify_config,
    allowed_tools: list[str],
    co_attestor,
    adversarial_safety_rate: float,
    adversarial_n_probes: int,
    backend_model_name: str,
    operating_point: LLMOperatingPoint | None = None,
    calibration_seed: int = 42,
    sign_key: bytes | str | None = None,
) -> ConstrainedLLMBundle:
    """Assemble and sign a :class:`ConstrainedLLMBundle`.

    All inputs come from the offline calibration pass --- thresholds
    from ``GMSJudge.calibrate``, fingerprint from the store at the
    moment calibration completed, and the safety rate from
    :func:`sweep_llm_adversarial`.
    """
    from knowlytix.harness.governance.store_fingerprint import compute_store_fingerprint
    fingerprint = compute_store_fingerprint(store)

    created = utcnow_iso()
    bundle_id = make_llm_bundle_id(store.config.store_path, created)

    tension_status = (
        "calibrated"
        if (verify_config.tau_contra is not None
            and verify_config.tau_ent is not None)
        else "degenerate"
    )

    bundle = ConstrainedLLMBundle(
        schema_version=LLM_BUNDLE_SCHEMA_VERSION,
        bundle_id=bundle_id,
        created_at=created,
        source_store_path=store.config.store_path,
        min_plausibility=verify_config.min_plausibility,
        tau_contra=verify_config.tau_contra,
        tau_ent=verify_config.tau_ent,
        tau_path=verify_config.tau_path,
        tension_status=tension_status,
        store_fingerprint=fingerprint,
        allowed_tools=tuple(allowed_tools),
        forbidden_relations=tuple(sorted(co_attestor.forbidden_relations)),
        numeric_bounds=dict(co_attestor.numeric_bounds),
        forbidden_tool_args={
            k: tuple(sorted(v))
            for k, v in co_attestor.forbidden_tool_args.items()
        },
        operating_point=operating_point or LLMOperatingPoint(),
        adversarial_safety_rate=adversarial_safety_rate,
        adversarial_n_probes=adversarial_n_probes,
        backend_model_name=backend_model_name,
        calibration_seed=calibration_seed,
    )
    return bundle.freeze(key=sign_key)


def bind_constrained_llm_from_bundle(
    bundle: ConstrainedLLMBundle,
    *,
    store,
    llm_backend,
):
    """Construct a runtime :class:`ConstrainedLLM` from a frozen bundle.

    The function refuses to bind when:

    * the bundle's signature does not verify (raised at load, not
      here --- this function trusts that ``bundle`` was produced by
      :meth:`ConstrainedLLMBundle.load`);
    * the live store fingerprint differs from the bundle's;
    * the bundle's measured safety rate is below the declared SLO
      floor.
    """
    bundle.assert_store_matches(store)
    bundle.assert_meets_slo()

    from knowlytix.knowledge.config import VerifyConfig
    from knowlytix.harness.governance.governed_llm import (
        ConstrainedLLM, LogicVerifier,
    )
    from knowlytix.harness.governance.llm_co_attestor import LLMCoAttestor

    verify_cfg = VerifyConfig(
        min_plausibility=bundle.min_plausibility,
        tau_contra=bundle.tau_contra,
        tau_ent=bundle.tau_ent,
        tau_path=bundle.tau_path,
    )
    verifier = LogicVerifier(store, verify_config=verify_cfg)

    co_attestor = LLMCoAttestor(
        forbidden_relations=frozenset(bundle.forbidden_relations),
        numeric_bounds=dict(bundle.numeric_bounds),
        forbidden_tool_args={
            k: frozenset(v)
            for k, v in bundle.forbidden_tool_args.items()
        },
    )

    return ConstrainedLLM(
        store, verifier, llm=llm_backend,
        allowed_tools=list(bundle.allowed_tools),
        co_attestor=co_attestor,
    )
