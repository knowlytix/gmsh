# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed artifact bundle for a behavioral contract.

A ``BehavioralContractBundle`` carries everything a runtime needs to
issue auditable contract verdicts without recomputing calibration:

  * a fingerprint of the FSM (states, transitions, tool permissions,
    invariant names, budget, human-approval flag) so the runtime
    refuses to bind when the live contract has drifted off the
    fingerprint;
  * the cohort hash of the systematic adversarial cohort that
    produced the measured detection rate, so a future cohort change
    invalidates the binding;
  * the holonomy co-attestor's calibrated threshold and reference
    sequence;
  * the operating point: minimum detection rate, maximum
    false-positive rate, both enforced at bind time;
  * an HMAC-SHA-256 signature over the canonical JSON payload.

The bundle is the contract between offline calibration (run the
sweep, calibrate the holonomy threshold, record the rates) and the
runtime contract (load the bundle, refuse to bind on fingerprint or
SLO failure).  Mirrors the Ch.10 :class:`GovernanceBundle` and the
Ch.11 :class:`ConstrainedLLMBundle`; the patterns are deliberately
the same so an operator can reason about all three artifacts the
same way.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from knowlytix.harness.governance.behaviors.contracts import (
    BehavioralContract,
)
from knowlytix.harness.governance.bundle_signing import (
    SigningMethod, resolve_signing_key, sign, verify,
)
from knowlytix.harness.governance.governance_bundle import BundleSignatureError


CONTRACT_BUNDLE_SCHEMA_VERSION = "1.0"


# -------------------------------------------------------------------- #
# Contract fingerprint                                                 #
# -------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContractFingerprint:
    """Component-wise digest of a contract's identity at calibration time.

    Each component is reported separately so a fingerprint mismatch
    points to *what* changed (transitions vs permissions vs
    invariants), not just *that* something changed.
    """

    states_hash: str
    transitions_hash: str
    permissions_hash: str
    invariants_hash: str
    budget: int
    require_human_approval: bool

    def matches(self, other: "ContractFingerprint") -> bool:
        return (
            self.states_hash == other.states_hash
            and self.transitions_hash == other.transitions_hash
            and self.permissions_hash == other.permissions_hash
            and self.invariants_hash == other.invariants_hash
            and self.budget == other.budget
            and self.require_human_approval == other.require_human_approval
        )

    def diff(self, other: "ContractFingerprint") -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for f in ("states_hash", "transitions_hash",
                  "permissions_hash", "invariants_hash"):
            mine = getattr(self, f)
            theirs = getattr(other, f)
            if mine != theirs:
                out[f] = (mine[:16], theirs[:16])
        if self.budget != other.budget:
            out["budget"] = (self.budget, other.budget)
        if self.require_human_approval != other.require_human_approval:
            out["require_human_approval"] = (
                self.require_human_approval,
                other.require_human_approval,
            )
        return out


def compute_contract_fingerprint(
    contract: BehavioralContract,
) -> ContractFingerprint:
    """Hash each component of a contract independently."""
    def _hash(items) -> str:
        h = hashlib.sha256()
        for item in items:
            h.update(str(item).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    states_canonical = sorted(s.value for s in contract.states)
    transitions_canonical = sorted(
        f"{src.value}->{dst.value}"
        for src, dsts in contract.transitions.items()
        for dst in sorted(dsts, key=lambda s: s.value)
    )
    permissions_canonical = sorted(
        f"{src.value}:{','.join(sorted(tools))}"
        for src, tools in contract.tool_permissions.items()
    )
    invariants_canonical = sorted(
        inv.__name__ for inv in contract.invariants
    )

    return ContractFingerprint(
        states_hash=_hash(states_canonical),
        transitions_hash=_hash(transitions_canonical),
        permissions_hash=_hash(permissions_canonical),
        invariants_hash=_hash(invariants_canonical),
        budget=contract.max_tool_calls,
        require_human_approval=contract.require_human_approval,
    )


# -------------------------------------------------------------------- #
# Operating point                                                      #
# -------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContractOperatingPoint:
    """SLO for a runtime-bound behavioral contract.

    ``min_detection_rate`` is the floor below which the bundle's
    measurement is considered too weak for deployment.  At runtime
    the sweep can be rerun against a candidate bundle; if the rate
    dips below the floor the candidate is rejected.

    ``max_false_positive_rate`` is the ceiling.  An FPR above this
    means the contract is rejecting too many compliant sequences ---
    a problem either with the contract or the cohort.
    """

    min_detection_rate: float = 1.0
    max_false_positive_rate: float = 0.0


# -------------------------------------------------------------------- #
# Bundle                                                               #
# -------------------------------------------------------------------- #


@dataclass
class BehavioralContractBundle:
    """Frozen artifact bundle for a behavioral contract."""

    schema_version: str
    bundle_id: str
    created_at: str
    contract_name: str

    fingerprint: ContractFingerprint
    cohort_hash: str

    # Co-attestor calibration.
    holonomy_threshold: float
    holonomy_reference_sequence: tuple[str, ...]
    holonomy_calibration_percentile: float

    # Measured rates from the sweep.
    detection_rate: float
    false_positive_rate: float
    safety_rate: float
    n_probes: int

    operating_point: ContractOperatingPoint = field(
        default_factory=ContractOperatingPoint
    )

    signature: str = ""
    signing_method: SigningMethod = "sha256"

    # ---- canonicalisation ---- #

    def canonical_payload(self) -> bytes:
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "contract_name": self.contract_name,
            "fingerprint": asdict(self.fingerprint),
            "cohort_hash": self.cohort_hash,
            "holonomy_threshold": self.holonomy_threshold,
            "holonomy_reference_sequence": list(
                self.holonomy_reference_sequence
            ),
            "holonomy_calibration_percentile": (
                self.holonomy_calibration_percentile
            ),
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "safety_rate": self.safety_rate,
            "n_probes": self.n_probes,
            "operating_point": asdict(self.operating_point),
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    # ---- sign / verify ---- #

    def freeze(
        self, key: bytes | str | None = None,
    ) -> "BehavioralContractBundle":
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

    # ---- bind-time refusal paths ---- #

    def assert_contract_matches(
        self, contract: BehavioralContract,
    ) -> None:
        live = compute_contract_fingerprint(contract)
        if not self.fingerprint.matches(live):
            diff = self.fingerprint.diff(live)
            raise BundleSignatureError(
                f"Contract fingerprint mismatch: bundle was calibrated "
                f"against a different contract. Differing components: "
                f"{sorted(diff.keys())}.  Recalibrate before binding."
            )

    def assert_meets_slo(self) -> None:
        op = self.operating_point
        if self.detection_rate < op.min_detection_rate:
            raise BundleSignatureError(
                f"Bundle's detection_rate ({self.detection_rate:.3f}) is "
                f"below the SLO floor ({op.min_detection_rate:.3f}). "
                f"Refuse to deploy."
            )
        if self.false_positive_rate > op.max_false_positive_rate:
            raise BundleSignatureError(
                f"Bundle's false_positive_rate "
                f"({self.false_positive_rate:.3f}) exceeds the SLO ceiling "
                f"({op.max_false_positive_rate:.3f}). Refuse to deploy."
            )

    # ---- persistence ---- #

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
    ) -> "BehavioralContractBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: BehavioralContractBundle = pickle.load(f)
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
# Builders                                                             #
# -------------------------------------------------------------------- #


def make_contract_bundle_id(contract_name: str, created_at: str) -> str:
    raw = f"{contract_name}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_behavioral_contract_bundle(
    *,
    contract: BehavioralContract,
    cohort,
    sweep_report,
    holonomy_attestor,
    holonomy_calibration_percentile: float,
    operating_point: ContractOperatingPoint | None = None,
    sign_key: bytes | str | None = None,
) -> BehavioralContractBundle:
    """Assemble and HMAC-sign a :class:`BehavioralContractBundle`.

    All inputs come from the offline calibration pass --- the sweep
    report from :func:`sweep_contract_adversarial`, the holonomy
    attestor after :meth:`HolonomyCoAttestor.calibrate_threshold`,
    and the contract itself for the fingerprint.
    """
    from knowlytix.harness.governance.behaviors.contract_adversarial import cohort_hash

    fingerprint = compute_contract_fingerprint(contract)
    created = utcnow_iso()
    bundle_id = make_contract_bundle_id(contract.name, created)

    bundle = BehavioralContractBundle(
        schema_version=CONTRACT_BUNDLE_SCHEMA_VERSION,
        bundle_id=bundle_id,
        created_at=created,
        contract_name=contract.name,
        fingerprint=fingerprint,
        cohort_hash=cohort_hash(cohort),
        holonomy_threshold=holonomy_attestor.threshold,
        holonomy_reference_sequence=tuple(
            holonomy_attestor.reference_sequence
        ),
        holonomy_calibration_percentile=holonomy_calibration_percentile,
        detection_rate=sweep_report.detection_rate,
        false_positive_rate=sweep_report.false_positive_rate,
        safety_rate=sweep_report.safety_rate,
        n_probes=sweep_report.n_total,
        operating_point=operating_point or ContractOperatingPoint(),
    )
    return bundle.freeze(key=sign_key)


def bind_contract_from_bundle(
    bundle: BehavioralContractBundle,
    *,
    contract: BehavioralContract,
):
    """Construct a runtime two-attestor evaluator from a bundle.

    Returns a :class:`HolonomyCoAttestor` configured with the
    bundle's calibrated threshold and reference, alongside the
    live contract used to instantiate the FSM evaluator.  The
    function refuses to bind when:

    * the live contract's fingerprint differs from the bundle's;
    * the bundle's measured detection rate is below the SLO floor;
    * the bundle's false-positive rate exceeds the SLO ceiling.
    """
    bundle.assert_contract_matches(contract)
    bundle.assert_meets_slo()

    from knowlytix.harness.governance.behaviors.contract_co_attestor import (
        HolonomyCoAttestor,
    )

    return HolonomyCoAttestor(
        reference_sequence=bundle.holonomy_reference_sequence,
        threshold=bundle.holonomy_threshold,
    )
