# SPDX-License-Identifier: Apache-2.0
"""k-of-n multi-signer bundle signing for policy bundles.

A high-stakes deployment never ships a bundle that a single
operator can sign alone. This module formalises the multi-party
signing workflow that real banks build on top of HSM-backed keys
or threshold-cryptography schemes.

The in-repo realisation is the workflow shape, not the cryptographic
primitive: each attestor signs the same canonical payload with
their own HMAC-SHA-256 key, and verification requires at least
``k`` of ``n`` signatures to validate. A deployment that wants
threshold-crypto (Shamir, BLS) replaces the HMAC primitive while
keeping the workflow API constant.

Workflow:

  1. Build the bundle (unsigned).
  2. Each of ``n`` deployment attestors calls
     ``sign_for(bundle, signer_id, key)`` with their own HMAC key.
  3. The bundle accumulates ``signatures: [(signer_id, sig), ...]``.
  4. ``verify_quorum(bundle, key_registry, k=2)`` returns True iff
     at least ``k`` signatures verify against the registered keys.
  5. ``runtime_load`` refuses to bind to a bundle whose verified
     signatures fall below ``k``.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from knowlytix.harness.governance.policy_bundle import PolicyBundle


@dataclass
class AttestorSignature:
    """One signer's signature on the bundle's canonical payload."""

    signer_id: str
    method: str       # "hmac-sha256" today; "shamir" / "bls" in real deployment
    signature: str

    def to_dict(self) -> dict:
        return {
            "signer_id": self.signer_id,
            "method": self.method,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AttestorSignature":
        return cls(**d)


def _hmac_hex(payload: bytes, key: bytes) -> str:
    if len(key) < 16:
        raise ValueError(
            f"signer key too short ({len(key)} bytes); use ≥ 16."
        )
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------


def sign_for(
    bundle: "PolicyBundle",
    signer_id: str,
    key: bytes | str,
) -> AttestorSignature:
    """Produce one attestor's signature for the bundle.

    The payload signed is the bundle's canonical_payload(), the same
    payload its single-signature scheme uses; this lets a deployment
    run both schemes side by side and roll over without rebuilding.
    """
    if isinstance(key, str):
        key = key.encode("utf-8")
    sig = _hmac_hex(bundle.canonical_payload(), key)
    return AttestorSignature(
        signer_id=signer_id, method="hmac-sha256", signature=sig,
    )


def attach_signatures(
    bundle: "PolicyBundle",
    signatures: Iterable[AttestorSignature],
) -> None:
    """Store an iterable of signatures on the bundle in canonical order."""
    sigs = sorted(signatures, key=lambda s: s.signer_id)
    bundle.attestor_signatures = [s.to_dict() for s in sigs]


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


@dataclass
class QuorumVerifyResult:
    n_signatures: int
    n_verified: int
    n_required: int
    verified_signers: list[str]
    rejected_signers: list[str]
    detail: str

    @property
    def quorum_met(self) -> bool:
        return self.n_verified >= self.n_required

    def summary(self) -> str:
        return (
            f"QuorumVerifyResult(n_sigs={self.n_signatures}, "
            f"verified={self.n_verified}/{self.n_required}, "
            f"signers={self.verified_signers}, "
            f"rejected={self.rejected_signers}, "
            f"quorum_met={self.quorum_met})"
        )


def verify_quorum(
    bundle: "PolicyBundle",
    key_registry: dict[str, bytes | str],
    *,
    k: int = 2,
) -> QuorumVerifyResult:
    """Verify that at least ``k`` of ``n`` attestor signatures validate.

    ``key_registry`` maps signer_id -> key (bytes or string). Each
    signature is checked against the registered key for its signer;
    signatures from unknown signers or with non-matching keys are
    rejected.
    """
    sigs_raw = getattr(bundle, "attestor_signatures", []) or []
    payload = bundle.canonical_payload()

    verified: list[str] = []
    rejected: list[str] = []

    seen_signers: set[str] = set()
    for raw in sigs_raw:
        sig = AttestorSignature.from_dict(raw)
        if sig.signer_id in seen_signers:
            rejected.append(f"{sig.signer_id}(duplicate)")
            continue
        seen_signers.add(sig.signer_id)

        key = key_registry.get(sig.signer_id)
        if key is None:
            rejected.append(f"{sig.signer_id}(unregistered)")
            continue
        if isinstance(key, str):
            key = key.encode("utf-8")
        try:
            expected = _hmac_hex(payload, key)
        except ValueError as exc:
            rejected.append(f"{sig.signer_id}({exc})")
            continue
        if hmac.compare_digest(sig.signature, expected):
            verified.append(sig.signer_id)
        else:
            rejected.append(f"{sig.signer_id}(bad-sig)")

    detail = (
        f"verified={verified}; rejected={rejected}; "
        f"need ≥{k} signatures"
    )
    return QuorumVerifyResult(
        n_signatures=len(sigs_raw),
        n_verified=len(verified),
        n_required=k,
        verified_signers=verified,
        rejected_signers=rejected,
        detail=detail,
    )


class QuorumSignatureError(RuntimeError):
    """Raised when a bundle fails k-of-n quorum verification."""


def assert_quorum(
    bundle: "PolicyBundle",
    key_registry: dict[str, bytes | str],
    *,
    k: int = 2,
) -> None:
    """Strict gate: raise if quorum is not met."""
    result = verify_quorum(bundle, key_registry, k=k)
    if not result.quorum_met:
        raise QuorumSignatureError(
            f"bundle quorum not met: {result.summary()}"
        )
