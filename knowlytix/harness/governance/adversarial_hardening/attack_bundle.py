# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed artifact for an adversarial-suite run.

An `AttackBundle` ships the canonical defense outcome alongside
its hardening evidence: snapshot fingerprint, defense rate,
cross-layer match rate, alt-validator agreement, attestor
signatures.

Mirrors :class:`PolicyBundle` / :class:`AuditTrailBundle` /
:class:`WorkflowBundle` in shape so an operator reasons about
all four artefacts the same way.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from knowlytix.harness.governance.bundle_signing import (
    SigningMethod, resolve_signing_key, sign, verify,
)
from knowlytix.harness.governance.governance_bundle import BundleSignatureError
from knowlytix.harness.governance.adversarial_hardening.attack_snapshot import (
    AttackSnapshot,
)


ATTACK_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AttackOperatingPoint:
    """SLO for an adversarial bundle at bind time."""

    min_defense_rate: float = 1.0
    min_layer_match_rate: float = 1.0
    require_alt_agreement: bool = True


@dataclass
class AttackBundle:
    """Frozen artifact bundle for one adversarial-suite run."""

    schema_version: str
    bundle_id: str
    created_at: str
    suite_id: str

    snapshot_fingerprint: str
    defense_rate: float
    n_attacks: int
    layer_match_rate: float
    alt_agreement_rate: float

    operating_point: AttackOperatingPoint = field(
        default_factory=AttackOperatingPoint,
    )

    signature: str = ""
    signing_method: SigningMethod = "sha256"
    attestor_signatures: list[dict] = field(default_factory=list)

    # ---- canonicalisation ---- #

    def canonical_payload(self) -> bytes:
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "suite_id": self.suite_id,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "defense_rate": self.defense_rate,
            "n_attacks": self.n_attacks,
            "layer_match_rate": self.layer_match_rate,
            "alt_agreement_rate": self.alt_agreement_rate,
            "operating_point": {
                "min_defense_rate":
                    self.operating_point.min_defense_rate,
                "min_layer_match_rate":
                    self.operating_point.min_layer_match_rate,
                "require_alt_agreement":
                    self.operating_point.require_alt_agreement,
            },
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    # ---- sign / verify ---- #

    def freeze(self, key: bytes | str | None = None) -> "AttackBundle":
        resolved = resolve_signing_key(key)
        self.signing_method = (
            "hmac-sha256" if resolved is not None else "sha256"
        )
        self.signature, _ = sign(self.canonical_payload(), key=resolved)
        return self

    def verify_signature(self, key: bytes | str | None = None) -> bool:
        resolved = resolve_signing_key(key)
        return verify(
            self.canonical_payload(), self.signature,
            self.signing_method, key=resolved,
        )

    # ---- bind-time refusal paths ---- #

    def assert_snapshot_matches(self, snapshot: AttackSnapshot) -> None:
        live = snapshot.fingerprint()
        if live != self.snapshot_fingerprint:
            raise BundleSignatureError(
                f"AttackSnapshot fingerprint mismatch: bundle "
                f"declares {self.snapshot_fingerprint[:12]}... but "
                f"live snapshot is {live[:12]}..."
            )

    def assert_meets_slo(self) -> None:
        op = self.operating_point
        if self.defense_rate < op.min_defense_rate:
            raise BundleSignatureError(
                f"defense_rate {self.defense_rate:.4f} below "
                f"floor {op.min_defense_rate:.4f}"
            )
        if self.layer_match_rate < op.min_layer_match_rate:
            raise BundleSignatureError(
                f"layer_match_rate {self.layer_match_rate:.4f} below "
                f"floor {op.min_layer_match_rate:.4f}"
            )
        if op.require_alt_agreement and self.alt_agreement_rate < 1.0:
            raise BundleSignatureError(
                f"alt-validator agreement {self.alt_agreement_rate:.4f} "
                f"< 1.0; primary and alt disagree on at least one attack"
            )

    # ---- persistence ---- #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, path: str | Path, *, key: bytes | str | None = None,
    ) -> "AttackBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: AttackBundle = pickle.load(f)
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
                f"Signature mismatch on bundle at {path}"
            )
        return bundle


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def make_attack_bundle_id(suite_id: str, created_at: str) -> str:
    raw = f"{suite_id}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_attack_bundle(
    *,
    snapshot: AttackSnapshot,
    layer_match_rate: float,
    alt_agreement_rate: float,
    operating_point: AttackOperatingPoint | None = None,
    sign_key: bytes | str | None = None,
) -> AttackBundle:
    """Assemble and HMAC-sign an `AttackBundle`."""
    created = utcnow_iso()
    bundle = AttackBundle(
        schema_version=ATTACK_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_attack_bundle_id(snapshot.suite_id, created),
        created_at=created,
        suite_id=snapshot.suite_id,
        snapshot_fingerprint=snapshot.fingerprint(),
        defense_rate=snapshot.defense_rate,
        n_attacks=snapshot.n_attacks,
        layer_match_rate=layer_match_rate,
        alt_agreement_rate=alt_agreement_rate,
        operating_point=operating_point or AttackOperatingPoint(),
    )
    return bundle.freeze(key=sign_key)


def bind_attack_from_bundle(
    bundle: AttackBundle, *, snapshot: AttackSnapshot,
) -> AttackBundle:
    """Bind a bundle to a live snapshot; raise on drift or SLO breach."""
    bundle.assert_snapshot_matches(snapshot)
    bundle.assert_meets_slo()
    return bundle
