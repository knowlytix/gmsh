# SPDX-License-Identifier: Apache-2.0
"""HMAC-signed compliance bundle.

Frozen artefact for one compliance evidence package. Carries:

  * the snapshot fingerprint (over the deterministic check list)
  * the overall status + pass rate
  * per-regulation pass rates
  * alt-validator agreement
  * HMAC-SHA-256 signature
  * multi-signer attestor signatures

`bind_compliance_from_bundle` refuses to bind a live snapshot
that diverges from the bundle's recorded values across:

1. snapshot fingerprint
2. overall_status (must match accepted set)
3. pass_rate (must meet floor)
4. alt-validator agreement (when required)
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from knowlytix.harness.governance.bundle_signing import (
    SigningMethod,
    resolve_signing_key,
    sign,
    verify,
)
from knowlytix.harness.governance.compliance_hardening.compliance_snapshot import (
    ComplianceSnapshot,
)
from knowlytix.harness.governance.governance_bundle import BundleSignatureError

COMPLIANCE_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ComplianceOperatingPoint:
    """SLO for a compliance bundle at bind time."""

    min_pass_rate: float = 0.80
    accepted_overall_statuses: tuple[str, ...] = ("COMPLIANT", "PARTIAL")
    require_alt_agreement: bool = True


@dataclass
class ComplianceBundle:
    """Frozen artifact bundle for one compliance evidence package."""

    schema_version: str
    bundle_id: str
    created_at: str
    suite_id: str

    snapshot_fingerprint: str
    overall_status: str
    pass_rate: float
    n_checks: int
    per_regulation_pass_rate: dict[str, float]
    alt_agreement_rate: float

    operating_point: ComplianceOperatingPoint = field(
        default_factory=ComplianceOperatingPoint,
    )

    signature: str = ""
    signing_method: SigningMethod = "sha256"
    attestor_signatures: list[dict] = field(default_factory=list)

    def canonical_payload(self) -> bytes:
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "suite_id": self.suite_id,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "overall_status": self.overall_status,
            "pass_rate": self.pass_rate,
            "n_checks": self.n_checks,
            "per_regulation_pass_rate": dict(
                sorted(self.per_regulation_pass_rate.items())
            ),
            "alt_agreement_rate": self.alt_agreement_rate,
            "operating_point": {
                "min_pass_rate": self.operating_point.min_pass_rate,
                "accepted_overall_statuses": list(
                    sorted(self.operating_point.accepted_overall_statuses)
                ),
                "require_alt_agreement":
                    self.operating_point.require_alt_agreement,
            },
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def freeze(self, key: bytes | str | None = None) -> "ComplianceBundle":
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

    def assert_snapshot_matches(self, snapshot: ComplianceSnapshot) -> None:
        live = snapshot.fingerprint()
        if live != self.snapshot_fingerprint:
            raise BundleSignatureError(
                f"ComplianceSnapshot fingerprint mismatch: bundle "
                f"declares {self.snapshot_fingerprint[:12]}... but "
                f"live snapshot is {live[:12]}..."
            )

    def assert_meets_slo(self) -> None:
        op = self.operating_point
        if self.pass_rate < op.min_pass_rate:
            raise BundleSignatureError(
                f"pass_rate {self.pass_rate:.4f} below "
                f"min_pass_rate {op.min_pass_rate:.4f}"
            )
        if self.overall_status not in op.accepted_overall_statuses:
            raise BundleSignatureError(
                f"overall_status {self.overall_status!r} not in "
                f"accepted set {sorted(op.accepted_overall_statuses)}"
            )
        if op.require_alt_agreement and self.alt_agreement_rate < 1.0:
            raise BundleSignatureError(
                f"alt_agreement_rate {self.alt_agreement_rate:.4f} "
                f"< 1.0; primary and alt disagree"
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, path: str | Path, *, key: bytes | str | None = None,
    ) -> "ComplianceBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: ComplianceBundle = pickle.load(f)
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


def make_compliance_bundle_id(suite_id: str, created_at: str) -> str:
    raw = f"{suite_id}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_compliance_bundle(
    *,
    snapshot: ComplianceSnapshot,
    alt_agreement_rate: float,
    operating_point: ComplianceOperatingPoint | None = None,
    sign_key: bytes | str | None = None,
) -> ComplianceBundle:
    """Assemble and HMAC-sign a `ComplianceBundle`."""
    created = utcnow_iso()
    bundle = ComplianceBundle(
        schema_version=COMPLIANCE_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_compliance_bundle_id(snapshot.suite_id, created),
        created_at=created,
        suite_id=snapshot.suite_id,
        snapshot_fingerprint=snapshot.fingerprint(),
        overall_status=snapshot.overall_status,
        pass_rate=snapshot.pass_rate,
        n_checks=snapshot.n_checks,
        per_regulation_pass_rate=dict(snapshot.per_regulation_pass_rate),
        alt_agreement_rate=alt_agreement_rate,
        operating_point=operating_point or ComplianceOperatingPoint(),
    )
    return bundle.freeze(key=sign_key)


def bind_compliance_from_bundle(
    bundle: ComplianceBundle, *, snapshot: ComplianceSnapshot,
) -> ComplianceBundle:
    """Bind a bundle to a live snapshot; raise on drift or SLO breach."""
    bundle.assert_snapshot_matches(snapshot)
    bundle.assert_meets_slo()
    return bundle
