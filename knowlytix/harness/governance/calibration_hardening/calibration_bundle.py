# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed artefact for a calibrated threshold configuration.

A `CalibrationBundle` ships the calibrated optimum + supporting
evidence (snapshot fingerprint, n_configs, alt-validator
agreement, optional cross-domain comparison) under HMAC-SHA-256
signing.

`bind_calibration_from_bundle` refuses to bind to a live
calibration that:

  * has a different snapshot fingerprint;
  * has a different optimum threshold vector;
  * has accuracy below the SLO floor;
  * has alt-validator disagreement (when required by the
    operating point).

Mirrors :class:`PolicyBundle`, :class:`AuditTrailBundle`,
:class:`WorkflowBundle`, :class:`AttackBundle`.
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
from knowlytix.harness.governance.calibration_hardening.calibration_snapshot import (
    CalibrationSnapshot,
)
from knowlytix.harness.governance.governance_bundle import BundleSignatureError

CALIBRATION_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CalibrationOperatingPoint:
    """SLO for a calibration bundle at bind time."""

    min_accuracy: float = 0.85
    require_alt_agreement: bool = True


@dataclass
class CalibrationBundle:
    """Frozen artifact bundle for one calibration run."""

    schema_version: str
    bundle_id: str
    created_at: str
    suite_id: str

    domain: str
    snapshot_fingerprint: str
    n_configs: int
    optimum_theta_sim: float
    optimum_theta_margin: float
    optimum_theta_esc: float
    optimum_d_max: int
    optimum_budget: float
    optimum_accuracy: float
    alt_agreement_rate: float

    operating_point: CalibrationOperatingPoint = field(
        default_factory=CalibrationOperatingPoint,
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
            "domain": self.domain,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "n_configs": self.n_configs,
            "optimum_theta_sim": self.optimum_theta_sim,
            "optimum_theta_margin": self.optimum_theta_margin,
            "optimum_theta_esc": self.optimum_theta_esc,
            "optimum_d_max": self.optimum_d_max,
            "optimum_budget": self.optimum_budget,
            "optimum_accuracy": self.optimum_accuracy,
            "alt_agreement_rate": self.alt_agreement_rate,
            "operating_point": {
                "min_accuracy": self.operating_point.min_accuracy,
                "require_alt_agreement":
                    self.operating_point.require_alt_agreement,
            },
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    # ---- sign / verify ---- #

    def freeze(
        self, key: bytes | str | None = None,
    ) -> "CalibrationBundle":
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

    def assert_snapshot_matches(self, snapshot: CalibrationSnapshot) -> None:
        live = snapshot.fingerprint()
        if live != self.snapshot_fingerprint:
            raise BundleSignatureError(
                f"CalibrationSnapshot fingerprint mismatch: bundle "
                f"declares {self.snapshot_fingerprint[:12]}... but "
                f"live snapshot is {live[:12]}..."
            )
        # Also check optimum threshold vector
        opt = snapshot.optimum
        if (opt.theta_sim != self.optimum_theta_sim
                or opt.theta_margin != self.optimum_theta_margin
                or opt.theta_esc != self.optimum_theta_esc
                or opt.d_max != self.optimum_d_max
                or opt.budget != self.optimum_budget):
            raise BundleSignatureError(
                f"Calibration optimum mismatch: bundle declares "
                f"({self.optimum_theta_sim}, {self.optimum_theta_margin}, "
                f"{self.optimum_theta_esc}, {self.optimum_d_max}, "
                f"{self.optimum_budget}) but live optimum is "
                f"({opt.theta_sim}, {opt.theta_margin}, "
                f"{opt.theta_esc}, {opt.d_max}, {opt.budget})"
            )

    def assert_meets_slo(self) -> None:
        op = self.operating_point
        if self.optimum_accuracy < op.min_accuracy:
            raise BundleSignatureError(
                f"optimum_accuracy {self.optimum_accuracy:.4f} below "
                f"min_accuracy {op.min_accuracy:.4f}"
            )
        if op.require_alt_agreement and self.alt_agreement_rate < 1.0:
            raise BundleSignatureError(
                f"alt_agreement_rate {self.alt_agreement_rate:.4f} "
                f"< 1.0; primary and alt disagree on the optimum"
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
    ) -> "CalibrationBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: CalibrationBundle = pickle.load(f)
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


def make_calibration_bundle_id(suite_id: str, created_at: str) -> str:
    raw = f"{suite_id}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_calibration_bundle(
    *,
    snapshot: CalibrationSnapshot,
    alt_agreement_rate: float,
    operating_point: CalibrationOperatingPoint | None = None,
    sign_key: bytes | str | None = None,
) -> CalibrationBundle:
    """Assemble and HMAC-sign a `CalibrationBundle`."""
    created = utcnow_iso()
    opt = snapshot.optimum
    bundle = CalibrationBundle(
        schema_version=CALIBRATION_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_calibration_bundle_id(snapshot.suite_id, created),
        created_at=created,
        suite_id=snapshot.suite_id,
        domain=snapshot.domain,
        snapshot_fingerprint=snapshot.fingerprint(),
        n_configs=snapshot.n_configs,
        optimum_theta_sim=opt.theta_sim,
        optimum_theta_margin=opt.theta_margin,
        optimum_theta_esc=opt.theta_esc,
        optimum_d_max=opt.d_max,
        optimum_budget=opt.budget,
        optimum_accuracy=opt.accuracy,
        alt_agreement_rate=alt_agreement_rate,
        operating_point=operating_point or CalibrationOperatingPoint(),
    )
    return bundle.freeze(key=sign_key)


def bind_calibration_from_bundle(
    bundle: CalibrationBundle,
    *,
    snapshot: CalibrationSnapshot,
) -> CalibrationBundle:
    """Bind a bundle to a live snapshot; raise on drift or SLO breach."""
    bundle.assert_snapshot_matches(snapshot)
    bundle.assert_meets_slo()
    return bundle
