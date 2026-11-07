# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed artifact bundle for an audit trail.

An ``AuditTrailBundle`` is what an operator ships when an audit
review asks ``what happened during this run?''  It carries:

* the chain-head hash of the sealed audit trail (so any subsequent
  edit to the underlying packets surfaces as a head mismatch);
* the deterministic metric fingerprint from
  :mod:`audit_replay` (so any change to the summary numbers
  surfaces as a fingerprint mismatch);
* the cohort safety rate from :mod:`audit_adversarial` (so a
  trail produced under a known-broken hardening pipeline is
  refused at bind time);
* the operating point: the minimum cohort safety rate and the
  required attestor coverage (``"chain"``, ``"cross"``, or
  ``"both"``);
* an HMAC-SHA-256 signature over the canonical JSON payload.

Mirrors :class:`GovernanceBundle` (Ch.10), :class:`PolicyBundle`
(Ch.12), :class:`BehavioralContractBundle` (Ch.13), and
:class:`ConstrainedLLMBundle` (Ch.11).  Same shape so an operator
can reason about all five artifacts the same way.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from knowlytix.harness.governance.bundle_signing import (
    SigningMethod,
    resolve_signing_key,
    sign,
    verify,
)
from knowlytix.harness.governance.governance_bundle import BundleSignatureError

AUDIT_BUNDLE_SCHEMA_VERSION = "1.0"


# -------------------------------------------------------------------- #
# Operating point                                                      #
# -------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuditOperatingPoint:
    """SLO for a sealed audit trail at bind time.

    ``min_cohort_safety_rate`` is the floor below which the bundle's
    measured adversarial safety rate is considered too weak for
    deployment --- an operator's hardening pipeline that catches
    only 80% of the standard tampers is producing audit logs whose
    integrity claims cannot be defended.

    ``required_attestor_coverage`` constrains which attestors the
    bundle must have run.  ``"chain"`` accepts trails sealed
    without cross-reference (e.g.\ when the runtime did not
    produce an independent decision log); ``"both"`` requires
    both.
    """

    min_cohort_safety_rate: float = 1.0
    required_attestor_coverage: str = "both"  # "chain" | "cross" | "both"


# -------------------------------------------------------------------- #
# Bundle                                                               #
# -------------------------------------------------------------------- #


@dataclass
class AuditTrailBundle:
    """Frozen artifact bundle for one sealed audit trail."""

    schema_version: str
    bundle_id: str
    created_at: str
    run_id: str

    # Chain head and metric fingerprint --- the two cryptographic
    # links to the underlying trail.
    chain_head_hash: str
    metrics_fingerprint: str
    n_packets: int

    # Independent measurements from the hardening pipeline.
    cohort_safety_rate: float
    cohort_n_probes: int
    attestors_run: tuple[str, ...]  # subset of ("chain", "cross")

    operating_point: AuditOperatingPoint = field(
        default_factory=AuditOperatingPoint
    )

    signature: str = ""
    signing_method: SigningMethod = "sha256"

    # Multi-signer (k-of-n) signatures, written by
    # ``policy_multisign.attach_signatures``. The single-signer
    # ``signature`` field remains the integrity-only / single-HMAC
    # path; the multi-signer field is the deployment requirement.
    attestor_signatures: list[dict] = field(default_factory=list)

    # ---- canonicalisation ---- #

    def canonical_payload(self) -> bytes:
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "chain_head_hash": self.chain_head_hash,
            "metrics_fingerprint": self.metrics_fingerprint,
            "n_packets": self.n_packets,
            "cohort_safety_rate": self.cohort_safety_rate,
            "cohort_n_probes": self.cohort_n_probes,
            "attestors_run": sorted(self.attestors_run),
            "operating_point": asdict(self.operating_point),
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    # ---- sign / verify ---- #

    def freeze(self, key: bytes | str | None = None) -> "AuditTrailBundle":
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

    def assert_chain_matches(self, packets) -> None:
        """Refuse if the live trail's head hash differs from the bundle's."""
        if not packets:
            live_head = ""
        else:
            live_head = packets[-1].content_hash
        if live_head != self.chain_head_hash:
            raise BundleSignatureError(
                f"Chain head mismatch: bundle declares "
                f"{self.chain_head_hash[:12]}... but live trail's head is "
                f"{live_head[:12]}...  The trail has been edited or a "
                f"different trail is being bound."
            )

    def assert_metrics_match(self, replay_result) -> None:
        """Refuse if the live metrics fingerprint differs from the bundle's."""
        from knowlytix.harness.governance.tracing_hardening.audit_replay import (
            metrics_fingerprint,
        )
        live = metrics_fingerprint(replay_result)
        if live != self.metrics_fingerprint:
            raise BundleSignatureError(
                f"Metrics fingerprint mismatch: bundle declares "
                f"{self.metrics_fingerprint[:12]}... but live replay "
                f"hashes to {live[:12]}...  Summary metrics have drifted."
            )

    def assert_meets_slo(self) -> None:
        op = self.operating_point
        if self.cohort_safety_rate < op.min_cohort_safety_rate:
            raise BundleSignatureError(
                f"Cohort safety rate {self.cohort_safety_rate:.3f} below "
                f"the SLO floor {op.min_cohort_safety_rate:.3f}.  "
                f"Refuse to deploy."
            )
        if op.required_attestor_coverage == "both":
            missing = {"chain", "cross"} - set(self.attestors_run)
            if missing:
                raise BundleSignatureError(
                    f"Required attestor coverage 'both' but bundle was "
                    f"produced without {sorted(missing)}."
                )
        elif op.required_attestor_coverage in ("chain", "cross"):
            if op.required_attestor_coverage not in self.attestors_run:
                raise BundleSignatureError(
                    f"Required attestor coverage "
                    f"{op.required_attestor_coverage!r} not present "
                    f"in attestors_run={sorted(self.attestors_run)}."
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
    ) -> "AuditTrailBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: AuditTrailBundle = pickle.load(f)
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
# Builder                                                              #
# -------------------------------------------------------------------- #


def make_audit_bundle_id(run_id: str, created_at: str) -> str:
    raw = f"{run_id}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_audit_trail_bundle(
    *,
    run_id: str,
    packets,
    cohort_report,
    attestors_run: tuple[str, ...] = ("chain", "cross"),
    operating_point: AuditOperatingPoint | None = None,
    sign_key: bytes | str | None = None,
) -> AuditTrailBundle:
    """Assemble and HMAC-sign an :class:`AuditTrailBundle`.

    All inputs come from the offline pipeline:
    :func:`seal_audit_trail` produced the chain head;
    :func:`replay_audit_metrics` produced the metrics fingerprint;
    :func:`sweep_audit_adversarial` produced the safety rate.
    """
    from knowlytix.harness.governance.tracing_hardening.audit_replay import (
        metrics_fingerprint,
        replay_audit_metrics,
    )
    from knowlytix.harness.governance.tracing_hardening.hash_chain import (
        chain_head_hash,
    )

    head = chain_head_hash(packets)
    metrics = replay_audit_metrics(run_id, packets)
    metrics_fp = metrics_fingerprint(metrics)

    created = utcnow_iso()
    bundle_id = make_audit_bundle_id(run_id, created)

    bundle = AuditTrailBundle(
        schema_version=AUDIT_BUNDLE_SCHEMA_VERSION,
        bundle_id=bundle_id,
        created_at=created,
        run_id=run_id,
        chain_head_hash=head,
        metrics_fingerprint=metrics_fp,
        n_packets=len(packets),
        cohort_safety_rate=cohort_report.safety_rate,
        cohort_n_probes=cohort_report.n_total,
        attestors_run=tuple(sorted(attestors_run)),
        operating_point=operating_point or AuditOperatingPoint(),
    )
    return bundle.freeze(key=sign_key)


def bind_audit_from_bundle(
    bundle: AuditTrailBundle,
    *,
    packets,
    replay_result,
):
    """Bind the bundle to a live trail and replay result.

    Raises :class:`BundleSignatureError` when:

    * the live trail's chain head differs from the bundle's;
    * the live replay's metrics fingerprint differs from the bundle's;
    * the cohort safety rate is below the SLO floor;
    * the required attestor coverage is unmet.
    """
    bundle.assert_chain_matches(packets)
    bundle.assert_metrics_match(replay_result)
    bundle.assert_meets_slo()
    return bundle
