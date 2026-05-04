# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed artifact for a complete governed workflow run.

A `WorkflowBundle` is what an operator ships when an audit review
asks ``what happened during this run, end to end?''  It carries
the workflow snapshot's fingerprint, the cohort safety rate, the
release-gate decision, the per-phase latency, and an HMAC-SHA-256
signature over the canonical payload.

`bind_workflow_from_bundle` refuses to bind to a live snapshot
that diverges from the bundle's recorded values.  Mirrors
`AuditTrailBundle` (Ch.14) and `PolicyBundle` (Ch.12) for the
workflow surface.
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
from knowlytix.harness.governance.workflow_hardening.workflow_snapshot import (
    WorkflowSnapshot,
)


WORKFLOW_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class WorkflowOperatingPoint:
    """SLO for a workflow at bind time.

    `min_pass_rate` floors the verification pass rate.
    `min_cohort_safety_rate` floors the adversarial sweep rate.
    `accepted_release_gates` restricts which gate verdicts are
    bind-compatible (e.g. a deployment may refuse to bind to a
    bundle whose gate is CONDITIONAL).
    """

    min_pass_rate: float = 0.95
    min_cohort_safety_rate: float = 1.0
    accepted_release_gates: tuple[str, ...] = ("APPROVED",)
    max_phase_latency_s: dict[str, float] = field(
        default_factory=lambda: {
            "data": 30.0,
            "training": 600.0,
            "diagnostics": 120.0,
            "fairness": 60.0,
        }
    )


@dataclass
class WorkflowBundle:
    """Frozen artifact bundle for one governed workflow run."""

    schema_version: str
    bundle_id: str
    created_at: str
    workflow_id: str

    snapshot_fingerprint: str
    release_gate: str
    pass_rate: float
    cumulative_claims: int
    cumulative_blocks: int

    cohort_safety_rate: float
    cohort_n_probes: int
    attestors_run: tuple[str, ...]    # ("snapshot", "cross_component", ...)

    per_phase_latency_s: dict[str, float]

    operating_point: WorkflowOperatingPoint = field(
        default_factory=WorkflowOperatingPoint,
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
            "workflow_id": self.workflow_id,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "release_gate": self.release_gate,
            "pass_rate": self.pass_rate,
            "cumulative_claims": self.cumulative_claims,
            "cumulative_blocks": self.cumulative_blocks,
            "cohort_safety_rate": self.cohort_safety_rate,
            "cohort_n_probes": self.cohort_n_probes,
            "attestors_run": sorted(self.attestors_run),
            "per_phase_latency_s": dict(
                sorted(self.per_phase_latency_s.items())
            ),
            "operating_point": {
                "min_pass_rate": self.operating_point.min_pass_rate,
                "min_cohort_safety_rate":
                    self.operating_point.min_cohort_safety_rate,
                "accepted_release_gates": list(
                    sorted(self.operating_point.accepted_release_gates)
                ),
                "max_phase_latency_s": dict(sorted(
                    self.operating_point.max_phase_latency_s.items()
                )),
            },
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    # ---- sign / verify ---- #

    def freeze(self, key: bytes | str | None = None) -> "WorkflowBundle":
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

    def assert_snapshot_matches(self, snapshot: WorkflowSnapshot) -> None:
        """Refuse if the live snapshot's fingerprint differs."""
        live = snapshot.fingerprint()
        if live != self.snapshot_fingerprint:
            raise BundleSignatureError(
                f"Workflow snapshot fingerprint mismatch: bundle "
                f"declares {self.snapshot_fingerprint[:12]}... but "
                f"live snapshot is {live[:12]}...  The workflow "
                f"produced different governance results than the "
                f"bundle was built on."
            )

    def assert_meets_slo(self) -> None:
        op = self.operating_point
        if self.pass_rate < op.min_pass_rate:
            raise BundleSignatureError(
                f"pass_rate {self.pass_rate:.4f} below SLO floor "
                f"{op.min_pass_rate:.4f}."
            )
        if self.cohort_safety_rate < op.min_cohort_safety_rate:
            raise BundleSignatureError(
                f"cohort_safety_rate {self.cohort_safety_rate:.4f} "
                f"below SLO floor {op.min_cohort_safety_rate:.4f}."
            )
        if self.release_gate not in op.accepted_release_gates:
            raise BundleSignatureError(
                f"release_gate {self.release_gate!r} not in accepted "
                f"set {sorted(op.accepted_release_gates)}."
            )
        for phase, lat in self.per_phase_latency_s.items():
            cap = op.max_phase_latency_s.get(phase)
            if cap is not None and lat > cap:
                raise BundleSignatureError(
                    f"phase {phase!r} latency {lat:.2f}s exceeds "
                    f"cap {cap:.2f}s."
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
    ) -> "WorkflowBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle: WorkflowBundle = pickle.load(f)
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
                f"Signature mismatch on bundle at {path}."
            )
        return bundle


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def make_workflow_bundle_id(workflow_id: str, created_at: str) -> str:
    raw = f"{workflow_id}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_workflow_bundle(
    *,
    snapshot: WorkflowSnapshot,
    cohort_report,
    per_phase_latency_s: dict[str, float],
    attestors_run: tuple[str, ...] = ("snapshot",),
    operating_point: WorkflowOperatingPoint | None = None,
    sign_key: bytes | str | None = None,
) -> WorkflowBundle:
    """Assemble and HMAC-sign a `WorkflowBundle`."""
    created = utcnow_iso()
    bundle = WorkflowBundle(
        schema_version=WORKFLOW_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_workflow_bundle_id(snapshot.workflow_id, created),
        created_at=created,
        workflow_id=snapshot.workflow_id,
        snapshot_fingerprint=snapshot.fingerprint(),
        release_gate=snapshot.release_gate,
        pass_rate=snapshot.pass_rate,
        cumulative_claims=snapshot.cumulative_claims,
        cumulative_blocks=snapshot.cumulative_blocks,
        cohort_safety_rate=cohort_report.safety_rate,
        cohort_n_probes=cohort_report.n_total,
        attestors_run=tuple(sorted(attestors_run)),
        per_phase_latency_s=dict(per_phase_latency_s),
        operating_point=operating_point or WorkflowOperatingPoint(),
    )
    return bundle.freeze(key=sign_key)


def bind_workflow_from_bundle(
    bundle: WorkflowBundle,
    *,
    snapshot: WorkflowSnapshot,
):
    """Bind a bundle to a live snapshot; raise if any drift detected."""
    bundle.assert_snapshot_matches(snapshot)
    bundle.assert_meets_slo()
    return bundle
