# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed bundle for the multi-agent governance surface.

A :class:`MultiAgentBundle` carries everything the runtime needs to
make audit-defensible role-based decisions:

  * per-role fingerprints (SHA-256 over allowed_tools + forbidden_tools
    + description) so the runtime refuses to bind to an orchestrator
    whose roles don't match
  * cohort hash (validated at calibration time)
  * tool-frequency reference distribution (PSI baseline)
  * handoff-latency baseline (ThresholdDistribution)
  * operating point (FAR/FRR/safety-rate floors, latency p99 ceiling)
  * exhaustive invariants proof hash
  * HMAC-SHA-256 signature

Mirrors :mod:`governed_agent.policy_bundle` for the multi-agent surface.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from knowlytix.harness.governance.agents.multi_agent import MultiAgentOrchestrator
from knowlytix.harness.governance.agents.multi_agent_calibration import (
    HandoffLatencyBaseline, PerAgentToolFrequency, fingerprint_orchestrator,
)
from knowlytix.harness.governance.bundle_signing import (
    SigningMethod, resolve_signing_key, sign, verify,
)
from knowlytix.harness.governance.governance_bundle import (
    BundleSignatureError, ThresholdDistribution,
)


MULTI_AGENT_BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class MultiAgentOperatingPoint:
    """SLO floors / ceilings for binding the orchestrator at runtime."""

    min_adversarial_safety_rate: float = 0.95
    min_ood_decision_match_rate: float = 0.95
    max_per_role_tool_psi: float = 0.10
    max_handoff_latency_p99_us: float = 1_000_000.0   # 1s
    min_uplift_over_naive: int = 0
    min_alt_validator_agreement: float = 1.0


@dataclass
class MultiAgentBundle:
    """Frozen artefact bundle for high-stakes multi-agent governance."""

    schema_version: str
    bundle_id: str
    created_at: str
    source_dataset: str   # e.g. "adult_income"

    # Per-role fingerprint: name -> SHA-256 over canonical role payload
    per_role_fingerprint: dict[str, str]

    # Calibration cohort hash (multi_agent_validation.cohort_fingerprint)
    cohort_hash: str
    cohort_size: int

    # Reference distributions
    per_agent_tool_frequency: dict | None = None  # PerAgentToolFrequency.to_dict
    handoff_latency_baseline: dict | None = None  # HandoffLatencyBaseline.to_dict
    handoff_latency_p99_dist: ThresholdDistribution | None = None

    # Operating point
    operating_point: MultiAgentOperatingPoint = field(
        default_factory=MultiAgentOperatingPoint,
    )

    # Calibration-time validation evidence
    validation_metrics: dict[str, float] = field(default_factory=dict)

    # Hash over the exhaustive invariants proof
    invariants_proof_hash: str = ""

    signature: str = ""
    signing_method: SigningMethod = "sha256"

    # Multi-signer (k-of-n) signatures
    attestor_signatures: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Canonical payload + signing
    # ------------------------------------------------------------------ #
    def canonical_payload(self) -> bytes:
        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_dataset": self.source_dataset,
            "per_role_fingerprint": dict(
                sorted(self.per_role_fingerprint.items())),
            "cohort_hash": self.cohort_hash,
            "cohort_size": self.cohort_size,
            "per_agent_tool_frequency": (
                self.per_agent_tool_frequency
                if self.per_agent_tool_frequency is not None else None
            ),
            "handoff_latency_baseline": (
                self.handoff_latency_baseline
                if self.handoff_latency_baseline is not None else None
            ),
            "handoff_latency_p99_dist": (
                asdict(self.handoff_latency_p99_dist)
                if self.handoff_latency_p99_dist is not None else None
            ),
            "operating_point": asdict(self.operating_point),
            "validation_metrics": dict(
                sorted(self.validation_metrics.items())),
            "invariants_proof_hash": self.invariants_proof_hash,
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True, default=str).encode("utf-8")

    def freeze(self, key: bytes | str | None = None) -> "MultiAgentBundle":
        resolved = resolve_signing_key(key)
        self.signing_method = (
            "hmac-sha256" if resolved is not None else "sha256"
        )
        self.signature, _ = sign(self.canonical_payload(), key=resolved)
        return self

    def verify(self, key: bytes | str | None = None) -> bool:
        resolved = resolve_signing_key(key)
        return verify(self.canonical_payload(),
                      self.signature, self.signing_method, key=resolved)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, path: str | Path, *, key: bytes | str | None = None,
    ) -> "MultiAgentBundle":
        from knowlytix.harness.governance.bundle_signing import BundleSigningError
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if not isinstance(bundle, cls):
            raise TypeError(
                f"Loaded object is {type(bundle).__name__}, "
                f"expected {cls.__name__}")
        try:
            ok = bundle.verify(key=key)
        except BundleSigningError as exc:
            raise BundleSignatureError(
                f"Refusing to load multi-agent bundle at {path}: {exc}"
            ) from exc
        if not ok:
            raise BundleSignatureError(
                f"Signature mismatch on multi-agent bundle at {path}: "
                f"signing_method={bundle.signing_method}.")
        return bundle


# ---------------------------------------------------------------------------
# Bundle production / binding
# ---------------------------------------------------------------------------

def make_bundle_id(source_dataset: str, cohort_hash: str,
                   created_at: str) -> str:
    raw = f"{source_dataset}|{cohort_hash}|{created_at}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_multi_agent_bundle(
    *,
    orch: MultiAgentOrchestrator,
    cohort,
    source_dataset: str = "adult_income",
    per_agent_freq: PerAgentToolFrequency | None = None,
    handoff_baseline: HandoffLatencyBaseline | None = None,
    handoff_p99_dist: ThresholdDistribution | None = None,
    operating_point: MultiAgentOperatingPoint | None = None,
    validation_metrics: dict[str, float] | None = None,
    invariants_resolution: bool = True,
) -> MultiAgentBundle:
    """Produce a calibrated, signed multi-agent bundle.

    Steps:
      1. Compute per-role fingerprints from the orchestrator.
      2. Compute the cohort fingerprint.
      3. Verify exhaustive invariants on the orchestrator; capture
         the proof hash.
      4. Sign the canonical payload (HMAC-SHA-256 with deployment
         key, else SHA-256).
    """
    from knowlytix.harness.governance.agents.multi_agent_invariants import (
        verify_multi_agent_invariants,
    )
    from knowlytix.harness.governance.agents.multi_agent_validation import (
        cohort_fingerprint,
    )

    # 1. Per-role fingerprints
    per_role_fp = fingerprint_orchestrator(orch)

    # 2. Cohort hash
    coh_hash = cohort_fingerprint(cohort) if cohort else ""

    # 3. Invariants proof
    inv_report = verify_multi_agent_invariants(orch)
    proof_payload = {
        "n_cells": inv_report.n_cells,
        "n_pass": inv_report.n_pass,
        "n_violations": inv_report.n_violations,
        "invariants_checked": sorted(inv_report.invariants_checked),
    }
    proof_hash = hashlib.sha256(
        json.dumps(proof_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if inv_report.n_violations > 0:
        raise RuntimeError(
            f"Refusing to build bundle: invariants proof failed with "
            f"{inv_report.n_violations} violations: "
            f"{inv_report.violations[0].description}"
        )

    bundle = MultiAgentBundle(
        schema_version=MULTI_AGENT_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_bundle_id(source_dataset, coh_hash, utcnow_iso()),
        created_at=utcnow_iso(),
        source_dataset=source_dataset,
        per_role_fingerprint=per_role_fp,
        cohort_hash=coh_hash,
        cohort_size=len(cohort) if cohort else 0,
        per_agent_tool_frequency=(
            per_agent_freq.to_dict() if per_agent_freq else None),
        handoff_latency_baseline=(
            handoff_baseline.to_dict() if handoff_baseline else None),
        handoff_latency_p99_dist=handoff_p99_dist,
        operating_point=(operating_point or MultiAgentOperatingPoint()),
        validation_metrics=validation_metrics or {},
        invariants_proof_hash=proof_hash,
    )
    return bundle.freeze()


def bind_orchestrator_from_bundle(
    orch: MultiAgentOrchestrator,
    bundle: MultiAgentBundle,
    *,
    cohort=None,
    adversarial_safety_rate: float | None = None,
    ood_decision_match_rate: float | None = None,
    alt_validator_agreement: float | None = None,
    handoff_latency_p99_us: float | None = None,
    uplift_over_naive: int | None = None,
) -> MultiAgentOrchestrator:
    """Bind an orchestrator to a verified bundle.

    Refuses to bind when:
      * any per-role fingerprint mismatches between the orchestrator
        and the bundle;
      * the cohort fingerprint mismatches (if cohort provided);
      * any operating-point floor / ceiling is violated.
    """
    from knowlytix.harness.governance.agents.multi_agent_validation import (
        cohort_fingerprint,
    )
    op = bundle.operating_point

    actual_fp = fingerprint_orchestrator(orch)
    for role, fp in bundle.per_role_fingerprint.items():
        if actual_fp.get(role) != fp:
            raise ValueError(
                f"Refusing to bind: per-role fingerprint mismatch for "
                f"'{role}'. Bundle: {fp[:16]}; "
                f"actual: {actual_fp.get(role, '<missing>')[:16]}"
            )
    # And no extra roles in the orchestrator that aren't in the bundle
    for role in actual_fp:
        if role not in bundle.per_role_fingerprint:
            raise ValueError(
                f"Refusing to bind: orchestrator has unbundled role "
                f"'{role}'."
            )

    if cohort is not None:
        actual_hash = cohort_fingerprint(cohort)
        if actual_hash != bundle.cohort_hash:
            raise ValueError(
                f"Refusing to bind: cohort fingerprint mismatch. "
                f"Bundle: {bundle.cohort_hash[:16]}; "
                f"actual: {actual_hash[:16]}.")

    if (adversarial_safety_rate is not None
            and adversarial_safety_rate < op.min_adversarial_safety_rate):
        raise ValueError(
            f"Refusing to bind: adversarial safety {adversarial_safety_rate:.4f}"
            f" < floor {op.min_adversarial_safety_rate:.4f}")

    if (ood_decision_match_rate is not None
            and ood_decision_match_rate < op.min_ood_decision_match_rate):
        raise ValueError(
            f"Refusing to bind: OOD decision-match {ood_decision_match_rate:.4f}"
            f" < floor {op.min_ood_decision_match_rate:.4f}")

    if (alt_validator_agreement is not None
            and alt_validator_agreement < op.min_alt_validator_agreement):
        raise ValueError(
            f"Refusing to bind: alt-validator agreement "
            f"{alt_validator_agreement:.4f} < floor "
            f"{op.min_alt_validator_agreement:.4f}")

    if (handoff_latency_p99_us is not None
            and handoff_latency_p99_us > op.max_handoff_latency_p99_us):
        raise ValueError(
            f"Refusing to bind: handoff latency p99 "
            f"{handoff_latency_p99_us:.1f}us > ceiling "
            f"{op.max_handoff_latency_p99_us:.1f}us")

    if (uplift_over_naive is not None
            and uplift_over_naive < op.min_uplift_over_naive):
        raise ValueError(
            f"Refusing to bind: uplift over naive baseline "
            f"{uplift_over_naive} < floor {op.min_uplift_over_naive}")

    # Stash bundle on the orchestrator for audit trail
    orch._bundle = bundle
    return orch
