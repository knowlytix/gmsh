# SPDX-License-Identifier: Apache-2.0
"""Frozen, signed bundle for the tiered release gate.

A :class:`ReleaseGateBundle` carries everything the runtime needs to
make audit-defensible release decisions:

  * the calibrated tier thresholds (per-cell ``ThresholdDistribution``)
  * the policy-declared baseline alongside, for provenance
  * the labeled cohort fingerprint (so the bundle binds to the
    calibration evidence base)
  * the SPC slope co-attestor configuration
  * an operating point (FAR/FRR/safety-rate floors)
  * exhaustive monotonicity proof hash
  * an HMAC-SHA-256 signature

The bundle ships *both* the calibrated thresholds (operating) and
the policy-declared defaults (provenance) so the auditor can see
how far the calibration moved each cell. ``bind_release_gate_from_bundle``
refuses to bind when the calibration cohort hash, the operating
point caps, or the signature don't match.

Mirrors :mod:`governed_agent.policy_bundle` for the release-gate
surface.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowlytix.harness.governance.bundle_signing import (
    SigningMethod,
    resolve_signing_key,
    sign,
    verify,
)
from knowlytix.harness.governance.governance_bundle import (
    BundleSignatureError,
    ThresholdDistribution,
)
from knowlytix.harness.governance.release_gates import (
    GateThresholds,
    TieredReleaseGate,
)

RELEASE_GATE_BUNDLE_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseGateOperatingPoint:
    """SLO floors / ceilings for binding the gate at runtime."""

    # Adversarial safety rate must meet or exceed this floor.
    min_adversarial_safety_rate: float = 0.95
    # OOD decision-match rate must meet or exceed this floor.
    min_ood_decision_match_rate: float = 0.95
    # Per-tier FAR ceiling.
    max_per_tier_far: float = 0.05
    # Per-tier FRR ceiling.
    max_per_tier_frr: float = 0.10
    # Required uplift over fixed-threshold baseline (correct pairs).
    min_uplift_over_fixed: int = 0
    # SPC slope-alert configuration: maximum acceptable rate of
    # slope-flagged runs in a healthy window. Above this, the
    # monitor should alarm.
    max_slope_alert_rate: float = 0.10


# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReleaseGateBundle:
    """Frozen artefact bundle for high-stakes release-gate enforcement."""

    schema_version: str
    bundle_id: str
    created_at: str
    source_dataset: str  # e.g. "adult_income"

    # Calibrated thresholds: per-tier, per-metric ThresholdDistribution.
    # Cells set to None mean the cohort had insufficient signal --- we
    # fall back to the policy-declared default for that cell.
    calibrated: dict[str, dict[str, ThresholdDistribution | None]]

    # Policy-declared baseline (provenance side-by-side).
    policy_declared: dict[str, dict[str, float]]

    # Cohort hash from release_gate_validation.cohort_fingerprint.
    cohort_hash: str
    cohort_size: int

    # SPC co-attestor config (slope thresholds per metric).
    spc_slope_config: dict[str, float] = field(default_factory=dict)

    # Operating point.
    operating_point: ReleaseGateOperatingPoint = field(
        default_factory=ReleaseGateOperatingPoint,
    )

    # Calibration-time validation evidence.
    validation_metrics: dict[str, float] = field(default_factory=dict)

    # Hash over the exhaustive monotonicity proof (resolution + n_pass +
    # n_violations).
    monotonicity_proof_hash: str = ""

    signature: str = ""
    signing_method: SigningMethod = "sha256"

    # Multi-signer signatures (k-of-n quorum).
    attestor_signatures: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Threshold materialization
    # ------------------------------------------------------------------ #

    def calibrated_tiers(self) -> dict[str, GateThresholds]:
        """Materialize the calibrated tier set for the runtime gate.

        Cells with ``None`` calibrated values fall back to the
        policy-declared default for that cell (recorded in
        ``policy_declared``). The returned ``GateThresholds`` carry
        ``is_calibrated=True`` iff at least one cell was actually
        calibrated.

        Defensive monotonicity projection is applied at materialization
        time so a downstream :class:`TieredReleaseGate` constructor
        cannot raise on a bundle whose raw bootstrap fits happen to be
        non-monotonic. The build path already projects + verifies
        invariants; this is the second layer that survives bundle
        replay against an unfamiliar runtime.
        """
        from knowlytix.harness.governance.release_gate_calibration import (
            enforce_monotonic_ordering,
        )
        raw: dict[str, GateThresholds] = {}
        for tier in TieredReleaseGate.TIER_ORDER:
            cal = self.calibrated.get(tier, {})
            base = self.policy_declared.get(tier, {})
            any_calibrated = any(v is not None for v in cal.values())

            def _v(name: str, default_key: str) -> float:
                d = cal.get(name)
                if d is not None:
                    return float(d.mean)
                return float(base.get(default_key, 0.0))

            raw[tier] = GateThresholds(
                min_accuracy=_v("min_accuracy", "min_accuracy"),
                min_confidence=_v("min_confidence", "min_confidence"),
                max_robustness_drop=_v(
                    "max_robustness_drop", "max_robustness_drop"),
                min_fairness_air=_v(
                    "min_fairness_air", "min_fairness_air"),
                is_calibrated=any_calibrated,
                source=f"bundle:{self.bundle_id}",
            )
        return enforce_monotonic_ordering(raw)

    # ------------------------------------------------------------------ #
    # Canonical payload + signing
    # ------------------------------------------------------------------ #
    def canonical_payload(self) -> bytes:
        cal_canonical: dict[str, dict[str, dict | None]] = {}
        for tier, cells in self.calibrated.items():
            row: dict[str, dict | None] = {}
            for name, dist in cells.items():
                row[name] = (asdict(dist) if dist is not None else None)
            cal_canonical[tier] = row

        d = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "source_dataset": self.source_dataset,
            "calibrated": {
                k: cal_canonical[k]
                for k in sorted(cal_canonical)
            },
            "policy_declared": {
                k: dict(sorted(v.items()))
                for k, v in sorted(self.policy_declared.items())
            },
            "cohort_hash": self.cohort_hash,
            "cohort_size": self.cohort_size,
            "spc_slope_config": dict(
                sorted(self.spc_slope_config.items())),
            "operating_point": {
                "min_adversarial_safety_rate":
                    self.operating_point.min_adversarial_safety_rate,
                "min_ood_decision_match_rate":
                    self.operating_point.min_ood_decision_match_rate,
                "max_per_tier_far":
                    self.operating_point.max_per_tier_far,
                "max_per_tier_frr":
                    self.operating_point.max_per_tier_frr,
                "min_uplift_over_fixed":
                    self.operating_point.min_uplift_over_fixed,
                "max_slope_alert_rate":
                    self.operating_point.max_slope_alert_rate,
            },
            "validation_metrics":
                dict(sorted(self.validation_metrics.items())),
            "monotonicity_proof_hash": self.monotonicity_proof_hash,
            "signing_method": self.signing_method,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def freeze(self, key: bytes | str | None = None) -> "ReleaseGateBundle":
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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, path: str | Path, *, key: bytes | str | None = None,
    ) -> "ReleaseGateBundle":
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
                f"Refusing to load release-gate bundle at {path}: {exc}"
            ) from exc
        if not ok:
            raise BundleSignatureError(
                f"Signature mismatch on release-gate bundle at {path}: "
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


def _spc_config_to_dict(config: Any) -> dict[str, float]:
    return {
        "min_acc_slope": float(getattr(config, "min_acc_slope", -0.005)),
        "min_conf_slope": float(getattr(config, "min_conf_slope", -0.005)),
        "min_fair_slope": float(getattr(config, "min_fair_slope", -0.005)),
        "max_rob_slope": float(getattr(config, "max_rob_slope", +0.005)),
        "min_runs_for_alert": float(getattr(
            config, "min_runs_for_alert", 5)),
    }


def build_release_gate_bundle(
    *,
    cohort: list,
    source_dataset: str = "adult_income",
    n_bootstrap: int = 200,
    seed: int = 42,
    spc_config: Any = None,
    operating_point: ReleaseGateOperatingPoint | None = None,
    validation_metrics: dict[str, float] | None = None,
    monotonicity_resolution: float = 0.10,
) -> ReleaseGateBundle:
    """Produce a calibrated, signed release-gate bundle from a cohort.

    Steps:

      1. Bootstrap-calibrate every (tier, metric) cell.
      2. Capture the policy-declared baseline.
      3. Compute the cohort fingerprint.
      4. Verify exhaustive monotonicity invariants on the calibrated
         tier set; capture the proof hash.
      5. Sign the canonical payload (HMAC-SHA-256 with deployment key,
         else SHA-256).
    """
    from knowlytix.harness.governance.release_gate_calibration import (
        calibrate_release_gate_thresholds,
        enforce_monotonic_ordering,
        policy_declared_baseline,
    )
    from knowlytix.harness.governance.release_gate_invariants import (
        verify_release_gate_invariants,
    )
    from knowlytix.harness.governance.release_gate_validation import cohort_fingerprint

    # 1. Calibrate
    cal = calibrate_release_gate_thresholds(
        cohort, n_bootstrap=n_bootstrap, seed=seed,
    )

    # 2. Materialize calibrated dict[tier][metric] -> ThresholdDistribution
    calibrated: dict[str, dict[str, ThresholdDistribution | None]] = {}
    for tier, cset in cal.items():
        calibrated[tier] = {
            "min_accuracy": cset.min_accuracy,
            "min_confidence": cset.min_confidence,
            "max_robustness_drop": cset.max_robustness_drop,
            "min_fairness_air": cset.min_fairness_air,
        }

    # 3. Policy-declared side-by-side
    declared = policy_declared_baseline()
    policy_declared: dict[str, dict[str, float]] = {}
    for tier, t in declared.items():
        policy_declared[tier] = {
            "min_accuracy": t.min_accuracy,
            "min_confidence": t.min_confidence,
            "max_robustness_drop": t.max_robustness_drop,
            "min_fairness_air": t.min_fairness_air,
        }

    # 4. Cohort fingerprint
    coh_hash = cohort_fingerprint(cohort)

    # 5. SPC config
    if spc_config is None:
        from knowlytix.harness.governance.release_gate_co_attestor import SPCSlopeConfig
        spc_config = SPCSlopeConfig()
    spc_dict = _spc_config_to_dict(spc_config)

    # 6. Materialize calibrated tiers + project to monotone, then
    # verify invariants.
    calibrated_gate_tiers: dict[str, GateThresholds] = {}
    for tier, cset in cal.items():
        calibrated_gate_tiers[tier] = cset.to_gate_thresholds(
            fallback=GateThresholds(
                min_accuracy=policy_declared[tier]["min_accuracy"],
                min_confidence=policy_declared[tier]["min_confidence"],
                max_robustness_drop=policy_declared[tier][
                    "max_robustness_drop"],
                min_fairness_air=policy_declared[tier]["min_fairness_air"],
            ),
            source="calibrated_pre_projection",
        )
    monotone_tiers = enforce_monotonic_ordering(calibrated_gate_tiers)
    test_gate = TieredReleaseGate(
        tiers=monotone_tiers, threshold_source="calibrated_test",
    )
    invariant_report = verify_release_gate_invariants(
        test_gate, resolution=monotonicity_resolution,
    )
    proof_payload = {
        "n_grid_points": invariant_report.n_grid_points,
        "n_pass": invariant_report.n_pass,
        "n_violations": invariant_report.n_violations,
        "invariants_checked": sorted(
            invariant_report.invariants_checked),
    }
    proof_hash = hashlib.sha256(
        json.dumps(proof_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if invariant_report.n_violations > 0:
        raise RuntimeError(
            f"Refusing to build bundle: monotonicity proof failed with "
            f"{invariant_report.n_violations} violations: "
            f"{invariant_report.violations[0].description}"
        )

    bundle = ReleaseGateBundle(
        schema_version=RELEASE_GATE_BUNDLE_SCHEMA_VERSION,
        bundle_id=make_bundle_id(source_dataset, coh_hash, utcnow_iso()),
        created_at=utcnow_iso(),
        source_dataset=source_dataset,
        calibrated=calibrated,
        policy_declared=policy_declared,
        cohort_hash=coh_hash,
        cohort_size=len(cohort),
        spc_slope_config=spc_dict,
        operating_point=(
            operating_point or ReleaseGateOperatingPoint()),
        validation_metrics=validation_metrics or {},
        monotonicity_proof_hash=proof_hash,
    )
    return bundle.freeze()


def bind_release_gate_from_bundle(
    bundle: ReleaseGateBundle,
    *,
    cohort: list | None = None,
    adversarial_safety_rate: float | None = None,
    ood_decision_match_rate: float | None = None,
    uplift_over_fixed: int | None = None,
) -> TieredReleaseGate:
    """Bind a TieredReleaseGate to a verified bundle.

    Refuses to bind when:

      * the bundle's signature doesn't verify, OR
      * (if ``cohort`` provided) the cohort fingerprint doesn't
        match the bundle's calibration cohort, OR
      * (if metrics provided) any operating-point floor is missed.

    Pass the ``cohort`` argument to enforce calibration-evidence-base
    integrity at bind time.
    """
    from knowlytix.harness.governance.release_gate_validation import cohort_fingerprint
    op = bundle.operating_point

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
            f"Refusing to bind: adversarial safety rate "
            f"{adversarial_safety_rate:.4f} below floor "
            f"{op.min_adversarial_safety_rate:.4f}.")

    if (ood_decision_match_rate is not None
            and ood_decision_match_rate < op.min_ood_decision_match_rate):
        raise ValueError(
            f"Refusing to bind: OOD decision-match rate "
            f"{ood_decision_match_rate:.4f} below floor "
            f"{op.min_ood_decision_match_rate:.4f}.")

    if (uplift_over_fixed is not None
            and uplift_over_fixed < op.min_uplift_over_fixed):
        raise ValueError(
            f"Refusing to bind: uplift over fixed-threshold baseline "
            f"{uplift_over_fixed} below floor "
            f"{op.min_uplift_over_fixed}.")

    return TieredReleaseGate.from_bundle(bundle)
