# SPDX-License-Identifier: Apache-2.0
"""Audit log + reason chain for release-gate decisions.

Each gate decision can be serialized to a record that:

  * names the bundle + cohort hash that bound the gate
  * captures the metric vector with finite/well-formed flags
  * captures the per-tier ``GateResult`` with full check breakdown
  * (optionally) captures the SPC co-attestor verdict
  * is JSON-serializable and replayable without the original gate

:func:`replay_audit_log` reconstructs the verdict string from a
saved record alone --- a property the auditor verifies during
the regression battery.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from knowlytix.harness.governance.release_gates import (
    GateResult,
    RunMetrics,
    TieredReleaseGate,
)


@dataclass
class ReleaseGateReasonChain:
    """Top-K reason triples explaining a gate verdict."""

    failures: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)


def build_release_gate_reason_chain(
    metrics: RunMetrics,
    result: GateResult,
    *,
    co_attestor_verdict: Any | None = None,
) -> ReleaseGateReasonChain:
    """Pull top reasons by category out of the gate + co-attestor."""
    failures = list(result.blocking_reasons)
    flags: list[str] = []
    passes: list[str] = []
    for name, info in result.checks.items():
        if info.get("passed"):
            v = info.get("value")
            try:
                v_s = f"{float(v):.4f}"
            except (TypeError, ValueError):
                v_s = repr(v)
            passes.append(
                f"{name}={v_s} >= {info['threshold']:.2f}"
                if name != "robustness_drop"
                else f"{name}={v_s} <= {info['threshold']:.2f}")
    if co_attestor_verdict is not None:
        if getattr(co_attestor_verdict, "verdict", None) == "flag":
            flags.extend(getattr(
                co_attestor_verdict, "flag_reasons", []))
    return ReleaseGateReasonChain(
        failures=failures, flags=flags, passes=passes,
    )


@dataclass
class ReleaseGateAuditRecord:
    """JSON-serializable audit record for a single gate decision."""

    run_id: str
    target_tier: str
    metrics: dict[str, float]
    metrics_finite: bool
    metrics_well_formed: bool
    well_formed_violations: list[str]
    gate_decisions: dict[str, dict]  # per-tier
    target_decision: dict
    reason_chain: dict
    bundle_id: str = ""
    cohort_hash: str = ""
    threshold_source: str = ""
    co_attestor_verdict: dict | None = None
    spc_slopes: dict[str, float] = field(default_factory=dict)
    signing_method: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)


def to_audit_record(
    metrics: RunMetrics,
    target_tier: str,
    gate: TieredReleaseGate,
    *,
    co_attestor_verdict: Any | None = None,
    bundle: Any | None = None,
) -> ReleaseGateAuditRecord:
    """Build a full audit record from a metric + gate + bundle.

    Captures the gate decision at every tier, not just the requested
    one --- the audit trail must show that the action_taking refusal
    was paired with a recommendation pass (or failure).
    """
    well_formed, viol = metrics.is_well_formed()
    finite = metrics.is_finite()
    decisions: dict[str, dict] = {}
    for tier, r in gate.check_all_tiers(metrics).items():
        decisions[tier] = {
            "passed": bool(r.passed),
            "blocking_reasons": list(r.blocking_reasons),
            "checks": {
                name: {
                    "value": info.get("value"),
                    "threshold": info.get("threshold"),
                    "passed": bool(info.get("passed", False)),
                    "finite": bool(info.get("finite", True)),
                }
                for name, info in r.checks.items()
            },
            "is_calibrated": bool(r.is_calibrated),
            "threshold_source": r.threshold_source,
        }
    target = decisions.get(target_tier, {})
    target_result = gate.check(metrics, target_tier)
    reason_chain = build_release_gate_reason_chain(
        metrics, target_result,
        co_attestor_verdict=co_attestor_verdict,
    )

    co_attestor_dict: dict | None = None
    spc_slopes: dict[str, float] = {}
    if co_attestor_verdict is not None:
        co_attestor_dict = {
            "verdict": getattr(co_attestor_verdict, "verdict", None),
            "structural_passed":
                bool(getattr(co_attestor_verdict,
                             "structural_passed", False)),
            "slope_alerted":
                bool(getattr(co_attestor_verdict,
                             "slope_alerted", False)),
            "blocking_reasons": list(getattr(
                co_attestor_verdict, "blocking_reasons", [])),
            "flag_reasons": list(getattr(
                co_attestor_verdict, "flag_reasons", [])),
        }
        spc_slopes = dict(getattr(
            co_attestor_verdict, "spc_slopes", {}) or {})

    return ReleaseGateAuditRecord(
        run_id=metrics.run_id or "",
        target_tier=target_tier,
        metrics={
            "accuracy": float(metrics.accuracy)
            if isinstance(metrics.accuracy, (int, float))
            else metrics.accuracy,
            "confidence": float(metrics.confidence)
            if isinstance(metrics.confidence, (int, float))
            else metrics.confidence,
            "robustness_drop": float(metrics.robustness_drop)
            if isinstance(metrics.robustness_drop, (int, float))
            else metrics.robustness_drop,
            "fairness_air": float(metrics.fairness_air)
            if isinstance(metrics.fairness_air, (int, float))
            else metrics.fairness_air,
        },
        metrics_finite=finite,
        metrics_well_formed=well_formed,
        well_formed_violations=viol,
        gate_decisions=decisions,
        target_decision=target,
        reason_chain={
            "failures": reason_chain.failures,
            "flags": reason_chain.flags,
            "passes": reason_chain.passes,
        },
        bundle_id=(getattr(bundle, "bundle_id", "")
                   if bundle is not None else ""),
        cohort_hash=(getattr(bundle, "cohort_hash", "")
                     if bundle is not None else ""),
        threshold_source=gate.threshold_source,
        co_attestor_verdict=co_attestor_dict,
        spc_slopes=spc_slopes,
        signing_method=(getattr(bundle, "signing_method", "")
                        if bundle is not None else ""),
    )


def replay_audit_log(record: dict | ReleaseGateAuditRecord) -> str:
    """Reconstruct the verdict string from a saved audit record alone.

    Returns one of:
      * ``"BLOCK"``  -- target tier refused (gate or upstream)
      * ``"FLAG"``   -- target tier passed but co-attestor flagged
      * ``"PASS"``   -- target tier passed and co-attestor passed (or
                        was absent)
    """
    if isinstance(record, ReleaseGateAuditRecord):
        d = record.to_dict()
    else:
        d = record
    if not d.get("metrics_well_formed", True):
        return "BLOCK"
    target = d.get("target_decision", {})
    if not target.get("passed", False):
        return "BLOCK"
    co = d.get("co_attestor_verdict") or {}
    if co.get("verdict") == "flag":
        return "FLAG"
    return "PASS"
