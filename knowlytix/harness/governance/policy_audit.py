# SPDX-License-Identifier: Apache-2.0
"""Reason-chain attachment and JSON audit log for policy decisions.

Mirrors :mod:`governed_agent.reason_chain` for the policy surface.
Every (decision, attestor_verdict) pair gets a structured audit
record with:

  * the probe (agent, action, args, context)
  * which layer fired and on what
  * top-K incident triples for the action and context entities
  * Layer 3 plausibility distance + threshold + CI
  * Layer 4 tension + threshold (when available)
  * the bundle id + signature so the verdict can be replayed
    cold from the log alone

The log is JSON-serialisable; ``replay_audit_log(log)`` reproduces
the verdict from the log without needing the live store.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.policy_bundle import PolicyBundle
    from knowlytix.harness.governance.policy_calibration import PolicyAttestorVerdict
    from knowlytix.harness.testing.policy import PolicyDecision, PolicyEngine


@dataclass
class IncidentTriple:
    head: str
    relation: str
    tail: str
    score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyReasonChain:
    """Structured reason-chain attached to a policy decision."""

    action_top_triples: list[IncidentTriple] = field(default_factory=list)
    context_top_triples: list[IncidentTriple] = field(default_factory=list)
    plausibility: float | None = None
    plausibility_threshold: float | None = None
    plausibility_ci: tuple[float, float] | None = None
    tension_signals: dict[str, float] = field(default_factory=dict)
    tau_contra: float | None = None

    def to_dict(self) -> dict:
        return {
            "action_top_triples": [
                t.to_dict() for t in self.action_top_triples
            ],
            "context_top_triples": [
                t.to_dict() for t in self.context_top_triples
            ],
            "plausibility": self.plausibility,
            "plausibility_threshold": self.plausibility_threshold,
            "plausibility_ci": (
                list(self.plausibility_ci)
                if self.plausibility_ci is not None else None
            ),
            "tension_signals": dict(self.tension_signals),
            "tau_contra": self.tau_contra,
        }


def build_policy_reason_chain(
    engine: "PolicyEngine",
    decision: "PolicyDecision",
    *,
    bundle: "PolicyBundle | None" = None,
    top_k: int = 5,
) -> PolicyReasonChain:
    """Build a top-K-corpus reason chain for a policy decision."""
    store = engine.store
    chain = PolicyReasonChain()

    # Top-K incident triples for the action entity
    try:
        action_triples = store.query_triples(head=decision.action) or []
        for t in action_triples[:top_k]:
            if isinstance(t, tuple) and len(t) == 3:
                h, r, tail = t
            elif isinstance(t, dict):
                h = t.get("head", decision.action)
                r = t.get("relation", "")
                tail = t.get("tail", "")
            else:
                h = getattr(t, "head", decision.action)
                r = getattr(t, "relation", "")
                tail = getattr(t, "tail", "")
            try:
                score = store.score_triple(h, r, tail)
                score_f = float(score) if score is not None else None
            except Exception:
                score_f = None
            chain.action_top_triples.append(
                IncidentTriple(h, r, tail, score_f)
            )
    except Exception:
        pass

    # Plausibility evidence — record the threshold even when no
    # plausibility score was computed (the audit record carries the
    # threshold the engine would have used had a score been
    # available).
    if "plausibility" in decision.gms_scores:
        chain.plausibility = float(decision.gms_scores["plausibility"])
    if bundle is not None and bundle.policy_geodesic is not None:
        chain.plausibility_threshold = bundle.policy_geodesic.mean
        chain.plausibility_ci = (
            bundle.policy_geodesic.ci_low,
            bundle.policy_geodesic.ci_hi,
        )
    else:
        chain.plausibility_threshold = float(getattr(
            engine, "PLAUSIBILITY_THRESHOLD",
            type(engine).PLAUSIBILITY_THRESHOLD,
        ))

    # Tension evidence
    for k, v in decision.gms_scores.items():
        if isinstance(k, str) and k.startswith("tension:"):
            chain.tension_signals[k] = float(v)
    if bundle is not None and bundle.tau_contra is not None:
        chain.tau_contra = bundle.tau_contra.mean
    else:
        chain.tau_contra = float(getattr(
            engine, "TENSION_THRESHOLD",
            type(engine).TENSION_THRESHOLD,
        ))

    return chain


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


@dataclass
class PolicyAuditRecord:
    """One end-to-end audit record for a policy decision.

    Every field is JSON-serialisable so a deployment can write the
    record straight to a tracing pipeline. ``replay()`` re-derives
    the attestor verdict from the record alone (no live store
    needed) so an auditor can replay decisions years later from the
    archived log.
    """

    bundle_id: str
    bundle_signature: str
    bundle_signing_method: str
    rules_yaml_hash: str

    probe_id: str
    agent: str
    action: str
    args: dict
    context: str

    # Engine output
    allowed: bool
    rules_checked: list[str]
    blocking_rules: list[str]
    gms_scores: dict[str, float]
    reason: str

    # Quorum verdict
    quorum_decision: str    # BLOCK | ALLOW | REVIEW
    quorum_required: int
    n_layers_blocking: int
    n_layers_signal_available: int
    layer1_block: bool
    layer2_block: bool
    layer3_block: bool | None
    layer4_block: bool | None
    quorum_evidence: str

    # Reason chain
    reason_chain: dict

    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def replay(self) -> str:
        """Re-derive the BLOCK/ALLOW/REVIEW verdict from this record alone.

        Useful in an auditor's offline review when the live store is
        no longer available. Returns the quorum decision the runtime
        would have produced.
        """
        n_avail = (
            2  # L1 + L2 always available
            + (1 if self.layer3_block is not None else 0)
            + (1 if self.layer4_block is not None else 0)
        )
        n_blocking = (
            int(self.layer1_block) + int(self.layer2_block)
            + int(self.layer3_block is True)
            + int(self.layer4_block is True)
        )
        if n_blocking == 0:
            return "ALLOW"
        if n_blocking >= self.quorum_required:
            return "BLOCK"
        return "REVIEW"


def to_audit_record(
    *,
    engine: "PolicyEngine",
    bundle: "PolicyBundle | None",
    probe_id: str,
    agent: str,
    decision: "PolicyDecision",
    verdict: "PolicyAttestorVerdict",
    args: dict | None = None,
    context: str = "",
    timestamp: str = "",
) -> PolicyAuditRecord:
    """Pack a (decision, verdict) into a frozen audit record."""
    chain = build_policy_reason_chain(
        engine, decision, bundle=bundle,
    )
    bundle_id = bundle.bundle_id if bundle is not None else ""
    sig = bundle.signature if bundle is not None else ""
    method = (bundle.signing_method
              if bundle is not None else "sha256")
    yaml_hash = bundle.rules_yaml_hash if bundle is not None else ""

    return PolicyAuditRecord(
        bundle_id=bundle_id,
        bundle_signature=sig,
        bundle_signing_method=method,
        rules_yaml_hash=yaml_hash,
        probe_id=probe_id,
        agent=agent,
        action=decision.action,
        args=dict(args or {}),
        context=context,
        allowed=decision.allowed,
        rules_checked=list(decision.rules_checked),
        blocking_rules=list(decision.blocking_rules),
        gms_scores={k: float(v) for k, v in decision.gms_scores.items()},
        reason=decision.reason,
        quorum_decision=verdict.decision,
        quorum_required=verdict.quorum_required,
        n_layers_blocking=verdict.n_layers_blocking,
        n_layers_signal_available=verdict.n_layers_signal_available,
        layer1_block=verdict.layer1_block,
        layer2_block=verdict.layer2_block,
        layer3_block=verdict.layer3_block,
        layer4_block=verdict.layer4_block,
        quorum_evidence=verdict.evidence,
        reason_chain=chain.to_dict(),
        timestamp=timestamp,
    )


def replay_audit_log(record: dict | PolicyAuditRecord) -> str:
    """Re-derive verdict from a record. Symmetric to ``to_audit_record``."""
    if isinstance(record, dict):
        record = PolicyAuditRecord(**record)
    return record.replay()
