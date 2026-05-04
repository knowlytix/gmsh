# SPDX-License-Identifier: Apache-2.0
"""Audit log + reason chain for multi-agent decisions.

Each tool-call decision can be serialized to a record that:

  * names the bundle + cohort hash bound to the orchestrator
  * captures the (role, tool, args_hash) tuple
  * captures the structural decision and (optionally) the
    co-attestor verdict
  * is JSON-serializable and replayable without the orchestrator
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from knowlytix.harness.governance.agents.multi_agent import (
    MultiAgentOrchestrator,
)


def _hash_args(args: dict) -> str:
    payload = json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class MultiAgentReasonChain:
    blocking_reasons: list[str] = field(default_factory=list)
    flag_reasons: list[str] = field(default_factory=list)


@dataclass
class MultiAgentAuditRecord:
    """JSON-serializable audit record for a single tool-call decision."""

    timestamp: float
    role_name: str
    tool_name: str
    args_hash: str
    structural_passed: bool
    decision: str   # "allow" | "block" | "deny"
    reason_chain: dict
    co_attestor_verdict: dict | None = None
    bundle_id: str = ""
    cohort_hash: str = ""
    signing_method: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)


def to_multi_agent_audit_record(
    orch: MultiAgentOrchestrator,
    role_name: str,
    tool_name: str,
    args: dict | None = None,
    *,
    co_attestor_verdict: Any = None,
    bundle: Any = None,
) -> MultiAgentAuditRecord:
    """Build an audit record for a (role, tool, args) decision."""
    args = args or {}
    agent = orch.agents.get(role_name)
    if agent is None:
        return MultiAgentAuditRecord(
            timestamp=time.time(),
            role_name=role_name,
            tool_name=tool_name,
            args_hash=_hash_args(args),
            structural_passed=False,
            decision="deny",
            reason_chain={
                "blocking_reasons": [f"unknown role '{role_name}'"],
                "flag_reasons": [],
            },
            co_attestor_verdict=None,
            bundle_id=(getattr(bundle, "bundle_id", "")
                       if bundle else ""),
            cohort_hash=(getattr(bundle, "cohort_hash", "")
                         if bundle else ""),
            signing_method=(getattr(bundle, "signing_method", "")
                            if bundle else ""),
        )
    structural = agent.can_execute(tool_name)
    if structural:
        decision = "allow"
        blocking_reasons: list[str] = []
    elif tool_name in agent.role.forbidden_tools:
        decision = "block"
        blocking_reasons = [f"{role_name} role explicitly forbids "
                            f"'{tool_name}'"]
    else:
        decision = "deny"
        blocking_reasons = [f"{role_name} role does not allow "
                            f"'{tool_name}' (default-deny)"]

    flag_reasons: list[str] = []
    co_dict: dict | None = None
    if co_attestor_verdict is not None:
        flag_reasons = list(getattr(
            co_attestor_verdict, "flag_reasons", []))
        co_dict = {
            "verdict": getattr(co_attestor_verdict, "verdict", None),
            "structural_passed": bool(getattr(
                co_attestor_verdict, "structural_passed", False)),
            "psi_value": float(getattr(
                co_attestor_verdict, "psi_value", float("nan"))),
            "blocking_reasons": list(getattr(
                co_attestor_verdict, "blocking_reasons", [])),
            "flag_reasons": flag_reasons,
        }

    return MultiAgentAuditRecord(
        timestamp=time.time(),
        role_name=role_name,
        tool_name=tool_name,
        args_hash=_hash_args(args),
        structural_passed=structural,
        decision=decision,
        reason_chain={
            "blocking_reasons": blocking_reasons,
            "flag_reasons": flag_reasons,
        },
        co_attestor_verdict=co_dict,
        bundle_id=(getattr(bundle, "bundle_id", "") if bundle else ""),
        cohort_hash=(getattr(bundle, "cohort_hash", "")
                     if bundle else ""),
        signing_method=(getattr(bundle, "signing_method", "")
                        if bundle else ""),
    )


def replay_multi_agent_audit_log(record: dict | MultiAgentAuditRecord) -> str:
    """Reconstruct the verdict from a saved audit record alone."""
    if isinstance(record, MultiAgentAuditRecord):
        d = record.to_dict()
    else:
        d = record
    decision = d.get("decision", "deny")
    co = d.get("co_attestor_verdict") or {}
    verdict = co.get("verdict")
    if verdict == "block" or decision in ("block", "deny"):
        return decision.upper()
    if verdict == "flag":
        return "FLAG"
    return "ALLOW"
