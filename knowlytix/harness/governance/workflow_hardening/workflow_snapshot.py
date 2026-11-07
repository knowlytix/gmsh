# SPDX-License-Identifier: Apache-2.0
"""Deterministic workflow snapshot for end-to-end replay regression.

The Adult Income workflow in Ch.16 produces a `PhaseResult` per
phase plus an aggregate `OverallVerdict`. Two of these views are
non-deterministic by construction (CUDA-trained model accuracies
vary at the 4th-5th decimal across runs); the rest --- extracted
triple counts, ENM keys, claim types, policy decisions, release
gate verdict --- must reproduce byte-identically.

This module captures the reproducible view as a `WorkflowSnapshot`
with a SHA-256 fingerprint over the canonical JSON. The runtime
runs the workflow, takes a snapshot, and gates against the
previously-blessed fingerprint. Any change that flips a claim
verdict, blocks an additional tool, or changes the release gate
shifts the fingerprint and is caught at ship time.

Mirrors :mod:`governed_agent.policy_replay` and
:mod:`governed_agent.tracing_hardening.audit_replay` for the
workflow surface.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from knowlytix.harness.governance.agents.data_agent import PhaseResult


# ---------------------------------------------------------------------------
# Per-phase canonical view
# ---------------------------------------------------------------------------


@dataclass
class PhaseSnapshot:
    """Deterministic per-phase summary, sortable for stable hashing."""

    phase_name: str
    tools_called: list[str]              # alphabetical-sorted
    n_triples: int
    n_enm: int
    n_policy_blocks: int
    n_claims: int
    claim_type_counts: dict[str, int]    # {"NUMERIC": 12, "FACT": 5}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_phase_result(cls, pr: "PhaseResult") -> "PhaseSnapshot":
        types = Counter(c.type.name for c in pr.claims)
        return cls(
            phase_name=pr.phase_name,
            tools_called=sorted(pr.tools_called),
            n_triples=pr.triples_extracted,
            n_enm=pr.enm_entries,
            n_policy_blocks=pr.policy_blocks,
            n_claims=len(pr.claims),
            claim_type_counts=dict(sorted(types.items())),
        )


# ---------------------------------------------------------------------------
# Aggregate workflow snapshot
# ---------------------------------------------------------------------------


@dataclass
class WorkflowSnapshot:
    """Audit-grade summary of a single workflow execution.

    The fingerprint is computed over the canonical JSON. Two
    consecutive executions of the same workflow must produce the
    same fingerprint or the deployment has drifted at the
    *governance* layer (CUDA-trained accuracies are reported in
    `floating_metrics` with explicit tolerance, but they do not
    contribute to the fingerprint).
    """

    workflow_id: str
    phase_snapshots: list[PhaseSnapshot]
    cumulative_triples: int
    cumulative_enm: int
    cumulative_blocks: int
    cumulative_claims: int
    n_passed: int
    n_total: int
    pass_rate: float                     # rounded to 6 decimals
    overall_status: str                  # "PASS" | "FAIL"
    severity_counts: dict[str, int]
    release_gate: str                    # APPROVED | CONDITIONAL | BLOCKED
    floating_metrics: dict[str, float] = field(default_factory=dict)
    # Floating metrics (model accuracies, AIR values, etc.) are
    # captured but excluded from the fingerprint --- they vary at
    # the 4th-5th decimal under CUDA non-determinism. The fingerprint
    # gates the *governance* layer; floating metrics get their own
    # tolerance check.

    def canonical_payload(self) -> bytes:
        """Bytes used for fingerprinting. Excludes floating_metrics."""
        d = {
            "workflow_id": self.workflow_id,
            "phase_snapshots": [
                ps.to_dict() for ps in self.phase_snapshots
            ],
            "cumulative_triples": self.cumulative_triples,
            "cumulative_enm": self.cumulative_enm,
            "cumulative_blocks": self.cumulative_blocks,
            "cumulative_claims": self.cumulative_claims,
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "pass_rate": self.pass_rate,
            "overall_status": self.overall_status,
            "severity_counts": dict(sorted(self.severity_counts.items())),
            "release_gate": self.release_gate,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()

    def summary(self) -> str:
        return (
            f"WorkflowSnapshot("
            f"id={self.workflow_id}, "
            f"phases={len(self.phase_snapshots)}, "
            f"claims={self.cumulative_claims}, "
            f"pass={self.n_passed}/{self.n_total}, "
            f"rate={self.pass_rate}, "
            f"gate={self.release_gate}, "
            f"fingerprint={self.fingerprint()[:16]}...)"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_workflow_snapshot(
    *,
    workflow_id: str,
    phase_results: list["PhaseResult"],
    overall_verdict: Any,
    release_gate: str,
    floating_metrics: dict[str, float] | None = None,
) -> WorkflowSnapshot:
    """Pack phase results + verdict into a deterministic snapshot."""
    phase_snaps = [PhaseSnapshot.from_phase_result(p) for p in phase_results]
    cumulative_triples = sum(p.n_triples for p in phase_snaps)
    cumulative_enm = sum(p.n_enm for p in phase_snaps)
    cumulative_blocks = sum(p.n_policy_blocks for p in phase_snaps)
    cumulative_claims = sum(p.n_claims for p in phase_snaps)

    severity_counts: dict[str, int] = {}
    if hasattr(overall_verdict, "severity_distribution"):
        severity_counts = {
            k: int(v) for k, v in
            sorted(overall_verdict.severity_distribution.items())
        }

    return WorkflowSnapshot(
        workflow_id=workflow_id,
        phase_snapshots=phase_snaps,
        cumulative_triples=cumulative_triples,
        cumulative_enm=cumulative_enm,
        cumulative_blocks=cumulative_blocks,
        cumulative_claims=cumulative_claims,
        n_passed=int(getattr(overall_verdict, "n_passed", 0)),
        n_total=int(getattr(overall_verdict, "n_total", 0)),
        pass_rate=round(
            float(getattr(overall_verdict, "pass_rate", 0.0)), 6
        ),
        overall_status=str(getattr(overall_verdict, "overall_status", "PASS")),
        severity_counts=severity_counts,
        release_gate=release_gate,
        floating_metrics=dict(floating_metrics or {}),
    )
