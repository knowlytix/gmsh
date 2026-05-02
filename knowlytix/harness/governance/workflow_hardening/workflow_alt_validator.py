# SPDX-License-Identifier: Apache-2.0
"""Independent second-implementation re-derivation of release gate.

Mirrors `policy_validator_alt` and `audit_alt_validator` for the
workflow surface. The primary release-gate decision is computed
during workflow execution by `generate_governance_report`; this
alt-validator re-derives the verdict from the snapshot's
fingerprinted fields alone, using a different code path (string
table lookup vs imperative branches).

If the two implementations agree on every cohort probe, the
auditor has structural evidence that the gate logic is not a
single-implementation artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.harness.governance.workflow_hardening.workflow_snapshot import (
    WorkflowSnapshot,
)


# ---------------------------------------------------------------------------
# Truth table for the release gate
# ---------------------------------------------------------------------------


# Sorted list of (any_block, pass_rate_bucket) -> gate.
# Buckets:
#   0: pass_rate < 0.80
#   1: 0.80 <= pass_rate < 0.95
#   2: pass_rate >= 0.95
_TRUTH_TABLE = {
    (True,  0): "BLOCKED",
    (True,  1): "BLOCKED",
    (True,  2): "BLOCKED",
    (False, 0): "BLOCKED",
    (False, 1): "CONDITIONAL",
    (False, 2): "APPROVED",
}


def _pass_rate_bucket(pass_rate: float) -> int:
    if pass_rate >= 0.95:
        return 2
    if pass_rate >= 0.80:
        return 1
    return 0


def alt_release_gate(
    *,
    cumulative_blocks: int,
    pass_rate: float,
) -> str:
    """Re-derive the release gate from the snapshot's pass_rate + blocks.

    This is intentionally not a thin wrapper around the primary
    `generate_governance_report` logic; it consults a string
    table keyed by `(any_block, bucket)` so disagreement reveals
    a logic divergence between the two implementations.
    """
    return _TRUTH_TABLE[(cumulative_blocks > 0, _pass_rate_bucket(pass_rate))]


# ---------------------------------------------------------------------------
# Agreement check
# ---------------------------------------------------------------------------


@dataclass
class WorkflowAltAgreementReport:
    n_snapshots: int
    n_agree: int
    disagreements: list[tuple[str, str, str]] = field(default_factory=list)
    # Each entry: (workflow_id, primary_gate, alt_gate)

    @property
    def agreement_rate(self) -> float:
        if self.n_snapshots == 0:
            return 0.0
        return self.n_agree / self.n_snapshots

    @property
    def all_agree(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        rows = [
            "WorkflowAltAgreementReport(",
            f"  n_snapshots={self.n_snapshots}",
            f"  n_agree={self.n_agree}",
            f"  agreement_rate={self.agreement_rate:.4f}",
            f"  disagreements={len(self.disagreements)}",
        ]
        for wid, primary, alt in self.disagreements[:5]:
            rows.append(
                f"    {wid}: primary={primary} alt={alt}"
            )
        rows.append(")")
        return "\n".join(rows)


def assert_workflow_alt_agreement(
    snapshots: list[WorkflowSnapshot],
) -> WorkflowAltAgreementReport:
    """Run primary + alt-validator on every snapshot; report agreement."""
    n_total = 0
    n_agree = 0
    disagreements: list[tuple[str, str, str]] = []
    for snap in snapshots:
        n_total += 1
        alt = alt_release_gate(
            cumulative_blocks=snap.cumulative_blocks,
            pass_rate=snap.pass_rate,
        )
        if alt == snap.release_gate:
            n_agree += 1
        else:
            disagreements.append((snap.workflow_id, snap.release_gate, alt))
    return WorkflowAltAgreementReport(
        n_snapshots=n_total,
        n_agree=n_agree,
        disagreements=disagreements,
    )
