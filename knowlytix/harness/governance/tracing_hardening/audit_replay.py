# SPDX-License-Identifier: Apache-2.0
"""Replay-regression battery for audit-trail summary metrics.

The hash chain proves the trail was not edited; the cross-reference
attestor proves the trail matches the runtime decisions.  Neither
proves that the \emph{summary metrics} computed over the trail ---
block rate, average plausibility, claim pass rate, lowest-margin
step --- reproduce byte-identically across runs.  A reporting bug
that subtly changes how block-rate is rounded, or that rebins
plausibility scores, can pass both attestors and still produce a
different summary number on every run.

This module captures a deterministic snapshot of the metrics so a
future CI run can assert ``the report is byte-identical to the
recorded baseline.''  The pattern mirrors :mod:`replay` (Ch.10),
:mod:`policy_replay` (Ch.12), and :mod:`contract_replay` (Ch.13);
the same shape across runtimes lets operators reason about all
four regression batteries the same way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from knowlytix.harness.governance.tracing import AuditPacket


@dataclass
class AuditReplayResult:
    """Per-metric audit snapshot.

    ``per_metric`` maps a metric name (block rate, claim pass rate,
    avg plausibility, etc.) to its canonicalized value.  ``head_hash``
    records the hash chain head so a comparison can verify the two
    snapshots came from the same trail before checking metric drift.
    """

    run_id: str
    n_packets: int
    head_hash: str
    per_metric: dict[str, float]


class AuditReplayRegressionError(AssertionError):
    """Raised when audit replay diverges from the baseline."""


def replay_audit_metrics(
    run_id: str,
    packets: list[AuditPacket],
) -> AuditReplayResult:
    """Compute deterministic summary metrics from a sealed audit trail.

    Metrics are rounded to 6 decimals where applicable so two
    replays produce byte-identical floats regardless of the order
    in which list arithmetic was performed.
    """
    n = len(packets)
    blocked = sum(1 for p in packets if p.policy_decision == "blocked")
    allowed = n - blocked
    block_rate = round(blocked / n, 6) if n else 0.0

    plaus_scores = [
        p.gms_scores.get("plausibility")
        for p in packets
        if p.gms_scores.get("plausibility") is not None
    ]
    avg_plaus = (
        round(sum(plaus_scores) / len(plaus_scores), 6)
        if plaus_scores else 0.0
    )

    n_claims = 0
    n_passed = 0
    for p in packets:
        if p.verification is not None:
            n_claims += p.verification.get("n_claims", 0)
            n_passed += p.verification.get("n_passed", 0)
    claim_pass_rate = (
        round(n_passed / n_claims, 6) if n_claims else 0.0
    )

    total_exec_ms = round(
        sum(p.execution_time_ms for p in packets), 6,
    )

    n_state_advances = sum(
        1 for p in packets if p.state_after and p.state_before != p.state_after
    )

    head = packets[-1].content_hash if packets else ""

    return AuditReplayResult(
        run_id=run_id,
        n_packets=n,
        head_hash=head,
        per_metric={
            "n_packets":          float(n),
            "n_allowed":          float(allowed),
            "n_blocked":          float(blocked),
            "block_rate":         block_rate,
            "avg_plausibility":   avg_plaus,
            "n_claims":           float(n_claims),
            "claim_pass_rate":    claim_pass_rate,
            "total_exec_ms":      total_exec_ms,
            "n_state_advances":   float(n_state_advances),
        },
    )


def assert_no_audit_regression(
    baseline: AuditReplayResult,
    candidate: AuditReplayResult,
) -> None:
    """Raise unless the candidate metrics match the baseline byte-for-byte.

    Comparison checks (in order):

    1. Same ``run_id`` (mismatched ids indicate the snapshots were
       taken from different runs; comparison is meaningless).
    2. Same ``head_hash`` (the trail itself is identical at the
       chain level; a different head means upstream tampering and
       the metric comparison is moot).
    3. Per-metric values match exactly.
    """
    if baseline.run_id != candidate.run_id:
        raise AuditReplayRegressionError(
            f"run_id differs: baseline={baseline.run_id!r}, "
            f"candidate={candidate.run_id!r}"
        )
    if baseline.head_hash != candidate.head_hash:
        raise AuditReplayRegressionError(
            f"chain head differs: baseline={baseline.head_hash[:12]}..., "
            f"candidate={candidate.head_hash[:12]}... --- the underlying "
            f"trail was modified between baseline and candidate."
        )

    diffs: list[tuple[str, float, float]] = []
    for k in sorted(set(baseline.per_metric) | set(candidate.per_metric)):
        b = baseline.per_metric.get(k)
        c = candidate.per_metric.get(k)
        if b != c:
            diffs.append((k, b if b is not None else float("nan"),
                          c if c is not None else float("nan")))

    if diffs:
        head = "\n  ".join(
            f"{name}: baseline={b} candidate={c}"
            for name, b, c in diffs[:10]
        )
        raise AuditReplayRegressionError(
            f"{len(diffs)} metrics diverge from baseline:\n  {head}"
        )


def metrics_fingerprint(result: AuditReplayResult) -> str:
    """Stable SHA-256 over the canonical JSON of the metrics dict.

    The bundle stores this so a runtime can verify metric integrity
    without re-running the replay.
    """
    payload = {
        "run_id": result.run_id,
        "n_packets": result.n_packets,
        "head_hash": result.head_hash,
        "per_metric": dict(sorted(result.per_metric.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
