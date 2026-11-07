# SPDX-License-Identifier: Apache-2.0
"""Replay regression for the tiered release gate.

A canonical run-stream is hashed via SHA-256 over each
:class:`GateResult`'s ``(tier, passed, blocking_reasons)`` triple
plus the per-check fail-finite flags. Two runs over the same gate
+ cohort must produce identical hashes; divergence is named per
probe and raised as :class:`ReleaseGateReplayRegressionError`.

The blessed hash file is written by Lab 15.4's runbook to
``governance_monograph/expected_release_gate_replay_hash.txt``.
The CI ship gate refuses to ship a bundle whose replay battery
hash diverges from the blessed value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from knowlytix.harness.governance.release_gate_validation import LabeledRun
from knowlytix.harness.governance.release_gates import (
    TieredReleaseGate,
)

# ---------------------------------------------------------------------------
# Snapshot + hash
# ---------------------------------------------------------------------------

def _snapshot_one(
    gate: TieredReleaseGate, lr: LabeledRun,
) -> dict:
    """Snapshot a single (gate, probe) decision."""
    results = gate.check_all_tiers(lr.metrics)
    snap = {
        "run_id": lr.run_id,
        "tier_truth": lr.tier_truth,
        "source_class": lr.source_class,
        "metrics": {
            "accuracy": lr.metrics.accuracy,
            "confidence": lr.metrics.confidence,
            "robustness_drop": lr.metrics.robustness_drop,
            "fairness_air": lr.metrics.fairness_air,
        },
        "decisions": {},
    }
    for tier, r in results.items():
        snap["decisions"][tier] = {
            "passed": bool(r.passed),
            "blocking_reasons": list(r.blocking_reasons),
            "checks": {
                name: {
                    "passed": bool(info.get("passed", False)),
                    "finite": bool(info.get("finite", True)),
                }
                for name, info in r.checks.items()
            },
        }
    return snap


def _hash_snapshots(snapshots: list[dict]) -> str:
    canon = json.dumps(
        sorted(snapshots, key=lambda d: d["run_id"]),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


# ---------------------------------------------------------------------------
# Replay battery
# ---------------------------------------------------------------------------

@dataclass
class ReleaseGateReplayResult:
    n_probes: int
    snapshots: list[dict]
    snapshots_hash: str
    bundle_id: str = ""
    cohort_hash: str = ""


def replay_release_gate_battery(
    gate: TieredReleaseGate,
    cohort: list[LabeledRun],
    *,
    bundle_id: str = "",
    cohort_hash: str = "",
) -> ReleaseGateReplayResult:
    """Run gate over cohort; return SHA-256 hash over per-probe snapshots."""
    snaps = [_snapshot_one(gate, lr) for lr in cohort]
    return ReleaseGateReplayResult(
        n_probes=len(snaps),
        snapshots=snaps,
        snapshots_hash=_hash_snapshots(snaps),
        bundle_id=bundle_id,
        cohort_hash=cohort_hash,
    )


# ---------------------------------------------------------------------------
# Comparison + regression assertion
# ---------------------------------------------------------------------------

@dataclass
class ReleaseGateReplayDelta:
    n_probes: int
    n_divergent: int
    divergent_probe_ids: list[str] = field(default_factory=list)
    matches: bool = True


def compare_release_gate_replays(
    a: ReleaseGateReplayResult,
    b: ReleaseGateReplayResult,
) -> ReleaseGateReplayDelta:
    by_id_a = {s["run_id"]: s for s in a.snapshots}
    by_id_b = {s["run_id"]: s for s in b.snapshots}
    divergent: list[str] = []
    for rid in sorted(by_id_a.keys() | by_id_b.keys()):
        if by_id_a.get(rid) != by_id_b.get(rid):
            divergent.append(rid)
    return ReleaseGateReplayDelta(
        n_probes=max(a.n_probes, b.n_probes),
        n_divergent=len(divergent),
        divergent_probe_ids=divergent,
        matches=(a.snapshots_hash == b.snapshots_hash
                 and len(divergent) == 0),
    )


class ReleaseGateReplayRegressionError(RuntimeError):
    """Replay regression detected: replay hash diverged from blessed."""


def assert_no_release_gate_regression(
    blessed_hash: str,
    current: ReleaseGateReplayResult,
    *,
    blessed_snapshots: list[dict] | None = None,
) -> None:
    """Raise if the current replay hash doesn't match ``blessed_hash``.

    If ``blessed_snapshots`` are provided, also report which probes
    diverged.
    """
    if current.snapshots_hash == blessed_hash:
        return
    if blessed_snapshots is not None:
        blessed_result = ReleaseGateReplayResult(
            n_probes=len(blessed_snapshots),
            snapshots=blessed_snapshots,
            snapshots_hash=blessed_hash,
        )
        delta = compare_release_gate_replays(blessed_result, current)
        diverging = ", ".join(delta.divergent_probe_ids[:5])
        raise ReleaseGateReplayRegressionError(
            f"Replay regression: blessed={blessed_hash[:16]} "
            f"current={current.snapshots_hash[:16]}; "
            f"{delta.n_divergent} divergent probes; "
            f"first 5: [{diverging}]"
        )
    raise ReleaseGateReplayRegressionError(
        f"Replay regression: blessed={blessed_hash[:16]} "
        f"current={current.snapshots_hash[:16]}"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_blessed_hash(path: str | Path, replay: ReleaseGateReplayResult,
                       *, approver: str = "", reason: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshots_hash": replay.snapshots_hash,
        "n_probes": replay.n_probes,
        "bundle_id": replay.bundle_id,
        "cohort_hash": replay.cohort_hash,
        "approver": approver,
        "reason": reason,
    }
    p.write_text(json.dumps(payload, indent=2))


def read_blessed_hash(path: str | Path) -> str:
    p = Path(path)
    payload = json.loads(p.read_text())
    return payload.get("snapshots_hash", "")
