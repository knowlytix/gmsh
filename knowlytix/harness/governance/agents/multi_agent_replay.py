# SPDX-License-Identifier: Apache-2.0
"""Replay regression for the multi-agent governance surface.

A canonical decision-stream is hashed via SHA-256 over each
``(role, tool, expected, observed)`` quadruple. Two replays of the
same orchestrator + cohort must produce identical hashes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from knowlytix.harness.governance.agents.multi_agent import MultiAgentOrchestrator
from knowlytix.harness.governance.agents.multi_agent_validation import (
    LabeledMultiAgentDecision,
    _observe,
)


def _snapshot(
    orch: MultiAgentOrchestrator,
    cohort: list[LabeledMultiAgentDecision],
) -> list[dict]:
    out = []
    for p in cohort:
        observed = _observe(orch, p)
        out.append({
            "probe_id": p.probe_id,
            "role": p.role_name,
            "tool": p.tool_name,
            "expected": p.expected_decision,
            "observed": observed,
            "match": observed == p.expected_decision,
        })
    return out


def _hash_snapshots(snapshots: list[dict]) -> str:
    canon = json.dumps(
        sorted(snapshots, key=lambda d: d["probe_id"]),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


@dataclass
class MultiAgentReplayResult:
    n_probes: int
    snapshots: list[dict]
    snapshots_hash: str
    bundle_id: str = ""
    cohort_hash: str = ""


def replay_multi_agent_battery(
    orch: MultiAgentOrchestrator,
    cohort: list[LabeledMultiAgentDecision],
    *,
    bundle_id: str = "",
    cohort_hash: str = "",
) -> MultiAgentReplayResult:
    """Run cohort over orchestrator; return SHA-256 hash over decisions."""
    snaps = _snapshot(orch, cohort)
    return MultiAgentReplayResult(
        n_probes=len(snaps),
        snapshots=snaps,
        snapshots_hash=_hash_snapshots(snaps),
        bundle_id=bundle_id,
        cohort_hash=cohort_hash,
    )


@dataclass
class MultiAgentReplayDelta:
    n_probes: int
    n_divergent: int
    divergent_probe_ids: list[str] = field(default_factory=list)
    matches: bool = True


def compare_multi_agent_replays(
    a: MultiAgentReplayResult, b: MultiAgentReplayResult,
) -> MultiAgentReplayDelta:
    by_id_a = {s["probe_id"]: s for s in a.snapshots}
    by_id_b = {s["probe_id"]: s for s in b.snapshots}
    divergent = []
    for pid in sorted(by_id_a.keys() | by_id_b.keys()):
        if by_id_a.get(pid) != by_id_b.get(pid):
            divergent.append(pid)
    return MultiAgentReplayDelta(
        n_probes=max(a.n_probes, b.n_probes),
        n_divergent=len(divergent),
        divergent_probe_ids=divergent,
        matches=(a.snapshots_hash == b.snapshots_hash
                 and len(divergent) == 0),
    )


class MultiAgentReplayRegressionError(RuntimeError):
    """Replay regression detected: replay hash diverged from blessed."""


def assert_no_multi_agent_regression(
    blessed_hash: str,
    current: MultiAgentReplayResult,
    *,
    blessed_snapshots: list[dict] | None = None,
) -> None:
    if current.snapshots_hash == blessed_hash:
        return
    if blessed_snapshots is not None:
        b = MultiAgentReplayResult(
            n_probes=len(blessed_snapshots),
            snapshots=blessed_snapshots,
            snapshots_hash=blessed_hash,
        )
        delta = compare_multi_agent_replays(b, current)
        diverging = ", ".join(delta.divergent_probe_ids[:5])
        raise MultiAgentReplayRegressionError(
            f"Replay regression: blessed={blessed_hash[:16]} "
            f"current={current.snapshots_hash[:16]}; "
            f"{delta.n_divergent} divergent probes; "
            f"first 5: [{diverging}]"
        )
    raise MultiAgentReplayRegressionError(
        f"Replay regression: blessed={blessed_hash[:16]} "
        f"current={current.snapshots_hash[:16]}"
    )


def write_blessed_hash(
    path: str | Path, replay: MultiAgentReplayResult,
    *, approver: str = "", reason: str = "",
) -> None:
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
