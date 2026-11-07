# SPDX-License-Identifier: Apache-2.0
"""Bundle-replay regression for policy decisions.

Mirrors :mod:`governed_agent.replay` for the policy surface. A frozen
labeled cohort produces a SHA-256 hash over canonical decision tuples;
a deployment refuses to ship a new bundle whose replay-battery hash
diverges from the blessed value, except after explicit deployment-team
review.

Workflow:

  1. Build a :class:`LabeledDecision` cohort once
     (``adult_income_labeled_cohort()``).
  2. Calibrate a bundle and run::

        old_result = replay_policy_battery(old_engine, cohort)
        new_result = replay_policy_battery(new_engine, cohort)
        delta      = compare_policy_replays(old_result, new_result)

  3. If ``delta.has_regressions``, the CI ship-gate refuses the new
     bundle. The deployment team reviews the per-probe verdict
     changes, updates the cohort or rolls back.

The cohort + the replay result are JSON-serialisable so they
diff cleanly in version control.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.policy_validation import LabeledDecision
    from knowlytix.harness.testing.policy import PolicyEngine


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------


@dataclass
class PolicyReplayResult:
    """Audit-grade summary of running a labeled cohort.

    The single ``cohort_hash`` is what gets checked into the
    deployment alongside the bundle: any change to the engine, the
    rule set, the calibration, or the cohort that flips a single
    verdict flips the hash and is caught at ship time.
    """

    cohort_size: int
    bundle_id: str
    bundle_signature: str
    snapshots: list[dict] = field(default_factory=list)
    cohort_hash: str = ""
    n_decision_matches: int = 0
    n_blocking_rules_matches: int = 0

    def summary(self) -> str:
        return (
            f"PolicyReplayResult(n={self.cohort_size}, "
            f"bundle={self.bundle_id[:8]}, "
            f"hash={self.cohort_hash}, "
            f"decision_matches={self.n_decision_matches}/"
            f"{self.cohort_size}, "
            f"blocking_rules_matches={self.n_blocking_rules_matches}/"
            f"{self.cohort_size})"
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "cohort_size": self.cohort_size,
                "bundle_id": self.bundle_id,
                "bundle_signature": self.bundle_signature,
                "snapshots": self.snapshots,
                "cohort_hash": self.cohort_hash,
                "n_decision_matches": self.n_decision_matches,
                "n_blocking_rules_matches": self.n_blocking_rules_matches,
            }, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyReplayResult":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


def policy_decision_snapshot(
    probe: "LabeledDecision",
    decision,
    verdict,
) -> dict:
    """Audit-grade summary of one (probe, decision, attestor) triple.

    Records only the deterministic fields. Floating-point gms_scores
    are quantised to 4 decimal places so the cohort hash is stable
    across CUDA non-determinism within calibration tolerance.
    """
    return {
        "probe_id": probe.probe_id,
        "agent": probe.agent,
        "action": probe.action,
        "context": probe.context,
        "expected_decision": probe.expected_decision,
        "source_class": probe.source_class,
        "actual_decision": verdict.decision,
        "blocking_rules": sorted(decision.blocking_rules),
        "n_blocking": len(decision.blocking_rules),
        "n_layers_blocking": verdict.n_layers_blocking,
        "n_layers_signal_available": verdict.n_layers_signal_available,
        "plausibility": (
            round(float(decision.gms_scores["plausibility"]), 4)
            if "plausibility" in decision.gms_scores else None
        ),
    }


def cohort_hash(snapshots: list[dict]) -> str:
    """Deterministic SHA-256 over canonical snapshots."""
    payload = json.dumps(snapshots, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def replay_policy_battery(
    engine: "PolicyEngine",
    cohort: list["LabeledDecision"],
    *,
    quorum_required: int = 2,
    bundle_id: str = "",
    bundle_signature: str = "",
) -> PolicyReplayResult:
    """Run every labeled probe and produce a single audit hash.

    Use this on every bundle rebuild: hash the cohort once, save
    alongside the bundle, refuse to ship a new bundle whose
    replay-battery hash differs.
    """
    from knowlytix.harness.governance.policy_calibration import check_with_quorum

    snapshots: list[dict] = []
    n_decision = 0
    n_rules = 0

    for probe in cohort:
        decision, verdict = check_with_quorum(
            engine, probe.action, probe.args, probe.context,
            quorum_required=quorum_required,
        )
        snap = policy_decision_snapshot(probe, decision, verdict)
        snapshots.append(snap)
        if snap["actual_decision"] == probe.expected_decision:
            n_decision += 1
        # blocking-rules match is informational: we record whether the
        # rules that fired this run are the same set as the cohort
        # snapshot would expect (cohort doesn't carry expected rule
        # IDs, so this is set true when at least one rule fired for
        # any expected-BLOCK probe).
        if (probe.expected_decision == "BLOCK"
                and snap["n_blocking"] > 0):
            n_rules += 1
        elif probe.expected_decision == "ALLOW":
            n_rules += 1  # not applicable; pass-through

    return PolicyReplayResult(
        cohort_size=len(cohort),
        bundle_id=bundle_id,
        bundle_signature=bundle_signature,
        snapshots=snapshots,
        cohort_hash=cohort_hash(snapshots),
        n_decision_matches=n_decision,
        n_blocking_rules_matches=n_rules,
    )


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass
class PolicyReplayDelta:
    """Per-probe changes between two replay runs."""

    n_total: int
    n_unchanged: int
    n_decision_changed: int
    n_blocking_rules_drifted: int
    old_hash: str
    new_hash: str
    changes: list[tuple[str, dict, dict]] = field(default_factory=list)
    rule_drifts: list[tuple[str, list[str], list[str]]] = field(
        default_factory=list,
    )

    @property
    def has_regressions(self) -> bool:
        return self.n_decision_changed > 0

    @property
    def hash_matches(self) -> bool:
        return self.old_hash == self.new_hash

    def summary(self) -> str:
        return (
            f"PolicyReplayDelta(n={self.n_total}, "
            f"unchanged={self.n_unchanged}, "
            f"decision_changed={self.n_decision_changed}, "
            f"rule_drifted={self.n_blocking_rules_drifted}, "
            f"hash_match={self.hash_matches})"
        )


def compare_policy_replays(
    old: PolicyReplayResult,
    new: PolicyReplayResult,
) -> PolicyReplayDelta:
    """Diff two replay results; report changed verdicts and rule drift."""
    if old.cohort_size != new.cohort_size:
        raise ValueError(
            f"Cohort size mismatch: old={old.cohort_size}, "
            f"new={new.cohort_size}"
        )
    by_probe_old = {s["probe_id"]: s for s in old.snapshots}
    by_probe_new = {s["probe_id"]: s for s in new.snapshots}
    common = sorted(set(by_probe_old) & set(by_probe_new))

    n_total = len(common)
    n_unchanged = 0
    changes: list[tuple[str, dict, dict]] = []
    rule_drifts: list[tuple[str, list[str], list[str]]] = []

    for pid in common:
        os = by_probe_old[pid]
        ns = by_probe_new[pid]
        if os["actual_decision"] != ns["actual_decision"]:
            changes.append((pid, os, ns))
            continue
        if os["blocking_rules"] != ns["blocking_rules"]:
            rule_drifts.append(
                (pid, os["blocking_rules"], ns["blocking_rules"])
            )
            continue
        n_unchanged += 1

    return PolicyReplayDelta(
        n_total=n_total,
        n_unchanged=n_unchanged,
        n_decision_changed=len(changes),
        n_blocking_rules_drifted=len(rule_drifts),
        old_hash=old.cohort_hash,
        new_hash=new.cohort_hash,
        changes=changes,
        rule_drifts=rule_drifts,
    )


class PolicyReplayRegressionError(RuntimeError):
    """Replay diff exceeds allowed regressions."""


def assert_no_policy_regression(
    delta: PolicyReplayDelta,
    *,
    allow_rule_drift: bool = True,
) -> None:
    """Strict gate: raise if any verdict changed (and optionally rule drift)."""
    problems: list[str] = []
    if delta.n_decision_changed > 0:
        sample = delta.changes[:5]
        problems.append(
            f"{delta.n_decision_changed} decision change(s); "
            f"first {len(sample)}: "
            + ", ".join(
                f"{pid}: {os['actual_decision']}->{ns['actual_decision']}"
                for pid, os, ns in sample
            )
        )
    if (not allow_rule_drift) and delta.n_blocking_rules_drifted > 0:
        problems.append(
            f"{delta.n_blocking_rules_drifted} probe(s) drifted "
            f"in blocking_rules"
        )
    if problems:
        raise PolicyReplayRegressionError("; ".join(problems))
