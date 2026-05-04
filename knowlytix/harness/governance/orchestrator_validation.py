# SPDX-License-Identifier: Apache-2.0
"""Held-out / out-of-distribution validation of a GovernedOrchestrator.

The single-run output of an orchestrator (one ``OrchestratorResult``)
is informative but does not characterize behavior under shift.
Bringing the orchestrator to a deployment-defensible standard
requires a labeled cohort drawn from a *different* distribution
than the calibration corpus, run through the orchestrator, with
per-terminal-state confusion and FAR / FRR reported.

This module mirrors the Ch 6 :func:`validate_gate` for the
orchestrator surface.  An auditor's first question is "where are
your failures?" --- this validator produces them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.harness.governance.orchestrator import (
        GovernedOrchestrator, Scenario,
    )


TerminalState = Literal["completed", "failed", "aborted"]


@dataclass
class LabeledPlan:
    """One ground-truth plan for end-to-end orchestrator validation.

    ``expected_state`` is the terminal state the orchestrator should
    reach when the plan is well-formed for the deployment context.
    ``expected_pass_rate`` is the lowest acceptable verification
    pass-rate (for COMPLETED plans).  ``expected_violations`` is
    the structural violation count (for FAILED / ABORTED plans).
    """

    plan: list[str]
    scenario: "Scenario"
    expected_state: TerminalState
    expected_pass_rate: float = 1.0
    expected_violations: int = 0
    provenance: str = ""


@dataclass
class OrchestratorOODReport:
    """Validation outcome on a held-out cohort.

    Fields:

      n_total          : plans evaluated
      per_state_n      : {expected_state: count}
      confusion        : {(expected, actual): count}
      far_per_state    : false-accept rate per non-COMPLETED state
                         (we computed COMPLETED when truth was not)
      frr_per_state    : false-reject rate per state
                         (we did NOT reach the expected state)
      pass_rate_ece    : expected calibration error of pass_rate on
                         COMPLETED plans
      abstain_rate     : fraction the orchestrator routed to ABORTED
      review_rate      : fraction routed to FAILED with high pass_rate
      mean_actual_pass : average actual pass_rate across COMPLETED runs
    """

    n_total: int
    per_state_n: dict[str, int]
    confusion: dict[tuple[str, str], int]
    far_per_state: dict[str, float]
    frr_per_state: dict[str, float]
    pass_rate_ece: float
    abstain_rate: float
    review_rate: float
    mean_actual_pass: float
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        rows = [
            f"n_total={self.n_total}",
            f"per_state_n={self.per_state_n}",
            f"abstain_rate={self.abstain_rate:.3f}",
            f"review_rate={self.review_rate:.3f}",
            f"pass_rate_ece={self.pass_rate_ece:.3f}",
            f"mean_actual_pass={self.mean_actual_pass:.3f}",
        ]
        if self.far_per_state:
            rows.append("far_per_state=" + ", ".join(
                f"{k}:{v:.3f}" for k, v in self.far_per_state.items()))
        if self.frr_per_state:
            rows.append("frr_per_state=" + ", ".join(
                f"{k}:{v:.3f}" for k, v in self.frr_per_state.items()))
        return "OrchestratorOODReport(\n  " + "\n  ".join(rows) + "\n)"


def _expected_calibration_error(
    p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10,
) -> float:
    """Standard ECE on a binary outcome."""
    if len(p_pred) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_pred)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p_pred >= edges[i]) & (p_pred <= edges[i + 1])
        else:
            mask = (p_pred >= edges[i]) & (p_pred < edges[i + 1])
        if not mask.any():
            continue
        bucket_p = p_pred[mask].mean()
        bucket_y = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bucket_p - bucket_y)
    return float(ece)


def validate_orchestrator(
    orch: "GovernedOrchestrator",
    cohort: Iterable[LabeledPlan],
    *,
    fresh_memory_factory=None,
) -> OrchestratorOODReport:
    """Run the full GovernedOrchestrator over a labeled cohort.

    ``fresh_memory_factory(plan_name)`` is called before each run to
    reset memory state when needed; pass ``None`` to reuse the
    orchestrator's existing memory.  Reusing memory across plans
    shares accumulated knowledge --- realistic for one user session,
    not for independent calibration.
    """
    cohort = list(cohort)
    if not cohort:
        raise ValueError("Empty cohort")

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    per_state_n: Counter[str] = Counter()
    actual_pass_collect: list[float] = []
    completed_pass_for_ece: list[float] = []
    completed_truth_for_ece: list[int] = []
    n_aborted = 0
    n_failed_high_pass = 0

    for lp in cohort:
        per_state_n[lp.expected_state] += 1
        if fresh_memory_factory is not None:
            mem = fresh_memory_factory(lp.provenance or "ood")
            orch.memory = mem
        result = orch.run(lp.plan, lp.scenario)
        actual_state = result.state.value
        confusion[(lp.expected_state, actual_state)] += 1
        actual_pass = (result.verdict.pass_rate
                       if result.verdict else 0.0)
        actual_pass_collect.append(actual_pass)
        if lp.expected_state == "completed":
            completed_pass_for_ece.append(actual_pass)
            completed_truth_for_ece.append(
                1 if actual_state == "completed" else 0,
            )
        if actual_state == "aborted":
            n_aborted += 1
        if (actual_state == "failed"
                and actual_pass >= 0.80):
            n_failed_high_pass += 1

    # FAR per state: how often was a non-COMPLETED true state mistakenly
    # treated as COMPLETED?
    far: dict[str, float] = {}
    for true_state in ("failed", "aborted"):
        n = per_state_n.get(true_state, 0)
        if n == 0:
            continue
        false_completed = confusion.get((true_state, "completed"), 0)
        far[true_state] = false_completed / n

    # FRR per state: how often did we miss the expected state?
    frr: dict[str, float] = {}
    for true_state in ("completed", "failed", "aborted"):
        n = per_state_n.get(true_state, 0)
        if n == 0:
            continue
        misses = sum(
            cnt for (e, a), cnt in confusion.items()
            if e == true_state and a != true_state
        )
        frr[true_state] = misses / n

    ece = _expected_calibration_error(
        np.asarray(completed_pass_for_ece),
        np.asarray(completed_truth_for_ece),
    ) if completed_pass_for_ece else 0.0
    n = len(cohort)
    return OrchestratorOODReport(
        n_total=n,
        per_state_n=dict(per_state_n),
        confusion=dict(confusion),
        far_per_state=far,
        frr_per_state=frr,
        pass_rate_ece=ece,
        abstain_rate=n_aborted / n,
        review_rate=n_failed_high_pass / n,
        mean_actual_pass=(float(np.mean(actual_pass_collect))
                           if actual_pass_collect else 0.0),
    )


# ---------------------------------------------------------------------------
# Default labeled cohort for the data-science orchestrator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Replay battery
# ---------------------------------------------------------------------------


@dataclass
class ReplayBatteryResult:
    """Audit-grade summary of running an entire labeled cohort.

    The single ``cohort_hash`` is what gets checked into the
    deployment alongside the bundle: any change to the orchestrator,
    its dependencies, or the bundle that flips a single verdict
    flips the hash and is caught at ship time.
    """

    cohort_size: int
    snapshots: list[dict]
    cohort_hash: str
    n_state_matches: int
    n_pass_rate_matches: int

    def summary(self) -> str:
        return (
            f"ReplayBatteryResult(n={self.cohort_size}, "
            f"hash={self.cohort_hash}, "
            f"state_matches={self.n_state_matches}/{self.cohort_size}, "
            f"pass_rate_matches={self.n_pass_rate_matches}/{self.cohort_size})"
        )


def orchestrator_snapshot(result, lp=None) -> dict:
    """Audit-grade summary of one orchestrator run."""
    return {
        "scenario_name":  lp.scenario.name if lp is not None else "",
        "expected_state": lp.expected_state if lp is not None else None,
        "state":          result.state.value,
        "n_steps":        len(result.steps_executed),
        "n_violations":   len(result.scenario_violations),
        "verdict":        (result.verdict.overall_status
                            if result.verdict else "N/A"),
        "n_passed":       result.verdict.n_passed if result.verdict else 0,
        "n_total":        result.verdict.n_total if result.verdict else 0,
        "pass_rate":      round(result.verdict.pass_rate, 3)
                            if result.verdict else 0.0,
        "tools":          [s["tool"] for s in result.steps_executed],
    }


def cohort_hash(snapshots: list[dict]) -> str:
    import hashlib
    import json
    payload = json.dumps(snapshots, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def orchestrator_replay_battery(
    orch,
    cohort: Iterable[LabeledPlan],
    *,
    fresh_memory_factory=None,
) -> ReplayBatteryResult:
    """Run every labeled plan and produce a single audit hash.

    Use this as the regression gate on every bundle rebuild: hash
    the cohort once, save alongside the bundle, refuse to ship a
    new bundle whose replay-battery hash differs.
    """
    cohort = list(cohort)
    snapshots: list[dict] = []
    n_state = 0
    n_pass = 0
    for lp in cohort:
        if fresh_memory_factory is not None:
            orch.memory = fresh_memory_factory(
                lp.provenance or lp.scenario.name,
            )
        result = orch.run(lp.plan, lp.scenario)
        snap = orchestrator_snapshot(result, lp=lp)
        snapshots.append(snap)
        if snap["state"] == lp.expected_state:
            n_state += 1
        if (lp.expected_state == "completed"
                and snap["pass_rate"] >= lp.expected_pass_rate - 1e-6):
            n_pass += 1
        elif lp.expected_state != "completed":
            n_pass += 1  # pass-rate not applicable
    return ReplayBatteryResult(
        cohort_size=len(cohort),
        snapshots=snapshots,
        cohort_hash=cohort_hash(snapshots),
        n_state_matches=n_state,
        n_pass_rate_matches=n_pass,
    )


def compare_replay_batteries(
    old: ReplayBatteryResult, new: ReplayBatteryResult,
) -> dict:
    """Diff two battery runs; report per-scenario state changes."""
    changes = []
    for o, n in zip(old.snapshots, new.snapshots):
        name = o.get("scenario_name", "")
        if (o["state"] != n["state"]
                or o["verdict"] != n["verdict"]
                or o["n_passed"] != n["n_passed"]):
            changes.append({
                "scenario": name,
                "old_state": o["state"], "new_state": n["state"],
                "old_verdict": o["verdict"], "new_verdict": n["verdict"],
                "old_passed": o["n_passed"], "new_passed": n["n_passed"],
            })
    return {
        "old_hash": old.cohort_hash,
        "new_hash": new.cohort_hash,
        "match": old.cohort_hash == new.cohort_hash,
        "n_changes": len(changes),
        "changes": changes,
    }


def default_labeled_cohort(
    Scenario,  # imported here to avoid circular import at module load
) -> list[LabeledPlan]:
    """Reusable labeled cohort covering every FSM path.

    The plans here are programmatic expectations a deployment team
    would write the same way: a happy path that should COMPLETE, a
    forbidden-tool plan that should FAIL, a budget overrun that
    should ABORT, etc.  This is intentionally minimal --- in
    production the cohort grows by every observed regression.
    """
    happy = [
        "load_dataset", "data_summary", "preprocess_data",
        "train_model", "diagnose_robustness", "explain_importance",
    ]
    return [
        LabeledPlan(
            plan=happy,
            scenario=Scenario(
                name="happy_canonical",
                required_tools=["load_dataset", "train_model"],
                forbidden_tools=["delete_record", "modify_raw_data"],
                expected_tool_sequence=["load_dataset", "train_model"],
                max_tool_calls=30,
            ),
            expected_state="completed",
            expected_pass_rate=1.0,
            provenance="canonical happy path",
        ),
        LabeledPlan(
            plan=happy,
            scenario=Scenario(
                name="happy_loose_budget",
                required_tools=["load_dataset"],
                forbidden_tools=[],
                max_tool_calls=50,
            ),
            expected_state="completed",
            expected_pass_rate=1.0,
            provenance="loose budget, only load_dataset required",
        ),
        LabeledPlan(
            plan=["load_dataset", "delete_record", "train_model"],
            scenario=Scenario(
                name="forbidden_present",
                required_tools=["load_dataset", "train_model"],
                forbidden_tools=["delete_record"],
                max_tool_calls=30,
            ),
            expected_state="failed",
            expected_violations=1,
            provenance="forbidden tool in plan",
        ),
        LabeledPlan(
            plan=happy,
            scenario=Scenario(
                name="budget_underrun_prep",
                required_tools=[],
                forbidden_tools=[],
                max_tool_calls=3,
            ),
            expected_state="aborted",
            expected_violations=1,
            provenance="budget exceeded at preparation",
        ),
        LabeledPlan(
            plan=happy,
            scenario=Scenario(
                name="budget_tiny",
                required_tools=[],
                forbidden_tools=[],
                max_tool_calls=2,
            ),
            expected_state="aborted",
            expected_violations=1,
            provenance="even smaller budget",
        ),
        LabeledPlan(
            plan=["data_summary", "preprocess_data"],
            scenario=Scenario(
                name="required_missing",
                required_tools=["load_dataset", "train_model"],
                forbidden_tools=[],
                max_tool_calls=30,
            ),
            expected_state="failed",
            expected_violations=2,
            provenance="required tools missing from plan",
        ),
        LabeledPlan(
            plan=["train_model", "load_dataset"],
            scenario=Scenario(
                name="wrong_sequence",
                required_tools=["load_dataset", "train_model"],
                forbidden_tools=[],
                expected_tool_sequence=["load_dataset", "train_model"],
                max_tool_calls=30,
            ),
            expected_state="failed",
            expected_violations=1,
            provenance="executed in wrong order",
        ),
        LabeledPlan(
            plan=["load_dataset"],
            scenario=Scenario(
                name="single_step",
                required_tools=["load_dataset"],
                forbidden_tools=[],
                max_tool_calls=10,
            ),
            expected_state="completed",
            expected_pass_rate=1.0,
            provenance="degenerate single-step plan",
        ),
    ]
