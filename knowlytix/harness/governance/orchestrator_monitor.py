# SPDX-License-Identifier: Apache-2.0
"""Cross-run drift detection for the GovernedOrchestrator.

The single-run drift signal addressed by ``DriftMonitor`` (PSI on
the tension-energy distribution) is the right tool for a continuous
verdict stream like ``RuntimeGate``. A run-based pipeline like the
orchestrator emits one terminal verdict per run, so the analogous
drift quantities are run-level statistics: how often does the run
end in COMPLETED, FAILED, or ABORTED?  How does the per-run
pass-rate distribution shift?  Are violations concentrating on
particular tools?

This module wraps a ``GovernanceBundle``'s :class:`OperatingPoint`
SLO around a rolling window of orchestrator runs and reports
violations the moment the window of recent runs crosses the
configured rates.

Distinct from ``DriftMonitor`` (which feeds individual verdicts),
``OrchestratorMonitor`` is fed once per :class:`OrchestratorResult`
and answers: *is the orchestrator's recent behavior consistent with
its operating point?*
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.governance_bundle import GovernanceBundle
    from knowlytix.harness.governance.orchestrator import OrchestratorResult


@dataclass
class RunSnapshot:
    """Audit-grade summary of one orchestrator run."""

    state: str
    n_steps: int
    n_violations: int
    verdict: str
    pass_rate: float
    n_passed: int
    n_total: int
    tools: tuple[str, ...]

    @classmethod
    def from_result(cls, result: "OrchestratorResult") -> "RunSnapshot":
        v = result.verdict
        return cls(
            state=result.state.value,
            n_steps=len(result.steps_executed),
            n_violations=len(result.scenario_violations),
            verdict=v.overall_status if v else "N/A",
            pass_rate=float(v.pass_rate) if v else 0.0,
            n_passed=v.n_passed if v else 0,
            n_total=v.n_total if v else 0,
            tools=tuple(s["tool"] for s in result.steps_executed),
        )


@dataclass
class OrchestratorMonitorState:
    n_observed: int
    window_full: bool
    completed_rate: float
    failed_rate: float
    aborted_rate: float
    abstain_rate: float
    review_rate: float
    mean_pass_rate: float
    slo_violation: str | None
    severity: str
    alert: bool
    recalibrate: bool

    def summary(self) -> str:
        slo = f", slo_violation={self.slo_violation}" if self.slo_violation else ""
        return (
            f"OrchestratorMonitorState(n={self.n_observed}, "
            f"window_full={self.window_full}, "
            f"COMPLETED={self.completed_rate:.3f}, "
            f"FAILED={self.failed_rate:.3f}, "
            f"ABORTED={self.aborted_rate:.3f}, "
            f"mean_pass_rate={self.mean_pass_rate:.3f}, "
            f"severity={self.severity}, alert={self.alert}, "
            f"recalibrate={self.recalibrate}{slo})"
        )


class OrchestratorMonitor:
    """Rolling-window monitor over orchestrator runs.

    Usage:

        monitor = OrchestratorMonitor(bundle, window_size=50)
        for run in stream:
            monitor.record(run)
            if monitor.should_recalibrate():
                trigger_offline_recalibration()

    The bundle's :class:`OperatingPoint` defines the SLO ceilings on
    abstain / review / combined-uncertain rates; this monitor maps
    those onto run-level quantities (ABORTED counts as ``abstain``
    --- the run ended without a verdict --- and a low pass-rate
    counts as ``review`` because the orchestrator's verdict is
    a low-confidence FAIL the deployment team should look at).
    """

    def __init__(
        self,
        bundle: "GovernanceBundle",
        *,
        window_size: int = 50,
        min_runs_for_alert: int = 10,
        recalibrate_requires_full_window: bool = True,
        low_pass_rate_threshold: float = 0.80,
    ) -> None:
        self.bundle = bundle
        self.window_size = window_size
        self.min_runs_for_alert = min_runs_for_alert
        self.recalibrate_requires_full_window = recalibrate_requires_full_window
        self.low_pass_rate_threshold = low_pass_rate_threshold
        self._buf: deque[RunSnapshot] = deque(maxlen=window_size)
        self._n_observed = 0

    def record(self, result_or_snapshot) -> None:
        if isinstance(result_or_snapshot, RunSnapshot):
            snap = result_or_snapshot
        else:
            snap = RunSnapshot.from_result(result_or_snapshot)
        self._buf.append(snap)
        self._n_observed += 1

    def state(self) -> OrchestratorMonitorState:
        n = len(self._buf)
        if n == 0:
            return OrchestratorMonitorState(
                n_observed=0, window_full=False,
                completed_rate=0.0, failed_rate=0.0, aborted_rate=0.0,
                abstain_rate=0.0, review_rate=0.0,
                mean_pass_rate=0.0,
                slo_violation=None, severity="warming_up",
                alert=False, recalibrate=False,
            )
        states = Counter(s.state for s in self._buf)
        completed_rate = states.get("completed", 0) / n
        failed_rate = states.get("failed", 0) / n
        aborted_rate = states.get("aborted", 0) / n
        # Map run terminal states onto SLO quantities.
        # ABORTED ~ abstain (no verdict reached).
        # FAILED with low pass_rate ~ review (low-confidence outcome).
        abstain_rate = aborted_rate
        review_rate = sum(
            1 for s in self._buf
            if s.state == "failed"
            and s.pass_rate >= self.low_pass_rate_threshold
        ) / n
        mean_pass = sum(s.pass_rate for s in self._buf) / n

        window_full = n >= self.window_size
        warming_up = n < self.min_runs_for_alert
        severity = "warming_up" if warming_up else "stable"
        alert = False
        recalibrate = False
        slo_violation: str | None = None

        op = getattr(self.bundle, "operating_point", None)
        if op is not None and not warming_up:
            violations: list[str] = []
            if abstain_rate > op.max_abstain_rate:
                violations.append(
                    f"aborted_rate={abstain_rate:.3f} > "
                    f"max_abstain_rate={op.max_abstain_rate:.3f}"
                )
            if review_rate > op.max_review_rate:
                violations.append(
                    f"failed_high_pass_rate={review_rate:.3f} > "
                    f"max_review_rate={op.max_review_rate:.3f}"
                )
            uncertain = abstain_rate + review_rate
            if uncertain > op.max_uncertain_rate:
                violations.append(
                    f"uncertain={uncertain:.3f} > "
                    f"max_uncertain_rate={op.max_uncertain_rate:.3f}"
                )
            if violations:
                slo_violation = "; ".join(violations)
                alert = True
                severity = "slo_breach"
                if (not self.recalibrate_requires_full_window
                        or window_full):
                    recalibrate = True

        return OrchestratorMonitorState(
            n_observed=self._n_observed,
            window_full=window_full,
            completed_rate=completed_rate,
            failed_rate=failed_rate,
            aborted_rate=aborted_rate,
            abstain_rate=abstain_rate,
            review_rate=review_rate,
            mean_pass_rate=mean_pass,
            slo_violation=slo_violation,
            severity=severity,
            alert=alert,
            recalibrate=recalibrate,
        )

    def should_alert(self) -> bool:
        return self.state().alert

    def should_recalibrate(self) -> bool:
        return self.state().recalibrate

    def reset(self) -> None:
        self._buf.clear()
        self._n_observed = 0
