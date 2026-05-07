# SPDX-License-Identifier: Apache-2.0
"""Drift monitor for the multi-agent governance surface.

Distinct from the calibration baseline: this monitor runs at the
**bundle** surface and computes PSI on the per-agent tool-call
distribution against the bundle's reference plus SLO on the
blocked-attempt rate. Severity:

  * ``stable``     -- PSI healthy on every agent
  * ``slow_drift`` -- PSI breach but no SLO violation yet
  * ``slo_breach`` -- blocked-attempt rate exceeds the operating-point
                      ceiling (means agents are repeatedly trying
                      forbidden tools)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


from knowlytix.harness.governance.agents.multi_agent import (
    MultiAgentOrchestrator,
)
from knowlytix.harness.governance.agents.multi_agent_bundle import MultiAgentBundle
from knowlytix.harness.governance.agents.multi_agent_co_attestor import _psi_one_role


Severity = Literal["stable", "slow_drift", "slo_breach"]


@dataclass
class MultiAgentMonitorState:
    severity: Severity
    psi_per_role: dict[str, float]
    blocked_attempt_rate: float
    n_runs: int
    alert: bool = False
    recalibrate: bool = False
    slo_violations: dict[str, str] = field(default_factory=dict)


class MultiAgentMonitor:
    """Rolling-window drift monitor for the multi-agent orchestrator.

    Construction:

      monitor = MultiAgentMonitor(
          orch=orch, bundle=bundle,
          window_size=50, min_runs_for_alert=20,
      )

    Per pipeline run:
      monitor.record(pipeline_result)
    Periodically:
      state = monitor.evaluate()
    """

    def __init__(
        self,
        orch: MultiAgentOrchestrator,
        bundle: MultiAgentBundle,
        *,
        window_size: int = 50,
        min_runs_for_alert: int = 20,
        psi_warn: float = 0.05,
        psi_alert: float = 0.10,
        max_blocked_attempt_rate: float = 0.05,
    ) -> None:
        self.orch = orch
        self.bundle = bundle
        self.window_size = int(window_size)
        self.min_runs_for_alert = int(min_runs_for_alert)
        self.psi_warn = float(psi_warn)
        self.psi_alert = float(psi_alert)
        self.max_blocked_rate = float(max_blocked_attempt_rate)

        # Reference distribution from the bundle
        ref = bundle.per_agent_tool_frequency or {}
        counts_by_role = ref.get("counts_by_role", {})
        self._reference_norm: dict[str, dict[str, float]] = {}
        for role, counts in counts_by_role.items():
            tot = max(1, sum(counts.values()))
            self._reference_norm[role] = {
                k: v / tot for k, v in counts.items()
            }

        self._observed: list[dict] = []  # {role: counts, blocks: int}

    def record_run(
        self,
        per_run_calls: dict[str, dict[str, int]],
        blocked_count: int,
    ) -> None:
        """Record a pipeline run.

        Parameters
        ----------
        per_run_calls : dict
            ``{role_name: {tool: count}}`` for the run.
        blocked_count : int
            Number of blocked tool attempts in the run.
        """
        self._observed.append({
            "by_role": {k: dict(v) for k, v in per_run_calls.items()},
            "blocked": int(blocked_count),
        })
        if len(self._observed) > self.window_size:
            self._observed = self._observed[-self.window_size:]

    def evaluate(self) -> MultiAgentMonitorState:
        n = len(self._observed)
        if n < self.min_runs_for_alert:
            return MultiAgentMonitorState(
                severity="stable",
                psi_per_role={},
                blocked_attempt_rate=0.0,
                n_runs=n,
                alert=False,
                recalibrate=False,
            )

        # Aggregate per-role counts across the window
        agg: dict[str, dict[str, int]] = {}
        total_blocked = 0
        total_calls = 0
        for run in self._observed:
            for role, counts in run["by_role"].items():
                cell = agg.setdefault(role, {})
                for tool, c in counts.items():
                    cell[tool] = cell.get(tool, 0) + c
                    total_calls += c
            total_blocked += run["blocked"]
        # PSI per role
        psi: dict[str, float] = {}
        for role, ref in self._reference_norm.items():
            obs = agg.get(role, {})
            psi[role] = _psi_one_role(ref, obs)

        denom = total_calls + total_blocked
        blocked_rate = (total_blocked / denom) if denom > 0 else 0.0

        slo_violations: dict[str, str] = {}
        if blocked_rate > self.max_blocked_rate:
            slo_violations["blocked_attempt_rate"] = (
                f"{blocked_rate:.4f} > ceiling {self.max_blocked_rate:.4f}")

        any_alert_psi = any(v > self.psi_alert for v in psi.values())
        any_warn_psi = any(v > self.psi_warn for v in psi.values())
        if slo_violations:
            severity: Severity = "slo_breach"
            alert = True
            recalibrate = True
        elif any_alert_psi:
            severity = "slow_drift"
            alert = True
            recalibrate = False
        elif any_warn_psi:
            severity = "slow_drift"
            alert = False
            recalibrate = False
        else:
            severity = "stable"
            alert = False
            recalibrate = False

        return MultiAgentMonitorState(
            severity=severity,
            psi_per_role=psi,
            blocked_attempt_rate=blocked_rate,
            n_runs=n,
            alert=alert,
            recalibrate=recalibrate,
            slo_violations=slo_violations,
        )
