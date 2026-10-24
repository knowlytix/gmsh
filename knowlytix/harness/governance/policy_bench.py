# SPDX-License-Identifier: Apache-2.0
"""Latency + drift benchmarks for the policy surface.

A high-stakes deployment must answer:

  * How much latency does enforcement add per tool call?
  * Does the drift monitor actually catch realistic shift?

This module gives both answers with deterministic, reproducible
benchmarks that run in the runbook and the CI ship gate.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.policy_validation import LabeledDecision
    from knowlytix.harness.testing.policy import PolicyEngine


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass
class LatencyReport:
    n_calls: int
    p50_us: float
    p95_us: float
    p99_us: float
    p_max_us: float
    p_mean_us: float

    def summary(self) -> str:
        return (
            f"LatencyReport(n={self.n_calls}, "
            f"p50={self.p50_us:.1f}μs, "
            f"p95={self.p95_us:.1f}μs, "
            f"p99={self.p99_us:.1f}μs, "
            f"max={self.p_max_us:.1f}μs, "
            f"mean={self.p_mean_us:.1f}μs)"
        )


def bench_check_with_quorum(
    engine: "PolicyEngine",
    probe: "LabeledDecision",
    *,
    n_calls: int = 10_000,
    quorum_required: int = 2,
    warmup: int = 100,
) -> LatencyReport:
    """Microbenchmark of ``check_with_quorum`` on one probe.

    Runs ``warmup`` untimed calls first to amortise JIT / cache
    effects, then ``n_calls`` timed calls. Reports p50/p95/p99/max
    in microseconds.

    Banks expect enforcement latency in the low-single-digit
    milliseconds at p99 for high-frequency tool calls. The runbook
    pins the SLO at p99 < 5ms; the CI ship gate refuses bundles
    that breach it.
    """
    from knowlytix.harness.governance.policy_calibration import check_with_quorum

    # Warmup
    for _ in range(warmup):
        check_with_quorum(
            engine, probe.action, probe.args, probe.context,
            quorum_required=quorum_required,
        )

    latencies_us: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter_ns()
        check_with_quorum(
            engine, probe.action, probe.args, probe.context,
            quorum_required=quorum_required,
        )
        dt = time.perf_counter_ns() - t0
        latencies_us.append(dt / 1000.0)

    latencies_us.sort()
    n = len(latencies_us)
    return LatencyReport(
        n_calls=n,
        p50_us=latencies_us[int(0.50 * n)],
        p95_us=latencies_us[int(0.95 * n)],
        p99_us=latencies_us[int(0.99 * n)],
        p_max_us=latencies_us[-1],
        p_mean_us=statistics.mean(latencies_us),
    )


# ---------------------------------------------------------------------------
# Drift simulation
# ---------------------------------------------------------------------------


@dataclass
class DriftSimulationReport:
    n_healthy: int
    n_perturbed: int
    healthy_state: str
    perturbed_state: str
    healthy_should_alert: bool
    perturbed_should_alert: bool
    healthy_should_recalibrate: bool
    perturbed_should_recalibrate: bool

    def summary(self) -> str:
        return (
            f"DriftSimulationReport(\n"
            f"  healthy: n={self.n_healthy} "
            f"severity={self.healthy_state} "
            f"alert={self.healthy_should_alert} "
            f"recalibrate={self.healthy_should_recalibrate}\n"
            f"  perturbed: n={self.n_perturbed} "
            f"severity={self.perturbed_state} "
            f"alert={self.perturbed_should_alert} "
            f"recalibrate={self.perturbed_should_recalibrate}\n"
            f")"
        )


def _perturb_probe(probe, perturbation: str = "alias_rotation"):
    """Generate a perturbed copy of a probe.

    Two perturbation modes:
      * ``alias_rotation``: agent identity rotated to a paraphrase
        not present in the corpus (model_risk_officer ->
        mr_officer); engine output stays correct (REVIEW for
        unknown-agent cases) but the L3 plausibility distribution
        shifts because the agent embedding is now untrained.
      * ``context_shift``: context phrasing rotated to a variant
        that does not match any rule's fnmatch pattern.
    """
    from knowlytix.harness.governance.policy_validation import LabeledDecision
    if perturbation == "alias_rotation":
        agent = probe.agent + "_paraphrased"
    elif perturbation == "context_shift":
        agent = probe.agent
    else:
        agent = probe.agent
    if perturbation == "context_shift":
        ctx = probe.context.replace("adult_income", "shifted_corpus")
    else:
        ctx = probe.context
    return LabeledDecision(
        probe_id=probe.probe_id + "-PERT",
        agent=agent, action=probe.action,
        args=probe.args, context=ctx,
        expected_decision=probe.expected_decision,
        source_class=probe.source_class + "_perturbed",
        provenance=f"perturbation: {perturbation}",
    )


def simulate_drift(
    engine: "PolicyEngine",
    bundle,
    cohort: list,
    *,
    n_healthy: int = 200,
    n_perturbed: int = 200,
    perturbation: str = "alias_rotation",
    quorum_required: int = 2,
) -> DriftSimulationReport:
    """Simulate a healthy stream then a perturbed stream; report both states.

    Drives ``PolicyMonitor`` through ``n_healthy`` calls on the
    intact cohort, then ``n_perturbed`` calls on a perturbed copy.
    The expected pattern: healthy=stable, perturbed=drift (PSI ≥
    0.10) and possibly recalibrate (PSI ≥ 0.20).

    Two states reported separately so the runbook can show that
    drift is *caught* by the monitor, not merely *possible*.
    """
    import random

    import numpy as np

    from knowlytix.harness.governance.policy_calibration import check_with_quorum
    from knowlytix.harness.governance.policy_monitor import PolicyMonitor
    rng = random.Random(42)

    # Sample baseline plausibility on the healthy cohort
    baseline_samples: list[float] = []
    for probe in cohort:
        decision, _ = check_with_quorum(
            engine, probe.action, probe.args, probe.context,
            quorum_required=quorum_required,
        )
        plaus = decision.gms_scores.get("plausibility")
        if plaus is not None:
            baseline_samples.append(float(plaus))

    monitor = PolicyMonitor(
        bundle, window_size=max(n_healthy, n_perturbed),
        min_runs_for_alert=20,
        psi_alert=0.10, psi_recalibrate=0.20,
    )
    if baseline_samples:
        monitor.set_baseline_plausibility(np.asarray(baseline_samples))

    # Healthy stream: sample uniformly across the full cohort so
    # the realised distribution matches the cohort's class mix.
    # Without this, a sequential slice of size ``n_healthy`` from
    # the front of the cohort returns mostly-authorized probes and
    # the monitor sees an artificially high review rate.
    healthy_probes = [rng.choice(cohort) for _ in range(n_healthy)]
    for probe in healthy_probes:
        decision, verdict = check_with_quorum(
            engine, probe.action, probe.args, probe.context,
            quorum_required=quorum_required,
        )
        monitor.record(decision, verdict)
    healthy_state = monitor.state()

    # Perturbed stream
    monitor_p = PolicyMonitor(
        bundle, window_size=max(n_healthy, n_perturbed),
        min_runs_for_alert=20,
        psi_alert=0.10, psi_recalibrate=0.20,
    )
    if baseline_samples:
        monitor_p.set_baseline_plausibility(
            np.asarray(baseline_samples)
        )

    # Drift-mode probe selection: oversample probes whose
    # ``expected_decision`` is BLOCK so the block rate shifts
    # measurably. A real production deployment that suddenly sees
    # a flood of high-risk action attempts (e.g. an upstream
    # planner regression) presents this exact shape to the monitor.
    if perturbation == "block_burst":
        block_probes = [p for p in cohort
                        if p.expected_decision == "BLOCK"]
        if not block_probes:
            block_probes = list(cohort)
        for _ in range(n_perturbed):
            probe = rng.choice(block_probes)
            decision, verdict = check_with_quorum(
                engine, probe.action, probe.args, probe.context,
                quorum_required=quorum_required,
            )
            monitor_p.record(decision, verdict)
    else:
        for _ in range(n_perturbed):
            probe = rng.choice(cohort)
            pp = _perturb_probe(probe, perturbation=perturbation)
            decision, verdict = check_with_quorum(
                engine, pp.action, pp.args, pp.context,
                quorum_required=quorum_required,
            )
            monitor_p.record(decision, verdict)
    perturbed_state = monitor_p.state()

    return DriftSimulationReport(
        n_healthy=healthy_state.n_observed,
        n_perturbed=perturbed_state.n_observed,
        healthy_state=healthy_state.severity,
        perturbed_state=perturbed_state.severity,
        healthy_should_alert=healthy_state.alert,
        perturbed_should_alert=perturbed_state.alert,
        healthy_should_recalibrate=healthy_state.recalibrate,
        perturbed_should_recalibrate=perturbed_state.recalibrate,
    )
