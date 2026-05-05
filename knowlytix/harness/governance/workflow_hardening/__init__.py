# SPDX-License-Identifier: Apache-2.0
"""Workflow-level hardening: snapshot, replay, cohort, bundle, invariants.

Mirrors :mod:`policy_hardening` and :mod:`tracing_hardening` for
the integrated Adult Income workflow surface (Ch.16). Public API:

* :class:`WorkflowSnapshot`, :class:`PhaseSnapshot`,
  :func:`build_workflow_snapshot` --- deterministic per-run summary
  with SHA-256 fingerprint over the canonical payload.
* :class:`WorkflowDelta`, :func:`compare_workflow_snapshots`,
  :func:`assert_no_workflow_regression` --- replay-regression diff.
* :class:`WorkflowAdversarialProbe`, :class:`WorkflowAdversarialReport`,
  :func:`build_workflow_adversarial_cohort_v1`,
  :func:`build_workflow_adversarial_cohort_v2`,
  :func:`sweep_workflow_adversarial` --- six-class cohort with
  parameterised v2 expansion.
* :class:`CrossComponentReport`, :class:`ComponentDisagreement`,
  :func:`cross_component_agreement` --- policy + contract + audit
  agreement check.
* :class:`WorkflowBundle`, :class:`WorkflowOperatingPoint`,
  :func:`build_workflow_bundle`, :func:`bind_workflow_from_bundle`
  --- HMAC-signed frozen artefact.
* :class:`WorkflowInvariantReport`, :func:`verify_workflow_invariants`
  --- exhaustive enumeration over the snapshot FSM.
* :class:`WorkflowAltAgreementReport`, :func:`alt_release_gate`,
  :func:`assert_workflow_alt_agreement` --- independent
  re-derivation of the release-gate verdict.
* :class:`WorkflowLatencyReport`, :func:`measure_phase_latencies`,
  :func:`bench_snapshot_fingerprint`, :func:`bench_alt_release_gate`
  --- per-phase and microbenchmarks.
"""

from knowlytix.harness.governance.workflow_hardening.workflow_snapshot import (
    PhaseSnapshot,
    WorkflowSnapshot,
    build_workflow_snapshot,
)
from knowlytix.harness.governance.workflow_hardening.workflow_replay import (
    WorkflowDelta,
    WorkflowReplayRegressionError,
    assert_no_workflow_regression,
    compare_workflow_snapshots,
)
from knowlytix.harness.governance.workflow_hardening.workflow_adversarial import (
    WorkflowAdversarialProbe,
    WorkflowAdversarialReport,
    build_workflow_adversarial_cohort_v1,
    build_workflow_adversarial_cohort_v2,
    sweep_workflow_adversarial,
)
from knowlytix.harness.governance.workflow_hardening.cross_component_agreement import (
    ComponentDisagreement,
    CrossComponentReport,
    cross_component_agreement,
)
from knowlytix.harness.governance.workflow_hardening.workflow_bundle import (
    WORKFLOW_BUNDLE_SCHEMA_VERSION,
    WorkflowBundle,
    WorkflowOperatingPoint,
    bind_workflow_from_bundle,
    build_workflow_bundle,
)
from knowlytix.harness.governance.workflow_hardening.workflow_invariants import (
    WorkflowInvariantReport,
    WorkflowInvariantViolation,
    gate_decision_from,
    verify_workflow_invariants,
)
from knowlytix.harness.governance.workflow_hardening.workflow_alt_validator import (
    WorkflowAltAgreementReport,
    alt_release_gate,
    assert_workflow_alt_agreement,
)
from knowlytix.harness.governance.workflow_hardening.workflow_bench import (
    PhaseLatencyRecord,
    SnapshotLatencyReport,
    WorkflowLatencyReport,
    bench_alt_release_gate,
    bench_snapshot_fingerprint,
    measure_phase_latencies,
)


__all__ = [
    "PhaseSnapshot",
    "WorkflowSnapshot",
    "build_workflow_snapshot",
    "WorkflowDelta",
    "WorkflowReplayRegressionError",
    "assert_no_workflow_regression",
    "compare_workflow_snapshots",
    "WorkflowAdversarialProbe",
    "WorkflowAdversarialReport",
    "build_workflow_adversarial_cohort_v1",
    "build_workflow_adversarial_cohort_v2",
    "sweep_workflow_adversarial",
    "ComponentDisagreement",
    "CrossComponentReport",
    "cross_component_agreement",
    "WORKFLOW_BUNDLE_SCHEMA_VERSION",
    "WorkflowBundle",
    "WorkflowOperatingPoint",
    "bind_workflow_from_bundle",
    "build_workflow_bundle",
    "WorkflowInvariantReport",
    "WorkflowInvariantViolation",
    "gate_decision_from",
    "verify_workflow_invariants",
    "WorkflowAltAgreementReport",
    "alt_release_gate",
    "assert_workflow_alt_agreement",
    "PhaseLatencyRecord",
    "SnapshotLatencyReport",
    "WorkflowLatencyReport",
    "bench_alt_release_gate",
    "bench_snapshot_fingerprint",
    "measure_phase_latencies",
]
