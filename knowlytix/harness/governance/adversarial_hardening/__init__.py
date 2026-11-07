# SPDX-License-Identifier: Apache-2.0
"""Adversarial-defense hardening: snapshot, cohort, bundle, invariants.

Mirrors :mod:`policy_hardening`, :mod:`tracing_hardening`, and
:mod:`workflow_hardening` for the Ch.18 adversarial surface.
Public API:

* :class:`AttackSnapshot`, :class:`AttackOutcome`,
  :func:`build_attack_snapshot` --- deterministic per-run summary
  with SHA-256 fingerprint over canonical outcomes.
* :class:`AttackProbe`, :class:`AttackCohortReport`,
  :func:`build_attack_cohort_v1`, :func:`build_attack_cohort_v2`,
  :func:`sweep_attack_cohort` --- 5-class parameterised cohort.
* :class:`CrossLayerReport`, :class:`LayerDisagreement`,
  :func:`cross_layer_agreement`, ``EXPECTED_LAYER_MAP`` ---
  expected-layer match check.
* :class:`AttackBundle`, :class:`AttackOperatingPoint`,
  :func:`build_attack_bundle`, :func:`bind_attack_from_bundle`
  --- HMAC-signed frozen artefact.
* :class:`AttackInvariantReport`, :func:`verify_attack_invariants`
  --- exhaustive enumeration over the snapshot FSM.
* :class:`AltAttackAgreementReport`, :func:`alt_blocked`,
  :func:`assert_attack_alt_agreement` --- independent re-derivation.
* :class:`AttackLatencyReport`,
  :func:`bench_attack_fingerprint`, :func:`bench_alt_blocked`
  --- microbenchmarks.
"""

from knowlytix.harness.governance.adversarial_hardening.attack_alt_validator import (
    AltAttackAgreementReport,
    AltAttackVerdict,
    alt_blocked,
    assert_attack_alt_agreement,
)
from knowlytix.harness.governance.adversarial_hardening.attack_bench import (
    AttackLatencyReport,
    bench_alt_blocked,
    bench_attack_fingerprint,
)
from knowlytix.harness.governance.adversarial_hardening.attack_bundle import (
    ATTACK_BUNDLE_SCHEMA_VERSION,
    AttackBundle,
    AttackOperatingPoint,
    bind_attack_from_bundle,
    build_attack_bundle,
)
from knowlytix.harness.governance.adversarial_hardening.attack_cohort import (
    AttackCohortReport,
    AttackProbe,
    build_attack_cohort_v1,
    build_attack_cohort_v2,
    sweep_attack_cohort,
)
from knowlytix.harness.governance.adversarial_hardening.attack_invariants import (
    AttackInvariantReport,
    AttackInvariantViolation,
    verify_attack_invariants,
)
from knowlytix.harness.governance.adversarial_hardening.attack_snapshot import (
    AttackOutcome,
    AttackSnapshot,
    build_attack_snapshot,
)
from knowlytix.harness.governance.adversarial_hardening.cross_layer_agreement import (
    EXPECTED_LAYER_MAP,
    CrossLayerReport,
    LayerDisagreement,
    cross_layer_agreement,
)

__all__ = [
    "AttackOutcome",
    "AttackSnapshot",
    "build_attack_snapshot",
    "AttackCohortReport",
    "AttackProbe",
    "build_attack_cohort_v1",
    "build_attack_cohort_v2",
    "sweep_attack_cohort",
    "CrossLayerReport",
    "EXPECTED_LAYER_MAP",
    "LayerDisagreement",
    "cross_layer_agreement",
    "ATTACK_BUNDLE_SCHEMA_VERSION",
    "AttackBundle",
    "AttackOperatingPoint",
    "bind_attack_from_bundle",
    "build_attack_bundle",
    "AttackInvariantReport",
    "AttackInvariantViolation",
    "verify_attack_invariants",
    "AltAttackAgreementReport",
    "AltAttackVerdict",
    "alt_blocked",
    "assert_attack_alt_agreement",
    "AttackLatencyReport",
    "bench_alt_blocked",
    "bench_attack_fingerprint",
]
