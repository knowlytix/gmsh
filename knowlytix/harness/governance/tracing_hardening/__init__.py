# SPDX-License-Identifier: Apache-2.0
"""Audit-trail hardening: hash chain, cross-reference attestor, signed bundle.

Public API:

* :func:`seal_audit_trail`, :func:`verify_audit_chain`,
  :func:`compute_packet_content_hash`, :func:`chain_head_hash`,
  :class:`ChainBreak` --- tamper-evident hash chain over
  :class:`AuditPacket` (Ch.14).
* :class:`RuntimePolicyEvent`, :class:`RuntimeContractEvent`,
  :class:`CrossAttestationBreak`, :func:`cross_attest_policy`,
  :func:`cross_attest_contract`, :func:`cross_attest_combined` ---
  cross-reference attestor that verifies the trail accurately
  reflects independent runtime decisions.
* :class:`AuditAdversarialProbe`, :class:`AuditAdversarialReport`,
  :func:`build_audit_adversarial_cohort`,
  :func:`sweep_audit_adversarial`, :func:`make_baseline_trail` ---
  systematic adversarial cohort exercising every block path.
* :class:`AuditReplayResult`, :class:`AuditReplayRegressionError`,
  :func:`replay_audit_metrics`, :func:`assert_no_audit_regression`,
  :func:`metrics_fingerprint` --- byte-identical metric replay.
* :class:`AuditTrailBundle`, :class:`AuditOperatingPoint`,
  :func:`build_audit_trail_bundle`, :func:`bind_audit_from_bundle` ---
  HMAC-signed frozen artifact with three refusal paths.
"""

from knowlytix.harness.governance.tracing_hardening.audit_adversarial import (
    AuditAdversarialProbe,
    AuditAdversarialReport,
    build_audit_adversarial_cohort,
    build_audit_adversarial_cohort_v2,
    make_baseline_trail,
    sweep_audit_adversarial,
)
from knowlytix.harness.governance.tracing_hardening.audit_alt_validator import (
    AltAgreementReport,
    AltChainBreak,
    alt_verify_chain,
    assert_chain_alt_agreement,
)
from knowlytix.harness.governance.tracing_hardening.audit_bench import (
    AuditLatencyReport,
    bench_replay_audit_metrics,
    bench_seal_audit_trail,
    bench_verify_audit_chain,
)
from knowlytix.harness.governance.tracing_hardening.audit_bundle import (
    AUDIT_BUNDLE_SCHEMA_VERSION,
    AuditOperatingPoint,
    AuditTrailBundle,
    bind_audit_from_bundle,
    build_audit_trail_bundle,
)
from knowlytix.harness.governance.tracing_hardening.audit_invariants import (
    AuditInvariantReport,
    AuditInvariantViolation,
    verify_audit_invariants,
)
from knowlytix.harness.governance.tracing_hardening.audit_replay import (
    AuditReplayRegressionError,
    AuditReplayResult,
    assert_no_audit_regression,
    metrics_fingerprint,
    replay_audit_metrics,
)
from knowlytix.harness.governance.tracing_hardening.cross_attestor import (
    CrossAttestationBreak,
    RuntimeContractEvent,
    RuntimePolicyEvent,
    cross_attest_combined,
    cross_attest_contract,
    cross_attest_policy,
)
from knowlytix.harness.governance.tracing_hardening.hash_chain import (
    ChainBreak,
    chain_head_hash,
    compute_packet_content_hash,
    seal_audit_trail,
    verify_audit_chain,
)

__all__ = [
    "AUDIT_BUNDLE_SCHEMA_VERSION",
    "AuditAdversarialProbe",
    "AuditAdversarialReport",
    "AuditOperatingPoint",
    "AuditReplayRegressionError",
    "AuditReplayResult",
    "AuditTrailBundle",
    "ChainBreak",
    "CrossAttestationBreak",
    "RuntimeContractEvent",
    "RuntimePolicyEvent",
    "assert_no_audit_regression",
    "bind_audit_from_bundle",
    "build_audit_adversarial_cohort",
    "build_audit_adversarial_cohort_v2",
    "build_audit_trail_bundle",
    "chain_head_hash",
    "compute_packet_content_hash",
    "cross_attest_combined",
    "cross_attest_contract",
    "cross_attest_policy",
    "make_baseline_trail",
    "metrics_fingerprint",
    "replay_audit_metrics",
    "seal_audit_trail",
    "sweep_audit_adversarial",
    "verify_audit_chain",
    "AltAgreementReport",
    "AltChainBreak",
    "alt_verify_chain",
    "assert_chain_alt_agreement",
    "AuditInvariantReport",
    "AuditInvariantViolation",
    "verify_audit_invariants",
    "AuditLatencyReport",
    "bench_replay_audit_metrics",
    "bench_seal_audit_trail",
    "bench_verify_audit_chain",
]
