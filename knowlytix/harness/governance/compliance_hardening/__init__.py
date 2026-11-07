# SPDX-License-Identifier: Apache-2.0
"""Compliance hardening: snapshot, invariants, alt, bundle, bench.

Mirrors the hardening pattern in :mod:`policy_hardening`,
:mod:`tracing_hardening`, :mod:`workflow_hardening`,
:mod:`adversarial_hardening`, and :mod:`calibration_hardening`
for the Ch.20 compliance-evidence surface. Public API:

* :class:`ComplianceSnapshot`, :class:`CheckRecord`,
  :func:`build_compliance_snapshot` --- deterministic fingerprint
  over the check list + status counts + per-regulation pass rates.
* :class:`ComplianceInvariantReport`,
  :func:`verify_compliance_invariants` --- N1-N5.
* :class:`AltComplianceAgreementReport`,
  :func:`alt_overall_status`, :func:`alt_pass_rate`,
  :func:`assert_compliance_alt_agreement` --- alt-validator.
* :class:`ComplianceBundle`, :class:`ComplianceOperatingPoint`,
  :func:`bind_compliance_from_bundle`,
  :func:`build_compliance_bundle` --- HMAC-signed artefact.
* :class:`ComplianceLatencyReport`,
  :func:`bench_compliance_fingerprint`,
  :func:`bench_alt_overall_status` --- microbenchmarks.
"""

from knowlytix.harness.governance.compliance_hardening.compliance_alt_validator import (
    AltComplianceAgreementReport,
    AltComplianceVerdict,
    alt_overall_status,
    alt_pass_rate,
    assert_compliance_alt_agreement,
)
from knowlytix.harness.governance.compliance_hardening.compliance_bench import (
    ComplianceLatencyReport,
    bench_alt_overall_status,
    bench_compliance_fingerprint,
)
from knowlytix.harness.governance.compliance_hardening.compliance_bundle import (
    COMPLIANCE_BUNDLE_SCHEMA_VERSION,
    ComplianceBundle,
    ComplianceOperatingPoint,
    bind_compliance_from_bundle,
    build_compliance_bundle,
)
from knowlytix.harness.governance.compliance_hardening.compliance_invariants import (
    ComplianceInvariantReport,
    ComplianceInvariantViolation,
    verify_compliance_invariants,
)
from knowlytix.harness.governance.compliance_hardening.compliance_snapshot import (
    CheckRecord,
    ComplianceSnapshot,
    build_compliance_snapshot,
)

__all__ = [
    "CheckRecord",
    "ComplianceSnapshot",
    "build_compliance_snapshot",
    "ComplianceInvariantReport",
    "ComplianceInvariantViolation",
    "verify_compliance_invariants",
    "AltComplianceAgreementReport",
    "AltComplianceVerdict",
    "alt_overall_status",
    "alt_pass_rate",
    "assert_compliance_alt_agreement",
    "COMPLIANCE_BUNDLE_SCHEMA_VERSION",
    "ComplianceBundle",
    "ComplianceOperatingPoint",
    "bind_compliance_from_bundle",
    "build_compliance_bundle",
    "ComplianceLatencyReport",
    "bench_alt_overall_status",
    "bench_compliance_fingerprint",
]
