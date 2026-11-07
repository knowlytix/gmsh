# SPDX-License-Identifier: Apache-2.0
"""Calibration hardening: snapshot, replay, invariants, alt, bundle, bench.

Mirrors :mod:`policy_hardening`, :mod:`tracing_hardening`,
:mod:`workflow_hardening`, and :mod:`adversarial_hardening` for
the Ch.19 threshold-calibration surface. Public API:

* :class:`CalibrationSnapshot`, :class:`CalibratedThreshold`,
  :func:`build_calibration_snapshot` --- deterministic fingerprint.
* :class:`CalibrationDelta`, :class:`CalibrationRegressionError`,
  :func:`compare_calibration_snapshots`,
  :func:`assert_no_calibration_regression` --- replay diff.
* :class:`CalibrationInvariantReport`,
  :func:`verify_calibration_invariants` --- FSM enumeration.
* :class:`AltCalibrationAgreementReport`,
  :func:`alt_best_config`,
  :func:`assert_calibration_alt_agreement` --- alt-validator.
* :class:`CalibrationBundle`, :class:`CalibrationOperatingPoint`,
  :func:`bind_calibration_from_bundle`,
  :func:`build_calibration_bundle` --- HMAC-signed artefact.
* :class:`CrossDomainReport`, :class:`ThresholdDrift`,
  :func:`compare_calibrations_across_domains` --- cross-domain
  comparison.
* :class:`CalibrationLatencyReport`,
  :func:`bench_calibration_fingerprint`,
  :func:`bench_alt_best_config` --- microbenchmarks.
"""

from knowlytix.harness.governance.calibration_hardening.calibration_alt_validator import (
    AltCalibrationAgreementReport,
    AltCalibrationVerdict,
    alt_best_config,
    assert_calibration_alt_agreement,
)
from knowlytix.harness.governance.calibration_hardening.calibration_bench import (
    CalibrationLatencyReport,
    bench_alt_best_config,
    bench_calibration_fingerprint,
)
from knowlytix.harness.governance.calibration_hardening.calibration_bundle import (
    CALIBRATION_BUNDLE_SCHEMA_VERSION,
    CalibrationBundle,
    CalibrationOperatingPoint,
    bind_calibration_from_bundle,
    build_calibration_bundle,
)
from knowlytix.harness.governance.calibration_hardening.calibration_invariants import (
    CalibrationInvariantReport,
    CalibrationInvariantViolation,
    verify_calibration_invariants,
)
from knowlytix.harness.governance.calibration_hardening.calibration_replay import (
    CalibrationDelta,
    CalibrationRegressionError,
    assert_no_calibration_regression,
    compare_calibration_snapshots,
)
from knowlytix.harness.governance.calibration_hardening.calibration_snapshot import (
    CalibratedThreshold,
    CalibrationSnapshot,
    build_calibration_snapshot,
)
from knowlytix.harness.governance.calibration_hardening.cross_domain_agreement import (
    CrossDomainReport,
    ThresholdDrift,
    compare_calibrations_across_domains,
)

__all__ = [
    "CalibratedThreshold",
    "CalibrationSnapshot",
    "build_calibration_snapshot",
    "CalibrationDelta",
    "CalibrationRegressionError",
    "assert_no_calibration_regression",
    "compare_calibration_snapshots",
    "CalibrationInvariantReport",
    "CalibrationInvariantViolation",
    "verify_calibration_invariants",
    "AltCalibrationAgreementReport",
    "AltCalibrationVerdict",
    "alt_best_config",
    "assert_calibration_alt_agreement",
    "CALIBRATION_BUNDLE_SCHEMA_VERSION",
    "CalibrationBundle",
    "CalibrationOperatingPoint",
    "bind_calibration_from_bundle",
    "build_calibration_bundle",
    "CrossDomainReport",
    "ThresholdDrift",
    "compare_calibrations_across_domains",
    "CalibrationLatencyReport",
    "bench_alt_best_config",
    "bench_calibration_fingerprint",
]
