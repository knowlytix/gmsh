# SPDX-License-Identifier: Apache-2.0
"""Deterministic snapshot of a compliance evidence package.

The Ch.20 `ComplianceEvidenceGenerator` produces a
`CompliancePackage` with a list of `ComplianceCheck`s and an
overall status. The package's `timestamp` is per-build; the
governance-grade fields that must reproduce byte-identically are:

  * the ordered tuple of (regulation, article, status, layer)
    per check
  * the overall_status
  * the pass_rate (rounded to 6 decimals)
  * the per-regulation pass-rate breakdown

This module fingerprints those fields. Two consecutive
`generate_package(...)` calls with the same input must produce
the same fingerprint.

Mirrors the snapshot pattern in :mod:`policy_replay`,
:mod:`workflow_hardening.workflow_snapshot`, etc.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.compliance import CompliancePackage


@dataclass
class CheckRecord:
    """Deterministic per-check record (no per-build evidence text)."""

    regulation: str
    article: str
    architecture_layer: str
    evidence_chapter: str
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComplianceSnapshot:
    """Audit-grade summary of a compliance package."""

    suite_id: str
    n_checks: int
    overall_status: str
    pass_rate: float
    per_regulation_pass_rate: dict[str, float]
    per_status_count: dict[str, int]
    checks: list[CheckRecord]                       # canonically sorted

    def canonical_payload(self) -> bytes:
        d = {
            "suite_id": self.suite_id,
            "n_checks": self.n_checks,
            "overall_status": self.overall_status,
            "pass_rate": self.pass_rate,
            "per_regulation_pass_rate": dict(
                sorted(self.per_regulation_pass_rate.items())
            ),
            "per_status_count": dict(
                sorted(self.per_status_count.items())
            ),
            "checks": [c.to_dict() for c in self.checks],
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()

    def summary(self) -> str:
        return (
            f"ComplianceSnapshot(id={self.suite_id}, "
            f"n_checks={self.n_checks}, "
            f"overall={self.overall_status}, "
            f"pass_rate={self.pass_rate}, "
            f"fingerprint={self.fingerprint()[:16]}...)"
        )


def build_compliance_snapshot(
    *,
    suite_id: str,
    package: "CompliancePackage",
) -> ComplianceSnapshot:
    """Pack a compliance package into a deterministic snapshot."""
    checks = sorted(
        (
            CheckRecord(
                regulation=c.regulation,
                article=c.article,
                architecture_layer=c.architecture_layer,
                evidence_chapter=c.evidence_chapter,
                status=c.status,
            )
            for c in package.checks
        ),
        key=lambda r: (r.regulation, r.article),
    )

    # Per-regulation pass rate
    by_reg: dict[str, list[str]] = {}
    for c in package.checks:
        by_reg.setdefault(c.regulation, []).append(c.status)
    per_reg_pr = {}
    for reg, statuses in by_reg.items():
        n_sat = sum(1 for s in statuses if s == "satisfied")
        per_reg_pr[reg] = round(n_sat / len(statuses), 6)

    per_status = dict(sorted(
        Counter(c.status for c in package.checks).items()
    ))

    return ComplianceSnapshot(
        suite_id=suite_id,
        n_checks=len(checks),
        overall_status=package.overall_status,
        pass_rate=round(package.pass_rate(), 6),
        per_regulation_pass_rate=per_reg_pr,
        per_status_count={k: int(v) for k, v in per_status.items()},
        checks=checks,
    )
