# SPDX-License-Identifier: Apache-2.0
"""Independent re-derivation of overall compliance status.

The primary `CompliancePackage.overall_status` is set by the
`ComplianceEvidenceGenerator.generate_package` method. This
alt-validator re-derives the overall status from the check list
alone using a different code path (reduce-style accumulator vs
the primary's iteration), so disagreement reveals a real
divergence between the two implementations.

Truth table:

    all checks satisfied         -> COMPLIANT
    any not_applicable           -> PARTIAL
    any partial (and not all sat)-> PARTIAL
    otherwise                    -> NON-COMPLIANT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class AltComplianceVerdict:
    primary_status: str
    alt_status: str
    primary_pass_rate: float
    alt_pass_rate: float
    agreement: bool


@dataclass
class AltComplianceAgreementReport:
    n_packages: int
    n_agree: int
    disagreements: list[AltComplianceVerdict] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        if self.n_packages == 0:
            return 0.0
        return self.n_agree / self.n_packages

    @property
    def all_agree(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        rows = [
            "AltComplianceAgreementReport(",
            f"  n_packages={self.n_packages}",
            f"  n_agree={self.n_agree}",
            f"  agreement_rate={self.agreement_rate:.4f}",
            f"  disagreements={len(self.disagreements)}",
        ]
        for d in self.disagreements[:5]:
            rows.append(
                f"    primary={d.primary_status} "
                f"alt={d.alt_status}"
            )
        rows.append(")")
        return "\n".join(rows)


def alt_overall_status(check_statuses: list[str]) -> str:
    """Re-derive overall status from the check status list.

    Alt algorithm: build a truth-table key from the unique
    status set, look up the verdict. Distinct from the primary's
    iteration-based decision.
    """
    if not check_statuses:
        return "COMPLIANT"
    unique = frozenset(check_statuses)
    if unique == {"satisfied"}:
        return "COMPLIANT"
    # Any not_applicable or partial -> PARTIAL
    if "not_applicable" in unique or "partial" in unique:
        return "PARTIAL"
    return "NON-COMPLIANT"


def alt_pass_rate(check_statuses: list[str]) -> float:
    """Re-derive pass_rate via a sum + divide."""
    if not check_statuses:
        return 0.0
    n_sat = sum(1 for s in check_statuses if s == "satisfied")
    return n_sat / len(check_statuses)


def assert_compliance_alt_agreement(
    packages: list,
) -> AltComplianceAgreementReport:
    """Run primary + alt on every package; report agreement."""
    n_total = 0
    n_agree = 0
    disagreements: list[AltComplianceVerdict] = []
    for pkg in packages:
        n_total += 1
        statuses = [c.status for c in pkg.checks]
        alt_status = alt_overall_status(statuses)
        alt_pr = alt_pass_rate(statuses)
        primary_pr = pkg.pass_rate()
        agree = (
            alt_status == pkg.overall_status
            and abs(alt_pr - primary_pr) < 1e-9
        )
        if agree:
            n_agree += 1
        else:
            disagreements.append(AltComplianceVerdict(
                primary_status=pkg.overall_status,
                alt_status=alt_status,
                primary_pass_rate=primary_pr,
                alt_pass_rate=alt_pr,
                agreement=False,
            ))
    return AltComplianceAgreementReport(
        n_packages=n_total,
        n_agree=n_agree,
        disagreements=disagreements,
    )
