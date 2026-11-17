# SPDX-License-Identifier: Apache-2.0
"""Gate decisions over an audit report, kept separate from reporting.

The decision is ``pass``, ``fail`` or ``indeterminate``. A finding at or above
``fail_on`` severity fails the gate. Otherwise the gate is indeterminate when a
required rule did not reach a definite outcome (unavailable, error, partial,
disabled or absent) or when the snapshot itself is unreliable: findings of the
``snapshot`` category (ART005, files changing during the read) never pass or
fail a gate, because the audited content is not one well-defined state.
An optional rule that raised (``error``) or stopped early (``partial``) also
makes the gate indeterminate: its findings are unknown, and extra rules passed
to ``audit(rules=...)`` are always optional. Unavailable and disabled optional
rules do not block; the decision lists every open optional rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from knowlytix.knowledge.kgcheck.models import (
    AuditReport,
    JsonObject,
    RuleStatus,
    Severity,
)

SNAPSHOT_CATEGORY = "snapshot"
_RANK = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2, Severity.SUGGESTION: 3}
_DEFINITE = frozenset({RuleStatus.PASSED, RuleStatus.FINDINGS, RuleStatus.NOT_APPLICABLE})
_UNKNOWN = frozenset({RuleStatus.ERROR, RuleStatus.PARTIAL})  # findings may be missing


class GateOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class GatePolicy:
    """What blocks. ``fail_on=None`` never fails on findings (report-only).

    ``required_rules=None`` uses the profile's required rules.
    """

    fail_on: Severity | None = Severity.ERROR
    required_rules: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.fail_on is not None:
            object.__setattr__(self, "fail_on", Severity(self.fail_on))
        if self.required_rules is not None:
            object.__setattr__(self, "required_rules", tuple(self.required_rules))


@dataclass(frozen=True)
class GateDecision:
    outcome: GateOutcome
    fail_on: Severity | None
    blocking_findings: tuple[str, ...] = ()
    unmet_required: Mapping[str, str] = field(default_factory=dict)
    snapshot_findings: tuple[str, ...] = ()
    open_optional: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unmet_required", MappingProxyType(dict(self.unmet_required)))
        object.__setattr__(self, "open_optional", MappingProxyType(dict(self.open_optional)))

    def to_dict(self) -> JsonObject:
        return {"outcome": self.outcome.value,
                "fail_on": self.fail_on.value if self.fail_on else None,
                "blocking_findings": list(self.blocking_findings),
                "unmet_required": dict(self.unmet_required),
                "snapshot_findings": list(self.snapshot_findings),
                "open_optional": dict(self.open_optional)}


def evaluate_gate(report: AuditReport, policy: GatePolicy | None = None) -> GateDecision:
    """The gate decision for ``report`` under ``policy`` (default: fail on errors)."""
    policy = policy or GatePolicy()
    required = _required(report, policy)
    statuses = {r.rule_id: r.status for r in report.results}
    unmet = {rid: statuses[rid].value if rid in statuses else "absent"
             for rid in required if statuses.get(rid) not in _DEFINITE}
    open_optional = {rid: s.value for rid, s in statuses.items()
                     if rid not in required and s not in _DEFINITE and s is not RuleStatus.DISABLED}
    snapshot = tuple(f.finding_id for f in report.findings() if f.category == SNAPSHOT_CATEGORY)
    blocking = tuple(f.finding_id for f in report.findings()
                     if f.category != SNAPSHOT_CATEGORY and f.suppression is None
                     and _blocks(f.severity, policy.fail_on))
    unknown = any(RuleStatus(s) in _UNKNOWN for s in open_optional.values())
    outcome = (GateOutcome.FAIL if blocking else
               GateOutcome.INDETERMINATE if unmet or snapshot or unknown else GateOutcome.PASS)
    return GateDecision(outcome, policy.fail_on, blocking, unmet, snapshot, open_optional)


def _blocks(severity: Severity, fail_on: Severity | None) -> bool:
    return fail_on is not None and _RANK[severity] <= _RANK[fail_on]


def _required(report: AuditReport, policy: GatePolicy) -> tuple[str, ...]:
    if policy.required_rules is not None:
        return policy.required_rules
    if report.profile is None:
        return ()
    from knowlytix.knowledge.kgcheck.config import load_profile
    return load_profile(report.profile).required_rules
