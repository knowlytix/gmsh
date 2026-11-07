# SPDX-License-Identifier: Apache-2.0
"""Compare two audit reports: what was added, resolved, unchanged or no longer evaluated.

Reports are compared in their canonical JSON form, so a saved report from an
earlier run can be compared with a new one. A finding counts as resolved only
when the later report ran the same version of its rule to a definite result
with full details and no longer lists it. When the rule did not reach a
definite result there (unavailable, error, partial, disabled or not run), the
finding is ``no_longer_evaluated``. When the other report truncated the rule's
details, its presence is ``unknown``. A finding that disappears while exactly
one finding of the same rule, discriminator and view appears is ``changed``:
it is the same issue on a different set of targets, as when a summary finding's
list of affected terms grows. When the rule's version differs, its
findings cannot be matched (identity includes the version) and the rule is
listed under ``rule_version_changed``. A different artifact contract, adapter
version or report format makes the whole comparison noncomparable; the
buckets are still computed and the reasons are stated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from knowlytix.knowledge.kgcheck.models import AuditReport, JsonObject

_DEFINITE = frozenset({"passed", "findings", "not_applicable"})
COMPARISON_VERSION = "kgcheck-comparison/1"


@dataclass(frozen=True)
class ReportComparison:
    """Finding ids by bucket, with the rules whose status or version changed."""

    comparable: bool
    reasons: tuple[str, ...]
    before: str
    after: str
    buckets: Mapping[str, tuple[str, ...]]
    rules: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {"comparison_version": COMPARISON_VERSION, "comparable": self.comparable,
                "reasons": list(self.reasons), "before": self.before, "after": self.after,
                "findings": {k: list(v) for k, v in self.buckets.items()},
                "counts": {k: len(v) for k, v in self.buckets.items()},
                "rules": {k: dict(v) for k, v in self.rules.items()}}


def _as_dict(report: AuditReport | Mapping) -> Mapping:
    if isinstance(report, AuditReport):
        from knowlytix.knowledge.kgcheck.reporting.json_report import report_to_dict
        return report_to_dict(report)
    return report


def compare_reports(before: AuditReport | Mapping, after: AuditReport | Mapping) -> ReportComparison:
    """How ``after`` differs from ``before``; either may be a report or its JSON form."""
    b, a = _as_dict(before), _as_dict(after)
    reasons = _noncomparable(b, a)
    results_b = {r["rule_id"]: r for r in b["results"]}
    results_a = {r["rule_id"]: r for r in a["results"]}
    findings_b = {f["finding_id"]: f for f in b["findings"]}
    findings_a = {f["finding_id"]: f for f in a["findings"]}
    changed = sorted(r for r in set(results_b) & set(results_a)
                     if results_b[r]["rule_version"] != results_a[r]["rule_version"])
    buckets: dict[str, list[str]] = {k: [] for k in (
        "added", "resolved", "changed", "unchanged", "suppressed", "no_longer_evaluated",
        "newly_evaluated", "unknown", "rule_version_changed")}
    for fid, f in sorted(findings_b.items()):
        if fid in findings_a:
            buckets["suppressed" if findings_a[fid].get("suppression") else "unchanged"].append(fid)
        else:
            buckets[_gone(f["rule_id"], results_a, changed)].append(fid)
    for fid, f in sorted(findings_a.items()):
        if fid not in findings_b:
            bucket = _new(f["rule_id"], results_b, changed)
            buckets["suppressed" if bucket == "added" and f.get("suppression") else bucket].append(fid)
    _pair_changed(buckets, findings_b, findings_a)
    rules = {r: {"before": results_b.get(r, {}).get("status"),
                 "after": results_a.get(r, {}).get("status"),
                 "version_changed": r in changed}
             for r in sorted(set(results_b) | set(results_a))
             if results_b.get(r, {}).get("status") != results_a.get(r, {}).get("status")
             or r in changed}
    return ReportComparison(not reasons, tuple(reasons), b["snapshot"]["snapshot_id"],
                            a["snapshot"]["snapshot_id"],
                            {k: tuple(v) for k, v in buckets.items()}, rules)


def _gone(rule_id: str, results_after: Mapping[str, Mapping], changed: list[str]) -> str:
    result = results_after.get(rule_id)
    if rule_id in changed:
        return "rule_version_changed"
    if result is None or result["status"] not in _DEFINITE:
        return "no_longer_evaluated"
    return "unknown" if result.get("details_truncated") else "resolved"


def _new(rule_id: str, results_before: Mapping[str, Mapping], changed: list[str]) -> str:
    result = results_before.get(rule_id)
    if rule_id in changed:
        return "rule_version_changed"
    if result is None or result["status"] not in _DEFINITE:
        return "newly_evaluated"
    return "unknown" if result.get("details_truncated") else "added"


def _issue(f: Mapping) -> tuple:
    return f["rule_id"], f.get("discriminator", ""), f.get("view")


def _pair_changed(buckets: dict[str, list[str]], before: Mapping, after: Mapping) -> None:
    """Move a resolved and an added finding of one issue to ``changed`` (see the module)."""
    gone: dict[tuple, list[str]] = {}
    new: dict[tuple, list[str]] = {}
    for fid in buckets["resolved"]:
        gone.setdefault(_issue(before[fid]), []).append(fid)
    for fid in buckets["added"]:
        new.setdefault(_issue(after[fid]), []).append(fid)
    for issue, ids in gone.items():
        if len(ids) == 1 and len(new.get(issue, ())) == 1 and issue[1]:
            buckets["resolved"].remove(ids[0])
            buckets["added"].remove(new[issue][0])
            buckets["changed"].append(f"{ids[0]}->{new[issue][0]}")


def _noncomparable(b: Mapping, a: Mapping) -> list[str]:
    reasons = []
    if b.get("report_version") != a.get("report_version"):
        reasons.append("report format differs")
    for key, what in (("contract_version", "artifact contract"), ("adapter", "adapter"),
                      ("adapter_version", "adapter version")):
        if b["snapshot"].get(key) != a["snapshot"].get(key):
            reasons.append(f"{what} differs: {b['snapshot'].get(key)} -> {a['snapshot'].get(key)}")
    if b.get("views") != a.get("views"):
        reasons.append(f"views differ: {b.get('views')} -> {a.get('views')}")
    if b.get("profile") != a.get("profile"):
        reasons.append(f"profiles differ: {b.get('profile')} -> {a.get('profile')}")
    return reasons
