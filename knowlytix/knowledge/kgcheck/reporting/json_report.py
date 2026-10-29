# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON form of an ``AuditReport`` (schema ``schemas/report.schema.json``).

Output is deterministic: results by rule id, findings by rule id then finding id,
fixed key order, strict JSON (no NaN). Everything outside ``run`` is the stable
payload; ``stable_payload`` drops ``run`` for comparing two audits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from pathlib import Path

from knowlytix.knowledge.kgcheck.frozen import thaw
from knowlytix.knowledge.kgcheck.gate import GateDecision
from knowlytix.knowledge.kgcheck.models import (
    ArtifactRecord,
    AuditReport,
    Capabilities,
    Diagnostic,
    Finding,
    JsonObject,
    RawRef,
    RuleResult,
)

REPORT_VERSION = "kgcheck-report/1"
REPORT_SCHEMA = "report.schema.json"


@dataclass(frozen=True)
class RunInfo:
    """What varies between runs; kept out of the stable payload."""

    created_at: str | None = None
    tool_version: str = ""
    configuration: JsonObject = field(default_factory=dict)


@cache
def _schema_text() -> str:
    return files("knowlytix.knowledge.kgcheck").joinpath("schemas", REPORT_SCHEMA).read_text(
        encoding="utf-8")


def report_schema() -> JsonObject:
    """The report JSON Schema (a fresh copy on every call)."""
    return json.loads(_schema_text())


def _location(ref: RawRef) -> JsonObject:
    return {"artifact": ref.artifact, "pointer": ref.pointer}


def _artifact(record: ArtifactRecord) -> JsonObject:
    return {"name": record.name, "present": record.present, "required": record.required,
            "size": record.size, "sha256": record.sha256}


def _capabilities(caps: Capabilities) -> JsonObject:
    return {k: v for k, v in vars(caps).items() if isinstance(v, bool)}


def _finding(f: Finding) -> JsonObject:
    return {"finding_id": f.finding_id, "rule_id": f.rule_id, "rule_version": f.rule_version,
            "category": f.category, "severity": f.severity.value,
            "view": f.view.value if f.view else None, "message": f.message, "basis": f.basis,
            "affected": list(f.affected), "locations": [_location(r) for r in f.locations],
            "witness": thaw(f.witness), "suggestion": f.suggestion,
            "heuristic_score": f.heuristic_score, "discriminator": f.discriminator,
            "suppression": thaw(f.suppression) if f.suppression is not None else None}


def _result(r: RuleResult) -> JsonObject:
    return {"rule_id": r.rule_id, "rule_version": r.rule_version, "status": r.status.value,
            "eligible": r.eligible, "tested": r.tested,
            "missing_prerequisites": list(r.missing_prerequisites), "error": r.error,
            "details_truncated": r.details_truncated, "metrics": thaw(r.metrics),
            "finding_ids": sorted(f.finding_id for f in r.findings)}


def _diagnostic(d: Diagnostic) -> JsonObject:
    return {"code": d.code, "message": d.message, "location": _location(d.location)}


def _summary(report: AuditReport) -> JsonObject:
    counts = report.finding_counts()
    return {"rules": len(report.results),
            "rules_by_status": {s.value: list(ids) for s, ids in report.by_status().items()},
            "findings": counts.total, "findings_listed": counts.listed,
            "findings_suppressed": _suppressed(report),
            "findings_by_severity": counts.by_severity,
            "findings_by_category": counts.by_category}


def _suppressed(report: AuditReport) -> int:
    """Suppressed findings, counted from each rule's exact total when it has one."""
    total = 0
    for r in report.results:
        exact = r.metrics.get("suppressed")
        total += exact if isinstance(exact, int) else sum(f.suppression is not None for f in r.findings)
    return total


def _snapshot(report: AuditReport) -> JsonObject:
    caps = report.capabilities
    return {"snapshot_id": report.snapshot_id, "contract_version": report.contract_version,
            "adapter": caps.adapter, "adapter_version": caps.adapter_version,
            "capabilities": _capabilities(caps),
            "artifacts": [_artifact(a) for a in sorted(report.artifacts, key=lambda a: a.name)]}


def _run(report: AuditReport, run: RunInfo) -> JsonObject:
    return {"created_at": run.created_at, "tool_version": run.tool_version,
            "runtime_s": {r.rule_id: r.runtime_s for r in report.results},
            "configuration": thaw(run.configuration)}


def report_to_dict(report: AuditReport, run: RunInfo | None = None,
                   gate: GateDecision | None = None) -> JsonObject:
    """The report as plain JSON-ready data, in canonical order, with an optional gate decision."""
    return {"report_version": REPORT_VERSION, "snapshot": _snapshot(report),
            "profile": report.profile, "views": [v.value for v in report.views],
            "complete": report.is_complete(), "gate": gate.to_dict() if gate else None,
            "summary": _summary(report),
            "results": [_result(r) for r in report.results],
            "findings": [_finding(f) for f in report.findings()],
            "diagnostics": [_diagnostic(d) for d in report.diagnostics],
            "run": _run(report, run or RunInfo())}


def stable_payload(report_json: JsonObject) -> JsonObject:
    """The report without its ``run`` section: equal for equal audits."""
    return {k: v for k, v in report_json.items() if k != "run"}


def dumps(report: AuditReport, run: RunInfo | None = None, gate: GateDecision | None = None) -> str:
    """Canonical JSON text: strict (no NaN), UTF-8, two-space indent, trailing newline."""
    return json.dumps(report_to_dict(report, run, gate), ensure_ascii=False, allow_nan=False,
                      indent=2) + "\n"


def write_json(report: AuditReport, path: str | Path, run: RunInfo | None = None,
               gate: GateDecision | None = None) -> Path:
    """Write the canonical JSON to ``path`` (parents created) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(report, run, gate), encoding="utf-8")
    return out
