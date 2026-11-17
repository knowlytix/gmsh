# SPDX-License-Identifier: Apache-2.0
"""Human-readable Markdown rendering of an audit report; the JSON report stays canonical.

Text that comes from the audited store (labels, source excerpts) is escaped
for Markdown and witnesses are printed in bounded code blocks, so store content
cannot change the report's structure. ``excerpts=False`` omits witnesses,
keeping rule, message and artifact locations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowlytix.knowledge.kgcheck.frozen import thaw
from knowlytix.knowledge.kgcheck.gate import GateDecision
from knowlytix.knowledge.kgcheck.models import (
    AuditReport,
    Finding,
    RuleResult,
    Severity,
)

_ESCAPE = re.compile(r"([\\`*_\[\]<>|#~])")
# GFM turns bare www./http(s):// text into links; a word joiner (U+2060) after the
# scheme or "www" keeps store text from becoming a clickable link.
_AUTOLINK = re.compile(r"(?i)\b(https?|ftp|www)(?=[:.])")
_WITNESS_CHARS = 1200
_SEVERITIES = (Severity.ERROR, Severity.WARNING, Severity.INFO, Severity.SUGGESTION)


def escape(text: str) -> str:
    """Markdown-inert text on one line."""
    escaped = _ESCAPE.sub(r"\\\1", " ".join(str(text).split()))
    return _AUTOLINK.sub("\\1\u2060", escaped)


def _code(text: str) -> str:
    fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{fence}json\n{text}\n{fence}"


def _inline(text: str) -> str:
    """A code span that holds ``text`` verbatim, whatever backticks it contains."""
    text = " ".join(str(text).split())
    ticks = "`" * (max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def _titles() -> dict[str, str]:
    from knowlytix.knowledge.kgcheck.registry import builtin_registry
    return {rid: r.spec.title for rid, r in builtin_registry().rules().items()}


def _header(report: AuditReport, gate: GateDecision | None) -> list[str]:
    caps = report.capabilities
    views = ", ".join(v.value for v in report.views) or "none"
    lines = ["# kgcheck audit report", "",
             f"- Snapshot: `{report.snapshot_id}` read by `{caps.adapter}` {caps.adapter_version}, "
             f"contract `{report.contract_version}`",
             f"- Profile: `{report.profile}`; view: `{views}`",
             f"- Execution: {'complete' if report.is_complete() else 'incomplete'}"]
    if gate is not None:
        lines.append(f"- Gate: **{gate.outcome.value}** (fail on "
                     f"{gate.fail_on.value if gate.fail_on else 'nothing, report only'})")
        lines += _gate_reasons(gate)
    return [*lines, ""]


def _gate_reasons(gate: GateDecision) -> list[str]:
    out = []
    if gate.blocking_findings:
        out.append(f"  - {len(gate.blocking_findings)} blocking finding(s)")
    if gate.unmet_required:
        out.append("  - required rules without a definite result: "
                   + ", ".join(f"{r} ({s})" for r, s in gate.unmet_required.items()))
    if gate.snapshot_findings:
        out.append("  - the store changed while it was read")
    if gate.open_optional:
        out.append("  - optional rules not run to completion: "
                   + ", ".join(f"{r} ({s})" for r, s in gate.open_optional.items()))
    return out


def _summary(report: AuditReport) -> list[str]:
    counts = report.finding_counts()
    rows = [f"| {s.value} | {counts.by_severity.get(s.value, 0)} |" for s in _SEVERITIES]
    note = ([f"{counts.listed} of {counts.total} findings are listed in detail.", ""]
            if counts.listed < counts.total else [])
    return ["## Findings by severity", "", "| Severity | Findings |", "|---|---|", *rows, "",
            *note]


def _rules(report: AuditReport) -> list[str]:
    titles = _titles()
    lines = ["## Rules", "", "| Rule | Title | Status | Eligible | Tested | Findings | Notes |",
             "|---|---|---|---|---|---|---|"]
    lines += [_rule_row(r, titles.get(r.rule_id, "")) for r in report.results]
    return [*lines, ""]


def _rule_row(r: RuleResult, title: str) -> str:
    total = r.metrics.get("findings_total", len(r.findings))
    notes = []
    if r.missing_prerequisites:
        notes.append("missing: " + ", ".join(r.missing_prerequisites))
    if r.error:
        notes.append(r.error)
    if r.details_truncated:
        notes.append(f"{len(r.findings)} of {total} findings listed")
    cells = [r.rule_id, title, r.status.value, _count(r.eligible), _count(r.tested), str(total),
             "; ".join(notes)]
    return "| " + " | ".join(escape(c) for c in cells) + " |"


def _count(n: int | None) -> str:
    return "" if n is None else str(n)


def _findings(report: AuditReport, excerpts: bool) -> list[str]:
    findings = report.findings()
    if not findings:
        return ["## Findings", "", "No findings.", ""]
    lines = ["## Findings", ""]
    for severity in _SEVERITIES:
        group = [f for f in findings if f.severity is severity]
        if group:
            lines += [f"### {severity.value.capitalize()} ({len(group)})", ""]
            lines += [line for f in group for line in _finding(f, excerpts)]
    return lines


def _finding(f: Finding, excerpts: bool) -> list[str]:
    where = ", ".join(_inline(f"{r.artifact}#{r.pointer}") for r in f.locations[:10])
    lines = [f"#### {f.rule_id}: {escape(f.message)}", "",
             f"- Finding {_inline(f.finding_id)}, category {escape(f.category)}"
             + (f", view {f.view.value}" if f.view else ""),
             f"- Basis: {escape(f.basis)}"]
    if where:
        lines.append(f"- Locations: {where}" + (" …" if len(f.locations) > 10 else ""))
    if f.suggestion:
        lines.append(f"- Review: {escape(f.suggestion)}")
    if f.suppression is not None:
        lines.append(f"- Suppressed: {escape(str(f.suppression['reason']))} "
                     f"(rule version {escape(str(f.suppression['rule_version']))})")
    if excerpts and f.witness:
        text = json.dumps(thaw(f.witness), ensure_ascii=False, indent=1)
        clipped = text if len(text) <= _WITNESS_CHARS else text[:_WITNESS_CHARS] + "\n…"
        lines += ["", _code(clipped)]
    return [*lines, ""]


def _diagnostics(report: AuditReport) -> list[str]:
    if not report.diagnostics:
        return []
    lines = ["## Read diagnostics", "", "| Code | Location | Message |", "|---|---|---|"]
    lines += [f"| {escape(d.code)} | {escape(d.location.artifact + '#' + d.location.pointer)} | "
              f"{escape(d.message)} |" for d in report.diagnostics]
    return [*lines, ""]


def render_markdown(report: AuditReport, gate: GateDecision | None = None, *,
                    excerpts: bool = True) -> str:
    """The report as Markdown, with the gate decision when one is given."""
    parts = [*_header(report, gate), *_summary(report), *_rules(report),
             *_findings(report, excerpts), *_diagnostics(report)]
    return "\n".join(parts).rstrip() + "\n"


def write_markdown(report: AuditReport, path: str | Path, gate: GateDecision | None = None, *,
                   excerpts: bool = True) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report, gate, excerpts=excerpts), encoding="utf-8")
    return out
