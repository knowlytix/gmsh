# SPDX-License-Identifier: Apache-2.0
"""GMSH Enhanced Report Sections — claim-level dashboard additions.

Generates HTML sections that extend the base DOE-GMS dashboard with
GMSH-specific views:

- Failure attribution heatmap (FailureCode x QuestionCategory)
- Claim type coverage table
- Sub-verdict detail table
- Release gate summary card
- Provenance chain view

These are appended to the base report by ``generate_enhanced_report()``.
"""

from __future__ import annotations

import html as _html
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.graphdoe.results import DOEBenchmarkResult
    from .audit import ReleaseGate
    from .verdict import TypedVerdict


def _e(text: str) -> str:
    """HTML-escape."""
    return _html.escape(str(text))


def _pct(n: float) -> str:
    return f"{n:.0%}" if n == n else "—"


def _gate_color(passed: bool) -> str:
    return "#34d399" if passed else "#f87171"


# ---------------------------------------------------------------------------
# 1. Release Gate Card
# ---------------------------------------------------------------------------

def build_release_gate_section(gate: "ReleaseGate") -> str:
    """Generate a release gate summary card."""
    status = "PASSED" if gate.passed else "BLOCKED"
    color = _gate_color(gate.passed)
    icon = "&#10003;" if gate.passed else "&#10007;"

    blockers_html = ""
    if gate.blocking_failures:
        blockers = "".join(
            f"<li>{_e(b)}</li>" for b in gate.blocking_failures)
        blockers_html = f"""
        <div style="margin-top:1rem;">
          <strong style="color:#f87171;">Blocking Issues:</strong>
          <ul style="margin:0.5rem 0 0 1.5rem; color:#fbbf24;">{blockers}</ul>
        </div>"""

    return f"""
    <div class="section">
      <h2>Release Gate</h2>
      <div class="card" style="border-left: 4px solid {color};">
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
          <span style="font-size:2rem; color:{color};">{icon}</span>
          <span style="font-size:1.5rem; font-weight:bold; color:{color};">{status}</span>
        </div>
        <table>
          <tr>
            <td style="color:#8b8fa3;">Accuracy</td>
            <td>{gate.actual_accuracy:.1%} / {gate.accuracy_threshold:.1%} threshold</td>
          </tr>
          <tr>
            <td style="color:#8b8fa3;">Geometric Confidence</td>
            <td>{gate.actual_geo_confidence:.3f} / {gate.geo_confidence_threshold:.3f} threshold</td>
          </tr>
          <tr>
            <td style="color:#8b8fa3;">Failure Codes</td>
            <td>{len(gate.actual_failures)} distinct codes, {sum(gate.actual_failures.values())} total</td>
          </tr>
        </table>
        {blockers_html}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# 2. Failure Heatmap (FailureCode x QuestionCategory)
# ---------------------------------------------------------------------------

def build_failure_heatmap_section(
    result: "DOEBenchmarkResult",
    typed_verdicts: dict[str, "TypedVerdict"],
) -> str:
    """Generate a failure code x category heatmap."""
    heatmap: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    categories = set()
    codes = set()

    for qr in result.questions:
        tv = typed_verdicts.get(qr.doe_qid)
        if not tv:
            continue
        categories.add(qr.category)
        for failure in tv.failures:
            code = failure.code.value
            codes.add(code)
            heatmap[code][qr.category] += 1

    if not codes:
        return """
        <div class="section">
          <h2>Failure Heatmap</h2>
          <div class="card"><p style="color:#8b8fa3;">No failures detected.</p></div>
        </div>"""

    sorted_cats = sorted(categories)
    sorted_codes = sorted(codes)

    # Find max for color scaling
    max_count = max(
        heatmap[c][cat]
        for c in sorted_codes for cat in sorted_cats
        if heatmap[c][cat] > 0
    ) if codes else 1

    header = "".join(
        f"<th style='writing-mode:vertical-lr; transform:rotate(180deg); "
        f"font-size:0.7rem; padding:0.5rem 0.3rem;'>{_e(cat)}</th>"
        for cat in sorted_cats)

    rows = []
    for code in sorted_codes:
        cells = []
        for cat in sorted_cats:
            count = heatmap[code][cat]
            if count == 0:
                cells.append(
                    "<td style='text-align:center; color:#2e3348;'>·</td>")
            else:
                intensity = min(1.0, count / max_count)
                r = int(248 * intensity + 30 * (1 - intensity))
                g = int(113 * intensity + 51 * (1 - intensity))
                b = int(113 * intensity + 72 * (1 - intensity))
                cells.append(
                    f"<td style='text-align:center; "
                    f"background:rgba({r},{g},{b},0.3); "
                    f"color:#f87171; font-weight:bold;'>{count}</td>")
        rows.append(
            f"<tr><td style='font-size:0.75rem; white-space:nowrap;'>"
            f"{_e(code)}</td>{''.join(cells)}</tr>")

    return f"""
    <div class="section">
      <h2>Failure Attribution Heatmap</h2>
      <div class="card">
        <div style="overflow-x:auto;">
          <table>
            <thead><tr><th>Failure Code</th>{header}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# 3. Claim Type Coverage
# ---------------------------------------------------------------------------

def build_claim_coverage_section(
    typed_verdicts: dict[str, "TypedVerdict"],
) -> str:
    """Generate a claim type coverage table."""
    type_counts: Counter = Counter()
    type_passed: Counter = Counter()
    type_failed: Counter = Counter()

    for tv in typed_verdicts.values():
        for claim in tv.claims:
            type_counts[claim.claim_type] += 1
        for sv in tv.sub_verdicts:
            # Find claim type for this sub-verdict
            claim_type = "unknown"
            for c in tv.claims:
                if c.claim_id == sv.claim_id:
                    claim_type = c.claim_type
                    break
            if sv.passed:
                type_passed[claim_type] += 1
            else:
                type_failed[claim_type] += 1

    if not type_counts:
        return """
        <div class="section">
          <h2>Claim Coverage</h2>
          <div class="card"><p style="color:#8b8fa3;">No typed claims extracted.</p></div>
        </div>"""

    rows = []
    for ctype in sorted(type_counts.keys()):
        total = type_counts[ctype]
        passed = type_passed.get(ctype, 0)
        failed = type_failed.get(ctype, 0)
        verified = passed + failed
        rate = passed / verified if verified > 0 else 0
        color = "#34d399" if rate >= 0.7 else "#fbbf24" if rate >= 0.4 else "#f87171"
        bar_width = int(rate * 100)

        rows.append(f"""
        <tr>
          <td style="font-weight:bold;">{_e(ctype)}</td>
          <td style="text-align:center;">{total}</td>
          <td style="text-align:center; color:#34d399;">{passed}</td>
          <td style="text-align:center; color:#f87171;">{failed}</td>
          <td>
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <div style="flex:1; background:#1a1d27; border-radius:3px; height:8px;">
                <div style="width:{bar_width}%; background:{color}; height:100%; border-radius:3px;"></div>
              </div>
              <span style="font-size:0.75rem; color:{color};">{rate:.0%}</span>
            </div>
          </td>
        </tr>""")

    return f"""
    <div class="section">
      <h2>Claim Type Coverage</h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th>Claim Type</th>
              <th style="text-align:center;">Total</th>
              <th style="text-align:center;">Passed</th>
              <th style="text-align:center;">Failed</th>
              <th>Pass Rate</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# 4. Verifier Detail Table
# ---------------------------------------------------------------------------

def build_verifier_detail_section(
    typed_verdicts: dict[str, "TypedVerdict"],
) -> str:
    """Generate a per-verifier summary table."""
    verifier_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "passed": 0, "methods": Counter()})

    for tv in typed_verdicts.values():
        for sv in tv.sub_verdicts:
            stats = verifier_stats[sv.verifier_name]
            stats["total"] += 1
            if sv.passed:
                stats["passed"] += 1
            if sv.gms_method_used:
                stats["methods"][sv.gms_method_used] += 1

    if not verifier_stats:
        return ""

    rows = []
    for name in sorted(verifier_stats.keys()):
        s = verifier_stats[name]
        rate = s["passed"] / s["total"] if s["total"] > 0 else 0
        color = "#34d399" if rate >= 0.7 else "#fbbf24" if rate >= 0.4 else "#f87171"
        methods = ", ".join(
            f"{m} ({n})" for m, n in s["methods"].most_common(3))

        rows.append(f"""
        <tr>
          <td style="font-weight:bold;">{_e(name)}</td>
          <td style="text-align:center;">{s['total']}</td>
          <td style="text-align:center; color:#34d399;">{s['passed']}</td>
          <td style="text-align:center; color:#f87171;">{s['total'] - s['passed']}</td>
          <td style="color:{color};">{rate:.0%}</td>
          <td style="font-size:0.75rem; color:#8b8fa3;">{_e(methods)}</td>
        </tr>""")

    return f"""
    <div class="section">
      <h2>Verifier Detail</h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th>Verifier</th>
              <th style="text-align:center;">Total</th>
              <th style="text-align:center;">Passed</th>
              <th style="text-align:center;">Failed</th>
              <th>Rate</th>
              <th>GMS Methods</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# 5. Provenance Chain View
# ---------------------------------------------------------------------------

def build_provenance_section(
    typed_verdicts: dict[str, "TypedVerdict"],
) -> str:
    """Generate a provenance summary showing extraction methods."""
    method_counts: Counter = Counter()
    has_provenance = 0
    total_claims = 0

    for tv in typed_verdicts.values():
        for claim in tv.claims:
            total_claims += 1
            if claim.provenance:
                has_provenance += 1
                method_counts[claim.provenance.extraction_method] += 1

    if total_claims == 0:
        return ""

    rows = []
    for method in sorted(method_counts.keys()):
        count = method_counts[method]
        pct = count / total_claims if total_claims > 0 else 0
        rows.append(f"""
        <tr>
          <td>{_e(method)}</td>
          <td style="text-align:center;">{count}</td>
          <td style="text-align:center;">{pct:.0%}</td>
        </tr>""")

    coverage_pct = has_provenance / total_claims if total_claims > 0 else 0
    coverage_color = ("#34d399" if coverage_pct >= 0.8
                      else "#fbbf24" if coverage_pct >= 0.5
                      else "#f87171")

    return f"""
    <div class="section">
      <h2>Provenance Chain</h2>
      <div class="card">
        <p style="margin-bottom:1rem;">
          <strong>Claims with provenance:</strong>
          <span style="color:{coverage_color};">{has_provenance}/{total_claims} ({coverage_pct:.0%})</span>
        </p>
        <table>
          <thead>
            <tr>
              <th>Extraction Method</th>
              <th style="text-align:center;">Count</th>
              <th style="text-align:center;">% of Claims</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Composite builder
# ---------------------------------------------------------------------------

def build_tiered_gate_section(tiered_gates: list) -> str:
    """Generate HTML for risk-tiered release gates.

    Parameters
    ----------
    tiered_gates : list[TieredReleaseGate]
        Release gate results per risk tier.

    Returns
    -------
    str
        HTML section with per-tier cards showing pass/fail status.
    """
    if not tiered_gates:
        return ""

    cards = []
    for tg in tiered_gates:
        tier_name = tg.tier.tier_name if tg.tier else "unknown"
        status = "PASSED" if tg.passed else "BLOCKED"
        color = "#34d399" if tg.passed else "#f87171"
        icon = "&#10003;" if tg.passed else "&#10007;"

        # Per-check rows
        check_rows = ""
        for check_name, check_data in (tg.checks or {}).items():
            check_passed = check_data.get("passed", False)
            check_color = "#34d399" if check_passed else "#f87171"
            threshold = check_data.get("threshold", "—")
            actual = check_data.get("actual", "—")
            if isinstance(threshold, float):
                threshold = f"{threshold:.3f}"
            if isinstance(actual, float):
                actual = f"{actual:.3f}"
            check_icon = "&#10003;" if check_passed else "&#10007;"
            check_rows += f"""
            <tr>
              <td style="color:#8b8fa3;">{_e(check_name)}</td>
              <td>{actual}</td>
              <td>{threshold}</td>
              <td style="color:{check_color};">{check_icon}</td>
            </tr>"""

        # Blocking reasons
        blockers_html = ""
        if tg.blocking_reasons:
            items = "".join(
                f"<li>{_e(r)}</li>" for r in tg.blocking_reasons)
            blockers_html = f"""
            <div style="margin-top:0.8rem;">
              <strong style="color:#f87171;">Blockers:</strong>
              <ul style="margin:0.3rem 0 0 1.5rem; color:#fbbf24; font-size:0.8rem;">
                {items}
              </ul>
            </div>"""

        cards.append(f"""
        <div class="card" style="border-left: 4px solid {color}; margin-bottom:1rem;">
          <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.8rem;">
            <span style="font-size:1.5rem; color:{color};">{icon}</span>
            <span style="font-size:1.1rem; font-weight:bold; color:{color};">{_e(tier_name).upper()}: {status}</span>
          </div>
          <table style="font-size:0.85rem;">
            <thead><tr><th>Check</th><th>Actual</th><th>Threshold</th><th>Status</th></tr></thead>
            <tbody>{check_rows}</tbody>
          </table>
          {blockers_html}
        </div>""")

    return f"""
    <div class="section">
      <h2>Risk-Tiered Release Gates</h2>
      {''.join(cards)}
    </div>"""


def build_enhanced_sections(
    result: "DOEBenchmarkResult",
    typed_verdicts: dict[str, "TypedVerdict"],
    gate: "ReleaseGate | None" = None,
    tiered_gates: list | None = None,
) -> str:
    """Build all enhanced HTML sections for the GMSH dashboard.

    Returns a single HTML string that can be appended to the base report.
    """
    parts = []

    if gate:
        parts.append(build_release_gate_section(gate))

    if tiered_gates:
        parts.append(build_tiered_gate_section(tiered_gates))

    parts.append(build_failure_heatmap_section(result, typed_verdicts))
    parts.append(build_claim_coverage_section(typed_verdicts))
    parts.append(build_verifier_detail_section(typed_verdicts))
    parts.append(build_provenance_section(typed_verdicts))

    return "\n".join(parts)
