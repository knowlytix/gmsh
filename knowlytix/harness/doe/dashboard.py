# SPDX-License-Identifier: Apache-2.0
"""Self-contained HTML report dashboard for DOE-GMS benchmark results.

Generates a single HTML file with embedded CSS — no external dependencies.

Usage::

    from knowlytix.harness.doe.dashboard import generate_report
    generate_report(result, analyzer, "report.html")

Or from CLI::

    python -m harness.doe report.md --report dashboard.html
"""

from __future__ import annotations

import html
from typing import Any

try:
    from knowlytix.harness.graphdoe.results import DOEBenchmarkResult
except ImportError:  # pragma: no cover
    # See harness/doe/benchmark.py for the graphdoe-optional rationale.
    class DOEBenchmarkResult:  # type: ignore[no-redef]
        """Stub surfaced when graphdoe is absent; report functions raise at use time."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "harness.doe.dashboard requires the Knowlytix-internal "
                "`graphdoe` package; install graphdoe to generate DOE reports."
            )

try:
    from knowlytix.harness.graphdoe.analysis import DOEAnalyzer
except ImportError:
    DOEAnalyzer = None


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #232735;
  --border: #2e3348;
  --text: #e4e6f0;
  --text-dim: #8b8fa3;
  --accent: #6c8cff;
  --accent2: #a78bfa;
  --green: #34d399;
  --red: #f87171;
  --amber: #fbbf24;
  --cyan: #22d3ee;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 2rem;
}

.container { max-width: 1400px; margin: 0 auto; }

h1 {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.25rem;
}

h2 {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--accent);
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

h3 {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--accent2);
  margin-bottom: 0.5rem;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 2px solid var(--border);
}

.header-left { flex: 1; }
.header-right {
  text-align: right;
  color: var(--text-dim);
  font-size: 0.8rem;
}

.subtitle {
  color: var(--text-dim);
  font-size: 0.85rem;
}

/* Cards grid */
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.2rem;
}

.card-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-dim);
  margin-bottom: 0.3rem;
}

.card-value {
  font-size: 1.6rem;
  font-weight: 700;
}

.card-sub {
  font-size: 0.75rem;
  color: var(--text-dim);
  margin-top: 0.2rem;
}

.card-value.green { color: var(--green); }
.card-value.amber { color: var(--amber); }
.card-value.red { color: var(--red); }
.card-value.accent { color: var(--accent); }
.card-value.cyan { color: var(--cyan); }

/* Section */
.section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}

th {
  text-align: left;
  padding: 0.5rem 0.7rem;
  border-bottom: 2px solid var(--border);
  color: var(--text-dim);
  font-weight: 600;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

th.right, td.right { text-align: right; }
th.center, td.center { text-align: center; }

td {
  padding: 0.45rem 0.7rem;
  border-bottom: 1px solid var(--border);
}

tr:hover { background: var(--surface2); }

.pass { color: var(--green); }
.fail { color: var(--red); }
.sig { color: var(--amber); font-weight: 700; }
.nosig { color: var(--text-dim); }

/* Bar chart (pure CSS) */
.bar-row {
  display: flex;
  align-items: center;
  margin-bottom: 0.4rem;
}

.bar-label {
  width: 160px;
  font-size: 0.78rem;
  text-align: right;
  padding-right: 0.8rem;
  color: var(--text-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bar-track {
  flex: 1;
  height: 22px;
  background: var(--surface2);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}

.bar-fill.green { background: var(--green); }
.bar-fill.accent { background: var(--accent); }
.bar-fill.amber { background: var(--amber); }
.bar-fill.red { background: var(--red); }
.bar-fill.cyan { background: var(--cyan); }

.bar-value {
  position: absolute;
  right: 8px;
  top: 2px;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text);
}

/* Two-column layout */
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

@media (max-width: 900px) {
  .two-col { grid-template-columns: 1fr; }
}

/* Detail table */
.detail-controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.filter-btn {
  background: var(--surface2);
  color: var(--text-dim);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.3rem 0.8rem;
  font-size: 0.75rem;
  cursor: pointer;
  font-family: inherit;
}

.filter-btn:hover, .filter-btn.active {
  background: var(--accent);
  color: var(--bg);
  border-color: var(--accent);
}

.detail-table-wrap {
  max-height: 500px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 4px;
}

.detail-table-wrap table th {
  position: sticky;
  top: 0;
  background: var(--surface);
  z-index: 1;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Tooltip */
.tt { position: relative; cursor: help; }
.tt:hover::after {
  content: attr(data-tip);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--surface2);
  color: var(--text);
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  font-size: 0.7rem;
  white-space: nowrap;
  border: 1px solid var(--border);
  z-index: 10;
}

/* Heatmap cell */
.heat {
  display: inline-block;
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  font-weight: 600;
  font-size: 0.78rem;
}

.footer {
  text-align: center;
  padding: 1.5rem 0 0.5rem;
  color: var(--text-dim);
  font-size: 0.7rem;
}
"""

# ---------------------------------------------------------------------------
# JavaScript (filter/sort for detail table)
# ---------------------------------------------------------------------------

_JS = """\
function filterTable(cat) {
  const rows = document.querySelectorAll('#detail-body tr');
  const btns = document.querySelectorAll('.filter-btn');
  btns.forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  rows.forEach(r => {
    if (cat === 'all') { r.style.display = ''; return; }
    if (cat === 'fail') { r.style.display = r.dataset.correct === '0' ? '' : 'none'; return; }
    r.style.display = r.dataset.category === cat ? '' : 'none';
  });
}
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _e(text: str) -> str:
    """HTML-escape."""
    return html.escape(str(text))


def _pct(n: float) -> str:
    return f"{n:.1%}"


def _acc_color(rate: float) -> str:
    if rate >= 0.9:
        return "green"
    if rate >= 0.7:
        return "amber"
    return "red"


def _heat_color(rate: float) -> str:
    """Return inline style for heatmap cell."""
    if rate >= 0.9:
        return "background:rgba(52,211,153,0.25);color:#34d399"
    if rate >= 0.7:
        return "background:rgba(251,191,36,0.2);color:#fbbf24"
    return "background:rgba(248,113,113,0.2);color:#f87171"


def _geo_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "amber"
    return "red"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header(result: DOEBenchmarkResult) -> str:
    acc = result.accuracy
    color = _acc_color(acc)
    return f"""\
<div class="header">
  <div class="header-left">
    <h1>DOE-GMS Benchmark Report</h1>
    <div class="subtitle">{_e(result.instance_name)} &mdash; {_e(result.model)}</div>
  </div>
  <div class="header-right">
    <div>{_e(result.timestamp)}</div>
    <div>Context: {_e(result.context_mode)}</div>
    <div>Factors: {', '.join(_e(f) for f in result.factors)}</div>
  </div>
</div>
"""


def _build_summary_cards(result: DOEBenchmarkResult, df) -> str:
    acc = result.accuracy
    color = _acc_color(acc)
    n_cats = df["category"].nunique() if df is not None and not df.empty else 0
    n_factors = len(result.factors)

    # Geometric scores if available
    geo_html = ""
    if df is not None and not df.empty and "_overall_confidence" in df.columns:
        geo = df["_overall_confidence"].mean()
        geo_html = f"""\
<div class="card">
  <div class="card-label">Geo Confidence</div>
  <div class="card-value {_geo_color(geo)}">{geo:.3f}</div>
  <div class="card-sub">weighted geometric score</div>
</div>"""

    stats = result.graph_stats
    graph_info = ""
    if stats:
        graph_info = f"""\
<div class="card">
  <div class="card-label">GMS Store</div>
  <div class="card-value accent">{stats.get('entities', 0)}</div>
  <div class="card-sub">{stats.get('triples', 0)} triples, {stats.get('enm_entries', 0)} ENM</div>
</div>"""

    return f"""\
<div class="cards">
  <div class="card">
    <div class="card-label">Accuracy</div>
    <div class="card-value {color}">{_pct(acc)}</div>
    <div class="card-sub">{result.total_correct} / {result.total_questions}</div>
  </div>
  <div class="card">
    <div class="card-label">Questions</div>
    <div class="card-value accent">{result.total_questions}</div>
    <div class="card-sub">{result.n_base_questions} base x {result.n_design_runs} runs</div>
  </div>
  <div class="card">
    <div class="card-label">Categories</div>
    <div class="card-value cyan">{n_cats}</div>
    <div class="card-sub">{n_factors} DOE factors</div>
  </div>
  {geo_html}
  {graph_info}
</div>
"""


def _build_category_section(df) -> str:
    if df is None or df.empty:
        return ""

    rows_html = ""
    bars_html = ""

    cats = sorted(df["category"].unique())
    has_geo = "_overall_confidence" in df.columns

    for cat in cats:
        sub = df[df["category"] == cat]
        correct = int(sub["correct"].sum())
        total = len(sub)
        rate = correct / total if total else 0
        color = _acc_color(rate)

        geo_td = ""
        tension_td = ""
        if has_geo:
            geo = sub["_overall_confidence"].mean()
            ten = sub["_tension"].mean()
            geo_td = f'<td class="right"><span class="heat" style="{_heat_color(geo)}">{geo:.3f}</span></td>'
            tension_td = f'<td class="right">{ten:.3f}</td>'

        rows_html += f"""\
<tr>
  <td>{_e(cat)}</td>
  <td class="right">{correct}</td>
  <td class="right">{total}</td>
  <td class="right"><span class="heat" style="{_heat_color(rate)}">{_pct(rate)}</span></td>
  {geo_td}{tension_td}
</tr>
"""

        bars_html += f"""\
<div class="bar-row">
  <div class="bar-label">{_e(cat)}</div>
  <div class="bar-track">
    <div class="bar-fill {color}" style="width:{rate*100:.1f}%"></div>
    <div class="bar-value">{_pct(rate)}</div>
  </div>
</div>
"""

    geo_th = ""
    if has_geo:
        geo_th = '<th class="right">Geo Conf</th><th class="right">Tension</th>'

    return f"""\
<div class="section">
  <h2>Category Breakdown</h2>
  <div class="two-col">
    <div>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th class="right">Correct</th>
            <th class="right">Total</th>
            <th class="right">Rate</th>
            {geo_th}
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div>
      <h3>Accuracy by Category</h3>
      {bars_html}
    </div>
  </div>
</div>
"""


def _build_geometric_section(df) -> str:
    if df is None or df.empty:
        return ""
    if "_overall_confidence" not in df.columns:
        return ""

    metrics = [
        ("Geodesic Confidence", "_geo_confidence",
         "exp(-mean_geodesic) on S^{m-1}"),
        ("Tension Score", "_tension",
         "Clifford tension energy (0=agree, 2=contradict)"),
        ("Holonomy Score", "_holonomy",
         "Path consistency defect (0=consistent)"),
        ("Overall Confidence", "_overall_confidence",
         "Weighted combination of all signals"),
    ]

    cards = ""
    for label, col, tip in metrics:
        if col not in df.columns:
            continue
        val = df[col].mean()
        if col == "_tension":
            color = _geo_color(1 - val / 2)
        elif col == "_holonomy":
            color = _geo_color(max(0, 1 - val))
        else:
            color = _geo_color(val)
        cards += f"""\
<div class="card">
  <div class="card-label tt" data-tip="{_e(tip)}">{_e(label)}</div>
  <div class="card-value {color}">{val:.4f}</div>
</div>
"""

    # Per-category geometric bars
    bars = ""
    if "_overall_confidence" in df.columns:
        cats = sorted(df["category"].unique())
        for cat in cats:
            sub = df[df["category"] == cat]
            geo = sub["_overall_confidence"].mean()
            color = _geo_color(geo)
            bars += f"""\
<div class="bar-row">
  <div class="bar-label">{_e(cat)}</div>
  <div class="bar-track">
    <div class="bar-fill {color}" style="width:{geo*100:.1f}%"></div>
    <div class="bar-value">{geo:.3f}</div>
  </div>
</div>
"""

    return f"""\
<div class="section">
  <h2>Geometric Scoring</h2>
  <div class="cards" style="margin-bottom:1.2rem">{cards}</div>
  <h3>Overall Confidence by Category</h3>
  {bars}
</div>
"""


def _build_factor_analysis(result: DOEBenchmarkResult, analyzer) -> str:
    if analyzer is None:
        return ""

    try:
        logistic_df = analyzer.run_logistic()
    except Exception:
        return ""

    if logistic_df is None or logistic_df.empty:
        return ""

    rows = ""
    for _, row in logistic_df.iterrows():
        factor = row.get("factor", "")
        dev = row.get("deviance", 0)
        pval = row.get("p_value", 1)
        pr2 = row.get("pseudo_r2", 0)
        sig = row.get("significant", False)
        n_levels = row.get("n_levels", 0)

        sig_cls = "sig" if sig else "nosig"
        sig_txt = "***" if pval < 0.001 else ("**" if pval < 0.01 else
                  ("*" if pval < 0.05 else ""))

        rows += f"""\
<tr>
  <td>{_e(factor)}</td>
  <td class="right">{dev:.2f}</td>
  <td class="right"><span class="{sig_cls}">{pval:.4f} {sig_txt}</span></td>
  <td class="right">{pr2:.4f}</td>
  <td class="center">{int(n_levels)}</td>
  <td class="center"><span class="{sig_cls}">{'YES' if sig else 'no'}</span></td>
</tr>
"""

    # Factor level heatmap
    heatmap = _build_factor_heatmap(result)

    # Failure analysis
    failure_html = _build_failure_analysis(analyzer)

    return f"""\
<div class="section">
  <h2>DOE Factor Analysis</h2>
  <h3>Analysis of Deviance (Logistic Regression)</h3>
  <table>
    <thead>
      <tr>
        <th>Factor</th>
        <th class="right">Deviance (G2)</th>
        <th class="right">p-value</th>
        <th class="right">Pseudo R2</th>
        <th class="center">Levels</th>
        <th class="center">Significant</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>
{heatmap}
{failure_html}
"""


def _build_factor_heatmap(result: DOEBenchmarkResult) -> str:
    try:
        df = result.to_dataframe()
    except Exception:
        return ""

    if df is None or df.empty:
        return ""

    factors = [f for f in result.factors if f in df.columns]
    if not factors:
        return ""

    rows = ""
    for factor in factors:
        levels = sorted(df[factor].unique())
        for level in levels:
            sub = df[df[factor] == level]
            rate = sub["correct"].mean()
            total = len(sub)
            correct = int(sub["correct"].sum())
            rows += f"""\
<tr>
  <td>{_e(factor)}</td>
  <td>{_e(str(level))}</td>
  <td class="right">{correct}/{total}</td>
  <td class="right"><span class="heat" style="{_heat_color(rate)}">{_pct(rate)}</span></td>
</tr>
"""

    return f"""\
<div class="section">
  <h2>Factor Level Accuracy</h2>
  <table>
    <thead>
      <tr><th>Factor</th><th>Level</th><th class="right">Score</th><th class="right">Rate</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""


def _build_failure_analysis(analyzer) -> str:
    try:
        failure_df = analyzer.failure_analysis()
    except Exception:
        return ""

    if failure_df is None or failure_df.empty:
        return ""

    # Top 15 failure modes sorted by failure rate
    top = failure_df.nlargest(15, "failure_rate")

    rows = ""
    for _, row in top.iterrows():
        factor = row.get("factor", "")
        level = row.get("level", "")
        n_total = int(row.get("n_total", 0))
        n_failed = int(row.get("n_failed", 0))
        rate = row.get("failure_rate", 0)
        ci_lo = row.get("ci_lo", 0)
        ci_hi = row.get("ci_hi", 1)

        bar_width = rate * 100
        ci_lo_pct = ci_lo * 100
        ci_hi_pct = ci_hi * 100

        rows += f"""\
<tr>
  <td>{_e(factor)}</td>
  <td>{_e(str(level))}</td>
  <td class="right">{n_failed}/{n_total}</td>
  <td class="right"><span class="heat" style="{_heat_color(1-rate)}">{_pct(rate)}</span></td>
  <td class="right" style="color:var(--text-dim)">[{_pct(ci_lo)}, {_pct(ci_hi)}]</td>
</tr>
"""

    return f"""\
<div class="section">
  <h2>Top Failure Modes</h2>
  <table>
    <thead>
      <tr>
        <th>Factor</th><th>Level</th>
        <th class="right">Failed/Total</th>
        <th class="right">Failure Rate</th>
        <th class="right">95% CI</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""


def _build_detail_table(result: DOEBenchmarkResult) -> str:
    if not result.questions:
        return ""

    # Build filter buttons
    cats = sorted(set(q.category for q in result.questions))
    cat_btns = '<button class="filter-btn active" onclick="filterTable(\'all\')">All</button>\n'
    cat_btns += '<button class="filter-btn" onclick="filterTable(\'fail\')">Failures Only</button>\n'
    for cat in cats:
        cat_btns += f'<button class="filter-btn" onclick="filterTable(\'{_e(cat)}\')">{_e(cat)}</button>\n'

    # Factor columns
    factors = result.factors
    factor_ths = "".join(f'<th>{_e(f)}</th>' for f in factors)

    has_geo = any(
        "_overall_confidence" in q.factor_assignment for q in result.questions
    )
    geo_th = '<th class="right">Geo</th>' if has_geo else ""

    rows = ""
    for q in result.questions:
        cls = "pass" if q.correct else "fail"
        icon = "PASS" if q.correct else "FAIL"

        factor_tds = ""
        for f in factors:
            val = q.factor_assignment.get(f, "")
            factor_tds += f"<td>{_e(str(val))}</td>"

        geo_td = ""
        if has_geo:
            gc = q.factor_assignment.get("_overall_confidence", "")
            if gc:
                gc_f = float(gc)
                geo_td = f'<td class="right"><span class="heat" style="{_heat_color(gc_f)}">{gc_f:.3f}</span></td>'
            else:
                geo_td = '<td class="right"></td>'

        gt_short = str(q.ground_truth)[:40]
        ans_short = str(q.answer)[:40]

        rows += f"""\
<tr data-category="{_e(q.category)}" data-correct="{'1' if q.correct else '0'}">
  <td class="{cls}">{icon}</td>
  <td>{_e(q.doe_qid)}</td>
  <td>{_e(q.category)}</td>
  {factor_tds}
  <td title="{_e(str(q.ground_truth))}">{_e(gt_short)}</td>
  <td title="{_e(str(q.answer))}">{_e(ans_short)}</td>
  {geo_td}
  <td title="{_e(q.detail)}">{_e(q.detail[:50])}</td>
</tr>
"""

    return f"""\
<div class="section">
  <h2>Question Detail</h2>
  <div class="detail-controls">{cat_btns}</div>
  <div class="detail-table-wrap">
    <table>
      <thead>
        <tr>
          <th></th>
          <th>ID</th>
          <th>Category</th>
          {factor_ths}
          <th>Ground Truth</th>
          <th>Answer</th>
          {geo_th}
          <th>Detail</th>
        </tr>
      </thead>
      <tbody id="detail-body">{rows}</tbody>
    </table>
  </div>
</div>
"""


def _build_tracing_section(tracing) -> str:
    """Build a summary of the DOE tracing GMS audit log."""
    if tracing is None:
        return ""

    try:
        stats = tracing.stats()
    except Exception:
        return ""

    entities = stats.get("entities", 0)
    triples = stats.get("triples", 0)
    relations = stats.get("relations", 0)
    enm = stats.get("enm_entries", 0)

    # Query runs from the tracing store
    run_rows = ""
    run_count = 0
    if tracing.store and tracing.store.doc_graph:
        graph = tracing.store.doc_graph
        # Find all run entities
        runs = []
        for h, r, t in graph.triples:
            if r == "is_type" and t == "doe_run":
                runs.append(h)

        for run_id in runs:
            run_count += 1
            run_triples = list(tracing.store.query_triples(head=run_id))
            run_info = {}
            for h, r, t in run_triples:
                run_info[r] = t

            model = run_info.get("target_model", "-")
            ctx = run_info.get("context_mode", "-")
            accuracy = run_info.get("accuracy", "-")
            total_q = run_info.get("total_questions", "-")
            total_c = run_info.get("total_correct", "-")
            started = run_info.get("started_at", "-")
            finished = run_info.get("finished_at", "-")

            # Collect factors
            factors_tested = [t for h, r, t in run_triples
                              if r == "tests_factor"]
            factors_str = ", ".join(factors_tested) if factors_tested else "-"

            acc_display = accuracy
            if accuracy != "-":
                try:
                    acc_f = float(accuracy)
                    acc_display = f'<span class="heat" style="{_heat_color(acc_f)}">{_pct(acc_f)}</span>'
                except ValueError:
                    pass

            run_rows += f"""\
<tr>
  <td>{_e(run_id)}</td>
  <td>{_e(model)}</td>
  <td>{acc_display}</td>
  <td class="right">{_e(str(total_c))}/{_e(str(total_q))}</td>
  <td>{_e(ctx)}</td>
  <td title="{_e(factors_str)}">{_e(factors_str[:40])}</td>
  <td>{_e(started)}</td>
</tr>
"""

        # Find analysis entries
        analyses = []
        for h, r, t in graph.triples:
            if r == "is_type" and t == "doe_analysis":
                analyses.append(h)

        analysis_rows = ""
        for aid in analyses:
            a_triples = list(tracing.store.query_triples(head=aid))
            a_info = {}
            sig_factors = []
            for h, r, t in a_triples:
                if r == "significant_factor":
                    sig_factors.append(t)
                elif r.endswith("_p_value"):
                    factor_name = r.replace("factor_", "").replace("_p_value", "")
                    a_info.setdefault(factor_name, {})["p_value"] = t
                elif r.endswith("_deviance"):
                    factor_name = r.replace("factor_", "").replace("_deviance", "")
                    a_info.setdefault(factor_name, {})["deviance"] = t
                elif r == "run":
                    a_info["_run_id"] = t
                elif r == "timestamp":
                    a_info["_timestamp"] = t

            run_ref = a_info.get("_run_id", "-")
            ts = a_info.get("_timestamp", "-")
            sig_str = ", ".join(sig_factors) if sig_factors else "none"
            n_factors = len([k for k in a_info if not k.startswith("_")])

            analysis_rows += f"""\
<tr>
  <td>{_e(aid)}</td>
  <td>{_e(run_ref)}</td>
  <td>{n_factors}</td>
  <td><span class="{'sig' if sig_factors else 'nosig'}">{_e(sig_str)}</span></td>
  <td>{_e(ts)}</td>
</tr>
"""

    run_table = ""
    if run_rows:
        run_table = f"""\
  <h3>Benchmark Runs</h3>
  <table>
    <thead>
      <tr>
        <th>Run ID</th><th>Model</th><th>Accuracy</th>
        <th class="right">Score</th><th>Context</th>
        <th>Factors</th><th>Started</th>
      </tr>
    </thead>
    <tbody>{run_rows}</tbody>
  </table>
"""

    analysis_table = ""
    if analysis_rows:
        analysis_table = f"""\
  <h3 style="margin-top:1.2rem">Statistical Analyses</h3>
  <table>
    <thead>
      <tr>
        <th>Analysis ID</th><th>Run</th><th>Factors</th>
        <th>Significant</th><th>Timestamp</th>
      </tr>
    </thead>
    <tbody>{analysis_rows}</tbody>
  </table>
"""

    # Tracing graph stats as a mini card row
    return f"""\
<div class="section">
  <h2>Tracing Agent Summary</h2>
  <div class="cards" style="margin-bottom:1.2rem">
    <div class="card">
      <div class="card-label">Traced Entities</div>
      <div class="card-value accent">{entities}</div>
    </div>
    <div class="card">
      <div class="card-label">Traced Triples</div>
      <div class="card-value accent">{triples}</div>
    </div>
    <div class="card">
      <div class="card-label">Relations</div>
      <div class="card-value cyan">{relations}</div>
    </div>
    <div class="card">
      <div class="card-label">ENM Entries</div>
      <div class="card-value cyan">{enm}</div>
    </div>
    <div class="card">
      <div class="card-label">Runs Traced</div>
      <div class="card-value green">{run_count}</div>
    </div>
  </div>
  {run_table}
  {analysis_table}
</div>
"""


def _build_comparison_section(results: list[DOEBenchmarkResult]) -> str:
    """Side-by-side comparison when multiple results provided."""
    if len(results) < 2:
        return ""

    # Overall accuracy comparison
    header_cells = "".join(
        f"<th class='right'>{_e(r.model)}</th>" for r in results
    )

    overall_row = ""
    for r in results:
        color = _acc_color(r.accuracy)
        overall_row += f'<td class="right"><span class="heat" style="{_heat_color(r.accuracy)}">{_pct(r.accuracy)}</span></td>'

    # Per-category
    all_cats = set()
    for r in results:
        df = r.to_dataframe()
        if df is not None and not df.empty:
            all_cats.update(df["category"].unique())

    cat_rows = ""
    for cat in sorted(all_cats):
        cat_rows += f"<tr><td>{_e(cat)}</td>"
        for r in results:
            df = r.to_dataframe()
            if df is not None and not df.empty and cat in df["category"].values:
                sub = df[df["category"] == cat]
                rate = sub["correct"].mean()
                cat_rows += f'<td class="right"><span class="heat" style="{_heat_color(rate)}">{_pct(rate)}</span></td>'
            else:
                cat_rows += '<td class="right center">-</td>'
        cat_rows += "</tr>"

    # Geometric comparison
    geo_rows = ""
    geo_metrics = [
        ("Geodesic", "_geo_confidence"),
        ("Tension", "_tension"),
        ("Holonomy", "_holonomy"),
        ("Overall", "_overall_confidence"),
    ]
    for label, col in geo_metrics:
        geo_rows += f"<tr><td>{label}</td>"
        for r in results:
            df = r.to_dataframe()
            if df is not None and not df.empty and col in df.columns:
                val = df[col].mean()
                geo_rows += f'<td class="right">{val:.4f}</td>'
            else:
                geo_rows += '<td class="right center">-</td>'
        geo_rows += "</tr>"

    return f"""\
<div class="section">
  <h2>Model Comparison</h2>
  <h3>Overall Accuracy</h3>
  <table>
    <thead><tr><th>Metric</th>{header_cells}</tr></thead>
    <tbody>
      <tr><td>Accuracy</td>{overall_row}</tr>
    </tbody>
  </table>

  <h3 style="margin-top:1.2rem">Per-Category</h3>
  <table>
    <thead><tr><th>Category</th>{header_cells}</tr></thead>
    <tbody>{cat_rows}</tbody>
  </table>

  <h3 style="margin-top:1.2rem">Geometric Scores</h3>
  <table>
    <thead><tr><th>Metric</th>{header_cells}</tr></thead>
    <tbody>{geo_rows}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    result: DOEBenchmarkResult,
    analyzer: DOEAnalyzer | None = None,
    output_path: str = "doe_report.html",
    extra_results: list[DOEBenchmarkResult] | None = None,
    tracing: Any | None = None,
) -> str:
    """Generate a self-contained HTML dashboard from benchmark results.

    Args:
        result: Primary benchmark result to display.
        analyzer: DOEAnalyzer with logistic regression results.
        output_path: Where to write the HTML file.
        extra_results: Additional results for model comparison.
        tracing: DOETracingGMS instance for audit log summary.

    Returns:
        The output_path for convenience.
    """
    try:
        df = result.to_dataframe()
    except Exception:
        df = None

    # Build all sections
    sections = [
        _build_header(result),
        _build_summary_cards(result, df),
    ]

    # Comparison section (if multiple results)
    all_results = [result] + (extra_results or [])
    if len(all_results) > 1:
        sections.append(_build_comparison_section(all_results))

    sections.extend([
        _build_category_section(df),
        _build_geometric_section(df),
        _build_factor_analysis(result, analyzer),
        _build_tracing_section(tracing),
        _build_detail_table(result),
    ])

    body = "\n".join(sections)

    page = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DOE-GMS Report &mdash; {_e(result.instance_name)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="container">
    {body}
    <div class="footer">
      Generated by DOE-GMS Benchmark &mdash; Geometric Memory System
    </div>
  </div>
  <script>{_JS}</script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)

    print(f"  Report saved to {output_path}")
    return output_path


def generate_enhanced_report(
    result: DOEBenchmarkResult,
    typed_verdicts: dict | None = None,
    gate=None,
    analyzer=None,
    output_path: str = "doe_report_enhanced.html",
    extra_results: list[DOEBenchmarkResult] | None = None,
    tracing: Any | None = None,
) -> str:
    """Generate an enhanced HTML dashboard with GMSH claim-level sections.

    Extends the base report with failure heatmap, claim coverage,
    verifier detail, release gate, and provenance chain views.

    Args:
        result: Primary benchmark result.
        typed_verdicts: Map from doe_qid to TypedVerdict.
        gate: ReleaseGate result (optional).
        analyzer: DOEAnalyzer (optional).
        output_path: Where to write the HTML file.
        extra_results: Additional results for comparison.
        tracing: DOETracingGMS instance.

    Returns:
        The output_path.
    """
    try:
        df = result.to_dataframe()
    except Exception:
        df = None

    sections = [
        _build_header(result),
        _build_summary_cards(result, df),
    ]

    all_results = [result] + (extra_results or [])
    if len(all_results) > 1:
        sections.append(_build_comparison_section(all_results))

    sections.extend([
        _build_category_section(df),
        _build_geometric_section(df),
        _build_factor_analysis(result, analyzer),
    ])

    # Enhanced GMSH sections
    if typed_verdicts:
        from .report_enhanced import build_enhanced_sections
        sections.append(build_enhanced_sections(result, typed_verdicts, gate))

    sections.extend([
        _build_tracing_section(tracing),
        _build_detail_table(result),
    ])

    body = "\n".join(sections)

    page = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GMSH Report &mdash; {_e(result.instance_name)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="container">
    {body}
    <div class="footer">
      Generated by GMSH Benchmark &mdash; Geometric Memory System Test Orchestrator
    </div>
  </div>
  <script>{_JS}</script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(page)

    print(f"  Enhanced report saved to {output_path}")
    return output_path


def generate_comparison_report(
    results: list[DOEBenchmarkResult],
    analyzers: list[DOEAnalyzer | None] | None = None,
    output_path: str = "doe_comparison.html",
    tracing: Any | None = None,
) -> str:
    """Generate a comparison dashboard from multiple benchmark results.

    Args:
        results: List of benchmark results (one per model).
        analyzers: Matching list of DOEAnalyzers (or None entries).
        output_path: Where to write the HTML file.
        tracing: DOETracingGMS instance for audit log summary.

    Returns:
        The output_path.
    """
    if not results:
        raise ValueError("No results to report")

    primary = results[0]
    extra = results[1:] if len(results) > 1 else None
    primary_analyzer = analyzers[0] if analyzers else None

    return generate_report(
        primary,
        analyzer=primary_analyzer,
        output_path=output_path,
        extra_results=extra,
        tracing=tracing,
    )
