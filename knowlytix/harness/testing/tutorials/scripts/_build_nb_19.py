# SPDX-License-Identifier: Apache-2.0
"""Build NB 19 — Campaigns, Release Gates, and Provenance — from cell
definitions, then execute.

Capstone for Part V (production lifecycle). Loads typed verdicts from a small
benchmark run on the bank model-risk corpus and applies risk-tiered release
gates (advisory / recommendation / action_taking) to determine which
deployment tiers the SUT can ship to. Demonstrates the audit-report and
executive-summary surface.

Cost: ~$0.30 / ~4 min wall clock.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_19.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "19_release_gates.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 19: Risk-Tiered Release Gates and Audit Reports

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch19.tex` — Campaigns, Release Gates, and Provenance
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch19_campaigns_release_gates.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §22
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; ch04 fixture; NB 00 (Quick Start), NB 10 (Verdicts & Severity)

**What you'll do.** Take a bank model-risk SUT, run a small typed-verdict benchmark, and apply the three canonical risk-tier profiles (`advisory`, `recommendation`, `action_taking`) to determine which deployment tiers it can ship to. The same agent may pass an advisory gate (suggestions to a human reviewer) while failing an action-taking gate (autonomous credit decisions). Then emit an audit-ready executive summary and a failure report.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import contextlib, io, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")

from _shared import (
    load_ch04_store, make_anthropic_client,
    DEFAULT_TUTORIAL_MODEL, seed_all,
)
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig, LLMEvaluator,
    AuditReporter, RiskTierProfile, TieredReleaseGate,
)
seed_all()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

A **risk tier** is a deployment context: the same model behaves very differently when it is suggesting options to a human analyst (`advisory`) versus autonomously approving credit applications (`action_taking`). Each tier carries its own thresholds on accuracy, geometric confidence, tension, holonomy, and failure budgets per failure code.

A **release gate** runs the SUT's typed verdicts through one tier's thresholds and returns a pass/fail with the blocking reasons. The same set of verdicts can pass the `advisory` gate and fail the `action_taking` gate simultaneously — that's the diagnostic signal.

The three canonical tiers (from `RiskTierProfile.advisory()` / `.recommendation()` / `.action_taking()`):

| Tier | Accuracy | Geo conf | Max tension | Hallucination budget | Required evaluators |
|---|---|---|---|---|---|
| `advisory` | 0.70 | 0.40 | 2.0 | unlimited | none |
| `recommendation` | 0.80 | 0.55 | 1.5 | 0 | none |
| `action_taking` | 0.95 | 0.70 | 1.0 | 0 | Process + Recovery |

For a bank model-risk agent, `action_taking` is the standard for credit decisions and AML flagging; `recommendation` is appropriate for analyst-facing suggestions; `advisory` for low-stakes pre-screening.\
"""))

cells.append(md("""\
## 3. Run a Typed-Verdict Benchmark

We use the same 78-question quick sweep as NB 00, but here we keep the run inline so the typed verdicts are fresh and the audit packet captures provenance from this notebook session. ~4 min wall time, ~$0.30 cost.\
"""))

cells.append(code("""\
client = make_anthropic_client()
config = DOEHarnessConfig(
    markdown_path="gmsh_testing_ext/lab_stores/ch04_ground_truth/documents/combined.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen",
    n_runs=2,
    max_per_category=3,
    cap_enabled=True,
    enable_typed_verdicts=True,
    enable_tracing=False,
    target_type="llm",
    target_model=DEFAULT_TUTORIAL_MODEL,
)
bench = DOEGMSBenchmark(config)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    bench.ingest()
    bench.generate_questions()

evaluator = LLMEvaluator(client=client, model=DEFAULT_TUTORIAL_MODEL)
t0 = time.time()
with contextlib.redirect_stdout(buf):
    result = bench.run(evaluator=evaluator)
elapsed = time.time() - t0
verdicts = bench.typed_verdicts or {}

acc = result.total_correct / result.total_questions if result.total_questions else 0.0
print(f"  wall time      : {elapsed:.1f}s")
print(f"  total questions: {result.total_questions}")
print(f"  correct        : {result.total_correct}")
print(f"  accuracy       : {acc:.1%}")
print(f"  typed verdicts : {len(verdicts)}")\
"""))

cells.append(md("""\
## 4. Define the Three Risk Tiers

`RiskTierProfile.advisory()`, `.recommendation()`, `.action_taking()` are the harness-shipped presets. You can also build a custom tier by instantiating `RiskTierProfile(tier_name=..., accuracy_threshold=..., ...)` or by loading a YAML via `config.risk_tier_profiles_path`.\
"""))

cells.append(code("""\
import pandas as pd

tiers = {
    "advisory": RiskTierProfile.advisory(),
    "recommendation": RiskTierProfile.recommendation(),
    "action_taking": RiskTierProfile.action_taking(),
}

rows = []
for name, t in tiers.items():
    rows.append({
        "tier": name,
        "accuracy_min": t.accuracy_threshold,
        "geo_conf_min": t.geo_confidence_threshold,
        "max_tension": t.max_tension,
        "max_holonomy": t.max_holonomy,
        "hallucination_budget": t.failure_budget.get("hallucination_detected", "∞"),
        "policy_violation_budget": t.failure_budget.get("policy_violation", "∞"),
        "required_evaluators": ", ".join(t.required_evaluators) or "—",
    })
pd.DataFrame(rows).set_index("tier")\
"""))

cells.append(md("""\
## 5. Apply Each Tier as a Release Gate

`AuditReporter.generate_tiered_release_gate(tier)` returns a `TieredReleaseGate` carrying the pass/fail verdict, the per-check breakdown (accuracy, confidence, tension, holonomy, failure-budget, regression), and the blocking reasons. Same SUT, three gates, three independent verdicts.\
"""))

cells.append(code("""\
reporter = AuditReporter(result=result, typed_verdicts=verdicts)

# generate_tiered_release_gate accepts a list of tier profiles and returns
# a list of TieredReleaseGate in the same order.
gate_list = reporter.generate_tiered_release_gate(list(tiers.values()))
gates: dict[str, TieredReleaseGate] = dict(zip(tiers.keys(), gate_list))

print(f"{'tier':16s} {'verdict':10s}  blocking reasons (first 2)")
print("-" * 76)
for name, gate in gates.items():
    verdict = "PASS" if gate.passed else "FAIL"
    reasons = "; ".join(gate.blocking_reasons[:2]) if gate.blocking_reasons else "—"
    print(f"{name:16s} {verdict:10s}  {reasons[:60]}")\
"""))

cells.append(md("""\
## 6. Inspect the Per-Check Breakdown

Each gate carries a `checks` dictionary with one entry per dimension the tier evaluates. For a failing gate, this is the audit trail — it shows exactly which threshold was missed and by how much.\
"""))

cells.append(code("""\
print(f"Per-check breakdown for the most restrictive tier ('action_taking'):")
print()
action_gate = gates["action_taking"]
for check_name, check_data in action_gate.checks.items():
    status = "PASS" if check_data.get("passed", False) else "FAIL"
    print(f"  [{status}] {check_name}")
    for k, v in check_data.items():
        if k == "passed":
            continue
        if isinstance(v, float):
            print(f"           {k:20s}: {v:.4f}")
        else:
            print(f"           {k:20s}: {v}")
    print()\
"""))

cells.append(md("""\
## 7. Executive Summary (Tiered)

`generate_tiered_executive_summary()` produces a one-page audit-ready summary across all three tiers — what an MRM committee or release manager actually needs to see. Format is markdown so it pastes directly into a release ticket.\
"""))

cells.append(code("""\
summary = reporter.generate_tiered_executive_summary(gate_list)
print(summary)\
"""))

cells.append(md("""\
## 8. Failure Report

When a gate fails, the failure report enumerates every blocking issue with the typed-verdict context that triggered it. This is the bridge from "the SUT failed `action_taking`" to "here are the specific questions that drove the failure."\
"""))

cells.append(code("""\
# generate_failure_report() returns a dict; iterate the categories
# that contributed blocking failures.
failure_report = reporter.generate_failure_report()
print(f"Failure report categories: {list(failure_report.keys())}")
print()
for k, v in failure_report.items():
    print(f"=== {k} ===")
    if isinstance(v, list):
        for item in v[:3]:
            print(f"  {str(item)[:140]}")
        if len(v) > 3:
            print(f"  ... ({len(v) - 3} more)")
    elif isinstance(v, dict):
        for sub_k, sub_v in list(v.items())[:5]:
            print(f"  {sub_k}: {str(sub_v)[:120]}")
    else:
        print(f"  {str(v)[:200]}")
    print()\
"""))

cells.append(md("""\
## 9. Visualization — Which Tiers Can This SUT Ship To?

A single chart that reads at a glance: the SUT passes every check up to a threshold and is blocked on tighter ones. Useful for committee slides and dashboards.\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt

names = list(gates.keys())
passed = [int(g.passed) for g in gates.values()]
n_blocked = [len(g.blocking_reasons) for g in gates.values()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

colors = ["#3aa56b" if p else "#c14a4a" for p in passed]
ax1.barh(names, [1] * len(names), color=colors)
for i, (name, p) in enumerate(zip(names, passed)):
    label = "PASS" if p else "FAIL"
    ax1.text(0.5, i, label, ha="center", va="center", color="white", fontweight="bold")
ax1.set_xlim(0, 1)
ax1.set_xticks([])
ax1.set_title(f"Release gate verdict per tier\\nSUT = {DEFAULT_TUTORIAL_MODEL}")

ax2.barh(names, n_blocked, color="steelblue")
ax2.set_xlabel("blocking reasons (count)")
ax2.set_title("Number of failing checks per tier")
for i, n in enumerate(n_blocked):
    ax2.text(n + 0.05, i, str(n), va="center")

plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 10. Take-Aways

1. **One SUT, three gates.** The same set of typed verdicts is evaluated independently against `advisory`, `recommendation`, and `action_taking` tiers — a SUT can ship to the lower tiers while failing the higher ones.
2. **`AuditReporter` is the front door.** Three method calls (`generate_tiered_release_gate`, `generate_tiered_executive_summary`, `generate_failure_report`) cover everything a release manager or MRM committee needs to see.
3. **Failure budgets are absolute.** `action_taking` and `recommendation` set `hallucination_detected: 0` — a single hallucination kills the gate regardless of overall accuracy. This is the right policy for autonomous-decision tiers.
4. **Geometric confidence is the second axis.** Accuracy alone is not enough; `geo_confidence_threshold` enforces that the SUT's correct answers are *also* the ones the geometric judge is most confident about, not lucky exact-string hits.
5. **The provenance chain captures the audit identity.** Even though we did not enable `enable_provenance=True` in this notebook, NB 19 shows the gate verdict; pair it with `TruthStore` (Ch19) and `enable_provenance=True` to get the full HMAC-signed audit trail for a bank deployment review.\
"""))

cells.append(md("""\
## 11. Next Steps

- **NB 20** — `20_calibration.ipynb` — load `CalibrationSnapshot` so `geo_confidence_threshold` enforces a calibrated cut, not a default (App G)
- **NB 22** — `22_failure_codes.ipynb` — the failure-code taxonomy and how the `failure_budget` dict consumes it (App C)
- **NB 23** — `23_operational_features.ipynb` — campaign management, artifact versioning, performance profiling, cost tracking (Ch19 supporting machinery)

For theory:
- `gmsh_testing_ext/chapters/ch19.tex` — derivation of risk-tier semantics, provenance chain, multi-signer attestation
- `gmsh_testing_ext/notebooks/ch19_campaigns_release_gates.ipynb` — textbook lab walkthrough with custom-tier exercises\
"""))


def build_and_execute() -> None:
    nb = new_notebook()
    nb.cells = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (spark-venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    print(f"Writing {NB_PATH} ({len(cells)} cells)...")
    NB_PATH.write_text(nbformat.writes(nb))

    print("Executing notebook end-to-end...")
    t0 = time.time()
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  execution wall time: {time.time() - t0:.1f}s")

    print(f"Writing executed notebook back to {NB_PATH}...")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
