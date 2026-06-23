# SPDX-License-Identifier: Apache-2.0
"""Build NB 15 — Oracle Framework (Ch15)."""
from __future__ import annotations
import time
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "15_oracles.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 15: Oracle Framework — Judging Behavior Geometrically

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch15.tex` — Oracles: Judging Agent Behavior Geometrically
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch15_oracles.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §18
**Prereqs:** NB 11 (SUT Adapters), NB 14 (Environment Simulation)

**What you'll do.** Build an `ExactOutcomeOracle` that checks whether a bank model-risk agent's final answer matches ground truth, a `TraceOracle` that checks whether the tool sequence followed the declared expected pattern, then compose them into a weighted `CompositeOracle`. Each oracle returns an `OracleVerdict` with `passed` / `confidence` / `evidence` and the composite aggregates them into a single release-gate decision.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")
from _shared import load_ch04_store, seed_all
from knowlytix.harness.testing import (
    Scenario, SUTTrace, ToolCallRecord,
    ExactOutcomeOracle, TraceOracle, CompositeOracle,
)
seed_all()
store = load_ch04_store()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

An **oracle** is a `Scenario × SUTTrace → OracleVerdict` function. The harness ships three families:

| Oracle | Compares | Use |
|---|---|---|
| `ExactOutcomeOracle` | Final answer vs `scenario.ground_truth` | Did the agent get the right answer? |
| `TraceOracle` | Tool sequence vs declared expected pattern | Did the agent do it the right way? |
| (composite) | Multiple oracles + weights | Release-gate aggregate |

`CompositeOracle(oracles=[(o1, w1), (o2, w2)], threshold=0.5, fail_on_any_critical=True)` weights individual verdicts, and any single oracle marked `critical=True` can short-circuit the composite to fail regardless of total weight.\
"""))

cells.append(md("""\
## 3. Build a Bank Model-Risk Scenario

A scenario is the declaration: what we asked, what the right answer is, and (optionally) what the expected tool pattern looks like.\
"""))

cells.append(code("""\
scenario = Scenario(
    scenario_id="bank_mr_001",
    name="MDL-001 Test AUC lookup",
    description="Verify the agent reads MDL-001 Test AUC and reports correctly.",
    version=1,
    tags=["bank", "model_risk", "exact_retrieval"],
    risk_tier="action_taking",
    question_text="What is the Test AUC of MDL-001?",
    ground_truth="0.8412",
    answer_type="float",
    # TraceOracle reads trace_pattern directly off the Scenario:
    trace_pattern="data_lookup > risk_check",
    expected_tool_sequence=["data_lookup", "risk_check"],
    required_tools=["data_lookup"],
)
print(f"scenario {scenario.scenario_id}: {scenario.name}")
print(f"  ground_truth   : {scenario.ground_truth}")
print(f"  trace_pattern  : {scenario.trace_pattern}")
print(f"  required_tools : {scenario.required_tools}")\
"""))

cells.append(md("""\
## 4. Two Hypothetical SUTTraces

Trace A: agent did the right thing (correct answer, correct tool order). Trace B: agent got a near-right answer but called the wrong tools.\
"""))

cells.append(code("""\
trace_good = SUTTrace(
    raw_output="0.8412",
    tool_calls=[
        ToolCallRecord(tool_name="data_lookup", arguments={"entity": "MDL-001"},
                       result='{"auc": 0.8412}', latency_ms=50.0),
        ToolCallRecord(tool_name="risk_check", arguments={"model_id": "MDL-001"},
                       result='{"risk_level": "High"}', latency_ms=30.0),
    ],
    latency_ms=85.0,
)

trace_bad = SUTTrace(
    raw_output="0.7234",   # wrong AUC (MDL-005's, not MDL-001's)
    tool_calls=[
        ToolCallRecord(tool_name="random_search", arguments={"q": "MDL-001"},
                       result="search_blob", latency_ms=400.0),
    ],
    latency_ms=410.0,
)
print(f"good trace: {trace_good.raw_output!r}  {len(trace_good.tool_calls)} tool calls")
print(f"bad  trace: {trace_bad.raw_output!r}  {len(trace_bad.tool_calls)} tool calls")\
"""))

cells.append(md("""\
## 5. ExactOutcomeOracle — Did the Answer Match?\
"""))

cells.append(code("""\
exact = ExactOutcomeOracle(store=store)
for label, t in (("good", trace_good), ("bad", trace_bad)):
    v = exact.evaluate(scenario, t)
    status = "PASS" if v.passed else "FAIL"
    print(f"  [{status}] {label:4s}  conf={v.confidence:.2f}  detail={v.detail[:100]}")\
"""))

cells.append(md("""\
## 6. TraceOracle — Did the Tool Sequence Match the Expected Pattern?\
"""))

cells.append(code("""\
trace_oracle = TraceOracle(store=store)
for label, t in (("good", trace_good), ("bad", trace_bad)):
    v = trace_oracle.evaluate(scenario, t)
    status = "PASS" if v.passed else "FAIL"
    print(f"  [{status}] {label:4s}  conf={v.confidence:.2f}  detail={v.detail[:100]}")
    print(f"        evidence: {v.evidence}")\
"""))

cells.append(md("""\
## 7. Composite Oracle — Weighted Aggregate\
"""))

cells.append(code("""\
composite = CompositeOracle(
    oracles=[(exact, 0.6), (trace_oracle, 0.4)],
    threshold=0.5,
    fail_on_any_critical=True,
)
for label, t in (("good", trace_good), ("bad", trace_bad)):
    v = composite.evaluate(scenario, t)
    status = "PASS" if v.passed else "FAIL"
    print(f"  [{status}] {label:4s}  composite_conf={v.confidence:.3f}  "
          f"detail={v.detail[:100]}")
    print(f"        raw_scores: {v.raw_scores}")\
"""))

cells.append(md("""\
## 8. Visualization — Per-Oracle Contribution\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

oracles_under_test = [("ExactOutcome", exact, 0.6), ("Trace", trace_oracle, 0.4)]
labels = ["good trace", "bad trace"]
data = []
for traceset in (trace_good, trace_bad):
    row = []
    for _, o, _ in oracles_under_test:
        v = o.evaluate(scenario, traceset)
        row.append(v.confidence)
    data.append(row)
data = np.array(data)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, data[:, 0], w, label=f"ExactOutcome (weight=0.6)", color="#3aa56b")
ax.bar(x + w/2, data[:, 1], w, label=f"Trace (weight=0.4)", color="#4a8cd1")
ax.axhline(0.5, color="gray", linestyle="--", label="composite threshold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel("oracle confidence")
ax.set_title("Per-oracle contribution on two traces")
ax.legend()
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 9. Take-Aways

1. **`Scenario × SUTTrace → OracleVerdict`** is the entire oracle contract. Stateless and pure; same inputs always yield the same verdict.
2. **`ExactOutcome` and `Trace` are orthogonal.** The good trace passes both; the bad trace fails both. A clever-but-lazy agent that gets the right answer through the wrong tools would pass `ExactOutcome` but fail `Trace`.
3. **`CompositeOracle` is the deployment glue.** Weighted aggregation + threshold + critical short-circuit. Configure once per risk tier, run the same composite at every release-gate evaluation.
4. **`evidence` is the audit trail.** Each `OracleVerdict.evidence` carries the raw measurements (tool sequence, scoring path, side-effect diff). Persist it alongside the typed verdict for replay.
5. **Build custom oracles by subclassing `OracleBase`.** The contract is `evaluate(scenario, trace, env_state_before, env_state_after) -> OracleVerdict`. A side-effect oracle reads `env_state_before` and `env_state_after` from `EnvironmentSimulator` (NB 14).\
"""))

cells.append(md("""\
## 10. Next Steps

- **NB 16** — `16_hallucination.ipynb` — a specialized oracle for hallucination detection (Ch16)
- **NB 17** — `17_multi_turn.ipynb` — oracles over multi-turn dialogs (Ch17)
- **NB 19** — `19_release_gates.ipynb` — wire the composite oracle into the release-gate audit (Ch19)

For theory: `gmsh_testing_ext/chapters/ch15.tex`.\
"""))


def build_and_execute():
    nb = new_notebook()
    nb.cells = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (spark-venv)",
                                      "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    print(f"Writing {NB_PATH} ({len(cells)} cells)...")
    NB_PATH.write_text(nbformat.writes(nb))
    print("Executing...")
    t0 = time.time()
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
