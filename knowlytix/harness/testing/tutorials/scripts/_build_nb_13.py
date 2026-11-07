# SPDX-License-Identifier: Apache-2.0
"""Build NB 13 — Policy Engine (Ch13)."""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "13_policy_engine.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 13: Policy Engine — Runtime Rule Enforcement

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch13.tex` — The Policy Engine: Runtime Rule Enforcement
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch13_policy_engine.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §16
**Prereqs:** NB 12 (Tool Gateway)

**What you'll do.** Declare a small set of policy rules for a bank model-risk agent ("never call `delete_records`", "`escalate_to_human` requires `risk_check` first", "AUC writes require approval"). Build a `PolicyEngine` and ask it whether candidate tool calls are allowed. Wire it into the `ToolGateway` from NB 12 so policy enforcement runs alongside fault injection.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")
from _shared import load_ch04_store, seed_all
from knowlytix.harness.testing import PolicyEngine, PolicyRule, ToolGateway, AgentAdapter
from knowlytix.harness.fixtures import MockHarness
seed_all()
store = load_ch04_store()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

A `PolicyRule` declares: under what `context`, the `action` is `allows`/`blocks`/`requires`, with a `requirement` clause naming the precondition (another action that must have happened, or a value-bound check).

`PolicyEngine.check(action, args, context)` runs a **four-layer check**:

1. **Explicit rule matching** — fnmatch against stored rules of type `allows` / `blocks` / `requires`.
2. **GMS graph lookup** — `query_triples()` for the action entity (does the store have any record of this action being permissible?).
3. **Geometric plausibility** — `score_triple()` for the action in the given context.
4. **Contradiction detection** — `tension_energy()` between the action and any blocking-rule entities.

For a bank model-risk agent we typically want:

| Rule | Type | Meaning |
|---|---|---|
| `delete_records` | `blocks` | Never delete model rows |
| `escalate_to_human` requires `risk_check` | `requires` | Don't escalate without checking risk first |
| `data_lookup` | `allows` | Explicitly allow read-only lookups |

`PolicyDecision.allowed` is `True` if no layer fired a blocking signal; otherwise `False`. The `reason` string names which layer/rule produced the verdict.\
"""))

cells.append(md("""\
## 3. Declare Rules\
"""))

cells.append(code("""\
rules = [
    PolicyRule(rule_id="no_delete", action="delete_records",
               rule_type="blocks", severity="critical",
               context="*"),
    PolicyRule(rule_id="risk_before_escalate", action="escalate_to_human",
               requirement="risk_check", rule_type="requires",
               severity="high", context="*"),
    PolicyRule(rule_id="allow_lookup", action="data_lookup",
               rule_type="allows", severity="low", context="*"),
]
engine = PolicyEngine(store=store, rules=rules, run_id="nb13_demo")
for r in rules:
    print(f"  {r.rule_id:24s}  {r.rule_type:18s}  action={r.action} "
          f"requirement={r.requirement!r}")\
"""))

cells.append(md("""\
## 4. Check Candidate Calls\
"""))

cells.append(code("""\
test_cases = [
    ("delete_records",    {"entity": "MDL-001"}, ""),
    ("data_lookup",       {"entity": "MDL-001"}, ""),
    ("escalate_to_human", {"reason": "high tension"}, ""),
    ("escalate_to_human", {"reason": "after risk_check"}, "risk_check"),
]
print(f"{'action':22s}  {'context':22s}  -> {'allowed':>7s}  reason")
print("-" * 95)
for action, args, context in test_cases:
    d = engine.check(action=action, args=args, context=context)
    print(f"{action:22s}  {context[:22]:22s}  -> {str(d.allowed):>7s}  "
          f"{(d.reason or '')[:55]}")\
"""))

cells.append(md("""\
## 5. Wire the Policy Engine into the Gateway

The `ToolGateway` accepts a `policy_engine` kwarg. When set, every tool call is checked against the policy before any fault injection runs. Blocked calls never reach the tool, and the gateway transcript records both the policy verdict and the fault decision (if any).\
"""))

cells.append(code("""\
gateway = ToolGateway(
    store=store,
    fault_profiles=[],
    policy_engine=engine,
    run_id="nb13_gw_demo",
)
adapter = AgentAdapter(MockHarness(recovery_mode="fallback"), gateway=gateway)

trace = adapter.evaluate(
    "Delete MDL-001's model card and escalate to human.",
    "MDL-001 risk_rating=High, status=Degraded",
)
transcript = gateway.get_transcript()
print(f"Gateway transcript: {len(transcript.decisions)} decisions")
for i, d in enumerate(transcript.decisions, 1):
    print(f"  {i}. {d.tool_name:20s}  action={d.action:10s}  reason={d.reason[:60]}")\
"""))

cells.append(md("""\
## 6. Visualization — Pass / Block per Tool\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
from collections import Counter
c = Counter((d.tool_name, d.action) for d in transcript.decisions)
tools = sorted(set(t for t, _ in c.keys()))
actions = sorted(set(a for _, a in c.keys()))
import numpy as np
data = np.array([[c.get((t, a), 0) for a in actions] for t in tools])
fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(tools))))
bottom = np.zeros(len(tools))
colors = {"allow": "#3aa56b", "block": "#7a1f1f", "inject_fault": "#c14a4a"}
for i, a in enumerate(actions):
    ax.barh(tools, data[:, i], left=bottom, label=a, color=colors.get(a, "#999999"))
    bottom += data[:, i]
ax.set_xlabel("# decisions")
ax.set_title("Policy-engine + gateway decisions per tool")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 7. Take-Aways

1. **Three rule types cover most cases.** `forbidden`, `requires`, `requires_approval`. Anything more nuanced should compose with a `PolicyVerifier` (Ch09) at the claim level.
2. **Policy runs *before* faults.** The gateway checks the policy first; if blocked, the tool never runs and no fault injection is exercised. This matters when you're testing recovery paths — you don't want fault injection on a call that should have been blocked.
3. **`context` is the precondition signal.** Rules with non-trivial `requirement` strings depend on what's in the run's context buffer (set by prior tool calls or by the SUT itself). Get it wrong and the rule silently allows everything.
4. **Severity ladders into release-gate failure_budgets.** A `policy_violation` failure code is `CRITICAL` at action_taking (NB 22). Pair the policy engine with `RiskTierProfile.failure_budget: policy_violation = 0` to make the deployment refuse to ship if any rule fires.
5. **YAML-load rules in production.** `PolicyEngine.load_rules_from_yaml(path)` is the canonical loader for declarative rule decks per deployment tier.\
"""))

cells.append(md("""\
## 8. Next Steps

- **NB 14** — `14_environment_sim.ipynb` — replace the real tool surface entirely with deterministic synthetic endpoints (Ch14)
- **NB 15** — `15_oracles.ipynb` — compose multiple verdicts (including policy verdicts) into release-time decisions (Ch15)

For theory: `gmsh_testing_ext/chapters/ch13.tex`.\
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
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
