# SPDX-License-Identifier: Apache-2.0
"""Build NB 17 — Stateful and Multi-Turn Testing (Ch17)."""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "17_multi_turn.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 17: Stateful and Multi-Turn Testing

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch17.tex` — Stateful and Multi-Turn Testing
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch17_multi_turn.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §20
**Prereqs:** NB 11 (SUT Adapters), NB 14 (Environment Sim), NB 15 (Oracles)

**What you'll do.** Wrap an agent in a `MultiTurnAdapter` and drive it through a 3-turn `InteractionScript` over a bank model-risk workflow: turn 1 looks up MDL-001, turn 2 reads its current status, turn 3 escalates if degraded. Each turn carries expected substrings + tool patterns. Read off per-turn pass/fail and the final teardown assertion.\
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
    MultiTurnAdapter, InteractionScript, TurnSpec,
    EnvironmentSimulator, ToolEndpointSpec, StateTransition,
)
from knowlytix.harness.fixtures import MockHarness
seed_all()
store = load_ch04_store()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

A multi-turn test is a sequence of `TurnSpec`s, each carrying:

- `user_message` — what the user says on this turn
- `expected_contains` / `expected_not_contains` — substring assertions on the agent's reply
- `expected_tool_calls` / `prohibited_tool_calls` — tool-use assertions per turn
- `expected_state_changes` — environment-state assertions when paired with `EnvironmentSimulator`
- `carry_forward` — keys carried from this turn's state into the next

The script as a whole carries `teardown_assertions` evaluated after the last turn.

Why multi-turn matters: a bank agent that gets every question right in isolation may still fail an audit if it loses track of MDL-001 between turns, or escalates without first having seen the risk_check result.\
"""))

cells.append(md("""\
## 3. Build the Script — Three Turns on MDL-001\
"""))

cells.append(code("""\
auc = store.lookup_enm("performance_metrics", "MDL-001/Test AUC")

# Environment with three endpoints
env = EnvironmentSimulator(tool_specs=[
    ToolEndpointSpec(
        tool_name="data_lookup",
        initial_state={"loaded": 0},
        default_response=f'{{"auc": {auc}, "status": "Degraded"}}',
        state_transitions=[
            StateTransition(
                trigger_action="data_lookup",
                trigger_condition="args.entity == 'MDL-001'",
                state_mutation={"loaded": 1, "last": "MDL-001"},
                response_template=f'{{"auc": {auc}, "status": "Degraded"}}',
            ),
        ],
    ),
    ToolEndpointSpec(
        tool_name="risk_check",
        default_response='{"risk_level": "High"}',
    ),
    ToolEndpointSpec(
        tool_name="escalate_to_human",
        default_response='{"ticket_id": "INC-12345"}',
    ),
])
env.reset(seed=42)

script = InteractionScript(
    script_id="bank_mdl001_workflow",
    name="MDL-001 escalation workflow",
    description="3-turn workflow: lookup -> risk check -> escalate.",
    setup_context="You are a model-risk analyst.",
    turns=[
        TurnSpec(
            turn_id="t1",
            user_message="Look up MDL-001 in the model inventory.",
            expected_tool_calls=["data_lookup"],
            expected_contains=["MDL-001"],
        ),
        TurnSpec(
            turn_id="t2",
            user_message="What is its current status?",
            expected_contains=["degraded"],
        ),
        TurnSpec(
            turn_id="t3",
            user_message="Open a remediation ticket via escalate_to_human.",
            expected_tool_calls=["escalate_to_human"],
            expected_contains=["INC", "ticket"],
        ),
    ],
    teardown_assertions=[
        {"check": "state.loaded == 1", "description": "MDL-001 must remain loaded"},
    ],
)
print(f"script: {len(script.turns)} turns")
for t in script.turns:
    print(f"  [{t.turn_id}] user='{t.user_message[:60]}'  "
          f"expects_tools={t.expected_tool_calls}")\
"""))

cells.append(md("""\
## 4. Execute the Script

Use `MockHarness` so the demo is deterministic; in production wrap a real agent in `MultiTurnAdapter` the same way.\
"""))

cells.append(code("""\
harness = MockHarness(recovery_mode="fallback")
adapter = MultiTurnAdapter(harness=harness, store=store)
t0 = time.time()
result = adapter.execute_script(script=script, environment=env)
elapsed = time.time() - t0
print(f"  wall time: {elapsed:.2f}s")
print(f"  result type: {type(result).__name__}")
# Inspect ScriptResult shape
print(f"  fields available:")
for name in dir(result):
    if not name.startswith("_"):
        val = getattr(result, name)
        if not callable(val):
            short = str(val)[:80]
            print(f"    {name:18s}: {short}")\
"""))

cells.append(md("""\
## 5. Per-Turn Breakdown\
"""))

cells.append(code("""\
import pandas as pd
rows = []
for i, tr in enumerate(getattr(result, "turn_results", []), 1):
    rows.append({
        "turn": tr.turn_id if hasattr(tr, 'turn_id') else f"t{i}",
        "passed": getattr(tr, "passed", None),
        "n_violations": len(getattr(tr, "violations", []) or []),
        "latency_ms": round(getattr(tr, "latency_ms", 0.0) or 0.0, 1),
        "raw_output": str(getattr(tr, "raw_output", ""))[:60],
    })
if rows:
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print("(no turn_results — adapter may report differently; inspect cell 4 output above)")\
"""))

cells.append(md("""\
## 6. Final State + Teardown\
"""))

cells.append(code("""\
final_state = env.get_state()
print(f"final environment state: {final_state}")
print()
print("teardown assertions:")
for a in script.teardown_assertions:
    print(f"  - check : {a.get('check')}")
    print(f"    description: {a.get('description')}")\
"""))

cells.append(md("""\
## 7. Take-Aways

1. **`InteractionScript` is the test artifact.** YAML-loadable, version-controlled, the canonical declaration of "what this stateful test exercises."
2. **State carries across turns.** `EnvironmentSimulator` retains its state between `handle_tool_call`s; `MultiTurnAdapter` retains agent memory between turns. Together they make compound workflow bugs visible.
3. **Per-turn assertions are the regression hooks.** When an agent regresses on turn 3 in a 5-turn workflow, the per-turn pass/fail tells you exactly where the regression starts.
4. **`teardown_assertions` are the invariants.** End-of-script global checks (e.g., "the agent never lost the loaded model" or "no ticket was opened on a non-degraded model") catch sequence-level violations that no single turn assertion can.
5. **Pair with `ConsensusOracle` for multi-agent.** NB 18 shows multi-agent consensus, which is multi-turn at a higher level of granularity (one agent's full answer is now one turn from the perspective of the consensus oracle).\
"""))

cells.append(md("""\
## 8. Next Steps

- **NB 18** — `18_multi_agent.ipynb` — multi-agent consensus + cross-document conflict detection (Ch18)
- **NB 14** — `14_environment_sim.ipynb` — deeper environment-simulator features (Ch14)

For theory: `gmsh_testing_ext/chapters/ch17.tex`.\
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
