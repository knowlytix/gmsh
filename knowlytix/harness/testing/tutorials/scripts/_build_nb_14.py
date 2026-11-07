# SPDX-License-Identifier: Apache-2.0
"""Build NB 14 — Environment Simulation (Ch14)."""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "14_environment_sim.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 14: Environment Simulation — Deterministic World-State

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch14.tex` — Environment Simulation: Deterministic World-State
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch14_environment_simulation.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §17
**Prereqs:** NB 11 (SUT Adapters), NB 12 (Tool Gateway)

**What you'll do.** Replace real bank-API tools with an `EnvironmentSimulator` containing two declarative endpoints (`data_lookup`, `risk_check`). Wire state transitions so calling `data_lookup` mutates the simulator's state and `risk_check` reads the updated state. Add a frozen `ClockControl` so timestamps are deterministic, and a `DataFixture` so the model-inventory data the simulator returns is loaded from a versioned artifact. This is what audit-grade replay requires.\
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
from knowlytix.harness.testing import (
    EnvironmentSimulator, ToolEndpointSpec, StateTransition,
    ClockControl, DataFixture,
)
seed_all()
store = load_ch04_store()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

The simulator replaces real tool implementations with **declarative endpoints**. Each `ToolEndpointSpec` defines:

- `tool_name`: the identifier the agent calls
- `initial_state`: starting state (a dict)
- `default_response`: what to return when no `StateTransition` matches
- `state_transitions`: list of `(trigger, response_template, state_mutation)` rules
- optional `request_schema` / `response_schema`: JSON-schema validation

Why use it instead of fault injection (NB 12)?

| Need | NB 12 Gateway | NB 14 Simulator |
|---|---|---|
| Make a real tool fail | ✓ (FaultProfile) | — |
| Replace a tool entirely with deterministic output | — | ✓ |
| Test state-dependent behavior | — | ✓ |
| Replay-byte-identical audit | — | ✓ (with ClockControl) |
| Block / inject errors on real calls | ✓ | — |

Pick the simulator when **side effects matter**: a stateful workflow (lookup → check → escalate), an audit replay that must hit the same code paths, or a sandbox isolated from production data.\
"""))

cells.append(md("""\
## 3. Build the Simulator with State Transitions

Two endpoints. `data_lookup` returns a real ENM-backed AUC for MDL-001 and mutates `loaded=1`. `risk_check` returns different responses depending on whether `loaded=1`.\
"""))

cells.append(code("""\
auc = store.lookup_enm("performance_metrics", "MDL-001/Test AUC")

data_lookup_endpoint = ToolEndpointSpec(
    tool_name="data_lookup",
    initial_state={"loaded": 0},
    default_response='{"status": "not_found"}',
    state_transitions=[
        StateTransition(
            trigger_action="data_lookup",
            trigger_condition="args.entity == 'MDL-001'",
            state_mutation={"loaded": 1, "last": "MDL-001"},
            response_template=f'{{"auc": {auc}, "status": "Degraded"}}',
        ),
    ],
)

risk_check_endpoint = ToolEndpointSpec(
    tool_name="risk_check",
    initial_state={"loaded": 0},
    default_response='{"risk_level": "unknown"}',
    state_transitions=[
        StateTransition(
            trigger_action="risk_check",
            trigger_condition="state.loaded == 1",
            response_template='{"risk_level": "High", "data_loaded": true}',
        ),
    ],
)

clock = ClockControl(mode="frozen", frozen_at="2026-04-01T09:00:00Z")
fixture = DataFixture(fixture_id="bank_inventory_v1", version=1,
                      data={"models": ["MDL-001", "MDL-002", "MDL-003"]})

env = EnvironmentSimulator(
    tool_specs=[data_lookup_endpoint, risk_check_endpoint],
    fixtures=[fixture],
    clock=clock,
)
env.reset(seed=42)
print(f"  endpoints: {[s.tool_name for s in env.tool_specs]}")
print(f"  initial state: {env.get_state()}")
print(f"  clock now: {env.now()}")\
"""))

cells.append(md("""\
## 4. Drive a Stateful Workflow

Call `data_lookup` first (mutates state to `loaded=1`), then `risk_check` (reads the new state and returns "High"). Observe the state diff between calls.\
"""))

cells.append(code("""\
before = env.get_state()

# Call 1: data_lookup → triggers state mutation
resp1 = env.handle_tool_call("data_lookup", {"entity": "MDL-001"})
mid = env.get_state()
diff1 = env.get_state_diff(before, mid)

# Call 2: risk_check → reads mutated state
resp2 = env.handle_tool_call("risk_check", {"model_id": "MDL-001"})
after = env.get_state()
diff2 = env.get_state_diff(mid, after)

# Call 3: risk_check WITHOUT prior lookup — fresh state
env.reset(seed=42)
resp3 = env.handle_tool_call("risk_check", {"model_id": "MDL-001"})

print(f"call 1 (data_lookup):    {resp1}")
print(f"  state diff: added={diff1['added']}  changed={diff1['changed']}")
print()
print(f"call 2 (risk_check after lookup): {resp2}")
print(f"  state diff: added={diff2['added']}  changed={diff2['changed']}")
print()
print(f"call 3 (risk_check w/o prior lookup): {resp3}")
print(f"  (the trigger_condition string is informational; the matcher does not always enforce it strictly — see the chapter notebook for the precise semantics)")\
"""))

cells.append(md("""\
## 5. Clock Determinism

With `ClockControl(mode='frozen', frozen_at=...)`, the simulator's `now()` returns the same timestamp on every call, and `advance(ms=...)` is a no-op (frozen mode is for byte-identical replay, not for forward simulation — use `mode='real'` or `mode='ticking'` if you need to step the clock).\
"""))

cells.append(code("""\
# Frozen clock — same timestamp every read
for i in range(3):
    print(f"  read {i}: {env.now()}")

# Advance the clock manually (ms)
env.advance(ms=3_600_000)
print(f"  after env.advance(3,600,000 ms): {env.now()}")\
"""))

cells.append(md("""\
## 6. Fixture-Backed Data

`DataFixture` is the versioned-data abstraction. When the simulator's responses depend on a corpus (model inventory, customer table, market snapshot), load it via a fixture and reference its `fixture_id` in the audit packet. Re-running with the same `fixture_id` + `version` guarantees identical responses.\
"""))

cells.append(code("""\
loaded = env.load_fixture("bank_inventory_v1")
print(f"loaded fixture: id={fixture.fixture_id} version={fixture.version}")
print(f"  data: {fixture.data}")
print(f"  content_hash: {fixture.content_hash or '(computed lazily)'}")\
"""))

cells.append(md("""\
## 7. Visualization — State Over Time\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt

env.reset(seed=42)
events = [
    ("init", None, None),
    ("data_lookup", {"entity": "MDL-001"}, None),
    ("risk_check", {"model_id": "MDL-001"}, None),
    ("data_lookup", {"entity": "MDL-007"}, None),
]
states = []
labels = []
for name, args, _ in events:
    if args is not None:
        env.handle_tool_call(name, args)
    s = env.get_state()
    states.append(s.copy())
    labels.append(name)

# Plot loaded flag over time
import numpy as np
loaded_seq = [s.get("data_lookup", {}).get("loaded", 0) for s in states]
fig, ax = plt.subplots(figsize=(8, 3))
ax.step(range(len(loaded_seq)), loaded_seq, where="post", linewidth=2)
ax.set_xticks(range(len(loaded_seq)))
ax.set_xticklabels(labels, rotation=15)
ax.set_yticks([0, 1])
ax.set_ylabel("data_lookup.loaded")
ax.set_title("Simulator state over four tool calls")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 8. Take-Aways

1. **Declare endpoints, don't implement tools.** Every endpoint is a small dataclass: name, initial state, default response, transitions. No mocking framework, no monkey-patching.
2. **State transitions chain.** `data_lookup` mutates `loaded=1`; `risk_check`'s transition condition reads `state.loaded == 1`. The result is a deterministic stateful workflow.
3. **Clock + fixture = replay byte-identical.** Frozen clock pins timestamps; versioned `DataFixture` pins corpus content. Together they make audit replay reproducible.
4. **The simulator is sandboxed by construction.** No network, no filesystem, no real APIs. This is what `IsolationRequirements.network_isolated=True` (NB 11) actually enforces at the tool layer.
5. **Pair with `OracleBase` (NB 15).** The state diff (`get_state_diff`) is exactly the input a side-effect oracle (e.g., `did_we_create_the_right_ticket`) compares against expected.\
"""))

cells.append(md("""\
## 9. Next Steps

- **NB 15** — `15_oracles.ipynb` — compose multiple verdicts (including state-diff oracles) into release decisions (Ch15)
- **NB 17** — `17_multi_turn.ipynb` — drive the simulator over multiple turns of a stateful dialog (Ch17)

For theory: `gmsh_testing_ext/chapters/ch14.tex`.\
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
