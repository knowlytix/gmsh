# SPDX-License-Identifier: Apache-2.0
"""Build NB 12 — Tool Gateway and Fault Injection (Ch12)."""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "12_tool_gateway.ipynb"


def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)


cells = []
cells.append(md("""\
# Tutorial 12: Tool Gateway and Fault Injection

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch12.tex` — The Tool Gateway: Fault Injection and Interception
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch12_tool_gateway.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §15
**Prereqs:** NB 11 (SUT Adapters)

**What you'll do.** Wire a `ToolGateway` in front of a `MockHarness`-backed agent. Configure three `FaultProfile`s that inject latency, errors, and mock responses on a bank model-risk tool surface. Inspect the gateway transcript that captures every decision the gateway made.\
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
    AgentAdapter, SUTTrace, ToolGateway, FaultProfile,
)
from knowlytix.harness.fixtures import MockHarness
seed_all()
store = load_ch04_store()
print(f"store: {store.stats()['entities']} entities, {store.stats()['triples']} triples")\
"""))

cells.append(md("""\
## 2. Concept

A `ToolGateway` is a per-run interceptor that sits between the agent's tool-use intentions and the actual tool implementation. It can:

- **Block** a call (policy-driven, when paired with `PolicyEngine` — NB 13)
- **Inject latency** (simulate slow downstream services)
- **Inject errors** (simulate outages, malformed responses)
- **Inject mock responses** (return canned data for deterministic tests)
- **Audit every decision** (the transcript is the audit trail)

Each `FaultProfile` carries:
- `tool_pattern`: glob over tool names (e.g., `risk_check`, `mode*`, `*`)
- `latency_ms`: extra latency to inject
- `error_rate`: probability of injecting an error
- `error_message`: payload returned on error
- `start_time` / `end_time`: schedule the fault to a clock window\
"""))

cells.append(md("""\
## 3. Configure Three Fault Profiles

A realistic bank model-risk test schedule: simulate a `risk_check` outage, add 500ms of latency to `data_lookup`, and mock `escalate_to_human`.\
"""))

cells.append(code("""\
fault_profiles = [
    FaultProfile(
        tool_pattern="risk_check",
        error_rate=1.0,
        error_message="Risk-check service outage (simulated)",
    ),
    FaultProfile(
        tool_pattern="data_lookup",
        latency_ms=500.0,
    ),
    FaultProfile(
        tool_pattern="escalate_to_human",
        error_rate=1.0,
        error_message="approval_queue=mocked",
    ),
]
for fp in fault_profiles:
    print(f"  pattern={fp.tool_pattern:24s} latency_ms={fp.latency_ms:>6.0f} "
          f"error_rate={fp.error_rate:.0%} message={fp.error_message!r}")\
"""))

cells.append(md("""\
## 4. Install the Gateway in Front of an Adapter\
"""))

cells.append(code("""\
gateway = ToolGateway(
    store=store,
    fault_profiles=fault_profiles,
    run_id="nb12_gateway_demo",
)
mock_harness = MockHarness(recovery_mode="fallback")
adapter = AgentAdapter(mock_harness, gateway=gateway)

# Run a question that should exercise multiple tools
trace = adapter.evaluate(
    "What is the current risk rating of MDL-001 and should we escalate?",
    "MDL-001 (CreditScore-XGB) is High-tier credit decisioning, AUC=0.8412",
)
print(f"  raw_output      : {trace.raw_output[:120]}")
print(f"  n tool_calls    : {len(trace.tool_calls)}")\
"""))

cells.append(md("""\
## 5. Inspect the Gateway Transcript\
"""))

cells.append(code("""\
transcript = gateway.get_transcript()
print(f"Gateway transcript: {len(transcript.decisions)} decisions")
print()
print(f"{'#':3s}  {'tool':18s}  {'action':12s}  {'latency':>9s}  reason")
print("-" * 90)
for i, d in enumerate(transcript.decisions, 1):
    lat = f"{d.latency_ms:.0f}ms" if hasattr(d, 'latency_ms') and d.latency_ms else "-"
    print(f"{i:3d}  {d.tool_name:18s}  {d.action:12s}  {lat:>9s}  {d.reason[:50]}")\
"""))

cells.append(md("""\
## 6. Run Without Faults for Contrast\
"""))

cells.append(code("""\
clean_gateway = ToolGateway(store=store, fault_profiles=[], run_id="nb12_clean")
clean_adapter = AgentAdapter(MockHarness(recovery_mode="fallback"), gateway=clean_gateway)
clean_trace = clean_adapter.evaluate(
    "What is the current risk rating of MDL-001 and should we escalate?",
    "MDL-001 is High-tier credit decisioning, AUC=0.8412",
)
clean_transcript = clean_gateway.get_transcript()
print(f"Clean run: {len(clean_transcript.decisions)} decisions, "
      f"all allow={sum(1 for d in clean_transcript.decisions if d.action=='allow')}/"
      f"{len(clean_transcript.decisions)}")\
"""))

cells.append(md("""\
## 7. Visualization — Action Distribution Across the Two Runs\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
from collections import Counter

def action_counts(t):
    return Counter(d.action for d in t.decisions)

c_fault = action_counts(transcript)
c_clean = action_counts(clean_transcript)
all_actions = sorted(set(c_fault) | set(c_clean))
import numpy as np
x = np.arange(len(all_actions))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, [c_fault.get(a, 0) for a in all_actions], w,
       label="with fault profiles", color="#c14a4a")
ax.bar(x + w/2, [c_clean.get(a, 0) for a in all_actions], w,
       label="clean (no faults)", color="#3aa56b")
ax.set_xticks(x); ax.set_xticklabels(all_actions)
ax.set_ylabel("# decisions")
ax.set_title("Gateway action distribution: with vs without fault profiles")
ax.legend()
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 8. Take-Aways

1. **Three injection types, one interface.** Latency, errors, mock responses — `FaultProfile` covers all three with the same dataclass.
2. **`tool_pattern` supports globs.** `risk_*`, `*_check`, `*` — useful when you want to slow down every external API call regardless of name.
3. **Transcript is the audit trail.** Every decision is recorded with the action taken and the reason. Persist it alongside the `DOETracingGMS` artifact for replay.
4. **`MockHarness` is the test surface, not the production surface.** The same `ToolGateway` wraps a production agent unchanged; we use MockHarness here so the demo is deterministic.
5. **Pair with `PolicyEngine` for blocking.** Fault injection simulates real failures; policy enforcement blocks intentional misuse. NB 13 shows the pairing.\
"""))

cells.append(md("""\
## 9. Next Steps

- **NB 13** — `13_policy_engine.ipynb` — block tool calls under policy rules (Ch13)
- **NB 14** — `14_environment_sim.ipynb` — full simulator with state transitions instead of fault injection (Ch14)

For theory: `gmsh_testing_ext/chapters/ch12.tex`, `gmsh_testing_ext/notebooks/ch12_tool_gateway.ipynb`.\
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
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
