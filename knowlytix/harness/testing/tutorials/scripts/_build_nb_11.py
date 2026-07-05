# SPDX-License-Identifier: Apache-2.0
"""Build NB 11 — Connecting Your SUT (Adapters, Targets, Traces) — from cell
definitions, then execute.

Migrated from knowlytix/harness/testing/tutorials/connecting_a_sut.ipynb and realigned to
the chapter-anchored thin-recipe template. Walks the four integration levels
on the bank model-risk corpus and includes a real-Claude SUT capstone.

Cost: ~$0.05 / ~2-3 min wall clock.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_11.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "11_sut_adapters.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 11: Connecting Your SUT — Adapters, Targets, and Traces

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch11.tex` — SUT Abstraction: Adapters, Targets, and Traces
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch11_sut_abstraction.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §14
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY` (for the Level 4 capstone); ch04 fixture; NB 00 (Quick Start)

**What you'll do.** Wire a System Under Test through the harness's uniform `evaluate(question, context) -> SUTTrace` contract. The notebook walks the four integration levels — passive observation, active interception, full simulation, and a real-LLM capstone — against the bank model-risk corpus (MDL-001 CreditScore-XGB, MDL-002 FraudNet-LSTM, etc.). By the end, you'll have built a `TargetProfile`, an `AgentAdapter` subclass, a `ToolGateway` capturing fault decisions, and an `EnvironmentSimulator` with deterministic synthetic endpoints — plus you'll have wrapped a real Claude Sonnet 4.6 agent through the same `SUTAdapter` contract to prove the contract is uniform.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")

from _shared import (
    load_ch04_store, make_anthropic_client,
    DEFAULT_TUTORIAL_MODEL, seed_all,
)
from knowlytix.harness.testing import (
    SUTAdapter, AgentAdapter, SUTTrace, ToolCallRecord,
    TargetProfile, IsolationRequirements,
    ToolGateway, FaultProfile,
    EnvironmentSimulator, ToolEndpointSpec, StateTransition,
)
seed_all()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

The harness only sees a SUT through one method:

```python
def evaluate(self, question: str, context: str) -> SUTTrace: ...
```

Whatever the underlying system is — a bare LLM, a tool-using agent, a RAG pipeline, a browser-driven UI — its adapter must accept a question and a context string and return a populated `SUTTrace`. The trace carries the answer (`raw_output`), the tool sequence (`tool_calls`), and timing metadata (`latency_ms`, `token_count`).

Four escalation levels for what the harness can verify:

- **Level 1 (passive observation).** The adapter emits a `SUTTrace`; the harness reads the answer and the observed tool sequence. No interception, no fault injection. Enough to verify final answers and tool ordering.
- **Level 2 (active interception).** A `ToolGateway` is installed so every tool call passes through gateway hooks first. The gateway can block calls (policy), inject faults (latency, errors, mock responses), and write a full audit transcript. Needed for recovery-path and policy verification.
- **Level 3 (full simulation).** Real tool implementations are replaced by an `EnvironmentSimulator` with declarative `ToolEndpointSpec` entries. Every tool call hits a deterministic synthetic endpoint. What audit-grade replay demands, but you pay the cost of reimplementing each tool's behavior.
- **Level 4 (real agent capstone).** The same adapter contract wraps a real Claude Sonnet 4.6 agent driving the bank model-risk corpus. Proves the abstraction layer is uniform — there is no special "production mode" wiring.

Start at Level 1. Move up only when the next level unlocks a verification you actually need.\
"""))

cells.append(md("""\
## 3. Load the Ground-Truth Store

We'll point both the gateway plausibility scorer and the simulator's `data_lookup` endpoint at the canonical ch04 bank inventory so the tool responses reflect actual recorded values.\
"""))

cells.append(code("""\
store = load_ch04_store()
print(f"store stats: {store.stats()}")

# Pull a few real values out for the simulator + stubs to return
mdl001_auc = store.lookup_enm("performance_metrics", "MDL-001/Test AUC")
mdl001_status = list(store.query_triples(head="mdl-001", relation="has_status"))[0][2]
print()
print(f"MDL-001 test_auc (ENM): {mdl001_auc}")
print(f"MDL-001 status (triple): {mdl001_status}")\
"""))

cells.append(md("""\
## 4. Build a `TargetProfile`

The target profile is the SUT's audit identity: name, version, type, isolation requirements. It is recorded in audit packets and replay manifests so reruns are reproducible.\
"""))

cells.append(code("""\
target = TargetProfile(
    name="ModelRiskAgent",
    version="0.1.0",
    target_type="agent",
    endpoint="local",
    model_name="claude-sonnet-4-6",
    capabilities=["tool_use"],
    timeout_ms=30_000,
    max_concurrent=1,
    isolation=IsolationRequirements(
        network_isolated=True,
        sandbox_filesystem=True,
        synthetic_data_only=True,
    ),
    metadata={"notebook": "11_sut_adapters", "corpus": "ch04_ground_truth"},
)
print(target.summary())
print(f"target_id: {target.target_id}")\
"""))

cells.append(md("""\
## 5. Stub Agent + `AgentAdapter` Subclass

The stub below pretends to call a `data_lookup` tool that returns a real ENM value from the bank inventory, then composes the answer. The adapter subclass translates the agent's internal log into the standard `ToolCallRecord` shape and packages the result as a `SUTTrace`.\
"""))

cells.append(code("""\
class StubBankAgent:
    \"\"\"A minimal stand-in for a tool-using model-risk agent. Returns real
    ENM-backed values from the ch04 store so the trace reflects production
    shape, not a hand-typed constant.\"\"\"

    def __init__(self, store):
        self.store = store
        self._records: list[dict] = []

    def ask(self, question: str, context: str) -> str:
        t0 = time.time()
        auc = self.store.lookup_enm("performance_metrics", "MDL-001/Test AUC")
        status = list(self.store.query_triples(
            head="mdl-001", relation="has_status"
        ))[0][2]
        self._records.append({
            "tool_name": "data_lookup",
            "arguments": {"entity": "MDL-001"},
            "result": f'{{"auc": {auc}, "status": "{status}"}}',
            "latency_ms": (time.time() - t0) * 1000,
        })
        return str(auc)

    @property
    def tool_calls(self) -> list[dict]:
        return list(self._records)

    def reset(self) -> None:
        self._records = []


class StubBankAdapter(AgentAdapter):
    \"\"\"Translate StubBankAgent records into a SUTTrace.\"\"\"

    def __init__(self, agent: StubBankAgent):
        self.agent = agent
        self.harness = None
        self.gateway = None

    def evaluate(self, question: str, context: str) -> SUTTrace:
        self.agent.reset()
        t0 = time.time()
        try:
            answer = self.agent.ask(question, context)
            return SUTTrace(
                raw_output=answer,
                tool_calls=[ToolCallRecord(**rec)
                            for rec in self.agent.tool_calls],
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            return SUTTrace(
                raw_output="",
                latency_ms=(time.time() - t0) * 1000,
                error=str(exc),
            )


agent = StubBankAgent(store)
adapter = StubBankAdapter(agent)
trace = adapter.evaluate(
    "What is the test AUC of MDL-001?",
    "MDL-001 inventory row.",
)
print("answer    :", trace.raw_output)
print("latency_ms:", round(trace.latency_ms, 3))
print("tool_calls:")
for tc in trace.tool_calls:
    print(" ", tc.tool_name, tc.arguments, "->", tc.result)\
"""))

cells.append(md("""\
## 6. Level 1 — Passive Observation

Call `adapter.evaluate(...)` and read the trace. No gateway, no simulator. The harness can verify the answer string and the observed tool sequence.\
"""))

cells.append(code("""\
level1_trace = adapter.evaluate(
    "What is the test AUC of MDL-001?",
    "MDL-001 row, see model_risk_assessment.md",
)
print("Level 1 inspectable:")
print(f"  raw_output   : {level1_trace.raw_output}")
print(f"  n tool_calls : {len(level1_trace.tool_calls)}")
print(f"  latency_ms   : {round(level1_trace.latency_ms, 3)}")\
"""))

cells.append(md("""\
## 7. Level 2 — Active Interception with `ToolGateway`

Install a `ToolGateway` in front of a `MockHarness`-backed adapter so every tool call passes through gateway hooks first. The gateway records an allow/block/inject decision per tool call. We use `MockHarness` here because it implements the `harness.hooks` interface that `ToolGateway.install` expects — it's a testing fixture, not a fallback path.

A `FaultProfile` is configured to inject a 100% error rate on `risk_check` calls; the gateway transcript should show the intercept.\
"""))

cells.append(code("""\
from knowlytix.harness.fixtures import MockHarness

mock_harness = MockHarness(recovery_mode="fallback")

fault = FaultProfile(
    tool_pattern="risk_check",
    error_rate=1.0,
    error_message="Simulated outage",
)
gateway = ToolGateway(
    store=store,
    fault_profiles=[fault],
    run_id="nb11_level2",
)
level2_adapter = AgentAdapter(mock_harness, gateway=gateway)

level2_trace = level2_adapter.evaluate(
    "What is the risk rating of MDL-001?",
    "MDL-001 CreditScore-XGB risk=High, AUC=0.8412",
)
transcript = gateway.get_transcript()

print("Level 2 — inspectable + intercept-able:")
print(f"  raw_output     : {level2_trace.raw_output[:80]}")
print(f"  n tool_calls   : {len(level2_trace.tool_calls)}")
print(f"  gateway decisions: {len(transcript.decisions)}")
for d in transcript.decisions:
    print(f"    {d.tool_name:14s}  action={d.action:8s}  reason={d.reason}")

assert len(transcript.decisions) >= 1, (
    "Gateway captured no decisions — adapter is not wired correctly"
)\
"""))

cells.append(md("""\
## 8. Level 3 — Full Simulation with `EnvironmentSimulator`

Replace the real tool implementations with declarative endpoints. Every tool call hits a deterministic synthetic endpoint; state changes are observable for side-effect oracles. The `data_lookup` endpoint below returns ENM-backed bank-inventory values so the simulation matches the real corpus.\
"""))

cells.append(code("""\
auc = store.lookup_enm("performance_metrics", "MDL-001/Test AUC")
env = EnvironmentSimulator(
    tool_specs=[
        ToolEndpointSpec(
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
        ),
        ToolEndpointSpec(
            tool_name="risk_check",
            default_response='{"risk_level": "High"}',
        ),
    ],
)
env.reset(seed=42)

before_state = env.get_state()
resp_a = env.handle_tool_call("data_lookup", {"entity": "MDL-001"})
resp_b = env.handle_tool_call("risk_check", {"model_id": "MDL-001"})
after_state = env.get_state()
diff = env.get_state_diff(before_state, after_state)

print("Level 3 — deterministic responses:")
print(f"  data_lookup -> {resp_a}")
print(f"  risk_check  -> {resp_b}")
print("Side-effect diff:")
print(f"  added  : {diff['added']}")
print(f"  changed: {diff['changed']}")\
"""))

cells.append(md("""\
## 9. Level 4 — Real Claude Agent through the Same Contract

The four-level escalation only matters if the abstraction holds at the production end. Here we wrap a real Claude Sonnet 4.6 call as an `LLMAdapter` and run the same `evaluate(question, context) -> SUTTrace` contract against the bank model-risk corpus. The harness sees no difference — the trace shape is identical to the stub's.\
"""))

cells.append(code("""\
from knowlytix.harness.testing import LLMAdapter
client = make_anthropic_client()

# LLMAdapter is the harness's bare-LLM wrapper — no tools, just a single
# Claude call producing a raw_output string. Same contract as StubBankAdapter.
class ClaudeBankAdapter(LLMAdapter):
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.harness = None
        self.gateway = None

    def evaluate(self, question: str, context: str) -> SUTTrace:
        t0 = time.time()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=("You are a model-risk analyst. Answer with just the "
                        "requested value (number, identifier, or one-word "
                        "status) prefixed with 'ANSWER: '."),
                messages=[{
                    "role": "user",
                    "content": f"<document>\\n{context}\\n</document>\\n\\n{question}",
                }],
            )
            raw = resp.content[0].text.strip()
            return SUTTrace(
                raw_output=raw,
                latency_ms=(time.time() - t0) * 1000,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        except Exception as exc:
            return SUTTrace(
                raw_output="",
                latency_ms=(time.time() - t0) * 1000,
                error=str(exc),
            )

claude_adapter = ClaudeBankAdapter(client, DEFAULT_TUTORIAL_MODEL)
level4_trace = claude_adapter.evaluate(
    "What is the Test AUC of MDL-001 in the model risk assessment document?",
    "MDL-001 (CreditScore-XGB) is a high-risk-tier credit-decisioning model. "
    "Its Test AUC on the credit panel hold-out is 0.8412.",
)
print("Level 4 — real Claude through the same contract:")
print(f"  raw_output      : {level4_trace.raw_output[:120]}")
print(f"  latency_ms      : {level4_trace.latency_ms:.0f}")
print(f"  input_tokens    : {level4_trace.input_tokens}")
print(f"  output_tokens   : {level4_trace.output_tokens}")
print()
print("Contract sanity: same SUTTrace shape as Level 1?", \
      isinstance(level4_trace, type(level1_trace)))\
"""))

cells.append(md("""\
## 10. Visualization — What Each Level Reveals\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

levels = ["L1\\nObservation", "L2\\nInterception", "L3\\nSimulation", "L4\\nReal LLM"]
inspectable = [1, 1, 1, 1]
fault_injectable = [0, 1, 1, 0]
deterministic = [0, 0, 1, 0]
real_agent_behavior = [0, 0, 0, 1]

x = np.arange(len(levels))
w = 0.2
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - 1.5*w, inspectable, w, label="inspectable")
ax.bar(x - 0.5*w, fault_injectable, w, label="fault-injectable")
ax.bar(x + 0.5*w, deterministic, w, label="deterministic")
ax.bar(x + 1.5*w, real_agent_behavior, w, label="real-agent behavior")
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylim(0, 1.2)
ax.set_ylabel("capability (1 = unlocked)")
ax.set_title("What each integration level reveals")
ax.legend(loc="upper left", ncol=2)
fig.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 11. Take-Aways

1. **One contract, four levels.** `evaluate(question, context) -> SUTTrace` is the entire SUT-side surface. Everything else (gateways, simulators, target profiles) lives in front of or behind that one method.
2. **TargetProfile is the audit identity.** Pin `name`, `version`, `model_name` per SUT revision so reruns are reproducible and audit packets can identify which build produced which result.
3. **MockHarness is a test fixture, not a fallback.** It implements the `harness.hooks` interface so you can exercise the gateway against something deterministic. The real fallback policy is in `_shared.require_anthropic_key()` — raise on missing key, no mock LLM.
4. **Each escalation costs something.** Level 1 is free. Level 2 costs gateway hook wiring. Level 3 costs reimplementing every tool. Level 4 costs the API budget. Climb only as high as the verification requires.
5. **The Claude adapter is one class.** Wrapping a real LLM through `LLMAdapter` is ~20 lines (see `ClaudeBankAdapter` in §9). For tool-using agents, swap `LLMAdapter` for `AgentAdapter` and populate `tool_calls`.\
"""))

cells.append(md("""\
## 12. Next Steps

- **NB 12** — `12_tool_gateway.ipynb` — full `ToolGateway` walkthrough: fault profiles, YAML loading, audit transcripts (Ch12)
- **NB 13** — `13_policy_engine.ipynb` — pair the gateway with `PolicyEngine` for runtime rule enforcement (Ch13)
- **NB 14** — `14_environment_sim.ipynb` — `EnvironmentSimulator` deep dive: state transitions, scheduled events, fixtures, clock control (Ch14)

For theory:
- `gmsh_testing_ext/chapters/ch11.tex` — SUT abstraction, adapter taxonomy, trace contract\
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
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  execution wall time: {time.time() - t0:.1f}s")

    print(f"Writing executed notebook back to {NB_PATH}...")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
