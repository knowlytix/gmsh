# GMSH Agentic Testing Harness — User Guide

A step-by-step guide to testing large-language-model (LLM) agents with the GMSH platform. This guide walks you through every feature with runnable examples.

> **Companion document:** For complete constructor signatures, all dataclass fields, method return types and enumerations, see the [API Reference](API_REFERENCE.md). Each tutorial below links to the relevant API section.

---

## Table of Contents

### Part I: Getting Started
1. [What is GMSH?](#1-what-is-gmsh)
2. [Quick Start](#2-quick-start)
3. [Core Concepts](#3-core-concepts)
3.5. [Connecting your SUT](#35-connecting-your-sut)
3.6. [Querying a Trained Store](#36-querying-a-trained-store)

### Part II: Agentic Harness (workstreams WS1–WS7)
4. [Tutorial 1: Your First Benchmark Run](#4-tutorial-1-your-first-benchmark-run)
5. [Tutorial 2: Tool Gateway and Fault Injection](#5-tutorial-2-tool-gateway-and-fault-injection)
6. [Tutorial 3: Policy Engine](#6-tutorial-3-policy-engine)
7. [Tutorial 4: Authored Test Scenarios](#7-tutorial-4-authored-test-scenarios)
8. [Tutorial 5: Run Lifecycle Management](#8-tutorial-5-run-lifecycle-management)
9. [Tutorial 6: Replay and Regression Testing](#9-tutorial-6-replay-and-regression-testing)
10. [Tutorial 7: Extended Evaluators (Process, Recovery, Interaction)](#10-tutorial-7-extended-evaluators)
11. [Tutorial 8: Risk-Tiered Release Gates](#11-tutorial-8-risk-tiered-release-gates)

### Part III: Black-Box Testing Platform
12. [Tutorial 9: Target Profiles](#12-tutorial-9-target-profiles)
13. [Tutorial 10: Environment Simulation](#13-tutorial-10-environment-simulation)
14. [Tutorial 11: Oracle Framework](#14-tutorial-11-oracle-framework)
15. [Tutorial 12: Severity Classification](#15-tutorial-12-severity-classification)
16. [Tutorial 13: Campaign Management](#16-tutorial-13-campaign-management)

### Part IV: Extended Test Generation
17. [Tutorial 14: Taxonomy Generators](#17-tutorial-14-taxonomy-generators)
18. [Tutorial 15: Multi-Turn Testing](#18-tutorial-15-multi-turn-testing)
19. [Tutorial 16: Safety and Isolation](#19-tutorial-16-safety-and-isolation)

### Part V: Scale and Artifact Management
20. [Tutorial 17: Parallel Execution](#20-tutorial-17-parallel-execution)
21. [Tutorial 18: Artifact Versioning](#21-tutorial-18-artifact-versioning)
22. [Tutorial 19: Browser and UI Testing](#22-tutorial-19-browser-and-ui-testing)

### Part VI: GMS-Native Behavioral Testing
23. [Tutorial 20: Performance Profiling](#23-tutorial-20-performance-profiling)
24. [Tutorial 21: Cost and Token Tracking](#24-tutorial-21-cost-and-token-tracking)
25. [Tutorial 22: Agent Memory Testing](#25-tutorial-22-agent-memory-testing)
26. [Tutorial 23: Hallucination Detection](#26-tutorial-23-hallucination-detection)
27. [Tutorial 24: Confidence Calibration](#27-tutorial-24-confidence-calibration)

### Part VII: Multi-Source and Advanced Testing
28. [Tutorial 25: Multi-Agent Consensus](#28-tutorial-25-multi-agent-consensus)
29. [Tutorial 26: Cross-Document Conflicts](#29-tutorial-26-cross-document-conflicts)
30. [Tutorial 27: Entity Disambiguation](#30-tutorial-27-entity-disambiguation)
31. [Tutorial 28: Paraphrase Invariance](#31-tutorial-28-paraphrase-invariance)
32. [Tutorial 29: Streaming Consistency](#32-tutorial-29-streaming-consistency)
33. [Tutorial 30: Live Dashboard](#33-tutorial-30-live-dashboard)

### Part VIII: Design Optimization
33.5. [Tutorial 31: Optimal-Design DOE (Sobol+Refine)](#335-tutorial-31-optimal-design-doe-sobolrefine)

### Reference
34. [End-to-End Walkthrough](#34-end-to-end-walkthrough)
35. [CLI Reference](#35-cli-reference)
36. [Configuration Reference](#36-configuration-reference)
37. [Troubleshooting](#37-troubleshooting)
38. [Common Recipes](#38-common-recipes)

---

## 1. What is GMSH?

> **Module:** Overview  
> **Key Concepts:** GMS knowledge graph, geometric verification, Design-of-Experiments (DOE) factor analysis  
> **Prereqs:** None — this is the starting point

GMSH (Geometric Memory System Harness) is an enterprise platform for **black-box testing of LLM agents**, where the system under test (SUT) is whatever you wire to the harness — an agent loop, a bare LLM call, a retrieval-augmented-generation (RAG) pipeline, or a browser-driven UI. It uses geometric knowledge graphs (GMS) as the ground-truth backbone to verify agent behavior across multiple dimensions:

- **Correctness** — Does the agent answer questions accurately?
- **Process compliance** — Does the agent call the right tools in the right order?
- **Fault tolerance** — Does the agent recover gracefully when tools fail?
- **Policy adherence** — Does the agent respect access controls and approval requirements?
- **Interaction quality** — Does the agent escalate and clarify appropriately?

All verification is anchored by **GMS geometric primitives** (geodesic distance, tension energy, holonomy defect) rather than string matching or heuristics.

### Architecture at a Glance

```
Your Agent (SUT)
      |
      v
[Tool Gateway] ── intercepts tool calls, injects faults
      |
[Policy Engine] ── enforces rules before execution
      |
      v
[DOE Benchmark] ── systematic question generation + evaluation
      |
      v
[Claim Compiler] ── decomposes answers into typed claims
      |
      v
[Verification Router]
      |
      ├── GraphVerifier      (fact accuracy)
      ├── ProcessEvaluator   (tool ordering)
      ├── RecoveryEvaluator  (fault recovery)
      ├── InteractionEvaluator (escalation quality)
      └── ... 5 more verifiers
      |
      v
[Release Gates] ── risk-tiered pass/fail determination
```

---

## 2. Quick Start

> **Module:** `knowlytix.harness.testing.benchmark`, `knowlytix.harness.testing.config`  
> **Import:** `from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig, make_evaluator`  
> **Key Classes:** `DOEGMSBenchmark`, `DOEHarnessConfig`, `make_evaluator`  
> **Prereqs:** None — standalone quickstart

### Prerequisites

```bash
# Activate the project environment
source ~/cluster/spark-venv/bin/activate

# Verify the installation
python -c "from knowlytix.harness.testing import DOEGMSBenchmark; print('OK')"
```

### 30-Second Demo

Run the full test suite to see everything working:

```bash
cd ~/jupyterlab/kg-memory
python demo/test_agentic_harness.py
```

This runs 54 tests covering all 7 workstreams and takes ~2 minutes on CPU.

### Minimal Python Example

```python
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig

# 1. Configure
config = DOEHarnessConfig(
    markdown_path="path/to/your/document.md",
    factor_group="quick_screen",
    n_runs=4,
    enable_typed_verdicts=True,
)

# 2. Ingest document into GMS knowledge graph
bench = DOEGMSBenchmark(config)
store = bench.ingest()
print(f"Store: {store.stats()}")

# 3. Generate test questions and run evaluation
def my_evaluator(question: str, context: str) -> str:
    """Replace with your agent's answer function."""
    return "your agent's answer here"

bench.generate_questions()
result = bench.run(evaluator=my_evaluator)
print(f"Accuracy: {result.accuracy:.1%}")
```

> **API Reference:** [DOEGMSBenchmark](API_REFERENCE.md#7-benchmark-orchestration) · [DOEHarnessConfig](API_REFERENCE.md#doeharnessconfig)

---

## 3. Core Concepts

> **Module:** Conceptual overview  
> **Key Concepts:** `GMSExpertStore`, geometric primitives, DOE factor analysis  
> **Prereqs:** None

### GMS Expert Store

A trained geometric knowledge graph built from your document. Contains:

- **Entities** — concepts from the document (e.g., "MDL-001", "XGBoost", "0.8412")
- **Triples** — relationships between entities (e.g., `("MDL-001", "has_test_auc", "0.8412")`)
- **ENM** — exact numeric memory for precise value lookup
- **Model** — trained geometric model on a hypersphere for computing distances

### GMS Geometric Primitives

The six core primitives form the complete verification vocabulary. Any claim an agent makes can be decomposed into one or more primitive checks.

| Primitive | What It Measures | Used For |
|-----------|-----------------|----------|
| `score_triple(h, r, t)` | Geodesic distance on $S^{m-1}$ | Fact verification, plausibility (failure mode FM-1: Hallucination) |
| `tension_energy(a, b)` | Inner-product energy in u-space | Contradiction detection, recovery quality |
| `check_holonomy(path, direct)` | Frobenius defect of composed rotors | Multi-hop consistency, process compliance (FM-4) |
| `lookup_enm(category, entity_id)` | SHA-256-verified exact numeric value | Numeric precision, state corruption (FM-6) |
| `link_predict(h, r, top_k)` | Beam search on the manifold | Distractor generation, candidate ranking |
| `query_triples(h, r, t)` | Graph pattern matching | Structural lookup, tool/policy verification (FM-2/3) |

The store also exposes `fuzzy_match_entity(name)` for mapping free-text entity references to graph entities, used internally by the extended evaluators and oracles.

### DOE Design Matrix

Questions are expanded via a **Design of Experiments** matrix (Sobol, Latin Hypercube Sampling — LHS, or Grid) that systematically varies factors like:

- Query phrasing style
- Context presentation mode
- Entity name format
- Adversarial perturbation level

This ensures you test your agent under diverse conditions, not just the happy path.

### Evaluator Function

The simplest way to connect your agent. A function with signature:

```python
def evaluator(question: str, context: str) -> str:
    """
    question: natural language question about the document
    context: relevant document section(s)
    returns: your agent's answer as a string
    """
```

> **API Reference:** [Core Store](API_REFERENCE.md#1-core-store) · [Document Graph](API_REFERENCE.md#6-document-graph) · [Geometric Generators](API_REFERENCE.md#8-geometric-generators)

---

## 3.5 Connecting your SUT

> **Module:** `knowlytix.harness.testing.adapters`, `knowlytix.harness.testing.target`, `knowlytix.harness.testing.gateway`, `knowlytix.harness.testing.environment`  
> **Import:** `from knowlytix.harness.testing import SUTAdapter, AgentAdapter, LLMAdapter, RAGAdapter, SUTTrace, ToolCallRecord, TargetProfile, IsolationRequirements, ToolGateway, FaultProfile, EnvironmentSimulator, ToolEndpointSpec`  
> **Key Classes:** `SUTAdapter`, `SUTTrace`, `ToolCallRecord`, `TargetProfile`, `ToolGateway`, `EnvironmentSimulator`  
> **Prereqs:** None — read this before Tutorial 1 if you are wiring a real agent  
> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch11.tex`](../../gmsh_testing_ext/chapters/ch11.tex) (SUT Abstraction: Adapters, Targets, and Traces)

This is a focused recipe for connecting a System Under Test to the harness. The information is also covered in §12 (Target Profiles), §13 (Environment Simulation), and §5 (Tool Gateway), but here it is collected end-to-end so you can stand up an integration in one read. We placed this as §3.5 (rather than renumbering §4–§38) to keep all existing API-Reference anchors stable.

A SUT is whatever you want to test: an agent loop, a bare LLM call, a RAG pipeline, a browser-driven UI, or something else. The harness only sees it through a uniform contract — `evaluate(question, context) -> SUTTrace` — and that one method is the bridge you are building.

### 3.5.A Identify the SUT — `TargetProfile` and `IsolationRequirements`

Every run is anchored to a versioned, declared target so audit reports, replay manifests, and campaign comparisons can identify which SUT produced which result. Build a `TargetProfile` once per SUT version:

```python
from knowlytix.harness.testing import TargetProfile, IsolationRequirements

target = TargetProfile(
    target_id="",                       # auto-generated as tgt_<uuid8>
    name="DS Agent v2.1",
    version="2.1.0",
    target_type="agent",                # "llm" | "agent" | "rag" | "browser"
    endpoint="local",                   # API URL or "local" for in-process
    model_name="claude-sonnet-4-20250514",
    capabilities=["tool_use", "multi_turn"],
    timeout_ms=60_000,
    max_concurrent=1,
    isolation=IsolationRequirements(
        network_isolated=True,
        sandbox_filesystem=True,
        synthetic_data_only=True,
        max_memory_mb=0,                # 0 = unlimited
        max_wall_time_s=0,
    ),
    metadata={"git_sha": "abc123"},
)
print(target.summary())
```

Or load from YAML:

```python
target = TargetProfile.from_yaml("my_test/target_profile.yaml")
```

The profile is recorded in audit packets and replay manifests. Pin `model_name` and `version` so a rerun with the same target ID is reproducible.

> **API Reference:** [Target Profiles](API_REFERENCE.md#24-target-profiles) · See also §12 for an extended walkthrough.

### 3.5.B Implement an `SUTAdapter`

The adapter is the bridge between the harness's `evaluate(question, context) -> SUTTrace` contract and your SUT's native API. The harness ships three subclasses; pick the closest match or write your own.

| Subclass | Use when | Key field populated |
|----------|----------|---------------------|
| `LLMAdapter`   | Direct LLM call, no tools | `raw_output`, `latency_ms`, `input_tokens`, `output_tokens` |
| `AgentAdapter` | Tool-using agent loop     | `raw_output`, `tool_calls`, `latency_ms` |
| `RAGAdapter`   | Retriever + generator     | `raw_output`, `retrieved_context`, `latency_ms` |

For browsers, streaming systems, or scripted multi-turn dialogs, the in-tree examples are `BrowserAdapter`, `StreamingAdapter`, and `MultiTurnAdapter`.

**`SUTTrace` fields** (from `knowlytix.harness.testing.adapters`):

| Field | Type | Meaning |
|-------|------|---------|
| `raw_output` | `str` | Final answer string from the SUT |
| `tool_calls` | `list[ToolCallRecord]` | Every tool invocation observed during the run |
| `latency_ms` | `float` | End-to-end wall-clock latency |
| `token_count` | `int` | Total tokens (input + output) |
| `input_tokens` | `int` | Prompt tokens |
| `output_tokens` | `int` | Completion tokens |
| `retrieved_context` | `str` | RAG retrieval window (RAG adapters only) |
| `error` | `str` | Non-empty if the adapter caught an exception |

**`ToolCallRecord` fields**:

| Field | Type | Meaning |
|-------|------|---------|
| `tool_name` | `str` | Registered tool identifier |
| `arguments` | `dict` | JSON arguments the SUT passed |
| `result` | `str` | Stringified tool response |
| `latency_ms` | `float` | Per-call latency |

**Key contract.** Every tool call your agent makes must end up appended to `SUTTrace.tool_calls`. There is no separate registration step — the adapter is what surfaces "registered tools" to the harness. If your agent keeps a tool log under a different key (for example, `MockHarness` uses `_tool_records` with `{tool, args, result, blocked}`), the adapter is responsible for translating that into `ToolCallRecord(tool_name=..., arguments=..., result=..., latency_ms=...)` before returning.

The shipped `AgentAdapter` does this translation automatically by attaching to `harness.hooks.on_tool_call` and `harness.hooks.on_tool_result`. If you write your own adapter from scratch, you do this translation yourself.

### 3.5.C Three integration levels

Pick the lowest level that answers the question you are testing for. Escalate only when the next level unlocks a verification you actually need.

| Level | What you do | What the harness can verify |
|-------|-------------|-----------------------------|
| **1. Passive observation** | Adapter emits a `SUTTrace`, nothing else | Final answers, tool sequences, latencies |
| **2. Active interception** | Install `ToolGateway` on the harness | All of the above + fault tolerance, recovery, policy compliance |
| **3. Full simulation** | Replace tool implementations with `EnvironmentSimulator` + `ToolEndpointSpec` | All of the above + deterministic state changes, time control |

#### Level 1: Passive observation

Wrap your agent in an `AgentAdapter` and pass it to the benchmark. No fault injection, no policy gate — the harness reads the trace and verifies the answer plus the observed tool sequence.

```python
from knowlytix.harness.testing import AgentAdapter, DOEGMSBenchmark, DOEHarnessConfig
from demo.mock_agent import MockHarness

harness = MockHarness(recovery_mode="fallback")
adapter = AgentAdapter(harness)               # gateway=None for Level 1

config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    factor_group="quick_screen",
    n_runs=4,
    device="cpu",
)
bench = DOEGMSBenchmark(config)
bench.ingest()
result = bench.run(evaluator=adapter)
```

#### Level 2: Active interception

Install a `ToolGateway` so the harness intercepts every tool call. Your SUT does not change — the gateway sits in front of it.

```python
from knowlytix.harness.testing import ToolGateway, FaultProfile

gateway = ToolGateway(
    store=bench.store,
    fault_profiles=[
        FaultProfile(tool_pattern="risk_check", error_rate=0.5,
                     error_message="Risk service unavailable"),
    ],
    run_id="run_001",
)
adapter = AgentAdapter(harness, gateway=gateway)
result = bench.run(evaluator=adapter)
transcript = gateway.get_transcript()
```

The transcript records every allow/block/mock/inject decision. Use Level 2 when you need to test recovery paths, policy enforcement, or audit traceability. See §5 (Tutorial 2) for fault-profile semantics and §6 (Tutorial 3) for policy integration.

#### Level 3: Full simulation

Replace the tool implementations entirely with an `EnvironmentSimulator`. Now the SUT's tool calls hit deterministic synthetic endpoints rather than real services.

```python
from knowlytix.harness.testing import EnvironmentSimulator, ToolEndpointSpec, StateTransition

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
                    state_mutation={"loaded": 1},
                    response_template='{"auc": 0.8412, "risk": "High"}',
                ),
            ],
        ),
        ToolEndpointSpec(
            tool_name="risk_check",
            default_response='{"risk_level": "unknown"}',
        ),
    ],
)
env.reset(seed=42)
response = env.handle_tool_call("data_lookup", {"entity": "MDL-001"})
```

Level 3 trades convenience for full reproducibility: you must reimplement every tool's behavior as a `ToolEndpointSpec`. Use Level 3 only when reproducibility-under-audit demands deterministic state, frozen clocks, or scheduled fault injection. See §13 (Tutorial 10) for the full simulator API.

**When to escalate.** Start at Level 1. Move to Level 2 when you need fault-recovery testing or policy gating. Move to Level 3 only when audit reproducibility demands deterministic state and time control — Level 3 forfeits real tool behavior in exchange for replay-byte-identical traces.

### 3.5.D End-to-end recipe — connecting an arbitrary agent

Here is the full path from "I have an agent class with tool methods" to "DOEGMSBenchmark runs against it and produces an HTML report." The example uses a stub agent so the recipe stands alone — substitute your own class.

```python
"""End-to-end SUT integration recipe."""
import time
from knowlytix.harness.testing import (
    AgentAdapter, SUTTrace, ToolCallRecord,
    TargetProfile, IsolationRequirements,
    DOEGMSBenchmark, DOEHarnessConfig,
)

# 1. Define your SUT (or import an existing agent class).
class MyAgent:
    """Tiny stub — replace with your real agent."""
    def __init__(self):
        self._records: list[dict] = []

    def ask(self, question: str, context: str) -> str:
        # Pretend to call a tool, then answer.
        self._records.append({
            "tool_name": "data_lookup",
            "arguments": {"q": question[:60]},
            "result": "stub-result",
            "latency_ms": 1.0,
        })
        return "stub-answer"

    @property
    def tool_calls(self) -> list[dict]:
        return list(self._records)

    def reset(self) -> None:
        self._records = []

# 2. Wrap it in an SUTAdapter so the harness sees the standard SUTTrace shape.
class MyAgentAdapter(AgentAdapter):
    """Adapter that translates MyAgent's records into SUTTrace."""
    def __init__(self, agent: MyAgent):
        self.agent = agent
        self.harness = None      # AgentAdapter base expects this attribute
        self.gateway = None

    def evaluate(self, question: str, context: str) -> SUTTrace:
        self.agent.reset()
        t0 = time.time()
        try:
            answer = self.agent.ask(question, context)
            latency = (time.time() - t0) * 1000
            return SUTTrace(
                raw_output=answer,
                tool_calls=[ToolCallRecord(**rec)
                            for rec in self.agent.tool_calls],
                latency_ms=latency,
            )
        except Exception as exc:
            return SUTTrace(
                raw_output="",
                latency_ms=(time.time() - t0) * 1000,
                error=str(exc),
            )

# 3. Identify the SUT.
target = TargetProfile(
    name="MyAgent",
    version="0.1.0",
    target_type="agent",
    endpoint="local",
    model_name="my-model",
    capabilities=["tool_use"],
    isolation=IsolationRequirements(synthetic_data_only=True),
)

# 4. Configure the benchmark.
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    factor_group="quick_screen",
    n_runs=4,
    device="cpu",
    enable_typed_verdicts=True,
)

# 5. Run the benchmark against the adapter.
adapter = MyAgentAdapter(MyAgent())
bench = DOEGMSBenchmark(config)
bench.ingest()
bench.generate_questions()
result = bench.run(evaluator=adapter)
bench.generate_report(
    result, bench.analyze(result),
    output_path="my_test/report.html",
)
print(f"Accuracy: {result.accuracy:.1%}")
```

That is the complete integration. Everything in §4 onward (factor analysis, claim verification, gateways, oracles, release gates) builds on this same `evaluate(question, context) -> SUTTrace` contract.

> **API Reference:** [SUT Adapters](API_REFERENCE.md#11-evaluators--adapters) · [Target Profiles](API_REFERENCE.md#24-target-profiles) · [Tool Gateway](API_REFERENCE.md#15-tool-gateway) · [Environment Simulator](API_REFERENCE.md#26-environment-simulator)

---

## 3.6 Querying a Trained Store

> **Module:** `knowlytix.knowledge.store`, `knowlytix.knowledge.config`  
> **Import:** `from knowlytix.knowledge.config import DocGMSConfig` · `from knowlytix.knowledge.store import GMSExpertStore`  
> **Companion Notebook:** [`tutorials/querying_a_trained_gms.ipynb`](tutorials/querying_a_trained_gms.ipynb)  
> **Prereqs:** §3 (the primitives table)

The six primitives in §3 are exposed as methods on a loaded `GMSExpertStore`. A trained store is a directory of eight files (`model.pt`, `model_dims.json`, `adapter.json`, `enm.json`, `triples.json`, `phase.json`, `metadata.json`, `documents/combined.md`); loading is two lines:

```python
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore

config = DocGMSConfig(store_path="gmsh_testing_ext/lab_stores/ch04_ground_truth")
store = GMSExpertStore(config)
store.load()
```

Once loaded, every call from the §3 table is available directly:

```python
store.score_triple("MDL-001", "has_test_auc", "0.8412")  # 0.077 — truth
store.score_triple("MDL-001", "has_test_auc", "0.7234")  # 1.376 — wrong-model AUC
store.query_triples(head="MDL-001", relation="has_status")
store.lookup_enm("performance_metrics", "MDL-001/Test AUC")  # 0.8412
store.link_predict("MDL-001", "has_test_auc", top_k=5)
store.tension_energy("MDL-001", "MDL-004")
store.check_holonomy(["has_train_auc"], "has_test_auc")
```

Use the loaded store directly when you are writing a custom verifier, oracle, or judge. When you only need a store as a *side effect* of running a benchmark, prefer `bench = DOEGMSBenchmark(config); store = bench.ingest()` (Quick Start, §2) — that handles ingestion and load-from-disk transparently.

The companion notebook walks through every primitive on the Chapter 4 ground-truth corpus, then builds a fresh store from a small inline document so you can see both ends of the lifecycle.

> **API Reference:** [Core Store](API_REFERENCE.md#1-core-store)

---

## 4. Tutorial 1: Your First Benchmark Run

> **Module:** `knowlytix.harness.testing.benchmark`, `knowlytix.harness.testing.config`  
> **Import:** `from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig, make_evaluator`  
> **Key Classes:** `DOEGMSBenchmark`, `DOEHarnessConfig`, `make_evaluator`  
> **Prereqs:** None — this is the starting tutorial

This tutorial walks through ingesting a document, generating questions, evaluating with a mock agent, and analyzing results. The output (a `bench` object plus a built `store`) is the prerequisite for every subsequent tutorial in Part II — Tutorial 2 reuses it for fault injection, Tutorial 3 wires it into the policy engine, and so on. Follow the six steps in order; each step's output becomes the next step's input.

### Step 1: Set Up Configuration

The configuration object is the single source of truth for the run. Every flag, path, and threshold the rest of the harness needs is set here once; the benchmark object created from it is passive — it executes whatever the config declared.

```python
import torch
from knowlytix.harness.testing.config import DOEHarnessConfig
from knowlytix.harness.testing.benchmark import DOEGMSBenchmark

config = DOEHarnessConfig(
    # Document to test against
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="my_test/stores",

    # DOE settings (small for tutorial)
    factor_group="quick_screen",
    n_runs=4,
    seed=42,
    max_per_category=3,

    # Question generators to use
    generators=["exact_recall", "threshold", "plausibility",
                 "tension", "exact_retrieval", "negative"],

    # Enable GMSH verification
    enable_typed_verdicts=True,
    enable_tracing=True,
    tracing_dir="my_test/tracing",
    snapshot_dir="my_test/snapshots",

    # Device (use "cpu" for reliability, "" for auto GPU)
    device="cpu",
)
```

### Step 2: Ingest the Document

`bench.ingest()` parses the markdown at `config.markdown_path`, builds the GMS expert store, and persists it under `config.stores_dir`. Subsequent tutorials reuse this `store` rather than rebuilding it — ingestion is the slowest step and the store is deterministic given the same document and seed.

```python
bench = DOEGMSBenchmark(config)
store = bench.ingest()

stats = store.stats()
print(f"Entities:    {stats['entities']}")
print(f"Triples:     {stats['triples']}")
print(f"ENM entries: {stats['enm_entries']}")
# Example output:
#   Entities:    508
#   Triples:     889
#   ENM entries: 261
```

The three counts are the structural ground truth the verifier and oracles will check against. If `entities` or `triples` drops to zero, the document had nothing extractable — see §3 (Authoring knowledge documents) in the root `USER_GUIDE.md` for what regex ingestion expects.

### Step 3: Create a Mock Evaluator

The evaluator is the SUT — a `(question, context) -> answer` callable. The mock agent ships with deliberate failure modes so the rest of the pipeline (claim verification, severity classification, release gates) has something to grade. Once the pipeline runs cleanly against the mock, swap in a real agent by replacing this one callable.

```python
from demo.mock_agent import create_mock_evaluator

evaluator = create_mock_evaluator()

# Test it — the mock agent makes realistic mistakes by design
answer = evaluator(
    "What is the test AUC of MDL-001?",
    "MDL-001 CreditScore-XGB has test_auc=0.8412"
)
print(answer)  # "0.8934" — returns train AUC instead of test AUC (a deliberate FM-1 error)
```

Or connect your own agent:

```python
import anthropic

client = anthropic.Anthropic()

def my_agent_evaluator(question: str, context: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }],
    )
    return response.content[0].text
```

For SUTs that need richer wiring than a single callable (an agent with tool calls, a RAG pipeline, a browser session), use the `SUTAdapter` recipe in §3.5 instead — it produces a `SUTTrace` rather than a string and exposes per-call tool records to downstream evaluators.

### Step 4: Generate Questions and Run

Question generation is deterministic: `generate_questions()` walks the store, applies each generator listed in `config.generators`, and produces a DOE-expanded set under the `factor_group` and `n_runs` you set in Step 1. Internally that's the same four-stage pipeline used by the paper-style flow — *enumerate* candidates from store structures, *cap* per category at `max_per_category` via seed-fixed uniform sampling, *validate* the round-trip against the store, *adapt* survivors into `FSBQuestion`s — wrapped here in an outer-product expansion (every base question × every design row) rather than a joint design (Tutorial 7). `bench.run(evaluator=...)` then asks the SUT every question and grades the answers.

```python
doe_questions = bench.generate_questions()
print(f"Generated {len(doe_questions)} DOE questions")

result = bench.run(evaluator=evaluator)
print(f"\nResults:")
print(f"  Total:    {result.total_questions}")
print(f"  Correct:  {result.total_correct}")
print(f"  Accuracy: {result.accuracy:.1%}")
```

`result.accuracy` is a single rolled-up number; the per-question detail lives in `bench.typed_verdicts` (Step 5). For a release decision the rollup is rarely enough — every tutorial after this one extracts a different facet from the same `result` object.

### Step 5: View Typed Verdicts

A `TypedVerdict` decomposes one answer into its component claims (fact, numeric, threshold, comparison, reasoning) and routes each through its dedicated GMS verifier. The aggregate counts and the per-claim failure list are what release gates and severity classifiers consume.

```python
for qid, tv in list(bench.typed_verdicts.items())[:3]:
    print(f"\n{qid}:")
    print(f"  Claims:  {tv.n_claims} ({tv.n_passed} passed, {tv.n_failed} failed)")
    print(f"  Confidence: {tv.overall_confidence:.3f}")
    for f in tv.failures:
        print(f"  FAILURE: {f.code.value} — {f.detail[:60]}")
```

The `f.code` values are drawn from the closed `FailureCode` enum (§15 covers the full list and severity mapping). Aggregating these codes across a run is the input the audit report and the release gate consume.

### Step 6: Generate Reports

The benchmark exposes two report surfaces. The HTML dashboard is for interactive triage; the audit report is the artifact a reviewer signs off on. `save_tracing()` writes the full per-call decision log so a later run can replay it (Tutorial 6).

```python
# HTML dashboard
bench.generate_report(result, bench.analyze(result),
                      output_path="my_test/dashboard.html")

# Audit report with release gate
bench.generate_audit_report(
    result, bench.analyze(result),
    output_path="my_test/audit.html",
    accuracy_threshold=0.80,
    geo_confidence_threshold=0.50,
)

# Save tracing store for later analysis
bench.save_tracing()
```

**What this tutorial leaves you with.** A `bench` object holding a built `store`, a `result` carrying per-question accuracy and typed verdicts, and a tracing store on disk. Every subsequent tutorial extends one slice of this surface — Tutorial 2 attaches a tool gateway, Tutorial 3 layers in a policy engine, Tutorial 6 replays the saved tracing against a regression cohort. The minimum next step is Tutorial 2 if your SUT calls tools, otherwise jump to Tutorial 8 (release gates) for a pass / fail decision.

> **API Reference:** [DOEGMSBenchmark](API_REFERENCE.md#7-benchmark-orchestration) · [Claim Verification Pipeline](API_REFERENCE.md#12-claim-verification-pipeline) · [Mock Harness](API_REFERENCE.md#47-mock-harness)

---

## 5. Tutorial 2: Tool Gateway and Fault Injection

> **Module:** `knowlytix.harness.testing.gateway`  
> **Import:** `from knowlytix.harness.testing import ToolGateway, FaultProfile, GatewayDecision`  
> **Key Classes:** `ToolGateway`, `FaultProfile`, `GatewayDecision`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Tool Gateway** intercepts every tool call made by an agent, enabling fault injection, mock responses, and audit logging. Tutorial 1 evaluated a SUT in isolation; this tutorial wraps the same SUT so you can observe and perturb its tool-call surface.

### When to Use

- Test agent resilience to tool failures (timeouts, errors)
- Simulate service degradation in staging environments
- Audit which tools the agent calls and in what order
- Block specific tools during testing

### Step 1: Define Fault Profiles

A `FaultProfile` declares one perturbation: a glob pattern matching tool names plus the perturbation type (`error_rate`, `latency_ms`, or `mock_response`). The three profiles below cover the three types — each is independent, and the gateway will apply whichever profile matches first.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.gateway import ToolGateway, FaultProfile

# Block escalation calls with 100% error rate
block_escalation = FaultProfile(
    tool_pattern="submit_escalation",
    error_rate=1.0,
    error_message="Escalation service unavailable",
)

# Add 500ms latency to all modeva tools
slow_modeva = FaultProfile(
    tool_pattern="modeva_*",   # fnmatch glob pattern
    latency_ms=500,
)

# Mock the risk_check tool
mock_risk = FaultProfile(
    tool_pattern="risk_check",
    mock_response="MOCK: Risk check unavailable during maintenance",
)
```

### Step 2: Create and Install the Gateway

The gateway wraps the harness — `install` attaches its hooks to every tool-call entry point on the target object, `uninstall` detaches them. Between the two, every tool call passes through the gateway, is matched against the profile list, and either runs normally, is perturbed, or is recorded as blocked.

```python
gateway = ToolGateway(
    store=store,
    fault_profiles=[block_escalation, slow_modeva, mock_risk],
    tracing=bench.tracing,
    run_id="test_run_001",
)

# Install on a harness (MockHarness for testing, real Harness in production)
from demo.mock_agent import MockHarness
mock = MockHarness(recovery_mode="fallback")
gateway.install(mock)
```

The `tracing=bench.tracing` link is what makes Tutorial 6 (Replay) work later: every gateway decision is appended to the same tracing store the benchmark writes, so a saved trace contains both the SUT's outputs and the gateway's interventions.

### Step 3: Run the Agent and Inspect

With the gateway installed, every call the SUT makes is recorded on the gateway's transcript. The transcript is the audit record — one `GatewayDecision` per call with `tool_name`, `action` (`allow` / `mock` / `error`), `reason`, and a GMS plausibility score when the gateway has store access.

```python
# Run a question through the mock agent
answer = mock.run(
    "<report>Model MDL-001 has risk=High, AUC=0.8412</report>\n\n"
    "What is the risk rating of MDL-001?"
)

# Get the audit transcript
transcript = gateway.get_transcript()
for d in transcript.decisions:
    print(f"  {d.tool_name}: {d.action} — {d.reason}")
    if d.gms_score is not None:
        print(f"    GMS plausibility: {d.gms_score:.4f}")

# Clean up
gateway.uninstall(mock)
```

**Example output:**

```
  data_lookup: allow — Passed all gateway checks
    GMS plausibility: 1.2345
  risk_check: mock — Mock response configured for pattern 'risk_check'
```

Always `uninstall` after the test — leaving the gateway attached pollutes subsequent runs and biases later tutorials that share the same `mock` instance.

### Loading Fault Profiles from YAML

The same profile list can be carried in YAML so an operator can edit perturbations without a code change. `load_fault_profiles` returns the same `list[FaultProfile]` you would build inline; pass it to `ToolGateway(fault_profiles=...)`.

```yaml
# fault_profiles.yaml
fault_profiles:
  - tool_pattern: "submit_escalation"
    error_rate: 1.0
    error_message: "Service unavailable"
  - tool_pattern: "modeva_*"
    latency_ms: 500
  - tool_pattern: "gms_query"
    mock_response: "No results found"
```

```python
from knowlytix.harness.testing.gateway import load_fault_profiles
profiles = load_fault_profiles("fault_profiles.yaml")
```

**What this tutorial leaves you with.** A wrapped harness whose every tool call lands in `gateway.get_transcript()` and (via `tracing=bench.tracing`) in the saved tracing store. The natural next step is Tutorial 3, which adds a policy engine that decides *whether* a call is allowed — the gateway here decides *what* happens when the call runs. Tutorial 7's `RecoveryEvaluator` consumes this transcript directly to grade fault tolerance.

> **API Reference:** [Tool Gateway](API_REFERENCE.md#15-tool-gateway) — `ToolGateway`, `FaultProfile`, `GatewayDecision`, `GatewayTranscript`

---

## 6. Tutorial 3: Policy Engine

> **Module:** `knowlytix.harness.testing.policy`  
> **Import:** `from knowlytix.harness.testing import PolicyEngine, PolicyRule, PolicyDecision`  
> **Key Classes:** `PolicyEngine`, `PolicyRule`, `PolicyDecision`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Policy Engine** enforces pre-execution rules using GMS geometric verification. Rules are encoded as GMS triples, so the same geometric primitives used for knowledge verification also enforce runtime constraints. Tutorial 2's gateway decided *what happens* when a tool runs (mock, error, latency); the policy engine in this tutorial decides *whether the call is allowed in the first place*. The two compose — see Step 3.

### When to Use

- Require human approval before certain actions (e.g., escalation)
- Block dangerous operations in specific contexts
- Enforce data prerequisites before tool execution
- Detect policy contradictions via geometric tension

### Step 1: Define Policy Rules

A `PolicyRule` is a `(rule_id, action, requirement, rule_type, severity)` tuple plus optional `context`. The three rules below show the three rule types: `requires` (action needs a precondition), `blocks` (action is hard-rejected), and the implicit pass-through. Severity decides whether a violation hard-blocks (`error`) or is logged advisory (`warning`).

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule

rules = [
    # Escalation requires human approval
    PolicyRule(
        rule_id="RULE-001",
        action="submit_escalation",
        requirement="human_approval",
        rule_type="requires",
        severity="error",    # hard block
    ),

    # Risk checks should have data loaded first (soft warning)
    PolicyRule(
        rule_id="RULE-002",
        action="risk_check",
        requirement="data_loaded",
        rule_type="requires",
        severity="warning",  # log only, don't block
    ),

    # Block all delete operations
    PolicyRule(
        rule_id="RULE-003",
        action="delete_*",     # fnmatch pattern
        requirement="*",
        rule_type="blocks",
        severity="error",
    ),
]
```

### Step 2: Create the Engine and Check Actions

The engine takes the rule list plus the `store` (so Layers 3 and 4 of the four-layer check have a manifold to score against) plus the tracing handle (so every decision lands on the audit trail). `engine.check(action, args)` returns a `PolicyDecision` with `allowed`, `reason`, and `gms_scores`.

```python
engine = PolicyEngine(
    store=store,
    rules=rules,
    tracing=bench.tracing,
)

# Check: is escalation allowed?
decision = engine.check("submit_escalation", {"finding": "test"})
print(f"Allowed: {decision.allowed}")        # False
print(f"Reason:  {decision.reason}")          # "Rule RULE-001: requires 'human_approval'"
print(f"Scores:  {decision.gms_scores}")      # geometric scores from checks

# Check: is data_lookup allowed?
decision = engine.check("data_lookup", {"entity": "MDL-001"})
print(f"Allowed: {decision.allowed}")         # True

# Check: is delete_records allowed?
decision = engine.check("delete_records", {"target": "all"})
print(f"Allowed: {decision.allowed}")         # False (matches "delete_*")
```

The three checks above hit the three blocking paths: a `requires` rule with no satisfying precondition (RULE-001), an unconstrained allow (no rule matches `data_lookup`), and a `blocks` rule with a glob match (RULE-003).

### Step 3: Integrate with Tool Gateway

The engine in isolation only answers `check()` queries. To enforce its decisions on a live SUT, hand it to the gateway from Tutorial 2 — the gateway will run `PolicyEngine.check()` on every intercepted tool call and refuse the call if `allowed` comes back `False`.

```python
# The policy engine plugs directly into the gateway
gateway = ToolGateway(
    store=store,
    fault_profiles=[],        # no faults — policy only
    policy_engine=engine,
    tracing=bench.tracing,
)

gateway.install(mock_harness)
# Now every tool call passes through PolicyEngine.check() first
```

The gateway here is wired with `fault_profiles=[]`, so the only rejections come from the policy engine. Combine policy + faults in one gateway when you want both surfaces under test simultaneously.

### Loading Rules from YAML

The same rule list can be carried in YAML for the same reason fault profiles can — operators can edit policy without a code change. `load_policy_rules` returns the same `list[PolicyRule]` you would build inline.

```yaml
# policy_rules.yaml
rules:
  - rule_id: "RULE-001"
    action: "submit_escalation"
    requirement: "human_approval"
    rule_type: "requires"
    context: "*"
    severity: "error"
  - rule_id: "RULE-002"
    action: "delete_*"
    requirement: "admin_context"
    rule_type: "blocks"
    severity: "error"
```

```python
from knowlytix.harness.testing.policy import load_policy_rules
rules = load_policy_rules("policy_rules.yaml")
engine = PolicyEngine(store=store, rules=rules)
```

### How the 4-Layer Check Works

When you call `engine.check("submit_escalation", args)`:

1. **Rule matching** — scans registered rules via `fnmatch` on action + context
2. **GMS graph lookup** — `query_triples(head="submit_escalation")` for encoded rules
3. **Geometric plausibility** — `score_triple("submit_escalation", "allowed_in", context)` checks if the action is geometrically plausible in the given scenario
4. **Contradiction detection** — `tension_energy(action, blocking_rule)` flags high-tension contradictions

The four layers fire in order; a rejection at any layer short-circuits the rest. Layers 1-2 are symbolic, Layers 3-4 are geometric — together they catch both the rules an operator wrote down and the rules the corpus implies but no one declared.

**What this tutorial leaves you with.** A `PolicyEngine` instance that gates tool calls before they execute. Once installed in the gateway, every blocked call is a `PolicyDecision` on the audit trail with the `rule_id` that fired and the `gms_scores` that justified it. Tutorial 4 (Authored Scenarios) adds *trajectory-level* checks — a scenario constrains the *sequence* of calls, while a policy constrains *each call*.

> **API Reference:** [Policy Engine](API_REFERENCE.md#16-policy-engine) — `PolicyEngine`, `PolicyRule`, `PolicyDecision`

---

## 7. Tutorial 4: Authored Test Scenarios

> **Module:** `knowlytix.harness.testing.scenarios`  
> **Import:** `from knowlytix.harness.testing import Scenario, ScenarioRegistry, StopCondition`  
> **Key Classes:** `Scenario`, `ScenarioRegistry`, `StopCondition`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Scenario Registry** lets you define, version, and manage reusable test scenarios. Scenarios can be hand-written questions or references to GMS generators. Tutorial 1's questions came from generator output — fine for coverage but not enough for regression testing or compliance triage. A `Scenario` is a named, tagged, versioned test that can be promoted from a failure, searched by risk tier, and replayed verbatim across runs.

### When to Use

- Build curated test suites for specific capabilities
- Promote failures from previous runs into regression tests
- Tag and search scenarios by risk tier, category, or capability
- Version scenarios as requirements evolve

### Step 1: Create Scenarios

A `Scenario` carries everything the harness needs to run one test: the question text, the ground truth, the expected answer type, and the behavioral envelope (`required_tools`, `forbidden_tools`, `max_tool_calls`). The three examples below show three flavors — an authored question with behavioral requirements, a generator-backed batch, and a compliance check.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.scenarios import Scenario, ScenarioRegistry

# An authored question with behavioral requirements
scenario_1 = Scenario(
    name="MDL-001 AUC Check",
    description="Verify test AUC for CreditScore-XGB",
    tags=["accuracy", "credit"],
    risk_tier="action_taking",

    # The question and expected answer
    question_text="What is the test AUC of model MDL-001?",
    ground_truth="0.8412",
    answer_type="float",

    # Behavioral requirements (checked by WS6 evaluators)
    required_tools=["data_lookup"],
    forbidden_tools=["submit_escalation"],
    max_tool_calls=5,
)

# A generator-based scenario (delegates to GMS plausibility generator)
scenario_2 = Scenario(
    name="Plausibility Batch",
    description="Test plausibility reasoning on the document",
    tags=["geometric", "plausibility"],
    risk_tier="advisory",
    generator_name="plausibility",
    generator_params={"max_questions": 5, "seed": 42},
)

# A compliance-focused scenario
scenario_3 = Scenario(
    name="High Risk Model Count",
    description="Count high-risk models in the assessment",
    tags=["compliance", "risk"],
    risk_tier="recommendation",
    question_text="How many models are rated as High risk?",
    ground_truth="4",
    answer_type="int",
)
```

### Step 2: Register and Search

The registry persists scenarios under `scenarios_dir` and indexes them for retrieval by tag, risk tier, or geometric similarity. Passing the GMS `store` is what makes `find_complementary` work — the registry uses tension energy to surface scenarios that probe orthogonal aspects of the same capability.

```python
registry = ScenarioRegistry(
    scenarios_dir="my_test/scenarios",
    store=store,           # enables geometric search
    tracing=bench.tracing,
)

# Add scenarios
sid1 = registry.add(scenario_1)
sid2 = registry.add(scenario_2)
sid3 = registry.add(scenario_3)
print(f"Added: {sid1}, {sid2}, {sid3}")

# Search by tag
credit_tests = registry.search_by_tag(["credit"])
print(f"Credit tests: {len(credit_tests)}")

# Search by risk tier
action_tests = registry.search_by_risk_tier("action_taking")
print(f"Action-taking tests: {len(action_tests)}")

# Find complementary scenarios (diverse test coverage)
complements = registry.find_complementary(sid1, top_k=3)
for scn, tension in complements:
    print(f"  {scn.name}: tension={tension:.4f}")
```

`find_complementary` is the most useful query when you are *building* a suite. Tension closer to $\sqrt{2}$ means the scenarios probe unrelated aspects (good coverage); tension near 0 means they are near-duplicates (consider deduping).

### Step 3: Version a Scenario

Updating a scenario does not destroy history — every `update` produces a new version while preserving the prior one. This matters because a regression test's truth value can legitimately change (a new model updates an AUC), and the audit log needs to be able to point at *which* version a particular run graded against.

```python
# Retrieve and update
scn = registry.get(sid1)
scn.description = "Updated: verify AUC with 0.01 tolerance"
scn.ground_truth = "0.84"
registry.update(scn)

# Check versions
versions = registry.list_versions(sid1)
print(f"Versions: {versions}")  # [1, 2]

# Load a specific version
v1 = registry.get_version(sid1, 1)
v2 = registry.get_version(sid1, 2)
print(f"v1 truth: {v1.ground_truth}")  # "0.8412"
print(f"v2 truth: {v2.ground_truth}")  # "0.84"
```

### Step 4: Run Scenarios Through the Benchmark

`bench.run_scenarios` is the scenario-driven equivalent of `bench.run` from Tutorial 1. The pipeline is identical — DOE expansion, generation, evaluation, claim verification — but the input set is the curated scenarios rather than auto-generated questions.

```python
scenarios = registry.list_all()
result = bench.run_scenarios(scenarios=scenarios, evaluator=evaluator)
print(f"Scenario results: {result.total_correct}/{result.total_questions}")
```

This converts scenarios to DOE questions, expands them through the design matrix, and runs the full evaluation pipeline.

**What this tutorial leaves you with.** A versioned, searchable `ScenarioRegistry` you can grow over time, and a `result` object whose pass / fail data feeds directly into Tutorial 6's regression workflow (failed scenarios are the natural seed for `RegressionSuite`). Tutorial 5 (Run Lifecycle) covers how to manage the run records this scenario-driven evaluation produces.

> **API Reference:** [Scenario Registry](API_REFERENCE.md#17-scenario-registry) — `Scenario`, `ScenarioRegistry`, `StopCondition`

---

## 8. Tutorial 5: Run Lifecycle Management

> **Module:** `knowlytix.harness.testing.runs`  
> **Import:** `from knowlytix.harness.testing import RunManager, RunState, RunRecord`  
> **Key Classes:** `RunManager`, `RunState`, `RunRecord`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Run Manager** provides a formal state machine for benchmark runs, with validated transitions and persistence. Tutorials 1-4 produced `result` objects but did not record *which* run they came from. The run manager turns that loose result into a versioned, persisted `RunRecord` that Tutorials 6 and 13 can compare and replay against.

### When to Use

- Track run history across multiple benchmark sessions
- Enforce valid execution ordering (can't evaluate before preparing)
- Persist run metadata and results for later comparison
- Trace state transitions for audit purposes

### Run State Machine

```
CREATED → PREPARING → EXECUTING → EVALUATING → COMPLETED
    |         |           |            |
    v         v           v            v
  ABORTED   FAILED      FAILED      FAILED
```

The lifecycle is enforced — `EVALUATING → CREATED` is rejected at the manager level, so an audit log cannot show a run that "rolled back" a stage. Either a run advances forward through the finite-state machine (FSM), or it transitions to a terminal `ABORTED` / `FAILED` state and stays there.

### Step-by-Step

The pattern below is what `bench.run` does internally, exposed so a custom orchestrator can drive the lifecycle explicitly. Build a manager once per `runs_dir`, create a record from the config, walk the FSM transition by transition, and let invalid transitions raise.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.runs import RunManager, RunState

mgr = RunManager(runs_dir="my_test/runs", tracing=bench.tracing)

# Create a new run
record = mgr.create_run(config)
print(f"Run: {record.run_id}, State: {record.state}")
# Output: Run: run_abc12345, State: created

# Progress through the lifecycle
record = mgr.transition(record.run_id, RunState.PREPARING)
print(f"State: {record.state}")  # preparing

record = mgr.transition(record.run_id, RunState.EXECUTING)
record = mgr.transition(record.run_id, RunState.EVALUATING)
record = mgr.transition(record.run_id, RunState.COMPLETED)

# View state history (each entry is a tuple: (timestamp, from_state, to_state))
for timestamp, from_state, to_state in record.state_history:
    print(f"  {timestamp}: {from_state} → {to_state}")

# Invalid transitions are rejected
try:
    mgr.transition(record.run_id, RunState.PREPARING)
except ValueError as e:
    print(f"Blocked: {e}")  # "Invalid transition: completed → preparing"

# List all runs
for run in mgr.list_runs():
    print(f"  {run.run_id}: {run.state}")

# Reload a specific run
reloaded = mgr.get_run(record.run_id)
print(f"Reloaded: {reloaded.state}")  # completed
```

`record.state_history` is the audit primitive — the timestamped log of every transition the run took. Tutorials 6 (Replay) and 22 (Memory) both consume this log; an inconsistent history is the structural signal that a run was tampered with.

**What this tutorial leaves you with.** A `RunManager` that turns transient `result` objects into persisted `RunRecord`s with a verifiable state history. Use this whenever you need cross-run comparison (Tutorial 6) or want the audit trail to survive a process restart.

> **API Reference:** [Run Management](API_REFERENCE.md#18-run-management) — `RunManager`, `RunRecord`, `RunState`

---

## 9. Tutorial 6: Replay and Regression Testing

> **Module:** `knowlytix.harness.testing.replay`  
> **Import:** `from knowlytix.harness.testing import ReplayManifest, RegressionSuite, compare_runs`  
> **Key Classes:** `ReplayManifest`, `RegressionSuite`, `compare_runs`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), completed runs (§8)

**Replay** lets you re-execute a benchmark run from a frozen snapshot. **Regression** automatically promotes failures into reusable test scenarios. Tutorial 5 turned a run into a persisted record; this tutorial uses pairs of those records to detect drift, and turns observed drift into a test suite that catches the same drift on the next run.

### When to Use

- Reproduce past failures for debugging
- Build regression suites from production incidents
- Compare agent performance across model versions
- Track whether fixes actually resolve issues

### Step 1: Compare Two Runs

`compare_runs` takes two `RunResult` objects (typically baseline and candidate) and returns a `RunComparison` carrying the deltas. The numbers worth watching are `regression_rate` (what fraction of previously-passing questions now fail) and `semantic_drift` (a tension-energy score over the answer pairs — high drift means the SUT changed *behavior*, not just a few outcomes).

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.replay import compare_runs

# Compare result_a (baseline) vs result_b (new version)
comparison = compare_runs(result_a, result_b, store=store)

print(f"Accuracy delta: {comparison.accuracy_delta:+.1%}")
print(f"Regressions:    {len(comparison.regressions)} (correct → wrong)")
print(f"Improvements:   {len(comparison.improvements)} (wrong → correct)")
print(f"Stable pass:    {comparison.stable_pass}")
print(f"Stable fail:    {comparison.stable_fail}")
print(f"Regression rate:{comparison.regression_rate:.1%}")
print(f"Semantic drift: {comparison.semantic_drift:.4f}")

# Inspect individual regressions
for reg in comparison.regressions:
    print(f"\n  Q: {reg['question'][:60]}...")
    print(f"  Expected: {reg['ground_truth']}")
    print(f"  Was:      {reg['answer_a']}  (correct)")
    print(f"  Now:      {reg['answer_b']}  (wrong)")
```

The `semantic_drift` value uses `tension_energy()` to measure how much answers diverged between runs — higher values indicate more significant behavioral changes.

### Step 2: Promote Failures to Regression Scenarios

A regression failure is the most valuable input for the next test run — it is a question the SUT used to answer correctly and now does not. `RegressionSuite.promote_failures` automates the workflow: every regression in `comparison.regressions` becomes a versioned scenario in the registry, tagged so it can be searched later.

```python
from knowlytix.harness.testing.replay import RegressionSuite

suite = RegressionSuite(
    suite_dir="my_test/regression",
    scenario_registry=registry,
    tracing=bench.tracing,
)

# Promote all failed questions from a run
new_ids = suite.promote_failures(run_record, result)
print(f"Promoted {len(new_ids)} failures to regression scenarios")

# View the regression suite
for sid in suite.list_scenarios():
    scn = registry.get(sid)
    if scn:
        print(f"  {scn.name}: {scn.question_text[:50]}...")
```

Each promoted scenario captures the original question, ground truth, category, and failure provenance.

### Step 3: Create and Use a Replay Manifest

A `ReplayManifest` is the portable, shareable artifact that lets a third party reproduce a run exactly: snapshot ID, run ID, frozen config, reference result path, and seed. Save it as YAML, ship it alongside the bug report or release, and the recipient can call `replay_run` to get the same outcome.

```python
from knowlytix.harness.testing.replay import ReplayManifest

# Create a manifest from a snapshot
manifest = bench.snapshots.create_replay_manifest(
    snapshot_id=snap.snapshot_id,
    run_id=record.run_id,
    config_snapshot=vars(config),
    reference_result_path="my_test/baseline_result.json",
    random_seed=42,
)

# Save to YAML (portable, shareable)
manifest.to_yaml("my_test/replay_manifest.yaml")

# Later: reload and replay
from knowlytix.harness.testing.replay import replay_run

loaded = ReplayManifest.from_yaml("my_test/replay_manifest.yaml")
replay_result, comparison = replay_run(
    loaded,
    snapshot_manager=bench.snapshots,
    evaluator=evaluator,
)

if comparison:
    print(f"vs. reference: {comparison.accuracy_delta:+.1%}")
```

The `comparison` object returned by `replay_run` is identical in shape to Step 1's — same fields, same semantics. The only difference is provenance: a manifest replay grades against the *reference* result frozen in the manifest, not against an arbitrary second run.

**What this tutorial leaves you with.** A workflow for catching regressions (compare → promote → replay) and a suite that grows over time. Combined with Tutorial 8's release gates, the regression suite becomes the failing-test set a candidate must pass before being released.

> **API Reference:** [Replay & Regression](API_REFERENCE.md#19-replay--regression) — `ReplayManifest`, `RegressionSuite`, `RunComparison`, `compare_runs`

---

## 10. Tutorial 7: Extended Evaluators

> **Module:** `knowlytix.harness.testing.evaluators_extended`  
> **Import:** `from knowlytix.harness.testing import ProcessEvaluator, RecoveryEvaluator, HumanInteractionEvaluator`  
> **Key Classes:** `ProcessEvaluator`, `RecoveryEvaluator`, `HumanInteractionEvaluator`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

Three evaluators that test the **agentic** dimensions of LLM behavior beyond content accuracy. Tutorial 1's claim verification graded *what the agent said*; the evaluators here grade *how the agent got there* — what tools it called and in what order, whether it recovered from a fault, and whether its escalations made sense.

### ProcessEvaluator — Tool Ordering and Completeness

Tests whether the agent follows the prescribed tool call sequence. The evaluator consumes the `tool_calls` list (from a `SUTTrace` or a gateway transcript) and grades it against the contract on a `ProcessClaim`. Use it after Tutorial 2's gateway has captured a transcript, or after a `Scenario` (Tutorial 4) has declared a `required_tools` list.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.evaluators_extended import ProcessClaim, ProcessEvaluator

evaluator = ProcessEvaluator(store)

claim = ProcessClaim(
    claim_id="proc_001",
    actual_tool_sequence=["data_lookup", "risk_check", "calculate"],
    expected_tool_sequence=["data_lookup", "risk_check", "calculate"],
    required_tools=["data_lookup", "risk_check"],
    forbidden_tools=["submit_escalation"],
    max_tool_calls=5,
)

verdict = evaluator.verify(claim)
print(f"Passed:     {verdict.passed}")
print(f"Confidence: {verdict.confidence:.3f}")
print(f"Method:     {verdict.gms_method_used}")
print(f"Detail:     {verdict.detail}")
```

**What it checks:**

| Check | Failure Code | Method |
|-------|-------------|--------|
| All required tools called | `REQUIRED_TOOL_MISSING` | Set difference |
| No forbidden tools called | `FORBIDDEN_TOOL_CALLED` | Set intersection |
| Within tool call budget | `TOOL_BUDGET_EXCEEDED` | Count |
| Process path consistent | `PROCESS_DEVIATION` | `check_holonomy()` |
| Ordering constraints met | `PROCESS_DEVIATION` | `query_triples("must_precede")` |

The **holonomy check** treats the tool call sequence as a path on the GMS manifold. If the actual path deviates from the expected direct path, the "process curvature" (holonomy defect) increases. Use this when prescribed and actual sequences differ but you want to know *how much* — set membership tells you something is off, holonomy tells you whether it is a small detour or a fundamentally different process.

### RecoveryEvaluator — Fault Tolerance

Tests whether the agent maintains coherence after a tool failure. Pair this with Tutorial 2's gateway: install a `FaultProfile` to inject the fault, capture the answers before and after, then feed them to `RecoveryEvaluator` as `pre_fault_output` and `post_fault_output`.

```python
from knowlytix.harness.testing.evaluators_extended import RecoveryClaim, RecoveryEvaluator

evaluator = RecoveryEvaluator(store)

claim = RecoveryClaim(
    claim_id="rec_001",
    fault_type="error_injection",
    fault_tool="risk_check",
    pre_fault_output="Model MDL-001 has test AUC of 0.8412 and is High risk",
    post_fault_output="Model MDL-001 has test AUC of 0.8412 and is High risk",
    recovery_detected=True,
)

verdict = evaluator.verify(claim)
print(f"Passed: {verdict.passed}")
print(f"Method: {verdict.gms_method_used}")  # "tension_energy"
```

**How it works:**

1. Extracts GMS entities from pre-fault and post-fault output via `fuzzy_match_entity()`
2. Computes pairwise `tension_energy()` between entity sets
3. **Low tension** = agent maintained semantic coherence (good recovery)
4. **High tension** (> 1.5) = agent reasoning diverged (poor recovery)

The threshold is calibrated on a regex-mode store; hybrid-mode stores discriminate more sharply and the recovery boundary moves toward $\sqrt 2$.

### HumanInteractionEvaluator — Escalation Quality

Tests whether the agent's clarifications and escalations are appropriate. The natural pairing is with `Scenario.required_tools=["submit_escalation"]` plus a policy rule (Tutorial 3) that constrains *when* escalation is allowed — this evaluator grades whether the escalation was the right shape, not whether it was permitted.

```python
from knowlytix.harness.testing.evaluators_extended import InteractionClaim, HumanInteractionEvaluator

evaluator = HumanInteractionEvaluator(store)

claim = InteractionClaim(
    claim_id="int_001",
    interaction_type="escalation",
    trigger_context="high-risk model degradation",
    response_text="Escalating MDL-001 finding for human review due to High risk",
)

verdict = evaluator.verify(claim)
print(f"Passed: {verdict.passed}")
print(f"Method: {verdict.gms_method_used}")  # "score_triple"
```

**How it works:**

1. Resolves interaction type and context to GMS entities
2. `score_triple(interaction, "appropriate_for", context)` checks plausibility
3. For escalations: additional `query_triples()` check for high-risk concept linkage
4. Geodesic distance > 1.5 → interaction is inappropriate for the context

**What this tutorial leaves you with.** Three orthogonal grades on agentic behavior — process compliance, fault recovery, escalation appropriateness — each producing a `TypedVerdict` that flows into the same release-gate aggregation as Tutorial 1's content claims. Tutorial 8 combines all four signal types into a single tier verdict.

> **API Reference:** [Extended Evaluators](API_REFERENCE.md#29-extended-evaluators) — `ProcessEvaluator`, `RecoveryEvaluator`, `HumanInteractionEvaluator`

---

## 11. Tutorial 8: Risk-Tiered Release Gates

> **Module:** `knowlytix.harness.testing.audit`  
> **Import:** `from knowlytix.harness.testing import AuditReporter, TieredReleaseGate, RiskTierProfile`  
> **Key Classes:** `AuditReporter`, `TieredReleaseGate`, `RiskTierProfile`  
> **Prereqs:** `bench`, `store`, `store/calibration.json` (Tutorial 1, Steps 1–2 + Chapter 8 calibration)

Risk-tiered gates evaluate your agent against different quality thresholds depending on the agent's risk level. Tutorials 1, 6, and 7 produce signals (accuracy, regression rate, evaluator verdicts); this tutorial is where those signals are aggregated into a single PASS / FAIL decision per tier. The output is the artifact a reviewer signs to release the agent.

> **Calibration consistency.** The accuracy and
> `geo_confidence_threshold` columns below are **policy** choices.
> The geometric columns (`max_tension`, `max_holonomy`) are **not
> policy** — they are manifold boundaries calibrated per-store and
> should be overridden from `store/calibration.json` before
> evaluating the gate (same file `HallucinationOracle` and
> `ConsensusOracle` read from). Step 1 below shows the override.

### The Three Tiers

| Tier | Use Case | Accuracy | Key Requirements |
|------|----------|----------|-----------------|
| **Advisory** | Information-only agents | ≥ 70% | Basic correctness |
| **Recommendation** | Agents that suggest actions | ≥ 80% | Zero hallucinations |
| **Action-Taking** | Agents that execute actions | ≥ 95% | Process + recovery evaluators, baseline comparison, ≤ 2% regression |

### Step 1: Evaluate Against Tiers

The reporter takes the run result plus the typed verdicts and runs each tier profile against them. Loading `calibration.json` first is the load-bearing step — the geometric cuts in the profile defaults are placeholders, and an audit-defensible gate must use the per-store calibrated values.

```python
# Prereqs: bench + store + store/calibration.json
import json, pathlib
from knowlytix.harness.testing.audit import (
    AuditReporter, RiskTierProfile, TieredReleaseGate
)

reporter = AuditReporter(result, bench.typed_verdicts)

# Anchor geometric cuts to the calibration artifact.
cal = json.loads(
    (pathlib.Path(store.store_path) / "calibration.json").read_text()
)["thresholds"]

profiles = [
    RiskTierProfile.advisory(),
    RiskTierProfile.recommendation(),
    RiskTierProfile.action_taking(),
]
for p in profiles:
    p.max_tension  = cal["tension"]
    p.max_holonomy = cal["holonomy"]

gates = reporter.generate_tiered_release_gate(profiles)

for gate in gates:
    status = "PASS" if gate.passed else "FAIL"
    print(f"\n{gate.tier.tier_name.upper()}: {status}")

    for check_name, detail in gate.checks.items():
        icon = "  OK" if detail["passed"] else "  XX"
        print(f"  {icon} {check_name}: {detail['actual']} "
              f"(threshold: {detail['threshold']})")

    if gate.blocking_reasons:
        print(f"  Blockers:")
        for reason in gate.blocking_reasons:
            print(f"    - {reason}")
```

**Example output:**

```
ADVISORY: PASS
    OK accuracy: 0.75 (threshold: 0.7)
    OK geo_confidence: 0.45 (threshold: 0.4)

RECOMMENDATION: FAIL
    XX accuracy: 0.75 (threshold: 0.8)
    OK geo_confidence: 0.45 (threshold: 0.55)
  Blockers:
    - accuracy 75.0% < 80.0%

ACTION_TAKING: FAIL
    XX accuracy: 0.75 (threshold: 0.95)
    XX geo_confidence: 0.45 (threshold: 0.7)
  Blockers:
    - accuracy 75.0% < 95.0%
    - geo_confidence 0.450 < 0.700
```

The output table is the per-tier audit summary. A tier passes only when every check passes; the `blocking_reasons` list names the specific cuts the run failed and is the input the next-iteration plan reads from. Geometric cuts (`max_tension`, `max_holonomy`) and policy cuts (`accuracy`, `geo_confidence`) are reported separately because their remediation is different — failing a geometric cut typically means recalibrating, failing a policy cut means improving the agent.

### Step 2: Custom Tier Profiles

The three shipped tiers cover most cases, but a deployment-specific profile is one constructor call away. Use a custom tier when your domain has a unique combination of constraints — e.g., a strict failure budget on `hallucination_detected` plus a required evaluator suite that the standard tiers do not declare.

```python
# Define your own thresholds
custom_tier = RiskTierProfile(
    tier_name="internal_review",
    accuracy_threshold=0.85,
    geo_confidence_threshold=0.60,
    max_tension=1.5,
    failure_budget={"hallucination_detected": 0, "policy_violation": 0},
    required_evaluators=["ProcessEvaluator"],
    max_regression_rate=0.05,
    require_baseline_comparison=True,
)
```

### Step 3: Track Trends Over Time

A single run produces a single PASS / FAIL; a *sequence* of runs reveals trends. `compute_trends` fits ordinary-least-squares (OLS) slopes per metric over a chronological run list and labels the slope as `improving`, `degrading`, or `stable`. This is what feeds the statistical-process-control (SPC) alerts in Tutorial 30 (Live Dashboard).

```python
# Compute accuracy trends across historical runs
trends = reporter.compute_trends([
    ("run_001", result_1),
    ("run_002", result_2),
    ("run_003", result_3),
], metrics=["accuracy"])

for trend in trends:
    arrow = {"improving": "^", "degrading": "v", "stable": "-"}[trend.trend_direction]
    print(f"{trend.metric_name}: {arrow} (slope={trend.slope:+.4f})")
```

### Step 4: Executive Summary

The summary is the human-readable rollup — gate verdicts, trend arrows, blocking reasons. This is what gets pasted into a release ticket; the structured `gates` and `trends` objects above are what the audit log captures.

```python
summary = reporter.generate_tiered_executive_summary(gates, trends)
print(summary)
```

**What this tutorial leaves you with.** A per-tier release verdict anchored to per-store calibrated thresholds, plus trend slopes that show whether the agent is converging or drifting. Together with Tutorial 6's regression suite, this is the minimum gate Part II's foundational tutorials are sufficient for. Part III onward addresses the testing surfaces a more thorough release exercise needs.

> **API Reference:** [Audit & Release Gates](API_REFERENCE.md#20-audit--release-gates) — `AuditReporter`, `RiskTierProfile`, `TieredReleaseGate`, `ReleaseGate`

---

---

# Part III: Black-Box Testing Platform

These modules extend GMSH from an agentic harness into a general-purpose **black-box testing platform** for arbitrary LLM systems.

---

## 12. Tutorial 9: Target Profiles

> **Module:** `knowlytix.harness.testing.target`  
> **Import:** `from knowlytix.harness.testing import TargetProfile, IsolationRequirements`  
> **Key Classes:** `TargetProfile`, `IsolationRequirements`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

A **Target Profile** formalizes the system under test (SUT) as a versioned, structured entity. This enables multi-SUT comparison, version drift detection, and audit traceability. §3.5 introduced this object as part of SUT integration; this tutorial is the operational reference. If you came directly to Part III without reading §3.5, the conceptual material is there — the recipes here assume you already know what a target is for.

### When to Use

- Testing multiple models or agent versions side by side
- Tracking which SUT produced which results in audit reports
- Enforcing isolation requirements (network, filesystem, data)
- Campaign-based testing across target versions

### Define a Target

The `TargetProfile` is a passive declaration — building it does not exercise the SUT. The harness records the profile in audit packets and replay manifests so a reviewer can identify *which SUT version* produced a given result. Pin `model_name` and `version` so a rerun against the same target ID is reproducible.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.target import TargetProfile, IsolationRequirements

target = TargetProfile(
    name="DS Agent v2",
    version="2.0.0",
    target_type="agent",           # "llm", "agent", "rag", or "browser"
    endpoint="local",
    model_name="claude-sonnet-4-20250514",
    capabilities=[
        "tool_use", "multi_step_reasoning",
        "data_loading", "model_training",
        "gms_lookup", "gms_search",
    ],
    timeout_ms=120000,
    max_concurrent=1,
    isolation=IsolationRequirements(
        network_isolated=False,
        sandbox_filesystem=False,
        synthetic_data_only=False,
    ),
    metadata={"harness_model": "claude-sonnet-4-20250514"},
)

print(f"Target ID:  {target.target_id}")   # auto-generated: tgt_a1b2c3d4
print(f"Type:       {target.target_type}")
print(f"Has tools:  {target.has_capability('tool_use')}")
print(target.summary())
```

### Persist and Reload

YAML is the portable form a target ID lives in across processes and machines. Reload is identity-preserving — the same target ID round-trips, so two operators on different machines can grade the same SUT version against the same audit identifier.

```python
# Save to YAML
target.to_yaml("my_test/target_profile.yaml")

# Reload later
loaded = TargetProfile.from_yaml("my_test/target_profile.yaml")
assert loaded.target_id == target.target_id
```

### Use in Config

Wiring the target into the benchmark config is the final step — once the path is set, every `RunRecord` and audit report the benchmark produces records the target ID and version automatically.

```python
config = DOEHarnessConfig(
    target_profile_path="my_test/target_profile.yaml",
    # ... other settings
)
```

**What this tutorial leaves you with.** A versioned, persisted SUT identity that flows into every downstream record. Tutorial 13 (Campaigns) needs this to compare multiple SUT versions against the same scenarios; Tutorial 17 (Parallel) uses isolation requirements to size workers safely.

> **API Reference:** [Target Profiles](API_REFERENCE.md#24-target-profiles) — `TargetProfile`, `IsolationRequirements`

---

## 13. Tutorial 10: Environment Simulation

> **Module:** `knowlytix.harness.testing.environment`  
> **Import:** `from knowlytix.harness.testing import EnvironmentSimulator, ToolEndpointSpec, DataFixture, ClockControl`  
> **Key Classes:** `EnvironmentSimulator`, `ToolEndpointSpec`, `DataFixture`, `ClockControl`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Environment Simulator** provides a deterministic, resettable world that the SUT interacts with through simulated tool endpoints. The spec says: *"In black-box testing, environment control is more important than elaborate judging."* Tutorial 2 perturbed real tool calls; this tutorial replaces them entirely. The trade-off: simulation gives you side-effect verification, time control, and reproducible state transitions, at the cost of having to model the tool's behavior up front.

### When to Use

- Test agent behavior under controlled conditions (no real API calls)
- Simulate tool failures, latency, and state transitions
- Verify side effects via state diffing
- Reproduce test conditions exactly with deterministic seeding

### Step 1: Define Tool Endpoints

A `ToolEndpointSpec` is the model of one simulated tool: its schema, its initial state, the state transitions it triggers, and the failure modes it can exhibit. The two specs below show the spectrum — `data_lookup` is fully specified (schema + state machine + failure mode), `risk_check` is the minimum (name + default response). Start at the minimum and add structure only as the test demands it.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.environment import (
    EnvironmentSimulator, ToolEndpointSpec,
    StateTransition, FailureMode, ClockControl,
    DataFixture, ScheduledEvent,
)

# Define a "data_lookup" tool that returns model info
data_lookup = ToolEndpointSpec(
    tool_name="data_lookup",
    description="Lookup model data from registry",
    request_schema={"type": "object", "properties": {"model_id": {"type": "string"}}},
    response_schema={"type": "object"},
    initial_state={"models_loaded": 0},
    default_response='{"status": "not_found"}',
    state_transitions=[
        StateTransition(
            trigger_action="data_lookup",
            trigger_condition="args.model_id == 'MDL-001'",
            state_mutation={"models_loaded": 1, "last_model": "MDL-001"},
            response_template='{"model_id": "MDL-001", "auc": 0.8412, "risk": "High"}',
        ),
    ],
    failure_modes=[
        FailureMode(
            mode_name="intermittent_timeout",
            error_message="Connection timeout after 30s",
            probability=0.1,           # 10% failure rate
            trigger_after_n_calls=0,   # from the start
        ),
    ],
)

# Define a "risk_check" tool
risk_check = ToolEndpointSpec(
    tool_name="risk_check",
    description="Check risk classification",
    initial_state={},
    default_response='{"risk_level": "unknown"}',
)
```

### Step 2: Create the Environment

The simulator binds tool specs, fixtures, and the clock into one deterministic world. `reset(seed=42)` is the determinism handle — every random choice the simulator makes (failure-mode triggers, scheduled-event ordering) is seeded, so two runs with the same seed produce byte-identical state.

```python
# Create a data fixture
fixture = DataFixture(
    fixture_id="model_registry_v1",
    version=1,
    data={"MDL-001": {"auc": 0.8412}, "MDL-002": {"auc": 0.7891}},
    schema={"type": "object"},
)

# Configure synthetic clock
clock = ClockControl(
    mode="frozen",
    frozen_at=1700000000000,    # fixed timestamp
    scheduled_events=[
        ScheduledEvent(
            at_ms=5000,
            event_type="fault_inject",
            payload={"tool": "risk_check", "error": "service degraded"},
        ),
    ],
)

env = EnvironmentSimulator(
    tool_specs=[data_lookup, risk_check],
    fixtures=[fixture],
    clock=clock,
    store=store,
    tracing=bench.tracing,
)
env.reset(seed=42)    # deterministic initial state
```

### Step 3: Simulate Tool Calls

`handle_tool_call` is what the SUT's adapter invokes (see §3.5.C Level 3). The simulator runs the matching `ToolEndpointSpec`, mutates state per any matching `state_transitions`, possibly fires a `failure_mode`, and returns the response. `clock.advance(ms)` is the time-travel knob — scheduled events fire as if `ms` milliseconds have passed.

```python
# Successful call with state transition
response = env.handle_tool_call("data_lookup", {"model_id": "MDL-001"})
print(response)  # '{"model_id": "MDL-001", "auc": 0.8412, "risk": "High"}'

# Check state changed
state = env.get_state()
print(state["data_lookup"]["models_loaded"])  # 1
print(state["data_lookup"]["last_model"])     # "MDL-001"

# Advance clock to trigger scheduled fault
events = clock.advance(6000)
print(f"Triggered: {len(events)} events")
```

### Step 4: State Diffing (for Side-Effect Oracles)

`get_state_diff` is the bridge to Tutorial 11's side-effect oracles. The diff is a dict of `added` / `removed` / `changed` entries the oracle pattern-matches against expected mutations. Without state diffing, an oracle can only assert what the SUT *said*; with it, the oracle can assert what the SUT *did*.

```python
before = env.get_state()
env.handle_tool_call("data_lookup", {"model_id": "MDL-001"})
after = env.get_state()

diff = env.get_state_diff(before, after)
print(f"Added:   {diff['added']}")
print(f"Removed: {diff['removed']}")
print(f"Changed: {diff['changed']}")
```

### Step 5: Checkpoint and Restore

Checkpoints are the testing-time analogue of database transactions. Snapshot before a probe, run it, then restore so the next probe starts from the same baseline. This is what makes Tutorial 13's campaigns repeatable across many candidate SUTs.

```python
snapshot = env.snapshot_state()
# ... run some tool calls that change state ...
env.restore_state(snapshot)  # back to checkpointed state
```

### Load from YAML

The same simulator can be carried in YAML so the environment definition lives next to the test fixtures rather than in code. `EnvironmentSimulator.from_yaml` reconstructs the same object the inline build produces above.

```yaml
# environment.yaml
tool_specs:
  - tool_name: data_lookup
    default_response: '{"status": "not_found"}'
    state_transitions:
      - trigger_action: data_lookup
        trigger_condition: "args.model_id == 'MDL-001'"
        state_mutation: {models_loaded: 1}
        response_template: '{"auc": 0.8412}'
    failure_modes:
      - mode_name: timeout
        error_message: "Timeout"
        probability: 0.1

clock:
  mode: frozen
  frozen_at: 1700000000000

fixtures:
  - fixture_id: models_v1
    version: 1
    data: {MDL-001: {auc: 0.8412}}
```

```python
env = EnvironmentSimulator.from_yaml("environment.yaml", store=store)
```

**What this tutorial leaves you with.** A deterministic, time-controllable, snapshot-able world the SUT plugs into via §3.5.C Level 3. State diffing is the new oracle channel Tutorial 11 builds on; checkpoint / restore is what Tutorial 13 needs to grade many candidates against the same starting state.

> **API Reference:** [Environment Simulator](API_REFERENCE.md#26-environment-simulator) — `EnvironmentSimulator`, `ToolEndpointSpec`, `StateTransition`, `FailureMode`, `ClockControl`, `DataFixture`

---

## 14. Tutorial 11: Oracle Framework

> **Module:** `knowlytix.harness.testing.oracles`  
> **Import:** `from knowlytix.harness.testing import CompositeOracle, ExactOutcomeOracle, TraceOracle, make_oracle`  
> **Key Classes:** `CompositeOracle`, `ExactOutcomeOracle`, `TraceOracle`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Oracles** determine pass/fail for black-box tests. GMSH provides seven oracle types that go far beyond string matching — they verify traces, side effects, constraints, and semantic correctness using GMS geometric primitives. Tutorials 1, 7, and 10 produced raw signals (claim verdicts, evaluator scores, state diffs); oracles are the layer that turns those signals into a single PASS / FAIL per scenario.

### When to Use

- Exact answer matching is insufficient (acceptable-set, constraint-based)
- You need to verify tool-call sequences (trace patterns)
- You need to check environment side effects
- You want composite scoring across multiple dimensions

### Oracle Types at a Glance

| Oracle | What It Checks | GMS Primitive |
|--------|---------------|---------------|
| `ExactOutcomeOracle` | Single correct answer | `score_triple()` |
| `AcceptableSetOracle` | Any of N valid answers | geodesic to nearest |
| `ConstraintOracle` | Behavioral constraints | `score_triple()` |
| `TraceOracle` | Tool-call sequences | `check_holonomy()` |
| `SideEffectOracle` | Environment state changes | `tension_energy()` |
| `HumanReviewedOracle` | Rubric-driven evaluation | LLM judge |
| `CompositeOracle` | Weighted combination | all |

Pick the lowest-cost oracle that answers the question. `ExactOutcomeOracle` is the cheapest and the right default; only escalate to `TraceOracle` when ordering matters, `SideEffectOracle` when a state mutation is part of the contract, or `CompositeOracle` when the run must satisfy several criteria at once.

### Example: Constraint Oracle

A `ConstraintOracle` is the right starting point when the contract is "this output must / must not contain certain things, must / must not call certain tools, must stay within budget." It is a structural check — no geometry — and produces a verdict in microseconds.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.oracles import ConstraintOracle

oracle = ConstraintOracle(store=store)

# Define constraints in the scenario
scenario_constraints = {
    "must_contain": ["MDL-001"],
    "must_not_contain": ["DELETE", "DROP"],
    "max_tool_calls": 5,
    "required_tools": ["data_lookup"],
    "forbidden_tools": ["submit_escalation"],
    "output_max_length": 500,
}

verdict = oracle.evaluate(
    scenario=scenario,
    trace=sut_trace,
    env_state_before=before_state,
    env_state_after=after_state,
)
print(f"Passed:     {verdict.passed}")
print(f"Confidence: {verdict.confidence:.3f}")
print(f"Detail:     {verdict.detail}")
```

### Example: Trace Oracle (Tool Sequence Verification)

`TraceOracle` consumes the same `tool_calls` list that Tutorial 7's `ProcessEvaluator` does, but at a higher level — patterns, not raw set membership. The geometric scoring is the same `check_holonomy` primitive, so the holonomy defect here is comparable to the one a process evaluator emits.

```python
from knowlytix.harness.testing.oracles import TraceOracle

oracle = TraceOracle(store=store)

# Trace pattern language:
#   "data_lookup > risk_check"     — ordering required
#   "!submit_escalation"           — prohibited step
#   "data_lookup > * > calculate"  — wildcard intermediate steps

# The oracle uses check_holonomy() to measure "process curvature"
# between the actual tool path and the expected direct path
verdict = oracle.evaluate(scenario, trace, before, after)
print(f"Holonomy defect: {verdict.raw_scores.get('holonomy', 0):.4f}")
```

### Example: Side-Effect Oracle

The state-diff channel from Tutorial 10 (`get_state_diff`) is the input to the side-effect oracle. The oracle reads `before` and `after` snapshots produced by `EnvironmentSimulator`, embeds the diff, and scores it with `tension_energy` against the expected effect — high tension means the SUT changed the world differently from what the contract asked for.

```python
from knowlytix.harness.testing.oracles import SideEffectOracle

oracle = SideEffectOracle(environment=env, store=store)

# Verifies that the environment changed in expected ways
# Uses tension_energy() between expected and actual effect embeddings
verdict = oracle.evaluate(scenario, trace, before, after)
for key, detail in verdict.raw_scores.items():
    print(f"  {key}: {detail}")
```

### Example: Composite Oracle

The composite is the combinator. Use it when no single oracle is sufficient — the typical pattern is "outcome AND trace" or "outcome AND side effects." Weights are convex; the verdict is `passed` iff the weighted sum exceeds `threshold`. `fail_on_any_critical=True` is the safety lever — if any sub-oracle returns a CRITICAL severity, the whole composite fails regardless of weighted sum.

```python
from knowlytix.harness.testing.oracles import CompositeOracle, ExactOutcomeOracle, TraceOracle

composite = CompositeOracle(
    oracles=[
        (ExactOutcomeOracle(store=store), 0.6),   # 60% weight on correctness
        (TraceOracle(store=store), 0.4),           # 40% weight on process
    ],
    threshold=0.5,
    fail_on_any_critical=True,
)

verdict = composite.evaluate(scenario, trace, before, after)
print(f"Overall: {'PASS' if verdict.passed else 'FAIL'} "
      f"(confidence={verdict.confidence:.3f})")
```

### Factory Function (from YAML/dict)

`make_oracle` is the factory the YAML loader uses. Pass a dict matching one of the seven oracle types and the factory returns the wired-up object. Use this when oracle definitions live in test fixtures rather than test code.

```python
from knowlytix.harness.testing.oracles import make_oracle

oracle = make_oracle(
    oracle_def={"type": "composite", "oracles": [
        {"type": "exact_outcome", "weight": 0.6},
        {"type": "trace", "weight": 0.4},
    ], "threshold": 0.5},
    store=store,
    environment=env,
)
```

**What this tutorial leaves you with.** A library of oracles that consume the same trace + state inputs your simulator produces. Together with Tutorial 10's environment, this is what makes black-box testing more rigorous than string matching — you grade behavior, not just text. Tutorial 12 (Severity) consumes oracle verdicts to assign per-failure severity for the release gate.

> **API Reference:** [Oracle Framework](API_REFERENCE.md#27-oracle-framework) — `CompositeOracle`, `ExactOutcomeOracle`, `TraceOracle`, `SideEffectOracle`, `ConstraintOracle`, `make_oracle`

---

## 15. Tutorial 12: Severity Classification

> **Module:** `knowlytix.harness.testing.severity`, `knowlytix.harness.testing.failures`  
> **Import:** `from knowlytix.harness.testing import SeverityClassifier, SeverityLevel, FailureCode`  
> **Key Classes:** `SeverityClassifier`, `SeverityLevel`, `FailureCode`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Severity Classifier** assigns one of four levels — Critical, High, Moderate, Low — to each failure using failure code, risk tier, and GMS geometric signals. Tutorials 1, 7, and 11 produce typed verdicts and oracle outputs carrying `FailureCode` enum values; this tutorial maps each code (plus its tier and geometry) to a severity an audit report can sort by.

### When to Use

- Prioritize which failures to fix first
- Gate releases on zero-critical-failure policies
- Report severity distributions to stakeholders
- Modulate severity using geometric distance (farther from known facts = more severe)

### Step 1: Classify Failures

`classify_code` is the single-failure entry point. Pass the `FailureCode`, the agent's risk tier, and (optionally) the raw GMS score that triggered the failure; the classifier returns an `Assessment` with the four sub-scores plus the final level. The four sub-scores let an audit report explain *why* a particular failure is CRITICAL rather than HIGH.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.severity import SeverityClassifier, SeverityLevel
from knowlytix.harness.testing.failures import FailureCode

classifier = SeverityClassifier(store=store)

# Classify a single failure by code and risk tier
assessment = classifier.classify_code(
    code=FailureCode.HALLUCINATION_DETECTED,
    risk_tier="action_taking",
    raw_score=1.8,    # high geodesic = far from known facts
)

print(f"Level:          {assessment.level.value}")       # CRITICAL
print(f"Impact:         {assessment.impact_score:.2f}")
print(f"Recoverability: {assessment.recoverability_score:.2f}")
print(f"Detectability:  {assessment.detectability_score:.2f}")
print(f"Composite:      {assessment.composite_score:.2f}")
print(f"Rationale:      {assessment.rationale}")
```

### Severity Mapping

| Failure Code | Base Level | Action-Taking Escalation |
|-------------|-----------|--------------------------|
| `hallucination_detected` | CRITICAL | CRITICAL |
| `policy_violation` | CRITICAL | CRITICAL |
| `fact_incorrect` | HIGH | HIGH |
| `threshold_violation` | HIGH | HIGH |
| `process_deviation` | MODERATE | HIGH |
| `tool_budget_exceeded` | MODERATE | HIGH |
| `exact_mismatch` | LOW | LOW |

### Step 2: Classify All Failures from a Run

`classify_all` is the bulk entry point — pass the failures from `bench.typed_verdicts` and the run's risk tier, and the classifier returns one assessment per failure. `count_by_level` rolls those into the per-level counts that feed the release gate's `failure_budget` check (Tutorial 8).

```python
# Classify all failures from typed verdicts
all_failures = []
for tv in bench.typed_verdicts.values():
    all_failures.extend(tv.failures)

assessments = classifier.classify_all(all_failures, risk_tier="recommendation")
counts = classifier.count_by_level(assessments)
print(f"Critical: {counts[SeverityLevel.CRITICAL]}")
print(f"High:     {counts[SeverityLevel.HIGH]}")
print(f"Moderate: {counts[SeverityLevel.MODERATE]}")
print(f"Low:      {counts[SeverityLevel.LOW]}")
```

### GMS Geometric Modulation

The classifier uses GMS signals to adjust severity:

- **High geodesic distance** (far from known entities) increases impact score — a hallucination about something entirely absent from the knowledge base is worse than one about a related concept.
- **High tension energy** decreases detectability score — contradictions with high tension are easier to spot, which is paradoxically less dangerous.
- **Risk tier escalation** — Moderate failures in `action_taking` tier are promoted to High, because action-taking agents have higher stakes.

**What this tutorial leaves you with.** A per-failure severity assignment and a per-level count. Both are inputs to Tutorial 8's release gate — the failure budget check refuses to release if Critical or High counts exceed tier thresholds — and both flow into the executive summary and the audit packet.

> **API Reference:** [Severity Classification](API_REFERENCE.md#25-severity-classification) · [Failure Attribution](API_REFERENCE.md#13-failure-attribution)

---

## 16. Tutorial 13: Campaign Management

> **Module:** `knowlytix.harness.testing.campaign`  
> **Import:** `from knowlytix.harness.testing import CampaignManager, CampaignConfig, CampaignType`  
> **Key Classes:** `CampaignManager`, `CampaignConfig`, `CampaignType`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), evaluator (§4)

A **Campaign** groups multiple benchmark runs under a single evaluation purpose — release validation, regression sweep, model comparison, or compliance certification. A single `bench.run` answers "did this SUT pass these scenarios *now*"; a campaign answers a higher-level question — "is this release *ready*", "did v2 regress against v1", "does this fail under adversarial pressure" — by orchestrating many runs and aggregating their gates.

### When to Use

- Validate a release across multiple scenarios and risk tiers
- Compare performance across model versions
- Run adversarial sweeps targeting specific weaknesses
- Certification audits requiring comprehensive evidence

### Campaign Types

| Type | Purpose |
|------|---------|
| `RELEASE_VALIDATION` | Full test suite before deployment |
| `REGRESSION_SWEEP` | Re-run known failure scenarios |
| `ADVERSARIAL_SWEEP` | Targeted attack scenarios |
| `MODEL_COMPARISON` | Side-by-side model evaluation |
| `COMPLIANCE_CERTIFICATION` | Regulatory evidence gathering |

Each type carries default scenarios, default tier, and default stop conditions. `RELEASE_VALIDATION` is the safest starting point; pick a more specific type only when you have a reason to scope the runs differently.

### Step 1: Configure a Campaign

`CampaignConfig` declares *which targets, which scenarios, at which tier, and how many runs each*. `stop_on_critical=True` is the safety knob — abort the entire campaign as soon as any run produces a CRITICAL severity (Tutorial 12), rather than burning compute on a doomed candidate.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.campaign import CampaignConfig, CampaignType, CampaignManager

campaign_config = CampaignConfig(
    name="DS Agent v2 Release Validation",
    campaign_type=CampaignType.RELEASE_VALIDATION,
    target_profiles=["tgt_a1b2c3d4"],
    scenario_tags=["accuracy", "compliance", "governance"],
    risk_tier="recommendation",
    n_runs=4,
    stop_on_critical=True,       # abort if any critical failure
    metadata={"release": "v2.0.0"},
)

# Save for reproducibility
campaign_config.to_yaml("campaigns/release_v2.yaml")
```

### Step 2: Execute the Campaign

The manager wires the campaign config to the benchmark and run-management infrastructure from earlier tutorials. The two factory callables are the integration points: `benchmark_factory` builds a fresh `DOEGMSBenchmark` per run, `evaluator_factory` returns the SUT callable for the named target. The result is a `CampaignResult` with the aggregate gate verdict and the per-run summaries.

```python
mgr = CampaignManager(
    campaigns_dir="campaigns",
    run_manager=run_mgr,
    scenario_registry=registry,
    tracing=bench.tracing,
)

campaign_id = mgr.create_campaign(campaign_config)

result = mgr.execute_campaign(
    campaign_id,
    benchmark_factory=lambda cfg: DOEGMSBenchmark(cfg),
    evaluator_factory=lambda target_id: my_evaluator,
)

print(f"Campaign: {result.campaign_id}")
print(f"Runs:     {len(result.run_summaries)}")
print(f"Accuracy: {result.aggregate_accuracy:.1%}")
print(f"Critical: {result.critical_failure_count}")
print(f"Gate:     {'PASS' if result.gate_passed else 'FAIL'}")
```

### Step 3: Compare Campaigns

The cross-campaign comparison is the multi-run analogue of Tutorial 6's `compare_runs`. Use it for "v2 vs v1" — same scenarios, same tier, two different SUT versions — to see whether the candidate moved the aggregate gate forward.

```python
comparison = mgr.compare_campaigns("cmp_abc12345", "cmp_def67890")
print(f"Accuracy delta: {comparison['accuracy_delta']:+.1%}")
print(f"Regressions:    {comparison['regressions']}")
```

**What this tutorial leaves you with.** A reusable, stoppable, comparable bundle of runs the audit log records as a single artifact. Campaigns are the natural unit for release / certification work — Tutorial 28 (Multi-Agent), Tutorial 29 (Cross-Doc), and Tutorial 21 (Artifact Versioning) are all designed to feed into a campaign run.

> **API Reference:** [Campaign Management](API_REFERENCE.md#28-campaign-management) — `CampaignManager`, `CampaignConfig`, `CampaignType`, `CampaignResult`

---

# Part IV: Extended Test Generation

---

## 17. Tutorial 14: Taxonomy Generators

> **Module:** `knowlytix.harness.testing.generators_taxonomy`  
> **Import:** `from knowlytix.harness.testing import BoundaryGenerator, AdversarialGenerator, taxonomy_generators`  
> **Key Classes:** `BoundaryGenerator`, `AdversarialGenerator`, `RobustnessGenerator`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

Five **taxonomy generators** systematically probe failure modes that standard question generators miss: boundaries, adversarial inputs, robustness, recovery, and governance. Tutorials 1-13 graded behavior on questions the geometric generators emit (well-posed, in-distribution); these five generators emit questions designed to *break* the SUT in specific, named ways.

### When to Use

- Standard generators only test the happy path
- You need to probe specific failure modes (off-by-one, injection, typos)
- Compliance requires evidence of adversarial and governance testing

### The Five Generators

| Generator | Category | What It Probes | GMS Primitive |
|-----------|----------|---------------|---------------|
| `BoundaryGenerator` | `boundary` | Phase encoder limits (exact, +epsilon, -epsilon) | `phase_encoders` |
| `AdversarialGenerator` | `adversarial` | Injection, contradictions, authority spoofing | `tension_energy()` |
| `RobustnessGenerator` | `robustness` | Typos, synonyms, truncation, rounding | `fuzzy_match_entity()` |
| `RecoveryTestGenerator` | `recovery_test` | Tool failure + graceful recovery | `tension_energy()` |
| `GovernanceTestGenerator` | `governance` | Forbidden tools, missing approvals, capability bounds | `score_triple()` |

Each generator emits questions in its own category, so a release campaign can require, say, "100 boundary, 50 adversarial, 50 governance" without the categories blurring into a single bag.

### Use in Configuration

The simplest activation: name the generators in `config.generators` alongside the standard ones. The benchmark expands the list to fill `max_per_category` slots from each generator, so a value of 10 yields up to 50 taxonomy probes total in addition to the geometric ones.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
config = DOEHarnessConfig(
    generators=[
        # Classic
        "exact_recall", "threshold", "plausibility", "tension",
        # Taxonomy (new)
        "boundary", "adversarial", "robustness",
        "recovery_test", "governance",
    ],
    max_per_category=10,
)
```

### Use Directly

Bypass the config when you want a generator's output as a separate cohort — for example, when building a regression suite from adversarial probes only. `taxonomy_generators()` returns all five as a list; or instantiate one for finer control. Each `generate(store)` call returns a list of typed questions with category metadata.

```python
from knowlytix.harness.testing.generators_taxonomy import (
    BoundaryGenerator, AdversarialGenerator,
    RobustnessGenerator, RecoveryTestGenerator,
    GovernanceTestGenerator, taxonomy_generators,
)

# All five at once
gens = taxonomy_generators()

# Or individually
boundary_gen = BoundaryGenerator()
questions = boundary_gen.generate(store)
print(f"Boundary questions: {len(questions)}")

# Each question carries metadata about what it probes
for q in questions[:3]:
    print(f"  {q.qid}: {q.natural_language[:60]}...")
    print(f"    boundary_type: {q.metadata.get('boundary_type')}")
```

### What Each Generator Produces

**BoundaryGenerator**: For each phase encoder in the GMS store, generates questions at the exact threshold, just above, and just below. Example: *"Is the test AUC of MDL-001 above 0.84?"* (at boundary) vs *"Is the test AUC of MDL-001 above 0.8411?"* (epsilon below).

**AdversarialGenerator**: Three strategies — (1) contradictory evidence injected into context, (2) competing instructions, (3) authoritative-sounding false claims. Uses `tension_energy()` to calibrate contradiction strength.

**RobustnessGenerator**: Applies perturbations — typos in entity names ("MDL-0O1"), synonym substitution, numeric rounding, context truncation. Tests whether `fuzzy_match_entity()` tolerance matches the agent's tolerance.

**RecoveryTestGenerator**: For each scenario with `required_tools`, creates a companion scenario where one tool fails, expecting the agent to recover gracefully.

**GovernanceTestGenerator**: Scenarios that tempt the agent to use forbidden tools, skip approval steps, or exceed tool budgets. Uses `score_triple(action, "requires", "approval")` for policy plausibility.

**What this tutorial leaves you with.** Five orthogonal probe families that turn known failure modes into systematic test categories. The natural pairings: `BoundaryGenerator` with phase-encoder thresholds (Tutorial 8), `AdversarialGenerator` with the `ADVERSARIAL_SWEEP` campaign type (Tutorial 13), `GovernanceTestGenerator` with the policy engine (Tutorial 3). The remaining tutorials (Part V onward) layer additional generator types on top of this taxonomy.

> **API Reference:** [Taxonomy Generators](API_REFERENCE.md#9-taxonomy-generators) — `BoundaryGenerator`, `AdversarialGenerator`, `RobustnessGenerator`, `RecoveryTestGenerator`, `GovernanceTestGenerator`

---

## 18. Tutorial 15: Multi-Turn Testing

> **Module:** `knowlytix.harness.testing.multiturn`  
> **Import:** `from knowlytix.harness.testing import MultiTurnAdapter, InteractionScript, TurnSpec`  
> **Key Classes:** `MultiTurnAdapter`, `InteractionScript`, `TurnSpec`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), evaluator (§4)

**Multi-turn interaction scripts** test conversation-level behavior: state carryover, per-turn assertions, and cross-turn consistency. Tutorials 1-17 graded the SUT one question at a time. Real agents are stateful — they remember earlier turns, build up context, and can contradict themselves three turns later. This tutorial is the surface for grading those failure modes explicitly.

### When to Use

- Agent behavior depends on conversation history
- Testing multi-step workflows (teach → test → verify)
- Checking cross-turn consistency (entity from turn 1 recalled in turn 3)
- Validating per-turn tool expectations

### Step 1: Define an Interaction Script

An `InteractionScript` is a list of `TurnSpec`s plus shared setup and teardown. Each turn carries its own assertions, expected tool calls, and `carry_forward` dict (entities to template into later turns). The three-turn pattern below — teach, distract, recall — is the canonical memory probe; replace it with your domain's natural dialog shape.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.multiturn import (
    TurnSpec, InteractionScript, MultiTurnAdapter,
    MultiTurnScenarioRunner,
)

script = InteractionScript(
    script_id="memory_test_001",
    name="3-Turn Memory Carryover",
    description="Teach a fact, distract, then recall",
    setup_context="You are a model risk analyst.",
    turns=[
        TurnSpec(
            turn_id="teach",
            user_message="Model MDL-001 has a test AUC of 0.8412.",
            expected_contains=["MDL-001"],
            expected_tool_calls=["data_lookup"],
            carry_forward={"model_id": "MDL-001", "auc": "0.8412"},
        ),
        TurnSpec(
            turn_id="distract",
            user_message="What is the capital of France?",
            expected_not_contains=["MDL-001"],  # shouldn't hallucinate
        ),
        TurnSpec(
            turn_id="recall",
            user_message="What was the test AUC of {{model_id}}?",
            expected_contains=["{{auc}}"],       # template substitution
            max_latency_ms=5000,
        ),
    ],
    teardown_assertions={"all_turns_passed": True},
)
```

### Step 2: Execute the Script

`MultiTurnAdapter` is the SUT-facing wrapper that walks the script turn by turn, applies template substitution from `carry_forward`, evaluates per-turn assertions, and accumulates the cross-turn consistency scores. Pass the same `harness`, `environment`, and `store` you used in earlier tutorials — the adapter participates in the same audit and tracing surface.

```python
adapter = MultiTurnAdapter(
    harness=my_harness,
    environment=env,
    store=store,
    tracing=bench.tracing,
)

result = adapter.execute_script(script)

print(f"Overall: {'PASS' if result.overall_passed else 'FAIL'}")
print(f"Consistency scores: {result.consistency_scores}")

for tr in result.turn_results:
    status = "OK" if tr.passed else "FAIL"
    print(f"  Turn {tr.turn_id}: {status} ({tr.latency_ms}ms)")
    if tr.assertions_failed:
        for v in tr.assertions_failed:
            print(f"    Failed: {v}")
```

### Load from YAML

The YAML form mirrors the inline build and is the right shape when scripts are versioned next to test fixtures. `InteractionScript.from_yaml` returns the same object the inline `InteractionScript(...)` call produces.

```yaml
# interaction_scripts/memory_test.yaml
script_id: memory_test_001
name: "3-Turn Memory Carryover"
setup_context: "You are a model risk analyst."
turns:
  - turn_id: teach
    user_message: "Model MDL-001 has a test AUC of 0.8412."
    expected_contains: ["MDL-001"]
  - turn_id: distract
    user_message: "What is the capital of France?"
  - turn_id: recall
    user_message: "What was the test AUC of MDL-001?"
    expected_contains: ["0.8412"]
```

```python
script = InteractionScript.from_yaml("interaction_scripts/memory_test.yaml")
```

### GMS Cross-Turn Consistency

The `consistency_scores` dict in `ScriptResult` uses `tension_energy()` between entities extracted from successive turns. A spike indicates the agent contradicted itself across the conversation. Low tension (< 0.3) indicates consistent behavior.

**What this tutorial leaves you with.** A reusable script format for stateful probes plus a per-turn / per-script result. Pair this with Tutorial 22 (Agent Memory Testing) for explicit memory probes and Tutorial 29 (Streaming Consistency) for streaming-specific cross-turn checks.

> **API Reference:** [Multi-Turn Interactions](API_REFERENCE.md#38-multi-turn-interactions) — `MultiTurnAdapter`, `InteractionScript`, `TurnSpec`, `ScriptResult`, `TurnResult`

---

## 19. Tutorial 16: Safety and Isolation

> **Module:** `knowlytix.harness.testing.safety`  
> **Import:** `from knowlytix.harness.testing import SafetyEnforcer, SafetyConfig, SyntheticDataFramework`  
> **Key Classes:** `SafetyEnforcer`, `SafetyConfig`, `SyntheticDataFramework`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Safety Enforcer** provides runtime protection — blocking destructive actions, enforcing sandbox mode, and generating synthetic test data. The policy engine in Tutorial 3 was *application-layer* policy ("this action requires approval"); the safety enforcer here is *infrastructure-layer* enforcement ("no real network calls, no file writes, no destructive tools, period"). Use both — they catch different classes of mistake.

### When to Use

- Testing agents that might make real API calls or file writes
- Enforcing network isolation in CI/CD pipelines
- Generating synthetic entities from GMS geometry (no real data exposure)
- Blocking destructive tool patterns (`delete_*`, `drop_*`)

### Step 1: Configure Safety

`SafetyConfig` declares the hard limits. `destructive_action_mode` is the dial: `block` refuses outright, `noop` lets the call appear to succeed but skips the side effect, `log` records the call without intervening. Default to `block` for any pre-release run; switch to `log` only when the failure mode you are studying *is* a destructive action.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.safety import SafetyConfig, SafetyEnforcer

safety_config = SafetyConfig(
    sandbox_mode=True,
    network_isolation=True,
    allowed_domains=["localhost"],
    max_tool_calls=100,
    max_file_writes=0,           # no file writes allowed
    max_network_calls=0,         # no network calls allowed
    blocked_tool_patterns=["delete_*", "drop_*", "rm_*"],
    allowed_tool_patterns=["data_lookup", "risk_check", "gms_*"],
    destructive_action_mode="block",  # "block", "noop", or "log"
)
```

### Step 2: Install on Harness

The enforcer attaches *before* the gateway in the call stack, so safety refusals fire before any policy or fault profile gets a chance. This ordering is intentional: an unsafe call should never reach a policy engine that might (incorrectly) approve it.

```python
enforcer = SafetyEnforcer(safety_config)
enforcer.install(my_harness)    # chains with ToolGateway — safety first

# Now all tool calls pass through safety checks before gateway
# Blocked calls return an error message, not a real execution

# When done:
enforcer.uninstall(my_harness)
```

### Load from YAML

```yaml
# safety_config.yaml
sandbox_mode: true
network_isolation: true
max_tool_calls: 100
max_file_writes: 0
blocked_tool_patterns: ["delete_*", "drop_*"]
destructive_action_mode: "block"
```

```python
config = SafetyConfig.from_yaml("safety_config.yaml")
```

### Synthetic Data Generation

The synthetic-data framework is the answer when sandbox mode forbids real data but the SUT needs *something* to operate on. It samples entities from the learned GMS embedding distribution, so the synthetic data has the same statistical shape as the real corpus — close enough to exercise the SUT, distinct enough that no real customer ID leaks.

```python
from knowlytix.harness.testing.safety import SyntheticDataFramework

synth = SyntheticDataFramework(store=store)

# Generate synthetic entities by sampling from the learned
# GMS embedding distribution on S^{m-1}
entities = synth.synthesize_entities(n=50, seed=42)
relations = synth.synthesize_relations(entities, density=0.3)
fixture = synth.synthesize_fixture(
    spec={"n_entities": 20, "n_relations": 40},
    seed=42,
)
print(f"Fixture: {fixture.fixture_id}, {len(fixture.data)} records")
```

**What this tutorial leaves you with.** A safety layer that intercepts before any policy runs, plus a synthetic-data generator that produces realistic-but-fake fixtures. Use this as the outermost wrapper in any pre-release run — Tutorial 17 (Parallel) and Tutorial 19 (Browser) both compose with the enforcer in the standard ordering.

> **API Reference:** [Safety Framework](API_REFERENCE.md#39-safety-framework) — `SafetyEnforcer`, `SafetyConfig`, `SyntheticDataFramework`

---

# Part V: Scale and Artifact Management

---

## 20. Tutorial 17: Parallel Execution

> **Module:** `knowlytix.harness.testing.parallel`  
> **Import:** `from knowlytix.harness.testing import ParallelExecutor, ParallelConfig`  
> **Key Classes:** `ParallelExecutor`, `ParallelConfig`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), evaluator (§4)

Run benchmark evaluations concurrently to reduce wall time. Earlier tutorials produced suites of hundreds of questions; with one worker the wall-clock cost grows linearly. The parallel executor is the bottom-layer concurrency primitive — the benchmark uses it automatically when `target.max_concurrent > 1`, but it is also exposed for direct use when you want explicit control.

### When to Use

- Large test suites (100+ questions) taking too long sequentially
- Target SUT supports concurrent requests
- CI/CD pipelines with time constraints

### Configuration

`ParallelConfig` exposes the four knobs that matter for stability under load — workers, batch size, per-question timeout, and retry behavior. The retry settings are what make a noisy SUT (rate limits, transient errors) testable without a single failure poisoning the whole run.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.parallel import ParallelConfig, ParallelExecutor

parallel_config = ParallelConfig(
    max_workers=4,                  # thread pool size
    batch_size=10,                  # questions per batch
    timeout_per_question_s=120,     # per-question timeout
    retry_on_error=True,
    max_retries=2,
    retry_delay_s=1.0,
)

executor = ParallelExecutor(
    config=parallel_config,
    on_progress=lambda done, total: print(f"  {done}/{total}"),
)

results = executor.execute_batch(
    questions=doe_questions,
    evaluate_fn=lambda q: my_evaluator(q.question.llm_prompt, context),
)

print(f"Completed: {len(results)}")
print(f"Errors:    {sum(1 for r in results if r.error)}")
```

### Typical Speedup

| Questions | Sequential | 4 Workers | Speedup |
|-----------|-----------|-----------|---------|
| 142 | ~33 min | ~8 min | 4.1x |
| 232 | ~44 min | ~11 min | 4.0x |

### Integration with DOEGMSBenchmark

The benchmark uses `ParallelExecutor` automatically when `config.max_parallel > 1`:

```python
config = DOEHarnessConfig(
    # ... other settings ...
    # (ParallelConfig is constructed internally from max_concurrent in TargetProfile)
)
```

For the typical campaign workflow (Tutorial 13), set `max_concurrent` on the `TargetProfile` and let the benchmark wire the executor — explicit `ParallelExecutor` use is for non-benchmark loops (e.g., grading a saved corpus offline).

**What this tutorial leaves you with.** A 4× speedup on parallel-friendly suites without changing test logic. Combine with Tutorial 17 (Safety) so concurrency does not mean concurrent destructive calls, and with Tutorial 13 (Campaigns) for batched candidate evaluation.

> **API Reference:** [Parallel Execution](API_REFERENCE.md#40-parallel-execution) — `ParallelExecutor`, `ParallelConfig`

---

## 21. Tutorial 18: Artifact Versioning

> **Module:** `knowlytix.harness.testing.versioning`  
> **Import:** `from knowlytix.harness.testing import ArtifactRegistry, VersionedArtifact`  
> **Key Classes:** `ArtifactRegistry`, `VersionedArtifact`  
> **Prereqs:** None (standalone)

**Artifact Versioning** tracks fault profiles, policy rules, and fixtures with SHA-256 integrity hashing and version history. Tutorials 2, 3, and 10 created artifacts (fault profiles, policy rules, data fixtures) and used them inline. The registries here turn those one-off objects into versioned, hashed, reloadable resources — what an audit-defensible build needs to answer "which exact rule set produced this verdict?"

### When to Use

- Reproducing past test configurations exactly
- Auditing which artifact versions were used in which runs
- Diffing artifact changes between versions
- Regulatory compliance requiring traceable test configurations

### Step 1: Create and Register Artifacts

A `VersionedArtifact` carries the content plus the integrity stamp. Registering it computes the SHA-256 over the canonical JSON, so any post-hoc edit changes the hash and `verify_integrity()` returns `False`. Pass real metadata (author, purpose) — these become the audit trail when the artifact is referenced by run ID.

```python
from knowlytix.harness.testing.versioning import (
    VersionedArtifact, ArtifactRegistry,
    FaultProfileRegistry, PolicyRuleRegistry, FixtureRegistry,
)

# Create a fault profile registry
fault_registry = FaultProfileRegistry("my_test/artifacts/faults")

artifact = VersionedArtifact(
    artifact_type="fault_profile",
    version=1,
    content={
        "tool_pattern": "submit_escalation",
        "error_rate": 1.0,
        "error_message": "Service unavailable",
    },
    author="test_engineer",
    metadata={"purpose": "escalation blocking"},
)

artifact_id = fault_registry.add(artifact)
print(f"Registered: {artifact_id}")
print(f"Hash: {artifact.content_hash}")
print(f"Integrity: {artifact.verify_integrity()}")
```

### Step 2: Version Tracking

`update` does not destroy the prior version. The history is preserved so a reviewer can see exactly when and how an artifact changed. This matters for the same reason scenario versioning matters in Tutorial 4 — a regression's truth value can legitimately drift, and the audit log needs to point at the version that was current when the run executed.

```python
# Update an artifact (creates new version)
loaded = fault_registry.get(artifact_id)
loaded.content["error_rate"] = 0.5
loaded.version = 2
fault_registry.update(loaded)

# List versions
versions = fault_registry.list_versions(artifact_id)
for v in versions:
    print(f"  v{v.version}: hash={v.content_hash[:12]}... "
          f"created={v.created_at}")
```

### Specialized Registries

The three specialized registries are typed wrappers — they share the same interface but carry domain-specific helpers (e.g., `PolicyRuleRegistry` knows how to construct a `PolicyRule` from the stored content). Use the specialized registry whenever the domain has a natural one; reach for the generic `ArtifactRegistry` only for ad-hoc artifacts.

```python
# Policy rules
policy_reg = PolicyRuleRegistry("my_test/artifacts/policies")

# Fixtures
fixture_reg = FixtureRegistry("my_test/artifacts/fixtures")
```

All registries share the same `ArtifactRegistry` interface (`add`, `get`, `list_versions`, `update`).

**What this tutorial leaves you with.** Versioned, hashed, reloadable artifacts with explicit lineage. Wire artifact IDs into `RunRecord.metadata` (Tutorial 5) so an audit query "which fault profile was used for run X" resolves byte-identically to a registry lookup.

> **API Reference:** [Versioning](API_REFERENCE.md#43-versioning) — `ArtifactRegistry`, `VersionedArtifact`, `FaultProfileRegistry`, `PolicyRuleRegistry`, `FixtureRegistry`

---

## 22. Tutorial 19: Browser and UI Testing

> **Module:** `knowlytix.harness.testing.browser_adapter`  
> **Import:** `from knowlytix.harness.testing import BrowserAdapter, UIScript, BrowserConfig`  
> **Key Classes:** `BrowserAdapter`, `UIScript`, `BrowserConfig`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Browser Adapter** enables testing agentic systems with web UIs using Playwright. This is the §3.5.B `BrowserAdapter` recipe in operational form: the SUT is a web application driven through a real browser, and the harness drives it through Playwright primitives — type, click, wait, assert, screenshot.

### When to Use

- Testing agents with browser-based interfaces
- Verifying UI-driven workflows (navigate, click, type, assert)
- Taking screenshots for visual regression testing

### Step 1: Define a UI Script

A `UIScript` is the browser equivalent of an `InteractionScript`: a list of `UIStep`s with selectors, actions, and assertions. Each step is one Playwright operation. The action vocabulary covers the six common cases — `type`, `click`, `wait`, `assert_text`, `screenshot`, plus `navigate`. Anything richer (multi-page workflows, file uploads) belongs in a custom step type.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.browser_adapter import (
    BrowserConfig, BrowserAdapter, UIStep, UIScript,
)

script = UIScript(
    script_id="login_flow",
    name="Login and Check Dashboard",
    setup_url="http://localhost:8080/login",
    steps=[
        UIStep(step_id="type_user", action="type",
               selector="#username", value="admin"),
        UIStep(step_id="type_pass", action="type",
               selector="#password", value="test123"),
        UIStep(step_id="click_login", action="click",
               selector="#login-button"),
        UIStep(step_id="wait_dash", action="wait",
               selector="#dashboard", timeout_ms=5000),
        UIStep(step_id="check_title", action="assert_text",
               selector="h1", value="Model Risk Dashboard"),
        UIStep(step_id="screenshot", action="screenshot",
               screenshot="dashboard.png"),
    ],
)
```

### Step 2: Execute

`BrowserConfig` chooses the browser engine and rendering mode. Use `headless=True` in CI; switch to `headless=False` for local debugging when you need to *see* the SUT misbehave. Screenshots are written to `screenshot_dir` keyed by `step_id`, so visual regression diffs are byte-stable across runs.

```python
config = BrowserConfig(
    base_url="http://localhost:8080",
    headless=True,
    browser_type="chromium",      # chromium, firefox, webkit
    screenshot_dir="my_test/screenshots",
)

adapter = BrowserAdapter(config)
result = adapter.execute_ui_script(script)

print(f"Overall: {'PASS' if result.overall_passed else 'FAIL'}")
for sr in result.step_results:
    status = "OK" if sr.success else "FAIL"
    print(f"  {sr.step_id}: {status} ({sr.latency_ms}ms)")
```

### Load from YAML

```yaml
# ui_scripts/login_flow.yaml
script_id: login_flow
name: "Login and Check Dashboard"
setup_url: "http://localhost:8080/login"
steps:
  - step_id: type_user
    action: type
    selector: "#username"
    value: admin
  - step_id: click_login
    action: click
    selector: "#login-button"
  - step_id: check_title
    action: assert_text
    selector: "h1"
    value: "Model Risk Dashboard"
```

```python
script = UIScript.from_yaml("ui_scripts/login_flow.yaml")
```

**Note**: Requires `playwright` package (`pip install playwright && playwright install`).

**What this tutorial leaves you with.** A scriptable, repeatable, screenshot-able browser test surface for agentic UIs. Combine with Tutorial 16 (Safety) and `network_isolation=True` so the browser cannot reach real services during a test run, and with Tutorial 17 (Parallel) once you need to fan out the same script across many sessions.

> **API Reference:** [Browser Adapter](API_REFERENCE.md#46-browser-adapter) — `BrowserAdapter`, `BrowserConfig`, `UIScript`, `UIStep`

---

# Part VI: GMS-Native Behavioral Testing

These modules use GMS geometric primitives — phase encoding on S^1, geodesic distance, tension energy — as the primary measurement mechanism, making them unique to GMSH.

---

## 23. Tutorial 20: Performance Profiling

> **Module:** `knowlytix.harness.testing.profiling`  
> **Import:** `from knowlytix.harness.testing import PerformanceProfiler, PerformanceReport`  
> **Key Classes:** `PerformanceProfiler`, `PerformanceReport`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Performance Profiling** encodes latency as a phase angle on S^1 and measures service-level-agreement (SLA) compliance via tension energy with the SLA target phase. Plain percentile reports answer "how slow"; the phase encoding here also gives you a single number — SLA tension — that compares cleanly with the geometric measurements the rest of the harness produces.

### How It Works

```
Latency → Phase mapping (S^1):
  0 ms     → 0°     (instant)
  SLA ms   → 90°    (at threshold)
  2×SLA ms → 180°   (severe violation, clamped)

SLA tension = 2·sin(|θ_observed - θ_SLA| / 2)
  Range [0, 2]: 0 = within SLA, ~2 = severe violation
```

The phase-encoding trick: an SLA breach lives at $\theta = \pi/2$, severe overrun at $\theta = \pi$. Tension energy turns the angular distance into a $[0, 2]$ score that aggregates the same way `tension_energy` does on entity pairs.

### Usage

`record(tool, latency_ms)` accumulates per-call latencies; `report()` emits the rollup. The mean and max SLA tension are the numbers worth tracking — the percentiles are the human-readable companion.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.profiling import PerformanceProfiler

profiler = PerformanceProfiler(store=store, sla_latency_ms=5000.0)

# Record latencies during a benchmark run
profiler.record("data_lookup", latency_ms=1200.0)
profiler.record("risk_check", latency_ms=8500.0)
profiler.record("gms_lookup", latency_ms=450.0)
profiler.record("data_lookup", latency_ms=4800.0)

report = profiler.report()
print(f"SLA pass rate:    {report.sla_pass_rate:.0%}")
print(f"Mean latency:     {report.mean_latency_ms:.0f} ms")
print(f"P95 latency:      {report.p95_latency_ms:.0f} ms")
print(f"P99 latency:      {report.p99_latency_ms:.0f} ms")
print(f"Mean SLA tension: {report.mean_sla_tension:.4f}")
print(f"Max SLA tension:  {report.max_sla_tension:.4f}")
```

### Enable in Config

Setting the flag wires the profiler into `bench.run` automatically; every tool call's latency is recorded against the per-target SLA from the target profile (Tutorial 9).

```python
config = DOEHarnessConfig(
    enable_performance_profiling=True,
    sla_latency_ms=5000.0,
)
```

**What this tutorial leaves you with.** Per-tool latency rollups plus a single SLA-tension number that travels alongside the other geometric scores. Combine with Tutorial 21 (Cost) for the full latency-and-spend picture, and with Tutorial 30 (Live Dashboard) to watch SLA tension drift in real time.

> **API Reference:** [Performance Profiling](API_REFERENCE.md#41-performance-profiling) — `PerformanceProfiler`, `PerformanceRecord`, `PerformanceReport`

---

## 24. Tutorial 21: Cost and Token Tracking

> **Module:** `knowlytix.harness.testing.cost_tracker`  
> **Import:** `from knowlytix.harness.testing import CostTracker, CostSummary`  
> **Key Classes:** `CostTracker`, `CostSummary`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Cost Tracking** encodes cumulative cost as a phase angle on S^1. Budget compliance is measured via tension with the budget phase. Same trick as Tutorial 20 — phase-encoding lets a budget overrun produce a number on the same `[0, 2]` scale as every other GMS measurement, so a release gate can refuse on tension regardless of which channel raised the alarm.

### How It Works

```
Cost → Phase mapping (S^1):
  $0        → 0°     (no spend)
  Budget    → 90°    (at budget)
  2×Budget  → 180°   (over budget, clamped)

Budget tension = 2·sin(|θ_cumulative - θ_budget| / 2)
```

### Usage

`record(tool, input_tokens, output_tokens)` accumulates per-call token usage; the tracker maps it to USD using the configured prices and reports both the absolute cost and the budget tension.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.cost_tracker import CostTracker

tracker = CostTracker(
    store=store,
    budget_usd=10.0,
    # Default pricing: $3/M input, $15/M output
    # Override: input_price_per_mtok=3.0, output_price_per_mtok=15.0
)

# Record token usage per tool call
tracker.record("gms_lookup", input_tokens=500, output_tokens=200)
tracker.record("gms_search", input_tokens=1200, output_tokens=800)
tracker.record("data_lookup", input_tokens=300, output_tokens=150)

summary = tracker.summary()
print(f"Total input tokens:  {summary.total_input_tokens}")
print(f"Total output tokens: {summary.total_output_tokens}")
print(f"Total cost:          ${summary.total_cost_usd:.4f}")
print(f"Budget:              ${summary.budget_usd:.2f}")
print(f"Budget tension:      {summary.budget_tension:.4f}")
print(f"Budget utilization:  {summary.budget_utilization:.2%}")
print(f"Records:             {summary.n_records}")
```

### Enable in Config

The flag wires the tracker into `bench.run`. Token counts are read from `SUTTrace.input_tokens` / `output_tokens`, so the SUT adapter (§3.5.B) is what makes this measurement possible — an adapter that does not surface token counts will under-report cost.

```python
config = DOEHarnessConfig(
    enable_cost_tracking=True,
    cost_budget_usd=10.0,
)
```

**What this tutorial leaves you with.** Per-tool token usage, USD cost, and a budget-tension number on the same scale as the other geometric measurements. Use this to gate runaway-cost agents — a tier in Tutorial 8 that requires `budget_tension < 0.5` enforces hard spend limits at release time.

> **API Reference:** [Cost Tracking](API_REFERENCE.md#42-cost-tracking) — `CostTracker`, `CostRecord`, `CostSummary`

---

## 25. Tutorial 22: Agent Memory Testing

> **Module:** `knowlytix.harness.testing.stateful`  
> **Import:** `from knowlytix.harness.testing import StatefulAgentTester, AgentMemoryProbe, MemoryAssessment`  
> **Key Classes:** `StatefulAgentTester`, `AgentMemoryProbe`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Stateful Agent Testing** probes memory retention across sessions using teach-then-test patterns scored by GMS geodesic distance. Tutorial 15 graded conversation-level state via assertion patterns; this tutorial is more clinical — it builds calibrated probes that exercise specific memory operations and grades each on the same geodesic scale the verifier uses for facts.

### How It Works

1. **Teach phase**: Feed the agent a fact triple (e.g., *"MDL-001 has test AUC 0.8412"*)
2. **Delay**: Insert distractor turns to test retention
3. **Recall phase**: Ask the agent to recall the fact
4. **Score**: `score_triple()` geodesic measures recall accuracy; `tension_energy()` measures session drift

### Usage

`build_probes` walks the GMS store and synthesizes one probe per chosen entity — the teach prompt asserts a fact, the recall prompt asks for it back. Run the agent against each probe, hand the result to `score_recall`, and the assessment summarizes accuracy plus the session-level drift.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.stateful import StatefulAgentTester, MemoryRecallOracle

tester = StatefulAgentTester(store=store)

# Build probes from GMS entities
probes = tester.build_probes(
    n_probes=10,
    delay_turns=3,       # 3 distractor turns between teach and recall
    seed=42,
)

print(f"Built {len(probes)} memory probes")
for p in probes[:3]:
    print(f"  Teach: {p.teach_prompt[:50]}...")
    print(f"  Recall: {p.recall_prompt[:50]}...")

# Run probes against your agent and score each recall
def agent_fn(prompt: str) -> str:
    return my_agent.run(prompt)

for probe in probes:
    answer = agent_fn(probe.recall_prompt)
    tester.score_recall(probe, answer)

assessment = tester.assess(probes)

print(f"\nMemory Assessment:")
print(f"  Recalled:             {assessment.n_recalled}/{assessment.n_probes}")
print(f"  Mean recall geodesic: {assessment.mean_recall_geodesic:.4f}")
print(f"  Mean session tension: {assessment.mean_session_tension:.4f}")
print(f"  Degradation detected: {assessment.degradation_detected}")
```

### As a Question Generator

The same probe-generation logic is also exposed as a benchmark generator, so memory probes can flow through the regular `bench.run` pipeline alongside the geometric and taxonomy generators.

```python
from knowlytix.harness.testing.stateful import StatefulAgentGenerator

gen = StatefulAgentGenerator()
questions = gen.generate(store)
# Produces teach-test question pairs
```

### Enable in Config

`stateful_session_count` controls how many independent test sessions the generator runs. More sessions = more statistical confidence in `degradation_detected`.

```python
config = DOEHarnessConfig(
    enable_stateful_testing=True,
    stateful_session_count=5,
)
```

**What this tutorial leaves you with.** A clinical memory-retention assessment over the SUT's session boundaries, expressed in the same geodesic units the rest of the harness uses. Pair with Tutorial 15 (Multi-Turn) when you need richer dialog patterns and with Tutorial 23 (Hallucination) when the failure mode you suspect is "the agent recalled, but recalled the wrong fact."

> **API Reference:** [Stateful Agent Testing](API_REFERENCE.md#32-stateful-agent-testing) — `StatefulAgentTester`, `AgentMemoryProbe`, `MemoryAssessment`, `MemoryRecallOracle`

---

## 26. Tutorial 23: Hallucination Detection

> **Module:** `knowlytix.harness.testing.hallucination`  
> **Import:** `from knowlytix.harness.testing import HallucinationOracle, HallucinationGenerator, geodesic_band`  
> **Key Classes:** `HallucinationOracle` (a Chapter 15 `OracleBase`)  
> **Prereqs:** `bench`, `store`, `store/calibration.json` (Tutorial 1, Steps 1–2 + Chapter 8 calibration)

**Hallucination Detection** is a thin layer over the Chapter 8–10 claim pipeline. It does not define a separate severity vocabulary or verdict type. Instead, it routes free-text response claims through `GraphVerifier` configured with the calibrated $\tau_{\text{geo}}$ from `store/calibration.json` and tags every non-pass as `FailureCode.HALLUCINATION_DETECTED`. The presentation bands below are cosmetic — for governance, the only level that matters is the binary pass / fail against $\tau_{\text{geo}}$.

### Presentation bands (anchored on calibrated $\tau_{\text{geo}}$ and geodesic ceiling)

Let $\Delta = (\text{ceiling} - \tau) / 3$ where the geodesic ceiling is 2.0.

| Geodesic | Band | Meaning |
|----------|------|---------|
| $d \le \tau$ | `grounded` | Consistent with the manifold |
| $\tau < d \le \tau + \Delta$ | `embellishment` | Minor drift beyond the source |
| $\tau + \Delta < d \le \tau + 2\Delta$ | `distortion` | Significant drift |
| $d > \tau + 2\Delta$ (incl. out-of-vocabulary OOV ceiling) | `fabrication` | Ungrounded |

Bands are presentation only. **Governance severity** comes from `SeverityClassifier` applied to `FailureCode.HALLUCINATION_DETECTED` (CRITICAL base severity in the Chapter 10 failure-code map).

### Usage

The oracle exposes three call shapes — one per output type the rest of the harness consumes. `assess_claim` returns a Chapter 9 `SubVerdict` for a single triple. `assess_triples` returns a Chapter 10 `TypedVerdict` aggregated over a batch (the audit-trail-ready form). `evaluate(scenario, trace)` returns a Chapter 15 `OracleVerdict` so the oracle composes with `TieredReleaseGate` like any other.

```python
# Prereqs: bench + store + store/calibration.json
from knowlytix.harness.testing.hallucination import (
    HallucinationOracle, geodesic_band,
)

# tau_geodesic is loaded from store/calibration.json by default.
oracle = HallucinationOracle(store=store)

# --- Single claim -> Chapter 9 SubVerdict ---
sv = oracle.assess_claim("MDL-001", "has_test_auc", "0.9500")
band = geodesic_band(sv.raw_score, oracle.tau_geodesic)
print(f"passed={sv.passed}, geodesic={sv.raw_score:.4f}, band={band}")

# --- Batch -> Chapter 10 TypedVerdict (audit trail ready) ---
verdict = oracle.assess_triples([
    ("MDL-001", "has_test_auc", "0.9500"),
    ("MDL-999", "has_algorithm", "QuantumBoost"),   # fabricated
])
print(f"passed={verdict.passed}; failures={len(verdict.failures)}")
for f in verdict.failures:
    print(f"  {f.code.value}: geodesic={f.raw_score:.4f}")

# --- Response-level -> Chapter 15 OracleVerdict ---
# Used inside TieredReleaseGate like any other oracle.
verdict = oracle.evaluate(scenario, trace)
```

### As a Question Generator

The companion generator emits questions designed to *invite* hallucination — entities at the manifold boundary (where the agent has weak grounding), near-miss entities (likely confusions), and open-ended elaboration prompts (where free generation is most tempted to fabricate).

```python
from knowlytix.harness.testing.hallucination import HallucinationGenerator

gen = HallucinationGenerator()
questions = gen.generate(store)
# Three strategies: boundary entities, near-miss entities, open-ended elaboration
```

### Enable in Config

The oracle does not need its own config flag for the threshold — it reads $\tau_{\text{geo}}$ from `store/calibration.json` like Tutorial 8 does for the release gate. This is the same calibrated cut everywhere; updating it in one place updates every consumer.

```python
config = DOEHarnessConfig(
    enable_hallucination_testing=True,
    # The oracle reads tau_geodesic from store/calibration.json; no config flag needed.
)
```

**What this tutorial leaves you with.** A calibrated detector that turns "did the agent fabricate" into a typed verdict on the same audit trail as every other claim check. The release gate (Tutorial 8) treats `HALLUCINATION_DETECTED` as CRITICAL by default — so a single failed probe is sufficient to block release at the action-taking tier.

> **API Reference:** [Hallucination Testing](API_REFERENCE.md#30-hallucination-testing) — `HallucinationOracle`, `HallucinationGenerator`, `geodesic_band`

---

## 27. Tutorial 24: Confidence Calibration

> **Module:** `knowlytix.harness.testing.calibration`  
> **Import:** `from knowlytix.harness.testing import CalibrationTester, CalibrationReport`  
> **Key Classes:** `CalibrationTester`, `CalibrationReport`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Calibration Testing** checks whether an agent's claimed confidence matches its actual correctness using S^1 phase encoding. An overconfident agent that says "0.95" when it is wrong is more dangerous than a humbler agent that says "0.6" when it is wrong — same accuracy, very different downstream behavior. Expected calibration error (ECE) is the standard metric; the phase encoding here gives a cleaner aggregate that composes with the rest of the harness's geometric measurements.

### How It Works

```
Confidence → Phase: 0 → 0°, 1.0 → 180°
Accuracy   → Phase: Incorrect → 0°, Correct → 180°

Calibration gap = angular distance between confidence phase and accuracy phase
  0 = perfectly calibrated, 1 = worst case
```

### Usage

`add(claimed_confidence, actual_correct)` records one observation. The tester does the bookkeeping — bins, gap calculations, ECE / MCE aggregation. The per-bin breakdown is what tells you *where* the calibration is broken — overconfident in the 0.8-0.9 band, underconfident below 0.5, etc.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.calibration import CalibrationTester

tester = CalibrationTester()

# Add results (from a benchmark run)
tester.add(claimed_confidence=0.95, actual_correct=True)
tester.add(claimed_confidence=0.90, actual_correct=True)
tester.add(claimed_confidence=0.85, actual_correct=False)  # overconfident!
tester.add(claimed_confidence=0.30, actual_correct=False)
tester.add(claimed_confidence=0.60, actual_correct=True)

report = tester.report()
print(f"ECE (Expected Calibration Error): {report.ece:.4f}")
print(f"MCE (Max Calibration Error):      {report.mce:.4f}")
print(f"Mean confidence: {report.mean_confidence:.2f}")
print(f"Mean accuracy:   {report.mean_accuracy:.2f}")
print(f"Overconfidence ratio: {report.overconfidence_ratio:.0%}")

# Per-bin breakdown
for b in report.bins:
    print(f"  [{b.bin_lower:.1f}-{b.bin_upper:.1f}]: "
          f"conf={b.mean_confidence:.2f} acc={b.mean_accuracy:.2f} "
          f"gap={b.gap:.3f} (n={b.n_samples})")
```

### Enable in Config

The flag wires the tester into `bench.run`. The agent must surface its claimed confidence — typically as a number in the answer payload or in `SUTTrace.metadata`. An agent that does not produce confidence values cannot be calibrated; this tester surfaces that as zero records, not a false PASS.

```python
config = DOEHarnessConfig(
    enable_calibration_testing=True,
)
```

**What this tutorial leaves you with.** ECE / MCE numbers plus the per-bin breakdown showing which confidence range is miscalibrated. Add a tier check on `overconfidence_ratio` in Tutorial 8 to refuse release for agents that lie about their confidence even when their accuracy is acceptable.

> **API Reference:** [Calibration Testing](API_REFERENCE.md#31-calibration-testing) — `CalibrationTester`, `CalibrationPoint`, `CalibrationBin`, `CalibrationReport`

---

# Part VII: Multi-Source and Advanced Testing

---

## 28. Tutorial 25: Multi-Agent Consensus

> **Module:** `knowlytix.harness.testing.multi_agent`  
> **Import:** `from knowlytix.harness.testing import ConsensusOracle, AgentRole`  
> **Key Classes:** `ConsensusOracle`, `AgentRole`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Multi-Agent Testing** evaluates interactions between multiple agents using `tension_energy()` for inter-agent consistency and `check_holonomy()` for delegation path correctness. Single-agent testing (every tutorial above) grades one SUT in isolation; this tutorial probes the *interaction* between agents — do they agree about a shared entity, do they delegate correctly, do their conclusions cohere.

### When to Use

- Multiple agents collaborate on a task
- Testing consensus quality across agents with different knowledge bases
- Verifying delegation patterns in multi-agent systems

### Usage

Each `AgentRole` declares its system prompt, allowed tools, and (importantly) its own `store_path`. The agents must have *different* knowledge stores for the consensus check to be meaningful — two agents trained on the same store will agree by construction. The `ConsensusOracle` reads tension energy between the agents' answers to grade agreement.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.multi_agent import (
    AgentRole, ConsensusOracle, MultiAgentGenerator,
)

# Define agent roles
roles = {
    "analyst": AgentRole(
        role_id="analyst",
        system_prompt="You are a credit risk analyst.",
        allowed_tools=["data_lookup", "risk_check"],
        store_path="stores/analyst_store",
    ),
    "reviewer": AgentRole(
        role_id="reviewer",
        system_prompt="You are a model validation reviewer.",
        allowed_tools=["data_lookup", "gms_lookup"],
        store_path="stores/reviewer_store",
    ),
}

# Create consensus oracle
oracle = ConsensusOracle(
    stores={"analyst": analyst_store, "reviewer": reviewer_store},
    tension_threshold=0.5,
)

# Evaluate consensus on an entity
result = oracle.evaluate_consensus("MDL-001")
print(f"Consensus reached: {result.consensus_reached}")
print(f"Mean tension:      {result.mean_tension:.4f}")
print(f"Process holonomy:  {result.process_holonomy:.4f}")

for pair, tension in result.inter_agent_tension.items():
    print(f"  {pair}: tension={tension:.4f}")
```

### Enable in Config

`multi_agent_roles_path` points to a YAML carrying the same `AgentRole` declarations. The benchmark walks the role list, runs the consensus oracle on every shared entity, and emits per-pair tension scores into the typed verdicts.

```python
config = DOEHarnessConfig(
    enable_multi_agent=True,
    multi_agent_roles_path="roles.yaml",
)
```

**What this tutorial leaves you with.** Per-entity consensus verdicts plus inter-agent tension scores. Use this when the production system itself is multi-agent — the same harness measurements apply, just routed across roles instead of one SUT. Tutorial 26 (Cross-Document) is the structural cousin: same `tension_energy` primitive, applied to *documents* instead of *agents*.

> **API Reference:** [Multi-Agent Testing](API_REFERENCE.md#35-multi-agent-testing) — `ConsensusOracle`, `AgentRole`, `InteractionResult`, `MultiAgentGenerator`

---

## 29. Tutorial 26: Cross-Document Conflicts

> **Module:** `knowlytix.harness.testing.cross_document`  
> **Import:** `from knowlytix.harness.testing import CrossDocumentAnalyzer, DocumentConflict`  
> **Key Classes:** `CrossDocumentAnalyzer`, `DocumentConflict`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Cross-Document Analysis** detects factual conflicts between multiple document sources by computing `tension_energy()` on shared entities. Tutorial 25 graded multi-agent agreement; this tutorial grades multi-document agreement. Same primitive (tension energy on shared entities), different unit of analysis. Use it before ingestion to vet a corpus, or as a release-time check against a versioned baseline document.

### Conflict Classification (by tension)

| Tension | Classification | Meaning |
|---------|---------------|---------|
| > 1.5 | contradiction | Direct factual conflict |
| 0.7 – 1.5 | disagreement | Significant difference |
| < 0.7 | minor_variance | Acceptable variation |

### Usage

The analyzer takes a `dict` mapping document IDs to their stores. `find_shared_entities` returns entities that appear in more than one store; `find_conflicts` ranks the shared entities by inter-store tension and returns the top-N. Each `DocumentConflict` carries the conflicting facts so a reviewer can see *what* the documents disagree about, not just that they disagree.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.cross_document import CrossDocumentAnalyzer

analyzer = CrossDocumentAnalyzer(
    stores={"report_2024": store_a, "report_2025": store_b},
    tension_threshold=0.5,
)

# Find shared entities
shared = analyzer.find_shared_entities()
print(f"Shared entities: {len(shared)}")

# Detect conflicts
conflicts = analyzer.find_conflicts(max_conflicts=50)
print(f"Conflicts found: {len(conflicts)}")

for c in conflicts:
    print(f"\n  Entity: {c.entity}")
    print(f"  Store A ({c.store_a_id}): {c.fact_a}")
    print(f"  Store B ({c.store_b_id}): {c.fact_b}")
    print(f"  Tension: {c.inter_store_tension:.4f}")
    print(f"  Type:    {c.conflict_type}")
```

### As a Question Generator

The companion generator emits questions targeted at conflict zones — exactly the entities the analyzer flagged as having high inter-document tension. This is the testing equivalent of "drill into the disagreement" — if the SUT cannot answer cleanly when sources disagree, this generator surfaces those failure modes.

```python
from knowlytix.harness.testing.cross_document import CrossDocumentGenerator

gen = CrossDocumentGenerator()
questions = gen.generate(store)
# Generates questions targeting entities in conflict zones
```

### Enable in Config

`additional_store_paths` is the list of stores to compare against the primary one. The benchmark wires the analyzer automatically and emits cross-document conflict findings into the typed verdicts.

```python
config = DOEHarnessConfig(
    enable_cross_document=True,
    additional_store_paths=["stores/report_2024", "stores/report_2025"],
)
```

**What this tutorial leaves you with.** A conflict-classified ranking of cross-document disagreements plus a generator that probes the conflict zones. Run this before promoting a document into a production knowledge base — high-tension conflicts are typically either real corpus drift (legitimate) or ingestion bugs (must fix).

> **API Reference:** [Cross-Document Analysis](API_REFERENCE.md#36-cross-document-analysis) — `CrossDocumentAnalyzer`, `DocumentConflict`, `CrossDocumentGenerator`

---

## 30. Tutorial 27: Entity Disambiguation

> **Module:** `knowlytix.harness.testing.disambiguation`  
> **Import:** `from knowlytix.harness.testing import DisambiguationOracle, AmbiguousEntity`  
> **Key Classes:** `DisambiguationOracle`, `AmbiguousEntity`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Disambiguation Testing** probes the agent's ability to distinguish between confusable entities using `score_triple()` geodesic separation. Real-world entity references are messy — "MDL-001" vs "MDL001" vs "Model 1" all point at the same thing some of the time, and at different things the rest of the time. This tutorial grades the SUT's resolution.

### Usage

`evaluate(head, relation, candidate_a, candidate_b, claimed)` scores both candidates against the head/relation pair and checks which one is geodesically closer. The verdict passes only if the SUT's `claimed` candidate matches the closer one — confusion between two near-duplicate entities is a hallucination dressed up as ambiguity.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.disambiguation import DisambiguationOracle

oracle = DisambiguationOracle(store=store)

# Evaluate disambiguation between two candidates
result = oracle.evaluate(
    head="credit_model",
    relation="has_auc",
    candidate_a="MDL-001",
    candidate_b="MDL-002",
    claimed="MDL-001",
)

print(f"Geodesic to A: {result['geodesic_a']:.4f}")
print(f"Geodesic to B: {result['geodesic_b']:.4f}")
print(f"Correct:       {result['correct_candidate']}")
print(f"Claimed:       {result['claimed']}")
print(f"Passed:        {result['passed']}")
```

### As a Question Generator

The generator walks `fuzzy_match_entity` collisions in the store and synthesizes one question per ambiguous-pair found. Pairs the corpus does not have are pairs the generator cannot construct — corpus quality drives test quality here.

```python
from knowlytix.harness.testing.disambiguation import DisambiguationGenerator

gen = DisambiguationGenerator()
questions = gen.generate(store)
# Finds ambiguous surface forms via fuzzy_match_entity()
# and generates disambiguation questions
```

### Enable in Config

```python
config = DOEHarnessConfig(
    enable_disambiguation_testing=True,
)
```

**What this tutorial leaves you with.** A focused failure-mode probe — does the SUT confuse near-duplicate entities. Use this alongside Tutorial 23 (Hallucination): both produce CRITICAL severities by default, and together they cover the two ways the SUT can produce a wrong fact (one it never knew, vs. one it confused with a similar fact).

> **API Reference:** [Disambiguation Testing](API_REFERENCE.md#37-disambiguation-testing) — `DisambiguationOracle`, `AmbiguousEntity`, `DisambiguationGenerator`

---

## 31. Tutorial 28: Paraphrase Invariance

> **Module:** `knowlytix.harness.testing.invariance`  
> **Import:** `from knowlytix.harness.testing import InvarianceTester, InvarianceResult`  
> **Key Classes:** `InvarianceTester`, `InvarianceResult`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Invariance Testing** checks that the agent gives consistent answers to paraphrased versions of the same question, measured via `tension_energy()` between answer entities. A robust SUT should answer "What is the AUC?" and "Tell me the AUC" the same way; a brittle one will give different answers and the tension energy spike will catch it.

### Usage

`add_paraphrase_result(label, response, group)` records one phrasing of a question. The tester batches results by `group` and computes pairwise tension energy across the responses in each group; `evaluate(group)` reports the worst pair and a binary `invariant` verdict against `tension_threshold`.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.invariance import InvarianceTester

tester = InvarianceTester(store=store, tension_threshold=0.3)

# Add paraphrased responses (same question, different phrasing)
tester.add_paraphrase_result("original", "The test AUC is 0.8412", group="q1")
tester.add_paraphrase_result("formal",   "Test AUC = 0.8412",     group="q1")
tester.add_paraphrase_result("casual",   "AUC around 0.84",       group="q1")

result = tester.evaluate("q1")
print(f"Paraphrases:    {result.n_paraphrases}")
print(f"Mean tension:   {result.mean_tension:.4f}")
print(f"Max tension:    {result.max_tension:.4f}")
print(f"Invariant:      {result.invariant}")  # True if all tensions < threshold

for pair, tension in result.pairwise_tensions.items():
    print(f"  {pair}: {tension:.4f}")
```

### As a Question Generator

The generator emits questions in *groups* — the original phrasing and several paraphrases sharing the same ground truth. Each group becomes one invariance test: the SUT should give consistent answers across all members.

```python
from knowlytix.harness.testing.invariance import InvarianceGenerator

gen = InvarianceGenerator()
questions = gen.generate(store)
# Generates groups of paraphrased questions
```

### Enable in Config

```python
config = DOEHarnessConfig(
    enable_invariance_testing=True,
)
```

**What this tutorial leaves you with.** A robustness check on phrasing brittleness. A SUT that fails invariance tests is one whose accuracy depends on prompt formatting — a serious release blocker because operators cannot guarantee phrasing in production.

> **API Reference:** [Invariance Testing](API_REFERENCE.md#33-invariance-testing) — `InvarianceTester`, `InvarianceResult`, `InvarianceGenerator`

---

## 32. Tutorial 29: Streaming Consistency

> **Module:** `knowlytix.harness.testing.streaming`  
> **Import:** `from knowlytix.harness.testing import StreamingAdapter, StreamConsistencyOracle`  
> **Key Classes:** `StreamingAdapter`, `StreamConsistencyOracle`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Streaming Testing** evaluates streaming LLM outputs for intermediate plausibility, self-contradiction, and reasoning coherence — chunk by chunk. Non-streaming tests grade the final answer; streaming tests grade the *path* the SUT took to that answer. A SUT that arrives at the right answer after asserting the wrong fact midway is a different beast from one that gets there cleanly — this tutorial surfaces the difference.

### What It Measures

| Metric | GMS Primitive | Meaning |
|--------|---------------|---------|
| Per-chunk plausibility | `score_triple()` | Is each chunk factually grounded? |
| Stream tension | `tension_energy()` | Does the stream contradict itself? |
| Chunk holonomy | `check_holonomy()` | Is the reasoning path coherent? |

### Usage

`StreamingAdapter` consumes the chunked output of an LLM streaming API and emits a `StreamingTrace` carrying per-chunk plausibility, cross-chunk tension, and a holonomy defect over the reasoning sequence. `StreamConsistencyOracle.evaluate_stream` produces the binary "consistent / inconsistent" verdict on top of those raw scores.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.streaming import StreamingAdapter, StreamConsistencyOracle

adapter = StreamingAdapter(
    client=anthropic_client,
    model="claude-sonnet-4-20250514",
    store=store,
)

trace = adapter.evaluate(
    question="What is the test AUC of MDL-001?",
    context=document_context,
)

print(f"Chunks:               {trace.n_chunks}")
print(f"First token:          {trace.first_token_ms} ms")
print(f"Total latency:        {trace.total_latency_ms} ms")
print(f"Mean plausibility:    {trace.mean_chunk_plausibility:.4f}")
print(f"Stream tension:       {trace.stream_tension:.4f}")
print(f"Chunk holonomy:       {trace.chunk_holonomy:.4f}")

# Check for mid-stream contradiction
oracle = StreamConsistencyOracle(store=store)
verdict = oracle.evaluate_stream(trace)
print(f"Consistent: {verdict['consistent']}")
```

### Enable in Config

```python
config = DOEHarnessConfig(
    enable_streaming_testing=True,
)
```

**What this tutorial leaves you with.** Mid-stream plausibility, cross-chunk tension, and reasoning holonomy — the three measurements that turn streaming from a UX feature into a verifiable behavior. A SUT that flickers between contradictory statements mid-stream now produces a measurable, gate-able failure mode.

> **API Reference:** [Streaming Testing](API_REFERENCE.md#34-streaming-testing) — `StreamingAdapter`, `StreamingTrace`, `StreamConsistencyOracle`

---

## 33. Tutorial 30: Live Dashboard

> **Module:** `knowlytix.harness.testing.live_dashboard`  
> **Import:** `from knowlytix.harness.testing import DashboardServer, DashboardSnapshot`  
> **Key Classes:** `DashboardServer`, `DashboardSnapshot`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Live Dashboard** provides real-time monitoring of benchmark runs via a WebSocket server that pulls all data from the GMS tracing store. Every other tutorial in this guide produces signals that land in `bench.tracing`; the dashboard is the consumer that surfaces those signals to a human as the run unfolds, so an operator can spot an emerging failure before the run completes.

### Usage

The dashboard exposes two surfaces. `DashboardSnapshot.from_tracing(...)` is a one-shot read — a single dict of the current state, useful for CLI status checks. `DashboardServer` is the live surface — start it on a port, point a browser at the WebSocket, and the server pushes a fresh snapshot whenever the tracing store changes.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.testing.live_dashboard import DashboardServer, DashboardSnapshot

# One-shot snapshot (no server needed)
snapshot = DashboardSnapshot.from_tracing(bench.tracing)
print(f"Active runs:  {snapshot.active_runs}")
print(f"Completed:    {snapshot.completed_runs}")
print(f"Accuracy:     {snapshot.accuracy:.1%}")
print(f"Questions:    {snapshot.total_questions}")

# Start WebSocket server for real-time monitoring
server = DashboardServer(tracing=bench.tracing, port=8765)
server.start()
# Connect a browser to ws://localhost:8765 for live updates
```

### Enable in Config

```python
config = DOEHarnessConfig(
    enable_live_dashboard=True,
    live_dashboard_port=8765,
)
```

The server polls the tracing store and pushes `DashboardSnapshot` updates over WebSocket. No separate data pipeline is needed — all data comes from GMS tracing queries.

**What this tutorial leaves you with.** A live operator view over any in-progress benchmark or campaign, fed by the same tracing store the audit log reads. This closes the loop on the testing surface: every signal Tutorials 1-29 produced is now observable in real time, gate-able at the end, and replayable after the fact.

> **API Reference:** [Live Dashboard](API_REFERENCE.md#45-live-dashboard) — `DashboardServer`, `DashboardSnapshot`

---

## 33.5. Tutorial 31: Optimal-Design DOE (Sobol+Refine)

> **Module:** `graphdoe.paper_api`, `graphdoe.design_matrix`, `knowlytix.harness.testing.bridge`  
> **Import:** `import graphdoe`  
> **Key API:** `graphdoe.design(questions=..., factor_names=[...], n_runs=N, method="sobol+refine")`  
> **Companion Notebook:** [`tutorials/optimal_design_doe.ipynb`](tutorials/optimal_design_doe.ipynb)  
> **Prereqs:** [Tutorial 1](#4-tutorial-1-your-first-benchmark-run), a CUDA-capable GPU recommended (CPU works for small `n_runs`)

A standard benchmark passively measures average accuracy on a fixed test set. The harness can do something stronger: under the same API budget, identify **which contextual factors of the evaluation drive failures** — clarity, noise, role, instruction conflict, paraphrase depth, etc. That requires *informative* design points across the factor space, not just numerous ones.

### Paper-style design — `n_runs` LLM calls, not `n_questions × n_runs`

The paper (`graphdoe/papers/paper_jasa.tex`) treats `base_question` as **one of the categorical factors** in the joint design matrix (cardinality = number of graph-verified base questions). Each design row picks both *which question* to ask and *how to present it*. Total LLM calls = `n_runs`.

> ⚠️ **Note**: `DOEGMSBenchmark.run()` does an **outer-product expansion** (`n_base × n_runs` calls), which is *not* the paper-style design. For paper semantics use `graphdoe.design()` directly and iterate the rows yourself, as below.

`graphdoe.design()` defaults to `method="sobol+refine"`: Sobol initialization, then GPU gradient descent on the **φ_p space-filling criterion** (LBFGS for `n_runs ≤ 500`, Adam otherwise) with cardinality-weighted distances. Same `n_runs`, much better minimum-distance separation between design points, more informative logistic regression on factor effects.

**Headline result from the FinStructBench case study** (n = 880 runs, 21-factor mixed-cardinality design, Claude Sonnet 4 evaluator):

| Metric | Sobol | Sobol+Refine |
|---|---|---|
| φ_p ↓ | 1.952 | **0.826** (2.4× better) |
| min_dist ↑ | 0.512 | **1.325** (2.6× better) |
| Pseudo-R² (joint logistic) | 0.752 | **0.825** |
| AIC | 470.5 | **380.0** (Δ = −90.5) |
| ROC AUC | 0.981 | **0.990** |

Same 880 LLM calls, same model, same factor catalog — the refined design produces a logistic factor-effect model that explains substantially more of the systematic variation in correctness.

### Step 1: Build the design (no API calls)

`graphdoe.design()` accepts a list of question strings (or any objects with `.natural_language`) plus the contextual-factor selection from the catalog. It returns a `DesignMatrix` whose `matrix` DataFrame has one row per LLM call and the `question` column already chosen.

```python
import graphdoe

CORE_FACTORS = [
    "clarity", "noise", "role", "answer_format", "reasoning_cue",
    "context_scope", "entity_aliasing", "instruction_conflict",
    "complexity", "style",
]

# 30 placeholder questions just to demonstrate quality metrics
placeholder_qs = [f"What is the value of metric {i}?" for i in range(30)]
demo = graphdoe.design(
    questions=placeholder_qs,
    factor_names=CORE_FACTORS,
    n_runs=300,
    method="sobol+refine",
    seed=42,
)
print(demo.matrix.shape)        # (300, 11) — one row per LLM call
print(demo.quality())           # phi_p, min_dist, mean_nn, mir_avg/max, coverage
```

Compare against `method="sobol"` at the same `n_runs` and factors to confirm the φ_p / min_dist gain on your specific factor set before paying for any LLM calls.

### Step 2: Pull real graph-verified base questions from a GMS store

The harness's `bridge.generate_base_questions` walks the knowledge graph and emits questions whose ground truth is derivable from triple traversal — graph-verified, 100% correct by construction.

Under the hood it's a four-stage pipeline, all seed-deterministic:

1. **Enumerate** — each named generator walks the store's structures (triples, sibling pairs, closed loops, multi-hop paths) and emits every candidate it can synthesize.
2. **Cap** — within each category the candidate list is reduced to `max_per_category` by `random.Random(seed).sample`: uniform without replacement, no difficulty weighting, no stratification.
3. **Validate** — each survivor is round-tripped (`gms_answer_fn(store) == ground_truth`); anything the store can't reproduce exactly is dropped, so a category may end up with **fewer** than `max_per_category` items.
4. **Adapt** — surviving questions are wrapped as `FSBQuestion`s ready for the design wrapper.

In the joint design `base_question` is one categorical factor, so its level set must be unique by surface form. Different generators occasionally synthesize the same prompt when entities and relations overlap — that's why Step 2 ends in a dedupe.

```python
import torch
from collections import Counter
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig
from knowlytix.harness.testing.bridge import generate_base_questions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",        # reuse pre-trained store
    factor_group="core",                    # not used in paper-style flow but set
    n_runs=300,                             #   so DOEHarnessConfig validates
    method="sobol+refine",
    seed=42,
    enable_tracing=False,                   # disable per-query GMS retrain
    device=str(DEVICE),
)
bench = DOEGMSBenchmark(config)
store = bench.ingest()                      # loads existing store; doesn't retrain

base_qs = generate_base_questions(
    store=store,
    generator_names=["exact_retrieval", "counting", "cross_reference",
                     "multi_hop", "boundary"],
    max_per_category=10,                    # per-generator cap (uniform random, seed-fixed)
    seed=42,
)
# Per-category counts after Cap + Validate.  Any category below
# max_per_category had candidates dropped by _validate.
print("Post-validate per category:",
      dict(Counter(q.category for q in base_qs)))

# Dedupe by natural_language; categorical factor levels must be unique
seen, unique_base = set(), []
for q in base_qs:
    if q.natural_language not in seen:
        seen.add(q.natural_language); unique_base.append(q)
    if len(unique_base) >= 30:
        break
print(f"{len(unique_base)} unique base questions")
```

Some generators (`plausibility`, `tension`, `holonomy`, ...) require a calibrated geometric judge before they can score; for tutorial scope use the simpler set above.

### Step 3: Build the paper-style design with the real questions

```python
design = graphdoe.design(
    questions=unique_base,                  # accepts FSBQuestion-like objects too
    factor_names=CORE_FACTORS,              # 10 contextual factors
    n_runs=300,                             # 300 LLM calls
    method="sobol+refine",
    seed=42,
)
df = design.matrix                          # shape (300, 11)
print(design.quality())
```

### Step 4: Graph baseline — verify ground-truth integrity

Cheap (no LLM calls). Only need to check the **unique** base questions; rephrasing leaves ground truth invariant.

```python
from knowlytix.benchmark.scorers import score_answer

n_ok = sum(
    score_answer(q.graph_answer_fn(store.graph), q.ground_truth).correct
    for q in unique_base
)
assert n_ok == len(unique_base), "Graph baseline must be 100% — ground truth issue otherwise"
```

### Step 5: LLM run — iterate the design rows manually

Per row: look up the chosen base question, apply the row's 10 contextual-factor levels via `QuestionRephraser`, send to Claude, score against ground truth.

```python
import os, anthropic
from graphdoe.question_rephraser import QuestionRephraser
from knowlytix.harness.governance.api_key import get_anthropic_api_key

client = anthropic.Anthropic(api_key=get_anthropic_api_key())
rephraser = QuestionRephraser(method="template")
base_q_lookup = {q.natural_language: q for q in unique_base}
with open(config.markdown_path) as f:
    document_context = f.read()

factor_cols = [c for c in df.columns if c != "question"]
rows = []
for i, row in df.iterrows():
    base_q = base_q_lookup[row["question"]]
    factor_assignment = {c: row[c] for c in factor_cols}
    rephrased = rephraser.rephrase(base_q, factor_assignment)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content":
                   f"<document>\n{document_context}\n</document>\n\n{rephrased.llm_prompt}"}],
    )
    answer = resp.content[0].text
    rows.append({
        "row": i, "base_qid": base_q.qid, "category": base_q.category,
        "question": row["question"], "ground_truth": str(base_q.ground_truth),
        "correct": bool(score_answer(answer, base_q.ground_truth).correct),
        **factor_assignment,
    })
results_df = pd.DataFrame(rows)
print(f"LLM accuracy: {results_df['correct'].mean():.1%}")
```

### Step 6: Factor-effect analysis — single-factor + joint model + FDR

`DOEAnalyzer` covers single-factor logistic + Wilson CIs + odds ratios + interactions, plus `run_joint_model()` (AIC/BIC/ROC AUC/Hosmer-Lemeshow) and `apply_fdr_correction()` (Benjamini-Hochberg) — the paper's full analysis stack.

```python
from graphdoe.analysis import DOEAnalyzer

ana = DOEAnalyzer.from_dataframe(results_df, factors=factor_cols + ["question"])

# Single-factor analysis of deviance
deviance = ana.run_logistic(metric="correct")

# Benjamini-Hochberg FDR correction across the family of factor tests
deviance_adj = ana.apply_fdr_correction(deviance, alpha=0.05, method="fdr_bh")
print(deviance_adj.sort_values("p_value_adj")[
    ["factor", "p_value", "p_value_adj", "significant_adj", "pseudo_r2"]
].head(10))

# Joint model — the paper's primary design-comparison signal
joint = ana.run_joint_model(metric="correct")
# joint == {pseudo_r2, aic, bic, roc_auc, hl_p_value, n_params, n_obs}
print(f"Joint:  pseudo-R²={joint['pseudo_r2']:.4f}  AIC={joint['aic']:.1f}  "
      f"BIC={joint['bic']:.1f}  ROC AUC={joint['roc_auc']:.4f}  "
      f"H-L p={joint['hl_p_value']:.3f}")
```

The paper's headline finding: under FDR correction, both designs detect `base_question` and one or two contextual factors. But the **joint model fit is materially better with Sobol+Refine** (pseudo-R² 0.825 vs 0.752, ΔAIC = −90.5). To replicate that comparison, run Steps 3–6 twice — once with `method="sobol"`, once with `method="sobol+refine"` — and contrast the four joint-model metrics.

### Step 7: Scaling to paper settings

```python
design = graphdoe.design(
    questions=unique_base,           # bump unique_base to 44 questions
    factor_group="comprehensive",    # 20 contextual factors, not 10
    n_runs=880,                      # paper scale; ≈45 min, ~$50–80 on Claude Sonnet
    method="sobol+refine",
    seed=42,
)
```

Budget the LLM run separately. The optimizer auto-switches LBFGS (`n_runs ≤ 500`, float64) → Adam (`n_runs > 500`, float32, lr=0.005, p=50, 500 iters) to keep memory tractable. With CUDA the refinement step itself takes seconds-to-minutes.

### Reference

- **Paper:** `graphdoe/papers/paper_jasa.tex` — full methodology, theoretical properties (Fisher information / Schur complement, fill-distance), and the 880-run application
- **API:** [`graphdoe.paper_api.design()`](../../graphdoe/paper_api.py), [`graphdoe.design_matrix.DesignMatrix`](../../graphdoe/design_matrix.py)
- **Notebook:** [`tutorials/optimal_design_doe.ipynb`](tutorials/optimal_design_doe.ipynb) — end-to-end runnable version of this tutorial

---

# Reference

---

## 34. End-to-End Walkthrough

> **Module:** `knowlytix.harness.testing.claims`, `knowlytix.harness.testing.verifiers`, `knowlytix.harness.testing.verdict`  
> **Import:** `from knowlytix.harness.testing import ClaimCompiler, VerificationRouter, TypedVerdict`  
> **Key Classes:** `ClaimCompiler`, `VerificationRouter`, `TypedVerdict`

This puts all 7 workstreams together in a single flow.

```python
import torch
from knowlytix.harness.testing.config import DOEHarnessConfig
from knowlytix.harness.testing.benchmark import DOEGMSBenchmark
from knowlytix.harness.testing.gateway import ToolGateway, FaultProfile
from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule
from knowlytix.harness.testing.scenarios import Scenario, ScenarioRegistry
from knowlytix.harness.testing.runs import RunManager, RunState
from knowlytix.harness.testing.replay import RegressionSuite, compare_runs
from knowlytix.harness.testing.audit import AuditReporter, RiskTierProfile
from demo.mock_agent import MockHarness, create_mock_evaluator


# ── 1. Configure and Ingest ──────────────────────────────────
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="e2e_test/stores",
    factor_group="quick_screen",
    n_runs=4,
    generators=["exact_recall", "plausibility", "tension",
                 "exact_retrieval", "negative"],
    enable_typed_verdicts=True,
    enable_tracing=True,
    tracing_dir="e2e_test/tracing",
    snapshot_dir="e2e_test/snapshots",
    device="cpu",
)

bench = DOEGMSBenchmark(config)
store = bench.ingest()
print(f"Store: {store.stats()['entities']} entities, "
      f"{store.stats()['triples']} triples")


# ── 2. Set Up Gateway + Policy (WS1 + WS2) ──────────────────
policy = PolicyEngine(store=store, rules=[
    PolicyRule(rule_id="R1", action="submit_escalation",
              requirement="human_approval", rule_type="requires"),
])

gateway = ToolGateway(
    store=store,
    fault_profiles=[
        FaultProfile(tool_pattern="submit_escalation",
                     error_rate=1.0, error_message="Blocked by test"),
    ],
    policy_engine=policy,
    tracing=bench.tracing,
    run_id="e2e_run",
)


# ── 3. Create Scenarios (WS3) ────────────────────────────────
registry = ScenarioRegistry("e2e_test/scenarios", store=store)
registry.add(Scenario(
    name="MDL-001 AUC",
    question_text="What is the test AUC of model MDL-001?",
    ground_truth="0.8412", answer_type="float",
    tags=["accuracy"], risk_tier="action_taking",
    required_tools=["data_lookup"], max_tool_calls=3,
))
registry.add(Scenario(
    name="Risk Count",
    question_text="How many models are rated as High risk?",
    ground_truth="4", answer_type="int",
    tags=["compliance"], risk_tier="recommendation",
))


# ── 4. Managed Run (WS4) ─────────────────────────────────────
run_mgr = RunManager("e2e_test/runs", tracing=bench.tracing)
record = run_mgr.create_run(config)
run_mgr.transition(record.run_id, RunState.PREPARING)

evaluator = create_mock_evaluator()
bench.generate_questions()

run_mgr.transition(record.run_id, RunState.EXECUTING)
result = bench.run(evaluator=evaluator)

run_mgr.transition(record.run_id, RunState.EVALUATING)
# ... typed verdicts computed during run ...

run_mgr.transition(record.run_id, RunState.COMPLETED)
print(f"Run {record.run_id}: {result.accuracy:.1%}")


# ── 5. Promote Failures to Regression Suite (WS5) ────────────
suite = RegressionSuite("e2e_test/regression", scenario_registry=registry)
new_scenarios = suite.promote_failures(record, result)
print(f"Promoted {len(new_scenarios)} failures to regression scenarios")


# ── 6. Risk-Tiered Release Gates (WS7) ───────────────────────
import json, pathlib
reporter = AuditReporter(result, bench.typed_verdicts)
# Anchor geometric cuts to store/calibration.json (same artifact
# HallucinationOracle and ConsensusOracle consume).
cal = json.loads(
    (pathlib.Path(store.store_path) / "calibration.json").read_text()
)["thresholds"]
profiles = [
    RiskTierProfile.advisory(),
    RiskTierProfile.recommendation(),
    RiskTierProfile.action_taking(),
]
for p in profiles:
    p.max_tension  = cal["tension"]
    p.max_holonomy = cal["holonomy"]
gates = reporter.generate_tiered_release_gate(profiles)

for gate in gates:
    status = "PASS" if gate.passed else "FAIL"
    print(f"  {gate.tier.tier_name}: {status}")


# ── 7. Reports ────────────────────────────────────────────────
bench.generate_report(result, bench.analyze(result),
                      output_path="e2e_test/dashboard.html")
bench.generate_audit_report(result, bench.analyze(result),
                            output_path="e2e_test/audit.html")
bench.save_tracing()
print("Done! Check e2e_test/ for reports.")
```

> **API Reference:** This walkthrough uses classes from across the API. See the full [API Reference](API_REFERENCE.md) for constructor signatures and all available methods.

---

## 35. CLI Reference

> **Module:** `knowlytix.harness.testing.failures`  
> **Import:** `from knowlytix.harness.testing import FailureCode, FailureAttribution, TypedVerdict`  
> **Key Classes:** `FailureCode`, `FailureAttribution`

### Standard Benchmark Run

```bash
# Basic run
python -m knowlytix.harness.testing run report.md

# With all GMSH features
python -m knowlytix.harness.testing run report.md \
    --typed-verdicts \
    --provenance \
    --snapshot-dir snapshots/ \
    --audit-report audit.html \
    --release-gate accuracy=0.8 geo=0.5 \
    --risk-tiers tier_profiles.yaml \
    --baseline-run run_abc12345 \
    --n-runs 64 \
    --target-model claude-opus-4-7

# From YAML config
python -m knowlytix.harness.testing run --config config.yaml
```

### Run Management

```bash
# List all tracked runs
python -m knowlytix.harness.testing runs list

# Show details for a specific run
python -m knowlytix.harness.testing runs show run_abc12345
```

### Regression and Replay

```bash
# Promote failures from a run to regression scenarios
python -m knowlytix.harness.testing regression promote run_abc12345

# List regression scenarios
python -m knowlytix.harness.testing regression list

# Replay a run from a manifest
python -m knowlytix.harness.testing replay manifest.yaml

# Compare two runs
python -m knowlytix.harness.testing compare result_a.json result_b.json
```

### Backwards Compatibility

The original single-command syntax still works:

```bash
python -m knowlytix.harness.testing report.md --n-runs 32
```

---

## 36. Configuration Reference

> **Module:** `knowlytix.harness.testing.adapters`  
> **Import:** `from knowlytix.harness.testing import SUTAdapter, LLMAdapter, AgentAdapter`  
> **Key Classes:** `SUTAdapter`, `LLMAdapter`, `AgentAdapter`

All configuration is done via `DOEHarnessConfig`. Fields can be set in Python or loaded from YAML.

### YAML Config File

```yaml
# config.yaml
document:
  markdown_path: "path/to/report.md"
  stores_dir: "gms_stores"

doe:
  factor_group: "quick_screen"
  n_runs: 32
  method: "sobol"
  seed: 42

target:
  target_type: "agent"
  target_model: "claude-opus-4-7"

gmsh:
  enable_typed_verdicts: true
  enable_provenance: true
  snapshot_dir: "snapshots"
  semantic_mode: "fallback"

agentic:
  enable_gateway: true
  gateway_fault_profiles_path: "fault_profiles.yaml"
  enable_policy_engine: true
  policy_rules_path: "policy_rules.yaml"
  scenarios_dir: "scenarios"
  runs_dir: "runs"
  enable_process_evaluation: true
  enable_recovery_evaluation: true

release:
  risk_tier_profiles_path: "tier_profiles.yaml"
  baseline_run_id: ""
```

### Full Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `markdown_path` | str | `""` | Path to the markdown document to test against |
| `stores_dir` | str | `"gms_stores"` | Directory for GMS store files |
| `factor_group` | str | `"quick_screen"` | DOE factor group name |
| `n_runs` | int | `32` | Number of DOE design runs per base question |
| `method` | str | `"sobol"` | DOE method: `"sobol"`, `"lhs"`, `"grid"` |
| `seed` | int | `42` | Random seed for reproducibility |
| `generators` | list | `[]` | Question generator names (empty = all) |
| `max_per_category` | int | `10` | Max questions per generator category |
| `target_type` | str | `"llm"` | `"llm"` or `"agent"` |
| `target_model` | str | `"claude-sonnet-4-20250514"` | Model name for LLM evaluation |
| `enable_typed_verdicts` | bool | `False` | Enable claim-level verification |
| `enable_tracing` | bool | `True` | Enable GMS tracing store |
| `enable_gateway` | bool | `False` | Enable tool gateway |
| `enable_policy_engine` | bool | `False` | Enable policy engine |
| `enable_process_evaluation` | bool | `False` | Enable process evaluator |
| `enable_recovery_evaluation` | bool | `False` | Enable recovery evaluator |
| `enable_interaction_evaluation` | bool | `False` | Enable interaction evaluator |
| `snapshot_dir` | str | `""` | Snapshot directory (empty = off) |
| `scenarios_dir` | str | `""` | Scenario registry directory |
| `runs_dir` | str | `""` | Run records directory |
| `regression_suite_dir` | str | `""` | Regression suite directory |
| `risk_tier_profiles_path` | str | `""` | YAML path for risk tier profiles |
| `baseline_run_id` | str | `""` | Baseline run ID for regression comparison |
| **Black-Box Testing** | | | |
| `target_profile_path` | str | `""` | YAML path for TargetProfile |
| `environment_yaml_path` | str | `""` | YAML path for EnvironmentSimulator |
| `campaigns_dir` | str | `""` | Directory for campaign configs/results |
| `enable_severity_classification` | bool | `False` | Enable severity classification |
| `safety_config_path` | str | `""` | YAML path for SafetyConfig |
| **Performance & Cost** | | | |
| `enable_performance_profiling` | bool | `False` | Enable latency SLA profiling |
| `sla_latency_ms` | float | `5000.0` | SLA target in milliseconds |
| `enable_cost_tracking` | bool | `False` | Enable token/cost tracking |
| `cost_budget_usd` | float | `10.0` | Budget cap in USD |
| **Behavioral Testing** | | | |
| `enable_stateful_testing` | bool | `False` | Enable memory probe testing |
| `stateful_session_count` | int | `5` | Number of teach-test sessions |
| `enable_hallucination_testing` | bool | `False` | Enable hallucination detection |
| `hallucination_tau_override` | float \| None | `None` | Override $\tau_{\text{geo}}$; default reads `store/calibration.json` |
| `enable_calibration_testing` | bool | `False` | Enable confidence calibration |
| **Multi-Source & Advanced** | | | |
| `enable_multi_agent` | bool | `False` | Enable multi-agent consensus |
| `multi_agent_roles_path` | str | `""` | YAML path for AgentRole definitions |
| `enable_cross_document` | bool | `False` | Enable cross-document conflict testing |
| `additional_store_paths` | list | `[]` | Paths to additional GMS stores |
| `enable_disambiguation_testing` | bool | `False` | Enable entity disambiguation |
| `enable_invariance_testing` | bool | `False` | Enable paraphrase invariance |
| `enable_streaming_testing` | bool | `False` | Enable streaming output testing |
| `enable_live_dashboard` | bool | `False` | Enable WebSocket live dashboard |
| `live_dashboard_port` | int | `8765` | Live dashboard port |
| `device` | str | `""` | `"cpu"`, `"cuda"`, or `""` (auto) |

> **API Reference:** [DOEHarnessConfig](API_REFERENCE.md#doeharnessconfig) documents all 60+ fields with types and defaults.

---

## 37. Troubleshooting

> **Module:** `knowlytix.harness.testing.audit`  
> **Import:** `from knowlytix.harness.testing import AuditReporter, TieredReleaseGate, RiskTierProfile`  
> **Key Classes:** `AuditReporter`, `TieredReleaseGate`

### CUDA Assertion Errors

**Symptom:** `RuntimeError: CUDA error: device-side assert triggered` during tracing store training.

**Fix:** Use CPU for the tracing store. Set `device="cpu"` in your config, or:

```python
config = DOEHarnessConfig(..., device="cpu")
```

GPU can still be used for the main GMS store by creating it separately.

### Index Out of Bounds in Generators

**Symptom:** `IndexError` during question generation with `link_prediction` or `multi_hop_transport` generators.

**Fix:** These generators require larger stores. Either:
- Use a larger document
- Exclude these generators: `generators=["exact_recall", "threshold", "plausibility", "tension", "exact_retrieval", "negative"]`

### No Questions Generated

**Symptom:** `bench.generate_questions()` produces 0 questions.

**Fix:** Check that:
1. The document was ingested successfully (`store.stats()` shows entities > 0)
2. The specified generators match your store content
3. `max_per_category` is > 0

### Policy Engine Blocking Everything

**Symptom:** All tool calls are blocked by the policy engine.

**Fix:** Check your rule definitions:
- Rules with `context="*"` match all contexts
- Rules with `severity="error"` are hard blocks; use `"warning"` for log-only
- `rule_type="requires"` blocks unless the requirement is met
- Use `engine.check("tool_name", {})` to test rules before connecting to the gateway

### Replay Fails with "Snapshot Not Found"

**Symptom:** `replay_run()` raises `ValueError: Snapshot not found`.

**Fix:** Ensure the snapshot directory from the original run is still accessible. Snapshots are file-based — if the directory was deleted, the snapshot is gone. Use `snap_mgr.list_snapshots()` to see available snapshots.

### Import Errors

**Symptom:** `ModuleNotFoundError` when importing harness modules.

**Fix:** Ensure you're in the project root and the environment is activated:

```bash
cd ~/jupyterlab/kg-memory
source ~/cluster/spark-venv/bin/activate
python -c "from knowlytix.harness.testing import DOEGMSBenchmark; print('OK')"
```

---

## 38. Common Recipes

Fifteen self-contained recipes covering the most frequent GMSH workflows. Every recipe includes all imports, defines all variables inline, uses `device="cpu"` and the standard document path `knowlytix/harness/testing/tests/model_risk_assessment.md`.

---

### Recipe 1: Ingest and Run a Basic Benchmark

```python
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig

# Step 1: Configure the benchmark
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="recipe_stores",
    factor_group="quick_screen",
    n_runs=4,
    seed=42,
    device="cpu",
)

# Step 2: Create benchmark and ingest the document
bench = DOEGMSBenchmark(config)
store = bench.ingest()

# Step 3: Define a mock evaluator (returns a plausible answer string)
def mock_eval(question, context):
    return "0.8412"

# Step 4: Generate DOE questions and run
bench.generate_questions()
result = bench.run(evaluator=mock_eval)

# Step 5: Inspect results
print(f"Accuracy: {result.accuracy:.1%}")  # float, e.g. "82.4%"
print(f"Questions: {len(result.questions)}")  # int
```

---

### Recipe 2: Connect a Claude Agent as Evaluator

```python
import anthropic
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig

# Step 1: Create the Anthropic client
client = anthropic.Anthropic()

# Step 2: Wrap the client as an evaluator function
def claude_evaluator(question: str, context: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": f"{context}\n\n{question}"}],
    )
    return response.content[0].text

# Step 3: Configure and run the benchmark
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="recipe_stores",
    n_runs=2,
    seed=42,
    device="cpu",
)
bench = DOEGMSBenchmark(config)
bench.ingest()
bench.generate_questions()
result = bench.run(evaluator=claude_evaluator)

print(f"Accuracy: {result.accuracy:.1%}")  # float
print(f"Questions evaluated: {len(result.questions)}")  # int
```

---

### Recipe 3: Inject Tool Faults and Verify Recovery

```python
from knowlytix.harness.testing import FaultProfile, ToolGateway
from demo.mock_agent import MockHarness

# Step 1: Create a mock harness (simulates an LLM agent)
harness = MockHarness(recovery_mode="fallback")

# Step 2: Define a fault profile (50% error rate on gms_query)
fault = FaultProfile(
    tool_pattern="gms_query",
    error_rate=0.5,
    error_message="Simulated timeout",
    latency_ms=200.0,
)

# Step 3: Install the gateway on the harness
gw = ToolGateway(fault_profiles=[fault])
gw.install(harness)

# Step 4: Run a question through the harness
output = harness.run("What is the model validation approach?")

# Step 5: Inspect the gateway transcript for fault/recovery events
transcript = gw.get_transcript()
gw.uninstall(harness)

print(f"Agent output: {output[:80]}...")  # str
print(f"Gateway decisions: {len(transcript)}")  # int
for dec in transcript[:3]:
    print(f"  {dec.tool_name}: {dec.action} — {dec.reason}")  # GatewayDecision
```

---

### Recipe 4: Create Policy Rules from YAML

```python
import yaml
from knowlytix.harness.testing import PolicyEngine, PolicyRule
from knowlytix.knowledge.store import GMSExpertStore

# Step 1: Define policy rules inline as a YAML string
yaml_str = """
rules:
  - rule_id: "RULE-001"
    action: "submit_escalation"
    requirement: "human_approval"
    rule_type: "requires"
    context: "*"
    severity: "error"
  - rule_id: "RULE-002"
    action: "delete_*"
    requirement: "admin_context"
    rule_type: "blocks"
    context: "production"
    severity: "error"
"""

# Step 2: Parse YAML and build PolicyRule objects
raw = yaml.safe_load(yaml_str)
rules = [PolicyRule.from_dict(r) for r in raw["rules"]]

# Step 3: Create a GMS store and policy engine
store = GMSExpertStore.from_markdown(
    "knowlytix/harness/testing/tests/model_risk_assessment.md", device="cpu"
)
engine = PolicyEngine(store=store, rules=rules)

# Step 4: Check an action against the policy
decision = engine.check("submit_escalation", {"ticket": 42})
print(f"Allowed: {decision.allowed}")  # bool
print(f"Blocking rules: {decision.blocking_rules}")  # list[str]
print(f"Reason: {decision.reason}")  # str
```

---

### Recipe 5: Build a Regression Suite from Failures

```python
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig,
    RegressionSuite, ScenarioRegistry,
)

# Step 1: Run a benchmark that will produce some failures
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="recipe_stores",
    n_runs=2,
    seed=42,
    device="cpu",
)
bench = DOEGMSBenchmark(config)
bench.ingest()
bench.generate_questions()

def bad_eval(q, c):
    return "I don't know"

result = bench.run(evaluator=bad_eval)

# Step 2: Create a scenario registry and regression suite
registry = ScenarioRegistry(registry_dir="recipe_scenarios")
suite = RegressionSuite(
    suite_dir="recipe_regression",
    scenario_registry=registry,
)

# Step 3: Promote failed questions into regression scenarios
promoted_ids = suite.promote_failures(run_record=None, result=result)
print(f"Promoted {len(promoted_ids)} failures")  # int
print(f"Suite scenarios: {suite.list_scenarios()}")  # list[str]
```

---

### Recipe 6: Composite Oracle (Correctness + Trace)

```python
from knowlytix.harness.testing import (
    CompositeOracle, ExactOutcomeOracle, TraceOracle,
)
from knowlytix.knowledge.store import GMSExpertStore

# Step 1: Create a GMS store for geometric scoring
store = GMSExpertStore.from_markdown(
    "knowlytix/harness/testing/tests/model_risk_assessment.md", device="cpu"
)

# Step 2: Build sub-oracles with weights
exact_oracle = ExactOutcomeOracle(store=store)
trace_oracle = TraceOracle(store=store)

# Step 3: Compose them (0.6 correctness + 0.4 trace compliance)
composite = CompositeOracle(
    oracles=[
        (exact_oracle, 0.6),
        (trace_oracle, 0.4),
    ],
    threshold=0.5,
)

# Step 4: Evaluate (normally called with Scenario + SUTTrace objects;
# here we show the oracle type and structure)
print(f"Oracle type: {composite.oracle_type}")  # str: "composite"
print(f"Sub-oracles: {len(composite._oracles)}")  # int: 2
print(f"Threshold: {composite._threshold}")  # float: 0.5
```

---

### Recipe 7: Multi-Turn Memory Test

```python
from knowlytix.harness.testing import InteractionScript, TurnSpec, MultiTurnAdapter
from demo.mock_agent import MockHarness

# Step 1: Define three conversation turns with assertions
turns = [
    TurnSpec(
        turn_id="turn_1",
        user_message="Remember: the validation threshold is 0.95.",
        expected_contains=["0.95"],
    ),
    TurnSpec(
        turn_id="turn_2",
        user_message="What data was loaded in the last session?",
        expected_not_contains=["error"],
    ),
    TurnSpec(
        turn_id="turn_3",
        user_message="What is the validation threshold I told you?",
        expected_contains=["0.95"],
    ),
]

# Step 2: Build an interaction script
script = InteractionScript(
    script_id="memory_test_001",
    name="Threshold recall test",
    turns=turns,
)

# Step 3: Execute via MultiTurnAdapter
harness = MockHarness()
adapter = MultiTurnAdapter(harness=harness)
result = adapter.execute_script(script)

# Step 4: Inspect results (TurnResult.passed is a property, not .success)
for tr in result.turn_results:
    print(f"{tr.turn_id}: passed={tr.passed}, latency={tr.latency_ms:.0f}ms")
print(f"Overall: {result.overall_passed}")  # bool
print(f"Consistency scores: {result.consistency_scores}")  # dict
```

---

### Recipe 8: Detect Hallucinations

```python
from knowlytix.harness.testing import HallucinationOracle, geodesic_band
from knowlytix.knowledge.store import GMSExpertStore

# Step 1: Create a GMS store (expects store/calibration.json from Ch8 calibration)
store = GMSExpertStore.from_markdown(
    "knowlytix/harness/testing/tests/model_risk_assessment.md", device="cpu"
)

# Step 2: Create the hallucination oracle
#   tau_geodesic is loaded from store/calibration.json; override only for experiments.
oracle = HallucinationOracle(store=store)

# Step 3: Define sample claims as (head, relation, tail) triples
claims = [
    ("model_validation", "requires", "backtesting"),
    ("XGBoost", "invented_by", "Alan_Turing"),       # likely hallucination
    ("risk_assessment", "includes", "stress_testing"),
]

# Step 4: Assess all claims; returns a Chapter 10 TypedVerdict
verdict = oracle.assess_triples(claims)

# Step 5: Inspect per-claim SubVerdicts + failure attributions
print(f"Claims assessed: {len(verdict.sub_verdicts)}")
print(f"Failures:        {len(verdict.failures)}")
for sv in verdict.sub_verdicts:
    band = geodesic_band(sv.raw_score, oracle.tau_geodesic)
    print(f"  passed={sv.passed}, geodesic={sv.raw_score:.3f}, band={band}")
```

---

### Recipe 9: Compare Two Model Versions

```python
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig, compare_runs

# Step 1: Configure a shared benchmark setup
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="recipe_stores",
    n_runs=2,
    seed=42,
    device="cpu",
)

# Step 2: Run with model A (conservative answers)
bench_a = DOEGMSBenchmark(config)
bench_a.ingest()
bench_a.generate_questions()
result_a = bench_a.run(evaluator=lambda q, c: "0.85")

# Step 3: Run with model B (different behavior)
bench_b = DOEGMSBenchmark(config)
bench_b.ingest()
bench_b.generate_questions()
result_b = bench_b.run(evaluator=lambda q, c: "0.9500")

# Step 4: Compare the two runs
comparison = compare_runs(result_a, result_b, store=bench_a.store)

# Step 5: Inspect deltas and regressions
print(f"Accuracy delta: {comparison.accuracy_delta:+.4f}")  # float
print(f"Regressions: {len(comparison.regressions)}")  # int
print(f"Improvements: {len(comparison.improvements)}")  # int
print(f"Semantic drift: {comparison.semantic_drift:.4f}")  # float
```

---

### Recipe 10: Configure Risk-Tiered Release Gates

```python
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig,
    AuditReporter, RiskTierProfile,
)

# Step 1: Run a benchmark
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="recipe_stores",
    n_runs=2,
    seed=42,
    device="cpu",
    enable_typed_verdicts=True,
)
bench = DOEGMSBenchmark(config)
bench.ingest()
bench.generate_questions()
result = bench.run(evaluator=lambda q, c: "0.85")

# Step 2: Build three risk-tier profiles, overriding the factory
# geometric defaults with the store's calibrated thresholds.
import json, pathlib
cal = json.loads(
    (pathlib.Path(bench.store.store_path) / "calibration.json").read_text()
)["thresholds"]

profiles = [
    RiskTierProfile.advisory(),        # 70% accuracy, low bar
    RiskTierProfile.recommendation(),  # 80% accuracy, no hallucinations
    RiskTierProfile.action_taking(),   # 95% accuracy, strict thresholds
]
for p in profiles:
    p.max_tension  = cal["tension"]    # same tau used by ConsensusOracle
    p.max_holonomy = cal["holonomy"]

# Step 3: Create the audit reporter and evaluate gates
reporter = AuditReporter(result=result, typed_verdicts=bench.typed_verdicts)
gates = reporter.generate_tiered_release_gate(profiles)

# Step 4: Print results for each tier
for gate in gates:
    tier = gate.tier.tier_name
    print(f"{tier}: passed={gate.passed}, "
          f"blocking={gate.blocking_reasons}")  # TieredReleaseGate
```

---

### Recipe 11: Run Taxonomy Generators

```python
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig, taxonomy_generators
from knowlytix.knowledge.store import GMSExpertStore

# Step 1: Create a GMS store
store = GMSExpertStore.from_markdown(
    "knowlytix/harness/testing/tests/model_risk_assessment.md", device="cpu"
)

# Step 2: Get all five taxonomy generators
generators = taxonomy_generators()

# Step 3: Generate questions from each generator
counts = {}
for gen in generators:
    questions = gen.generate(store)
    counts[gen.category] = len(questions)

# Step 4: Print counts by category
total = sum(counts.values())
print(f"Total taxonomy questions: {total}")  # int
for category, n in counts.items():
    print(f"  {category}: {n}")  # str: int
# Categories: boundary, adversarial, robustness, recovery_test, governance
```

---

### Recipe 12: Environment Simulator from Code

```python
from knowlytix.harness.testing import EnvironmentSimulator, ToolEndpointSpec
from knowlytix.harness.testing.environment import StateTransition

# Step 1: Define a state transition for a "create_ticket" tool
transition = StateTransition(
    trigger_action="create_ticket",
    trigger_condition="",
    state_mutation={"last_ticket": "{{args.title}}", "ticket_count": 1},
    response_template="Ticket created: {{args.title}}",
)

# Step 2: Define the tool endpoint spec
spec = ToolEndpointSpec(
    tool_name="create_ticket",
    description="Creates a support ticket",
    state_transitions=[transition],
    initial_state={"ticket_count": 0, "last_ticket": ""},
    default_response="OK",
)

# Step 3: Build the environment simulator
env = EnvironmentSimulator(tool_specs=[spec])
env.reset(seed=42)

# Step 4: Handle a tool call and observe state changes
before = env.get_state()
response = env.handle_tool_call("create_ticket", {"title": "Bug #101"})
after = env.get_state()

print(f"Response: {response}")  # str
print(f"State after: {after}")  # dict
print(f"Diff: {env.get_state_diff(before, after)}")  # dict
```

---

### Recipe 13: Track Costs with Phase-Encoded SLA

```python
from knowlytix.harness.testing import CostTracker, CostSummary

# Step 1: Create a cost tracker with a $5 budget (no GMS store needed)
tracker = CostTracker(store=None, budget_usd=5.0)

# Step 2: Record several LLM calls
tracker.record("gms_query",   input_tokens=2000, output_tokens=500)
tracker.record("gms_search",  input_tokens=1500, output_tokens=300)
tracker.record("gms_lookup",  input_tokens=800,  output_tokens=100)

# Step 3: Get the cost summary
summary = tracker.summary()

# Step 4: Inspect budget tension and utilization
print(f"Total cost: ${summary.total_cost_usd:.4f}")  # float
print(f"Budget utilization: {summary.budget_utilization:.2%}")  # float
print(f"Budget tension: {summary.budget_tension:.4f}")  # float (0=under, ~1.41=at limit)
print(f"Within budget: {tracker.within_budget}")  # bool
print(f"Records: {summary.n_records}")  # int
print(f"Per-tool breakdown: {summary.per_tool}")  # dict[str, dict]
```

---

### Recipe 14: Parallel Campaign Execution

```python
from knowlytix.harness.testing import (
    CampaignConfig, CampaignManager, CampaignType,
    DOEGMSBenchmark, DOEHarnessConfig,
    ScenarioRegistry,
)
from knowlytix.harness.testing.runs import RunManager

# Step 1: Set up supporting infrastructure
registry = ScenarioRegistry(registry_dir="recipe_scenarios")
run_mgr = RunManager(runs_dir="recipe_runs")

# Step 2: Create the campaign manager
mgr = CampaignManager(
    campaigns_dir="recipe_campaigns",
    run_manager=run_mgr,
    scenario_registry=registry,
)

# Step 3: Define a campaign configuration
campaign_cfg = CampaignConfig(
    name="Release v2 validation",
    campaign_type=CampaignType.RELEASE_VALIDATION,
    risk_tier="recommendation",
    n_runs=2,
    base_config={
        "markdown_path": "knowlytix/harness/testing/tests/model_risk_assessment.md",
        "stores_dir": "recipe_stores",
        "device": "cpu",
        "seed": 42,
    },
)

# Step 4: Create and execute the campaign
campaign_id = mgr.create_campaign(campaign_cfg)

def bench_factory(cfg_dict):
    cfg = DOEHarnessConfig(**cfg_dict)
    return DOEGMSBenchmark(cfg)

result = mgr.execute_campaign(campaign_id, benchmark_factory=bench_factory)

print(f"Campaign: {result.campaign_id}")  # str
print(f"Accuracy: {result.aggregate_accuracy:.1%}")  # float
print(f"Runs: {len(result.run_summaries)}")  # int
```

---

### Recipe 15: Full Pipeline (Ingest, Test, Gate, Report)

```python
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig,
    AuditReporter, RiskTierProfile,
)

# Step 1: Configure with typed verdicts enabled
config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="recipe_stores",
    n_runs=4,
    seed=42,
    device="cpu",
    enable_typed_verdicts=True,
)

# Step 2: Ingest and generate
bench = DOEGMSBenchmark(config)
bench.ingest()
bench.generate_questions()

# Step 3: Run with a simple evaluator
result = bench.run(evaluator=lambda q, c: "0.85")

# Step 4: Collect typed verdicts
verdicts = bench.typed_verdicts  # dict[str, TypedVerdict]

# Step 5: Build the audit reporter
reporter = AuditReporter(result=result, typed_verdicts=verdicts)

# Step 6: Release gate check (single tier)
gate = reporter.generate_release_gate(
    accuracy_threshold=0.75,
    geo_confidence_threshold=0.40,
)
print(f"Release gate: {'PASS' if gate.passed else 'FAIL'}")  # str

# Step 7: Executive summary
summary = reporter.generate_executive_summary()
print(summary[:200])  # str (first 200 chars of summary text)
```
