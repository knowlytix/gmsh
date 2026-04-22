# GMSH Agentic Testing Harness — User Guide

A step-by-step guide to testing LLM agents with the GMSH platform. This guide walks you through every feature with runnable examples.

> **Companion document:** For complete constructor signatures, all dataclass fields, method return types and enumerations, see the [API Reference](API_REFERENCE.md). Each tutorial below links to the relevant API section.

---

## Table of Contents

### Part I: Getting Started
1. [What is GMSH?](#1-what-is-gmsh)
2. [Quick Start](#2-quick-start)
3. [Core Concepts](#3-core-concepts)

### Part II: Agentic Harness (WS1–WS7)
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

### Reference
34. [End-to-End Walkthrough](#34-end-to-end-walkthrough)
35. [CLI Reference](#35-cli-reference)
36. [Configuration Reference](#36-configuration-reference)
37. [Troubleshooting](#37-troubleshooting)
38. [Common Recipes](#38-common-recipes)

---

## 1. What is GMSH?

> **Module:** Overview  
> **Key Concepts:** GMS knowledge graph, geometric verification, DOE factor analysis  
> **Prereqs:** None — this is the starting point

GMSH (Geometric Memory System Harness) is an enterprise platform for **black-box testing of LLM agents**. It uses geometric knowledge graphs (GMS) as the ground-truth backbone to verify agent behavior across multiple dimensions:

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

> **Module:** `harness.doe.benchmark`, `harness.doe.config`  
> **Import:** `from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig, make_evaluator`  
> **Key Classes:** `DOEGMSBenchmark`, `DOEHarnessConfig`, `make_evaluator`  
> **Prereqs:** None — standalone quickstart

### Prerequisites

```bash
# Activate the project environment
source ~/cluster/spark-venv/bin/activate

# Verify the installation
python -c "from knowlytix.harness.doe import DOEGMSBenchmark; print('OK')"
```

### 30-Second Demo

Post-install, run the bundled smoke check to see every subsystem come up green:

```bash
knowlytix-smoke
```

This runs the 5-step post-install check (LLM config, license verify, GMS load,
MockHarness ingest, smoke notebook path). Full benchmark examples live in the
public tutorials at `knowlytix/gmsh/tutorials/` — see the tutorials README for
the current exercise list.

### Minimal Python Example

```python
from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig

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
| `score_triple(h, r, t)` | Geodesic distance on $S^{m-1}$ | Fact verification, plausibility (FM-1: Hallucination) |
| `tension_energy(a, b)` | Inner-product energy in u-space | Contradiction detection, recovery quality |
| `check_holonomy(path, direct)` | Frobenius defect of composed rotors | Multi-hop consistency, process compliance (FM-4) |
| `lookup_enm(category, entity_id)` | SHA-256-verified exact numeric value | Numeric precision, state corruption (FM-6) |
| `link_predict(h, r, top_k)` | Beam search on the manifold | Distractor generation, candidate ranking |
| `query_triples(h, r, t)` | Graph pattern matching | Structural lookup, tool/policy verification (FM-2/3) |

The store also exposes `fuzzy_match_entity(name)` for mapping free-text entity references to graph entities, used internally by the extended evaluators and oracles.

### DOE Design Matrix

Questions are expanded via a **Design of Experiments** matrix (Sobol, LHS, or Grid) that systematically varies factors like:

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

## 4. Tutorial 1: Your First Benchmark Run

> **Module:** `harness.doe.benchmark`, `harness.doe.config`  
> **Import:** `from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig, make_evaluator`  
> **Key Classes:** `DOEGMSBenchmark`, `DOEHarnessConfig`, `make_evaluator`  
> **Prereqs:** None — this is the starting tutorial

This tutorial walks through ingesting a document, generating questions, evaluating with a mock agent, and analyzing results.

### Step 1: Set Up Configuration

```python
import torch
from knowlytix.harness.doe.config import DOEHarnessConfig
from knowlytix.harness.doe.benchmark import DOEGMSBenchmark

config = DOEHarnessConfig(
    # Document to test against
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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

### Step 3: Create a Mock Evaluator

For testing purposes, use the built-in mock agent:

```python
from knowlytix.harness.fixtures import create_mock_evaluator

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

### Step 4: Generate Questions and Run

```python
doe_questions = bench.generate_questions()
print(f"Generated {len(doe_questions)} DOE questions")

result = bench.run(evaluator=evaluator)
print(f"\nResults:")
print(f"  Total:    {result.total_questions}")
print(f"  Correct:  {result.total_correct}")
print(f"  Accuracy: {result.accuracy:.1%}")
```

### Step 5: View Typed Verdicts

```python
for qid, tv in list(bench.typed_verdicts.items())[:3]:
    print(f"\n{qid}:")
    print(f"  Claims:  {tv.n_claims} ({tv.n_passed} passed, {tv.n_failed} failed)")
    print(f"  Confidence: {tv.overall_confidence:.3f}")
    for f in tv.failures:
        print(f"  FAILURE: {f.code.value} — {f.detail[:60]}")
```

### Step 6: Generate Reports

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

> **API Reference:** [DOEGMSBenchmark](API_REFERENCE.md#7-benchmark-orchestration) · [Claim Verification Pipeline](API_REFERENCE.md#12-claim-verification-pipeline) · [Mock Harness](API_REFERENCE.md#47-mock-harness)

---

## 5. Tutorial 2: Tool Gateway and Fault Injection

> **Module:** `harness.doe.gateway`  
> **Import:** `from knowlytix.harness.doe import ToolGateway, FaultProfile, GatewayDecision`  
> **Key Classes:** `ToolGateway`, `FaultProfile`, `GatewayDecision`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Tool Gateway** intercepts every tool call made by an agent, enabling fault injection, mock responses, and audit logging.

### When to Use

- Test agent resilience to tool failures (timeouts, errors)
- Simulate service degradation in staging environments
- Audit which tools the agent calls and in what order
- Block specific tools during testing

### Step 1: Define Fault Profiles

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.gateway import ToolGateway, FaultProfile

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

```python
gateway = ToolGateway(
    store=store,
    fault_profiles=[block_escalation, slow_modeva, mock_risk],
    tracing=bench.tracing,
    run_id="test_run_001",
)

# Install on a harness (MockHarness for testing, real Harness in production)
from knowlytix.harness.fixtures import MockHarness
mock = MockHarness(recovery_mode="fallback")
gateway.install(mock)
```

### Step 3: Run the Agent and Inspect

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

### Loading Fault Profiles from YAML

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
from knowlytix.harness.doe.gateway import load_fault_profiles
profiles = load_fault_profiles("fault_profiles.yaml")
```

> **API Reference:** [Tool Gateway](API_REFERENCE.md#15-tool-gateway) — `ToolGateway`, `FaultProfile`, `GatewayDecision`, `GatewayTranscript`

---

## 6. Tutorial 3: Policy Engine

> **Module:** `harness.doe.policy`  
> **Import:** `from knowlytix.harness.doe import PolicyEngine, PolicyRule, PolicyDecision`  
> **Key Classes:** `PolicyEngine`, `PolicyRule`, `PolicyDecision`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Policy Engine** enforces pre-execution rules using GMS geometric verification. Rules are encoded as GMS triples, so the same geometric primitives used for knowledge verification also enforce runtime constraints.

### When to Use

- Require human approval before certain actions (e.g., escalation)
- Block dangerous operations in specific contexts
- Enforce data prerequisites before tool execution
- Detect policy contradictions via geometric tension

### Step 1: Define Policy Rules

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.policy import PolicyEngine, PolicyRule

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

### Step 3: Integrate with Tool Gateway

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

### Loading Rules from YAML

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
from knowlytix.harness.doe.policy import load_policy_rules
rules = load_policy_rules("policy_rules.yaml")
engine = PolicyEngine(store=store, rules=rules)
```

### How the 4-Layer Check Works

When you call `engine.check("submit_escalation", args)`:

1. **Rule matching** — scans registered rules via `fnmatch` on action + context
2. **GMS graph lookup** — `query_triples(head="submit_escalation")` for encoded rules
3. **Geometric plausibility** — `score_triple("submit_escalation", "allowed_in", context)` checks if the action is geometrically plausible in the given scenario
4. **Contradiction detection** — `tension_energy(action, blocking_rule)` flags high-tension contradictions

> **API Reference:** [Policy Engine](API_REFERENCE.md#16-policy-engine) — `PolicyEngine`, `PolicyRule`, `PolicyDecision`

---

## 7. Tutorial 4: Authored Test Scenarios

> **Module:** `harness.doe.scenarios`  
> **Import:** `from knowlytix.harness.doe import Scenario, ScenarioRegistry, StopCondition`  
> **Key Classes:** `Scenario`, `ScenarioRegistry`, `StopCondition`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Scenario Registry** lets you define, version, and manage reusable test scenarios. Scenarios can be hand-written questions or references to GMS generators.

### When to Use

- Build curated test suites for specific capabilities
- Promote failures from previous runs into regression tests
- Tag and search scenarios by risk tier, category, or capability
- Version scenarios as requirements evolve

### Step 1: Create Scenarios

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.scenarios import Scenario, ScenarioRegistry

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

### Step 3: Version a Scenario

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

```python
scenarios = registry.list_all()
result = bench.run_scenarios(scenarios=scenarios, evaluator=evaluator)
print(f"Scenario results: {result.total_correct}/{result.total_questions}")
```

This converts scenarios to DOE questions, expands them through the design matrix, and runs the full evaluation pipeline.

> **API Reference:** [Scenario Registry](API_REFERENCE.md#17-scenario-registry) — `Scenario`, `ScenarioRegistry`, `StopCondition`

---

## 8. Tutorial 5: Run Lifecycle Management

> **Module:** `harness.doe.runs`  
> **Import:** `from knowlytix.harness.doe import RunManager, RunState, RunRecord`  
> **Key Classes:** `RunManager`, `RunState`, `RunRecord`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Run Manager** provides a formal state machine for benchmark runs, with validated transitions and persistence.

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

### Step-by-Step

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.runs import RunManager, RunState

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

> **API Reference:** [Run Management](API_REFERENCE.md#18-run-management) — `RunManager`, `RunRecord`, `RunState`

---

## 9. Tutorial 6: Replay and Regression Testing

> **Module:** `harness.doe.replay`  
> **Import:** `from knowlytix.harness.doe import ReplayManifest, RegressionSuite, compare_runs`  
> **Key Classes:** `ReplayManifest`, `RegressionSuite`, `compare_runs`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), completed runs (§8)

**Replay** lets you re-execute a benchmark run from a frozen snapshot. **Regression** automatically promotes failures into reusable test scenarios.

### When to Use

- Reproduce past failures for debugging
- Build regression suites from production incidents
- Compare agent performance across model versions
- Track whether fixes actually resolve issues

### Step 1: Compare Two Runs

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.replay import compare_runs

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

```python
from knowlytix.harness.doe.replay import RegressionSuite

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

```python
from knowlytix.harness.doe.replay import ReplayManifest

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
from knowlytix.harness.doe.replay import replay_run

loaded = ReplayManifest.from_yaml("my_test/replay_manifest.yaml")
replay_result, comparison = replay_run(
    loaded,
    snapshot_manager=bench.snapshots,
    evaluator=evaluator,
)

if comparison:
    print(f"vs. reference: {comparison.accuracy_delta:+.1%}")
```

> **API Reference:** [Replay & Regression](API_REFERENCE.md#19-replay--regression) — `ReplayManifest`, `RegressionSuite`, `RunComparison`, `compare_runs`

---

## 10. Tutorial 7: Extended Evaluators

> **Module:** `harness.doe.evaluators_extended`  
> **Import:** `from knowlytix.harness.doe import ProcessEvaluator, RecoveryEvaluator, HumanInteractionEvaluator`  
> **Key Classes:** `ProcessEvaluator`, `RecoveryEvaluator`, `HumanInteractionEvaluator`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

Three evaluators that test the **agentic** dimensions of LLM behavior beyond content accuracy.

### ProcessEvaluator — Tool Ordering and Completeness

Tests whether the agent follows the prescribed tool call sequence.

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.evaluators_extended import ProcessClaim, ProcessEvaluator

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

The **holonomy check** treats the tool call sequence as a path on the GMS manifold. If the actual path deviates from the expected direct path, the "process curvature" (holonomy defect) increases.

### RecoveryEvaluator — Fault Tolerance

Tests whether the agent maintains coherence after a tool failure.

```python
from knowlytix.harness.doe.evaluators_extended import RecoveryClaim, RecoveryEvaluator

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

### HumanInteractionEvaluator — Escalation Quality

Tests whether the agent's clarifications and escalations are appropriate.

```python
from knowlytix.harness.doe.evaluators_extended import InteractionClaim, HumanInteractionEvaluator

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

> **API Reference:** [Extended Evaluators](API_REFERENCE.md#29-extended-evaluators) — `ProcessEvaluator`, `RecoveryEvaluator`, `HumanInteractionEvaluator`

---

## 11. Tutorial 8: Risk-Tiered Release Gates

> **Module:** `harness.doe.audit`  
> **Import:** `from knowlytix.harness.doe import AuditReporter, TieredReleaseGate, RiskTierProfile`  
> **Key Classes:** `AuditReporter`, `TieredReleaseGate`, `RiskTierProfile`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

Risk-tiered gates evaluate your agent against different quality thresholds depending on the agent's risk level.

### The Three Tiers

| Tier | Use Case | Accuracy | Key Requirements |
|------|----------|----------|-----------------|
| **Advisory** | Information-only agents | ≥ 70% | Basic correctness |
| **Recommendation** | Agents that suggest actions | ≥ 80% | Zero hallucinations |
| **Action-Taking** | Agents that execute actions | ≥ 95% | Process + recovery evaluators, baseline comparison, ≤ 2% regression |

### Step 1: Evaluate Against Tiers

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.audit import (
    AuditReporter, RiskTierProfile, TieredReleaseGate
)

reporter = AuditReporter(result, bench.typed_verdicts)

gates = reporter.generate_tiered_release_gate([
    RiskTierProfile.advisory(),
    RiskTierProfile.recommendation(),
    RiskTierProfile.action_taking(),
])

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

### Step 2: Custom Tier Profiles

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

```python
summary = reporter.generate_tiered_executive_summary(gates, trends)
print(summary)
```

> **API Reference:** [Audit & Release Gates](API_REFERENCE.md#20-audit--release-gates) — `AuditReporter`, `RiskTierProfile`, `TieredReleaseGate`, `ReleaseGate`

---

---

# Part III: Black-Box Testing Platform

These modules extend GMSH from an agentic harness into a general-purpose **black-box testing platform** for arbitrary LLM systems.

---

## 12. Tutorial 9: Target Profiles

> **Module:** `harness.doe.target`  
> **Import:** `from knowlytix.harness.doe import TargetProfile, IsolationRequirements`  
> **Key Classes:** `TargetProfile`, `IsolationRequirements`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

A **Target Profile** formalizes the system under test (SUT) as a versioned, structured entity. This enables multi-SUT comparison, version drift detection, and audit traceability.

### When to Use

- Testing multiple models or agent versions side by side
- Tracking which SUT produced which results in audit reports
- Enforcing isolation requirements (network, filesystem, data)
- Campaign-based testing across target versions

### Define a Target

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.target import TargetProfile, IsolationRequirements

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

```python
# Save to YAML
target.to_yaml("my_test/target_profile.yaml")

# Reload later
loaded = TargetProfile.from_yaml("my_test/target_profile.yaml")
assert loaded.target_id == target.target_id
```

### Use in Config

```python
config = DOEHarnessConfig(
    target_profile_path="my_test/target_profile.yaml",
    # ... other settings
)
```

> **API Reference:** [Target Profiles](API_REFERENCE.md#24-target-profiles) — `TargetProfile`, `IsolationRequirements`

---

## 13. Tutorial 10: Environment Simulation

> **Module:** `harness.doe.environment`  
> **Import:** `from knowlytix.harness.doe import EnvironmentSimulator, ToolEndpointSpec, DataFixture, ClockControl`  
> **Key Classes:** `EnvironmentSimulator`, `ToolEndpointSpec`, `DataFixture`, `ClockControl`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Environment Simulator** provides a deterministic, resettable world that the SUT interacts with through simulated tool endpoints. The spec says: *"In black-box testing, environment control is more important than elaborate judging."*

### When to Use

- Test agent behavior under controlled conditions (no real API calls)
- Simulate tool failures, latency, and state transitions
- Verify side effects via state diffing
- Reproduce test conditions exactly with deterministic seeding

### Step 1: Define Tool Endpoints

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.environment import (
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

```python
snapshot = env.snapshot_state()
# ... run some tool calls that change state ...
env.restore_state(snapshot)  # back to checkpointed state
```

### Load from YAML

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

> **API Reference:** [Environment Simulator](API_REFERENCE.md#26-environment-simulator) — `EnvironmentSimulator`, `ToolEndpointSpec`, `StateTransition`, `FailureMode`, `ClockControl`, `DataFixture`

---

## 14. Tutorial 11: Oracle Framework

> **Module:** `harness.doe.oracles`  
> **Import:** `from knowlytix.harness.doe import CompositeOracle, ExactOutcomeOracle, TraceOracle, make_oracle`  
> **Key Classes:** `CompositeOracle`, `ExactOutcomeOracle`, `TraceOracle`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Oracles** determine pass/fail for black-box tests. GMSH provides seven oracle types that go far beyond string matching — they verify traces, side effects, constraints, and semantic correctness using GMS geometric primitives.

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

### Example: Constraint Oracle

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.oracles import ConstraintOracle

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

```python
from knowlytix.harness.doe.oracles import TraceOracle

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

```python
from knowlytix.harness.doe.oracles import SideEffectOracle

oracle = SideEffectOracle(environment=env, store=store)

# Verifies that the environment changed in expected ways
# Uses tension_energy() between expected and actual effect embeddings
verdict = oracle.evaluate(scenario, trace, before, after)
for key, detail in verdict.raw_scores.items():
    print(f"  {key}: {detail}")
```

### Example: Composite Oracle

```python
from knowlytix.harness.doe.oracles import CompositeOracle, ExactOutcomeOracle, TraceOracle

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

```python
from knowlytix.harness.doe.oracles import make_oracle

oracle = make_oracle(
    oracle_def={"type": "composite", "oracles": [
        {"type": "exact_outcome", "weight": 0.6},
        {"type": "trace", "weight": 0.4},
    ], "threshold": 0.5},
    store=store,
    environment=env,
)
```

> **API Reference:** [Oracle Framework](API_REFERENCE.md#27-oracle-framework) — `CompositeOracle`, `ExactOutcomeOracle`, `TraceOracle`, `SideEffectOracle`, `ConstraintOracle`, `make_oracle`

---

## 15. Tutorial 12: Severity Classification

> **Module:** `harness.doe.severity`, `harness.doe.failures`  
> **Import:** `from knowlytix.harness.doe import SeverityClassifier, SeverityLevel, FailureCode`  
> **Key Classes:** `SeverityClassifier`, `SeverityLevel`, `FailureCode`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Severity Classifier** assigns one of four levels — Critical, High, Moderate, Low — to each failure using failure code, risk tier, and GMS geometric signals.

### When to Use

- Prioritize which failures to fix first
- Gate releases on zero-critical-failure policies
- Report severity distributions to stakeholders
- Modulate severity using geometric distance (farther from known facts = more severe)

### Step 1: Classify Failures

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.severity import SeverityClassifier, SeverityLevel
from knowlytix.harness.doe.failures import FailureCode

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

> **API Reference:** [Severity Classification](API_REFERENCE.md#25-severity-classification) · [Failure Attribution](API_REFERENCE.md#13-failure-attribution)

---

## 16. Tutorial 13: Campaign Management

> **Module:** `harness.doe.campaign`  
> **Import:** `from knowlytix.harness.doe import CampaignManager, CampaignConfig, CampaignType`  
> **Key Classes:** `CampaignManager`, `CampaignConfig`, `CampaignType`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), evaluator (§4)

A **Campaign** groups multiple benchmark runs under a single evaluation purpose — release validation, regression sweep, model comparison, or compliance certification.

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

### Step 1: Configure a Campaign

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.campaign import CampaignConfig, CampaignType, CampaignManager

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

```python
comparison = mgr.compare_campaigns("cmp_abc12345", "cmp_def67890")
print(f"Accuracy delta: {comparison['accuracy_delta']:+.1%}")
print(f"Regressions:    {comparison['regressions']}")
```

> **API Reference:** [Campaign Management](API_REFERENCE.md#28-campaign-management) — `CampaignManager`, `CampaignConfig`, `CampaignType`, `CampaignResult`

---

# Part IV: Extended Test Generation

---

## 17. Tutorial 14: Taxonomy Generators

> **Module:** `harness.doe.generators_taxonomy`  
> **Import:** `from knowlytix.harness.doe import BoundaryGenerator, AdversarialGenerator, taxonomy_generators`  
> **Key Classes:** `BoundaryGenerator`, `AdversarialGenerator`, `RobustnessGenerator`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

Five **taxonomy generators** systematically probe failure modes that standard question generators miss: boundaries, adversarial inputs, robustness, recovery, and governance.

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

### Use in Configuration

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

```python
from knowlytix.harness.doe.generators_taxonomy import (
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

> **API Reference:** [Taxonomy Generators](API_REFERENCE.md#9-taxonomy-generators) — `BoundaryGenerator`, `AdversarialGenerator`, `RobustnessGenerator`, `RecoveryTestGenerator`, `GovernanceTestGenerator`

---

## 18. Tutorial 15: Multi-Turn Testing

> **Module:** `harness.doe.multiturn`  
> **Import:** `from knowlytix.harness.doe import MultiTurnAdapter, InteractionScript, TurnSpec`  
> **Key Classes:** `MultiTurnAdapter`, `InteractionScript`, `TurnSpec`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), evaluator (§4)

**Multi-turn interaction scripts** test conversation-level behavior: state carryover, per-turn assertions, and cross-turn consistency.

### When to Use

- Agent behavior depends on conversation history
- Testing multi-step workflows (teach → test → verify)
- Checking cross-turn consistency (entity from turn 1 recalled in turn 3)
- Validating per-turn tool expectations

### Step 1: Define an Interaction Script

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.multiturn import (
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

> **API Reference:** [Multi-Turn Interactions](API_REFERENCE.md#38-multi-turn-interactions) — `MultiTurnAdapter`, `InteractionScript`, `TurnSpec`, `ScriptResult`, `TurnResult`

---

## 19. Tutorial 16: Safety and Isolation

> **Module:** `harness.doe.safety`  
> **Import:** `from knowlytix.harness.doe import SafetyEnforcer, SafetyConfig, SyntheticDataFramework`  
> **Key Classes:** `SafetyEnforcer`, `SafetyConfig`, `SyntheticDataFramework`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Safety Enforcer** provides runtime protection — blocking destructive actions, enforcing sandbox mode, and generating synthetic test data.

### When to Use

- Testing agents that might make real API calls or file writes
- Enforcing network isolation in CI/CD pipelines
- Generating synthetic entities from GMS geometry (no real data exposure)
- Blocking destructive tool patterns (`delete_*`, `drop_*`)

### Step 1: Configure Safety

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.safety import SafetyConfig, SafetyEnforcer

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

```python
from knowlytix.harness.doe.safety import SyntheticDataFramework

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

> **API Reference:** [Safety Framework](API_REFERENCE.md#39-safety-framework) — `SafetyEnforcer`, `SafetyConfig`, `SyntheticDataFramework`

---

# Part V: Scale and Artifact Management

---

## 20. Tutorial 17: Parallel Execution

> **Module:** `harness.doe.parallel`  
> **Import:** `from knowlytix.harness.doe import ParallelExecutor, ParallelConfig`  
> **Key Classes:** `ParallelExecutor`, `ParallelConfig`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2), evaluator (§4)

Run benchmark evaluations concurrently to reduce wall time.

### When to Use

- Large test suites (100+ questions) taking too long sequentially
- Target SUT supports concurrent requests
- CI/CD pipelines with time constraints

### Configuration

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.parallel import ParallelConfig, ParallelExecutor

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

> **API Reference:** [Parallel Execution](API_REFERENCE.md#40-parallel-execution) — `ParallelExecutor`, `ParallelConfig`

---

## 21. Tutorial 18: Artifact Versioning

> **Module:** `harness.doe.versioning`  
> **Import:** `from knowlytix.harness.doe import ArtifactRegistry, VersionedArtifact`  
> **Key Classes:** `ArtifactRegistry`, `VersionedArtifact`  
> **Prereqs:** None (standalone)

**Artifact Versioning** tracks fault profiles, policy rules, and fixtures with SHA-256 integrity hashing and version history.

### When to Use

- Reproducing past test configurations exactly
- Auditing which artifact versions were used in which runs
- Diffing artifact changes between versions
- Regulatory compliance requiring traceable test configurations

### Step 1: Create and Register Artifacts

```python
from knowlytix.harness.doe.versioning import (
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

```python
# Policy rules
policy_reg = PolicyRuleRegistry("my_test/artifacts/policies")

# Fixtures
fixture_reg = FixtureRegistry("my_test/artifacts/fixtures")
```

All registries share the same `ArtifactRegistry` interface (`add`, `get`, `list_versions`, `update`).

> **API Reference:** [Versioning](API_REFERENCE.md#43-versioning) — `ArtifactRegistry`, `VersionedArtifact`, `FaultProfileRegistry`, `PolicyRuleRegistry`, `FixtureRegistry`

---

## 22. Tutorial 19: Browser and UI Testing

> **Module:** `harness.doe.browser_adapter`  
> **Import:** `from knowlytix.harness.doe import BrowserAdapter, UIScript, BrowserConfig`  
> **Key Classes:** `BrowserAdapter`, `UIScript`, `BrowserConfig`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

The **Browser Adapter** enables testing agentic systems with web UIs using Playwright.

### When to Use

- Testing agents with browser-based interfaces
- Verifying UI-driven workflows (navigate, click, type, assert)
- Taking screenshots for visual regression testing

### Step 1: Define a UI Script

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.browser_adapter import (
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

> **API Reference:** [Browser Adapter](API_REFERENCE.md#46-browser-adapter) — `BrowserAdapter`, `BrowserConfig`, `UIScript`, `UIStep`

---

# Part VI: GMS-Native Behavioral Testing

These modules use GMS geometric primitives — phase encoding on S^1, geodesic distance, tension energy — as the primary measurement mechanism, making them unique to GMSH.

---

## 23. Tutorial 20: Performance Profiling

> **Module:** `harness.doe.profiling`  
> **Import:** `from knowlytix.harness.doe import PerformanceProfiler, PerformanceReport`  
> **Key Classes:** `PerformanceProfiler`, `PerformanceReport`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Performance Profiling** encodes latency as a phase angle on S^1 and measures SLA compliance via tension energy with the SLA target phase.

### How It Works

```
Latency → Phase mapping (S^1):
  0 ms     → 0°     (instant)
  SLA ms   → 90°    (at threshold)
  2×SLA ms → 180°   (severe violation, clamped)

SLA tension = 2·sin(|θ_observed - θ_SLA| / 2)
  Range [0, 2]: 0 = within SLA, ~2 = severe violation
```

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.profiling import PerformanceProfiler

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

```python
config = DOEHarnessConfig(
    enable_performance_profiling=True,
    sla_latency_ms=5000.0,
)
```

> **API Reference:** [Performance Profiling](API_REFERENCE.md#41-performance-profiling) — `PerformanceProfiler`, `PerformanceRecord`, `PerformanceReport`

---

## 24. Tutorial 21: Cost and Token Tracking

> **Module:** `harness.doe.cost_tracker`  
> **Import:** `from knowlytix.harness.doe import CostTracker, CostSummary`  
> **Key Classes:** `CostTracker`, `CostSummary`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Cost Tracking** encodes cumulative cost as a phase angle on S^1. Budget compliance is measured via tension with the budget phase.

### How It Works

```
Cost → Phase mapping (S^1):
  $0        → 0°     (no spend)
  Budget    → 90°    (at budget)
  2×Budget  → 180°   (over budget, clamped)

Budget tension = 2·sin(|θ_cumulative - θ_budget| / 2)
```

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.cost_tracker import CostTracker

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

```python
config = DOEHarnessConfig(
    enable_cost_tracking=True,
    cost_budget_usd=10.0,
)
```

> **API Reference:** [Cost Tracking](API_REFERENCE.md#42-cost-tracking) — `CostTracker`, `CostRecord`, `CostSummary`

---

## 25. Tutorial 22: Agent Memory Testing

> **Module:** `harness.doe.stateful`  
> **Import:** `from knowlytix.harness.doe import StatefulAgentTester, AgentMemoryProbe, MemoryAssessment`  
> **Key Classes:** `StatefulAgentTester`, `AgentMemoryProbe`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Stateful Agent Testing** probes memory retention across sessions using teach-then-test patterns scored by GMS geodesic distance.

### How It Works

1. **Teach phase**: Feed the agent a fact triple (e.g., *"MDL-001 has test AUC 0.8412"*)
2. **Delay**: Insert distractor turns to test retention
3. **Recall phase**: Ask the agent to recall the fact
4. **Score**: `score_triple()` geodesic measures recall accuracy; `tension_energy()` measures session drift

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.stateful import StatefulAgentTester, MemoryRecallOracle

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

```python
from knowlytix.harness.doe.stateful import StatefulAgentGenerator

gen = StatefulAgentGenerator()
questions = gen.generate(store)
# Produces teach-test question pairs
```

### Enable in Config

```python
config = DOEHarnessConfig(
    enable_stateful_testing=True,
    stateful_session_count=5,
)
```

> **API Reference:** [Stateful Agent Testing](API_REFERENCE.md#32-stateful-agent-testing) — `StatefulAgentTester`, `AgentMemoryProbe`, `MemoryAssessment`, `MemoryRecallOracle`

---

## 26. Tutorial 23: Hallucination Detection

> **Module:** `harness.doe.hallucination`  
> **Import:** `from knowlytix.harness.doe import HallucinationOracle, HallucinationReport`  
> **Key Classes:** `HallucinationOracle`, `HallucinationReport`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Hallucination Detection** identifies fabricated claims by measuring their geodesic distance from known entities in the GMS manifold. Claims far from the manifold surface are hallucinations.

### Severity Tiers (by geodesic distance)

| Geodesic | Severity | Meaning |
|----------|----------|---------|
| > 1.5 | fabrication | Entirely invented content |
| 0.7 – 1.5 | distortion | Significantly altered facts |
| 0.5 – 0.7 | embellishment | Minor additions beyond source |
| < 0.5 | grounded | Consistent with known facts |

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.hallucination import HallucinationOracle

oracle = HallucinationOracle(store=store, geodesic_threshold=0.5)

# Assess extracted claims as (head, relation, tail) triples
report = oracle.assess_triples([
    ("MDL-001", "has_test_auc", "0.9500"),
    ("MDL-001", "trained_on", "10M records"),
])

print(f"Claims analyzed:    {report.n_claims}")
print(f"Hallucinated:       {report.n_hallucinated}")
print(f"Hallucination rate: {report.hallucination_rate:.0%}")
print(f"Mean geodesic:      {report.mean_geodesic:.4f}")
print(f"Max geodesic:       {report.max_geodesic:.4f}")
print(f"Severity counts:    {report.severity_counts}")

for a in report.assessments:
    if a.severity != "grounded":
        print(f"\n  HALLUCINATION [{a.severity}]:")
        print(f"    Claim: ({a.head}, {a.relation}, {a.tail})")
        print(f"    Geodesic: {a.geodesic_distance:.4f}")
        print(f"    Tension vs nearest: {a.tension_vs_nearest:.4f}")
```

### As a Question Generator

```python
from knowlytix.harness.doe.hallucination import HallucinationGenerator

gen = HallucinationGenerator()
questions = gen.generate(store)
# Three strategies: boundary entities, near-miss entities, open-ended elaboration
```

### Enable in Config

```python
config = DOEHarnessConfig(
    enable_hallucination_testing=True,
    hallucination_geodesic_threshold=1.0,
)
```

> **API Reference:** [Hallucination Testing](API_REFERENCE.md#30-hallucination-testing) — `HallucinationOracle`, `HallucinationAssessment`, `HallucinationReport`, `HallucinationGenerator`

---

## 27. Tutorial 24: Confidence Calibration

> **Module:** `harness.doe.calibration`  
> **Import:** `from knowlytix.harness.doe import CalibrationTester, CalibrationReport`  
> **Key Classes:** `CalibrationTester`, `CalibrationReport`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Calibration Testing** checks whether an agent's claimed confidence matches its actual correctness using S^1 phase encoding.

### How It Works

```
Confidence → Phase: 0 → 0°, 1.0 → 180°
Accuracy   → Phase: Incorrect → 0°, Correct → 180°

Calibration gap = angular distance between confidence phase and accuracy phase
  0 = perfectly calibrated, 1 = worst case
```

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.calibration import CalibrationTester

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

```python
config = DOEHarnessConfig(
    enable_calibration_testing=True,
)
```

> **API Reference:** [Calibration Testing](API_REFERENCE.md#31-calibration-testing) — `CalibrationTester`, `CalibrationPoint`, `CalibrationBin`, `CalibrationReport`

---

# Part VII: Multi-Source and Advanced Testing

---

## 28. Tutorial 25: Multi-Agent Consensus

> **Module:** `harness.doe.multi_agent`  
> **Import:** `from knowlytix.harness.doe import ConsensusOracle, AgentRole`  
> **Key Classes:** `ConsensusOracle`, `AgentRole`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Multi-Agent Testing** evaluates interactions between multiple agents using `tension_energy()` for inter-agent consistency and `check_holonomy()` for delegation path correctness.

### When to Use

- Multiple agents collaborate on a task
- Testing consensus quality across agents with different knowledge bases
- Verifying delegation patterns in multi-agent systems

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.multi_agent import (
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

```python
config = DOEHarnessConfig(
    enable_multi_agent=True,
    multi_agent_roles_path="roles.yaml",
)
```

> **API Reference:** [Multi-Agent Testing](API_REFERENCE.md#35-multi-agent-testing) — `ConsensusOracle`, `AgentRole`, `InteractionResult`, `MultiAgentGenerator`

---

## 29. Tutorial 26: Cross-Document Conflicts

> **Module:** `harness.doe.cross_document`  
> **Import:** `from knowlytix.harness.doe import CrossDocumentAnalyzer, DocumentConflict`  
> **Key Classes:** `CrossDocumentAnalyzer`, `DocumentConflict`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Cross-Document Analysis** detects factual conflicts between multiple document sources by computing `tension_energy()` on shared entities.

### Conflict Classification (by tension)

| Tension | Classification | Meaning |
|---------|---------------|---------|
| > 1.5 | contradiction | Direct factual conflict |
| 0.7 – 1.5 | disagreement | Significant difference |
| < 0.7 | minor_variance | Acceptable variation |

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.cross_document import CrossDocumentAnalyzer

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

```python
from knowlytix.harness.doe.cross_document import CrossDocumentGenerator

gen = CrossDocumentGenerator()
questions = gen.generate(store)
# Generates questions targeting entities in conflict zones
```

### Enable in Config

```python
config = DOEHarnessConfig(
    enable_cross_document=True,
    additional_store_paths=["stores/report_2024", "stores/report_2025"],
)
```

> **API Reference:** [Cross-Document Analysis](API_REFERENCE.md#36-cross-document-analysis) — `CrossDocumentAnalyzer`, `DocumentConflict`, `CrossDocumentGenerator`

---

## 30. Tutorial 27: Entity Disambiguation

> **Module:** `harness.doe.disambiguation`  
> **Import:** `from knowlytix.harness.doe import DisambiguationOracle, AmbiguousEntity`  
> **Key Classes:** `DisambiguationOracle`, `AmbiguousEntity`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Disambiguation Testing** probes the agent's ability to distinguish between confusable entities using `score_triple()` geodesic separation.

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.disambiguation import DisambiguationOracle

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

```python
from knowlytix.harness.doe.disambiguation import DisambiguationGenerator

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

> **API Reference:** [Disambiguation Testing](API_REFERENCE.md#37-disambiguation-testing) — `DisambiguationOracle`, `AmbiguousEntity`, `DisambiguationGenerator`

---

## 31. Tutorial 28: Paraphrase Invariance

> **Module:** `harness.doe.invariance`  
> **Import:** `from knowlytix.harness.doe import InvarianceTester, InvarianceResult`  
> **Key Classes:** `InvarianceTester`, `InvarianceResult`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Invariance Testing** checks that the agent gives consistent answers to paraphrased versions of the same question, measured via `tension_energy()` between answer entities.

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.invariance import InvarianceTester

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

```python
from knowlytix.harness.doe.invariance import InvarianceGenerator

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

> **API Reference:** [Invariance Testing](API_REFERENCE.md#33-invariance-testing) — `InvarianceTester`, `InvarianceResult`, `InvarianceGenerator`

---

## 32. Tutorial 29: Streaming Consistency

> **Module:** `harness.doe.streaming`  
> **Import:** `from knowlytix.harness.doe import StreamingAdapter, StreamConsistencyOracle`  
> **Key Classes:** `StreamingAdapter`, `StreamConsistencyOracle`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Streaming Testing** evaluates streaming LLM outputs for intermediate plausibility, self-contradiction, and reasoning coherence — chunk by chunk.

### What It Measures

| Metric | GMS Primitive | Meaning |
|--------|---------------|---------|
| Per-chunk plausibility | `score_triple()` | Is each chunk factually grounded? |
| Stream tension | `tension_energy()` | Does the stream contradict itself? |
| Chunk holonomy | `check_holonomy()` | Is the reasoning path coherent? |

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.streaming import StreamingAdapter, StreamConsistencyOracle

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

> **API Reference:** [Streaming Testing](API_REFERENCE.md#34-streaming-testing) — `StreamingAdapter`, `StreamingTrace`, `StreamConsistencyOracle`

---

## 33. Tutorial 30: Live Dashboard

> **Module:** `harness.doe.live_dashboard`  
> **Import:** `from knowlytix.harness.doe import DashboardServer, DashboardSnapshot`  
> **Key Classes:** `DashboardServer`, `DashboardSnapshot`  
> **Prereqs:** `bench`, `store` (Tutorial 1, Steps 1–2)

**Live Dashboard** provides real-time monitoring of benchmark runs via a WebSocket server that pulls all data from the GMS tracing store.

### Usage

```python
# Prereqs: bench + store (Tutorial 1, Steps 1-2)
from knowlytix.harness.doe.live_dashboard import DashboardServer, DashboardSnapshot

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

> **API Reference:** [Live Dashboard](API_REFERENCE.md#45-live-dashboard) — `DashboardServer`, `DashboardSnapshot`

---

# Reference

---

## 34. End-to-End Walkthrough

> **Module:** `harness.doe.claims`, `harness.doe.verifiers`, `harness.doe.verdict`  
> **Import:** `from knowlytix.harness.doe import ClaimCompiler, VerificationRouter, TypedVerdict`  
> **Key Classes:** `ClaimCompiler`, `VerificationRouter`, `TypedVerdict`

This puts all 7 workstreams together in a single flow.

```python
import torch
from knowlytix.harness.doe.config import DOEHarnessConfig
from knowlytix.harness.doe.benchmark import DOEGMSBenchmark
from knowlytix.harness.doe.gateway import ToolGateway, FaultProfile
from knowlytix.harness.doe.policy import PolicyEngine, PolicyRule
from knowlytix.harness.doe.scenarios import Scenario, ScenarioRegistry
from knowlytix.harness.doe.runs import RunManager, RunState
from knowlytix.harness.doe.replay import RegressionSuite, compare_runs
from knowlytix.harness.doe.audit import AuditReporter, RiskTierProfile
from knowlytix.harness.fixtures import MockHarness, create_mock_evaluator


# ── 1. Configure and Ingest ──────────────────────────────────
config = DOEHarnessConfig(
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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
reporter = AuditReporter(result, bench.typed_verdicts)
gates = reporter.generate_tiered_release_gate([
    RiskTierProfile.advisory(),
    RiskTierProfile.recommendation(),
    RiskTierProfile.action_taking(),
])

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

> **Module:** `harness.doe.failures`  
> **Import:** `from knowlytix.harness.doe import FailureCode, FailureAttribution, TypedVerdict`  
> **Key Classes:** `FailureCode`, `FailureAttribution`

### Standard Benchmark Run

```bash
# Basic run
python -m harness.doe run report.md

# With all GMSH features
python -m harness.doe run report.md \
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
python -m harness.doe run --config config.yaml
```

### Run Management

```bash
# List all tracked runs
python -m harness.doe runs list

# Show details for a specific run
python -m harness.doe runs show run_abc12345
```

### Regression and Replay

```bash
# Promote failures from a run to regression scenarios
python -m harness.doe regression promote run_abc12345

# List regression scenarios
python -m harness.doe regression list

# Replay a run from a manifest
python -m harness.doe replay manifest.yaml

# Compare two runs
python -m harness.doe compare result_a.json result_b.json
```

### Backwards Compatibility

The original single-command syntax still works:

```bash
python -m harness.doe report.md --n-runs 32
```

---

## 36. Configuration Reference

> **Module:** `harness.doe.adapters`  
> **Import:** `from knowlytix.harness.doe import SUTAdapter, LLMAdapter, AgentAdapter`  
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
| `hallucination_geodesic_threshold` | float | `1.0` | Geodesic above = hallucinated |
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

> **Module:** `harness.doe.audit`  
> **Import:** `from knowlytix.harness.doe import AuditReporter, TieredReleaseGate, RiskTierProfile`  
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
python -c "from knowlytix.harness.doe import DOEGMSBenchmark; print('OK')"
```

---

## 38. Common Recipes

Fifteen self-contained recipes covering the most frequent GMSH workflows. Every recipe includes all imports, defines all variables inline, uses `device="cpu"` and the standard document path `harness/doe/tests/model_risk_assessment.md`.

---

### Recipe 1: Ingest and Run a Basic Benchmark

```python
from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig

# Step 1: Configure the benchmark
config = DOEHarnessConfig(
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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
from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig

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
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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
from knowlytix.harness.doe import FaultProfile, ToolGateway
from knowlytix.harness.fixtures import MockHarness

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
from knowlytix.harness.doe import PolicyEngine, PolicyRule
from knowlytix.doc.store import GMSExpertStore

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
    "harness/doe/tests/model_risk_assessment.md", device="cpu"
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
from knowlytix.harness.doe import (
    DOEGMSBenchmark, DOEHarnessConfig,
    RegressionSuite, ScenarioRegistry,
)

# Step 1: Run a benchmark that will produce some failures
config = DOEHarnessConfig(
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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
from knowlytix.harness.doe import (
    CompositeOracle, ExactOutcomeOracle, TraceOracle,
)
from knowlytix.doc.store import GMSExpertStore

# Step 1: Create a GMS store for geometric scoring
store = GMSExpertStore.from_markdown(
    "harness/doe/tests/model_risk_assessment.md", device="cpu"
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
from knowlytix.harness.doe import InteractionScript, TurnSpec, MultiTurnAdapter
from knowlytix.harness.fixtures import MockHarness

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
from knowlytix.harness.doe import HallucinationOracle, HallucinationReport
from knowlytix.doc.store import GMSExpertStore

# Step 1: Create a GMS store
store = GMSExpertStore.from_markdown(
    "harness/doe/tests/model_risk_assessment.md", device="cpu"
)

# Step 2: Create the hallucination oracle
oracle = HallucinationOracle(store=store, geodesic_threshold=1.0)

# Step 3: Define sample claims as (head, relation, tail) triples
claims = [
    ("model_validation", "requires", "backtesting"),
    ("XGBoost", "invented_by", "Alan_Turing"),       # likely hallucination
    ("risk_assessment", "includes", "stress_testing"),
]

# Step 4: Assess all claims (signature: list[tuple[str,str,str]])
report = oracle.assess_triples(claims)

# Step 5: Print the hallucination report
print(f"Claims assessed: {report.n_claims}")  # int
print(f"Hallucinated: {report.n_hallucinated}")  # int
print(f"Hallucination rate: {report.hallucination_rate:.2%}")  # float
print(f"Severity counts: {report.severity_counts}")  # dict[str, int]
for a in report.assessments:
    print(f"  {a.claim_text}: geodesic={a.geodesic_distance:.3f}, "
          f"severity={a.severity}")  # HallucinationAssessment
```

---

### Recipe 9: Compare Two Model Versions

```python
from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig, compare_runs

# Step 1: Configure a shared benchmark setup
config = DOEHarnessConfig(
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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
from knowlytix.harness.doe import (
    DOEGMSBenchmark, DOEHarnessConfig,
    AuditReporter, RiskTierProfile,
)

# Step 1: Run a benchmark
config = DOEHarnessConfig(
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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

# Step 2: Build three risk-tier profiles
profiles = [
    RiskTierProfile.advisory(),        # 70% accuracy, low bar
    RiskTierProfile.recommendation(),  # 80% accuracy, no hallucinations
    RiskTierProfile.action_taking(),   # 95% accuracy, strict thresholds
]

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
from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig, taxonomy_generators
from knowlytix.doc.store import GMSExpertStore

# Step 1: Create a GMS store
store = GMSExpertStore.from_markdown(
    "harness/doe/tests/model_risk_assessment.md", device="cpu"
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
from knowlytix.harness.doe import EnvironmentSimulator, ToolEndpointSpec
from knowlytix.harness.doe.environment import StateTransition

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
from knowlytix.harness.doe import CostTracker, CostSummary

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
from knowlytix.harness.doe import (
    CampaignConfig, CampaignManager, CampaignType,
    DOEGMSBenchmark, DOEHarnessConfig,
    ScenarioRegistry,
)
from knowlytix.harness.doe.runs import RunManager

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
        "markdown_path": "harness/doe/tests/model_risk_assessment.md",
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
from knowlytix.harness.doe import (
    DOEGMSBenchmark, DOEHarnessConfig,
    AuditReporter, RiskTierProfile,
)

# Step 1: Configure with typed verdicts enabled
config = DOEHarnessConfig(
    markdown_path="harness/doe/tests/model_risk_assessment.md",
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
