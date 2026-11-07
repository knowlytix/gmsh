# Knowlytix Harness Governance -- API Reference

Complete API reference for the `knowlytix.harness.governance` subpackage.  Every public
class, method and function is documented with constructor parameters,
return types and usage notes.

The reference is grouped into the original core surface (Ch 1-11) and
the audit-defensible hardening modules (Ch 12-20) added during the
banking-grade pass.

---

## Table of Contents

### Core surface

- [orchestrator](#orchestrator) -- End-to-end governed pipeline
- [governed_gateway](#governed_gateway) -- Triple-gate tool execution
- [governed_llm](#governed_llm) -- Constrained LLM integration
- [memory](#memory) -- Geometric agent memory
- [verifier](#verifier) -- Typed claim verification
- [extractors](#extractors) -- Three-tier knowledge extraction
- [tools](#tools) -- Sklearn tool definitions
- [session](#session) -- Mutable session state
- [planner](#planner) -- GMS-guided workflow planning
- [reasoner](#reasoner) -- Multi-hop geometric reasoning
- [agents/data_agent](#agentsdata_agent) -- Phase-based pipeline runner
- [agents/multi_agent](#agentsmulti_agent) -- Multi-agent coordination
- [agents/adversarial](#agentsadversarial) -- Adversarial attack suite
- [release_gates](#release_gates) -- Tiered deployment gates and SPC
- [compliance](#compliance) -- Regulatory compliance evidence
- [tracing](#tracing) -- Governance audit telemetry
- [calibration](#calibration) -- Threshold tuning
- [behaviors/contracts](#behaviorscontracts) -- FSM behavioral contracts
- [policies/loader](#policiesloader) -- YAML policy loading
- [knowledge](#knowledge) -- Domain knowledge documents

### Runtime governance (Ch 12 -- comparison gate)

- [runtime_gate](#runtime_gate) -- Read-only verdict gate
- [governance_bundle](#governance_bundle) -- Frozen calibrated bundle
- [bundle_signing](#bundle_signing) -- HMAC-SHA-256 bundle signing
- [store_fingerprint](#store_fingerprint) -- Store identity digests
- [offline_validator](#offline_validator) -- Bootstrap & isotonic calibration
- [ood_validation](#ood_validation) -- Held-out cohort validation
- [replay](#replay) -- Bundle regression-replay
- [drift_monitor](#drift_monitor) -- PSI / abstain-rate drift
- [comparison_policy](#comparison_policy) -- Rule-based second attestor
- [reason_chain](#reason_chain) -- Audit-grade evidence
- [naive_baseline](#naive_baseline) -- Opposition / Jaccard baselines
- [plausibility_calibration](#plausibility_calibration) -- Geodesic + presence two-attestor
- [support_calibration](#support_calibration) -- Calibrated support floors
- [adversarial_sweep](#adversarial_sweep) -- Bundle adversarial cohort
- [api_key](#api_key) -- Anthropic API key resolution

### Policy hardening (Ch 12 v2)

- [policy_audit](#policy_audit) -- Reason-chain audit records
- [policy_bundle](#policy_bundle) -- Frozen policy bundle
- [policy_calibration](#policy_calibration) -- Bootstrap thresholds + quorum
- [policy_invariants](#policy_invariants) -- Property-based verification
- [policy_validator_alt](#policy_validator_alt) -- Independent quorum validator
- [policy_multisign](#policy_multisign) -- k-of-n attestor signatures
- [policy_validation](#policy_validation) -- Held-out OOD cohort
- [policy_replay](#policy_replay) -- Replay with audit hashes
- [policy_monitor](#policy_monitor) -- Rolling drift monitor
- [policy_numeric](#policy_numeric) -- Numeric two-attestor
- [policy_adversarial](#policy_adversarial) -- 200-probe adversarial sweep
- [policy_baselines](#policy_baselines) -- Prompt / Jaccard baselines
- [policy_bench](#policy_bench) -- Latency benchmarks

### Plan hardening

- [plan_policy](#plan_policy) -- Declarative plan rule pack
- [plan_adversarial](#plan_adversarial) -- LLM + structural attacks
- [plan_benchmark](#plan_benchmark) -- Labeled plan cohort

### Orchestrator hardening (Ch 16)

- [orchestrator_policy](#orchestrator_policy) -- Pre-run plan gate
- [orchestrator_validation](#orchestrator_validation) -- Held-out plan cohort
- [orchestrator_monitor](#orchestrator_monitor) -- SLO drift monitor
- [orchestrator_adversarial](#orchestrator_adversarial) -- 6-class plan attacks
- [orchestrator_baselines](#orchestrator_baselines) -- Jaccard / random verifier

### Release gate hardening (Ch 15)

- [release_gate_co_attestor](#release_gate_co_attestor) -- SPC slope co-attestor
- [release_gate_validator_alt](#release_gate_validator_alt) -- Truth-table alt
- [release_gate_invariants](#release_gate_invariants) -- Grid invariants
- [release_gate_calibration](#release_gate_calibration) -- Bootstrap thresholds
- [release_gate_adversarial](#release_gate_adversarial) -- 200-probe sweep
- [release_gate_baselines](#release_gate_baselines) -- Fixed / random
- [release_gate_bench](#release_gate_bench) -- Latency benchmarks

### LLM governance (Ch 11)

- [llm_bundle](#llm_bundle) -- Frozen ConstrainedLLM bundle
- [llm_co_attestor](#llm_co_attestor) -- Rule-based second attestor
- [llm_adversarial](#llm_adversarial) -- Adversarial cohort sweep

### Behavioral contract hardening (Ch 13)

- [contract_bundle](#contract_bundle) -- Frozen contract bundle
- [contract_co_attestor](#contract_co_attestor) -- Holonomy co-attestor
- [contract_adversarial](#contract_adversarial) -- Systematic violation cohort
- [contract_replay](#contract_replay) -- Per-probe replay snapshots

### Multi-agent hardening (Ch 17)

- [multi_agent_audit](#multi_agent_audit) -- Per-call audit records
- [multi_agent_bundle](#multi_agent_bundle) -- Frozen orchestrator bundle
- [multi_agent_monitor](#multi_agent_monitor) -- PSI drift monitor
- [multi_agent_replay](#multi_agent_replay) -- Cohort replay
- [multi_agent_co_attestor](#multi_agent_co_attestor) -- Role + PSI co-attestor
- [multi_agent_validator_alt](#multi_agent_validator_alt) -- Boolean alt-validator
- [multi_agent_invariants](#multi_agent_invariants) -- Lattice invariants
- [multi_agent_calibration](#multi_agent_calibration) -- Tool-frequency baselines
- [multi_agent_adversarial](#multi_agent_adversarial) -- 10-class attacks
- [multi_agent_baselines](#multi_agent_baselines) -- Single-agent / no-perm
- [multi_agent_bench](#multi_agent_bench) -- Latency benchmarks

### Document Ingestion

- [Three-mode ingestion (regex / hybrid / llm_only)](#document-ingestion)

### Key Dependencies

- [knowlytix.knowledge / knowlytix.core / knowlytix.benchmark / knowlytix.harness](#key-dependencies)

---

## orchestrator

`knowlytix/harness/governance/orchestrator.py`

### `RunState(Enum)`

Lifecycle states for a governed run.

| Value | Description |
|-------|-------------|
| `CREATED` | Initial state |
| `PREPARING` | Validating plan against scenario contract |
| `EXECUTING` | Running tools through governed gateway |
| `EVALUATING` | Verifying claims against GMS store |
| `COMPLETED` | All steps executed, verification passed |
| `FAILED` | Contract violation or verification failure |
| `ABORTED` | Budget exceeded or pre-execution block |

### `InvalidTransitionError`

Exception raised by `RunManager.transition()` when a transition violates the FSM.

### `RunManager`

FSM enforcer for the run lifecycle.  Logs every transition with a timestamp.

```python
RunManager()
```

**Attributes:** `state: RunState`, `transitions: list[tuple[RunState, RunState, float]]`

| Method | Returns | Description |
|--------|---------|-------------|
| `transition(new_state)` | `None` | Enforce valid state transition; raises `InvalidTransitionError` on illegal transition |
| `is_terminal()` | `bool` | `True` if state is `COMPLETED`, `FAILED` or `ABORTED` |
| `transition_log()` | `list[str]` | Human-readable transition history |

### `Scenario`

Behavioral contract (boundary conditions) for a governed run.

```python
@dataclass
class Scenario:
    name: str                           # scenario identifier
    required_tools: list[str] = []      # tools that MUST appear in plan
    forbidden_tools: list[str] = []     # tools that MUST NOT appear
    expected_tool_sequence: list[str] = []  # required ordering
    max_tool_calls: int = 30            # budget cap
```

### `SignalVector`

Six-component execution signal computed per step.

```python
@dataclass
class SignalVector:
    sim: float              # plausibility (inverted geodesic)
    ver: float              # verification success
    depth: float            # normalized step progress
    nov: float              # novelty (1.0 if new triples extracted)
    margin: float           # confidence margin from plausibility gate
    signal_unavailable: bool = False  # True when no plausibility source configured
```

| Method | Returns | Description |
|--------|---------|-------------|
| `as_tuple()` | `tuple[float, ...]` | Five-element numeric tuple (excludes `signal_unavailable`) |

### `OrchestratorResult`

Complete result from a governed pipeline run.

```python
@dataclass
class OrchestratorResult:
    state: RunState                    # final lifecycle state
    steps_executed: list[dict]         # per-step records
    verdict: OverallVerdict | None     # verification verdict
    trace: list[dict]                  # full execution trace
    scenario_violations: list[str]     # contract violations
    transition_log: list[str]          # state transition history
    total_time_ms: float               # wall-clock execution time
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_audit_log(bundle=None)` | `dict` | JSON-serializable audit record; embeds `GovernanceBundle` id+signature when bound |

### `GovernedOrchestrator`

End-to-end governed pipeline executor.

```python
GovernedOrchestrator(
    gateway: GovernedToolGateway,
    memory: AgentMemory,
    verifier: ClaimVerifier,
    session: SessionState,
    store: GMSExpertStore = None,
    device: str = "cpu",
    dataset_name: str = "iris",
    model_name: str = "RF_100",
    model_type: str = "RandomForestClassifier",
    plausibility_threshold: float | None = None,
    policy_store: Any = None,
    policy_threshold: float | None = None,
    agent_identity: str | None = None,
    plan_policy: OrchestratorPolicy | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `run(plan, scenario)` | `OrchestratorResult` | Execute governed pipeline: validate plan against scenario, run tools through gateway, extract triples, verify claims |
| `format_result(result)` | `str` | Pretty-print result with lifecycle transitions, signal vectors and verdict |
| `from_bundle(bundle, ...)` (classmethod) | `GovernedOrchestrator` | Construct orchestrator from a signed `GovernanceBundle`, binding calibrated thresholds |
| `bundle` (property) | `GovernanceBundle \| None` | The bundle bound at construction (if any) |

---

## governed_gateway

`knowlytix/harness/governance/governed_gateway.py`

### `GatewayResult`

Outcome of a governed tool execution.

```python
@dataclass
class GatewayResult:
    blocked: bool                      # whether call was blocked
    reason: str                        # human-readable explanation
    display: str                       # tool output string
    raw: dict                          # structured result dict
    gate_results: dict                 # per-gate pass/fail record
    policy_decision: PolicyDecision | None
    plausibility_score: float | None   # GMS geodesic distance
    fault_injected: bool               # whether fault was injected
    execution_time_ms: float           # wall-clock time
```

### `GovernedTranscriptEntry`

Single governed execution record for audit trail.

```python
@dataclass
class GovernedTranscriptEntry:
    tool_name: str
    args: dict
    context: str
    blocked: bool
    reason: str
    gate_results: dict
    plausibility_score: float | None
    policy_decision_summary: str
    fault_injected: bool
    execution_time_ms: float
    timestamp: str
```

### `GovernedToolGateway`

Triple-gate execution engine.

```python
GovernedToolGateway(
    session: SessionState,
    store: GMSExpertStore,
    policy: PolicyEngine | None = None,
    theta_plausibility: float = 1.5,    # hard-block threshold
    tool_schemas: list[dict] | None = None,
    device: str = "cuda",
    plausibility_relation: str = "should_call",  # relation used for L3 scoring
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(tool_name, args, context="default")` | `GatewayResult` | Execute tool through three gates: schema validation, policy check, GMS plausibility |
| `add_policy_rule(rule)` | `None` | Register a `PolicyRule` |
| `add_fault_profile(profile)` | `None` | Attach a `FaultProfile` for fault injection |
| `transcript` (property) | `list[GovernedTranscriptEntry]` | Chronological execution history |
| `transcript_summary()` | `dict` | Aggregate counts: total calls, blocked, passed, fault injections, block rate, per-gate blocks |

**Three gates applied in order:**

1. **Schema validation** -- checks required parameters and types
   against `TOOL_DEFINITIONS` JSON schemas
2. **Policy engine** -- four-layer check (fnmatch pattern, graph triple
   lookup, geometric plausibility, contradiction via tension energy)
3. **GMS plausibility** -- geodesic distance on $S^{m-1}$; hard-blocks
   if distance > `theta_plausibility`

---

## governed_llm

`knowlytix/harness/governance/governed_llm.py`

### `Proposal`

Proposed tool call with governance metadata.

```python
@dataclass
class Proposal:
    tool: str
    args: dict
    reason: str
    approved: bool = False
    score: float | None = None         # plausibility score
    rejection_reason: str = ""
```

### `GatedClaim`

Factual claim subject to contradiction gating.

```python
@dataclass
class GatedClaim:
    text: str                          # human-readable claim
    triple: ExtractedTriple            # knowledge triple
    approved: bool = False
    tension: float | None = None       # contradiction energy
    rejection_reason: str = ""
```

### `MockConstrainedLLM`

Deterministic mock (no API key required).

```python
MockConstrainedLLM(
    store: GMSExpertStore,
    verifier: LogicVerifier,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `interpret(tool_name, raw_result, context="")` | `str` | **Channel 1 (ungated):** interpret tool output as narrative |
| `propose_next(context, completed)` | `list[Proposal]` | **Channel 2 (plausibility-gated):** propose next tool calls |
| `extract_claims(narrative)` | `list[GatedClaim]` | **Channel 3 (contradiction-gated):** extract factual claims |

### `ConstrainedLLM`

Full implementation with real LLM backend.  Five fail-closed paths:
geometric (low plausibility / high tension), forbidden tool, backend
error, co-attestor block, and missing API key.

```python
ConstrainedLLM(
    store: GMSExpertStore,
    verifier: LogicVerifier,
    llm: Any,                          # LLM client (e.g. Anthropic)
    allowed_tools: list[str] | None = None,
    co_attestor: LLMCoAttestor | None = None,
)
```

Same three-channel interface as `MockConstrainedLLM`.

### Helper Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `format_proposal_report(proposals)` | `str` | Display proposals with approval status |
| `format_claim_report(claims)` | `str` | Display claims with approval status |

---

## memory

`knowlytix/harness/governance/memory.py`

### `CompareResult`

Result of comparing two entities on the manifold.

```python
@dataclass
class CompareResult:
    entity_a: str
    entity_b: str
    tension_energy: float | None       # 0=agreement, sqrt(2)=irrelevant, 2=contradiction
    interpretation: str                # "agreement" / "irrelevance" / "contradiction"
    metric_deltas: dict                # numeric metric differences
```

### `AgentMemory`

High-level agent memory wrapping GMS infrastructure.

```python
AgentMemory(
    store_path: str,
    knowledge_path: str | None = None,
    device: str = "cuda",
    epochs: int = 500,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `store_findings(triples, enm_entries=None)` | `LearnResult` | Store triples and optional ENM entries |
| `recall_entity(entity)` | `dict` | Retrieve all triples and ENM values for entity |
| `recall_metric(category, entity_id)` | `float \| None` | Exact numeric recall from ENM |
| `compare_entities(entity_a, entity_b)` | `CompareResult` | Geometric comparison via tension energy |
| `find_related(entity, relation, top_k=5)` | `list[tuple[str, float]]` | Link prediction neighbors |
| `check_contradiction(head, relation, tail)` | `float` | Contradiction score (tension energy) |
| `memory_stats()` | `dict` | Summary: entity count, triple count, ENM count |
| `save()` | `None` | Persist store to disk |
| `bootstrap(...)` (classmethod) | `AgentMemory` | One-shot training of a fresh store with the full GMS loss stack |

---

## verifier

`knowlytix/harness/governance/verifier.py`

### `ClaimType(Enum)`

| Value | Routing | Evidence |
|-------|---------|----------|
| `FACT` | `score_triple()` | Geodesic distance < threshold |
| `NUMERIC` | `lookup_enm()` | Relative difference <= tolerance |
| `THRESHOLD` | `lookup_enm()` + inequality | ENM value vs limit |
| `COMPARISON` | ENM comparison | Metric ordering |
| `REASONING` | `check_holonomy()` | Holonomy defect < threshold |

### `Claim`

Typed claim for verification.

```python
@dataclass
class Claim:
    type: ClaimType
    text: str                          # human-readable claim
    metadata: dict                     # type-specific routing fields
```

**Metadata fields by type:**

| Type | Required Keys |
|------|--------------|
| `FACT` | `head`, `relation`, `tail` |
| `NUMERIC` | `category`, `entity_id`, `claimed_value` |
| `THRESHOLD` | `category`, `entity_id`, `operator` (`ge`, `le`, `gt`, `lt`), `limit` |
| `COMPARISON` | `category`, `entity_a`, `entity_b`, `metric`, `direction` (`higher`, `lower`) |
| `REASONING` | `relation_path` (list of triples), `direct_relation` |

### `ClaimVerdict`

Result from verifying a single claim.

```python
@dataclass
class ClaimVerdict:
    claim: Claim
    passed: bool
    score: float                       # continuous confidence [0, 1]
    severity: str                      # "info", "warning", "error", "critical"
    evidence: str                      # which GMS primitive was used
    raw_signal: float | None           # geodesic, tension or holonomy value
    gms_method: str                    # the GMS method called
```

### `OverallVerdict`

Aggregate verdict across all claims.

```python
@dataclass
class OverallVerdict:
    claim_verdicts: list[ClaimVerdict]
    n_passed: int
    n_total: int
    overall_status: str                # "PASS" or "FAIL"
    pass_rate: float
    severity_distribution: dict[str, int]
```

### `ClaimVerifier`

Route claims to GMS primitives for evidence-based verification.

```python
ClaimVerifier(
    store: GMSExpertStore,
    device: str = "cuda",
    plausibility_threshold: float = 1.2,   # theta for fact claims
    numeric_tolerance: float = 0.01,       # relative difference for numeric claims
    holonomy_threshold: float = 0.5,       # defect limit for reasoning claims
    numeric_range_index: dict | None = None,  # secondary attestor for NUMERIC claims
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `verify_claim(claim)` | `ClaimVerdict` | Verify single claim by type |
| `verify_all(claims)` | `OverallVerdict` | Batch verification with aggregation |
| `format_verdict(verdict)` | `str` | Format single verdict with evidence |
| `format_overall(overall)` | `str` | Format aggregate verdict |

---

## extractors

`knowlytix/harness/governance/extractors.py`

### `ENMEntry`

Exact numerical memory entry.

```python
@dataclass
class ENMEntry:
    category: str          # e.g., "metric", "robustness", "fairness"
    entity_id: str         # e.g., "rf_100_test_accuracy"
    value: float
```

### `ExtractionResult`

Unified extraction output from any tier.  `n_facts` is auto-computed
from `len(triples) + len(enm_entries)` in `__post_init__`.

```python
@dataclass
class ExtractionResult:
    triples: list[ExtractedTriple]
    enm_entries: list[ENMEntry]
    tier: int              # 1 (deterministic), 2 (regex), 3 (LLM)
    tool_name: str
    n_facts: int = 0       # auto-set in __post_init__
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `extract_from_tool(tool_name, args, raw_result, display_string="")` | `ExtractionResult` | Main entry point; tries Tier 1, falls back to Tier 2, then Tier 3 |
| `compare_extraction_accuracy(ground_truth, extraction)` | `dict` | Compare extracted numeric values against known ground truth; returns `matched`, `mismatched`, `missing`, `accuracy`, `details` |

**Tier 1 handlers** (deterministic, dispatched by tool name):
`_tier1_load_dataset`, `_tier1_train_model`,
`_tier1_diagnose_robustness`, `_tier1_diagnose_fairness`,
`_tier1_explain_importance`, `_tier1_leaderboard`

Tools without a Tier 1 handler fall through to Tier 2 (regex) or Tier 3 (simple/LLM parse).

---

## tools

`knowlytix/harness/governance/tools.py`

### `TOOL_DEFINITIONS`

`list[dict]` -- JSON schema declarations for all tools.

**Data tools:** `load_dataset`, `data_summary`, `eda_correlation`,
`preprocess_data`, `feature_select`

**Training tools:** `add_model`, `train_model`, `predict`

**Evaluation tools:** `leaderboard`, `diagnose_accuracy`,
`diagnose_robustness`, `diagnose_fairness`

**Explanation tools:** `explain_importance`, `explain_pdp`, `explain_shap`

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `execute_tool(tool_name, args, session)` | `tuple[str, dict]` | Execute tool; returns `(display_string, raw_result_dict)` |

---

## session

`knowlytix/harness/governance/session.py`

### `SessionState`

Mutable session state persisting across tool calls.

```python
@dataclass
class SessionState:
    datasets: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    preprocessors: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
```

**`datasets` value structure:**
`{X_train, X_test, y_train, y_test, feature_names, target_name, task_type, df}`

**`models` value structure:**
`{estimator, model_type, params, dataset}`

---

## planner

`knowlytix/harness/governance/planner.py`

### `PlannedStep`

One step in a workflow plan.

```python
@dataclass
class PlannedStep:
    tool: str
    confidence: float      # GMS link prediction confidence
    reason: str            # explanation for recommendation
```

### `PlanValidation`

Result of validating a plan.  Carries calibrated/policy fields when
the planner has been bound via `calibrate()` / `attach_policy()`.

```python
@dataclass
class PlanValidation:
    holonomy: float                            # holonomy defect of "precedes" chain
    prerequisites_satisfied: bool
    violations: list[str]
    verdict: str | None = None                 # "accept" / "review" / "reject"
    confidence: float | None = None
    calibrated_features: dict[str, float] | None = None
    policy_verdict: str | None = None          # "ALLOW" / "BLOCK" from PlanPolicy
    policy_reasons: list[str] = []
```

### `WorkflowPlanner`

GMS-guided step selection.

```python
WorkflowPlanner(
    store: GMSExpertStore,
    device: str | torch.device = "cpu",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `next_step(completed_steps, context="")` | `PlannedStep` | Recommend the single next step via `link_predict(last, "precedes")` with prerequisite filtering; falls back to `CANONICAL_ORDER` |
| `full_plan(goal, context="", max_steps=15)` | `list[PlannedStep]` | Iteratively call `next_step` until goal is reached or budget exhausted |
| `validate_plan(plan)` | `PlanValidation` | Check holonomy defect on the `precedes` chain and prerequisite satisfaction; emits calibrated verdict if `calibrate()` has been called |
| `calibrate(labeled_plans, ...)` | `None` | Fit a logistic regressor over plan features for calibrated `accept/review/reject` verdicts |
| `load_calibration(path)` | `None` | Load classifier + thresholds from disk |
| `attach_policy(policy)` | `None` | Bind a `PlanPolicy` so `validate_plan` runs declarative rules in addition to FSM/holonomy checks |
| `is_step_ready(step, completed)` | `tuple[bool, list[str]]` | `(ready, missing_prereqs)` |
| `get_workflow_steps()` | `list[str]` | Return `CANONICAL_ORDER` |
| `get_phase(step)` | `str` | Phase label for a step (Data, Training, Diagnostics, Explanation, Evaluation, Deployment) |

**Constants:**
- `CANONICAL_ORDER` -- 13-step fallback ordering
- `PREREQUISITES` -- required prior steps per tool
- `STEP_PHASE` -- phase assignments (Data, Training, Diagnostics, Explanation, Evaluation, Deployment)

---

## reasoner

`knowlytix/harness/governance/reasoner.py`

### `CalibratedThresholds`

Frozen calibrated threshold set produced by `GMSJudge.calibrate()` and
consumed by `GeometricReasoner`.  Every value is fitted from labeled
graph samples by a logistic regression, never hardcoded.
`tension_status` reports whether the 3-class tension fit was usable
(`"ok"`, `"degenerate"`, `"insufficient_data"`,
`"insufficient_classes"`).  When the status is not `"ok"` the geometric
arm of `GeometricReasoner.detect_contradictions` refuses to run and
the caller must rely on schema-declared contradictions only.

```python
@dataclass(frozen=True)
class CalibratedThresholds:
    tau_plausibility: float | None     # geodesic plausibility cut
    tau_ent: float | None              # agree / unrelated cut on tension
    tau_contra: float | None           # unrelated / contradict cut on tension
    tension_status: str                # "ok" / "degenerate" / "insufficient_*"
    diagnostics: dict = field(default_factory=dict)
                                       # carries every other key from the
                                       # judge thresholds (e.g. path_transport,
                                       # holonomy) for audit traceability

    @property
    def tension_ok(self) -> bool: ...

    @classmethod
    def from_judge(cls, judge) -> "CalibratedThresholds": ...
```

### `CalibrationNotUsable`

Exception raised when reasoner construction is attempted with
calibration metadata that does not meet quality floors.

### `ChainLink`

One hop in a reasoning chain.

```python
@dataclass
class ChainLink:
    head: str
    relation: str
    tail: str
    score: float | None        # geodesic distance
    plausible: bool | None     # score <= threshold
```

### `ChainValidation`

Result of validating a multi-hop reasoning chain.

```python
@dataclass
class ChainValidation:
    links: list[ChainLink]
    all_plausible: bool
    mean_score: float | None
    holonomy_defect: float | None      # consistency check
    consistent: bool | None            # holonomy <= threshold
```

### `Contradiction`

Pair of contradictory entities.  Carries audit provenance: which arm
(schema-declared `opposite_of` / `is_functional`, learned geometric
tension above `tau_contra`, or both) flagged the pair.  Both arms
firing on the same pair is the strongest evidence and `provenances`
preserves that as a set rather than collapsing to a single string.

```python
@dataclass
class Contradiction:
    entity_a: str
    entity_b: str
    tension: float                                 # tension energy
    provenances: frozenset = frozenset({"geometric"})
                                                   # subset of {"schema", "geometric"}
    threshold: float | None = None                 # calibrated tau_contra
                                                   # (None when only schema fired)
    interpretation: str = ""                       # "agreement" / "irrelevant"
                                                   # / "contradiction"
    schema_relations: tuple[str, str] | None = None
                                                   # opposing relation pair, when
                                                   # the schema arm fired
    shared_tail: str | None = None                 # tail both relations point at,
                                                   # when the schema arm fired
```

`Contradiction.provenance` (singular, property) collapses
`provenances` to a single string for legacy callers:
`"schema+geometric"`, `"schema"`, or `"geometric"`.

### `Explanation`

Supporting and contradicting evidence for a claim.

```python
@dataclass
class Explanation:
    head: str
    relation: str
    tail: str
    direct_score: float | None
    supporting_paths: list[ScoredPath]
    contradicting_entities: list[Contradiction]
    verdict: str               # "supported" / "unsupported" / "contradicted"
```

### `GeometricReasoner`

Agent-level inference API.  Requires a `CalibratedThresholds` object;
backwards-compatible scalar properties `plausibility_threshold` and
`tension_contradiction` are exposed for legacy callers.

```python
GeometricReasoner(
    store: GMSExpertStore,
    thresholds: CalibratedThresholds,
    device: str | torch.device = "cuda",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `is_plausible(head, relation, tail)` | `tuple[bool, float] \| None` | `(plausible, geodesic)` via `score_triple`; None if entities unknown |
| `find_path(source_entities, target_entities, max_hops=4)` | `SearchResult \| None` | BFS between entity groups via `GeometricSearch.connect_groups` |
| `validate_chain(chain)` | `ChainValidation` | Score each `(h, r, t)` link and compute holonomy defect |
| `detect_contradictions(entities, threshold=None)` | `list[Contradiction]` | Pairwise tension energy above threshold, sorted descending |
| `recommend_next(context, relation, top_k=5)` | `list[tuple[str, float]]` | Link prediction via `store.link_predict` |
| `explain_why(head, relation, tail, max_hops=4)` | `Explanation` | Direct score + supporting paths + contradicting entities + verdict |
| `get_all_tensions(entities)` | `list[tuple[str, str, float, str]]` | All pairwise tensions with interpretation label |
| `score_triple_set(triples)` | `float` | Geometric confidence `exp(-mean_geodesic)` via `GeometricSearch` |

---

## agents/data_agent

`knowlytix/harness/governance/agents/data_agent.py`

### `PhaseResult`

Outcome of executing a pipeline phase.

```python
@dataclass
class PhaseResult:
    phase_name: str
    tools_called: list[str]
    triples_extracted: int
    enm_entries: int
    policy_blocks: int
    trace: list[dict]
    claims: list[Claim]
    elapsed_s: float
```

### `PhaseRunner`

Orchestrates phases with extraction and verification.

```python
PhaseRunner(
    session: SessionState,
    memory: AgentMemory,
    policy_rules: list[dict] | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `run_phase(phase_name, tool_sequence)` | `PhaseResult` | Execute a phase (list of `(tool_name, args)` tuples) |
| `cumulative_stats()` | `dict` | Aggregate statistics across all phases |

**Predefined phase generators:**

| Function | Returns | Description |
|----------|---------|-------------|
| `adult_data_phase()` | `list[tuple]` | load, summarize, correlate, preprocess |
| `adult_training_phase()` | `list[tuple]` | train 5 classifiers |
| `adult_diagnostics_phase()` | `list[tuple]` | robustness + feature importance |
| `adult_fairness_phase()` | `list[tuple]` | AIR on protected attributes |

### `generate_governance_report`

```python
generate_governance_report(
    phase_results: list[PhaseResult],
    verification_result: OverallVerdict,
    session: SessionState,
    cumulative: dict,
) -> str
```

Builds a complete markdown-formatted governance report (executive summary, per-phase stats, verification verdict, model inventory) from pipeline outputs. `cumulative` is the dict returned by `PhaseRunner.cumulative_stats()`.

---

## agents/multi_agent

`knowlytix/harness/governance/agents/multi_agent.py`

### `AgentRole`

Definition of an agent's identity and tool permissions.  Frozen and
post-init-validated: `allowed_tools` and `forbidden_tools` are coerced
to `frozenset`, and any overlap raises immediately.

```python
@dataclass(frozen=True)
class AgentRole:
    name: str                          # e.g., "data_engineer"
    allowed_tools: frozenset[str]      # tools this agent may call
    forbidden_tools: frozenset[str] = frozenset()
    description: str = ""
```

### Predefined Roles

Shared `AgentRole` instances for the reference Adult Income pipeline:

| Constant | Allowed tools | Purpose |
|----------|---------------|---------|
| `DATA_AGENT_ROLE` | load_dataset, data_summary, eda_correlation, preprocess_data, feature_select | Data-phase loading and preprocessing |
| `TRAINING_AGENT_ROLE` | train_model, predict, leaderboard | Model training and ranking |
| `DIAGNOSTICS_AGENT_ROLE` | diagnose_robustness, explain_importance, explain_pdp, explain_shap | Robustness and explanation diagnostics |
| `FAIRNESS_AGENT_ROLE` | diagnose_fairness | Fairness auditing |

### `HandoffRecord`, `ToolCallRecord`, `AgentStats`

```python
@dataclass
class HandoffRecord:
    from_agent: str
    to_agent: str
    step_num: int
    shared_keys: list[str]             # session keys passed at handoff
    timestamp: float

@dataclass
class ToolCallRecord:
    agent_name: str
    tool_name: str
    args: dict
    display: str
    raw: dict
    n_triples: int
    timestamp: float

@dataclass
class AgentStats:
    tools_called: int
    triples_learned: int
    tool_names: list[str]
    blocked_attempts: int
```

### `PipelineResult`

```python
@dataclass
class PipelineResult:
    phases_completed: list[str]
    handoffs: list[HandoffRecord]
    tool_calls: list[ToolCallRecord]
    per_agent_stats: dict[str, AgentStats]
    boundary_violations: list[str]     # cross-role permission breaches
```

### `GovernedAgent`

A single tool-scoped agent that enforces its role and tracks domain
access.  Identity attributes (`role`, `name`) are immutable: a custom
`__setattr__` raises if any agent attempts to mutate them after
construction (role-hijack defense).

```python
GovernedAgent(
    role: AgentRole,
    session: SessionState,             # shared across agents
    memory: AgentMemory | None = None, # optional shared memory
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `can_execute(tool_name)` | `bool` | True iff tool not forbidden and tool is allowed by role |
| `execute(tool_name, args)` | `tuple[str, dict]` | Run tool if permitted; raises if blocked |
| `domain_stats()` | `AgentStats` | Current per-agent counters |

### `MultiAgentOrchestrator`

Coordinate a team of `GovernedAgent` instances with enforced handoffs and
store-isolation checks.

```python
MultiAgentOrchestrator(
    agents: dict[str, GovernedAgent],  # name -> agent
    shared_session: SessionState,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `handoff(from_agent, to_agent, step_num, context_keys)` | `HandoffRecord` | Record an explicit control handoff, filtering session keys to those named in `context_keys` |
| `run_pipeline(phases)` | `PipelineResult` | Execute a sequence of `(agent_name, tool_name, args)` tuples; boundary violations recorded but do not abort |
| `test_boundary(agent_name, forbidden_tool, args)` | `ToolCallRecord` | Probe that a non-owning agent is blocked from a tool outside its role |
| `test_store_isolation(agent_a, agent_b, shared_store_keys)` | `dict` | (static) Report which session keys each agent wrote, detecting unintended cross-agent leakage |
| `format_handoffs()` | `str` | Pretty-print the handoff log |
| `format_agent_stats()` | `str` | Tabular per-agent summary |

---

## agents/adversarial

`knowlytix/harness/governance/agents/adversarial.py`

Red-team probes that attempt to subvert a governed pipeline and verify
that the six-layer defense (role permissions, policy rules, memory
tension checks, budget caps, identity guards) holds.

### `AttackResult`

```python
@dataclass
class AttackResult:
    attack_name: str
    description: str
    blocked: bool                      # True = defense held
    defense_layer: str                 # "role", "policy", "memory", "budget", "identity"
    mechanism: str                     # short explanation of why blocked
    evidence: str                      # detailed trace/output
```

### `AdversarialSuite`

Runs the five canonical attacks and reports which defenses fired.

```python
AdversarialSuite(
    session: SessionState,
    memory: AgentMemory,
    verifier: ClaimVerifier | None = None,
    policy_rules: list[PolicyRule] | None = None,
    gateway: GovernedToolGateway | None = None,
    budget_limit: int = 30,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `attack_skip_fairness()` | `AttackResult` | Attempts to reach DEPLOYMENT_READY without `diagnose_fairness` |
| `attack_forbidden_tool()` | `AttackResult` | Data agent tries to call `train_model` |
| `attack_poison_memory()` | `AttackResult` | Injects a contradictory triple into memory |
| `attack_budget_exhaustion()` | `AttackResult` | Loops noop tool calls past `budget_limit` |
| `attack_model_identity()` | `AttackResult` | Queries memory for a metric attributed to a non-existent model |
| `run_all()` | `list[AttackResult]` | Run all five attacks |
| `format_results(results)` (static) | `str` | Tabular summary |

### `TextGuardrail`

A naive keyword-based guardrail used as a comparison baseline.

```python
TextGuardrail(
    blocked_tools: list[str],
    banned_keywords: list[str],
    budget_limit: int = 30,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `check(tool_name, args, call_count)` | `tuple[bool, str]` | `(allowed, reason)` |
| `simulate_attacks()` | `list[dict]` | Demonstrate that the guardrail misses attacks the FSM contract catches |

---

## release_gates

`knowlytix/harness/governance/release_gates.py`

### `GateThresholds`

Per-tier deployment thresholds.  `is_calibrated` and `source` track
provenance when thresholds are produced by `release_gate_calibration`.

```python
@dataclass
class GateThresholds:
    min_accuracy: float
    min_confidence: float
    max_robustness_drop: float
    min_fairness_air: float
    is_calibrated: bool = False
    source: str = "policy_declared"
```

### `GateResult`

Outcome of checking one tier.

```python
@dataclass
class GateResult:
    tier: str
    passed: bool
    checks: dict[str, dict]            # per-check details
    blocking_reasons: list[str]
    is_calibrated: bool = False
    threshold_source: str = "policy_declared"
```

### `RunMetrics`

Metrics from a single pipeline run.

```python
@dataclass
class RunMetrics:
    accuracy: float
    confidence: float
    robustness_drop: float
    fairness_air: float
    tool_calls: int
    policy_blocks: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `from_orchestrator_result(result)` (classmethod) | `RunMetrics` | Extract metrics from an `OrchestratorResult`'s verdict + trace |

### `RunTrend`

Track metrics across runs for SPC drift detection via OLS slope.

```python
@dataclass
class RunTrend:
    metrics: list[RunMetrics] = []
```

Append by calling `trend.metrics.append(run_metrics)` directly.

| Method | Returns | Description |
|--------|---------|-------------|
| `slope(field_name)` | `float` | OLS slope of the metric over run index; 0.0 if fewer than 2 points |
| `mean(field_name)` | `float` | Arithmetic mean across runs |
| `alert(field_name, threshold)` | `bool` | True when slope is *below* threshold (drift detected) |

### `TieredReleaseGate`

Three-tier release gate with monotonically increasing stringency.

**Default thresholds (`TieredReleaseGate.TIERS`):**

| Tier | Min Accuracy | Min Confidence | Max Robustness Drop | Min Fairness AIR |
|------|-------------|----------------|---------------------|------------------|
| advisory | 0.70 | 0.30 | unconstrained (1.0) | unconstrained (0.0) |
| recommendation | 0.80 | 0.50 | 0.10 | unconstrained (0.0) |
| action_taking | 0.85 | 0.70 | 0.05 | 0.80 |

Canonical ordering: `TIER_ORDER = ["advisory", "recommendation", "action_taking"]`.

| Method | Returns | Description |
|--------|---------|-------------|
| `check(metrics, tier)` | `GateResult` | Check metrics against tier; raises `ValueError` on unknown tier |
| `check_all_tiers(metrics)` | `dict[str, GateResult]` | Check all three tiers |
| `verify_monotonicity(metrics)` | `bool` | Assert that a stricter tier passing implies every looser tier passes |
| `format_gate_result(result)` (static) | `str` | Human-readable gate outcome |

---

## compliance

`knowlytix/harness/governance/compliance.py`

### `ComplianceCheck`

Single regulatory check.

```python
@dataclass
class ComplianceCheck:
    requirement: str
    regulation: str                    # "SR 11-7" or "EU AI Act"
    article: str                       # e.g., "Conceptual Soundness", "Article 9"
    architecture_layer: str
    evidence_chapter: str
    status: str                        # "satisfied", "partial", "not_applicable"
    evidence_summary: str
```

### `CompliancePackage`

Aggregated compliance evidence.

```python
@dataclass
class CompliancePackage:
    checks: list[ComplianceCheck]
    overall_status: str                # "COMPLIANT", "PARTIAL", "NON-COMPLIANT"
    summary: str
    timestamp: str
```

| Method | Returns | Description |
|--------|---------|-------------|
| `pass_rate()` | `float` | Fraction of satisfied checks |
| `to_markdown()` | `str` | Audit-ready markdown report |

### `ComplianceEvidenceGenerator`

Map pipeline results to regulatory checks.

```python
ComplianceEvidenceGenerator()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `generate_package(pipeline_results)` | `CompliancePackage` | Generate full evidence package |
| `sr_11_7_checks(pipeline_results)` | `list[ComplianceCheck]` | SR 11-7 checks only |
| `eu_ai_act_checks(pipeline_results)` | `list[ComplianceCheck]` | EU AI Act checks only |

**Expected `pipeline_results` dict keys:**

```python
{
    "models_trained": int,
    "verification_pass_rate": float,
    "policy_violations": int,
    "contract_violations": int,
    "spc_alerts": int,
    "tracing_enabled": bool,
    "snapshot_count": int,
    "audit_entries": int,
    "risk_assessment_done": bool,
    "fairness_checked": bool,
    "human_oversight_enabled": bool,
}
```

### `SnapshotManager`

Version governance state for change management.

```python
SnapshotManager()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `take_snapshot(label, state)` | `int` | Deep-copy the state and return the new snapshot index |
| `compare(snap_a, snap_b)` | `dict` | Structured diff with `added`, `removed`, `changed`, `unchanged`, `summary` |
| `summary()` | `str` | Human-readable list of all snapshots with fingerprints |

### `ProvenanceRegistry`

Artifact lineage tracking with SHA-256 fingerprints.

```python
ProvenanceRegistry()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `register(name, artifact_type, source, metadata=None, parent=None)` | `str` | Register artifact, return 16-char SHA-256 fingerprint |
| `get_lineage(name)` | `list[dict]` | Trace ancestor chain from root to the named artifact |
| `summary()` | `str` | Human-readable summary with per-type counts and parent links |

---

## tracing

`knowlytix/harness/governance/tracing.py`

### `AuditPacket`

Per-tool-call provenance record.  `prev_hash` and `content_hash`
implement a tamper-evident hash chain over the audit trail.

```python
@dataclass
class AuditPacket:
    step_num: int
    tool_name: str
    policy_decision: str               # "allowed" or "blocked"
    blocking_rules: list
    gms_scores: dict
    verification: dict | None
    state_before: str
    state_after: str
    timestamp: float
    execution_time_ms: float
    gate_results: dict
    args: dict
    result_summary: str
    prev_hash: str = ""
    content_hash: str = ""
```

### `GovernanceTracer`

Audit-grade governance telemetry.  Optionally persists to a
`DOETracingGMS` store for geometric durability.

```python
GovernanceTracer(
    tracing_store: DOETracingGMS | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `start_run(run_id, description="")` | `None` | Mark run start |
| `end_run(run_id)` | `None` | Mark run end |
| `trace_tool_call(run_id, step_num, tool_name, args, result_summary, policy_decision, blocking_rules=None, gms_scores=None, gate_results=None, execution_time_ms=0.0, state_before="", state_after="")` | `AuditPacket` | Record a tool call and persist to DOETracingGMS if attached |
| `trace_verification(run_id, step_num, claims, verdicts)` | `None` | Attach verification verdicts to the most recent packet for this step |
| `trace_state_transition(run_id, from_state, to_state)` | `None` | Record FSM transition |
| `get_audit_trail(run_id)` | `list[AuditPacket]` | Full chronological audit trail |
| `query_blocked_calls(run_id)` | `list[AuditPacket]` | Packets with `policy_decision == "blocked"` |
| `query_allowed_calls(run_id)` | `list[AuditPacket]` | Packets with `policy_decision == "allowed"` |
| `get_verifications(run_id)` | `list[dict]` | All verification records |
| `get_transitions(run_id)` | `list[dict]` | All state transitions |
| `summary_report(run_id)` | `str` | Formatted governance audit summary |
| `dashboard_table(run_id)` | `list[dict]` | Per-step dashboard rows |

---

## calibration

`knowlytix/harness/governance/calibration.py`

### `ThresholdConfig`

Threshold vector for governance parameters.

```python
@dataclass
class ThresholdConfig:
    theta_sim: float = 1.5             # plausibility threshold
    theta_margin: float = 0.3          # verification confidence
    theta_esc: float = 0.5             # escalation (holonomy)
    d_max: int = 30                    # budget cap
    budget: float = 1.0                # normalized budget
```

### `CalibrationResult`

Result from one configuration.

```python
@dataclass
class CalibrationResult:
    config: ThresholdConfig
    accuracy: float
    false_accepts: int
    false_rejects: int
    tool_calls: int
```

### `LabeledClaim`

Claim with ground truth for calibration.

```python
@dataclass
class LabeledClaim:
    claim: Claim
    expected_pass: bool
```

### `ThresholdCalibrator`

Systematic threshold tuning via parameter sweep and grid search.

```python
ThresholdCalibrator(
    store: GMSExpertStore,
    session: SessionState | None = None,
    device: str = "cpu",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `sweep(param_name, values, test_claims, base_config=None)` | `list[CalibrationResult]` | Vary a single parameter |
| `grid_search(param_grid, test_claims)` | `list[CalibrationResult]` | Full factorial grid search |
| `sensitivity_analysis(results)` (static) | `dict[str, float]` | Per-parameter accuracy range across grouped results |
| `best_config(results)` (static) | `CalibrationResult` | Highest accuracy, ties broken by fewer FA then fewer FR |
| `results_table(results)` (static) | `str` | Formatted text table of configurations |

---

## behaviors/contracts

`knowlytix/harness/governance/behaviors/contracts.py`

### `ContractState(Enum)`

Workflow FSM states.

| Value | Description |
|-------|-------------|
| `IDLE` | No data loaded |
| `DATA_LOADED` | Dataset available |
| `PREPROCESSED` | Data preprocessed |
| `MODEL_TRAINED` | At least one model trained |
| `DIAGNOSED` | Diagnostics completed |
| `FAIR_CHECKED` | Fairness assessment done |
| `COMPARED` | Models compared |
| `DEPLOYMENT_READY` | All checks passed |
| `HUMAN_REVIEW` | Awaiting human approval |

### `SequenceValidation`

Result of validating a tool sequence against a contract.

```python
@dataclass
class SequenceValidation:
    valid: bool
    violations: list[str]
    states_visited: list[ContractState]
    tools_checked: int
    tools_blocked: int
    invariant_violations: list[str]
```

### `BehavioralContract`

FSM-based structural enforcement.

```python
@dataclass
class BehavioralContract:
    name: str
    states: list[ContractState]
    transitions: dict[ContractState, set[ContractState]]
    tool_permissions: dict[ContractState, set[str]]
    invariants: list[Callable] = []   # (state, history) -> str | None
    max_tool_calls: int = 30
    require_human_approval: bool = False
```

| Method | Returns | Description |
|--------|---------|-------------|
| `is_transition_valid(from_state, to_state)` | `bool` | True if `to_state in transitions[from_state]`; self-loops always allowed |
| `allowed_tools(current_state)` | `set[str]` | Tools permitted in the given state |
| `check_invariants(state, history)` | `list[str]` | Run every invariant and return violation descriptions |
| `transition_table()` | `list[dict[str, str]]` | Tabular dump of `{from, to, tools}` rows |

### `ProcessEvaluator`

Validates a tool sequence against a contract.

```python
ProcessEvaluator(contract: BehavioralContract)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `validate_sequence(sequence, contract=None)` | `SequenceValidation` | Pure-FSM walk: permission + transition + budget + invariant + human-approval checks |
| `check_holonomy(actual, expected, store=None)` | `float` | GMS holonomy defect when `precedes` is trained; otherwise normalized Levenshtein distance in [0, 2] |

### Predefined Contract Factories

| Factory | Budget | Required invariants | Human Approval | Use Case |
|---------|--------|---------------------|----------------|----------|
| `advisory_contract()` | 50 | none | No | Low-risk, exploratory |
| `recommendation_contract()` | 30 | no_train_without_preprocess | No | Standard analysis |
| `action_taking_contract()` | 20 | no_train_without_preprocess, no_deploy_without_fairness, no_deploy_without_robustness | Yes | High-risk, must pass through HUMAN_REVIEW |

### `TextGuardrail` (comparison baseline)

Naive keyword-ordering guardrail used in Lab 13.2.

```python
TextGuardrail()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `check_sequence(sequence)` | `tuple[bool, list[str]]` | `(passed, violations)` |

### Module-level constants

- `TOOL_STATE_ADVANCE: dict[str, ContractState]` -- mapping from tool name to the contract state that tool advances the FSM into (e.g. `load_dataset → DATA_LOADED`, `preprocess_data → PREPROCESSED`).

---

## policies/loader

`knowlytix/harness/governance/policies/loader.py`

| Function | Returns | Description |
|----------|---------|-------------|
| `load_policy_yaml(path)` | `list[PolicyRule]` | Load policy rules from YAML file |
| `rules_to_triples(rules)` | `list[ExtractedTriple]` | Convert policy rules to GMS triples |
| `ingest_rules_to_store(rules, store, *, learner=None, ...)` | `None` | Convert rules to triples and ingest into a GMSExpertStore via `RuntimeLearner` |

---

## knowledge

`knowlytix/harness/governance/knowledge/`

Markdown domain-knowledge documents ingested into the GMS store as priors. Plain files, not a Python API.

| File | Purpose |
|------|---------|
| `california_housing_domain.md` | California Housing domain facts (features, geography, typical ranges) |
| `data_science_practices.md` | Reference practices for preprocessing, training, diagnostics, fairness |

Loaded via `docknowlytix.core.extract.extract_triples_simple` (or Tier 3 LLM extraction) and committed to the store before a governed run begins, so the reasoner has domain priors to verify against.

---

# Runtime governance (Ch 12 -- comparison gate)

The modules below implement the runtime entity-comparison gate
introduced in Ch 12 of the agent-builder monograph.  An offline
validator produces a frozen, signed `GovernanceBundle`; a
`RuntimeGate` consumes the bundle to issue verdicts on entity pairs
with bounded latency and full reason chains.

---

## runtime_gate

`knowlytix/harness/governance/runtime_gate.py`

### `Verdict` (Literal type)

```python
Verdict = Literal["agreement", "irrelevance", "contradiction", "review", "abstain"]
```

### `RuntimeVerdict`

One verdict from `RuntimeGate.verdict`, suitable for emitting as
telemetry with full audit trail.

```python
@dataclass
class RuntimeVerdict:
    entity_a: str
    entity_b: str
    verdict: Verdict
    p_contradiction: float | None
    tension_energy: float | None
    reason: str
    bundle_id: str
    bundle_signature: str
    latency_ms: float
    features: dict[str, float] = {}
    reason_chain: ReasonChain | None = None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert verdict to dictionary for telemetry |

### `RuntimeGate`

Read-only consumer of a `GovernanceBundle` that issues entity-comparison
verdicts.

```python
RuntimeGate(
    bundle: GovernanceBundle,
    store: GMSExpertStore,
    *,
    policy: ComparisonPolicy | None = None,
    require_two_attestors: bool = True,
    min_triples: int = 2,
    min_relations: int = 1,
    review_p_low: float = 0.35,
    review_p_high: float = 0.65,
    verify_store_fingerprint: bool = True,
    include_reason_chain: bool = True,
    reason_chain_top_k: int = 5,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `verdict(a, b)` | `RuntimeVerdict` | Compute verdict for entity pair through bounded-latency pipeline |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `serve(store, bundle_path, *, policy=None, min_triples=2, min_relations=1)` | `RuntimeGate` | Convenience: load bundle from disk and bind to store |

---

## governance_bundle

`knowlytix/harness/governance/governance_bundle.py`

### `OperatingPoint`

SLO for the runtime gate's deferral behavior.

```python
@dataclass(frozen=True)
class OperatingPoint:
    max_abstain_rate: float = 0.20
    max_review_rate: float = 0.20
    max_uncertain_rate: float = 0.30
```

### `EntitySupport`

Per-entity coverage statistics computed at bundle build time.

```python
@dataclass(frozen=True)
class EntitySupport:
    triple_count: int
    relation_count: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `is_supported(min_triples, min_relations)` | `bool` | Check if entity meets support thresholds |

### `ThresholdDistribution`

Bootstrap distribution of a calibrated cut.

```python
@dataclass(frozen=True)
class ThresholdDistribution:
    mean: float
    std: float
    ci_low: float
    ci_hi: float
    n_bootstrap: int
```

### `GovernanceBundle`

Frozen artifact bundle for entity-comparison governance, carrying
calibrated thresholds and store metadata.

```python
@dataclass
class GovernanceBundle:
    schema_version: str
    bundle_id: str
    created_at: str
    source_store_path: str
    tau_ent: ThresholdDistribution | None
    tau_contra: ThresholdDistribution | None
    isotonic_x: np.ndarray | None
    isotonic_y: np.ndarray | None
    support_index: dict[str, EntitySupport]
    opposition_table: dict[tuple[str, str], str]
    drift_baseline_te: np.ndarray
    drift_baseline_te_quantiles: tuple[float, ...]
    validation_metrics: dict[str, float]
    tau_plausibility: ThresholdDistribution | None = None
    store_fingerprint: StoreFingerprint | None = None
    support_thresholds: CalibratedSupportThresholds | None = None
    operating_point: OperatingPoint | None = None
    numeric_range_index: dict | None = None
    signature: str = ""
    signing_method: SigningMethod = "sha256"
```

| Property | Returns | Description |
|----------|---------|-------------|
| `is_tension_capable` | `bool` | True iff bundle carries calibrated tension thresholds |
| `is_plausibility_capable` | `bool` | True iff bundle carries a calibrated plausibility threshold |
| `isotonic` | `IsotonicRegression` | Rebuild sklearn isotonic regressor from frozen x/y arrays |

| Method | Returns | Description |
|--------|---------|-------------|
| `support_for(entity)` | `EntitySupport \| None` | Retrieve support statistics for entity |
| `is_declared_opposition(a, b)` | `bool` | Check if pair is declared opposite via relations |
| `canonical_payload()` | `bytes` | Bytes used for signing (excludes signature itself) |
| `freeze(key=None)` | `GovernanceBundle` | Compute and stamp signature |
| `compute_signature(key=None)` | `str` | Recompute signature for current state (tests) |
| `verify(key=None)` | `bool` | Verify bundle signature |
| `save(path)` | `None` | Pickle bundle to disk |
| `load(path, *, key=None)` (classmethod) | `GovernanceBundle` | Load and verify bundle from disk |

### `BundleSignatureError`

Raised when bundle signature does not match contents.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `make_bundle_id(source_store_path, created_at)` | `str` | Deterministic short id from store path + timestamp |
| `utcnow_iso()` | `str` | Current UTC time in ISO format |

### Constants

- `BUNDLE_SCHEMA_VERSION: str = "1.4"` -- current schema version

---

## bundle_signing

`knowlytix/harness/governance/bundle_signing.py`

### `SigningMethod` (Literal type)

```python
SigningMethod = Literal["sha256", "hmac-sha256"]
```

### `BundleSigningError`

Raised when bundle signature does not authenticate.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `resolve_signing_key(key=None)` | `bytes \| None` | Resolve signing key from argument or environment |
| `sign(payload, key=None)` | `tuple[str, SigningMethod]` | Sign payload bytes; returns `(signature_hex, method)` |
| `verify(payload, signature, method, key=None)` | `bool` | Verify signature with constant-time comparison |

### Constants

- `SIGNING_KEY_ENV_VAR: str = "GOVERNED_AGENT_SIGNING_KEY"` -- environment variable name

---

## store_fingerprint

`knowlytix/harness/governance/store_fingerprint.py`

### `StoreFingerprint`

Component-wise digests of a store's identity at calibration time.

```python
@dataclass(frozen=True)
class StoreFingerprint:
    entity_hash: str
    relation_hash: str
    triple_hash: str
    model_hash: str
    n_entities: int
    n_relations: int
    n_triples: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `matches(other)` | `bool` | Check if all hash components match |
| `diff(other)` | `dict[str, tuple[str, str]]` | Return differing components as `{component: (mine, other)}` |
| `to_dict()` | `dict` | Convert fingerprint to dictionary |

### `StoreFingerprintMismatch`

Raised when a runtime store does not match the bundle's calibration-time store.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_store_fingerprint(store)` | `StoreFingerprint` | Compute four digests over a store's identity |

---

## offline_validator

`knowlytix/harness/governance/offline_validator.py`

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bootstrap_thresholds(samples, *, n_bootstrap=200, seed=42)` | `tuple[ThresholdDistribution, ThresholdDistribution]` | Resample cohort K times; returns `(tau_ent_dist, tau_contra_dist)` |
| `fit_isotonic(samples)` | `tuple[np.ndarray, np.ndarray]` | Fit isotonic regression `tension_energy → P(contradict)` |
| `build_numeric_range_index(store, *, pad=0.0)` | `dict[str, tuple[float, float, int]]` | Compute (min, max, n_obs) per ENM category |
| `build_bundle(store, *, source_store_path, n_bootstrap=200, seed=42, operating_point=None)` | `GovernanceBundle` | Assemble and sign GovernanceBundle from trained store |

---

## ood_validation

`knowlytix/harness/governance/ood_validation.py`

### `GroundTruth` (Literal type)

```python
GroundTruth = Literal["agree", "unrelated", "contradict"]
```

### `LabeledPair`

One ground-truth pair for end-to-end gate validation.

```python
@dataclass
class LabeledPair:
    a: str
    b: str
    label: GroundTruth
    provenance: str = ""
```

### `OODValidationReport`

Validation outcome on a held-out cohort.

```python
@dataclass
class OODValidationReport:
    n_total: int
    per_class_n: dict[str, int]
    confusion: dict[tuple[str, str], int]
    far_per_class: dict[str, float]
    frr_per_class: dict[str, float]
    ece: float
    abstain_rate: float
    review_rate: float
    tau_used: tuple[float, float]
    extra: dict[str, float] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable summary of validation results |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `validate_calibration(bundle, samples)` | `OODValidationReport` | Test bundle's isotonic mapping against labeled samples (calibration-only, no store) |
| `validate_gate(gate, pairs)` | `OODValidationReport` | Run full RuntimeGate over labeled entity pairs (end-to-end) |
| `shift_samples(samples, *, delta_per_class=None, noise=0.0, seed=0)` | `list[CalibrationSample]` | Apply synthetic distribution shift to a labeled cohort |

---

## replay

`knowlytix/harness/governance/replay.py`

### `ReplayCohort`

Versioned, frozen cohort of labeled pairs for regression testing
bundle updates.

```python
@dataclass(frozen=True)
class ReplayCohort:
    name: str
    version: str
    pairs: tuple[tuple[str, str, str, str], ...]  # (a, b, label, provenance)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `from_labeled_pairs(pairs, *, name, version)` (classmethod) | `ReplayCohort` | Construct from `LabeledPair` list |
| `to_labeled_pairs()` | `list[LabeledPair]` | Convert back to `LabeledPair` objects |
| `save(path)` | `None` | Save cohort as JSON |
| `load(path)` (classmethod) | `ReplayCohort` | Load cohort from JSON |

### `ReplayResult`

Verdicts produced by one bundle on a replay cohort.

```python
@dataclass
class ReplayResult:
    cohort_name: str
    cohort_version: str
    bundle_id: str
    bundle_signature: str
    verdicts: dict[tuple[str, str], dict]
```

| Method | Returns | Description |
|--------|---------|-------------|
| `save(path)` | `None` | Save replay result as JSON |
| `load(path)` (classmethod) | `ReplayResult` | Load replay result from JSON |

### `ReplayDelta`

Per-pair changes between two replay results.

```python
@dataclass
class ReplayDelta:
    cohort_name: str
    n_total: int
    n_unchanged: int
    n_verdict_changed: int
    n_verdict_drift: int
    changes: list[tuple[tuple[str, str], dict, dict]] = []
    drifts: list[tuple[tuple[str, str], float, float]] = []
```

| Property | Returns | Description |
|----------|---------|-------------|
| `has_regressions` | `bool` | True if any pair changed verdict or drifted significantly |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable summary of delta |

### `ReplayRegressionError`

Raised when replay diff exceeds allowed regressions.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `replay_bundle(gate, cohort)` | `ReplayResult` | Run every pair in cohort through gate |
| `compare_replays(old, new, *, drift_threshold_te=0.20, drift_threshold_p=0.15)` | `ReplayDelta` | Diff two replay results |
| `assert_no_regression(delta, *, allow_drift=True)` | `None` | Strict gate: raise if any verdict changed |

---

## drift_monitor

`knowlytix/harness/governance/drift_monitor.py`

### `DriftState`

Snapshot of current drift monitor state.

```python
@dataclass
class DriftState:
    n_observed: int
    window_full: bool
    psi_te: float | None
    abstain_rate: float
    review_rate: float
    abstain_rate_delta: float | None
    alert: bool
    recalibrate: bool
    severity: str
    slo_violation: str | None = None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable summary of drift state |

### `DriftMonitor`

Stateful, ring-buffered drift detector for a `RuntimeGate`.

```python
DriftMonitor(
    bundle: GovernanceBundle,
    *,
    window_size: int = 500,
    alert_threshold: float = 0.10,
    recalibrate_threshold: float = 0.20,
    abstain_drift_threshold: float = 0.10,
    baseline_abstain_rate: float = 0.0,
    n_bins: int = 10,
    min_te_samples: int = 50,
    recalibrate_requires_full_window: bool = True,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `record(verdict)` | `None` | Ingest a `RuntimeVerdict` for drift tracking |
| `record_te(te, verdict="geometry")` | `None` | Lightweight ingest path for tests |
| `state()` | `DriftState` | Compute current drift state snapshot |
| `should_alert()` | `bool` | True if alert threshold exceeded |
| `should_recalibrate()` | `bool` | True if recalibration threshold exceeded |
| `reset()` | `None` | Clear all buffered observations |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `population_stability_index(baseline, current, *, n_bins=10, smoothing=1e-4)` | `float` | Compute PSI between two empirical distributions |

---

## comparison_policy

`knowlytix/harness/governance/comparison_policy.py`

### `PolicySignal` (Literal type)

```python
PolicySignal = Literal["contradiction", "agreement", "abstain"]
```

### `RuleResult`

One rule's verdict on an entity pair.

```python
@dataclass
class RuleResult:
    signal: PolicySignal | None
    reason: str = ""
```

### `ComparisonRule`

Predicate + metadata for one rule on entity comparison.

```python
@dataclass
class ComparisonRule:
    rule_id: str
    description: str
    severity: Literal["blocking", "advisory"]
    predicate: Callable[[str, str, GMSExpertStore, GovernanceBundle], RuleResult]
```

| Method | Returns | Description |
|--------|---------|-------------|
| `__call__(a, b, store, bundle)` | `RuleResult` | Invoke the rule's predicate |

### `ComparisonPolicyVerdict`

```python
@dataclass
class ComparisonPolicyVerdict:
    verdict: PolicySignal
    fired_rules: list[str] = []
    reasons: list[str] = []
    contradicting_rules: list[str] = []
    agreeing_rules: list[str] = []
```

### `ComparisonPolicy`

Set of rules that produce a second-attestor verdict for entity comparison.

| Method | Returns | Description |
|--------|---------|-------------|
| `__init__(rules)` | `None` | Initialize with list of rules |
| `check(a, b, store, bundle)` | `ComparisonPolicyVerdict` | Evaluate entity pair and return verdict |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `declared_opposition_rule()` | `ComparisonRule` | Rule checking declared opposition table |
| `functional_relation_rule()` | `ComparisonRule` | Rule checking functional relation disagreement |
| `shared_target_rule(min_shared=2)` | `ComparisonRule` | Rule checking shared `(relation, target)` pairs |
| `default_comparison_policy(min_shared_for_agreement=2)` | `ComparisonPolicy` | Minimal default policy with three built-in rules |

---

## reason_chain

`knowlytix/harness/governance/reason_chain.py`

### `ReasonChain`

Audit-grade evidence for one entity-comparison verdict.

```python
@dataclass
class ReasonChain:
    entity_a: str
    entity_b: str
    a_top_triples: list[tuple[str, str, str]] = []
    b_top_triples: list[tuple[str, str, str]] = []
    shared_targets: list[tuple[str, str]] = []
    opposing_signals: list[tuple[str, str, str]] = []
    n_shared_targets: int = 0
    n_opposing_signals: int = 0
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable reason-chain summary |
| `to_dict()` | `dict` | Convert reason-chain to dictionary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_reason_chain(a, b, store, top_k_triples=5)` | `ReasonChain` | Compute reason chain for A vs B comparison via pure graph lookup |

---

## naive_baseline

`knowlytix/harness/governance/naive_baseline.py`

### `NaiveOppositionBaseline`

Verdict from the opposition table only.

```python
NaiveOppositionBaseline(
    bundle: GovernanceBundle,
    name: str = "naive_opposition",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(a, b)` | `str` | Predict verdict from opposition table |

### `JaccardBaseline`

Jaccard similarity over incident triples.

```python
JaccardBaseline(
    triples: list,
    theta_agree: float = 0.4,
    theta_unrelated: float = 0.05,
    name: str = "jaccard",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(a, b)` | `str` | Predict verdict via Jaccard similarity |

### `BaselineContrastReport`

```python
@dataclass
class BaselineContrastReport:
    cohort_size: int
    baseline_name: str
    gate_correct: int
    baseline_correct: int
    gate_abstain: int
    baseline_abstain: int
    gate_accuracy_on_decided: float
    baseline_accuracy_on_decided: float
    uplift_correct_pairs: int
    examples_gate_correct_baseline_wrong: list = []
    examples_baseline_correct_gate_wrong: list = []
```

| Method | Returns | Description |
|--------|---------|-------------|
| `uplift_accuracy()` | `float` | Gate accuracy minus baseline accuracy |
| `summary()` | `str` | Human-readable contrast report |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `compare_to_gate(gate, baseline, cohort, keep_examples=5)` | `BaselineContrastReport` | Run gate and baseline over cohort |

---

## plausibility_calibration

`knowlytix/harness/governance/plausibility_calibration.py`

### `GraphPresenceResult`

```python
@dataclass
class GraphPresenceResult:
    present: bool
    matched_head: str | None
    matched_tail: str | None
    matched_triple: tuple[str, str, str] | None
    reason: str
```

| Method | Returns | Description |
|--------|---------|-------------|
| `signal()` | `str` | Return `"agreement"` if present else `"abstain"` |

### `TwoAttestorVerdict`

```python
@dataclass
class TwoAttestorVerdict:
    passed: bool
    geodesic_pass: bool
    presence_pass: bool
    geodesic: float | None
    threshold: float
    threshold_ci: tuple[float, float]
    presence_reason: str
    decision: str               # "PASS" | "REVIEW" | "FAIL"
    evidence: str
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert verdict to dictionary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bootstrap_plausibility_threshold(store, n_bootstrap=200, seed=42)` | `ThresholdDistribution` | Bootstrap geodesic threshold |
| `check_graph_presence(store, head, relation, tail)` | `GraphPresenceResult` | Check if corpus asserts this triple |
| `verify_fact_two_attestor(store, head, relation, tail, threshold, require_concurrence=True)` | `TwoAttestorVerdict` | Verify fact with geodesic + graph-presence concurrence |

---

## support_calibration

`knowlytix/harness/governance/support_calibration.py`

### `CalibratedSupportThresholds`

```python
@dataclass(frozen=True)
class CalibratedSupportThresholds:
    min_triples: int
    min_relations: int
    target_accuracy: float
    coverage_accuracy: float
    n_pairs: int
    coverage_pairs: int
    rationale: str
```

| Method | Returns | Description |
|--------|---------|-------------|
| `coverage_rate()` | `float` | Ratio of coverage pairs to total pairs |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `calibrate_support_thresholds(store, support_index, tau_ent, tau_contra, target_accuracy=0.85, seed=42, fallback_min_triples=2, fallback_min_relations=1)` | `CalibratedSupportThresholds` | Sweep candidate floors and select smallest preserving accuracy |

---

## adversarial_sweep

`knowlytix/harness/governance/adversarial_sweep.py`

### `AdversarialPair`

```python
@dataclass(frozen=True)
class AdversarialPair:
    a: str
    b: str
    attack_type: str
    safe_verdicts: tuple[str, ...]
    provenance: str = ""
```

### `AdversarialReport`

```python
@dataclass
class AdversarialReport:
    n_total: int
    per_attack_n: dict[str, int]
    per_attack_safe: dict[str, int]
    per_attack_unsafe: dict[str, list[tuple[AdversarialPair, str, str]]] = {}
    symmetry_violations: list[tuple[str, str, str, str]] = []
```

| Method | Returns | Description |
|--------|---------|-------------|
| `safety_rate()` | `float` | Ratio of safe verdicts to total probes |
| `summary()` | `str` | Human-readable report summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_adversarial_cohort(bundle, seed=42, n_per_type=20, thin_threshold=2)` | `list[AdversarialPair]` | Generate attack pairs from bundle |
| `sweep_adversarial(gate, cohort, check_symmetry=True, keep_unsafe_examples=5)` | `AdversarialReport` | Run gate over cohort and report safety |

---

## api_key

`knowlytix/harness/governance/api_key.py`

### `AnthropicAPIKeyError`

Raised when no Anthropic API key can be resolved.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `get_anthropic_api_key()` | `str` | Return Anthropic API key; resolves from `ANTHROPIC_API_KEY` env or `pass(1)` store |

---

# Policy hardening (Ch 12 v2)

The `policy_*` modules add audit-defensible hardening on top of the
core `knowlytix.harness.testing.policy.PolicyEngine`: bundle-signed thresholds,
property-based invariants, alt-validators, multi-signer signatures,
adversarial cohorts, baselines, and microbenchmarks.

---

## policy_audit

`knowlytix/harness/governance/policy_audit.py`

### `IncidentTriple`

Top-K incident triple from policy decision audit.

```python
@dataclass
class IncidentTriple:
    head: str
    relation: str
    tail: str
    score: float | None = None
```

### `PolicyReasonChain`

Structured reason-chain attached to a policy decision.

```python
@dataclass
class PolicyReasonChain:
    action_top_triples: list[IncidentTriple] = []
    context_top_triples: list[IncidentTriple] = []
    plausibility: float | None = None
    plausibility_threshold: float | None = None
    plausibility_ci: tuple[float, float] | None = None
    tension_signals: dict[str, float] = {}
    tau_contra: float | None = None
```

### `PolicyAuditRecord`

End-to-end JSON-serializable audit record for a policy decision.

```python
@dataclass
class PolicyAuditRecord:
    bundle_id: str
    bundle_signature: str
    bundle_signing_method: str
    rules_yaml_hash: str
    probe_id: str
    agent: str
    action: str
    args: dict
    context: str
    allowed: bool
    rules_checked: list[str]
    blocking_rules: list[str]
    gms_scores: dict[str, float]
    reason: str
    quorum_decision: str
    quorum_required: int
    n_layers_blocking: int
    n_layers_signal_available: int
    layer1_block: bool
    layer2_block: bool
    layer3_block: bool | None
    layer4_block: bool | None
    quorum_evidence: str
    reason_chain: dict
    timestamp: str = ""
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert audit record to dictionary |
| `to_json()` | `str` | Serialize audit record to JSON string |
| `replay()` | `str` | Re-derive `BLOCK`/`ALLOW`/`REVIEW` verdict from this record alone |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_policy_reason_chain(engine, decision, bundle=None, top_k=5)` | `PolicyReasonChain` | Build top-K reason chain for a policy decision |
| `to_audit_record(engine, bundle, probe_id, agent, decision, verdict, args=None, context="", timestamp="")` | `PolicyAuditRecord` | Pack `(decision, verdict)` into a frozen audit record |
| `replay_audit_log(record)` | `str` | Re-derive verdict from a record; symmetric to `to_audit_record` |

---

## policy_bundle

`knowlytix/harness/governance/policy_bundle.py`

### `PolicyOperatingPoint`

SLO for the policy engine's deferral and refusal behavior.

```python
@dataclass(frozen=True)
class PolicyOperatingPoint:
    max_review_rate: float = 0.20
    max_block_rate: float = 0.80
    min_block_rate: float = 0.10
```

| Method | Returns | Description |
|--------|---------|-------------|
| `violates(block_rate, review_rate)` | `bool` | Check if rates violate the operating point |

### `PolicyBundle`

Frozen artifact bundle for high-stakes policy enforcement.

```python
@dataclass
class PolicyBundle:
    schema_version: str
    bundle_id: str
    created_at: str
    source_store_path: str
    rules: list[dict]
    rules_yaml_hash: str
    policy_geodesic: ThresholdDistribution | None = None
    policy_geodesic_empirical: ThresholdDistribution | None = None
    tau_contra: ThresholdDistribution | None = None
    plausibility_relation: str = "uses_tool"
    numeric_range_index: dict[str, tuple[float, float, int]] | None = None
    store_fingerprint: StoreFingerprint | None = None
    operating_point: PolicyOperatingPoint = PolicyOperatingPoint()
    quorum_required: int = 2
    validation_metrics: dict[str, float] = {}
    signature: str = ""
    signing_method: SigningMethod = "sha256"
    attestor_signatures: list[dict] = []
```

| Property | Returns | Description |
|----------|---------|-------------|
| `is_layer3_capable` | `bool` | True if `policy_geodesic` is set |
| `is_layer4_capable` | `bool` | True if `tau_contra` is set |

| Method | Returns | Description |
|--------|---------|-------------|
| `canonical_payload()` | `bytes` | Deterministic JSON payload for signing |
| `freeze(key=None)` | `PolicyBundle` | Compute and stamp the signature |
| `verify(key=None)` | `bool` | Verify the signature on the bundle |
| `save(path)` | `None` | Persist bundle to disk via pickle |
| `load(path, key=None)` (classmethod) | `PolicyBundle` | Load and verify bundle from disk |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_policy_bundle(engine, rules, source_store_path, relation="uses_tool", n_bootstrap=200, seed=42, operating_point=None, quorum_required=2, validation_metrics=None, numeric_range_index=None)` | `PolicyBundle` | Produce a calibrated, signed policy bundle |
| `bind_engine_from_bundle(bundle, store, verify_store_fingerprint=True)` | `PolicyEngine` | Construct a `PolicyEngine` from a verified bundle |

### Constants

- `POLICY_BUNDLE_SCHEMA_VERSION: str = "1.0"`

---

## policy_calibration

`knowlytix/harness/governance/policy_calibration.py`

### `PolicyAttestorVerdict`

```python
@dataclass
class PolicyAttestorVerdict:
    decision: str               # "BLOCK" | "ALLOW" | "REVIEW"
    n_layers_blocking: int
    n_layers_signal_available: int
    layer1_block: bool
    layer2_block: bool
    layer3_block: bool | None
    layer4_block: bool | None
    blocking_rules: list[str]
    quorum_required: int
    evidence: str
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert verdict to dictionary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bootstrap_policy_geodesic_threshold(store, relation="uses_tool", n_bootstrap=200, seed=42)` | `ThresholdDistribution` | Bootstrap policy geodesic threshold |
| `bootstrap_policy_geodesic_empirical(store, relation="uses_tool", n_bootstrap=200, seed=42)` | `ThresholdDistribution \| None` | Always-empirical bootstrap on labeled pairs |
| `bootstrap_policy_tau_contra(store, n_bootstrap=200, seed=42)` | `ThresholdDistribution \| None` | Bootstrap `tau_contra` for Layer 4 |
| `derive_attestor_verdict(decision, quorum_required=2)` | `PolicyAttestorVerdict` | Convert `PolicyDecision` to quorum verdict |
| `check_with_quorum(engine, action, args, context, quorum_required=2)` | `tuple[PolicyDecision, PolicyAttestorVerdict]` | Run engine and return `(decision, verdict)` |

---

## policy_invariants

`knowlytix/harness/governance/policy_invariants.py`

### `InvariantViolation`

```python
@dataclass
class InvariantViolation:
    invariant: str
    state: dict
    detail: str
```

### `InvariantReport`

Exhaustive property-based verification results over layer state space.

```python
@dataclass
class InvariantReport:
    n_states: int
    n_passed: int
    violations: list[InvariantViolation]
```

| Property | Returns | Description |
|----------|---------|-------------|
| `all_pass` | `bool` | True iff no violations found |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable report |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `verify_quorum_invariants(quorum_required=2)` | `InvariantReport` | Enumerate all layer states and verify I1-I5 invariants |
| `verify_replay_symmetry()` | `InvariantReport` | Verify I5 (`to_audit_record` + `replay` symmetry) |

---

## policy_validator_alt

`knowlytix/harness/governance/policy_validator_alt.py`

### `AlternativeVerdict`

Truth-table-driven quorum verdict from independent re-implementation.

```python
@dataclass
class AlternativeVerdict:
    decision: str
    layer1: bool
    layer2: bool
    layer3: bool | None
    layer4: bool | None
    n_block: int
```

### `AgreementReport`

Cohort agreement check results between primary and alt implementations.

```python
@dataclass
class AgreementReport:
    n_total: int
    n_agree: int
    disagreements: list[tuple[str, str, str]]
```

| Property | Returns | Description |
|----------|---------|-------------|
| `agreement_rate` | `float` | Fraction of probes where implementations agree |
| `all_agree` | `bool` | True iff no disagreements |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable report |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `alt_check(blocking_rules, gms_scores, *, quorum=2, tau_contra=1.7)` | `AlternativeVerdict` | Stand-alone quorum check from raw decision fields |
| `alt_check_from_audit_record(record)` | `AlternativeVerdict` | Replay verdict from archived audit record |
| `assert_engine_alt_agreement(engine, cohort, *, quorum_required=2)` | `AgreementReport` | Run engine + alt-validator on every probe |

---

## policy_multisign

`knowlytix/harness/governance/policy_multisign.py`

### `AttestorSignature`

One signer's signature on the bundle's canonical payload.

```python
@dataclass
class AttestorSignature:
    signer_id: str
    method: str
    signature: str
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(d)` (classmethod) | `AttestorSignature` | Deserialize from dictionary |

### `QuorumVerifyResult`

Aggregated result of k-of-n signature verification.

```python
@dataclass
class QuorumVerifyResult:
    n_signatures: int
    n_verified: int
    n_required: int
    verified_signers: list[str]
    rejected_signers: list[str]
    detail: str
```

| Property | Returns | Description |
|----------|---------|-------------|
| `quorum_met` | `bool` | True iff verified count meets requirement |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable summary |

### `QuorumSignatureError`

Raised when a bundle fails k-of-n quorum verification.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `sign_for(bundle, signer_id, key)` | `AttestorSignature` | Produce one attestor's HMAC-SHA-256 signature |
| `attach_signatures(bundle, signatures)` | `None` | Store signatures on bundle in canonical order |
| `verify_quorum(bundle, key_registry, *, k=2)` | `QuorumVerifyResult` | Verify at least `k` of `n` signatures validate |
| `assert_quorum(bundle, key_registry, *, k=2)` | `None` | Raise `QuorumSignatureError` if quorum not met |

---

## policy_validation

`knowlytix/harness/governance/policy_validation.py`

### `ExpectedDecision` (Literal type)

```python
ExpectedDecision = Literal["BLOCK", "ALLOW", "REVIEW"]
```

### `LabeledDecision`

```python
@dataclass
class LabeledDecision:
    probe_id: str
    agent: str
    action: str
    args: dict
    context: str
    expected_decision: ExpectedDecision
    source_class: str
    provenance: str = ""
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert to dictionary |
| `from_dict(d)` (classmethod) | `LabeledDecision` | Construct from dictionary |

### `PolicyOODReport`

```python
@dataclass
class PolicyOODReport:
    n_total: int
    per_class_n: dict[str, int]
    confusion: dict[tuple[str, str], int]
    far_per_class: dict[str, float]
    frr_per_class: dict[str, float]
    review_rate: float
    ece: float
    mismatches: list[tuple[str, str, str, str]] = []
```

| Property | Returns | Description |
|----------|---------|-------------|
| `overall_far` | `float` | Mean false-accept rate across classes |
| `overall_frr` | `float` | Mean false-reject rate across classes |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable report summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `validate_policy(engine, cohort, quorum_required=2, keep_examples=8)` | `PolicyOODReport` | Run engine over labeled cohort and report metrics |
| `adult_income_labeled_cohort()` | `list[LabeledDecision]` | Default 32-probe Adult Income cohort, deterministic |
| `large_adult_income_cohort(seed=42, n_context_paraphrases=4)` | `list[LabeledDecision]` | 1000+ probe cohort for production validation |
| `cohort_fingerprint(cohort)` | `str` | Deterministic SHA-256 over the cohort |
| `save_cohort(cohort, path)` | `None` | Persist cohort to disk as JSONL |
| `load_cohort(path)` | `list[LabeledDecision]` | Load cohort from disk |

---

## policy_replay

`knowlytix/harness/governance/policy_replay.py`

### `PolicyReplayResult`

```python
@dataclass
class PolicyReplayResult:
    cohort_size: int
    bundle_id: str
    bundle_signature: str
    snapshots: list[dict] = []
    cohort_hash: str = ""
    n_decision_matches: int = 0
    n_blocking_rules_matches: int = 0
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable result summary |
| `save(path)` | `None` | Persist result to disk as JSON |
| `load(path)` (classmethod) | `PolicyReplayResult` | Load result from disk |

### `PolicyReplayDelta`

```python
@dataclass
class PolicyReplayDelta:
    n_total: int
    n_unchanged: int
    n_decision_changed: int
    n_blocking_rules_drifted: int
    old_hash: str
    new_hash: str
    changes: list[tuple[str, dict, dict]] = []
    rule_drifts: list[tuple[str, list[str], list[str]]] = []
```

| Property | Returns | Description |
|----------|---------|-------------|
| `has_regressions` | `bool` | True if any verdict changed |
| `hash_matches` | `bool` | True if cohort hashes match |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable delta summary |

### `PolicyReplayRegressionError`

Raised when replay diff exceeds allowed regressions.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `policy_decision_snapshot(probe, decision, verdict)` | `dict` | Audit-grade summary of one `(probe, decision, attestor)` triple |
| `cohort_hash(snapshots)` | `str` | Deterministic SHA-256 over canonical snapshots |
| `replay_policy_battery(engine, cohort, quorum_required=2, bundle_id="", bundle_signature="")` | `PolicyReplayResult` | Run every labeled probe and produce single audit hash |
| `compare_policy_replays(old, new)` | `PolicyReplayDelta` | Diff two replay results |
| `assert_no_policy_regression(delta, allow_rule_drift=True)` | `None` | Strict gate: raise if any verdict changed |

---

## policy_monitor

`knowlytix/harness/governance/policy_monitor.py`

### `PolicyMonitorState`

```python
@dataclass
class PolicyMonitorState:
    n_observed: int
    window_full: bool
    block_rate: float
    review_rate: float
    allow_rate: float
    psi_plausibility: float | None
    severity: str
    alert: bool
    recalibrate: bool
    slo_violation: str | None = None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable state summary |

### `PolicyMonitor`

Stateful rolling-window monitor for a deployed `PolicyEngine`.

```python
PolicyMonitor(
    bundle: PolicyBundle,
    window_size: int = 500,
    min_runs_for_alert: int = 50,
    psi_alert: float = 0.10,
    psi_recalibrate: float = 0.20,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `set_baseline_plausibility(samples)` | `None` | Stamp baseline L3 plausibility distribution |
| `record(decision, verdict)` | `None` | Record one runtime `(decision, verdict)` pair |
| `state()` | `PolicyMonitorState` | Get current monitor state |
| `should_alert()` | `bool` | True if alert condition triggered |
| `should_recalibrate()` | `bool` | True if recalibration recommended |

---

## policy_numeric

`knowlytix/harness/governance/policy_numeric.py`

### `NumericObservation`

```python
@dataclass(frozen=True)
class NumericObservation:
    requirement: str
    value: float
    accepted: bool
```

### `NumericRangeVerdict`

```python
@dataclass
class NumericRangeVerdict:
    requirement: str
    value: float
    boolean_pass: bool
    range_pass: bool
    range_min: float | None
    range_max: float | None
    n_observations: int
    decision: str
    evidence: str
```

### `NumericObservationLog`

Append-only telemetry log of numeric requirement observations.

```python
@dataclass
class NumericObservationLog:
    path: str | None = None
    observations: list[NumericObservation] = None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `append(obs)` | `None` | Append one observation |
| `append_many(obs)` | `None` | Append multiple observations |
| `per_requirement_counts()` | `dict[str, int]` | Counts per requirement |
| `is_ready_for(requirement, min_n=30)` | `bool` | True when log has enough observations to displace synthetic defaults |
| `build_index(min_n=30)` | `dict[str, tuple[float, float, int]]` | Build `(min, max, n)` index from log |
| `merge_with_synthetic(synthetic, min_n=30)` | `dict[str, tuple[float, float, int]]` | Compose real telemetry with synthetic fallback |
| `save(path=None)` | `None` | Persist log as JSONL |
| `load(path)` (classmethod) | `NumericObservationLog` | Load log from disk |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_numeric_range_index(observations)` | `dict[str, tuple[float, float, int]]` | Aggregate observations into `(min, max, n)` index |
| `check_numeric_range(requirement, value, boolean_pass, index, margin=0.0)` | `NumericRangeVerdict` | Two-attestor decision on state-tracking requirement |
| `adult_income_numeric_observations()` | `list[NumericObservation]` | Synthetic but plausible observations for default cohort |

---

## policy_adversarial

`knowlytix/harness/governance/policy_adversarial.py`

### `AdversarialPolicyProbe`

```python
@dataclass(frozen=True)
class AdversarialPolicyProbe:
    probe_id: str
    agent: str
    action: str
    args: dict
    context: str
    attack_type: str
    safe_decisions: tuple[str, ...]   # subset of {"BLOCK", "REVIEW", "ALLOW"}
    provenance: str = ""
    inject_rules: tuple[dict, ...] = ()
```

### `AdversarialPolicyReport`

```python
@dataclass
class AdversarialPolicyReport:
    n_total: int
    per_attack_n: dict[str, int]
    per_attack_safe: dict[str, int]
    per_attack_unsafe: dict[str, list[tuple[AdversarialPolicyProbe, str, str]]] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `safety_rate()` | `float` | Ratio of safe decisions to total probes |
| `summary()` | `str` | Human-readable report summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_policy_adversarial_cohort(n_per_type=5, seed=42)` | `list[AdversarialPolicyProbe]` | Generate adversarial probes across 10 attack classes |
| `sweep_policy_adversarial(engine, cohort, quorum_required=2, keep_unsafe_examples=5)` | `AdversarialPolicyReport` | Run engine over adversarial cohort |

---

## policy_baselines

`knowlytix/harness/governance/policy_baselines.py`

### `BaselineDecision`

```python
@dataclass
class BaselineDecision:
    action: str
    allowed: bool
    blocking_rules: list[str] = []
    reason: str = ""
```

### `PromptGuardrailBaseline`

Regex/keyword guardrails as written into an LLM system prompt.

| Method | Returns | Description |
|--------|---------|-------------|
| `check(action, args, context)` | `BaselineDecision` | Check action against hardcoded rules |

### `JaccardPolicyBaseline`

Token-overlap heuristic over YAML rules.

```python
JaccardPolicyBaseline(
    rules: list[PolicyRule],
    threshold: float = 0.5,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `check(action, args, context)` | `BaselineDecision` | Check action via Jaccard similarity |

### `BaselineContrastReport`

```python
@dataclass
class BaselineContrastReport:
    n_total: int
    engine_correct: int
    baseline_correct: int
    uplift: int
    per_class_engine: dict[str, int] = {}
    per_class_baseline: dict[str, int] = {}
    per_class_n: dict[str, int] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `engine_accuracy()` | `float` | Engine accuracy ratio |
| `baseline_accuracy()` | `float` | Baseline accuracy ratio |
| `summary()` | `str` | Human-readable contrast report |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `compare_to_engine(engine, baseline, cohort, quorum_required=2)` | `BaselineContrastReport` | Run engine and baseline on cohort |

---

## policy_bench

`knowlytix/harness/governance/policy_bench.py`

### `LatencyReport`

```python
@dataclass
class LatencyReport:
    n_calls: int
    p50_us: float
    p95_us: float
    p99_us: float
    p_max_us: float
    p_mean_us: float
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable latency summary |

### `DriftSimulationReport`

```python
@dataclass
class DriftSimulationReport:
    n_healthy: int
    n_perturbed: int
    healthy_state: str
    perturbed_state: str
    healthy_should_alert: bool
    perturbed_should_alert: bool
    healthy_should_recalibrate: bool
    perturbed_should_recalibrate: bool
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable drift simulation summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bench_check_with_quorum(engine, probe, n_calls=10_000, quorum_required=2, warmup=100)` | `LatencyReport` | Microbenchmark `check_with_quorum` |
| `simulate_drift(engine, bundle, cohort, n_healthy=200, n_perturbed=200, perturbation="alias_rotation", quorum_required=2)` | `DriftSimulationReport` | Simulate healthy vs perturbed streams |

---

# Plan hardening

---

## plan_policy

`knowlytix/harness/governance/plan_policy.py`

### `PlanRule`

```python
@dataclass
class PlanRule:
    rule_id: str
    description: str
    predicate: Callable[[list[str]], bool]
    severity: str = "blocking"
```

### `PolicyVerdict`

```python
@dataclass
class PolicyVerdict:
    allowed: bool
    reasons: list[str] = []
    rules_checked: int = 0
    blocking_failures: list[str] = []
    advisory_failures: list[str] = []
```

### `PlanPolicy`

Run a declarative rule pack against a plan.

```python
PlanPolicy(rules: list[PlanRule])
```

| Method | Returns | Description |
|--------|---------|-------------|
| `check_plan(plan)` | `PolicyVerdict` | Evaluate plan against all rules |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `data_science_rules()` | `list[PlanRule]` | Standard eleven-rule pack for data-science workflow |

---

## plan_adversarial

`knowlytix/harness/governance/plan_adversarial.py`

### `AdversarialPlan`

```python
@dataclass
class AdversarialPlan:
    plan: list[str]
    source: str               # "llm:<bias>" or "mutate:<strategy>"
    bias: str = ""
    severity: int = 0
    notes: str = ""
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert plan to dictionary |

### Constants

- `CANONICAL_STEPS: list[str]` -- standard workflow step names
- `BIAS_PROMPTS: dict[str, str]` -- system prompts for LLM bias induction

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `generate_llm_plans(bias, n=20, model_id="claude-opus-4-6", cache_path=None, temperature=0.8, max_tokens=300)` | `list[AdversarialPlan]` | Generate plans via Claude under bias prompt |
| `mutate_plan(plan, severity=1, rng=None)` | `list[str]` | Apply random mutations to a plan |
| `build_mutation_cohort(base_plan=None, n_per_severity=20, seed=42)` | `list[AdversarialPlan]` | Structured mutation cohort at severities 1-5 |
| `build_adversarial_cohort(llm_cache_path, bias_counts=None, structural_per_severity=20, seed=42, regenerate_llm=False, model_id="claude-opus-4-6")` | `list[AdversarialPlan]` | Assemble LLM + structural adversaries |

---

## plan_benchmark

`knowlytix/harness/governance/plan_benchmark.py`

### `PlanRecord`

```python
@dataclass
class PlanRecord:
    plan: list[str]
    label: str                  # "valid" | "invalid"
    plan_class: str
    source: str
    severity: int = 0
    notes: str = ""
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert to dictionary with `"class"` key |

### Constants

- `CANONICAL_VALID_PLAN: list[str]` -- reference valid plan
- `STEP_ALIASES: dict[str, list[str]]` -- canonical steps and aliases
- `HALLUCINATED_STEPS: list[str]` -- plausible but non-canonical steps
- `DEFAULT_COUNTS: dict[str, int]` -- default class counts (~400)

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_cohort(seed=42, n_per_class=None)` | `list[PlanRecord]` | Build deterministic labeled plan cohort |
| `save_cohort(cohort, path)` | `None` | Write cohort to JSON-Lines file |
| `load_cohort(path)` | `list[PlanRecord]` | Load cohort from JSON-Lines file |
| `cohort_summary(cohort)` | `dict` | Per-class counts and label split |

---

# Orchestrator hardening (Ch 16)

---

## orchestrator_policy

`knowlytix/harness/governance/orchestrator_policy.py`

### `OrchestratorRule`

```python
@dataclass(frozen=True)
class OrchestratorRule:
    rule_id: str
    description: str
    predicate: Callable[[list[str], Any], bool]
    severity: str = "blocking"
```

### `OrchestratorPolicyVerdict`

```python
@dataclass
class OrchestratorPolicyVerdict:
    allowed: bool
    reasons: list[str] = []
    rules_checked: int = 0
    blocking_failures: list[str] = []
    advisory_failures: list[str] = []
```

### `OrchestratorPolicy`

```python
OrchestratorPolicy(rules: list[OrchestratorRule])
```

| Method | Returns | Description |
|--------|---------|-------------|
| `check(plan, scenario)` | `OrchestratorPolicyVerdict` | Evaluate `(plan, scenario)` against all rules |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `default_orchestrator_rules()` | `list[OrchestratorRule]` | Default eight-rule pack for data-science orchestrator |

---

## orchestrator_validation

`knowlytix/harness/governance/orchestrator_validation.py`

### `TerminalState` (Literal type)

```python
TerminalState = Literal["completed", "failed", "aborted"]
```

### `LabeledPlan`

```python
@dataclass
class LabeledPlan:
    plan: list[str]
    scenario: Scenario
    expected_state: TerminalState
    expected_pass_rate: float = 1.0
    expected_violations: int = 0
    provenance: str = ""
```

### `OrchestratorOODReport`

```python
@dataclass
class OrchestratorOODReport:
    n_total: int
    per_state_n: dict[str, int]
    confusion: dict[tuple[str, str], int]
    far_per_state: dict[str, float]
    frr_per_state: dict[str, float]
    pass_rate_ece: float
    abstain_rate: float
    review_rate: float
    mean_actual_pass: float
    extra: dict[str, float] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable report summary |

### `ReplayBatteryResult`

```python
@dataclass
class ReplayBatteryResult:
    cohort_size: int
    snapshots: list[dict]
    cohort_hash: str
    n_state_matches: int
    n_pass_rate_matches: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable result summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `validate_orchestrator(orch, cohort, fresh_memory_factory=None)` | `OrchestratorOODReport` | Run full `GovernedOrchestrator` over labeled cohort and report metrics |
| `orchestrator_snapshot(result, lp=None)` | `dict` | Audit-grade summary of one orchestrator run |
| `cohort_hash(snapshots)` | `str` | Deterministic SHA-256 over canonical snapshots |
| `orchestrator_replay_battery(orch, cohort, fresh_memory_factory=None)` | `ReplayBatteryResult` | Run every labeled plan and produce single audit hash |
| `compare_replay_batteries(old, new)` | `dict` | Diff two battery runs; report per-scenario state changes |
| `default_labeled_cohort(Scenario)` | `list[LabeledPlan]` | Reusable minimal labeled cohort covering FSM paths |

---

## orchestrator_monitor

`knowlytix/harness/governance/orchestrator_monitor.py`

### `RunSnapshot`

```python
@dataclass
class RunSnapshot:
    state: str
    n_steps: int
    n_violations: int
    verdict: str
    pass_rate: float
    n_passed: int
    n_total: int
    tools: tuple[str, ...]
```

| Method | Returns | Description |
|--------|---------|-------------|
| `from_result(result)` (classmethod) | `RunSnapshot` | Construct from `OrchestratorResult` |

### `OrchestratorMonitorState`

```python
@dataclass
class OrchestratorMonitorState:
    n_observed: int
    window_full: bool
    completed_rate: float
    failed_rate: float
    aborted_rate: float
    abstain_rate: float
    review_rate: float
    mean_pass_rate: float
    slo_violation: str | None
    severity: str
    alert: bool
    recalibrate: bool
```

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable state summary |

### `OrchestratorMonitor`

Rolling-window monitor over orchestrator runs tracking SLO compliance.

```python
OrchestratorMonitor(
    bundle,
    window_size: int = 50,
    min_runs_for_alert: int = 10,
    recalibrate_requires_full_window: bool = True,
    low_pass_rate_threshold: float = 0.80,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `record(result_or_snapshot)` | `None` | Record one run |
| `state()` | `OrchestratorMonitorState` | Get current monitor state |
| `should_alert()` | `bool` | True if alert condition triggered |
| `should_recalibrate()` | `bool` | True if recalibration recommended |
| `reset()` | `None` | Clear buffer and reset monitor |

---

## orchestrator_adversarial

`knowlytix/harness/governance/orchestrator_adversarial.py`

### `AdversarialPlan`

```python
@dataclass(frozen=True)
class AdversarialPlan:
    plan: tuple[str, ...]
    scenario_name: str
    scenario_required: tuple[str, ...]
    scenario_forbidden: tuple[str, ...]
    scenario_expected_seq: tuple[str, ...]
    scenario_max_tool_calls: int
    attack_type: str
    safe_states: tuple[str, ...]
    provenance: str = ""
    dataset_name_override: str | None = None
```

### `AdversarialReport`

```python
@dataclass
class AdversarialReport:
    n_total: int
    per_attack_n: dict[str, int]
    per_attack_safe: dict[str, int]
    per_attack_unsafe: dict[str, list[tuple[AdversarialPlan, str, str]]] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `safety_rate()` | `float` | Ratio of safe terminal states to total probes |
| `summary()` | `str` | Human-readable report summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_orchestrator_adversarial_cohort(n_per_type=5, seed=42)` | `list[AdversarialPlan]` | Generate attack plans across 6 classes |
| `sweep_orchestrator_adversarial(orch, cohort, Scenario, fresh_memory_factory=None, keep_unsafe_examples=5)` | `AdversarialReport` | Run orchestrator over adversarial cohort |

---

## orchestrator_baselines

`knowlytix/harness/governance/orchestrator_baselines.py`

### `JaccardClaimVerifier`

Surface-level baseline: Jaccard token similarity, no embeddings.

```python
JaccardClaimVerifier(
    store: GMSExpertStore,
    theta: float = 0.6,
    name: str = "jaccard_baseline",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `verify_claim(claim)` | `ClaimVerdict` | Verify fact or numeric claim |
| `verify_all(claims)` | `OverallVerdict` | Verify all claims |

### `RandomClaimVerifier`

```python
RandomClaimVerifier(
    p_pass: float = 0.5,
    seed: int = 42,
    name: str = "random_baseline",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `verify_claim(claim)` | `ClaimVerdict` | Random PASS with probability `p_pass` |
| `verify_all(claims)` | `OverallVerdict` | Verify all claims |

### `BaselineComparisonReport`

```python
@dataclass
class BaselineComparisonReport:
    cohort_size: int
    baseline_name: str
    calibrated_correct: int
    baseline_correct: int
    calibrated_state_match: int
    baseline_state_match: int
    calibrated_pass_rate_avg: float
    baseline_pass_rate_avg: float
    examples_calibrated_correct_baseline_wrong: list = []
    examples_baseline_correct_calibrated_wrong: list = []
```

| Method | Returns | Description |
|--------|---------|-------------|
| `uplift_correct()` | `int` | Calibrated correct minus baseline correct |
| `uplift_state()` | `int` | Calibrated state match minus baseline state match |
| `summary()` | `str` | Human-readable comparison report |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `compare_to_baseline(orch_calibrated, orch_baseline, cohort, keep_examples=5, fresh_memory_factory=None)` | `BaselineComparisonReport` | Run same cohort through two orchestrators |

---

# Release gate hardening (Ch 15)

---

## release_gate_co_attestor

`knowlytix/harness/governance/release_gate_co_attestor.py`

### `Verdict` (Literal type)

```python
Verdict = Literal["pass", "flag", "block"]
```

### `SPCSlopeConfig`

```python
@dataclass(frozen=True)
class SPCSlopeConfig:
    min_acc_slope: float = -0.005
    min_conf_slope: float = -0.005
    min_fair_slope: float = -0.005
    max_rob_slope: float = +0.005
    min_runs_for_alert: int = 5
```

| Method | Returns | Description |
|--------|---------|-------------|
| `per_metric()` | `dict[str, tuple[str, float]]` | Return metric thresholds with direction (`"below"`/`"above"`) |

### `CoAttestorVerdict`

```python
@dataclass
class CoAttestorVerdict:
    verdict: Verdict
    structural_passed: bool
    slope_alerted: bool
    blocking_reasons: list[str]
    flag_reasons: list[str]
    structural_result: GateResult
    spc_slopes: dict[str, float] = {}
```

### `SPCSlopeCoAttestor`

```python
SPCSlopeCoAttestor(
    gate: TieredReleaseGate,
    config: SPCSlopeConfig | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(metrics, tier, trend=None)` | `CoAttestorVerdict` | Three-state verdict on this run |
| `calibrate_from_canonical(canonical_runs, *, target_quantile=0.05)` | `SPCSlopeConfig` | Fit slope thresholds to healthy baseline distribution |

---

## release_gate_validator_alt

`knowlytix/harness/governance/release_gate_validator_alt.py`

### `AltAgreementReport`

```python
@dataclass
class AltAgreementReport:
    n: int
    n_agree: int
    agreement_rate: float
    disagreements: list[dict] = []
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `alt_check(metrics, t)` | `bool` | Independent boolean truth-table check of gate thresholds |
| `assert_gate_alt_agreement(gate, cohort, *, raise_on_disagreement=True)` | `AltAgreementReport` | Compare gate verdicts to alt-validator on every `(probe, tier)` |

---

## release_gate_invariants

`knowlytix/harness/governance/release_gate_invariants.py`

### `InvariantViolation`

```python
@dataclass
class InvariantViolation:
    invariant: str
    description: str
    offending_input: dict | None = None
```

### `InvariantReport`

```python
@dataclass
class InvariantReport:
    n_grid_points: int
    n_pass: int
    n_violations: int
    violations: list[InvariantViolation] = []
    invariants_checked: list[str] = []
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `verify_release_gate_invariants(gate=None, *, resolution=0.10)` | `InvariantReport` | Exhaustively verify five invariants on gate over metric grid |

---

## release_gate_calibration

`knowlytix/harness/governance/release_gate_calibration.py`

### `CalibratedThresholdSet`

```python
@dataclass(frozen=True)
class CalibratedThresholdSet:
    tier: str
    min_accuracy: ThresholdDistribution | None
    min_confidence: ThresholdDistribution | None
    max_robustness_drop: ThresholdDistribution | None
    min_fairness_air: ThresholdDistribution | None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_gate_thresholds(fallback, source)` | `GateThresholds` | Materialize calibrated values with fallback |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bootstrap_metric_threshold(cohort, metric_name, target_tier, n_bootstrap=200, seed=42)` | `ThresholdDistribution \| None` | Bootstrap single per-metric threshold |
| `calibrate_release_gate_thresholds(cohort, n_bootstrap=200, seed=42)` | `dict[str, CalibratedThresholdSet]` | Bootstrap-calibrate every `(tier, metric)` cell |
| `enforce_monotonic_ordering(tiers)` | `dict[str, GateThresholds]` | Repair thresholds to enforce monotonicity |
| `policy_declared_baseline()` | `dict[str, GateThresholds]` | Return shipped class defaults |

---

## release_gate_adversarial

`knowlytix/harness/governance/release_gate_adversarial.py`

### `AdversarialRunProbe`

```python
@dataclass
class AdversarialRunProbe:
    probe_id: str
    attack_class: str
    target_tier: str
    metrics: RunMetrics | None = None
    custom_tiers: dict[str, GateThresholds] | None = None
    trend: RunTrend | None = None
    safe_states: tuple[str, ...] = ("gate_block",)
    description: str = ""
```

### `AdversarialReleaseGateReport`

```python
@dataclass
class AdversarialReleaseGateReport:
    n_probes: int
    n_safe: int
    safety_rate: float
    per_class: dict[str, dict[str, float]] = {}
    per_probe: list[dict] = []
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_release_gate_adversarial_cohort(n_per_class=20, seed=42, gate=None)` | `list[AdversarialRunProbe]` | Generate 200 probes across 10 attack classes |
| `sweep_release_gate_adversarial(gate, cohort, spc_alert_threshold=-0.005)` | `AdversarialReleaseGateReport` | Run gate over adversarial cohort |

---

## release_gate_baselines

`knowlytix/harness/governance/release_gate_baselines.py`

### `FixedThresholdBaseline`

Single-cutoff baseline on accuracy only.

```python
FixedThresholdBaseline(min_accuracy: float = 0.80)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `decide(m, tier)` | `bool` | True if `accuracy >= threshold` |

### `RandomBaseline`

```python
RandomBaseline(seed: int = 42, p_pass: float = 0.5)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `decide(m, tier)` | `bool` | Random pass with probability `p_pass` |

### `BaselineContrastReport`

```python
@dataclass
class BaselineContrastReport:
    baseline_name: str
    n_probes: int
    per_tier: dict[str, dict[str, int]] = {}
    n_correct_gate: int = 0
    n_correct_baseline: int = 0
    uplift: int = 0
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `compare_to_gate(gate, cohort, baseline)` | `BaselineContrastReport` | Run gate and baseline on cohort |

---

## release_gate_bench

`knowlytix/harness/governance/release_gate_bench.py`

### `ReleaseGateLatencyReport`

```python
@dataclass
class ReleaseGateLatencyReport:
    operation: str
    n: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    slo_us: float
    slo_violations: int
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bench_check_all_tiers(gate, n=1000, slo_us=1000.0, seed=42)` | `ReleaseGateLatencyReport` | Microbenchmark `gate.check_all_tiers` |
| `bench_co_attestor(gate, n=1000, slo_us=2000.0, seed=42)` | `ReleaseGateLatencyReport` | Microbenchmark SPC slope co-attestor |
| `bench_audit_record(gate, n=1000, slo_us=5000.0, seed=42)` | `ReleaseGateLatencyReport` | Microbenchmark audit-record materialization |

---

# LLM governance (Ch 11)

---

## llm_bundle

`knowlytix/harness/governance/llm_bundle.py`

### `LLMOperatingPoint`

```python
@dataclass(frozen=True)
class LLMOperatingPoint:
    adversarial_floor: float = 1.0
    max_backend_error_rate: float = 0.05
```

### `ConstrainedLLMBundle`

Frozen artifact bundle for a constrained-LLM deployment.

```python
@dataclass
class ConstrainedLLMBundle:
    schema_version: str
    bundle_id: str
    created_at: str
    source_store_path: str
    min_plausibility: float | None
    tau_contra: float | None
    tau_ent: float | None
    tau_path: float | None
    tension_status: str               # "calibrated" | "degenerate"
    store_fingerprint: StoreFingerprint
    allowed_tools: tuple[str, ...]
    forbidden_relations: tuple[str, ...]
    numeric_bounds: dict[str, tuple[float, float]] = {}
    forbidden_tool_args: dict[str, tuple[str, ...]] = {}
    operating_point: LLMOperatingPoint = LLMOperatingPoint()
    adversarial_safety_rate: float = 0.0
    adversarial_n_probes: int = 0
    backend_model_name: str = ""
    calibration_seed: int = 42
    signature: str = ""
    signing_method: SigningMethod = "sha256"
```

| Method | Returns | Description |
|--------|---------|-------------|
| `canonical_payload()` | `bytes` | Bytes used for signing (excludes signature field) |
| `freeze(key=None)` | `ConstrainedLLMBundle` | Compute and stamp HMAC-SHA-256 signature |
| `verify_signature(key=None)` | `bool` | Verify signature against canonical payload |
| `assert_store_matches(store)` | `None` | Refuse to bind if live store fingerprint differs |
| `assert_meets_slo()` | `None` | Refuse to bind if measured safety rate < SLO floor |
| `save(path)` | `None` | Persist bundle to pickle |
| `load(path, key=None)` (classmethod) | `ConstrainedLLMBundle` | Load and verify bundle |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `make_llm_bundle_id(source_store_path, created_at)` | `str` | Content-addressed bundle ID |
| `utcnow_iso()` | `str` | ISO timestamp at UTC now |
| `build_constrained_llm_bundle(store, verify_config, allowed_tools, co_attestor, adversarial_safety_rate, adversarial_n_probes, backend_model_name, operating_point=None, calibration_seed=42, sign_key=None)` | `ConstrainedLLMBundle` | Assemble, fingerprint, and sign a constrained-LLM bundle |
| `bind_constrained_llm_from_bundle(bundle, store, llm_backend)` | `ConstrainedLLM` | Construct runtime `ConstrainedLLM` from a frozen bundle; refuses on store/SLO mismatch |

### Constants

- `LLM_BUNDLE_SCHEMA_VERSION: str = "1.0"`

---

## llm_co_attestor

`knowlytix/harness/governance/llm_co_attestor.py`

### `LLMCoAttestor`

Rule-based second attestor for `ConstrainedLLM` with deterministic blocking rules.

```python
@dataclass
class LLMCoAttestor:
    forbidden_relations: frozenset[str] = frozenset()
    numeric_bounds: dict[str, tuple[float, float]] = {}
    forbidden_tool_args: dict[str, frozenset[str]] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `check_proposal(context, tool, args)` | `dict` | Rule decision for proposed tool call: `{allowed: bool, reason: str}` |
| `check_claim(triple)` | `dict` | Rule decision for extracted claim: `{allowed: bool, reason: str}` |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `default_ds_co_attestor()` | `LLMCoAttestor` | Default rules for data-science workflow |

---

## llm_adversarial

`knowlytix/harness/governance/llm_adversarial.py`

### `LLMAdversarialProbe`

```python
@dataclass(frozen=True)
class LLMAdversarialProbe:
    name: str
    probe_type: str
    factory: Callable[[], ConstrainedLLM]
    invocation: Callable[[ConstrainedLLM], list]
    safe_predicate: Callable[[list], tuple[bool, str]]
```

### `LLMAdversarialReport`

```python
@dataclass
class LLMAdversarialReport:
    n_total: int
    per_type_n: dict[str, int]
    per_type_safe: dict[str, int]
    per_type_unsafe: dict[str, list[tuple[str, str]]] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `safety_rate()` | `float` | Ratio of safe outcomes to total probes |
| `summary()` | `str` | Human-readable report summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_llm_adversarial_cohort(trained_store_factory, empty_store_factory, verifier_factory, co_attestor_factory=None, allowed_tools=None)` | `list[LLMAdversarialProbe]` | Build adversarial cohort for `ConstrainedLLM` |
| `sweep_llm_adversarial(cohort)` | `LLMAdversarialReport` | Execute cohort and report per-type safety |

---

# Behavioral contract hardening (Ch 13)

---

## contract_bundle

`knowlytix/harness/governance/behaviors/contract_bundle.py`

### `ContractFingerprint`

```python
@dataclass(frozen=True)
class ContractFingerprint:
    states_hash: str
    transitions_hash: str
    permissions_hash: str
    invariants_hash: str
    budget: int
    require_human_approval: bool
```

| Method | Returns | Description |
|--------|---------|-------------|
| `matches(other)` | `bool` | Check if all components match another fingerprint |
| `diff(other)` | `dict[str, tuple]` | Report differing components as `(mine, theirs)` tuples |

### `ContractOperatingPoint`

```python
@dataclass(frozen=True)
class ContractOperatingPoint:
    min_detection_rate: float = 1.0
    max_false_positive_rate: float = 0.0
```

### `BehavioralContractBundle`

```python
@dataclass
class BehavioralContractBundle:
    schema_version: str
    bundle_id: str
    created_at: str
    contract_name: str
    fingerprint: ContractFingerprint
    cohort_hash: str
    holonomy_threshold: float
    holonomy_reference_sequence: tuple[str, ...]
    holonomy_calibration_percentile: float
    detection_rate: float
    false_positive_rate: float
    safety_rate: float
    n_probes: int
    operating_point: ContractOperatingPoint = ContractOperatingPoint()
    signature: str = ""
    signing_method: SigningMethod = "sha256"
```

| Method | Returns | Description |
|--------|---------|-------------|
| `canonical_payload()` | `bytes` | Bytes used for signing |
| `freeze(key=None)` | `BehavioralContractBundle` | Compute and stamp HMAC-SHA-256 signature |
| `verify_signature(key=None)` | `bool` | Verify signature |
| `assert_contract_matches(contract)` | `None` | Refuse to bind if live contract fingerprint differs |
| `assert_meets_slo()` | `None` | Refuse to bind if detection rate < floor or FPR > ceiling |
| `save(path)` | `None` | Persist bundle to pickle |
| `load(path, key=None)` (classmethod) | `BehavioralContractBundle` | Load and verify bundle |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_contract_fingerprint(contract)` | `ContractFingerprint` | Hash each component of a contract independently |
| `make_contract_bundle_id(contract_name, created_at)` | `str` | Content-addressed bundle ID |
| `utcnow_iso()` | `str` | ISO timestamp at UTC now |
| `build_behavioral_contract_bundle(contract, cohort, sweep_report, holonomy_attestor, holonomy_calibration_percentile, operating_point=None, sign_key=None)` | `BehavioralContractBundle` | Assemble and HMAC-sign a contract bundle |
| `bind_contract_from_bundle(bundle, contract)` | `HolonomyCoAttestor` | Construct runtime evaluator from bundle |

### Constants

- `CONTRACT_BUNDLE_SCHEMA_VERSION: str = "1.0"`

---

## contract_co_attestor

`knowlytix/harness/governance/behaviors/contract_co_attestor.py`

### `TwoAttestorContractVerdict`

```python
@dataclass(frozen=True)
class TwoAttestorContractVerdict:
    decision: Literal["pass", "flag", "block"]
    fsm_valid: bool
    fsm_violations: tuple[str, ...]
    holonomy_score: float | None
    holonomy_threshold: float | None
    flagged_reason: str | None
```

### `HolonomyCoAttestor`

Continuous-deviation co-attestor for FSM-valid sequences using holonomy / edit distance.

```python
@dataclass
class HolonomyCoAttestor:
    reference_sequence: tuple[str, ...]
    threshold: float = 0.5
    store: object | None = None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `score(actual)` | `float` | Holonomy score for actual sequence |
| `verdict(contract, actual, evaluator=None)` | `TwoAttestorContractVerdict` | Run FSM + holonomy and return combined verdict |
| `calibrate_threshold(valid_sequences, *, percentile=0.95)` | `float` | Set threshold to percentile quantile of scores |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `canonical_compliant_workflows()` | `list[list[str]]` | Operator-curated reference workflows for DS contracts |

---

## contract_adversarial

`knowlytix/harness/governance/behaviors/contract_adversarial.py`

### `ContractAdversarialProbe`

```python
@dataclass(frozen=True)
class ContractAdversarialProbe:
    name: str
    sequence: tuple[str, ...]
    expected_valid: bool
    expected_violation_kind: str | None
    provenance: str = ""
```

### `ContractAdversarialReport`

```python
@dataclass
class ContractAdversarialReport:
    n_total: int
    per_kind_n: dict[str, int]
    per_kind_safe: dict[str, int]
    per_kind_unsafe: dict[str, list[tuple[str, str]]] = {}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `safety_rate()` | `float` | Overall safety rate |
| `detection_rate()` | `float` | Fraction of violation probes detected |
| `false_positive_rate()` | `float` | Fraction of valid probes rejected |
| `summary()` | `str` | Human-readable report summary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_contract_adversarial_cohort(contract)` | `list[ContractAdversarialProbe]` | Generate systematic cohort for contract |
| `sweep_contract_adversarial(contract, cohort, evaluator=None)` | `ContractAdversarialReport` | Validate every probe and report per-kind safety |
| `cohort_hash(cohort)` | `str` | SHA-256 over canonical probe tuples |

---

## contract_replay

`knowlytix/harness/governance/behaviors/contract_replay.py`

### `ContractReplayResult`

```python
@dataclass
class ContractReplayResult:
    contract_name: str
    contract_fingerprint: str
    cohort_hash: str
    per_probe: dict[str, str]   # probe_name -> SHA-256 of validation outcome
```

### `ContractReplayRegressionError`

Raised when replay diverges from the baseline.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `replay_contract(contract, cohort, evaluator=None)` | `ContractReplayResult` | Run every probe through contract and snapshot verdicts as hashes |
| `assert_no_contract_regression(baseline, candidate)` | `None` | Raise if candidate snapshot differs from baseline |

---

# Multi-agent hardening (Ch 17)

---

## multi_agent_audit

`knowlytix/harness/governance/agents/multi_agent_audit.py`

### `MultiAgentReasonChain`

```python
@dataclass
class MultiAgentReasonChain:
    blocking_reasons: list[str] = []
    flag_reasons: list[str] = []
```

### `MultiAgentAuditRecord`

JSON-serializable audit record for a single tool-call decision.

```python
@dataclass
class MultiAgentAuditRecord:
    timestamp: float
    role_name: str
    tool_name: str
    args_hash: str
    structural_passed: bool
    decision: str                # "allow" | "block" | "deny"
    reason_chain: dict
    co_attestor_verdict: dict | None = None
    bundle_id: str = ""
    cohort_hash: str = ""
    signing_method: str = ""
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert to dictionary representation |
| `to_json(**kwargs)` | `str` | Serialize to JSON string with sorted keys |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `to_multi_agent_audit_record(orch, role_name, tool_name, args=None, co_attestor_verdict=None, bundle=None)` | `MultiAgentAuditRecord` | Build audit record for a `(role, tool, args)` decision |
| `replay_multi_agent_audit_log(record)` | `str` | Reconstruct verdict from saved audit record alone; returns `"ALLOW"`, `"FLAG"`, `"BLOCK"`, or `"DENY"` |

---

## multi_agent_bundle

`knowlytix/harness/governance/agents/multi_agent_bundle.py`

### `MultiAgentOperatingPoint`

```python
@dataclass(frozen=True)
class MultiAgentOperatingPoint:
    min_adversarial_safety_rate: float = 0.95
    min_ood_decision_match_rate: float = 0.95
    max_per_role_tool_psi: float = 0.10
    max_handoff_latency_p99_us: float = 1_000_000.0
    min_uplift_over_naive: int = 0
    min_alt_validator_agreement: float = 1.0
```

### `MultiAgentBundle`

```python
@dataclass
class MultiAgentBundle:
    schema_version: str
    bundle_id: str
    created_at: str
    source_dataset: str
    per_role_fingerprint: dict[str, str]
    cohort_hash: str
    cohort_size: int
    per_agent_tool_frequency: dict | None = None
    handoff_latency_baseline: dict | None = None
    handoff_latency_p99_dist: ThresholdDistribution | None = None
    operating_point: MultiAgentOperatingPoint = MultiAgentOperatingPoint()
    validation_metrics: dict[str, float] = {}
    invariants_proof_hash: str = ""
    signature: str = ""
    signing_method: SigningMethod = "sha256"
    attestor_signatures: list[dict] = []
```

| Method | Returns | Description |
|--------|---------|-------------|
| `canonical_payload()` | `bytes` | Bytes used for signing |
| `freeze(key=None)` | `MultiAgentBundle` | Compute and stamp HMAC-SHA-256 signature |
| `verify(key=None)` | `bool` | Verify signature |
| `save(path)` | `None` | Persist bundle to pickle |
| `load(path, key=None)` (classmethod) | `MultiAgentBundle` | Load and verify bundle |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `make_bundle_id(source_dataset, cohort_hash, created_at)` | `str` | Content-addressed bundle ID |
| `utcnow_iso()` | `str` | ISO timestamp at UTC now |
| `build_multi_agent_bundle(orch, cohort, source_dataset="adult_income", per_agent_freq=None, handoff_baseline=None, handoff_p99_dist=None, operating_point=None, validation_metrics=None, invariants_resolution=True)` | `MultiAgentBundle` | Produce a calibrated, signed multi-agent bundle |
| `bind_orchestrator_from_bundle(orch, bundle, cohort=None, adversarial_safety_rate=None, ood_decision_match_rate=None, alt_validator_agreement=None, handoff_latency_p99_us=None, uplift_over_naive=None)` | `MultiAgentOrchestrator` | Bind orchestrator to verified bundle; refuses on fingerprint/cohort/operating-point mismatch |

### Constants

- `MULTI_AGENT_BUNDLE_SCHEMA_VERSION: str = "1.0"`

---

## multi_agent_monitor

`knowlytix/harness/governance/agents/multi_agent_monitor.py`

### `Severity` (Literal type)

```python
Severity = Literal["stable", "slow_drift", "slo_breach"]
```

### `MultiAgentMonitorState`

```python
@dataclass
class MultiAgentMonitorState:
    severity: Severity
    psi_per_role: dict[str, float]
    blocked_attempt_rate: float
    n_runs: int
    alert: bool = False
    recalibrate: bool = False
    slo_violations: dict[str, str] = {}
```

### `MultiAgentMonitor`

```python
MultiAgentMonitor(
    orch: MultiAgentOrchestrator,
    bundle: MultiAgentBundle,
    *,
    window_size: int = 50,
    min_runs_for_alert: int = 20,
    psi_warn: float = 0.05,
    psi_alert: float = 0.10,
    max_blocked_attempt_rate: float = 0.05,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `record_run(per_run_calls, blocked_count)` | `None` | Record a pipeline run with per-role tool counts and block count |
| `evaluate()` | `MultiAgentMonitorState` | Compute PSI per role and blocked-attempt rate over window |

---

## multi_agent_replay

`knowlytix/harness/governance/agents/multi_agent_replay.py`

### `MultiAgentReplayResult`

```python
@dataclass
class MultiAgentReplayResult:
    n_probes: int
    snapshots: list[dict]
    snapshots_hash: str
    bundle_id: str = ""
    cohort_hash: str = ""
```

### `MultiAgentReplayDelta`

```python
@dataclass
class MultiAgentReplayDelta:
    n_probes: int
    n_divergent: int
    divergent_probe_ids: list[str] = []
    matches: bool = True
```

### `MultiAgentReplayRegressionError`

Raised when replay hash diverges from blessed baseline.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `replay_multi_agent_battery(orch, cohort, bundle_id="", cohort_hash="")` | `MultiAgentReplayResult` | Run cohort over orchestrator; return SHA-256 hash over all decisions |
| `compare_multi_agent_replays(a, b)` | `MultiAgentReplayDelta` | Compare two replays; report divergent probes and match status |
| `assert_no_multi_agent_regression(blessed_hash, current, blessed_snapshots=None)` | `None` | Raise if current snapshot hash diverges from blessed baseline |
| `write_blessed_hash(path, replay, approver="", reason="")` | `None` | Write blessed hash and metadata to JSON file |
| `read_blessed_hash(path)` | `str` | Read blessed hash from JSON file |

---

## multi_agent_co_attestor

`knowlytix/harness/governance/agents/multi_agent_co_attestor.py`

### `Verdict` (Literal type)

```python
Verdict = Literal["pass", "flag", "block"]
```

### `MultiAgentCoAttestorVerdict`

```python
@dataclass
class MultiAgentCoAttestorVerdict:
    verdict: Verdict
    structural_passed: bool
    psi_value: float
    blocking_reasons: list[str]
    flag_reasons: list[str]
    role_name: str
    tool_name: str
```

### `MultiAgentCoAttestorConfig`

```python
@dataclass(frozen=True)
class MultiAgentCoAttestorConfig:
    flag_threshold: float = 0.10
    block_threshold: float = float("inf")
    min_window_calls: int = 5
```

### `MultiAgentCoAttestor`

```python
MultiAgentCoAttestor(
    orch: MultiAgentOrchestrator,
    reference_freq: PerAgentToolFrequency | None = None,
    config: MultiAgentCoAttestorConfig | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(role_name, tool_name, args=None, recent_window=None)` | `MultiAgentCoAttestorVerdict` | Three-state verdict on `(role, tool)` call |

---

## multi_agent_validator_alt

`knowlytix/harness/governance/agents/multi_agent_validator_alt.py`

### `MultiAgentAltAgreementReport`

```python
@dataclass
class MultiAgentAltAgreementReport:
    n: int
    n_agree: int
    agreement_rate: float
    disagreements: list[dict] = []
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `alt_can_execute(*, tool_name, allowed_tools, forbidden_tools)` | `bool` | Independent boolean re-implementation of `can_execute` |
| `assert_orchestrator_alt_agreement(orch, cohort=None, *, raise_on_disagreement=True)` | `MultiAgentAltAgreementReport` | Compare every `(role, tool)` decision between orchestrator and alt |

---

## multi_agent_invariants

`knowlytix/harness/governance/agents/multi_agent_invariants.py`

### `MultiAgentInvariantViolation`

```python
@dataclass
class MultiAgentInvariantViolation:
    invariant: str
    description: str
    offending_input: dict | None = None
```

### `MultiAgentInvariantReport`

```python
@dataclass
class MultiAgentInvariantReport:
    n_cells: int
    n_pass: int
    n_violations: int
    violations: list[MultiAgentInvariantViolation] = []
    invariants_checked: list[str] = []
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `verify_multi_agent_invariants(orch, *, tool_vocab=...)` | `MultiAgentInvariantReport` | Exhaustively verify five invariants on orchestrator |

---

## multi_agent_calibration

`knowlytix/harness/governance/agents/multi_agent_calibration.py`

### `PerAgentToolFrequency`

```python
@dataclass(frozen=True)
class PerAgentToolFrequency:
    counts_by_role: dict[str, dict[str, int]]
    total_calls: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `normalized()` | `dict[str, dict[str, float]]` | Per-role probability distributions |
| `to_dict()` | `dict` | Convert to dictionary |

### `HandoffLatencyBaseline`

```python
@dataclass(frozen=True)
class HandoffLatencyBaseline:
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    n_handoffs: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert to dictionary |

### `PerRoleFingerprint`

```python
@dataclass(frozen=True)
class PerRoleFingerprint:
    role_name: str
    fingerprint: str
```

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert to dictionary |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `calibrate_per_agent_tool_frequency(results)` | `PerAgentToolFrequency` | Aggregate tool-call counts from pipeline runs |
| `calibrate_handoff_latency_baseline(results)` | `HandoffLatencyBaseline \| None` | Compute p50/p95/p99 inter-handoff intervals |
| `bootstrap_handoff_latency(results, n_bootstrap=200, seed=42)` | `ThresholdDistribution \| None` | Bootstrap the p99 handoff latency |
| `fingerprint_role(role)` | `PerRoleFingerprint` | Compute deterministic SHA-256 fingerprint |
| `fingerprint_orchestrator(orch)` | `dict[str, str]` | Map `role_name -> fingerprint` for all agents |

---

## multi_agent_adversarial

`knowlytix/harness/governance/agents/multi_agent_adversarial.py`

### `AdversarialMultiAgentProbe`

```python
@dataclass
class AdversarialMultiAgentProbe:
    probe_id: str
    attack_class: str
    target_role: str | None
    tool_name: str | None
    args: dict = {}
    payload: dict = {}
    safe_states: tuple[str, ...] = ("role_block",)
    description: str = ""
```

### `MultiAgentAdversarialReport`

```python
@dataclass
class MultiAgentAdversarialReport:
    n_probes: int
    n_safe: int
    safety_rate: float
    per_class: dict[str, dict[str, float]] = {}
    per_probe: list[dict] = []
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `build_multi_agent_adversarial_cohort(n_per_class=20, seed=42)` | `list[AdversarialMultiAgentProbe]` | Generate 200 probes across 10 attack classes |
| `sweep_multi_agent_adversarial(orch, cohort)` | `MultiAgentAdversarialReport` | Run adversarial cohort against orchestrator |

---

## multi_agent_baselines

`knowlytix/harness/governance/agents/multi_agent_baselines.py`

### `SingleAgentBaseline`

One agent for everything; no role separation.

| Method | Returns | Description |
|--------|---------|-------------|
| `decide(probe)` | `str` | Always `"allow"` |

### `NoPermissionCheckBaseline`

Multi-agent shell but no permission enforcement.

| Method | Returns | Description |
|--------|---------|-------------|
| `decide(probe)` | `str` | Always `"allow"` |

### `PromptOnlyBaseline`

Permission stated in role description but not enforced.

| Method | Returns | Description |
|--------|---------|-------------|
| `decide(probe)` | `str` | Always `"allow"` |

### `MultiAgentBaselineContrastReport`

```python
@dataclass
class MultiAgentBaselineContrastReport:
    baseline_name: str
    n_probes: int
    n_correct_orchestrator: int = 0
    n_correct_baseline: int = 0
    uplift: int = 0
    per_decision: dict[str, dict[str, int]] = {}
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `compare_to_orchestrator(orchestrator_decisions, cohort, baseline)` | `MultiAgentBaselineContrastReport` | Pair-uplift contrast between orchestrator and baseline |

---

## multi_agent_bench

`knowlytix/harness/governance/agents/multi_agent_bench.py`

### `MultiAgentLatencyReport`

```python
@dataclass
class MultiAgentLatencyReport:
    operation: str
    n: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    slo_us: float
    slo_violations: int
```

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `bench_can_execute(orch, n=1000, slo_us=50.0, seed=42)` | `MultiAgentLatencyReport` | Microbenchmark `agent.can_execute` |
| `bench_handoff(orch, n=500, slo_us=200.0)` | `MultiAgentLatencyReport` | Microbenchmark `orchestrator.handoff` |
| `bench_test_boundary(orch, n=500, slo_us=200.0)` | `MultiAgentLatencyReport` | Microbenchmark `orch.test_boundary` |

---

## Document Ingestion

Three ingestion modes, one configuration knob, one callable seam. The
modes correspond to the three-way comparison reported in
FinStructBench paper §6.5.

### Mode contract

| Mode | LLM | Recall vs regex | Use for |
|------|-----|-----------------|---------|
| `"regex"` | not invoked | baseline (100%) | reproducible offline / CI builds |
| `"hybrid"` | required | strict superset; +22% ENM, +14% triples; **0** value drift on shared keys | production; **the orchestrator default** |
| `"llm_only"` | required | ≈ 12.5% ENM recall, near-zero overlap | paper reproduction, debugging |

`hybrid` is bit-identical to `regex` on every shared numeric value; the
LLM contributes only structural decisions (column classification,
entity-column detection) and prose-derived triples — it never produces
numeric values that bypass the SHA-256-hashed register.

### `LLMCallable`

`knowlytix.benchmark.llm_caller.LLMCallable = Callable[[str], str]` — the
single seam every LLM-touching helper accepts. Anything that takes a
prompt and returns a string works: Anthropic, OpenAI, Bedrock, vLLM,
Ollama, mocks. Build one with the convenience constructor:

```python
from knowlytix.benchmark.llm_caller import make_anthropic_callable
llm_callable = make_anthropic_callable()           # bound to Claude Sonnet
```

Or wrap an existing `docknowlytix.core.llm_backend.LLMBackend` directly:

```python
from docknowlytix.core.ingest import _llm_backend_to_callable
llm_callable = _llm_backend_to_callable(my_backend)
```

### `knowlytix.benchmark.ingest.ingest_markdown`

Standalone parser. Library-level default is `"regex"` (conservative).

```python
ingest_markdown(
    path: str,
    document_entity: str | None = None,
    *,
    mode: Literal["regex", "hybrid", "llm_only"] = "regex",
    llm_callable: Callable[[str], str] | None = None,
) -> DocumentGraph
```

| Argument | Required when | Behaviour |
|----------|---------------|-----------|
| `mode="regex"` | always | no LLM; deterministic |
| `mode="hybrid"` | always | regex backbone + LLM at three decision points |
| `mode="llm_only"` | always | LLM extracts everything; regex disabled |
| `llm_callable` | `mode != "regex"` | raises `ValueError` if absent (no silent fallback) |

Deprecated arguments retained for one release with `DeprecationWarning`:
`mode="default"` (alias for `"regex"`), and the `(llm_client, llm_model)`
pair (replaced by `llm_callable`).

### `docknowlytix.core.config.DocGMSConfig.ingest_mode`

Orchestrator-level default. **The bank-grade entry point uses `"hybrid"`.**

```python
@dataclass
class DocGMSConfig:
    ...
    ingest_mode: str = "hybrid"   # "regex" | "hybrid" | "llm_only"
```

Set explicitly in code so a reviewer can tell from the configuration
which path produced any given store.

### `docknowlytix.core.ingest.ingest_document`

Orchestrator. Reads `config.ingest_mode`, builds an `LLMCallable` from
the supplied `LLMBackend`, plumbs both into `ingest_markdown`. Raises
`ValueError` if `ingest_mode != "regex"` and `llm` is `None` — silent
degradation would hide that the LLM-augmented extraction never ran.

```python
ingest_document(
    store: GMSExpertStore,
    document_path: str,
    llm: LLMBackend | None,
    config: DocGMSConfig,
    device: torch.device,
) -> IngestResult
```

### `knowlytix.benchmark.llm_ingest`

Five LLM helpers used by `hybrid` and `llm_only` modes; all take the
callable directly.

| Function | Used in | Role |
|----------|---------|------|
| `llm_classify_columns(llm_callable, headers, rows)` | hybrid fallback | reclassify columns when row-sampling heuristic is ambiguous |
| `llm_detect_entity_column(llm_callable, headers, rows, col_types)` | hybrid fallback | identify the entity column when the name heuristic fails |
| `llm_extract_relations(llm_callable, section_name, text)` | hybrid | extract triples from prose paragraphs the regex cannot reach |
| `llm_extract_table(llm_callable, section_name, subsection, table_text)` | llm_only | extract every ENM entry and triple from a table |
| `llm_extract_bullets_and_prose(llm_callable, section_name, text)` | llm_only | same, for non-tabular content |

### Tests

`tests/test_ingest_modes.py` — 17 hermetic tests covering every mode
contract, every error path, the SHA-256 round-trip, and the
orchestrator wiring. Plus 2 opt-in real-Anthropic tests gated on
`ANTHROPIC_API_KEY` (marker: `requires_anthropic`).

---

## Key Dependencies

### knowlytix.knowledge

| Class/Function | Module | Description |
|---------------|--------|-------------|
| `GMSExpertStore` | `docknowlytix.core.store` | Geometric knowledge graph with trained model |
| `DocGMSConfig` | `docknowlytix.core.config` | Store configuration; **`ingest_mode` default `"hybrid"`** |
| `ingest_document` | `docknowlytix.core.ingest` | Three-mode orchestrator (see Document Ingestion above) |
| `RuntimeLearner` | `docknowlytix.core.learn` | Two-phase learning (sync + async GD) |
| `LearnConfig` | `docknowlytix.core.config` | Learning parameters (`n_steps` controls convergence) |
| `ExtractedTriple` | `docknowlytix.core.extract` | Knowledge triple representation |
| `extract_triples_simple` | `docknowlytix.core.extract` | LLM-based Tier 3 extraction |

### knowlytix.core

| Class | Module | Description |
|-------|--------|-------------|
| `GeometryConfig` | `knowlytix.core.config` | Manifold dimension (`m`) |
| `TrainConfig` | `knowlytix.core.config` | Training parameters (epochs, device) |
| `GeometricSearch` | `knowlytix.core.graph.search` | Multi-hop path finding |
| `RelationalTransport` | `knowlytix.core.graph.transport` | Relational rotation on $S^{m-1}$ |
| `ExactNumericalMemory` | `knowlytix.core.memory.enm` | SHA-256 backed numeric store |
| `DualEmbedding` | `knowlytix.core.graph.embeddings` | Dual semantic + logical embeddings on $S^{m-1}$ |
| `CayleyRotor` | `knowlytix.core.geometry.cayley` | Cayley-parameterized SO(d) relation rotor |

### knowlytix.benchmark

| Class/Function | Module | Description |
|---------------|--------|-------------|
| `DocumentGraph` | `knowlytix.benchmark.graph` | Three-store ground-truth (ENM · triples · phase encoders) |
| `PhaseEncoder` | `knowlytix.benchmark.graph` | `encode()` (angle on $S^1$) + `check_inequality()` |
| `ingest_markdown` | `knowlytix.benchmark.ingest` | Three-mode markdown parser (regex / hybrid / llm_only) |
| `LLMCallable` | `knowlytix.benchmark.llm_caller` | Type alias `Callable[[str], str]` — the LLM seam |
| `make_anthropic_callable` | `knowlytix.benchmark.llm_caller` | Convenience factory bound to Claude Sonnet |
| `llm_classify_columns`, `llm_detect_entity_column`, `llm_extract_relations`, `llm_extract_table`, `llm_extract_bullets_and_prose` | `knowlytix.benchmark.llm_ingest` | Five LLM helpers used by hybrid / llm_only modes |

### knowlytix.harness

| Class | Module | Description |
|-------|--------|-------------|
| `PolicyEngine` | `knowlytix.harness.testing.policy` | Four-layer policy checking |
| `PolicyRule` | `knowlytix.harness.testing.policy` | Single policy rule definition |
| `PolicyDecision` | `knowlytix.harness.testing.policy` | Result of policy check |
| `ToolGateway` | `knowlytix.harness.testing.gateway` | Base tool execution layer |
| `FaultProfile` | `knowlytix.harness.testing.failures` | Fault injection configuration |
