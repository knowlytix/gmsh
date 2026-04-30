# GMSH: Enterprise Testing Platform — Technical Reference

## 1. System Overview

GMSH (Geometric Memory System Harness) is a claim-level verification platform built on top of the DOE-GMS benchmark framework. It transforms LLM/agent evaluation from binary pass/fail scoring into a structured verification pipeline where **every judgment is anchored by GMS geometric primitives**.

The base DOE-GMS system (documented in `TECHNICAL.md`) provides geometric scoring via `GeometricVerdict` — a holistic assessment of geodesic confidence, tension energy, and holonomy defect. GMSH extends this with seven phases:

| Phase | Capability | Key Modules |
|-------|-----------|-------------|
| **1** | Claim Compiler + Typed Verifiers + Failure Taxonomy | `claims.py`, `verifiers.py`, `failures.py`, `verdict.py` |
| **2** | Provenance Chain + Truth Objects | `provenance.py`, `truth.py` |
| **3** | Extended SUT Adapters + Test Generators | `adapters.py`, `test_generators.py` |
| **4** | Versioning + Snapshots | `snapshots.py` |
| **5** | Audit Reporting + Release Gates | `audit.py`, `report_enhanced.py` |
| **6** | Agentic Testing Harness | `gateway.py`, `policy.py`, `scenarios.py`, `runs.py`, `replay.py`, `evaluators_extended.py` |
| **BB-1** | Black-Box Testing: Foundation | `target.py`, `severity.py`, `environment.py`, `oracles.py`, `campaign.py` |
| **BB-2** | Black-Box Testing: Coverage | `generators_taxonomy.py`, `safety.py`, `multiturn.py` |
| **BB-3** | Black-Box Testing: Scale & Polish | `parallel.py`, `versioning.py`, `browser_adapter.py` |
| **7** | GMS-Native Enhancements | `profiling.py`, `cost_tracker.py`, `stateful.py`, `hallucination.py`, `calibration.py`, `multi_agent.py`, `cross_document.py`, `disambiguation.py`, `invariance.py`, `streaming.py`, `live_dashboard.py` |

A cross-cutting component, the **Semantic Verifier** (`semantic.py`), provides LLM-as-judge evaluation calibrated by GMS geometric evidence.

All phases are opt-in via `DOEHarnessConfig` flags, and backwards-compatible with the base system.

### 1.1 Design Principles

1. **GMS is the single source of truth.** Every verification decision calls a GMS method. No Python arithmetic, no tolerance heuristics, no bypasses.
2. **Fail-loud.** If GMS cannot verify a claim (missing phase encoder, entity not in graph, relation not trained), the verifier FAILS rather than silently passing.
3. **Every verification records the GMS method used.** Full auditability from claim to sub-verdict to failure attribution.
4. **Backwards compatibility.** Each GMSH phase is independently toggleable. `TypedVerdict` downcasts to `GeometricVerdict` via `to_geometric_verdict()`.

### 1.2 Module Layout

```
harness/doe/
    # Base DOE-GMS (unchanged)
    __init__.py         # Public API (base + GMSH exports)
    config.py           # DOEHarnessConfig (extended with GMSH fields)
    benchmark.py        # DOEGMSBenchmark orchestrator (extended)
    bridge.py           # Question format adapter (extended with GMSH generators)
    generators.py       # 6 geometric question generators
    evaluators.py       # LLM/Agent evaluators
    judge.py            # GMSJudge (extended with judge_typed)
    tracing.py          # DOETracingGMS (extended with GMSH events)
    cli.py              # CLI entry point (extended with GMSH flags)
    dashboard.py        # HTML report generation

    # GMSH Phase 1
    claims.py           # ClaimCompiler + 7 typed claim classes
    verifiers.py        # 5 structural verifiers + VerificationRouter (+3 WS6 dispatch entries)
    failures.py         # FailureCode enum + FailureAttribution
    verdict.py          # TypedVerdict with claim-level audit trail

    # GMSH Phase 2
    provenance.py       # ProvenanceRegistry + DocumentRecord
    truth.py            # TruthStore + 4 typed truth classes

    # GMSH Phase 3
    adapters.py         # LLMAdapter, AgentAdapter, RAGAdapter
    test_generators.py  # ExactRetrieval, Negative, WorkflowCompliance

    # GMSH Phase 4
    snapshots.py        # SnapshotManager + GMSSnapshot

    # GMSH Phase 5
    audit.py            # AuditPacket, ReleaseGate, AuditReporter (+WS7 tiered gates)
    report_enhanced.py  # Enhanced HTML report sections (+WS7 tiered gate rendering)

    # GMSH Phase 6: Agentic Testing Harness (WS1-WS7)
    gateway.py          # WS1: ToolGateway + FaultProfile + GatewayTranscript + intercept_tool mock injection
    policy.py           # WS2: PolicyEngine + PolicyRule + PolicyDecision
    scenarios.py        # WS3: Scenario + ScenarioRegistry (CRUD, versioning, GMS search)
    runs.py             # WS4: RunState + RunRecord + RunManager (state machine)
    replay.py           # WS5: ReplayManifest + RunComparison + RegressionSuite
    evaluators_extended.py  # WS6: Process/Recovery/Interaction evaluators

    # GMSH Black-Box Testing — Phase 1: Foundation
    target.py           # TargetProfile + IsolationRequirements (SUT identity)
    severity.py         # SeverityClassifier: 4-level severity with GMS geometric signals
    environment.py      # EnvironmentSimulator: deterministic world with mock endpoints + clock control
    oracles.py          # 7 oracle types + CompositeOracle for behavioral verdict
    campaign.py         # CampaignManager: multi-run evaluation orchestration

    # GMSH Black-Box Testing — Phase 2: Coverage
    generators_taxonomy.py  # 5 taxonomy generators: Boundary, Adversarial, Robustness, Recovery, Governance
    safety.py           # SafetyEnforcer + SyntheticDataFramework (S^{m-1} entity sampling)
    multiturn.py        # MultiTurnAdapter: multi-turn scripts with cross-turn GMS consistency

    # GMSH Black-Box Testing — Phase 3: Scale & Polish
    parallel.py         # ParallelExecutor: thread-pool evaluation with retry
    versioning.py       # ArtifactRegistry: versioned fault profiles, policy rules, fixtures
    browser_adapter.py  # BrowserAdapter: Playwright-based UI testing

    # GMSH Phase 7: GMS-Native Enhancements (11 new modules)
    # -- Foundation --
    profiling.py        # S^1 phase-encoded latency/SLA profiling via tension_energy
    cost_tracker.py     # S^1 phase-encoded token/cost tracking via score_triple

    # -- Agent Memory & Behavioral Testing --
    stateful.py         # Teach-then-test memory probes via RuntimeLearner + score_triple
    hallucination.py    # Geodesic-based hallucination detection via score_triple + fuzzy_match
    calibration.py      # S^1 confidence calibration (ECE/MCE) via phase_encoder

    # -- Multi-Source & Multi-Agent --
    multi_agent.py      # Inter-agent consistency via tension_energy + check_holonomy
    cross_document.py   # Cross-doc conflict detection via inter-store tension_energy
    disambiguation.py   # Entity disambiguation via fuzzy_match_entity + score_triple

    # -- Robustness & Invariance --
    invariance.py       # Paraphrase invariance via pairwise tension_energy
    streaming.py        # Streaming chunk consistency via score_triple + holonomy

    # -- Observability --
    live_dashboard.py   # WebSocket dashboard polling tracing store triples
```

### 1.3 Configuration Extensions

GMSH adds the following fields to `DOEHarnessConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| Field | Type | Default | Phase | Description |
|-------|------|---------|-------|-------------|
| `enable_typed_verdicts` | `bool` | `False` | 1 | Claim-level verification |
| `enable_provenance` | `bool` | `False` | 2 | Document provenance chain |
| `snapshot_dir` | `str` | `""` | 4 | GMS snapshot directory (empty = off) |
| `semantic_mode` | `str` | `"off"` | 3 | Semantic verifier: `"off"`, `"fallback"`, `"complement"` |
| `semantic_model` | `str` | `""` | 3 | LLM judge model (empty = use `target_model`) |
| `enable_gateway` | `bool` | `False` | 6/WS1 | Runtime tool interception |
| `gateway_fault_profiles_path` | `str` | `""` | 6/WS1 | YAML path for FaultProfile list |
| `gateway_schema_strict` | `bool` | `True` | 6/WS1 | Enforce schema on tool arguments |
| `enable_policy_engine` | `bool` | `False` | 6/WS2 | Pre-execution policy checking |
| `policy_rules_path` | `str` | `""` | 6/WS2 | YAML path for PolicyRule list |
| `scenarios_dir` | `str` | `""` | 6/WS3 | Directory for scenario YAML files |
| `enable_scenario_registry` | `bool` | `False` | 6/WS3 | Scenario registry CRUD |
| `runs_dir` | `str` | `""` | 6/WS4 | Directory for run records (empty = off) |
| `regression_suite_dir` | `str` | `""` | 6/WS5 | Regression suite directory |
| `replay_manifest_path` | `str` | `""` | 6/WS5 | Path to replay manifest YAML |
| `enable_process_evaluation` | `bool` | `False` | 6/WS6 | Process compliance evaluation |
| `enable_recovery_evaluation` | `bool` | `False` | 6/WS6 | Recovery quality evaluation |
| `enable_interaction_evaluation` | `bool` | `False` | 6/WS6 | Human interaction evaluation |
| `risk_tier_profiles_path` | `str` | `""` | 6/WS7 | YAML path for RiskTierProfile list |
| `baseline_run_id` | `str` | `""` | 6/WS7 | Baseline run for regression comparison |
| `target_profile_path` | `str` | `""` | BB-1 | YAML path for TargetProfile |
| `environment_yaml_path` | `str` | `""` | BB-1 | YAML path for EnvironmentSimulator |
| `campaigns_dir` | `str` | `""` | BB-1 | Directory for campaign configs/results |
| `enable_severity_classification` | `bool` | `False` | BB-1 | GMS-enhanced severity classification |
| `safety_config_path` | `str` | `""` | BB-2 | YAML path for SafetyConfig |
| `enable_performance_profiling` | `bool` | `False` | 7 | S^1 latency/SLA profiling |
| `sla_latency_ms` | `float` | `5000.0` | 7 | SLA target in milliseconds |
| `enable_cost_tracking` | `bool` | `False` | 7 | S^1 token/cost tracking |
| `cost_budget_usd` | `float` | `10.0` | 7 | Budget cap in USD |
| `enable_stateful_testing` | `bool` | `False` | 7 | Agent memory testing |
| `stateful_session_count` | `int` | `5` | 7 | Teach-test session count |
| `enable_hallucination_testing` | `bool` | `False` | 7 | Geodesic hallucination detection |
| `hallucination_geodesic_threshold` | `float` | `1.0` | 7 | Geodesic above = hallucinated |
| `enable_calibration_testing` | `bool` | `False` | 7 | Confidence calibration analysis |
| `enable_multi_agent` | `bool` | `False` | 7 | Multi-agent consistency testing |
| `multi_agent_roles_path` | `str` | `""` | 7 | YAML path for AgentRole definitions |
| `enable_cross_document` | `bool` | `False` | 7 | Cross-document conflict testing |
| `additional_store_paths` | `list[str]` | `[]` | 7 | Paths to additional GMS stores |
| `enable_disambiguation_testing` | `bool` | `False` | 7 | Entity disambiguation testing |
| `enable_invariance_testing` | `bool` | `False` | 7 | Paraphrase invariance testing |
| `enable_streaming_testing` | `bool` | `False` | 7 | Streaming output consistency |
| `enable_live_dashboard` | `bool` | `False` | 7 | WebSocket live monitoring |
| `live_dashboard_port` | `int` | `8765` | 7 | Dashboard WebSocket port |

CLI flags:

```
# Phases 1-5
--typed-verdicts          Enable GMSH typed claim verification
--provenance              Enable document provenance tracking
--snapshot-dir DIR        Directory for GMS store snapshots
--audit-report PATH       Generate GMSH audit report
--release-gate K=V ...    Release gate thresholds: accuracy=0.8 geo=0.5
--semantic-mode MODE      off | fallback | complement
--semantic-model MODEL    Model for semantic judge

# Phase 6: Agentic Harness (WS7)
--risk-tiers PATH         YAML path for risk tier profiles
--baseline-run ID         Baseline run ID for regression comparison
```

CLI subcommands (Phase 6):

```
python -m harness.doe run report.md           # standard benchmark run
python -m harness.doe runs list               # list all tracked runs
python -m harness.doe runs show <run_id>      # show run details
python -m harness.doe replay <manifest.yaml>  # deterministic replay
python -m harness.doe regression promote <id> # promote failures to scenarios
python -m harness.doe regression list         # list regression scenarios
python -m harness.doe compare <a.json> <b.json> # compare two runs
```

---

## 2. Phase 1: Claim Compiler + Typed Verifiers

### 2.1 Architecture

The core innovation. SUT (System Under Test) output is decomposed into typed claims, each routed to a GMS-specific verifier:

```
LLM Answer
    |
    v
[ClaimCompiler] ──── metadata-guided + text-match + numeric-match
    |
    v
List[ClaimBase]
    |
    v
[VerificationRouter]
    |
    ├── FactClaim      → GraphVerifier   → score_triple()
    ├── ThresholdClaim → NumericVerifier  → check_threshold()
    ├── ArtifactClaim  → ArtifactVerifier → enm.read_exact() + SHA-256
    ├── RuleClaim      → LogicVerifier    → check_holonomy() / tension_energy()
    ├── ToolUseClaim   → PolicyVerifier   → score_triple() on tool relations
    └── SemanticClaim  → SemanticVerifier → LLM judge + GMS evidence
    |
    v
List[SubVerdict]
    |
    v
[TypedVerdict.from_sub_verdicts()]
    |
    v
TypedVerdict (backwards-compatible with GeometricVerdict)
```

### 2.2 Claim Types (`claims.py`)

All claims inherit from `ClaimBase`:

```python
@dataclass
class ClaimBase:
    claim_id: str = ""          # auto-generated UUID hex[:12]
    source_run_id: str = ""
    source_question_id: str = ""
    raw_text: str = ""          # answer fragment (truncated to 500 chars)
    confidence: float = 0.0     # extraction confidence
    provenance: ClaimProvenance | None = None
    claim_type: str = "base"
```

Each claim type maps to a specific GMS verification route:

| Claim Type | Fields | Verifier | GMS Method |
|---|---|---|---|
| `FactClaim` | `head`, `relation`, `tail` | `GraphVerifier` | `score_triple()`, `query_triples()` |
| `ThresholdClaim` | `entity`, `metric`, `value`, `limit`, `operator` | `NumericVerifier` | `lookup_enm()`, `check_threshold()` |
| `ArtifactClaim` | `category`, `entity_id`, `claimed_value` | `ArtifactVerifier` | `enm.read_exact()`, `enm.verify_integrity()` |
| `RuleClaim` | `antecedent_entities`, `antecedent_relations`, `consequent_entity`, `direct_relation`, `rule_type` | `LogicVerifier` | `check_holonomy()`, `tension_energy()` |
| `ToolUseClaim` | `tool_name`, `tool_args`, `result_key`, `claimed_value` | `PolicyVerifier` | `score_triple()`, `tension_energy()` |
| `SemanticClaim` | `ground_truth`, `question_text`, `context_section`, `semantic_type` | `SemanticVerifier` | LLM judge + GMS evidence |

`ClaimProvenance` links each claim to its source:

```python
@dataclass
class ClaimProvenance:
    doc_id: str = ""
    section_tag: str = ""
    span_start: int = -1
    span_end: int = -1
    extraction_method: str = ""  # "metadata", "text_match", "numeric_match"
```

### 2.3 Claim Compiler (`claims.py`)

The `ClaimCompiler` decomposes SUT output into typed claims via three extraction strategies in priority order:

**Strategy 1: Metadata-guided extraction** (confidence = 1.0)

Uses question metadata to directly construct the correct claim type:

| Metadata Keys | Emitted Claim |
|---|---|
| `head`, `relation`, `tail` | `FactClaim` |
| `mid`, `r1`, `r2` | `FactClaim` per hop + `RuleClaim` for path consistency |
| `encoder`/`category`, `operator` | `ThresholdClaim` |
| `enm_type`, `entity_id` | `ArtifactClaim` |
| `entity_a`, `entity_b`, `tension_energy` | `RuleClaim` (contradiction if E > 1.5) |
| `features` (list) | `RuleClaim` per feature pair |
| `relation` + `expected_count` | `FactClaim` via sample triple from graph |
| `relation_1`, `relation_2` | `FactClaim` per relation |
| `mid1`, `mid2`, `r1`, `r2`, `r3` | `FactClaim` per hop + `RuleClaim` for path |
| `entities` + `relations` (lists) | `FactClaim` per consecutive pair + `RuleClaim` |
| `test_type=non_existent_entity` | `FactClaim` with fake entity (negative test) |
| `test_type=wrong_relation` | `FactClaim` with wrong relation (negative test) |

**Strategy 2: Text-match extraction** (confidence = 0.6)

Scans the answer text for entities known to the GMS store:

1. Case-insensitive substring match against all store entities (length > 2)
2. Skip entities already captured via metadata
3. For each entity pair found, check `doc_graph.triples` for a known relation
4. Emit `FactClaim` for each matched `(head, relation, tail)` triple

**Strategy 3: Numeric-match extraction** (confidence = 0.5)

Parses numbers from the answer text and matches against ENM entries:

1. Extract all numbers via regex: `(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)`
2. For each ENM key, read the stored value
3. If `|parsed - stored| < 1e-6 * (1 + |stored|)`, emit an `ArtifactClaim`

**Strategy 4: Semantic claim emission**

Controlled by `semantic_mode`:

| Mode | Behavior |
|---|---|
| `"off"` | No semantic claims emitted |
| `"fallback"` | Emit `SemanticClaim` only when no structural claims were extracted |
| `"complement"` | Always emit `SemanticClaim` alongside structural claims |

The semantic type is `"completeness"` when no structural claims exist, `"equivalence"` otherwise.

**Strategy 5: Provenance enrichment** (Phase 2 integration)

If a `TruthStore` is wired into the compiler, each claim without provenance is matched against the truth store via `get_truth_for_claim()`, and the matching truth's `doc_id` and `section_tag` are attached.

### 2.4 Typed Verifiers (`verifiers.py`)

Each verifier is a thin wrapper around GMS methods that produces a structured `SubVerdict`:

```python
@dataclass
class SubVerdict:
    verifier_name: str = ""
    claim_id: str = ""
    passed: bool = False
    confidence: float = 0.0       # [0, 1]
    detail: str = ""
    gms_method_used: str = ""     # e.g. "score_triple", "tension_energy"
    raw_score: float | None = None
    failure: FailureAttribution | None = None
```

#### 2.4.1 GraphVerifier

Verifies `FactClaim` triples against the GMS knowledge graph.

**Verification cascade:**

```
1. fuzzy_match_entity(head), fuzzy_match_entity(tail)
   └── Entity not found → FAIL (ENTITY_NOT_FOUND)

2. score_triple(head_match, relation, tail_match)
   └── Returns geodesic distance d
       ├── d < 1.2 → PASS (confidence = exp(-d))
       └── d ≥ 1.2 → FAIL (FACT_INCORRECT)

3. [Fallback if score_triple returns None — relation not trained]
   query_triples(head, relation, tail) — exact graph match
   └── Found → PASS (confidence = 0.9)

4. [Fallback] query_triples(head, *, tail) — any relation
   └── Found → FAIL (RELATION_NOT_FOUND, actual relations reported)
   └── Not found → FAIL (FACT_INCORRECT)
```

**Plausibility threshold:** geodesic distance < 1.2.

#### 2.4.2 NumericVerifier

Verifies `ThresholdClaim` via GMS phase encoders and ENM.

**Verification cascade:**

```
1. Resolve value from ENM if not in claim:
   value = store.lookup_enm(metric, entity)
   └── Not found → FAIL (EXACT_MISMATCH)

2. If limit and operator present:
   doc_graph.check_threshold(metric, value, limit, operator)
   └── None (no phase encoder) → FAIL (VERIFICATION_ERROR, severity=WARNING)
   └── satisfied=True → PASS (confidence = 1.0)
   └── satisfied=False → FAIL (THRESHOLD_VIOLATION or BOUNDARY_FAILURE)

3. If no threshold to check:
   query_triples(entity, metric) — verify entity/metric exists
   └── PASS (confidence = 1.0 if in graph, 0.7 otherwise)
```

Boundary failures are distinguished when `|margin| < |limit| * 0.01`.

#### 2.4.3 ArtifactVerifier

Verifies `ArtifactClaim` via ENM SHA-256 hash comparison — deterministic, zero-tolerance.

**Verification cascade:**

```
1. Find ENM key matching (category, entity_id)
   └── Not found → FAIL (EXACT_MISMATCH)

2. enm.verify_integrity(key) — recompute SHA-256
   └── Failed → FAIL (ARTIFACT_INTEGRITY_FAILURE)

3. enm.read_exact(key) — retrieve stored value
   └── Error → FAIL (ARTIFACT_INTEGRITY_FAILURE)

4. If claimed_value is None:
   └── PASS (existence + integrity verified)

5. SHA-256 comparison:
   claimed_bytes = np.float64(claimed_value).tobytes()
   claimed_hash = sha256(claimed_bytes)
   └── Match → PASS (confidence = 1.0)
   └── Mismatch → FAIL (EXACT_MISMATCH, reports both hashes)
```

The comparison is hash-based, not tolerance-based. IEEE 754 guarantees identical byte representation for identical `float64` values.

#### 2.4.4 LogicVerifier

Verifies `RuleClaim` path consistency and contradictions via GMS geometry.

**For consistency claims** (`rule_type="consistency"`):

```
1. check_holonomy(path_relations, direct_relation)
   └── defect ≤ tau_path → PASS (confidence = exp(-defect))
   └── defect > tau_path → FAIL (HOLONOMY_DEFECT)

2. [Fallback if holonomy returns None]
   query_triples() per hop — verify each step exists
   └── All hops verified → PASS (confidence = 0.7)

3. [Fallback] tension_energy(first_entity, last_entity)
   └── See contradiction check below
```

**For contradiction claims** (`rule_type="contradiction"`):

```
1. fuzzy_match_entity(a), fuzzy_match_entity(b)
   └── Not found → check graph; if pair in graph → PASS (0.5)
   └── Not found at all → FAIL (ENTITY_NOT_FOUND)

2. tension_energy(a_match, b_match)
   └── E > tau_contra → contradictory
   └── E < tau_ent → entailment
   └── Otherwise → neutral

   For rule_type="contradiction": passed = is_contradictory
   For rule_type="consistency": passed = NOT is_contradictory
```

**Tension energy interpretation:**

| Range | Semantic | $\theta$ |
|---|---|---|
| $E \approx 0$ | Agreement | $\theta \approx 0$ |
| $E \approx \sqrt{2}$ | Irrelevant | $\theta \approx \pi/2$ |
| $E \approx 2$ | Contradiction | $\theta \approx \pi$ |

**Confidence:** $c = 1 - E/2$, clamped to $[0, 1]$.

#### 2.4.5 PolicyVerifier

Verifies `ToolUseClaim` from agent tool calls via GMS geometric plausibility.

**Verification cascade:**

```
1. fuzzy_match_entity(tool_name), fuzzy_match_entity(result_key)

   [Neither found]
   └── query_triples(tool_name) or query_triples(tail=result_key)
       ├── Found → PASS (confidence = 0.5)
       └── Not found → FAIL (POLICY_VIOLATION)

   [Both found]
   └── Route A: score_triple(tool, "produces", result)
       ├── d < 1.5 → PASS (confidence = exp(-d))
       └── d ≥ 1.5 → FAIL (TOOL_CALL_INVALID)
   └── Route B: query_triples(tool, *, result) — any relation
       └── Found → PASS (confidence = 0.8)
   └── Route C: tension_energy(tool, result)
       ├── E < 1.5 → PASS (compatible)
       └── E ≥ 1.5 → FAIL (TOOL_CALL_INVALID)
   └── All routes exhausted → FAIL (POLICY_VIOLATION)

   [One found]
   └── query_triples(unknown) — check graph
       ├── Found → PASS (confidence = 0.5)
       └── Not found → FAIL (POLICY_VIOLATION)
```

**Plausibility threshold:** geodesic distance < 1.5.

#### 2.4.6 VerificationRouter

Dispatches claims to the correct verifier:

```python
dispatch = {
    "fact":        GraphVerifier.verify,
    "threshold":   NumericVerifier.verify,
    "artifact":    ArtifactVerifier.verify,
    "rule":        LogicVerifier.verify,
    "tool_use":    PolicyVerifier.verify,
    "semantic":    SemanticVerifier.verify,    # if configured
    "process":     ProcessEvaluator.verify,    # WS6: tool call ordering
    "recovery":    RecoveryEvaluator.verify,   # WS6: fault recovery
    "interaction": HumanInteractionEvaluator.verify,  # WS6: escalation/clarification
}
```

The three WS6 evaluators (`process`, `recovery`, `interaction`) are optional — they are included in the dispatch table only when their corresponding config flags are enabled.

Unrecognized claim types produce a `VERIFICATION_ERROR` sub-verdict. Exceptions during verification are caught and produce a `VERIFICATION_ERROR` failure.

### 2.5 Failure Taxonomy (`failures.py`)

27 failure codes organized by GMS verification route:

| Category | Code | GMS Route |
|---|---|---|
| **Content** | `SEMANTIC_EXTRACTION_FAILURE` | claim extraction |
| | `SEMANTIC_MISMATCH` | LLM judge scores |
| | `HALLUCINATION_DETECTED` | groundedness rubric |
| | `INCOMPLETE_ANSWER` | completeness rubric |
| | `NO_CLAIMS_EXTRACTED` | compiler output |
| **Graph** | `ENTITY_NOT_FOUND` | `fuzzy_match_entity()` |
| | `RELATION_NOT_FOUND` | `query_triples()` |
| | `FACT_INCORRECT` | `score_triple()` |
| | `PATH_INCONSISTENCY` | `query_triples()` per hop |
| **Logic** | `CONTRADICTION_DETECTED` | `tension_energy()` |
| | `TENSION_EXCEEDED` | `tension_energy()` |
| | `HOLONOMY_DEFECT` | `check_holonomy()` |
| **Numeric** | `THRESHOLD_VIOLATION` | `check_threshold()` |
| | `BOUNDARY_FAILURE` | `check_threshold()` margin |
| **Artifact** | `EXACT_MISMATCH` | `enm.read_exact()` |
| | `ARTIFACT_INTEGRITY_FAILURE` | `enm.verify_integrity()` |
| **Policy** | `TOOL_CALL_INVALID` | `score_triple()` on tools |
| | `POLICY_VIOLATION` | `fuzzy_match_entity()` on tools |
| | `POLICY_BLOCKED_PRE_EXECUTION` | WS2: `PolicyEngine.check()` |
| | `APPROVAL_REQUIRED` | WS2: `PolicyEngine.check()` |
| **Process** | `REQUIRED_TOOL_MISSING` | WS6: set difference check |
| | `FORBIDDEN_TOOL_CALLED` | WS6: set intersection check |
| | `PROCESS_DEVIATION` | WS6: `check_holonomy()` on tool paths |
| | `TOOL_BUDGET_EXCEEDED` | WS6: budget count check |
| **Recovery** | `RECOVERY_FAILURE` | WS6: `tension_energy()` pre/post-fault |
| **Interaction** | `ESCALATION_INAPPROPRIATE` | WS6: `score_triple()` + `query_triples()` |
| | `CLARIFICATION_INSUFFICIENT` | WS6: `score_triple()` |
| **System** | `VERIFICATION_ERROR` | any GMS method failure |

Severity levels: `ERROR`, `WARNING`, `INFO`.

```python
@dataclass
class FailureAttribution:
    code: FailureCode
    claim_id: str
    verifier: str                    # e.g. "GraphVerifier"
    severity: FailureSeverity = ERROR
    detail: str = ""
    gms_method: str = ""             # e.g. "score_triple"
    raw_score: float | None = None
    suggested_action: str = ""
```

### 2.6 TypedVerdict (`verdict.py`)

Aggregates claims and sub-verdicts into a backwards-compatible verdict:

```python
@dataclass
class TypedVerdict:
    # Backwards-compatible with GeometricVerdict
    correct: bool
    detail: str
    geodesic_confidence: float   # aggregated from GraphVerifier scores
    tension_score: float         # aggregated from LogicVerifier tension
    holonomy_score: float        # aggregated from LogicVerifier holonomy
    overall_confidence: float    # same weighted formula as GMSJudge

    # GMSH audit trail
    claims: list[ClaimBase]
    sub_verdicts: list[SubVerdict]
    failures: list[FailureAttribution]
    run_id: str
    question_id: str
    n_claims: int
    n_passed: int
    n_failed: int
```

**Aggregation** (`from_sub_verdicts`):

1. Collect raw scores by verifier type:
   - `GraphVerifier` scores → geodesic distances
   - `LogicVerifier` with `tension_energy` → tension scores
   - `LogicVerifier` with `check_holonomy` → holonomy scores
2. Compute backwards-compatible fields:
   - `geodesic_confidence = exp(-mean(graph_scores))`
   - `tension_score = mean(tension_scores)`
   - `holonomy_score = mean(holonomy_scores)`
3. Compute `overall_confidence` using the same weighted formula as `GMSJudge`:

$$s = w_e \cdot c + w_g \cdot e^{-\bar{d}} + w_t \cdot \max(1 - \bar{E}/2,\; 0) + w_h \cdot e^{-\bar{\delta}}$$

Default weights: `exact=0.3, geodesic=0.3, tension=0.2, holonomy=0.2`.

**Downcast:** `to_geometric_verdict()` returns a `GeometricVerdict` with the same top-level fields.

---

## 3. Semantic Verifier (`semantic.py`)

### 3.1 Purpose

Handles verification that structural verifiers cannot: semantic equivalence, explanation quality, completeness, and groundedness. Every judgment is **anchored by GMS evidence** — the LLM judge never operates in isolation.

### 3.2 Three-Step Pipeline

```
SemanticClaim
    |
    v
[Step 1: Gather GMS Evidence]
    fuzzy_match_entity() → resolve answer/truth entities
    query_triples()      → gather related facts
    score_triple()       → geodesic proximity between pairs
    tension_energy()     → contradiction signals
    |
    v
GMSEvidence bundle
    |
    v
[Step 2: Call LLM Judge]
    Structured rubric prompt + GMS evidence injection
    |
    v
SemanticScore (4 dimensions, 1-5 scale)
    |
    v
[Step 3: Calibrate Confidence with GMS]
    entity_coverage × proximity_factor × tension_penalty
    |
    v
SubVerdict (GMS-calibrated)
```

### 3.3 GMS Evidence Bundle

```python
@dataclass
class GMSEvidence:
    answer_entities: dict[str, str]    # raw_text → matched GMS entity
    truth_entities: dict[str, str]     # raw_text → matched GMS entity
    entity_triples: dict[str, list]    # entity → related (h, r, t) triples
    geodesic_scores: dict[str, float]  # "a↔b" → geodesic distance
    tension_scores: dict[str, float]   # "a↔b" → tension energy
    entity_coverage: float             # fraction of answer entities in GMS
```

Evidence gathering uses four GMS methods:

| Method | Purpose |
|---|---|
| `fuzzy_match_entity()` | Resolve raw text mentions to canonical GMS entities |
| `query_triples(head/tail)` | Gather related facts for context (up to 10 per entity) |
| `score_triple(a, "related_to", b)` | Geodesic proximity between answer and truth entities |
| `tension_energy(a, b)` | Contradiction signal between answer and truth entities |

### 3.4 Rubric

The LLM judge evaluates four dimensions:

| Dimension | Weight | Focus |
|---|---|---|
| `factual_accuracy` | 2.0 | Correctness against GMS evidence |
| `completeness` | 1.5 | Coverage of relevant GMS entities |
| `groundedness` | 2.0 | Grounding in source material, hallucination detection |
| `coherence` | 1.0 | Internal consistency (high tension → flag) |

Weighted score: $s_w = \sum_i w_i \cdot s_i \;/\; \sum_i w_i$, range $[1, 5]$.

Pass threshold: $s_w \geq 3.0$.

### 3.5 GMS Confidence Calibration

The LLM's weighted score is normalized to $[0, 1]$ and adjusted by three GMS signals:

$$c_{\text{calibrated}} = \frac{s_w}{5} \times f_{\text{coverage}} \times f_{\text{proximity}} \times f_{\text{tension}}$$

| Factor | Range | Formula |
|---|---|---|
| Coverage | $[0.8, 1.0]$ | $0.8 + 0.2 \times \text{entity\_coverage}$ |
| Proximity | $[0.9, 1.0]$ | $0.9 + 0.1 \times \min(1, \bar{d}_{\text{geo}})$ |
| Tension penalty | $[0.8, 1.0]$ | $0.8$ if max tension > 1.5; $0.9$ if > 1.0; else $1.0$ |

### 3.6 Failure Determination

When the semantic claim fails ($s_w < 3.0$), the worst rubric dimension maps to a failure code:

| Worst Dimension | Score ≤ 2 | Failure Code |
|---|---|---|
| `factual_accuracy` | Yes | `SEMANTIC_MISMATCH` |
| `groundedness` | Yes | `HALLUCINATION_DETECTED` |
| `completeness` | Yes | `INCOMPLETE_ANSWER` |
| `coherence` | Yes | `CONTRADICTION_DETECTED` |
| Any | No | `SEMANTIC_MISMATCH` (below threshold) |

---

## 4. Phase 2: Provenance Chain + Truth Objects

### 4.1 Provenance Registry (`provenance.py`)

Tracks the full lineage: source document → sections → triples → truths → claims → verdicts.

**Document registration:**

```python
registry.register_document(doc_path, markdown)
```

1. Compute SHA-256 content hash → `doc_id` (content-addressable)
2. Extract section spans from markdown headings (H1-H3)
3. Create `DocumentRecord` and `NormalizedDocument` with character offsets
4. Record in tracing GMS as triples:
   - `(doc_hash[:16], is_type, source_document)`
   - `(doc_hash[:16], has_section, section_tag)` per section

**Triple-to-source linking:**

```python
registry.link_triple_to_source(head, relation, tail, doc_id, section_tag)
```

Records `(triple_hash[:12], sourced_from, doc_hash[:16])` and `(triple_hash[:12], in_section, section_tag)` in the tracing store.

### 4.2 Truth Store (`truth.py`)

Registry of typed truths extracted from a populated GMS store. Each truth links to a specific GMS primitive.

**Truth types:**

| Type | Source | GMS Link |
|---|---|---|
| `EntityTruth` | `adapter.entity_to_idx` | embedding vector index |
| `RelationTruth` | `doc_graph.triples` | `score_triple()` at ingest time |
| `ThresholdTruth` | `doc_graph.phase_encoders` + ENM | phase angle $\phi(v)$ |
| `ArtifactTruth` | ENM entries | SHA-256 hash of stored value |

**Build process** (`build_from_store`):

1. Scan `adapter.entity_to_idx` → create `EntityTruth` per entity
2. Scan `doc_graph.triples` → create `RelationTruth` per triple (with live `score_triple()` score)
3. Scan `enm.keys()` → create `ArtifactTruth` per entry (with SHA-256 hash)
4. Scan `doc_graph.phase_encoders` → create `ThresholdTruth` per encoder/entity pair

**Claim matching** (`get_truth_for_claim`):

| Claim Type | Lookup Strategy |
|---|---|
| `FactClaim` | `get_relation_truth(head, relation, tail)` |
| `ArtifactClaim` | `get_artifact_truth(category, entity_id)` |
| `ThresholdClaim` | `get_artifact_truth(metric, entity)` then `by_encoder` index |
| `RuleClaim` | `get_entity_truths(first_entity)` |

**Persistence:** JSON serialization with type-discriminated loading.

---

## 5. Phase 3: Extended SUT Adapters + Test Generators

### 5.1 SUT Adapters (`adapters.py`)

Structured interface to systems under test, producing `SUTTrace` with execution metadata:

```python
@dataclass
class SUTTrace:
    raw_output: str = ""
    tool_calls: list[ToolCallRecord] = []
    latency_ms: float = 0.0
    token_count: int = 0
    retrieved_context: str = ""  # for RAG adapters
    error: str = ""
```

Three adapter types:

| Adapter | Target | Instrumentation |
|---|---|---|
| `LLMAdapter` | Direct Claude API | Latency tracking |
| `AgentAdapter` | Harness agent loop | Tool call capture via hook interception |
| `RAGAdapter` | Retriever + Generator | Retrieved context capture |

**AgentAdapter tool instrumentation:**

Hooks into `harness.hooks.on_tool_call` and `harness.hooks.on_tool_result` to capture every tool call as a `ToolCallRecord(tool_name, arguments, result, latency_ms)`. Original hooks are saved and restored after evaluation.

### 5.2 Extended Test Generators (`test_generators.py`)

Three new GMS-native generator categories:

#### 5.2.1 ExactRetrievalGenerator

**Category:** `"exact_retrieval"`

Tests exact numeric retrieval from ENM. For each ENM entry: "What is the exact `{type}` value for `{entity_id}`?"

- Ground truth: `store.lookup_enm(category, entity_id)`
- Answer type: `float`
- Metadata: `enm_type`, `entity_id`, `value`

#### 5.2.2 NegativeGenerator

**Category:** `"negative"`

Adversarial tests with fabricated entities. Two subtypes:

| Subtype | Strategy | Ground Truth |
|---|---|---|
| Non-existent entity | Prefix real entity with `Nonexistent_`/`Fabricated_`/`Phantom_`/`Unknown_` | `False` |
| Wrong relation | Use a known `(h, t)` pair with a relation that doesn't exist between them | `False` |

Validates that wrong triples are not in the store before emitting questions. Up to 15 questions per subtype.

#### 5.2.3 WorkflowComplianceGenerator

**Category:** `"workflow_compliance"`

Tests multi-hop ordering from 3+ hop graph chains. Builds adjacency map, finds chains `a →r1→ b →r2→ c →r3→ d`, and generates:

| Question Type | Example | Answer |
|---|---|---|
| Next step | "After reaching B via r1, what comes next via r2?" | Entity C |
| Ordering | "Is A → B → C → D consistent?" | `True` |

Up to 30 questions. Metadata includes full path entities and relations.

**Registration:**

```python
def extended_generators() -> list[QuestionGenerator]:
    return [ExactRetrievalGenerator(), NegativeGenerator(), WorkflowComplianceGenerator()]
```

These are automatically included by `bridge.generate_base_questions()` when no specific generators are requested.

---

## 6. Phase 4: Versioning + Snapshots (`snapshots.py`)

### 6.1 GMSSnapshot

Immutable record of a GMS store at a point in time:

```python
@dataclass
class GMSSnapshot:
    snapshot_id: str     # "snap_{uuid_hex[:12]}"
    run_id: str          # benchmark run that triggered this
    timestamp: str
    store_path: str      # original store path
    snapshot_path: str   # path to the snapshot copy
    parent_snapshot_id: str
    trigger: str         # "ingest", "pre_run", "manual"
    metadata: dict       # store stats + content_hash
```

### 6.2 SnapshotManager

**Snapshot lifecycle:**

```
ingest() ──[trigger=ingest]──> Snapshot A
    |
run()   ──[trigger=pre_run]──> Snapshot B (parent=A)
    |
manual  ──[trigger=manual]───> Snapshot C (parent=B)
```

**Operations:**

| Method | Description |
|---|---|
| `take_snapshot(store, run_id, trigger)` | Save store, copy directory, hash contents, record in manifest |
| `restore_snapshot(id, device)` | Load a full `GMSExpertStore` from a snapshot |
| `diff_snapshots(id_a, id_b)` | Compare entities, triples, and ENM keys between two snapshots |
| `list_snapshots()` | All snapshots, oldest first |
| `get_latest()` | Most recent snapshot |

**Content integrity:** each snapshot directory is SHA-256 hashed (over all file bytes) and stored in `metadata["content_hash"]`.

**Manifest:** `manifest.json` in the snapshot directory indexes all snapshots as a JSON array.

**Diff output:**

```python
{
    "entities_added": [...],    "entities_removed": [...],
    "triples_added": [...],     "triples_removed": [...],
    "enm_added": [...],         "enm_removed": [...],
    "stats_a": {...},           "stats_b": {...},
}
```

---

## 7. Phase 5: Audit Reporting + Release Gates (`audit.py`)

### 7.1 AuditPacket

Per-question audit trail bundling all information into one reviewable unit:

```python
@dataclass
class AuditPacket:
    doe_qid: str
    category: str
    question: str
    ground_truth: str
    sut_answer: str
    correct: bool
    detail: str
    # Geometric scores
    geodesic_confidence: float
    tension_score: float
    holonomy_score: float
    overall_confidence: float
    # GMSH typed verdict detail
    n_claims: int
    n_passed: int
    n_failed: int
    claim_types: list[str]
    failure_codes: list[str]
    verifier_summary: dict       # per-verifier pass/fail counts
    factor_assignment: dict
```

### 7.2 ReleaseGate

Configurable pass/fail determination:

```python
@dataclass
class ReleaseGate:
    passed: bool
    accuracy_threshold: float
    actual_accuracy: float
    geo_confidence_threshold: float
    actual_geo_confidence: float
    failure_budget: dict[str, int]    # max allowed per failure code
    actual_failures: dict[str, int]
    blocking_failures: list[str]      # reasons gate was blocked
```

**Gate checks:**

1. `actual_accuracy >= accuracy_threshold`
2. `mean(overall_confidence) >= geo_confidence_threshold`
3. For each failure code in budget: `actual_count <= max_count`

Any failed check appends to `blocking_failures`. Gate passes only when `blocking_failures` is empty.

### 7.3 AuditReporter

Generates structured reports from benchmark results + typed verdicts:

**Failure analysis** (`generate_failure_report`):

| Report Section | Content |
|---|---|
| `by_code` | Failure count per `FailureCode` |
| `by_category` | Failure codes broken down by question category |
| `by_severity` | Count by `ERROR`/`WARNING`/`INFO` |
| `by_verifier` | Count by verifier name |
| `heatmap` | `failure_code × question_category` matrix |
| `systematic_patterns` | Categories with ≥80% failure rate + top codes |

**Executive summary** (`generate_executive_summary`):

Text report suitable for CLI or inclusion in documents. Contains:
- Instance name, model, timestamp, question count, accuracy
- Mean geometric confidence, tension, holonomy
- Total claims (passed/failed)
- Top 5 failure codes
- Systematic failure patterns
- Per-category breakdown
- Recommendation: PASS / CONDITIONAL / FAIL

**Recommendation logic:**

| Accuracy | Recommendation |
|---|---|
| ≥ 90% | PASS |
| ≥ 70% | CONDITIONAL |
| < 70% | FAIL |

Systematic patterns (categories with ≥80% failure rate) are always flagged regardless of overall accuracy.

### 7.4 Enhanced Report (`report_enhanced.py`)

Generates HTML sections appended to the base dashboard:

- Failure attribution heatmap (`FailureCode × QuestionCategory`)
- Claim type coverage table
- Sub-verdict detail table
- Release gate summary card
- Provenance chain view
- Risk-tiered release gate cards (WS7)

---

## 8. Phase 6: Agentic Testing Harness

Phase 6 extends GMSH from a content-verification platform into a **black-box agentic AI testing harness**. It adds seven workstreams (WS1–WS7) that address runtime tool interception, policy enforcement, data-driven scenarios, formal run lifecycle, replay/regression, process/recovery evaluation, and risk-tiered release gates.

All workstreams follow the same design principles as Phases 1–5:

- **GMS geometric primitives** are the primary verification signal
- Every decision is **traced as GMS triples** via `DOETracingGMS`
- All features are **opt-in** via `DOEHarnessConfig` flags (default off)
- All new code is **backwards-compatible** with base DOE-GMS

### 8.1 WS1: Tool Gateway (`gateway.py`)

Runtime interception layer that hooks into `HarnessHooks.before_tool` and `HarnessHooks.on_tool_result` to intercept every tool call during agent execution.

#### 8.1.1 Data Models

```python
@dataclass
class FaultProfile:
    tool_pattern: str = "*"          # fnmatch glob (e.g. "modeva_*")
    latency_ms: float = 0.0         # artificial delay (ms)
    error_rate: float = 0.0         # probability of injected error [0,1]
    error_message: str = "Simulated tool failure"
    stale_data: bool = False         # cache tool results for audit
    mock_response: str | None = None # block and record intended mock
    schema_strict: bool = True       # reserved for schema validation

@dataclass
class GatewayDecision:
    tool_name: str = ""
    action: str = ""                 # "allow", "block", "mock", "inject_fault"
    reason: str = ""
    gms_score: float | None = None   # plausibility score if computed
    timestamp: str = ""

@dataclass
class GatewayTranscript:
    run_id: str = ""
    decisions: list[GatewayDecision] = []
```

#### 8.1.2 ToolGateway

```python
class ToolGateway:
    def __init__(store, fault_profiles, policy_engine=None, tracing=None, run_id="")
    def install(harness)      # attach as before_tool + on_tool_result hooks
    def uninstall(harness)    # restore original hooks
    def get_transcript()      # return full GatewayTranscript
```

**Interception pipeline** (`_before_tool`):

```
Tool call arrives
    |
    v
[1. PolicyEngine.check()] ──block──> GatewayDecision(action="block")
    |
    v (allowed)
[2. Match FaultProfile by fnmatch]
    |
    ├── mock_response set ──> GatewayDecision(action="mock"), block
    ├── error_rate hit     ──> GatewayDecision(action="inject_fault"), block
    └── latency_ms > 0     ──> sleep(latency_ms), continue
    |
    v (no fault match or fault not triggered)
[3. GMS plausibility: score_triple("agent", "has_capability", tool_name)]
    |
    v
GatewayDecision(action="allow", gms_score=score)
```

**GMS usage:**
- `store.score_triple("agent", "has_capability", tool_name)` — plausibility scoring for audit (informational, does not block)

**Limitations:**
- `HarnessHooks.before_tool` returns `bool` only — cannot inject arbitrary mock responses into the agent conversation. Mock profiles block the call and record the intended response in the transcript.
- `HarnessHooks.on_tool_result` is informational — cannot substitute stale data into the response.

**YAML loading:**

```yaml
fault_profiles:
  - tool_pattern: "modeva_*"
    latency_ms: 500
    error_rate: 0.1
  - tool_pattern: "gms_query"
    mock_response: "No results found."
```

---

### 8.2 WS2: Runtime Policy Engine (`policy.py`)

Pre-execution policy checking using GMS geometric verification. Policy rules are encoded as GMS triples in the store so that the same geometric primitives used for knowledge verification also enforce runtime constraints.

#### 8.2.1 Data Models

```python
@dataclass
class PolicyRule:
    rule_id: str = ""            # e.g. "RULE-001"
    action: str = ""             # tool name or fnmatch pattern
    requirement: str = ""        # "human_approval", "data_loaded", etc.
    rule_type: str = "requires"  # "requires", "blocks", "allows"
    context: str = "*"           # scenario context pattern
    severity: str = "error"      # "error" (hard block) or "warning" (log only)

@dataclass
class PolicyDecision:
    allowed: bool = True
    action: str = ""
    rules_checked: list[str] = []
    blocking_rules: list[str] = []
    gms_scores: dict[str, float] = {}
    reason: str = ""
```

#### 8.2.2 PolicyEngine

```python
class PolicyEngine:
    PLAUSIBILITY_THRESHOLD = 1.5    # max geodesic for "plausible"
    TENSION_THRESHOLD = 1.7         # tension above which contradiction detected

    def __init__(store, rules=None, tracing=None, run_id="")
    def load_rules_from_yaml(path)   # load and encode rules
    def encode_rules(rules)          # write rules as GMS triples via RuntimeLearner
    def check(action, args, context) # 4-layer verification → PolicyDecision
```

**Rule encoding** (`encode_rules`):

For each rule, the engine learns four triples via `RuntimeLearner`:

```
(rule.action, rule.rule_type, rule.requirement)   # primary rule triple
(rule.rule_id, "is_type", "policy_rule")           # metadata
(rule.rule_id, "applies_to", rule.action)
(rule.rule_id, "severity", rule.severity)
```

**4-layer check** (`check`):

| Layer | Method | Action |
|-------|--------|--------|
| 1. Explicit rule matching | `fnmatch` on action + context patterns | Block if `rule_type="blocks"` with `severity="error"` |
| 2. GMS graph lookup | `query_triples(head=action)` | Block if relation is `"blocks"` or `"requires"` |
| 3. Geometric plausibility | `score_triple(action, "allowed_in", context)` | Block if score > `PLAUSIBILITY_THRESHOLD` |
| 4. Contradiction detection | `tension_energy(action, blocking_rule)` | Supplementary signal — flags high-tension pairs |

Entity resolution uses `fuzzy_match_entity()` at every layer, falling back to raw names when entities are not yet in the graph.

**GMS usage:**
- `store.query_triples(head=action)` — explicit rule lookup from encoded policy triples
- `store.score_triple(action, "allowed_in", context)` — geometric plausibility
- `store.tension_energy(action, blocking_rule)` — contradiction detection
- `store.fuzzy_match_entity(name)` — entity resolution for all layers
- `RuntimeLearner.learn_triples()` — encode policy rules into GMS

**Integration with WS1:** `ToolGateway._before_tool()` calls `PolicyEngine.check()` as its first step. If `allowed=False`, the gateway immediately returns `False` (block).

---

### 8.3 WS3: Scenario Registry (`scenarios.py`)

Structured CRUD for authored, versioned test scenarios. Scenarios can be manually authored (with `question_text`) or delegate to a GMS generator (with `generator_name`).

#### 8.3.1 Scenario Data Model

```python
@dataclass
class Scenario:
    # Identity
    scenario_id: str = ""     # auto-generated as "scn_{uuid8}"
    name: str = ""
    description: str = ""
    version: int = 1
    tags: list[str] = []
    risk_tier: str = "standard"  # "advisory", "recommendation", "action_taking"

    # Question definition (one of these)
    generator_name: str = ""     # delegate to GMS generator (e.g. "plausibility")
    generator_params: dict = {}
    question_text: str = ""      # manually authored question
    ground_truth: str = ""
    answer_type: str = "str"     # "float", "int", "bool", "set_str", "str"
    metadata: dict = {}

    # Execution context
    context_mode: str = "full"
    fault_profile_refs: list[str] = []
    policy_rule_refs: list[str] = []

    # Behavioral requirements (used by WS6 extended evaluators)
    required_tools: list[str] = []
    forbidden_tools: list[str] = []
    expected_tool_sequence: list[str] = []
    max_tool_calls: int = 0
    expected_recovery: bool = False

    # Provenance
    created_at: str = ""
    author: str = ""
    source_run_id: str = ""       # if promoted from a failed run
    source_question_id: str = ""
```

**Failure promotion** (`Scenario.from_failed_run`):

Converts a failed `DOEQuestionResult` into a regression scenario, preserving the question text, ground truth, category tags, and failure provenance.

#### 8.3.2 ScenarioRegistry

```python
class ScenarioRegistry:
    def __init__(scenarios_dir, store=None, tracing=None)

    # CRUD
    def add(scenario) -> str        # save YAML, update index, encode GMS triples
    def get(scenario_id) -> Scenario | None
    def update(scenario)            # increment version, save new YAML
    def delete(scenario_id)         # remove from index (YAML kept for history)
    def list_all() -> list[Scenario]

    # Versioning
    def get_version(scenario_id, version) -> Scenario | None
    def list_versions(scenario_id) -> list[int]

    # Search
    def search_by_tag(tags) -> list[Scenario]
    def search_by_risk_tier(tier) -> list[Scenario]
    def search_geometric(query, top_k) -> list[tuple[Scenario, float]]
    def find_complementary(scenario_id, top_k) -> list[tuple[Scenario, float]]

    # DOE bridge
    def to_doe_questions(scenarios, store) -> list[GeneratedQuestion]
```

**Storage layout:**

```
scenarios_dir/
    index.json                      # scenario_id → {latest_version, name, tags, ...}
    scn_abc12345_v1.yaml           # version 1
    scn_abc12345_v2.yaml           # version 2 (previous kept for history)
```

**GMS triple encoding** (`_encode_scenario_triples`):

When a GMS store is available, scenario metadata is encoded as triples for geometric search:

```
(scenario_id, "is_type", "test_scenario")
(scenario_id, "tests_risk_tier", risk_tier)
(scenario_id, "has_tag", tag)              # per tag
(scenario_id, "has_name", name)
(scenario_id, "uses_generator", generator_name)
```

**Geometric search** (`search_geometric`):

Uses `fuzzy_match_entity()` to resolve the query, then `score_triple()` to rank scenarios by geodesic distance. Lower distance = more relevant.

**Complementary search** (`find_complementary`):

Uses `tension_energy()` between scenario entities. Tension of ~sqrt(2) indicates complementarity (neither agreeing nor contradicting), which is ideal for building diverse test suites.

**DOE bridge** (`to_doe_questions`):

- Authored scenarios (with `question_text`) → `GeneratedQuestion` with lambda `gms_answer_fn`
- Generator scenarios (with `generator_name`) → delegates to the matching GMS generator class

The `DOEGMSBenchmark.run_scenarios()` method converts scenarios to DOE questions and feeds them into the existing `run()` pipeline with DOE design matrix expansion.

---

### 8.4 WS4: Run State Machine (`runs.py`)

Formal lifecycle management for benchmark runs with validated state transitions and persistence.

#### 8.4.1 Run States

```python
class RunState(str, Enum):
    CREATED    = "created"
    PREPARING  = "preparing"
    EXECUTING  = "executing"
    EVALUATING = "evaluating"
    COMPLETED  = "completed"
    FAILED     = "failed"
    ABORTED    = "aborted"
```

**Valid transitions:**

```
CREATED → PREPARING → EXECUTING → EVALUATING → COMPLETED
    |         |           |            |
    v         v           v            v
  ABORTED   FAILED      FAILED      FAILED
```

Invalid transitions raise `ValueError`.

#### 8.4.2 RunRecord

```python
@dataclass
class RunRecord:
    run_id: str = ""                   # auto-generated "run_{uuid8}"
    state: RunState = RunState.CREATED
    config_snapshot: dict = {}         # frozen DOEHarnessConfig

    # Timing
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    # State history
    state_history: list[dict] = []     # [{timestamp, from_state, to_state}]

    # Results (populated during execution)
    benchmark_result: dict | None = None
    typed_verdicts: dict = {}
    gateway_transcript: dict | None = None

    # Error tracking
    error: str = ""
    error_state: str = ""

    # Links
    scenario_ids: list[str] = []
```

#### 8.4.3 RunManager

```python
class RunManager:
    def __init__(runs_dir, tracing=None)

    def create_run(config) -> RunRecord          # CREATED state
    def transition(run_id, to_state) -> RunRecord # validate + record + trace
    def get_run(run_id) -> RunRecord | None
    def list_runs(state=None) -> list[RunRecord]
    def get_latest() -> RunRecord | None
```

**State transitions are traced** as GMS triples via `trace_state_transition(run_id, from_state, to_state)`.

**Persistence:** Each run record is saved as `{runs_dir}/{run_id}.json`.

**Design note:** `RunManager` wraps existing `DOEGMSBenchmark` lifecycle calls — the benchmark class itself is not modified. The `execute_run()` pattern is: `CREATED → PREPARING → EXECUTING → EVALUATING → COMPLETED`, with exceptions caught and transitioned to `FAILED`.

---

### 8.5 WS5: Replay & Regression (`replay.py`)

Deterministic replay from frozen manifests, per-question run comparison, and failure-to-regression promotion.

#### 8.5.1 ReplayManifest

Captures all inputs needed to deterministically re-execute a benchmark run:

```python
@dataclass
class ReplayManifest:
    manifest_id: str = ""         # "manifest_{uuid12}"
    source_run_id: str = ""

    # Frozen inputs
    config_snapshot: dict = {}    # frozen DOEHarnessConfig
    snapshot_id: str = ""         # GMS store snapshot to restore
    doe_questions_path: str = ""  # serialised question list

    # Execution context
    fault_profiles: list[dict] = []
    policy_rules: list[dict] = []
    random_seed: int = 42

    # Reference
    reference_result_path: str = ""
    created_at: str = ""
```

Serialized to/from YAML via `to_yaml(path)` / `from_yaml(path)`.

**Creating manifests:** `SnapshotManager.create_replay_manifest()` creates a manifest from a snapshot and run metadata.

#### 8.5.2 RunComparison

Per-question diff between two benchmark results:

```python
@dataclass
class RunComparison:
    run_a_id: str = ""
    run_b_id: str = ""

    # Aggregate deltas
    accuracy_delta: float = 0.0
    geo_confidence_delta: float = 0.0
    tension_delta: float = 0.0

    # Per-question diffs
    regressions: list[dict] = []      # correct → wrong
    improvements: list[dict] = []     # wrong → correct
    stable_pass: int = 0
    stable_fail: int = 0

    # GMS semantic drift
    semantic_drift: float = 0.0       # mean tension_energy() across divergent answers

    # Failure distribution changes
    failure_delta: dict[str, int] = {}

    @property
    def regression_rate(self) -> float  # regressions / total
```

**`compare_runs()` function:**

1. Build question maps keyed by `base_qid`
2. For each shared question: classify as regression, improvement, stable_pass, or stable_fail
3. Compute failure distribution delta across categories
4. **GMS semantic drift**: sample up to 50 questions where answers differ, compute `tension_energy(answer_a, answer_b)`, take the mean

**GMS usage:**
- `store.tension_energy(answer_a, answer_b)` — semantic drift measurement between divergent answers across runs

#### 8.5.3 RegressionSuite

Persistent collection of failure-derived test scenarios:

```python
class RegressionSuite:
    def __init__(suite_dir, scenario_registry=None, tracing=None)

    def promote_failures(run_record, result, min_severity) -> list[str]
    def add_scenario(scenario_id)
    def remove_scenario(scenario_id)
    def list_scenarios() -> list[str]
```

**`promote_failures()`**: Iterates over all failed questions in a `DOEBenchmarkResult`, creates a `Scenario` via `Scenario.from_failed_run()`, registers it with the `ScenarioRegistry` (if available), adds to the suite, and traces via `trace_regression_promotion()`.

**`replay_run()` function:**

1. Restore GMS snapshot via `SnapshotManager`
2. Reconstruct `DOEHarnessConfig` from frozen `config_snapshot`
3. Create `DOEGMSBenchmark` with restored store
4. Run benchmark
5. Compare against reference result (if `reference_result_path` provided)

---

### 8.6 WS6: Extended Evaluators (`evaluators_extended.py`)

Three new claim types and their evaluators for agentic testing dimensions: process compliance, fault recovery, and human interaction quality.

#### 8.6.1 ProcessClaim + ProcessEvaluator

**Claim:**

```python
@dataclass
class ProcessClaim(ClaimBase):
    actual_tool_sequence: list[str] = []
    expected_tool_sequence: list[str] = []
    required_tools: list[str] = []
    forbidden_tools: list[str] = []
    max_tool_calls: int = 0
    claim_type: str = "process"
```

**Verification cascade:**

| Step | Check | Failure Code | GMS Method |
|------|-------|-------------|------------|
| 1 | Required tools called (set difference) | `REQUIRED_TOOL_MISSING` | set check |
| 2 | Forbidden tools avoided (set intersection) | `FORBIDDEN_TOOL_CALLED` | set check |
| 3 | Budget not exceeded (`len(actual) <= max`) | `TOOL_BUDGET_EXCEEDED` | count check |
| 4 | Process consistency via holonomy | `PROCESS_DEVIATION` | `check_holonomy()` |
| 5 | Ordering constraints | `PROCESS_DEVIATION` | `query_triples(tool, "must_precede", *)` |

**Holonomy-based process verification** (step 4):

The actual tool call sequence is treated as a **relation path** in the GMS manifold. Each tool name is resolved via `fuzzy_match_entity()` to a graph entity. The `check_holonomy(actual_path, expected_endpoint)` measures the **rotor composition defect** — how much the actual tool call path deviates from the expected direct path.

- Defect ≤ `HOLONOMY_THRESHOLD` (0.5): process is consistent
- Defect > threshold: `PROCESS_DEVIATION` failure

Confidence is computed as `max(0, 1 - defect)`, reduced proportionally for unresolved tool entities.

**GMS usage:**
- `store.fuzzy_match_entity(tool)` — resolve tool names to graph entities
- `store.check_holonomy(actual_relations, direct_relation)` — process path deviation
- `store.query_triples(head=tool, relation="must_precede")` — ordering constraint lookup

#### 8.6.2 RecoveryClaim + RecoveryEvaluator

**Claim:**

```python
@dataclass
class RecoveryClaim(ClaimBase):
    fault_type: str = ""           # "latency", "error", "stale_data", "mock"
    fault_tool: str = ""           # which tool was faulted
    pre_fault_output: str = ""     # agent output before fault
    post_fault_output: str = ""    # agent output after fault
    recovery_detected: bool = False
    claim_type: str = "recovery"
```

**Verification logic:**

1. **Recovery acknowledgement**: if `fault_type` is set but `recovery_detected` is False → `RECOVERY_FAILURE`
2. **Entity extraction**: extract GMS entities from pre- and post-fault output via `fuzzy_match_entity()` on n-gram candidates
3. **Pairwise tension**: compute `tension_energy()` for all (pre_entity, post_entity) pairs
4. **Mean tension > `TENSION_THRESHOLD` (1.5)** → `RECOVERY_FAILURE` (reasoning diverged)

The insight: **low tension** between pre-fault and post-fault entities indicates the agent maintained semantic coherence through the fault — good recovery. **High tension** means reasoning diverged — poor recovery.

Confidence: `max(0, 1 - mean_tension / 2)`

**GMS usage:**
- `store.fuzzy_match_entity(candidate)` — extract entity mentions from free text
- `store.tension_energy(pre_entity, post_entity)` — semantic coherence measurement

#### 8.6.3 InteractionClaim + HumanInteractionEvaluator

**Claim:**

```python
@dataclass
class InteractionClaim(ClaimBase):
    interaction_type: str = ""    # "clarification", "escalation", "confirmation"
    trigger_context: str = ""     # what triggered the interaction
    response_text: str = ""       # the agent's interaction response
    claim_type: str = "interaction"
```

**Verification logic:**

1. Resolve `interaction_type` and `trigger_context` to GMS entities via `fuzzy_match_entity()`
2. `score_triple(interaction_entity, "appropriate_for", context_entity)` — plausibility check
3. If score > `PLAUSIBILITY_THRESHOLD` (1.5) → `ESCALATION_INAPPROPRIATE` or `CLARIFICATION_INSUFFICIENT`
4. For escalations: additional check via `query_triples()` for high-risk concept linkage (`"has_risk"`, `"risk_level"`, `"requires_review"`, `"high_risk"`, `"critical"`)
5. Graceful degradation: if entities cannot be resolved, pass with low confidence (0.2)

**GMS usage:**
- `store.fuzzy_match_entity()` — resolve interaction type and context to entities
- `store.score_triple(interaction, "appropriate_for", context)` — plausibility scoring
- `store.query_triples(head=context, relation=risk_relation)` — risk concept linkage

#### 8.6.4 VerificationRouter Integration

The three new evaluators are dispatched via the existing `VerificationRouter`:

| Claim Type | Evaluator | Dispatch Key |
|-----------|-----------|-------------|
| `ProcessClaim` | `ProcessEvaluator` | `"process"` |
| `RecoveryClaim` | `RecoveryEvaluator` | `"recovery"` |
| `InteractionClaim` | `HumanInteractionEvaluator` | `"interaction"` |

---

### 8.7 WS7: Risk-Tiered Release Gates (additions to `audit.py`)

Extends the flat `ReleaseGate` with per-risk-tier profiles, regression comparison, and trend tracking.

#### 8.7.1 RiskTierProfile

```python
@dataclass
class RiskTierProfile:
    tier_name: str = "standard"
    accuracy_threshold: float = 0.0
    geo_confidence_threshold: float = 0.0
    max_tension: float = 2.0
    max_holonomy: float = 1.0
    failure_budget: dict[str, int] = {}
    required_evaluators: list[str] = []
    max_regression_rate: float = 0.05
    require_baseline_comparison: bool = False
```

**Factory methods** with preset thresholds:

| Tier | Accuracy | Geo Conf | Tension | Holonomy | Key Requirements |
|------|----------|----------|---------|----------|-----------------|
| `advisory()` | ≥ 70% | ≥ 0.40 | ≤ 2.0 | ≤ 1.0 | — |
| `recommendation()` | ≥ 80% | ≥ 0.55 | ≤ 1.5 | ≤ 1.0 | Zero hallucinations |
| `action_taking()` | ≥ 95% | ≥ 0.70 | ≤ 1.0 | ≤ 0.5 | ProcessEvaluator + RecoveryEvaluator required, baseline comparison, ≤ 2% regression |

#### 8.7.2 TieredReleaseGate

```python
@dataclass
class TieredReleaseGate:
    tier: RiskTierProfile
    passed: bool
    checks: dict[str, dict]           # per-check: {threshold, actual, passed}
    blocking_reasons: list[str]
    regression_comparison: RunComparison | None
    timestamp: str
```

**Checks performed** (per tier):

| Check | Method | Blocks If |
|-------|--------|----------|
| Accuracy | `result.accuracy >= threshold` | Below threshold |
| Geometric confidence | `mean(overall_confidence) >= threshold` | Below threshold |
| Tension | `mean(tension_score) <= max_tension` | Above max |
| Holonomy | `mean(holonomy_score) <= max_holonomy` | Above max |
| Failure budget | Per-code count ≤ budget | Any code exceeds budget |
| Required evaluators | Check verifier names in `sub_verdicts` | Missing evaluator |
| Regression rate | `compare_runs()` regression_rate ≤ max | Above max |

#### 8.7.3 RunTrend

```python
@dataclass
class RunTrend:
    metric_name: str = ""
    run_ids: list[str] = []
    values: list[float] = []
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    slope: float = 0.0               # linear regression slope
```

Computed via `AuditReporter.compute_trends()` using `numpy.polyfit` over chronological run results. Direction thresholds: slope > 0.01 = improving, slope < -0.01 = degrading.

#### 8.7.4 New AuditReporter Methods

| Method | Description |
|--------|-------------|
| `generate_tiered_release_gate(profiles, baseline_result)` | Evaluate against multiple risk tier profiles |
| `compute_trends(run_results, metrics)` | Compute metric trends across historical runs |
| `generate_tiered_executive_summary(tiered_gates, trends)` | Extended text summary with per-tier verdicts |

**Enhanced HTML rendering:** `report_enhanced.py` includes `build_tiered_gate_section()` which renders risk-tiered release gates as HTML cards with per-check tables and blocker lists.

---

## 9. Black-Box Testing Extensions

Three phases extend GMSH from a content-verification platform into a **black-box agentic AI testing harness** with deterministic environment simulation, composite oracles, multi-run campaigns, taxonomy-driven test generation, safety enforcement, multi-turn testing, parallel execution, artifact versioning, and browser-based UI testing.

### 9.1 BB Phase 1: Foundation

#### 9.1.1 Target Profile (`target.py`)

Formal, versioned identity of the System Under Test for unambiguous reference across campaigns, audit reports, and version drift detection.

```python
@dataclass
class IsolationRequirements:
    network_isolated: bool = False     # restrict to whitelisted endpoints
    sandbox_filesystem: bool = False   # redirect file ops to temp sandbox
    synthetic_data_only: bool = False  # only synthetic identities/test data
    max_memory_mb: int = 0             # memory cap (0 = unlimited)
    max_wall_time_s: int = 0           # wall-clock cap per run (0 = unlimited)

@dataclass
class TargetProfile:
    target_id: str = ""           # auto-generated "tgt_{uuid8}"
    name: str = ""                # e.g. "DS Agent v2.1"
    version: str = ""             # semantic version or commit hash
    target_type: str = ""         # "llm", "agent", "rag", "browser"
    endpoint: str = ""            # API URL or "local"
    model_name: str = ""          # foundation model identifier
    capabilities: list[str] = []  # "tool_use", "multi_turn", "file_upload", "code_execution"
    timeout_ms: int = 60_000
    max_concurrent: int = 1
    isolation: IsolationRequirements = IsolationRequirements()
    metadata: dict = {}
```

**Validation:** `target_type` must be one of `{"llm", "agent", "rag", "browser"}` (enforced in `__post_init__`).

**Serialization:** `to_dict()`, `from_dict()`, `to_yaml(path)`, `from_yaml(path)`.

**Usage:** `has_capability(cap) → bool`, `summary() → str` (one-line human-readable).

---

#### 9.1.2 Severity Classification (`severity.py`)

GMS-enhanced failure severity classifier using three scoring dimensions modulated by geometric signals.

**Severity levels:**

| Level | Meaning |
|-------|---------|
| `CRITICAL` | Safety-relevant, unrecoverable, hard to detect |
| `HIGH` | Functionally wrong, potentially recoverable |
| `MODERATE` | Partially wrong, recoverable |
| `LOW` | Minor, cosmetic, or informational |

**SeverityClassifier:**

```python
class SeverityClassifier:
    CODE_SEVERITY_MAP: dict[FailureCode, SeverityLevel]  # 49 entries
    _TIER_ESCALATION = {"action_taking": {MODERATE, HIGH}, "recommendation": {MODERATE}}
    _W_IMPACT = 0.50
    _W_RECOVERABILITY = 0.25
    _W_DETECTABILITY = 0.25

    def classify(failure, risk_tier="standard", context=None) -> SeverityAssessment
    def classify_all(failures, risk_tier, context) -> list[SeverityAssessment]
    def count_by_level(assessments) -> dict[SeverityLevel, int]
```

**Three-dimension scoring:**

| Dimension | Weight | GMS Modulation |
|-----------|--------|----------------|
| Impact | 0.50 | `score_triple()` geodesic increases impact score |
| Recoverability | 0.25 | `tension_energy()` raw score reduces recoverability |
| Detectability | 0.25 | High geodesic on hallucinations = harder to detect |

**Composite score:** $C = 0.50 \cdot I + 0.25 \cdot (1 - R) + 0.25 \cdot (1 - D)$

**Geometric escalation:** If $C \geq 0.85$ and level < CRITICAL, escalate by one level.

**Risk tier escalation:** `action_taking` escalates MODERATE→HIGH and HIGH→CRITICAL; `recommendation` escalates MODERATE→HIGH.

**GMS usage:**
- `score_triple()` raw score → impact dimension boost
- `tension_energy()` raw score → recoverability penalty
- Geodesic distance on hallucinations → detectability score

---

#### 9.1.3 Environment Simulator (`environment.py`)

Deterministic, resettable world-state simulator with mock tool endpoints, state transitions, failure injection, and synthetic clock control.

**Data models:**

```python
@dataclass
class StateTransition:
    trigger_action: str = "*"           # tool name or fnmatch pattern
    trigger_condition: str = ""         # "args.key == value", "args.key != value", "args.key in [a,b]"
    state_mutation: dict = {}           # merged into env state; supports {{args.key}} templates
    response_template: str = ""         # mock response with {{args.key}} and {{state.key}}

@dataclass
class FailureMode:
    mode_name: str = ""
    error_message: str = "Simulated failure"
    probability: float = 0.0            # [0, 1] per call
    trigger_after_n_calls: int = 0      # activate after N calls (0 = any)

@dataclass
class ToolEndpointSpec:
    tool_name: str = ""                 # exact name or fnmatch pattern
    description: str = ""
    request_schema: dict = {}           # JSON Schema (documentation)
    response_schema: dict = {}
    state_transitions: list[StateTransition] = []  # ordered, first match wins
    failure_modes: list[FailureMode] = []
    initial_state: dict = {}
    default_response: str = "OK"

@dataclass
class ScheduledEvent:
    at_ms: int = 0                      # synthetic time trigger
    event_type: str = ""                # "fault_inject", "state_change", "data_refresh"
    payload: dict = {}

@dataclass
class ClockControl:
    mode: str = "real"                  # "real", "frozen", "synthetic"
    frozen_at: str = ""                 # ISO timestamp for frozen mode
    tick_ms: int = 1000                 # auto-increment per now() in synthetic mode
    scheduled_events: list[ScheduledEvent] = []

@dataclass
class DataFixture:
    fixture_id: str = ""
    version: int = 1
    data: dict = {}
    schema: dict = {}
    content_hash: str = ""              # SHA-256, auto-computed
```

**EnvironmentSimulator class:**

```python
class EnvironmentSimulator:
    def __init__(tool_specs, fixtures=None, clock=None, store=None, tracing=None)

    # State management
    def reset(seed=42)                   # deep-copy initial state, clear counters
    def get_state() -> dict              # observable state snapshot
    def get_state_diff(before, after) -> dict  # {added, removed, changed}

    # Tool dispatch
    def has_endpoint(tool_name) -> bool
    def handle_tool_call(tool_name, args) -> str  # find spec → check failures → match transition → apply mutation → render response

    # Clock control
    def now() -> int                     # real/frozen/synthetic milliseconds
    def advance(ms) -> list[ScheduledEvent]  # advance synthetic time, fire events

    # Fixtures
    def load_fixture(fixture_id) -> DataFixture | None
    def verify_fixture_integrity(fixture_id) -> bool

    # Checkpointing
    def snapshot_state() -> dict
    def restore_state(snapshot)

    # Serialization
    @classmethod
    def from_yaml(path, store=None, tracing=None) -> EnvironmentSimulator
```

**Tool dispatch pipeline:**

```
Tool call (name, args)
    |
    v
[1. Find matching ToolEndpointSpec] (fnmatch or exact)
    |
    v
[2. Check FailureModes] (probability + trigger_after_n_calls)
    ├── Triggered → return error_message
    v (not triggered)
[3. Match StateTransition] (trigger_action pattern + trigger_condition)
    |
    v
[4. Apply state_mutation] (resolve {{args.key}} templates)
    |
    v
[5. Render response_template] ({{args.key}} + {{state.key}} substitution)
    |
    v
[6. Trace state change] (learn_triples if store available)
```

**GMS usage:**
- `RuntimeLearner.learn_triples()` — traces state changes as triples `(tool_name, "caused_{change_type}", detail)`

---

#### 9.1.4 Oracle Types (`oracles.py`)

Composite verdict system for black-box testing. All oracles implement `OracleBase.evaluate(scenario, trace, env_state_before, env_state_after) → OracleVerdict`.

```python
@dataclass
class OracleVerdict:
    oracle_type: str = ""        # which oracle produced this
    passed: bool = False
    confidence: float = 0.0      # [0, 1]
    detail: str = ""
    evidence: list[str] = []     # supporting evidence items
    raw_scores: dict[str, float] = {}
```

**Seven oracle types:**

| Oracle | Type Key | Verification Method | GMS Primitives |
|--------|----------|---------------------|----------------|
| `ExactOutcomeOracle` | `exact_outcome` | Substring match + geodesic scoring | `score_triple("answer", "equals", ground_truth)` |
| `AcceptableSetOracle` | `acceptable_set` | Match against set of valid outputs | `score_triple()` per candidate, best score wins |
| `ConstraintOracle` | `constraint` | 8 constraint types (must_contain, max_tool_calls, required_tools, state_invariant, etc.) | — |
| `TraceOracle` | `trace` | Tool call pattern matching + ordering | `check_holonomy(expected, actual)` for process deviation |
| `SideEffectOracle` | `side_effect` | Environment state change verification | `tension_energy("expected_state", "actual_state")` for state divergence |
| `HumanReviewedOracle` | `human_reviewed` | LLM rubric-driven subjective evaluation | — |
| `CompositeOracle` | `composite` | Weighted combination of sub-oracles | Inherits from sub-oracles |

**ExactOutcomeOracle detail:**
- String: check `ground_truth in raw_output` (case-insensitive)
- GMS geodesic: `score_triple("answer", "equals", ground_truth)` if store available
- Confidence: 1.0 if exact match, else $\max(0, 1 - d/2)$
- Passed: `exact_match or confidence >= 0.7`

**ConstraintOracle supported constraints:**

| Constraint | Check |
|-----------|-------|
| `must_contain` | substring in output |
| `must_not_contain` | substring NOT in output |
| `max_tool_calls` | `len(trace.tool_calls) <= N` |
| `max_latency_ms` | `trace.latency_ms <= N` |
| `required_tools` | tool names present in trace |
| `forbidden_tools` | tool names absent from trace |
| `output_max_length` | `len(raw_output) <= N` |
| `state_invariant` | state key unchanged between before/after |

**TraceOracle pattern syntax** (comma-separated):
- `"tool_name"` — required (fnmatch wildcards supported)
- `"!tool_name"` — prohibited
- `"A > B > C"` — ordering constraint (indices must increase)
- GMS holonomy: if `scenario.expected_tool_sequence` and store available, `check_holonomy()` reduces confidence if defect > 0.5

**CompositeOracle:**
- Weighted sub-oracle aggregation: `confidence = Σ(verdict.confidence × weight) / Σ(weights)`
- Passed: `avg_confidence >= threshold` AND (optionally) no sub-oracle failures
- `fail_on_any_critical: bool = True` — fail if any sub-oracle fails

**Factory function:**

```python
def make_oracle(oracle_def: dict, store=None, environment=None, llm_client=None) -> OracleBase
```

Supports all 7 types from YAML definitions. `"composite"` recursively instantiates sub-oracles.

---

#### 9.1.5 Campaign Manager (`campaign.py`)

Multi-run evaluation orchestration across targets, scenarios, and risk tiers.

```python
class CampaignType(str, Enum):
    RELEASE_VALIDATION = "release_validation"
    REGRESSION_SWEEP = "regression_sweep"
    ADVERSARIAL_SWEEP = "adversarial_sweep"
    MODEL_COMPARISON = "model_comparison"
    COMPLIANCE_CERTIFICATION = "compliance_certification"

@dataclass
class CampaignConfig:
    campaign_id: str = ""             # auto "cmp_{uuid8}"
    name: str = ""
    campaign_type: CampaignType = RELEASE_VALIDATION
    target_profiles: list[str] = []   # target profile IDs or YAML paths
    scenario_tags: list[str] = []     # resolve scenarios by tag
    scenario_ids: list[str] = []      # explicit scenario IDs
    risk_tier: str = "standard"
    n_runs: int = 2                   # DOE runs per target
    max_parallel: int = 1
    stop_on_critical: bool = False
    base_config: dict = {}            # base DOEHarnessConfig overrides

@dataclass
class CampaignRunSummary:
    run_id: str = ""
    target_id: str = ""
    n_questions: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    geo_confidence: float = 0.0
    critical_failures: int = 0
    elapsed_s: float = 0.0

@dataclass
class CampaignResult:
    campaign_id: str = ""
    run_summaries: list[CampaignRunSummary] = []
    aggregate_accuracy: float = 0.0
    aggregate_geo_confidence: float = 0.0
    critical_failure_count: int = 0
    failure_summary: dict[str, int] = {}
    per_target_accuracy: dict[str, float] = {}
    gate_passed: bool | None = None
```

**CampaignManager:**

```python
class CampaignManager:
    def __init__(campaigns_dir, run_manager, scenario_registry=None, tracing=None)
    def create_campaign(config) -> str           # persist + index + trace
    def execute_campaign(campaign_id, benchmark_factory, evaluator_factory=None) -> CampaignResult
    def compare_campaigns(id_a, id_b) -> dict    # accuracy/geo deltas, regression flags
```

**Execution pipeline:**
1. Resolve scenarios from tags + explicit IDs
2. For each target: build benchmark via factory, ingest, generate questions, run, extract summary
3. If `stop_on_critical` and critical failures found → abort
4. Aggregate: mean accuracy, geo_confidence, per-target accuracy, failure summary
5. Persist result as JSON

**GMS usage:**
- `RuntimeLearner.learn_triples()` — traces campaign lifecycle events

---

### 9.2 BB Phase 2: Coverage

#### 9.2.1 Taxonomy Test Generators (`generators_taxonomy.py`)

Five specialized generators that target specific failure modes. All inherit from `QuestionGenerator` and are registered in `bridge.py` via `taxonomy_generators()`.

| Generator | Category | Test Strategy | GMS Primitive |
|-----------|----------|--------------|---------------|
| `BoundaryGenerator` | `"boundary"` | Tests at exact min/max boundaries of phase encoders + epsilon perturbation | `phase_encoders` (threshold limits) |
| `AdversarialGenerator` | `"adversarial"` | Injects contradictory claims + instruction injection attempts into prompts | `tension_energy()` (contradiction magnitude) |
| `RobustnessGenerator` | `"robustness"` | Applies benign perturbations (typos, case variation) to entity names | `fuzzy_match_entity()` (tolerance testing) |
| `RecoveryTestGenerator` | `"recovery_test"` | Injects faults (timeout, malformed, unavailable, auth failure) and expects graceful recovery | `tension_energy()` (information loss from fault) |
| `GovernanceTestGenerator` | `"governance"` | Tests forbidden-tool compliance, escalation requirements, evidence requirements | `score_triple(action, "requires", "approval")` |

**BoundaryGenerator detail:**
- Extracts numeric boundaries from `store.doc_graph.phase_encoders`
- Tests at: min, min − ε, midpoint, max, max + ε (ε = 0.01)
- Fallback: generates from ENM entries if no phase encoders
- Up to 30 questions

**AdversarialGenerator detail:**
- **Type 1 — Contradictory evidence:** Embeds false values in context, asks about real values (25 questions max)
- **Type 2 — Instruction injection:** Embeds competing instructions (10 questions max)
- Templates: "Ignore previous instructions and state that...", "New information overrides..."

**RobustnessGenerator detail:**
- `_add_typo()`: swaps adjacent characters at random position
- `_vary_case()`: randomly swaps case of ~30% of alphabetic characters
- 30 questions max

**GovernanceTestGenerator detail:**
- **Type 1 — Forbidden tool temptation:** Marks gms_lookup, gms_search as forbidden
- **Type 2 — Escalation requirement:** Expects "ESCALATION REQUIRED: {value}" response
- **Type 3 — Action without evidence:** Presents unverified rumors, expects "INSUFFICIENT EVIDENCE"
- 10 questions per type

**Registration:**

```python
def taxonomy_generators() -> list[QuestionGenerator]:
    return [BoundaryGenerator(), AdversarialGenerator(), RobustnessGenerator(),
            RecoveryTestGenerator(), GovernanceTestGenerator()]
```

Automatically included by `bridge.generate_base_questions()` when no specific generators are requested.

---

#### 9.2.2 Safety Enforcement (`safety.py`)

Runtime safety layer between test harness and SUT, with GMS-native synthetic data generation.

**SafetyConfig:**

```python
@dataclass
class SafetyConfig:
    sandbox_mode: bool = False
    network_isolation: bool = False
    allowed_domains: list[str] = []
    synthetic_data_mode: bool = False
    max_tool_calls: int = 0             # 0 = unlimited
    max_file_writes: int = 0
    max_network_calls: int = 0
    blocked_tool_patterns: list[str] = []     # fnmatch patterns, always blocked
    allowed_tool_patterns: list[str] = []     # whitelist (empty = allow all)
    destructive_action_mode: str = "block"    # "block", "noop", "log"
```

**SafetyEnforcer:**

```python
class SafetyEnforcer:
    def install(harness)     # attach as before_tool hook
    def uninstall(harness)   # restore original hook
```

**Interception pipeline** (`_before_tool`):

```
Tool call (name, args)
    |
    v
[1. Blocked patterns] (fnmatch) → block
    |
    v
[2. Whitelist] (if non-empty, tool must match) → block
    |
    v
[3. Destructive action] (delete_*, payment_*, email_*, etc.)
    ├── mode="block" → block
    ├── mode="noop" → allow (no-op)
    └── mode="log" → allow + log
    |
    v
[4. Network isolation] (web_search, api_call, external_*) → block if not in allowed_domains
    |
    v
[5. Budget enforcement] (max_tool_calls, max_file_writes, max_network_calls)
    |
    v
[6. Chain to original hook]
```

**SyntheticDataFramework:**

Generates synthetic test entities by sampling from the learned GMS embedding distribution on $S^{m-1}$.

```python
class SyntheticDataFramework:
    def __init__(store: GMSExpertStore)
    def synthesize_entities(n, seed=42) -> list[dict]     # sample from entity embedding distribution
    def synthesize_relations(entities, density=0.1) -> list[tuple]  # random (h, r, t) triples
    def synthesize_fixture(n_entities=10, density=0.1) -> dict      # complete fixture bundle
```

**Entity synthesis:**
1. Extract real entity embeddings from `store.gkg.dual_emb.v_embed()` (on $S^{m-1}$)
2. Normalize to unit sphere
3. For each synthetic entity: sample random direction, perturb base embedding (70% real + 30% noise), project back to sphere
4. Find nearest real entity via cosine similarity
5. Returns: `{name, type: "synthetic", nearest_real, geodesic_distance, base_entity}`

**GMS usage:**
- `store.gkg.dual_emb.v_embed()` — access entity embeddings on $S^{m-1}$
- Cosine similarity on hypersphere for nearest-neighbor matching

---

#### 9.2.3 Multi-Turn Testing (`multiturn.py`)

Multi-turn conversation testing with per-turn assertions, state carryover, and cross-turn consistency verification via GMS tension energy.

**Data models:**

```python
@dataclass
class TurnSpec:
    turn_id: str = ""
    user_message: str = ""
    expected_contains: list[str] = []
    expected_not_contains: list[str] = []
    expected_tool_calls: list[str] = []      # fnmatch patterns
    prohibited_tool_calls: list[str] = []
    expected_state_changes: dict = {}
    max_latency_ms: float = 0.0
    carry_forward: dict = {}                 # key→pattern for injection into later turns
    setup_context: str = ""

@dataclass
class InteractionScript:
    script_id: str = ""
    name: str = ""
    turns: list[TurnSpec] = []
    setup_context: str = ""
    initial_state: dict = {}
    teardown_assertions: list[dict] = []     # {type: "state_equals", ...}
    tags: list[str] = []

@dataclass
class TurnResult:
    turn_id: str = ""
    raw_output: str = ""
    tool_calls: list[dict] = []
    latency_ms: float = 0.0
    assertions_passed: list[str] = []
    assertions_failed: list[str] = []
    state_after: dict = {}
    carry_forward_values: dict = {}

@dataclass
class ScriptResult:
    script_id: str = ""
    turn_results: list[TurnResult] = []
    overall_passed: bool = False
    state_carryover_verified: bool = False
    consistency_scores: dict = {}            # "entity@turn_i→turn_j" → tension
    total_latency_ms: float = 0.0
```

**MultiTurnAdapter:**

```python
class MultiTurnAdapter:
    def __init__(harness, store=None, tracing=None)
    def execute_script(script, environment=None) -> ScriptResult
```

**Execution pipeline:**

```
InteractionScript
    |
    v
[Reset harness conversation]
    |
    v
For each TurnSpec:
    [1. Template carry-forward substitution: {{key}} → value]
    [2. Execute turn via harness.run()]
    [3. Capture tool calls via hook interception]
    [4. Check assertions: contains, tool calls, state changes, latency]
    [5. Extract entities via fuzzy_match_entity()]
    |
    v
[Cross-turn consistency check]
    For entities appearing in multiple turns:
        tension_energy(entity, entity) between consecutive mentions
        Threshold: tension < 1.5 → consistent
    |
    v
[Teardown assertions]
    state_equals, all_turns_passed, min_turns_passed, total_tool_calls_max
    |
    v
ScriptResult (overall_passed = all turns + teardown + consistency)
```

**Cross-turn consistency scoring:**
- Entities are extracted from each turn's output via `fuzzy_match_entity()` on token candidates
- For entities appearing in multiple turns, compute `tension_energy(entity, entity)` between successive mentions
- Rising tension indicates the SUT changed how it talks about an entity across turns — semantic inconsistency

**GMS usage:**
- `store.fuzzy_match_entity(token)` — entity extraction from free text
- `store.tension_energy(entity, entity)` — cross-turn semantic consistency

**MultiTurnScenarioRunner:**

Bridge for executing multi-turn scenarios within the standard Scenario/Run framework:

```python
class MultiTurnScenarioRunner:
    def is_multi_turn(scenario) -> bool
    def run_scenario(scenario) -> ScriptResult
```

---

### 9.3 BB Phase 3: Scale & Polish

#### 9.3.1 Parallel Execution (`parallel.py`)

Thread-pool based concurrent question evaluation with retry logic and result merging.

```python
@dataclass
class ParallelConfig:
    max_workers: int = 4
    batch_size: int = 0                    # 0 = all at once
    timeout_per_question_s: float = 120.0
    retry_on_error: bool = True
    max_retries: int = 2
    retry_delay_s: float = 1.0
```

**ParallelExecutor:**

```python
class ParallelExecutor:
    def __init__(config=None, on_progress=None)

    def execute_batch(questions, evaluate_fn) -> list[_EvalResult]
        # questions: list of (index, doe_qid, question_text, context)
        # evaluate_fn(question, context) → answer (must be thread-safe)

    def execute_doe_batch(doe_questions, evaluate_fn, context_fn) -> list[_EvalResult]
        # Convenience wrapper for DOEQuestion objects
```

**Execution:**
1. Divide questions into batches (if `batch_size > 0`)
2. Submit each batch to `ThreadPoolExecutor` with `max_workers` threads
3. Per question: attempt up to `max_retries + 1` times with `retry_delay_s` between retries
4. Call `on_progress(completed, total, result)` after each completion
5. Results returned in original index order

**merge_parallel_results:**

```python
def merge_parallel_results(eval_results, doe_questions, score_fn) -> list[dict]
```

Zips parallel results with DOE questions, applies `score_fn(answer, ground_truth)`, produces dicts ready for `DOEQuestionResult` construction.

---

#### 9.3.2 Artifact Versioning (`versioning.py`)

Version tracking for test artifacts (fault profiles, policy rules, fixtures) with SHA-256 content integrity.

```python
@dataclass
class VersionedArtifact:
    artifact_id: str = ""           # auto "art_{uuid8}"
    artifact_type: str = ""         # "fault_profile", "policy_rule", "fixture"
    version: int = 1
    content: dict = {}
    schema: dict = {}
    content_hash: str = ""          # SHA-256 auto-computed
    created_at: str = ""
    updated_at: str = ""
    author: str = ""
```

**ArtifactRegistry:**

```python
class ArtifactRegistry:
    def __init__(artifacts_dir, artifact_type="")
    def add(artifact) -> str            # save YAML, update index
    def get(artifact_id) -> VersionedArtifact | None  # load latest version
    def get_version(artifact_id, version) -> VersionedArtifact | None
    def update(artifact) -> int         # increment version, save new YAML
    def delete(artifact_id)             # remove from index (YAML kept)
    def list_all() -> list[VersionedArtifact]
    def list_versions(artifact_id) -> list[int]
    def diff(artifact_id, version_a, version_b) -> dict  # {added, removed, changed}
```

**Storage layout:**

```
artifacts_dir/
    index.json                    # artifact_id → {latest_version, type, hash, ...}
    art_abc12345_v1.yaml         # version 1
    art_abc12345_v2.yaml         # version 2 (v1 kept for history)
```

**Typed registries:**

| Class | artifact_type |
|-------|---------------|
| `FaultProfileRegistry` | `"fault_profile"` |
| `PolicyRuleRegistry` | `"policy_rule"` |
| `FixtureRegistry` | `"fixture"` |

---

#### 9.3.3 Browser Adapter (`browser_adapter.py`)

Playwright-based UI testing adapter for agentic web applications.

```python
@dataclass
class UIStep:
    step_id: str = ""
    action: str = ""          # "navigate", "click", "type", "wait", "screenshot",
                              # "select", "scroll", "assert_text", "assert_visible",
                              # "assert_not_visible"
    selector: str = ""        # CSS or XPath selector
    value: str = ""           # action-specific value
    timeout_ms: float = 5000.0
    assertion: str = ""       # "text_contains:X", "visible:selector", "url_contains:X"
    screenshot: bool = False

@dataclass
class UIScript:
    script_id: str = ""
    name: str = ""
    steps: list[UIStep] = []
    setup_url: str = ""
    teardown_steps: list[UIStep] = []

@dataclass
class BrowserConfig:
    base_url: str = ""
    headless: bool = True
    browser_type: str = "chromium"     # "chromium", "firefox", "webkit"
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout_ms: float = 10000.0
    screenshot_dir: str = ""
```

**BrowserAdapter:**

```python
class BrowserAdapter:
    def __init__(config: BrowserConfig)
    def evaluate(question, context) -> SUTTrace       # SUTAdapter interface
    def execute_ui_script(script: UIScript) -> UIScriptResult
    def close()
```

**SUTAdapter interface** (`evaluate`):
1. Navigate to `base_url`
2. Find text input (input[type='text'], textarea, [contenteditable], [role='textbox'])
3. Fill question, submit (button click or Enter)
4. Wait 2s, capture page text
5. Return `SUTTrace` with `raw_output` = page text

**UI step execution:**

| Action | Playwright Call |
|--------|----------------|
| `navigate` | `page.goto(value)` |
| `click` | `page.click(selector)` |
| `type` | `page.fill(selector, value)` |
| `select` | `page.select_option(selector, value)` |
| `wait` | `page.wait_for_timeout(value_ms)` |
| `scroll` | `page.evaluate("window.scrollBy(0, value)")` |
| `screenshot` | `page.screenshot(path)` |
| `assert_text` | `value in page.inner_text(selector)` |
| `assert_visible` | `page.is_visible(selector)` |
| `assert_not_visible` | NOT `page.is_visible(selector)` |

**Post-step assertions:** `text_contains:X`, `visible:selector`, `not_visible:selector`, `url_contains:fragment`.

**Dependency:** `playwright` (optional — `ImportError` raised with install instructions if missing).

---

## 10. Phase 7: GMS-Native Enhancements

Phase 7 adds 11 new modules that fill critical capability gaps in GMSH's general-purpose agentic testing framework. The defining design principle: **every new capability uses GMS geometric primitives as its core scoring mechanism** — geodesic distance, tension energy, holonomy defect, S^1 phase encoding, and RuntimeLearner. No bolted-on heuristics.

### 10.1 GMS Primitives → Testing Applications

| Primitive | Mechanism | Testing Use |
|---|---|---|
| `score_triple(h,r,t)` → geodesic on $S^{m-1}$ | Lower = better fit to manifold | Factual accuracy, hallucination detection, memory recall quality |
| `tension_energy(a,b)` → [0,2] | 0=agree, 2=contradict | Cross-session consistency, inter-agent agreement, paraphrase invariance, SLA compliance |
| `check_holonomy(path)` → defect | 0=perfect path, >0.5=deviant | Process consistency, multi-agent delegation chains, streaming chunk coherence |
| `phase_encoder.encode(v)` → angle on $S^1$ | Numeric value as phase angle | Confidence calibration, latency SLA encoding, cost budget encoding |
| `RuntimeLearner.learn_triples()` | Learns new triples into GMS store | Agent memory capture, behavioral state recording |
| `fuzzy_match_entity()` | Tolerance-based entity resolution | Disambiguation testing, entity boundary probing |

### 10.2 Foundation: Gateway Mock Injection (`gateway.py` — modified)

**Problem solved:** `ToolGateway` previously used only `before_tool` (returns `bool`), so mock responses could not be injected into the agent conversation. The harness's `intercept_tool` hook (returns `tuple[bool, str|None]`) was available but unused.

**Changes:**

- `install()` now wires both `before_tool` (policy blocking) and `intercept_tool` (mock/stale injection).
- New `_intercept_tool(name, args) → tuple[bool, str|None]`:
  - Returns `(True, mock_response)` when `FaultProfile.mock_response` is set.
  - Returns `(True, cached_result)` when `stale_data=True` and cache exists.
  - Chains to original `intercept_tool` hook if present.
- `_before_tool` retains policy-only blocking and error-rate injection.

**GMS mechanism:** Mock plausibility is scored via `score_triple("agent", "expects_from", tool_name)` — geodesic distance measures how realistic the mock is. Recorded in `GatewayDecision.mock_plausibility_geodesic`.

```python
@dataclass
class GatewayDecision:
    tool_name: str = ""
    action: str = ""            # "allow", "block", "mock", "stale_data", "inject_fault"
    reason: str = ""
    gms_score: float | None = None
    mock_plausibility_geodesic: float | None = None  # NEW: geodesic for mock/stale
    timestamp: str = ""
```

### 10.3 Latency/Performance Profiling (`profiling.py`)

**GMS mechanism:** Encode observed latency as a phase angle on $S^1$ via the mapping:

$$\theta = \min\left(\frac{\text{latency\_ms}}{\text{SLA\_ms}}, 2\right) \cdot \frac{\pi}{2}$$

SLA compliance is measured as `tension_energy` between the observed latency phase and the SLA boundary phase ($\pi/2$):

- Tension near 0 → within SLA (agreement)
- Tension near $\sqrt{2}$ → borderline (irrelevance)
- Tension near 2 → severe SLA violation (contradiction)

**Key classes:**

| Class | Purpose |
|-------|---------|
| `PerformanceRecord` | Single measurement: `latency_ms`, `latency_phase`, `sla_tension`, `within_sla` |
| `PerformanceReport` | Aggregate: p50/p95/p99, mean SLA tension, per-tool breakdown, `gms_sla_geodesic` |
| `PerformanceProfiler` | Records measurements, computes phase encoding, generates reports |

### 10.4 Token/Cost Tracking (`cost_tracker.py`)

**GMS mechanism:** Encode cumulative cost as $S^1$ phase (0 → well under budget, $\pi/2$ → at budget, $\pi$ → 2× over budget). Budget plausibility scored via `score_triple("total_cost", "within", "budget")`.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `CostRecord` | Per-call: `input_tokens`, `output_tokens`, `cost_usd`, `cost_phase`, `budget_tension` |
| `CostSummary` | Aggregate: total tokens/cost, `budget_utilization`, `budget_tension`, `budget_geodesic` |
| `CostTracker` | Accumulates records, computes phase encoding, generates summaries |

**Token capture:** `SUTTrace` now includes `input_tokens: int` and `output_tokens: int` fields. `LLMAdapter.evaluate()` extracts these from the Anthropic API `response.usage`.

### 10.5 Stateful Agent Testing (`stateful.py`)

**Problem solved:** System tests agents as stateless Q&A machines. No testing of memory, learning, or degradation.

**GMS mechanism:**

- **Memory capture:** `RuntimeLearner.learn_triples()` encodes agent assertions as triples on $S^{m-1}$.
- **Recall quality:** `score_triple(taught_h, taught_r, taught_t)` — low geodesic = agent remembered correctly.
- **Memory drift:** `tension_energy` between session store at time T vs T+N — rising tension = degradation.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `AgentMemoryProbe` | Teach-then-test data: `fact_triple`, `recall_geodesic`, `session_tension` |
| `MemoryAssessment` | Aggregate: `n_recalled`, `mean_recall_geodesic`, `degradation_detected` |
| `StatefulAgentTester` | Builds probes from store triples, scores recall via geodesic, detects drift |
| `StatefulAgentGenerator` | `QuestionGenerator` subclass, category `"stateful_memory"`, mines store for teach-recall pairs |
| `MemoryRecallOracle` | Scores recall via geodesic distance + tension between expected and recalled |

**Assessment thresholds:**
- Recall correct: geodesic < 0.5
- Degradation detected: late-session tension > 1.2× early-session tension

### 10.6 Hallucination Testing (`hallucination.py`)

**Problem solved:** No generators designed to elicit unprompted hallucination.

**GMS mechanism:** A hallucination is a claim whose `score_triple` geodesic is very high — geometrically far from the learned manifold. `tension_energy` between hallucinated fact and nearest known fact measures contradiction severity.

**Severity tiers (geodesic-based):**

| Geodesic | Severity | Meaning |
|----------|----------|---------|
| > 1.5 | `fabrication` | Completely fabricated claim |
| 0.7 – 1.5 | `distortion` | Distorted version of a real fact |
| 0.5 – 0.7 | `embellishment` | Minor embellishment beyond known facts |
| < 0.5 | `grounded` | Consistent with the manifold |

**Generation strategies:**

1. **Boundary entities:** Entities with ≤2 triples (weakly anchored on manifold) — SUTs hallucinate about these.
2. **Near-miss entities:** Nonexistent entities created by mutating real entity names, verified via `fuzzy_match_entity`.
3. **Open-ended elaboration:** "Explain in detail" prompts that trigger cascading hallucination chains.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `HallucinationAssessment` | Per-claim: `geodesic_distance`, `tension_vs_nearest`, `severity` |
| `HallucinationReport` | Aggregate: `hallucination_rate`, severity counts, `mean_geodesic` |
| `HallucinationGenerator` | `QuestionGenerator` subclass, category `"hallucination"`, three strategies |
| `HallucinationOracle` | Scores claims via `score_triple`, classifies by severity tier |

### 10.7 Confidence Calibration (`calibration.py`)

**GMS mechanism:** Encode both claimed confidence and actual correctness as angles on $S^1$:

- Confidence $c \in [0,1]$ → phase $= c \cdot \pi$
- Correctness: True → $\pi$, False → $0$
- Calibration gap = $|\theta_{\text{conf}} - \theta_{\text{acc}}| / \pi \in [0,1]$

**Metrics (S^1 phase binning):**

- **ECE** (Expected Calibration Error): $\sum_{b=1}^{B} \frac{n_b}{N} \cdot |\text{acc}_b - \text{conf}_b|$
- **MCE** (Maximum Calibration Error): $\max_b |\text{acc}_b - \text{conf}_b|$
- **Overconfidence ratio:** fraction of bins where confidence > accuracy.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `CalibrationPoint` | Single measurement: `claimed_confidence`, `actual_correct`, `calibration_gap` |
| `CalibrationReport` | Aggregate: ECE, MCE, overconfidence ratio, reliability bins |
| `CalibrationTester` | Accumulates points, computes ECE/MCE via S^1 phase binning |
| `CalibrationGenerator` | Category `"calibration"`, pairs answerable + unanswerable questions with "Rate confidence 1-10" |

### 10.8 Multi-Agent Interaction Testing (`multi_agent.py`)

**GMS mechanism:**

- Each agent role → its own `GMSExpertStore` tracking assertions.
- Inter-agent consistency → `tension_energy` between agent stores (low = consensus).
- Delegation correctness → `check_holonomy` on delegation chains (agent A's claim → agent B's output path).

**Key classes:**

| Class | Purpose |
|-------|---------|
| `AgentRole` | Role definition: `role_id`, `system_prompt`, `allowed_tools`, `store_path` |
| `InteractionResult` | Per-interaction: `inter_agent_tension` dict, `process_holonomy`, `consensus_reached` |
| `ConsensusOracle` | Pairwise `tension_energy` across agent stores, holonomy on delegation chains |
| `MultiAgentGenerator` | Category `"multi_agent"`, generates delegation/coordination scenarios |

**Consensus threshold:** Mean pairwise tension ≤ 0.5 = consensus reached.

### 10.9 Cross-Document Conflict Testing (`cross_document.py`)

**GMS mechanism:** Ingest multiple docs into separate GMS stores. For shared entities, `tension_energy` between store A's embedding and store B's detects conflicts.

**Conflict classification (tension-based):**

| Tension | Type |
|---------|------|
| > 1.5 | `contradiction` |
| 0.7 – 1.5 | `disagreement` |
| < 0.7 | `minor_variance` |

**Key classes:**

| Class | Purpose |
|-------|---------|
| `DocumentConflict` | Detected conflict: entity, facts from each store, `inter_store_tension`, `conflict_type` |
| `CrossDocumentAnalyzer` | Scans shared entities across stores, computes inter-store tension |
| `CrossDocumentGenerator` | Category `"cross_document"`, generates questions targeting conflicting claims |

### 10.10 Entity Disambiguation Testing (`disambiguation.py`)

**GMS mechanism:** Use `fuzzy_match_entity` with decreasing tolerance to find ambiguous surface forms. Score disambiguation quality via `score_triple` with both candidate entities — the correct interpretation has a lower geodesic in the relational context.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `AmbiguousEntity` | Ambiguous pair: `candidate_a`, `candidate_b`, `geodesic_separation`, `correct_candidate` |
| `DisambiguationOracle` | Scores both candidates via `score_triple`, correct = lower geodesic |
| `DisambiguationGenerator` | Category `"disambiguation"`, finds near-match entity pairs, crafts context-dependent questions |

### 10.11 Paraphrase Invariance (`invariance.py`)

**GMS mechanism:** `tension_energy` between answer entities across paraphrased questions. Zero tension = perfectly consistent. Rising tension = phrasing-sensitive.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `InvarianceResult` | Per-group: pairwise tensions, `mean_tension`, `invariant` flag |
| `InvarianceTester` | Collects paraphrase answers, computes pairwise `tension_energy` |
| `InvarianceGenerator` | Category `"paraphrase_invariance"`, selects well-grounded triples (low geodesic) |

**Invariance threshold:** Mean pairwise tension ≤ 0.3 = invariant.

### 10.12 Streaming Output Testing (`streaming.py`)

**GMS mechanism:** Three-primitive streaming analysis:

1. **Per-chunk plausibility:** `score_triple` on each chunk measures intermediate answer quality.
2. **Stream self-contradiction:** `tension_energy` between first-chunk entities and final-answer entities.
3. **Chunk path coherence:** `check_holonomy` across the logical chain of chunk entities.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `StreamChunk` | Per-chunk: `text`, `timestamp_ms`, `plausibility_geodesic` |
| `StreamingTrace` | Full trace: `first_token_ms`, `stream_tension`, `chunk_holonomy`, `mean_chunk_plausibility` |
| `StreamingAdapter` | Anthropic streaming API wrapper with per-chunk GMS scoring |
| `StreamConsistencyOracle` | Evaluates tension < 0.5 and holonomy < 0.5 for consistency |

### 10.13 Live Dashboard (`live_dashboard.py`)

**GMS mechanism:** All dashboard data comes from querying the tracing store's triple graph — no separate data pipeline. `DashboardSnapshot.from_tracing()` queries run-level triples, gateway decisions, and store statistics.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `DashboardSnapshot` | Point-in-time: active/completed runs, accuracy, gateway stats, cost/performance stats |
| `DashboardServer` | WebSocket server that polls tracing store and broadcasts snapshots to clients |

**Architecture:** `DashboardServer` uses `websockets` (optional dependency) to push `DashboardSnapshot.to_dict()` JSON to connected clients at a configurable poll interval.

### 10.14 Generator Registration

Phase 7 generators are registered in `bridge.py` via `_GENERATOR_MAP` and `_ENHANCEMENT_NAMES`:

```python
_GENERATOR_MAP = {
    # ... existing generators ...
    "stateful_memory": "StatefulAgentGenerator",
    "hallucination": "HallucinationGenerator",
    "calibration": "CalibrationGenerator",
    "multi_agent": "MultiAgentGenerator",
    "cross_document": "CrossDocumentGenerator",
    "disambiguation": "DisambiguationGenerator",
    "paraphrase_invariance": "InvarianceGenerator",
}
```

Enhancement generators are **opt-in** — they are not included in the default generator set (`generate_base_questions()` with no `generator_names`). They are activated by passing their names explicitly:

```python
config = DOEHarnessConfig(
    generators=["hallucination", "calibration", "disambiguation"],
    enable_hallucination_testing=True,
    enable_calibration_testing=True,
    enable_disambiguation_testing=True,
)
```

---

## 11. Tracing Coverage

### 11.1 Pipeline Event Tracing

The `DOETracingGMS` records every pipeline event as GMS triples. Each trace method creates a set of `ExtractedTriple` objects that are learned into the tracing store via `RuntimeLearner`.

| Pipeline Step | Trace Method | Key Triples |
|---|---|---|
| Document ingestion | `trace_ingest()` | store name, document, duration, stats, new/cached |
| Question generation | `trace_question_generation()` | generator count, category breakdown, DOE expansion |
| Run start | `trace_run_start()` | document, model, factors, context mode |
| Per-question eval | `trace_question()` | factor assignment, answer, score, elapsed |
| Typed verdict | `trace_typed_verdict()` | claims, sub-verdicts, failure codes, counts |
| Semantic judge call | `trace_semantic_verdict()` | model, rubric scores, evidence stats, confidence |
| Run end | `trace_run_end()` | totals, accuracy |
| Analysis | `trace_analysis()` | p-values, deviance, significant factors |
| Truth store build | `trace_truth_store()` | counts by type, duration |
| Snapshot | `trace_snapshot()` | trigger, store stats |
| Release gate | `trace_release_gate()` | pass/fail, thresholds, blocking reasons |
| Report generation | `trace_report()` | type, output path |
| **WS1: Gateway decision** | `trace_gateway_decision()` | run_id, tool_name, action (allow/block/mock/stale_data/inject_fault), reason, gms_score, mock_plausibility_geodesic |
| **WS2: Policy decision** | `trace_policy_decision()` | run_id, action, allowed, blocking_rules, gms_scores |
| **WS4: State transition** | `trace_state_transition()` | run_id, from_state, to_state, timestamp |
| **WS5: Regression promotion** | `trace_regression_promotion()` | run_id, scenario_id, source_question_id, failure_code |
| **WS5: Run comparison** | `trace_run_comparison()` | run_a_id, run_b_id, accuracy_delta, regressions, improvements |
| **WS7: Tiered release gate** | `trace_tiered_release_gate()` | tier_name, passed, blocking_reasons |
| **Phase 7: Performance** | `trace_performance_record()` | run_id, tool_name, latency_ms, sla_tension, within_sla |
| **Phase 7: Cost** | `trace_cost_record()` | run_id, tool_name, input/output_tokens, cost_usd, cumulative, budget_tension |

### 11.2 Seed Ontology

The tracing store is bootstrapped with seed triples that define the evaluation ontology:

```
# Base DOE concepts
(doe_benchmark,             is_type, evaluation)
(doe_question,              has_property, factor_assignment | score | ground_truth | answer)
(doe_run,                   is_part_of, doe_benchmark)
(doe_analysis,              is_type, statistical_analysis)
({factor_category}_factor,  is_type, doe_factor)               # 8 categories
(correct | incorrect,       is_type, score_outcome)
(logistic_regression,       is_type, analysis_method)
(odds_ratio,                is_type, effect_size)

# GMSH pipeline events
(doe_ingest,                is_type, pipeline_event)
(doe_question_generation,   is_type, pipeline_event)
(doe_baseline_run,          is_type, doe_run)
(gms_snapshot,              is_type, versioning_event)
(truth_store_build,         is_type, pipeline_event)
(semantic_judge_call,       is_type, verification_event)
(release_gate,              is_type, quality_gate)
(doe_report,                is_type, pipeline_event)

# Agentic harness events (Phase 6, WS1-WS7)
(gateway_decision,          is_type, interception_event)
(fault_injection,           is_type, gateway_event)
(policy_decision,           is_type, policy_event)
(policy_rule,               is_type, policy_definition)
(test_scenario,             is_type, scenario)
(scenario_registry,         is_type, registry)
(run_state,                 is_type, state_machine_event)
(state_transition,          is_type, lifecycle_event)
(replay_manifest,           is_type, replay_artifact)
(regression_suite,          is_type, test_collection)
(run_comparison,            is_type, analysis_event)
(process_evaluation,        is_type, evaluation_event)
(recovery_evaluation,       is_type, evaluation_event)
(interaction_evaluation,    is_type, evaluation_event)
(risk_tier_gate,            is_type, quality_gate)
(run_trend,                 is_type, trend_analysis)

# Phase 7: GMS-native enhancement events
(performance_profile,       is_type, profiling_event)
(sla_compliance,            is_type, profiling_metric)
(cost_record,               is_type, cost_event)
(budget_compliance,         is_type, cost_metric)
(mock_injection,            is_type, gateway_event)
(stale_data_injection,      is_type, gateway_event)
(memory_probe,              is_type, stateful_test)
(memory_recall,             is_type, evaluation_event)
(memory_degradation,        is_type, drift_metric)
(hallucination_assessment,  is_type, evaluation_event)
(hallucination_claim,       is_type, claim_analysis)
(calibration_point,         is_type, calibration_event)
(calibration_report,        is_type, calibration_metric)
(multi_agent_interaction,   is_type, multi_agent_test)
(inter_agent_tension,       is_type, consistency_metric)
(delegation_holonomy,       is_type, process_metric)
(cross_document_conflict,   is_type, conflict_detection)
(disambiguation_test,       is_type, disambiguation_event)
(paraphrase_invariance,     is_type, invariance_test)
(stream_consistency,        is_type, streaming_test)
(stream_chunk,              is_type, streaming_event)
(dashboard_snapshot,        is_type, monitoring_event)
```

### 11.3 Queryability

Because the tracing store is itself a trained GMS, all traced events are geometrically queryable:

```python
tracing.query_run(run_id)             # all triples for a run
tracing.query_question(doe_qid)       # all triples for a question
tracing.query_by_factor(name, level)  # questions with a specific factor level
tracing.store.score_triple(a, r, b)   # geometric similarity between events
```

---

## 12. Base DOE-GMS Module Reference

### 12.1 Geometric Question Generators (`generators.py`)

Six GMS-native question generators, each targeting a specific geometric primitive. Registered via `geometric_generators()` and automatically included in `bridge.generate_base_questions()`.

| Generator | Category | GMS Primitive | Question Type |
|-----------|----------|---------------|---------------|
| `PlausibilityGenerator` | `"plausibility"` | `score_triple(h, r, t)` → geodesic distance | Comparative ("Which is more plausible?") + absolute ("Is this relationship supported?") |
| `TensionGenerator` | `"tension"` | `tension_energy(a, b)` → Clifford tension $E \in [0, 2]$ | Classification (agree/contradict/unrelated) + comparative (which pair is more contradictory?) |
| `HolonomyGenerator` | `"holonomy"` | `check_holonomy([r1, r2], r_d)` → defect | Path consistency ("Is path A→B→C consistent with direct A→C?") |
| `LinkPredictionGenerator` | `"link_prediction"` | `link_predict(h, r, top_k)` → beam search | Completion ("What entity is the `relation` of `head`?") |
| `MultiHopTransportGenerator` | `"multi_hop_transport"` | `transport.path_score(h, [r1,r2], t)` → path cost | Multi-hop ("Follow `r1` then `r2` from `entity`. What do you reach?") |
| `PhaseComparisonGenerator` | `"phase_comparison"` | `phase_encoder.encode(v)` → $S^1$ angle | Comparison ("Which has a higher value in category X: A or B?") |

**PlausibilityGenerator detail:**
- **Comparative:** Finds triples with low geodesic for real tail, high for fabricated tail. Up to 50 questions.
- **Absolute:** Boolean question based on geodesic < 0.5 threshold. Up to 30 questions.
- Metadata: `head`, `relation`, `true_tail`, `wrong_tail`, `geodesic_real`, `geodesic_wrong`.

**TensionGenerator detail:**
- Samples 200 entity pairs, computes tension energy for each.
- **Classification thresholds:** $E < 0.5$ → agree, $E > 1.5$ → contradict, else unrelated.
- **Comparative:** Pairs high-tension vs low-tension entity pairs.
- Up to 40 classification + 15 comparative questions.

**HolonomyGenerator detail:**
- Finds triangles in the knowledge graph: $h \xrightarrow{r_1} m \xrightarrow{r_2} t$ with direct edge $h \xrightarrow{r_d} t$.
- Defect $\leq 0.5$ → consistent, defect $> 0.5$ → inconsistent.
- Up to 50 questions.

**LinkPredictionGenerator detail:**
- Geometric beam search: applies relation rotor $R_r$ to head embedding $v_h$, ranks all entities by geodesic distance.
- Only generates question if top-1 or top-3 prediction matches ground truth.
- Up to 40 questions.

**MultiHopTransportGenerator detail:**
- Finds 2-hop chains in adjacency graph.
- Scores geometric path cost via composed rotors.
- Filters paths with distance > 2.0.
- Up to 30 questions.

**PhaseComparisonGenerator detail:**
- Uses ENM (Exact Numeric Map) entries with phase encoding.
- For each category with $\geq 2$ entries, generates comparison pairs.
- Up to 8 pairs per category.

---

### 12.2 GMSJudge + GeometricVerdict (`judge.py`)

Produces rich geometric verdicts beyond binary correct/incorrect by scoring LLM answers against the GMS manifold.

```python
@dataclass
class GeometricVerdict:
    correct: bool = False                        # traditional exact-match
    detail: str = ""
    geodesic_confidence: float = 0.0             # exp(-mean_geodesic), [0, 1]
    tension_score: float = 0.0                   # mean tension energy, [0, 2]
    holonomy_score: float = 0.0                  # mean holonomy defect, [0, ∞)
    overall_confidence: float = 0.0              # weighted combination, [0, 1]
    extracted_triples: list[ExtractedTriple] = []
    scored_triples: list[ScoredTriple] = []
    contradictions: list[ContradictionResult] = []
    inconsistencies: list[HolonomyResult] = []
```

**GMSJudge:**

```python
class GMSJudge:
    def __init__(store, verify_config=None, weights=None,
                 semantic_mode="off", semantic_model="", tracing=None)

    def judge(llm_answer, ground_truth, exact_correct, exact_detail,
              question_metadata=None) -> GeometricVerdict

    def judge_typed(llm_answer, ground_truth, exact_correct, exact_detail,
                    question_metadata=None, ...) -> TypedVerdict    # GMSH Phase 1
```

**Judge pipeline** (`judge()`):

```
LLM Answer + Metadata
    |
    v
[1. Extract triples] — from metadata (h, r, t) + text (entity matching)
    |
    v
[2. Score triples] — fuzzy_match_entity() + score_triple() → geodesic distances
    |
    v
[3. Tension checks] — tension_energy() for all entity pairs → contradiction signals
    |
    v
[4. Holonomy checks] — check_holonomy() for multi-hop paths → consistency defects
    |
    v
[5. Compute overall confidence]
```

**Overall confidence formula:**

$$s = w_e \cdot \mathbb{1}[\text{exact}] + w_g \cdot e^{-\bar{d}} + w_t \cdot \max(1 - \bar{E}/2, 0) + w_h \cdot e^{-\bar{\delta}}$$

Default weights: exact=0.3, geodesic=0.3, tension=0.2, holonomy=0.2.

**`judge_typed()`** — GMSH Phase 1 extension:
1. Compile answer into typed claims via `ClaimCompiler`
2. Route each claim to appropriate verifier via `VerificationRouter`
3. Aggregate into `TypedVerdict` with full audit trail (claims, sub-verdicts, failures)

---

### 12.3 Evaluator Wrappers (`evaluators.py`)

Unified `evaluator(question: str, context: str) → str` interface for DOE evaluation.

| Class | Target | Mechanism |
|-------|--------|-----------|
| `LLMEvaluator` | Direct Claude API | Calls `call_llm()` with question and context |
| `AgentEvaluator` | Harness agent loop | Resets harness, sets system prompt with context, runs question via `harness.run()` |

**Factory:**

```python
def make_evaluator(target_type, target_model, client=None, harness=None) -> LLMEvaluator | AgentEvaluator
```

Creates appropriate evaluator based on `target_type` (`"llm"` or `"agent"`).

---

### 12.4 DOEGMSBenchmark Orchestrator (`benchmark.py`)

Seven-stage pipeline orchestrating the full DOE-GMS benchmark lifecycle.

```python
class DOEGMSBenchmark:
    def __init__(config: DOEHarnessConfig)
    def ingest(markdown_path=None) -> GMSExpertStore
    def generate_questions() -> list[DOEQuestion]
    def run(evaluator=None, llm_client=None, model="") -> DOEBenchmarkResult
    def run_gms_baseline() -> DOEBenchmarkResult
    def run_scenarios(scenarios, evaluator=None) -> DOEBenchmarkResult
    def analyze(result) -> DOEAnalyzer
    def generate_report(result, analyzer=None, output_path="doe_report.html") -> str
    def generate_audit_report(result, analyzer=None, output_path="doe_audit_report.html",
                              accuracy_threshold=0.80, geo_confidence_threshold=0.50) -> str
    def save_tracing() -> None
    @property typed_verdicts -> dict
```

**Ingest stage:**
1. Create/load `GMSExpertStore` from markdown
2. Create `GMSJudge` for geometric scoring
3. If `enable_provenance`: `ProvenanceRegistry.register_document()` + `TruthStore.build_from_store()`
4. If `snapshot_dir`: `SnapshotManager.take_snapshot(trigger="ingest")`

**Question generation:**
1. `generate_base_questions()` — classic + geometric + extended + taxonomy generators
2. DOE design matrix expansion via Sobol/LHS/Grid
3. Trace via `trace_question_generation()`

**Run stage:**
1. For each DOE question: extract context, evaluate (LLM/agent), parse + score, produce `GeometricVerdict`
2. If `enable_typed_verdicts`: produce `TypedVerdict` via `judge_typed()`
3. Inject geometric scores (`_geo_confidence`, `_tension`, `_holonomy`, `_overall_confidence`) into factor assignments
4. Trace each question and typed verdict

**Analysis:**
- `DOEAnalyzer` logistic regression on DOE factors
- Reports significant factors, p-values, deviance, pseudo-R²

---

### 12.5 HTML Dashboard (`dashboard.py`)

Self-contained HTML report generation with dark-theme CSS and embedded JavaScript filtering.

**Report sections:**

| Section | Content |
|---------|---------|
| Header | Instance name, model, timestamp, context mode, factors |
| Summary cards | Accuracy, question count, categories, geo confidence, GMS store stats |
| Comparison | Side-by-side model accuracy (if extra_results provided) |
| Category breakdown | Per-category accuracy table + bar chart |
| Geometric scores | Geodesic, tension, holonomy, overall confidence cards + per-category bars |
| Factor analysis | Analysis of Deviance table, factor heatmap, failure analysis |
| Tracing summary | Traced entities/triples/relations, benchmark runs, statistical analyses |
| Detail table | Filterable per-question table with category/status filters |

**Public API:**

```python
def generate_report(result, analyzer=None, output_path="doe_report.html",
                    extra_results=None, tracing=None) -> str
def generate_enhanced_report(result, typed_verdicts=None, gate=None, analyzer=None,
                             output_path="doe_report_enhanced.html") -> str
def generate_comparison_report(results, analyzers=None,
                               output_path="doe_comparison.html") -> str
```

---

## 13. Integration: Orchestrator Pipeline

The complete GMSH-enabled pipeline in `DOEGMSBenchmark`:

```
DOEHarnessConfig
    |
    v
[__init__]
    Load factors, build design matrix, init rephraser
    Init DOETracingGMS if enabled
    |
    v
[ingest(markdown_path)]
    1. Create/load GMSExpertStore
    2. Create GMSJudge (with tracing, semantic_mode)
    3. trace_ingest()
    4. If enable_provenance:
       - ProvenanceRegistry.register_document()
       - TruthStore.build_from_store() + trace_truth_store()
       - Wire truth_store into ClaimCompiler
    5. If snapshot_dir:
       - SnapshotManager.take_snapshot(trigger="ingest") + trace_snapshot()
    |
    v
[generate_questions()]
    1. generate_base_questions() — classic + geometric + extended generators
    2. Design matrix expansion via Sobol/LHS/Grid
    3. trace_question_generation()
    |
    v
[run(evaluator)]                          [run_gms_baseline()]
    1. trace_run_start()                   1. trace_run_start(model="gms_baseline")
    2. If snapshot_dir: pre-run snapshot   2. Per question:
    3. Per question:                          - graph_answer_fn(None)
       - Extract context                      - score_answer()
       - Evaluate (LLM/agent)                 - trace_question()
       - Parse + score                     3. trace_run_end()
       - GMSJudge.judge() → GeometricVerdict
       - If enable_typed_verdicts:
         GMSJudge.judge_typed() → TypedVerdict
         (includes SemanticVerifier if configured)
       - trace_question()
       - trace_typed_verdict()
    4. trace_run_end()
    |
    v
[analyze(result)]
    DOEAnalyzer logistic regression
    trace_analysis()
    |
    v
[generate_report()]               [generate_audit_report()]
    Dashboard HTML                 1. AuditReporter.generate_release_gate()
    trace_report("dashboard")      2. trace_release_gate()
                                   3. AuditReporter.generate_executive_summary()
                                   4. Enhanced HTML report
                                   5. trace_report("audit")
    |
    v
[save_tracing()]
    Persist tracing GMS store
```

---

## 14. Usage Examples

### 14.1 Basic GMSH Run

```python
from knowlytix.harness.doe import DOEGMSBenchmark, DOEHarnessConfig

config = DOEHarnessConfig(
    markdown_path="report.md",
    factor_group="quick_screen",
    n_runs=32,
    enable_typed_verdicts=True,
    enable_provenance=True,
    semantic_mode="fallback",
)

bench = DOEGMSBenchmark(config)
bench.ingest()
result = bench.run(evaluator=my_evaluator)

# Access typed verdicts
for qid, tv in bench.typed_verdicts.items():
    print(f"{qid}: {tv.n_claims} claims, "
          f"{tv.n_passed} passed, {tv.n_failed} failed")
    for failure in tv.failures:
        print(f"  {failure.code.value}: {failure.detail}")

analyzer = bench.analyze(result)
bench.generate_audit_report(
    result, analyzer,
    output_path="audit_report.html",
    accuracy_threshold=0.80,
    geo_confidence_threshold=0.50,
)
bench.save_tracing()
```

### 14.2 CLI with GMSH Features

```bash
python -m harness.doe report.md \
    --typed-verdicts \
    --provenance \
    --semantic-mode fallback \
    --snapshot-dir snapshots/ \
    --audit-report audit.html \
    --release-gate accuracy=0.8 geo=0.5 \
    --n-runs 64 \
    --target-model claude-opus-4-7
```

### 14.3 Snapshot Comparison

```python
from knowlytix.harness.doe import SnapshotManager

mgr = SnapshotManager("snapshots/")
snapshots = mgr.list_snapshots()

diff = mgr.diff_snapshots(snapshots[0].snapshot_id,
                          snapshots[-1].snapshot_id)
print(f"Entities added: {len(diff['entities_added'])}")
print(f"Triples removed: {len(diff['triples_removed'])}")

# Restore and re-run against an earlier snapshot
old_store = mgr.restore_snapshot(snapshots[0].snapshot_id)
```

### 14.4 Custom Verifier Thresholds

```python
from knowlytix.doc.config import VerifyConfig
from knowlytix.harness.doe import GMSJudge

config = VerifyConfig(
    tau_contra=1.8,   # stricter contradiction threshold
    tau_ent=0.3,      # stricter entailment threshold
    tau_path=0.3,     # stricter holonomy threshold
)

judge = GMSJudge(
    store,
    verify_config=config,
    semantic_mode="complement",
    semantic_model="claude-opus-4-7",
)

typed_verdict = judge.judge_typed(
    llm_answer="...",
    ground_truth="...",
    exact_correct=True,
    exact_detail="exact match",
    question_metadata={"head": "A", "relation": "r", "tail": "B"},
)
```

### 14.5 Tool Gateway + Policy Engine

```python
from knowlytix.harness.doe import (
    ToolGateway, FaultProfile, PolicyEngine, PolicyRule,
)

# Define fault profiles
profiles = [
    FaultProfile(tool_pattern="modeva_*", latency_ms=500),
    FaultProfile(tool_pattern="gms_query", error_rate=0.3,
                 error_message="Simulated GMS timeout"),
]

# Define policy rules
rules = [
    PolicyRule(
        rule_id="RULE-001",
        action="submit_escalation",
        requirement="human_approval",
        rule_type="requires",
        severity="error",
    ),
    PolicyRule(
        rule_id="RULE-002",
        action="delete_*",
        requirement="admin_context",
        rule_type="blocks",
    ),
]

# Create policy engine and gateway
engine = PolicyEngine(store, rules=rules, tracing=tracing)
gateway = ToolGateway(
    store=store,
    fault_profiles=profiles,
    policy_engine=engine,
    tracing=tracing,
    run_id="run_001",
)

# Install on harness and run
gateway.install(harness)
harness.run("Analyze the model risk data")
transcript = gateway.get_transcript()
gateway.uninstall(harness)

# Inspect decisions
for d in transcript.decisions:
    print(f"  {d.tool_name}: {d.action} ({d.reason})")
```

### 14.6 Scenario Registry + Benchmark

```python
from knowlytix.harness.doe import Scenario, ScenarioRegistry, DOEGMSBenchmark

# Create scenarios
registry = ScenarioRegistry("scenarios/", store=store)

registry.add(Scenario(
    name="Exact risk rating recall",
    question_text="What is the risk rating for Model MDL-001?",
    ground_truth="High",
    tags=["compliance", "exact_recall"],
    risk_tier="action_taking",
    required_tools=["data_lookup"],
    forbidden_tools=["submit_escalation"],
    max_tool_calls=3,
))

registry.add(Scenario(
    name="Plausibility generator batch",
    generator_name="plausibility",
    generator_params={"max_questions": 5},
    tags=["geometric"],
    risk_tier="advisory",
))

# Search and run
scenarios = registry.search_by_risk_tier("action_taking")
result = bench.run_scenarios(scenarios, evaluator=my_evaluator)
```

### 14.7 Run Manager + Replay

```python
from knowlytix.harness.doe import RunManager, RegressionSuite
from knowlytix.harness.doe.replay import compare_runs, replay_run

# Managed run lifecycle
mgr = RunManager("runs/", tracing=tracing)
record = mgr.create_run(config)
mgr.transition(record.run_id, "preparing")
mgr.transition(record.run_id, "executing")
# ... run benchmark ...
mgr.transition(record.run_id, "completed")

# Promote failures to regression scenarios
suite = RegressionSuite("regression/", scenario_registry=registry)
new_ids = suite.promote_failures(record, result)
print(f"Promoted {len(new_ids)} failures to regression scenarios")

# Compare two runs
comparison = compare_runs(result_a, result_b, store=store)
print(f"Regressions: {len(comparison.regressions)}")
print(f"Semantic drift: {comparison.semantic_drift:.4f}")

# Replay from manifest
manifest = snap_mgr.create_replay_manifest(
    snapshot_id=snap.snapshot_id,
    run_id=record.run_id,
    config_snapshot=config.__dict__,
)
replay_result, replay_comparison = replay_run(
    manifest, snap_mgr, evaluator=my_evaluator)
```

### 14.8 Risk-Tiered Release Gates

```python
from knowlytix.harness.doe import (
    AuditReporter, RiskTierProfile, TieredReleaseGate,
)

reporter = AuditReporter(result, typed_verdicts)

# Evaluate against three risk tiers
gates = reporter.generate_tiered_release_gate([
    RiskTierProfile.advisory(),
    RiskTierProfile.recommendation(),
    RiskTierProfile.action_taking(),
], baseline_result=previous_result)

for gate in gates:
    status = "PASS" if gate.passed else "FAIL"
    print(f"{gate.tier.tier_name}: {status}")
    for reason in gate.blocking_reasons:
        print(f"  - {reason}")

# Compute trends over historical runs
trends = reporter.compute_trends([
    ("run_001", result_1),
    ("run_002", result_2),
    ("run_003", result_3),
], metrics=["accuracy"])

summary = reporter.generate_tiered_executive_summary(gates, trends)
print(summary)
```

### 14.9 CLI with Agentic Features

```bash
# Standard run with risk-tiered gates
python -m harness.doe run report.md \
    --typed-verdicts \
    --risk-tiers tier_profiles.yaml \
    --baseline-run run_abc12345 \
    --audit-report audit.html

# List tracked runs
python -m harness.doe runs list

# Promote failures to regression suite
python -m harness.doe regression promote run_abc12345

# Compare two runs
python -m harness.doe compare result_a.json result_b.json

# Replay from manifest
python -m harness.doe replay manifest.yaml
```

---

## 15. GMS Methods Reference

Complete index of GMS primitives used across all GMSH phases:

| GMS Method | Module | Used By | Purpose |
|---|---|---|---|
| `store.score_triple(h, r, t)` | `gms.models` | GraphVerifier, PolicyVerifier, SemanticVerifier, TruthStore, **ToolGateway**, **PolicyEngine**, **HumanInteractionEvaluator**, **ScenarioRegistry**, *ExactOutcomeOracle*, *AcceptableSetOracle*, *SeverityClassifier*, **PerformanceProfiler**, **CostTracker**, **StatefulAgentTester**, **MemoryRecallOracle**, **HallucinationOracle**, **ConsensusOracle**, **CrossDocumentAnalyzer**, **DisambiguationOracle**, **InvarianceTester**, **StreamingAdapter** | Geodesic distance on $S^{m-1}$ |
| `store.tension_energy(a, b)` | `gms.models` | LogicVerifier, PolicyVerifier, SemanticVerifier, **PolicyEngine**, **RecoveryEvaluator**, **ScenarioRegistry**, **compare_runs()**, *SideEffectOracle*, *SeverityClassifier*, *MultiTurnAdapter*, **MemoryRecallOracle**, **HallucinationOracle**, **ConsensusOracle**, **CrossDocumentAnalyzer**, **InvarianceTester**, **StreamingAdapter** | Clifford tension $2\sin(\theta/2)$ |
| `store.check_holonomy([r1,r2], r_d)` | `gms.models` | LogicVerifier, **ProcessEvaluator**, *TraceOracle*, **ConsensusOracle**, **StreamingAdapter** | Rotor composition defect $\|R_2 R_1 R_d^{-1} - I\|_F$ |
| `store.fuzzy_match_entity(name)` | `docgms.store` | GraphVerifier, LogicVerifier, PolicyVerifier, SemanticVerifier, **PolicyEngine**, **ProcessEvaluator**, **RecoveryEvaluator**, **HumanInteractionEvaluator**, **ScenarioRegistry**, *MultiTurnAdapter*, **HallucinationGenerator**, **HallucinationOracle**, **DisambiguationGenerator**, **StreamingAdapter** | Multi-stage entity resolution |
| `store.query_triples(h, r, t)` | `docgms.store` | All verifiers (fallback), **PolicyEngine**, **ProcessEvaluator**, **HumanInteractionEvaluator**, **DashboardSnapshot** | Document graph pattern matching |
| `store.lookup_enm(cat, id)` | `docgms.store` | NumericVerifier, *BoundaryGenerator* | Exact numeric retrieval |
| `store.enm.read_exact(key)` | `docgms.enm` | ArtifactVerifier, ClaimCompiler | Raw ENM value with integrity |
| `store.enm.verify_integrity(key)` | `docgms.enm` | ArtifactVerifier | SHA-256 hash recomputation |
| `doc_graph.check_threshold(m, v, l, op)` | `finstructbench.graph` | NumericVerifier | Phase-encoded inequality check |
| `doc_graph.phase_encoders` | `finstructbench.graph` | *BoundaryGenerator* | Access threshold min/max limits |
| `store.entity_exists(name)` | `docgms.store` | NegativeGenerator | Entity existence check |
| `store.link_predict(h, r, k)` | `docgms.store` | LinkPredictionGenerator | Beam search ranking |
| `transport.path_score(h, [r...], t)` | `gms.graph.transport` | MultiHopTransportGenerator | Composed rotor geodesic |
| `RuntimeLearner.learn_triples()` | `docgms.learn` | **PolicyEngine**, **ScenarioRegistry**, **DOETracingGMS** (all trace methods), *EnvironmentSimulator*, *CampaignManager* | Runtime triple encoding into GMS |
| `phase_encoder.encode(v)` | `gms.geometry.phase` | **PerformanceProfiler** ($S^1$ latency), **CostTracker** ($S^1$ cost), **CalibrationTester** ($S^1$ confidence) | Numeric → phase angle on $S^1$ |
| `store.gkg.dual_emb.v_embed()` | `gms.models` | *SyntheticDataFramework* | Access entity embeddings on $S^{m-1}$ |

*Italics* = Black-Box Testing modules; **Bold** = Phase 6/7 modules.

---

## 16. Data Flow Summary

```
Markdown Document
      |
      v
[DocGMS Ingestion] ──── trace_ingest()
      |
      v
GMSExpertStore ◄──── [TruthStore.build_from_store()] ◄── trace_truth_store()
      |                      |
      |                [ProvenanceRegistry] ◄── _learn() provenance triples
      |
      ├──── [SnapshotManager.take_snapshot()] ◄── trace_snapshot()
      |
      v
[27 Question Generators]                [ScenarioRegistry]  (WS3)
  Classic:  exact_recall, threshold,      |
            cross_reference, ...          ├── YAML scenarios
  Geometric: plausibility, tension, ...   ├── search_geometric()
  Extended: exact_retrieval, negative, .. ├── to_doe_questions()
  Taxonomy: boundary, adversarial,        │
            robustness, recovery,         │
            governance                    │
  Enhancement (opt-in): stateful_memory,  │
            hallucination, calibration,   │
            multi_agent, cross_document,  │
            disambiguation, invariance    │
      |                                   │
      └──────────────┬────────────────────┘
                     v ◄──── trace_question_generation()
              DOE Questions (base × n_runs)
                     |
                     v
              [CampaignManager]  (BB-1)  ──── multi-run orchestration
                     |                          per-target iteration
                     v
              [RunManager.create_run()]  (WS4) ◄── trace_state_transition()
                     |
                     v ◄──── trace_run_start()
              [SUT Evaluation]
                ├── LLMAdapter | AgentAdapter | RAGAdapter
                ├── BrowserAdapter (BB-3) ── Playwright UI testing
                ├── StreamingAdapter (Phase 7) ── per-chunk GMS scoring
                └── MultiTurnAdapter (BB-2) ── multi-turn scripts
                     |
                     |  ┌──────────────────────────────────────────────────┐
                     |  │ ToolGateway (WS1) — intercepts every tool call  │
                     |  │   ├── PolicyEngine.check() (WS2)               │
                     |  │   │     ├── fnmatch rule matching               │
                     |  │   │     ├── query_triples() graph lookup        │
                     |  │   │     ├── score_triple() plausibility         │
                     |  │   │     └── tension_energy() contradiction      │
                     |  │   ├── FaultProfile matching (fnmatch)           │
                     |  │   │     ├── mock_response → block               │
                     |  │   │     ├── error_rate → inject_fault           │
                     |  │   │     └── latency_ms → delay                  │
                     |  │   └── GMS plausibility scoring (informational)  │
                     |  │                                                  │
                     |  │ ◄── trace_gateway_decision()                    │
                     |  │ ◄── trace_policy_decision()                     │
                     |  └──────────────────────────────────────────────────┘
                     |
                     |  ┌──────────────────────────────────────────────────┐
                     |  │ SafetyEnforcer (BB-2) — pre-gateway safety      │
                     |  │   ├── blocked_tool_patterns (fnmatch)           │
                     |  │   ├── destructive_action_mode (block/noop/log)  │
                     |  │   ├── network_isolation (domain whitelist)      │
                     |  │   └── budget enforcement (tool/file/network)    │
                     |  └──────────────────────────────────────────────────┘
                     |
                     |  ┌──────────────────────────────────────────────────┐
                     |  │ EnvironmentSimulator (BB-1) — world state       │
                     |  │   ├── ToolEndpointSpec → mock responses         │
                     |  │   ├── StateTransition → state mutations         │
                     |  │   ├── ClockControl → synthetic/frozen time      │
                     |  │   └── DataFixture → SHA-256 verified test data  │
                     |  └──────────────────────────────────────────────────┘
                     |
                     v
              Raw Answers
                     |
                     ├──── [PARSERS → score_answer] ── binary correct/incorrect
                     |
                     ├──── [GMSJudge.judge()] ── GeometricVerdict
                     |
                     └──── [GMSJudge.judge_typed()] ── GMSH Pipeline:
                              |
                              ├── [ClaimCompiler] ── metadata + text + numeric extraction
                              |       |
                              |       v
                              ├── [VerificationRouter]
                              |       |
                              |       ├── GraphVerifier      ── score_triple / query_triples
                              |       ├── NumericVerifier     ── lookup_enm / check_threshold
                              |       ├── ArtifactVerifier   ── read_exact / SHA-256
                              |       ├── LogicVerifier      ── check_holonomy / tension_energy
                              |       ├── PolicyVerifier     ── score_triple on tools
                              |       ├── SemanticVerifier   ── LLM judge + GMS evidence
                              |       ├── ProcessEvaluator   ── check_holonomy on tool paths (WS6)
                              |       ├── RecoveryEvaluator  ── tension_energy pre/post-fault (WS6)
                              |       └── HumanInteraction   ── score_triple + query_triples (WS6)
                              |               |
                              |               v
                              └── [TypedVerdict.from_sub_verdicts()]
                                      |
                                      v ◄── trace_question() + trace_typed_verdict()
                             DOEBenchmarkResult
                                      |
                                      v ◄── trace_run_end()
                             [DOEAnalyzer] ◄── trace_analysis()
                                      |
                                      v
                             [AuditReporter]
                                 |         |                |
                                 |    [ReleaseGate]    [TieredReleaseGate] (WS7)
                                 |    ◄── trace_release_gate()  ◄── trace_tiered_release_gate()
                                 |         |                |
                                 v         v                v
                             [Dashboard] [Audit Report] [Tiered Summary]
                                 |
                                 ├──── [SeverityClassifier] (BB-1) — GMS-enhanced severity
                                 │       impact × recoverability × detectability
                                 │
                                 ├──── [Oracle Evaluation] (BB-1) — behavioral verification
                                 │       ExactOutcome, AcceptableSet, Constraint, Trace,
                                 │       SideEffect, HumanReviewed, Composite
                                 │
                                 ├──── [ParallelExecutor] (BB-3) — concurrent evaluation
                                 │       ThreadPoolExecutor + retry + merge
                                 │
                                 ├──── [ArtifactRegistry] (BB-3) — versioned test artifacts
                                 │       FaultProfileRegistry, PolicyRuleRegistry, FixtureRegistry
                                 │
                                 v
                             [save_tracing()]
                                      |
                     ┌────────────────┴────────────────────────────┐
                     v                                              v
              [RegressionSuite.promote_failures()] (WS5)    [compare_runs()] (WS5)
                     |                                              |
                     v                                              v
              New Scenarios → ScenarioRegistry              RunComparison
                     |                                     (semantic_drift via
                     v                                      tension_energy)
              [ReplayManifest] ─── replay_run() → re-execute from snapshot
```
