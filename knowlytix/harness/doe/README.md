# GMSH: Geometric Memory System Harness

Enterprise testing platform for evaluating LLM agents using **Geometric Memory Systems (GMS)** as the ground-truth backbone.

GMSH goes beyond binary pass/fail scoring. It decomposes agent output into typed, verifiable claims and anchors every verification decision to GMS geometric primitives — geodesic distance, tension energy, holonomy defects — on the unit hypersphere S^{m-1}.

## Why GMSH

| Problem | GMSH Solution |
|---------|---------------|
| LLM evaluations rely on string matching or LLM-as-judge with no ground truth | Every verdict is computed from a trained geometric model |
| No systematic coverage of test factors | Design of Experiments (Sobol/LHS) expands questions across factor dimensions |
| Agent tool calls are unaudited | Tool Gateway intercepts, logs, and optionally blocks or injects faults |
| No regression tracking across runs | Snapshot diffing, replay manifests, and failure-to-scenario promotion |
| Release decisions are ad-hoc | Risk-tiered release gates with configurable thresholds per deployment tier |

## Quick Start

```python
from harness.doe import DOEGMSBenchmark, DOEHarnessConfig

config = DOEHarnessConfig(
    markdown_path="your_document.md",
    n_runs=32,
)

bench = DOEGMSBenchmark(config)
bench.ingest()
result = bench.run(evaluator=your_evaluator)
bench.analyze(result).print_report()
```

Or from the CLI:

```bash
python -m harness.doe --markdown your_document.md --n-runs 32
```

## Architecture

GMSH is built in ten phases, each opt-in via config flags:

```
Phase 1     Claim Compiler + Typed Verifiers + Failure Taxonomy
Phase 2     Provenance Chain + Truth Objects
Phase 3     Extended SUT Adapters + Test Generators
Phase 4     Versioning + Snapshots
Phase 5     Audit Reporting + Release Gates
Phase 6     Agentic Testing Harness (7 workstreams)
BB Phase 1  Black-Box Testing Foundation (target, severity, environment, oracles, campaigns)
BB Phase 2  Black-Box Testing Coverage (taxonomy generators, safety, multi-turn)
BB Phase 3  Black-Box Testing Scale (parallel execution, versioning, browser adapter)
Phase 7     GMS-Native Enhancements (11 modules)
```

### GMS Geometric Primitives

These are the verification building blocks. Every evaluator and verifier calls one or more of these:

| Primitive | What it measures |
|-----------|-----------------|
| `score_triple(h, r, t)` | Geodesic distance — how plausible is this fact? |
| `tension_energy(a, b)` | Clifford tension — do these entities contradict? |
| `check_holonomy(path)` | Rotor composition defect — is this reasoning path consistent? |
| `phase_encoder.encode(v)` | Phase angle on S^1 — numeric → geometric (latency, cost, confidence) |
| `lookup_enm(key)` | Exact numerical memory — SHA-256 verified numeric recall |
| `link_predict(h, r, k)` | Geometric beam search — what entities complete this relation? |
| `fuzzy_match_entity(text)` | Multi-stage entity resolution |
| `RuntimeLearner.learn_triples()` | Runtime triple encoding — agent memory, state, policy rules |

## Module Layout

```
harness/doe/
    # Core
    config.py               Configuration (60+ fields, all with defaults)
    benchmark.py            7-stage orchestrator
    bridge.py               Question format adapter (27 generators registered)
    generators.py           6 geometric question generators
    evaluators.py           LLM/Agent evaluator wrappers
    judge.py                GeometricVerdict + TypedVerdict scoring
    tracing.py              Full event tracing as GMS triples
    cli.py                  CLI entry point
    dashboard.py            HTML report generation

    # Phase 1: Claim-Level Verification
    claims.py               7 claim types + ClaimCompiler
    verifiers.py            5 verifiers + VerificationRouter
    failures.py             27 failure codes + FailureAttribution
    verdict.py              TypedVerdict (backwards-compatible)

    # Phase 2: Provenance
    provenance.py           Document lineage (SHA-256 content-addressed)
    truth.py                4 typed truth classes + TruthStore

    # Phase 3: Extended Testing
    adapters.py             LLM, Agent, RAG adapters with token tracking
    test_generators.py      ExactRetrieval, Negative, WorkflowCompliance

    # Phase 4: Versioning
    snapshots.py            GMS store snapshots with diff

    # Phase 5: Audit
    audit.py                Audit packets + release gates + risk-tiered gates
    report_enhanced.py      Enhanced HTML dashboard sections

    # Phase 6: Agentic Harness
    gateway.py              WS1 — Tool interception + fault/mock injection
    policy.py               WS2 — Pre-execution policy engine
    scenarios.py            WS3 — Scenario registry (CRUD, versioning, search)
    runs.py                 WS4 — Run state machine
    replay.py               WS5 — Replay manifests + regression suites
    evaluators_extended.py  WS6 — Process, Recovery, Interaction evaluators

    # Black-Box Testing — Foundation
    target.py               TargetProfile + IsolationRequirements
    severity.py             GMS-enhanced 4-level severity classifier
    environment.py          Deterministic world simulator + mock endpoints
    oracles.py              7 oracle types + CompositeOracle
    campaign.py             Multi-run campaign orchestration

    # Black-Box Testing — Coverage
    generators_taxonomy.py  5 taxonomy generators (boundary, adversarial, etc.)
    safety.py               SafetyEnforcer + SyntheticDataFramework
    multiturn.py            Multi-turn scripts + cross-turn GMS consistency

    # Black-Box Testing — Scale & Polish
    parallel.py             Thread-pool concurrent evaluation
    versioning.py           Versioned artifact registry (fault profiles, etc.)
    browser_adapter.py      Playwright-based UI testing

    # Phase 7: GMS-Native Enhancements
    profiling.py            S^1 latency/SLA profiling
    cost_tracker.py         S^1 token/cost tracking
    stateful.py             Agent memory teach-then-test
    hallucination.py        Geodesic hallucination detection
    calibration.py          S^1 confidence calibration
    multi_agent.py          Inter-agent tension/holonomy testing
    cross_document.py       Cross-document conflict detection
    disambiguation.py       Entity disambiguation via fuzzy_match
    invariance.py           Paraphrase invariance testing
    streaming.py            Streaming chunk consistency
    live_dashboard.py       WebSocket live monitoring

    # Cross-Cutting
    semantic.py             LLM-as-judge calibrated by GMS evidence
```

## Phase 6: Agentic Testing Harness

Seven workstreams extend GMSH for black-box agent evaluation:

| WS | Name | What it does |
|----|------|-------------|
| 1 | Tool Gateway | Intercepts agent tool calls at runtime. Applies fault profiles (latency, errors, stale data, mocks). Produces auditable transcripts. |
| 2 | Policy Engine | Pre-execution policy checking with 4 layers: pattern matching, graph lookup, plausibility scoring, contradiction detection. |
| 3 | Scenario Registry | Authored test scenarios with CRUD, versioning, tagging, and geometric search via GMS link prediction. |
| 4 | Run State Machine | Formal lifecycle tracking: CREATED -> PREPARING -> EXECUTING -> EVALUATING -> COMPLETED/FAILED/ABORTED. |
| 5 | Replay & Regression | Deterministic replay from snapshots. Failure promotion to regression scenarios. Cross-run semantic drift detection. |
| 6 | Extended Evaluators | Process compliance (holonomy), fault recovery (tension energy), human interaction quality (score_triple). |
| 7 | Risk-Tiered Gates | Per-tier release profiles (advisory / recommendation / action-taking) with regression comparison and trend tracking. |

All agentic features are **off by default** and enabled via `DOEHarnessConfig` flags.

## Configuration

All fields have sensible defaults. A minimal config needs only `markdown_path`:

```yaml
document:
  markdown_path: "report.md"

doe:
  n_runs: 32
  method: sobol

# Enable features as needed
gmsh:
  enable_typed_verdicts: true
  enable_gateway: true
  enable_policy_engine: true
```

See `USER_GUIDE.md` Section 14 for the full configuration reference.

## Failure Taxonomy

GMSH defines 27 structured failure codes across 8 categories:

- **Factual**: unsupported claim, entity missing, relation missing
- **Numeric**: threshold/artifact wrong, phase encoder missing
- **Logic**: rule invalid, holonomy defect
- **Tool**: tool not called, wrong args, hallucinated tool
- **Semantic**: low similarity, weak explanation
- **Policy**: blocked pre-execution, approval required
- **Process**: required tool missing, forbidden tool called, deviation, budget exceeded
- **Recovery/Interaction**: recovery failure, escalation inappropriate, clarification insufficient

Every failure includes `FailureAttribution` linking it to the GMS primitive that detected it.

## Documentation

| Document | Purpose |
|----------|---------|
| `GMSH_TECHNICAL.md` | Full technical reference — data models, algorithms, GMS integration points |
| `USER_GUIDE.md` | Step-by-step tutorials with runnable code examples for all features |
| `TECHNICAL.md` | Base DOE-GMS benchmark reference |

## Design Principles

1. **GMS is the single source of truth.** Every verification calls a GMS method. No heuristics, no bypasses.
2. **Fail-loud.** If GMS cannot verify a claim, the verifier fails rather than silently passing.
3. **Full auditability.** Every decision records the GMS method used and is traced as geometric triples.
4. **Backwards-compatible.** Each phase is independently toggleable. `TypedVerdict` downcasts to `GeometricVerdict`.
5. **Opt-in complexity.** All agentic features default to off. Enable only what you need.
