# GMSH Harness — Quick Reference

All public classes are re-exported from `harness.doe`.
Use `from harness.doe import X` for everything listed below.

## Task Lookup

### Run a benchmark
    from harness.doe import DOEGMSBenchmark, DOEHarnessConfig

### Connect an LLM evaluator
    from harness.doe import LLMEvaluator, AgentEvaluator, make_evaluator

### Inject tool faults
    from harness.doe import ToolGateway, FaultProfile

### Enforce policy rules
    from harness.doe import PolicyEngine, PolicyRule

### Create test scenarios
    from harness.doe import Scenario, ScenarioRegistry

### Track run lifecycle
    from harness.doe import RunManager, RunState, RunRecord

### Replay/regression testing
    from harness.doe import ReplayManifest, RegressionSuite, compare_runs

### Evaluate tool ordering
    from harness.doe import ProcessEvaluator, ProcessClaim

### Evaluate fault recovery
    from harness.doe import RecoveryEvaluator, RecoveryClaim

### Evaluate escalation quality
    from harness.doe import HumanInteractionEvaluator, InteractionClaim

### Release gate checks
    from harness.doe import AuditReporter, RiskTierProfile, TieredReleaseGate

### Define target profiles
    from harness.doe import TargetProfile, IsolationRequirements

### Simulate tool environments
    from harness.doe import EnvironmentSimulator, ToolEndpointSpec, DataFixture, ClockControl

### Oracle-based pass/fail
    from harness.doe import CompositeOracle, ExactOutcomeOracle, TraceOracle, make_oracle

### Classify failure severity
    from harness.doe import SeverityClassifier, SeverityLevel, FailureCode

### Run campaigns
    from harness.doe import CampaignManager, CampaignConfig, CampaignType

### Taxonomy test generation
    from harness.doe import BoundaryGenerator, AdversarialGenerator, taxonomy_generators

### Multi-turn testing
    from harness.doe import MultiTurnAdapter, InteractionScript, TurnSpec

### Safety and isolation
    from harness.doe import SafetyEnforcer, SafetyConfig, SyntheticDataFramework

### Parallel execution
    from harness.doe import ParallelExecutor, ParallelConfig

### Artifact versioning
    from harness.doe import ArtifactRegistry, VersionedArtifact

### Performance profiling
    from harness.doe import PerformanceProfiler, PerformanceReport

### Cost tracking
    from harness.doe import CostTracker, CostSummary

### Agent memory testing
    from harness.doe import StatefulAgentTester, AgentMemoryProbe, MemoryAssessment

### Hallucination detection
    from harness.doe import HallucinationOracle, HallucinationReport

### Confidence calibration
    from harness.doe import CalibrationTester, CalibrationReport

### Multi-agent consensus
    from harness.doe import ConsensusOracle, AgentRole

### Cross-document conflicts
    from harness.doe import CrossDocumentAnalyzer, DocumentConflict

### Entity disambiguation
    from harness.doe import DisambiguationOracle, AmbiguousEntity

### Paraphrase invariance
    from harness.doe import InvarianceTester, InvarianceResult

### Streaming consistency
    from harness.doe import StreamingAdapter, StreamConsistencyOracle

### Live dashboard
    from harness.doe import DashboardServer, DashboardSnapshot

### Browser/UI testing
    from harness.doe import BrowserAdapter, UIScript, BrowserConfig

### Mock agent for testing
    from knowlytix.harness.fixtures import MockHarness, create_mock_evaluator

### Claim verification pipeline
    from harness.doe import ClaimCompiler, VerificationRouter, TypedVerdict

### Typed verdicts and failures
    from harness.doe import FailureCode, FailureAttribution, TypedVerdict

## Generator Names

Valid strings for `DOEHarnessConfig.generators` (empty list = all):

| Group        | Names | Notes |
|--------------|-------|-------|
| Geometric    | `plausibility`, `tension`, `holonomy`, `link_prediction`, `multi_hop_transport`, `phase_comparison`, `conditional`, `procedural` | Default |
| Taxonomy     | `boundary`, `adversarial`, `robustness`, `recovery_test`, `governance` | Default |
| Classic (legacy) | `exact_recall`, `threshold`, `cross_reference`, `contradiction`, `multi_hop`, `counting` | Superseded by geometric; available by name |
| Extended (legacy) | `exact_retrieval`, `negative`, `workflow_compliance` | Superseded by geometric; available by name |
| Enhancement  | `stateful_memory`, `hallucination`, `calibration`, `multi_agent`, `cross_document`, `disambiguation`, `paraphrase_invariance` | Opt-in via config flags |

13 default generators (geometric + taxonomy). 27 total available by name.

## Config Quick Reference

`DOEHarnessConfig` enable flags and what they activate:

| Flag | Module |
|------|--------|
| `enable_typed_verdicts` | Claim-level verification (ClaimCompiler, TypedVerdict) |
| `enable_provenance` | Document provenance chain |
| `enable_tracing` | DOE tracing store (on by default) |
| `enable_gateway` | Tool fault injection (ToolGateway) |
| `enable_policy_engine` | Policy rule enforcement |
| `enable_scenario_registry` | YAML-driven scenario loading |
| `enable_process_evaluation` | Tool-ordering evaluation |
| `enable_recovery_evaluation` | Fault-recovery evaluation |
| `enable_interaction_evaluation` | Human-escalation evaluation |
| `enable_severity_classification` | Failure severity tagging |
| `enable_performance_profiling` | Latency/SLA profiling |
| `enable_cost_tracking` | Token/USD cost tracking |
| `enable_stateful_testing` | Agent memory probe sessions |
| `enable_hallucination_testing` | Geodesic hallucination detection |
| `enable_calibration_testing` | Confidence calibration |
| `enable_multi_agent` | Multi-agent consensus testing |
| `enable_cross_document` | Cross-document conflict detection |
| `enable_disambiguation_testing` | Entity disambiguation |
| `enable_invariance_testing` | Paraphrase invariance |
| `enable_streaming_testing` | Streaming output consistency |
| `enable_live_dashboard` | WebSocket live dashboard |

Key non-boolean config: `semantic_mode` ("off"/"fallback"/"complement"), `risk_tier_profiles_path`, `target_profile_path`, `runs_dir`, `campaigns_dir`.

## External Imports

These are NOT re-exported from `harness.doe`:

```python
from docgms.store import GMSExpertStore
from knowlytix.harness.fixtures import MockHarness, create_mock_evaluator
```

## Docs

- [USER_GUIDE.md](USER_GUIDE.md) -- end-to-end walkthrough
- [API_REFERENCE.md](API_REFERENCE.md) -- full class/method reference
