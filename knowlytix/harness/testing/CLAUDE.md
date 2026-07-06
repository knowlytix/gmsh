# GMSH Harness — Quick Reference

All public classes are re-exported from `knowlytix.harness.testing`.
Use `from knowlytix.harness.testing import X` for everything listed below.

## Task Lookup

### Run a benchmark
    from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig

### Connect an LLM evaluator
    from knowlytix.harness.testing import LLMEvaluator, AgentEvaluator, make_evaluator

### Inject tool faults
    from knowlytix.harness.testing import ToolGateway, FaultProfile

### Enforce policy rules
    from knowlytix.harness.testing import PolicyEngine, PolicyRule

### Create test scenarios
    from knowlytix.harness.testing import Scenario, ScenarioRegistry

### Track run lifecycle
    from knowlytix.harness.testing import RunManager, RunState, RunRecord

### Replay/regression testing
    from knowlytix.harness.testing import ReplayManifest, RegressionSuite, compare_runs

### Evaluate tool ordering
    from knowlytix.harness.testing import ProcessEvaluator, ProcessClaim

### Evaluate fault recovery
    from knowlytix.harness.testing import RecoveryEvaluator, RecoveryClaim

### Evaluate escalation quality
    from knowlytix.harness.testing import HumanInteractionEvaluator, InteractionClaim

### Release gate checks
    from knowlytix.harness.testing import AuditReporter, RiskTierProfile, TieredReleaseGate

### Define target profiles
    from knowlytix.harness.testing import TargetProfile, IsolationRequirements

### Simulate tool environments
    from knowlytix.harness.testing import EnvironmentSimulator, ToolEndpointSpec, DataFixture, ClockControl

### Oracle-based pass/fail
    from knowlytix.harness.testing import CompositeOracle, ExactOutcomeOracle, TraceOracle, make_oracle

### Classify failure severity
    from knowlytix.harness.testing import SeverityClassifier, SeverityLevel, FailureCode

### Run campaigns
    from knowlytix.harness.testing import CampaignManager, CampaignConfig, CampaignType

### Taxonomy test generation
    from knowlytix.harness.testing import BoundaryGenerator, AdversarialGenerator, taxonomy_generators

### Multi-turn testing
    from knowlytix.harness.testing import MultiTurnAdapter, InteractionScript, TurnSpec

### Safety and isolation
    from knowlytix.harness.testing import SafetyEnforcer, SafetyConfig, SyntheticDataFramework

### Parallel execution
    from knowlytix.harness.testing import ParallelExecutor, ParallelConfig

### Artifact versioning
    from knowlytix.harness.testing import ArtifactRegistry, VersionedArtifact

### Performance profiling
    from knowlytix.harness.testing import PerformanceProfiler, PerformanceReport

### Cost tracking
    from knowlytix.harness.testing import CostTracker, CostSummary

### Agent memory testing
    from knowlytix.harness.testing import StatefulAgentTester, AgentMemoryProbe, MemoryAssessment

### Hallucination detection
    from knowlytix.harness.testing import HallucinationOracle, HallucinationGenerator

### Confidence calibration
    from knowlytix.harness.testing import CalibrationTester, CalibrationReport

### Multi-agent consensus
    from knowlytix.harness.testing import ConsensusOracle, AgentRole

### Cross-document conflicts
    from knowlytix.harness.testing import CrossDocumentAnalyzer, DocumentConflict

### Entity disambiguation
    from knowlytix.harness.testing import DisambiguationOracle, AmbiguousEntity

### Paraphrase invariance
    from knowlytix.harness.testing import InvarianceTester, InvarianceResult

### Streaming consistency
    from knowlytix.harness.testing import StreamingAdapter, StreamConsistencyOracle

### Live dashboard
    from knowlytix.harness.testing import DashboardServer, DashboardSnapshot

### Browser/UI testing
    from knowlytix.harness.testing import BrowserAdapter, UIScript, BrowserConfig

### Mock agent for testing
    from knowlytix.harness.fixtures import MockHarness, create_mock_evaluator

### Claim verification pipeline
    from knowlytix.harness.testing import ClaimCompiler, VerificationRouter, TypedVerdict

### Answer-driven claim extraction (precision)
    from knowlytix.harness.testing import GeometricClaimExtractor

### Typed verdicts and failures
    from knowlytix.harness.testing import FailureCode, FailureAttribution, TypedVerdict

### Answer completeness (recall axis)
    from knowlytix.harness.testing import CompletenessEvaluator, CompletenessReport

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

These are NOT re-exported from `knowlytix.harness.testing`:

```python
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.harness.fixtures import MockHarness, create_mock_evaluator
```

## Docs

- [USER_GUIDE.md](USER_GUIDE.md) -- end-to-end walkthrough
- [API_GUIDE.md](API_GUIDE.md) -- full class/method reference
