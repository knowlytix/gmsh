# GMSH API Reference

Complete API reference for all public classes, dataclasses, enumerations and
factory functions exported by the GMSH package.

> **Version** 0.5.0 &mdash; generated 2026-04-11

---

## Table of Contents

| # | Section | Key Classes |
|---|---------|-------------|
| 1 | [Core Store](#1-core-store) | `GMSExpertStore` |
| 2 | [Document Ingestion](#2-document-ingestion) | `IngestResult`, `ingest_document` |
| 3 | [Runtime Learning](#3-runtime-learning) | `RuntimeLearner`, `LearnResult` |
| 4 | [Verification Primitives](#4-verification-primitives) | `verify_answer`, `ContradictionResult`, `HolonomyResult` |
| 5 | [Query Engine](#5-query-engine) | `QueryEngine`, `QueryResult` |
| 6 | [Document Graph](#6-document-graph) | `DocumentGraph`, `ENMKey`, `PhaseEncoder` |
| 7 | [Benchmark Orchestration](#7-benchmark-orchestration) | `DOEGMSBenchmark`, `DOEHarnessConfig` |
| 8 | [Geometric Generators](#8-geometric-generators) | 6 generators + `geometric_generators()` |
| 9 | [Taxonomy Generators](#9-taxonomy-generators) | `BoundaryGenerator`, `AdversarialGenerator`, `RobustnessGenerator` |
| 10 | [Geometric Judge](#10-geometric-judge) | `GMSJudge`, `GeometricVerdict` |
| 11 | [Evaluators & Adapters](#11-evaluators--adapters) | `LLMEvaluator`, `AgentEvaluator`, `SUTAdapter`, `LLMAdapter` |
| 12 | [Claim Verification Pipeline](#12-claim-verification-pipeline) | `ClaimCompiler`, `VerificationRouter`, `TypedVerdict` |
| 13 | [Failure Attribution](#13-failure-attribution) | `FailureCode`, `FailureAttribution`, `FailureSeverity` |
| 14 | [Semantic Verification](#14-semantic-verification) | `SemanticVerifier`, `RubricDimension`, `SemanticScore` |
| 15 | [Tool Gateway](#15-tool-gateway) | `ToolGateway`, `FaultProfile`, `GatewayDecision` |
| 16 | [Policy Engine](#16-policy-engine) | `PolicyEngine`, `PolicyRule`, `PolicyDecision` |
| 17 | [Scenario Registry](#17-scenario-registry) | `Scenario`, `ScenarioRegistry`, `StopCondition` |
| 18 | [Run Management](#18-run-management) | `RunManager`, `RunRecord`, `RunState` |
| 19 | [Replay & Regression](#19-replay--regression) | `ReplayManifest`, `RegressionSuite`, `RunComparison` |
| 20 | [Audit & Release Gates](#20-audit--release-gates) | `AuditReporter`, `ReleaseGate`, `TieredReleaseGate`, `RiskTierProfile` |
| 21 | [Snapshots](#21-snapshots) | `SnapshotManager`, `GMSSnapshot` |
| 22 | [Provenance](#22-provenance) | `ProvenanceRegistry`, `DocumentRecord`, `SectionSpan` |
| 23 | [Truth Store](#23-truth-store) | `TruthStore`, `EntityTruth`, `RelationTruth`, `ArtifactTruth` |
| 24 | [Target Profiles](#24-target-profiles) | `TargetProfile`, `IsolationRequirements` |
| 25 | [Severity Classification](#25-severity-classification) | `SeverityClassifier`, `SeverityLevel`, `SeverityAssessment` |
| 26 | [Environment Simulator](#26-environment-simulator) | `EnvironmentSimulator`, `ToolEndpointSpec`, `DataFixture` |
| 27 | [Oracle Framework](#27-oracle-framework) | `CompositeOracle`, `ExactOutcomeOracle`, `TraceOracle` |
| 28 | [Campaign Management](#28-campaign-management) | `CampaignManager`, `CampaignConfig`, `CampaignResult` |
| 29 | [Extended Evaluators](#29-extended-evaluators) | `ProcessEvaluator`, `RecoveryEvaluator`, `HumanInteractionEvaluator` |
| 30 | [Hallucination Testing](#30-hallucination-testing) | `HallucinationOracle`, `HallucinationReport` |
| 31 | [Calibration Testing](#31-calibration-testing) | `CalibrationTester`, `CalibrationReport` |
| 32 | [Stateful Agent Testing](#32-stateful-agent-testing) | `StatefulAgentTester`, `MemoryAssessment` |
| 33 | [Invariance Testing](#33-invariance-testing) | `InvarianceTester`, `InvarianceResult` |
| 34 | [Streaming Testing](#34-streaming-testing) | `StreamingAdapter`, `StreamingTrace` |
| 35 | [Multi-Agent Testing](#35-multi-agent-testing) | `ConsensusOracle`, `InteractionResult` |
| 36 | [Cross-Document Analysis](#36-cross-document-analysis) | `CrossDocumentAnalyzer`, `DocumentConflict` |
| 37 | [Disambiguation Testing](#37-disambiguation-testing) | `DisambiguationOracle`, `AmbiguousEntity` |
| 38 | [Multi-Turn Interactions](#38-multi-turn-interactions) | `MultiTurnAdapter`, `InteractionScript`, `ScriptResult` |
| 39 | [Safety Framework](#39-safety-framework) | `SafetyEnforcer`, `SafetyConfig`, `SyntheticDataFramework` |
| 40 | [Parallel Execution](#40-parallel-execution) | `ParallelExecutor`, `ParallelConfig` |
| 41 | [Performance Profiling](#41-performance-profiling) | `PerformanceProfiler`, `PerformanceReport` |
| 42 | [Cost Tracking](#42-cost-tracking) | `CostTracker`, `CostSummary` |
| 43 | [Versioning](#43-versioning) | `ArtifactRegistry`, `VersionedArtifact` |
| 44 | [Tracing](#44-tracing) | `DOETracingGMS` |
| 45 | [Live Dashboard](#45-live-dashboard) | `DashboardServer`, `DashboardSnapshot` |
| 46 | [Browser Adapter](#46-browser-adapter) | `BrowserAdapter`, `UIScript`, `BrowserConfig` |
| 47 | [Mock Harness](#47-mock-harness) | `MockHarness`, `FullMockHarness` |

---

## Quick Import Reference

Use this table to find the right import for any class or function. Everything below (except where noted) is re-exported from `harness.doe`.

### Table 1: By Module

#### Core

| Name | Source Module | Import |
|------|--------------|--------|
| `DOEHarnessConfig` | `config` | `from knowlytix.harness.doe import DOEHarnessConfig` |
| `DOEGMSBenchmark` | `benchmark` | `from knowlytix.harness.doe import DOEGMSBenchmark` |
| `generate_base_questions` | `bridge` | `from knowlytix.harness.doe import generate_base_questions` |
| `adapt_question` | `bridge` | `from knowlytix.harness.doe import adapt_question` |
| `scenario_to_fsb_question` | `bridge` | `from knowlytix.harness.doe import scenario_to_fsb_question` |
| `LLMEvaluator` | `evaluators` | `from knowlytix.harness.doe import LLMEvaluator` |
| `AgentEvaluator` | `evaluators` | `from knowlytix.harness.doe import AgentEvaluator` |
| `make_evaluator` | `evaluators` | `from knowlytix.harness.doe import make_evaluator` |
| `PlausibilityGenerator` | `generators` | `from knowlytix.harness.doe import PlausibilityGenerator` |
| `TensionGenerator` | `generators` | `from knowlytix.harness.doe import TensionGenerator` |
| `HolonomyGenerator` | `generators` | `from knowlytix.harness.doe import HolonomyGenerator` |
| `LinkPredictionGenerator` | `generators` | `from knowlytix.harness.doe import LinkPredictionGenerator` |
| `MultiHopTransportGenerator` | `generators` | `from knowlytix.harness.doe import MultiHopTransportGenerator` |
| `PhaseComparisonGenerator` | `generators` | `from knowlytix.harness.doe import PhaseComparisonGenerator` |
| `geometric_generators` | `generators` | `from knowlytix.harness.doe import geometric_generators` |
| `GMSJudge` | `judge` | `from knowlytix.harness.doe import GMSJudge` |
| `GeometricVerdict` | `judge` | `from knowlytix.harness.doe import GeometricVerdict` |
| `create_judge` | `judge` | `from knowlytix.harness.doe import create_judge` |
| `DOETracingGMS` | `tracing` | `from knowlytix.harness.doe import DOETracingGMS` |
| `generate_report` | `dashboard` | `from knowlytix.harness.doe import generate_report` |
| `generate_comparison_report` | `dashboard` | `from knowlytix.harness.doe import generate_comparison_report` |
| `generate_enhanced_report` | `dashboard` | `from knowlytix.harness.doe import generate_enhanced_report` |

#### Phase 1: Claim Verification

| Name | Source Module | Import |
|------|--------------|--------|
| `ClaimBase` | `claims` | `from knowlytix.harness.doe import ClaimBase` |
| `ClaimProvenance` | `claims` | `from knowlytix.harness.doe import ClaimProvenance` |
| `FactClaim` | `claims` | `from knowlytix.harness.doe import FactClaim` |
| `ThresholdClaim` | `claims` | `from knowlytix.harness.doe import ThresholdClaim` |
| `RuleClaim` | `claims` | `from knowlytix.harness.doe import RuleClaim` |
| `ArtifactClaim` | `claims` | `from knowlytix.harness.doe import ArtifactClaim` |
| `ToolUseClaim` | `claims` | `from knowlytix.harness.doe import ToolUseClaim` |
| `SemanticClaim` | `claims` | `from knowlytix.harness.doe import SemanticClaim` |
| `ClaimCompiler` | `claims` | `from knowlytix.harness.doe import ClaimCompiler` |
| `SubVerdict` | `verifiers` | `from knowlytix.harness.doe import SubVerdict` |
| `GraphVerifier` | `verifiers` | `from knowlytix.harness.doe import GraphVerifier` |
| `NumericVerifier` | `verifiers` | `from knowlytix.harness.doe import NumericVerifier` |
| `ArtifactVerifier` | `verifiers` | `from knowlytix.harness.doe import ArtifactVerifier` |
| `LogicVerifier` | `verifiers` | `from knowlytix.harness.doe import LogicVerifier` |
| `PolicyVerifier` | `verifiers` | `from knowlytix.harness.doe import PolicyVerifier` |
| `VerificationRouter` | `verifiers` | `from knowlytix.harness.doe import VerificationRouter` |
| `SemanticVerifier` | `semantic` | `from knowlytix.harness.doe import SemanticVerifier` |
| `GMSEvidence` | `semantic` | `from knowlytix.harness.doe import GMSEvidence` |
| `SemanticScore` | `semantic` | `from knowlytix.harness.doe import SemanticScore` |
| `RubricDimension` | `semantic` | `from knowlytix.harness.doe import RubricDimension` |
| `DEFAULT_RUBRIC` | `semantic` | `from knowlytix.harness.doe import DEFAULT_RUBRIC` |
| `FailureCode` | `failures` | `from knowlytix.harness.doe import FailureCode` |
| `FailureSeverity` | `failures` | `from knowlytix.harness.doe import FailureSeverity` |
| `FailureAttribution` | `failures` | `from knowlytix.harness.doe import FailureAttribution` |
| `TypedVerdict` | `verdict` | `from knowlytix.harness.doe import TypedVerdict` |

#### Phase 2: Provenance & Truth

| Name | Source Module | Import |
|------|--------------|--------|
| `DocumentRecord` | `provenance` | `from knowlytix.harness.doe import DocumentRecord` |
| `SectionSpan` | `provenance` | `from knowlytix.harness.doe import SectionSpan` |
| `NormalizedDocument` | `provenance` | `from knowlytix.harness.doe import NormalizedDocument` |
| `ProvenanceRegistry` | `provenance` | `from knowlytix.harness.doe import ProvenanceRegistry` |
| `TruthBase` | `truth` | `from knowlytix.harness.doe import TruthBase` |
| `EntityTruth` | `truth` | `from knowlytix.harness.doe import EntityTruth` |
| `RelationTruth` | `truth` | `from knowlytix.harness.doe import RelationTruth` |
| `ThresholdTruth` | `truth` | `from knowlytix.harness.doe import ThresholdTruth` |
| `ArtifactTruth` | `truth` | `from knowlytix.harness.doe import ArtifactTruth` |
| `TruthStore` | `truth` | `from knowlytix.harness.doe import TruthStore` |

#### Phase 3: SUT Adapters & Test Generators

| Name | Source Module | Import |
|------|--------------|--------|
| `SUTAdapter` | `adapters` | `from knowlytix.harness.doe import SUTAdapter` |
| `SUTTrace` | `adapters` | `from knowlytix.harness.doe import SUTTrace` |
| `ToolCallRecord` | `adapters` | `from knowlytix.harness.doe import ToolCallRecord` |
| `LLMAdapter` | `adapters` | `from knowlytix.harness.doe import LLMAdapter` |
| `AgentAdapter` | `adapters` | `from knowlytix.harness.doe import AgentAdapter` |
| `RAGAdapter` | `adapters` | `from knowlytix.harness.doe import RAGAdapter` |
| `make_adapter` | `adapters` | `from knowlytix.harness.doe import make_adapter` |
| `ExactRetrievalGenerator` | `test_generators` | `from knowlytix.harness.doe import ExactRetrievalGenerator` |
| `NegativeGenerator` | `test_generators` | `from knowlytix.harness.doe import NegativeGenerator` |
| `WorkflowComplianceGenerator` | `test_generators` | `from knowlytix.harness.doe import WorkflowComplianceGenerator` |
| `extended_generators` | `test_generators` | `from knowlytix.harness.doe import extended_generators` |

#### Phase 4: Versioning & Snapshots

| Name | Source Module | Import |
|------|--------------|--------|
| `GMSSnapshot` | `snapshots` | `from knowlytix.harness.doe import GMSSnapshot` |
| `SnapshotManager` | `snapshots` | `from knowlytix.harness.doe import SnapshotManager` |

#### Phase 5: Audit & Reporting

| Name | Source Module | Import |
|------|--------------|--------|
| `AuditPacket` | `audit` | `from knowlytix.harness.doe import AuditPacket` |
| `ReleaseGate` | `audit` | `from knowlytix.harness.doe import ReleaseGate` |
| `AuditReporter` | `audit` | `from knowlytix.harness.doe import AuditReporter` |
| `RiskTierProfile` | `audit` | `from knowlytix.harness.doe import RiskTierProfile` |
| `TieredReleaseGate` | `audit` | `from knowlytix.harness.doe import TieredReleaseGate` |
| `RunTrend` | `audit` | `from knowlytix.harness.doe import RunTrend` |
| `generate_report` | `dashboard` | `from knowlytix.harness.doe import generate_report` |
| `generate_comparison_report` | `dashboard` | `from knowlytix.harness.doe import generate_comparison_report` |
| `generate_enhanced_report` | `dashboard` | `from knowlytix.harness.doe import generate_enhanced_report` |

#### Agentic Harness

| Name | Source Module | Import |
|------|--------------|--------|
| `ToolGateway` | `gateway` | `from knowlytix.harness.doe import ToolGateway` |
| `FaultProfile` | `gateway` | `from knowlytix.harness.doe import FaultProfile` |
| `GatewayDecision` | `gateway` | `from knowlytix.harness.doe import GatewayDecision` |
| `GatewayTranscript` | `gateway` | `from knowlytix.harness.doe import GatewayTranscript` |
| `PolicyEngine` | `policy` | `from knowlytix.harness.doe import PolicyEngine` |
| `PolicyRule` | `policy` | `from knowlytix.harness.doe import PolicyRule` |
| `PolicyDecision` | `policy` | `from knowlytix.harness.doe import PolicyDecision` |
| `Scenario` | `scenarios` | `from knowlytix.harness.doe import Scenario` |
| `ScenarioRegistry` | `scenarios` | `from knowlytix.harness.doe import ScenarioRegistry` |
| `RunState` | `runs` | `from knowlytix.harness.doe import RunState` |
| `RunRecord` | `runs` | `from knowlytix.harness.doe import RunRecord` |
| `RunManager` | `runs` | `from knowlytix.harness.doe import RunManager` |
| `ReplayManifest` | `replay` | `from knowlytix.harness.doe import ReplayManifest` |
| `RunComparison` | `replay` | `from knowlytix.harness.doe import RunComparison` |
| `RegressionSuite` | `replay` | `from knowlytix.harness.doe import RegressionSuite` |
| `compare_runs` | `replay` | `from knowlytix.harness.doe import compare_runs` |
| `replay_run` | `replay` | `from knowlytix.harness.doe import replay_run` |
| `ProcessClaim` | `evaluators_extended` | `from knowlytix.harness.doe import ProcessClaim` |
| `RecoveryClaim` | `evaluators_extended` | `from knowlytix.harness.doe import RecoveryClaim` |
| `InteractionClaim` | `evaluators_extended` | `from knowlytix.harness.doe import InteractionClaim` |
| `ProcessEvaluator` | `evaluators_extended` | `from knowlytix.harness.doe import ProcessEvaluator` |
| `RecoveryEvaluator` | `evaluators_extended` | `from knowlytix.harness.doe import RecoveryEvaluator` |
| `HumanInteractionEvaluator` | `evaluators_extended` | `from knowlytix.harness.doe import HumanInteractionEvaluator` |

#### Black-Box Phase 1: Foundation

| Name | Source Module | Import |
|------|--------------|--------|
| `TargetProfile` | `target` | `from knowlytix.harness.doe import TargetProfile` |
| `IsolationRequirements` | `target` | `from knowlytix.harness.doe import IsolationRequirements` |
| `SeverityLevel` | `severity` | `from knowlytix.harness.doe import SeverityLevel` |
| `SeverityAssessment` | `severity` | `from knowlytix.harness.doe import SeverityAssessment` |
| `SeverityClassifier` | `severity` | `from knowlytix.harness.doe import SeverityClassifier` |
| `EnvironmentSimulator` | `environment` | `from knowlytix.harness.doe import EnvironmentSimulator` |
| `ToolEndpointSpec` | `environment` | `from knowlytix.harness.doe import ToolEndpointSpec` |
| `ClockControl` | `environment` | `from knowlytix.harness.doe import ClockControl` |
| `DataFixture` | `environment` | `from knowlytix.harness.doe import DataFixture` |
| `StateTransition` | `environment` | `from knowlytix.harness.doe import StateTransition` |
| `FailureMode` | `environment` | `from knowlytix.harness.doe import FailureMode` |
| `ScheduledEvent` | `environment` | `from knowlytix.harness.doe import ScheduledEvent` |
| `OracleVerdict` | `oracles` | `from knowlytix.harness.doe import OracleVerdict` |
| `OracleBase` | `oracles` | `from knowlytix.harness.doe import OracleBase` |
| `ExactOutcomeOracle` | `oracles` | `from knowlytix.harness.doe import ExactOutcomeOracle` |
| `AcceptableSetOracle` | `oracles` | `from knowlytix.harness.doe import AcceptableSetOracle` |
| `ConstraintOracle` | `oracles` | `from knowlytix.harness.doe import ConstraintOracle` |
| `TraceOracle` | `oracles` | `from knowlytix.harness.doe import TraceOracle` |
| `SideEffectOracle` | `oracles` | `from knowlytix.harness.doe import SideEffectOracle` |
| `HumanReviewedOracle` | `oracles` | `from knowlytix.harness.doe import HumanReviewedOracle` |
| `CompositeOracle` | `oracles` | `from knowlytix.harness.doe import CompositeOracle` |
| `make_oracle` | `oracles` | `from knowlytix.harness.doe import make_oracle` |
| `CampaignType` | `campaign` | `from knowlytix.harness.doe import CampaignType` |
| `CampaignConfig` | `campaign` | `from knowlytix.harness.doe import CampaignConfig` |
| `CampaignResult` | `campaign` | `from knowlytix.harness.doe import CampaignResult` |
| `CampaignRunSummary` | `campaign` | `from knowlytix.harness.doe import CampaignRunSummary` |
| `CampaignManager` | `campaign` | `from knowlytix.harness.doe import CampaignManager` |

#### Black-Box Phase 2: Coverage

| Name | Source Module | Import |
|------|--------------|--------|
| `BoundaryGenerator` | `generators_taxonomy` | `from knowlytix.harness.doe import BoundaryGenerator` |
| `AdversarialGenerator` | `generators_taxonomy` | `from knowlytix.harness.doe import AdversarialGenerator` |
| `RobustnessGenerator` | `generators_taxonomy` | `from knowlytix.harness.doe import RobustnessGenerator` |
| `RecoveryTestGenerator` | `generators_taxonomy` | `from knowlytix.harness.doe import RecoveryTestGenerator` |
| `GovernanceTestGenerator` | `generators_taxonomy` | `from knowlytix.harness.doe import GovernanceTestGenerator` |
| `taxonomy_generators` | `generators_taxonomy` | `from knowlytix.harness.doe import taxonomy_generators` |
| `SafetyConfig` | `safety` | `from knowlytix.harness.doe import SafetyConfig` |
| `SafetyEnforcer` | `safety` | `from knowlytix.harness.doe import SafetyEnforcer` |
| `SyntheticDataFramework` | `safety` | `from knowlytix.harness.doe import SyntheticDataFramework` |
| `TurnSpec` | `multiturn` | `from knowlytix.harness.doe import TurnSpec` |
| `InteractionScript` | `multiturn` | `from knowlytix.harness.doe import InteractionScript` |
| `TurnResult` | `multiturn` | `from knowlytix.harness.doe import TurnResult` |
| `ScriptResult` | `multiturn` | `from knowlytix.harness.doe import ScriptResult` |
| `MultiTurnAdapter` | `multiturn` | `from knowlytix.harness.doe import MultiTurnAdapter` |
| `MultiTurnScenarioRunner` | `multiturn` | `from knowlytix.harness.doe import MultiTurnScenarioRunner` |

#### Black-Box Phase 3: Scale & Polish

| Name | Source Module | Import |
|------|--------------|--------|
| `ParallelConfig` | `parallel` | `from knowlytix.harness.doe import ParallelConfig` |
| `ParallelExecutor` | `parallel` | `from knowlytix.harness.doe import ParallelExecutor` |
| `merge_parallel_results` | `parallel` | `from knowlytix.harness.doe import merge_parallel_results` |
| `VersionedArtifact` | `versioning` | `from knowlytix.harness.doe import VersionedArtifact` |
| `ArtifactRegistry` | `versioning` | `from knowlytix.harness.doe import ArtifactRegistry` |
| `FaultProfileRegistry` | `versioning` | `from knowlytix.harness.doe import FaultProfileRegistry` |
| `PolicyRuleRegistry` | `versioning` | `from knowlytix.harness.doe import PolicyRuleRegistry` |
| `FixtureRegistry` | `versioning` | `from knowlytix.harness.doe import FixtureRegistry` |
| `UIStep` | `browser_adapter` | `from knowlytix.harness.doe import UIStep` |
| `UIScript` | `browser_adapter` | `from knowlytix.harness.doe import UIScript` |
| `UIStepResult` | `browser_adapter` | `from knowlytix.harness.doe import UIStepResult` |
| `UIScriptResult` | `browser_adapter` | `from knowlytix.harness.doe import UIScriptResult` |
| `BrowserConfig` | `browser_adapter` | `from knowlytix.harness.doe import BrowserConfig` |
| `BrowserAdapter` | `browser_adapter` | `from knowlytix.harness.doe import BrowserAdapter` |
| `StopCondition` | `scenarios` | `from knowlytix.harness.doe import StopCondition` |

#### GMS-Native Phase 1: Foundation

| Name | Source Module | Import |
|------|--------------|--------|
| `PerformanceRecord` | `profiling` | `from knowlytix.harness.doe import PerformanceRecord` |
| `PerformanceReport` | `profiling` | `from knowlytix.harness.doe import PerformanceReport` |
| `PerformanceProfiler` | `profiling` | `from knowlytix.harness.doe import PerformanceProfiler` |
| `CostRecord` | `cost_tracker` | `from knowlytix.harness.doe import CostRecord` |
| `CostSummary` | `cost_tracker` | `from knowlytix.harness.doe import CostSummary` |
| `CostTracker` | `cost_tracker` | `from knowlytix.harness.doe import CostTracker` |

#### GMS-Native Phase 2: Agent Memory

| Name | Source Module | Import |
|------|--------------|--------|
| `AgentMemoryProbe` | `stateful` | `from knowlytix.harness.doe import AgentMemoryProbe` |
| `MemoryAssessment` | `stateful` | `from knowlytix.harness.doe import MemoryAssessment` |
| `StatefulAgentTester` | `stateful` | `from knowlytix.harness.doe import StatefulAgentTester` |
| `StatefulAgentGenerator` | `stateful` | `from knowlytix.harness.doe import StatefulAgentGenerator` |
| `MemoryRecallOracle` | `stateful` | `from knowlytix.harness.doe import MemoryRecallOracle` |
| `HallucinationAssessment` | `hallucination` | `from knowlytix.harness.doe import HallucinationAssessment` |
| `HallucinationReport` | `hallucination` | `from knowlytix.harness.doe import HallucinationReport` |
| `HallucinationGenerator` | `hallucination` | `from knowlytix.harness.doe import HallucinationGenerator` |
| `HallucinationOracle` | `hallucination` | `from knowlytix.harness.doe import HallucinationOracle` |
| `CalibrationPoint` | `calibration` | `from knowlytix.harness.doe import CalibrationPoint` |
| `CalibrationReport` | `calibration` | `from knowlytix.harness.doe import CalibrationReport` |
| `CalibrationTester` | `calibration` | `from knowlytix.harness.doe import CalibrationTester` |
| `CalibrationGenerator` | `calibration` | `from knowlytix.harness.doe import CalibrationGenerator` |

#### GMS-Native Phase 3: Multi-Source

| Name | Source Module | Import |
|------|--------------|--------|
| `AgentRole` | `multi_agent` | `from knowlytix.harness.doe import AgentRole` |
| `InteractionResult` | `multi_agent` | `from knowlytix.harness.doe import InteractionResult` |
| `ConsensusOracle` | `multi_agent` | `from knowlytix.harness.doe import ConsensusOracle` |
| `MultiAgentGenerator` | `multi_agent` | `from knowlytix.harness.doe import MultiAgentGenerator` |
| `DocumentConflict` | `cross_document` | `from knowlytix.harness.doe import DocumentConflict` |
| `CrossDocumentAnalyzer` | `cross_document` | `from knowlytix.harness.doe import CrossDocumentAnalyzer` |
| `CrossDocumentGenerator` | `cross_document` | `from knowlytix.harness.doe import CrossDocumentGenerator` |
| `AmbiguousEntity` | `disambiguation` | `from knowlytix.harness.doe import AmbiguousEntity` |
| `DisambiguationOracle` | `disambiguation` | `from knowlytix.harness.doe import DisambiguationOracle` |
| `DisambiguationGenerator` | `disambiguation` | `from knowlytix.harness.doe import DisambiguationGenerator` |

#### GMS-Native Phase 4: Robustness

| Name | Source Module | Import |
|------|--------------|--------|
| `InvarianceResult` | `invariance` | `from knowlytix.harness.doe import InvarianceResult` |
| `InvarianceTester` | `invariance` | `from knowlytix.harness.doe import InvarianceTester` |
| `InvarianceGenerator` | `invariance` | `from knowlytix.harness.doe import InvarianceGenerator` |
| `StreamChunk` | `streaming` | `from knowlytix.harness.doe import StreamChunk` |
| `StreamingTrace` | `streaming` | `from knowlytix.harness.doe import StreamingTrace` |
| `StreamingAdapter` | `streaming` | `from knowlytix.harness.doe import StreamingAdapter` |
| `StreamConsistencyOracle` | `streaming` | `from knowlytix.harness.doe import StreamConsistencyOracle` |

#### GMS-Native Phase 5: Observability

| Name | Source Module | Import |
|------|--------------|--------|
| `DashboardSnapshot` | `live_dashboard` | `from knowlytix.harness.doe import DashboardSnapshot` |
| `DashboardServer` | `live_dashboard` | `from knowlytix.harness.doe import DashboardServer` |

#### External

| Name | Source Module | Import |
|------|--------------|--------|
| `GMSExpertStore` | `docgms.store` | `from knowlytix.doc.store import GMSExpertStore` |
| `MockHarness` | `knowlytix.harness.fixtures` | `from knowlytix.harness.fixtures import MockHarness` |
| `create_mock_evaluator` | `knowlytix.harness.fixtures` | `from knowlytix.harness.fixtures import create_mock_evaluator` |

### Table 2: Alphabetical Index

| Name | Kind | Import |
|------|------|--------|
| `AcceptableSetOracle` | class | `from knowlytix.harness.doe import AcceptableSetOracle` |
| `adapt_question` | function | `from knowlytix.harness.doe import adapt_question` |
| `AdversarialGenerator` | class | `from knowlytix.harness.doe import AdversarialGenerator` |
| `AgentAdapter` | class | `from knowlytix.harness.doe import AgentAdapter` |
| `AgentEvaluator` | class | `from knowlytix.harness.doe import AgentEvaluator` |
| `AgentMemoryProbe` | class | `from knowlytix.harness.doe import AgentMemoryProbe` |
| `AgentRole` | class | `from knowlytix.harness.doe import AgentRole` |
| `AmbiguousEntity` | class | `from knowlytix.harness.doe import AmbiguousEntity` |
| `ArtifactClaim` | class | `from knowlytix.harness.doe import ArtifactClaim` |
| `ArtifactRegistry` | class | `from knowlytix.harness.doe import ArtifactRegistry` |
| `ArtifactTruth` | class | `from knowlytix.harness.doe import ArtifactTruth` |
| `ArtifactVerifier` | class | `from knowlytix.harness.doe import ArtifactVerifier` |
| `AuditPacket` | class | `from knowlytix.harness.doe import AuditPacket` |
| `AuditReporter` | class | `from knowlytix.harness.doe import AuditReporter` |
| `BoundaryGenerator` | class | `from knowlytix.harness.doe import BoundaryGenerator` |
| `BrowserAdapter` | class | `from knowlytix.harness.doe import BrowserAdapter` |
| `BrowserConfig` | class | `from knowlytix.harness.doe import BrowserConfig` |
| `CalibrationGenerator` | class | `from knowlytix.harness.doe import CalibrationGenerator` |
| `CalibrationPoint` | class | `from knowlytix.harness.doe import CalibrationPoint` |
| `CalibrationReport` | class | `from knowlytix.harness.doe import CalibrationReport` |
| `CalibrationTester` | class | `from knowlytix.harness.doe import CalibrationTester` |
| `CampaignConfig` | class | `from knowlytix.harness.doe import CampaignConfig` |
| `CampaignManager` | class | `from knowlytix.harness.doe import CampaignManager` |
| `CampaignResult` | class | `from knowlytix.harness.doe import CampaignResult` |
| `CampaignRunSummary` | class | `from knowlytix.harness.doe import CampaignRunSummary` |
| `CampaignType` | enum | `from knowlytix.harness.doe import CampaignType` |
| `ClaimBase` | class | `from knowlytix.harness.doe import ClaimBase` |
| `ClaimCompiler` | class | `from knowlytix.harness.doe import ClaimCompiler` |
| `ClaimProvenance` | class | `from knowlytix.harness.doe import ClaimProvenance` |
| `ClockControl` | class | `from knowlytix.harness.doe import ClockControl` |
| `compare_runs` | function | `from knowlytix.harness.doe import compare_runs` |
| `CompositeOracle` | class | `from knowlytix.harness.doe import CompositeOracle` |
| `ConsensusOracle` | class | `from knowlytix.harness.doe import ConsensusOracle` |
| `ConstraintOracle` | class | `from knowlytix.harness.doe import ConstraintOracle` |
| `CostRecord` | class | `from knowlytix.harness.doe import CostRecord` |
| `CostSummary` | class | `from knowlytix.harness.doe import CostSummary` |
| `CostTracker` | class | `from knowlytix.harness.doe import CostTracker` |
| `create_judge` | function | `from knowlytix.harness.doe import create_judge` |
| `create_mock_evaluator` | function | `from knowlytix.harness.fixtures import create_mock_evaluator` |
| `CrossDocumentAnalyzer` | class | `from knowlytix.harness.doe import CrossDocumentAnalyzer` |
| `CrossDocumentGenerator` | class | `from knowlytix.harness.doe import CrossDocumentGenerator` |
| `DashboardServer` | class | `from knowlytix.harness.doe import DashboardServer` |
| `DashboardSnapshot` | class | `from knowlytix.harness.doe import DashboardSnapshot` |
| `DataFixture` | class | `from knowlytix.harness.doe import DataFixture` |
| `DEFAULT_RUBRIC` | constant | `from knowlytix.harness.doe import DEFAULT_RUBRIC` |
| `DisambiguationGenerator` | class | `from knowlytix.harness.doe import DisambiguationGenerator` |
| `DisambiguationOracle` | class | `from knowlytix.harness.doe import DisambiguationOracle` |
| `DocumentConflict` | class | `from knowlytix.harness.doe import DocumentConflict` |
| `DocumentRecord` | class | `from knowlytix.harness.doe import DocumentRecord` |
| `DOEGMSBenchmark` | class | `from knowlytix.harness.doe import DOEGMSBenchmark` |
| `DOEHarnessConfig` | class | `from knowlytix.harness.doe import DOEHarnessConfig` |
| `DOETracingGMS` | class | `from knowlytix.harness.doe import DOETracingGMS` |
| `EntityTruth` | class | `from knowlytix.harness.doe import EntityTruth` |
| `EnvironmentSimulator` | class | `from knowlytix.harness.doe import EnvironmentSimulator` |
| `ExactOutcomeOracle` | class | `from knowlytix.harness.doe import ExactOutcomeOracle` |
| `ExactRetrievalGenerator` | class | `from knowlytix.harness.doe import ExactRetrievalGenerator` |
| `extended_generators` | function | `from knowlytix.harness.doe import extended_generators` |
| `FactClaim` | class | `from knowlytix.harness.doe import FactClaim` |
| `FailureAttribution` | class | `from knowlytix.harness.doe import FailureAttribution` |
| `FailureCode` | enum | `from knowlytix.harness.doe import FailureCode` |
| `FailureMode` | class | `from knowlytix.harness.doe import FailureMode` |
| `FailureSeverity` | enum | `from knowlytix.harness.doe import FailureSeverity` |
| `FaultProfile` | class | `from knowlytix.harness.doe import FaultProfile` |
| `FaultProfileRegistry` | class | `from knowlytix.harness.doe import FaultProfileRegistry` |
| `FixtureRegistry` | class | `from knowlytix.harness.doe import FixtureRegistry` |
| `GatewayDecision` | class | `from knowlytix.harness.doe import GatewayDecision` |
| `GatewayTranscript` | class | `from knowlytix.harness.doe import GatewayTranscript` |
| `generate_base_questions` | function | `from knowlytix.harness.doe import generate_base_questions` |
| `generate_comparison_report` | function | `from knowlytix.harness.doe import generate_comparison_report` |
| `generate_enhanced_report` | function | `from knowlytix.harness.doe import generate_enhanced_report` |
| `generate_report` | function | `from knowlytix.harness.doe import generate_report` |
| `geometric_generators` | function | `from knowlytix.harness.doe import geometric_generators` |
| `GeometricVerdict` | class | `from knowlytix.harness.doe import GeometricVerdict` |
| `GMSEvidence` | class | `from knowlytix.harness.doe import GMSEvidence` |
| `GMSExpertStore` | class | `from knowlytix.doc.store import GMSExpertStore` |
| `GMSJudge` | class | `from knowlytix.harness.doe import GMSJudge` |
| `GMSSnapshot` | class | `from knowlytix.harness.doe import GMSSnapshot` |
| `GovernanceTestGenerator` | class | `from knowlytix.harness.doe import GovernanceTestGenerator` |
| `GraphVerifier` | class | `from knowlytix.harness.doe import GraphVerifier` |
| `HallucinationAssessment` | class | `from knowlytix.harness.doe import HallucinationAssessment` |
| `HallucinationGenerator` | class | `from knowlytix.harness.doe import HallucinationGenerator` |
| `HallucinationOracle` | class | `from knowlytix.harness.doe import HallucinationOracle` |
| `HallucinationReport` | class | `from knowlytix.harness.doe import HallucinationReport` |
| `HolonomyGenerator` | class | `from knowlytix.harness.doe import HolonomyGenerator` |
| `HumanInteractionEvaluator` | class | `from knowlytix.harness.doe import HumanInteractionEvaluator` |
| `HumanReviewedOracle` | class | `from knowlytix.harness.doe import HumanReviewedOracle` |
| `InteractionClaim` | class | `from knowlytix.harness.doe import InteractionClaim` |
| `InteractionResult` | class | `from knowlytix.harness.doe import InteractionResult` |
| `InteractionScript` | class | `from knowlytix.harness.doe import InteractionScript` |
| `InvarianceGenerator` | class | `from knowlytix.harness.doe import InvarianceGenerator` |
| `InvarianceResult` | class | `from knowlytix.harness.doe import InvarianceResult` |
| `InvarianceTester` | class | `from knowlytix.harness.doe import InvarianceTester` |
| `IsolationRequirements` | class | `from knowlytix.harness.doe import IsolationRequirements` |
| `LinkPredictionGenerator` | class | `from knowlytix.harness.doe import LinkPredictionGenerator` |
| `LLMAdapter` | class | `from knowlytix.harness.doe import LLMAdapter` |
| `LLMEvaluator` | class | `from knowlytix.harness.doe import LLMEvaluator` |
| `LogicVerifier` | class | `from knowlytix.harness.doe import LogicVerifier` |
| `make_adapter` | function | `from knowlytix.harness.doe import make_adapter` |
| `make_evaluator` | function | `from knowlytix.harness.doe import make_evaluator` |
| `make_oracle` | function | `from knowlytix.harness.doe import make_oracle` |
| `MemoryAssessment` | class | `from knowlytix.harness.doe import MemoryAssessment` |
| `MemoryRecallOracle` | class | `from knowlytix.harness.doe import MemoryRecallOracle` |
| `merge_parallel_results` | function | `from knowlytix.harness.doe import merge_parallel_results` |
| `MockHarness` | class | `from knowlytix.harness.fixtures import MockHarness` |
| `MultiAgentGenerator` | class | `from knowlytix.harness.doe import MultiAgentGenerator` |
| `MultiHopTransportGenerator` | class | `from knowlytix.harness.doe import MultiHopTransportGenerator` |
| `MultiTurnAdapter` | class | `from knowlytix.harness.doe import MultiTurnAdapter` |
| `MultiTurnScenarioRunner` | class | `from knowlytix.harness.doe import MultiTurnScenarioRunner` |
| `NegativeGenerator` | class | `from knowlytix.harness.doe import NegativeGenerator` |
| `NormalizedDocument` | class | `from knowlytix.harness.doe import NormalizedDocument` |
| `NumericVerifier` | class | `from knowlytix.harness.doe import NumericVerifier` |
| `OracleBase` | class | `from knowlytix.harness.doe import OracleBase` |
| `OracleVerdict` | class | `from knowlytix.harness.doe import OracleVerdict` |
| `ParallelConfig` | class | `from knowlytix.harness.doe import ParallelConfig` |
| `ParallelExecutor` | class | `from knowlytix.harness.doe import ParallelExecutor` |
| `PerformanceProfiler` | class | `from knowlytix.harness.doe import PerformanceProfiler` |
| `PerformanceRecord` | class | `from knowlytix.harness.doe import PerformanceRecord` |
| `PerformanceReport` | class | `from knowlytix.harness.doe import PerformanceReport` |
| `PhaseComparisonGenerator` | class | `from knowlytix.harness.doe import PhaseComparisonGenerator` |
| `PlausibilityGenerator` | class | `from knowlytix.harness.doe import PlausibilityGenerator` |
| `PolicyDecision` | class | `from knowlytix.harness.doe import PolicyDecision` |
| `PolicyEngine` | class | `from knowlytix.harness.doe import PolicyEngine` |
| `PolicyRule` | class | `from knowlytix.harness.doe import PolicyRule` |
| `PolicyRuleRegistry` | class | `from knowlytix.harness.doe import PolicyRuleRegistry` |
| `PolicyVerifier` | class | `from knowlytix.harness.doe import PolicyVerifier` |
| `ProcessClaim` | class | `from knowlytix.harness.doe import ProcessClaim` |
| `ProcessEvaluator` | class | `from knowlytix.harness.doe import ProcessEvaluator` |
| `ProvenanceRegistry` | class | `from knowlytix.harness.doe import ProvenanceRegistry` |
| `RAGAdapter` | class | `from knowlytix.harness.doe import RAGAdapter` |
| `RecoveryClaim` | class | `from knowlytix.harness.doe import RecoveryClaim` |
| `RecoveryEvaluator` | class | `from knowlytix.harness.doe import RecoveryEvaluator` |
| `RecoveryTestGenerator` | class | `from knowlytix.harness.doe import RecoveryTestGenerator` |
| `RegressionSuite` | class | `from knowlytix.harness.doe import RegressionSuite` |
| `RelationTruth` | class | `from knowlytix.harness.doe import RelationTruth` |
| `ReleaseGate` | class | `from knowlytix.harness.doe import ReleaseGate` |
| `ReplayManifest` | class | `from knowlytix.harness.doe import ReplayManifest` |
| `replay_run` | function | `from knowlytix.harness.doe import replay_run` |
| `RiskTierProfile` | class | `from knowlytix.harness.doe import RiskTierProfile` |
| `RobustnessGenerator` | class | `from knowlytix.harness.doe import RobustnessGenerator` |
| `RubricDimension` | class | `from knowlytix.harness.doe import RubricDimension` |
| `RuleClaim` | class | `from knowlytix.harness.doe import RuleClaim` |
| `RunComparison` | class | `from knowlytix.harness.doe import RunComparison` |
| `RunManager` | class | `from knowlytix.harness.doe import RunManager` |
| `RunRecord` | class | `from knowlytix.harness.doe import RunRecord` |
| `RunState` | enum | `from knowlytix.harness.doe import RunState` |
| `RunTrend` | class | `from knowlytix.harness.doe import RunTrend` |
| `SafetyConfig` | class | `from knowlytix.harness.doe import SafetyConfig` |
| `SafetyEnforcer` | class | `from knowlytix.harness.doe import SafetyEnforcer` |
| `scenario_to_fsb_question` | function | `from knowlytix.harness.doe import scenario_to_fsb_question` |
| `Scenario` | class | `from knowlytix.harness.doe import Scenario` |
| `ScenarioRegistry` | class | `from knowlytix.harness.doe import ScenarioRegistry` |
| `ScheduledEvent` | class | `from knowlytix.harness.doe import ScheduledEvent` |
| `ScriptResult` | class | `from knowlytix.harness.doe import ScriptResult` |
| `SectionSpan` | class | `from knowlytix.harness.doe import SectionSpan` |
| `SemanticClaim` | class | `from knowlytix.harness.doe import SemanticClaim` |
| `SemanticScore` | class | `from knowlytix.harness.doe import SemanticScore` |
| `SemanticVerifier` | class | `from knowlytix.harness.doe import SemanticVerifier` |
| `SeverityAssessment` | class | `from knowlytix.harness.doe import SeverityAssessment` |
| `SeverityClassifier` | class | `from knowlytix.harness.doe import SeverityClassifier` |
| `SeverityLevel` | enum | `from knowlytix.harness.doe import SeverityLevel` |
| `SideEffectOracle` | class | `from knowlytix.harness.doe import SideEffectOracle` |
| `SnapshotManager` | class | `from knowlytix.harness.doe import SnapshotManager` |
| `StateTransition` | class | `from knowlytix.harness.doe import StateTransition` |
| `StatefulAgentGenerator` | class | `from knowlytix.harness.doe import StatefulAgentGenerator` |
| `StatefulAgentTester` | class | `from knowlytix.harness.doe import StatefulAgentTester` |
| `StopCondition` | class | `from knowlytix.harness.doe import StopCondition` |
| `StreamChunk` | class | `from knowlytix.harness.doe import StreamChunk` |
| `StreamConsistencyOracle` | class | `from knowlytix.harness.doe import StreamConsistencyOracle` |
| `StreamingAdapter` | class | `from knowlytix.harness.doe import StreamingAdapter` |
| `StreamingTrace` | class | `from knowlytix.harness.doe import StreamingTrace` |
| `SubVerdict` | class | `from knowlytix.harness.doe import SubVerdict` |
| `SUTAdapter` | class | `from knowlytix.harness.doe import SUTAdapter` |
| `SUTTrace` | class | `from knowlytix.harness.doe import SUTTrace` |
| `SyntheticDataFramework` | class | `from knowlytix.harness.doe import SyntheticDataFramework` |
| `TargetProfile` | class | `from knowlytix.harness.doe import TargetProfile` |
| `taxonomy_generators` | function | `from knowlytix.harness.doe import taxonomy_generators` |
| `TensionGenerator` | class | `from knowlytix.harness.doe import TensionGenerator` |
| `ThresholdClaim` | class | `from knowlytix.harness.doe import ThresholdClaim` |
| `ThresholdTruth` | class | `from knowlytix.harness.doe import ThresholdTruth` |
| `TieredReleaseGate` | class | `from knowlytix.harness.doe import TieredReleaseGate` |
| `ToolCallRecord` | class | `from knowlytix.harness.doe import ToolCallRecord` |
| `ToolEndpointSpec` | class | `from knowlytix.harness.doe import ToolEndpointSpec` |
| `ToolGateway` | class | `from knowlytix.harness.doe import ToolGateway` |
| `ToolUseClaim` | class | `from knowlytix.harness.doe import ToolUseClaim` |
| `TraceOracle` | class | `from knowlytix.harness.doe import TraceOracle` |
| `TruthBase` | class | `from knowlytix.harness.doe import TruthBase` |
| `TruthStore` | class | `from knowlytix.harness.doe import TruthStore` |
| `TurnResult` | class | `from knowlytix.harness.doe import TurnResult` |
| `TurnSpec` | class | `from knowlytix.harness.doe import TurnSpec` |
| `TypedVerdict` | class | `from knowlytix.harness.doe import TypedVerdict` |
| `UIScript` | class | `from knowlytix.harness.doe import UIScript` |
| `UIScriptResult` | class | `from knowlytix.harness.doe import UIScriptResult` |
| `UIStep` | class | `from knowlytix.harness.doe import UIStep` |
| `UIStepResult` | class | `from knowlytix.harness.doe import UIStepResult` |
| `VerificationRouter` | class | `from knowlytix.harness.doe import VerificationRouter` |
| `VersionedArtifact` | class | `from knowlytix.harness.doe import VersionedArtifact` |
| `WorkflowComplianceGenerator` | class | `from knowlytix.harness.doe import WorkflowComplianceGenerator` |

---

## 1. Core Store

**Module:** `docgms.store`

### GMSExpertStore

The central knowledge store. Embeds entities and relations on the unit
hypersphere S^{m-1} using dual v-space (semantic) and u-space (logical)
representations with Cayley rotor rotations.

```python
class GMSExpertStore:
    def __init__(
        self,
        config: DocGMSConfig,
        device: torch.device | None = None,
    )
```

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `store_path` | `str` | Resolved directory for persisted store files |
| `triples` | `list` | Raw list of `(head_idx, rel_idx, tail_idx)` triples |

**Geometric Primitives**

| Method | Signature | Returns |
|--------|-----------|---------|
| `score_triple` | `(head: str, rel: str, tail: str)` | `float \| None` |
| `tension_energy` | `(entity_a: str, entity_b: str)` | `float \| None` |
| `check_holonomy` | `(relation_path: list[str], direct_relation: str)` | `float \| None` |
| `is_path_consistent` | `(relation_path: list[str], direct_relation: str)` | `bool \| None` |
| `lookup_enm` | `(category: str, entity_id: str)` | `float \| None` |
| `link_predict` | `(head: str, relation: str, top_k: int = 10)` | `list[tuple[str, float]]` |
| `query_triples` | `(head=None, relation=None, tail=None)` | `list[tuple]` |

**Utility Methods**

| Method | Signature | Returns |
|--------|-----------|---------|
| `exists` | `()` | `bool` |
| `save` | `()` | `None` |
| `load` | `()` | `bool` |
| `entity_exists` | `(name: str)` | `bool` |
| `relation_exists` | `(name: str)` | `bool` |
| `fuzzy_match_entity` | `(name: str)` | `str \| None` |
| `find_contradictions` | `()` | `list` |
| `stats` | `()` | `dict` |

---

## 2. Document Ingestion

**Module:** `docgms.ingest`

### IngestResult

```python
@dataclass
class IngestResult:
    document: str
    new_entities: int = 0
    new_triples: int = 0
    new_enm: int = 0
    training_epochs: int = 0
    is_first: bool = False
    elapsed_seconds: float = 0.0
```

### ingest_document

```python
def ingest_document(
    store: GMSExpertStore,
    document_path: str,
    llm: LLMBackend | None,
    config: DocGMSConfig,
    device: torch.device,
) -> IngestResult
```

Parses a Markdown document, extracts triples and numeric entries via
LLM-assisted extraction, trains the GMS embeddings, and populates the store.

---

## 3. Runtime Learning

**Module:** `docgms.learn`

### LearnResult

```python
@dataclass
class LearnResult:
    accepted: bool = False
    reason: str = ""
    new_entities: list[str] = field(default_factory=list)
    new_relations: list[str] = field(default_factory=list)
    n_triples_added: int = 0
    training_queued: bool = False
    scores_before: list[float] = field(default_factory=list)
    scores_after: list[float] = field(default_factory=list)
    n_steps_run: int = 0
```

### RuntimeLearner

Incrementally adds new triples and ENM entries to a live store, expanding
embedding matrices and queueing background training.

```python
class RuntimeLearner:
    def __init__(self, store: GMSExpertStore, config: LearnConfig)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `learn_triples` | `(new_triples: list[ExtractedTriple], device: torch.device)` | `LearnResult` |
| `learn_enm` | `(category: str, entity_id: str, value: float)` | `bool` |
| `flush_training` | `()` | `None` |
| `wait_for_training` | `(timeout: float = 300.0)` | `bool` |
| `training_status` | `()` | `str` |

---

## 4. Verification Primitives

**Module:** `docgms.verify`

### ContradictionResult

```python
@dataclass
class ContradictionResult:
    entity_a: str
    entity_b: str
    tension_energy: float
    is_contradiction: bool
    is_entailment: bool
```

### HolonomyResult

```python
@dataclass
class HolonomyResult:
    path: list[str]
    direct: str
    defect: float
    is_consistent: bool
```

### VerificationReport

```python
@dataclass
class VerificationReport:
    contradictions: list[ContradictionResult] = field(default_factory=list)
    inconsistencies: list[HolonomyResult] = field(default_factory=list)
    overall_consistent: bool = True
    confidence: float = 1.0
    detail: str = ""
```

### Module-level Functions

```python
def check_contradictions(
    entity_pairs: list[tuple[str, str]],
    store: GMSExpertStore,
    config: VerifyConfig,
) -> list[ContradictionResult]

def check_holonomy(
    scored_triples: list[ScoredTriple],
    store: GMSExpertStore,
    config: VerifyConfig,
) -> list[HolonomyResult]

def verify_answer(
    scored_triples: list[ScoredTriple],
    store: GMSExpertStore,
    config: VerifyConfig,
) -> VerificationReport
```

---

## 5. Query Engine

**Module:** `docgms.query`

### QueryResult

```python
@dataclass
class QueryResult:
    answer: str = ""
    source: str = ""
    source_detail: str = ""
    confidence: float = 0.0
    confidence_method: str = ""
    extracted_triples: list[ScoredTriple] = field(default_factory=list)
    plausibility: PlausibilityReport | None = None
    verification: VerificationReport | None = None
    learned: bool = False
    learn_result: LearnResult | None = None
    gms_context: str = ""
    web_sources: list = field(default_factory=list)
    triples_merged: int = 0
```

### QueryEngine

```python
class QueryEngine:
    def __init__(
        self,
        store: GMSExpertStore,
        llm: LLMBackend,
        config: DocGMSConfig,
    )
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `query` | `(question: str, mode: str = "gms_llm")` | `QueryResult` |
| `query_batch` | `(questions: list[str], mode: str = "gms_llm")` | `list[QueryResult]` |

**Query modes:** `"gms_only"`, `"llm_only"`, `"gms_llm"` (default), `"web_agent"`.

---

## 6. Document Graph

**Module:** `finstructbench.graph`

### ENMKey

```python
@dataclass(frozen=True)
class ENMKey:
    type: str
    id: str
```

### ENMEntry

```python
@dataclass
class ENMEntry:
    key: ENMKey
    value: float
    hash: bytes

    @staticmethod
    def compute_hash(value: float) -> bytes
```

### PhaseEncoder

Maps numeric values to phase angles on S^1 for inequality reasoning.

```python
class PhaseEncoder:
    def __init__(self, v_min: float = 0.0, v_max: float = 1.0)

    def encode(self, value: float) -> float
    def check_inequality(
        self, value: float, limit: float, op: str,
    ) -> tuple[bool, float]
```

### DocumentGraph

In-memory graph used during ingestion before training to GMS embeddings.

```python
class DocumentGraph:
    def __init__(self)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `store_value` | `(category: str, entity_id: str, value: float)` | `None` |
| `lookup` | `(category: str, entity_id: str)` | `float \| None` |
| `add_triple` | `(head: str, relation: str, tail: str)` | `None` |
| `query_triples` | `(head=None, relation=None, tail=None)` | `list[tuple]` |
| `add_phase_encoder` | `(name: str, v_min: float = 0.0, v_max: float = 1.0)` | `None` |
| `check_threshold` | `(encoder_name: str, value: float, limit: float, op: str)` | `tuple[bool, float]` |
| `find_contradictions` | `()` | `list[tuple[str, str, str]]` |
| `stats` | `()` | `dict` |

---

## 7. Benchmark Orchestration

**Module:** `harness.doe.benchmark`, `harness.doe.config`

### DOEHarnessConfig

Central configuration dataclass with 60+ fields across 15 functional groups.

```python
@dataclass
class DOEHarnessConfig:
    # ── Document & Store ──
    markdown_path: str = ""
    stores_dir: str = "gms_stores"
    store_name: str = ""

    # ── DOE Design ──
    factor_group: str = "quick_screen"
    factor_names: list[str] = field(default_factory=list)
    n_runs: int = 32
    method: str = "sobol"
    seed: int = 42

    # ── Question Generation ──
    rephrase_method: str = "template"
    rephrase_model: str = "claude-sonnet-4-20250514"
    max_per_category: int = 10
    generators: list[str] = field(default_factory=list)

    # ── Target ──
    target_type: str = "llm"
    target_model: str = "claude-sonnet-4-20250514"
    context_mode: str = "full"

    # ── Tracing ──
    tracing_dir: str = "doe_tracing_store"
    enable_tracing: bool = True

    # ── Ingestion ──
    ingest_epochs: int = 200

    # ── Typed Verification ──
    enable_typed_verdicts: bool = False
    enable_provenance: bool = False
    snapshot_dir: str = ""
    semantic_mode: str = "off"
    semantic_model: str = ""

    # ── Tool Gateway ──
    enable_gateway: bool = False
    gateway_fault_profiles_path: str = ""
    gateway_schema_strict: bool = True

    # ── Policy Engine ──
    enable_policy_engine: bool = False
    policy_rules_path: str = ""

    # ── Scenarios & Runs ──
    scenarios_dir: str = ""
    enable_scenario_registry: bool = False
    runs_dir: str = ""
    regression_suite_dir: str = ""
    replay_manifest_path: str = ""

    # ── Extended Evaluators ──
    enable_process_evaluation: bool = False
    enable_recovery_evaluation: bool = False
    enable_interaction_evaluation: bool = False
    risk_tier_profiles_path: str = ""
    baseline_run_id: str = ""

    # ── Black-Box Testing ──
    target_profile_path: str = ""
    environment_yaml_path: str = ""
    campaigns_dir: str = ""
    enable_severity_classification: bool = False
    safety_config_path: str = ""

    # ── Performance & Cost ──
    enable_performance_profiling: bool = False
    sla_latency_ms: float = 5000.0
    enable_cost_tracking: bool = False
    cost_budget_usd: float = 10.0

    # ── Behavioral Testing ──
    enable_stateful_testing: bool = False
    stateful_session_count: int = 5
    enable_hallucination_testing: bool = False
    hallucination_geodesic_threshold: float = 1.0
    enable_calibration_testing: bool = False
    enable_multi_agent: bool = False
    multi_agent_roles_path: str = ""
    enable_cross_document: bool = False
    additional_store_paths: list[str] = field(default_factory=list)
    enable_disambiguation_testing: bool = False
    enable_invariance_testing: bool = False
    enable_streaming_testing: bool = False

    # ── Live Dashboard ──
    enable_live_dashboard: bool = False
    live_dashboard_port: int = 8765

    # ── Device ──
    device: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "DOEHarnessConfig"

    def resolve_store_name(self) -> str
```

### DOEGMSBenchmark

Top-level orchestrator: ingest, generate, run, analyze, report.

```python
class DOEGMSBenchmark:
    def __init__(self, config: DOEHarnessConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DOEGMSBenchmark"
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `ingest` | `(markdown_path: str \| None = None)` | `GMSExpertStore` |
| `generate_questions` | `()` | `list[DOEQuestion]` |
| `run` | `(evaluator=None, llm_client=None, model: str = "")` | `DOEBenchmarkResult` |
| `run_gms_baseline` | `()` | `DOEBenchmarkResult` |
| `run_scenarios` | `(scenarios: list, evaluator=None, llm_client=None, model: str = "")` | `DOEBenchmarkResult` |
| `analyze` | `(result: DOEBenchmarkResult)` | `DOEAnalyzer` |
| `generate_report` | `(result, analyzer=None, output_path="doe_report.html", extra_results=None)` | `str` |
| `generate_audit_report` | `(result, analyzer=None, output_path="doe_audit_report.html", accuracy_threshold=0.80, geo_confidence_threshold=0.50, failure_budget=None)` | `str` |
| `save_tracing` | `()` | `None` |

| Property | Type |
|----------|------|
| `typed_verdicts` | `dict` |

```python
@staticmethod
def print_results(result: DOEBenchmarkResult) -> None
```

---

## 8. Geometric Generators

**Module:** `harness.doe.generators`

All generators inherit from `QuestionGenerator` and implement a single method:

```python
def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

| Generator | Category | GMS Primitive |
|-----------|----------|---------------|
| `PlausibilityGenerator` | `plausibility` | `score_triple` |
| `TensionGenerator` | `tension` | `tension_energy` |
| `HolonomyGenerator` | `holonomy` | `check_holonomy` |
| `LinkPredictionGenerator` | `link_prediction` | `link_predict` |
| `MultiHopTransportGenerator` | `multi_hop` | `check_holonomy` + `score_triple` |
| `PhaseComparisonGenerator` | `phase_comparison` | `lookup_enm` |

**Factory function:**

```python
def geometric_generators() -> list[QuestionGenerator]
```

Returns all six generators in canonical order.

---

## 9. Taxonomy Generators

**Module:** `harness.doe.generators_taxonomy`

Extended generators for boundary, adversarial, robustness, recovery and
governance testing. Same `generate(store) -> list[GeneratedQuestion]` interface.

| Generator | Category | Description |
|-----------|----------|-------------|
| `BoundaryGenerator` | `boundary` | Near-threshold ENM values (epsilon=0.01) |
| `AdversarialGenerator` | `adversarial` | Entity/relation swaps to create plausible-looking falsehoods |
| `RobustnessGenerator` | `robustness` | Typos, case variations, format noise |
| `RecoveryTestGenerator` | `recovery` | Questions requiring tool-error recovery |
| `GovernanceTestGenerator` | `governance` | Policy-constrained scenarios |

**Factory function:**

```python
def taxonomy_generators() -> list[QuestionGenerator]
```

---

## 10. Geometric Judge

**Module:** `harness.doe.judge`

### GeometricVerdict

```python
@dataclass
class GeometricVerdict:
    correct: bool = False
    detail: str = ""
    geodesic_confidence: float = 0.0
    tension_score: float = 0.0
    holonomy_score: float = 0.0
    overall_confidence: float = 0.0
    extracted_triples: list[ExtractedTriple] = field(default_factory=list)
    scored_triples: list[ScoredTriple] = field(default_factory=list)
    contradictions: list[ContradictionResult] = field(default_factory=list)
    inconsistencies: list[HolonomyResult] = field(default_factory=list)
    n_triples_scored: int = 0
    n_tension_checks: int = 0
    n_holonomy_checks: int = 0
```

### GMSJudge

```python
class GMSJudge:
    def __init__(
        self,
        store: GMSExpertStore,
        verify_config: VerifyConfig | None = None,
        weights: dict | None = None,
        semantic_mode: str = "off",
        semantic_model: str = "",
        tracing=None,
    )
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `judge` | `(llm_answer: str, ground_truth, exact_correct: bool, exact_detail: str, question_metadata: dict \| None = None)` | `GeometricVerdict` |
| `judge_typed` | `(llm_answer: str, ground_truth, exact_correct: bool, exact_detail: str, question_metadata=None, question_category="", run_id="", question_id="", question_text="")` | `TypedVerdict` |

**Factory function:**

```python
def create_judge(store: GMSExpertStore, **kwargs) -> GMSJudge
```

---

## 11. Evaluators & Adapters

**Module:** `harness.doe.evaluators`, `harness.doe.adapters`

### LLMEvaluator

```python
class LLMEvaluator:
    def __init__(self, client: anthropic.Anthropic, model: str)
    def __call__(self, question: str, context: str) -> str
```

### AgentEvaluator

```python
class AgentEvaluator:
    def __init__(self, harness: Harness)
    def __call__(self, question: str, context: str) -> str
```

### make_evaluator

```python
def make_evaluator(
    target_type: str,
    target_model: str,
    client: anthropic.Anthropic | None = None,
    harness: Harness | None = None,
) -> LLMEvaluator | AgentEvaluator
```

### SUTTrace

System-under-test execution trace, returned by all adapters.

```python
@dataclass
class SUTTrace:
    raw_output: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved_context: str = ""
    error: str = ""
```

### ToolCallRecord

```python
@dataclass
class ToolCallRecord:
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    result: str = ""
    latency_ms: float = 0.0
```

### SUTAdapter (abstract)

```python
class SUTAdapter(ABC):
    @abstractmethod
    def evaluate(self, question: str, context: str) -> SUTTrace
```

### Concrete Adapters

| Adapter | Constructor |
|---------|-------------|
| `LLMAdapter` | `(client: anthropic.Anthropic, model: str)` |
| `AgentAdapter` | `(harness: Harness, gateway=None)` |
| `RAGAdapter` | `(retriever: Callable, generator: Callable)` |

**Factory function:**

```python
def make_adapter(
    target_type: str,
    target_model: str,
    client: anthropic.Anthropic | None = None,
    harness: Harness | None = None,
) -> SUTAdapter
```

---

## 12. Claim Verification Pipeline

**Module:** `harness.doe.claims`, `harness.doe.verifiers`, `harness.doe.verdict`

### Claim Types

All claims inherit from `ClaimBase`:

```python
@dataclass
class ClaimBase:
    claim_id: str = ""
    source_run_id: str = ""
    source_question_id: str = ""
    raw_text: str = ""
    confidence: float = 0.0
    provenance: ClaimProvenance | None = None
    claim_type: str = "base"
```

```python
@dataclass
class ClaimProvenance:
    doc_id: str = ""
    section_tag: str = ""
    span_start: int = -1
    span_end: int = -1
    extraction_method: str = ""
```

| Claim Type | Extra Fields |
|------------|-------------|
| `FactClaim` | `head`, `relation`, `tail` |
| `ThresholdClaim` | `entity`, `metric`, `value`, `limit`, `operator` |
| `RuleClaim` | `antecedent_entities`, `antecedent_relations`, `consequent_entity`, `direct_relation`, `rule_type` |
| `ArtifactClaim` | `category`, `entity_id`, `claimed_value` |
| `ToolUseClaim` | `tool_name`, `tool_args`, `result_key`, `claimed_value` |
| `SemanticClaim` | `ground_truth`, `question_text`, `context_section`, `semantic_type` |

### ClaimCompiler

```python
class ClaimCompiler:
    def __init__(
        self,
        store: GMSExpertStore,
        truth_store=None,
        semantic_mode: str = "off",
    )

    def compile(
        self,
        llm_answer: str,
        question_metadata: dict | None = None,
        question_category: str = "",
        run_id: str = "",
        question_id: str = "",
        ground_truth: str = "",
        question_text: str = "",
    ) -> list[ClaimBase]
```

### Verifiers

Each verifier produces a `SubVerdict`:

```python
@dataclass
class SubVerdict:
    verifier_name: str = ""
    claim_id: str = ""
    passed: bool = False
    confidence: float = 0.0
    detail: str = ""
    gms_method_used: str = ""
    raw_score: float | None = None
    failure: FailureAttribution | None = None
```

| Verifier | Claim Type | GMS Method | Threshold |
|----------|-----------|------------|-----------|
| `GraphVerifier` | `FactClaim` | `score_triple` | 1.2 (plausibility) |
| `NumericVerifier` | `ThresholdClaim` | `lookup_enm` | exact match |
| `ArtifactVerifier` | `ArtifactClaim` | `lookup_enm` | SHA-256 hash |
| `LogicVerifier` | `RuleClaim` | `check_holonomy`, `tension_energy` | config-based |
| `PolicyVerifier` | `ToolUseClaim` | `score_triple` | 1.5 (plausibility) |

### VerificationRouter

```python
class VerificationRouter:
    def __init__(
        self,
        store: GMSExpertStore,
        config: VerifyConfig | None = None,
        semantic_verifier=None,
        process_evaluator=None,
        recovery_evaluator=None,
        interaction_evaluator=None,
    )

    def route(self, claim: ClaimBase) -> SubVerdict
    def route_all(self, claims: list[ClaimBase]) -> list[SubVerdict]
```

### TypedVerdict

```python
@dataclass
class TypedVerdict:
    # Backwards-compatible geometric fields
    correct: bool = False
    detail: str = ""
    geodesic_confidence: float = 0.0
    tension_score: float = 0.0
    holonomy_score: float = 0.0
    overall_confidence: float = 0.0

    # Typed verification fields
    claims: list[ClaimBase] = field(default_factory=list)
    sub_verdicts: list[SubVerdict] = field(default_factory=list)
    failures: list[FailureAttribution] = field(default_factory=list)

    # Provenance
    run_id: str = ""
    question_id: str = ""
    timestamp: str = ""

    # Counts
    n_claims: int = 0
    n_passed: int = 0
    n_failed: int = 0

    @classmethod
    def from_sub_verdicts(
        cls,
        claims: list[ClaimBase],
        sub_verdicts: list[SubVerdict],
        exact_correct: bool,
        exact_detail: str,
        run_id: str = "",
        question_id: str = "",
        weights: dict | None = None,
    ) -> "TypedVerdict"

    def to_geometric_verdict(self) -> GeometricVerdict

    @property
    def failure_codes(self) -> list[str]

    @property
    def verifier_summary(self) -> dict[str, dict]
```

---

## 13. Failure Attribution

**Module:** `harness.doe.failures`

### FailureCode

```python
class FailureCode(str, Enum):
    # Semantic
    SEMANTIC_EXTRACTION_FAILURE = "semantic_extraction_failure"
    SEMANTIC_MISMATCH           = "semantic_mismatch"
    HALLUCINATION_DETECTED      = "hallucination_detected"
    INCOMPLETE_ANSWER           = "incomplete_answer"
    NO_CLAIMS_EXTRACTED         = "no_claims_extracted"

    # Graph
    ENTITY_NOT_FOUND            = "entity_not_found"
    RELATION_NOT_FOUND          = "relation_not_found"
    FACT_INCORRECT              = "fact_incorrect"

    # Consistency
    PATH_INCONSISTENCY          = "path_inconsistency"
    CONTRADICTION_DETECTED      = "contradiction_detected"
    TENSION_EXCEEDED            = "tension_exceeded"
    HOLONOMY_DEFECT             = "holonomy_defect"

    # Numeric
    THRESHOLD_VIOLATION         = "threshold_violation"
    BOUNDARY_FAILURE            = "boundary_failure"
    EXACT_MISMATCH              = "exact_mismatch"
    ARTIFACT_INTEGRITY_FAILURE  = "artifact_integrity_failure"

    # Tool / Policy
    TOOL_CALL_INVALID           = "tool_call_invalid"
    POLICY_VIOLATION            = "policy_violation"
    POLICY_BLOCKED_PRE_EXECUTION = "policy_blocked_pre_execution"
    APPROVAL_REQUIRED           = "approval_required"
    REQUIRED_TOOL_MISSING       = "required_tool_missing"
    FORBIDDEN_TOOL_CALLED       = "forbidden_tool_called"

    # Process / Recovery
    PROCESS_DEVIATION           = "process_deviation"
    TOOL_BUDGET_EXCEEDED        = "tool_budget_exceeded"
    RECOVERY_FAILURE            = "recovery_failure"
    ESCALATION_INAPPROPRIATE    = "escalation_inappropriate"
    CLARIFICATION_INSUFFICIENT  = "clarification_insufficient"

    # Internal
    VERIFICATION_ERROR          = "verification_error"
```

### FailureSeverity

```python
class FailureSeverity(str, Enum):
    ERROR   = "error"
    WARNING = "warning"
    INFO    = "info"
```

### FailureAttribution

```python
@dataclass
class FailureAttribution:
    code: FailureCode
    claim_id: str
    verifier: str
    severity: FailureSeverity = FailureSeverity.ERROR
    detail: str = ""
    gms_method: str = ""
    raw_score: float | None = None
    suggested_action: str = ""
    severity_assessment: object | None = None
```

---

## 14. Semantic Verification

**Module:** `harness.doe.semantic`

### RubricDimension

```python
@dataclass
class RubricDimension:
    name: str
    description: str
    weight: float = 1.0
```

### SemanticScore

```python
@dataclass
class SemanticScore:
    scores: dict[str, int] = field(default_factory=dict)
    reasoning: str = ""
    raw_response: str = ""

    @property
    def weighted_score(self) -> float
```

### GMSEvidence

```python
@dataclass
class GMSEvidence:
    answer_entities: dict[str, str] = field(default_factory=dict)
    truth_entities: dict[str, str] = field(default_factory=dict)
    entity_triples: dict[str, list[tuple]] = field(default_factory=dict)
    geodesic_scores: dict[str, float] = field(default_factory=dict)
    tension_scores: dict[str, float] = field(default_factory=dict)
    entity_coverage: float = 0.0
```

### SemanticVerifier

```python
class SemanticVerifier:
    def __init__(
        self,
        store: GMSExpertStore,
        client=None,
        model: str = "claude-sonnet-4-20250514",
        rubric: list[RubricDimension] | None = None,
        pass_threshold: float = 3.0,
        tracing=None,
    )

    def verify(self, claim: SemanticClaim) -> SubVerdict
```

---

## 15. Tool Gateway

**Module:** `harness.doe.gateway`

### FaultProfile

```python
@dataclass
class FaultProfile:
    tool_pattern: str = "*"
    latency_ms: float = 0.0
    error_rate: float = 0.0
    error_message: str = "Simulated tool failure"
    stale_data: bool = False
    mock_response: str | None = None
    schema_strict: bool = True

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "FaultProfile"
```

### GatewayDecision

```python
@dataclass
class GatewayDecision:
    tool_name: str = ""
    action: str = ""
    reason: str = ""
    gms_score: float | None = None
    mock_plausibility_geodesic: float | None = None
    timestamp: str = ""
```

### GatewayTranscript

```python
@dataclass
class GatewayTranscript:
    run_id: str = ""
    decisions: list[GatewayDecision] = field(default_factory=list)
```

### ToolGateway

Intercepts tool calls, injects faults, validates schemas and logs decisions.

```python
class ToolGateway:
    def __init__(
        self,
        store: GMSExpertStore,
        fault_profiles: list[FaultProfile] | None = None,
        policy_engine: Any = None,
        tracing: DOETracingGMS | None = None,
        run_id: str = "",
    )

    def install(self, harness: Harness) -> None
    def uninstall(self, harness: Harness) -> None
    def get_transcript(self) -> GatewayTranscript
```

---

## 16. Policy Engine

**Module:** `harness.doe.policy`

### PolicyRule

```python
@dataclass
class PolicyRule:
    rule_id: str = ""
    action: str = ""
    requirement: str = ""
    rule_type: str = "requires"    # "requires" | "forbids" | "approves"
    context: str = "*"
    severity: str = "error"

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "PolicyRule"
```

### PolicyDecision

```python
@dataclass
class PolicyDecision:
    allowed: bool = True
    action: str = ""
    rules_checked: list[str] = field(default_factory=list)
    blocking_rules: list[str] = field(default_factory=list)
    gms_scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
```

### PolicyEngine

```python
class PolicyEngine:
    PLAUSIBILITY_THRESHOLD = 1.5
    TENSION_THRESHOLD = 1.7

    def __init__(
        self,
        store: GMSExpertStore,
        rules: list[PolicyRule] | None = None,
        tracing: DOETracingGMS | None = None,
        run_id: str = "",
    )

    def load_rules_from_yaml(self, path: str) -> list[PolicyRule]
    def encode_rules(self, rules: list[PolicyRule]) -> None
    def check(self, action: str, args: dict, context: str = "") -> PolicyDecision
```

---

## 17. Scenario Registry

**Module:** `harness.doe.scenarios`

### StopCondition

```python
@dataclass
class StopCondition:
    condition_type: str = ""
    value: Any = ""
    action: str = "abort"

    @classmethod
    def from_dict(cls, d: dict) -> StopCondition
    def to_dict(self) -> dict
```

### Scenario

Full scenario specification with 33 fields covering question definition,
tool constraints, oracle configuration, interaction scripts and stop
conditions. Key fields shown; see source for the complete list.

```python
@dataclass
class Scenario:
    scenario_id: str = ""
    name: str = ""
    description: str = ""
    version: int = 1
    tags: list[str] = field(default_factory=list)
    risk_tier: str = "standard"
    generator_name: str = ""
    generator_params: dict = field(default_factory=dict)
    question_text: str = ""
    ground_truth: str = ""
    answer_type: str = "str"
    metadata: dict = field(default_factory=dict)
    context_mode: str = "full"
    fault_profile_refs: list[str] = field(default_factory=list)
    policy_rule_refs: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_tool_sequence: list[str] = field(default_factory=list)
    max_tool_calls: int = 0
    expected_recovery: bool = False
    oracle_definition: dict = field(default_factory=dict)
    expected_side_effects: dict = field(default_factory=dict)
    trace_pattern: str = ""
    constraints: list[dict] = field(default_factory=list)
    prohibited_outcomes: list[str] = field(default_factory=list)
    required_evidence_items: list[str] = field(default_factory=list)
    interaction_script: dict = field(default_factory=dict)
    environment_fixture_id: str = ""
    stop_conditions: list[dict] = field(default_factory=list)
    severity_overrides: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    author: str = ""
    source_run_id: str = ""
    source_question_id: str = ""

    def to_dict(self) -> dict
    def to_yaml(self) -> str
    @classmethod
    def from_dict(cls, data: dict) -> Scenario
    @classmethod
    def from_yaml_file(cls, path: str) -> Scenario
    @classmethod
    def from_failed_run(cls, run_id, question_result, failure_code="") -> Scenario
```

### ScenarioRegistry

```python
class ScenarioRegistry:
    def __init__(
        self,
        scenarios_dir: str,
        store: GMSExpertStore | None = None,
        tracing: DOETracingGMS | None = None,
    )
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `add` | `(scenario: Scenario)` | `str` |
| `get` | `(scenario_id: str)` | `Scenario \| None` |
| `update` | `(scenario: Scenario)` | `None` |
| `delete` | `(scenario_id: str)` | `None` |
| `list_all` | `()` | `list[Scenario]` |
| `get_version` | `(scenario_id: str, version: int)` | `Scenario \| None` |
| `list_versions` | `(scenario_id: str)` | `list[int]` |
| `search_by_tag` | `(tags: list[str])` | `list[Scenario]` |
| `search_by_risk_tier` | `(tier: str)` | `list[Scenario]` |
| `search_geometric` | `(query: str, top_k: int = 10)` | `list[tuple[Scenario, float]]` |
| `find_complementary` | `(scenario_id: str, top_k: int = 5)` | `list[tuple[Scenario, float]]` |
| `to_doe_questions` | `(scenarios: list[Scenario], store: GMSExpertStore)` | `list[GeneratedQuestion]` |

---

## 18. Run Management

**Module:** `harness.doe.runs`

### RunState

```python
class RunState(Enum):
    CREATED    = "created"
    PREPARING  = "preparing"
    EXECUTING  = "executing"
    EVALUATING = "evaluating"
    COMPLETED  = "completed"
    FAILED     = "failed"
    ABORTED    = "aborted"
```

### RunRecord

```python
@dataclass
class RunRecord:
    run_id: str = ""
    state: RunState = RunState.CREATED
    config_snapshot: dict = field(default_factory=dict)
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    state_history: list[tuple[str, str, str]] = field(default_factory=list)
    result_path: str = ""
    n_questions: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    error: str = ""
    error_state: str = ""
    scenario_ids: list[str] = field(default_factory=list)
    target_id: str = ""
    campaign_id: str = ""

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord"
    @property
    def is_terminal(self) -> bool
```

### RunManager

```python
class RunManager:
    INDEX_FILE = "index.json"

    def __init__(self, runs_dir: str, tracing: DOETracingGMS | None = None)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_run` | `(config: DOEHarnessConfig)` | `RunRecord` |
| `transition` | `(run_id: str, to_state: RunState)` | `RunRecord` |
| `execute_run` | `(run_id: str, benchmark: DOEGMSBenchmark, evaluator=None, llm_client=None, model: str = "")` | `RunRecord` |
| `get_run` | `(run_id: str)` | `RunRecord \| None` |
| `list_runs` | `(state: RunState \| None = None)` | `list[RunRecord]` |
| `get_latest` | `()` | `RunRecord \| None` |

---

## 19. Replay & Regression

**Module:** `harness.doe.replay`

### ReplayManifest

```python
@dataclass
class ReplayManifest:
    manifest_id: str = ""
    source_run_id: str = ""
    config_snapshot: dict = field(default_factory=dict)
    snapshot_id: str = ""
    doe_questions_path: str = ""
    fault_profiles: list[dict] = field(default_factory=list)
    policy_rules: list[dict] = field(default_factory=list)
    random_seed: int = 42
    reference_result_path: str = ""
    created_at: str = ""

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "ReplayManifest"
    def to_yaml(self, path: str) -> None
    @classmethod
    def from_yaml(cls, path: str) -> "ReplayManifest"
```

### RunComparison

```python
@dataclass
class RunComparison:
    run_a_id: str = ""
    run_b_id: str = ""
    accuracy_delta: float = 0.0
    geo_confidence_delta: float = 0.0
    tension_delta: float = 0.0
    regressions: list[dict] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)
    stable_pass: int = 0
    stable_fail: int = 0
    semantic_drift: float = 0.0
    failure_delta: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict
    @property
    def regression_rate(self) -> float
```

### RegressionSuite

```python
class RegressionSuite:
    SUITE_FILE = "suite.json"

    def __init__(
        self,
        suite_dir: str,
        scenario_registry=None,
        tracing: DOETracingGMS | None = None,
    )

    def promote_failures(
        self, run_record, result: DOEBenchmarkResult, min_severity: str = "error",
    ) -> list[str]
    def add_scenario(self, scenario_id: str) -> None
    def remove_scenario(self, scenario_id: str) -> None
    def list_scenarios(self) -> list[str]
```

### Module-level Functions

```python
def compare_runs(
    result_a: DOEBenchmarkResult,
    result_b: DOEBenchmarkResult,
    store: GMSExpertStore | None = None,
) -> RunComparison

def replay_run(
    manifest: ReplayManifest,
    snapshot_manager: SnapshotManager,
    evaluator=None,
    llm_client=None,
    model: str = "",
    device=None,
) -> tuple[DOEBenchmarkResult, RunComparison | None]
```

---

## 20. Audit & Release Gates

**Module:** `harness.doe.audit`

### AuditPacket

```python
@dataclass
class AuditPacket:
    doe_qid: str = ""
    category: str = ""
    question: str = ""
    ground_truth: str = ""
    sut_answer: str = ""
    correct: bool = False
    detail: str = ""
    geodesic_confidence: float = 0.0
    tension_score: float = 0.0
    holonomy_score: float = 0.0
    overall_confidence: float = 0.0
    n_claims: int = 0
    n_passed: int = 0
    n_failed: int = 0
    claim_types: list[str] = field(default_factory=list)
    failure_codes: list[str] = field(default_factory=list)
    verifier_summary: dict = field(default_factory=dict)
    factor_assignment: dict = field(default_factory=dict)

    def to_dict(self) -> dict
```

### ReleaseGate

```python
@dataclass
class ReleaseGate:
    passed: bool = False
    accuracy_threshold: float = 0.0
    actual_accuracy: float = 0.0
    geo_confidence_threshold: float = 0.0
    actual_geo_confidence: float = 0.0
    failure_budget: dict[str, int] = field(default_factory=dict)
    actual_failures: dict[str, int] = field(default_factory=dict)
    blocking_failures: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict
```

### RiskTierProfile

Three built-in factory methods for standard risk tiers.

```python
@dataclass
class RiskTierProfile:
    tier_name: str = "standard"
    accuracy_threshold: float = 0.0
    geo_confidence_threshold: float = 0.0
    max_tension: float = 2.0
    max_holonomy: float = 1.0
    failure_budget: dict[str, int] = field(default_factory=dict)
    required_evaluators: list[str] = field(default_factory=list)
    max_regression_rate: float = 0.05
    require_baseline_comparison: bool = False

    @classmethod
    def advisory(cls) -> "RiskTierProfile"          # >= 70% accuracy

    @classmethod
    def recommendation(cls) -> "RiskTierProfile"     # >= 80% accuracy

    @classmethod
    def action_taking(cls) -> "RiskTierProfile"      # >= 95% accuracy

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "RiskTierProfile"
```

### TieredReleaseGate

```python
@dataclass
class TieredReleaseGate:
    tier: RiskTierProfile = field(default_factory=RiskTierProfile)
    passed: bool = False
    checks: dict[str, dict] = field(default_factory=dict)
    blocking_reasons: list[str] = field(default_factory=list)
    regression_comparison: object = None
    timestamp: str = ""

    def to_dict(self) -> dict
```

### RunTrend

```python
@dataclass
class RunTrend:
    metric_name: str = ""
    run_ids: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    trend_direction: str = "stable"
    slope: float = 0.0

    def to_dict(self) -> dict
```

### AuditReporter

```python
class AuditReporter:
    def __init__(
        self,
        result: DOEBenchmarkResult,
        typed_verdicts: dict[str, TypedVerdict],
        truth_store: TruthStore | None = None,
    )
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `generate_audit_packets` | `()` | `list[AuditPacket]` |
| `generate_failure_report` | `()` | `dict` |
| `generate_release_gate` | `(accuracy_threshold=0.80, geo_confidence_threshold=0.50, failure_budget=None)` | `ReleaseGate` |
| `generate_executive_summary` | `()` | `str` |
| `generate_tiered_release_gate` | `(profiles: list[RiskTierProfile], baseline_result=None)` | `list[TieredReleaseGate]` |
| `compute_trends` | `(run_results: list[tuple[str, DOEBenchmarkResult]], metrics=None)` | `list[RunTrend]` |
| `generate_tiered_executive_summary` | `(tiered_gates: list[TieredReleaseGate], trends=None)` | `str` |

---

## 21. Snapshots

**Module:** `harness.doe.snapshots`

### GMSSnapshot

```python
@dataclass
class GMSSnapshot:
    snapshot_id: str = ""
    run_id: str = ""
    timestamp: str = ""
    store_path: str = ""
    snapshot_path: str = ""
    parent_snapshot_id: str = ""
    trigger: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> GMSSnapshot
```

### SnapshotManager

```python
class SnapshotManager:
    def __init__(
        self,
        snapshot_dir: str,
        tracing: DOETracingGMS | None = None,
    )
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `take_snapshot` | `(store: GMSExpertStore, run_id="", trigger="manual")` | `GMSSnapshot` |
| `restore_snapshot` | `(snapshot_id: str, device=None)` | `GMSExpertStore` |
| `list_snapshots` | `()` | `list[GMSSnapshot]` |
| `diff_snapshots` | `(id_a: str, id_b: str)` | `dict` |
| `get_latest` | `()` | `GMSSnapshot \| None` |
| `create_replay_manifest` | `(snapshot_id, run_id="", config_snapshot=None, doe_questions_path="", fault_profiles=None, policy_rules=None, reference_result_path="", random_seed=42)` | `ReplayManifest` |

---

## 22. Provenance

**Module:** `harness.doe.provenance`

### SectionSpan

```python
@dataclass
class SectionSpan:
    tag: str
    heading: str = ""
    level: int = 1
    char_start: int = 0
    char_end: int = 0

    @property
    def length(self) -> int
```

### DocumentRecord

```python
@dataclass
class DocumentRecord:
    doc_id: str
    source_path: str
    content_hash: str
    ingested_at: str = ""
    version: int = 1
    parent_doc_id: str = ""
    title: str = ""
    sections: list[SectionSpan] = field(default_factory=list)
```

### NormalizedDocument

```python
@dataclass
class NormalizedDocument:
    doc_id: str
    markdown: str
    sections: list[SectionSpan] = field(default_factory=list)

    def get_section_text(self, tag: str) -> str | None
    def section_for_offset(self, offset: int) -> SectionSpan | None
```

### ProvenanceRegistry

```python
class ProvenanceRegistry:
    def __init__(self, tracing: DOETracingGMS | None = None)

    def register_document(self, doc_path: str, markdown: str) -> DocumentRecord
    def link_triple_to_source(
        self, head: str, relation: str, tail: str, doc_id: str, section_tag: str,
    ) -> None
    def get_provenance(self, head: str, relation: str, tail: str) -> tuple[str, str] | None
    def get_document(self, doc_id: str) -> DocumentRecord | None
    def get_normalized(self, doc_id: str) -> NormalizedDocument | None

    @property
    def documents(self) -> list[DocumentRecord]
```

---

## 23. Truth Store

**Module:** `harness.doe.truth`

### Truth Types

All truth types inherit from `TruthBase`:

```python
@dataclass
class TruthBase:
    truth_id: str = ""
    doc_id: str = ""
    section_tag: str = ""
    confidence: float = 1.0
    truth_type: str = "base"
```

| Type | Extra Fields |
|------|-------------|
| `EntityTruth` | `entity`, `entity_type`, `gms_idx` |
| `RelationTruth` | `head`, `relation`, `tail`, `gms_score` |
| `ThresholdTruth` | `entity`, `metric`, `value`, `encoder_name`, `phase_angle` |
| `ArtifactTruth` | `category`, `entity_id`, `exact_value`, `enm_hash` |

### TruthStore

```python
class TruthStore:
    def __init__(self, store: GMSExpertStore, tracing=None)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `build_from_store` | `()` | `None` |
| `get` | `(truth_id: str)` | `TruthBase \| None` |
| `get_entity_truths` | `(entity: str)` | `list[EntityTruth]` |
| `get_relation_truth` | `(head: str, relation: str, tail: str)` | `RelationTruth \| None` |
| `get_artifact_truth` | `(category: str, entity_id: str)` | `ArtifactTruth \| None` |
| `get_truth_for_claim` | `(claim: ClaimBase)` | `TruthBase \| None` |
| `save` | `(path: str)` | `None` |
| `load` | `(path: str)` | `None` |

| Property | Returns |
|----------|---------|
| `stats` | `str` |
| `all_truths` | `list[TruthBase]` |

---

## 24. Target Profiles

**Module:** `harness.doe.target`

### IsolationRequirements

```python
@dataclass
class IsolationRequirements:
    network_isolated: bool = False
    sandbox_filesystem: bool = False
    synthetic_data_only: bool = False
    max_memory_mb: int = 0               # 0 = unlimited
    max_wall_time_s: int = 0             # 0 = unlimited

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> IsolationRequirements
```

### TargetProfile

```python
@dataclass
class TargetProfile:
    target_id: str = ""                  # auto-generated tgt_<uuid8>
    name: str = ""
    version: str = ""
    target_type: str = "agent"           # "llm" | "agent" | "rag" | "browser"
    endpoint: str = "local"
    model_name: str = ""
    capabilities: list[str] = field(default_factory=list)
    timeout_ms: int = 60_000
    max_concurrent: int = 1
    isolation: IsolationRequirements = field(default_factory=IsolationRequirements)
    metadata: dict = field(default_factory=dict)
    created_at: str = ""                 # auto-generated if empty

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> TargetProfile
    @classmethod
    def from_yaml(cls, path: str) -> TargetProfile
    def to_yaml(self, path: str) -> None
    def has_capability(self, cap: str) -> bool
    def summary(self) -> str
```

---

## 25. Severity Classification

**Module:** `harness.doe.severity`

### SeverityLevel

```python
class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MODERATE = "moderate"
    LOW      = "low"
```

### SeverityAssessment

```python
@dataclass
class SeverityAssessment:
    level: SeverityLevel = SeverityLevel.LOW
    impact_score: float = 0.0
    recoverability_score: float = 0.0
    detectability_score: float = 0.0
    composite_score: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> SeverityAssessment
```

### SeverityClassifier

```python
class SeverityClassifier:
    def __init__(
        self,
        store: GMSExpertStore | None = None,
        custom_map: dict[FailureCode, SeverityLevel] | None = None,
    )

    def classify(
        self,
        failure: FailureAttribution,
        risk_tier: str = "standard",
        context: dict | None = None,
    ) -> SeverityAssessment

    def classify_code(
        self,
        code: FailureCode,
        risk_tier: str = "standard",
        raw_score: float | None = None,
    ) -> SeverityAssessment

    def classify_all(
        self,
        failures: list[FailureAttribution],
        risk_tier: str = "standard",
        context: dict | None = None,
    ) -> list[SeverityAssessment]

    @staticmethod
    def count_by_level(
        assessments: list[SeverityAssessment],
    ) -> dict[SeverityLevel, int]
```

---

## 26. Environment Simulator

**Module:** `harness.doe.environment`

### StateTransition

```python
@dataclass
class StateTransition:
    trigger_action: str = "*"
    trigger_condition: str = ""
    state_mutation: dict = field(default_factory=dict)
    response_template: str = ""

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> StateTransition
```

### FailureMode

```python
@dataclass
class FailureMode:
    mode_name: str = ""
    error_message: str = "Simulated failure"
    probability: float = 0.0
    trigger_after_n_calls: int = 0

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> FailureMode
```

### ToolEndpointSpec

```python
@dataclass
class ToolEndpointSpec:
    tool_name: str = ""
    description: str = ""
    request_schema: dict = field(default_factory=dict)
    response_schema: dict = field(default_factory=dict)
    state_transitions: list[StateTransition] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)
    initial_state: dict = field(default_factory=dict)
    default_response: str = "OK"

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> ToolEndpointSpec
```

### ScheduledEvent / ClockControl

```python
@dataclass
class ScheduledEvent:
    at_ms: int = 0
    event_type: str = ""
    payload: dict = field(default_factory=dict)

@dataclass
class ClockControl:
    mode: str = "real"               # "real" | "frozen" | "ticking"
    frozen_at: str = ""
    tick_ms: int = 1000
    scheduled_events: list[ScheduledEvent] = field(default_factory=list)
```

### DataFixture

```python
@dataclass
class DataFixture:
    fixture_id: str = ""
    version: int = 1
    data: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)
    content_hash: str = ""

    def verify_integrity(self) -> bool
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> DataFixture
```

### EnvironmentSimulator

```python
class EnvironmentSimulator:
    def __init__(
        self,
        tool_specs: list[ToolEndpointSpec],
        fixtures: list[DataFixture] | None = None,
        clock: ClockControl | None = None,
        store: GMSExpertStore | None = None,
        tracing: DOETracingGMS | None = None,
    )
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `reset` | `(seed: int = 42)` | `None` |
| `get_state` | `()` | `dict` |
| `get_state_diff` | `(before: dict, after: dict)` | `dict` |
| `has_endpoint` | `(tool_name: str)` | `bool` |
| `handle_tool_call` | `(tool_name: str, args: dict)` | `str` |
| `now` | `()` | `int` |
| `advance` | `(ms: int)` | `list[ScheduledEvent]` |
| `load_fixture` | `(fixture_id: str)` | `DataFixture \| None` |
| `verify_fixture_integrity` | `(fixture_id: str)` | `bool` |
| `snapshot_state` | `()` | `dict` |
| `restore_state` | `(snapshot: dict)` | `None` |

```python
@classmethod
def from_yaml(
    cls,
    path: str,
    store: GMSExpertStore | None = None,
    tracing: DOETracingGMS | None = None,
) -> EnvironmentSimulator
```

---

## 27. Oracle Framework

**Module:** `harness.doe.oracles`

### OracleVerdict

```python
@dataclass
class OracleVerdict:
    oracle_type: str = ""
    passed: bool = False
    confidence: float = 0.0
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict
```

### OracleBase (abstract)

```python
class OracleBase(ABC):
    oracle_type: str = "base"

    @abstractmethod
    def evaluate(
        self,
        scenario: Scenario,
        trace: SUTTrace,
        env_state_before: dict | None = None,
        env_state_after: dict | None = None,
    ) -> OracleVerdict
```

### Concrete Oracles

| Oracle | Type | Description |
|--------|------|-------------|
| `ExactOutcomeOracle` | `exact_outcome` | String/numeric equality against ground truth |
| `AcceptableSetOracle` | `acceptable_set` | Answer within an allowed set |
| `ConstraintOracle` | `constraint` | Prohibited outcome and required evidence checks |
| `TraceOracle` | `trace` | Tool-call sequence validation via holonomy |
| `SideEffectOracle` | `side_effect` | Environment state-diff verification |
| `HumanReviewedOracle` | `human_reviewed` | LLM-as-judge with configurable rubric |

### CompositeOracle

Weighted combination of multiple oracles.

```python
class CompositeOracle(OracleBase):
    def __init__(
        self,
        oracles: list[tuple[OracleBase, float]],
        threshold: float = 0.5,
        fail_on_any_critical: bool = True,
    )

    def evaluate(self, scenario, trace, env_state_before=None, env_state_after=None) -> OracleVerdict
```

### make_oracle

```python
def make_oracle(
    oracle_def: dict,
    store: GMSExpertStore | None = None,
    environment: EnvironmentSimulator | None = None,
    llm_client: Any = None,
) -> OracleBase
```

---

## 28. Campaign Management

**Module:** `harness.doe.campaign`

### CampaignType

```python
class CampaignType(Enum):
    RELEASE_VALIDATION        = "release_validation"
    REGRESSION_SWEEP          = "regression_sweep"
    ADVERSARIAL_SWEEP         = "adversarial_sweep"
    MODEL_COMPARISON          = "model_comparison"
    COMPLIANCE_CERTIFICATION  = "compliance_certification"
```

### CampaignConfig

```python
@dataclass
class CampaignConfig:
    campaign_id: str = ""                # auto-generated cmp_<uuid8>
    name: str = ""
    description: str = ""
    campaign_type: CampaignType = CampaignType.RELEASE_VALIDATION
    target_profiles: list[str] = field(default_factory=list)
    scenario_tags: list[str] = field(default_factory=list)
    scenario_ids: list[str] = field(default_factory=list)
    risk_tier: str = "standard"
    n_runs: int = 2
    max_parallel: int = 1
    stop_on_critical: bool = False
    base_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    created_at: str = ""                 # auto-generated if empty

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> CampaignConfig
    @classmethod
    def from_yaml(cls, path: str) -> CampaignConfig
    def to_yaml(self, path: str) -> None
```

### CampaignRunSummary

```python
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
```

### CampaignResult

```python
@dataclass
class CampaignResult:
    campaign_id: str = ""
    campaign_type: str = ""
    run_summaries: list[CampaignRunSummary] = field(default_factory=list)
    aggregate_accuracy: float = 0.0
    aggregate_geo_confidence: float = 0.0
    total_questions: int = 0
    total_correct: int = 0
    critical_failure_count: int = 0
    failure_summary: dict[str, int] = field(default_factory=dict)
    per_target_accuracy: dict[str, float] = field(default_factory=dict)
    gate_passed: bool | None = None
    started_at: str = ""
    completed_at: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict
```

### CampaignManager

```python
class CampaignManager:
    def __init__(
        self,
        campaigns_dir: str,
        run_manager: RunManager,
        scenario_registry: ScenarioRegistry | None = None,
        tracing: DOETracingGMS | None = None,
    )
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_campaign` | `(config: CampaignConfig)` | `str` |
| `get_campaign` | `(campaign_id: str)` | `CampaignConfig \| None` |
| `list_campaigns` | `()` | `list[CampaignConfig]` |
| `execute_campaign` | `(campaign_id, benchmark_factory, evaluator_factory=None)` | `CampaignResult` |
| `compare_campaigns` | `(id_a: str, id_b: str)` | `dict` |

---

## 29. Extended Evaluators

**Module:** `harness.doe.evaluators_extended`

### ProcessClaim / RecoveryClaim / InteractionClaim

```python
@dataclass
class ProcessClaim:
    actual_tool_sequence: list[str] = field(default_factory=list)
    expected_tool_sequence: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    max_tool_calls: int = 0
    claim_type: str = "process"

@dataclass
class RecoveryClaim:
    fault_type: str = ""
    fault_tool: str = ""
    pre_fault_output: str = ""
    post_fault_output: str = ""
    recovery_detected: bool = False
    claim_type: str = "recovery"

@dataclass
class InteractionClaim:
    interaction_type: str = ""
    trigger_context: str = ""
    response_text: str = ""
    claim_type: str = "interaction"
```

### ProcessEvaluator

```python
class ProcessEvaluator:
    HOLONOMY_THRESHOLD = 0.5

    def __init__(self, store: GMSExpertStore)
    def verify(self, claim: ProcessClaim) -> SubVerdict
```

### RecoveryEvaluator

```python
class RecoveryEvaluator:
    TENSION_THRESHOLD = 1.5

    def __init__(self, store: GMSExpertStore)
    def verify(self, claim: RecoveryClaim) -> SubVerdict
```

### HumanInteractionEvaluator

```python
class HumanInteractionEvaluator:
    PLAUSIBILITY_THRESHOLD = 1.5

    def __init__(self, store: GMSExpertStore)
    def verify(self, claim: InteractionClaim) -> SubVerdict
```

---

## 30. Hallucination Testing

**Module:** `harness.doe.hallucination`

### HallucinationAssessment

```python
@dataclass
class HallucinationAssessment:
    claim_text: str = ""
    head: str = ""
    relation: str = ""
    tail: str = ""
    geodesic_distance: float = 0.0
    tension_vs_nearest: float | None = None
    severity: str = "grounded"
```

### HallucinationReport

```python
@dataclass
class HallucinationReport:
    n_claims: int = 0
    n_hallucinated: int = 0
    hallucination_rate: float = 0.0
    mean_geodesic: float = 0.0
    max_geodesic: float = 0.0
    severity_counts: dict = field(default_factory=dict)
    assessments: list[HallucinationAssessment] = field(default_factory=list)

    def to_dict(self) -> dict
```

### HallucinationOracle

```python
class HallucinationOracle:
    oracle_type = "hallucination"

    def __init__(self, store: GMSExpertStore, geodesic_threshold: float = 1.0)

    def assess_claim(self, head: str, relation: str, tail: str) -> HallucinationAssessment
    def assess_triples(self, claims: list[tuple[str, str, str]]) -> HallucinationReport
```

### HallucinationGenerator

```python
class HallucinationGenerator(QuestionGenerator):
    category = "hallucination"
    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

---

## 31. Calibration Testing

**Module:** `harness.doe.calibration`

### CalibrationPoint

```python
@dataclass
class CalibrationPoint:
    question_id: str = ""
    claimed_confidence: float = 0.0
    actual_correct: bool = False
    confidence_phase: float = 0.0
    accuracy_phase: float = 0.0
    calibration_gap: float = 0.0
```

### CalibrationBin

```python
@dataclass
class CalibrationBin:
    bin_lower: float = 0.0
    bin_upper: float = 0.0
    n_samples: int = 0
    mean_confidence: float = 0.0
    mean_accuracy: float = 0.0
    gap: float = 0.0
```

### CalibrationReport

```python
@dataclass
class CalibrationReport:
    n_points: int = 0
    ece: float = 0.0            # Expected Calibration Error
    mce: float = 0.0            # Maximum Calibration Error
    mean_confidence: float = 0.0
    mean_accuracy: float = 0.0
    overconfidence_ratio: float = 0.0
    bins: list[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> dict
```

### CalibrationTester

```python
class CalibrationTester:
    def __init__(self, n_bins: int = 10)

    def add(
        self,
        claimed_confidence: float,
        actual_correct: bool,
        question_id: str = "",
    ) -> CalibrationPoint

    def report(self) -> CalibrationReport

    @property
    def points(self) -> list[CalibrationPoint]

    def reset(self) -> None
```

### CalibrationGenerator

```python
class CalibrationGenerator(QuestionGenerator):
    category = "calibration"
    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

---

## 32. Stateful Agent Testing

**Module:** `harness.doe.stateful`

### AgentMemoryProbe

```python
@dataclass
class AgentMemoryProbe:
    probe_id: str = ""
    fact_triple: tuple[str, str, str] = ("", "", "")
    teach_prompt: str = ""
    recall_prompt: str = ""
    delay_turns: int = 0
    expected_answer: str = ""
    recall_geodesic: float | None = None
    session_tension: float | None = None
```

### MemoryAssessment

```python
@dataclass
class MemoryAssessment:
    n_probes: int = 0
    n_recalled: int = 0
    mean_recall_geodesic: float = 0.0
    mean_session_tension: float = 0.0
    degradation_detected: bool = False
    probes: list[AgentMemoryProbe] = field(default_factory=list)

    def to_dict(self) -> dict
```

### StatefulAgentTester

```python
class StatefulAgentTester:
    def __init__(self, store: GMSExpertStore)

    def build_probes(
        self,
        n_probes: int = 10,
        delay_turns: int = 5,
        seed: int = 42,
    ) -> list[AgentMemoryProbe]

    def score_recall(
        self,
        probe: AgentMemoryProbe,
        recalled_answer: str,
    ) -> AgentMemoryProbe

    def measure_session_tension(
        self,
        entity_a: str,
        entity_b: str,
    ) -> float | None

    def assess(self, probes: list[AgentMemoryProbe]) -> MemoryAssessment
```

### StatefulAgentGenerator

```python
class StatefulAgentGenerator(QuestionGenerator):
    category = "stateful_memory"
    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

### MemoryRecallOracle

```python
class MemoryRecallOracle:
    oracle_type = "memory_recall"

    def __init__(self, store: GMSExpertStore, geodesic_threshold: float = 0.5)

    def evaluate(
        self,
        head: str,
        relation: str,
        expected_tail: str,
        recalled_answer: str,
    ) -> dict
```

---

## 33. Invariance Testing

**Module:** `harness.doe.invariance`

### InvarianceResult

```python
@dataclass
class InvarianceResult:
    question_group: str = ""
    n_paraphrases: int = 0
    answers: list[str] = field(default_factory=list)
    pairwise_tensions: dict = field(default_factory=dict)
    mean_tension: float = 0.0
    max_tension: float = 0.0
    invariant: bool = True

    def to_dict(self) -> dict
```

### InvarianceTester

```python
class InvarianceTester:
    def __init__(self, store: GMSExpertStore, tension_threshold: float = 0.3)

    def add_paraphrase_result(
        self,
        variant_id: str,
        answer: str,
        group: str = "",
    ) -> None

    def evaluate(self, group: str) -> InvarianceResult
    def evaluate_all(self) -> list[InvarianceResult]
    def reset(self) -> None
```

### InvarianceGenerator

```python
class InvarianceGenerator(QuestionGenerator):
    category = "paraphrase_invariance"
    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

---

## 34. Streaming Testing

**Module:** `harness.doe.streaming`

### StreamChunk

```python
@dataclass
class StreamChunk:
    index: int = 0
    text: str = ""
    cumulative_text: str = ""
    timestamp_ms: float = 0.0
    plausibility_geodesic: float | None = None
```

### StreamingTrace

```python
@dataclass
class StreamingTrace:
    raw_output: str = ""
    chunks: list[StreamChunk] = field(default_factory=list)
    first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    stream_tension: float | None = None
    chunk_holonomy: float | None = None
    mean_chunk_plausibility: float = 0.0
    n_chunks: int = 0
    error: str = ""

    def to_dict(self) -> dict
```

### StreamingAdapter

```python
class StreamingAdapter:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        store: GMSExpertStore | None = None,
    )

    def evaluate(self, question: str, context: str) -> StreamingTrace
```

### StreamConsistencyOracle

```python
class StreamConsistencyOracle:
    oracle_type = "stream_consistency"

    def __init__(
        self,
        store: GMSExpertStore,
        tension_threshold: float = 0.5,
        holonomy_threshold: float = 0.5,
    )

    def evaluate_stream(self, trace: StreamingTrace) -> dict
```

---

## 35. Multi-Agent Testing

**Module:** `harness.doe.multi_agent`

### AgentRole

```python
@dataclass
class AgentRole:
    role_id: str = ""
    system_prompt: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    store_path: str = ""

    def to_dict(self) -> dict
```

### InteractionResult

```python
@dataclass
class InteractionResult:
    scenario_id: str = ""
    agent_outputs: dict = field(default_factory=dict)
    inter_agent_tension: dict = field(default_factory=dict)
    process_holonomy: float | None = None
    consensus_reached: bool = False
    mean_tension: float = 0.0

    def to_dict(self) -> dict
```

### ConsensusOracle

```python
class ConsensusOracle:
    def __init__(
        self,
        stores: dict[str, GMSExpertStore] | None = None,
        tension_threshold: float = 0.5,
    )

    def evaluate_consensus(self, entity: str) -> InteractionResult

    def evaluate_delegation_holonomy(
        self,
        delegation_chain: list[str],
        direct_relation: str,
        store: GMSExpertStore,
    ) -> float | None
```

### MultiAgentGenerator

```python
class MultiAgentGenerator(QuestionGenerator):
    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

---

## 36. Cross-Document Analysis

**Module:** `harness.doe.cross_document`

### DocumentConflict

```python
@dataclass
class DocumentConflict:
    entity: str = ""
    store_a_id: str = ""
    store_b_id: str = ""
    fact_a: str = ""
    fact_b: str = ""
    inter_store_tension: float = 0.0
    conflict_type: str = "minor_variance"

    def to_dict(self) -> dict
```

### CrossDocumentAnalyzer

```python
class CrossDocumentAnalyzer:
    def __init__(
        self,
        stores: dict[str, GMSExpertStore] | None = None,
        tension_threshold: float = 0.5,
    )

    def find_shared_entities(self) -> dict[str, list[str]]
    def find_conflicts(self, max_conflicts: int = 50) -> list[DocumentConflict]
```

### CrossDocumentGenerator

```python
class CrossDocumentGenerator(QuestionGenerator):
    def __init__(self, stores: dict[str, GMSExpertStore] | None = None)
    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

---

## 37. Disambiguation Testing

**Module:** `harness.doe.disambiguation`

### AmbiguousEntity

```python
@dataclass
class AmbiguousEntity:
    surface_form: str = ""
    candidate_a: str = ""
    candidate_b: str = ""
    geodesic_a: float | None = None
    geodesic_b: float | None = None
    geodesic_separation: float = 0.0
    correct_candidate: str = ""
```

### DisambiguationOracle

```python
class DisambiguationOracle:
    def __init__(self, store: GMSExpertStore)

    def evaluate(
        self,
        head: str,
        relation: str,
        candidate_a: str,
        candidate_b: str,
        claimed: str,
    ) -> dict
```

### DisambiguationGenerator

```python
class DisambiguationGenerator(QuestionGenerator):
    def generate(self, store: GMSExpertStore) -> list[GeneratedQuestion]
```

---

## 38. Multi-Turn Interactions

**Module:** `harness.doe.multiturn`

### TurnSpec

```python
@dataclass
class TurnSpec:
    turn_id: str = ""
    user_message: str = ""
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)
    prohibited_tool_calls: list[str] = field(default_factory=list)
    expected_state_changes: dict = field(default_factory=dict)
    max_latency_ms: float = 0.0
    carry_forward: dict = field(default_factory=dict)
    setup_context: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> TurnSpec
```

### InteractionScript

```python
@dataclass
class InteractionScript:
    script_id: str = ""
    name: str = ""
    description: str = ""
    turns: list[TurnSpec] = field(default_factory=list)
    setup_context: str = ""
    initial_state: dict = field(default_factory=dict)
    teardown_assertions: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> InteractionScript
    @classmethod
    def from_yaml(cls, path: str) -> InteractionScript
    def to_dict(self) -> dict
```

### TurnResult

```python
@dataclass
class TurnResult:
    turn_id: str = ""
    raw_output: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    assertions_passed: list[str] = field(default_factory=list)
    assertions_failed: list[str] = field(default_factory=list)
    state_after: dict = field(default_factory=dict)
    carry_forward_values: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool
```

### ScriptResult

```python
@dataclass
class ScriptResult:
    script_id: str = ""
    turn_results: list[TurnResult] = field(default_factory=list)
    overall_passed: bool = False
    teardown_passed: list[str] = field(default_factory=list)
    teardown_failed: list[str] = field(default_factory=list)
    state_carryover_verified: bool = True
    consistency_scores: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict
```

### MultiTurnAdapter

```python
class MultiTurnAdapter:
    def __init__(
        self,
        harness: Harness,
        store: GMSExpertStore | None = None,
        tracing: DOETracingGMS | None = None,
    )

    def execute_script(
        self,
        script: InteractionScript,
        environment: EnvironmentSimulator | None = None,
    ) -> ScriptResult
```

### MultiTurnScenarioRunner

```python
class MultiTurnScenarioRunner:
    def __init__(
        self,
        adapter: MultiTurnAdapter,
        environment: EnvironmentSimulator | None = None,
    )

    def is_multi_turn(self, scenario) -> bool
    def run_scenario(self, scenario) -> ScriptResult
```

---

## 39. Safety Framework

**Module:** `harness.doe.safety`

### SafetyConfig

```python
@dataclass
class SafetyConfig:
    sandbox_mode: bool = False
    network_isolation: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    synthetic_data_mode: bool = False
    max_tool_calls: int = 0              # 0 = unlimited
    max_file_writes: int = 0             # 0 = unlimited
    max_network_calls: int = 0           # 0 = unlimited
    blocked_tool_patterns: list[str] = field(default_factory=list)
    allowed_tool_patterns: list[str] = field(default_factory=list)
    destructive_action_mode: str = "block"  # "block" | "noop" | "log"

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> SafetyConfig
    @classmethod
    def from_yaml(cls, path: str) -> SafetyConfig
```

### SafetyEnforcer

```python
class SafetyEnforcer:
    def __init__(
        self,
        config: SafetyConfig,
        tracing: DOETracingGMS | None = None,
    )

    def install(self, harness: Harness) -> None
    def uninstall(self, harness: Harness) -> None
    def reset_counters(self) -> None
    def get_blocked_log(self) -> list[dict]
    def get_stats(self) -> dict
```

### SyntheticDataFramework

```python
class SyntheticDataFramework:
    def __init__(self, store: GMSExpertStore)

    def synthesize_entities(self, n: int, seed: int = 42) -> list[dict]
    def synthesize_relations(
        self,
        entities: list[str],
        density: float = 0.1,
        seed: int = 42,
    ) -> list[tuple[str, str, str]]
    def synthesize_fixture(
        self,
        n_entities: int = 10,
        relation_density: float = 0.1,
        seed: int = 42,
    ) -> dict
```

---

## 40. Parallel Execution

**Module:** `harness.doe.parallel`

### ParallelConfig

```python
@dataclass
class ParallelConfig:
    max_workers: int = 4
    batch_size: int = 0                  # 0 = all at once
    timeout_per_question_s: float = 120.0
    retry_on_error: bool = True
    max_retries: int = 2
    retry_delay_s: float = 1.0

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> ParallelConfig
```

### ParallelExecutor

```python
class ParallelExecutor:
    def __init__(
        self,
        config: ParallelConfig | None = None,
        on_progress: Callable[[int, int, _EvalResult], None] | None = None,
    )

    def execute_batch(
        self,
        questions: list[tuple[int, str, str, str]],
        evaluate_fn: Callable[[str, str], str],
    ) -> list[_EvalResult]

    def execute_doe_batch(
        self,
        doe_questions: list,
        evaluate_fn: Callable[[str, str], str],
        context_fn: Callable[[Any], str],
    ) -> list[_EvalResult]
```

### merge_parallel_results

```python
def merge_parallel_results(
    eval_results: list[_EvalResult],
    doe_questions: list,
    score_fn: Callable[[str, str], Any],
) -> list[dict]
```

---

## 41. Performance Profiling

**Module:** `harness.doe.profiling`

### PerformanceRecord

```python
@dataclass
class PerformanceRecord:
    tool_name: str = ""
    latency_ms: float = 0.0
    latency_phase: float = 0.0
    sla_phase: float = 0.0
    sla_tension: float = 0.0
    within_sla: bool = True
    metadata: dict = field(default_factory=dict)
```

### PerformanceReport

```python
@dataclass
class PerformanceReport:
    n_records: int = 0
    sla_latency_ms: float = 0.0
    sla_pass_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_sla_tension: float = 0.0
    max_sla_tension: float = 0.0
    gms_sla_geodesic: float | None = None
    per_tool: dict = field(default_factory=dict)

    def to_dict(self) -> dict
```

### PerformanceProfiler

```python
class PerformanceProfiler:
    def __init__(
        self,
        store: GMSExpertStore | None = None,
        sla_latency_ms: float = 5000.0,
    )

    def record(
        self,
        tool_name: str,
        latency_ms: float,
        metadata: dict | None = None,
    ) -> PerformanceRecord

    def report(self, run_id: str = "") -> PerformanceReport

    @property
    def records(self) -> list[PerformanceRecord]

    def reset(self) -> None
```

---

## 42. Cost Tracking

**Module:** `harness.doe.cost_tracker`

### CostRecord

```python
@dataclass
class CostRecord:
    tool_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0
    cost_phase: float = 0.0
    budget_phase: float = 0.0
    budget_tension: float = 0.0
    within_budget: bool = True
    timestamp: str = ""                  # auto-generated if empty
```

### CostSummary

```python
@dataclass
class CostSummary:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_usd: float = 0.0
    budget_utilization: float = 0.0
    budget_tension: float = 0.0
    budget_geodesic: float | None = None
    n_records: int = 0
    per_tool: dict = field(default_factory=dict)

    def to_dict(self) -> dict
```

### CostTracker

```python
class CostTracker:
    def __init__(
        self,
        store: GMSExpertStore | None = None,
        budget_usd: float = 10.0,
        input_price_per_mtok: float = 3.0,   # $/M input tokens
        output_price_per_mtok: float = 15.0, # $/M output tokens
    )

    def record(
        self,
        tool_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = None,
    ) -> CostRecord

    def summary(self) -> CostSummary

    @property
    def records(self) -> list[CostRecord]
    @property
    def cumulative_cost(self) -> float
    @property
    def within_budget(self) -> bool

    def reset(self) -> None
```

---

## 43. Versioning

**Module:** `harness.doe.versioning`

### VersionedArtifact

```python
@dataclass
class VersionedArtifact:
    artifact_id: str = ""
    artifact_type: str = ""
    version: int = 1
    content: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)
    content_hash: str = ""
    created_at: str = ""
    updated_at: str = ""
    author: str = ""
    metadata: dict = field(default_factory=dict)

    def verify_integrity(self) -> bool
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> VersionedArtifact
    def to_yaml(self) -> str
    @classmethod
    def from_yaml_file(cls, path: str) -> VersionedArtifact
```

### ArtifactRegistry

```python
class ArtifactRegistry:
    def __init__(self, artifacts_dir: str, artifact_type: str = "")
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `add` | `(artifact: VersionedArtifact)` | `str` |
| `get` | `(artifact_id: str)` | `VersionedArtifact \| None` |
| `get_version` | `(artifact_id: str, version: int)` | `VersionedArtifact \| None` |
| `update` | `(artifact: VersionedArtifact)` | `int` |
| `delete` | `(artifact_id: str)` | `None` |
| `list_all` | `()` | `list[VersionedArtifact]` |
| `list_versions` | `(artifact_id: str)` | `list[int]` |
| `diff` | `(artifact_id: str, version_a: int, version_b: int)` | `dict` |

### Specialized Registries

```python
class FaultProfileRegistry(ArtifactRegistry):
    def __init__(self, artifacts_dir: str)

class PolicyRuleRegistry(ArtifactRegistry):
    def __init__(self, artifacts_dir: str)

class FixtureRegistry(ArtifactRegistry):
    def __init__(self, artifacts_dir: str)
```

---

## 44. Tracing

**Module:** `harness.doe.tracing`

### DOETracingGMS

GMS-backed tracing store that records every event as a triple in its own
expert store.

```python
class DOETracingGMS:
    def __init__(self, stores_dir: str, device: torch.device)
```

**Core Trace Methods**

| Method | Key Parameters |
|--------|---------------|
| `trace_run_start` | `run_id, target_model, context_mode, factors, initial_state` |
| `trace_question` | `doe_qid, run_id, base_qid, category, question_text, answer, correct` |
| `trace_run_end` | `run_id, total, correct` |
| `trace_analysis` | `run_id, analysis_results` |
| `trace_ingest` | `document_path, store_name, source_type, elapsed, stats` |
| `trace_question_generation` | `generator_name, n_doe_questions, n_runs, category_counts, factors` |

**Extended Trace Methods**

| Method | Key Parameters |
|--------|---------------|
| `trace_semantic_verdict` | `doe_qid, verdict details` |
| `trace_truth_store` | `store_name, elapsed, truth_counts, total` |
| `trace_release_gate` | `run_id, passed, ...` |
| `trace_report` | `run_id, report_type, output_path` |
| `trace_typed_verdict` | `doe_qid, typed_verdict, run_id` |
| `trace_snapshot` | `snapshot_id, run_id, store_stats` |
| `trace_gateway_decision` | `run_id, tool_name, action, reason, ...` |
| `trace_policy_decision` | `run_id, action, allowed, blocking_rules, gms_scores` |
| `trace_state_transition` | `run_id, from_state, to_state` |
| `trace_regression_promotion` | `run_id, scenario_id, source_question_id, failure_code` |
| `trace_run_comparison` | `run_id_a, run_id_b, ...` |
| `trace_tiered_release_gate` | gate details |
| `trace_performance_record` | profiling details |
| `trace_cost_record` | cost details |

**Query Methods**

| Method | Signature | Returns |
|--------|-----------|---------|
| `query_run` | `(run_id: str)` | `list` |
| `query_by_factor` | `(factor_name: str, level: str)` | `list` |
| `query_question` | `(doe_qid: str)` | `list` |

**Persistence**

| Method / Property | Returns |
|-------------------|---------|
| `save()` | `None` |
| `stats()` | `dict` |
| `store` (property) | `GMSExpertStore` |

---

## 45. Live Dashboard

**Module:** `harness.doe.live_dashboard`

### DashboardSnapshot

```python
@dataclass
class DashboardSnapshot:
    timestamp: str = ""
    active_runs: list[str] = field(default_factory=list)
    completed_runs: list[str] = field(default_factory=list)
    total_questions: int = 0
    total_correct: int = 0
    accuracy: float = 0.0
    recent_events: list[dict] = field(default_factory=list)
    gateway_stats: dict = field(default_factory=dict)
    performance_stats: dict = field(default_factory=dict)
    cost_stats: dict = field(default_factory=dict)
    store_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict

    @classmethod
    def from_tracing(cls, tracing: DOETracingGMS) -> DashboardSnapshot
```

### DashboardServer

WebSocket-based server streaming real-time run progress.

```python
class DashboardServer:
    def __init__(
        self,
        tracing: DOETracingGMS,
        port: int = 8765,
        poll_interval_s: float = 2.0,
    )

    def start(self) -> None
    def snapshot(self) -> DashboardSnapshot
```

---

## 46. Browser Adapter

**Module:** `harness.doe.browser_adapter`

### BrowserConfig

```python
@dataclass
class BrowserConfig:
    base_url: str = ""
    headless: bool = True
    browser_type: str = "chromium"
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout_ms: float = 10000.0
    screenshot_dir: str = ""

    def to_dict(self) -> dict
```

### UIStep / UIStepResult

```python
@dataclass
class UIStep:
    step_id: str = ""
    action: str = ""           # "navigate" | "click" | "fill" | "wait" | ...
    selector: str = ""
    value: str = ""
    timeout_ms: float = 5000.0
    assertion: str = ""
    screenshot: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> UIStep

@dataclass
class UIStepResult:
    step_id: str = ""
    action: str = ""
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0
    screenshot_path: str = ""
    assertion_passed: bool = True
    assertion_detail: str = ""
```

### UIScript / UIScriptResult

```python
@dataclass
class UIScript:
    script_id: str = ""
    name: str = ""
    description: str = ""
    steps: list[UIStep] = field(default_factory=list)
    setup_url: str = ""
    teardown_steps: list[UIStep] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> UIScript
    @classmethod
    def from_yaml(cls, path: str) -> UIScript
    def to_dict(self) -> dict

@dataclass
class UIScriptResult:
    script_id: str = ""
    step_results: list[UIStepResult] = field(default_factory=list)
    overall_passed: bool = False
    total_latency_ms: float = 0.0
    screenshots: list[str] = field(default_factory=list)
    page_text: str = ""

    def to_dict(self) -> dict
```

### BrowserAdapter

```python
class BrowserAdapter:
    def __init__(self, config: BrowserConfig)

    def close(self) -> None
    def evaluate(self, question: str, context: str) -> SUTTrace
    def execute_ui_script(self, script: UIScript) -> UIScriptResult
```

---

## 47. Mock Harness

**Module:** `knowlytix.harness.fixtures` (re-exports from `knowlytix.harness.fixtures.mock_agent` and `.mock_agent_full`)

### MockHarness

Deterministic mock agent for testing without LLM calls.

```python
class MockHarness:
    def __init__(
        self,
        tools: dict[str, dict] | None = None,
        recovery_mode: str = "retry",
        deliberate_errors: bool = False,
    )

    def run(self, prompt: str) -> str
```

**Factory:**

```python
def create_mock_evaluator(
    recovery_mode: str = "retry",
    deliberate_errors: bool = False,
) -> Callable[[str, str], str]
```

### FullMockHarness

Extended mock with confidence modes, role identities, teach/recall memory
and streaming support.

```python
class FullMockHarness(MockHarness):
    def __init__(
        self,
        environment=None,
        confidence_mode: bool = False,
        role_id: str = "",
        expertise: str = "",
        **kwargs,
    )

    def teach(self, key: str, value: str) -> None
    def recall(self, key: str) -> str | None
    def set_confidence(self, level: int) -> None
    def run(self, prompt: str) -> str
    def stream_response(self, prompt: str) -> list[str]
    def reset_conversation(self) -> None
    def get_conversation_history(self) -> list[dict[str, str]]
    def get_all_tool_names(self) -> list[str]
```

**Factories:**

```python
def create_full_evaluator(...) -> Callable[[str, str], str]

def create_mock_stream_client(
    harness: FullMockHarness | None = None,
) -> MockStreamClient

def create_agent_for_role(
    role_id: str,
    expertise: str = "",
    recovery_mode: str = "retry",
    confidence_mode: bool = False,
) -> FullMockHarness

def create_mock_environment() -> EnvironmentSimulator

def create_multi_turn_script() -> InteractionScript
```

---

## Bridge Functions

**Module:** `harness.doe.bridge`

Utility functions to convert between the DOE harness question format and the
FinStructBench evaluation format.

```python
def adapt_question(
    eq: EvalQuestion,
    store: GMSExpertStore,
) -> FSBQuestion

def generate_base_questions(
    store: GMSExpertStore,
    generator_names: list[str] | None = None,
    max_per_category: int = 10,
    seed: int = 42,
) -> list[FSBQuestion]

def scenario_to_fsb_question(
    scenario: Scenario,
    store: GMSExpertStore | None = None,
) -> FSBQuestion | None
```

---

## External Dependencies (graphdoe)

The following classes are provided by the `graphdoe` package and are used
throughout the GMSH pipeline:

| Class | Description |
|-------|-------------|
| `DesignMatrix` | Space-filling DOE designs (Sobol, LHS, Grid) |
| `DOEAnalyzer` | Logistic regression analysis of DOE results |
| `DOEBenchmarkResult` | Container for per-question results and factor assignments |
| `QuestionRephraser` | Template and LLM-based question paraphrasing |
| `QuestionGenerator` | Abstract base class for all question generators |
| `GeneratedQuestion` | Single generated question with metadata and ground truth |
