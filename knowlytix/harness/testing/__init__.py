# SPDX-License-Identifier: Apache-2.0
"""DOE-GMS Benchmark: systematic LLM/agent testing with graph-verified
ground truth and Design of Experiments factor analysis.

Quick start::

    from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig

    config = DOEHarnessConfig(
        markdown_path="report.md",
        factor_group="query_core",
        n_runs=32,
    )
    bench = DOEGMSBenchmark(config)
    bench.ingest()
    result = bench.run(evaluator=my_evaluator)
    analyzer = bench.analyze(result)
    analyzer.print_report()
"""

from .config import DOEHarnessConfig
from .benchmark import DOEGMSBenchmark
from .bridge import generate_base_questions, adapt_question, scenario_to_fsb_question
from .evaluators import LLMEvaluator, AgentEvaluator, make_evaluator
from .generators import (
    PlausibilityGenerator, TensionGenerator, HolonomyGenerator,
    LinkPredictionGenerator, MultiHopTransportGenerator,
    PhaseComparisonGenerator, ConditionalGenerator,
    ProceduralGenerator, geometric_generators,
)
from .judge import GMSJudge, GeometricVerdict, create_judge
from .tracing import DOETracingGMS
from .dashboard import generate_report, generate_comparison_report

# Typed Verification: Claim Compiler + Typed Verifiers + Failure Taxonomy
from .claims import (
    ClaimBase, ClaimProvenance, FactClaim, ThresholdClaim,
    RuleClaim, ArtifactClaim, ToolUseClaim, SemanticClaim,
    ClaimSchema, FactClaimSchema, ThresholdClaimSchema,
    ArtifactClaimSchema, RuleClaimSchema,
    ClaimExtractor, RuleBasedClaimExtractor,
    StructuredClaimExtractor, LLMClaimExtractor,
    ClaimCompiler,
)
from .verifiers import (
    SubVerdict, GraphVerifier, NumericVerifier, ArtifactVerifier,
    LogicVerifier, PolicyVerifier, VerificationRouter,
)
from .semantic import (
    SemanticVerifier, GMSEvidence, SemanticScore,
    RubricDimension, DEFAULT_RUBRIC,
)
from .failures import FailureCode, FailureSeverity, FailureAttribution
from .verdict import (
    TypedVerdict,
    load_overall_confidence_weights,
    DEFAULT_OVERALL_CONFIDENCE_WEIGHTS,
)

# Provenance + Truth: Provenance Chain + Truth Objects
from .provenance import (
    DocumentRecord, SectionSpan, NormalizedDocument, ProvenanceRegistry,
)
from .truth import (
    TruthBase, EntityTruth, RelationTruth, ThresholdTruth,
    ArtifactTruth, TruthStore,
)

# SUT Surface: Extended SUT Adapters + Test Generators
from .adapters import (
    SUTAdapter, SUTTrace, ToolCallRecord,
    LLMAdapter, AgentAdapter, RAGAdapter, make_adapter,
)
from .test_generators import (
    ExactRetrievalGenerator, NegativeGenerator,
    WorkflowComplianceGenerator, extended_generators,
)

# Versioning + Snapshots: + Snapshots
from .snapshots import GMSSnapshot, SnapshotManager

# Audit + Reporting: + Enhanced Reporting
from .audit import (
    AuditPacket, ReleaseGate, AuditReporter,
    RiskTierProfile, TieredReleaseGate, RunTrend,
)
from .dashboard import generate_enhanced_report

# GMSH Agentic Harness Extensions
from .gateway import ToolGateway, FaultProfile, GatewayDecision, GatewayTranscript
from .policy import PolicyEngine, PolicyRule, PolicyDecision
from .scenarios import Scenario, ScenarioRegistry
from .runs import RunState, RunRecord, RunManager
from .replay import (
    ReplayManifest, RunComparison, RegressionSuite,
    compare_runs, replay_run,
)
from .evaluators_extended import (
    ProcessClaim, RecoveryClaim, InteractionClaim,
    ProcessEvaluator, RecoveryEvaluator, HumanInteractionEvaluator,
)

# Black-Box Testing — Foundation
from .target import TargetProfile, IsolationRequirements
from .severity import SeverityLevel, SeverityAssessment, SeverityClassifier
from .environment import (
    EnvironmentSimulator, ToolEndpointSpec, ClockControl,
    DataFixture, StateTransition, FailureMode, ScheduledEvent,
)
from .oracles import (
    OracleVerdict, OracleBase, ExactOutcomeOracle, AcceptableSetOracle,
    ConstraintOracle, TraceOracle, SideEffectOracle,
    HumanReviewedOracle, CompositeOracle, make_oracle,
    DEFAULT_COMPOSITE_ORACLE_WEIGHTS, load_composite_oracle_weights,
)
from .campaign import (
    CampaignType, CampaignConfig, CampaignResult,
    CampaignRunSummary, CampaignManager,
)

# Black-Box Testing — Coverage
from .generators_taxonomy import (
    BoundaryGenerator, AdversarialGenerator, RobustnessGenerator,
    RecoveryTestGenerator, GovernanceTestGenerator, taxonomy_generators,
)
from .safety import SafetyConfig, SafetyEnforcer, SyntheticDataFramework
from .multiturn import (
    TurnSpec, InteractionScript, TurnResult, ScriptResult,
    MultiTurnAdapter, MultiTurnScenarioRunner,
)

# Black-Box Testing — Scale & Polish
from .parallel import ParallelConfig, ParallelExecutor, merge_parallel_results
from .versioning import (
    VersionedArtifact, ArtifactRegistry,
    FaultProfileRegistry, PolicyRuleRegistry, FixtureRegistry,
)
from .browser_adapter import (
    UIStep, UIScript, UIStepResult, UIScriptResult,
    BrowserConfig, BrowserAdapter,
)
from .scenarios import StopCondition

# GMS-Native Enhancements — Profiling & Cost
from .profiling import (
    PerformanceRecord, PerformanceReport, PerformanceProfiler,
)
from .cost_tracker import (
    CostRecord, CostSummary, CostTracker,
)

# GMS-Native Enhancements — Agent Memory & Behavioral Testing
from .stateful import (
    AgentMemoryProbe, MemoryAssessment,
    StatefulAgentTester, StatefulAgentGenerator, MemoryRecallOracle,
)
from .hallucination import (
    HallucinationGenerator, HallucinationOracle, geodesic_band,
)
from .calibration import (
    CalibrationPoint, CalibrationReport,
    CalibrationTester, CalibrationGenerator,
)

# GMS-Native Enhancements — Multi-Source & Multi-Agent
from .multi_agent import (
    AgentRole, InteractionResult,
    ConsensusOracle, MultiAgentGenerator,
)
from .cross_document import (
    DocumentConflict, CrossDocumentAnalyzer, CrossDocumentGenerator,
)
from .disambiguation import (
    AmbiguousEntity, DisambiguationOracle, DisambiguationGenerator,
)

# GMS-Native Enhancements — Robustness & Invariance
from .invariance import (
    InvarianceResult, InvarianceTester, InvarianceGenerator,
)
from .streaming import (
    StreamChunk, StreamingTrace,
    StreamingAdapter, StreamConsistencyOracle,
)

# GMS-Native Enhancements — Observability
from .live_dashboard import DashboardSnapshot, DashboardServer

# Flat env-driven settings (lightweight; does not pull harness.core)
from knowlytix.harness.settings import GMSHSettings

# Public API declaration. Anything not listed here is internal and may change
# without notice. `from knowlytix.harness.testing import *` yields exactly these symbols.
__all__ = [
    # Core benchmark entry points
    "DOEGMSBenchmark",
    "DOEHarnessConfig",
    "GMSHSettings",
    # Bridge helpers
    "adapt_question",
    "generate_base_questions",
    "scenario_to_fsb_question",
    # Evaluators
    "AgentEvaluator",
    "HumanInteractionEvaluator",
    "InteractionClaim",
    "LLMEvaluator",
    "ProcessClaim",
    "ProcessEvaluator",
    "RecoveryClaim",
    "RecoveryEvaluator",
    "make_evaluator",
    # Classic + geometric generators
    "HolonomyGenerator",
    "LinkPredictionGenerator",
    "MultiHopTransportGenerator",
    "PhaseComparisonGenerator",
    "PlausibilityGenerator",
    "TensionGenerator",
    "geometric_generators",
    # Judges + verdicts
    "GMSJudge",
    "GeometricVerdict",
    "TypedVerdict",
    "create_judge",
    # Tracing + reporting
    "DOETracingGMS",
    "generate_comparison_report",
    "generate_enhanced_report",
    "generate_report",
    # Claim compiler + verifiers
    "ArtifactClaim",
    "ArtifactVerifier",
    "ClaimBase",
    "ClaimCompiler",
    "ClaimProvenance",
    "FactClaim",
    "GraphVerifier",
    "LogicVerifier",
    "NumericVerifier",
    "PolicyVerifier",
    "RuleClaim",
    "SemanticClaim",
    "SubVerdict",
    "ThresholdClaim",
    "ToolUseClaim",
    "VerificationRouter",
    # Semantic verifier
    "DEFAULT_RUBRIC",
    "GMSEvidence",
    "RubricDimension",
    "SemanticScore",
    "SemanticVerifier",
    # Failure taxonomy
    "FailureAttribution",
    "FailureCode",
    "FailureSeverity",
    # Provenance + truth
    "ArtifactTruth",
    "DocumentRecord",
    "EntityTruth",
    "NormalizedDocument",
    "ProvenanceRegistry",
    "RelationTruth",
    "SectionSpan",
    "ThresholdTruth",
    "TruthBase",
    "TruthStore",
    # SUT adapters + test generators
    "AgentAdapter",
    "ExactRetrievalGenerator",
    "LLMAdapter",
    "NegativeGenerator",
    "RAGAdapter",
    "SUTAdapter",
    "SUTTrace",
    "ToolCallRecord",
    "WorkflowComplianceGenerator",
    "extended_generators",
    "make_adapter",
    # Versioning + snapshots
    "ArtifactRegistry",
    "FaultProfileRegistry",
    "FixtureRegistry",
    "GMSSnapshot",
    "PolicyRuleRegistry",
    "SnapshotManager",
    "VersionedArtifact",
    # Audit + release gating
    "AuditPacket",
    "AuditReporter",
    "ReleaseGate",
    "RiskTierProfile",
    "RunTrend",
    "TieredReleaseGate",
    # Agentic extensions
    "FaultProfile",
    "GatewayDecision",
    "GatewayTranscript",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyRule",
    "RunManager",
    "RunRecord",
    "RunState",
    "Scenario",
    "ScenarioRegistry",
    "StopCondition",
    "ToolGateway",
    # Replay + regression
    "RegressionSuite",
    "ReplayManifest",
    "RunComparison",
    "compare_runs",
    "replay_run",
    # Black-box foundation
    "ClockControl",
    "DataFixture",
    "EnvironmentSimulator",
    "FailureMode",
    "IsolationRequirements",
    "ScheduledEvent",
    "StateTransition",
    "TargetProfile",
    "ToolEndpointSpec",
    # Severity
    "SeverityAssessment",
    "SeverityClassifier",
    "SeverityLevel",
    # Oracles
    "AcceptableSetOracle",
    "CompositeOracle",
    "ConstraintOracle",
    "ExactOutcomeOracle",
    "HumanReviewedOracle",
    "OracleBase",
    "OracleVerdict",
    "SideEffectOracle",
    "TraceOracle",
    "make_oracle",
    # Campaigns
    "CampaignConfig",
    "CampaignManager",
    "CampaignResult",
    "CampaignRunSummary",
    "CampaignType",
    # Taxonomy generators + safety
    "AdversarialGenerator",
    "BoundaryGenerator",
    "GovernanceTestGenerator",
    "RecoveryTestGenerator",
    "RobustnessGenerator",
    "SafetyConfig",
    "SafetyEnforcer",
    "SyntheticDataFramework",
    "taxonomy_generators",
    # Multi-turn
    "InteractionScript",
    "MultiTurnAdapter",
    "MultiTurnScenarioRunner",
    "ScriptResult",
    "TurnResult",
    "TurnSpec",
    # Parallel + browser
    "BrowserAdapter",
    "BrowserConfig",
    "ParallelConfig",
    "ParallelExecutor",
    "UIScript",
    "UIScriptResult",
    "UIStep",
    "UIStepResult",
    "merge_parallel_results",
    # Profiling + cost
    "CostRecord",
    "CostSummary",
    "CostTracker",
    "PerformanceProfiler",
    "PerformanceRecord",
    "PerformanceReport",
    # Stateful + hallucination + calibration
    "AgentMemoryProbe",
    "CalibrationGenerator",
    "CalibrationPoint",
    "CalibrationReport",
    "CalibrationTester",
    "HallucinationGenerator",
    "HallucinationOracle",
    "MemoryAssessment",
    "MemoryRecallOracle",
    "StatefulAgentGenerator",
    "StatefulAgentTester",
    # Multi-source + multi-agent
    "AgentRole",
    "AmbiguousEntity",
    "ConsensusOracle",
    "CrossDocumentAnalyzer",
    "CrossDocumentGenerator",
    "DisambiguationGenerator",
    "DisambiguationOracle",
    "DocumentConflict",
    "InteractionResult",
    "MultiAgentGenerator",
    # Robustness + streaming
    "InvarianceGenerator",
    "InvarianceResult",
    "InvarianceTester",
    "StreamChunk",
    "StreamConsistencyOracle",
    "StreamingAdapter",
    "StreamingTrace",
    # Observability
    "DashboardServer",
    "DashboardSnapshot",
]
