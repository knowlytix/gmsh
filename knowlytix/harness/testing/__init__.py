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

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("knowlytix-harness")
except _PkgNotFound:  # source tree (dev install is the `gms` distribution)
    __version__ = "0.0.0.dev0"

# Flat env-driven settings (lightweight; does not pull harness.core)
from knowlytix.harness.settings import GMSHSettings

# SUT Surface: Extended SUT Adapters + Test Generators
from .adapters import (
    AgentAdapter,
    LLMAdapter,
    RAGAdapter,
    SUTAdapter,
    SUTTrace,
    ToolCallRecord,
    make_adapter,
)

# Audit + Reporting: + Enhanced Reporting
from .audit import (
    AuditPacket,
    AuditReporter,
    ReleaseGate,
    RiskTierProfile,
    RunTrend,
    TieredReleaseGate,
)
from .benchmark import DOEGMSBenchmark
from .bridge import adapt_question, generate_base_questions, scenario_to_fsb_question
from .browser_adapter import (
    BrowserAdapter,
    BrowserConfig,
    UIScript,
    UIScriptResult,
    UIStep,
    UIStepResult,
)
from .calibration import (
    CalibrationGenerator,
    CalibrationPoint,
    CalibrationReport,
    CalibrationTester,
)
from .campaign import (
    CampaignConfig,
    CampaignManager,
    CampaignResult,
    CampaignRunSummary,
    CampaignType,
)

# Typed Verification: Claim Compiler + Typed Verifiers + Failure Taxonomy
from .claims import (
    ArtifactClaim,
    ArtifactClaimSchema,
    ClaimBase,
    ClaimCompiler,
    ClaimExtractor,
    ClaimProvenance,
    ClaimSchema,
    FactClaim,
    FactClaimSchema,
    GeometricClaimExtractor,
    LLMClaimExtractor,
    RuleBasedClaimExtractor,
    RuleClaim,
    RuleClaimSchema,
    SemanticClaim,
    StructuredClaimExtractor,
    ThresholdClaim,
    ThresholdClaimSchema,
    ToolUseClaim,
)
from .completeness import CompletenessEvaluator, CompletenessReport
from .config import DOEHarnessConfig
from .cost_tracker import (
    CostRecord,
    CostSummary,
    CostTracker,
)
from .cross_document import (
    CrossDocumentAnalyzer,
    CrossDocumentGenerator,
    DocumentConflict,
)
from .dashboard import (
    generate_comparison_report,
    generate_enhanced_report,
    generate_report,
)
from .disambiguation import (
    AmbiguousEntity,
    DisambiguationGenerator,
    DisambiguationOracle,
)
from .environment import (
    ClockControl,
    DataFixture,
    EnvironmentSimulator,
    FailureMode,
    ScheduledEvent,
    StateTransition,
    ToolEndpointSpec,
)
from .evaluators import AgentEvaluator, LLMEvaluator, make_evaluator
from .evaluators_extended import (
    HumanInteractionEvaluator,
    InteractionClaim,
    ProcessClaim,
    ProcessEvaluator,
    RecoveryClaim,
    RecoveryEvaluator,
)
from .failures import FailureAttribution, FailureCode, FailureSeverity

# GMSH Agentic Harness Extensions
from .gateway import FaultProfile, GatewayDecision, GatewayTranscript, ToolGateway
from .generators import (
    ConditionalGenerator,
    HolonomyGenerator,
    LinkPredictionGenerator,
    MultiHopTransportGenerator,
    PhaseComparisonGenerator,
    PlausibilityGenerator,
    ProceduralGenerator,
    TensionGenerator,
    geometric_generators,
)

# Black-Box Testing — Coverage
from .generators_taxonomy import (
    AdversarialGenerator,
    BoundaryGenerator,
    GovernanceTestGenerator,
    RecoveryTestGenerator,
    RobustnessGenerator,
    taxonomy_generators,
)
from .hallucination import (
    HallucinationGenerator,
    HallucinationOracle,
    geodesic_band,
)

# GMS-Native Enhancements — Robustness & Invariance
from .invariance import (
    InvarianceGenerator,
    InvarianceResult,
    InvarianceTester,
)
from .judge import GeometricVerdict, GMSJudge, create_judge

# GMS-Native Enhancements — Observability
from .live_dashboard import DashboardServer, DashboardSnapshot

# GMS-Native Enhancements — Multi-Source & Multi-Agent
from .multi_agent import (
    AgentRole,
    ConsensusOracle,
    InteractionResult,
    MultiAgentGenerator,
)
from .multiturn import (
    InteractionScript,
    MultiTurnAdapter,
    MultiTurnScenarioRunner,
    ScriptResult,
    TurnResult,
    TurnSpec,
)
from .oracles import (
    DEFAULT_COMPOSITE_ORACLE_WEIGHTS,
    AcceptableSetOracle,
    CompositeOracle,
    ConstraintOracle,
    ExactOutcomeOracle,
    HumanReviewedOracle,
    OracleBase,
    OracleVerdict,
    SideEffectOracle,
    TraceOracle,
    load_composite_oracle_weights,
    make_oracle,
)

# Black-Box Testing — Scale & Polish
from .parallel import ParallelConfig, ParallelExecutor, merge_parallel_results
from .policy import PolicyDecision, PolicyEngine, PolicyRule

# GMS-Native Enhancements — Profiling & Cost
from .profiling import (
    PerformanceProfiler,
    PerformanceRecord,
    PerformanceReport,
)

# Provenance + Truth: Provenance Chain + Truth Objects
from .provenance import (
    DocumentRecord,
    NormalizedDocument,
    ProvenanceRegistry,
    SectionSpan,
)
from .replay import (
    RegressionSuite,
    ReplayManifest,
    RunComparison,
    compare_runs,
    replay_run,
)
from .runs import RunManager, RunRecord, RunState
from .safety import SafetyConfig, SafetyEnforcer, SyntheticDataFramework
from .scenarios import Scenario, ScenarioRegistry, StopCondition
from .semantic import (
    DEFAULT_RUBRIC,
    GMSEvidence,
    RubricDimension,
    SemanticScore,
    SemanticVerifier,
)
from .severity import SeverityAssessment, SeverityClassifier, SeverityLevel

# Versioning + Snapshots: + Snapshots
from .snapshots import GMSSnapshot, SnapshotManager

# GMS-Native Enhancements — Agent Memory & Behavioral Testing
from .stateful import (
    AgentMemoryProbe,
    MemoryAssessment,
    MemoryRecallOracle,
    StatefulAgentGenerator,
    StatefulAgentTester,
)
from .streaming import (
    StreamChunk,
    StreamConsistencyOracle,
    StreamingAdapter,
    StreamingTrace,
)

# Black-Box Testing — Foundation
from .target import IsolationRequirements, TargetProfile
from .test_generators import (
    ExactRetrievalGenerator,
    NegativeGenerator,
    WorkflowComplianceGenerator,
    extended_generators,
)
from .tracing import DOETracingGMS
from .triple_completeness import (
    TripleCompletenessReport,
    TripleCompletenessScorer,
    triples_from_rag_answer,
)
from .truth import (
    ArtifactTruth,
    EntityTruth,
    RelationTruth,
    ThresholdTruth,
    TruthBase,
    TruthStore,
)
from .verdict import (
    DEFAULT_OVERALL_CONFIDENCE_WEIGHTS,
    TypedVerdict,
    load_overall_confidence_weights,
)
from .verifiers import (
    ArtifactVerifier,
    GraphVerifier,
    LogicVerifier,
    NumericVerifier,
    PolicyVerifier,
    SubVerdict,
    VerificationRouter,
)
from .versioning import (
    ArtifactRegistry,
    FaultProfileRegistry,
    FixtureRegistry,
    PolicyRuleRegistry,
    VersionedArtifact,
)

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
    "CompletenessReport",
    "CompletenessEvaluator",
    "TripleCompletenessReport",
    "TripleCompletenessScorer",
    "triples_from_rag_answer",
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
    "GeometricClaimExtractor",
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
