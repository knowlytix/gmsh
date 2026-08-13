# SPDX-License-Identifier: Apache-2.0
"""knowlytix.harness.governance — high-stakes runtime governance over a GMS store.

Three-layer architecture for deploying geometric entity-comparison
verdicts into high-stakes settings:

  * **Offline validation**  — produces a frozen, signed
    :class:`GovernanceBundle` carrying bootstrap-calibrated thresholds,
    isotonic ``te → P(contra)`` mapping, support index,
    declared-opposition table, drift baseline, store fingerprint,
    calibrated support floors, and an operating-point SLO.
  * **Runtime gate**        — :class:`RuntimeGate` consumes a bundle
    and produces bounded-latency, deterministic verdicts with full
    reason chains, two-attestor concurrence, and telemetry.
  * **Live monitoring**     — :class:`DriftMonitor` consumes runtime
    telemetry, detects PSI drift, abstain-rate shifts, and SLO
    violations, and signals when offline recalibration is needed.

Top-level imports (``from knowlytix.harness.governance import X``):
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _dist_version

# --- Adversarial sweep --------------------------------------------------- #
from knowlytix.harness.governance.adversarial_sweep import (
    AdversarialPair,
    AdversarialReport,
    build_adversarial_cohort,
    sweep_adversarial,
)

# --- Multi-agent governance hardening (Ch 17) --------------------------- #
from knowlytix.harness.governance.agents.multi_agent import (
    DATA_AGENT_ROLE,
    DIAGNOSTICS_AGENT_ROLE,
    FAIRNESS_AGENT_ROLE,
    TRAINING_AGENT_ROLE,
    AgentRole,
    AgentStats,
    GovernedAgent,
    HandoffRecord,
    MultiAgentOrchestrator,
    PipelineResult,
    ToolCallRecord,
)
from knowlytix.harness.governance.agents.multi_agent_adversarial import (
    AdversarialMultiAgentProbe,
    MultiAgentAdversarialReport,
    build_multi_agent_adversarial_cohort,
    sweep_multi_agent_adversarial,
)
from knowlytix.harness.governance.agents.multi_agent_audit import (
    MultiAgentAuditRecord,
    MultiAgentReasonChain,
    replay_multi_agent_audit_log,
    to_multi_agent_audit_record,
)
from knowlytix.harness.governance.agents.multi_agent_baselines import (
    MultiAgentBaselineContrastReport,
    NoPermissionCheckBaseline,
    PromptOnlyBaseline,
    SingleAgentBaseline,
    compare_to_orchestrator,
)
from knowlytix.harness.governance.agents.multi_agent_bench import (
    MultiAgentLatencyReport,
    bench_can_execute,
    bench_handoff,
    bench_test_boundary,
)
from knowlytix.harness.governance.agents.multi_agent_bundle import (
    MULTI_AGENT_BUNDLE_SCHEMA_VERSION,
    MultiAgentBundle,
    MultiAgentOperatingPoint,
    bind_orchestrator_from_bundle,
    build_multi_agent_bundle,
)
from knowlytix.harness.governance.agents.multi_agent_calibration import (
    HandoffLatencyBaseline,
    PerAgentToolFrequency,
    PerRoleFingerprint,
    bootstrap_handoff_latency,
    calibrate_handoff_latency_baseline,
    calibrate_per_agent_tool_frequency,
    fingerprint_orchestrator,
    fingerprint_role,
)
from knowlytix.harness.governance.agents.multi_agent_co_attestor import (
    MultiAgentCoAttestor,
    MultiAgentCoAttestorConfig,
    MultiAgentCoAttestorVerdict,
)
from knowlytix.harness.governance.agents.multi_agent_invariants import (
    MultiAgentInvariantReport,
    MultiAgentInvariantViolation,
    verify_multi_agent_invariants,
)
from knowlytix.harness.governance.agents.multi_agent_monitor import (
    MultiAgentMonitor,
    MultiAgentMonitorState,
)
from knowlytix.harness.governance.agents.multi_agent_replay import (
    MultiAgentReplayDelta,
    MultiAgentReplayRegressionError,
    MultiAgentReplayResult,
    assert_no_multi_agent_regression,
    compare_multi_agent_replays,
    replay_multi_agent_battery,
)
from knowlytix.harness.governance.agents.multi_agent_replay import (
    read_blessed_hash as read_multi_agent_blessed_hash,
)
from knowlytix.harness.governance.agents.multi_agent_replay import (
    write_blessed_hash as write_multi_agent_blessed_hash,
)
from knowlytix.harness.governance.agents.multi_agent_validation import (
    LabeledMultiAgentDecision,
    MultiAgentOODReport,
    adult_income_multi_agent_cohort,
    large_adult_income_multi_agent_cohort,
    validate_multi_agent,
)
from knowlytix.harness.governance.agents.multi_agent_validation import (
    cohort_fingerprint as multi_agent_cohort_fingerprint,
)
from knowlytix.harness.governance.agents.multi_agent_validation import (
    load_cohort as load_multi_agent_cohort,
)
from knowlytix.harness.governance.agents.multi_agent_validation import (
    save_cohort as save_multi_agent_cohort,
)
from knowlytix.harness.governance.agents.multi_agent_validator_alt import (
    MultiAgentAltAgreementReport,
    alt_can_execute,
    assert_orchestrator_alt_agreement,
)

# --- Bundle integrity / provenance --------------------------------------- #
from knowlytix.harness.governance.bundle_signing import (
    SIGNING_KEY_ENV_VAR,
    BundleSigningError,
    resolve_signing_key,
)

# --- Two-attestor policy ------------------------------------------------- #
from knowlytix.harness.governance.comparison_policy import (
    ComparisonPolicy,
    ComparisonPolicyVerdict,
    ComparisonRule,
    RuleResult,
    declared_opposition_rule,
    default_comparison_policy,
    functional_relation_rule,
    shared_target_rule,
)

# --- Drift detection ----------------------------------------------------- #
from knowlytix.harness.governance.drift_monitor import (
    DriftMonitor,
    DriftState,
    population_stability_index,
)

# --- Offline artifact bundle --------------------------------------------- #
from knowlytix.harness.governance.governance_bundle import (
    BUNDLE_SCHEMA_VERSION,
    BundleSignatureError,
    EntitySupport,
    GovernanceBundle,
    OperatingPoint,
    ThresholdDistribution,
)
from knowlytix.harness.governance.governed_gateway import GovernedToolGateway
from knowlytix.harness.governance.llm_adversarial import (
    LLMAdversarialProbe,
    LLMAdversarialReport,
    build_llm_adversarial_cohort,
    sweep_llm_adversarial,
)
from knowlytix.harness.governance.llm_bundle import (
    LLM_BUNDLE_SCHEMA_VERSION,
    ConstrainedLLMBundle,
    LLMOperatingPoint,
    bind_constrained_llm_from_bundle,
    build_constrained_llm_bundle,
)

# --- Constrained-LLM hardening (Ch 11) ---------------------------------- #
from knowlytix.harness.governance.llm_co_attestor import (
    LLMCoAttestor,
    default_ds_co_attestor,
)
from knowlytix.harness.governance.memory import AgentMemory

# --- Naive-baseline contrast -------------------------------------------- #
from knowlytix.harness.governance.naive_baseline import (
    Baseline,
    BaselineContrastReport,
    JaccardBaseline,
    NaiveOppositionBaseline,
    compare_to_gate,
)

# --- Bundle production --------------------------------------------------- #
from knowlytix.harness.governance.offline_validator import (
    bootstrap_thresholds,
    build_bundle,
    fit_isotonic,
)

# --- OOD held-out validation -------------------------------------------- #
from knowlytix.harness.governance.ood_validation import (
    LabeledPair,
    OODValidationReport,
    shift_samples,
    validate_calibration,
    validate_gate,
)

# --- Orchestrator + monitor --------------------------------------------- #
from knowlytix.harness.governance.orchestrator import (
    GovernedOrchestrator,
    OrchestratorResult,
    RunManager,
    RunState,
    Scenario,
    SignalVector,
)
from knowlytix.harness.governance.orchestrator_adversarial import (
    AdversarialPlan as OrchestratorAdversarialPlan,
)
from knowlytix.harness.governance.orchestrator_adversarial import (
    AdversarialReport as OrchestratorAdversarialReport,
)
from knowlytix.harness.governance.orchestrator_adversarial import (
    build_orchestrator_adversarial_cohort,
    sweep_orchestrator_adversarial,
)
from knowlytix.harness.governance.orchestrator_baselines import (
    BaselineComparisonReport,
    JaccardClaimVerifier,
    RandomClaimVerifier,
    compare_to_baseline,
)
from knowlytix.harness.governance.orchestrator_monitor import (
    OrchestratorMonitor,
    OrchestratorMonitorState,
    RunSnapshot,
)
from knowlytix.harness.governance.orchestrator_policy import (
    OrchestratorPolicy,
    OrchestratorPolicyVerdict,
    OrchestratorRule,
    default_orchestrator_rules,
)
from knowlytix.harness.governance.orchestrator_validation import (
    LabeledPlan,
    OrchestratorOODReport,
    ReplayBatteryResult,
    cohort_hash,
    compare_replay_batteries,
    default_labeled_cohort,
    orchestrator_replay_battery,
    orchestrator_snapshot,
    validate_orchestrator,
)

# --- Bootstrap-calibrated plausibility threshold + two-attestor --------- #
from knowlytix.harness.governance.plausibility_calibration import (
    GraphPresenceResult,
    TwoAttestorVerdict,
    bootstrap_plausibility_threshold,
    check_graph_presence,
    verify_fact_two_attestor,
)
from knowlytix.harness.governance.policy_adversarial import (
    AdversarialPolicyProbe,
    AdversarialPolicyReport,
    build_policy_adversarial_cohort,
    sweep_policy_adversarial,
)
from knowlytix.harness.governance.policy_audit import (
    IncidentTriple,
    PolicyAuditRecord,
    PolicyReasonChain,
    build_policy_reason_chain,
    replay_audit_log,
    to_audit_record,
)
from knowlytix.harness.governance.policy_baselines import (
    BaselineContrastReport as PolicyBaselineContrastReport,
)
from knowlytix.harness.governance.policy_baselines import (
    BaselineDecision,
    JaccardPolicyBaseline,
    PromptGuardrailBaseline,
    compare_to_engine,
)
from knowlytix.harness.governance.policy_bench import (
    DriftSimulationReport,
    LatencyReport,
    bench_check_with_quorum,
    simulate_drift,
)
from knowlytix.harness.governance.policy_bundle import (
    POLICY_BUNDLE_SCHEMA_VERSION,
    PolicyBundle,
    PolicyOperatingPoint,
    bind_engine_from_bundle,
    build_policy_bundle,
)

# --- Policy hardening (Ch 12) ------------------------------------------- #
from knowlytix.harness.governance.policy_calibration import (
    PolicyAttestorVerdict,
    bootstrap_policy_geodesic_empirical,
    bootstrap_policy_geodesic_threshold,
    bootstrap_policy_tau_contra,
    check_with_quorum,
    derive_attestor_verdict,
)
from knowlytix.harness.governance.policy_invariants import (
    InvariantReport,
    InvariantViolation,
    verify_quorum_invariants,
    verify_replay_symmetry,
)
from knowlytix.harness.governance.policy_monitor import (
    PolicyMonitor,
    PolicyMonitorState,
)
from knowlytix.harness.governance.policy_multisign import (
    AttestorSignature,
    QuorumSignatureError,
    QuorumVerifyResult,
    assert_quorum,
    attach_signatures,
    sign_for,
    verify_quorum,
)
from knowlytix.harness.governance.policy_numeric import (
    NumericObservation,
    NumericObservationLog,
    NumericRangeVerdict,
    adult_income_numeric_observations,
    build_numeric_range_index,
    check_numeric_range,
)
from knowlytix.harness.governance.policy_replay import (
    PolicyReplayDelta,
    PolicyReplayRegressionError,
    PolicyReplayResult,
    assert_no_policy_regression,
    compare_policy_replays,
    policy_decision_snapshot,
    replay_policy_battery,
)
from knowlytix.harness.governance.policy_validation import (
    LabeledDecision,
    PolicyOODReport,
    adult_income_labeled_cohort,
    cohort_fingerprint,
    large_adult_income_cohort,
    load_cohort,
    save_cohort,
    validate_policy,
)
from knowlytix.harness.governance.policy_validator_alt import (
    AgreementReport,
    AlternativeVerdict,
    alt_check,
    alt_check_from_audit_record,
    assert_engine_alt_agreement,
)
from knowlytix.harness.governance.reason_chain import (
    ReasonChain,
    build_reason_chain,
)
from knowlytix.harness.governance.release_gate_adversarial import (
    AdversarialReleaseGateReport,
    AdversarialRunProbe,
    build_release_gate_adversarial_cohort,
    sweep_release_gate_adversarial,
)
from knowlytix.harness.governance.release_gate_audit import (
    ReleaseGateAuditRecord,
    ReleaseGateReasonChain,
    build_release_gate_reason_chain,
)
from knowlytix.harness.governance.release_gate_audit import (
    replay_audit_log as replay_release_gate_audit_log,
)
from knowlytix.harness.governance.release_gate_audit import (
    to_audit_record as to_release_gate_audit_record,
)
from knowlytix.harness.governance.release_gate_baselines import (
    BaselineContrastReport as ReleaseGateBaselineContrastReport,
)
from knowlytix.harness.governance.release_gate_baselines import (
    FixedThresholdBaseline,
    RandomBaseline,
)
from knowlytix.harness.governance.release_gate_baselines import (
    compare_to_gate as compare_to_release_gate,
)
from knowlytix.harness.governance.release_gate_bench import (
    ReleaseGateLatencyReport,
    bench_audit_record,
    bench_check_all_tiers,
    bench_co_attestor,
)
from knowlytix.harness.governance.release_gate_bundle import (
    RELEASE_GATE_BUNDLE_SCHEMA_VERSION,
    ReleaseGateBundle,
    ReleaseGateOperatingPoint,
    bind_release_gate_from_bundle,
    build_release_gate_bundle,
)
from knowlytix.harness.governance.release_gate_calibration import (
    CalibratedThresholdSet,
    bootstrap_metric_threshold,
    calibrate_release_gate_thresholds,
    enforce_monotonic_ordering,
    policy_declared_baseline,
)
from knowlytix.harness.governance.release_gate_co_attestor import (
    CoAttestorVerdict,
    SPCSlopeCoAttestor,
    SPCSlopeConfig,
)
from knowlytix.harness.governance.release_gate_invariants import (
    InvariantReport as ReleaseGateInvariantReport,
)
from knowlytix.harness.governance.release_gate_invariants import (
    InvariantViolation as ReleaseGateInvariantViolation,
)
from knowlytix.harness.governance.release_gate_invariants import (
    verify_release_gate_invariants,
)
from knowlytix.harness.governance.release_gate_monitor import (
    ReleaseGateMonitor,
    ReleaseGateMonitorState,
)
from knowlytix.harness.governance.release_gate_monitor import (
    population_stability_index as release_gate_psi,
)
from knowlytix.harness.governance.release_gate_replay import (
    ReleaseGateReplayDelta,
    ReleaseGateReplayRegressionError,
    ReleaseGateReplayResult,
    assert_no_release_gate_regression,
    compare_release_gate_replays,
    replay_release_gate_battery,
)
from knowlytix.harness.governance.release_gate_replay import (
    read_blessed_hash as read_release_gate_blessed_hash,
)
from knowlytix.harness.governance.release_gate_replay import (
    write_blessed_hash as write_release_gate_blessed_hash,
)
from knowlytix.harness.governance.release_gate_validation import (
    LabeledRun,
    ReleaseGateOODReport,
    adult_income_labeled_run_cohort,
    large_adult_income_run_cohort,
    relabel_cohort_against_gate,
    validate_release_gate,
)
from knowlytix.harness.governance.release_gate_validation import (
    cohort_fingerprint as release_gate_cohort_fingerprint,
)
from knowlytix.harness.governance.release_gate_validation import (
    load_cohort as load_release_gate_cohort,
)
from knowlytix.harness.governance.release_gate_validation import (
    save_cohort as save_release_gate_cohort,
)
from knowlytix.harness.governance.release_gate_validator_alt import (
    AltAgreementReport as ReleaseGateAltAgreementReport,
)
from knowlytix.harness.governance.release_gate_validator_alt import (
    alt_check as alt_check_release_gate,
)
from knowlytix.harness.governance.release_gate_validator_alt import (
    assert_gate_alt_agreement,
)

# --- Release-gate hardening (Ch 15) ------------------------------------- #
from knowlytix.harness.governance.release_gates import (
    GateResult,
    GateThresholds,
    RunMetrics,
    RunTrend,
    TieredReleaseGate,
)

# --- Bundle-replay regression ------------------------------------------- #
from knowlytix.harness.governance.replay import (
    ReplayCohort,
    ReplayDelta,
    ReplayRegressionError,
    ReplayResult,
    assert_no_regression,
    compare_replays,
    replay_bundle,
)

# --- Runtime gate -------------------------------------------------------- #
from knowlytix.harness.governance.runtime_gate import (
    RuntimeGate,
    RuntimeVerdict,
    serve,
)
from knowlytix.harness.governance.session import SessionState
from knowlytix.harness.governance.store_fingerprint import (
    StoreFingerprint,
    StoreFingerprintMismatch,
    compute_store_fingerprint,
)

# --- Calibrated support floors ------------------------------------------ #
from knowlytix.harness.governance.support_calibration import (
    CalibratedSupportThresholds,
    calibrate_support_thresholds,
)
from knowlytix.harness.governance.verifier import ClaimVerifier

try:
    __version__ = _dist_version("knowlytix-harness")
except _PkgNotFound:  # source tree (dev install is the `gms` distribution)
    __version__ = "0.0.0.dev0"

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BundleSignatureError",
    "BundleSigningError",
    "EntitySupport",
    "GovernanceBundle",
    "OperatingPoint",
    "SIGNING_KEY_ENV_VAR",
    "ThresholdDistribution",
    "resolve_signing_key",
    "bootstrap_thresholds",
    "build_bundle",
    "fit_isotonic",
    "StoreFingerprint",
    "StoreFingerprintMismatch",
    "compute_store_fingerprint",
    "ReasonChain",
    "RuntimeGate",
    "RuntimeVerdict",
    "build_reason_chain",
    "serve",
    "ComparisonPolicy",
    "ComparisonPolicyVerdict",
    "ComparisonRule",
    "RuleResult",
    "declared_opposition_rule",
    "default_comparison_policy",
    "functional_relation_rule",
    "shared_target_rule",
    "LabeledPair",
    "OODValidationReport",
    "shift_samples",
    "validate_calibration",
    "validate_gate",
    "AdversarialPair",
    "AdversarialReport",
    "build_adversarial_cohort",
    "sweep_adversarial",
    "DriftMonitor",
    "DriftState",
    "population_stability_index",
    "CalibratedSupportThresholds",
    "calibrate_support_thresholds",
    "GraphPresenceResult",
    "TwoAttestorVerdict",
    "bootstrap_plausibility_threshold",
    "check_graph_presence",
    "verify_fact_two_attestor",
    "OrchestratorMonitor",
    "OrchestratorMonitorState",
    "RunSnapshot",
    "LabeledPlan",
    "OrchestratorOODReport",
    "ReplayBatteryResult",
    "cohort_hash",
    "compare_replay_batteries",
    "default_labeled_cohort",
    "orchestrator_replay_battery",
    "orchestrator_snapshot",
    "validate_orchestrator",
    "OrchestratorAdversarialPlan",
    "OrchestratorAdversarialReport",
    "build_orchestrator_adversarial_cohort",
    "sweep_orchestrator_adversarial",
    "BaselineComparisonReport",
    "JaccardClaimVerifier",
    "RandomClaimVerifier",
    "compare_to_baseline",
    "OrchestratorPolicy",
    "OrchestratorPolicyVerdict",
    "OrchestratorRule",
    "default_orchestrator_rules",
    "GovernedOrchestrator",
    "OrchestratorResult",
    "Scenario",
    "SignalVector",
    "RunState",
    "RunManager",
    "SessionState",
    "GovernedToolGateway",
    "ClaimVerifier",
    "AgentMemory",
    "ReplayCohort",
    "ReplayDelta",
    "ReplayRegressionError",
    "ReplayResult",
    "assert_no_regression",
    "compare_replays",
    "replay_bundle",
    "Baseline",
    "BaselineContrastReport",
    "JaccardBaseline",
    "NaiveOppositionBaseline",
    "compare_to_gate",
    # --- Policy hardening (Ch 12) ---
    "PolicyAttestorVerdict",
    "bootstrap_policy_geodesic_empirical",
    "bootstrap_policy_geodesic_threshold",
    "bootstrap_policy_tau_contra",
    "check_with_quorum",
    "derive_attestor_verdict",
    "LabeledDecision",
    "PolicyOODReport",
    "adult_income_labeled_cohort",
    "cohort_fingerprint",
    "large_adult_income_cohort",
    "load_cohort",
    "save_cohort",
    "validate_policy",
    "AdversarialPolicyProbe",
    "AdversarialPolicyReport",
    "build_policy_adversarial_cohort",
    "sweep_policy_adversarial",
    "POLICY_BUNDLE_SCHEMA_VERSION",
    "PolicyBundle",
    "PolicyOperatingPoint",
    "bind_engine_from_bundle",
    "build_policy_bundle",
    "PolicyReplayDelta",
    "PolicyReplayRegressionError",
    "PolicyReplayResult",
    "assert_no_policy_regression",
    "compare_policy_replays",
    "policy_decision_snapshot",
    "replay_policy_battery",
    "PolicyBaselineContrastReport",
    "BaselineDecision",
    "JaccardPolicyBaseline",
    "PromptGuardrailBaseline",
    "compare_to_engine",
    "PolicyMonitor",
    "PolicyMonitorState",
    "IncidentTriple",
    "PolicyAuditRecord",
    "PolicyReasonChain",
    "build_policy_reason_chain",
    "replay_audit_log",
    "to_audit_record",
    "NumericObservation",
    "NumericObservationLog",
    "NumericRangeVerdict",
    "adult_income_numeric_observations",
    "build_numeric_range_index",
    "check_numeric_range",
    "InvariantReport",
    "InvariantViolation",
    "verify_quorum_invariants",
    "verify_replay_symmetry",
    "AgreementReport",
    "AlternativeVerdict",
    "alt_check",
    "alt_check_from_audit_record",
    "assert_engine_alt_agreement",
    "AttestorSignature",
    "QuorumSignatureError",
    "QuorumVerifyResult",
    "assert_quorum",
    "attach_signatures",
    "sign_for",
    "verify_quorum",
    "DriftSimulationReport",
    "LatencyReport",
    "bench_check_with_quorum",
    "simulate_drift",
    # --- Release-gate hardening (Ch 15) ---
    "GateResult",
    "GateThresholds",
    "RunMetrics",
    "RunTrend",
    "TieredReleaseGate",
    "CalibratedThresholdSet",
    "bootstrap_metric_threshold",
    "calibrate_release_gate_thresholds",
    "enforce_monotonic_ordering",
    "policy_declared_baseline",
    "LabeledRun",
    "ReleaseGateOODReport",
    "adult_income_labeled_run_cohort",
    "release_gate_cohort_fingerprint",
    "large_adult_income_run_cohort",
    "load_release_gate_cohort",
    "relabel_cohort_against_gate",
    "save_release_gate_cohort",
    "validate_release_gate",
    "AdversarialReleaseGateReport",
    "AdversarialRunProbe",
    "build_release_gate_adversarial_cohort",
    "sweep_release_gate_adversarial",
    "CoAttestorVerdict",
    "SPCSlopeCoAttestor",
    "SPCSlopeConfig",
    "ReleaseGateInvariantReport",
    "ReleaseGateInvariantViolation",
    "verify_release_gate_invariants",
    "ReleaseGateBaselineContrastReport",
    "FixedThresholdBaseline",
    "RandomBaseline",
    "compare_to_release_gate",
    "RELEASE_GATE_BUNDLE_SCHEMA_VERSION",
    "ReleaseGateBundle",
    "ReleaseGateOperatingPoint",
    "bind_release_gate_from_bundle",
    "build_release_gate_bundle",
    "ReleaseGateReplayDelta",
    "ReleaseGateReplayRegressionError",
    "ReleaseGateReplayResult",
    "assert_no_release_gate_regression",
    "compare_release_gate_replays",
    "read_release_gate_blessed_hash",
    "replay_release_gate_battery",
    "write_release_gate_blessed_hash",
    "ReleaseGateAuditRecord",
    "ReleaseGateReasonChain",
    "build_release_gate_reason_chain",
    "replay_release_gate_audit_log",
    "to_release_gate_audit_record",
    "ReleaseGateMonitor",
    "ReleaseGateMonitorState",
    "release_gate_psi",
    "ReleaseGateLatencyReport",
    "bench_audit_record",
    "bench_check_all_tiers",
    "bench_co_attestor",
    "ReleaseGateAltAgreementReport",
    "alt_check_release_gate",
    "assert_gate_alt_agreement",
    # --- Multi-agent governance (Ch 17) ---
    "AgentRole",
    "GovernedAgent",
    "MultiAgentOrchestrator",
    "HandoffRecord",
    "PipelineResult",
    "AgentStats",
    "ToolCallRecord",
    "DATA_AGENT_ROLE",
    "TRAINING_AGENT_ROLE",
    "DIAGNOSTICS_AGENT_ROLE",
    "FAIRNESS_AGENT_ROLE",
    "LabeledMultiAgentDecision",
    "MultiAgentOODReport",
    "adult_income_multi_agent_cohort",
    "multi_agent_cohort_fingerprint",
    "large_adult_income_multi_agent_cohort",
    "load_multi_agent_cohort",
    "save_multi_agent_cohort",
    "validate_multi_agent",
    "HandoffLatencyBaseline",
    "PerAgentToolFrequency",
    "PerRoleFingerprint",
    "bootstrap_handoff_latency",
    "calibrate_handoff_latency_baseline",
    "calibrate_per_agent_tool_frequency",
    "fingerprint_orchestrator",
    "fingerprint_role",
    "AdversarialMultiAgentProbe",
    "MultiAgentAdversarialReport",
    "build_multi_agent_adversarial_cohort",
    "sweep_multi_agent_adversarial",
    "MultiAgentCoAttestor",
    "MultiAgentCoAttestorConfig",
    "MultiAgentCoAttestorVerdict",
    "MultiAgentInvariantReport",
    "MultiAgentInvariantViolation",
    "verify_multi_agent_invariants",
    "MultiAgentBaselineContrastReport",
    "NoPermissionCheckBaseline",
    "PromptOnlyBaseline",
    "SingleAgentBaseline",
    "compare_to_orchestrator",
    "MULTI_AGENT_BUNDLE_SCHEMA_VERSION",
    "MultiAgentBundle",
    "MultiAgentOperatingPoint",
    "bind_orchestrator_from_bundle",
    "build_multi_agent_bundle",
    "MultiAgentReplayDelta",
    "MultiAgentReplayRegressionError",
    "MultiAgentReplayResult",
    "assert_no_multi_agent_regression",
    "compare_multi_agent_replays",
    "read_multi_agent_blessed_hash",
    "replay_multi_agent_battery",
    "write_multi_agent_blessed_hash",
    "MultiAgentAuditRecord",
    "MultiAgentReasonChain",
    "replay_multi_agent_audit_log",
    "to_multi_agent_audit_record",
    "MultiAgentMonitor",
    "MultiAgentMonitorState",
    "MultiAgentLatencyReport",
    "bench_can_execute",
    "bench_handoff",
    "bench_test_boundary",
    "MultiAgentAltAgreementReport",
    "alt_can_execute",
    "assert_orchestrator_alt_agreement",
]
