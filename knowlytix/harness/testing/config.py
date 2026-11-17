# SPDX-License-Identifier: Apache-2.0
"""Configuration for the DOE testing harness."""

from dataclasses import dataclass, field

import yaml


@dataclass
class DOEHarnessConfig:
    """Full configuration for a DOE-GMS benchmark run.

    Attributes:
        markdown_path: Path to the source document to ingest.
        stores_dir: Directory holding built GMS stores.
        store_name: Store name; auto-derived from the markdown filename when empty.
        factor_group: Named DOE factor group to sweep.
        factor_names: Explicit factor names to include; empty uses the group.
        n_runs: Number of DOE design points.
        method: DOE sampling method, "sobol", "lhs" or "grid".
        seed: Random seed.
        rephrase_method: Question-rephrasing method, "template" or "llm".
        rephrase_model: LLM used when rephrase_method is "llm".
        max_per_category: Cap on generated questions per category.
        generators: Question generators to run; empty runs all.
        target_type: System under test, "llm" or "agent".
        target_model: Model id for the target under test.
        context_mode: Context supplied to the target, "full", "oracle_section" or "noisy_section".
        tracing_dir: Directory for the tracing store.
        enable_tracing: Whether to record execution traces.
        ingest_epochs: Training epochs for store ingestion.
        ingest_mode: Overrides the DocGMSConfig ingest mode when non-empty ("hybrid", "regex" or "llm_only").
        cap_enabled: Whether to rebuild the store with spherical-cap admissibility.
        cap_rho_max: Maximum cap radius in radians when cap_enabled.
        cap_use_diag: Whether the cap uses a per-relation diagonal scale.
        loss_config_path: YAML path whose loss/cap/train overrides merge before training; empty disables.
        claim_extraction: Claim-extraction mode for typed verdicts, "rule" or "geometric".
        enable_typed_verdicts: Whether to run claim-level verification.
        enable_provenance: Whether to build the document provenance chain.
        snapshot_dir: Directory for GMS snapshots; empty disables.
        semantic_mode: Semantic-judge mode, "off", "fallback" or "complement".
        semantic_model: LLM judge model; empty uses target_model.
        enable_gateway: Whether to install the tool-interception gateway.
        gateway_fault_profiles_path: YAML path defining fault profiles for the gateway.
        gateway_schema_strict: Whether the gateway enforces strict tool-call schemas.
        enable_policy_engine: Whether to run the policy engine.
        policy_rules_path: YAML path defining policy rules.
        scenarios_dir: Directory of scenario YAML files.
        enable_scenario_registry: Whether to load the scenario registry.
        runs_dir: Directory for run records; empty disables.
        regression_suite_dir: Directory holding the regression suite.
        replay_manifest_path: Path to a replay manifest.
        enable_process_evaluation: Whether to run process evaluators.
        enable_recovery_evaluation: Whether to run recovery evaluators.
        enable_interaction_evaluation: Whether to run interaction evaluators.
        risk_tier_profiles_path: YAML path defining risk-tier release profiles.
        baseline_run_id: Prior run id used for regression comparison.
        target_profile_path: YAML path defining a black-box target profile.
        environment_yaml_path: YAML path defining the environment simulator.
        campaigns_dir: Directory for campaign configs and results.
        enable_severity_classification: Whether to classify failure severity.
        safety_config_path: YAML path defining the safety config.
        enable_performance_profiling: Whether to profile latency.
        sla_latency_ms: Latency SLA target in milliseconds.
        enable_cost_tracking: Whether to track token and dollar cost.
        cost_budget_usd: Cost budget cap in USD.
        enable_stateful_testing: Whether to run stateful teach-test sessions.
        stateful_session_count: Number of teach-test sessions.
        enable_hallucination_testing: Whether to run the hallucination oracle.
        hallucination_tau_override: Overrides the calibrated geodesic threshold; None uses calibration.
        enable_calibration_testing: Whether to run confidence-calibration tests.
        enable_multi_agent: Whether to run multi-agent tests.
        multi_agent_roles_path: YAML path defining agent roles.
        enable_cross_document: Whether to run cross-document conflict tests.
        additional_store_paths: Extra store paths for cross-document testing.
        enable_disambiguation_testing: Whether to run entity-disambiguation tests.
        enable_invariance_testing: Whether to run paraphrase-invariance tests.
        enable_streaming_testing: Whether to run streaming-output tests.
        enable_live_dashboard: Whether to serve the live dashboard.
        live_dashboard_port: Port for the live dashboard.
        device: Torch device; empty selects cuda when available.
    """

    # -- Document --
    markdown_path: str = ""
    stores_dir: str = "gms_stores"
    store_name: str = ""  # auto-derived from markdown filename if empty

    # -- DOE --
    factor_group: str = "quick_screen"
    factor_names: list[str] = field(default_factory=list)
    n_runs: int = 32
    method: str = "sobol"  # "sobol", "lhs", "grid"
    seed: int = 42
    rephrase_method: str = "template"  # "template" or "llm"
    rephrase_model: str = "claude-sonnet-4-20250514"

    # -- Question generation --
    max_per_category: int = 10
    generators: list[str] = field(default_factory=list)  # empty = all

    # -- Target under test --
    target_type: str = "llm"  # "llm" or "agent"
    target_model: str = "claude-sonnet-4-20250514"
    context_mode: str = "full"  # "full", "oracle_section", "noisy_section"

    # -- Tracing --
    tracing_dir: str = "doe_tracing_store"
    enable_tracing: bool = True

    # -- Ingestion --
    ingest_epochs: int = 200

    # ``DocGMSConfig.ingest_mode`` -- "hybrid" (default), "regex", or
    # "llm_only". Overrides the DocGMSConfig default when non-empty so a
    # caller can pin byte-deterministic ingest without reaching into
    # DocGMSConfig directly. Empty string = leave the DocGMSConfig
    # default in place. testing-ext sets this to "regex" so the
    # fixtures match the chapter's manual Stage-1 walkthrough.
    ingest_mode: str = ""

    # -- Cap admissibility (paper.tex §7) -------------------------------
    # ``cap_enabled=True`` rebuilds the GMSExpertStore with a relation-
    # conditioned spherical-cap admissibility region rather than a single
    # post-hoc threshold. This is the regime that resolves the 1-N
    # centroid floor on relations with multiple authorised tails (see
    # gmsh_testing_ext/HARDENING_PLAN.md §2.1). Default is False so that
    # build-side callers (governance USER_GUIDE tutorial notebooks at
    # knowlytix/harness/governance/tutorials/notebooks/, finstructbench)
    # keep their existing fixtures byte-stable; testing-ext sets True
    # explicitly in its DOEHarnessConfig usage.
    cap_enabled: bool = False
    cap_rho_max: float = 1.5707963267948966   # pi/2
    cap_use_diag: bool = True                 # matches agent-builder ch04_ds_policy

    # -- Loss / training overrides ---------------------------------------
    # YAML path whose contents merge into the LossConfig, CapLossConfig,
    # and TrainConfig dataclasses before training. Useful for pushing
    # corpus-specific tuning (e.g., higher lambda_tension when the corpus
    # has dense schema declarations and the default 2.0 weight isn't
    # enough to drive contradict pairs past the 3-class calibration gate)
    # without changing library defaults that other fixtures depend on.
    # Empty string = no override; library defaults apply. The YAML is
    # merged via OmegaConf so partial overrides are fine; only the keys
    # present in the file are touched. Schema:
    #
    #     loss:
    #       lambda_tension: 4.0
    #     cap:
    #       lambda_pos: 6.0
    #     train:
    #       epochs: 800
    #
    # Top-level keys must match the GMSConfig schema.
    loss_config_path: str = ""

    # Claim extraction for typed verdicts: "rule" (head/relation from the
    # generator schema, tail scraped from the answer) or "geometric" (parse the
    # answer itself into store-bound triples via GeometricQueryParser and score
    # each against the GMS -- answer-driven verification).
    claim_extraction: str = "rule"

    # -- GMSH enhancements --
    enable_typed_verdicts: bool = False  # claim-level verification
    enable_provenance: bool = False      # document provenance chain
    snapshot_dir: str = ""               # GMS snapshots (empty = off)
    semantic_mode: str = "off"           # "off", "fallback", "complement"
    semantic_model: str = ""             # LLM judge model (empty = target_model)

    # -- Tool Gateway (WS1) --
    enable_gateway: bool = False
    gateway_fault_profiles_path: str = ""   # YAML path for FaultProfile list
    gateway_schema_strict: bool = True

    # -- Policy Engine (WS2) --
    enable_policy_engine: bool = False
    policy_rules_path: str = ""             # YAML path for PolicyRule list

    # -- Scenario Registry (WS3) --
    scenarios_dir: str = ""                 # directory for scenario YAML files
    enable_scenario_registry: bool = False

    # -- Run State Machine (WS4) --
    runs_dir: str = ""                      # directory for run records (empty = off)

    # -- Replay & Regression (WS5) --
    regression_suite_dir: str = ""
    replay_manifest_path: str = ""

    # -- Extended Evaluators (WS6) --
    enable_process_evaluation: bool = False
    enable_recovery_evaluation: bool = False
    enable_interaction_evaluation: bool = False

    # -- Risk-Tiered Release Gates (WS7) --
    risk_tier_profiles_path: str = ""       # YAML path for RiskTierProfile list
    baseline_run_id: str = ""               # for regression comparison

    # -- Target Profile (Black-Box Testing) --
    target_profile_path: str = ""           # YAML path for TargetProfile

    # -- Environment Simulator --
    environment_yaml_path: str = ""         # YAML path for EnvironmentSimulator

    # -- Campaigns --
    campaigns_dir: str = ""                 # directory for campaign configs/results

    # -- Severity Classification --
    enable_severity_classification: bool = False

    # -- Safety --
    safety_config_path: str = ""            # YAML path for SafetyConfig

    # -- Performance Profiling (Gap 7) --
    enable_performance_profiling: bool = False
    sla_latency_ms: float = 5000.0         # SLA target in milliseconds

    # -- Cost/Token Tracking (Gap 12) --
    enable_cost_tracking: bool = False
    cost_budget_usd: float = 10.0          # budget cap in USD

    # -- Stateful Agent Testing (Gap 2) --
    enable_stateful_testing: bool = False
    stateful_session_count: int = 5        # number of teach-test sessions

    # -- Hallucination Testing (Gap 4) --
    enable_hallucination_testing: bool = False
    # HallucinationOracle loads tau_geodesic from store/calibration.json by default.
    # Set this only to override the calibrated value for sweeps/experiments.
    hallucination_tau_override: float | None = None

    # -- Confidence Calibration (Gap 5) --
    enable_calibration_testing: bool = False

    # -- Multi-Agent Testing (Gap 3) --
    enable_multi_agent: bool = False
    multi_agent_roles_path: str = ""       # YAML path for AgentRole definitions

    # -- Cross-Document Conflict Testing (Gap 6) --
    enable_cross_document: bool = False
    additional_store_paths: list[str] = field(default_factory=list)

    # -- Entity Disambiguation Testing (Gap 8) --
    enable_disambiguation_testing: bool = False

    # -- Paraphrase Invariance (Gap 11) --
    enable_invariance_testing: bool = False

    # -- Streaming Output Testing (Gap 9) --
    enable_streaming_testing: bool = False

    # -- Live Dashboard (Gap 10) --
    enable_live_dashboard: bool = False
    live_dashboard_port: int = 8765

    # -- Device --
    device: str = ""  # empty = auto (cuda if available)

    @classmethod
    def from_yaml(cls, path: str) -> "DOEHarnessConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        flat: dict = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)

        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in known}
        return cls(**filtered)

    def resolve_store_name(self) -> str:
        """Derive store_name from markdown_path if not set."""
        if self.store_name:
            return self.store_name
        import os
        base = os.path.splitext(os.path.basename(self.markdown_path))[0]
        return base.replace(" ", "_").lower()
