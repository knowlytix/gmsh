# SPDX-License-Identifier: Apache-2.0
"""Configuration for the DOE testing harness."""

from dataclasses import dataclass, field

import yaml


@dataclass
class DOEHarnessConfig:
    """Full configuration for a DOE-GMS benchmark run."""

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
