# SPDX-License-Identifier: Apache-2.0
"""Policy GMS store builder.

The policy verifier in :class:`PolicyEngine` needs geometrically
trained embeddings for governance-domain entities (agents, tools,
workflows, standards) and relations (``uses_tool``, ``complies_with``,
``violates``).  Mixing that corpus into the content store pollutes
content calibration, so we keep it in a **separate** GMS and wire
PolicyEngine to it independently.

Usage:

    from knowlytix.harness.testing.policy_store import build_policy_store
    policy_store, policy_judge = build_policy_store()
    engine = PolicyEngine(
        store=policy_store,
        judge=policy_judge,
        rules=my_rules,
    )

By default the store is cached under ``gms_agent_builder_ext/lab_stores/
policy_governance/`` and reloaded on subsequent calls.
"""

from __future__ import annotations

import os

import torch

from knowlytix.knowledge.config import DocGMSConfig, LearnConfig
from knowlytix.knowledge.extract import ExtractedTriple
from knowlytix.knowledge.learn import RuntimeLearner
from knowlytix.knowledge.store import GMSExpertStore

from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.core.config import GeometryConfig, EmbeddingConfig
from knowlytix.core.graph.transport import RelationalTransport
from knowlytix.core.memory.compression import CompressionMemory
from knowlytix.core.memory.router import MemoryRouter
from knowlytix.core.config import CapLossConfig
from knowlytix.core.train_finstructbench import GraphToGMS, populate_enm, train_gms

from .judge import GMSJudge


# ---------------------------------------------------------------------------
# Semantic corpus — seeded on top of the markdown-derived triples.
# Kept here (not in the markdown) because ingest_markdown only produces
# has_X / passes / fails / in_section relations; custom relation names
# must be written directly through RuntimeLearner.
# ---------------------------------------------------------------------------

TOOL_USE_TRIPLES: list[tuple[str, str, str]] = [
    # model_risk_analyst: read-only / diagnostic tools only
    ("model_risk_analyst", "uses_tool", "data_lookup"),
    ("model_risk_analyst", "uses_tool", "risk_check"),
    ("model_risk_analyst", "uses_tool", "send_notification"),

    # compliance_auditor: review + raise cases
    ("compliance_auditor", "uses_tool", "data_lookup"),
    ("compliance_auditor", "uses_tool", "risk_check"),
    ("compliance_auditor", "uses_tool", "submit_escalation"),

    # governance_officer: high-authority actions
    ("governance_officer", "uses_tool", "approve_action"),
    ("governance_officer", "uses_tool", "revoke_access"),
    ("governance_officer", "uses_tool", "submit_escalation"),
    ("governance_officer", "uses_tool", "send_notification"),

    # ops_engineer: deploy / notify (no approvals, no destructive)
    ("ops_engineer", "uses_tool", "data_lookup"),
    ("ops_engineer", "uses_tool", "risk_check"),
    ("ops_engineer", "uses_tool", "send_notification"),

    # escalation_manager: raise + notify + approve
    ("escalation_manager", "uses_tool", "submit_escalation"),
    ("escalation_manager", "uses_tool", "send_notification"),
    ("escalation_manager", "uses_tool", "approve_action"),
]

OPPOSING_TRIPLES: list[tuple[str, str, str]] = [
    # audit_standard_sox
    ("sox_compliant_workflow", "complies_with", "audit_standard_sox"),
    ("reviewed_pipeline", "complies_with", "audit_standard_sox"),
    ("legacy_workflow_v1", "violates", "audit_standard_sox"),
    ("deprecated_workflow_v2", "violates", "audit_standard_sox"),

    # audit_standard_bcbs
    ("bcbs_compliant_workflow", "complies_with", "audit_standard_bcbs"),
    ("validated_pipeline", "complies_with", "audit_standard_bcbs"),
    ("deprecated_workflow_v2", "violates", "audit_standard_bcbs"),
    ("shadow_pipeline", "violates", "audit_standard_bcbs"),

    # regulatory_threshold
    ("conservative_lending_model", "complies_with", "regulatory_threshold"),
    ("standard_risk_model", "complies_with", "regulatory_threshold"),
    ("aggressive_growth_model", "violates", "regulatory_threshold"),
    ("unaudited_workflow_model", "violates", "regulatory_threshold"),

    # governance_charter
    ("reviewed_pipeline", "complies_with", "governance_charter"),
    ("validated_pipeline", "complies_with", "governance_charter"),
    ("shadow_pipeline", "violates", "governance_charter"),
    ("unreviewed_deployment", "violates", "governance_charter"),
]


DEFAULT_MARKDOWN = os.path.join(
    os.path.dirname(__file__), "tests", "policy_governance.md"
)

DEFAULT_DS_MARKDOWN = os.path.join(
    os.path.dirname(__file__), "tests", "policy_governance_datascience.md"
)


# ---------------------------------------------------------------------------
# Data-science governance corpus — seeded on top of the
# policy_governance_datascience.md markdown.  Relations: uses_tool,
# reads_data, writes_data, complies_with, violates, requires_check,
# escalates_to, can_approve, restricted_to_tier.
# ---------------------------------------------------------------------------

DS_USES_TOOL_TRIPLES: list[tuple[str, str, str]] = [
    # data layer
    ("data_engineer", "uses_tool", "load_dataset"),
    ("data_engineer", "uses_tool", "data_summary"),
    ("data_engineer", "uses_tool", "preprocess_data"),
    ("data_steward", "uses_tool", "load_dataset"),
    ("data_steward", "uses_tool", "data_summary"),
    ("schema_curator", "uses_tool", "load_dataset"),
    ("schema_curator", "uses_tool", "data_summary"),

    # EDA and features
    ("eda_analyst", "uses_tool", "load_dataset"),
    ("eda_analyst", "uses_tool", "data_summary"),
    ("eda_analyst", "uses_tool", "eda_correlation"),
    ("eda_analyst", "uses_tool", "explain_pdp"),
    ("eda_analyst", "uses_tool", "explain_shap"),
    ("feature_engineer", "uses_tool", "preprocess_data"),
    ("feature_engineer", "uses_tool", "data_summary"),
    ("feature_engineer", "uses_tool", "eda_correlation"),
    ("feature_selector", "uses_tool", "feature_select"),
    ("feature_selector", "uses_tool", "train_model"),
    ("feature_selector", "uses_tool", "diagnose_stability"),

    # modelers
    ("modeler_tree", "uses_tool", "train_model"),
    ("modeler_tree", "uses_tool", "predict"),
    ("modeler_tree", "uses_tool", "leaderboard"),
    ("modeler_tree", "uses_tool", "explain_pfi"),
    ("modeler_tree", "uses_tool", "explain_shap"),
    ("modeler_linear", "uses_tool", "train_model"),
    ("modeler_linear", "uses_tool", "predict"),
    ("modeler_linear", "uses_tool", "leaderboard"),
    ("modeler_linear", "uses_tool", "explain_pfi"),
    ("modeler_linear", "uses_tool", "explain_pdp"),
    ("modeler_nn", "uses_tool", "train_model"),
    ("modeler_nn", "uses_tool", "predict"),
    ("modeler_nn", "uses_tool", "leaderboard"),
    ("modeler_nn", "uses_tool", "explain_shap"),
    ("modeler_nn", "uses_tool", "explain_pdp"),
    ("ensembler", "uses_tool", "train_model"),
    ("ensembler", "uses_tool", "predict"),
    ("ensembler", "uses_tool", "leaderboard"),

    # reviewers — narrow tool authority by pillar
    ("validation_reviewer", "uses_tool", "leaderboard"),
    ("validation_reviewer", "uses_tool", "diagnose_accuracy"),
    ("validation_reviewer", "uses_tool", "predict"),
    ("weakness_reviewer", "uses_tool", "diagnose_weakness"),
    ("weakness_reviewer", "uses_tool", "train_model"),
    ("weakness_reviewer", "uses_tool", "predict"),
    ("weakness_reviewer", "uses_tool", "explain_pfi"),
    ("resilience_reviewer", "uses_tool", "diagnose_resilience"),
    ("resilience_reviewer", "uses_tool", "train_model"),
    ("resilience_reviewer", "uses_tool", "predict"),
    ("stability_reviewer", "uses_tool", "diagnose_stability"),
    ("stability_reviewer", "uses_tool", "train_model"),
    ("stability_reviewer", "uses_tool", "explain_pfi"),
    ("stability_reviewer", "uses_tool", "explain_shap"),
    ("robustness_reviewer", "uses_tool", "diagnose_robustness"),
    ("robustness_reviewer", "uses_tool", "predict"),
    ("fairness_reviewer", "uses_tool", "diagnose_fairness"),
    ("fairness_reviewer", "uses_tool", "predict"),
    ("explainability_reviewer", "uses_tool", "explain_pfi"),
    ("explainability_reviewer", "uses_tool", "explain_pdp"),
    ("explainability_reviewer", "uses_tool", "explain_shap"),

    # ops / approval / incident
    ("mlops_engineer", "uses_tool", "trigger_retrain"),
    ("deploy_operator", "uses_tool", "deploy_model"),
    ("approval_officer", "uses_tool", "approve_model"),
    ("model_risk_officer", "uses_tool", "approve_model"),
    ("model_risk_officer", "uses_tool", "deploy_model"),
    ("incident_responder", "uses_tool", "trigger_retrain"),
    ("incident_responder", "uses_tool", "predict"),
    ("incident_responder", "uses_tool", "diagnose_accuracy"),
    ("retrain_agent", "uses_tool", "train_model"),
    ("retrain_agent", "uses_tool", "trigger_retrain"),
]

DS_READS_DATA_TRIPLES: list[tuple[str, str, str]] = [
    # data layer has broad read access
    ("data_engineer", "reads_data", "public"),
    ("data_engineer", "reads_data", "internal"),
    ("data_engineer", "reads_data", "confidential"),
    ("data_engineer", "reads_data", "pii"),
    ("data_engineer", "reads_data", "regulated"),
    ("data_steward", "reads_data", "pii"),
    ("data_steward", "reads_data", "regulated"),
    ("schema_curator", "reads_data", "public"),
    ("schema_curator", "reads_data", "internal"),
    ("schema_curator", "reads_data", "confidential"),

    # EDA / features — engineered (internal) only
    ("eda_analyst", "reads_data", "internal"),
    ("feature_engineer", "reads_data", "internal"),
    ("feature_engineer", "reads_data", "confidential"),
    ("feature_selector", "reads_data", "internal"),

    # modelers — engineered only
    ("modeler_tree", "reads_data", "internal"),
    ("modeler_linear", "reads_data", "internal"),
    ("modeler_nn", "reads_data", "internal"),
    ("ensembler", "reads_data", "internal"),

    # reviewers — engineered (validation may see raw confidential)
    ("validation_reviewer", "reads_data", "internal"),
    ("validation_reviewer", "reads_data", "confidential"),
    ("weakness_reviewer", "reads_data", "internal"),
    ("resilience_reviewer", "reads_data", "internal"),
    ("stability_reviewer", "reads_data", "internal"),
    ("robustness_reviewer", "reads_data", "internal"),
    ("fairness_reviewer", "reads_data", "internal"),
    ("explainability_reviewer", "reads_data", "internal"),

    # ops / approval
    ("mlops_engineer", "reads_data", "internal"),
    ("deploy_operator", "reads_data", "internal"),
    ("approval_officer", "reads_data", "internal"),
    ("approval_officer", "reads_data", "confidential"),
    ("model_risk_officer", "reads_data", "public"),
    ("model_risk_officer", "reads_data", "internal"),
    ("model_risk_officer", "reads_data", "confidential"),
    ("model_risk_officer", "reads_data", "pii"),
    ("model_risk_officer", "reads_data", "regulated"),
    ("incident_responder", "reads_data", "internal"),
    ("incident_responder", "reads_data", "confidential"),
    ("incident_responder", "reads_data", "regulated"),
    ("retrain_agent", "reads_data", "internal"),
    ("audit_logger", "reads_data", "public"),
    ("audit_logger", "reads_data", "internal"),
    ("audit_logger", "reads_data", "confidential"),
    ("audit_logger", "reads_data", "pii"),
    ("audit_logger", "reads_data", "regulated"),
]

DS_WRITES_DATA_TRIPLES: list[tuple[str, str, str]] = [
    ("data_engineer", "writes_data", "internal"),
    ("data_engineer", "writes_data", "confidential"),
    ("schema_curator", "writes_data", "internal"),
    ("feature_engineer", "writes_data", "internal"),
    ("feature_selector", "writes_data", "internal"),
    ("modeler_tree", "writes_data", "internal"),
    ("modeler_linear", "writes_data", "internal"),
    ("modeler_nn", "writes_data", "internal"),
    ("ensembler", "writes_data", "internal"),
    ("mlops_engineer", "writes_data", "internal"),
    ("retrain_agent", "writes_data", "internal"),
    ("audit_logger", "writes_data", "public"),
]

DS_COMPLIES_WITH_TRIPLES: list[tuple[str, str, str]] = [
    # high_risk_classification — the exemplar
    ("high_risk_classification", "complies_with", "weakness_identification_standard"),
    ("high_risk_classification", "complies_with", "resilience_standard"),
    ("high_risk_classification", "complies_with", "stability_standard"),
    ("high_risk_classification", "complies_with", "model_validation_standard"),
    ("high_risk_classification", "complies_with", "fairness_standard"),
    ("high_risk_classification", "complies_with", "explainability_standard"),
    ("high_risk_classification", "complies_with", "monitoring_standard"),

    # regulated_lending_model
    ("regulated_lending_model", "complies_with", "weakness_identification_standard"),
    ("regulated_lending_model", "complies_with", "resilience_standard"),
    ("regulated_lending_model", "complies_with", "stability_standard"),
    ("regulated_lending_model", "complies_with", "fairness_standard"),
    ("regulated_lending_model", "complies_with", "explainability_standard"),
    ("regulated_lending_model", "complies_with", "approval_standard"),

    # clinical_triage_model
    ("clinical_triage_model", "complies_with", "weakness_identification_standard"),
    ("clinical_triage_model", "complies_with", "resilience_standard"),
    ("clinical_triage_model", "complies_with", "stability_standard"),
    ("clinical_triage_model", "complies_with", "explainability_standard"),
    ("clinical_triage_model", "complies_with", "privacy_standard"),
    ("clinical_triage_model", "complies_with", "model_validation_standard"),

    # conservative_credit_model
    ("conservative_credit_model", "complies_with", "weakness_identification_standard"),
    ("conservative_credit_model", "complies_with", "resilience_standard"),
    ("conservative_credit_model", "complies_with", "stability_standard"),
    ("conservative_credit_model", "complies_with", "model_validation_standard"),

    # reviewed_fraud_detection
    ("reviewed_fraud_detection", "complies_with", "weakness_identification_standard"),
    ("reviewed_fraud_detection", "complies_with", "robustness_standard"),
    ("reviewed_fraud_detection", "complies_with", "monitoring_standard"),
    ("reviewed_fraud_detection", "complies_with", "change_control_standard"),

    # monitored_churn_model
    ("monitored_churn_model", "complies_with", "resilience_standard"),
    ("monitored_churn_model", "complies_with", "monitoring_standard"),
    ("monitored_churn_model", "complies_with", "data_quality_standard"),

    # validated_baseline_model
    ("validated_baseline_model", "complies_with", "model_validation_standard"),
    ("validated_baseline_model", "complies_with", "data_quality_standard"),
    ("validated_baseline_model", "complies_with", "change_control_standard"),
]

DS_VIOLATES_TRIPLES: list[tuple[str, str, str]] = [
    # The three pillar-violators are pedagogically named
    ("aggregate_only_workflow", "violates", "weakness_identification_standard"),
    ("aggregate_only_workflow", "violates", "explainability_standard"),
    ("snapshot_only_workflow", "violates", "resilience_standard"),
    ("snapshot_only_workflow", "violates", "monitoring_standard"),
    ("unstable_lasso_workflow", "violates", "stability_standard"),
    ("unstable_lasso_workflow", "violates", "explainability_standard"),

    # Governance hygiene failures
    ("shadow_pipeline", "violates", "change_control_standard"),
    ("shadow_pipeline", "violates", "approval_standard"),
    ("shadow_pipeline", "violates", "monitoring_standard"),
    ("legacy_workflow_v1", "violates", "model_validation_standard"),
    ("legacy_workflow_v1", "violates", "monitoring_standard"),
    ("unreviewed_deployment", "violates", "approval_standard"),
    ("unreviewed_deployment", "violates", "change_control_standard"),
    ("unmonitored_legacy_model", "violates", "monitoring_standard"),
    ("unmonitored_legacy_model", "violates", "resilience_standard"),
    ("overfit_trial_model", "violates", "model_validation_standard"),
    ("overfit_trial_model", "violates", "stability_standard"),
]

DS_REQUIRES_CHECK_TRIPLES: list[tuple[str, str, str]] = [
    # high_risk_classification — all three pillars plus fairness, accuracy
    ("high_risk_classification", "requires_check", "diagnose_weakness"),
    ("high_risk_classification", "requires_check", "diagnose_resilience"),
    ("high_risk_classification", "requires_check", "diagnose_stability"),
    ("high_risk_classification", "requires_check", "diagnose_fairness"),
    ("high_risk_classification", "requires_check", "diagnose_accuracy"),

    # regulated_lending_model — all three pillars plus fairness, explain
    ("regulated_lending_model", "requires_check", "diagnose_weakness"),
    ("regulated_lending_model", "requires_check", "diagnose_resilience"),
    ("regulated_lending_model", "requires_check", "diagnose_stability"),
    ("regulated_lending_model", "requires_check", "diagnose_fairness"),
    ("regulated_lending_model", "requires_check", "explain_shap"),
    ("regulated_lending_model", "requires_check", "diagnose_accuracy"),

    # clinical_triage_model — three pillars plus explain
    ("clinical_triage_model", "requires_check", "diagnose_weakness"),
    ("clinical_triage_model", "requires_check", "diagnose_resilience"),
    ("clinical_triage_model", "requires_check", "diagnose_stability"),
    ("clinical_triage_model", "requires_check", "explain_pfi"),
    ("clinical_triage_model", "requires_check", "explain_shap"),
    ("clinical_triage_model", "requires_check", "diagnose_accuracy"),

    # conservative_credit_model — three pillars plus accuracy
    ("conservative_credit_model", "requires_check", "diagnose_weakness"),
    ("conservative_credit_model", "requires_check", "diagnose_resilience"),
    ("conservative_credit_model", "requires_check", "diagnose_stability"),
    ("conservative_credit_model", "requires_check", "diagnose_accuracy"),

    # reviewed_fraud_detection — weakness + robustness
    ("reviewed_fraud_detection", "requires_check", "diagnose_weakness"),
    ("reviewed_fraud_detection", "requires_check", "diagnose_robustness"),
    ("reviewed_fraud_detection", "requires_check", "diagnose_accuracy"),

    # monitored_churn_model — minimal tier-3 set
    ("monitored_churn_model", "requires_check", "diagnose_resilience"),
    ("monitored_churn_model", "requires_check", "diagnose_accuracy"),

    # validated_baseline_model — tier-3 minimum
    ("validated_baseline_model", "requires_check", "diagnose_accuracy"),
]

DS_ESCALATES_TO_TRIPLES: list[tuple[str, str, str]] = [
    # reviewer escalation to model_risk_officer (terminal authority)
    ("validation_reviewer", "escalates_to", "model_risk_officer"),
    ("weakness_reviewer", "escalates_to", "model_risk_officer"),
    ("resilience_reviewer", "escalates_to", "model_risk_officer"),
    ("stability_reviewer", "escalates_to", "model_risk_officer"),
    ("robustness_reviewer", "escalates_to", "model_risk_officer"),
    ("fairness_reviewer", "escalates_to", "model_risk_officer"),
    ("explainability_reviewer", "escalates_to", "model_risk_officer"),

    # modeler escalation to validation
    ("modeler_tree", "escalates_to", "validation_reviewer"),
    ("modeler_linear", "escalates_to", "validation_reviewer"),
    ("modeler_nn", "escalates_to", "validation_reviewer"),
    ("ensembler", "escalates_to", "validation_reviewer"),

    # incident chain
    ("mlops_engineer", "escalates_to", "incident_responder"),
    ("incident_responder", "escalates_to", "model_risk_officer"),

    # approval + access
    ("approval_officer", "escalates_to", "model_risk_officer"),
    ("data_steward", "escalates_to", "model_risk_officer"),
]

DS_CAN_APPROVE_TRIPLES: list[tuple[str, str, str]] = [
    # tier assignments for approval authority
    ("approval_officer", "can_approve", "tier_2"),
    ("approval_officer", "can_approve", "tier_3"),
    ("model_risk_officer", "can_approve", "tier_1"),
    ("model_risk_officer", "can_approve", "tier_2"),
    ("model_risk_officer", "can_approve", "tier_3"),
    ("deploy_operator", "can_approve", "tier_3"),
]

DS_RESTRICTED_TO_TIER_TRIPLES: list[tuple[str, str, str]] = [
    # tier-1 (restricted)
    ("approve_model", "restricted_to_tier", "tier_1"),
    ("deploy_model", "restricted_to_tier", "tier_1"),
    # tier-2 (diagnostics + retrain trigger)
    ("trigger_retrain", "restricted_to_tier", "tier_2"),
    ("diagnose_weakness", "restricted_to_tier", "tier_2"),
    ("diagnose_resilience", "restricted_to_tier", "tier_2"),
    ("diagnose_stability", "restricted_to_tier", "tier_2"),
    ("diagnose_robustness", "restricted_to_tier", "tier_2"),
    ("diagnose_fairness", "restricted_to_tier", "tier_2"),
    # tier-3 (modeling + explanation)
    ("preprocess_data", "restricted_to_tier", "tier_3"),
    ("feature_select", "restricted_to_tier", "tier_3"),
    ("train_model", "restricted_to_tier", "tier_3"),
    ("predict", "restricted_to_tier", "tier_3"),
    ("leaderboard", "restricted_to_tier", "tier_3"),
    ("diagnose_accuracy", "restricted_to_tier", "tier_3"),
    ("explain_pfi", "restricted_to_tier", "tier_3"),
    ("explain_pdp", "restricted_to_tier", "tier_3"),
    ("explain_shap", "restricted_to_tier", "tier_3"),
    # tier-4 (read-only)
    ("load_dataset", "restricted_to_tier", "tier_4"),
    ("data_summary", "restricted_to_tier", "tier_4"),
    ("eda_correlation", "restricted_to_tier", "tier_4"),
]

DS_ALL_TRIPLES: list[tuple[str, str, str]] = (
    DS_USES_TOOL_TRIPLES
    + DS_READS_DATA_TRIPLES
    + DS_WRITES_DATA_TRIPLES
    + DS_COMPLIES_WITH_TRIPLES
    + DS_VIOLATES_TRIPLES
    + DS_REQUIRES_CHECK_TRIPLES
    + DS_ESCALATES_TO_TRIPLES
    + DS_CAN_APPROVE_TRIPLES
    + DS_RESTRICTED_TO_TIER_TRIPLES
)

# Corpus registry — maps corpus name to markdown path, seed triple set,
# and default on-disk store name.  Seeds are injected into the doc_graph
# BEFORE main training so they go through the full GMSLoss (hard
# negatives + tension pairs), not the weak incremental loop.
_CORPUS_REGISTRY = {
    "generic": {
        "markdown": DEFAULT_MARKDOWN,
        "triples": TOOL_USE_TRIPLES + OPPOSING_TRIPLES,
        "default_store_name": "policy_governance",
    },
    "data_science": {
        "markdown": DEFAULT_DS_MARKDOWN,
        "triples": DS_ALL_TRIPLES,
        "default_store_name": "policy_governance_datascience",
    },
}


# ---------------------------------------------------------------------------


def _seed_semantic_corpus(
    store: GMSExpertStore,
    device: torch.device,
    triples: list[tuple[str, str, str]],
    *,
    n_steps: int = 50,
    freeze_existing: bool = True,
    verbose: bool = True,
) -> int:
    """Write the given triple set into the store via RuntimeLearner.

    Returns the number of triples actually accepted.
    """
    learn_cfg = store.config.learn if store.config else LearnConfig()
    # Override the incremental-training budget and freeze flag so the
    # caller controls how hard the seeding pass gets to push.
    learn_cfg.n_steps = n_steps
    learn_cfg.freeze_existing = freeze_existing
    learner = RuntimeLearner(store, learn_cfg)

    all_triples = [
        ExtractedTriple(head=h, relation=r, tail=t) for (h, r, t) in triples
    ]
    before = store.stats()
    result = learner.learn_triples(all_triples, device)
    learner.flush_training()
    learner.wait_for_training(timeout=600)
    after = store.stats()

    if verbose:
        print(
            f"  Seeded semantic corpus: "
            f"accepted={result.accepted} "
            f"entities {before['entities']}->{after['entities']} "
            f"relations {before['relations']}->{after['relations']} "
            f"triples {before['triples']}->{after['triples']} "
            f"(n_steps={n_steps}, freeze_existing={freeze_existing})"
        )
    return after["triples"] - before["triples"]


# ---------------------------------------------------------------------------


def build_policy_store(
    *,
    corpus: str = "generic",
    markdown_path: str | None = None,
    stores_dir: str = "gms_agent_builder_ext/lab_stores",
    store_name: str | None = None,
    ingest_epochs: int = 500,
    d_v: int = 256,
    d_u: int = 256,
    m: int = 128,
    device: torch.device | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
    seed: int = 42,
    loss_mode: str = "cap",
    cap_config: CapLossConfig | None = None,
    embedding: EmbeddingConfig | None = None,
) -> tuple[GMSExpertStore, GMSJudge]:
    """Build (or load) a policy governance GMS and calibrated judge.

    The build path parses the corpus markdown, injects the corpus's
    seeded semantic triples into the doc_graph, then runs a SINGLE
    main-training pass (full ``GMSLoss``: hard negatives + tension
    pairs) on the combined set.  This produces discriminative
    embeddings on the seeded relations --- a weak positives-only
    incremental pass does not.

    Parameters
    ----------
    corpus
        ``"generic"`` (default, back-compat) uses the small
        enterprise-risk corpus.  ``"data_science"`` uses the 24-agent
        / 20-tool / 15-workflow data-science corpus with nine seeded
        relations.
    markdown_path
        Override the bootstrap markdown for the selected corpus.
    stores_dir, store_name
        Cache location.  ``store_name`` defaults to the corpus's own.
    ingest_epochs
        Training epochs for the combined ingest + seed set.
    d_v, d_u, m
        Embedding geometry.  Defaults 256/256/128 match DocGMS.
    force_rebuild
        If True, wipe any existing on-disk store and retrain.
    seed
        Seed for calibration sampling.

    Returns
    -------
    (store, judge)
        ``store`` holds the trained policy embeddings.
        ``judge`` exposes ``policy_geodesic`` and ``tau_contra`` in
        ``judge._thresholds`` — which :class:`PolicyEngine` reads to
        override its hardcoded defaults.
    """
    if corpus not in _CORPUS_REGISTRY:
        raise ValueError(
            f"unknown corpus {corpus!r}; "
            f"expected one of {sorted(_CORPUS_REGISTRY)}"
        )
    entry = _CORPUS_REGISTRY[corpus]
    effective_markdown = markdown_path or entry["markdown"]
    effective_store_name = store_name or entry["default_store_name"]
    seed_triples = entry["triples"]

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    store_path = os.path.join(stores_dir, effective_store_name)

    if force_rebuild and os.path.isdir(store_path):
        import shutil
        shutil.rmtree(store_path)
        if verbose:
            print(f"  Removed existing policy store at {store_path}")

    # Build the DocGMS config with the caller's geometry + epochs.
    doc_config = DocGMSConfig(store_path=store_path)
    doc_config.geometry = GeometryConfig(d_v=d_v, d_u=d_u, m=m, d=m)
    doc_config.train.epochs = ingest_epochs
    if embedding is not None:
        doc_config.embedding = embedding
    store = GMSExpertStore(doc_config, device)

    if store.exists():
        if verbose:
            print(f"  Loading existing policy store: {store_path}")
        store.load()
    else:
        # Integrated build: parse markdown -> inject seeded semantic
        # triples into the doc_graph BEFORE training.  This ensures the
        # seeded triples go through the full GMSLoss (hard negatives +
        # tension pairs) during main training, instead of the weak
        # positives-only loss that RuntimeLearner._incremental_train
        # applies in the post-hoc seeding path.
        if verbose:
            print(
                f"  Building policy store: {store_path}\n"
                f"    Config: d_v={d_v} d_u={d_u} m={m} "
                f"epochs={ingest_epochs}"
            )
            print(f"    Markdown:     {effective_markdown}")
            print(f"    Seed triples: {len(seed_triples)}")

        doc_graph = ingest_markdown(effective_markdown)

        before_triples = len(doc_graph.triples)
        for h, r, t in seed_triples:
            doc_graph.add_triple(h, r, t)
        after_triples = len(doc_graph.triples)
        if verbose:
            print(
                f"    Injected seed triples: "
                f"{before_triples} -> {after_triples} "
                f"({after_triples - before_triples} new)"
            )

        adapter = GraphToGMS(doc_graph)
        if verbose:
            print(
                f"    Adapter: {adapter.num_entities} entities, "
                f"{adapter.num_relations} relations, "
                f"{len(doc_graph.triples)} triples"
            )

        effective_cap = cap_config if loss_mode == "cap" else None
        model = train_gms(
            adapter,
            device,
            epochs=ingest_epochs,
            batch_size=doc_config.train.batch_size,
            num_neg=doc_config.train.neg_samples,
            lr=doc_config.train.lr,
            lr_riemannian=doc_config.train.lr_riemannian,
            geometry=doc_config.geometry,
            loss=doc_config.loss,
            loss_mode=loss_mode,
            cap=effective_cap,
            seed=seed,
            embedding=doc_config.embedding,
        )
        model.eval()

        if verbose:
            print("\n  Populating ENM register...")
        enm = populate_enm(doc_graph, adapter)

        transport = RelationalTransport(model)
        compression = CompressionMemory(k=doc_config.memory.k, d=m)
        router = MemoryRouter(enm, compression)

        store.model = model
        store.adapter = adapter
        store.enm = enm
        store.doc_graph = doc_graph
        store.transport = transport
        store.compression = compression
        store.router = router
        with open(effective_markdown) as _f:
            store.markdown = _f.read()

        store.save()
        if verbose:
            print(f"  Store saved to {store_path}")

    # Step 3: calibrate both content-style thresholds (geodesic,
    # tension, holonomy) and the policy-specific geodesic cut on the
    # uses_tool relation (present in both corpora).
    judge = GMSJudge(store=store)
    if verbose:
        print("\n  Calibrating policy judge...")
    judge.calibrate(seed=seed)
    judge.calibrate_policy(relation="uses_tool", seed=seed)

    if verbose:
        thr = judge._thresholds or {}
        print(
            f"  Policy thresholds: "
            f"policy_geodesic={thr.get('policy_geodesic')} "
            f"tau_contra={thr.get('tau_contra')} "
            f"tension={thr.get('tension')}"
        )

    return store, judge


__all__ = [
    "build_policy_store",
    "TOOL_USE_TRIPLES",
    "OPPOSING_TRIPLES",
    "DEFAULT_MARKDOWN",
    "DEFAULT_DS_MARKDOWN",
    "DS_USES_TOOL_TRIPLES",
    "DS_READS_DATA_TRIPLES",
    "DS_WRITES_DATA_TRIPLES",
    "DS_COMPLIES_WITH_TRIPLES",
    "DS_VIOLATES_TRIPLES",
    "DS_REQUIRES_CHECK_TRIPLES",
    "DS_ESCALATES_TO_TRIPLES",
    "DS_CAN_APPROVE_TRIPLES",
    "DS_RESTRICTED_TO_TIER_TRIPLES",
    "DS_ALL_TRIPLES",
]
