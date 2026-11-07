# SPDX-License-Identifier: Apache-2.0
"""kgcheck: read-only audits of OntoDisc ontologies, knowledge graphs and store artifacts.

Phases 0 and A of ``docs/plans/ontodisc_kg_audit_library_plan.md``: the artifact
contract, a lossless store reader, the rule engine with the ``ontodisc_default``
profile (ART, VOC, STR, EVD, NUM and COV rules), JSON and Markdown reports, gate
decisions and the CLI ``python -m knowlytix.knowledge.kgcheck audit <store>``.
Phase B adds the optional ``audit_schema.json`` sidecar, which declares node
kinds, types, descriptions, hierarchy and relation properties that a store does
not record (``kgcheck.audit_schema``), the rules that use it (VOC005-006,
STR003-007, DEF001-003, MOD001-002, META001), ``audit_ontology`` with the
``taxonomy`` profile, and ``draft_audit_schema``, which drafts a sidecar of
inferred declarations for review. Phase C adds typed constraints declared in
the sidecar (CON001-CON006, STR008), unit and competency checks (NUM003,
COV002), a rerun of the producer's grounding (EVD004, through
``knowlytix.discovery.ontodisc.audit``), and optional RDF input and SHACL
validation (``audit_rdf``, CON007; ``kgcheck-rdf`` extra) with the ``typed_kg``
and ``rdf_shacl`` profiles. Phase E adds the optional OWL reasoner rule (CON008,
HermiT through the ``kgcheck-owl`` extra), incremental audits that reuse a
rule's result when nothing it read changed (``audit_incremental``), the
optional model review of evidence (``semantic_review_rule``, EVD005), and
harnesses that measure a reviewer and calibrate heuristic rules on labeled
data. An audit reads JSON, Markdown and, for RDF input,
the named graph file only: it never loads model weights, uses a GPU or calls an
LLM unless the caller supplies a reviewer, and it never changes the store.

    from knowlytix.knowledge.kgcheck import AuditConfig, GatePolicy, audit_store
    report = audit_store("runs/my_store", AuditConfig(view="stored"))
    report.write_json("reports/kg_audit.json")
    decision = report.evaluate_gate(GatePolicy(fail_on="error"))
"""

from knowlytix.knowledge.kgcheck.adapters import read_ontology, read_rdf, read_store
from knowlytix.knowledge.kgcheck.audit_schema import SIDECAR, sidecar_schema
from knowlytix.knowledge.kgcheck.config import (
    AuditConfig,
    ConfigError,
    load_profile,
    profile_names,
)
from knowlytix.knowledge.kgcheck.contract import (
    ArtifactSpec,
    artifact_contract,
    artifact_schema,
    artifact_specs,
    contract_version,
)
from knowlytix.knowledge.kgcheck.draft import draft_audit_schema
from knowlytix.knowledge.kgcheck.engine import (
    audit,
    audit_ontology,
    audit_rdf,
    audit_store,
    audit_triples,
)
from knowlytix.knowledge.kgcheck.evaluation import (
    LabeledClaim,
    calibrate_rule,
    calibrated_profile,
    evaluate_reviewer,
    labeling_sheet,
)
from knowlytix.knowledge.kgcheck.gate import (
    GateDecision,
    GateOutcome,
    GatePolicy,
    evaluate_gate,
)
from knowlytix.knowledge.kgcheck.incremental import AuditCache, audit_incremental
from knowlytix.knowledge.kgcheck.models import (
    AuditReport,
    AuditSnapshot,
    CompetencyQuestion,
    Declaration,
    DeclKey,
    DeclSource,
    Diagnostic,
    DuplicateFindingError,
    Finding,
    GraphView,
    NodeKind,
    RawRef,
    RuleResult,
    RuleResultError,
    RuleStatus,
    SemanticsDecl,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import (
    Outcome,
    Rule,
    RuleContext,
    RuleRegistry,
    RuleSpec,
    builtin_registry,
)
from knowlytix.knowledge.kgcheck.reporting import (
    REPORT_VERSION,
    ReportComparison,
    RunInfo,
    compare_reports,
    render_markdown,
    report_schema,
    report_to_dict,
    stable_payload,
    write_json,
    write_markdown,
)
from knowlytix.knowledge.kgcheck.review import SemanticReviewer, semantic_review_rule

__all__ = [
    # artifact contract
    "ArtifactSpec",
    "artifact_contract",
    "artifact_schema",
    "artifact_specs",
    "contract_version",
    # reading a store or an ontology
    "read_ontology",
    "read_rdf",
    "read_store",
    # audit schema sidecar
    "SIDECAR",
    "draft_audit_schema",
    "sidecar_schema",
    # auditing
    "audit",
    "audit_ontology",
    "audit_rdf",
    "audit_store",
    "audit_triples",
    "compare_reports",
    "ReportComparison",
    "audit_incremental",
    "AuditCache",
    "AuditConfig",
    "ConfigError",
    "load_profile",
    "profile_names",
    # rules
    "Outcome",
    "Rule",
    "RuleContext",
    "RuleRegistry",
    "RuleSpec",
    "builtin_registry",
    # semantic review and evaluation (advisory, caller-supplied model)
    "SemanticReviewer",
    "semantic_review_rule",
    "LabeledClaim",
    "evaluate_reviewer",
    "calibrate_rule",
    "calibrated_profile",
    "labeling_sheet",
    # gate
    "GateDecision",
    "GateOutcome",
    "GatePolicy",
    "evaluate_gate",
    # data model
    "AuditReport",
    "AuditSnapshot",
    "CompetencyQuestion",
    "Declaration",
    "DeclKey",
    "DeclSource",
    "Diagnostic",
    "DuplicateFindingError",
    "Finding",
    "GraphView",
    "NodeKind",
    "RawRef",
    "RuleResult",
    "RuleResultError",
    "RuleStatus",
    "SemanticsDecl",
    "Severity",
    "AssertionSemanticReviewer",
    "audit_quality",
    "QualityReport",
    "QUALITY_VERSION",
    # reports
    "REPORT_VERSION",
    "RunInfo",
    "report_schema",
    "render_markdown",
    "report_to_dict",
    "stable_payload",
    "write_json",
    "write_markdown",
]

from knowlytix.knowledge.kgcheck.assertion_review import AssertionSemanticReviewer
from knowlytix.knowledge.kgcheck.quality import (
    QUALITY_VERSION,
    QualityReport,
    audit_quality,
)
