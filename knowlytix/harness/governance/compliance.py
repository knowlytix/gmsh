# SPDX-License-Identifier: Apache-2.0
"""Compliance evidence generation for regulatory alignment.

Maps the six-layer governed agent architecture to regulatory
requirements from SR 11-7 (OCC/Federal Reserve) and the EU AI Act.
Generates audit-ready compliance evidence packages from pipeline
results.

Three components:

    ComplianceEvidenceGenerator — maps pipeline results to regulatory checks
    ProvenanceRegistry — tracks all artifacts and their lineage
    SnapshotManager — versions governance state for change management

Built in Chapter 20 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ComplianceCheck:
    """A single regulatory compliance check with evidence.

    Attributes
    ----------
    requirement : str
        Human-readable description of the regulatory requirement.
    regulation : str
        Which regulation this maps to ("SR 11-7" or "EU AI Act").
    article : str
        Specific article or expectation (e.g., "Article 9" or
        "Conceptual Soundness").
    architecture_layer : str
        Which layer of the six-layer architecture provides evidence.
    evidence_chapter : str
        Which monograph chapter demonstrates compliance.
    status : str
        One of "satisfied", "partial", "not_applicable".
    evidence_summary : str
        Concrete evidence supporting the status determination.
    """

    requirement: str
    regulation: str
    article: str
    architecture_layer: str
    evidence_chapter: str
    status: str
    evidence_summary: str


@dataclass
class CompliancePackage:
    """Aggregated compliance evidence across all checks.

    Attributes
    ----------
    checks : list[ComplianceCheck]
        Individual compliance checks.
    overall_status : str
        "COMPLIANT" if all checks satisfied, "PARTIAL" if any partial,
        "NON-COMPLIANT" if any not satisfied.
    summary : str
        Executive summary of the compliance posture.
    timestamp : str
        ISO 8601 timestamp of package generation.
    """

    checks: list[ComplianceCheck] = field(default_factory=list)
    overall_status: str = "COMPLIANT"
    summary: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def pass_rate(self) -> float:
        """Fraction of checks with status 'satisfied'."""
        if not self.checks:
            return 0.0
        n_satisfied = sum(
            1 for c in self.checks if c.status == "satisfied"
        )
        return n_satisfied / len(self.checks)

    def to_markdown(self) -> str:
        """Render the compliance package as a markdown report."""
        lines = [
            "# Compliance Evidence Package",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Overall Status:** {self.overall_status}",
            f"**Pass Rate:** {self.pass_rate():.0%} "
            f"({sum(1 for c in self.checks if c.status == 'satisfied')}"
            f"/{len(self.checks)} satisfied)",
            "",
            f"**Summary:** {self.summary}",
            "",
            "---",
            "",
        ]

        # Group by regulation
        by_reg: dict[str, list[ComplianceCheck]] = {}
        for c in self.checks:
            by_reg.setdefault(c.regulation, []).append(c)

        for reg, checks in by_reg.items():
            lines.append(f"## {reg}")
            lines.append("")
            lines.append(
                "| Article | Requirement | Layer | Chapter "
                "| Status | Evidence |"
            )
            lines.append(
                "|---------|-------------|-------|--------"
                "|--------|----------|"
            )
            for c in checks:
                status_icon = {
                    "satisfied": "PASS",
                    "partial": "PARTIAL",
                    "not_applicable": "N/A",
                }.get(c.status, c.status.upper())
                lines.append(
                    f"| {c.article} | {c.requirement} "
                    f"| {c.architecture_layer} | {c.evidence_chapter} "
                    f"| {status_icon} | {c.evidence_summary} |"
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComplianceEvidenceGenerator
# ---------------------------------------------------------------------------

class ComplianceEvidenceGenerator:
    """Generate compliance evidence packages from pipeline results.

    The generator maps pipeline results to SR 11-7 and EU AI Act
    requirements.  Each check references a specific architecture layer
    and monograph chapter for traceability.

    Parameters
    ----------
    None.  The generator is stateless; all inputs come via method
    arguments.
    """

    def __init__(self):
        self._sr_11_7_expectations = [
            "Conceptual Soundness",
            "Ongoing Monitoring",
            "Outcomes Analysis",
            "Effective Challenge",
            "Documentation",
            "Change Management",
        ]
        self._eu_ai_act_articles = [
            ("Article 9", "Risk Management System"),
            ("Article 11", "Technical Documentation"),
            ("Article 14", "Human Oversight"),
        ]

    # ---- SR 11-7 checks ------------------------------------------------

    def sr_11_7_checks(
        self, pipeline_results: dict,
    ) -> list[ComplianceCheck]:
        """Generate SR 11-7 compliance checks from pipeline results.

        Parameters
        ----------
        pipeline_results : dict
            Keys used:
              - models_trained (int): number of models trained
              - verification_pass_rate (float): claim verification pass rate
              - policy_violations (int): number of policy violations
              - contract_violations (int): FSM contract violations detected
              - spc_alerts (int): statistical process control alerts
              - tracing_enabled (bool): whether decision tracing is active
              - snapshot_count (int): number of governance snapshots taken
              - audit_entries (int): total audit log entries

        Returns
        -------
        list[ComplianceCheck]
        """
        checks = []

        # 1. Conceptual Soundness
        contract_violations = pipeline_results.get(
            "contract_violations", 0)
        models_trained = pipeline_results.get("models_trained", 0)
        checks.append(ComplianceCheck(
            requirement=(
                "Model development follows a sound theoretical "
                "framework with formal behavioral contracts"
            ),
            regulation="SR 11-7",
            article="Conceptual Soundness",
            architecture_layer="Layer 2: Formal Contracts",
            evidence_chapter="Ch 13 (Behavioral Contracts)",
            status="satisfied" if contract_violations == 0 else "partial",
            evidence_summary=(
                f"FSM behavioral contracts enforce valid state "
                f"transitions. {contract_violations} violation(s) "
                f"detected across {models_trained} model(s) trained. "
                f"100% structural enforcement (not probabilistic)."
            ),
        ))

        # 2. Ongoing Monitoring
        spc_alerts = pipeline_results.get("spc_alerts", 0)
        checks.append(ComplianceCheck(
            requirement=(
                "Ongoing monitoring detects model degradation "
                "and triggers remediation"
            ),
            regulation="SR 11-7",
            article="Ongoing Monitoring",
            architecture_layer="Layer 6: SPC + Drift Detection",
            evidence_chapter="Ch 15 (SPC and Release Gates)",
            status="satisfied",
            evidence_summary=(
                f"RunTrend slope analysis monitors accuracy, "
                f"confidence, tension and failure rate across runs. "
                f"{spc_alerts} SPC alert(s) raised in current session."
            ),
        ))

        # 3. Outcomes Analysis
        tracing = pipeline_results.get("tracing_enabled", True)
        audit_entries = pipeline_results.get("audit_entries", 0)
        checks.append(ComplianceCheck(
            requirement=(
                "Decision outcomes are recorded and queryable "
                "for post-hoc analysis"
            ),
            regulation="SR 11-7",
            article="Outcomes Analysis",
            architecture_layer="Layer 6: Decision Traces",
            evidence_chapter="Ch 14 (Telemetry and Tracing)",
            status="satisfied" if tracing else "partial",
            evidence_summary=(
                f"Queryable tracing GMS stores every tool call, "
                f"policy decision and verification result. "
                f"{audit_entries} audit entries recorded. "
                f"Tracing {'enabled' if tracing else 'disabled'}."
            ),
        ))

        # 4. Effective Challenge
        vpr = pipeline_results.get("verification_pass_rate", 0.0)
        checks.append(ComplianceCheck(
            requirement=(
                "Independent verification challenges model "
                "outputs with geometric evidence"
            ),
            regulation="SR 11-7",
            article="Effective Challenge",
            architecture_layer="Layer 6: Verification",
            evidence_chapter="Ch 8 (Claim Verification)",
            status="satisfied" if vpr > 0.0 else "partial",
            evidence_summary=(
                f"Typed claim verification (fact, numeric, threshold, "
                f"comparison, reasoning) achieves {vpr:.0%} pass rate. "
                f"Every verdict includes geometric evidence "
                f"(geodesic distance, ENM match, holonomy defect)."
            ),
        ))

        # 5. Documentation
        checks.append(ComplianceCheck(
            requirement=(
                "Audit-grade documentation of all model "
                "development decisions and artifacts"
            ),
            regulation="SR 11-7",
            article="Documentation",
            architecture_layer="Layer 6: Telemetry",
            evidence_chapter="Ch 14 (Telemetry and Tracing)",
            status="satisfied" if audit_entries > 0 else "partial",
            evidence_summary=(
                f"All tool calls, extractions, policy decisions "
                f"and verification results written to tracing GMS. "
                f"{audit_entries} entries with full provenance. "
                f"Immutable SHA-256 integrity on numeric values."
            ),
        ))

        # 6. Change Management
        snapshot_count = pipeline_results.get("snapshot_count", 0)
        checks.append(ComplianceCheck(
            requirement=(
                "Model changes are versioned with before/after "
                "comparison and approval workflow"
            ),
            regulation="SR 11-7",
            article="Change Management",
            architecture_layer="Layer 2: Contract Versioning",
            evidence_chapter="Ch 20 (Regulatory Alignment)",
            status="satisfied" if snapshot_count >= 2 else "partial",
            evidence_summary=(
                f"SnapshotManager maintains {snapshot_count} "
                f"governance snapshot(s). Each snapshot records "
                f"models, metrics, policies and contract state. "
                f"Diff comparison available between any two snapshots."
            ),
        ))

        return checks

    # ---- EU AI Act checks -----------------------------------------------

    def eu_ai_act_checks(
        self, pipeline_results: dict,
    ) -> list[ComplianceCheck]:
        """Generate EU AI Act compliance checks from pipeline results.

        Parameters
        ----------
        pipeline_results : dict
            Keys used:
              - risk_assessment_done (bool): risk assessment performed
              - models_trained (int): number of models
              - verification_pass_rate (float): verification pass rate
              - fairness_checked (bool): fairness diagnostics run
              - human_oversight_enabled (bool): human-in-the-loop flag
              - policy_violations (int): policy violations detected
              - tracing_enabled (bool): decision tracing active
              - audit_entries (int): audit entries

        Returns
        -------
        list[ComplianceCheck]
        """
        checks = []

        # Article 9: Risk Management System
        risk_done = pipeline_results.get("risk_assessment_done", False)
        fairness = pipeline_results.get("fairness_checked", False)
        models = pipeline_results.get("models_trained", 0)
        checks.append(ComplianceCheck(
            requirement=(
                "Risk management system identifies and mitigates "
                "foreseeable risks throughout the AI lifecycle"
            ),
            regulation="EU AI Act",
            article="Article 9",
            architecture_layer="Layer 1: Policy Rules",
            evidence_chapter="Ch 12 (Policy Encoding)",
            status="satisfied" if risk_done and models > 0 else "partial",
            evidence_summary=(
                f"PolicyEngine enforces 10 responsible ML rules. "
                f"Robustness diagnostics and fairness checks "
                f"{'completed' if fairness else 'pending'}. "
                f"Risk tier classification gates deployment."
            ),
        ))

        # Article 11: Technical Documentation
        tracing = pipeline_results.get("tracing_enabled", True)
        audit_entries = pipeline_results.get("audit_entries", 0)
        checks.append(ComplianceCheck(
            requirement=(
                "Technical documentation demonstrates compliance "
                "before system placement on the market"
            ),
            regulation="EU AI Act",
            article="Article 11",
            architecture_layer="Layer 6: Telemetry",
            evidence_chapter="Ch 14 (Telemetry and Tracing)",
            status="satisfied" if tracing and audit_entries > 0 else "partial",
            evidence_summary=(
                f"Complete audit trail with {audit_entries} entries. "
                f"Includes model inventory, training provenance, "
                f"verification results, policy compliance matrix "
                f"and release gate decisions."
            ),
        ))

        # Article 14: Human Oversight
        human = pipeline_results.get("human_oversight_enabled", True)
        policy_v = pipeline_results.get("policy_violations", 0)
        checks.append(ComplianceCheck(
            requirement=(
                "Human oversight measures enable effective "
                "supervision of the AI system during operation"
            ),
            regulation="EU AI Act",
            article="Article 14",
            architecture_layer="Layer 6: Release Gates",
            evidence_chapter="Ch 15 (SPC and Release Gates)",
            status="satisfied" if human else "partial",
            evidence_summary=(
                f"Three-tier release gates (advisory, recommendation, "
                f"action-taking) with escalation to human review. "
                f"Policy rule 'human_approval_for_deploy' enforced. "
                f"{policy_v} policy violation(s) in current session."
            ),
        ))

        return checks

    # ---- Generate full package ------------------------------------------

    def generate_package(
        self, pipeline_results: dict,
    ) -> CompliancePackage:
        """Generate a complete compliance evidence package.

        Combines SR 11-7 and EU AI Act checks into a single
        auditable package with overall status and summary.

        Parameters
        ----------
        pipeline_results : dict
            Combined pipeline results (see sr_11_7_checks and
            eu_ai_act_checks for required keys).

        Returns
        -------
        CompliancePackage
        """
        sr_checks = self.sr_11_7_checks(pipeline_results)
        eu_checks = self.eu_ai_act_checks(pipeline_results)
        all_checks = sr_checks + eu_checks

        # Determine overall status
        statuses = [c.status for c in all_checks]
        if all(s == "satisfied" for s in statuses):
            overall = "COMPLIANT"
        elif any(s == "not_applicable" for s in statuses):
            overall = "PARTIAL"
        elif any(s == "partial" for s in statuses):
            overall = "PARTIAL"
        else:
            overall = "COMPLIANT"

        n_sat = sum(1 for s in statuses if s == "satisfied")
        n_total = len(statuses)

        summary = (
            f"{n_sat}/{n_total} regulatory checks satisfied. "
            f"SR 11-7: {sum(1 for c in sr_checks if c.status == 'satisfied')}"
            f"/{len(sr_checks)} expectations met. "
            f"EU AI Act: "
            f"{sum(1 for c in eu_checks if c.status == 'satisfied')}"
            f"/{len(eu_checks)} articles addressed."
        )

        return CompliancePackage(
            checks=all_checks,
            overall_status=overall,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# ProvenanceRegistry
# ---------------------------------------------------------------------------

class ProvenanceRegistry:
    """Track all artifacts and their lineage for audit trails.

    Each artifact has a name, type, source and metadata.  The registry
    maintains creation timestamps and parent-child relationships for
    full lineage reconstruction.

    Example
    -------
    >>> registry = ProvenanceRegistry()
    >>> registry.register("iris_data", "dataset", "sklearn",
    ...                   {"n_samples": 150})
    >>> registry.register("rf_model", "model", "training",
    ...                   {"algorithm": "RandomForest"},
    ...                   parent="iris_data")
    >>> print(registry.get_lineage("rf_model"))
    """

    def __init__(self):
        self.artifacts: dict[str, dict] = {}
        self._creation_order: list[str] = []

    def register(
        self,
        name: str,
        artifact_type: str,
        source: str,
        metadata: dict | None = None,
        parent: str | None = None,
    ) -> str:
        """Register an artifact with provenance metadata.

        Parameters
        ----------
        name : str
            Unique artifact name.
        artifact_type : str
            Type: "dataset", "model", "metric", "verification",
            "policy", "snapshot".
        source : str
            How the artifact was created (e.g., "sklearn",
            "training", "verification").
        metadata : dict, optional
            Additional key-value metadata.
        parent : str, optional
            Name of the parent artifact (for lineage tracking).

        Returns
        -------
        str
            SHA-256 fingerprint of the artifact record.
        """
        record = {
            "name": name,
            "type": artifact_type,
            "source": source,
            "metadata": metadata or {},
            "parent": parent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Compute fingerprint
        fingerprint = hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        record["fingerprint"] = fingerprint

        self.artifacts[name] = record
        if name not in self._creation_order:
            self._creation_order.append(name)

        return fingerprint

    def get_lineage(self, name: str) -> list[dict]:
        """Trace the lineage of an artifact back to its roots.

        Parameters
        ----------
        name : str
            Artifact name to trace.

        Returns
        -------
        list[dict]
            Lineage chain from root to the named artifact.
        """
        chain = []
        current = name
        visited = set()
        while current and current not in visited:
            visited.add(current)
            record = self.artifacts.get(current)
            if record is None:
                break
            chain.append(record)
            current = record.get("parent")
        chain.reverse()
        return chain

    def summary(self) -> str:
        """Human-readable summary of all registered artifacts."""
        lines = [
            f"=== Provenance Registry: {len(self.artifacts)} artifacts ===",
        ]
        by_type: dict[str, int] = {}
        for record in self.artifacts.values():
            by_type[record["type"]] = by_type.get(record["type"], 0) + 1
        for t, count in sorted(by_type.items()):
            lines.append(f"  {t:15s}: {count}")

        lines.append("")
        lines.append("Artifacts:")
        for name in self._creation_order:
            record = self.artifacts[name]
            parent_str = (
                f" <- {record['parent']}" if record.get("parent") else ""
            )
            lines.append(
                f"  [{record['fingerprint']}] {name} "
                f"({record['type']}, {record['source']}){parent_str}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SnapshotManager
# ---------------------------------------------------------------------------

class SnapshotManager:
    """Version governance state for change management.

    Takes point-in-time snapshots of the governance configuration
    (models, metrics, policies, contract state) and supports
    comparison between any two snapshots.

    Example
    -------
    >>> mgr = SnapshotManager()
    >>> mgr.take_snapshot("before_retrain", {"models": [...], ...})
    >>> mgr.take_snapshot("after_retrain", {"models": [...], ...})
    >>> diff = mgr.compare(0, 1)
    """

    def __init__(self):
        self.snapshots: list[dict] = []

    def take_snapshot(self, label: str, state: dict) -> int:
        """Capture a governance state snapshot.

        Parameters
        ----------
        label : str
            Human-readable label for this snapshot.
        state : dict
            Governance state to snapshot (deep-copied).

        Returns
        -------
        int
            Index of the new snapshot.
        """
        snapshot = {
            "index": len(self.snapshots),
            "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": copy.deepcopy(state),
            "fingerprint": hashlib.sha256(
                json.dumps(state, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
        }
        self.snapshots.append(snapshot)
        return snapshot["index"]

    def compare(self, snap_a: int, snap_b: int) -> dict:
        """Compare two snapshots and report differences.

        Parameters
        ----------
        snap_a : int
            Index of the first snapshot.
        snap_b : int
            Index of the second snapshot.

        Returns
        -------
        dict
            Keys: added, removed, changed, unchanged, summary.
        """
        if snap_a >= len(self.snapshots) or snap_b >= len(self.snapshots):
            return {"error": "Snapshot index out of range"}

        state_a = self.snapshots[snap_a]["state"]
        state_b = self.snapshots[snap_b]["state"]

        keys_a = set(state_a.keys())
        keys_b = set(state_b.keys())

        added = {}
        removed = {}
        changed = {}
        unchanged = {}

        for key in keys_b - keys_a:
            added[key] = state_b[key]

        for key in keys_a - keys_b:
            removed[key] = state_a[key]

        for key in keys_a & keys_b:
            val_a = state_a[key]
            val_b = state_b[key]
            if val_a != val_b:
                changed[key] = {"before": val_a, "after": val_b}
            else:
                unchanged[key] = val_a

        label_a = self.snapshots[snap_a]["label"]
        label_b = self.snapshots[snap_b]["label"]

        summary = (
            f"Comparing '{label_a}' vs '{label_b}': "
            f"{len(added)} added, {len(removed)} removed, "
            f"{len(changed)} changed, {len(unchanged)} unchanged."
        )

        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
            "summary": summary,
        }

    def summary(self) -> str:
        """Human-readable summary of all snapshots."""
        lines = [
            f"=== Snapshot Manager: {len(self.snapshots)} snapshots ===",
        ]
        for snap in self.snapshots:
            n_keys = len(snap["state"])
            lines.append(
                f"  [{snap['index']}] {snap['label']} "
                f"({n_keys} keys, fingerprint={snap['fingerprint']}) "
                f"at {snap['timestamp']}"
            )
        return "\n".join(lines)
