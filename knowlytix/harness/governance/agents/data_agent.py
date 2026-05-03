# SPDX-License-Identifier: Apache-2.0
"""Phase-based pipeline runner for governed data science workflows.

Wires together all components from Chapters 3-8 into a cohesive
execution pipeline.  Each phase is a named sequence of tool calls
that runs through the governed infrastructure:

    execute_tool -> extract_from_tool -> memory.store_findings

Four predefined phases cover the Adult Income classification workflow:

    1. DataPhase     -- load, summarize, correlate, preprocess
    2. TrainingPhase -- train 5 classifiers, produce leaderboard
    3. DiagnosticsPhase -- robustness + feature importance for top 2
    4. FairnessPhase -- AIR on sex and race for the top model

Built in Chapter 16 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from knowlytix.harness.governance.extractors import ExtractionResult, extract_from_tool
from knowlytix.harness.governance.memory import AgentMemory
from knowlytix.harness.governance.session import SessionState
from knowlytix.harness.governance.tools import execute_tool
from knowlytix.harness.governance.verifier import Claim, ClaimType


# ---------------------------------------------------------------------------
# Phase result data model
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    """Outcome of a single phase execution.

    Attributes
    ----------
    phase_name : str
        Human-readable phase label (e.g. "DataPhase").
    tools_called : list[str]
        Names of tools that executed successfully.
    triples_extracted : int
        Total knowledge triples extracted across all tool calls.
    enm_entries : int
        Total ENM (Exact Numerical Memory) entries stored.
    policy_blocks : int
        Number of tool calls that were blocked by policy rules.
    trace : list[dict]
        Per-tool-call execution records with timings and extraction
        counts, useful for governance reporting.
    claims : list[Claim]
        Typed claims accumulated during the phase for later verification.
    elapsed_s : float
        Wall-clock seconds for the entire phase.
    """

    phase_name: str = ""
    tools_called: list[str] = field(default_factory=list)
    triples_extracted: int = 0
    enm_entries: int = 0
    policy_blocks: int = 0
    trace: list[dict] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

class PhaseRunner:
    """Run a sequence of tool calls with extraction and memory storage.

    Orchestrates the governed pipeline for each tool call:

    1. ``execute_tool`` -- run the sklearn tool and capture output
    2. ``extract_from_tool`` -- apply three-tier extraction
    3. ``memory.store_findings`` -- commit triples and ENM values
    4. Accumulate claims for post-phase verification

    Parameters
    ----------
    session : SessionState
        Mutable session state shared across all tool calls.
    memory : AgentMemory
        GMS-backed memory store for triples and exact numerics.
    policy_rules : list[dict], optional
        Policy rules to evaluate (lightweight check; the full
        GovernedToolGateway from Ch 7 can be used for stricter gating).
    """

    def __init__(
        self,
        session: SessionState,
        memory: AgentMemory,
        policy_rules: list[dict] | None = None,
    ):
        self.session = session
        self.memory = memory
        self.policy_rules = policy_rules or []
        self._total_tools = 0
        self._total_triples = 0
        self._total_enm = 0
        self._total_blocks = 0

    # ----- Policy check (lightweight) ------------------------------------

    def _check_policy(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Lightweight policy check against loaded rules.

        Returns (allowed, reason).  If no rules match the tool_name,
        execution is allowed by default.
        """
        for rule in self.policy_rules:
            action = rule.get("action", "")
            if action != "*" and action != tool_name:
                continue

            rule_type = rule.get("rule_type", "requires")
            requirement = rule.get("requirement", "")

            if rule_type == "blocks":
                # Check if blocking condition is met
                if requirement == "raw_data_protection" and tool_name == "delete_raw_data":
                    return False, f"Policy {rule.get('rule_id', '?')}: {requirement}"

            elif rule_type == "requires":
                # Check session-level requirements
                if requirement == "preprocessing_done" and tool_name == "train_model":
                    dataset = args.get("dataset", "")
                    if dataset and dataset not in self.session.preprocessors:
                        return False, (
                            f"Policy {rule.get('rule_id', '?')}: "
                            f"preprocessing not done for '{dataset}'"
                        )
                if requirement == "min_two_models_trained" and tool_name == "leaderboard":
                    dataset = args.get("dataset", "")
                    n_models = sum(
                        1 for m in self.session.models.values()
                        if m.get("dataset") == dataset
                    )
                    if n_models < 2:
                        return False, (
                            f"Policy {rule.get('rule_id', '?')}: "
                            f"need >= 2 models, have {n_models}"
                        )

        return True, ""

    # ----- Phase execution -----------------------------------------------

    def run_phase(
        self,
        phase_name: str,
        tool_sequence: list[tuple[str, dict]],
    ) -> PhaseResult:
        """Execute a named phase as a sequence of governed tool calls.

        Parameters
        ----------
        phase_name : str
            Human-readable label for the phase.
        tool_sequence : list[tuple[str, dict]]
            Ordered list of (tool_name, arguments) pairs.

        Returns
        -------
        PhaseResult
            Aggregated outcome including extraction counts and claims.
        """
        result = PhaseResult(phase_name=phase_name)
        t0 = time.time()

        for tool_name, args in tool_sequence:
            entry = {
                "tool": tool_name,
                "args": {k: v for k, v in args.items()
                         if k not in ("X_train", "X_test")},
                "blocked": False,
                "reason": "",
                "triples": 0,
                "enm": 0,
            }

            # 1. Policy check
            allowed, reason = self._check_policy(tool_name, args)
            if not allowed:
                entry["blocked"] = True
                entry["reason"] = reason
                result.policy_blocks += 1
                result.trace.append(entry)
                continue

            # 2. Execute tool
            t_start = time.time()
            display, raw = execute_tool(tool_name, args, self.session)
            entry["elapsed_ms"] = round((time.time() - t_start) * 1000, 1)

            if isinstance(raw, dict) and "error" in raw:
                entry["error"] = raw["error"]
                result.trace.append(entry)
                continue

            result.tools_called.append(tool_name)

            # 3. Three-tier extraction
            extraction = extract_from_tool(
                tool_name, args, raw, display_string=display,
            )
            entry["triples"] = len(extraction.triples)
            entry["enm"] = len(extraction.enm_entries)
            entry["tier"] = extraction.tier

            result.triples_extracted += len(extraction.triples)
            result.enm_entries += len(extraction.enm_entries)

            # 4. Store in memory
            if extraction.triples or extraction.enm_entries:
                triple_tuples = [
                    (t.head, t.relation, t.tail) for t in extraction.triples
                ]
                enm_tuples = [
                    (e.category, e.entity_id, e.value)
                    for e in extraction.enm_entries
                ]
                self.memory.store_findings(triple_tuples, enm_tuples)

            # 5. Accumulate claims from extraction
            claims = _extraction_to_claims(tool_name, args, extraction)
            result.claims.extend(claims)

            result.trace.append(entry)

        result.elapsed_s = round(time.time() - t0, 2)

        # Update running totals
        self._total_tools += len(result.tools_called)
        self._total_triples += result.triples_extracted
        self._total_enm += result.enm_entries
        self._total_blocks += result.policy_blocks

        return result

    # ----- Cumulative statistics -----------------------------------------

    def cumulative_stats(self) -> dict:
        """Return cumulative statistics across all phases."""
        mem_stats = self.memory.memory_stats()
        return {
            "total_tools": self._total_tools,
            "total_triples": self._total_triples,
            "total_enm_entries": self._total_enm,
            "total_policy_blocks": self._total_blocks,
            "memory_entities": mem_stats.get("entities", 0),
            "memory_relations": mem_stats.get("relations", 0),
            "memory_triples": mem_stats.get("triples", 0),
            "memory_enm": mem_stats.get("enm_entries", 0),
        }


# ---------------------------------------------------------------------------
# Claim generation from extraction results
# ---------------------------------------------------------------------------

def _extraction_to_claims(
    tool_name: str,
    args: dict,
    extraction: ExtractionResult,
) -> list[Claim]:
    """Convert extraction results into typed claims for verification.

    Generates NUMERIC claims for every ENM entry and FACT claims for
    key knowledge triples.
    """
    claims: list[Claim] = []

    # Numeric claims from ENM entries
    for enm in extraction.enm_entries:
        claims.append(Claim(
            type=ClaimType.NUMERIC,
            text=f"{enm.entity_id} = {enm.value}",
            metadata={
                "category": enm.category,
                "entity_id": enm.entity_id,
                "claimed_value": enm.value,
            },
        ))

    # Fact claims from key triples
    for t in extraction.triples:
        if t.relation in (
            "is_type", "trained_on", "has_task_type", "has_target",
            "is_best_model_by", "diagnosed_for", "passes_4_5_rule",
            "is_top_driver_for",
        ):
            claims.append(Claim(
                type=ClaimType.FACT,
                text=f"{t.head} {t.relation} {t.tail}",
                metadata={
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                },
            ))

    return claims


# ---------------------------------------------------------------------------
# Predefined phase sequences for Adult Income
# ---------------------------------------------------------------------------

def adult_data_phase() -> list[tuple[str, dict]]:
    """Phase 1: Load Adult Income, summarize, correlate, preprocess."""
    return [
        ("load_dataset", {"name": "adult", "builtin": "adult",
                          "random_state": 42}),
        ("data_summary", {"dataset": "adult"}),
        ("eda_correlation", {"dataset": "adult"}),
        ("preprocess_data", {"dataset": "adult",
                             "impute_method": "median",
                             "scale_method": "standard",
                             "encode_method": "ordinal"}),
    ]


def adult_training_phase() -> list[tuple[str, dict]]:
    """Phase 2: Train 5 classifiers and produce leaderboard."""
    models = [
        ("LR",  "LogisticRegression",
         {"max_iter": 1000, "random_state": 42}),
        ("RF",  "RandomForestClassifier",
         {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
        ("GB",  "GradientBoostingClassifier",
         {"n_estimators": 100, "random_state": 42}),
        ("DT",  "DecisionTreeClassifier",
         {"max_depth": 5, "random_state": 42}),
        ("SVC", "SVC",
         {"kernel": "rbf", "random_state": 42, "probability": True}),
    ]
    seq: list[tuple[str, dict]] = []
    for name, mtype, params in models:
        seq.append(("train_model", {
            "dataset": "adult",
            "model_type": mtype,
            "model_name": name,
            "params": params,
        }))
    seq.append(("leaderboard", {"dataset": "adult"}))
    return seq


def adult_diagnostics_phase(
    top_models: list[str] | None = None,
) -> list[tuple[str, dict]]:
    """Phase 3: Robustness + feature importance for top 2 models.

    Parameters
    ----------
    top_models : list[str], optional
        Names of the top 2 models from the leaderboard.
        Defaults to ["GB", "RF"] if not provided.
    """
    if top_models is None:
        top_models = ["GB", "RF"]
    seq: list[tuple[str, dict]] = []
    for model in top_models[:2]:
        seq.append(("diagnose_robustness", {
            "dataset": "adult",
            "model_name": model,
        }))
        seq.append(("explain_importance", {
            "dataset": "adult",
            "model_name": model,
        }))
    return seq


def adult_fairness_phase(
    top_model: str | None = None,
) -> list[tuple[str, dict]]:
    """Phase 4: Fairness on sex and race for the top model.

    Parameters
    ----------
    top_model : str, optional
        Name of the top model. Defaults to "GB" if not provided.
    """
    if top_model is None:
        top_model = "GB"
    return [
        ("diagnose_fairness", {
            "dataset": "adult",
            "model_name": top_model,
            "protected_attribute": "sex",
        }),
        ("diagnose_fairness", {
            "dataset": "adult",
            "model_name": top_model,
            "protected_attribute": "race",
        }),
    ]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_governance_report(
    phase_results: list[PhaseResult],
    verification_result: Any,
    session: SessionState,
    cumulative: dict,
) -> str:
    """Build a markdown governance report from pipeline results.

    Parameters
    ----------
    phase_results : list[PhaseResult]
        Results from each phase.
    verification_result : OverallVerdict
        Result from ClaimVerifier.verify_all().
    session : SessionState
        Session containing trained models and results.
    cumulative : dict
        Cumulative statistics from PhaseRunner.cumulative_stats().

    Returns
    -------
    str
        Complete markdown-formatted governance report.
    """
    lines: list[str] = []

    # --- Executive Summary ---
    lines.append("# Governance Report: Adult Income Classification")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("- **Dataset:** Adult Income (OpenML)")
    n_models = len(session.models)
    lines.append(f"- **Models trained:** {n_models}")
    lines.append(f"- **Total tool calls:** {cumulative['total_tools']}")
    lines.append(f"- **Knowledge triples extracted:** {cumulative['total_triples']}")
    lines.append(f"- **ENM entries stored:** {cumulative['total_enm_entries']}")
    lines.append(f"- **Policy blocks:** {cumulative['total_policy_blocks']}")
    verdict_status = getattr(verification_result, 'overall_status', 'N/A')
    pass_rate = getattr(verification_result, 'pass_rate', 0.0)
    lines.append(f"- **Verification verdict:** {verdict_status} "
                 f"({pass_rate:.1%} pass rate)")
    lines.append("")

    # --- Phase Summary ---
    lines.append("## Phase Summary")
    lines.append("")
    lines.append("| Phase | Tools | Triples | ENM | Blocks | Time (s) |")
    lines.append("|-------|------:|--------:|----:|-------:|---------:|")
    for pr in phase_results:
        lines.append(
            f"| {pr.phase_name} | {len(pr.tools_called)} | "
            f"{pr.triples_extracted} | {pr.enm_entries} | "
            f"{pr.policy_blocks} | {pr.elapsed_s:.1f} |"
        )
    lines.append("")

    # --- Model Comparison Table ---
    lines.append("## Model Comparison")
    lines.append("")
    header = "| Model | Type | Accuracy | F1 (macro) | AUC | Train-Test Gap |"
    lines.append(header)
    lines.append("|-------|------|----------|------------|-----|----------------|")
    for mname, minfo in session.models.items():
        metrics = session.results.get(mname, {})
        mtype = minfo.get("model_type", "?")
        acc = metrics.get("test_accuracy", "N/A")
        f1 = metrics.get("test_f1_macro", "N/A")
        auc = metrics.get("test_auc", metrics.get("test_auc_ovr", "N/A"))
        gap = metrics.get("train_test_gap", "N/A")
        lines.append(
            f"| {mname} | {mtype} | "
            f"{acc if isinstance(acc, str) else f'{acc:.4f}'} | "
            f"{f1 if isinstance(f1, str) else f'{f1:.4f}'} | "
            f"{auc if isinstance(auc, str) else f'{auc:.4f}'} | "
            f"{gap if isinstance(gap, str) else f'{gap:.4f}'} |"
        )
    lines.append("")

    # --- Robustness Analysis ---
    lines.append("## Robustness Analysis")
    lines.append("")
    for mname in session.models:
        rob_key = f"{mname}_robustness"
        if rob_key in session.results:
            lines.append(f"**{mname}:**")
            rob = session.results[rob_key]
            lines.append("")
            lines.append("| Noise Level | Accuracy |")
            lines.append("|-------------|----------|")
            for level, vals in rob.items():
                acc = vals.get("accuracy", "N/A")
                lines.append(
                    f"| {level} | "
                    f"{acc if isinstance(acc, str) else f'{acc:.4f}'} |"
                )
            lines.append("")

    # --- Fairness Analysis ---
    lines.append("## Fairness Analysis")
    lines.append("")
    for mname in session.models:
        fair_key = f"{mname}_fairness"
        if fair_key in session.results:
            fair = session.results[fair_key]
            attr = fair.get("protected_attribute", "?")
            air = fair.get("adverse_impact_ratio", "N/A")
            dp = fair.get("demographic_parity_diff", "N/A")
            passes = fair.get("air_passes_4_5_rule", "N/A")
            lines.append(f"**{mname} -- {attr}:**")
            lines.append(f"- Adverse Impact Ratio: {air}")
            lines.append(f"- Demographic Parity Diff: {dp}")
            lines.append(f"- Passes 4/5 Rule: {passes}")
            rates = fair.get("group_positive_rates", {})
            if rates:
                lines.append(f"- Group selection rates: {rates}")
            lines.append("")

    # --- Policy Compliance ---
    lines.append("## Policy Compliance")
    lines.append("")
    lines.append(f"- Total policy blocks: {cumulative['total_policy_blocks']}")
    lines.append("- All phases completed with governance tracing")
    lines.append("")

    # --- Verification Summary ---
    lines.append("## Verification Summary")
    lines.append("")
    n_passed = getattr(verification_result, 'n_passed', 0)
    n_total = getattr(verification_result, 'n_total', 0)
    lines.append(f"- Claims verified: {n_passed}/{n_total}")
    lines.append(f"- Pass rate: {pass_rate:.1%}")
    sev = getattr(verification_result, 'severity_distribution', {})
    if sev:
        lines.append(f"- Severity distribution: {sev}")
    lines.append("")

    # --- Release Gate Decision ---
    lines.append("## Release Gate Decision")
    lines.append("")
    if verdict_status == "PASS" and pass_rate >= 0.8:
        lines.append("**APPROVED** -- All governance checks passed. "
                      "Model is cleared for deployment review.")
    elif pass_rate >= 0.6:
        lines.append("**CONDITIONAL** -- Partial governance compliance. "
                      "Manual review required before deployment.")
    else:
        lines.append("**BLOCKED** -- Insufficient governance compliance. "
                      "Remediation required.")
    lines.append("")

    return "\n".join(lines)
