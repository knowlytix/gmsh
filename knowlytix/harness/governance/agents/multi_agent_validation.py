# SPDX-License-Identifier: Apache-2.0
"""OOD validation for the multi-agent governance surface.

A :class:`LabeledMultiAgentDecision` carries a ``(role_name, tool_name)``
pair plus the ground-truth verdict the orchestrator should produce
(``allow`` for permitted calls, ``block`` for explicitly forbidden,
``deny`` for default-deny on unknown). The cohort is labeled by
construction from the AgentRole tool sets, so truth is deterministic
and audit-defensible.

Adult Income context
--------------------
The cohort is grounded in the Adult Income study used across
Ch 6/7/8/10/11/12/15/16. Each probe carries the surrounding pipeline
state (``dataset='adult'``, model_name in {LR, RF, GB, DT, SVC}) so
the verdict is reproducible given the predefined role set.

A high-stakes deployment regenerates the labeled cohort whenever
roles are edited, then re-validates against the deployed orchestrator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from knowlytix.harness.governance.agents.multi_agent import (
    DATA_AGENT_ROLE,
    DIAGNOSTICS_AGENT_ROLE,
    FAIRNESS_AGENT_ROLE,
    TRAINING_AGENT_ROLE,
    AgentRole,
    MultiAgentOrchestrator,
)

# All Adult Income tool names the pipeline knows about.
_ADULT_TOOL_VOCAB = (
    "load_dataset", "data_summary", "eda_correlation", "preprocess_data",
    "feature_select", "train_model", "predict", "leaderboard",
    "diagnose_robustness", "diagnose_fairness", "diagnose_accuracy",
    "diagnose_resilience", "diagnose_reliability",
    "explain_importance", "explain_pdp", "explain_shap", "explain_ale",
    "interpret_effects", "interpret_fi",
)

_ADULT_MODELS = ("LR", "RF", "GB", "DT", "SVC")


# ---------------------------------------------------------------------------
# Labeled probe + report
# ---------------------------------------------------------------------------

@dataclass
class LabeledMultiAgentDecision:
    """A (role, tool) pair paired with its ground-truth verdict.

    Attributes
    ----------
    probe_id : str
        Stable identifier for traceability.
    role_name : str
        Name of the role being probed (matches a key in
        ``orchestrator.agents``).
    tool_name : str
        Tool the agent is asked to execute.
    args : dict
        Tool arguments the probe supplies.
    expected_decision : str
        Ground truth: "allow" | "block" | "deny".
        - "allow" : tool in role.allowed_tools, must succeed.
        - "block" : tool in role.forbidden_tools, must raise
          PermissionError citing explicit forbidden.
        - "deny"  : tool not in either set, must raise
          PermissionError citing default-deny on unknown.
    source_class : str
        Free-form provenance bucket.
    """

    probe_id: str
    role_name: str
    tool_name: str
    args: dict
    expected_decision: str   # "allow" | "block" | "deny"
    source_class: str = "unspecified"


@dataclass
class MultiAgentOODReport:
    n: int
    decision_matches: int
    per_decision: dict[str, dict[str, int]]
    per_role: dict[str, dict[str, int]] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    cohort_hash: str = ""


def _observe(
    orch: MultiAgentOrchestrator, probe: LabeledMultiAgentDecision,
) -> str:
    """Run a probe and return the observed decision string.

    Returns one of ``"allow"`` (tool ran), ``"block"``,
    ``"deny"`` --- the latter two distinguished by whether the
    PermissionError mentions ``explicitly forbidden`` or
    ``default-deny``.
    """
    agent = orch.agents.get(probe.role_name)
    if agent is None:
        return "unknown_role"
    blocked, msg = orch.test_boundary(
        probe.role_name, probe.tool_name, probe.args)
    if not blocked:
        return "allow"
    if "explicitly forbidden" in msg:
        return "block"
    return "deny"


def validate_multi_agent(
    orch: MultiAgentOrchestrator,
    cohort: list[LabeledMultiAgentDecision],
) -> MultiAgentOODReport:
    """Run every probe against the orchestrator; report per-cell counts.

    Decision-match rule: the observed decision must equal the
    probe's ``expected_decision``. Per-role and per-class
    breakdown stratify the rate so a reviewer sees which role
    or source class slipped.
    """
    n = len(cohort)
    if n == 0:
        return MultiAgentOODReport(0, 0, {}, {}, {})

    decision_matches = 0
    per_decision: dict[str, dict[str, int]] = {
        d: {"tp": 0, "fp": 0, "n": 0} for d in ("allow", "block", "deny")
    }
    per_role: dict[str, dict[str, int]] = {}
    per_class: dict[str, dict[str, int]] = {}

    for probe in cohort:
        observed = _observe(orch, probe)
        per_decision.setdefault(
            probe.expected_decision, {"tp": 0, "fp": 0, "n": 0})
        per_decision[probe.expected_decision]["n"] += 1
        if observed == probe.expected_decision:
            decision_matches += 1
            per_decision[probe.expected_decision]["tp"] += 1
        else:
            per_decision[probe.expected_decision].setdefault(
                "miss_to:" + observed, 0)
            per_decision[probe.expected_decision][
                "miss_to:" + observed] += 1

        rcell = per_role.setdefault(
            probe.role_name, {"n": 0, "matches": 0})
        rcell["n"] += 1
        if observed == probe.expected_decision:
            rcell["matches"] += 1

        ccell = per_class.setdefault(
            probe.source_class, {"n": 0, "matches": 0})
        ccell["n"] += 1
        if observed == probe.expected_decision:
            ccell["matches"] += 1

    per_class_out: dict[str, dict[str, float]] = {
        sc: {
            "n": cell["n"],
            "match_rate": (cell["matches"] / cell["n"]
                           if cell["n"] > 0 else 0.0),
        }
        for sc, cell in per_class.items()
    }

    return MultiAgentOODReport(
        n=n,
        decision_matches=decision_matches,
        per_decision=per_decision,
        per_role=per_role,
        per_class=per_class_out,
        cohort_hash=cohort_fingerprint(cohort),
    )


# ---------------------------------------------------------------------------
# Adult Income labeled cohort
# ---------------------------------------------------------------------------

_DEFAULT_ROLES = {
    "DataAgent": DATA_AGENT_ROLE,
    "TrainingAgent": TRAINING_AGENT_ROLE,
    "DiagnosticsAgent": DIAGNOSTICS_AGENT_ROLE,
    "FairnessAgent": FAIRNESS_AGENT_ROLE,
}


def _expected_for(role: AgentRole, tool: str) -> str:
    if tool in role.forbidden_tools:
        return "block"
    if tool in role.allowed_tools:
        return "allow"
    return "deny"


def adult_income_multi_agent_cohort(
    roles: dict[str, AgentRole] | None = None,
) -> list[LabeledMultiAgentDecision]:
    """Hand-curated 30-probe Adult Income cohort.

    Spans (role, tool) combinations that exercise:
      - Permitted calls (positive controls): each role's allowed tools
        on its canonical Adult Income arguments.
      - Forbidden calls (negative controls): each role attempting
        a tool from another role's allowed set.
      - Default-deny (negative controls): each role attempting an
        unknown tool name.

    Mirrors :func:`governed_agent.release_gate_validation.adult_income_labeled_run_cohort`
    for the multi-agent surface.
    """
    if roles is None:
        roles = _DEFAULT_ROLES

    out: list[LabeledMultiAgentDecision] = []
    idx = 0

    # Permitted calls: every role x every allowed tool with canonical args
    canonical_args = {
        "load_dataset":     {"name": "adult", "builtin": "adult"},
        "data_summary":     {"dataset": "adult"},
        "eda_correlation":  {"dataset": "adult"},
        "preprocess_data":  {"dataset": "adult"},
        "train_model":      {"dataset": "adult",
                             "model_type": "RandomForestClassifier",
                             "model_name": "RF"},
        "predict":          {"dataset": "adult", "model_name": "RF"},
        "leaderboard":      {"dataset": "adult"},
        "feature_select":   {"dataset": "adult"},
        "diagnose_robustness": {"dataset": "adult", "model_name": "GB"},
        "diagnose_fairness":   {"dataset": "adult", "model_name": "GB",
                                "protected_attribute": "sex"},
        "explain_importance":  {"dataset": "adult", "model_name": "GB"},
        "explain_pdp":         {"dataset": "adult", "model_name": "GB"},
        "explain_shap":        {"dataset": "adult", "model_name": "GB"},
    }

    for role_name, role in roles.items():
        for tool in sorted(role.allowed_tools):
            args = canonical_args.get(tool, {})
            out.append(LabeledMultiAgentDecision(
                probe_id=f"adult_perm_{idx:03d}",
                role_name=role_name, tool_name=tool, args=args,
                expected_decision="allow",
                source_class=f"permitted_{role_name}",
            ))
            idx += 1

    # Forbidden calls: every role attempts a tool from its forbidden set
    for role_name, role in roles.items():
        for tool in sorted(role.forbidden_tools):
            args = canonical_args.get(tool, {})
            out.append(LabeledMultiAgentDecision(
                probe_id=f"adult_forb_{idx:03d}",
                role_name=role_name, tool_name=tool, args=args,
                expected_decision="block",
                source_class=f"forbidden_{role_name}",
            ))
            idx += 1

    # Default-deny: every role attempts an unknown tool name
    unknown_tools = ["train_model_v2", "drop_table", "delete_user",
                     "exec_arbitrary_code"]
    for role_name in roles:
        for tool in unknown_tools:
            out.append(LabeledMultiAgentDecision(
                probe_id=f"adult_deny_{idx:03d}",
                role_name=role_name, tool_name=tool, args={},
                expected_decision="deny",
                source_class=f"unknown_{role_name}",
            ))
            idx += 1

    return out


def large_adult_income_multi_agent_cohort(
    *,
    n_per_class: int = 60,
    seed: int = 42,
    roles: dict[str, AgentRole] | None = None,
) -> list[LabeledMultiAgentDecision]:
    """Programmatic >=1000-probe Adult Income cohort for serious audit.

    Sample n_per_class probes from each of the 12 (role, decision)
    strata: 4 roles x {allow, block, deny}. Total: 4 * 3 * n_per_class
    (default n_per_class=60 -> 720 probes). For >=1000 use n_per_class=90.
    """
    if roles is None:
        roles = _DEFAULT_ROLES
    rng = np.random.default_rng(seed)
    out: list[LabeledMultiAgentDecision] = []
    idx = 0

    canonical_args = {
        "load_dataset":     {"name": "adult", "builtin": "adult"},
        "data_summary":     {"dataset": "adult"},
        "eda_correlation":  {"dataset": "adult"},
        "preprocess_data":  {"dataset": "adult"},
        "train_model":      {"dataset": "adult"},
        "predict":          {"dataset": "adult"},
        "leaderboard":      {"dataset": "adult"},
        "feature_select":   {"dataset": "adult"},
        "diagnose_robustness": {"dataset": "adult"},
        "diagnose_fairness":   {"dataset": "adult"},
        "explain_importance":  {"dataset": "adult"},
        "explain_pdp":         {"dataset": "adult"},
        "explain_shap":        {"dataset": "adult"},
    }

    unknown_seeds = ["v2", "v3", "alt", "shadow", "_internal",
                     "_test", "_admin", "_super", "_legacy",
                     "_helper", "_debug"]

    for role_name, role in roles.items():
        # ---- allow stratum ----
        allow_pool = sorted(role.allowed_tools)
        if not allow_pool:
            allow_pool = ["data_summary"]   # fallback
        for k in range(n_per_class):
            tool = allow_pool[k % len(allow_pool)]
            args = dict(canonical_args.get(tool, {}))
            # vary model_name slightly for tools that take it
            if "model_name" in args or tool == "train_model":
                args.setdefault(
                    "model_name", _ADULT_MODELS[k % len(_ADULT_MODELS)])
            if tool == "train_model":
                args.setdefault("model_type", "RandomForestClassifier")
            if tool == "diagnose_fairness":
                args.setdefault("protected_attribute",
                                "sex" if k % 2 == 0 else "race")
            out.append(LabeledMultiAgentDecision(
                probe_id=f"adult_synth_{idx:05d}",
                role_name=role_name, tool_name=tool, args=args,
                expected_decision="allow",
                source_class=f"synth_permitted_{role_name}",
            ))
            idx += 1

        # ---- block stratum ----
        forb_pool = sorted(role.forbidden_tools)
        if not forb_pool:
            # Empty forbidden set means we can only test default-deny
            forb_pool = ["train_model"]
        for k in range(n_per_class):
            tool = forb_pool[k % len(forb_pool)]
            args = dict(canonical_args.get(tool, {}))
            if tool == "train_model":
                args.setdefault("model_type", "RandomForestClassifier")
                args.setdefault("model_name",
                                _ADULT_MODELS[k % len(_ADULT_MODELS)])
            out.append(LabeledMultiAgentDecision(
                probe_id=f"adult_synth_{idx:05d}",
                role_name=role_name, tool_name=tool, args=args,
                expected_decision="block",
                source_class=f"synth_forbidden_{role_name}",
            ))
            idx += 1

        # ---- deny stratum ----
        for k in range(n_per_class):
            base = ["train", "load", "predict", "explain"][
                k % 4]
            suffix = unknown_seeds[k % len(unknown_seeds)]
            tool = f"{base}{suffix}"
            args = {}
            out.append(LabeledMultiAgentDecision(
                probe_id=f"adult_synth_{idx:05d}",
                role_name=role_name, tool_name=tool, args=args,
                expected_decision="deny",
                source_class=f"synth_unknown_{role_name}",
            ))
            idx += 1

    return out


# ---------------------------------------------------------------------------
# Cohort fingerprint + persistence
# ---------------------------------------------------------------------------

def cohort_fingerprint(
    cohort: list[LabeledMultiAgentDecision],
) -> str:
    """Deterministic SHA-256 over the canonical cohort representation."""
    payload = []
    for p in cohort:
        payload.append({
            "probe_id": p.probe_id,
            "role_name": p.role_name,
            "tool_name": p.tool_name,
            "args": dict(sorted(p.args.items())) if p.args else {},
            "expected_decision": p.expected_decision,
            "source_class": p.source_class,
        })
    canon = json.dumps(
        sorted(payload, key=lambda d: d["probe_id"]),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def save_cohort(
    cohort: list[LabeledMultiAgentDecision], path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(probe) for probe in cohort]
    p.write_text(json.dumps(payload, indent=2))


def load_cohort(path: str | Path) -> list[LabeledMultiAgentDecision]:
    p = Path(path)
    payload = json.loads(p.read_text())
    return [LabeledMultiAgentDecision(**d) for d in payload]
