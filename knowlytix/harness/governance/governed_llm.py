# SPDX-License-Identifier: Apache-2.0
"""Constrained LLM integration for policy-governed agents.

Implements the "LLM proposes, GMS approves" pattern with three controlled
channels:

    1. **Interpretation (ungated):** LLM reads tool results and produces
       narrative.  Display-only --- does not affect agent state.
    2. **Proposals (plausibility-gated):** LLM proposes next tool calls.
       Each goes through ``LogicVerifier.verify_action()``.  Rejected
       proposals are not executed.
    3. **Claims (contradiction-gated):** LLM extracts factual claims.
       Each goes through ``LogicVerifier.verify_domain_triples()``.
       Rejected claims are not learned into GMS.

Fail-closed design: if the GMS store has no trained model, **all**
proposals are rejected.  The agent falls back to pure GMS-driven
execution, preventing the "authority leak" where an LLM operates
without geometric verification.

Built in Chapter 11 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from knowlytix.knowledge.extract import ExtractedTriple

# Reuse the production LogicVerifier from knowlytix.harness._verifier so
# governance can import it without a private-tree dep.
from knowlytix.harness._verifier import LogicVerifier  # noqa: F401

if TYPE_CHECKING:
    from knowlytix.knowledge.llm_backend import LLMBackend
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.governance.llm_co_attestor import LLMCoAttestor


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #

@dataclass
class Proposal:
    """A proposed tool call with governance metadata."""
    tool: str
    args: dict
    reason: str
    approved: bool = False
    score: float | None = None
    rejection_reason: str = ""


@dataclass
class GatedClaim:
    """A factual claim extracted from LLM narrative, subject to gating."""
    text: str
    triple: ExtractedTriple
    approved: bool = False
    tension: float | None = None
    rejection_reason: str = ""


# ------------------------------------------------------------------ #
# Deterministic data-science workflow templates
# ------------------------------------------------------------------ #

# Canonical tool ordering for a supervised-learning pipeline.
# MockConstrainedLLM uses this to generate proposals.
_WORKFLOW_STEPS = [
    "load_dataset",
    "preprocess_data",
    "train_model",
    "evaluate_model",
    "diagnose_accuracy",
    "explain_model",
]

# Template narratives for Channel 1 (interpretation), keyed by tool name.
_NARRATIVE_TEMPLATES: dict[str, str] = {
    "load_dataset": (
        "The dataset has been loaded successfully.  It contains "
        "{n_rows} rows and {n_features} features.  The target variable "
        "is '{target}'.  Initial inspection shows the data is ready for "
        "preprocessing."
    ),
    "preprocess_data": (
        "Preprocessing is complete.  The training set has {n_train} "
        "samples and the test set has {n_test} samples.  Features have "
        "been scaled and missing values handled."
    ),
    "train_model": (
        "The model '{model_name}' has been trained on the training set.  "
        "Training completed in {n_epochs} epochs.  The model is ready "
        "for evaluation on the held-out test set."
    ),
    "evaluate_model": (
        "Evaluation on the test set shows accuracy={accuracy:.4f} and "
        "weighted F1={f1:.4f}.  These metrics will be stored in the "
        "knowledge graph for downstream verification."
    ),
    "diagnose_accuracy": (
        "Accuracy diagnosis reveals {diagnosis}.  The train-test gap "
        "is {gap:.4f}, which {gap_assessment}."
    ),
    "explain_model": (
        "Feature importance analysis identifies {top_feature} as the "
        "most influential feature, contributing {importance:.1%} to "
        "model predictions.  This aligns with domain knowledge about "
        "the dataset."
    ),
}

# Default fill values for narrative templates when raw_result is sparse.
_DEFAULTS: dict[str, dict] = {
    "load_dataset": {
        "n_rows": 150, "n_features": 4, "target": "species",
    },
    "preprocess_data": {
        "n_train": 120, "n_test": 30,
    },
    "train_model": {
        "model_name": "RandomForest", "n_epochs": 100,
    },
    "evaluate_model": {
        "accuracy": 1.0, "f1": 1.0,
    },
    "diagnose_accuracy": {
        "diagnosis": "no overfitting detected",
        "gap": 0.0, "gap_assessment": "indicates good generalization",
    },
    "explain_model": {
        "top_feature": "petal_length", "importance": 0.45,
    },
}

# Template claims for Channel 3, keyed by tool name.
# Each entry is a list of (text, head, relation, tail[, numeric]).
_CLAIM_TEMPLATES: dict[str, list[tuple]] = {
    "load_dataset": [
        ("Dataset loaded with 150 rows",
         "iris_dataset", "has_row_count", "150", 150.0),
        ("Dataset has 4 features",
         "iris_dataset", "has_feature_count", "4", 4.0),
    ],
    "preprocess_data": [
        ("Train-test split uses 80-20 ratio",
         "preprocessing", "uses_split_ratio", "0.8", 0.8),
    ],
    "train_model": [
        ("RandomForest trained on iris dataset",
         "RandomForest", "trained_on", "iris_dataset", None),
        ("RandomForest uses 100 estimators",
         "RandomForest", "has_n_estimators", "100", 100.0),
    ],
    "evaluate_model": [
        ("RandomForest test accuracy is 1.0",
         "RandomForest", "has_accuracy", "1.0", 1.0),
        ("RandomForest weighted F1 is 1.0",
         "RandomForest", "has_f1", "1.0", 1.0),
    ],
    "diagnose_accuracy": [
        ("No overfitting detected in RandomForest",
         "RandomForest", "has_diagnosis", "no_overfitting", None),
        ("Train-test gap is 0.0",
         "RandomForest", "has_train_test_gap", "0.0", 0.0),
    ],
    "explain_model": [
        ("petal_length is the most important feature",
         "petal_length", "is_most_important_for", "RandomForest", None),
        ("RandomForest requires StandardScaler",
         "RandomForest", "requires", "StandardScaler", None),
    ],
}


# ------------------------------------------------------------------ #
# MockConstrainedLLM — deterministic, no API keys
# ------------------------------------------------------------------ #

class MockConstrainedLLM:
    """Deterministic mock that simulates LLM behavior for the tutorial.

    Uses pattern matching and template-based generation rather than
    actual LLM calls.  All outputs are repeatable across runs.

    Parameters
    ----------
    store : GMSExpertStore
        The GMS store providing geometric verification.
    verifier : LogicVerifier
        Wraps the existing ``LogicVerifier`` from
        ``agents.ds_agent.agents.base``.
    """

    def __init__(self, store: "GMSExpertStore", verifier: LogicVerifier):
        self.store = store
        self.verifier = verifier

    # ── helpers ──

    def _has_trained_model(self) -> bool:
        """True if the store has a trained GKG model."""
        return self.store.model is not None

    # ── Channel 1: Interpretation (ungated) ──

    def interpret(
        self,
        tool_name: str,
        raw_result: dict,
        context: str = "",
    ) -> str:
        """Produce a narrative interpretation of a tool result.

        This channel is **ungated** --- it produces display-only text
        that does not affect agent state or the knowledge graph.

        Parameters
        ----------
        tool_name : str
            The tool whose result is being interpreted.
        raw_result : dict
            The structured result returned by the tool.
        context : str
            Optional context string (e.g. current phase).

        Returns
        -------
        str
            A human-readable narrative interpretation.
        """
        template = _NARRATIVE_TEMPLATES.get(tool_name)
        if template is None:
            return (
                f"Tool '{tool_name}' completed successfully.  "
                f"Result contains {len(raw_result)} fields."
            )

        # Merge defaults with actual results
        defaults = _DEFAULTS.get(tool_name, {})
        values = {**defaults, **raw_result}

        try:
            return template.format(**values)
        except (KeyError, ValueError):
            # Graceful fallback if template keys don't match
            return template.format(**defaults)

    # ── Channel 2: Proposals (plausibility-gated) ──

    def propose_next(
        self,
        context: str,
        completed: list[str],
    ) -> list[Proposal]:
        """Propose next tool calls based on completed steps.

        Each proposal is gated through ``LogicVerifier.verify_action()``.
        If the store has no trained model, **all** proposals are rejected
        (fail-closed design).

        Parameters
        ----------
        context : str
            The current workflow context entity (e.g. ``"training_phase"``).
        completed : list[str]
            List of tool names already executed.

        Returns
        -------
        list[Proposal]
            Proposals with ``approved`` and ``score`` fields populated.
        """
        # Determine candidate tools from canonical ordering
        candidates = []
        for step in _WORKFLOW_STEPS:
            if step not in completed:
                candidates.append(step)
        if not candidates:
            return []

        # Also inject a deliberately bad proposal for demonstration
        # The mock LLM sometimes proposes a tool that has already been run
        proposals = []
        for tool in candidates[:3]:  # propose up to 3 next tools
            reason = self._generate_reason(tool, completed)
            proposals.append(Proposal(
                tool=tool,
                args=self._generate_args(tool),
                reason=reason,
            ))

        # Add a dubious proposal: re-running the last completed tool
        if completed:
            last = completed[-1]
            proposals.append(Proposal(
                tool=last,
                args=self._generate_args(last),
                reason=f"Re-run {last} with different parameters for comparison",
            ))

        # Gate each proposal through LogicVerifier
        self._gate_proposals(proposals, context)

        return proposals

    def _gate_proposals(
        self, proposals: list[Proposal], context: str
    ) -> None:
        """Apply plausibility gating to each proposal in place."""
        # Fail-closed: if no trained model, reject everything
        if not self._has_trained_model():
            for p in proposals:
                p.approved = False
                p.score = None
                p.rejection_reason = "fail-closed: no trained GMS model"
            return

        for p in proposals:
            result = self.verifier.verify_action(
                context, p.tool, "should_call"
            )
            p.score = result.get("score")
            if result["plausible"]:
                p.approved = True
            else:
                p.approved = False
                p.rejection_reason = (
                    f"implausible (score={result['score']:.4f}, "
                    f"reason={result['reason']})"
                )

    def _generate_reason(self, tool: str, completed: list[str]) -> str:
        """Generate a deterministic reason for proposing a tool."""
        reasons = {
            "load_dataset": "Need to load the dataset before any analysis",
            "preprocess_data": "Data must be preprocessed before training",
            "train_model": "Ready to train the model on preprocessed data",
            "evaluate_model": "Model trained; need to evaluate on test set",
            "diagnose_accuracy": (
                "Evaluation complete; check for overfitting"
            ),
            "explain_model": (
                "Generate feature importance for interpretability"
            ),
        }
        return reasons.get(tool, f"Next logical step after {completed[-1] if completed else 'start'}")

    def _generate_args(self, tool: str) -> dict:
        """Generate deterministic arguments for a tool proposal."""
        args_map = {
            "load_dataset": {"name": "iris"},
            "preprocess_data": {"test_size": 0.2, "scaler": "standard"},
            "train_model": {"model": "RandomForest", "n_estimators": 100},
            "evaluate_model": {"metrics": ["accuracy", "f1"]},
            "diagnose_accuracy": {"check_overfitting": True},
            "explain_model": {"method": "feature_importance"},
        }
        return args_map.get(tool, {})

    # ── Channel 3: Claims (contradiction-gated) ──

    def extract_claims(self, narrative: str) -> list[GatedClaim]:
        """Extract factual claims from a narrative and gate each.

        Each claim is gated through contradiction detection via
        ``LogicVerifier.verify_domain_triples()``.  Claims that
        contradict existing knowledge are rejected.

        Parameters
        ----------
        narrative : str
            The LLM-generated narrative to extract claims from.

        Returns
        -------
        list[GatedClaim]
            Claims with ``approved`` and ``tension`` fields populated.
        """
        # Determine which tool produced this narrative by keyword matching
        tool_name = self._detect_tool_from_narrative(narrative)
        templates = _CLAIM_TEMPLATES.get(tool_name, [])

        if not templates:
            return []

        # Build GatedClaim objects
        claims = []
        for entry in templates:
            text, head, relation, tail = entry[:4]
            numeric = entry[4] if len(entry) > 4 else None
            triple = ExtractedTriple(
                head=head, relation=relation, tail=tail,
                numeric_value=numeric,
            )
            claims.append(GatedClaim(text=text, triple=triple))

        # Gate through contradiction detection
        self._gate_claims(claims)

        return claims

    def _gate_claims(self, claims: list[GatedClaim]) -> None:
        """Apply contradiction gating to each claim in place."""
        # Fail-closed: if no trained model, reject everything
        if not self._has_trained_model():
            for c in claims:
                c.approved = False
                c.tension = None
                c.rejection_reason = "fail-closed: no trained GMS model"
            return

        # Build ExtractedTriple list for batch verification
        triples = [c.triple for c in claims]
        result = self.verifier.verify_domain_triples(triples)

        # Map results back to GatedClaim objects
        accepted_set = set()
        for t in result["accepted"]:
            accepted_set.add((t.head, t.relation, t.tail))

        rejection_map = {}
        for rej in result["rejected"]:
            key = rej["triple"]
            rejection_map[key] = rej

        for c in claims:
            triple_key = (c.triple.head, c.triple.relation, c.triple.tail)
            if triple_key in accepted_set:
                c.approved = True
                # Check tension for informational purposes
                tension_result = self.verifier.check_tension(
                    c.triple.head, c.triple.tail
                )
                c.tension = tension_result.get("tension")
            else:
                c.approved = False
                rej_info = rejection_map.get(triple_key, {})
                c.tension = rej_info.get("tension")
                c.rejection_reason = rej_info.get(
                    "reason", "rejected by contradiction gate"
                )
                if rej_info.get("score") is not None:
                    c.rejection_reason = (
                        f"{rej_info['reason']} "
                        f"(score={rej_info['score']:.4f})"
                    )

    def _detect_tool_from_narrative(self, narrative: str) -> str:
        """Detect which tool produced a narrative via keyword matching."""
        text = narrative.lower()
        if "loaded" in text or "dataset" in text and "rows" in text:
            return "load_dataset"
        if "preprocessing" in text or "train-test" in text:
            return "preprocess_data"
        if "trained" in text and "model" in text:
            return "train_model"
        if "accuracy" in text and "f1" in text:
            return "evaluate_model"
        if "diagnosis" in text or "overfitting" in text:
            return "diagnose_accuracy"
        if "feature importance" in text or "most important" in text:
            return "explain_model"
        return ""


# ------------------------------------------------------------------ #
# GovernedLLM — LLM-call helper extracted from agents/ds_agent/
# (private tree never shipped in the wheel or mirrored to gmsh).
# Extracted here following the same pattern as LogicVerifier/GMS-215.
# ------------------------------------------------------------------ #

_GOVERNED_SYSTEM = """\
You are the {phase} agent in an interpretable ML pipeline, governed by \
a Geometric Memory System (GMS) that enforces policies.

## Your Role
You interpret structured results and propose next actions. You do NOT \
perform verification — the GMS handles all structured checks (numeric \
comparison, threshold evaluation, set membership). You provide:
1. **Narrative interpretation** of what the results mean
2. **Adaptive proposals** for additional investigation when assessments trigger

## Policy Constraints (from behavior specification)
{policy_context}

## Current Domain Knowledge
{domain_context}

## Response Format
Respond in this exact format:

NARRATIVE:
<1-3 sentences interpreting the results>

PROPOSALS:
<one per line: TOOL: <tool_name> | REASON: <why> | PARAMS: <key=value,...>>
<or "none" if no additional investigation needed>

CLAIMS:
<one per line: TRIPLE: <head> | <relation> | <tail> [| <numeric_value>]>
<factual claims from your interpretation — will be verified by GMS>
"""

_GOVERNED_USER = """\
## Step Context
Phase: {phase}, Step: {step_num}, Tool: {tool_name}

## Tool Result
{result_summary}

## Assessments Triggered
{assessments}

## Question
Based on the results and triggered assessments, provide:
1. A narrative interpretation of what these results mean
2. Any proposals for additional investigation (only tools from allowed list: {allowed_tools})
3. Any factual claims you can extract from the results
"""


def _try_float(val: object) -> float | None:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class GovernedLLM:
    """Policy-constrained LLM call helper.

    Extracted from ``agents.ds_agent.agents.base`` so the shipped
    ``knowlytix-harness`` wheel and the public gmsh mirror can use it
    without depending on the private ``agents/`` tree.
    """

    def __init__(
        self,
        llm: "LLMBackend",
        verifier: LogicVerifier | None = None,
        allowed_tools: list[str] | None = None,
    ):
        self.llm = llm
        self.verifier = verifier
        self.allowed_tools = allowed_tools or []

    def interpret_and_propose(
        self,
        phase_name: str,
        step_num: int,
        tool_name: str,
        result_text: str,
        assessments: dict[str, bool],
        policy_context: str,
        domain_context: str,
    ) -> dict:
        system = _GOVERNED_SYSTEM.format(
            phase=phase_name,
            policy_context=policy_context or "(no policy loaded)",
            domain_context=domain_context or "(no prior knowledge)",
        )
        triggered = [f"  - {k}: {v}" for k, v in assessments.items() if v]
        not_triggered = [f"  - {k}: {v}" for k, v in assessments.items() if not v]
        assess_text = ""
        if triggered:
            assess_text += "Triggered:\n" + "\n".join(triggered)
        if not_triggered:
            assess_text += "\nNot triggered:\n" + "\n".join(not_triggered)
        if not assess_text:
            assess_text = "(no assessments evaluated)"
        user = _GOVERNED_USER.format(
            phase=phase_name,
            step_num=step_num,
            tool_name=tool_name,
            result_summary=result_text[:2000],
            assessments=assess_text,
            allowed_tools=", ".join(self.allowed_tools),
        )
        try:
            raw = self.llm.call(system, user, max_tokens=1024)
        except Exception as e:
            return {
                "narrative": "", "proposals": [], "claims": [],
                "raw_response": "", "error": f"{type(e).__name__}: {e}",
            }
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> dict:
        result: dict = {
            "narrative": "", "proposals": [], "claims": [],
            "raw_response": raw,
        }
        narr = re.search(
            r"NARRATIVE:\s*\n(.*?)(?=\nPROPOSALS:|\nCLAIMS:|\Z)",
            raw, re.DOTALL,
        )
        if narr:
            result["narrative"] = narr.group(1).strip()
        prop = re.search(r"PROPOSALS:\s*\n(.*?)(?=\nCLAIMS:|\Z)", raw, re.DOTALL)
        if prop:
            text = prop.group(1).strip()
            if text.lower() != "none":
                for line in text.split("\n"):
                    p = self._parse_proposal(line)
                    if p:
                        result["proposals"].append(p)
        claim = re.search(r"CLAIMS:\s*\n(.*?)$", raw, re.DOTALL)
        if claim:
            for line in claim.group(1).strip().split("\n"):
                t = self._parse_claim(line)
                if t:
                    result["claims"].append(t)
        return result

    def _parse_proposal(self, line: str) -> dict | None:
        m = re.match(
            r"TOOL:\s*(\S+)\s*\|\s*REASON:\s*(.*?)\s*(?:\|\s*PARAMS:\s*(.*))?$",
            line.strip(),
        )
        if not m:
            return None
        tool = m.group(1)
        params: dict[str, str] = {}
        if m.group(3):
            for kv in m.group(3).split(","):
                kv = kv.strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[k.strip()] = v.strip()
        return {
            "tool": tool,
            "reason": m.group(2).strip(),
            "params": params,
            "allowed_tool": tool in self.allowed_tools,
        }

    def _parse_claim(self, line: str) -> ExtractedTriple | None:
        m = re.match(r"TRIPLE:\s*(.+)", line.strip())
        if not m:
            return None
        parts = [p.strip() for p in m.group(1).split("|")]
        if len(parts) < 3:
            return None
        return ExtractedTriple(
            head=parts[0], relation=parts[1], tail=parts[2],
            numeric_value=_try_float(parts[3]) if len(parts) >= 4 else None,
        )

    def verify_proposals(
        self, proposals: list[dict], context_entity: str,
    ) -> list[dict]:
        if not self.verifier or not proposals:
            return proposals
        verified = []
        for p in proposals:
            check = self.verifier.verify_action(
                context_entity, p["tool"], "should_call",
            )
            p["verification"] = check
            if check["plausible"]:
                verified.append(p)
        return verified

    def verify_claims(self, claims: list[ExtractedTriple]) -> dict:
        if not self.verifier or not claims:
            return {
                "accepted": claims, "rejected": [],
                "n_accepted": len(claims), "n_rejected": 0,
            }
        return self.verifier.verify_domain_triples(claims)


# ------------------------------------------------------------------ #
# ConstrainedLLM — full implementation with real LLM (requires API)
# ------------------------------------------------------------------ #

class ConstrainedLLM:
    """Policy-constrained LLM integration with real LLM backend.

    Same interface as :class:`MockConstrainedLLM` but uses actual LLM
    calls via :class:`~knowlytix.knowledge.llm_backend.LLMBackend`.

    .. note::

       This class requires an API key.  For the tutorial, use
       :class:`MockConstrainedLLM` instead.

    Parameters
    ----------
    store : GMSExpertStore
        The GMS store providing geometric verification.
    verifier : LogicVerifier
        The logic verifier for plausibility gating.
    llm : LLMBackend
        The LLM backend for generating narratives and proposals.
    allowed_tools : list[str]
        Tools the LLM is permitted to propose.
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        verifier: LogicVerifier,
        llm: "LLMBackend",
        allowed_tools: list[str] | None = None,
        co_attestor: "LLMCoAttestor | None" = None,
    ):
        self.store = store
        self.verifier = verifier
        self.llm = llm
        self.allowed_tools = allowed_tools or list(_WORKFLOW_STEPS)
        self.co_attestor = co_attestor
        self._governed = GovernedLLM(
            llm=llm,
            verifier=verifier,
            allowed_tools=self.allowed_tools,
        )

    def _has_trained_model(self) -> bool:
        return self.store.model is not None

    def interpret(
        self,
        tool_name: str,
        raw_result: dict,
        context: str = "",
    ) -> str:
        """Channel 1: LLM interprets tool results (ungated)."""
        result_text = "\n".join(
            f"  {k}: {v}" for k, v in raw_result.items()
        )
        resp = self._governed.interpret_and_propose(
            phase_name=context or "analysis",
            step_num=0,
            tool_name=tool_name,
            result_text=result_text,
            assessments={},
            policy_context="",
            domain_context="",
        )
        return resp.get("narrative", "")

    def propose_next(
        self,
        context: str,
        completed: list[str],
    ) -> list[Proposal]:
        """Channel 2: LLM proposes next tools (plausibility-gated).

        Block paths exercised in order:

        1. Fail-closed when the GMS store has no trained model ---
           short-circuits before any LLM call.
        2. Backend error (rate limit, timeout, malformed JSON,
           any exception in :meth:`LLMBackend.call`) --- the LLM
           response is empty and we emit a single rejection-marked
           Proposal so callers can see the failure in the audit log
           rather than silently receiving zero proposals.
        3. Forbidden tool --- the LLM proposed a tool not in
           ``allowed_tools``.  Recorded as a rejected Proposal so
           the audit trail captures the LLM's propensity to suggest
           forbidden actions.
        4. Co-attestor block --- a registered
           :class:`LLMCoAttestor` rule rejected the proposal (e.g.
           policy or numeric range).  Applied before the geometric
           check so a misconfigured manifold cannot let a
           policy-violating proposal through.
        5. Geometric (LogicVerifier) block --- the proposal's
           geodesic distance exceeds the calibrated plausibility
           threshold.
        """
        if not self._has_trained_model():
            return [
                Proposal(
                    tool="none", args={},
                    reason="fail-closed: no trained GMS model",
                    approved=False, score=None,
                    rejection_reason="fail-closed: no trained GMS model",
                )
            ]

        result_text = (
            f"Completed tools: {', '.join(completed) or 'none'}\n"
            f"Context: {context}"
        )
        resp = self._governed.interpret_and_propose(
            phase_name=context,
            step_num=len(completed),
            tool_name=completed[-1] if completed else "start",
            result_text=result_text,
            assessments={},
            policy_context="",
            domain_context="",
        )

        # Backend-failure path: GovernedLLM.interpret_and_propose
        # captures every exception from llm.call into resp["error"].
        # We surface it as a fail-closed rejection so the audit log
        # records the failure rather than receiving a silent empty
        # proposal list.
        if resp.get("error"):
            return [
                Proposal(
                    tool="none", args={},
                    reason=f"backend_error: {resp['error']}",
                    approved=False, score=None,
                    rejection_reason=(
                        f"backend_error: {resp['error']}"
                    ),
                )
            ]

        proposals = []
        for p in resp.get("proposals", []):
            proposal = Proposal(
                tool=p["tool"],
                args=p.get("params", {}),
                reason=p.get("reason", ""),
            )

            # 3. Forbidden tool: not in allowed_tools.
            if not p.get("allowed_tool", True):
                proposal.approved = False
                proposal.score = None
                proposal.rejection_reason = (
                    f"forbidden_tool: '{proposal.tool}' "
                    f"not in allowed_tools"
                )
                proposals.append(proposal)
                continue

            # 4. Co-attestor (rule-based) check.
            if self.co_attestor is not None:
                co = self.co_attestor.check_proposal(
                    context, proposal.tool, proposal.args
                )
                if not co["allowed"]:
                    proposal.approved = False
                    proposal.score = None
                    proposal.rejection_reason = (
                        f"co_attestor: {co['reason']}"
                    )
                    proposals.append(proposal)
                    continue

            # 5. Geometric (LogicVerifier) check.
            check = self.verifier.verify_action(
                context, proposal.tool, "should_call"
            )
            proposal.score = check.get("score")
            proposal.approved = check["plausible"]
            if not proposal.approved:
                proposal.rejection_reason = (
                    f"implausible (score={check['score']}, "
                    f"reason={check['reason']})"
                )
            proposals.append(proposal)

        return proposals

    def extract_claims(self, narrative: str) -> list[GatedClaim]:
        """Channel 3: LLM extracts claims (contradiction-gated).

        Block paths exercised in order:

        1. Fail-closed when no trained model --- short-circuits
           before any LLM call.
        2. Backend error --- the call to the LLM failed and the
           response is empty.  We return ``[]`` so callers do not
           treat this as ``no claims`` (which would be wrong); the
           empty list paired with the absence of a narrative
           signals the failure.
        3. Co-attestor (rule-based) block --- numeric range
           violation, forbidden relation, etc.
        4. Geometric (LogicVerifier) block --- contradiction with
           an existing in-manifold triple.
        """
        if not self._has_trained_model():
            return []

        resp = self._governed.interpret_and_propose(
            phase_name="claim_extraction",
            step_num=0,
            tool_name="narrative",
            result_text=narrative,
            assessments={},
            policy_context="",
            domain_context="",
        )

        if resp.get("error"):
            return []

        claims = []
        for triple in resp.get("claims", []):
            gc = GatedClaim(
                text=f"{triple.head} {triple.relation} {triple.tail}",
                triple=triple,
            )
            claims.append(gc)

        if not claims:
            return claims

        # 3. Co-attestor pass: blocks on rule violations before
        # any geometric check.  The geometric gate gets only the
        # claims the rule pack accepted.
        co_blocked: dict[tuple, str] = {}
        if self.co_attestor is not None:
            for c in claims:
                co = self.co_attestor.check_claim(c.triple)
                if not co["allowed"]:
                    key = (c.triple.head, c.triple.relation, c.triple.tail)
                    co_blocked[key] = co["reason"]

        # 4. Geometric pass on remaining claims.
        remaining = [
            c for c in claims
            if (c.triple.head, c.triple.relation, c.triple.tail)
            not in co_blocked
        ]
        accepted_set: set = set()
        rejection_map: dict = {}
        if remaining:
            triples = [c.triple for c in remaining]
            result = self.verifier.verify_domain_triples(triples)
            accepted_set = {
                (t.head, t.relation, t.tail) for t in result["accepted"]
            }
            rejection_map = {
                rej["triple"]: rej for rej in result["rejected"]
            }

        # Decide each claim's verdict.
        for c in claims:
            key = (c.triple.head, c.triple.relation, c.triple.tail)
            if key in co_blocked:
                c.approved = False
                c.tension = None
                c.rejection_reason = (
                    f"co_attestor: {co_blocked[key]}"
                )
            elif key in accepted_set:
                c.approved = True
                te = self.verifier.check_tension(
                    c.triple.head, c.triple.tail
                )
                c.tension = te.get("tension")
            else:
                c.approved = False
                rej = rejection_map.get(key, {})
                c.tension = rej.get("tension")
                c.rejection_reason = rej.get(
                    "reason", "rejected by contradiction gate"
                )

        return claims


# ------------------------------------------------------------------ #
# Helper: format governance report
# ------------------------------------------------------------------ #

def format_proposal_report(proposals: list[Proposal]) -> str:
    """Format a list of proposals for display."""
    lines = [
        f"{'Tool':<22s} {'Approved':<10s} {'Score':<10s} {'Reason'}",
        "-" * 70,
    ]
    for p in proposals:
        score_str = f"{p.score:.4f}" if p.score is not None else "N/A"
        status = "YES" if p.approved else "NO"
        lines.append(
            f"{p.tool:<22s} {status:<10s} {score_str:<10s} {p.reason}"
        )
        if p.rejection_reason:
            lines.append(f"  -> REJECTED: {p.rejection_reason}")
    return "\n".join(lines)


def format_claim_report(claims: list[GatedClaim]) -> str:
    """Format a list of gated claims for display."""
    lines = [
        f"{'Claim':<45s} {'Approved':<10s} {'Tension':<10s}",
        "-" * 70,
    ]
    for c in claims:
        t_str = f"{c.tension:.4f}" if c.tension is not None else "N/A"
        status = "YES" if c.approved else "NO"
        lines.append(f"{c.text:<45s} {status:<10s} {t_str:<10s}")
        if c.rejection_reason:
            lines.append(f"  -> REJECTED: {c.rejection_reason}")
    return "\n".join(lines)
