# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts: FSM-based structural enforcement for governed agents.

A behavioral contract defines an FSM whose states correspond to workflow
milestones (data loaded, preprocessed, model trained, ...) and whose
transitions are the *only* legal moves.  Tool permissions are scoped to
each state so that a tool may execute *only* when the FSM is in a state
that explicitly allows it.  This is structural enforcement: if a
transition is not in the table it *cannot happen*, period.

Three risk-tier contracts are pre-defined:
  - **ADVISORY**        -- loose, 50 tool calls, most transitions allowed
  - **RECOMMENDATION**  -- standard, 30 tool calls, preprocessing required
  - **ACTION_TAKING**   -- strict, 20 tool calls, mandatory fairness +
                           robustness, human approval before deployment

The ``ProcessEvaluator`` validates that an actual tool sequence conforms
to a contract.  It supports a pure-FSM path (no GMS dependency) and an
optional GMS holonomy path when a trained store is available.

Built in Chapter 13 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Contract state enumeration
# ---------------------------------------------------------------------------

class ContractState(Enum):
    """States in the model-development lifecycle FSM."""

    IDLE = "idle"
    DATA_LOADED = "data_loaded"
    PREPROCESSED = "preprocessed"
    MODEL_TRAINED = "model_trained"
    DIAGNOSED = "diagnosed"
    FAIR_CHECKED = "fair_checked"
    COMPARED = "compared"
    DEPLOYMENT_READY = "deployment_ready"
    HUMAN_REVIEW = "human_review"


# ---------------------------------------------------------------------------
# Tool-to-state mapping (which state does a tool advance the FSM to?)
# ---------------------------------------------------------------------------

TOOL_STATE_ADVANCE: dict[str, ContractState] = {
    "load_dataset":         ContractState.DATA_LOADED,
    "data_summary":         ContractState.DATA_LOADED,
    "eda_correlation":      ContractState.DATA_LOADED,
    "preprocess_data":      ContractState.PREPROCESSED,
    "feature_select":       ContractState.PREPROCESSED,
    "train_model":          ContractState.MODEL_TRAINED,
    "predict":              ContractState.MODEL_TRAINED,
    "leaderboard":          ContractState.COMPARED,
    "diagnose_robustness":  ContractState.DIAGNOSED,
    "diagnose_fairness":    ContractState.FAIR_CHECKED,
    "explain_importance":   ContractState.DIAGNOSED,
    "explain_pdp":          ContractState.DIAGNOSED,
    "explain_shap":         ContractState.DIAGNOSED,
    "recall_memory":        None,  # no state change
    "save_finding":         None,  # no state change
}


# ---------------------------------------------------------------------------
# Sequence validation result
# ---------------------------------------------------------------------------

@dataclass
class SequenceValidation:
    """Result of validating a tool sequence against a contract.

    Attributes
    ----------
    valid : bool
        True if every tool was allowed and every transition legal.
    violations : list[str]
        Human-readable descriptions of each violation detected.
    states_visited : list[ContractState]
        The FSM states traversed during validation.
    tools_checked : int
        Total tools processed.
    tools_blocked : int
        Tools that would have been blocked.
    invariant_violations : list[str]
        Invariant conditions that were broken.
    """

    valid: bool = True
    violations: list[str] = field(default_factory=list)
    states_visited: list[ContractState] = field(default_factory=list)
    tools_checked: int = 0
    tools_blocked: int = 0
    invariant_violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BehavioralContract
# ---------------------------------------------------------------------------

@dataclass
class BehavioralContract:
    """FSM-based behavioral contract with structural enforcement.

    Parameters
    ----------
    name : str
        Human-readable contract name.
    states : list[ContractState]
        All states reachable in this contract.
    transitions : dict[ContractState, set[ContractState]]
        Permitted state transitions.  A transition from ``s`` to ``t``
        is legal only if ``t`` is in ``transitions[s]``.
    tool_permissions : dict[ContractState, set[str]]
        The set of tools callable when the FSM is in a given state.
    invariants : list[Callable]
        Functions ``(state, history) -> str | None``.  If the function
        returns a non-None string, the invariant is violated and the
        string describes the violation.
    max_tool_calls : int
        Hard budget for total tool calls in one session.
    require_human_approval : bool
        If True, the FSM must pass through HUMAN_REVIEW before
        reaching DEPLOYMENT_READY.
    """

    name: str
    states: list[ContractState]
    transitions: dict[ContractState, set[ContractState]]
    tool_permissions: dict[ContractState, set[str]]
    invariants: list[Callable] = field(default_factory=list)
    max_tool_calls: int = 30
    require_human_approval: bool = False

    # -- core enforcement methods -----------------------------------------

    def is_transition_valid(
        self,
        from_state: ContractState,
        to_state: ContractState,
    ) -> bool:
        """Return True if transitioning from *from_state* to *to_state* is legal.

        A transition is valid if and only if *to_state* appears in the
        set ``self.transitions[from_state]``.  Self-loops (staying in
        the same state) are always permitted.

        Parameters
        ----------
        from_state : ContractState
            Current FSM state.
        to_state : ContractState
            Proposed next state.

        Returns
        -------
        bool
            True if the transition is structurally permitted.
        """
        if from_state == to_state:
            return True  # self-loops always permitted
        allowed = self.transitions.get(from_state, set())
        return to_state in allowed

    def allowed_tools(self, current_state: ContractState) -> set[str]:
        """Return the set of tools permitted in *current_state*.

        Parameters
        ----------
        current_state : ContractState
            The current FSM state.

        Returns
        -------
        set[str]
            Tool names that may be called.  Empty set if the state has
            no tool permissions defined.
        """
        return self.tool_permissions.get(current_state, set())

    def check_invariants(
        self,
        state: ContractState,
        history: list[str],
    ) -> list[str]:
        """Check all invariants and return a list of violations.

        Parameters
        ----------
        state : ContractState
            The current FSM state.
        history : list[str]
            The tool call history up to this point.

        Returns
        -------
        list[str]
            A (possibly empty) list of violation descriptions.
        """
        violations = []
        for inv in self.invariants:
            result = inv(state, history)
            if result is not None:
                violations.append(result)
        return violations

    def transition_table(self) -> list[dict[str, str]]:
        """Return the FSM transition table as a list of dicts.

        Each dict has keys ``from``, ``to`` and ``tools`` (comma-joined
        list of tools allowed in the ``from`` state).

        Returns
        -------
        list[dict[str, str]]
            One entry per permitted transition.
        """
        rows = []
        for src in self.states:
            dests = sorted(self.transitions.get(src, set()),
                           key=lambda s: s.value)
            tools_here = sorted(self.tool_permissions.get(src, set()))
            for dst in dests:
                rows.append({
                    "from": src.value,
                    "to": dst.value,
                    "tools": ", ".join(tools_here) if tools_here else "(none)",
                })
        return rows


# ---------------------------------------------------------------------------
# ProcessEvaluator
# ---------------------------------------------------------------------------

class ProcessEvaluator:
    """Validate tool sequences against a behavioral contract.

    Provides two validation paths:
      1. **Pure-FSM** (``validate_sequence``): walks the FSM for each
         tool, checking transitions and permissions structurally.
      2. **GMS holonomy** (``check_holonomy``): when a GMS store is
         available, also scores the geometric consistency of the
         actual vs. expected sequence.

    Parameters
    ----------
    contract : BehavioralContract
        The behavioral contract to enforce.
    """

    def __init__(self, contract: BehavioralContract) -> None:
        self.contract = contract

    # -- pure-FSM validation ----------------------------------------------

    def validate_sequence(
        self,
        sequence: list[str],
        contract: BehavioralContract | None = None,
    ) -> SequenceValidation:
        """Validate *sequence* against *contract* using pure FSM logic.

        Walks through *sequence* tool by tool, enforcing:
          - The tool is permitted in the current state.
          - The state transition triggered by the tool is legal.
          - The total call count does not exceed the budget.
          - All invariants hold after each step.

        Parameters
        ----------
        sequence : list[str]
            Ordered list of tool names to validate.
        contract : BehavioralContract or None
            Contract to check against.  Defaults to ``self.contract``.

        Returns
        -------
        SequenceValidation
            Detailed validation result.
        """
        if contract is None:
            contract = self.contract

        result = SequenceValidation()
        current_state = ContractState.IDLE
        result.states_visited.append(current_state)
        history: list[str] = []

        for idx, tool in enumerate(sequence):
            result.tools_checked += 1

            # Budget check
            if idx >= contract.max_tool_calls:
                result.valid = False
                result.violations.append(
                    f"Step {idx}: budget exceeded "
                    f"({idx + 1} > {contract.max_tool_calls})"
                )
                result.tools_blocked += 1
                continue

            # Permission check
            permitted = contract.allowed_tools(current_state)
            if tool not in permitted:
                result.valid = False
                result.violations.append(
                    f"Step {idx}: tool '{tool}' not permitted in "
                    f"state '{current_state.value}' "
                    f"(allowed: {sorted(permitted)})"
                )
                result.tools_blocked += 1
                continue

            # Compute the next state
            next_state = TOOL_STATE_ADVANCE.get(tool)
            if next_state is None:
                # Tool does not advance state (e.g. recall_memory)
                next_state = current_state

            # Transition check
            if not contract.is_transition_valid(current_state, next_state):
                result.valid = False
                result.violations.append(
                    f"Step {idx}: transition "
                    f"'{current_state.value}' -> '{next_state.value}' "
                    f"(triggered by '{tool}') is not permitted"
                )
                result.tools_blocked += 1
                continue

            # Advance state
            if next_state != current_state:
                current_state = next_state
                result.states_visited.append(current_state)

            history.append(tool)

            # Invariant check
            inv_violations = contract.check_invariants(current_state, history)
            if inv_violations:
                result.valid = False
                result.invariant_violations.extend(inv_violations)
                for v in inv_violations:
                    result.violations.append(
                        f"Step {idx}: invariant violated -- {v}"
                    )

        # Human-approval check
        if contract.require_human_approval:
            if (ContractState.DEPLOYMENT_READY in result.states_visited
                    and ContractState.HUMAN_REVIEW
                    not in result.states_visited):
                result.valid = False
                result.violations.append(
                    "Contract requires HUMAN_REVIEW before "
                    "DEPLOYMENT_READY, but HUMAN_REVIEW was never reached"
                )

        return result

    # -- GMS holonomy validation ------------------------------------------

    def check_holonomy(
        self,
        actual: list[str],
        expected: list[str],
        store: "GMSExpertStore | None" = None,
    ) -> float:
        """Score the geometric consistency of *actual* vs *expected* sequence.

        When a GMS store is available and the ``precedes`` relation is
        trained, uses ``store.check_holonomy`` to compose rotors along
        both sequences and measure their defect.

        When GMS is unavailable, falls back to a normalized edit-distance
        metric scaled to [0, 2] (0 = identical, 2 = maximally different).

        Parameters
        ----------
        actual : list[str]
            The actual tool sequence that was executed.
        expected : list[str]
            The expected (contract-compliant) tool sequence.
        store : GMSExpertStore or None
            Optional GMS store with trained relation embeddings.

        Returns
        -------
        float
            Holonomy-like defect in [0, 2].  Lower is better.
        """
        # Attempt GMS holonomy if store is available
        if store is not None:
            try:
                if store.relation_exists("precedes"):
                    actual_rels = ["precedes"] * max(len(actual) - 1, 1)
                    expected_rels = ["precedes"] * max(len(expected) - 1, 1)
                    defect_actual = store.check_holonomy(
                        actual_rels, "precedes"
                    )
                    defect_expected = store.check_holonomy(
                        expected_rels, "precedes"
                    )
                    if defect_actual is not None and defect_expected is not None:
                        return abs(defect_actual - defect_expected)
            except Exception:
                pass

        # Fallback: normalized Levenshtein edit distance
        return self._edit_distance_metric(actual, expected)

    @staticmethod
    def _edit_distance_metric(a: list[str], b: list[str]) -> float:
        """Normalized edit distance scaled to [0, 2].

        Uses the standard dynamic-programming Levenshtein algorithm.
        The result is ``2 * edit_distance / max(len(a), len(b))``.
        """
        n, m = len(a), len(b)
        if n == 0 and m == 0:
            return 0.0
        if n == 0 or m == 0:
            return 2.0

        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp

        max_len = max(n, m)
        return 2.0 * dp[m] / max_len


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------

def _invariant_no_deploy_without_fairness(
    state: ContractState,
    history: list[str],
) -> str | None:
    """Deployment readiness requires a prior fairness check."""
    if state == ContractState.DEPLOYMENT_READY:
        if "diagnose_fairness" not in history:
            return (
                "DEPLOYMENT_READY reached without diagnose_fairness "
                "in the tool history"
            )
    return None


def _invariant_no_deploy_without_robustness(
    state: ContractState,
    history: list[str],
) -> str | None:
    """Deployment readiness requires a prior robustness check."""
    if state == ContractState.DEPLOYMENT_READY:
        if "diagnose_robustness" not in history:
            return (
                "DEPLOYMENT_READY reached without diagnose_robustness "
                "in the tool history"
            )
    return None


def _invariant_no_train_without_preprocess(
    state: ContractState,
    history: list[str],
) -> str | None:
    """Model training requires prior preprocessing."""
    if state == ContractState.MODEL_TRAINED:
        if "preprocess_data" not in history:
            return (
                "MODEL_TRAINED reached without preprocess_data "
                "in the tool history"
            )
    return None


# ---------------------------------------------------------------------------
# Pre-defined risk-tier contracts
# ---------------------------------------------------------------------------

# Shared tool sets
_DATA_TOOLS = {
    "load_dataset", "data_summary", "eda_correlation",
}
_PREPROCESS_TOOLS = {
    "preprocess_data", "feature_select",
}
_TRAINING_TOOLS = {
    "train_model", "predict",
}
_DIAGNOSTIC_TOOLS = {
    "diagnose_robustness", "diagnose_fairness",
    "explain_importance", "explain_pdp", "explain_shap",
}
_COMPARISON_TOOLS = {
    "leaderboard",
}
_MEMORY_TOOLS = {
    "recall_memory", "save_finding",
}

ALL_TOOLS = (
    _DATA_TOOLS | _PREPROCESS_TOOLS | _TRAINING_TOOLS
    | _DIAGNOSTIC_TOOLS | _COMPARISON_TOOLS | _MEMORY_TOOLS
)


def advisory_contract() -> BehavioralContract:
    """Create the Advisory (loose) behavioral contract.

    Most transitions are allowed.  Tool permissions are broad.
    Budget: 50 tool calls.  No human approval required.
    """
    S = ContractState
    return BehavioralContract(
        name="advisory",
        states=list(S),
        transitions={
            S.IDLE:             {S.DATA_LOADED},
            S.DATA_LOADED:      {S.PREPROCESSED, S.MODEL_TRAINED,
                                 S.DATA_LOADED},
            S.PREPROCESSED:     {S.MODEL_TRAINED, S.PREPROCESSED,
                                 S.DATA_LOADED},
            S.MODEL_TRAINED:    {S.DIAGNOSED, S.FAIR_CHECKED,
                                 S.COMPARED, S.MODEL_TRAINED,
                                 S.PREPROCESSED, S.DEPLOYMENT_READY},
            S.DIAGNOSED:        {S.FAIR_CHECKED, S.COMPARED,
                                 S.DIAGNOSED, S.MODEL_TRAINED,
                                 S.DEPLOYMENT_READY},
            S.FAIR_CHECKED:     {S.COMPARED, S.DIAGNOSED,
                                 S.FAIR_CHECKED, S.MODEL_TRAINED,
                                 S.DEPLOYMENT_READY},
            S.COMPARED:         {S.DEPLOYMENT_READY, S.COMPARED,
                                 S.MODEL_TRAINED, S.DIAGNOSED,
                                 S.FAIR_CHECKED},
            S.DEPLOYMENT_READY: {S.HUMAN_REVIEW},
            S.HUMAN_REVIEW:     set(),
        },
        tool_permissions={
            S.IDLE:             _DATA_TOOLS | _MEMORY_TOOLS,
            S.DATA_LOADED:      (ALL_TOOLS - {"leaderboard"}),
            S.PREPROCESSED:     (ALL_TOOLS - {"leaderboard"}),
            S.MODEL_TRAINED:    ALL_TOOLS,
            S.DIAGNOSED:        ALL_TOOLS,
            S.FAIR_CHECKED:     ALL_TOOLS,
            S.COMPARED:         ALL_TOOLS,
            S.DEPLOYMENT_READY: _MEMORY_TOOLS,
            S.HUMAN_REVIEW:     _MEMORY_TOOLS,
        },
        invariants=[],
        max_tool_calls=50,
        require_human_approval=False,
    )


def recommendation_contract() -> BehavioralContract:
    """Create the Recommendation (standard) behavioral contract.

    Requires preprocessing before training.  Diagnostics must come
    after training.  Budget: 30 tool calls.
    """
    S = ContractState
    return BehavioralContract(
        name="recommendation",
        states=list(S),
        transitions={
            S.IDLE:             {S.DATA_LOADED},
            S.DATA_LOADED:      {S.PREPROCESSED, S.DATA_LOADED},
            S.PREPROCESSED:     {S.MODEL_TRAINED, S.PREPROCESSED},
            S.MODEL_TRAINED:    {S.DIAGNOSED, S.FAIR_CHECKED,
                                 S.COMPARED, S.MODEL_TRAINED},
            S.DIAGNOSED:        {S.FAIR_CHECKED, S.COMPARED,
                                 S.DIAGNOSED, S.DEPLOYMENT_READY},
            S.FAIR_CHECKED:     {S.COMPARED, S.FAIR_CHECKED,
                                 S.DIAGNOSED, S.DEPLOYMENT_READY},
            S.COMPARED:         {S.DEPLOYMENT_READY, S.COMPARED},
            S.DEPLOYMENT_READY: {S.HUMAN_REVIEW},
            S.HUMAN_REVIEW:     set(),
        },
        tool_permissions={
            S.IDLE:             _DATA_TOOLS | _MEMORY_TOOLS,
            S.DATA_LOADED:      _DATA_TOOLS | _PREPROCESS_TOOLS | _MEMORY_TOOLS,
            S.PREPROCESSED:     (_DATA_TOOLS | _PREPROCESS_TOOLS
                                 | _TRAINING_TOOLS | _MEMORY_TOOLS),
            S.MODEL_TRAINED:    ALL_TOOLS,
            S.DIAGNOSED:        (_DIAGNOSTIC_TOOLS | _COMPARISON_TOOLS
                                 | _MEMORY_TOOLS),
            S.FAIR_CHECKED:     (_DIAGNOSTIC_TOOLS | _COMPARISON_TOOLS
                                 | _MEMORY_TOOLS),
            S.COMPARED:         _COMPARISON_TOOLS | _MEMORY_TOOLS,
            S.DEPLOYMENT_READY: _MEMORY_TOOLS,
            S.HUMAN_REVIEW:     _MEMORY_TOOLS,
        },
        invariants=[_invariant_no_train_without_preprocess],
        max_tool_calls=30,
        require_human_approval=False,
    )


def action_taking_contract() -> BehavioralContract:
    """Create the Action-Taking (strict) behavioral contract.

    Mandatory fairness and robustness checks before deployment.
    Human approval required before DEPLOYMENT_READY.
    Budget: 20 tool calls.
    """
    S = ContractState
    return BehavioralContract(
        name="action_taking",
        states=list(S),
        transitions={
            S.IDLE:             {S.DATA_LOADED},
            S.DATA_LOADED:      {S.PREPROCESSED},
            S.PREPROCESSED:     {S.MODEL_TRAINED},
            S.MODEL_TRAINED:    {S.DIAGNOSED},
            S.DIAGNOSED:        {S.FAIR_CHECKED},
            S.FAIR_CHECKED:     {S.COMPARED},
            S.COMPARED:         {S.HUMAN_REVIEW},
            S.HUMAN_REVIEW:     {S.DEPLOYMENT_READY},
            S.DEPLOYMENT_READY: set(),
        },
        tool_permissions={
            S.IDLE:             _DATA_TOOLS | _MEMORY_TOOLS,
            S.DATA_LOADED:      _DATA_TOOLS | _PREPROCESS_TOOLS | _MEMORY_TOOLS,
            S.PREPROCESSED:     _TRAINING_TOOLS | _MEMORY_TOOLS,
            S.MODEL_TRAINED:    {"diagnose_robustness", "explain_importance",
                                 "explain_pdp", "explain_shap"} | _MEMORY_TOOLS,
            S.DIAGNOSED:        {"diagnose_fairness"} | _MEMORY_TOOLS,
            S.FAIR_CHECKED:     _COMPARISON_TOOLS | _MEMORY_TOOLS,
            S.COMPARED:         _MEMORY_TOOLS,
            S.HUMAN_REVIEW:     _MEMORY_TOOLS,
            S.DEPLOYMENT_READY: _MEMORY_TOOLS,
        },
        invariants=[
            _invariant_no_train_without_preprocess,
            _invariant_no_deploy_without_fairness,
            _invariant_no_deploy_without_robustness,
        ],
        max_tool_calls=20,
        require_human_approval=True,
    )


# ---------------------------------------------------------------------------
# Text-based guardrail (for comparison in Lab 13.2)
# ---------------------------------------------------------------------------

class TextGuardrail:
    """A naive text-based guardrail that checks tool ordering via keywords.

    This intentionally mimics the kind of loose, probabilistic check
    that prompt-based guardrails perform.  It uses substring matching
    on a set of textual rules and misses violations that require
    tracking FSM state.
    """

    def __init__(self) -> None:
        self.rules = [
            "train_model must come after preprocess_data",
            "diagnose_fairness must come after train_model",
            "leaderboard must come after train_model",
        ]

    def check_sequence(self, sequence: list[str]) -> tuple[bool, list[str]]:
        """Check *sequence* against textual ordering rules.

        Returns
        -------
        tuple[bool, list[str]]
            ``(passed, violations)`` where *passed* is True only if no
            violations were detected.
        """
        violations = []
        seen: set[str] = set()

        for tool in sequence:
            # Rule 1: train_model after preprocess_data
            if tool == "train_model" and "preprocess_data" not in seen:
                violations.append(
                    "train_model called before preprocess_data"
                )
            # Rule 2: diagnose_fairness after train_model
            if tool == "diagnose_fairness" and "train_model" not in seen:
                violations.append(
                    "diagnose_fairness called before train_model"
                )
            # Rule 3: leaderboard after train_model
            if tool == "leaderboard" and "train_model" not in seen:
                violations.append(
                    "leaderboard called before train_model"
                )
            seen.add(tool)

        return (len(violations) == 0, violations)
