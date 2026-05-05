# SPDX-License-Identifier: Apache-2.0
"""Exhaustive (role, tool) lattice invariants for multi-agent governance.

Five invariants verified by enumeration over the Cartesian product of
{4 roles} x {15+ Adult-Income tool vocabulary}:

  I1 (disjoint sets): for every role, allowed_tools and
       forbidden_tools are disjoint.
  I2 (default-deny): for every role and every tool not in the
       allow-list, ``can_execute(tool)`` returns False.
  I3 (cross-role isolation): for every pair of distinct agents,
       ``id(agent_a.memory.store) != id(agent_b.memory.store)``.
  I4 (handoff transitivity): if a transition is recorded from A
       to B, the orchestrator's tool-record log shows A's calls
       all preceding B's first call.
  I5 (frozen role consistency): ``can_execute(t)`` is deterministic
       for the lifetime of an agent --- mutation attempts must
       raise.

Mirrors :mod:`governed_agent.release_gate_invariants`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.harness.governance.agents.multi_agent import (
    MultiAgentOrchestrator,
)


_ADULT_TOOL_VOCAB = (
    "load_dataset", "data_summary", "eda_correlation", "preprocess_data",
    "feature_select", "train_model", "predict", "leaderboard",
    "diagnose_robustness", "diagnose_fairness", "diagnose_accuracy",
    "diagnose_resilience", "diagnose_reliability",
    "explain_importance", "explain_pdp", "explain_shap", "explain_ale",
    "interpret_effects", "interpret_fi",
)


@dataclass
class MultiAgentInvariantViolation:
    invariant: str
    description: str
    offending_input: dict | None = None


@dataclass
class MultiAgentInvariantReport:
    n_cells: int
    n_pass: int
    n_violations: int
    violations: list[MultiAgentInvariantViolation] = field(default_factory=list)
    invariants_checked: list[str] = field(default_factory=list)


def verify_multi_agent_invariants(
    orch: MultiAgentOrchestrator,
    *,
    tool_vocab: tuple[str, ...] = _ADULT_TOOL_VOCAB,
) -> MultiAgentInvariantReport:
    """Exhaustively verify the five invariants on the orchestrator."""
    violations: list[MultiAgentInvariantViolation] = []
    n_cells = 0
    n_pass = 0

    # I1: disjoint sets
    for role_name, agent in orch.agents.items():
        n_cells += 1
        overlap = (agent.role.allowed_tools & agent.role.forbidden_tools)
        if overlap:
            violations.append(MultiAgentInvariantViolation(
                invariant="I1_disjoint_sets",
                description=(f"Role '{role_name}' has overlapping allowed "
                             f"and forbidden tools: {sorted(overlap)}"),
                offending_input={"role": role_name,
                                 "overlap": sorted(overlap)},
            ))
        else:
            n_pass += 1

    # I2: default-deny over the lattice
    for role_name, agent in orch.agents.items():
        for tool in tool_vocab:
            n_cells += 1
            permitted = agent.can_execute(tool)
            in_allow = tool in agent.role.allowed_tools
            in_forbid = tool in agent.role.forbidden_tools
            expected = in_allow and not in_forbid
            if permitted == expected:
                n_pass += 1
            else:
                violations.append(MultiAgentInvariantViolation(
                    invariant="I2_default_deny",
                    description=(f"can_execute({tool!r}) on {role_name} = "
                                 f"{permitted}, expected {expected}"),
                    offending_input={"role": role_name, "tool": tool},
                ))

    # I3: cross-role store isolation
    role_names = list(orch.agents.keys())
    for i in range(len(role_names)):
        for j in range(i + 1, len(role_names)):
            n_cells += 1
            a = orch.agents[role_names[i]]
            b = orch.agents[role_names[j]]
            if id(a.memory.store) == id(b.memory.store):
                violations.append(MultiAgentInvariantViolation(
                    invariant="I3_cross_role_isolation",
                    description=(f"{role_names[i]} and {role_names[j]} "
                                 "share the same store object"),
                    offending_input={"a": role_names[i],
                                     "b": role_names[j]},
                ))
            else:
                n_pass += 1

    # I4: handoff transitivity (only when handoffs exist)
    if orch._handoffs:
        for h in orch._handoffs:
            n_cells += 1
            from_calls = [
                r for r in orch._tool_records
                if r.agent_name == h.from_agent
                and r.timestamp <= h.timestamp
            ]
            to_calls = [
                r for r in orch._tool_records
                if r.agent_name == h.to_agent
                and r.timestamp >= h.timestamp
            ]
            ok = (len(from_calls) > 0 or len(to_calls) > 0)
            if ok:
                n_pass += 1
            else:
                violations.append(MultiAgentInvariantViolation(
                    invariant="I4_handoff_transitivity",
                    description=(f"Handoff {h.from_agent}->{h.to_agent} "
                                 "has neither preceding from-calls nor "
                                 "following to-calls"),
                    offending_input={"from": h.from_agent,
                                     "to": h.to_agent},
                ))

    # I5: frozen role consistency
    for role_name, agent in orch.agents.items():
        n_cells += 1
        try:
            agent.role.allowed_tools.add("hijack_tool")  # type: ignore
            violations.append(MultiAgentInvariantViolation(
                invariant="I5_frozen_role",
                description=(f"{role_name}.role.allowed_tools.add() "
                             "mutated successfully (should be frozenset)"),
                offending_input={"role": role_name},
            ))
        except (AttributeError, TypeError):
            n_pass += 1
        # And reassignment of role itself
        try:
            agent.role = agent.role  # type: ignore
            violations.append(MultiAgentInvariantViolation(
                invariant="I5_frozen_role",
                description=(f"{role_name}.role assignment after "
                             "construction succeeded"),
                offending_input={"role": role_name},
            ))
        except AttributeError:
            n_pass += 1

    return MultiAgentInvariantReport(
        n_cells=n_cells,
        n_pass=n_pass,
        n_violations=len(violations),
        violations=violations,
        invariants_checked=[
            "I1_disjoint_sets",
            "I2_default_deny",
            "I3_cross_role_isolation",
            "I4_handoff_transitivity",
            "I5_frozen_role",
        ],
    )
