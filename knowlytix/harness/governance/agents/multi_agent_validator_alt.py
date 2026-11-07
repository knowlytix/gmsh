# SPDX-License-Identifier: Apache-2.0
"""Independent boolean re-implementation of the multi-agent permission check.

Used as a second-opinion attestor: if the production
:class:`GovernedAgent.can_execute` and this alt-validator disagree on
any (role, tool) pair, the bundle build refuses to ship.

The alt-validator does NOT call into ``governed_agent.agents.multi_agent``;
it re-derives the verdict from raw role data via direct set membership
checks. A bug in the production class would have to also exist in this
module to escape detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.harness.governance.agents.multi_agent import MultiAgentOrchestrator


def alt_can_execute(
    *,
    tool_name: str,
    allowed_tools: frozenset[str] | set[str],
    forbidden_tools: frozenset[str] | set[str],
) -> bool:
    """Independent boolean re-implementation of can_execute.

    Whitelist semantics with explicit forbidden list. Empty / None
    tool_name fails closed.
    """
    if not isinstance(tool_name, str) or not tool_name:
        return False
    if tool_name in forbidden_tools:
        return False
    return tool_name in allowed_tools


@dataclass
class MultiAgentAltAgreementReport:
    n: int
    n_agree: int
    agreement_rate: float
    disagreements: list[dict] = field(default_factory=list)


def assert_orchestrator_alt_agreement(
    orch: MultiAgentOrchestrator,
    cohort: list[tuple[str, str]] | None = None,
    *,
    raise_on_disagreement: bool = True,
) -> MultiAgentAltAgreementReport:
    """Compare every (role, tool) decision between orchestrator and alt.

    If ``cohort`` is None, enumerate every (role, tool) over the
    Adult-Income vocabulary.
    """
    if cohort is None:
        from knowlytix.harness.governance.agents.multi_agent_invariants import (
            _ADULT_TOOL_VOCAB,
        )
        cohort = [
            (rn, tool)
            for rn in orch.agents
            for tool in _ADULT_TOOL_VOCAB
        ]

    n = 0
    n_agree = 0
    disagreements: list[dict] = []

    for role_name, tool in cohort:
        agent = orch.agents.get(role_name)
        if agent is None:
            continue
        n += 1
        prod = agent.can_execute(tool)
        alt = alt_can_execute(
            tool_name=tool,
            allowed_tools=agent.role.allowed_tools,
            forbidden_tools=agent.role.forbidden_tools,
        )
        if prod == alt:
            n_agree += 1
        else:
            disagreements.append({
                "role": role_name,
                "tool": tool,
                "production": bool(prod),
                "alt": bool(alt),
            })

    rate = (n_agree / n) if n > 0 else 0.0
    report = MultiAgentAltAgreementReport(
        n=n, n_agree=n_agree, agreement_rate=rate,
        disagreements=disagreements,
    )
    if raise_on_disagreement and disagreements:
        first = disagreements[0]
        raise RuntimeError(
            f"alt-validator disagreement on {len(disagreements)}/{n} "
            f"pairs; first: role={first['role']} tool={first['tool']} "
            f"prod={first['production']} alt={first['alt']}"
        )
    return report
