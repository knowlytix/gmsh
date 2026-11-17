# SPDX-License-Identifier: Apache-2.0
"""Naive baselines for the multi-agent governance surface.

The audit-defensible bar requires that the role-separated multi-agent
orchestrator demonstrably outperforms naive alternatives.

  * :class:`SingleAgentBaseline` -- one agent does everything; no
    role separation. Should fail every privilege_escalation probe.
  * :class:`NoPermissionCheckBaseline` -- multi-agent but
    ``can_execute`` always returns True. Should also fail privilege
    probes.

:func:`compare_to_orchestrator` reports per-decision uplift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.harness.governance.agents.multi_agent_validation import (
    LabeledMultiAgentDecision,
)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class _Baseline:
    name: str = "baseline"

    def decide(self, probe: LabeledMultiAgentDecision) -> str:
        raise NotImplementedError


class SingleAgentBaseline(_Baseline):
    """One agent for everything: any tool is allowed.

    This is the no-multi-agent control. Every probe gets ``allow``,
    so privilege_escalation, forbidden, and even unknown-tool probes
    all "succeed" --- precisely the failure mode the orchestrator
    is built to prevent.
    """

    name = "single_agent_no_separation"

    def decide(self, probe: LabeledMultiAgentDecision) -> str:
        return "allow"


class NoPermissionCheckBaseline(_Baseline):
    """Multi-agent shell but no permission enforcement.

    Captures the mistake of having roles defined as documentation
    only. Every probe gets ``allow``.
    """

    name = "no_permission_check"

    def decide(self, probe: LabeledMultiAgentDecision) -> str:
        return "allow"


class PromptOnlyBaseline(_Baseline):
    """Permission stated in role description text but not enforced.

    Models the "we told the LLM what it can do" approach.
    Effectively allows everything.
    """

    name = "prompt_only"

    def decide(self, probe: LabeledMultiAgentDecision) -> str:
        return "allow"


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentBaselineContrastReport:
    baseline_name: str
    n_probes: int
    n_correct_orchestrator: int = 0
    n_correct_baseline: int = 0
    uplift: int = 0
    per_decision: dict[str, dict[str, int]] = field(default_factory=dict)


def compare_to_orchestrator(
    orchestrator_decisions: dict[str, str],
    cohort: list[LabeledMultiAgentDecision],
    baseline: _Baseline,
) -> MultiAgentBaselineContrastReport:
    """Pair-uplift contrast between orchestrator and baseline.

    ``orchestrator_decisions`` maps probe_id -> observed decision
    (``allow`` / ``block`` / ``deny``). The caller produces this map
    by running the cohort through the orchestrator.
    """
    per_decision: dict[str, dict[str, int]] = {
        d: {"orch_correct": 0, "base_correct": 0, "n": 0}
        for d in ("allow", "block", "deny")
    }
    n_correct_orch = 0
    n_correct_base = 0
    for probe in cohort:
        observed = orchestrator_decisions.get(probe.probe_id, "unknown")
        baseline_observed = baseline.decide(probe)
        cell = per_decision.setdefault(
            probe.expected_decision, {"orch_correct": 0,
                                      "base_correct": 0, "n": 0})
        cell["n"] += 1
        if observed == probe.expected_decision:
            cell["orch_correct"] += 1
            n_correct_orch += 1
        if baseline_observed == probe.expected_decision:
            cell["base_correct"] += 1
            n_correct_base += 1

    return MultiAgentBaselineContrastReport(
        baseline_name=baseline.name,
        n_probes=len(cohort),
        n_correct_orchestrator=n_correct_orch,
        n_correct_baseline=n_correct_base,
        uplift=n_correct_orch - n_correct_base,
        per_decision=per_decision,
    )
