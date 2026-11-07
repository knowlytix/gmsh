# SPDX-License-Identifier: Apache-2.0
"""Cross-component agreement check for the integrated workflow.

The workflow runs through three governance components in
parallel: the policy engine (Ch.12) decides allow/block per tool
call, the behavioral-contract engine (Ch.13) tracks state
transitions, and the audit tracer (Ch.14) records the outcome.
For the integrated pipeline to be trustworthy, the three views
must agree on every gated event.

This module is the workflow-level analog of the cross-attestor
in `tracing_hardening.cross_attestor`: same shape, same break
classification, but checks across *components* rather than
across *layers within one component*.

A break here is a serious finding: it means one governance layer
silently disagreed with another, and the audit log no longer
faithfully reflects the runtime decisions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentDisagreement:
    """One mismatch between two governance components."""

    step_num: int
    tool_name: str
    component_a: str
    component_b: str
    detail: str


@dataclass
class CrossComponentReport:
    n_steps_checked: int
    n_agreements: int
    disagreements: list[ComponentDisagreement]

    @property
    def all_agree(self) -> bool:
        return not self.disagreements

    @property
    def agreement_rate(self) -> float:
        if self.n_steps_checked == 0:
            return 0.0
        return self.n_agreements / self.n_steps_checked

    def summary(self) -> str:
        rows = [
            "CrossComponentReport(",
            f"  n_steps_checked={self.n_steps_checked}",
            f"  n_agreements={self.n_agreements}",
            f"  agreement_rate={self.agreement_rate:.4f}",
            f"  disagreements={len(self.disagreements)}",
        ]
        for d in self.disagreements[:5]:
            rows.append(
                f"    step {d.step_num} ({d.tool_name}): "
                f"{d.component_a} vs {d.component_b}: {d.detail}"
            )
        rows.append(")")
        return "\n".join(rows)


def cross_component_agreement(
    *,
    policy_decisions: list[dict],
    contract_transitions: list[dict],
    audit_packets: list,
) -> CrossComponentReport:
    """Verify policy + contract + audit all agree per step.

    Each input is a list keyed by step_num:

      * `policy_decisions[i]` -> {"step_num", "tool_name",
                                  "decision" ("allowed"/"blocked"),
                                  "blocking_rules": tuple}
      * `contract_transitions[i]` -> {"step_num", "valid": bool,
                                      "from_state", "to_state"}
      * `audit_packets[i]` is a `tracing.AuditPacket`

    Disagreements are reported per-step. Three checks per step:

      1. policy.decision == audit.policy_decision
      2. policy.blocking_rules == audit.blocking_rules
      3. if contract is invalid, audit.state_after must equal
         contract.from_state (the rejected transition did not
         take place).
    """
    p_by_step = {p["step_num"]: p for p in policy_decisions}
    c_by_step = {c["step_num"]: c for c in contract_transitions}
    a_by_step = {pkt.step_num: pkt for pkt in audit_packets}

    steps = sorted(set(p_by_step) | set(c_by_step) | set(a_by_step))
    disagreements: list[ComponentDisagreement] = []
    agreements = 0

    for s in steps:
        p = p_by_step.get(s)
        c = c_by_step.get(s)
        a = a_by_step.get(s)

        # Audit-vs-policy: decision must match.
        if p is not None and a is not None:
            if p["decision"] != a.policy_decision:
                disagreements.append(ComponentDisagreement(
                    step_num=s,
                    tool_name=p["tool_name"],
                    component_a="policy",
                    component_b="audit",
                    detail=(
                        f"policy={p['decision']} but "
                        f"audit={a.policy_decision}"
                    ),
                ))
                continue

            # Compare blocking rules (sorted, sets-equal)
            p_rules = tuple(sorted(p.get("blocking_rules", ())))
            a_rules = tuple(sorted(a.blocking_rules))
            if p_rules != a_rules:
                disagreements.append(ComponentDisagreement(
                    step_num=s,
                    tool_name=p["tool_name"],
                    component_a="policy",
                    component_b="audit",
                    detail=(
                        f"policy rules={list(p_rules)} but "
                        f"audit rules={list(a_rules)}"
                    ),
                ))
                continue

        # Contract-vs-audit: if contract rejected, audit must show
        # state_before == state_after (no transition).
        if c is not None and a is not None and not c.get("valid", True):
            if a.state_before != a.state_after:
                disagreements.append(ComponentDisagreement(
                    step_num=s,
                    tool_name=getattr(a, "tool_name", "?"),
                    component_a="contract",
                    component_b="audit",
                    detail=(
                        f"contract invalid but audit advanced "
                        f"{a.state_before}->{a.state_after}"
                    ),
                ))
                continue

        agreements += 1

    return CrossComponentReport(
        n_steps_checked=len(steps),
        n_agreements=agreements,
        disagreements=disagreements,
    )
