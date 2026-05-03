# SPDX-License-Identifier: Apache-2.0
"""Cross-reference attestor: does the audit trail accurately reflect runtime decisions?

The hash chain in :mod:`hash_chain` proves that nobody edited the
audit trail \emph{after} it was sealed.  It does not prove that the
trail captured \emph{everything that actually happened} during the
run.  A misconfigured tracer that silently drops every blocked call
produces a perfectly self-consistent chain --- and a worthless audit
log.

This module supplies the second attestor.  Given an independent
record of what the runtime actually decided (the policy engine's
decision log, the orchestrator's contract verdicts, the tool
gateway's gate ledger), it cross-references the audit trail
packet-by-packet:

* Every policy ``blocked`` decision must appear in the trail with
  the same blocking rules.
* Every contract violation must appear in the trail with matching
  ``state_before`` / ``state_after``.
* Every gateway-allowed tool must appear with the corresponding
  gate-results dictionary, and conversely.

Any mismatch is reported as a :class:`CrossAttestationBreak` with
the runtime record and the trail packet that should have matched
(or the absence of one).  In a release pipeline, a non-empty break
list is a hard fail: the audit log is incomplete or wrong, and the
run cannot ship.

Why a separate module rather than baking the check into
:meth:`GovernanceTracer.summary_report`: the cross-reference
attestor is auxiliary --- it is meaningful only when the runtime
also produced an independent decision log.  Bundling it into the
tracer would imply every runtime always has one, which is not
true for the simpler labs in this chapter.  Keeping it separate
mirrors the
``LogicVerifier``+\,``LLMCoAttestor`` split in Chapter~11 and the
``BehavioralContract``+\,``HolonomyCoAttestor`` split in
Chapter~13.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowlytix.harness.governance.tracing import AuditPacket


@dataclass(frozen=True)
class RuntimePolicyEvent:
    """Independent record of one policy-engine decision.

    The runtime produces these alongside calling the tracer; the
    cross-reference attestor checks that every event has a matching
    audit packet.
    """

    step_num: int
    tool_name: str
    decision: str  # "allowed" or "blocked"
    blocking_rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeContractEvent:
    """Independent record of one behavioral-contract transition."""

    step_num: int
    from_state: str
    to_state: str
    valid: bool
    violation: str = ""


@dataclass(frozen=True)
class CrossAttestationBreak:
    """One mismatch between the runtime record and the audit trail.

    ``kind`` is one of:

    * ``"missing_packet"``   --- runtime recorded an event with no
      matching audit packet.
    * ``"extra_packet"``     --- audit packet for a step that the
      runtime never recorded (suggests a forged or stale packet).
    * ``"decision_mismatch"`` --- step is recorded on both sides but
      with differing ``policy_decision``.
    * ``"rules_mismatch"``    --- decision matches but
      ``blocking_rules`` differ.
    * ``"transition_mismatch"`` --- behavioral-contract transition
      and the audit packet's ``state_before``/``state_after``
      disagree.
    """

    step_num: int
    kind: str
    detail: str


def cross_attest_policy(
    runtime_events: list[RuntimePolicyEvent],
    audit_packets: list[AuditPacket],
) -> list[CrossAttestationBreak]:
    """Confirm every runtime policy decision has a matching audit packet.

    The match key is ``(step_num, tool_name)``.  An audit packet
    that the runtime never recorded is reported as ``extra_packet``
    --- a forged or stale entry that the runtime did not ask for.
    """
    breaks: list[CrossAttestationBreak] = []
    pkt_by_key = {(p.step_num, p.tool_name): p for p in audit_packets}
    rt_by_key = {(e.step_num, e.tool_name): e for e in runtime_events}

    for ev in runtime_events:
        key = (ev.step_num, ev.tool_name)
        pkt = pkt_by_key.get(key)
        if pkt is None:
            breaks.append(CrossAttestationBreak(
                step_num=ev.step_num,
                kind="missing_packet",
                detail=(
                    f"runtime recorded {ev.decision} on '{ev.tool_name}' "
                    f"but no audit packet found"
                ),
            ))
            continue

        if pkt.policy_decision != ev.decision:
            breaks.append(CrossAttestationBreak(
                step_num=ev.step_num,
                kind="decision_mismatch",
                detail=(
                    f"runtime={ev.decision} but audit={pkt.policy_decision} "
                    f"for tool '{ev.tool_name}'"
                ),
            ))

        rt_rules = tuple(sorted(ev.blocking_rules))
        pkt_rules = tuple(sorted(pkt.blocking_rules))
        if rt_rules != pkt_rules:
            breaks.append(CrossAttestationBreak(
                step_num=ev.step_num,
                kind="rules_mismatch",
                detail=(
                    f"runtime rules={list(rt_rules)} but "
                    f"audit rules={list(pkt_rules)} for '{ev.tool_name}'"
                ),
            ))

    for pkt in audit_packets:
        key = (pkt.step_num, pkt.tool_name)
        if key not in rt_by_key:
            breaks.append(CrossAttestationBreak(
                step_num=pkt.step_num,
                kind="extra_packet",
                detail=(
                    f"audit packet for '{pkt.tool_name}' (step "
                    f"{pkt.step_num}) has no matching runtime event"
                ),
            ))

    return breaks


def cross_attest_contract(
    runtime_events: list[RuntimeContractEvent],
    audit_packets: list[AuditPacket],
) -> list[CrossAttestationBreak]:
    """Confirm every contract transition is reflected in the audit trail.

    For each runtime contract event the matching audit packet (by
    ``step_num``) must report ``state_before`` and ``state_after``
    consistent with the runtime's transition.
    """
    breaks: list[CrossAttestationBreak] = []
    pkt_by_step = {p.step_num: p for p in audit_packets}

    for ev in runtime_events:
        pkt = pkt_by_step.get(ev.step_num)
        if pkt is None:
            breaks.append(CrossAttestationBreak(
                step_num=ev.step_num,
                kind="missing_packet",
                detail=(
                    f"contract transition "
                    f"{ev.from_state}->{ev.to_state} has no audit packet"
                ),
            ))
            continue

        if (pkt.state_before != ev.from_state
                or pkt.state_after != ev.to_state):
            breaks.append(CrossAttestationBreak(
                step_num=ev.step_num,
                kind="transition_mismatch",
                detail=(
                    f"runtime {ev.from_state}->{ev.to_state} but audit "
                    f"{pkt.state_before}->{pkt.state_after}"
                ),
            ))

    return breaks


def cross_attest_combined(
    *,
    policy_events: list[RuntimePolicyEvent],
    contract_events: list[RuntimeContractEvent],
    audit_packets: list[AuditPacket],
) -> list[CrossAttestationBreak]:
    """Run both attestors and concatenate the breaks for a single report."""
    return (
        cross_attest_policy(policy_events, audit_packets)
        + cross_attest_contract(contract_events, audit_packets)
    )
