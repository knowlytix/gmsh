# SPDX-License-Identifier: Apache-2.0
"""GovernanceTracer --- audit-grade telemetry for governed agent pipelines.

Wraps the existing :class:`~knowlytix.harness.testing.tracing.DOETracingGMS` and adds
governance-specific helpers: per-step AuditPackets, verification tracing,
state-transition recording, and queryable governance reports.

Built in Chapter 14 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.testing.tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# AuditPacket --- per-tool-call provenance record
# ---------------------------------------------------------------------------

@dataclass
class AuditPacket:
    """Complete provenance record for a single governed tool call.

    Parameters
    ----------
    step_num : int
        Sequential step number within the run.
    tool_name : str
        Name of the tool that was called.
    policy_decision : str
        ``"allowed"`` or ``"blocked"``.
    blocking_rules : list[str]
        Names of rules that blocked the call (empty if allowed).
    gms_scores : dict[str, float]
        GMS scores relevant to this decision (plausibility, etc.).
    verification : dict or None
        Verification results if claims were checked after this step.
    state_before : str
        Pipeline state before this tool call.
    state_after : str
        Pipeline state after this tool call.
    timestamp : float
        Unix timestamp of the event.
    execution_time_ms : float
        Wall-clock execution time in milliseconds.
    gate_results : dict
        Per-gate pass/fail record from the gateway.
    args : dict
        Arguments passed to the tool.
    result_summary : str
        Brief summary of the tool result.
    """

    step_num: int = 0
    tool_name: str = ""
    policy_decision: str = "allowed"
    blocking_rules: list = field(default_factory=list)
    gms_scores: dict = field(default_factory=dict)
    verification: dict | None = None
    state_before: str = ""
    state_after: str = ""
    timestamp: float = 0.0
    execution_time_ms: float = 0.0
    gate_results: dict = field(default_factory=dict)
    args: dict = field(default_factory=dict)
    result_summary: str = ""

    # Hash-chain fields populated by ``seal_audit_trail``.  Default
    # to the empty string so unsealed packets remain
    # JSON-serializable and ``verify_audit_chain`` can detect when a
    # caller forgets to seal.
    prev_hash: str = ""
    content_hash: str = ""


# ---------------------------------------------------------------------------
# GovernanceTracer
# ---------------------------------------------------------------------------

class GovernanceTracer:
    """Audit-grade governance tracer wrapping DOETracingGMS.

    Provides a high-level API for recording tool calls, verification
    results and state transitions.  Every event is stored both in an
    in-memory audit trail (for fast querying) and, optionally, in a
    DOETracingGMS store (for geometric persistence).

    Parameters
    ----------
    tracing_store : DOETracingGMS or None
        Optional tracing store for geometric persistence.  If None,
        events are recorded only in memory.
    """

    def __init__(self, tracing_store: "DOETracingGMS | None" = None):
        self._store = tracing_store
        self._packets: dict[str, list[AuditPacket]] = {}
        self._verifications: dict[str, list[dict]] = {}
        self._transitions: dict[str, list[dict]] = {}
        self._run_meta: dict[str, dict] = {}

    # -- Run lifecycle ----------------------------------------------------

    def start_run(self, run_id: str, description: str = "") -> None:
        """Mark the start of a governed run.

        Parameters
        ----------
        run_id : str
            Unique identifier for this run.
        description : str
            Optional description.
        """
        self._packets[run_id] = []
        self._verifications[run_id] = []
        self._transitions[run_id] = []
        self._run_meta[run_id] = {
            "start_time": time.time(),
            "description": description,
        }

    def end_run(self, run_id: str) -> None:
        """Mark the end of a governed run.

        Parameters
        ----------
        run_id : str
            Run identifier.
        """
        if run_id in self._run_meta:
            self._run_meta[run_id]["end_time"] = time.time()
            self._run_meta[run_id]["elapsed_s"] = (
                self._run_meta[run_id]["end_time"]
                - self._run_meta[run_id]["start_time"]
            )

    # -- Tool call tracing ------------------------------------------------

    def trace_tool_call(
        self,
        run_id: str,
        step_num: int,
        tool_name: str,
        args: dict,
        result_summary: str,
        policy_decision: str,
        blocking_rules: list[str] | None = None,
        gms_scores: dict[str, float] | None = None,
        gate_results: dict | None = None,
        execution_time_ms: float = 0.0,
        state_before: str = "",
        state_after: str = "",
    ) -> AuditPacket:
        """Record a tool call and return an AuditPacket.

        Parameters
        ----------
        run_id : str
            Run identifier.
        step_num : int
            Sequential step number.
        tool_name : str
            Tool that was called.
        args : dict
            Arguments passed.
        result_summary : str
            Brief description of the result.
        policy_decision : str
            ``"allowed"`` or ``"blocked"``.
        blocking_rules : list[str] or None
            Rules that blocked the call.
        gms_scores : dict[str, float] or None
            GMS scores for this decision.
        gate_results : dict or None
            Per-gate pass/fail record.
        execution_time_ms : float
            Execution time in milliseconds.
        state_before : str
            Pipeline state before the call.
        state_after : str
            Pipeline state after the call.

        Returns
        -------
        AuditPacket
            The recorded audit packet.
        """
        packet = AuditPacket(
            step_num=step_num,
            tool_name=tool_name,
            policy_decision=policy_decision,
            blocking_rules=blocking_rules or [],
            gms_scores=gms_scores or {},
            state_before=state_before,
            state_after=state_after,
            timestamp=time.time(),
            execution_time_ms=execution_time_ms,
            gate_results=gate_results or {},
            args=args,
            result_summary=result_summary,
        )

        if run_id not in self._packets:
            self._packets[run_id] = []
        self._packets[run_id].append(packet)

        # Persist to DOETracingGMS if available
        if self._store is not None:
            action = "allowed" if policy_decision == "allowed" else "blocked"
            plausibility = (gms_scores or {}).get("plausibility")
            self._store.trace_gateway_decision(
                run_id=run_id,
                tool_name=tool_name,
                action=action,
                reason=result_summary[:200],
                gms_score=plausibility,
            )
            if policy_decision == "blocked" and blocking_rules:
                self._store.trace_policy_decision(
                    run_id=run_id,
                    action=tool_name,
                    allowed=False,
                    blocking_rules=blocking_rules,
                    gms_scores=gms_scores or {},
                )

        return packet

    # -- Verification tracing ---------------------------------------------

    def trace_verification(
        self,
        run_id: str,
        step_num: int,
        claims: list[dict],
        verdicts: list[dict],
    ) -> None:
        """Record verification results for a step.

        Parameters
        ----------
        run_id : str
            Run identifier.
        step_num : int
            Step number being verified.
        claims : list[dict]
            Claims that were verified.
        verdicts : list[dict]
            Verdict results.
        """
        record = {
            "step_num": step_num,
            "claims": claims,
            "verdicts": verdicts,
            "timestamp": time.time(),
        }
        if run_id not in self._verifications:
            self._verifications[run_id] = []
        self._verifications[run_id].append(record)

        # Update the most recent packet for this step
        if run_id in self._packets:
            for pkt in reversed(self._packets[run_id]):
                if pkt.step_num == step_num:
                    n_passed = sum(1 for v in verdicts if v.get("passed", False))
                    pkt.verification = {
                        "n_claims": len(claims),
                        "n_passed": n_passed,
                        "pass_rate": n_passed / len(claims) if claims else 1.0,
                        "verdicts": verdicts,
                    }
                    break

    # -- State transition tracing -----------------------------------------

    def trace_state_transition(
        self,
        run_id: str,
        from_state: str,
        to_state: str,
    ) -> None:
        """Record a state machine transition.

        Parameters
        ----------
        run_id : str
            Run identifier.
        from_state : str
            State before the transition.
        to_state : str
            State after the transition.
        """
        record = {
            "from_state": from_state,
            "to_state": to_state,
            "timestamp": time.time(),
        }
        if run_id not in self._transitions:
            self._transitions[run_id] = []
        self._transitions[run_id].append(record)

        # Persist to DOETracingGMS if available
        if self._store is not None:
            self._store.trace_state_transition(run_id, from_state, to_state)

    # -- Query API --------------------------------------------------------

    def get_audit_trail(self, run_id: str) -> list[AuditPacket]:
        """Return the full audit trail for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        list[AuditPacket]
            Chronological list of audit packets.
        """
        return self._packets.get(run_id, [])

    def query_blocked_calls(self, run_id: str) -> list[AuditPacket]:
        """Return only the blocked tool calls for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        list[AuditPacket]
            Audit packets where policy_decision == "blocked".
        """
        return [
            p for p in self._packets.get(run_id, [])
            if p.policy_decision == "blocked"
        ]

    def query_allowed_calls(self, run_id: str) -> list[AuditPacket]:
        """Return only the allowed tool calls for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        list[AuditPacket]
            Audit packets where policy_decision == "allowed".
        """
        return [
            p for p in self._packets.get(run_id, [])
            if p.policy_decision == "allowed"
        ]

    def get_verifications(self, run_id: str) -> list[dict]:
        """Return all verification records for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        list[dict]
            Verification records.
        """
        return self._verifications.get(run_id, [])

    def get_transitions(self, run_id: str) -> list[dict]:
        """Return all state transitions for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        list[dict]
            State transition records.
        """
        return self._transitions.get(run_id, [])

    # -- Summary report ---------------------------------------------------

    def summary_report(self, run_id: str) -> str:
        """Generate a human-readable summary report for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        str
            Formatted summary report.
        """
        packets = self._packets.get(run_id, [])
        verifs = self._verifications.get(run_id, [])
        transitions = self._transitions.get(run_id, [])
        meta = self._run_meta.get(run_id, {})

        total = len(packets)
        blocked = sum(1 for p in packets if p.policy_decision == "blocked")
        allowed = total - blocked

        # Execution time
        total_exec_ms = sum(p.execution_time_ms for p in packets)

        # Verification stats
        total_claims = 0
        total_passed = 0
        all_scores = []
        for v in verifs:
            for verdict in v.get("verdicts", []):
                total_claims += 1
                if verdict.get("passed", False):
                    total_passed += 1
                score = verdict.get("score")
                if score is not None:
                    all_scores.append(score)

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Find lowest-confidence step
        lowest_step = None
        lowest_score = float("inf")
        for pkt in packets:
            if pkt.verification is not None:
                rate = pkt.verification.get("pass_rate", 1.0)
                if rate < lowest_score:
                    lowest_score = rate
                    lowest_step = pkt.step_num

        # Elapsed time
        elapsed = meta.get("elapsed_s", 0.0)

        lines = [
            f"{'=' * 60}",
            f"  GOVERNANCE AUDIT REPORT  --  Run: {run_id}",
            f"{'=' * 60}",
            "",
            f"  Total tool calls:       {total}",
            f"  Allowed:                {allowed}",
            f"  Blocked:                {blocked}",
            f"  Block rate:             {blocked/total:.1%}" if total > 0 else "  Block rate:             N/A",
            "",
            f"  Total claims verified:  {total_claims}",
            f"  Claims passed:          {total_passed}",
            f"  Avg verification score: {avg_score:.3f}",
            "",
            f"  State transitions:      {len(transitions)}",
            f"  Total execution time:   {total_exec_ms:.1f} ms",
            f"  Run elapsed time:       {elapsed:.2f} s",
            "",
        ]

        if lowest_step is not None:
            lines.append(
                f"  Lowest-confidence step: {lowest_step} "
                f"(pass_rate={lowest_score:.3f})"
            )

        # Policy violations
        violations = [p for p in packets if p.policy_decision == "blocked"]
        if violations:
            lines.append("")
            lines.append("  --- Policy Violations ---")
            for v in violations:
                rules = ", ".join(v.blocking_rules) if v.blocking_rules else "plausibility"
                lines.append(
                    f"  Step {v.step_num}: {v.tool_name} "
                    f"blocked by [{rules}]"
                )

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    # -- Governance dashboard table ---------------------------------------

    def dashboard_table(self, run_id: str) -> list[dict]:
        """Return a list of dicts suitable for tabular display.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        list[dict]
            One dict per step with dashboard columns.
        """
        rows = []
        for pkt in self._packets.get(run_id, []):
            plaus = pkt.gms_scores.get("plausibility", None)
            verif_rate = None
            if pkt.verification is not None:
                verif_rate = pkt.verification.get("pass_rate")
            rows.append({
                "Step": pkt.step_num,
                "Tool": pkt.tool_name,
                "Decision": pkt.policy_decision,
                "Blocking Rules": ", ".join(pkt.blocking_rules) if pkt.blocking_rules else "",
                "Plausibility": f"{plaus:.4f}" if plaus is not None else "N/A",
                "Gate: Schema": pkt.gate_results.get("schema", ""),
                "Gate: Policy": pkt.gate_results.get("policy", ""),
                "Gate: Plausibility": pkt.gate_results.get("plausibility", ""),
                "Verification": f"{verif_rate:.1%}" if verif_rate is not None else "",
                "Time (ms)": f"{pkt.execution_time_ms:.1f}",
                "State": f"{pkt.state_before} -> {pkt.state_after}",
            })
        return rows
