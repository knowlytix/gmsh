# SPDX-License-Identifier: Apache-2.0
"""GovernedToolGateway --- triple-gate execution layer for policy-governed agents.

Extends the GMSH :class:`~knowlytix.harness.testing.gateway.ToolGateway` and
:class:`~knowlytix.harness.testing.policy.PolicyEngine` with a **hard plausibility gate**
that blocks tool calls whose GMS geodesic distance exceeds a configurable
threshold.  The existing gateway logs plausibility but never blocks on it;
this module makes the gate enforceable.

The three gates executed in order:

1. **Schema gate** -- validates tool arguments against the declared JSON
   schema (type checking, required parameters).
2. **Policy gate** -- delegates to ``PolicyEngine.check()`` which performs
   four-layer verification: explicit rules, GMS graph lookup, geometric
   plausibility and contradiction detection.
3. **Plausibility gate** -- scores ``(context, "should_call", tool_name)``
   on the GMS manifold; if the geodesic exceeds ``theta_plausibility``
   the call is blocked as implausible.

Fault injection profiles can be attached to inject errors, latency or
mock responses --- delegating to the underlying ``FaultProfile`` dataclass.

Built in Chapter 7 of "How to Build Policy-Governed AI Agents."

Usage::

    from knowlytix.harness.governance.governed_gateway import GovernedToolGateway, GatewayResult
    from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule
    from knowlytix.harness.testing.gateway import FaultProfile

    gateway = GovernedToolGateway(
        session=state,
        store=my_store,
        policy=policy_engine,
        theta_plausibility=1.5,
    )
    result = gateway.execute("train_model", {"dataset": "iris", ...}, context="training")
    if result.blocked:
        print(result.reason)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowlytix.harness.testing.gateway import (
    FaultProfile,
    ToolGateway,
)
from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule, PolicyDecision

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.governance.session import SessionState


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GatewayResult:
    """Outcome of a governed tool execution attempt.

    Parameters
    ----------
    blocked : bool
        True if the tool call was blocked by any gate.
    reason : str
        Human-readable explanation of the outcome.
    display : str
        Display string produced by the tool (empty if blocked).
    raw : dict
        Raw result dict from the tool (empty if blocked).
    gate_results : dict
        Per-gate pass/fail record, e.g.
        ``{"schema": True, "policy": True, "plausibility": False}``.
    policy_decision : PolicyDecision or None
        Full ``PolicyDecision`` from the policy gate, if executed.
    plausibility_score : float or None
        GMS geodesic distance from the plausibility gate.
    fault_injected : bool
        True if a fault profile triggered an injected error.
    execution_time_ms : float
        Wall-clock time for the full execute() call (milliseconds).
    """

    blocked: bool = False
    reason: str = ""
    display: str = ""
    raw: dict = field(default_factory=dict)
    gate_results: dict = field(default_factory=dict)
    policy_decision: PolicyDecision | None = None
    plausibility_score: float | None = None
    fault_injected: bool = False
    execution_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Transcript entry for governed execution
# ---------------------------------------------------------------------------

@dataclass
class GovernedTranscriptEntry:
    """A single governed execution record with per-gate detail.

    Parameters
    ----------
    tool_name : str
        Name of the tool that was called.
    args : dict
        Arguments passed to the tool.
    context : str
        Execution context (e.g. ``"model_training"``).
    blocked : bool
        Whether the call was blocked.
    reason : str
        Explanation of the outcome.
    gate_results : dict
        Per-gate pass/fail record.
    plausibility_score : float or None
        GMS geodesic distance from the plausibility gate.
    policy_decision_summary : str
        One-line summary of the policy decision.
    fault_injected : bool
        True if a fault profile fired.
    execution_time_ms : float
        Wall-clock time in milliseconds.
    timestamp : str
        ISO-ish timestamp; auto-filled if empty.
    """

    tool_name: str = ""
    args: dict = field(default_factory=dict)
    context: str = ""
    blocked: bool = False
    reason: str = ""
    gate_results: dict = field(default_factory=dict)
    plausibility_score: float | None = None
    policy_decision_summary: str = ""
    fault_injected: bool = False
    execution_time_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Schema validation helper
# ---------------------------------------------------------------------------

def _validate_schema(tool_name: str, args: dict, schemas: list[dict]) -> tuple[bool, str]:
    """Validate *args* against the declared JSON schema for *tool_name*.

    Checks that required parameters are present and that provided
    values have a JSON-compatible type (string, number, integer,
    boolean, array, object).  This is intentionally lightweight ---
    full JSON Schema validation is a separate concern.

    Parameters
    ----------
    tool_name : str
        Name of the tool to validate against.
    args : dict
        Arguments supplied by the caller.
    schemas : list[dict]
        The ``TOOL_DEFINITIONS`` list containing all tool schemas.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` if valid, ``(False, reason)`` if invalid.
    """
    schema = None
    for s in schemas:
        if s["name"] == tool_name:
            schema = s
            break

    if schema is None:
        return False, f"Unknown tool: {tool_name}"

    input_schema = schema.get("input_schema", {})
    required = input_schema.get("required", [])
    properties = input_schema.get("properties", {})

    # Check required parameters
    for param in required:
        if param not in args:
            return False, f"Missing required parameter: {param}"

    # Type checking
    _type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for param, value in args.items():
        if param in properties:
            expected_type = properties[param].get("type")
            if expected_type and expected_type in _type_map:
                py_type = _type_map[expected_type]
                if not isinstance(value, py_type):
                    # Allow int for number
                    if expected_type == "number" and isinstance(value, (int, float)):
                        continue
                    return (
                        False,
                        f"Parameter '{param}' expected type "
                        f"'{expected_type}', got {type(value).__name__}",
                    )

    return True, ""


# ---------------------------------------------------------------------------
# GovernedToolGateway
# ---------------------------------------------------------------------------

class GovernedToolGateway:
    """Triple-gate tool execution layer with hard plausibility enforcement.

    Wraps the existing :class:`~knowlytix.harness.testing.gateway.ToolGateway` (for
    fault injection) and :class:`~knowlytix.harness.testing.policy.PolicyEngine` (for
    four-layer policy checking), adding a mandatory plausibility gate
    that blocks tool calls whose GMS geodesic distance exceeds
    ``theta_plausibility``.

    Parameters
    ----------
    session : SessionState
        Mutable session state shared across tool calls.
    store : GMSExpertStore
        GMS store used for plausibility scoring and policy graph queries.
    policy : PolicyEngine or None
        Pre-configured policy engine.  If None, a fresh engine is
        created from *store* with no rules.
    theta_plausibility : float
        Maximum geodesic distance for the plausibility gate.  Tool calls
        scoring above this threshold are blocked.  Default 1.5.
    tool_schemas : list[dict] or None
        Tool definitions for schema validation.  If None, imports
        ``TOOL_DEFINITIONS`` from ``governed_agent.tools``.
    device : str
        PyTorch device string (``"cuda"`` or ``"cpu"``).

    Examples
    --------
    >>> gateway = GovernedToolGateway(session, store, policy, theta_plausibility=1.5)
    >>> result = gateway.execute("train_model", {"dataset": "iris", ...}, "training")
    >>> result.blocked
    False
    """

    def __init__(
        self,
        session: "SessionState",
        store: "GMSExpertStore",
        policy: PolicyEngine | None = None,
        theta_plausibility: float = 1.5,
        tool_schemas: list[dict] | None = None,
        device: str = "cuda",
        plausibility_relation: str = "should_call",
    ):
        self.session = session
        self.store = store
        self.theta_plausibility = theta_plausibility
        self.plausibility_relation = plausibility_relation
        self.device = device

        # Policy engine
        if policy is not None:
            self.policy = policy
        else:
            self.policy = PolicyEngine(store=store)

        # Underlying ToolGateway for fault injection
        self._gw = ToolGateway(store=store)

        # Tool schemas for gate 1
        if tool_schemas is not None:
            self._schemas = tool_schemas
        else:
            from knowlytix.harness.governance.tools import TOOL_DEFINITIONS
            self._schemas = TOOL_DEFINITIONS

        # Transcript log
        self._transcript: list[GovernedTranscriptEntry] = []

    # -- Public execution method -------------------------------------------

    def execute(
        self,
        tool_name: str,
        args: dict,
        context: str = "default",
    ) -> GatewayResult:
        """Execute a tool through the triple-gate pipeline.

        Gate order:

        1. **Schema validation** -- checks required params and types.
        2. **Policy check** -- delegates to ``PolicyEngine.check()``.
        3. **Plausibility check** -- scores ``(context, "should_call",
           tool_name)`` via ``store.score_triple()``.

        After all gates pass the tool is executed via
        ``governed_agent.tools.execute_tool``.  If a ``FaultProfile``
        matches the tool, fault injection (error rate, latency) is
        applied before execution.

        Parameters
        ----------
        tool_name : str
            Name of the tool to call.
        args : dict
            Arguments to pass to the tool handler.
        context : str
            Execution context string used for policy and plausibility
            checks.  Defaults to ``"default"``.

        Returns
        -------
        GatewayResult
            Structured result with per-gate outcomes, display string
            and raw result.
        """
        t0 = time.time()
        gate_results: dict[str, bool] = {}

        # ==============================================================
        # Gate 1: Schema validation
        # ==============================================================
        valid, schema_reason = _validate_schema(
            tool_name, args, self._schemas,
        )
        gate_results["schema"] = valid
        if not valid:
            result = GatewayResult(
                blocked=True,
                reason=f"Schema validation failed: {schema_reason}",
                gate_results=gate_results,
                execution_time_ms=(time.time() - t0) * 1000,
            )
            self._record(tool_name, args, context, result)
            return result

        # ==============================================================
        # Gate 2: Policy engine check (HARD block)
        # ==============================================================
        policy_decision = self.policy.check(
            action=tool_name, args=args, context=context,
        )
        gate_results["policy"] = policy_decision.allowed
        if not policy_decision.allowed:
            result = GatewayResult(
                blocked=True,
                reason=f"Policy violation: {policy_decision.reason}",
                gate_results=gate_results,
                policy_decision=policy_decision,
                execution_time_ms=(time.time() - t0) * 1000,
            )
            self._record(tool_name, args, context, result)
            return result

        # ==============================================================
        # Gate 3: GMS plausibility (HARD block if > theta)
        # ==============================================================
        plausibility_score = self._score_plausibility(context, tool_name)
        if plausibility_score is not None:
            plausible = plausibility_score <= self.theta_plausibility
        else:
            plausible = True  # fail-open when scoring is unavailable
        gate_results["plausibility"] = plausible

        if not plausible:
            result = GatewayResult(
                blocked=True,
                reason=(
                    f"Implausible tool call: distance({context}, "
                    f"{self.plausibility_relation}, {tool_name}) = "
                    f"{plausibility_score:.4f} > "
                    f"theta={self.theta_plausibility}"
                ),
                gate_results=gate_results,
                policy_decision=policy_decision,
                plausibility_score=plausibility_score,
                execution_time_ms=(time.time() - t0) * 1000,
            )
            self._record(tool_name, args, context, result)
            return result

        # ==============================================================
        # Fault injection check (from ToolGateway profiles)
        # ==============================================================
        fault_injected = False
        profile = self._gw._match_fault_profile(tool_name)
        if profile is not None:
            # Latency injection
            if profile.latency_ms > 0:
                time.sleep(profile.latency_ms / 1000.0)

            # Error injection
            if profile.error_rate > 0 and random.random() < profile.error_rate:
                fault_injected = True
                gate_results["fault_injection"] = False
                result = GatewayResult(
                    blocked=True,
                    reason=f"Fault injected: {profile.error_message}",
                    gate_results=gate_results,
                    policy_decision=policy_decision,
                    plausibility_score=plausibility_score,
                    fault_injected=True,
                    execution_time_ms=(time.time() - t0) * 1000,
                )
                self._record(tool_name, args, context, result)
                return result

        gate_results["fault_injection"] = True  # no fault fired

        # ==============================================================
        # Execute the tool
        # ==============================================================
        from knowlytix.harness.governance.tools import execute_tool

        display, raw = execute_tool(tool_name, args, self.session)

        result = GatewayResult(
            blocked=False,
            reason="All gates passed",
            display=display,
            raw=raw if isinstance(raw, dict) else {"result": raw},
            gate_results=gate_results,
            policy_decision=policy_decision,
            plausibility_score=plausibility_score,
            fault_injected=False,
            execution_time_ms=(time.time() - t0) * 1000,
        )
        self._record(tool_name, args, context, result)
        return result

    # -- Rule and profile management ---------------------------------------

    def add_policy_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule to the underlying PolicyEngine.

        Parameters
        ----------
        rule : PolicyRule
            The rule to encode and register.
        """
        self.policy.encode_rules([rule])

    def add_fault_profile(self, profile: FaultProfile) -> None:
        """Attach a fault injection profile to the underlying ToolGateway.

        Parameters
        ----------
        profile : FaultProfile
            Fault profile specifying the tool pattern, error rate,
            latency and other injection parameters.
        """
        self._gw.fault_profiles.append(profile)

    # -- Transcript access -------------------------------------------------

    @property
    def transcript(self) -> list[GovernedTranscriptEntry]:
        """Return the full governed execution transcript.

        Returns
        -------
        list[GovernedTranscriptEntry]
            Chronological list of all execution attempts and outcomes.
        """
        return self._transcript

    def transcript_summary(self) -> dict:
        """Return an aggregate summary of the transcript.

        Returns
        -------
        dict
            Summary with counts of total calls, blocked calls,
            per-gate block counts and fault injections.
        """
        total = len(self._transcript)
        blocked = sum(1 for e in self._transcript if e.blocked)
        faults = sum(1 for e in self._transcript if e.fault_injected)

        gate_blocks: dict[str, int] = {}
        for entry in self._transcript:
            for gate, passed in entry.gate_results.items():
                if not passed:
                    gate_blocks[gate] = gate_blocks.get(gate, 0) + 1

        return {
            "total_calls": total,
            "blocked": blocked,
            "passed": total - blocked,
            "fault_injections": faults,
            "block_rate": blocked / total if total > 0 else 0.0,
            "gate_blocks": gate_blocks,
        }

    # -- Internal helpers --------------------------------------------------

    def _score_plausibility(
        self, context: str, tool_name: str,
    ) -> float | None:
        """Score the plausibility of calling *tool_name* in *context*.

        Queries ``store.score_triple(context, "should_call", tool_name)``.
        Returns None if scoring is unavailable.

        Parameters
        ----------
        context : str
            The execution context.
        tool_name : str
            The tool being evaluated.

        Returns
        -------
        float or None
            Geodesic distance, or None on error.
        """
        try:
            score = self.store.score_triple(
                context, self.plausibility_relation, tool_name,
            )
            return float(score) if score is not None else None
        except Exception:
            return None

    def _record(
        self,
        tool_name: str,
        args: dict,
        context: str,
        result: GatewayResult,
    ) -> None:
        """Append a transcript entry for the given execution result.

        Parameters
        ----------
        tool_name : str
            The tool that was called.
        args : dict
            Arguments passed.
        context : str
            Execution context.
        result : GatewayResult
            The gateway result.
        """
        policy_summary = ""
        if result.policy_decision is not None:
            pd_ = result.policy_decision
            if pd_.allowed:
                policy_summary = "allowed"
            else:
                policy_summary = (
                    f"blocked by {pd_.blocking_rules}"
                )

        entry = GovernedTranscriptEntry(
            tool_name=tool_name,
            args=args,
            context=context,
            blocked=result.blocked,
            reason=result.reason,
            gate_results=result.gate_results,
            plausibility_score=result.plausibility_score,
            policy_decision_summary=policy_summary,
            fault_injected=result.fault_injected,
            execution_time_ms=result.execution_time_ms,
        )
        self._transcript.append(entry)
