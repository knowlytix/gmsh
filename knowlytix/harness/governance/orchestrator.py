# SPDX-License-Identifier: Apache-2.0
"""GovernedOrchestrator --- RunManager, Scenarios and behavioral contracts.

Wires together the components built in Chapters 3--8:
  - GovernedToolGateway  (Ch 7): triple-gate execution
  - extract_from_tool    (Ch 4): three-tier knowledge extraction
  - AgentMemory          (Ch 5): geometric memory accumulation
  - ClaimVerifier        (Ch 8): typed claim verification
  - SessionState         (Ch 3): mutable session state

The orchestrator takes a **plan** (list of tool names) as input ---
it does NOT perform its own planning.  Planning is the responsibility
of the WorkflowPlanner (Ch 9).

Key abstractions:
  - RunState / RunManager: finite-state machine with guarded transitions
  - Scenario: behavioral contract (required tools, forbidden tools, budget)
  - GovernedOrchestrator.run(): the main execution loop

Built in Chapter 10 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from knowlytix.harness.governance.extractors import ExtractionResult, extract_from_tool
from knowlytix.harness.governance.verifier import (
    Claim,
    ClaimType,
    ClaimVerifier,
    OverallVerdict,
)

if TYPE_CHECKING:
    from knowlytix.harness.governance.governed_gateway import (
        GatewayResult,
        GovernedToolGateway,
    )
    from knowlytix.harness.governance.memory import AgentMemory
    from knowlytix.harness.governance.orchestrator_policy import OrchestratorPolicy
    from knowlytix.harness.governance.session import SessionState


# ---------------------------------------------------------------------------
# RunState FSM
# ---------------------------------------------------------------------------

class RunState(str, Enum):
    """Lifecycle states for a governed orchestrator run."""

    CREATED = "created"
    PREPARING = "preparing"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


VALID_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED:    {RunState.PREPARING},
    RunState.PREPARING:  {RunState.EXECUTING, RunState.ABORTED},
    RunState.EXECUTING:  {RunState.EVALUATING, RunState.ABORTED},
    RunState.EVALUATING: {RunState.COMPLETED, RunState.FAILED},
}


class InvalidTransitionError(Exception):
    """Raised when a RunManager transition violates the FSM."""
    pass


# ---------------------------------------------------------------------------
# Scenario (behavioral contract)
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Behavioral contract defining what the orchestrator may and may not do.

    Attributes
    ----------
    name : str
        Human-readable scenario identifier.
    required_tools : list[str]
        Tools that MUST appear in the plan.  A plan missing any of
        these triggers a scenario violation.
    forbidden_tools : list[str]
        Tools that MUST NOT appear in the plan.  A plan containing
        any of these triggers a scenario violation and the step is
        skipped.
    expected_tool_sequence : list[str]
        If non-empty, the executed tools must appear in this order
        (not necessarily contiguously).
    max_tool_calls : int
        Maximum number of tool calls before the run is aborted.
    """

    name: str = "default"
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_tool_sequence: list[str] = field(default_factory=list)
    max_tool_calls: int = 30


# ---------------------------------------------------------------------------
# Signal vector
# ---------------------------------------------------------------------------

@dataclass
class SignalVector:
    """Five-component signal vector computed at each execution step.

    Components mapped to GMS primitives:
      sim    -- best score_triple() score in current context
      ver    -- 1.0 if extraction succeeded, 0.0 otherwise
      depth  -- normalized step index (step / total_steps)
      nov    -- 1.0 if new entities were added, 0.0 otherwise
      margin -- execution confidence from gateway plausibility score
    """

    sim: float = 0.0
    ver: float = 0.0
    depth: float = 0.0
    nov: float = 0.0
    margin: float = 0.0
    signal_unavailable: bool = False

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Return the signal vector as a plain tuple."""
        return (self.sim, self.ver, self.depth, self.nov, self.margin)

    def __repr__(self) -> str:
        if self.signal_unavailable:
            return (
                f"sigma(sim=N/A, ver={self.ver:.3f}, "
                f"depth={self.depth:.3f}, nov={self.nov:.3f}, "
                f"margin=N/A, signal_unavailable=True)"
            )
        return (
            f"sigma(sim={self.sim:.3f}, ver={self.ver:.3f}, "
            f"depth={self.depth:.3f}, nov={self.nov:.3f}, "
            f"margin={self.margin:.3f})"
        )


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------

class RunManager:
    """Finite-state machine enforcing valid lifecycle transitions.

    Every transition is logged with a timestamp.  Invalid transitions
    raise ``InvalidTransitionError`` --- no silent state corruption.

    Attributes
    ----------
    state : RunState
        The current lifecycle state.
    transitions : list[tuple[RunState, RunState, float]]
        Log of (from_state, to_state, timestamp) for every transition.
    """

    def __init__(self) -> None:
        self.state = RunState.CREATED
        self.transitions: list[tuple[RunState, RunState, float]] = []

    def transition(self, new_state: RunState) -> None:
        """Attempt a state transition.

        Parameters
        ----------
        new_state : RunState
            The target state.

        Raises
        ------
        InvalidTransitionError
            If the transition from the current state to *new_state*
            is not in ``VALID_TRANSITIONS``.
        """
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self.state.value} "
                f"to {new_state.value}.  "
                f"Allowed: {sorted(s.value for s in allowed)}"
            )
        old = self.state
        self.state = new_state
        self.transitions.append((old, new_state, time.time()))

    def is_terminal(self) -> bool:
        """Return True if the current state is terminal."""
        return self.state in (
            RunState.COMPLETED, RunState.FAILED, RunState.ABORTED,
        )

    def transition_log(self) -> list[str]:
        """Return a human-readable transition log."""
        lines = []
        for old, new, ts in self.transitions:
            lines.append(
                f"  {old.value:12s} -> {new.value:12s}  "
                f"at {time.strftime('%H:%M:%S', time.localtime(ts))}"
            )
        return lines


# ---------------------------------------------------------------------------
# OrchestratorResult
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """Result of a governed orchestrator run.

    Attributes
    ----------
    state : RunState
        Final lifecycle state.
    steps_executed : list[dict]
        Per-step records with tool name, args, display output,
        extraction summary and signal vector.
    verdict : OverallVerdict or None
        Verification verdict from the evaluation phase.
    trace : list[dict]
        Full execution trace including gateway details.
    scenario_violations : list[str]
        List of scenario contract violations detected.
    transition_log : list[str]
        Human-readable lifecycle transition log.
    total_time_ms : float
        Wall-clock time for the entire run in milliseconds.
    """

    state: RunState = RunState.CREATED
    steps_executed: list[dict] = field(default_factory=list)
    verdict: OverallVerdict | None = None
    trace: list[dict] = field(default_factory=list)
    scenario_violations: list[str] = field(default_factory=list)
    transition_log: list[str] = field(default_factory=list)
    total_time_ms: float = 0.0

    def to_audit_log(self, bundle=None) -> dict:
        """Produce a JSON-serialisable, fully-replayable audit log.

        The log carries every signal an auditor needs to reconstruct
        the verdict six months later: terminal state, transitions,
        scenario violations, per-step reason chain (incident triples
        + extracted triples + plausibility distance), per-claim
        verdicts.  When a bundle is supplied, its id and signature
        are included so the audit ties the verdict to a specific
        calibration.
        """
        log: dict = {
            "state": self.state.value,
            "transitions": list(self.transition_log),
            "scenario_violations": list(self.scenario_violations),
            "total_time_ms": self.total_time_ms,
            "n_steps_with_unavailable_signal": sum(
                1 for s in self.steps_executed
                if getattr(s.get("signal"), "signal_unavailable", False)
            ),
            "steps": [
                {
                    "step": s["step"],
                    "tool": s["tool"],
                    "blocked": s["blocked"],
                    "n_facts": s.get("n_facts", 0),
                    "extraction_tier": s.get("extraction_tier", 0),
                    "signal": {
                        "sim": (None if s["signal"].signal_unavailable
                                else s["signal"].sim),
                        "ver": s["signal"].ver,
                        "depth": s["signal"].depth,
                        "nov": s["signal"].nov,
                        "margin": (None if s["signal"].signal_unavailable
                                   else s["signal"].margin),
                        "unavailable": s["signal"].signal_unavailable,
                    },
                }
                for s in self.steps_executed
            ],
            "trace": list(self.trace),
        }
        if self.verdict is not None:
            log["verdict"] = {
                "overall_status": self.verdict.overall_status,
                "n_passed": self.verdict.n_passed,
                "n_total": self.verdict.n_total,
                "pass_rate": self.verdict.pass_rate,
                "severity_distribution": dict(
                    self.verdict.severity_distribution),
                "claim_verdicts": [
                    {
                        "type": cv.claim.type.value if hasattr(cv.claim.type, "value") else str(cv.claim.type),
                        "passed": cv.passed,
                        "score": cv.score,
                        "severity": cv.severity,
                        "evidence": cv.evidence,
                        "gms_method": cv.gms_method,
                    }
                    for cv in self.verdict.claim_verdicts
                ],
            }
        if bundle is not None:
            log["bundle_id"] = bundle.bundle_id
            log["bundle_signature"] = bundle.signature[:16]
            log["signing_method"] = bundle.signing_method
        return log


# ---------------------------------------------------------------------------
# Default tool argument templates
# ---------------------------------------------------------------------------

def _default_args_for_tool(
    tool_name: str,
    dataset_name: str = "iris",
    model_name: str = "RF_100",
    model_type: str = "RandomForestClassifier",
) -> dict:
    """Return reasonable default arguments for a tool.

    These defaults allow the orchestrator to execute a plan given only
    tool names.  In production the planner would supply full arguments.
    """
    _templates: dict[str, dict] = {
        "load_dataset": {
            "name": dataset_name,
            "builtin": dataset_name,
        },
        "data_summary": {
            "dataset": dataset_name,
        },
        "eda_correlation": {
            "dataset": dataset_name,
        },
        "preprocess_data": {
            "dataset": dataset_name,
        },
        "feature_select": {
            "dataset": dataset_name,
            "method": "f_classif",
        },
        "train_model": {
            "dataset": dataset_name,
            "model_type": model_type,
            "model_name": model_name,
        },
        "predict": {
            "dataset": dataset_name,
            "model_name": model_name,
        },
        "leaderboard": {
            "dataset": dataset_name,
        },
        "diagnose_robustness": {
            "dataset": dataset_name,
            "model_name": model_name,
        },
        "diagnose_fairness": {
            "dataset": dataset_name,
            "model_name": model_name,
            "protected_attribute": "feature_0",
        },
        "explain_importance": {
            "dataset": dataset_name,
            "model_name": model_name,
        },
        "explain_pdp": {
            "dataset": dataset_name,
            "model_name": model_name,
        },
    }
    return _templates.get(tool_name, {})


# ---------------------------------------------------------------------------
# GovernedOrchestrator
# ---------------------------------------------------------------------------

class GovernedOrchestrator:
    """End-to-end governed pipeline executor.

    Wires together the gateway, extractors, memory and verifier into a
    single ``run()`` method that takes a plan (list of tool names) and
    a behavioral scenario contract.

    Lifecycle: CREATED -> PREPARING -> EXECUTING -> EVALUATING ->
    COMPLETED | FAILED.  If a scenario violation is detected during
    PREPARING or a budget is exhausted during EXECUTING, the run
    transitions to ABORTED.

    Parameters
    ----------
    gateway : GovernedToolGateway
        The triple-gate execution layer (Ch 7).
    memory : AgentMemory
        Geometric memory for storing findings (Ch 5).
    verifier : ClaimVerifier
        Typed claim verifier (Ch 8).
    session : SessionState
        Mutable session state (Ch 3).
    store : GMSExpertStore
        The underlying GMS knowledge store.
    device : str
        PyTorch device string.
    dataset_name : str
        Default dataset name for tool argument templates.
    model_name : str
        Default model name for tool argument templates.
    model_type : str
        Default model type for tool argument templates.
    """

    def __init__(
        self,
        gateway: "GovernedToolGateway",
        memory: "AgentMemory",
        verifier: ClaimVerifier,
        session: "SessionState",
        store: Any = None,
        device: str = "cpu",
        dataset_name: str = "iris",
        model_name: str = "RF_100",
        model_type: str = "RandomForestClassifier",
        plausibility_threshold: float | None = None,
        policy_store: Any = None,
        policy_threshold: float | None = None,
        agent_identity: str | None = None,
        plan_policy: "OrchestratorPolicy | None" = None,
    ):
        self.gateway = gateway
        self.memory = memory
        self.verifier = verifier
        self.session = session
        self.store = store
        self.device = device
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_type = model_type
        # Calibrated plausibility cut applied to the run's internal
        # memory-store verifier. When None the verifier falls back to
        # the built-in cohort calibration (which fails closed on a
        # newly-seeded memory store), so a real value should be
        # supplied for any non-trivial run. Recover it via:
        #   judge = GMSJudge(memory.store); judge.calibrate()
        #   tau = CalibratedThresholds.from_judge(judge).tau_plausibility
        self.plausibility_threshold = plausibility_threshold
        # Optional calibrated tool-context plausibility (gate 3 on the
        # policy manifold).  When all three of policy_store /
        # policy_threshold / agent_identity are supplied the orchestrator
        # scores (agent_identity, "uses_tool", tool_name) against the
        # cap-trained policy store on every step and reports the
        # calibrated value in the signal vector.  When any is None the
        # signal stays "unavailable" with NaN sim/margin (the honest
        # behaviour; a deployment can SLO-cap the unavailable rate).
        self.policy_store = policy_store
        self.policy_threshold = policy_threshold
        self.agent_identity = agent_identity
        # Optional structural co-attestor: a pure-Python
        # ``OrchestratorPolicy`` rule pack that runs against the
        # (plan, scenario) tuple BEFORE EXECUTING.  Mirrors the
        # ``PlanPolicy`` co-attestor pattern from Chapter~9: independent
        # of geometry, deterministic, no learning.  When provided and
        # the policy returns ``allowed=False``, the orchestrator
        # transitions to FAILED with the policy reasons recorded in
        # the trace --- the run never reaches EXECUTING.  When None,
        # the orchestrator runs without the structural check (current
        # behaviour, suitable for tutorials).
        self.plan_policy = plan_policy
        # Optional bundle binding: when set, the orchestrator carries
        # the audit-grade artifact (signature, fingerprint, bootstrap
        # CIs, operating point) for every run.
        self._bundle = None

    @classmethod
    def from_bundle(
        cls,
        bundle,
        *,
        gateway: "GovernedToolGateway",
        memory: "AgentMemory",
        verifier: ClaimVerifier,
        session: "SessionState",
        store: Any = None,
        device: str = "cpu",
        dataset_name: str = "iris",
        model_name: str = "RF_100",
        model_type: str = "RandomForestClassifier",
    ) -> "GovernedOrchestrator":
        """Construct an orchestrator from a signed GovernanceBundle.

        Verifies the bundle's signature and (when present) the store
        fingerprint, then forwards the calibrated plausibility
        threshold to the orchestrator's internal verifier.  The
        bundle is retained on the instance so per-run reports include
        ``bundle_id`` and signature.
        """
        if not bundle.verify():
            raise ValueError(
                "Bundle signature does not verify. Refusing to bind."
            )
        if not bundle.is_plausibility_capable:
            raise ValueError(
                "Bundle has no tau_plausibility — orchestrator cannot "
                "bind a fail-closed verifier. Rebuild the bundle on a "
                "store with viable geodesic calibration."
            )
        if (store is not None and bundle.store_fingerprint is not None):
            from knowlytix.harness.governance.store_fingerprint import (
                StoreFingerprintMismatch,
                compute_store_fingerprint,
            )
            current = compute_store_fingerprint(store)
            if not bundle.store_fingerprint.matches(current):
                diff = bundle.store_fingerprint.diff(current)
                raise StoreFingerprintMismatch(
                    "Orchestrator store does not match the bundle's "
                    "calibration-time store. Components differing: "
                    f"{sorted(diff.keys())}."
                )
        # Forward the bundle's numeric range index to the verifier
        # so the run's internal mem_verifier inherits the second
        # attestor on numeric claims.  We rebuild a verifier rather
        # than mutate the caller's instance to keep the binding
        # explicit at the audit-trail level. For baseline verifiers
        # (Jaccard, Random) that lack ClaimVerifier's calibration
        # attributes, pass the supplied verifier through unchanged
        # so the from_bundle binding is non-destructive.
        if isinstance(verifier, ClaimVerifier):
            verifier_bound = ClaimVerifier(
                verifier.store,
                device=verifier.device,
                plausibility_threshold=bundle.tau_plausibility.mean,
                numeric_tolerance=verifier.numeric_tolerance,
                holonomy_threshold=verifier.holonomy_threshold,
                numeric_range_index=getattr(
                    bundle, "numeric_range_index", None,
                ),
            )
        else:
            verifier_bound = verifier
        instance = cls(
            gateway=gateway, memory=memory, verifier=verifier_bound,
            session=session, store=store, device=device,
            dataset_name=dataset_name, model_name=model_name,
            model_type=model_type,
            plausibility_threshold=bundle.tau_plausibility.mean,
        )
        instance._bundle = bundle
        instance._numeric_range_index = (
            getattr(bundle, "numeric_range_index", None)
        )
        return instance

    @property
    def bundle(self):
        return self._bundle

    # ----- Main execution loop --------------------------------------------

    def run(
        self,
        plan: list[str],
        scenario: Scenario,
    ) -> OrchestratorResult:
        """Execute a governed pipeline.

        Parameters
        ----------
        plan : list[str]
            Ordered list of tool names to execute.
        scenario : Scenario
            Behavioral contract to enforce.

        Returns
        -------
        OrchestratorResult
            Full result with steps, verdict, trace and violations.
        """
        t0 = time.time()
        rm = RunManager()
        violations: list[str] = []
        steps: list[dict] = []
        trace: list[dict] = []
        claims: list[Claim] = []

        # ==============================================================
        # Phase 1: PREPARING --- validate plan against scenario
        # ==============================================================
        rm.transition(RunState.PREPARING)

        # Check required tools
        for req in scenario.required_tools:
            if req not in plan:
                violations.append(
                    f"Required tool '{req}' missing from plan"
                )

        # Check forbidden tools
        forbidden_in_plan = [t for t in plan if t in scenario.forbidden_tools]
        for fb in forbidden_in_plan:
            violations.append(
                f"Forbidden tool '{fb}' present in plan"
            )

        # Check budget
        if len(plan) > scenario.max_tool_calls:
            violations.append(
                f"Plan has {len(plan)} steps but budget is "
                f"{scenario.max_tool_calls}"
            )
            rm.transition(RunState.ABORTED)
            return OrchestratorResult(
                state=rm.state,
                steps_executed=steps,
                verdict=None,
                trace=trace,
                scenario_violations=violations,
                transition_log=rm.transition_log(),
                total_time_ms=(time.time() - t0) * 1000,
            )

        # Structural co-attestor: pure-Python rule pack runs against
        # (plan, scenario) before any tool fires.  Same architecture
        # as Chapter~9's PlanPolicy --- independent of geometry,
        # deterministic, audit-grade.  Catches unknown tools,
        # circular dependencies, hallucinated dataset names, etc.
        # before the EXECUTING phase has a chance to compound the
        # damage.
        if self.plan_policy is not None:
            policy_verdict = self.plan_policy.check(
                plan=list(plan), scenario=scenario,
            )
            trace.append({
                "phase": "preparing",
                "plan_policy": {
                    "allowed": policy_verdict.allowed,
                    "rules_checked": policy_verdict.rules_checked,
                    "blocking_failures": list(policy_verdict.blocking_failures),
                    "advisory_failures": list(policy_verdict.advisory_failures),
                    "reasons": list(policy_verdict.reasons),
                },
            })
            if not policy_verdict.allowed:
                for reason in policy_verdict.reasons:
                    violations.append(f"plan_policy: {reason}")
                rm.transition(RunState.ABORTED)
                return OrchestratorResult(
                    state=rm.state,
                    steps_executed=steps,
                    verdict=None,
                    trace=trace,
                    scenario_violations=violations,
                    transition_log=rm.transition_log(),
                    total_time_ms=(time.time() - t0) * 1000,
                )

        # ==============================================================
        # Phase 2: EXECUTING --- run each tool through the gateway
        # ==============================================================
        rm.transition(RunState.EXECUTING)

        # Filter out forbidden tools (skip them, record violation)
        executable_plan = [
            t for t in plan if t not in scenario.forbidden_tools
        ]

        tool_call_count = 0

        for step_idx, tool_name in enumerate(executable_plan):
            # Budget guard
            if tool_call_count >= scenario.max_tool_calls:
                violations.append(
                    f"Budget exhausted at step {step_idx} "
                    f"({tool_call_count}/{scenario.max_tool_calls})"
                )
                rm.transition(RunState.ABORTED)
                return OrchestratorResult(
                    state=rm.state,
                    steps_executed=steps,
                    verdict=None,
                    trace=trace,
                    scenario_violations=violations,
                    transition_log=rm.transition_log(),
                    total_time_ms=(time.time() - t0) * 1000,
                )

            # Build default arguments for this tool
            args = _default_args_for_tool(
                tool_name,
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                model_type=self.model_type,
            )

            # Execute through the governed gateway
            gw_result = self.gateway.execute(
                tool_name, args, context="orchestrator",
            )
            tool_call_count += 1

            # Tool-level execution error: when the tool itself reports
            # a failure via raw["error"] (e.g. load_dataset called with
            # an unknown builtin), the gateway sees the call as
            # successful (gates passed) but the tool returned an error
            # rather than usable output.  In a high-stakes deployment
            # the orchestrator must catch this — letting downstream
            # extraction and verification continue on an errored
            # result produces "completed with 0 claims" which the
            # default PASS-on-empty-claims path then signs off on.
            # That is a silent acceptance of a failure mode.  Abort
            # instead, with the error reason recorded in the trace.
            tool_error = (
                gw_result.raw.get("error")
                if isinstance(gw_result.raw, dict) else None
            )
            if tool_error:
                violations.append(
                    f"Tool '{tool_name}' returned error: "
                    f"{str(tool_error)[:120]}"
                )
                steps.append({
                    "step": step_idx, "tool": tool_name, "args": args,
                    "blocked": False, "reason": f"tool_error: {tool_error}",
                    "display": gw_result.display[:200] if gw_result.display else "",
                    "extraction_tier": 0, "n_facts": 0,
                    "signal": self._compute_signal(
                        step_idx=step_idx,
                        total_steps=len(executable_plan),
                        extraction=None,
                        gw_result=gw_result,
                        tool_name=tool_name,
                    ),
                })
                rm.transition(RunState.ABORTED)
                return OrchestratorResult(
                    state=rm.state,
                    steps_executed=steps,
                    verdict=None,
                    trace=trace,
                    scenario_violations=violations,
                    transition_log=rm.transition_log(),
                    total_time_ms=(time.time() - t0) * 1000,
                )

            # Extract knowledge from the result
            extraction: ExtractionResult | None = None
            if not gw_result.blocked:
                extraction = extract_from_tool(
                    tool_name, args,
                    raw_result=gw_result.raw,
                    display_string=gw_result.display,
                )

                # Store findings in geometric memory
                if extraction and extraction.n_facts > 0:
                    triple_tuples = [
                        (t.head, t.relation, t.tail)
                        for t in extraction.triples
                    ]
                    enm_tuples = [
                        (e.category, e.entity_id, e.value)
                        for e in extraction.enm_entries
                    ]
                    try:
                        self.memory.store_findings(
                            triple_tuples, enm_tuples or None,
                        )
                    except Exception:
                        pass  # non-fatal: memory write failure

                # Build claims from extraction for later verification
                claims.extend(
                    self._claims_from_extraction(tool_name, extraction)
                )

            # Compute signal vector (uses calibrated policy_store
            # tool-context score when available; falls back to gateway
            # plausibility; honestly reports unavailable when neither).
            sigma = self._compute_signal(
                step_idx=step_idx,
                total_steps=len(executable_plan),
                extraction=extraction,
                gw_result=gw_result,
                tool_name=tool_name,
            )

            # Record step
            step_record = {
                "step": step_idx,
                "tool": tool_name,
                "args": args,
                "blocked": gw_result.blocked,
                "reason": gw_result.reason,
                "display": gw_result.display[:200] if gw_result.display else "",
                "extraction_tier": extraction.tier if extraction else 0,
                "n_facts": extraction.n_facts if extraction else 0,
                "signal": sigma,
            }
            steps.append(step_record)

            # Reason-chain enrichment: which entities did this tool
            # touch, and which corpus triples are incident to them?
            # Pure graph lookup, deterministic, audit-grade.
            incident_triples = self._step_incident_triples(
                tool_name, args, top_k=5,
            )

            # Trace entry (full detail)
            trace.append({
                "step": step_idx,
                "tool": tool_name,
                "blocked": gw_result.blocked,
                "gate_results": gw_result.gate_results,
                "plausibility_score": gw_result.plausibility_score,
                "execution_time_ms": gw_result.execution_time_ms,
                "extraction_tier": extraction.tier if extraction else 0,
                "n_triples": len(extraction.triples) if extraction else 0,
                "n_enm": len(extraction.enm_entries) if extraction else 0,
                "signal_vector": sigma.as_tuple(),
                "signal_unavailable": sigma.signal_unavailable,
                "incident_triples": incident_triples,
                "extracted_triples": [
                    [t.head, t.relation, t.tail]
                    for t in (extraction.triples if extraction else [])
                ],
            })

        # ==============================================================
        # Phase 3: EVALUATING --- verify collected claims
        # ==============================================================
        rm.transition(RunState.EVALUATING)

        # Check expected tool sequence
        if scenario.expected_tool_sequence:
            seq_violations = self._check_sequence(
                [s["tool"] for s in steps if not s["blocked"]],
                scenario.expected_tool_sequence,
            )
            violations.extend(seq_violations)

        # Verify claims against the *memory* store (which contains both
        # pre-ingested knowledge and runtime findings) so that entities
        # created during execution are available for verification.
        if claims:
            mem_verifier = ClaimVerifier(
                self.memory.store, device=self.device,
                plausibility_threshold=self.plausibility_threshold,
                numeric_range_index=getattr(
                    self, "_numeric_range_index", None,
                ),
            )
            verdict = mem_verifier.verify_all(claims)
        else:
            verdict = OverallVerdict(
                overall_status="PASS",
                n_passed=0,
                n_total=0,
                pass_rate=1.0,
            )

        # ==============================================================
        # Phase 4: Terminal state
        # ==============================================================
        if verdict.overall_status == "PASS" and not violations:
            rm.transition(RunState.COMPLETED)
        else:
            rm.transition(RunState.FAILED)

        return OrchestratorResult(
            state=rm.state,
            steps_executed=steps,
            verdict=verdict,
            trace=trace,
            scenario_violations=violations,
            transition_log=rm.transition_log(),
            total_time_ms=(time.time() - t0) * 1000,
        )

    # ----- Reason-chain helpers -------------------------------------------

    def _step_incident_triples(
        self, tool_name: str, args: dict, *, top_k: int = 5,
    ) -> list[list[str]]:
        """Return up to ``top_k`` corpus triples incident to the tool's
        argument entities (string-typed values).

        Pure graph lookup: no embeddings, no scoring.  Used to attach
        an audit-grade reason chain to every step's trace entry.
        """
        if (self.store is None or self.store.doc_graph is None):
            return []
        # Pull string-valued args as candidate entity names.
        candidates: list[str] = []
        for k, v in args.items():
            if isinstance(v, str):
                candidates.append(v)
        if not candidates:
            return []
        triples = list(self.store.doc_graph.triples)
        out: list[list[str]] = []
        for cand in candidates:
            for h, r, t in triples:
                if h == cand or t == cand:
                    out.append([h, r, t])
                    if len(out) >= top_k:
                        return out
        return out

    # ----- Signal vector computation --------------------------------------

    def _compute_signal(
        self,
        step_idx: int,
        total_steps: int,
        extraction: ExtractionResult | None,
        gw_result: "GatewayResult",
        tool_name: str | None = None,
    ) -> SignalVector:
        """Compute the five-component signal vector for a step.

        Components:
          sim    -- plausibility score (inverted: lower geodesic =
                    higher similarity)
          ver    -- 1.0 if extraction produced facts, 0.0 otherwise
          depth  -- normalized step index
          nov    -- 1.0 if new triples were extracted, 0.0 otherwise
          margin -- confidence margin from plausibility gate

        Source of plausibility (precedence):
          1. Calibrated tool-context score from the policy store —
             ``policy_store.score_triple(agent_identity, "uses_tool",
             tool_name)`` — when all three of policy_store /
             agent_identity / tool_name are available and the score
             is not None.  This is the gate-3 signal on the
             authorisation manifold; the score is bounded by the
             cap-trained ``rho_uses_tool`` from
             :func:`build_policy_store(..., loss_mode="cap")`.
          2. Gateway plausibility score (``gw_result.plausibility_score``)
             from the content store's gate 3.  Untrained on the
             ``should_call`` relation in most setups, so usually None.
          3. ``signal_unavailable=True`` with NaN values when neither
             source returns a score — honest reporting that no
             calibrated signal is available, instead of the 0.500
             fail-open default that hides a calibration gap.
        """
        # 1. Calibrated tool-context score (preferred).
        plaus = None
        plaus_source = "none"
        if (
            self.policy_store is not None
            and self.agent_identity is not None
            and tool_name is not None
        ):
            try:
                plaus = self.policy_store.score_triple(
                    self.agent_identity, "uses_tool", tool_name,
                )
                if plaus is not None:
                    plaus_source = "policy_uses_tool"
            except Exception:
                plaus = None
        # 2. Fall back to gateway's plausibility score.
        if plaus is None:
            plaus = gw_result.plausibility_score
            if plaus is not None:
                plaus_source = "gateway_should_call"
        # 3. Unavailable: NaN sim/margin, flag set so operating point
        #    can SLO-cap the rate of unavailable-signal steps.
        signal_unavailable = plaus is None
        if signal_unavailable:
            sim = float("nan")
            margin = float("nan")
        else:
            sim = math.exp(-plaus)
            margin = max(0.0, 1.0 - plaus / 2.0)  # normalize to [0, 1]
        self._last_plausibility_source = plaus_source

        # ver: extraction success
        ver = 1.0 if (extraction and extraction.n_facts > 0) else 0.0

        # depth: normalized progress
        depth = (step_idx + 1) / total_steps if total_steps > 0 else 0.0

        # nov: novelty (did we get new triples?)
        nov = 1.0 if (
            extraction and len(extraction.triples) > 0
        ) else 0.0

        return SignalVector(
            sim=sim if signal_unavailable else round(sim, 4),
            ver=round(ver, 4),
            depth=round(depth, 4),
            nov=round(nov, 4),
            margin=margin if signal_unavailable else round(margin, 4),
            signal_unavailable=signal_unavailable,
        )

    # ----- Claim construction from extraction -----------------------------

    @staticmethod
    def _claims_from_extraction(
        tool_name: str,
        extraction: ExtractionResult | None,
    ) -> list[Claim]:
        """Build typed claims from an extraction result.

        Creates FACT claims for graph triples and NUMERIC claims for
        ENM entries.  These claims will be verified in the EVALUATING
        phase.
        """
        if extraction is None or extraction.n_facts == 0:
            return []

        claims: list[Claim] = []

        # FACT claims from triples
        for triple in extraction.triples[:5]:  # cap to avoid explosion
            claims.append(Claim(
                type=ClaimType.FACT,
                text=f"{triple.head} {triple.relation} {triple.tail}",
                metadata={
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail,
                },
            ))

        # NUMERIC claims from ENM entries
        for enm in extraction.enm_entries[:5]:  # cap
            entity_id = f"{enm.entity_id}"
            claims.append(Claim(
                type=ClaimType.NUMERIC,
                text=f"{enm.entity_id} = {enm.value}",
                metadata={
                    "category": enm.category,
                    "entity_id": entity_id,
                    "claimed_value": enm.value,
                },
            ))

        return claims

    # ----- Sequence checking ----------------------------------------------

    @staticmethod
    def _check_sequence(
        executed: list[str],
        expected: list[str],
    ) -> list[str]:
        """Check that executed tools follow the expected subsequence.

        The expected sequence must appear in order (but not necessarily
        contiguously) within the executed list.

        Returns a list of violation messages.
        """
        violations = []
        exp_idx = 0
        for tool in executed:
            if exp_idx < len(expected) and tool == expected[exp_idx]:
                exp_idx += 1
        if exp_idx < len(expected):
            missing = expected[exp_idx:]
            violations.append(
                f"Expected sequence violation: tools {missing} "
                f"not executed in order"
            )
        return violations

    # ----- Pretty printing ------------------------------------------------

    @staticmethod
    def format_result(result: OrchestratorResult) -> str:
        """Format an orchestrator result for display."""
        lines = [
            "=== GovernedOrchestrator Result ===",
            f"  Final state: {result.state.value}",
            f"  Steps executed: {len(result.steps_executed)}",
            f"  Total time: {result.total_time_ms:.0f} ms",
            "",
            "--- Lifecycle Transitions ---",
        ]
        lines.extend(result.transition_log)

        if result.scenario_violations:
            lines.append("")
            lines.append("--- Scenario Violations ---")
            for v in result.scenario_violations:
                lines.append(f"  [!] {v}")

        if result.verdict:
            lines.append("")
            lines.append("--- Verification Verdict ---")
            lines.append(
                f"  Status: {result.verdict.overall_status}"
            )
            lines.append(
                f"  Passed: {result.verdict.n_passed}/"
                f"{result.verdict.n_total} "
                f"({result.verdict.pass_rate:.1%})"
            )

        # Honest reporting of unavailable-signal steps.
        n_unavailable = sum(
            1 for s in result.steps_executed
            if getattr(s.get("signal"), "signal_unavailable", False)
        )
        if n_unavailable > 0:
            n_steps = max(1, len(result.steps_executed))
            lines.append("")
            lines.append("--- Signal Availability ---")
            lines.append(
                f"  Steps with unavailable plausibility signal: "
                f"{n_unavailable}/{n_steps} "
                f"({n_unavailable / n_steps:.1%})"
            )

        lines.append("")
        lines.append("--- Steps ---")
        for s in result.steps_executed:
            status = "BLOCKED" if s["blocked"] else "OK"
            sigma = s["signal"]
            lines.append(
                f"  [{s['step']:2d}] {s['tool']:25s} "
                f"{status:7s} | tier={s['extraction_tier']} "
                f"facts={s['n_facts']:2d} | {sigma}"
            )

        return "\n".join(lines)
