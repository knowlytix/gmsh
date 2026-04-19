# SPDX-License-Identifier: Apache-2.0
"""GMSH Tool Gateway — runtime tool interception for agentic testing.

Intercepts every tool call in the Harness via ``HarnessHooks.before_tool``,
``HarnessHooks.intercept_tool``, and ``HarnessHooks.on_tool_result``,
enabling:

- **Policy enforcement**: block tool calls that violate policy rules.
- **Fault injection**: simulate latency, errors, and stale data.
- **Mock responses**: inject arbitrary mock responses via ``intercept_tool``.
- **Stale data substitution**: return cached results instead of live calls.
- **GMS plausibility**: score each tool call against the knowledge graph,
  and measure mock plausibility via geodesic distance.
- **Transcript capture**: full audit trail of every gateway decision.

Fault profiles can be loaded from YAML for repeatable chaos testing.

Usage::

    from knowlytix.harness.doe.gateway import ToolGateway, FaultProfile

    gw = ToolGateway(
        store=my_store,
        fault_profiles=[FaultProfile(tool_pattern="modeva_*", latency_ms=500)],
        tracing=tracing_gms,
        run_id="run_001",
    )
    gw.install(harness)
    harness.run("Load BikeSharing and train XGB")
    transcript = gw.get_transcript()
    gw.uninstall(harness)

The gateway uses the ``intercept_tool`` hook (returns ``tuple[bool, str|None]``)
for mock response injection and stale data substitution.  The ``before_tool``
hook (returns ``bool``) handles policy-only blocking where no mock is needed.
"""

from __future__ import annotations

import fnmatch
import random
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore
    from knowlytix.harness.core import Harness
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FaultProfile:
    """Fault injection profile matched against tool names.

    Parameters
    ----------
    tool_pattern : str
        ``fnmatch`` glob pattern applied to tool names (e.g. ``"modeva_*"``).
    latency_ms : float
        Artificial delay injected before allowing the call (milliseconds).
        Zero means no delay.
    error_rate : float
        Probability in ``[0, 1]`` that the call is blocked with an
        injected error.
    error_message : str
        Message recorded when an error is injected.
    stale_data : bool
        If True, the gateway caches tool results for this pattern.
        See module-level Limitations for caveats.
    mock_response : str or None
        If set, the tool call is blocked and this value is recorded in the
        transcript as the intended response.
    schema_strict : bool
        If True, enforce schema validation on tool arguments (reserved
        for future use).
    """

    tool_pattern: str = "*"
    latency_ms: float = 0.0
    error_rate: float = 0.0
    error_message: str = "Simulated tool failure"
    stale_data: bool = False
    mock_response: str | None = None
    schema_strict: bool = True

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "tool_pattern": self.tool_pattern,
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
            "error_message": self.error_message,
            "stale_data": self.stale_data,
            "mock_response": self.mock_response,
            "schema_strict": self.schema_strict,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FaultProfile":
        """Deserialize from a plain dictionary."""
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class GatewayDecision:
    """Record of a single gateway interception decision.

    Parameters
    ----------
    tool_name : str
        Name of the intercepted tool.
    action : str
        One of ``"allow"``, ``"block"``, ``"mock"``, ``"stale_data"``,
        ``"inject_fault"``.
    reason : str
        Human-readable explanation of the decision.
    gms_score : float or None
        GMS plausibility score for the tool call, if computed.
    mock_plausibility_geodesic : float or None
        When action is ``"mock"`` or ``"stale_data"``, geodesic distance
        from ``score_triple("agent", "expects_from", tool_name)`` measures
        how realistic the injected response is.  Lower = more plausible.
    timestamp : str
        ISO-ish timestamp; auto-filled if empty.
    """

    tool_name: str = ""
    action: str = ""
    reason: str = ""
    gms_score: float | None = None
    mock_plausibility_geodesic: float | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class GatewayTranscript:
    """Ordered log of all gateway decisions for a single run.

    Parameters
    ----------
    run_id : str
        Identifier for the benchmark or agent run.
    decisions : list[GatewayDecision]
        Chronological list of interception decisions.
    """

    run_id: str = ""
    decisions: list[GatewayDecision] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ToolGateway
# ---------------------------------------------------------------------------

class ToolGateway:
    """Runtime tool interception layer for agent testing.

    Installs itself into a :class:`~harness.core.Harness` via the
    ``HarnessHooks`` callback interface, intercepting every tool call
    to enforce policies, inject faults, and capture an audit transcript.

    Parameters
    ----------
    store : GMSExpertStore
        GMS store used for plausibility scoring.
    fault_profiles : list[FaultProfile] or None
        Ordered list of fault injection profiles.  The first matching
        profile (by ``fnmatch``) wins.
    policy_engine : object or None
        Optional policy engine with a ``check(tool_name, tool_args,
        run_id) -> bool`` method.  When provided, calls that fail the
        policy check are blocked.
    tracing : DOETracingGMS or None
        If provided, every decision is traced as GMS triples.
    run_id : str
        Identifier for the current run (used in tracing and transcript).
    """

    def __init__(
        self,
        store: "GMSExpertStore",
        fault_profiles: list[FaultProfile] | None = None,
        policy_engine: Any = None,
        tracing: "DOETracingGMS | None" = None,
        run_id: str = "",
    ):
        self.store = store
        self.fault_profiles = fault_profiles or []
        self.policy_engine = policy_engine
        self.tracing = tracing
        self.run_id = run_id
        self._transcript = GatewayTranscript(run_id=run_id)
        self._orig_before_tool = None
        self._orig_intercept_tool = None
        self._orig_on_tool_result = None
        self._stale_cache: dict[str, str] = {}

    # -- Install / Uninstall ------------------------------------------------

    def install(self, harness: "Harness") -> None:
        """Attach the gateway to a Harness, chaining with existing hooks.

        Wires both ``before_tool`` (policy blocking) and ``intercept_tool``
        (mock response / stale data injection).

        Parameters
        ----------
        harness : Harness
            The orchestration harness whose hooks will be intercepted.
        """
        self._orig_before_tool = harness.hooks.before_tool
        self._orig_intercept_tool = harness.hooks.intercept_tool
        self._orig_on_tool_result = harness.hooks.on_tool_result

        harness.hooks.before_tool = self._before_tool
        harness.hooks.intercept_tool = self._intercept_tool
        harness.hooks.on_tool_result = self._on_tool_result

    def uninstall(self, harness: "Harness") -> None:
        """Restore the original harness hooks.

        Parameters
        ----------
        harness : Harness
            The harness to restore.
        """
        harness.hooks.before_tool = self._orig_before_tool
        harness.hooks.intercept_tool = self._orig_intercept_tool
        harness.hooks.on_tool_result = self._orig_on_tool_result
        self._orig_before_tool = None
        self._orig_intercept_tool = None
        self._orig_on_tool_result = None

    # -- Hook implementations ----------------------------------------------

    def _before_tool(self, tool_name: str, tool_args: dict) -> bool:
        """Hook called before every tool execution.

        Handles policy blocking and error-rate fault injection.
        Mock responses and stale-data substitution are handled by
        ``_intercept_tool`` instead.

        Returns
        -------
        bool
            True to allow the tool call, False to block it.
        """
        # 1. Policy engine check
        if self.policy_engine is not None:
            try:
                allowed = self.policy_engine.check(
                    tool_name, tool_args, self.run_id,
                )
            except Exception as exc:
                allowed = True  # fail-open: do not block on engine errors
                print(f"  [Gateway] Policy engine error for '{tool_name}': {exc}")

            if not allowed:
                decision = GatewayDecision(
                    tool_name=tool_name,
                    action="block",
                    reason="Blocked by policy engine",
                )
                self._record_decision(decision)
                return self._chain_before_tool(tool_name, tool_args, False)

        # 2. Error injection — probabilistic block from fault profile
        profile = self._match_fault_profile(tool_name)
        if profile is not None:
            if profile.error_rate > 0 and random.random() < profile.error_rate:
                decision = GatewayDecision(
                    tool_name=tool_name,
                    action="inject_fault",
                    reason=profile.error_message,
                )
                self._record_decision(decision)
                return self._chain_before_tool(tool_name, tool_args, False)

            # 2b. Latency injection — delay then allow
            if profile.latency_ms > 0:
                time.sleep(profile.latency_ms / 1000.0)

        # 3. GMS plausibility score (informational, does not block)
        gms_score = self._score_plausibility(tool_name)

        # 4. Allow
        decision = GatewayDecision(
            tool_name=tool_name,
            action="allow",
            reason="Passed all gateway checks",
            gms_score=gms_score,
        )
        self._record_decision(decision)
        return self._chain_before_tool(tool_name, tool_args, True)

    def _intercept_tool(
        self, tool_name: str, tool_args: dict,
    ) -> tuple[bool, str | None]:
        """Hook called to intercept a tool call with an injected response.

        Checks fault profiles for mock responses and stale-data
        substitution.  When a mock or stale result is injected, the
        gateway scores mock plausibility via ``score_triple("agent",
        "expects_from", tool_name)`` — a low geodesic means the mock
        is geometrically realistic on the learned manifold.

        Returns
        -------
        tuple[bool, str | None]
            ``(True, response)`` to intercept, ``(False, None)`` to pass
            through to real execution.
        """
        profile = self._match_fault_profile(tool_name)

        if profile is not None:
            # Mock response injection
            if profile.mock_response is not None:
                geodesic = self._score_mock_plausibility(tool_name)
                decision = GatewayDecision(
                    tool_name=tool_name,
                    action="mock",
                    reason=(
                        f"Mock response injected for pattern "
                        f"'{profile.tool_pattern}'"
                    ),
                    mock_plausibility_geodesic=geodesic,
                )
                self._record_decision(decision)
                return True, profile.mock_response

            # Stale data substitution
            if profile.stale_data and tool_name in self._stale_cache:
                geodesic = self._score_mock_plausibility(tool_name)
                decision = GatewayDecision(
                    tool_name=tool_name,
                    action="stale_data",
                    reason=(
                        f"Stale cached result substituted for pattern "
                        f"'{profile.tool_pattern}'"
                    ),
                    mock_plausibility_geodesic=geodesic,
                )
                self._record_decision(decision)
                return True, self._stale_cache[tool_name]

        # Chain to original intercept_tool hook if present
        if self._orig_intercept_tool is not None:
            return self._orig_intercept_tool(tool_name, tool_args)

        return False, None

    def _on_tool_result(self, name: str, result_str: str, raw: Any) -> None:
        """Hook called after every tool execution.

        Caches results for stale-data profiles and chains to the original
        ``on_tool_result`` hook.

        Parameters
        ----------
        name : str
            Tool name.
        result_str : str
            Stringified tool result.
        raw : Any
            Raw result object.
        """
        # Cache for stale-data profiles
        profile = self._match_fault_profile(name)
        if profile is not None and profile.stale_data:
            self._stale_cache[name] = result_str

        # Chain to original hook
        if self._orig_on_tool_result is not None:
            self._orig_on_tool_result(name, result_str, raw)

    # -- Internal helpers ---------------------------------------------------

    def _chain_before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        gateway_allows: bool,
    ) -> bool:
        """Chain with the original ``before_tool`` hook.

        If the gateway blocks the call, return False immediately.
        Otherwise, delegate to the original hook (if any).
        """
        if not gateway_allows:
            return False
        if self._orig_before_tool is not None:
            return self._orig_before_tool(tool_name, tool_args)
        return True

    def _match_fault_profile(self, tool_name: str) -> FaultProfile | None:
        """Return the first fault profile whose pattern matches *tool_name*.

        Parameters
        ----------
        tool_name : str
            Name of the tool to match.

        Returns
        -------
        FaultProfile or None
            The first matching profile, or None if no profile matches.
        """
        for profile in self.fault_profiles:
            if fnmatch.fnmatch(tool_name, profile.tool_pattern):
                return profile
        return None

    def _score_plausibility(self, tool_name: str) -> float | None:
        """Score the plausibility of a tool call using the GMS store.

        Queries ``store.score_triple()`` with the tool name as a
        "has_capability" relation.  Returns None if scoring fails.

        Parameters
        ----------
        tool_name : str
            Name of the tool to score.

        Returns
        -------
        float or None
            GMS plausibility score, or None on error.
        """
        try:
            score = self.store.score_triple("agent", "has_capability", tool_name)
            return float(score) if score is not None else None
        except Exception:
            return None

    def _score_mock_plausibility(self, tool_name: str) -> float | None:
        """Score mock plausibility via geodesic distance.

        Uses ``score_triple("agent", "expects_from", tool_name)`` to
        measure how realistic a mock/stale response is for this tool.
        Lower geodesic = the tool is well-anchored on the manifold,
        so the agent is likely to accept the injected response.

        Parameters
        ----------
        tool_name : str
            Name of the tool being mocked.

        Returns
        -------
        float or None
            Geodesic distance (lower = more plausible), or None on error.
        """
        try:
            score = self.store.score_triple(
                "agent", "expects_from", tool_name,
            )
            return float(score) if score is not None else None
        except Exception:
            return None

    def _record_decision(self, decision: GatewayDecision) -> None:
        """Append a decision to the transcript and trace it.

        Parameters
        ----------
        decision : GatewayDecision
            The gateway decision to record.
        """
        self._transcript.decisions.append(decision)

        if self.tracing is not None:
            try:
                self.tracing.trace_gateway_decision(
                    run_id=self.run_id,
                    tool_name=decision.tool_name,
                    action=decision.action,
                    reason=decision.reason,
                    gms_score=decision.gms_score,
                    mock_plausibility_geodesic=decision.mock_plausibility_geodesic,
                )
            except Exception as exc:
                print(f"  [Gateway] Tracing error: {exc}")

    # -- Public API ---------------------------------------------------------

    def get_transcript(self) -> GatewayTranscript:
        """Return the full gateway transcript for this run.

        Returns
        -------
        GatewayTranscript
            All decisions recorded since the gateway was created.
        """
        return self._transcript


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_fault_profiles(path: str) -> list[FaultProfile]:
    """Load fault profiles from a YAML file.

    The file should contain a top-level ``fault_profiles`` key with a
    list of profile objects::

        fault_profiles:
          - tool_pattern: "modeva_*"
            latency_ms: 500
            error_rate: 0.1
          - tool_pattern: "gms_query"
            mock_response: "No results found."

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    list[FaultProfile]
        Deserialized fault profiles.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the file does not contain a ``fault_profiles`` key.
    """
    import yaml  # deferred import — yaml is only needed for this helper

    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "fault_profiles" not in data:
        raise ValueError(
            f"YAML file must contain a top-level 'fault_profiles' key: {path}"
        )

    return [FaultProfile.from_dict(entry) for entry in data["fault_profiles"]]
