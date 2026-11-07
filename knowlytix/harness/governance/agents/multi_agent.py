# SPDX-License-Identifier: Apache-2.0
"""Multi-agent wiring for governed data science teams.

Provides role-based agent isolation with shared session state and
handoff protocols.  Each GovernedAgent owns a separate domain GMS
(AgentMemory) for capability isolation, while sharing a single
SessionState for data flow.

Key design decisions:
  - Role permissions are STRUCTURAL (set-based check), not advisory.
    A tool call outside the allowed set raises PermissionError.
  - Domain stores are isolated: one agent cannot write triples into
    another agent's AgentMemory.
  - The orchestrator records every handoff as a HandoffRecord for
    auditability.

Built in Chapter 17 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from knowlytix.harness.governance.extractors import extract_from_tool
from knowlytix.harness.governance.memory import AgentMemory
from knowlytix.harness.governance.session import SessionState
from knowlytix.harness.governance.tools import execute_tool

# ── Data Classes ────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentRole:
    """Defines what an agent is allowed and forbidden to do.

    Frozen by design: production deployment must not mutate role
    permissions at runtime --- a hardening guard against the
    ``frozen_role_mutation`` adversarial class. The
    ``allowed_tools`` and ``forbidden_tools`` sets must also be
    *frozen* (``frozenset``); a mutable ``set`` would let callers
    extend the allow-list after construction even on a frozen
    dataclass. The ``__post_init__`` enforces frozenset.
    """
    name: str
    allowed_tools: frozenset[str]
    forbidden_tools: frozenset[str]
    description: str

    def __post_init__(self) -> None:
        # Coerce to frozenset so callers can pass plain sets at
        # construction time without sacrificing immutability.
        if not isinstance(self.allowed_tools, frozenset):
            object.__setattr__(self, "allowed_tools",
                               frozenset(self.allowed_tools))
        if not isinstance(self.forbidden_tools, frozenset):
            object.__setattr__(self, "forbidden_tools",
                               frozenset(self.forbidden_tools))
        # allowed and forbidden must be disjoint --- a tool listed in
        # both is ambiguous; default-deny resolves it but the audit
        # trail must record this as a configuration error.
        overlap = self.allowed_tools & self.forbidden_tools
        if overlap:
            raise ValueError(
                f"AgentRole '{self.name}' has overlapping allowed and "
                f"forbidden tools: {sorted(overlap)}"
            )


@dataclass
class HandoffRecord:
    """Audit trail for inter-agent handoffs."""
    from_agent: str
    to_agent: str
    shared_keys: list[str]
    timestamp: float


@dataclass
class ToolCallRecord:
    """Record of a single tool execution within the pipeline."""
    agent_name: str
    tool_name: str
    args: dict
    display: str
    raw: dict
    n_triples: int
    timestamp: float


@dataclass
class AgentStats:
    """Per-agent statistics accumulated during pipeline execution."""
    tools_called: int = 0
    triples_learned: int = 0
    tool_names: list[str] = field(default_factory=list)
    blocked_attempts: int = 0


@dataclass
class PipelineResult:
    """Outcome of a complete multi-agent pipeline run."""
    tool_records: list[ToolCallRecord]
    handoff_records: list[HandoffRecord]
    agent_stats: dict[str, AgentStats]
    total_tools: int = 0
    total_triples: int = 0
    total_handoffs: int = 0


# ── Predefined Roles ────────────────────────────────────────────────

DATA_AGENT_ROLE = AgentRole(
    name="DataAgent",
    allowed_tools={"load_dataset", "data_summary", "eda_correlation",
                   "preprocess_data"},
    forbidden_tools={"train_model", "diagnose_robustness",
                     "diagnose_fairness", "explain_importance",
                     "explain_pdp", "explain_shap"},
    description="Loads and prepares data. Cannot train or diagnose.",
)

TRAINING_AGENT_ROLE = AgentRole(
    name="TrainingAgent",
    allowed_tools={"train_model", "predict", "leaderboard",
                   "feature_select"},
    forbidden_tools={"load_dataset", "diagnose_fairness",
                     "preprocess_data"},
    description="Trains models and builds leaderboards. Cannot load data or check fairness.",
)

DIAGNOSTICS_AGENT_ROLE = AgentRole(
    name="DiagnosticsAgent",
    allowed_tools={"diagnose_robustness", "explain_importance",
                   "explain_pdp", "explain_shap"},
    forbidden_tools={"train_model", "load_dataset", "preprocess_data"},
    description="Runs robustness and explainability diagnostics. Cannot train or load.",
)

FAIRNESS_AGENT_ROLE = AgentRole(
    name="FairnessAgent",
    allowed_tools={"diagnose_fairness"},
    forbidden_tools={"train_model", "load_dataset", "preprocess_data",
                     "predict"},
    description="Runs fairness diagnostics only. Most restricted role.",
)


# ── GovernedAgent ───────────────────────────────────────────────────

class GovernedAgent:
    """An agent with its own domain store and structural tool permissions.

    Parameters
    ----------
    role : AgentRole
        Defines allowed and forbidden tools.
    session : SessionState
        Shared mutable session state (datasets, models, results).
    memory : AgentMemory
        Agent-private domain GMS. Each agent owns a separate store.
    """

    # Identity-immutable attributes: ``role``, ``session``, ``memory``
    # are set exactly once at construction. Reassignment after
    # construction raises ``AttributeError`` --- this defeats the
    # ``role_hijack`` adversarial class where an attacker tries
    # ``agent.role = TRAINING_AGENT_ROLE`` to assume privilege.
    _IDENTITY_ATTRS = frozenset({"role", "session", "memory"})

    def __init__(
        self,
        role: AgentRole,
        session: SessionState,
        memory: AgentMemory,
    ):
        # Bypass our own __setattr__ guard for the initial assignment.
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "stats", AgentStats())
        object.__setattr__(self, "_call_log",
                           [])  # type: list[ToolCallRecord]
        object.__setattr__(self, "_initialised", True)

    def __setattr__(self, name: str, value: object) -> None:
        if (getattr(self, "_initialised", False)
                and name in self._IDENTITY_ATTRS):
            raise AttributeError(
                f"GovernedAgent.{name} is identity-immutable "
                f"after construction (role-hijack defense)."
            )
        object.__setattr__(self, name, value)

    # ── Permission Check ────────────────────────────────────────

    def can_execute(self, tool_name: str) -> bool:
        """Return True iff this agent's role permits the tool.

        Whitelist semantics (deny by default): the tool must be
        explicitly listed in ``allowed_tools``. A tool that is in
        neither set is *forbidden* --- this defeats the
        ``tool_alias_attack`` adversarial class where a hostile
        caller registers a new tool name to bypass a stale
        forbidden-set.

        ``forbidden_tools`` is retained as documentation: it states
        *why* the tool is denied, and lets the audit log distinguish
        an explicit deny from a default-deny.
        """
        if not isinstance(tool_name, str) or not tool_name:
            return False
        if tool_name in self.role.forbidden_tools:
            return False
        return tool_name in self.role.allowed_tools

    # ── Tool Execution ──────────────────────────────────────────

    def execute(
        self, tool_name: str, args: dict,
    ) -> tuple[str, dict]:
        """Execute a tool with structural role enforcement.

        Parameters
        ----------
        tool_name : str
            Name of the tool to call.
        args : dict
            Arguments to pass to the tool.

        Returns
        -------
        tuple[str, dict]
            (display_string, raw_dict) from the tool.

        Raises
        ------
        PermissionError
            If the tool is not in the agent's allowed set, OR is
            in the forbidden set, OR is not a recognized tool name
            (default-deny on unknown).
        """
        if not self.can_execute(tool_name):
            self.stats.blocked_attempts += 1
            reason = ("explicitly forbidden"
                      if tool_name in self.role.forbidden_tools
                      else "not in allow-list (default-deny)")
            raise PermissionError(
                f"[{self.role.name}] Tool '{tool_name}' is not permitted "
                f"({reason}). Allowed: {sorted(self.role.allowed_tools)}"
            )

        display, raw = execute_tool(tool_name, args, self.session)

        # Extract triples and store in this agent's domain memory
        extraction = extract_from_tool(tool_name, args, raw, display)
        n_triples = 0
        if extraction.n_facts > 0:
            triples = [
                (t.head, t.relation, t.tail) for t in extraction.triples
            ]
            enm = [
                (e.category, e.entity_id, e.value)
                for e in extraction.enm_entries
            ]
            if triples:
                self.memory.store_findings(triples, enm or None)
                n_triples = len(triples)

        # Record stats
        self.stats.tools_called += 1
        self.stats.triples_learned += n_triples
        self.stats.tool_names.append(tool_name)

        record = ToolCallRecord(
            agent_name=self.role.name,
            tool_name=tool_name,
            args=args,
            display=display,
            raw=raw,
            n_triples=n_triples,
            timestamp=time.time(),
        )
        self._call_log.append(record)

        return display, raw

    # ── Domain Store Isolation ──────────────────────────────────

    def domain_stats(self) -> dict:
        """Return statistics from this agent's private domain store."""
        return self.memory.memory_stats()

    def __repr__(self) -> str:
        return (
            f"GovernedAgent(role={self.role.name!r}, "
            f"tools_called={self.stats.tools_called}, "
            f"triples={self.stats.triples_learned})"
        )


# ── MultiAgentOrchestrator ─────────────────────────────────────────

class MultiAgentOrchestrator:
    """Coordinates multiple GovernedAgents with handoff protocols.

    Parameters
    ----------
    agents : dict[str, GovernedAgent]
        Mapping from agent name to GovernedAgent instance.
    shared_session : SessionState
        The shared session state all agents operate on.
    """

    def __init__(
        self,
        agents: dict[str, GovernedAgent],
        shared_session: SessionState,
    ):
        self.agents = agents
        self.shared_session = shared_session
        self._handoffs: list[HandoffRecord] = []
        self._tool_records: list[ToolCallRecord] = []

    # ── Handoff ─────────────────────────────────────────────────

    def handoff(self, from_agent: str, to_agent: str) -> HandoffRecord:
        """Record a handoff between two agents.

        Inspects the shared session to determine what data keys
        are available for the receiving agent.

        Parameters
        ----------
        from_agent : str
            Name of the handing-off agent.
        to_agent : str
            Name of the receiving agent.

        Returns
        -------
        HandoffRecord
            Audit record of the handoff.
        """
        shared_keys = []
        if self.shared_session.datasets:
            shared_keys.append(
                f"datasets:{list(self.shared_session.datasets.keys())}"
            )
        if self.shared_session.models:
            shared_keys.append(
                f"models:{list(self.shared_session.models.keys())}"
            )
        if self.shared_session.results:
            shared_keys.append(
                f"results:{list(self.shared_session.results.keys())}"
            )

        record = HandoffRecord(
            from_agent=from_agent,
            to_agent=to_agent,
            shared_keys=shared_keys,
            timestamp=time.time(),
        )
        self._handoffs.append(record)
        return record

    # ── Pipeline Execution ──────────────────────────────────────

    def run_pipeline(
        self,
        phases: list[tuple[str, list[tuple[str, dict]]]],
    ) -> PipelineResult:
        """Run a multi-agent pipeline with automatic handoffs.

        Parameters
        ----------
        phases : list of (agent_name, tool_calls)
            Each phase is (agent_name, [(tool_name, args), ...]).
            The orchestrator inserts a handoff record between
            consecutive phases with different agents.

        Returns
        -------
        PipelineResult
            Full audit trail of tools, handoffs and per-agent stats.
        """
        all_records: list[ToolCallRecord] = []
        prev_agent_name: str | None = None

        for agent_name, tool_calls in phases:
            agent = self.agents[agent_name]

            # Insert handoff if switching agents
            if prev_agent_name is not None and prev_agent_name != agent_name:
                self.handoff(prev_agent_name, agent_name)

            for tool_name, args in tool_calls:
                display, raw = agent.execute(tool_name, args)
                # Copy the record from the agent's log
                record = agent._call_log[-1]
                all_records.append(record)
                self._tool_records.append(record)

            prev_agent_name = agent_name

        # Build per-agent stats
        agent_stats = {
            name: ag.stats for name, ag in self.agents.items()
        }

        return PipelineResult(
            tool_records=all_records,
            handoff_records=list(self._handoffs),
            agent_stats=agent_stats,
            total_tools=len(all_records),
            total_triples=sum(
                s.triples_learned for s in agent_stats.values()
            ),
            total_handoffs=len(self._handoffs),
        )

    # ── Boundary Enforcement Testing ────────────────────────────

    def test_boundary(
        self, agent_name: str, tool_name: str, args: dict,
    ) -> tuple[bool, str]:
        """Attempt a tool call and return whether it was blocked.

        Returns
        -------
        tuple[bool, str]
            (blocked, message). blocked is True if PermissionError
            was raised.
        """
        agent = self.agents[agent_name]
        try:
            agent.execute(tool_name, args)
            return False, f"{agent_name} executed {tool_name} (allowed)"
        except PermissionError as e:
            return True, str(e)

    # ── Cross-Store Isolation Test ──────────────────────────────

    @staticmethod
    def test_store_isolation(
        agent_a: GovernedAgent,
        agent_b: GovernedAgent,
    ) -> tuple[bool, str]:
        """Verify that two agents have distinct domain stores.

        Returns
        -------
        tuple[bool, str]
            (isolated, message). isolated is True if stores are
            distinct objects.
        """
        if agent_a.memory.store is agent_b.memory.store:
            return False, (
                f"{agent_a.role.name} and {agent_b.role.name} "
                "share the SAME store object -- isolation violated"
            )
        return True, (
            f"{agent_a.role.name} and {agent_b.role.name} "
            "have distinct store objects -- isolation preserved"
        )

    # ── Reporting ───────────────────────────────────────────────

    def format_handoffs(self) -> str:
        """Format handoff records as a human-readable table."""
        lines = ["=" * 60, "Handoff Records", "=" * 60]
        for i, h in enumerate(self._handoffs, 1):
            lines.append(
                f"  [{i}] {h.from_agent} -> {h.to_agent}"
            )
            for k in h.shared_keys:
                lines.append(f"      {k}")
        return "\n".join(lines)

    def format_agent_stats(self) -> str:
        """Format per-agent stats as a human-readable table."""
        lines = ["=" * 60, "Per-Agent Statistics", "=" * 60]
        for name, agent in self.agents.items():
            s = agent.stats
            ds = agent.domain_stats()
            lines.append(f"  {name}:")
            lines.append(f"    Tools called:    {s.tools_called}")
            lines.append(f"    Tool names:      {s.tool_names}")
            lines.append(f"    Triples learned: {s.triples_learned}")
            lines.append(f"    Blocked:         {s.blocked_attempts}")
            lines.append(
                f"    Domain store:    entities={ds.get('entities', 'N/A')}, "
                f"triples={ds.get('triples', 'N/A')}, "
                f"enm={ds.get('enm_entries', 'N/A')}"
            )
        return "\n".join(lines)
