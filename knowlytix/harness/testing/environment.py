# SPDX-License-Identifier: Apache-2.0
"""GMSH Environment Simulator — deterministic world-state for black-box testing.

In black-box testing, environment control is more important than elaborate
judging.  This module provides a deterministic, resettable world that the
System Under Test (SUT) interacts with through simulated tool endpoints.

Key capabilities:

- **Deterministic reset** — snap back to a known state between runs
- **Mock tool endpoints** — request/response schema, state transitions,
  failure modes, and response templates
- **Clock control** — frozen time, synthetic progression, scheduled events
- **Side-effect sandbox** — observable environment state changes for the
  side-effect oracle
- **Data fixtures** — versioned test data bundles with integrity hashes
- **GMS integration** — state changes encoded as triples for geometric
  consistency checking

Usage::

    env = EnvironmentSimulator.from_yaml("test_env.yaml")
    env.reset(seed=42)

    before = env.get_state()
    response = env.handle_tool_call("create_ticket", {"priority": "high"})
    after = env.get_state()
    diff = env.get_state_diff(before, after)
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore

    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StateTransition:
    """Trigger condition and state mutation for a simulated tool endpoint.

    Attributes
    ----------
    trigger_action : str
        Tool name or ``fnmatch`` pattern that activates this transition.
    trigger_condition : str
        Simple predicate on tool arguments.  Supported forms:
        ``"args.key == value"``, ``"args.key != value"``,
        ``"args.key in [a, b]"``, or ``""`` (always match).
    state_mutation : dict
        Key-value pairs to merge into the environment state.
        Values may contain ``{{args.key}}`` templates for argument
        substitution.
    response_template : str
        Template for the mock response.  Supports ``{{args.key}}``
        and ``{{state.key}}`` substitutions.
    """

    trigger_action: str = "*"
    trigger_condition: str = ""
    state_mutation: dict = field(default_factory=dict)
    response_template: str = ""

    def to_dict(self) -> dict:
        return {
            "trigger_action": self.trigger_action,
            "trigger_condition": self.trigger_condition,
            "state_mutation": dict(self.state_mutation),
            "response_template": self.response_template,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateTransition:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class FailureMode:
    """Named failure mode for a simulated tool endpoint.

    Attributes
    ----------
    mode_name : str
        Human-readable failure identifier.
    error_message : str
        Error message returned when this failure fires.
    probability : float
        Probability ``[0, 1]`` per call that this failure activates.
    trigger_after_n_calls : int
        If > 0, failure activates only after this many calls to the
        endpoint.  0 means any call can fail.
    """

    mode_name: str = ""
    error_message: str = "Simulated failure"
    probability: float = 0.0
    trigger_after_n_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "mode_name": self.mode_name,
            "error_message": self.error_message,
            "probability": self.probability,
            "trigger_after_n_calls": self.trigger_after_n_calls,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailureMode:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ToolEndpointSpec:
    """Schema for a simulated tool endpoint.

    Attributes
    ----------
    tool_name : str
        Exact name or ``fnmatch`` pattern for matching.
    description : str
        Human-readable description of what this endpoint simulates.
    request_schema : dict
        JSON Schema for valid arguments (for documentation; not enforced
        by default).
    response_schema : dict
        JSON Schema for responses (for documentation).
    state_transitions : list[StateTransition]
        Ordered list of state transitions.  First matching transition wins.
    failure_modes : list[FailureMode]
        Named failure modes with probabilities.
    initial_state : dict
        Per-endpoint initial state (merged into global state on reset).
    default_response : str
        Response returned when no transition matches.
    """

    tool_name: str = ""
    description: str = ""
    request_schema: dict = field(default_factory=dict)
    response_schema: dict = field(default_factory=dict)
    state_transitions: list[StateTransition] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)
    initial_state: dict = field(default_factory=dict)
    default_response: str = "OK"

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "request_schema": dict(self.request_schema),
            "response_schema": dict(self.response_schema),
            "state_transitions": [t.to_dict() for t in self.state_transitions],
            "failure_modes": [f.to_dict() for f in self.failure_modes],
            "initial_state": dict(self.initial_state),
            "default_response": self.default_response,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ToolEndpointSpec:
        d = dict(d)
        transitions_raw = d.pop("state_transitions", [])
        failures_raw = d.pop("failure_modes", [])
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        obj = cls(**filtered)
        obj.state_transitions = [StateTransition.from_dict(t) for t in transitions_raw]
        obj.failure_modes = [FailureMode.from_dict(f) for f in failures_raw]
        return obj


# ---------------------------------------------------------------------------
# Clock control
# ---------------------------------------------------------------------------

@dataclass
class ScheduledEvent:
    """Event triggered at a specific synthetic time.

    Attributes
    ----------
    at_ms : int
        Synthetic time (milliseconds) when this event fires.
    event_type : str
        One of ``"fault_inject"``, ``"state_change"``, ``"data_refresh"``.
    payload : dict
        Event-specific data (e.g. state mutations, fixture ID).
    """

    at_ms: int = 0
    event_type: str = ""
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "at_ms": self.at_ms,
            "event_type": self.event_type,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScheduledEvent:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ClockControl:
    """Synthetic time control.

    Attributes
    ----------
    mode : str
        ``"real"`` — use ``time.time()``; ``"frozen"`` — always return
        ``frozen_at``; ``"synthetic"`` — advance by ``tick_ms`` per call.
    frozen_at : str
        ISO timestamp for frozen mode.
    tick_ms : int
        Increment per ``now()`` call in synthetic mode.
    scheduled_events : list[ScheduledEvent]
        Events fired when synthetic time reaches ``at_ms``.
    """

    mode: str = "real"
    frozen_at: str = ""
    tick_ms: int = 1000
    scheduled_events: list[ScheduledEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "frozen_at": self.frozen_at,
            "tick_ms": self.tick_ms,
            "scheduled_events": [e.to_dict() for e in self.scheduled_events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClockControl:
        d = dict(d)
        events_raw = d.pop("scheduled_events", [])
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        obj = cls(**filtered)
        obj.scheduled_events = [ScheduledEvent.from_dict(e) for e in events_raw]
        return obj


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@dataclass
class DataFixture:
    """Versioned test data bundle with integrity hash.

    Attributes
    ----------
    fixture_id : str
        Unique fixture identifier.
    version : int
        Monotonic version number.
    data : dict
        The test data (documents, records, policies, etc.).
    schema : dict
        JSON Schema describing the data shape.
    content_hash : str
        SHA-256 of the JSON-serialized data.  Auto-computed if empty.
    """

    fixture_id: str = ""
    version: int = 1
    data: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.data:
            self.content_hash = hashlib.sha256(
                json.dumps(self.data, sort_keys=True).encode()
            ).hexdigest()

    def verify_integrity(self) -> bool:
        """Check that data matches the stored hash."""
        if not self.content_hash:
            return True
        expected = hashlib.sha256(
            json.dumps(self.data, sort_keys=True).encode()
        ).hexdigest()
        return expected == self.content_hash

    def to_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "version": self.version,
            "data": dict(self.data),
            "schema": dict(self.schema),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataFixture:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Environment Simulator
# ---------------------------------------------------------------------------

class EnvironmentSimulator:
    """Deterministic world-state simulator for black-box agent testing.

    Manages a set of simulated tool endpoints, each with state transitions
    and failure modes.  Tool calls from the SUT are routed through
    :meth:`handle_tool_call` which applies transitions, fires failures,
    and maintains observable environment state.

    Parameters
    ----------
    tool_specs : list[ToolEndpointSpec]
        Simulated tool endpoint definitions.
    fixtures : list[DataFixture] or None
        Versioned test data bundles.
    clock : ClockControl or None
        Time control configuration.
    store : GMSExpertStore or None
        Optional GMS store for geometric state-change encoding.
    tracing : DOETracingGMS or None
        Optional tracing store for audit.
    """

    def __init__(
        self,
        tool_specs: list[ToolEndpointSpec],
        fixtures: list[DataFixture] | None = None,
        clock: ClockControl | None = None,
        store: GMSExpertStore | None = None,
        tracing: DOETracingGMS | None = None,
    ):
        self.tool_specs = tool_specs
        self.fixtures = {f.fixture_id: f for f in (fixtures or [])}
        self.clock = clock or ClockControl()
        self.store = store
        self.tracing = tracing

        # Mutable state
        self._state: dict[str, Any] = {}
        self._initial_state: dict[str, Any] = {}
        self._call_counts: dict[str, int] = {}
        self._synthetic_time_ms: int = 0
        self._fired_events: set[int] = set()  # indices of fired scheduled events
        self._rng = random.Random(42)

        # Build initial state from all endpoint specs
        for spec in self.tool_specs:
            self._initial_state.update(copy.deepcopy(spec.initial_state))
        # Include fixture data in initial state
        for fid, fixture in self.fixtures.items():
            self._initial_state[f"fixture:{fid}"] = copy.deepcopy(fixture.data)

        # Initialize to initial state
        self._state = copy.deepcopy(self._initial_state)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, seed: int = 42) -> None:
        """Deterministic reset to initial state.

        Parameters
        ----------
        seed : int
            Random seed for failure mode sampling.
        """
        self._state = copy.deepcopy(self._initial_state)
        self._call_counts = {}
        self._synthetic_time_ms = 0
        self._fired_events = set()
        self._rng = random.Random(seed)

    def get_state(self) -> dict:
        """Return a deep copy of the full observable environment state."""
        return copy.deepcopy(self._state)

    def get_state_diff(self, before: dict, after: dict) -> dict:
        """Compute the difference between two environment states.

        Returns
        -------
        dict
            Keys: ``"added"`` (new keys), ``"removed"`` (deleted keys),
            ``"changed"`` (keys with different values).
        """
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        added = {k: after[k] for k in after_keys - before_keys}
        removed = {k: before[k] for k in before_keys - after_keys}
        changed = {}
        for k in before_keys & after_keys:
            if before[k] != after[k]:
                changed[k] = {"before": before[k], "after": after[k]}

        return {"added": added, "removed": removed, "changed": changed}

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def has_endpoint(self, tool_name: str) -> bool:
        """Check if a tool has a simulated endpoint."""
        return self._find_spec(tool_name) is not None

    def handle_tool_call(self, tool_name: str, args: dict) -> str:
        """Simulate tool execution.

        1. Find matching ToolEndpointSpec
        2. Increment call counter
        3. Check failure modes (probabilistic)
        4. Match and apply first qualifying state transition
        5. Render response template
        6. Trace state change if GMS store available

        Parameters
        ----------
        tool_name : str
            Tool being called.
        args : dict
            Tool arguments from the SUT.

        Returns
        -------
        str
            Simulated tool response.

        Raises
        ------
        KeyError
            If no endpoint spec matches the tool name.
        """
        spec = self._find_spec(tool_name)
        if spec is None:
            raise KeyError(
                f"No simulated endpoint for tool '{tool_name}'. "
                f"Available: {[s.tool_name for s in self.tool_specs]}"
            )

        # Track call count
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1
        n_calls = self._call_counts[tool_name]

        # Check failure modes
        failure = self._check_failure(spec, n_calls)
        if failure is not None:
            self._trace_state_change(tool_name, "failure", failure.mode_name)
            return f"ERROR: {failure.error_message}"

        # Match state transition
        transition = self._match_transition(spec, args)
        if transition is not None:
            # Apply state mutation
            resolved_mutation = self._resolve_templates(
                transition.state_mutation, args
            )
            self._state.update(resolved_mutation)

            # Render response
            response = self._render_template(
                transition.response_template, args
            )
            self._trace_state_change(
                tool_name, "transition", str(list(resolved_mutation.keys()))
            )
            return response

        # No transition matched — return default response
        return self._render_template(spec.default_response, args)

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def now(self) -> int:
        """Current time in milliseconds.

        Behavior depends on ``clock.mode``:
        - ``"real"`` — wall-clock milliseconds
        - ``"frozen"`` — always returns ``frozen_at`` parsed as ms
        - ``"synthetic"`` — auto-increments by ``tick_ms`` each call
        """
        if self.clock.mode == "frozen":
            return self._parse_frozen_time()
        elif self.clock.mode == "synthetic":
            self._synthetic_time_ms += self.clock.tick_ms
            return self._synthetic_time_ms
        else:
            return int(time.time() * 1000)

    def advance(self, ms: int) -> list[ScheduledEvent]:
        """Advance synthetic clock by *ms* milliseconds.

        Returns any scheduled events that fire during the interval.

        Parameters
        ----------
        ms : int
            Milliseconds to advance.

        Returns
        -------
        list[ScheduledEvent]
            Events that fired during this interval.
        """
        old_time = self._synthetic_time_ms
        self._synthetic_time_ms += ms
        fired = []

        for i, event in enumerate(self.clock.scheduled_events):
            if i in self._fired_events:
                continue
            if old_time < event.at_ms <= self._synthetic_time_ms:
                self._fired_events.add(i)
                self._apply_scheduled_event(event)
                fired.append(event)

        return fired

    # ------------------------------------------------------------------
    # Fixture management
    # ------------------------------------------------------------------

    def load_fixture(self, fixture_id: str) -> DataFixture | None:
        """Retrieve a fixture by ID."""
        return self.fixtures.get(fixture_id)

    def verify_fixture_integrity(self, fixture_id: str) -> bool:
        """Verify a fixture's data integrity."""
        fixture = self.fixtures.get(fixture_id)
        if fixture is None:
            return False
        return fixture.verify_integrity()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def snapshot_state(self) -> dict:
        """Capture full simulator state for checkpoint/restore."""
        return {
            "state": copy.deepcopy(self._state),
            "call_counts": dict(self._call_counts),
            "synthetic_time_ms": self._synthetic_time_ms,
            "fired_events": list(self._fired_events),
        }

    def restore_state(self, snapshot: dict) -> None:
        """Restore simulator state from a snapshot."""
        self._state = copy.deepcopy(snapshot["state"])
        self._call_counts = dict(snapshot["call_counts"])
        self._synthetic_time_ms = snapshot["synthetic_time_ms"]
        self._fired_events = set(snapshot["fired_events"])

    @classmethod
    def from_yaml(
        cls,
        path: str,
        store: GMSExpertStore | None = None,
        tracing: DOETracingGMS | None = None,
    ) -> EnvironmentSimulator:
        """Load an environment simulator from a YAML configuration.

        Expected YAML structure::

            environment:
              clock:
                mode: frozen
                frozen_at: "2026-01-15T10:00:00"
              fixtures:
                - fixture_id: "fx_001"
                  version: 1
                  data: { ... }
              tool_specs:
                - tool_name: "create_ticket"
                  initial_state:
                    tickets: []
                  state_transitions:
                    - trigger_action: "create_ticket"
                      state_mutation:
                        last_ticket: "{{args.title}}"
                      response_template: "Ticket created: {{args.title}}"
                  failure_modes:
                    - mode_name: "timeout"
                      probability: 0.1
                      error_message: "Service timeout"

        Parameters
        ----------
        path : str
            Path to the YAML file.
        store : GMSExpertStore or None
            Optional GMS store.
        tracing : DOETracingGMS or None
            Optional tracing store.
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        if "environment" in raw:
            raw = raw["environment"]

        clock_raw = raw.get("clock", {})
        clock = ClockControl.from_dict(clock_raw) if clock_raw else ClockControl()

        fixtures_raw = raw.get("fixtures", [])
        fixtures = [DataFixture.from_dict(f) for f in fixtures_raw]

        specs_raw = raw.get("tool_specs", [])
        tool_specs = [ToolEndpointSpec.from_dict(s) for s in specs_raw]

        return cls(
            tool_specs=tool_specs,
            fixtures=fixtures,
            clock=clock,
            store=store,
            tracing=tracing,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_spec(self, tool_name: str) -> ToolEndpointSpec | None:
        """Find the first endpoint spec matching the tool name."""
        for spec in self.tool_specs:
            if fnmatch.fnmatch(tool_name, spec.tool_name):
                return spec
            if spec.tool_name == tool_name:
                return spec
        return None

    def _check_failure(
        self, spec: ToolEndpointSpec, n_calls: int,
    ) -> FailureMode | None:
        """Check failure modes for this call."""
        for fm in spec.failure_modes:
            if fm.trigger_after_n_calls > 0 and n_calls < fm.trigger_after_n_calls:
                continue
            if fm.probability > 0 and self._rng.random() < fm.probability:
                return fm
        return None

    def _match_transition(
        self, spec: ToolEndpointSpec, args: dict,
    ) -> StateTransition | None:
        """Return the first matching state transition."""
        for trans in spec.state_transitions:
            # Check action pattern
            if not fnmatch.fnmatch(spec.tool_name, trans.trigger_action):
                if trans.trigger_action != "*":
                    continue

            # Check condition
            if trans.trigger_condition and not self._eval_condition(
                trans.trigger_condition, args
            ):
                continue

            return trans
        return None

    def _eval_condition(self, condition: str, args: dict) -> bool:
        """Evaluate a simple condition predicate.

        Supports:
        - ``"args.key == value"``
        - ``"args.key != value"``
        - ``"args.key in [a, b, c]"``
        - ``""`` (always true)
        """
        condition = condition.strip()
        if not condition:
            return True

        # Parse "args.key OP value"
        match = re.match(
            r"args\.(\w+)\s*(==|!=|in)\s*(.+)", condition
        )
        if not match:
            return True  # unrecognized condition → pass

        key, op, value_str = match.groups()
        actual = args.get(key)

        if op == "==":
            return str(actual) == value_str.strip().strip("'\"")
        elif op == "!=":
            return str(actual) != value_str.strip().strip("'\"")
        elif op == "in":
            # Parse [a, b, c] list
            items = [
                s.strip().strip("'\"")
                for s in value_str.strip().strip("[]").split(",")
            ]
            return str(actual) in items

        return True

    def _resolve_templates(self, mutation: dict, args: dict) -> dict:
        """Resolve ``{{args.key}}`` templates in mutation values."""
        resolved = {}
        for k, v in mutation.items():
            if isinstance(v, str):
                resolved[k] = self._render_template(v, args)
            else:
                resolved[k] = v
        return resolved

    def _render_template(self, template: str, args: dict) -> str:
        """Render ``{{args.key}}`` and ``{{state.key}}`` placeholders."""
        if not template:
            return ""

        def replace_match(m: re.Match) -> str:
            namespace, key = m.group(1), m.group(2)
            if namespace == "args":
                return str(args.get(key, f"<{key}>"))
            elif namespace == "state":
                return str(self._state.get(key, f"<{key}>"))
            return m.group(0)

        return re.sub(r"\{\{(\w+)\.(\w+)\}\}", replace_match, template)

    def _parse_frozen_time(self) -> int:
        """Parse frozen_at string to milliseconds since epoch."""
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(self.clock.frozen_at)
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return 0

    def _apply_scheduled_event(self, event: ScheduledEvent) -> None:
        """Apply a scheduled event to the environment state."""
        if event.event_type == "state_change":
            self._state.update(event.payload)
        elif event.event_type == "data_refresh":
            fixture_id = event.payload.get("fixture_id", "")
            if fixture_id in self.fixtures:
                fixture_key = f"fixture:{fixture_id}"
                self._state[fixture_key] = copy.deepcopy(
                    self.fixtures[fixture_id].data
                )
        # fault_inject events are handled by the gateway layer

    def _trace_state_change(
        self, tool_name: str, change_type: str, detail: str,
    ) -> None:
        """Trace a state change in the GMS tracing store."""
        if self.tracing is None:
            return
        try:
            self.tracing.store.learn_triples(
                [(tool_name, f"caused_{change_type}", detail)],
            )
        except Exception:
            pass  # tracing failures are non-fatal
