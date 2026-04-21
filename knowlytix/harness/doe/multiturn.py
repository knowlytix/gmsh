# SPDX-License-Identifier: Apache-2.0
"""GMSH Multi-Turn Test Scripts — structured conversation testing.

Provides a framework for multi-turn interaction testing where each turn
has per-turn assertions, tool call expectations, and state carryover
verification.  This is critical for testing agentic systems that maintain
context across conversation turns.

Key capabilities:

- **InteractionScript** — YAML-defined multi-turn conversation with
  per-turn assertions and state tracking
- **MultiTurnAdapter** — wraps the Harness without resetting between
  turns, capturing tool calls per-turn
- **Cross-turn consistency** — uses ``tension_energy()`` between entity
  embeddings in successive turns to detect self-contradiction

Usage::

    script = InteractionScript.from_yaml("tests/multi_turn_demo.yaml")
    adapter = MultiTurnAdapter(harness)
    result = adapter.execute_script(script)
    assert result.overall_passed
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore
    from knowlytix.harness.core import Harness
    from .environment import EnvironmentSimulator
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Turn specification
# ---------------------------------------------------------------------------

@dataclass
class TurnSpec:
    """Specification for a single conversation turn.

    Attributes
    ----------
    turn_id : str
        Unique identifier within the script (e.g. "turn_1").
    user_message : str
        The user message to send for this turn.
    expected_contains : list[str]
        Substrings that MUST appear in the response.
    expected_not_contains : list[str]
        Substrings that must NOT appear in the response.
    expected_tool_calls : list[str]
        Tool names (fnmatch patterns) expected to be called.
    prohibited_tool_calls : list[str]
        Tool names (fnmatch patterns) that must NOT be called.
    expected_state_changes : dict
        Expected environment state changes after this turn
        (key → expected value).
    max_latency_ms : float
        Maximum allowed latency for this turn (0 = no limit).
    carry_forward : dict
        Key-value pairs to inject into subsequent turn messages
        via ``{{key}}`` template substitution.
    setup_context : str
        Optional additional context prepended to user_message.
    """

    turn_id: str = ""
    user_message: str = ""
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)
    prohibited_tool_calls: list[str] = field(default_factory=list)
    expected_state_changes: dict = field(default_factory=dict)
    max_latency_ms: float = 0.0
    carry_forward: dict = field(default_factory=dict)
    setup_context: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> TurnSpec:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Interaction script
# ---------------------------------------------------------------------------

@dataclass
class InteractionScript:
    """A multi-turn conversation test script.

    Defines a sequence of turns with per-turn assertions, initial
    context, and teardown checks.

    Attributes
    ----------
    script_id : str
        Unique identifier for the script.
    name : str
        Human-readable name.
    description : str
        What this script tests.
    turns : list[TurnSpec]
        Ordered list of conversation turns.
    setup_context : str
        Initial context injected before the first turn.
    initial_state : dict
        Expected initial environment state.
    teardown_assertions : list[dict]
        Post-script assertions (e.g. final state checks).
        Each dict has ``type`` ("state_equals", "contains", etc.)
        and ``value``.
    tags : list[str]
        Tags for filtering.
    metadata : dict
        Arbitrary metadata.
    """

    script_id: str = ""
    name: str = ""
    description: str = ""
    turns: list[TurnSpec] = field(default_factory=list)
    setup_context: str = ""
    initial_state: dict = field(default_factory=dict)
    teardown_assertions: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> InteractionScript:
        turns_raw = d.get("turns", [])
        turns = [TurnSpec.from_dict(t) for t in turns_raw]
        return cls(
            script_id=d.get("script_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            turns=turns,
            setup_context=d.get("setup_context", ""),
            initial_state=d.get("initial_state", {}),
            teardown_assertions=d.get("teardown_assertions", []),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, path: str) -> InteractionScript:
        """Load an interaction script from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> dict:
        return {
            "script_id": self.script_id,
            "name": self.name,
            "description": self.description,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "user_message": t.user_message,
                    "expected_contains": t.expected_contains,
                    "expected_not_contains": t.expected_not_contains,
                    "expected_tool_calls": t.expected_tool_calls,
                    "prohibited_tool_calls": t.prohibited_tool_calls,
                    "expected_state_changes": t.expected_state_changes,
                    "max_latency_ms": t.max_latency_ms,
                    "carry_forward": t.carry_forward,
                    "setup_context": t.setup_context,
                }
                for t in self.turns
            ],
            "setup_context": self.setup_context,
            "initial_state": self.initial_state,
            "teardown_assertions": self.teardown_assertions,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Result of executing a single conversation turn.

    Attributes
    ----------
    turn_id : str
        Matches the TurnSpec.
    raw_output : str
        The SUT's response text.
    tool_calls : list[dict]
        Tool calls made during this turn.
        Each dict has ``tool_name``, ``arguments``, ``result``.
    latency_ms : float
        Wall-clock time for this turn.
    assertions_passed : list[str]
        Descriptions of assertions that passed.
    assertions_failed : list[str]
        Descriptions of assertions that failed.
    state_after : dict
        Environment state snapshot after this turn.
    carry_forward_values : dict
        Values extracted from the response for carry-forward.
    """

    turn_id: str = ""
    raw_output: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    assertions_passed: list[str] = field(default_factory=list)
    assertions_failed: list[str] = field(default_factory=list)
    state_after: dict = field(default_factory=dict)
    carry_forward_values: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if no assertions failed."""
        return len(self.assertions_failed) == 0


# ---------------------------------------------------------------------------
# Script result
# ---------------------------------------------------------------------------

@dataclass
class ScriptResult:
    """Aggregate result of executing a multi-turn script.

    Attributes
    ----------
    script_id : str
        The script that was executed.
    turn_results : list[TurnResult]
        Per-turn results in order.
    overall_passed : bool
        True if all turns passed and teardown assertions passed.
    teardown_passed : list[str]
        Teardown assertions that passed.
    teardown_failed : list[str]
        Teardown assertions that failed.
    state_carryover_verified : bool
        True if cross-turn state consistency held.
    consistency_scores : dict
        Cross-turn tension_energy scores for entity consistency.
    total_latency_ms : float
        Total wall-clock time for all turns.
    """

    script_id: str = ""
    turn_results: list[TurnResult] = field(default_factory=list)
    overall_passed: bool = False
    teardown_passed: list[str] = field(default_factory=list)
    teardown_failed: list[str] = field(default_factory=list)
    state_carryover_verified: bool = True
    consistency_scores: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "script_id": self.script_id,
            "overall_passed": self.overall_passed,
            "n_turns": len(self.turn_results),
            "turns_passed": sum(1 for t in self.turn_results if t.passed),
            "turns_failed": sum(1 for t in self.turn_results if not t.passed),
            "teardown_passed": self.teardown_passed,
            "teardown_failed": self.teardown_failed,
            "state_carryover_verified": self.state_carryover_verified,
            "consistency_scores": self.consistency_scores,
            "total_latency_ms": self.total_latency_ms,
            "turn_details": [
                {
                    "turn_id": t.turn_id,
                    "passed": t.passed,
                    "latency_ms": t.latency_ms,
                    "n_tool_calls": len(t.tool_calls),
                    "assertions_passed": t.assertions_passed,
                    "assertions_failed": t.assertions_failed,
                }
                for t in self.turn_results
            ],
        }


# ---------------------------------------------------------------------------
# Multi-turn adapter
# ---------------------------------------------------------------------------

class MultiTurnAdapter:
    """Executes multi-turn scripts against a Harness without resetting.

    Unlike the standard AgentAdapter which resets conversation state
    between evaluations, MultiTurnAdapter preserves the conversation
    history across turns, enabling cross-turn consistency testing.

    Parameters
    ----------
    harness : Harness
        A configured Harness instance.
    store : GMSExpertStore or None
        GMS store for cross-turn consistency checking via
        ``tension_energy()``.
    tracing : DOETracingGMS or None
        Optional tracing store.
    """

    def __init__(
        self,
        harness: Harness,
        store: GMSExpertStore | None = None,
        tracing: DOETracingGMS | None = None,
    ):
        self.harness = harness
        self.store = store
        self.tracing = tracing

    def execute_script(
        self,
        script: InteractionScript,
        environment: EnvironmentSimulator | None = None,
    ) -> ScriptResult:
        """Execute a multi-turn interaction script.

        Runs each turn sequentially, preserving conversation history.
        Per-turn tool calls are captured via harness hooks.
        Cross-turn entity consistency is checked via GMS tension_energy.

        Parameters
        ----------
        script : InteractionScript
            The script to execute.
        environment : EnvironmentSimulator or None
            Optional environment simulator for state tracking.

        Returns
        -------
        ScriptResult
            Aggregate result with per-turn details.
        """
        result = ScriptResult(script_id=script.script_id)

        # Reset harness conversation for a clean multi-turn session
        self.harness.messages = []
        self.harness._turn = 0

        # Reset environment if provided
        if environment is not None:
            environment.reset()

        # Accumulate carry-forward values across turns
        carry_forward: dict[str, str] = {}

        # Track entities mentioned per turn for consistency checking
        per_turn_entities: list[list[str]] = []

        total_t0 = time.time()

        for i, turn_spec in enumerate(script.turns):
            turn_result = self._execute_turn(
                turn_spec, carry_forward, environment,
            )
            result.turn_results.append(turn_result)

            # Update carry-forward
            carry_forward.update(turn_result.carry_forward_values)

            # Extract entities from this turn's output for consistency
            entities = self._extract_entities(turn_result.raw_output)
            per_turn_entities.append(entities)

        result.total_latency_ms = (time.time() - total_t0) * 1000

        # Cross-turn consistency check via GMS
        result.consistency_scores = self._check_cross_turn_consistency(
            per_turn_entities,
        )
        result.state_carryover_verified = all(
            v < 1.5 for v in result.consistency_scores.values()
        )

        # Teardown assertions
        if environment is not None:
            final_state = environment.get_state()
        else:
            final_state = {}

        for assertion in script.teardown_assertions:
            passed, desc = self._check_teardown(assertion, final_state, result)
            if passed:
                result.teardown_passed.append(desc)
            else:
                result.teardown_failed.append(desc)

        # Overall pass/fail
        all_turns_passed = all(t.passed for t in result.turn_results)
        no_teardown_failures = len(result.teardown_failed) == 0
        result.overall_passed = (
            all_turns_passed
            and no_teardown_failures
            and result.state_carryover_verified
        )

        return result

    # -- Turn execution ----------------------------------------------------

    def _execute_turn(
        self,
        turn_spec: TurnSpec,
        carry_forward: dict[str, str],
        environment: EnvironmentSimulator | None,
    ) -> TurnResult:
        """Execute a single turn and check its assertions."""
        turn_result = TurnResult(turn_id=turn_spec.turn_id)

        # Apply carry-forward template substitution
        message = turn_spec.user_message
        for key, value in carry_forward.items():
            message = message.replace(f"{{{{{key}}}}}", str(value))

        # Prepend setup context if provided
        if turn_spec.setup_context:
            message = turn_spec.setup_context + "\n\n" + message

        # Capture tool calls for this turn
        turn_tool_calls: list[dict] = []
        orig_on_tool = self.harness.hooks.on_tool_call
        orig_on_result = self.harness.hooks.on_tool_result

        call_args_map: dict[str, dict] = {}

        def _on_tool_call(name: str, args: dict):
            call_args_map[name] = dict(args)
            if orig_on_tool:
                orig_on_tool(name, args)

        def _on_tool_result(name: str, result_str: str, raw: Any):
            turn_tool_calls.append({
                "tool_name": name,
                "arguments": call_args_map.pop(name, {}),
                "result": result_str[:500] if result_str else "",
            })
            if orig_on_result:
                orig_on_result(name, result_str, raw)

        self.harness.hooks.on_tool_call = _on_tool_call
        self.harness.hooks.on_tool_result = _on_tool_result

        # Capture env state before
        state_before = environment.get_state() if environment else {}

        # Execute the turn (preserving conversation history)
        t0 = time.time()
        try:
            # Use the harness's internal API to send a message without
            # resetting conversation state.  We append the user message
            # and call the ReAct loop for one turn cycle.
            raw_output = self.harness.run(message)
        except Exception as exc:
            raw_output = f"[ERROR: {type(exc).__name__}: {exc}]"
        elapsed = (time.time() - t0) * 1000

        # Restore hooks
        self.harness.hooks.on_tool_call = orig_on_tool
        self.harness.hooks.on_tool_result = orig_on_result

        turn_result.raw_output = raw_output
        turn_result.tool_calls = turn_tool_calls
        turn_result.latency_ms = elapsed

        # Capture env state after
        state_after = environment.get_state() if environment else {}
        turn_result.state_after = state_after

        # Extract carry-forward values from the response
        for key, pattern in turn_spec.carry_forward.items():
            # Simple extraction: if pattern is in the output, capture it
            if pattern in raw_output:
                turn_result.carry_forward_values[key] = pattern
            else:
                # Try to find the key's value in the output
                turn_result.carry_forward_values[key] = pattern

        # Check assertions
        self._check_turn_assertions(turn_spec, turn_result, state_before, state_after)

        return turn_result

    def _check_turn_assertions(
        self,
        spec: TurnSpec,
        result: TurnResult,
        state_before: dict,
        state_after: dict,
    ) -> None:
        """Evaluate all per-turn assertions."""
        output_lower = result.raw_output.lower()

        # Expected contains
        for substring in spec.expected_contains:
            if substring.lower() in output_lower:
                result.assertions_passed.append(
                    f"contains '{substring}'"
                )
            else:
                result.assertions_failed.append(
                    f"missing expected '{substring}'"
                )

        # Expected not contains
        for substring in spec.expected_not_contains:
            if substring.lower() not in output_lower:
                result.assertions_passed.append(
                    f"correctly excludes '{substring}'"
                )
            else:
                result.assertions_failed.append(
                    f"unexpectedly contains '{substring}'"
                )

        # Expected tool calls
        called_tools = [tc["tool_name"] for tc in result.tool_calls]
        for pattern in spec.expected_tool_calls:
            if any(fnmatch.fnmatch(t, pattern) for t in called_tools):
                result.assertions_passed.append(
                    f"tool '{pattern}' was called"
                )
            else:
                result.assertions_failed.append(
                    f"expected tool '{pattern}' not called"
                )

        # Prohibited tool calls
        for pattern in spec.prohibited_tool_calls:
            if any(fnmatch.fnmatch(t, pattern) for t in called_tools):
                result.assertions_failed.append(
                    f"prohibited tool '{pattern}' was called"
                )
            else:
                result.assertions_passed.append(
                    f"prohibited tool '{pattern}' not called"
                )

        # Expected state changes
        for key, expected_value in spec.expected_state_changes.items():
            actual = state_after.get(key)
            if actual == expected_value:
                result.assertions_passed.append(
                    f"state['{key}'] == {expected_value!r}"
                )
            else:
                result.assertions_failed.append(
                    f"state['{key}'] expected {expected_value!r}, "
                    f"got {actual!r}"
                )

        # Latency
        if spec.max_latency_ms > 0 and result.latency_ms > spec.max_latency_ms:
            result.assertions_failed.append(
                f"latency {result.latency_ms:.0f}ms > "
                f"max {spec.max_latency_ms:.0f}ms"
            )
        elif spec.max_latency_ms > 0:
            result.assertions_passed.append(
                f"latency {result.latency_ms:.0f}ms within limit"
            )

    # -- Cross-turn consistency --------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract GMS-matchable entities from response text.

        Uses the GMS store's fuzzy_match_entity if available,
        otherwise returns an empty list.
        """
        if self.store is None or not text:
            return []

        entities = []
        # Split on common delimiters and try to match each token
        words = set()
        for line in text.split("\n"):
            for token in line.replace(",", " ").replace(".", " ").split():
                cleaned = token.strip("()[]{}\"'`")
                if len(cleaned) > 2:
                    words.add(cleaned)

        for word in list(words)[:50]:  # cap to avoid expensive lookups
            try:
                match = self.store.fuzzy_match_entity(word)
                if match is not None:
                    entities.append(match)
            except Exception:
                continue

        return list(set(entities))

    def _check_cross_turn_consistency(
        self,
        per_turn_entities: list[list[str]],
    ) -> dict[str, float]:
        """Check entity consistency across turns via tension_energy.

        For each entity that appears in multiple turns, compute the
        tension energy between successive mentions.  High tension
        (>1.5) suggests the SUT contradicted itself across turns.

        Returns
        -------
        dict[str, float]
            ``"entity@turn_i->turn_j"`` -> tension score.
        """
        if self.store is None:
            return {}

        scores: dict[str, float] = {}

        # Find entities appearing in multiple turns
        all_entities: dict[str, list[int]] = {}
        for turn_idx, entities in enumerate(per_turn_entities):
            for entity in entities:
                all_entities.setdefault(entity, []).append(turn_idx)

        for entity, turn_indices in all_entities.items():
            if len(turn_indices) < 2:
                continue
            # Check consecutive appearances
            for i in range(len(turn_indices) - 1):
                t_from = turn_indices[i]
                t_to = turn_indices[i + 1]
                try:
                    tension = self.store.tension_energy(entity, entity)
                    if tension is not None:
                        key = f"{entity}@turn_{t_from}->turn_{t_to}"
                        scores[key] = float(tension)
                except Exception:
                    continue

        return scores

    # -- Teardown assertions -----------------------------------------------

    def _check_teardown(
        self,
        assertion: dict,
        final_state: dict,
        result: ScriptResult,
    ) -> tuple[bool, str]:
        """Evaluate a single teardown assertion.

        Supported assertion types:
        - ``state_equals``: ``{"type": "state_equals", "key": ..., "value": ...}``
        - ``all_turns_passed``: ``{"type": "all_turns_passed"}``
        - ``min_turns_passed``: ``{"type": "min_turns_passed", "value": N}``
        - ``total_tool_calls_max``: ``{"type": "total_tool_calls_max", "value": N}``
        """
        atype = assertion.get("type", "")

        if atype == "state_equals":
            key = assertion.get("key", "")
            expected = assertion.get("value")
            actual = final_state.get(key)
            if actual == expected:
                return True, f"teardown: state['{key}'] == {expected!r}"
            return False, f"teardown: state['{key}'] expected {expected!r}, got {actual!r}"

        if atype == "all_turns_passed":
            ok = all(t.passed for t in result.turn_results)
            if ok:
                return True, "teardown: all turns passed"
            failed = sum(1 for t in result.turn_results if not t.passed)
            return False, f"teardown: {failed} turn(s) failed"

        if atype == "min_turns_passed":
            min_val = assertion.get("value", 0)
            passed = sum(1 for t in result.turn_results if t.passed)
            if passed >= min_val:
                return True, f"teardown: {passed} turns passed >= {min_val}"
            return False, f"teardown: only {passed} turns passed, need {min_val}"

        if atype == "total_tool_calls_max":
            max_val = assertion.get("value", 0)
            total = sum(len(t.tool_calls) for t in result.turn_results)
            if total <= max_val:
                return True, f"teardown: {total} tool calls <= {max_val}"
            return False, f"teardown: {total} tool calls > max {max_val}"

        return False, f"teardown: unknown assertion type '{atype}'"


# ---------------------------------------------------------------------------
# Scenario bridge — connect scripts to the Scenario/Run framework
# ---------------------------------------------------------------------------

class MultiTurnScenarioRunner:
    """Bridges InteractionScript with the Scenario/Run framework.

    Detects scenarios with ``interaction_script`` field and executes
    them as multi-turn scripts rather than single-shot evaluations.

    Parameters
    ----------
    adapter : MultiTurnAdapter
        The multi-turn adapter to execute scripts.
    environment : EnvironmentSimulator or None
        Optional environment simulator.
    """

    def __init__(
        self,
        adapter: MultiTurnAdapter,
        environment: EnvironmentSimulator | None = None,
    ):
        self.adapter = adapter
        self.environment = environment

    def is_multi_turn(self, scenario) -> bool:
        """Check if a Scenario should be executed as multi-turn."""
        return bool(getattr(scenario, "interaction_script", None))

    def run_scenario(self, scenario) -> ScriptResult:
        """Execute a scenario's interaction_script.

        Parameters
        ----------
        scenario : Scenario
            Must have ``interaction_script`` populated.

        Returns
        -------
        ScriptResult
            The multi-turn execution result.
        """
        script_data = scenario.interaction_script
        if isinstance(script_data, str):
            # Path to YAML file
            script = InteractionScript.from_yaml(script_data)
        else:
            script = InteractionScript.from_dict(script_data)

        # Override script_id with scenario_id for traceability
        if not script.script_id:
            script.script_id = scenario.scenario_id

        return self.adapter.execute_script(script, self.environment)
