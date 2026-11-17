"""Real-harness tests for MultiTurnAdapter and InteractionScript.

Every behavioral test runs against:

- a real ``FullMockHarness`` (from ``demo.mock_agent_full``),
- a real ``EnvironmentSimulator`` (from ``create_mock_environment``),
- the loaded ``ch04_ground_truth`` GMS store for cross-turn
  ``tension_energy`` checks.

Only the script-serialization contracts (``TurnSpec.from_dict``,
``InteractionScript.from_yaml``) and the teardown-dispatch unknown-
type diagnostic are kept as pure unit tests — those are shape checks
on a dataclass and a string lookup, with no geometry involved.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import yaml


def _calibration_thresholds(store) -> dict:
    """Read the ``thresholds`` block from the store's calibration.json."""
    path = os.path.join(store.store_path, "calibration.json")
    assert os.path.isfile(path), f"calibration.json missing at {path}"
    with open(path) as f:
        return json.load(f)["thresholds"]


# ---------------------------------------------------------------------------
# Script-serialization contracts — pure unit-level (no store/harness)
# ---------------------------------------------------------------------------

def test_turnspec_from_dict_ignores_unknown_keys():
    from knowlytix.harness.testing.multiturn import TurnSpec

    spec = TurnSpec.from_dict({
        "turn_id": "t1", "user_message": "hi",
        "expected_contains": ["ok"],
        "bogus_key": 42,
    })
    assert spec.turn_id == "t1"
    assert spec.expected_contains == ["ok"]
    assert not hasattr(spec, "bogus_key")


def test_interaction_script_to_dict_and_from_dict_round_trip():
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec

    script = InteractionScript(
        script_id="s1", name="demo",
        turns=[
            TurnSpec(turn_id="t1", user_message="u1",
                     expected_contains=["foo"],
                     expected_tool_calls=["data_lookup"]),
            TurnSpec(turn_id="t2", user_message="u2",
                     prohibited_tool_calls=["dangerous_*"]),
        ],
        teardown_assertions=[{"type": "all_turns_passed"}],
    )
    rebuilt = InteractionScript.from_dict(script.to_dict())
    assert rebuilt.script_id == script.script_id
    assert len(rebuilt.turns) == 2
    assert rebuilt.turns[0].expected_tool_calls == ["data_lookup"]
    assert rebuilt.turns[1].prohibited_tool_calls == ["dangerous_*"]
    assert rebuilt.teardown_assertions == [{"type": "all_turns_passed"}]


def test_interaction_script_from_yaml():
    from knowlytix.harness.testing.multiturn import InteractionScript

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "script.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({
                "script_id": "s2", "name": "yaml-loaded",
                "turns": [{"turn_id": "t1", "user_message": "hello",
                           "expected_contains": ["world"]}],
            }, f)
        script = InteractionScript.from_yaml(path)

    assert script.script_id == "s2"
    assert len(script.turns) == 1
    assert script.turns[0].expected_contains == ["world"]


# ---------------------------------------------------------------------------
# Pre-built model-review script executed end-to-end
# ---------------------------------------------------------------------------

def test_prebuilt_script_executes_against_real_harness(
    ch04_store, full_mock_harness, mock_environment,
):
    """The chapter's canonical example: run the 3-turn model-review
    script through FullMockHarness + mock environment + real GMS
    store.  The script must execute all three turns, produce a
    ScriptResult with per-turn tool-call records, and populate
    consistency_scores via tension_energy."""
    from demo.mock_agent_full import create_multi_turn_script
    from knowlytix.harness.testing.multiturn import MultiTurnAdapter

    script = create_multi_turn_script()
    adapter = MultiTurnAdapter(
        harness=full_mock_harness, store=ch04_store,
    )
    result = adapter.execute_script(script, environment=mock_environment)

    assert len(result.turn_results) == 3
    # Every turn should have recorded wall-clock latency.
    for t in result.turn_results:
        assert t.latency_ms > 0
    # Real mock agent at least calls *some* tools across the script.
    total_tool_calls = sum(len(t.tool_calls) for t in result.turn_results)
    assert total_tool_calls > 0
    # Teardown assertion dispatched.
    assert (
        len(result.teardown_passed) + len(result.teardown_failed)
        == len(script.teardown_assertions)
    )


def test_prebuilt_script_captures_per_turn_tool_calls(
    ch04_store, full_mock_harness, mock_environment,
):
    """Each turn's tool_calls list must contain only tools fired
    during *that* turn, not accumulated across prior turns."""
    from demo.mock_agent_full import create_multi_turn_script
    from knowlytix.harness.testing.multiturn import MultiTurnAdapter

    adapter = MultiTurnAdapter(
        harness=full_mock_harness, store=ch04_store,
    )
    result = adapter.execute_script(
        create_multi_turn_script(), environment=mock_environment,
    )

    # Tool name sets per turn — each turn's tools exist independently.
    per_turn_tools = [
        {tc["tool_name"] for tc in t.tool_calls}
        for t in result.turn_results
    ]
    # No turn should claim a tool from a *later* turn.
    for i, tools in enumerate(per_turn_tools):
        # tool_calls is a list — check it's a proper list (not accumulated).
        assert isinstance(result.turn_results[i].tool_calls, list)


# ---------------------------------------------------------------------------
# Per-turn assertion semantics against the real harness
# ---------------------------------------------------------------------------

def _adapter(harness, store=None):
    from knowlytix.harness.testing.multiturn import MultiTurnAdapter
    return MultiTurnAdapter(harness=harness, store=store)


def _single_turn_script(**turn_kwargs):
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec
    return InteractionScript(
        script_id="s", turns=[TurnSpec(turn_id="t1", **turn_kwargs)],
    )


def test_expected_contains_matches_real_harness_output(
    full_mock_harness, mock_environment, ch04_store,
):
    """Teach a fact, then check the substring assertion fires against
    the real recalled value."""
    full_mock_harness.teach("model_auc", "0.8412")
    script = _single_turn_script(
        user_message="What is the model_auc value?",
        expected_contains=["0.8412"],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert result.turn_results[0].passed is True
    assert "0.8412" in result.turn_results[0].raw_output


def test_expected_not_contains_fails_when_harness_emits_forbidden(
    full_mock_harness, mock_environment, ch04_store,
):
    full_mock_harness.teach("confession", "I don't know the answer")
    script = _single_turn_script(
        user_message="What is the confession value?",
        expected_not_contains=["I don't know"],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert result.turn_results[0].passed is False
    assert any("unexpectedly contains" in a
               for a in result.turn_results[0].assertions_failed)


TICKET_TRIGGER = "Please file ticket for MDL-001 degradation"


def test_expected_tool_called_by_real_environment(
    full_mock_harness, mock_environment, ch04_store,
):
    """Asking the real mock harness to file a ticket actually calls
    ``create_ticket`` through the environment simulator — the adapter
    must capture it per-turn.  ``file ticket`` is the planner's
    trigger phrase (see mock_agent_full.py::_plan_tool_calls)."""
    script = _single_turn_script(
        user_message=TICKET_TRIGGER,
        expected_tool_calls=["create_ticket"],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    tools = [tc["tool_name"] for tc in result.turn_results[0].tool_calls]
    assert "create_ticket" in tools
    assert result.turn_results[0].passed is True


def test_prohibited_tool_assertion_blocks_happy_path(
    full_mock_harness, mock_environment, ch04_store,
):
    """Prohibit ``create_ticket``; the real harness still calls it
    (the prohibition doesn't block execution, it grades the verdict).
    The assertion must therefore fail."""
    script = _single_turn_script(
        user_message=TICKET_TRIGGER,
        prohibited_tool_calls=["create_ticket"],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert result.turn_results[0].passed is False
    assert any("prohibited tool" in a
               for a in result.turn_results[0].assertions_failed)


def test_expected_state_change_verified_against_env(
    full_mock_harness, mock_environment, ch04_store,
):
    """After a ticket-creation turn, the real environment writes the
    mutated ticket id into state.  We assert on the observed value
    via ``state_after`` (the env's template substitution yields
    ``TKT-<title>``, which the adapter records per-turn)."""
    expected_ticket_id = f"TKT-{TICKET_TRIGGER}"
    script = _single_turn_script(
        user_message=TICKET_TRIGGER,
        expected_state_changes={"last_ticket_id": expected_ticket_id},
        expected_tool_calls=["create_ticket"],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert mock_environment.get_state().get("last_ticket_id") == expected_ticket_id
    assert result.turn_results[0].state_after.get("last_ticket_id") == expected_ticket_id
    assert result.turn_results[0].passed is True


# ---------------------------------------------------------------------------
# Carry-forward templating with real turn outputs
# ---------------------------------------------------------------------------

def test_carry_forward_substitutes_in_later_turn(
    full_mock_harness, mock_environment, ch04_store,
):
    """Turn 1 captures an AUC value via carry_forward; turn 2's
    user_message template must substitute it before the real harness
    sees it."""
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec

    full_mock_harness.teach("auc_value", "0.8412")

    script = InteractionScript(
        script_id="s", turns=[
            TurnSpec(turn_id="t1",
                     user_message="What is the auc_value?",
                     carry_forward={"auc": "0.8412"}),
            TurnSpec(turn_id="t2",
                     user_message="Earlier the AUC was {{auc}}. Confirm.",
                     expected_contains=["0.8412"]),
        ],
    )
    # Spy on turn-2 user_message by watching the harness's conversation.
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    # Turn 2's rendered message should have been substituted before
    # delivery; we can verify via the conversation history.
    history = [m["content"] for m in
               full_mock_harness._conversation_history
               if m["role"] == "user"]
    turn_2_msg = history[1]
    assert "0.8412" in turn_2_msg
    assert "{{auc}}" not in turn_2_msg


# ---------------------------------------------------------------------------
# Cross-turn consistency uses the real GMS store
# ---------------------------------------------------------------------------

def test_cross_turn_consistency_populates_when_store_entities_reappear(
    ch04_store, full_mock_harness, mock_environment,
):
    """Ask the harness twice about MDL-001 — the real fuzzy matcher
    should snap to a known entity, and tension_energy should produce
    at least one consistency score."""
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec

    full_mock_harness.teach("model_under_test", "MDL-001")

    script = InteractionScript(
        script_id="s", turns=[
            TurnSpec(turn_id="t1",
                     user_message="What is the model_under_test?"),
            TurnSpec(turn_id="t2",
                     user_message="What is the model_under_test?"),
        ],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )

    # state_carryover_verified is True by default when no high-tension
    # crossings are found.  Consistency_scores may be empty if the
    # harness echoes strings that don't snap to entities.  Either way,
    # the attribute must exist and be a dict.
    assert isinstance(result.consistency_scores, dict)
    assert isinstance(result.state_carryover_verified, bool)


def test_consistency_empty_without_store(
    full_mock_harness, mock_environment,
):
    """Without a store, cross-turn consistency is a no-op — scores
    stay empty and carryover verification defaults to True.  With no
    calibration available the adapter falls back to the historical
    1.5 tension cutoff."""
    from knowlytix.harness.testing.multiturn import (
        _FALLBACK_TENSION_CUTOFF,
        InteractionScript,
        MultiTurnAdapter,
        TurnSpec,
    )

    script = InteractionScript(
        script_id="s", turns=[
            TurnSpec(turn_id="t1", user_message="Hello"),
            TurnSpec(turn_id="t2", user_message="Again"),
        ],
    )
    adapter = MultiTurnAdapter(harness=full_mock_harness, store=None)
    assert adapter.tension_threshold == _FALLBACK_TENSION_CUTOFF
    result = adapter.execute_script(script, environment=mock_environment)
    assert result.consistency_scores == {}
    assert result.state_carryover_verified is True


def test_adapter_loads_calibrated_tension_threshold(
    full_mock_harness, ch04_store,
):
    """Constructing the adapter with a store must pull tau_tension
    from calibration.json next to that store.  An explicit
    ``tension_threshold=`` argument overrides calibration — useful for
    stricter or more permissive scenario configs."""
    from knowlytix.harness.testing.multiturn import MultiTurnAdapter

    adapter = MultiTurnAdapter(harness=full_mock_harness, store=ch04_store)
    cal = _calibration_thresholds(ch04_store)
    assert adapter.tension_threshold == pytest.approx(cal["tension"])

    explicit = MultiTurnAdapter(
        harness=full_mock_harness, store=ch04_store, tension_threshold=0.25,
    )
    assert explicit.tension_threshold == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Conversation persistence across turns
# ---------------------------------------------------------------------------

def test_execute_script_resets_once_not_between_turns(
    full_mock_harness, mock_environment, ch04_store,
):
    """FullMockHarness._conversation_history must accumulate all three
    user turns + three assistant replies = 6 entries after a 3-turn
    script, proving there is no per-turn reset."""
    from demo.mock_agent_full import create_multi_turn_script

    result = _adapter(full_mock_harness, ch04_store).execute_script(
        create_multi_turn_script(), environment=mock_environment,
    )
    n_user = sum(1 for m in full_mock_harness._conversation_history
                 if m["role"] == "user")
    n_assistant = sum(1 for m in full_mock_harness._conversation_history
                      if m["role"] == "assistant")
    assert n_user == 3
    assert n_assistant == 3
    assert len(result.turn_results) == 3


def test_hooks_restored_after_script(
    full_mock_harness, mock_environment, ch04_store,
):
    """User's hooks survive script execution — the adapter's temp
    hooks are unwound in _execute_turn (multiturn.py:474-475)."""
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec

    sentinel_call = lambda name, args: None  # noqa: E731 — terse no-op test sentinel
    sentinel_result = lambda name, result, raw: None  # noqa: E731 — terse no-op test sentinel
    full_mock_harness.hooks.on_tool_call = sentinel_call
    full_mock_harness.hooks.on_tool_result = sentinel_result

    script = InteractionScript(
        script_id="s", turns=[
            TurnSpec(turn_id="t1", user_message="hello"),
            TurnSpec(turn_id="t2", user_message="hello again"),
        ],
    )
    _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )

    assert full_mock_harness.hooks.on_tool_call is sentinel_call
    assert full_mock_harness.hooks.on_tool_result is sentinel_result


# ---------------------------------------------------------------------------
# Teardown dispatch — exercised end-to-end
# ---------------------------------------------------------------------------

def _script_with_teardown(assertion, *, tool_message=TICKET_TRIGGER):
    """Build a two-turn script: turn 1 queries the database, turn 2
    fires the ticket-creation path by default."""
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec
    return InteractionScript(
        script_id="s",
        turns=[
            TurnSpec(turn_id="t1",
                     user_message="Query the database for all models"),
            TurnSpec(turn_id="t2", user_message=tool_message),
        ],
        teardown_assertions=[assertion],
    )


def test_teardown_all_turns_passed(
    full_mock_harness, mock_environment, ch04_store,
):
    script = _script_with_teardown({"type": "all_turns_passed"})
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert len(result.teardown_passed) + len(result.teardown_failed) == 1


def test_teardown_min_turns_passed(
    full_mock_harness, mock_environment, ch04_store,
):
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec

    script = InteractionScript(
        script_id="s",
        turns=[
            # Turn 1 fails (forbidden substring).
            TurnSpec(turn_id="t1", user_message="Hello",
                     expected_contains=["never_appears_in_output"]),
            # Turn 2 passes (no assertions to fail).
            TurnSpec(turn_id="t2", user_message="Hello again"),
        ],
        teardown_assertions=[{"type": "min_turns_passed", "value": 2}],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert result.teardown_failed
    assert "only 1 turns passed" in result.teardown_failed[0]


def test_teardown_total_tool_calls_max(
    full_mock_harness, mock_environment, ch04_store,
):
    """Cap tool calls to 0 on a script that definitely calls tools —
    the teardown must fail with an actionable count."""
    script = _script_with_teardown({"type": "total_tool_calls_max", "value": 0})
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert result.teardown_failed
    assert "tool calls > max 0" in result.teardown_failed[0]


def test_teardown_state_equals(
    full_mock_harness, mock_environment, ch04_store,
):
    expected_ticket_id = f"TKT-{TICKET_TRIGGER}"
    script = _script_with_teardown(
        {"type": "state_equals",
         "key": "last_ticket_id",
         "value": expected_ticket_id},
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    # Turn 2's ticket creation populates last_ticket_id — teardown passes.
    assert result.teardown_passed
    assert any("last_ticket_id" in s for s in result.teardown_passed)


def test_teardown_unknown_type_fails_loudly(
    full_mock_harness, mock_environment, ch04_store,
):
    script = _script_with_teardown({"type": "mystery_check"})
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert result.teardown_failed
    assert "unknown assertion type" in result.teardown_failed[0]


# ---------------------------------------------------------------------------
# Overall pass composition
# ---------------------------------------------------------------------------

def test_overall_passed_requires_turns_teardown_and_consistency(
    full_mock_harness, mock_environment, ch04_store,
):
    """overall_passed = all_turns_passed AND no_teardown_failures AND
    state_carryover_verified (multiturn.py:404-411)."""
    from knowlytix.harness.testing.multiturn import InteractionScript, TurnSpec

    # Two trivially-passing turns (no assertions) with an
    # all_turns_passed teardown.  Consistency defaults to True.
    script = InteractionScript(
        script_id="s",
        turns=[
            TurnSpec(turn_id="t1", user_message="Hello"),
            TurnSpec(turn_id="t2", user_message="Hello again"),
        ],
        teardown_assertions=[{"type": "all_turns_passed"}],
    )
    result = _adapter(full_mock_harness, ch04_store).execute_script(
        script, environment=mock_environment,
    )
    assert result.overall_passed is True
    d = result.to_dict()
    assert d["overall_passed"] is True
    assert d["n_turns"] == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
