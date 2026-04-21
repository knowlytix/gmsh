# SPDX-License-Identifier: Apache-2.0
"""Full-featured mock agentic system for comprehensive GMSTO testing.

Extends the base :class:`MockHarness` with every GMSTO testing capability:

1. **Multi-turn memory** — preserves conversation state across turns
2. **Stateful memory** — teach-then-recall for StatefulAgentTester
3. **Streaming output** — simulates chunk-by-chunk responses
4. **Confidence expression** — appends confidence ratings for CalibrationTester
5. **Environment-aware** — routes tool calls through EnvironmentSimulator
6. **Multi-agent roles** — persona-specific mock agents for multi-agent testing

Usage::

    from knowlytix.harness.fixtures import FullMockHarness, create_full_evaluator

    # Basic usage (drop-in replacement for MockHarness)
    harness = FullMockHarness()
    answer = harness.run("What is the AUC of MDL-001?")

    # Multi-turn (conversation persists across calls)
    harness.run("Tell me about MDL-001")
    harness.run("What is its risk rating?")  # context preserved

    # Stateful memory (teach → recall)
    harness.teach("project_deadline", "2026-06-15")
    harness.run("When is the project deadline?")  # → "2026-06-15"

    # Streaming (mock Anthropic streaming client)
    client = MockStreamClient(harness)
    with client.messages.stream(model="x", messages=[...]) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="")

    # Environment-aware
    from knowlytix.harness.doe.environment import EnvironmentSimulator
    harness = FullMockHarness(environment=env_simulator)

    # Multi-agent roles
    analyst = create_agent_for_role("analyst", expertise="credit risk")
    reviewer = create_agent_for_role("reviewer", expertise="compliance")
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Callable

from knowlytix.harness.fixtures.mock_agent import (
    MockHarness, _MODEL_DATA, _HANDLERS,
)

__all__ = [
    "FullMockHarness",
    "create_full_evaluator",
    "MockStreamClient",
    "create_mock_stream_client",
    "create_agent_for_role",
    "create_mock_environment",
    "create_multi_turn_script",
    "ENVIRONMENT_TOOLS",
]


# ---------------------------------------------------------------------------
# Additional tools for environment / BB testing
# ---------------------------------------------------------------------------

ENVIRONMENT_TOOLS = {
    "create_ticket": {
        "description": "Create a support or risk ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "description": {"type": "string"},
            },
            "required": ["title"],
        },
        "category": "workflow",
    },
    "query_database": {
        "description": "Query model validation records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "filter": {"type": "string"},
            },
            "required": ["table"],
        },
        "category": "data",
    },
    "approve_action": {
        "description": "Submit an action for approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string"},
                "target": {"type": "string"},
                "justification": {"type": "string"},
            },
            "required": ["action_type", "target"],
        },
        "category": "compliance",
    },
    "send_notification": {
        "description": "Send a notification to stakeholders.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message": {"type": "string"},
                "channel": {"type": "string", "enum": ["email", "slack", "sms"]},
            },
            "required": ["recipient", "message"],
        },
        "category": "communication",
    },
}


def _handle_create_ticket(args: dict, context: str) -> tuple[str, dict | None]:
    title = args.get("title", "Untitled")
    priority = args.get("priority", "medium")
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    return (
        f"Ticket {ticket_id} created: [{priority.upper()}] {title}",
        {"ticket_id": ticket_id, "title": title, "priority": priority},
    )


def _handle_query_database(args: dict, context: str) -> tuple[str, dict | None]:
    table = args.get("table", "")
    filter_str = args.get("filter", "")
    # Return mock data matching the model data
    if "model" in table.lower() or "mdl" in filter_str.upper():
        records = [
            {"model_id": mid, "name": d["name"], "risk": d["risk"],
             "test_auc": d["test_auc"]}
            for mid, d in _MODEL_DATA.items()
        ]
        if filter_str:
            records = [r for r in records if filter_str.upper() in str(r).upper()]
        return f"Query returned {len(records)} records: {records}", {"records": records}
    return f"Table '{table}' returned 0 records", {"records": []}


def _handle_approve_action(args: dict, context: str) -> tuple[str, dict | None]:
    action = args.get("action_type", "")
    target = args.get("target", "")
    justification = args.get("justification", "")
    return (
        f"Action '{action}' on '{target}' submitted for approval. "
        f"Justification: {justification or 'None provided'}",
        {"action_type": action, "target": target, "status": "pending_approval"},
    )


def _handle_send_notification(args: dict, context: str) -> tuple[str, dict | None]:
    recipient = args.get("recipient", "")
    message = args.get("message", "")
    channel = args.get("channel", "email")
    return (
        f"Notification sent to {recipient} via {channel}: {message[:100]}",
        {"recipient": recipient, "channel": channel, "sent": True},
    )


_FULL_HANDLERS = {
    **_HANDLERS,
    "create_ticket": _handle_create_ticket,
    "query_database": _handle_query_database,
    "approve_action": _handle_approve_action,
    "send_notification": _handle_send_notification,
}


# ---------------------------------------------------------------------------
# FullMockHarness
# ---------------------------------------------------------------------------

class FullMockHarness(MockHarness):
    """Extended mock with multi-turn, stateful memory, streaming, and confidence.

    Drop-in replacement for MockHarness with additional capabilities:

    - ``teach(key, value)`` / ``recall(key)`` for stateful memory testing
    - ``stream_response(prompt)`` yields text chunks for streaming tests
    - ``confidence_mode=True`` appends ``[Confidence: N/10]`` to answers
    - ``environment`` routes tool calls through EnvironmentSimulator
    - Conversation history preserved across ``run()`` calls
    """

    def __init__(
        self,
        environment=None,
        confidence_mode: bool = False,
        role_id: str = "",
        expertise: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Add environment tools
        self._tools.update(ENVIRONMENT_TOOLS)

        # Stateful memory store (for teach/recall)
        self._taught_facts: dict[str, str] = {}

        # Multi-turn conversation history
        self._conversation_history: list[dict[str, str]] = []

        # Environment simulator integration
        self._environment = environment

        # Confidence mode
        self._confidence_mode = confidence_mode
        self._confidence_level: int = 8  # default confidence 1-10

        # Multi-agent role
        self.role_id = role_id
        self.expertise = expertise

    # -- Stateful memory ---------------------------------------------------

    def teach(self, key: str, value: str) -> None:
        """Teach a fact for later recall (stateful memory testing)."""
        self._taught_facts[key.lower().strip()] = value

    def recall(self, key: str) -> str | None:
        """Recall a previously taught fact."""
        return self._taught_facts.get(key.lower().strip())

    def set_confidence(self, level: int) -> None:
        """Set the confidence level (1-10) for subsequent answers."""
        self._confidence_level = max(1, min(10, level))

    # -- Tool execution (environment-aware) --------------------------------

    def _execute_tool(self, name: str, args: dict) -> tuple[str, Any]:
        """Execute a tool, optionally routing through EnvironmentSimulator."""
        # Fire on_tool_call hook
        if self.hooks.on_tool_call:
            self.hooks.on_tool_call(name, args)

        # Check before_tool hook (gateway/policy interception point)
        if self.hooks.before_tool:
            allowed = self.hooks.before_tool(name, args)
            if not allowed:
                self._tool_records.append({
                    "tool": name, "args": args,
                    "result": "Tool call skipped by policy hook.",
                    "blocked": True,
                })
                return "Tool call skipped by policy hook.", None

        # Check intercept_tool hook (gateway mock injection point)
        if self.hooks.intercept_tool:
            intercepted, mock_response = self.hooks.intercept_tool(name, args)
            if intercepted:
                result = mock_response or f"Tool {name} intercepted."
                raw = {"source": "intercept", "tool": name}
                if self.hooks.on_tool_result:
                    self.hooks.on_tool_result(name, result, raw)
                self._tool_records.append({
                    "tool": name, "args": args,
                    "result": result, "blocked": False,
                })
                return result, raw

        # Route through environment simulator if available and it has the endpoint
        if self._environment is not None and self._environment.has_endpoint(name):
            result = self._environment.handle_tool_call(name, args)
            raw = {"source": "environment", "tool": name}
        else:
            # Use full handler map
            handler = _FULL_HANDLERS.get(name)
            if handler is None:
                result = f"Unknown tool: {name}"
                raw = None
            else:
                result, raw = handler(args, self._context)

        # Fire on_tool_result hook
        if self.hooks.on_tool_result:
            self.hooks.on_tool_result(name, result, raw)

        self._tool_records.append({
            "tool": name, "args": args, "result": result, "blocked": False,
        })
        return result, raw

    # -- Extended tool planning --------------------------------------------

    def _plan_tool_calls(self, prompt: str) -> list[tuple[str, dict]]:
        """Extended tool planning with environment-aware tools."""
        calls = super()._plan_tool_calls(prompt)

        prompt_lower = prompt.lower()

        # Pattern: ticket/issue creation
        if any(w in prompt_lower for w in ["create ticket", "file ticket", "open issue"]):
            calls.append(("create_ticket", {
                "title": prompt[:100], "priority": "medium",
            }))

        # Pattern: database query
        if any(w in prompt_lower for w in ["query", "database", "records", "all models"]):
            calls.append(("query_database", {
                "table": "model_inventory", "filter": "",
            }))

        # Pattern: approval
        if any(w in prompt_lower for w in ["approve", "authorization", "sign off"]):
            calls.append(("approve_action", {
                "action_type": "model_deployment",
                "target": "MDL-001",
                "justification": prompt[:100],
            }))

        # Pattern: notification
        if any(w in prompt_lower for w in ["notify", "alert", "send message"]):
            calls.append(("send_notification", {
                "recipient": "risk_committee",
                "message": prompt[:200],
                "channel": "email",
            }))

        return calls

    # -- Main run (multi-turn + stateful + confidence) ---------------------

    def run(self, prompt: str) -> str:
        """Run with conversation history, stateful memory, and confidence."""
        self._conversation_history.append({"role": "user", "content": prompt})

        # Check if the question matches a taught fact
        answer = self._check_taught_facts(prompt)

        if answer is None:
            # Fall back to standard mock agent behavior
            answer = self._run_standard(prompt)

        # Append confidence if enabled
        if self._confidence_mode:
            answer = f"{answer} [Confidence: {self._confidence_level}/10]"

        self._conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def _run_standard(self, prompt: str) -> str:
        """Standard MockHarness run logic (duplicated to avoid double-history)."""
        self._tool_records = []

        # Extract context from prompt
        ctx_match = re.search(
            r"<report>\s*(.*?)\s*</report>", prompt, re.DOTALL)
        if ctx_match:
            self._context = ctx_match.group(1)
        else:
            # Preserve existing context for multi-turn
            if not self._context:
                self._context = prompt

        # Extract question
        q_match = re.search(
            r"Answer the following question.*?\n\n(.+)", prompt, re.DOTALL)
        question = q_match.group(1).strip() if q_match else prompt

        # Plan and execute tools
        planned_calls = self._plan_tool_calls(question)
        tool_results = []
        blocked_tools = []

        for tool_name, tool_args in planned_calls:
            self._turn += 1
            result, raw = self._execute_tool(tool_name, tool_args)

            if "skipped by policy" in result:
                blocked_tools.append(tool_name)
                recovery = self._recover_from_blocked(
                    tool_name, tool_args, question)
                tool_results.append(recovery)
            else:
                tool_results.append(result)

        return self._compose_answer(question, tool_results)

    def _check_taught_facts(self, prompt: str) -> str | None:
        """Check if the prompt matches any taught fact.

        Matches either the stored key verbatim OR the key with underscores
        turned into spaces, so ``teach("project_deadline", ...)`` resolves
        against a natural-language prompt like "What is the project deadline?"
        — the documented pattern in the module docstring.
        """
        prompt_lower = prompt.lower()
        for key, value in self._taught_facts.items():
            if key in prompt_lower or key.replace("_", " ") in prompt_lower:
                return value
        return None

    # -- Streaming ---------------------------------------------------------

    def stream_response(self, prompt: str) -> list[str]:
        """Generate streaming chunks from a response.

        Returns a list of text chunks that, when concatenated, form
        the complete response.  Used by MockStreamClient.
        """
        answer = self.run(prompt)
        words = answer.split()
        chunks = []
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i + 3])
            if i > 0:
                chunk = " " + chunk
            chunks.append(chunk)
        return chunks if chunks else [answer]

    # -- Multi-turn helpers ------------------------------------------------

    def reset_conversation(self) -> None:
        """Reset conversation history (for new multi-turn sessions)."""
        self._conversation_history = []
        self.messages = []
        self._turn = 0
        self._context = ""

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Return the conversation history."""
        return list(self._conversation_history)

    # -- Accessors ---------------------------------------------------------

    @property
    def tool_records(self) -> list[dict]:
        return self._tool_records

    def get_all_tool_names(self) -> list[str]:
        """Return names of all available tools."""
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Mock Anthropic Streaming Client
# ---------------------------------------------------------------------------

class _MockContent:
    def __init__(self, text: str):
        self.text = text


class _MockUsage:
    def __init__(self, input_tokens: int = 150, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockResponse:
    """Mock Anthropic API response."""
    def __init__(self, text: str, input_tokens: int = 150, output_tokens: int = 50):
        self.content = [_MockContent(text)]
        self.usage = _MockUsage(input_tokens, output_tokens)


class _MockStream:
    """Context manager that yields text chunks."""

    def __init__(self, harness: FullMockHarness, prompt: str):
        self._harness = harness
        self._prompt = prompt
        self._chunks: list[str] = []

    def __enter__(self):
        self._chunks = self._harness.stream_response(self._prompt)
        return self

    def __exit__(self, *args):
        pass

    @property
    def text_stream(self):
        for chunk in self._chunks:
            time.sleep(0.001)  # simulate tiny latency between chunks
            yield chunk


class _MockMessages:
    """Mock Anthropic messages API."""

    def __init__(self, harness: FullMockHarness):
        self._harness = harness

    def create(self, **kwargs) -> _MockResponse:
        """Simulate messages.create()."""
        messages = kwargs.get("messages", [])
        content = messages[0]["content"] if messages else ""
        answer = self._harness.run(content)
        return _MockResponse(answer)

    def stream(self, **kwargs) -> _MockStream:
        """Simulate messages.stream() — returns a context manager."""
        messages = kwargs.get("messages", [])
        content = messages[0]["content"] if messages else ""
        return _MockStream(self._harness, content)


class MockStreamClient:
    """Mock Anthropic client with streaming support.

    Drop-in replacement for ``anthropic.Anthropic`` in tests that
    need streaming or standard message creation without API calls.

    Usage::

        client = MockStreamClient(harness)
        # Standard call
        resp = client.messages.create(model="x", messages=[...])
        # Streaming call
        with client.messages.stream(model="x", messages=[...]) as s:
            for chunk in s.text_stream:
                print(chunk)
    """

    def __init__(self, harness: FullMockHarness | None = None):
        if harness is None:
            harness = FullMockHarness()
        self.messages = _MockMessages(harness)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_full_evaluator(
    confidence_mode: bool = False,
    recovery_mode: str = "retry",
    deliberate_errors: bool = False,
    environment=None,
) -> Callable[[str, str], str]:
    """Create a full-featured mock evaluator for DOEGMSBenchmark.run().

    Returns ``(question, context) -> answer``.
    """
    harness = FullMockHarness(
        recovery_mode=recovery_mode,
        deliberate_errors=deliberate_errors,
        confidence_mode=confidence_mode,
        environment=environment,
    )

    def evaluator(question: str, context: str) -> str:
        harness._tool_records = []
        harness._turn = 0
        prompt = (
            f"You have been given the following report as context:\n\n"
            f"<report>\n{context}\n</report>\n\n"
            f"Answer the following question using the report and any tools "
            f"available to you.\n\n{question}"
        )
        return harness.run(prompt)

    evaluator.__repr__ = lambda: "FullMockAgentEvaluator()"
    return evaluator


def create_mock_stream_client(
    harness: FullMockHarness | None = None,
) -> MockStreamClient:
    """Create a mock streaming client for StreamingAdapter tests."""
    return MockStreamClient(harness)


def create_agent_for_role(
    role_id: str,
    expertise: str = "",
    recovery_mode: str = "retry",
    confidence_mode: bool = False,
) -> FullMockHarness:
    """Create a mock agent with a specific role persona.

    Used for multi-agent interaction testing. Each role gets its
    own FullMockHarness instance with role metadata.

    Parameters
    ----------
    role_id : str
        Role identifier (e.g. "analyst", "reviewer", "approver").
    expertise : str
        Domain expertise hint.
    recovery_mode : str
        Recovery behavior when tools are blocked.
    confidence_mode : bool
        Whether to append confidence ratings.
    """
    harness = FullMockHarness(
        role_id=role_id,
        expertise=expertise,
        recovery_mode=recovery_mode,
        confidence_mode=confidence_mode,
    )
    return harness


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def create_mock_environment():
    """Create a pre-configured EnvironmentSimulator for testing.

    Returns an EnvironmentSimulator with tool endpoints for:
    - create_ticket (with state transitions)
    - query_database
    - approve_action (with state transitions)
    - send_notification

    Does NOT require a GMS store — suitable for isolated testing.
    """
    from knowlytix.harness.doe.environment import (
        EnvironmentSimulator, ToolEndpointSpec, StateTransition,
        FailureMode, ClockControl, DataFixture,
    )

    ticket_spec = ToolEndpointSpec(
        tool_name="create_ticket",
        description="Create a support ticket",
        initial_state={"tickets_created": 0, "last_ticket_id": ""},
        state_transitions=[
            StateTransition(
                trigger_action="*",
                state_mutation={
                    "tickets_created": "{{state.tickets_created}}+1",
                    "last_ticket_id": "TKT-{{args.title}}",
                },
                response_template="Ticket created: {{args.title}} [{{args.priority}}]",
            ),
        ],
        failure_modes=[
            FailureMode(
                mode_name="service_unavailable",
                error_message="Ticket service unavailable",
                probability=0.0,  # disabled by default
            ),
        ],
        default_response="Ticket created successfully",
    )

    query_spec = ToolEndpointSpec(
        tool_name="query_database",
        description="Query model validation records",
        initial_state={"query_count": 0},
        state_transitions=[
            StateTransition(
                trigger_action="*",
                response_template="Query on {{args.table}}: 5 records found",
            ),
        ],
        default_response="Query executed: 0 records",
    )

    approve_spec = ToolEndpointSpec(
        tool_name="approve_action",
        description="Submit action for approval",
        initial_state={"pending_approvals": 0, "last_action": ""},
        state_transitions=[
            StateTransition(
                trigger_action="*",
                state_mutation={
                    "last_action": "{{args.action_type}}",
                    "approval_status": "pending",
                },
                response_template="Action '{{args.action_type}}' on '{{args.target}}' queued for approval",
            ),
        ],
        default_response="Action submitted for approval",
    )

    notify_spec = ToolEndpointSpec(
        tool_name="send_notification",
        description="Send stakeholder notification",
        initial_state={"notifications_sent": 0},
        state_transitions=[
            StateTransition(
                trigger_action="*",
                response_template="Notification sent to {{args.recipient}} via {{args.channel}}",
            ),
        ],
        default_response="Notification sent",
    )

    fixture = DataFixture(
        fixture_id="model_inventory",
        version=1,
        data={
            "models": [
                {"id": "MDL-001", "name": "CreditScore-XGB", "risk": "High"},
                {"id": "MDL-002", "name": "FraudNet-LSTM", "risk": "High"},
                {"id": "MDL-003", "name": "AML-GBM", "risk": "High"},
                {"id": "MDL-004", "name": "MarketVaR-GARCH", "risk": "High"},
                {"id": "MDL-005", "name": "ChurnPredict-RF", "risk": "Medium"},
            ],
        },
    )

    clock = ClockControl(mode="frozen", frozen_at="2026-01-15T10:00:00")

    return EnvironmentSimulator(
        tool_specs=[ticket_spec, query_spec, approve_spec, notify_spec],
        fixtures=[fixture],
        clock=clock,
    )


def create_multi_turn_script():
    """Create a sample multi-turn interaction script for testing.

    Returns an InteractionScript with 3 turns testing:
    - Turn 1: Ask about a model (expects data_lookup tool)
    - Turn 2: Follow-up about risk (expects risk_check tool)
    - Turn 3: Create a ticket (expects create_ticket tool)
    """
    from knowlytix.harness.doe.multiturn import InteractionScript, TurnSpec

    return InteractionScript(
        script_id="mt_model_review",
        name="Model Review Workflow",
        description="Multi-turn model risk assessment workflow",
        turns=[
            TurnSpec(
                turn_id="turn_1",
                user_message=(
                    "<report>Model MDL-001 CreditScore-XGB has test AUC 0.8412"
                    " and is rated High risk with Degraded status.</report>\n\n"
                    "Answer the following question using the report and any "
                    "tools available to you.\n\n"
                    "What is the test AUC of MDL-001?"
                ),
                expected_contains=["0.8412"],
                expected_tool_calls=["data_lookup"],
            ),
            TurnSpec(
                turn_id="turn_2",
                user_message=(
                    "Answer the following question using the report and any "
                    "tools available to you.\n\n"
                    "What is the risk rating of MDL-001?"
                ),
                expected_contains=["High"],
                expected_tool_calls=["risk_check"],
            ),
            TurnSpec(
                turn_id="turn_3",
                user_message=(
                    "Answer the following question using the report and any "
                    "tools available to you.\n\n"
                    "Create a ticket for the MDL-001 degradation finding"
                ),
                expected_tool_calls=["create_ticket"],
            ),
        ],
        setup_context="Model risk assessment session for MDL-001 CreditScore-XGB",
        teardown_assertions=[
            {"type": "min_turns_passed", "value": 2},
        ],
        tags=["model_review", "multi_turn"],
    )
