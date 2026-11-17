# SPDX-License-Identifier: Apache-2.0
"""Mock agentic system for GMSH harness testing.

Provides a deterministic mock agent that simulates tool-using agentic
behavior WITHOUT requiring LLM API calls. The mock agent:

1. Registers realistic tools (lookup, query, calculate, risk_check)
2. Answers questions by parsing the context for relevant information
3. Makes tool calls that are captured by the ToolGateway and PolicyEngine
4. Exhibits configurable recovery behavior when tools are blocked
5. Can be used as both an evaluator function and via AgentAdapter

Usage::

    from knowlytix.harness.fixtures import MockHarness, create_mock_evaluator

    # As an evaluator callable
    evaluator = create_mock_evaluator()
    answer = evaluator("What is the test AUC of MDL-001?", context)

    # As a Harness for AgentAdapter testing
    harness = MockHarness()
    adapter = AgentAdapter(harness, gateway=my_gateway)
    trace = adapter.evaluate("question", "context")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["MockHarness", "create_mock_evaluator", "MOCK_TOOLS"]


# ---------------------------------------------------------------------------
# Mock tool definitions
# ---------------------------------------------------------------------------

MOCK_TOOLS = {
    "data_lookup": {
        "description": "Look up a specific value from the report data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity to look up"},
                "field": {"type": "string", "description": "Field name"},
            },
            "required": ["entity"],
        },
        "category": "data",
    },
    "risk_check": {
        "description": "Check the risk rating and status of a model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model identifier"},
            },
            "required": ["model_id"],
        },
        "category": "compliance",
    },
    "calculate": {
        "description": "Perform a numeric calculation on report data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
                "context_values": {"type": "object", "description": "Named values"},
            },
            "required": ["expression"],
        },
        "category": "analysis",
    },
    "submit_escalation": {
        "description": "Escalate a finding for human review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "finding": {"type": "string", "description": "Finding to escalate"},
                "severity": {"type": "string", "description": "low, medium, high"},
            },
            "required": ["finding"],
        },
        "category": "compliance",
    },
    "recall_memory": {
        "description": "Query the geometric memory system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query string"},
            },
            "required": ["query"],
        },
        "category": "memory",
    },
}

# Model data extracted from the test document for deterministic answers
_MODEL_DATA = {
    "MDL-001": {
        "name": "CreditScore-XGB", "domain": "Credit Decisioning",
        "risk": "High", "status": "Degraded", "algorithm": "XGBoost (depth=6)",
        "features": 142, "train_auc": 0.8934, "test_auc": 0.8412,
        "gini": 0.6824, "ks": 0.4891, "brier": 0.1523,
    },
    "MDL-002": {
        "name": "FraudNet-LSTM", "domain": "Fraud Detection",
        "risk": "High", "status": "Watch", "algorithm": "LSTM (3-layer)",
        "features": 87, "train_auc": 0.9567, "test_auc": 0.9234,
        "gini": 0.8468, "ks": 0.6712, "brier": 0.0834,
    },
    "MDL-003": {
        "name": "AML-GBM", "domain": "Anti-Money Laundering",
        "risk": "High", "status": "Degraded", "algorithm": "Gradient Boosting",
        "features": 234, "train_auc": 0.9123, "test_auc": 0.8756,
        "gini": 0.7512, "ks": 0.5234, "brier": 0.1245,
    },
    "MDL-004": {
        "name": "MarketVaR-GARCH", "domain": "Market Risk VaR",
        "risk": "High", "status": "Stable", "algorithm": "GARCH(1,1)",
        "features": 15, "train_auc": 0.8234, "test_auc": 0.8012,
        "gini": 0.6024, "ks": 0.3891, "brier": 0.1876,
    },
    "MDL-005": {
        "name": "ChurnPredict-RF", "domain": "Customer Analytics",
        "risk": "Medium", "status": "Degraded", "algorithm": "Random Forest",
        "features": 53, "train_auc": 0.7891, "test_auc": 0.7234,
        "gini": 0.4468, "ks": 0.3123, "brier": 0.2134,
    },
}


# ---------------------------------------------------------------------------
# Mock tool handlers
# ---------------------------------------------------------------------------

def _handle_data_lookup(args: dict, context: str) -> tuple[str, dict | None]:
    """Look up entity data from the built-in model database."""
    entity = args.get("entity", "").upper()
    field_name = args.get("field", "")

    for model_id, data in _MODEL_DATA.items():
        if model_id in entity or data["name"].lower() in entity.lower():
            if field_name and field_name in data:
                return f"{field_name}: {data[field_name]}", data
            return str(data), data

    # Fall back to context search
    if entity and context:
        for line in context.split("\n"):
            if entity.lower() in line.lower():
                return line.strip(), {"source": "context", "line": line}

    return f"No data found for entity: {entity}", None


def _handle_risk_check(args: dict, context: str) -> tuple[str, dict | None]:
    """Check risk rating for a model."""
    model_id = args.get("model_id", "").upper()
    if not model_id.startswith("MDL-"):
        model_id = f"MDL-{model_id}"

    data = _MODEL_DATA.get(model_id)
    if data:
        result = {
            "model_id": model_id,
            "name": data["name"],
            "risk_rating": data["risk"],
            "status": data["status"],
            "domain": data["domain"],
        }
        return (
            f"Model {model_id} ({data['name']}): "
            f"Risk={data['risk']}, Status={data['status']}, "
            f"Domain={data['domain']}",
            result,
        )
    return f"Model {model_id} not found in assessment", None


def _handle_calculate(args: dict, context: str) -> tuple[str, dict | None]:
    """Evaluate a simple arithmetic expression."""
    expr = args.get("expression", "")
    ctx_vals = args.get("context_values", {})

    # Simple safe evaluation of arithmetic
    namespace = {"__builtins__": {}}
    namespace.update({k: float(v) for k, v in ctx_vals.items()
                      if isinstance(v, (int, float, str))})
    try:
        result = eval(expr, namespace)
        return f"Result: {result}", {"expression": expr, "result": result}
    except Exception as e:
        return f"Calculation error: {e}", None


def _handle_escalation(args: dict, context: str) -> tuple[str, dict | None]:
    """Submit an escalation (mock — just records it)."""
    finding = args.get("finding", "")
    severity = args.get("severity", "medium")
    return (
        f"Escalation submitted: [{severity.upper()}] {finding}",
        {"finding": finding, "severity": severity, "submitted": True},
    )


def _handle_recall(args: dict, context: str) -> tuple[str, dict | None]:
    """Mock memory recall — returns a summary from context."""
    query = args.get("query", "")
    matches = []
    for line in context.split("\n"):
        if any(w.lower() in line.lower() for w in query.split() if len(w) > 3):
            matches.append(line.strip())
    if matches:
        return "\n".join(matches[:5]), {"matches": len(matches)}
    return f"No relevant memory found for: {query}", None


_HANDLERS = {
    "data_lookup": _handle_data_lookup,
    "risk_check": _handle_risk_check,
    "calculate": _handle_calculate,
    "submit_escalation": _handle_escalation,
    "recall_memory": _handle_recall,
}


# ---------------------------------------------------------------------------
# Mock Harness — simulates agentic tool-use without LLM API
# ---------------------------------------------------------------------------

@dataclass
class _MockHarnessConfig:
    model: str = "mock-agent-v1"
    max_turns: int = 5
    device: str = ""


@dataclass
class _MockHooks:
    on_tool_call: Callable | None = None
    on_tool_result: Callable | None = None
    on_learn: Callable | None = None
    on_turn: Callable | None = None
    before_tool: Callable | None = None
    intercept_tool: Callable | None = None


class MockHarness:
    """A mock Harness that simulates tool-using agent behavior.

    Compatible with AgentAdapter and ToolGateway — exposes the same
    hook interface as the real Harness without requiring LLM API calls.

    The agent follows a deterministic script:
    1. Parse the question to identify what's being asked
    2. Make tool calls based on question patterns
    3. Compose an answer from tool results
    4. Handle blocked tools via recovery/fallback
    """

    def __init__(
        self,
        tools: dict[str, dict] | None = None,
        recovery_mode: str = "retry",
        deliberate_errors: bool = False,
    ):
        self.config = _MockHarnessConfig()
        self.hooks = _MockHooks()
        self.messages: list[dict] = []
        self._turn = 0
        self._context = ""

        # Configurable behavior
        self._recovery_mode = recovery_mode  # "retry", "fallback", "escalate"
        self._deliberate_errors = deliberate_errors

        # Tool registry (simplified — just needs the handler map)
        self._tools = dict(MOCK_TOOLS)
        if tools:
            self._tools.update(tools)

        self._tool_records: list[dict] = []

    def _execute_tool(self, name: str, args: dict) -> tuple[str, Any]:
        """Execute a tool through the hook pipeline."""
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

        # Execute
        handler = _HANDLERS.get(name)
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

    def _plan_tool_calls(self, prompt: str) -> list[tuple[str, dict]]:
        """Decide which tools to call based on the question content."""
        prompt_lower = prompt.lower()
        calls: list[tuple[str, dict]] = []

        # Pattern 1: Model-specific questions → data_lookup + risk_check
        model_match = re.search(r"(MDL-\d{3})", prompt, re.IGNORECASE)
        if model_match:
            model_id = model_match.group(1).upper()
            calls.append(("data_lookup", {"entity": model_id}))
            calls.append(("risk_check", {"model_id": model_id}))

        # Pattern 2: Numeric/metric questions → data_lookup + calculate
        elif any(w in prompt_lower for w in [
            "auc", "gini", "brier", "ks statistic", "accuracy",
            "precision", "features", "training samples",
        ]):
            # Try to extract entity
            entity = "MDL-001"  # default
            for mid in _MODEL_DATA:
                if mid.lower() in prompt_lower:
                    entity = mid
                    break
                name = _MODEL_DATA[mid]["name"].lower()
                if name in prompt_lower:
                    entity = mid
                    break
            calls.append(("data_lookup", {"entity": entity}))

        # Pattern 3: Risk/compliance questions → risk_check
        elif any(w in prompt_lower for w in [
            "risk", "compliance", "degraded", "high-risk", "status",
        ]):
            calls.append(("risk_check", {"model_id": "MDL-001"}))

        # Pattern 4: Questions requiring calculation
        elif any(w in prompt_lower for w in [
            "difference", "gap", "ratio", "average", "total", "how many",
            "count",
        ]):
            calls.append(("recall_memory", {"query": prompt[:100]}))
            calls.append(("calculate", {
                "expression": "1 + 1",
                "context_values": {},
            }))

        # Pattern 5: Escalation scenarios (high-risk findings)
        elif any(w in prompt_lower for w in [
            "escalat", "approve", "review", "override",
        ]):
            calls.append(("submit_escalation", {
                "finding": prompt[:200],
                "severity": "high",
            }))

        # Default: memory recall
        if not calls:
            calls.append(("recall_memory", {"query": prompt[:100]}))

        return calls

    def _recover_from_blocked(
        self, tool_name: str, args: dict, prompt: str,
    ) -> str:
        """Handle blocked tool calls based on recovery mode."""
        if self._recovery_mode == "escalate":
            result, _ = self._execute_tool("submit_escalation", {
                "finding": f"Tool {tool_name} was blocked during: {prompt[:100]}",
                "severity": "high",
            })
            return f"Escalated blocked tool '{tool_name}'. {result}"

        elif self._recovery_mode == "fallback":
            # Try memory recall as fallback
            result, _ = self._execute_tool("recall_memory", {
                "query": prompt[:100],
            })
            return f"Tool '{tool_name}' blocked; falling back to memory. {result}"

        else:  # "retry" — just note it
            return f"Tool '{tool_name}' was blocked."

    def run(self, prompt: str) -> str:
        """Run the mock agent loop.

        Simulates the Harness.run() interface:
        1. Plan tool calls from question
        2. Execute tools through hooks
        3. Handle blocked tools via recovery
        4. Compose answer from results
        """
        self._tool_records = []

        # Extract context from prompt
        ctx_match = re.search(
            r"<report>\s*(.*?)\s*</report>", prompt, re.DOTALL)
        if ctx_match:
            self._context = ctx_match.group(1)
        else:
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

        # Compose final answer
        answer = self._compose_answer(question, tool_results)
        return answer

    def _compose_answer(self, question: str, tool_results: list[str]) -> str:
        """Compose a natural-language answer from tool results."""
        question_lower = question.lower()

        # Try to extract a direct answer from tool results
        for result in tool_results:
            # Numeric extraction
            if any(w in question_lower for w in [
                "auc", "gini", "brier", "ks", "score", "value",
            ]):
                numbers = re.findall(r"\d+\.\d+", result)
                if numbers:
                    # Return the most relevant number
                    for key in ["test_auc", "gini", "ks", "brier", "train_auc"]:
                        if key.replace("_", " ") in question_lower or \
                           key.replace("_", "") in question_lower:
                            for num in numbers:
                                return num
                    return numbers[0]

            # Boolean questions
            if question_lower.startswith(("is ", "does ", "are ", "has ")):
                if "High" in result or "Degraded" in result:
                    return "True"
                if "Low" in result or "Stable" in result:
                    return "False"

            # String extraction for name/status/domain
            if "risk" in question_lower and "Risk=" in result:
                match = re.search(r"Risk=(\w+)", result)
                if match:
                    return match.group(1)

            if "status" in question_lower and "Status=" in result:
                match = re.search(r"Status=(\w+)", result)
                if match:
                    return match.group(1)

            if "domain" in question_lower and "Domain=" in result:
                match = re.search(r"Domain=([\w\s]+?)(?:,|$)", result)
                if match:
                    return match.group(1).strip()

            # Count questions
            if "how many" in question_lower:
                numbers = re.findall(r"\b(\d+)\b", result)
                if numbers:
                    return numbers[0]

        # Fallback: return the most informative tool result
        if tool_results:
            longest = max(tool_results, key=len)
            return longest[:500]

        return "Unable to determine the answer from available data."


# ---------------------------------------------------------------------------
# Evaluator factory
# ---------------------------------------------------------------------------

def create_mock_evaluator(
    recovery_mode: str = "retry",
    deliberate_errors: bool = False,
) -> Callable[[str, str], str]:
    """Create a mock evaluator function compatible with DOEGMSBenchmark.run().

    Returns a callable ``(question, context) -> answer`` that simulates
    agentic behavior with tool calls and recovery logic.
    """
    harness = MockHarness(
        recovery_mode=recovery_mode,
        deliberate_errors=deliberate_errors,
    )

    def evaluator(question: str, context: str) -> str:
        harness.messages = []
        harness._turn = 0
        prompt = (
            f"You have been given the following report as context:\n\n"
            f"<report>\n{context}\n</report>\n\n"
            f"Answer the following question using the report and any tools "
            f"available to you.\n\n{question}"
        )
        return harness.run(prompt)

    evaluator.__repr__ = lambda: "MockAgentEvaluator()"
    return evaluator
