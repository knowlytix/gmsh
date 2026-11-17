# SPDX-License-Identifier: Apache-2.0
"""Harness: standalone agent orchestration layer.

Owns the full loop: prompt → Claude → tool calls → results → memory → back.
Replaces Claude Code as the harness — your code controls execution.

Architecture:
    ┌─────────────────────────────────────────┐
    │              Harness                     │
    │  ┌───────────┐  ┌───────────────────┐   │
    │  │ToolRegistry│  │  GMS Memory       │   │
    │  │ gms_*     │  │  session store    │   │
    │  │ modeva_*  │  │  modeva store     │   │
    │  │ custom    │  │  runtime learner  │   │
    │  └───────────┘  └───────────────────┘   │
    │  ┌───────────┐  ┌───────────────────┐   │
    │  │ ReAct Loop │  │  Policy Verifier  │   │
    │  │ any LLM    │  │  LogicVerifier    │   │
    │  └───────────┘  └───────────────────┘   │
    └─────────────────────────────────────────┘

The loop runs in one of two modes:
  * Native tool-calling over the Anthropic SDK (default — no ``backend`` given).
  * Provider-agnostic text protocol over an injected ``LLMBackend`` (any LLM,
    including open-weight endpoints without native tool-calling).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from knowlytix.knowledge.llm_backend import LLMBackend

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.learn import RuntimeLearner
from knowlytix.knowledge.store import GMSExpertStore

from ._modeva_tools import SessionState
from .memory import (
    bootstrap_session_store,
    extract_triples_from_result,
    gather_memory_context,
)
from .registry import ToolRegistry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    """Harness configuration."""
    # LLM
    model: str = "claude-opus-4-7"
    max_tokens: int = 4096
    max_turns: int = 30
    truncate_result_at: int = 8000
    # Text-protocol mode only: how many malformed replies to reprompt before
    # giving up on a turn.
    max_parse_retries: int = 2

    # GMS stores
    stores_dir: str = "gms_stores"
    session_store_path: str = "harness_session_store"
    modeva_store_path: str = ""
    gms_model_name: str = "claude-sonnet-4-20250514"

    # Device override (e.g., "cpu" when GPU is occupied by vLLM)
    device: str = ""

    # Memory
    auto_learn: bool = True
    memory_context_lines: int = 30

    # Policy
    enable_policy: bool = True
    policy_behaviors_dir: str = ""

    # Context management
    max_message_tokens: int = 180_000   # compress beyond this
    context_summary_model: str = ""     # if set, use this model for summarization


# ---------------------------------------------------------------------------
# Callback hooks (optional)
# ---------------------------------------------------------------------------

@dataclass
class HarnessHooks:
    """Optional hooks for observability and control."""
    on_tool_call: Callable[[str, dict], None] | None = None
    on_tool_result: Callable[[str, str, Any], None] | None = None
    on_learn: Callable[[int, str], None] | None = None
    on_turn: Callable[[int, str], None] | None = None
    before_tool: Callable[[str, dict], bool] | None = None  # return False to skip
    # intercept_tool: return (True, response_str) to bypass real execution,
    # or (False, None) to let the registry handle the call.
    intercept_tool: Callable[[str, dict], tuple[bool, str | None]] | None = None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Harness:
    """Standalone orchestration layer: owns the ReAct loop, tool routing,
    GMS memory, and policy verification.

    Usage (native Anthropic tool-calling):
        h = Harness(HarnessConfig(stores_dir="gms_stores"))
        h.register_gms_tools()
        h.register_modeva_tools()
        response = h.run("Load BikeSharing and train XGB depth-2")

    Usage (provider-agnostic, any LLM via an injected backend):
        backend = build_backend(agent_llm_config)   # an LLMBackend
        h = Harness(HarnessConfig(model=backend.model_name), backend=backend)
        h.register_modeva_tools()
        response = h.run("Load BikeSharing and train XGB depth-2")
    """

    def __init__(self, config: HarnessConfig | None = None,
                 hooks: HarnessHooks | None = None,
                 backend: "LLMBackend | None" = None,
                 control_plane: Any = None):
        self.config = config or HarnessConfig()
        self.hooks = hooks or HarnessHooks()
        # Optional control plane: a domain object that governs execution against
        # an ingested process spec. Duck-typed (kept generic here): it may expose
        #   check_precondition(tool_name, tool_args, state) -> (ok: bool, reason)
        #   review_result(tool_name, tool_args, raw_result, state) -> str | None
        #   outstanding_obligations(state, user_request) -> list[str]
        # None -> unchanged, ungoverned behavior.
        self.control_plane = control_plane
        self.device = torch.device(self.config.device if self.config.device else
                                    ("cuda" if torch.cuda.is_available() else "cpu"))

        # LLM path. When a backend is injected the loop is provider-agnostic
        # (text protocol); otherwise it falls back to the Anthropic SDK with
        # native tool-calling.
        self.backend = backend
        self.client = None
        if backend is None:
            import anthropic
            self.client = anthropic.Anthropic()

        # Tool registry
        self.registry = ToolRegistry()

        # MoDeva session state (live DataSet/ModelZoo/Model objects)
        self.state = SessionState()

        # Conversation history
        self.messages: list[dict] = []

        # GMS memory stores
        self._session_store: GMSExpertStore | None = None
        self._modeva_store: GMSExpertStore | None = None
        self._learner: RuntimeLearner | None = None
        self._policy_verifier = None

        # Turn counter
        self._turn = 0

        self._init_memory()

        if self.backend is not None:
            print(f"  [Harness] Backend: {self.backend.model_name} "
                  f"(text-protocol loop)")
        else:
            print(f"  [Harness] Model: {self.config.model} (native tool-calling)")
        print(f"  [Harness] Device: {self.device}")
        print("  [Harness] Ready.")

    # ----- Setup -----

    def _init_memory(self):
        """Initialize GMS session store and optional MoDeva store."""
        cfg = self.config

        # Session store (grows with each interaction)
        session_config = DocGMSConfig(store_path=cfg.session_store_path)
        self._session_store = GMSExpertStore(session_config, self.device)
        if not self._session_store.load():
            print("  [Harness] Bootstrapping session store...")
            bootstrap_session_store(self._session_store, session_config, self.device)
        self._learner = RuntimeLearner(self._session_store, session_config.learn)

        session_stats = self._session_store.stats()
        print(f"  [Harness] Session GMS: {session_stats['entities']} entities, "
              f"{session_stats['triples']} triples")

        # MoDeva store (static, read-only reference knowledge)
        if cfg.modeva_store_path and os.path.isdir(cfg.modeva_store_path):
            modeva_config = DocGMSConfig(store_path=cfg.modeva_store_path)
            self._modeva_store = GMSExpertStore(modeva_config, self.device)
            if self._modeva_store.load():
                stats = self._modeva_store.stats()
                print(f"  [Harness] MoDeva GMS: {stats['entities']} entities, "
                      f"{stats['triples']} triples")
            else:
                self._modeva_store = None

    def _init_policy(self):
        """Initialize policy verifier from domain GMS if available."""
        if not self.config.enable_policy:
            return
        if self._session_store is None:
            return
        try:
            from knowlytix.knowledge.config import VerifyConfig

            from ._verifier import LogicVerifier
            self._policy_verifier = LogicVerifier(self._session_store, VerifyConfig())
        except Exception:
            pass

    def register_gms_tools(self, stores_dir: str | None = None):
        """Register all GMS expert system tools."""
        sd = stores_dir or self.config.stores_dir
        self.registry.register_gms_tools(sd, self.device, self.config.gms_model_name)

    def register_modeva_tools(self):
        """Register all MoDeva tools."""
        self.registry.register_modeva_tools(self.state)

    def register_tool(self, name: str, description: str, input_schema: dict,
                      handler: Callable, category: str = "custom"):
        """Register a custom tool."""
        self.registry.register_fn(name, description, input_schema, handler, category)

    # ----- System prompt -----

    def _build_system_prompt(self, user_request: str) -> str:
        """Build system prompt with injected GMS memory context."""
        dataset_name = self._current_dataset_name()
        model_names = self._current_model_names()

        modeva_ctx, session_ctx = gather_memory_context(
            self._modeva_store, self._session_store,
            dataset_name, model_names, user_request,
        )

        parts = [self._base_system_prompt()]

        if modeva_ctx:
            parts.append(
                f"\n## MoDeva Knowledge (from Geometric Memory)\n"
                f"```\n{modeva_ctx}\n```"
            )
        if session_ctx:
            parts.append(
                f"\n## Session Memory (prior findings)\n"
                f"```\n{session_ctx}\n```"
            )

        # List available tool categories
        categories = set()
        for t in self.registry._tools.values():
            categories.add(t.category)
        if categories:
            parts.append(f"\n## Available Tool Categories: {', '.join(sorted(categories))}")

        return "\n".join(parts)

    def _base_system_prompt(self) -> str:
        return """\
You are an AI agent powered by a Geometric Memory System (GMS) with access \
to data science, knowledge management, and custom tools.

## Your Capabilities
- GMS tools: create, ingest, query, search, lookup expert systems
- MoDeva tools: load data, train models, diagnose, explain, interpret
- Custom tools: any additional tools registered by the user

## Principles
- Report exact metric values — no vague qualitative statements.
- Flag issues: overfitting (large train-test gap), fairness violations (AIR < 0.8), weak regions.
- For FANOVA interpretation, tree depth MUST be 2.
- Suggest next steps based on findings.
- Use recall_memory when you need to check prior findings or methodology.

## Memory
You have a geometric memory system that accumulates knowledge across sessions. \
Verified facts from your interactions are merged into the store. \
Use it to remember past experiments and methodology."""

    def _build_text_system_prompt(self, user_request: str) -> str:
        """System prompt for the text-protocol loop: base guidance + injected
        GMS memory + the full tool menu + the ReAct response protocol."""
        parts = [self._base_system_prompt()]

        if self._session_store is not None:
            modeva_ctx, session_ctx = gather_memory_context(
                self._modeva_store, self._session_store,
                self._current_dataset_name(), self._current_model_names(),
                user_request,
            )
            if modeva_ctx:
                parts.append(
                    f"\n## MoDeva Knowledge (from Geometric Memory)\n"
                    f"```\n{modeva_ctx}\n```"
                )
            if session_ctx:
                parts.append(
                    f"\n## Session Memory (prior findings)\n"
                    f"```\n{session_ctx}\n```"
                )

        parts.append("\n## Available Tools\n" + self._render_tool_menu())
        parts.append("\n" + _REACT_PROTOCOL)
        return "\n".join(parts)

    def _render_tool_menu(self) -> str:
        """One line per registered tool: name, category, description, params."""
        lines = []
        for t in self.registry._tools.values():
            schema = t.input_schema or {}
            props = schema.get("properties", {}) or {}
            required = set(schema.get("required", []) or [])
            params = []
            for pname, pspec in props.items():
                tag = pname + ("*" if pname in required else "")
                if isinstance(pspec, dict) and pspec.get("enum"):
                    tag += f" in {pspec['enum']}"
                params.append(tag)
            pstr = ", ".join(params) if params else "(none)"
            lines.append(f"- {t.name} [{t.category}]: {t.description}\n"
                         f"    params: {pstr}")
        return "\n".join(lines)

    # ----- Recall memory (special tool) -----

    def _handle_recall_memory(self, args: dict) -> str:
        """Handle the recall_memory meta-tool."""
        query = args.get("query", "")
        dataset_name = self._current_dataset_name()
        model_names = self._current_model_names()

        m_ctx, s_ctx = gather_memory_context(
            self._modeva_store, self._session_store,
            dataset_name, model_names, query,
        )
        result = ""
        if m_ctx:
            result += f"MODEVA KNOWLEDGE:\n{m_ctx}\n\n"
        if s_ctx:
            result += f"SESSION MEMORY:\n{s_ctx}\n\n"
        return result or "No relevant knowledge found in memory."

    # ----- Learning -----

    def _learn_from_result(self, tool_name: str, args: dict, raw_result: Any):
        """Extract triples from tool result and learn into session GMS."""
        if not self.config.auto_learn or self._learner is None or raw_result is None:
            return

        dataset_name = args.get("dataset_name", self._current_dataset_name())
        model_name = args.get("model_name")

        triples = extract_triples_from_result(raw_result, dataset_name, model_name)
        if not triples:
            return

        try:
            lr = self._learner.learn_triples(triples, self.device)
            if lr is not None and lr.accepted:
                n = lr.n_triples_added
                print(f"  [Memory] Learned {n} triples from {tool_name}")
                if self.hooks.on_learn:
                    self.hooks.on_learn(n, tool_name)
            elif lr is not None and lr.reason:
                print(f"  [Memory] Rejected: {lr.reason[:80]}")
        except Exception as e:
            print(f"  [Memory] Learning error: {e}")

    # ----- Policy verification -----

    def _verify_tool_call(self, tool_name: str, args: dict) -> bool:
        """Check if a tool call is policy-allowed. Returns True if OK."""
        if self._policy_verifier is None:
            return True

        # Check if the tool call is geometrically plausible
        context = self._current_dataset_name() or "agent"
        result = self._policy_verifier.verify_action(context, tool_name)
        if not result["plausible"] and result["score"] is not None:
            print(f"  [Policy] Tool '{tool_name}' flagged as implausible "
                  f"(score={result['score']:.4f})")
            # Log but don't block — the agent may have valid reasons
        return True

    def _obligations_block(self, user_request: str) -> list[str]:
        """Control-plane obligations that are not yet satisfied. A non-empty
        list means the agent may not finish; the loop reports these back and
        continues instead of returning the final answer."""
        if self.control_plane is None:
            return []
        try:
            return list(self.control_plane.outstanding_obligations(
                self.state, user_request) or [])
        except Exception as e:
            print(f"  [ControlPlane] obligation check error: {e}")
            return []

    # ----- Context management -----

    def _estimate_message_tokens(self) -> int:
        """Rough token estimate for conversation history."""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(json.dumps(block)) // 4
                    elif hasattr(block, "text"):
                        total += len(getattr(block, "text", "")) // 4
                    else:
                        total += 100  # estimate for tool_use blocks
        return total

    def _compress_context(self):
        """Drop older messages if context is too large, keeping system coherence."""
        est = self._estimate_message_tokens()
        if est <= self.config.max_message_tokens:
            return

        # Keep first user message + last N exchanges
        if len(self.messages) <= 4:
            return

        # Remove middle messages, keeping first and last 4
        kept = self.messages[:1] + self.messages[-4:]
        summary = (
            f"[Context compressed: {len(self.messages)} messages → {len(kept)}. "
            f"Estimated {est} tokens → ~{self._estimate_message_tokens()}]"
        )
        self.messages = [
            {"role": "user", "content": summary},
        ] + kept
        print(f"  [Harness] Context compressed: {est} → {self._estimate_message_tokens()} est. tokens")

    # ----- Main loop -----

    def run(self, user_request: str) -> str:
        """Process a user request through the full agent loop.

        Dispatches to the provider-agnostic text-protocol loop when a
        ``backend`` was injected, otherwise to the native Anthropic loop.
        Both build a system prompt with GMS memory context, execute tool
        calls under the same hooks/policy/learning path, and return the
        final response text.
        """
        self._turn = 0

        # Register recall_memory if not already there
        if not self.registry.get("recall_memory"):
            self.registry.register_fn(
                "recall_memory",
                "Query the geometric memory system for relevant knowledge.",
                {"type": "object", "properties": {
                    "query": {"type": "string", "description": "Question for memory"},
                }, "required": ["query"]},
                lambda args, ctx: (self._handle_recall_memory(args), None),
                category="memory",
            )

        # Initialize policy if not done
        if self._policy_verifier is None:
            self._init_policy()

        if self.backend is not None:
            return self._run_text(user_request)
        return self._run_native(user_request)

    def _dispatch_tool(self, tool_name: str, tool_args: dict) -> str:
        """Run one tool call through hooks, policy, memory learning and
        truncation, returning the result string shown back to the model.

        Shared by both the native and text-protocol loops so a single code
        path governs execution regardless of the LLM.
        """
        print(f"  [Tool] {tool_name}({_summarize_args(tool_args)})")

        if self.hooks.on_tool_call:
            self.hooks.on_tool_call(tool_name, tool_args)

        # Hook: before_tool can skip execution
        if self.hooks.before_tool and not self.hooks.before_tool(tool_name, tool_args):
            return "Tool call skipped by policy hook."

        # Policy check (log-only)
        self._verify_tool_call(tool_name, tool_args)

        # Control-plane precondition gate: refuse a premature / out-of-order
        # call before it executes (e.g. diagnosing an unfitted model).
        if self.control_plane is not None:
            try:
                ok, reason = self.control_plane.check_precondition(
                    tool_name, tool_args, self.state)
            except Exception as e:               # never let the gate crash a run
                ok, reason = True, ""
                print(f"  [ControlPlane] precondition check error: {e}")
            if not ok:
                msg = f"BLOCKED by control plane: {reason}"
                print(f"  [ControlPlane] {msg}")
                if self.hooks.on_tool_result:
                    self.hooks.on_tool_result(tool_name, msg, None)
                return msg

        # Special handling for recall_memory
        if tool_name == "recall_memory":
            return self._handle_recall_memory(tool_args)

        # Hook: intercept_tool can bypass real execution
        # (used by EnvironmentSimulator for mock tool endpoints)
        if self.hooks.intercept_tool:
            intercepted, intercept_response = self.hooks.intercept_tool(
                tool_name, tool_args,
            )
            if intercepted and intercept_response is not None:
                if self.hooks.on_tool_result:
                    self.hooks.on_tool_result(tool_name, intercept_response, None)
                return intercept_response

        # Execute via registry
        result_str, raw_result = self.registry.execute(
            tool_name, tool_args, context=self.state,
        )

        # Learn from result
        self._learn_from_result(tool_name, tool_args, raw_result)

        # Truncate long results
        if len(result_str) > self.config.truncate_result_at:
            result_str = result_str[:self.config.truncate_result_at] + "\n... [truncated]"

        # Control-plane outcome check (non-blocking; records evidence).
        if self.control_plane is not None:
            try:
                note = self.control_plane.review_result(
                    tool_name, tool_args, raw_result, self.state)
            except Exception as e:
                note = None
                print(f"  [ControlPlane] result review error: {e}")
            if note:
                result_str += f"\n[control-plane check] {note}"
                print(f"  [ControlPlane] {note}")

        if self.hooks.on_tool_result:
            self.hooks.on_tool_result(tool_name, result_str, raw_result)

        return result_str

    def _run_native(self, user_request: str) -> str:
        """ReAct loop over the Anthropic SDK with native tool-calling."""
        system = self._build_system_prompt(user_request)
        tools = self.registry.to_anthropic_tools()

        self.messages.append({"role": "user", "content": user_request})

        for turn in range(self.config.max_turns):
            self._turn = turn + 1

            if self.hooks.on_turn:
                self.hooks.on_turn(self._turn, "start")

            # Compress context if needed
            self._compress_context()

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system,
                tools=tools,
                messages=self.messages,
            )

            assistant_content = response.content
            self.messages.append({"role": "assistant", "content": assistant_content})

            # Done?
            if response.stop_reason == "end_turn":
                final_text = "".join(
                    b.text for b in assistant_content if hasattr(b, "text")
                )
                blocked = self._obligations_block(user_request)
                if blocked:
                    print(f"  [ControlPlane] finish blocked — {len(blocked)} "
                          f"obligation(s) outstanding")
                    self.messages.append({"role": "user", "content": (
                        "Cannot finish yet — the required backbone is incomplete. "
                        "Outstanding obligations:\n- " + "\n- ".join(blocked)
                        + "\nComplete these, then finish.")})
                    system = self._build_system_prompt(user_request)
                    continue
                return final_text

            # Process tool calls
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in assistant_content:
                    if block.type != "tool_use":
                        continue
                    content = self._dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    })

                self.messages.append({"role": "user", "content": tool_results})

                # Refresh system prompt with updated memory
                system = self._build_system_prompt(user_request)

        return "[Harness] Reached maximum tool call turns."

    def _run_text(self, user_request: str) -> str:
        """Provider-agnostic ReAct loop over an injected ``LLMBackend``.

        Drives a text protocol (THOUGHT / ACTION+PARAMS / FINAL) instead of
        native tool-calling, so it works with any text LLM — including
        open-weight endpoints served without a tool-call parser.
        """
        system = self._build_text_system_prompt(user_request)
        transcript = [f"USER REQUEST:\n{user_request}"]
        retries = 0

        for turn in range(self.config.max_turns):
            self._turn = turn + 1

            if self.hooks.on_turn:
                self.hooks.on_turn(self._turn, "start")

            user_prompt = "\n\n".join(transcript) + "\n\n" + _REACT_REMINDER
            text = self.backend.call(system, user_prompt, self.config.max_tokens)

            kind, first, second = _parse_react_action(text)

            if kind == "final":
                blocked = self._obligations_block(user_request)
                if blocked:
                    print(f"  [ControlPlane] finish blocked — {len(blocked)} "
                          f"obligation(s) outstanding")
                    transcript.append(f"ASSISTANT:\n{text}")
                    transcript.append(
                        "OBSERVATION:\nCannot finish yet — the required backbone "
                        "is incomplete. Outstanding obligations:\n- "
                        + "\n- ".join(blocked)
                        + "\nComplete these, then produce your FINAL answer.")
                    system = self._build_text_system_prompt(user_request)
                    transcript = self._trim_transcript(transcript)
                    continue
                return first

            if kind == "error":
                retries += 1
                if retries > self.config.max_parse_retries:
                    return ("[Harness] No valid action after "
                            f"{retries} attempts. Last model output:\n{text[:500]}")
                transcript.append(f"ASSISTANT:\n{text}")
                transcript.append(
                    f"OBSERVATION:\nFORMAT ERROR — {first}. Reply with exactly "
                    "one ACTION+PARAMS block, or a FINAL block.")
                continue

            # kind == "action"
            retries = 0
            tool_name, tool_args = first, second
            result_str = self._dispatch_tool(tool_name, tool_args)

            transcript.append(f"ASSISTANT:\n{text}")
            transcript.append(f"OBSERVATION [{tool_name}]:\n{result_str}")

            # Refresh memory context and keep the transcript within budget.
            system = self._build_text_system_prompt(user_request)
            transcript = self._trim_transcript(transcript)

        return "[Harness] Reached maximum tool call turns."

    def _trim_transcript(self, transcript: list[str]) -> list[str]:
        """Keep the running transcript within the configured token budget
        (rough char/4 estimate), always retaining the original request and
        the most recent exchanges."""
        budget_chars = self.config.max_message_tokens * 4
        if sum(len(x) for x in transcript) <= budget_chars or len(transcript) <= 5:
            return transcript
        head = transcript[:1]
        tail = transcript[-4:]
        dropped = len(transcript) - len(head) - len(tail)
        note = [f"[transcript trimmed: {dropped} earlier step(s) omitted]"]
        return head + note + tail

    # ----- Multi-turn conversation -----

    def chat(self, message: str) -> str:
        """Continue the conversation (keeps message history)."""
        return self.run(message)

    def reset(self):
        """Clear conversation history (keeps memory and state)."""
        self.messages = []
        self._turn = 0

    # ----- Persistence -----

    def save(self):
        """Persist session GMS and flush background training."""
        if self._learner:
            self._learner.flush_training()
            self._learner.wait_for_training(timeout=60)
        if self._session_store:
            self._session_store.save()
            stats = self._session_store.stats()
            print(f"  [Harness] Session saved: {stats['entities']} entities, "
                  f"{stats['triples']} triples")

    # ----- Helpers -----

    def _current_dataset_name(self) -> str | None:
        names = list(self.state.datasets.keys())
        return names[-1] if names else None

    def _current_model_names(self) -> list[str] | None:
        names = list(self.state.trained_models.keys())
        return names if names else None

    @property
    def session_stats(self) -> dict:
        if self._session_store:
            return self._session_store.stats()
        return {}

    @property
    def tools(self) -> list[str]:
        return self.registry.list_names()


def _summarize_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 30:
            v = v[:30] + "..."
        parts.append(f"{k}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Text protocol (provider-agnostic ReAct over a plain text LLM)
# ---------------------------------------------------------------------------

_REACT_PROTOCOL = """\
## How to act
Respond with EXACTLY ONE of the following blocks, and nothing else.

To call a tool:
THOUGHT: <one or two sentences of reasoning>
ACTION: <exact tool name from the tool list above>
PARAMS: {<single-line JSON object of arguments; use {} if none>}

To finish and give the final answer:
THOUGHT: <one or two sentences>
FINAL: <your complete answer to the user's request>

Rules:
- Emit exactly one ACTION per response; never more than one.
- PARAMS must be a single valid JSON object on one line.
- Use only tool names listed above, spelled exactly.
- After each ACTION you receive an OBSERVATION with the result; use it to
  choose the next step.
- Emit FINAL only once you have enough information to answer."""

_REACT_REMINDER = (
    "Reply with one THOUGHT + ACTION + PARAMS block, or a THOUGHT + FINAL "
    "block. Nothing else."
)

_ACTION_RE = re.compile(r"ACTION:\s*(.+)")
_PARAMS_RE = re.compile(r"PARAMS:\s*(\{.*\})", re.DOTALL)
_FINAL_RE = re.compile(r"FINAL:\s*(.+)", re.DOTALL)


def _parse_react_action(text: str):
    """Parse a ReAct response.

    Returns one of:
      ("final", answer_str, None)
      ("action", tool_name, params_dict)
      ("error", reason_str, None)
    """
    t = (text or "").strip()
    if not t:
        return ("error", "empty model response", None)

    final_m = _FINAL_RE.search(t)
    action_m = _ACTION_RE.search(t)

    # Prefer whichever marker appears first, so a FINAL that trails an
    # unrelated word cannot pre-empt an actual ACTION and vice versa.
    if final_m and (action_m is None or final_m.start() < action_m.start()):
        return ("final", final_m.group(1).strip(), None)

    if action_m:
        tool = action_m.group(1).strip().split()[0].strip(" `\"'")
        if not tool:
            return ("error", "ACTION had no tool name", None)
        params: Any = {}
        params_m = _PARAMS_RE.search(t)
        if params_m:
            try:
                params = json.loads(params_m.group(1))
            except Exception:
                return ("error", "PARAMS was not valid JSON", None)
        if not isinstance(params, dict):
            return ("error", "PARAMS must be a JSON object", None)
        return ("action", tool, params)

    return ("error", "response had no ACTION or FINAL block", None)
