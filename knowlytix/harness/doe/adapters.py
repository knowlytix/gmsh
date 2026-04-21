# SPDX-License-Identifier: Apache-2.0
"""GMSH SUT Adapters — structured interface to systems under test.

Each adapter wraps a different SUT type and produces a ``SUTTrace``
that captures the raw output, tool calls, latency, and token count.
This trace feeds into the ClaimCompiler for ToolUseClaim extraction.

Adapter types:
    LLMAdapter   — direct Claude API call
    AgentAdapter — Harness agent loop with tool call instrumentation
    RAGAdapter   — retriever + generator pipeline
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic
    from knowlytix.harness.core import Harness


# ---------------------------------------------------------------------------
# Trace data models
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """Record of a single tool call during SUT execution."""

    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    result: str = ""
    latency_ms: float = 0.0


@dataclass
class SUTTrace:
    """Full execution trace from a system under test.

    Carries both the answer and the execution metadata needed for
    claim compilation and audit.
    """

    raw_output: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved_context: str = ""  # for RAG adapters
    error: str = ""


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class SUTAdapter(ABC):
    """Abstract base for System Under Test adapters.

    All adapters expose the same interface so they can be used
    interchangeably by the benchmark orchestrator.
    """

    @abstractmethod
    def evaluate(self, question: str, context: str) -> SUTTrace:
        """Run the SUT on a question with context and return a trace."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# LLMAdapter — direct API call
# ---------------------------------------------------------------------------

class LLMAdapter(SUTAdapter):
    """Wraps a direct Claude API call as a SUT adapter.

    Parameters
    ----------
    client : anthropic.Anthropic
        Anthropic API client.
    model : str
        Model to evaluate.
    """

    def __init__(self, client: "anthropic.Anthropic", model: str):
        self.client = client
        self.model = model

    def evaluate(self, question: str, context: str) -> SUTTrace:
        t0 = time.time()
        try:
            from knowlytix.bench.eval.llm_caller import SYSTEM_PROMPT
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"<report>\n{context}\n</report>\n\n"
                        f"{question}"
                    ),
                }],
            )
            raw = response.content[0].text
            elapsed = (time.time() - t0) * 1000
            # Extract token usage from API response
            in_tok = getattr(response.usage, "input_tokens", 0) or 0
            out_tok = getattr(response.usage, "output_tokens", 0) or 0
            return SUTTrace(
                raw_output=raw,
                latency_ms=elapsed,
                token_count=in_tok + out_tok,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return SUTTrace(
                raw_output="",
                latency_ms=elapsed,
                error=str(e),
            )

    def __repr__(self) -> str:
        return f"LLMAdapter(model={self.model!r})"


# ---------------------------------------------------------------------------
# AgentAdapter — Harness loop with tool instrumentation
# ---------------------------------------------------------------------------

class AgentAdapter(SUTAdapter):
    """Wraps the Harness agent loop with tool call instrumentation.

    Hooks into the Harness's callback system to capture every tool call
    as a ToolCallRecord, enabling ToolUseClaim extraction by the
    ClaimCompiler.

    Parameters
    ----------
    harness : Harness
        A configured Harness instance with tools registered.
    gateway : ToolGateway or None
        Optional tool gateway for runtime interception, fault injection,
        and policy enforcement.  Installed before each evaluation and
        uninstalled afterward.
    """

    def __init__(self, harness: "Harness", gateway=None):
        self.harness = harness
        self.gateway = gateway

    def evaluate(self, question: str, context: str) -> SUTTrace:
        tool_records: list[ToolCallRecord] = []

        # Install gateway if provided (adds before_tool hook)
        if self.gateway is not None:
            self.gateway.install(self.harness)

        # Save original hooks
        orig_on_tool = self.harness.hooks.on_tool_call
        orig_on_result = self.harness.hooks.on_tool_result

        call_times: dict[str, float] = {}
        call_args: dict[str, dict] = {}

        def _on_tool_call(name: str, args: dict):
            call_times[name] = time.time()
            call_args[name] = args
            if orig_on_tool:
                orig_on_tool(name, args)

        def _on_tool_result(name: str, result_str: str, raw: Any):
            elapsed = (time.time() - call_times.pop(name, time.time())) * 1000
            tool_records.append(ToolCallRecord(
                tool_name=name,
                arguments=call_args.pop(name, {}),
                result=result_str[:500] if result_str else "",
                latency_ms=elapsed,
            ))
            if orig_on_result:
                orig_on_result(name, result_str, raw)

        # Install hooks
        self.harness.hooks.on_tool_call = _on_tool_call
        self.harness.hooks.on_tool_result = _on_tool_result

        # Reset conversation for clean evaluation
        self.harness.messages = []
        self.harness._turn = 0

        prompt = (
            f"You have a GMS knowledge base loaded with structured entities, "
            f"relationships and numeric facts about this domain. The following "
            f"report provides background context:\n\n"
            f"<report>\n{context}\n</report>\n\n"
            f"IMPORTANT: Before answering, use gms_lookup on the key entities "
            f"in the question to retrieve verified facts from the knowledge "
            f"base. Do NOT answer from the report text alone — the GMS store "
            f"is the authoritative source.\n\n"
            f"Steps:\n"
            f"1. Identify the key entities in the question.\n"
            f"2. Use gms_lookup to retrieve their known relationships.\n"
            f"3. If the question involves connections between entities, use "
            f"gms_search.\n"
            f"4. Answer concisely. If the question specifies a format "
            f"(e.g. ANSWER: X), respond in exactly that format.\n\n"
            f"Question: {question}"
        )

        t0 = time.time()
        try:
            raw = self.harness.run(prompt)
            elapsed = (time.time() - t0) * 1000
            trace = SUTTrace(
                raw_output=raw,
                tool_calls=tool_records,
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            trace = SUTTrace(
                raw_output="",
                tool_calls=tool_records,
                latency_ms=elapsed,
                error=str(e),
            )
        finally:
            # Restore original hooks
            self.harness.hooks.on_tool_call = orig_on_tool
            self.harness.hooks.on_tool_result = orig_on_result
            # Uninstall gateway
            if self.gateway is not None:
                self.gateway.uninstall(self.harness)

        return trace

    def __repr__(self) -> str:
        return f"AgentAdapter(model={self.harness.config.model!r})"


# ---------------------------------------------------------------------------
# RAGAdapter — retriever + generator
# ---------------------------------------------------------------------------

class RAGAdapter(SUTAdapter):
    """Adapter for retrieval-augmented generation systems.

    Parameters
    ----------
    retriever : callable
        ``retriever(question, context) -> retrieved_text``
    generator : callable
        ``generator(question, retrieved_text) -> answer``
    """

    def __init__(
        self,
        retriever: Callable[[str, str], str],
        generator: Callable[[str, str], str],
    ):
        self.retriever = retriever
        self.generator = generator

    def evaluate(self, question: str, context: str) -> SUTTrace:
        t0 = time.time()
        try:
            retrieved = self.retriever(question, context)
            raw = self.generator(question, retrieved)
            elapsed = (time.time() - t0) * 1000
            return SUTTrace(
                raw_output=raw,
                latency_ms=elapsed,
                retrieved_context=retrieved[:1000] if retrieved else "",
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return SUTTrace(
                raw_output="",
                latency_ms=elapsed,
                error=str(e),
            )

    def __repr__(self) -> str:
        return "RAGAdapter()"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_adapter(
    target_type: str,
    target_model: str,
    client: "anthropic.Anthropic | None" = None,
    harness: "Harness | None" = None,
) -> SUTAdapter:
    """Create the appropriate SUT adapter.

    Parameters
    ----------
    target_type : str
        ``"llm"`` for direct API, ``"agent"`` for Harness loop.
    target_model : str
        Model name (used by LLMAdapter).
    client : anthropic.Anthropic or None
        Required for LLMAdapter.
    harness : Harness or None
        Required for AgentAdapter.
    """
    if target_type == "llm":
        if client is None:
            import anthropic as anth
            client = anth.Anthropic()
        return LLMAdapter(client, target_model)

    if target_type == "agent":
        if harness is None:
            raise ValueError("AgentAdapter requires a Harness instance")
        return AgentAdapter(harness)

    raise ValueError(f"Unknown target_type: {target_type!r}")
