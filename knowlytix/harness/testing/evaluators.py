# SPDX-License-Identifier: Apache-2.0
"""Target evaluator wrappers for DOE benchmarks.

Both wrappers expose the same interface::

    evaluator(question: str, context: str) -> str

so they can be used interchangeably by the benchmark orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.llm_caller import call_llm

if TYPE_CHECKING:
    import anthropic
    from knowlytix.harness.core import Harness


class LLMEvaluator:
    """Wraps a raw Claude API call as a DOE evaluator.

    Parameters
    ----------
    client : anthropic.Anthropic
        Anthropic API client.
    model : str
        Model to evaluate (e.g. ``"claude-sonnet-4-20250514"``).
    """

    def __init__(self, client: "anthropic.Anthropic", model: str):
        self.client = client
        self.model = model

    def __call__(self, question: str, context: str) -> str:
        return call_llm(self.client, context, question, self.model)

    def __repr__(self) -> str:
        return f"LLMEvaluator(model={self.model!r})"


class AgentEvaluator:
    """Wraps the Harness agent loop as a DOE evaluator.

    The agent receives the question as a user request and the markdown
    context is injected into the system prompt.  The agent can use any
    registered tools (GMS query, MoDeva, etc.) to arrive at an answer.

    Parameters
    ----------
    harness : Harness
        A configured Harness instance with tools registered.
    """

    def __init__(self, harness: "Harness"):
        self.harness = harness

    def __call__(self, question: str, context: str) -> str:
        # Reset conversation so each question starts clean
        self.harness.messages = []
        self.harness._turn = 0

        # Build a combined prompt that provides context + question
        prompt = (
            f"You have been given the following report as context:\n\n"
            f"<report>\n{context}\n</report>\n\n"
            f"Answer the following question using the report and any tools "
            f"available to you.\n\n{question}"
        )
        return self.harness.run(prompt)

    def __repr__(self) -> str:
        return f"AgentEvaluator(model={self.harness.config.model!r})"


def make_evaluator(
    target_type: str,
    target_model: str,
    client: "anthropic.Anthropic | None" = None,
    harness: "Harness | None" = None,
) -> LLMEvaluator | AgentEvaluator:
    """Factory to create the appropriate evaluator.

    Parameters
    ----------
    target_type : str
        ``"llm"`` for direct API calls or ``"agent"`` for Harness loop.
    target_model : str
        Model name (used by LLMEvaluator).
    client : anthropic.Anthropic or None
        Required when target_type is ``"llm"``.
    harness : Harness or None
        Required when target_type is ``"agent"``.
    """
    if target_type == "llm":
        if client is None:
            import anthropic as anth
            client = anth.Anthropic()
        return LLMEvaluator(client, target_model)

    if target_type == "agent":
        if harness is None:
            raise ValueError("AgentEvaluator requires a Harness instance")
        return AgentEvaluator(harness)

    raise ValueError(f"Unknown target_type: {target_type!r}. Use 'llm' or 'agent'.")
