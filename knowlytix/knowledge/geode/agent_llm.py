# SPDX-License-Identifier: Apache-2.0
"""GEODE agent LLM — a wireable, local-by-default actor entry point.

The GEODE actor's runtime LLM is **local and user-wireable**: a caller passes
the LLM of their choice (a model name, an :class:`LLMBackend`, or a bare
``str -> str`` callable) and :func:`agent_callable` returns the
``Callable[[str], str]`` the loop calls for every step. The default is a local
**Qwen 4B** (:data:`QWEN_4B`); no hosted/frontier model runs inside the agent
loop unless the caller explicitly opts in with ``allow_frontier=True``. Keeping
the actor local by default is how the small-model-viability premise (geometry +
the loop compensate for a small model) stays honest rather than a large frontier
model quietly doing the work.
"""

from __future__ import annotations

from typing import Callable

# Default actor model: a local Qwen 4B instruct checkpoint.
QWEN_4B = "Qwen/Qwen3-4B-Instruct-2507"
# Retained for back-compat; some callers / tests still reference the 3B id.
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_AGENT_MODEL = QWEN_4B

# Substrings that mark a hosted/frontier model. The actor stays local by
# default, so a model name carrying one of these is rejected unless the caller
# passes ``allow_frontier=True``. This is an integrity guard, not a hard lock —
# users wire the local LLM of their choice freely.
_FRONTIER_MARKERS = (
    "claude", "gpt-", "gpt4", "gpt-4", "gpt-3", "o1-", "o3-",
    "gemini", "anthropic/", "openai/", "azure/", "bedrock/", "vertex_ai/",
    "cohere", "mistral/", "command-r",
)


def _check_local_model(model_name: str, *, allow_frontier: bool) -> None:
    """Reject a hosted/frontier model name unless explicitly allowed."""
    if allow_frontier:
        return
    low = model_name.lower()
    if any(m in low for m in _FRONTIER_MARKERS):
        raise ValueError(
            f"GEODE actor model {model_name!r} looks like a hosted/frontier "
            "model. The agent runtime is local by default so a large model does "
            "not quietly do the work. Pass allow_frontier=True to override, or "
            "wire a local model (e.g. a Qwen checkpoint)."
        )


def agent_callable(
    llm: "Callable[[str], str] | object | str | None" = None,
    *,
    model_name: str = DEFAULT_AGENT_MODEL,
    device: str = "cuda",
    system: str = "",
    max_tokens: int = 256,
    allow_frontier: bool = False,
) -> Callable[[str], str]:
    """Return the ``Callable[[str], str]`` actor the GEODE loop calls each step.

    Wire the LLM of your choice via ``llm``:

    * a bare ``str -> str`` callable — returned unchanged (you own the wiring);
    * an :class:`~knowlytix.knowledge.llm_backend.LLMBackend` (anything with a
      ``.call(system, user, max_tokens)`` method) — wrapped into a callable;
    * ``None`` — a local :class:`LocalTransformersBackend` is built for
      ``model_name`` (default :data:`QWEN_4B`).

    The heavy backend import is deferred to call time. A hosted/frontier model
    name is rejected unless ``allow_frontier=True`` (see :data:`_FRONTIER_MARKERS`).
    """
    # Already a plain callable (and not a backend object): the caller owns it.
    if callable(llm) and not hasattr(llm, "call"):
        return llm  # type: ignore[return-value]

    # A backend instance (has .call): wrap its .call into the str->str actor.
    if llm is not None and hasattr(llm, "call"):
        backend = llm
        return lambda prompt: backend.call(
            system=system, user=prompt, max_tokens=max_tokens)

    # A model name string overrides the default model.
    if isinstance(llm, str) and llm.strip():
        model_name = llm.strip()

    _check_local_model(model_name, allow_frontier=allow_frontier)
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend

    backend = LocalTransformersBackend(model_name, device=device)
    return lambda prompt: backend.call(system=system, user=prompt,
                                       max_tokens=max_tokens)


def qwen_agent_callable(
    model_name: str = DEFAULT_AGENT_MODEL,
    device: str = "cuda",
    *,
    system: str = "",
    max_tokens: int = 256,
    allow_frontier: bool = False,
) -> Callable[[str], str]:
    """Back-compat constructor for a local-model actor callable.

    Thin wrapper over :func:`agent_callable` that builds a local backend for
    ``model_name`` (default :data:`QWEN_4B`). Kept so existing callers keep
    working; new code should prefer :func:`agent_callable`, which also accepts a
    pre-built backend or callable.
    """
    return agent_callable(model_name=model_name, device=device, system=system,
                          max_tokens=max_tokens, allow_frontier=allow_frontier)
