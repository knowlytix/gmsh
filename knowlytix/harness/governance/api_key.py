# SPDX-License-Identifier: Apache-2.0
"""Anthropic API-key resolution helper.

Single canonical entry point for the codebase --- resolves a key from
the standard ``ANTHROPIC_API_KEY`` environment variable, falling back
to ``pass show anthropic/api-key`` (a Unix password-store integration
common on developer workstations).  Raises a clear error if neither
source provides a key.

Used by :class:`knowlytix.knowledge.llm_backend.AnthropicBackend`,
:func:`governed_agent.plan_adversarial.generate_llm_plans`, and any
other library code that talks to Anthropic.  Keeping the resolution
logic in one place avoids drift across multiple callers.
"""

from __future__ import annotations

import os
import subprocess


__all__ = ["get_anthropic_api_key", "AnthropicAPIKeyError"]


class AnthropicAPIKeyError(RuntimeError):
    """Raised when no Anthropic API key can be resolved.

    Production deployments treat this as fatal: there is no safe
    fall-back to a mock LLM, because the gating layers downstream
    rely on real LLM behavior to be exercised.  A missing key means
    the run cannot proceed; surface the failure rather than masking
    it with canned text.
    """


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key.

    Resolution order:

    1. ``ANTHROPIC_API_KEY`` environment variable.
    2. ``pass show anthropic/api-key`` (Unix ``pass(1)`` password
       store).

    Raises
    ------
    AnthropicAPIKeyError
        If neither source provides a key.  The message names both
        sources so the caller knows where to set it.
    """
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    try:
        result = subprocess.run(
            ["pass", "show", "anthropic/api-key"],
            capture_output=True, text=True, check=True,
        )
        key = result.stdout.strip().splitlines()[0]
        if key:
            return key
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    raise AnthropicAPIKeyError(
        "Anthropic API key not found.  Set ANTHROPIC_API_KEY in the "
        "environment or configure `pass(1)` with "
        "`pass insert anthropic/api-key`.  Production runs do NOT "
        "fall back to a mock LLM --- the gating layers must exercise "
        "real LLM behavior to be meaningful."
    )
