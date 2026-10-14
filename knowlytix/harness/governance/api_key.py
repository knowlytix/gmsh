# SPDX-License-Identifier: Apache-2.0
"""Anthropic API-key resolution helper.

Single canonical entry point for the codebase --- resolves a key from
the standard ``ANTHROPIC_API_KEY`` environment variable, then falls
back to a per-user file at ``~/.anthropic_key`` (or
``~/.config/anthropic/api-key``), and finally to ``pass show
anthropic/api-key`` (a Unix password-store integration common on
developer workstations).  Raises a clear error if no source provides
a key.

Used by :class:`knowlytix.knowledge.llm_backend.AnthropicBackend`,
:func:`governed_agent.plan_adversarial.generate_llm_plans`, and any
other library code that talks to Anthropic.  Keeping the resolution
logic in one place avoids drift across multiple callers.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["get_anthropic_api_key", "AnthropicAPIKeyError"]


# Per-user files checked after the env var and before `pass(1)`.  The
# first existing, non-empty file wins.  Both forms are common dev
# conventions: ``~/.anthropic_key`` is the flat dotfile pattern;
# ``~/.config/anthropic/api-key`` follows the XDG base-dir pattern.
_KEY_FILE_CANDIDATES: tuple[Path, ...] = (
    Path.home() / ".anthropic_key",
    Path.home() / ".config" / "anthropic" / "api-key",
)


class AnthropicAPIKeyError(RuntimeError):
    """Raised when no Anthropic API key can be resolved.

    Production deployments treat this as fatal: there is no safe
    fall-back to a mock LLM, because the gating layers downstream
    rely on real LLM behavior to be exercised.  A missing key means
    the run cannot proceed; surface the failure rather than masking
    it with canned text.
    """


def _read_key_file(path: Path) -> str | None:
    """Return the first non-empty line of ``path``, or ``None`` if the
    file is missing, unreadable, or empty.  Trailing whitespace is
    stripped so a file written with a trailing newline still resolves.
    """
    try:
        text = path.read_text()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return None
    for line in text.splitlines():
        key = line.strip()
        if key:
            return key
    return None


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key.

    Resolution order:

    1. ``ANTHROPIC_API_KEY`` environment variable.
    2. ``~/.anthropic_key`` file (first non-empty line).
    3. ``~/.config/anthropic/api-key`` file (first non-empty line).
    4. ``pass show anthropic/api-key`` (Unix ``pass(1)`` password
       store).

    Raises
    ------
    AnthropicAPIKeyError
        If no source provides a key.  The message names every source
        so the caller knows where to set it.
    """
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    for candidate in _KEY_FILE_CANDIDATES:
        key = _read_key_file(candidate)
        if key:
            return key
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
        "environment, write the key to ~/.anthropic_key (or "
        "~/.config/anthropic/api-key), or configure `pass(1)` with "
        "`pass insert anthropic/api-key`.  Production runs do NOT "
        "fall back to a mock LLM --- the gating layers must exercise "
        "real LLM behavior to be meaningful."
    )
