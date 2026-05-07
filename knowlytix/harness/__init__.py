# SPDX-License-Identifier: Apache-2.0
"""Harness: standalone orchestration layer for GMS + MoDeva tools.

Usage:
    from knowlytix.harness import Harness

    h = Harness(stores_dir="gms_stores")
    h.register_gms_tools()
    h.register_modeva_tools()
    response = h.run("Load BikeSharing, train XGB depth-2, run FANOVA")

Loading is lazy: ``from knowlytix.harness import Harness`` triggers the heavy
``harness.core`` import (anthropic SDK, modeva, etc.); ``from
harness.settings import GMSHSettings`` does not. This keeps the flat
settings view usable in minimal wheel-test environments that don't
have the DS-agent optional deps installed.
"""

from __future__ import annotations

__version__ = "0.2.0"

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Harness, HarnessConfig
    from .registry import ToolDef, ToolRegistry
    from .settings import GMSHSettings

    # Populated lazily by __getattr__ below.  Annotated here so static
    # checkers see `harness.settings` as `GMSHSettings`, not `Any`.
    settings: GMSHSettings


__all__ = ["Harness", "HarnessConfig", "ToolDef", "ToolRegistry", "settings"]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy loader: defer `.core` / `.registry` imports.

    Each symbol is resolved on first attribute access and cached in the
    module's globals so subsequent lookups are zero-overhead. Anything
    outside `__all__` falls through to a normal ``AttributeError``.
    """

    if name in {"Harness", "HarnessConfig"}:
        from .core import Harness, HarnessConfig

        globals()["Harness"] = Harness
        globals()["HarnessConfig"] = HarnessConfig
        return globals()[name]
    if name in {"ToolDef", "ToolRegistry"}:
        from .registry import ToolDef, ToolRegistry

        globals()["ToolDef"] = ToolDef
        globals()["ToolRegistry"] = ToolRegistry
        return globals()[name]
    if name == "settings":
        # Licensed settings singleton — pulls caps from knowlytix.core._caps.get_caps()
        # which fails closed until gms has verified a license key.  Lazy
        # on purpose: minimal wheel-test environments don't always have
        # the license gate wired up, and this lets them skip the check
        # unless they actually need the settings.
        from .settings import GMSHSettings

        globals()["settings"] = GMSHSettings.from_license()
        return globals()["settings"]
    raise AttributeError(f"module 'harness' has no attribute {name!r}")
