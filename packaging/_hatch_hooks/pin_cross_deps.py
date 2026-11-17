# SPDX-License-Identifier: Apache-2.0
"""Shared hatchling metadata hook: pin knowlytix-* cross-deps to the exact
hatch-vcs version at build time.

The dependent wheels (knowledge/benchmark/harness) declare their sibling
knowlytix-* dependencies as BARE names (no specifier).  This hook rewrites
each to ``name==<resolved-version>`` so the lockstep cross-pins always match
the wheel's own git-tag-derived version — no version literal to hand-edit.

Wired per package via::

    [tool.hatch.metadata.hooks.custom]
    path = "../_hatch_hooks/pin_cross_deps.py"
"""

from __future__ import annotations

import os

from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging.requirements import Requirement

_SIBLINGS = frozenset(
    {
        "knowlytix-core",
        "knowlytix-knowledge",
        "knowlytix-benchmark",
        "knowlytix-harness",
    }
)


class PinCrossDepsMetadataHook(MetadataHookInterface):
    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        version = metadata.get("version") or self._scm_version()
        pinned: list[str] = []
        for dep in metadata.get("dependencies", []):
            name = Requirement(dep).name
            if name in _SIBLINGS:
                # Force the exact lockstep pin regardless of any specifier
                # already present — the git tag is the single source of truth.
                pinned.append(f"{name}=={version}")
            else:
                pinned.append(dep)
        metadata["dependencies"] = pinned

    def _scm_version(self) -> str:
        # Fallback if the version source hasn't populated metadata yet.
        # self.root is packaging/<pkg>/; the repo .git is two levels up.
        from setuptools_scm import get_version

        return get_version(root=os.path.join(self.root, "..", ".."))
