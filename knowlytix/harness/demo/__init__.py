# SPDX-License-Identifier: Apache-2.0
"""Keyless-demo quickstart notebook, shipped with knowlytix-harness.

``demo_quickstart.ipynb`` runs end-to-end with no license key and no LLM API
key (``KNOWLYTIX_DEMO_MODE=1``), exercising the bundled demo corpus in
``knowlytix.benchmark.fixtures.demo``.  Regenerate via
``scripts/build_demo_notebook.py``.  Resolve the shipped path with
:func:`notebook_path`.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__all__ = ["notebook_path"]


def notebook_path() -> Path:
    """Absolute, canonical path to the bundled ``demo_quickstart.ipynb``.

    Mirrors :func:`knowlytix.harness._smoke.notebook_path` — the
    Traversable→Path coercion + symlink resolution kept in one expression so
    the demo CI test (and any CLI surface) track together if
    ``importlib.resources`` semantics tighten in a future Python.
    """
    return Path(str(files("knowlytix.harness.demo") / "demo_quickstart.ipynb")).resolve()
