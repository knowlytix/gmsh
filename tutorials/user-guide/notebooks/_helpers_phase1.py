"""Phase 1B helpers used by NB 01-05 (foundations notebooks).

Phase 0 ``_shared.py`` is treated as stable — these helpers extend it
without modification.

Helpers
-------

bootstrap_runtime
    Idempotently prepends the cluster-venv site-packages and the
    repo root to ``sys.path`` so that the default ``python3``
    kernel can import ``torch``, ``docgms``, ``governed_agent`` etc.

draw_six_layer_stack
    Draws the six-layer policy-governed-agent architecture as a
    vertical stack of labeled rectangles on a matplotlib axis.

draw_three_gate_flow
    Draws the three sequential gates (schema -> policy ->
    plausibility) inside the ``GovernedToolGateway`` as a horizontal
    flow on a matplotlib axis.

plot_pass_fail_grid
    Imshow-based grid that renders one binary cell per diagnostic
    check, with PASS/FAIL labels.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Iterable


# ---------------------------------------------------------------------------
# 1. Runtime bootstrap (the default kernel runs /usr/bin/python3, which does
#    not have torch installed; the cluster venv does. Notebooks call this in
#    cell 1 before importing _shared.)
# ---------------------------------------------------------------------------

def bootstrap_runtime(repo: str = "~/jupyterlab/kg-memory") -> None:
    """Prepend cluster-venv site-packages and *repo* to ``sys.path``.

    Idempotent: if a path is already on ``sys.path`` it is left alone.
    Also installs a no-op stub for the optional ``modeva`` package so
    that ``harness/__init__.py`` (which transitively imports
    ``agents.ds_agent.tools``) loads even on machines where ``modeva``
    is not installed.  The stub is only registered if the real
    package cannot be imported.
    """
    for p in glob.glob(
        "/home/asudjianto/cluster/spark-venv/lib/python*/site-packages"
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    repo_abs = os.path.expanduser(repo)
    if repo_abs not in sys.path:
        sys.path.insert(0, repo_abs)
    _install_modeva_stub_if_missing()


def _install_modeva_stub_if_missing() -> None:
    """Provide a minimal ``modeva`` module if the real one is unavailable.

    Only the names imported at module load time by
    ``agents/ds_agent/tools.py`` are needed (``DataSet``, ``ModelZoo``,
    ``TestSuite``); they are placeholder classes that are never
    instantiated by NB 01-05.
    """
    import importlib
    try:
        importlib.import_module("modeva")
        return
    except ImportError:
        pass
    import types
    stub = types.ModuleType("modeva")

    class _Placeholder:
        """Stub class used only as a type-hint anchor."""

        def __init__(self, *_, **__):
            raise RuntimeError(
                "modeva is not installed in this environment; "
                "the stub forbids instantiation."
            )

    stub.DataSet = _Placeholder
    stub.ModelZoo = _Placeholder
    stub.TestSuite = _Placeholder
    sys.modules["modeva"] = stub


# ---------------------------------------------------------------------------
# 2. Architecture diagrams (NB 02)
# ---------------------------------------------------------------------------

_LAYERS = [
    ("Layer 6", "Governance Stack",
     "verifier, tracer, SPC gates, compliance"),
    ("Layer 5", "Memory + Reasoning",
     "GMSExpertStore, AgentMemory, GeometricReasoner"),
    ("Layer 4", "Extraction",
     "three-tier (regex / hybrid / llm-only)"),
    ("Layer 3", "Tool Definitions",
     "TOOL_DEFINITIONS, schema-validated"),
    ("Layer 2", "Formal Contracts",
     "BehavioralContract finite-state machines"),
    ("Layer 1", "Policy Rules",
     "PolicyEngine: declarative + calibrated GMS"),
]


def draw_six_layer_stack(ax, layers: Iterable[tuple] = _LAYERS) -> None:
    """Draw the six-layer architecture as a vertical stack of boxes."""
    from matplotlib.patches import Rectangle

    layers = list(layers)
    n = len(layers)
    box_w, box_h = 1.0, 0.9
    for i, (tag, name, sub) in enumerate(layers):
        y = (n - 1 - i) * box_h
        ax.add_patch(Rectangle(
            (0.0, y), box_w, box_h,
            facecolor="#dde6f0", edgecolor="#244",
            linewidth=1.2,
        ))
        ax.text(0.04, y + box_h * 0.62, f"{tag}: {name}",
                fontsize=10, fontweight="bold", va="center")
        ax.text(0.04, y + box_h * 0.25, sub,
                fontsize=8, color="#333", va="center")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, n * box_h + 0.1)
    ax.set_aspect("auto")
    ax.set_axis_off()
    ax.set_title("Six layers", fontsize=11)


_GATES = [
    ("Schema",       "required params + types"),
    ("Policy",       "PolicyEngine.check"),
    ("Plausibility", "geodesic on $S^{m-1}$"),
]


def draw_three_gate_flow(ax, gates: Iterable[tuple] = _GATES) -> None:
    """Draw the three-gate flow as a horizontal sequence of boxes."""
    from matplotlib.patches import Rectangle, FancyArrow

    gates = list(gates)
    n = len(gates)
    box_w, box_h = 1.0, 0.6
    gap = 0.25
    for i, (name, sub) in enumerate(gates):
        x = i * (box_w + gap)
        ax.add_patch(Rectangle(
            (x, 0.0), box_w, box_h,
            facecolor="#e6f0dd", edgecolor="#262",
            linewidth=1.2,
        ))
        ax.text(x + box_w / 2, box_h * 0.62, f"{i + 1}. {name}",
                fontsize=10, fontweight="bold", ha="center", va="center")
        ax.text(x + box_w / 2, box_h * 0.25, sub,
                fontsize=8, color="#222", ha="center", va="center")
        if i < n - 1:
            ax.add_patch(FancyArrow(
                x + box_w + 0.02, box_h / 2, gap - 0.06, 0,
                width=0.02, head_width=0.07, head_length=0.05,
                length_includes_head=True, color="#262",
            ))
    total = n * box_w + (n - 1) * gap
    ax.set_xlim(-0.1, total + 0.1)
    ax.set_ylim(-0.1, box_h + 0.2)
    ax.set_aspect("auto")
    ax.set_axis_off()
    ax.set_title("Three gates inside GovernedToolGateway", fontsize=11)


# ---------------------------------------------------------------------------
# 3. Pass/fail grid (NB 05)
# ---------------------------------------------------------------------------

def plot_pass_fail_grid(ax, names, passed, *, title: str = "") -> None:
    """One row per check; green=PASS, red=FAIL; label on the right."""
    import numpy as np

    n = len(names)
    arr = np.array(passed, dtype=int).reshape(n, 1)
    ax.imshow(
        arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
        extent=(0, 1, n - 0.5, -0.5),
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks([0.5])
    ax.set_xticklabels(["check"])
    for i, ok in enumerate(passed):
        ax.text(0.5, i, "PASS" if ok else "FAIL",
                ha="center", va="center",
                color="black", fontsize=9, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=11)
