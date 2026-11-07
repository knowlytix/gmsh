# SPDX-License-Identifier: Apache-2.0
"""Phase 5C helpers used by NB 26 (Compliance).

Phase 0 ``_shared.py`` is treated as stable; this module adds a single
helper for drawing a provenance lineage DAG. Imported only by NB 26.
"""

from __future__ import annotations

from typing import Iterable


def draw_lineage_dag(ax, lineage, *, title: str = "") -> None:
    """Draw a left-to-right provenance chain on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lineage : Iterable of dict
        Records as returned by ``ProvenanceRegistry.get_lineage`` —
        each dict carries at least ``name``, ``type``, ``fingerprint``.
    title : str, optional
        Axes title.
    """
    from matplotlib.patches import FancyArrow, Rectangle

    nodes = list(lineage)
    if not nodes:
        ax.text(0.5, 0.5, "(empty lineage)", ha="center", va="center")
        ax.set_axis_off()
        return

    # color per node-type
    palette = {
        "dataset":    "#dde6f0",
        "model":      "#e6f0dd",
        "metric":     "#f0e6dd",
        "verification": "#f0ddee",
        "policy":     "#dde6e6",
        "snapshot":   "#e6dde6",
    }
    box_w, box_h = 1.4, 0.85
    gap = 0.45

    n = len(nodes)
    for i, rec in enumerate(nodes):
        x = i * (box_w + gap)
        face = palette.get(rec.get("type", ""), "#eeeeee")
        ax.add_patch(Rectangle(
            (x, 0.0), box_w, box_h,
            facecolor=face, edgecolor="#244", linewidth=1.2,
        ))
        ax.text(
            x + box_w / 2, box_h * 0.70,
            str(rec.get("name", "?")),
            fontsize=10, fontweight="bold", ha="center", va="center",
        )
        ax.text(
            x + box_w / 2, box_h * 0.45,
            f"[{rec.get('type', '?')}]",
            fontsize=8, color="#333", ha="center", va="center",
        )
        ax.text(
            x + box_w / 2, box_h * 0.18,
            f"sha256:{rec.get('fingerprint', '?')[:8]}",
            fontsize=7, color="#555", family="monospace",
            ha="center", va="center",
        )
        if i < n - 1:
            ax.add_patch(FancyArrow(
                x + box_w + 0.02, box_h / 2, gap - 0.06, 0,
                width=0.02, head_width=0.08, head_length=0.06,
                length_includes_head=True, color="#244",
            ))

    total = n * box_w + (n - 1) * gap
    ax.set_xlim(-0.1, total + 0.1)
    ax.set_ylim(-0.1, box_h + 0.2)
    ax.set_aspect("auto")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)


def draw_truth_chain(ax, stages: Iterable[tuple], *, title: str = "") -> None:
    """Draw the document -> section -> triple -> truth -> claim -> verdict chain.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    stages : Iterable of (label, sub) tuples
        Each tuple is one stage in the provenance chain. ``label`` is
        the stage name (e.g. ``"document"``); ``sub`` is the concrete
        artifact reference (e.g. ``"data_science_practices.md"``).
    title : str, optional
        Axes title.
    """
    from matplotlib.patches import FancyArrow, Rectangle

    items = list(stages)
    n = len(items)
    box_w, box_h = 1.55, 0.7
    gap = 0.35

    for i, (label, sub) in enumerate(items):
        x = i * (box_w + gap)
        ax.add_patch(Rectangle(
            (x, 0.0), box_w, box_h,
            facecolor="#dde6f0", edgecolor="#244", linewidth=1.2,
        ))
        ax.text(
            x + box_w / 2, box_h * 0.66, label,
            fontsize=10, fontweight="bold", ha="center", va="center",
        )
        ax.text(
            x + box_w / 2, box_h * 0.30, sub,
            fontsize=7, color="#333", ha="center", va="center",
        )
        if i < n - 1:
            ax.add_patch(FancyArrow(
                x + box_w + 0.02, box_h / 2, gap - 0.06, 0,
                width=0.02, head_width=0.07, head_length=0.05,
                length_includes_head=True, color="#244",
            ))

    total = n * box_w + (n - 1) * gap
    ax.set_xlim(-0.1, total + 0.1)
    ax.set_ylim(-0.1, box_h + 0.2)
    ax.set_aspect("auto")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)
