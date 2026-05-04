"""Phase 4c helpers — orchestrator timeline and role-permission matrix plots.

Used by NB 20 (orchestrator) and NB 21 (multi-agent). Extends the
shared helpers in ``_shared.py`` without modification.

Helpers
-------

plot_orchestrator_timeline
    Horizontal timeline of orchestrator steps; each step is colored by
    its attestor verdict (OK, BLOCKED, ABORTED) and labeled with the
    tool name and execution-time bar.

plot_role_permission_matrix
    Imshow-based matrix with one row per agent role and one column per
    tool, colored allow / forbidden / default-deny. Reads directly from
    ``AgentRole.allowed_tools`` and ``AgentRole.forbidden_tools``.
"""

from __future__ import annotations

from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# 1. Orchestrator step timeline (NB 20)
# ---------------------------------------------------------------------------

_VERDICT_COLOR = {
    "OK":      "#4caf50",   # green
    "BLOCKED": "#f4a536",   # amber
    "ABORTED": "#d33",      # red
    "SKIPPED": "#888",      # gray
}


def step_verdict(step: dict) -> str:
    """Classify a step record from ``OrchestratorResult.steps_executed``.

    Returns one of ``"OK"``, ``"BLOCKED"``, ``"ABORTED"``, ``"SKIPPED"``.
    """
    if step.get("blocked"):
        return "BLOCKED"
    reason = (step.get("reason") or "")
    if isinstance(reason, str) and reason.startswith("tool_error"):
        return "ABORTED"
    return "OK"


def plot_orchestrator_timeline(ax, steps_executed: Sequence[dict],
                               *, title: str = "") -> None:
    """Draw a horizontal timeline of orchestrator steps.

    Each step occupies one row; the bar width is fixed (the timing
    field is not always populated for tool-error aborts). The bar
    color encodes the attestor verdict: green=OK, amber=BLOCKED,
    red=ABORTED, gray=SKIPPED.
    """
    from matplotlib.patches import Rectangle

    n = len(steps_executed)
    box_h = 0.7
    for i, step in enumerate(steps_executed):
        verdict = step_verdict(step)
        color = _VERDICT_COLOR.get(verdict, "#888")
        y = (n - 1 - i)
        ax.add_patch(Rectangle(
            (0.0, y), 1.0, box_h,
            facecolor=color, edgecolor="#222",
            linewidth=1.0, alpha=0.85,
        ))
        tool = step.get("tool", "?")
        n_facts = step.get("n_facts", 0)
        ax.text(0.04, y + box_h * 0.55,
                f"[{step.get('step', i):2d}] {tool}",
                fontsize=9, fontweight="bold", va="center")
        ax.text(0.04, y + box_h * 0.18,
                f"{verdict} · facts={n_facts}",
                fontsize=8, color="#222", va="center")
    # Legend swatches across the top.
    legend_x = 1.05
    for j, (label, color) in enumerate(_VERDICT_COLOR.items()):
        y = (n - 1) - j * 0.4
        ax.add_patch(Rectangle(
            (legend_x, y), 0.12, 0.25,
            facecolor=color, edgecolor="#222", linewidth=0.8,
        ))
        ax.text(legend_x + 0.15, y + 0.12, label,
                fontsize=8, va="center")
    ax.set_xlim(-0.05, 1.6)
    ax.set_ylim(-0.2, n + 0.2)
    ax.set_aspect("auto")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)


# ---------------------------------------------------------------------------
# 2. Role-permission matrix (NB 21)
# ---------------------------------------------------------------------------

def plot_role_permission_matrix(ax, agents_by_name: dict,
                                tools: Iterable[str],
                                *, title: str = "") -> None:
    """Draw an agent-role x tool permission matrix.

    For each (role, tool) cell:
      - 1.0 (allow)         — tool in role.allowed_tools
      - -1.0 (explicit deny) — tool in role.forbidden_tools
      - 0.0 (default-deny)   — tool in neither set

    Cells are colored on a diverging colormap so the three classes are
    visually distinct.
    """
    import numpy as np

    role_names = list(agents_by_name.keys())
    tool_list = list(tools)
    arr = np.zeros((len(role_names), len(tool_list)), dtype=float)
    for i, role_name in enumerate(role_names):
        agent = agents_by_name[role_name]
        allowed = agent.role.allowed_tools
        forbidden = agent.role.forbidden_tools
        for j, tool in enumerate(tool_list):
            if tool in allowed:
                arr[i, j] = 1.0
            elif tool in forbidden:
                arr[i, j] = -1.0
            else:
                arr[i, j] = 0.0
    im = ax.imshow(
        arr, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0,
    )
    ax.set_xticks(range(len(tool_list)))
    ax.set_xticklabels(tool_list, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(role_names)))
    ax.set_yticklabels(role_names, fontsize=9)
    for i in range(len(role_names)):
        for j in range(len(tool_list)):
            v = arr[i, j]
            if v > 0.5:
                ax.text(j, i, "A", ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")
            elif v < -0.5:
                ax.text(j, i, "F", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            else:
                ax.text(j, i, "·", ha="center", va="center",
                        fontsize=10, color="#444")
    if title:
        ax.set_title(title, fontsize=11)
    return im
