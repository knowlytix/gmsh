# SPDX-License-Identifier: Apache-2.0
"""Phase 4a helpers used by NB 16 (Gateway) and NB 17 (Contracts).

Phase 0 ``_shared.py`` and Phase 1B ``_helpers_phase1.py`` are treated
as stable.  These helpers extend them without modification.

Helpers
-------

plot_gate_decision_grid
    Heat-style grid: one row per call, three columns (schema, policy,
    plausibility); green=pass, red=blocked, gray=not reached.

plot_visited_states_grid
    Heat-style grid: one row per contract, columns are FSM states in
    canonical order; cells filled in if visited during validation.

plot_contract_validity_bar
    Two-bar plot per contract: total tools checked vs tools blocked.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Per-call gate decision grid (NB 16)
# ---------------------------------------------------------------------------

GATE_NAMES = ("schema", "policy", "plausibility")


def plot_gate_decision_grid(ax, rows, *, title: str = "") -> None:
    """Render a (call, gate) verdict grid.

    Each row in *rows* is a tuple ``(label, gate_results)`` where
    ``gate_results`` is the dict from ``GatewayResult.gate_results``.
    A gate that is absent from the dict is rendered as 'not reached'
    (gray); a present True is green (PASS); a present False is red
    (BLOCK).  The leftmost column is the call label.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows)
    g = len(GATE_NAMES)
    # 0 = not reached, 1 = pass, 2 = block
    arr = np.zeros((n, g), dtype=int)
    text = [["" for _ in range(g)] for _ in range(n)]
    for i, (_, gr) in enumerate(rows):
        for j, gate in enumerate(GATE_NAMES):
            if gate not in gr:
                arr[i, j] = 0
                text[i][j] = "—"
            elif gr[gate]:
                arr[i, j] = 1
                text[i][j] = "PASS"
            else:
                arr[i, j] = 2
                text[i][j] = "BLOCK"

    cmap = plt.cm.colors.ListedColormap(
        ["#cccccc", "#a5d99b", "#f4a8a0"]
    )
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(g))
    ax.set_xticklabels(list(GATE_NAMES), fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    for i in range(n):
        for j in range(g):
            ax.text(j, i, text[i][j], ha="center", va="center",
                    fontsize=8, fontweight="bold", color="#222")
    if title:
        ax.set_title(title, fontsize=11)


# ---------------------------------------------------------------------------
# 2. Visited-states grid (NB 17)
# ---------------------------------------------------------------------------

CANONICAL_STATES = (
    "idle",
    "data_loaded",
    "preprocessed",
    "model_trained",
    "diagnosed",
    "fair_checked",
    "compared",
    "human_review",
    "deployment_ready",
)


def plot_visited_states_grid(ax, rows, *, title: str = "") -> None:
    """Render a (contract, state) visited grid.

    Each row in *rows* is a tuple ``(contract_label, visited_states)``
    where ``visited_states`` is an iterable of state-name strings (the
    ``.value`` attribute of ``ContractState``).  Visited cells are
    colored; unvisited are blank.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows)
    s = len(CANONICAL_STATES)
    arr = np.zeros((n, s), dtype=int)
    for i, (_, visited) in enumerate(rows):
        visited_set = set(visited)
        for j, name in enumerate(CANONICAL_STATES):
            if name in visited_set:
                arr[i, j] = 1

    cmap = plt.cm.colors.ListedColormap(["#f0f0f0", "#7fb8d9"])
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(s))
    ax.set_xticklabels(list(CANONICAL_STATES), fontsize=8, rotation=30,
                       ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    for i in range(n):
        for j in range(s):
            mark = "X" if arr[i, j] else ""
            ax.text(j, i, mark, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#222")
    if title:
        ax.set_title(title, fontsize=11)


# ---------------------------------------------------------------------------
# 3. Per-contract validity bar (NB 17)
# ---------------------------------------------------------------------------

def plot_contract_validity_bar(ax, labels, checked, blocked,
                               *, title: str = "") -> None:
    """Side-by-side bar of tools_checked vs tools_blocked per contract."""
    import numpy as np

    n = len(labels)
    x = np.arange(n)
    width = 0.35
    ax.bar(x - width / 2, checked, width, label="checked",
           color="tab:blue")
    ax.bar(x + width / 2, blocked, width, label="blocked",
           color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
