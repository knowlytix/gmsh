# SPDX-License-Identifier: Apache-2.0
"""GMSH Cost Tracker — GMS-native token and cost accounting.

Encodes cumulative cost as a phase angle on S^1 and scores budget
plausibility via ``score_triple("total_cost", "within", "budget")``.

- Phase near 0 → well under budget
- Phase near pi/2 → at budget boundary
- Phase near pi → severely over budget

Tension energy between current cost phase and budget phase gives a
continuous compliance metric composable with other GMS scores.

Usage::

    tracker = CostTracker(store, budget_usd=10.0)
    tracker.record("gms_query", input_tokens=500, output_tokens=200)
    tracker.record("gms_lookup", input_tokens=100, output_tokens=50)
    summary = tracker.summary()
    print(summary.total_cost_usd, summary.budget_tension)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Pricing (per-million tokens, USD) — sensible defaults
# ---------------------------------------------------------------------------

# Default pricing for Claude models (approximate, per 1M tokens)
DEFAULT_PRICING = {
    "input_per_mtok": 3.0,    # $3/M input tokens
    "output_per_mtok": 15.0,  # $15/M output tokens
}


# ---------------------------------------------------------------------------
# Phase encoding helpers
# ---------------------------------------------------------------------------

def _cost_to_phase(cost_usd: float, budget_usd: float) -> float:
    """Map cumulative cost to an angle on S^1.

    0 USD → 0, budget → pi/2, 2*budget → pi.  Clamped at pi.
    """
    if budget_usd <= 0:
        return 0.0
    ratio = cost_usd / budget_usd
    return min(ratio, 2.0) * (math.pi / 2.0)


def _tension_from_phases(phase_a: float, phase_b: float) -> float:
    """Tension energy between two S^1 phases.  E = 2 sin(|Δθ|/2)."""
    return 2.0 * math.sin(abs(phase_a - phase_b) / 2.0)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CostRecord:
    """A single token/cost measurement with phase encoding.

    Attributes
    ----------
    tool_name : str
        Tool or operation that consumed tokens.
    input_tokens : int
        Input (prompt) tokens consumed.
    output_tokens : int
        Output (completion) tokens consumed.
    cost_usd : float
        Estimated cost in USD for this call.
    cumulative_cost_usd : float
        Running total cost after this record.
    cost_phase : float
        Cumulative cost encoded as S^1 angle.
    budget_phase : float
        Budget boundary encoded as S^1 angle (always pi/2).
    budget_tension : float
        Tension energy between cost phase and budget phase.
    within_budget : bool
        Whether cumulative cost <= budget.
    timestamp : str
        ISO timestamp.
    """

    tool_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0
    cost_phase: float = 0.0
    budget_phase: float = 0.0
    budget_tension: float = 0.0
    within_budget: bool = True
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class CostSummary:
    """Aggregate cost summary with GMS budget plausibility.

    Attributes
    ----------
    total_input_tokens : int
        Total input tokens across all records.
    total_output_tokens : int
        Total output tokens across all records.
    total_tokens : int
        Sum of input + output tokens.
    total_cost_usd : float
        Total estimated cost in USD.
    budget_usd : float
        Budget cap in USD.
    budget_utilization : float
        Fraction of budget consumed (can exceed 1.0).
    budget_tension : float
        Final tension energy vs budget boundary.
    budget_geodesic : float or None
        GMS ``score_triple("total_cost", "within", "budget")``
        geodesic — how plausible it is that cost is within budget.
    n_records : int
        Number of cost records.
    per_tool : dict[str, dict]
        Per-tool token and cost breakdown.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_usd: float = 0.0
    budget_utilization: float = 0.0
    budget_tension: float = 0.0
    budget_geodesic: float | None = None
    n_records: int = 0
    per_tool: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_usd": self.budget_usd,
            "budget_utilization": round(self.budget_utilization, 4),
            "budget_tension": round(self.budget_tension, 4),
            "budget_geodesic": (
                round(self.budget_geodesic, 4)
                if self.budget_geodesic is not None else None
            ),
            "n_records": self.n_records,
            "per_tool": self.per_tool,
        }


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """GMS-native token and cost tracker with S^1 budget encoding.

    Parameters
    ----------
    store : GMSExpertStore or None
        GMS store for budget plausibility scoring.  If None, the
        ``budget_geodesic`` field in summaries is omitted.
    budget_usd : float
        Budget cap in USD.  Maps to pi/2 on S^1.
    input_price_per_mtok : float
        Price per million input tokens (USD).
    output_price_per_mtok : float
        Price per million output tokens (USD).
    """

    def __init__(
        self,
        store: "GMSExpertStore | None" = None,
        budget_usd: float = 10.0,
        input_price_per_mtok: float = DEFAULT_PRICING["input_per_mtok"],
        output_price_per_mtok: float = DEFAULT_PRICING["output_per_mtok"],
    ):
        self.store = store
        self.budget_usd = budget_usd
        self.input_price_per_mtok = input_price_per_mtok
        self.output_price_per_mtok = output_price_per_mtok
        self._budget_phase = math.pi / 2.0
        self._records: list[CostRecord] = []
        self._cumulative_cost: float = 0.0

    def record(
        self,
        tool_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = None,
    ) -> CostRecord:
        """Record a token consumption event.

        Parameters
        ----------
        tool_name : str
            Tool or operation name.
        input_tokens : int
            Input tokens consumed.
        output_tokens : int
            Output tokens consumed.
        cost_usd : float or None
            Override cost.  If None, computed from token counts and
            pricing rates.

        Returns
        -------
        CostRecord
            The encoded cost record.
        """
        if cost_usd is None:
            cost_usd = (
                (input_tokens / 1_000_000.0) * self.input_price_per_mtok
                + (output_tokens / 1_000_000.0) * self.output_price_per_mtok
            )

        self._cumulative_cost += cost_usd
        phase = _cost_to_phase(self._cumulative_cost, self.budget_usd)
        tension = _tension_from_phases(phase, self._budget_phase)

        rec = CostRecord(
            tool_name=tool_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cumulative_cost_usd=self._cumulative_cost,
            cost_phase=phase,
            budget_phase=self._budget_phase,
            budget_tension=tension,
            within_budget=self._cumulative_cost <= self.budget_usd,
        )
        self._records.append(rec)
        return rec

    def summary(self) -> CostSummary:
        """Generate an aggregate cost summary.

        Returns
        -------
        CostSummary
            Aggregate token/cost statistics with GMS budget scores.
        """
        if not self._records:
            return CostSummary(budget_usd=self.budget_usd)

        total_in = sum(r.input_tokens for r in self._records)
        total_out = sum(r.output_tokens for r in self._records)
        total_cost = self._cumulative_cost

        # Final budget tension
        final_phase = _cost_to_phase(total_cost, self.budget_usd)
        budget_tension = _tension_from_phases(final_phase, self._budget_phase)

        # Per-tool breakdown
        per_tool: dict[str, dict] = {}
        tool_groups: dict[str, list[CostRecord]] = {}
        for r in self._records:
            tool_groups.setdefault(r.tool_name, []).append(r)

        for tname, recs in tool_groups.items():
            per_tool[tname] = {
                "n_calls": len(recs),
                "input_tokens": sum(r.input_tokens for r in recs),
                "output_tokens": sum(r.output_tokens for r in recs),
                "cost_usd": round(sum(r.cost_usd for r in recs), 6),
            }

        # GMS budget plausibility
        geodesic = None
        if self.store is not None:
            try:
                score = self.store.score_triple(
                    "total_cost", "within", "budget",
                )
                geodesic = float(score) if score is not None else None
            except Exception:
                pass

        return CostSummary(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_tokens=total_in + total_out,
            total_cost_usd=total_cost,
            budget_usd=self.budget_usd,
            budget_utilization=(
                total_cost / self.budget_usd if self.budget_usd > 0 else 0.0
            ),
            budget_tension=budget_tension,
            budget_geodesic=geodesic,
            n_records=len(self._records),
            per_tool=per_tool,
        )

    @property
    def records(self) -> list[CostRecord]:
        """Return all cost records."""
        return list(self._records)

    @property
    def cumulative_cost(self) -> float:
        """Current cumulative cost in USD."""
        return self._cumulative_cost

    @property
    def within_budget(self) -> bool:
        """Whether cumulative cost is within budget."""
        return self._cumulative_cost <= self.budget_usd

    def reset(self) -> None:
        """Clear all records and reset cumulative cost."""
        self._records.clear()
        self._cumulative_cost = 0.0
