# SPDX-License-Identifier: Apache-2.0
"""GMSH Live Dashboard — real-time monitoring via tracing store queries.

All dashboard data comes from querying the tracing store's triple graph —
no separate data pipeline.  The dashboard periodically queries
``tracing.store.query_triples()`` for live run state.

Provides:
- ``DashboardServer``: WebSocket server that polls the tracing store
  and pushes updates to connected clients.
- ``DashboardSnapshot``: point-in-time snapshot of run metrics derived
  from GMS triples.

The WebSocket server is optional and requires the ``websockets`` package.
For non-interactive use, ``DashboardSnapshot.from_tracing()`` can be
called directly.

Usage::

    # Non-interactive snapshot
    snapshot = DashboardSnapshot.from_tracing(tracing)
    print(snapshot.to_dict())

    # WebSocket server
    server = DashboardServer(tracing, port=8765)
    server.start()  # blocks until interrupted
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Dashboard snapshot
# ---------------------------------------------------------------------------

@dataclass
class DashboardSnapshot:
    """Point-in-time snapshot of run metrics from the tracing store.

    All fields are derived from querying GMS triples — no separate
    data pipeline.

    Attributes
    ----------
    timestamp : str
        ISO timestamp of snapshot.
    active_runs : list[str]
        Run IDs currently in progress.
    completed_runs : list[str]
        Run IDs that have finished.
    total_questions : int
        Total questions evaluated across all runs.
    total_correct : int
        Total correct answers.
    accuracy : float
        Overall accuracy.
    recent_events : list[dict]
        Last N events from the tracing store.
    gateway_stats : dict
        Gateway decision counts by action type.
    performance_stats : dict
        Latency and SLA summary (if profiling enabled).
    cost_stats : dict
        Token and budget summary (if cost tracking enabled).
    store_stats : dict
        Tracing store size metrics.
    """

    timestamp: str = ""
    active_runs: list[str] = field(default_factory=list)
    completed_runs: list[str] = field(default_factory=list)
    total_questions: int = 0
    total_correct: int = 0
    accuracy: float = 0.0
    recent_events: list[dict] = field(default_factory=list)
    gateway_stats: dict = field(default_factory=dict)
    performance_stats: dict = field(default_factory=dict)
    cost_stats: dict = field(default_factory=dict)
    store_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "active_runs": self.active_runs,
            "completed_runs": self.completed_runs,
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "accuracy": round(self.accuracy, 4),
            "recent_events": self.recent_events,
            "gateway_stats": self.gateway_stats,
            "performance_stats": self.performance_stats,
            "cost_stats": self.cost_stats,
            "store_stats": self.store_stats,
        }

    @classmethod
    def from_tracing(cls, tracing: "DOETracingGMS") -> "DashboardSnapshot":
        """Build a snapshot by querying the tracing store.

        Parameters
        ----------
        tracing : DOETracingGMS
            The tracing store to query.

        Returns
        -------
        DashboardSnapshot
            Current state derived from GMS triples.
        """
        store = tracing.store

        # Query run-level triples
        active_runs: list[str] = []
        completed_runs: list[str] = []
        total_q = 0
        total_c = 0

        try:
            run_triples = store.query_triples("doe_run", limit=100)
            for triple in run_triples:
                run_id = triple.get("head", "")
                rel = triple.get("relation", "")
                tail = triple.get("tail", "")
                if rel == "accuracy":
                    completed_runs.append(run_id)
                    try:
                        total_c += int(float(tail) * total_q)
                    except (ValueError, TypeError):
                        pass
                elif rel == "total_questions":
                    try:
                        total_q += int(float(tail))
                    except (ValueError, TypeError):
                        pass
                elif rel == "started_at" and run_id not in completed_runs:
                    active_runs.append(run_id)
        except Exception:
            pass

        accuracy = total_c / total_q if total_q > 0 else 0.0

        # Gateway stats
        gw_stats: dict[str, int] = {}
        try:
            gw_triples = store.query_triples("gateway_decision", limit=200)
            for triple in gw_triples:
                if triple.get("relation") == "action":
                    action = triple.get("tail", "unknown")
                    gw_stats[action] = gw_stats.get(action, 0) + 1
        except Exception:
            pass

        # Store-level stats
        store_stats = {}
        try:
            store_stats = tracing.stats()
        except Exception:
            pass

        return cls(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            active_runs=active_runs,
            completed_runs=completed_runs,
            total_questions=total_q,
            total_correct=total_c,
            accuracy=accuracy,
            gateway_stats=gw_stats,
            store_stats=store_stats,
        )


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

class DashboardServer:
    """WebSocket server that pushes live run metrics to clients.

    Polls the tracing store at a configurable interval and broadcasts
    ``DashboardSnapshot`` to all connected WebSocket clients.

    Requires the ``websockets`` package (optional dependency).

    Parameters
    ----------
    tracing : DOETracingGMS
        Tracing store to query for live data.
    port : int
        WebSocket server port (default 8765).
    poll_interval_s : float
        Seconds between tracing store polls (default 2.0).
    """

    def __init__(
        self,
        tracing: "DOETracingGMS",
        port: int = 8765,
        poll_interval_s: float = 2.0,
    ):
        self.tracing = tracing
        self.port = port
        self.poll_interval_s = poll_interval_s
        self._clients: set = set()

    def start(self) -> None:
        """Start the WebSocket server (blocking).

        Raises
        ------
        ImportError
            If ``websockets`` is not installed.
        """
        try:
            import asyncio

            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for DashboardServer. "
                "Install with: pip install websockets"
            )

        async def _handler(websocket, path):
            self._clients.add(websocket)
            try:
                async for _msg in websocket:
                    pass  # clients don't send messages
            finally:
                self._clients.discard(websocket)

        async def _broadcast():
            while True:
                snapshot = DashboardSnapshot.from_tracing(self.tracing)
                data = json.dumps(snapshot.to_dict())
                if self._clients:
                    disconnected = set()
                    for client in self._clients:
                        try:
                            await client.send(data)
                        except Exception:
                            disconnected.add(client)
                    self._clients -= disconnected
                await asyncio.sleep(self.poll_interval_s)

        async def _main():
            async with websockets.serve(_handler, "0.0.0.0", self.port):
                print(f"  [Dashboard] WebSocket server on ws://0.0.0.0:{self.port}")
                await _broadcast()

        asyncio.run(_main())

    def snapshot(self) -> DashboardSnapshot:
        """Get a non-async snapshot (for programmatic use)."""
        return DashboardSnapshot.from_tracing(self.tracing)
