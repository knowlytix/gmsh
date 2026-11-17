# SPDX-License-Identifier: Apache-2.0
"""Run State Machine for the GMSH agentic testing harness.

Wraps the existing DOEGMSBenchmark lifecycle with formal state tracking,
persistence, and tracing.  RunManager orchestrates transitions through:

    CREATED -> PREPARING -> EXECUTING -> EVALUATING -> COMPLETED

Each run is persisted as a JSON file under ``runs_dir/`` and indexed in
``runs_dir/index.json``.  Every state transition is validated against
``VALID_TRANSITIONS`` and traced via ``DOETracingGMS.trace_state_transition()``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark import DOEGMSBenchmark
    from .config import DOEHarnessConfig
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Run state enum
# ---------------------------------------------------------------------------

class RunState(str, Enum):
    """Lifecycle states for a benchmark run."""

    CREATED = "created"
    PREPARING = "preparing"        # ingesting, generating questions
    EXECUTING = "executing"        # running SUT evaluations
    EVALUATING = "evaluating"      # scoring, typed verdicts, audit
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.PREPARING, RunState.ABORTED},
    RunState.PREPARING: {RunState.EXECUTING, RunState.FAILED, RunState.ABORTED},
    RunState.EXECUTING: {RunState.EVALUATING, RunState.FAILED, RunState.ABORTED},
    RunState.EVALUATING: {RunState.COMPLETED, RunState.FAILED, RunState.ABORTED},
    RunState.COMPLETED: set(),
    RunState.FAILED: set(),
    RunState.ABORTED: set(),
}


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Persistent record of a single benchmark run."""

    run_id: str = ""
    state: RunState = RunState.CREATED
    config_snapshot: dict = field(default_factory=dict)

    # Timing
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    # State history: list of (timestamp, from_state, to_state)
    state_history: list[tuple[str, str, str]] = field(default_factory=list)

    # Results (populated during / after run)
    result_path: str = ""
    n_questions: int = 0
    n_correct: int = 0
    accuracy: float = 0.0

    # Error info
    error: str = ""
    error_state: str = ""

    # Scenario references
    scenario_ids: list[str] = field(default_factory=list)

    # Black-box testing enhancements
    target_id: str = ""        # TargetProfile reference
    campaign_id: str = ""      # Campaign reference

    def __post_init__(self):
        if not self.run_id:
            date_str = time.strftime("%Y%m%d")
            short_uuid = uuid.uuid4().hex[:8]
            self.run_id = f"run_{date_str}_{short_uuid}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the record to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "state": self.state.value if isinstance(self.state, RunState) else self.state,
            "config_snapshot": self.config_snapshot,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "state_history": [list(entry) for entry in self.state_history],
            "result_path": self.result_path,
            "n_questions": self.n_questions,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "error": self.error,
            "error_state": self.error_state,
            "scenario_ids": self.scenario_ids,
            "target_id": self.target_id,
            "campaign_id": self.campaign_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        """Deserialize a record from a dict."""
        state_raw = d.get("state", "created")
        try:
            state = RunState(state_raw)
        except ValueError:
            state = RunState.CREATED

        history_raw = d.get("state_history", [])
        state_history = [tuple(entry) for entry in history_raw]

        return cls(
            run_id=d.get("run_id", ""),
            state=state,
            config_snapshot=d.get("config_snapshot", {}),
            created_at=d.get("created_at", ""),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
            state_history=state_history,
            result_path=d.get("result_path", ""),
            n_questions=d.get("n_questions", 0),
            n_correct=d.get("n_correct", 0),
            accuracy=d.get("accuracy", 0.0),
            error=d.get("error", ""),
            error_state=d.get("error_state", ""),
            scenario_ids=d.get("scenario_ids", []),
            target_id=d.get("target_id", ""),
            campaign_id=d.get("campaign_id", ""),
        )

    @property
    def is_terminal(self) -> bool:
        """True if the run has reached a terminal state."""
        return self.state in {RunState.COMPLETED, RunState.FAILED, RunState.ABORTED}


# ---------------------------------------------------------------------------
# Run manager
# ---------------------------------------------------------------------------

class RunManager:
    """Manages the lifecycle of benchmark runs with state tracking.

    Parameters
    ----------
    runs_dir : str
        Directory where run record JSON files are stored.
    tracing : DOETracingGMS or None
        If provided, state transitions are traced as GMS triples.
    """

    INDEX_FILE = "index.json"

    def __init__(
        self,
        runs_dir: str,
        tracing: "DOETracingGMS | None" = None,
    ):
        self.runs_dir = runs_dir
        self.tracing = tracing
        self._index: dict[str, str] = {}   # run_id -> json filename

        os.makedirs(runs_dir, exist_ok=True)
        self._load_index()

    # ----- Public API -----------------------------------------------------

    def create_run(self, config: "DOEHarnessConfig") -> RunRecord:
        """Create a new run record in CREATED state.

        Parameters
        ----------
        config : DOEHarnessConfig
            The benchmark configuration.  Snapshotted as a dict for
            reproducibility.

        Returns
        -------
        RunRecord
            The freshly created record.
        """
        record = RunRecord(
            config_snapshot=dataclasses.asdict(config),
        )
        self._persist(record)
        return record

    def transition(self, run_id: str, to_state: RunState) -> RunRecord:
        """Transition a run to a new state.

        Parameters
        ----------
        run_id : str
            The run to transition.
        to_state : RunState
            The target state.

        Returns
        -------
        RunRecord
            The updated record.

        Raises
        ------
        ValueError
            If the run does not exist or the transition is invalid.
        """
        record = self.get_run(run_id)
        if record is None:
            raise ValueError(f"Run not found: {run_id}")

        from_state = record.state
        allowed = VALID_TRANSITIONS.get(from_state, set())
        if to_state not in allowed:
            raise ValueError(
                f"Invalid transition: {from_state.value} -> {to_state.value} "
                f"(allowed: {sorted(s.value for s in allowed)})"
            )

        # Record transition
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        record.state_history.append((timestamp, from_state.value, to_state.value))
        record.state = to_state

        # Update timing fields
        if to_state == RunState.PREPARING:
            record.started_at = timestamp
        if to_state in {RunState.COMPLETED, RunState.FAILED, RunState.ABORTED}:
            record.completed_at = timestamp

        # Trace via GMS
        if self.tracing:
            self.tracing.trace_state_transition(
                run_id=run_id,
                from_state=from_state.value,
                to_state=to_state.value,
            )

        self._persist(record)
        return record

    def execute_run(
        self,
        run_id: str,
        benchmark: "DOEGMSBenchmark",
        evaluator=None,
        llm_client=None,
        model: str = "",
    ) -> RunRecord:
        """Execute the full benchmark lifecycle with state transitions.

        Wraps the existing ``DOEGMSBenchmark`` pipeline:

        1. PREPARING  -- ingest + generate questions
        2. EXECUTING  -- run the SUT evaluations
        3. EVALUATING -- analyze results, persist
        4. COMPLETED  -- done

        Any unhandled exception triggers a transition to FAILED with the
        error message captured on the record.

        Parameters
        ----------
        run_id : str
            Run to execute (must be in CREATED state).
        benchmark : DOEGMSBenchmark
            The benchmark instance (not modified).
        evaluator : callable or None
            Evaluator function ``(question, context) -> answer``.
        llm_client : object or None
            Anthropic client for direct LLM calls.
        model : str
            Model name when using ``llm_client``.

        Returns
        -------
        RunRecord
            The final record (COMPLETED or FAILED).
        """
        record = self.get_run(run_id)
        if record is None:
            raise ValueError(f"Run not found: {run_id}")

        try:
            # -- PREPARING: ingest + generate questions ----------------------
            self.transition(run_id, RunState.PREPARING)
            benchmark.ingest()
            benchmark.generate_questions()

            # -- EXECUTING: run the SUT evaluations --------------------------
            self.transition(run_id, RunState.EXECUTING)
            result = benchmark.run(evaluator=evaluator, llm_client=llm_client, model=model)

            # -- EVALUATING: analyze results, persist ------------------------
            self.transition(run_id, RunState.EVALUATING)
            benchmark.analyze(result)

            # Save the benchmark result alongside the run record
            result_filename = f"{run_id}_result.json"
            result_path = os.path.join(self.runs_dir, result_filename)
            result.save(result_path)

            # Update record with result summary
            record = self.get_run(run_id)
            record.result_path = result_path
            record.n_questions = result.total_questions
            record.n_correct = result.total_correct
            record.accuracy = result.accuracy
            self._persist(record)

            # -- COMPLETED ---------------------------------------------------
            record = self.transition(run_id, RunState.COMPLETED)

        except Exception as exc:
            # Capture error details on the record
            record = self.get_run(run_id)
            if record is not None and not record.is_terminal:
                record.error = f"{type(exc).__name__}: {exc}"
                record.error_state = record.state.value
                self._persist(record)
                try:
                    record = self.transition(run_id, RunState.FAILED)
                except ValueError:
                    # Already in a terminal state — just persist the error
                    record.state = RunState.FAILED
                    record.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._persist(record)

        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a run record by ID.

        Parameters
        ----------
        run_id : str
            The run identifier.

        Returns
        -------
        RunRecord or None
            The record, or None if not found.
        """
        filename = self._index.get(run_id)
        if filename is None:
            return None

        path = os.path.join(self.runs_dir, filename)
        if not os.path.isfile(path):
            return None

        with open(path) as f:
            data = json.load(f)
        return RunRecord.from_dict(data)

    def list_runs(self, state: RunState | None = None) -> list[RunRecord]:
        """List all runs, optionally filtered by state.

        Parameters
        ----------
        state : RunState or None
            If provided, only return runs in this state.

        Returns
        -------
        list[RunRecord]
            Run records sorted by created_at (oldest first).
        """
        records: list[RunRecord] = []
        for run_id in self._index:
            record = self.get_run(run_id)
            if record is None:
                continue
            if state is not None and record.state != state:
                continue
            records.append(record)
        records.sort(key=lambda r: r.created_at)
        return records

    def get_latest(self) -> RunRecord | None:
        """Return the most recently created run, or None.

        Returns
        -------
        RunRecord or None
        """
        all_runs = self.list_runs()
        return all_runs[-1] if all_runs else None

    # ----- Internal helpers -----------------------------------------------

    def _persist(self, record: RunRecord):
        """Save a run record to its JSON file and update the index."""
        filename = f"{record.run_id}.json"
        path = os.path.join(self.runs_dir, filename)
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        self._index[record.run_id] = filename
        self._save_index()

    def _load_index(self) -> dict:
        """Load the run index from disk."""
        index_path = os.path.join(self.runs_dir, self.INDEX_FILE)
        if os.path.isfile(index_path):
            with open(index_path) as f:
                data = json.load(f)
            self._index = data.get("runs", {})
        else:
            self._index = {}
        return self._index

    def _save_index(self):
        """Persist the run index to disk."""
        index_path = os.path.join(self.runs_dir, self.INDEX_FILE)
        data = {
            "runs": self._index,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)
