# SPDX-License-Identifier: Apache-2.0
"""GMSH Replay & Regression — deterministic replay and run comparison (WS5).

Provides:

- **ReplayManifest**: frozen inputs for deterministic re-execution of a
  benchmark run (config, snapshot, questions, seed).
- **RunComparison**: per-question diff between two DOE benchmark results
  with aggregate deltas and GMS semantic-drift measurement.
- **RegressionSuite**: persistent collection of failure-derived test
  scenarios that guard against regressions.
- **compare_runs()**: build a ``RunComparison`` from two result files.
- **replay_run()**: restore a snapshot, re-run the benchmark, and
  optionally compare against a reference result.

Storage layout::

    suite_dir/
        suite.json          # name, scenario_ids, created_at
    manifest.yaml           # ReplayManifest (portable)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore
    from knowlytix.harness.graphdoe.results import DOEBenchmarkResult, DOEQuestionResult
    from .snapshots import SnapshotManager
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# ReplayManifest
# ---------------------------------------------------------------------------

@dataclass
class ReplayManifest:
    """Frozen inputs that allow a benchmark run to be replayed exactly."""

    manifest_id: str = ""
    source_run_id: str = ""

    # Frozen inputs
    config_snapshot: dict = field(default_factory=dict)
    snapshot_id: str = ""
    doe_questions_path: str = ""      # path to serialised question list

    # Execution context
    fault_profiles: list[dict] = field(default_factory=list)
    policy_rules: list[dict] = field(default_factory=list)
    random_seed: int = 42

    # Reference results
    reference_result_path: str = ""

    created_at: str = ""

    def __post_init__(self):
        if not self.manifest_id:
            self.manifest_id = f"manifest_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "source_run_id": self.source_run_id,
            "config_snapshot": self.config_snapshot,
            "snapshot_id": self.snapshot_id,
            "doe_questions_path": self.doe_questions_path,
            "fault_profiles": self.fault_profiles,
            "policy_rules": self.policy_rules,
            "random_seed": self.random_seed,
            "reference_result_path": self.reference_result_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayManifest":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_yaml(self, path: str) -> None:
        """Persist the manifest to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False,
                           sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ReplayManifest":
        """Load a manifest from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# RunComparison
# ---------------------------------------------------------------------------

@dataclass
class RunComparison:
    """Per-question diff between two benchmark results."""

    run_a_id: str = ""
    run_b_id: str = ""

    # Aggregate deltas
    accuracy_delta: float = 0.0       # run_b.accuracy - run_a.accuracy
    geo_confidence_delta: float = 0.0
    tension_delta: float = 0.0

    # Per-question diffs
    regressions: list[dict] = field(default_factory=list)    # correct -> wrong
    improvements: list[dict] = field(default_factory=list)   # wrong -> correct
    stable_pass: int = 0
    stable_fail: int = 0

    # GMS semantic drift
    semantic_drift: float = 0.0

    # Failure distribution changes
    failure_delta: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_a_id": self.run_a_id,
            "run_b_id": self.run_b_id,
            "accuracy_delta": self.accuracy_delta,
            "geo_confidence_delta": self.geo_confidence_delta,
            "tension_delta": self.tension_delta,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "stable_pass": self.stable_pass,
            "stable_fail": self.stable_fail,
            "semantic_drift": self.semantic_drift,
            "failure_delta": self.failure_delta,
            "regression_rate": self.regression_rate,
        }

    @property
    def regression_rate(self) -> float:
        """Fraction of total questions that regressed (correct -> wrong)."""
        total = (
            len(self.regressions)
            + len(self.improvements)
            + self.stable_pass
            + self.stable_fail
        )
        if total == 0:
            return 0.0
        return len(self.regressions) / total


# ---------------------------------------------------------------------------
# RegressionSuite
# ---------------------------------------------------------------------------

class RegressionSuite:
    """Persistent collection of failure-derived test scenarios.

    Parameters
    ----------
    suite_dir : str
        Directory where ``suite.json`` is stored.
    scenario_registry : ScenarioRegistry or None
        If provided, promoted scenarios are also added to the registry.
    tracing : DOETracingGMS or None
        If provided, promotions are traced as GMS triples.
    """

    SUITE_FILE = "suite.json"

    def __init__(
        self,
        suite_dir: str,
        scenario_registry=None,
        tracing: "DOETracingGMS | None" = None,
    ):
        self.suite_dir = suite_dir
        self.scenario_registry = scenario_registry
        self.tracing = tracing
        self._scenario_ids: list[str] = []
        self._name: str = ""
        self._created_at: str = ""

        os.makedirs(suite_dir, exist_ok=True)
        self._load_or_init()

    # ----- Public API --------------------------------------------------------

    def promote_failures(
        self,
        run_record,
        result: "DOEBenchmarkResult",
        min_severity: str = "error",
    ) -> list[str]:
        """Promote failed questions from a run to regression scenarios.

        Parameters
        ----------
        run_record : RunRecord
            The run that produced *result*.
        result : DOEBenchmarkResult
            Benchmark result whose failures will be promoted.
        min_severity : str
            Minimum severity threshold (reserved for future filtering).

        Returns
        -------
        list[str]
            Newly created scenario_ids.
        """
        # Lazy import to avoid circular dependencies
        from .scenarios import Scenario

        new_ids: list[str] = []
        for q in result.questions:
            if q.correct:
                continue

            scenario = Scenario.from_failed_run(
                run_id=run_record.run_id,
                question_result=q,
                failure_code="",
            )

            # Register with the scenario registry if available
            if self.scenario_registry is not None:
                try:
                    self.scenario_registry.add(scenario)
                except Exception:
                    pass  # degrade gracefully

            self.add_scenario(scenario.scenario_id)
            new_ids.append(scenario.scenario_id)

            # Trace the promotion
            if self.tracing:
                self.tracing.trace_regression_promotion(
                    run_id=run_record.run_id,
                    scenario_id=scenario.scenario_id,
                    source_question_id=getattr(q, "doe_qid", ""),
                    failure_code="",
                )

        return new_ids

    def add_scenario(self, scenario_id: str) -> None:
        """Add a scenario to the suite (idempotent)."""
        if scenario_id not in self._scenario_ids:
            self._scenario_ids.append(scenario_id)
            self._save()

    def remove_scenario(self, scenario_id: str) -> None:
        """Remove a scenario from the suite."""
        if scenario_id in self._scenario_ids:
            self._scenario_ids.remove(scenario_id)
            self._save()

    def list_scenarios(self) -> list[str]:
        """Return all scenario IDs in the suite."""
        return list(self._scenario_ids)

    # ----- Persistence -------------------------------------------------------

    def _load(self) -> dict:
        """Load suite.json and return its contents."""
        path = os.path.join(self.suite_dir, self.SUITE_FILE)
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _save(self) -> None:
        """Persist suite.json to disk."""
        path = os.path.join(self.suite_dir, self.SUITE_FILE)
        data = {
            "name": self._name,
            "scenario_ids": self._scenario_ids,
            "created_at": self._created_at,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_or_init(self) -> None:
        """Load existing suite or initialise a new one."""
        data = self._load()
        if data:
            self._name = data.get("name", "")
            self._scenario_ids = data.get("scenario_ids", [])
            self._created_at = data.get("created_at", "")
        else:
            self._name = os.path.basename(self.suite_dir)
            self._created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()


# ---------------------------------------------------------------------------
# compare_runs()
# ---------------------------------------------------------------------------

def compare_runs(
    result_a: "DOEBenchmarkResult",
    result_b: "DOEBenchmarkResult",
    store: "GMSExpertStore | None" = None,
) -> RunComparison:
    """Build a per-question diff between two benchmark results.

    Parameters
    ----------
    result_a : DOEBenchmarkResult
        The baseline (earlier) result.
    result_b : DOEBenchmarkResult
        The newer result to compare against the baseline.
    store : GMSExpertStore or None
        If provided, semantic drift is estimated via ``tension_energy``.

    Returns
    -------
    RunComparison
    """

    # Build question maps keyed by base_qid (first result per base_qid wins)
    def _build_map(result: "DOEBenchmarkResult") -> dict[str, "DOEQuestionResult"]:
        qmap: dict = {}
        for q in result.questions:
            if q.base_qid not in qmap:
                qmap[q.base_qid] = q
        return qmap

    map_a = _build_map(result_a)
    map_b = _build_map(result_b)

    shared_qids = sorted(set(map_a.keys()) & set(map_b.keys()))

    regressions: list[dict] = []
    improvements: list[dict] = []
    stable_pass = 0
    stable_fail = 0

    # Track failure category distribution for both runs
    fail_cats_a: dict[str, int] = {}
    fail_cats_b: dict[str, int] = {}

    for qid in shared_qids:
        qa = map_a[qid]
        qb = map_b[qid]

        if qa.correct and not qb.correct:
            regressions.append({
                "base_qid": qid,
                "category": qa.category,
                "question": qa.question,
                "answer_a": qa.answer,
                "answer_b": qb.answer,
                "ground_truth": qa.ground_truth,
            })
        elif not qa.correct and qb.correct:
            improvements.append({
                "base_qid": qid,
                "category": qa.category,
                "question": qa.question,
                "answer_a": qa.answer,
                "answer_b": qb.answer,
                "ground_truth": qa.ground_truth,
            })
        elif qa.correct and qb.correct:
            stable_pass += 1
        else:
            stable_fail += 1

        # Failure distribution
        if not qa.correct:
            cat = qa.category or "unknown"
            fail_cats_a[cat] = fail_cats_a.get(cat, 0) + 1
        if not qb.correct:
            cat = qb.category or "unknown"
            fail_cats_b[cat] = fail_cats_b.get(cat, 0) + 1

    # Failure distribution delta
    all_cats = sorted(set(fail_cats_a.keys()) | set(fail_cats_b.keys()))
    failure_delta = {
        cat: fail_cats_b.get(cat, 0) - fail_cats_a.get(cat, 0)
        for cat in all_cats
    }

    # Semantic drift via GMS tension energy
    semantic_drift = 0.0
    if store is not None:
        tensions: list[float] = []
        for qid in shared_qids[:50]:  # sample for efficiency
            qa = map_a[qid]
            qb = map_b[qid]
            if qa.answer and qb.answer and qa.answer != qb.answer:
                try:
                    t = store.tension_energy(qa.answer, qb.answer)
                    if t is not None:
                        tensions.append(t)
                except Exception:
                    pass  # degrade gracefully without store support
        if tensions:
            semantic_drift = sum(tensions) / len(tensions)

    accuracy_delta = result_b.accuracy - result_a.accuracy

    return RunComparison(
        run_a_id=getattr(result_a, "instance_name", ""),
        run_b_id=getattr(result_b, "instance_name", ""),
        accuracy_delta=accuracy_delta,
        regressions=regressions,
        improvements=improvements,
        stable_pass=stable_pass,
        stable_fail=stable_fail,
        semantic_drift=semantic_drift,
        failure_delta=failure_delta,
    )


# ---------------------------------------------------------------------------
# replay_run()
# ---------------------------------------------------------------------------

def replay_run(
    manifest: ReplayManifest,
    snapshot_manager: "SnapshotManager",
    evaluator=None,
    llm_client=None,
    model: str = "",
    device=None,
) -> tuple["DOEBenchmarkResult", RunComparison | None]:
    """Deterministically replay a benchmark run from a frozen manifest.

    Parameters
    ----------
    manifest : ReplayManifest
        The frozen inputs describing the run to replay.
    snapshot_manager : SnapshotManager
        Manager used to restore the GMS store snapshot.
    evaluator : callable or None
        ``(question, context) -> answer`` evaluator function.
    llm_client : object or None
        Anthropic client for direct LLM calls.
    model : str
        Model name when using *llm_client*.
    device : torch.device or None
        Device for model loading (defaults to CPU inside snapshot_manager).

    Returns
    -------
    tuple[DOEBenchmarkResult, RunComparison | None]
        The replay result, and a comparison against the reference if a
        ``reference_result_path`` was provided in the manifest.
    """
    from knowlytix.harness.graphdoe.results import DOEBenchmarkResult
    from .benchmark import DOEGMSBenchmark
    from .config import DOEHarnessConfig

    # 1. Restore GMS snapshot
    store = snapshot_manager.restore_snapshot(manifest.snapshot_id, device)

    # 2. Reconstruct config from frozen snapshot (filter to known fields only)
    known_fields = {f.name for f in DOEHarnessConfig.__dataclass_fields__.values()}
    safe_config = {
        k: v for k, v in manifest.config_snapshot.items()
        if k in known_fields
    }
    # Override seed for determinism
    safe_config["seed"] = manifest.random_seed
    config = DOEHarnessConfig(**safe_config)

    # 3. Create benchmark with the restored store
    benchmark = DOEGMSBenchmark(config)
    benchmark.store = store

    # 4. Run the benchmark
    result = benchmark.run(evaluator=evaluator, llm_client=llm_client, model=model)

    # 5. Compare against reference if available
    comparison: RunComparison | None = None
    if manifest.reference_result_path and os.path.isfile(manifest.reference_result_path):
        reference = DOEBenchmarkResult.load(manifest.reference_result_path)
        comparison = compare_runs(reference, result, store)

    return result, comparison
