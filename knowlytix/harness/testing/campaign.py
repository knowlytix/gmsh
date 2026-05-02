# SPDX-License-Identifier: Apache-2.0
"""GMSH Campaign — multi-run evaluation orchestration.

A **campaign** groups multiple benchmark runs under a single evaluation
purpose:

- **Release validation** — run the full suite against a candidate version
- **Regression sweep** — re-run failure-derived scenarios
- **Adversarial sweep** — targeted adversarial test suites
- **Model comparison** — same suite against multiple SUT versions
- **Compliance certification** — formal audit-quality evidence gathering

Each campaign resolves scenarios (by tag or ID), creates runs via
:class:`~knowlytix.harness.testing.runs.RunManager`, and aggregates results into
campaign-level metrics and reporting.

Usage::

    mgr = CampaignManager(
        campaigns_dir="campaigns",
        runs_dir="runs",
        run_manager=run_manager,
        scenario_registry=registry,
    )
    config = CampaignConfig.from_yaml("release_v2.yaml")
    campaign_id = mgr.create_campaign(config)
    result = mgr.execute_campaign(campaign_id, benchmark_factory)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .runs import RunManager
    from .scenarios import ScenarioRegistry
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Campaign type
# ---------------------------------------------------------------------------

class CampaignType(str, Enum):
    """Purpose classification for a campaign."""

    RELEASE_VALIDATION = "release_validation"
    REGRESSION_SWEEP = "regression_sweep"
    ADVERSARIAL_SWEEP = "adversarial_sweep"
    MODEL_COMPARISON = "model_comparison"
    COMPLIANCE_CERTIFICATION = "compliance_certification"


# ---------------------------------------------------------------------------
# Campaign configuration
# ---------------------------------------------------------------------------

@dataclass
class CampaignConfig:
    """Configuration for a test campaign.

    Attributes
    ----------
    campaign_id : str
        Unique identifier (auto-generated as ``cmp_<uuid8>`` if empty).
    name : str
        Human-readable campaign name.
    description : str
        What this campaign tests and why.
    campaign_type : CampaignType
        Purpose classification.
    target_profiles : list[str]
        Target profile IDs or YAML paths to test against.
    scenario_tags : list[str]
        Tags used to resolve scenarios from the registry.
    scenario_ids : list[str]
        Explicit scenario IDs to include.
    risk_tier : str
        Risk tier for release gate evaluation.
    n_runs : int
        Number of DOE runs per target.
    max_parallel : int
        Maximum parallel runs (1 = sequential).
    stop_on_critical : bool
        If True, abort the campaign on the first critical failure.
    base_config : dict
        DOEHarnessConfig as a dict (merged with per-run overrides).
    metadata : dict
        Arbitrary key-value metadata.
    created_at : str
        ISO-ish creation timestamp.
    """

    campaign_id: str = ""
    name: str = ""
    description: str = ""
    campaign_type: CampaignType = CampaignType.RELEASE_VALIDATION
    target_profiles: list[str] = field(default_factory=list)
    scenario_tags: list[str] = field(default_factory=list)
    scenario_ids: list[str] = field(default_factory=list)
    risk_tier: str = "standard"
    n_runs: int = 2
    max_parallel: int = 1
    stop_on_critical: bool = False
    base_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.campaign_id:
            self.campaign_id = f"cmp_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(self.campaign_type, str):
            self.campaign_type = CampaignType(self.campaign_type)

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "description": self.description,
            "campaign_type": self.campaign_type.value,
            "target_profiles": list(self.target_profiles),
            "scenario_tags": list(self.scenario_tags),
            "scenario_ids": list(self.scenario_ids),
            "risk_tier": self.risk_tier,
            "n_runs": self.n_runs,
            "max_parallel": self.max_parallel,
            "stop_on_critical": self.stop_on_critical,
            "base_config": dict(self.base_config),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CampaignConfig:
        d = dict(d)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> CampaignConfig:
        """Load campaign configuration from YAML.

        The YAML may have a top-level ``campaign`` key or be flat::

            campaign:
              name: "Release v2.1 validation"
              campaign_type: release_validation
              target_profiles: ["tgt_abc123"]
              scenario_tags: ["compliance", "boundary"]
              risk_tier: action_taking
              n_runs: 2
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "campaign" in raw and isinstance(raw["campaign"], dict):
            raw = raw["campaign"]
        return cls.from_dict(raw)

    def to_yaml(self, path: str) -> None:
        """Save campaign configuration to YAML."""
        with open(path, "w") as f:
            yaml.safe_dump(
                {"campaign": self.to_dict()}, f,
                default_flow_style=False, sort_keys=False,
            )


# ---------------------------------------------------------------------------
# Campaign result
# ---------------------------------------------------------------------------

@dataclass
class CampaignRunSummary:
    """Summary of a single run within a campaign.

    Attributes
    ----------
    run_id : str
        Run identifier.
    target_id : str
        Which SUT target this run tested.
    n_questions : int
        Number of questions evaluated.
    n_correct : int
        Number of correct answers.
    accuracy : float
        Accuracy percentage.
    geo_confidence : float
        Mean geometric confidence.
    critical_failures : int
        Count of critical-severity failures.
    elapsed_s : float
        Wall-clock seconds.
    """

    run_id: str = ""
    target_id: str = ""
    n_questions: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    geo_confidence: float = 0.0
    critical_failures: int = 0
    elapsed_s: float = 0.0


@dataclass
class CampaignResult:
    """Aggregated results for a completed campaign.

    Attributes
    ----------
    campaign_id : str
        Campaign identifier.
    campaign_type : str
        Campaign type value.
    run_summaries : list[CampaignRunSummary]
        Per-run summaries.
    aggregate_accuracy : float
        Mean accuracy across all runs.
    aggregate_geo_confidence : float
        Mean geometric confidence across all runs.
    total_questions : int
        Total questions evaluated across all runs.
    total_correct : int
        Total correct across all runs.
    critical_failure_count : int
        Total critical failures across all runs.
    failure_summary : dict[str, int]
        Failure code → count across all runs.
    per_target_accuracy : dict[str, float]
        Target ID → mean accuracy (for model comparison campaigns).
    gate_passed : bool or None
        Overall release gate result (None if not evaluated).
    started_at : str
        Campaign start time.
    completed_at : str
        Campaign end time.
    elapsed_s : float
        Total wall-clock seconds.
    """

    campaign_id: str = ""
    campaign_type: str = ""
    run_summaries: list[CampaignRunSummary] = field(default_factory=list)
    aggregate_accuracy: float = 0.0
    aggregate_geo_confidence: float = 0.0
    total_questions: int = 0
    total_correct: int = 0
    critical_failure_count: int = 0
    failure_summary: dict[str, int] = field(default_factory=dict)
    per_target_accuracy: dict[str, float] = field(default_factory=dict)
    gate_passed: bool | None = None
    started_at: str = ""
    completed_at: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "campaign_type": self.campaign_type,
            "run_summaries": [
                {
                    "run_id": s.run_id,
                    "target_id": s.target_id,
                    "n_questions": s.n_questions,
                    "n_correct": s.n_correct,
                    "accuracy": round(s.accuracy, 4),
                    "geo_confidence": round(s.geo_confidence, 4),
                    "critical_failures": s.critical_failures,
                    "elapsed_s": round(s.elapsed_s, 2),
                }
                for s in self.run_summaries
            ],
            "aggregate_accuracy": round(self.aggregate_accuracy, 4),
            "aggregate_geo_confidence": round(self.aggregate_geo_confidence, 4),
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "critical_failure_count": self.critical_failure_count,
            "failure_summary": dict(self.failure_summary),
            "per_target_accuracy": {
                k: round(v, 4) for k, v in self.per_target_accuracy.items()
            },
            "gate_passed": self.gate_passed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# Campaign manager
# ---------------------------------------------------------------------------

class CampaignManager:
    """Orchestrate multi-run evaluation campaigns.

    Parameters
    ----------
    campaigns_dir : str
        Directory for campaign configuration and result persistence.
    run_manager : RunManager
        Manages individual run lifecycles.
    scenario_registry : ScenarioRegistry or None
        For resolving scenario_tags to scenario lists.
    tracing : DOETracingGMS or None
        For tracing campaign events as GMS triples.
    """

    INDEX_FILE = "index.json"

    def __init__(
        self,
        campaigns_dir: str,
        run_manager: RunManager,
        scenario_registry: ScenarioRegistry | None = None,
        tracing: DOETracingGMS | None = None,
    ):
        self.campaigns_dir = campaigns_dir
        self.run_manager = run_manager
        self.scenario_registry = scenario_registry
        self.tracing = tracing
        self._index: dict[str, str] = {}  # campaign_id -> yaml filename

        os.makedirs(campaigns_dir, exist_ok=True)
        self._load_index()

    # -- Public API -----------------------------------------------------------

    def create_campaign(self, config: CampaignConfig) -> str:
        """Create and persist a new campaign.

        Parameters
        ----------
        config : CampaignConfig
            Campaign configuration.

        Returns
        -------
        str
            The campaign ID.
        """
        # Persist config
        filename = f"{config.campaign_id}.yaml"
        config.to_yaml(os.path.join(self.campaigns_dir, filename))
        self._index[config.campaign_id] = filename
        self._save_index()

        # Trace
        if self.tracing:
            try:
                self.tracing.store.learn_triples([
                    (config.campaign_id, "is_type", "test_campaign"),
                    (config.campaign_id, "has_campaign_type",
                     config.campaign_type.value),
                ])
            except Exception:
                pass

        return config.campaign_id

    def get_campaign(self, campaign_id: str) -> CampaignConfig | None:
        """Load a campaign configuration by ID."""
        filename = self._index.get(campaign_id)
        if filename is None:
            return None
        path = os.path.join(self.campaigns_dir, filename)
        if not os.path.exists(path):
            return None
        return CampaignConfig.from_yaml(path)

    def list_campaigns(self) -> list[CampaignConfig]:
        """List all campaigns."""
        configs = []
        for cid in sorted(self._index.keys()):
            config = self.get_campaign(cid)
            if config is not None:
                configs.append(config)
        return configs

    def execute_campaign(
        self,
        campaign_id: str,
        benchmark_factory: Callable[[dict], Any],
        evaluator_factory: Callable[[Any], Any] | None = None,
    ) -> CampaignResult:
        """Execute all runs in a campaign and aggregate results.

        Parameters
        ----------
        campaign_id : str
            Campaign to execute.
        benchmark_factory : callable
            ``(config_dict) -> DOEGMSBenchmark`` — creates a benchmark
            instance from a configuration dictionary.
        evaluator_factory : callable or None
            ``(benchmark) -> evaluator`` — creates an evaluator for
            each run.  If None, the benchmark's default evaluator is
            used.

        Returns
        -------
        CampaignResult
            Aggregated campaign results.
        """
        config = self.get_campaign(campaign_id)
        if config is None:
            raise ValueError(f"Campaign not found: {campaign_id}")

        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()

        # Resolve scenarios
        scenario_ids = list(config.scenario_ids)
        if config.scenario_tags and self.scenario_registry:
            for tag in config.scenario_tags:
                matches = self.scenario_registry.search_by_tag(tag)
                for scn in matches:
                    if scn.scenario_id not in scenario_ids:
                        scenario_ids.append(scn.scenario_id)

        # Determine targets
        targets = config.target_profiles or ["default"]

        run_summaries: list[CampaignRunSummary] = []
        all_failures: dict[str, int] = {}
        abort = False

        for target_id in targets:
            if abort:
                break

            # Build config for this target
            run_config = dict(config.base_config)
            run_config["n_runs"] = config.n_runs

            # Create and execute benchmark
            try:
                benchmark = benchmark_factory(run_config)
            except Exception:
                run_summaries.append(CampaignRunSummary(
                    target_id=target_id,
                    run_id=f"failed_{uuid.uuid4().hex[:8]}",
                ))
                continue

            evaluator = None
            if evaluator_factory is not None:
                evaluator = evaluator_factory(benchmark)

            # Run the benchmark
            run_t0 = time.time()
            try:
                if hasattr(benchmark, "ingest"):
                    benchmark.ingest(run_config.get("markdown_path", ""))

                if hasattr(benchmark, "generate_questions"):
                    benchmark.generate_questions()

                result = None
                if evaluator is not None and hasattr(benchmark, "run"):
                    result = benchmark.run(evaluator)
                elif hasattr(benchmark, "run"):
                    result = benchmark.run()

                run_elapsed = time.time() - run_t0

                # Extract summary from result
                summary = CampaignRunSummary(
                    target_id=target_id,
                    run_id=getattr(result, "run_id", f"run_{uuid.uuid4().hex[:8]}"),
                    n_questions=getattr(result, "n_questions",
                                        len(getattr(result, "results", []))),
                    n_correct=getattr(result, "n_correct", 0),
                    accuracy=getattr(result, "accuracy", 0.0),
                    geo_confidence=getattr(result, "mean_geo_confidence", 0.0),
                    elapsed_s=run_elapsed,
                )

                # Count failures from result if available
                if hasattr(result, "failure_counts"):
                    for code, count in result.failure_counts.items():
                        all_failures[code] = all_failures.get(code, 0) + count

                run_summaries.append(summary)

                # Trace run completion
                if self.tracing:
                    try:
                        self.tracing.store.learn_triples([
                            (campaign_id, "contains_run", summary.run_id),
                            (summary.run_id, "tested_target", target_id),
                        ])
                    except Exception:
                        pass

                # Stop-on-critical check
                if config.stop_on_critical and summary.critical_failures > 0:
                    abort = True

            except Exception:
                run_elapsed = time.time() - run_t0
                run_summaries.append(CampaignRunSummary(
                    target_id=target_id,
                    run_id=f"error_{uuid.uuid4().hex[:8]}",
                    elapsed_s=run_elapsed,
                ))

        # Aggregate results
        completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        total_elapsed = time.time() - t0

        total_questions = sum(s.n_questions for s in run_summaries)
        total_correct = sum(s.n_correct for s in run_summaries)
        total_critical = sum(s.critical_failures for s in run_summaries)

        accuracies = [s.accuracy for s in run_summaries if s.n_questions > 0]
        geo_confs = [s.geo_confidence for s in run_summaries if s.n_questions > 0]

        # Per-target aggregation
        per_target: dict[str, list[float]] = {}
        for s in run_summaries:
            if s.n_questions > 0:
                per_target.setdefault(s.target_id, []).append(s.accuracy)
        per_target_accuracy = {
            tid: sum(accs) / len(accs)
            for tid, accs in per_target.items()
        }

        if not accuracies:
            agg_accuracy = 0.0
            agg_geo_conf = 0.0
            print(f"  [Campaign] WARNING: No runs produced results "
                  f"(all {len(run_summaries)} runs had n_questions=0)")
        else:
            agg_accuracy = sum(accuracies) / len(accuracies)
            agg_geo_conf = sum(geo_confs) / len(geo_confs)

        campaign_result = CampaignResult(
            campaign_id=campaign_id,
            campaign_type=config.campaign_type.value,
            run_summaries=run_summaries,
            aggregate_accuracy=agg_accuracy,
            aggregate_geo_confidence=agg_geo_conf,
            total_questions=total_questions,
            total_correct=total_correct,
            critical_failure_count=total_critical,
            failure_summary=all_failures,
            per_target_accuracy=per_target_accuracy,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_s=total_elapsed,
        )

        # Persist result
        result_path = os.path.join(
            self.campaigns_dir, f"{campaign_id}_result.json",
        )
        with open(result_path, "w") as f:
            json.dump(campaign_result.to_dict(), f, indent=2)

        return campaign_result

    def compare_campaigns(
        self, id_a: str, id_b: str,
    ) -> dict:
        """Compare two campaign results.

        Parameters
        ----------
        id_a : str
            First campaign ID (baseline).
        id_b : str
            Second campaign ID (candidate).

        Returns
        -------
        dict
            Comparison with accuracy_delta, geo_confidence_delta,
            per_target deltas, and regression flags.
        """
        result_a = self._load_result(id_a)
        result_b = self._load_result(id_b)

        if result_a is None or result_b is None:
            return {"error": "One or both campaign results not found."}

        acc_delta = result_b["aggregate_accuracy"] - result_a["aggregate_accuracy"]
        geo_delta = (
            result_b["aggregate_geo_confidence"]
            - result_a["aggregate_geo_confidence"]
        )

        # Per-target comparison
        target_deltas = {}
        all_targets = set(result_a.get("per_target_accuracy", {}).keys()) | \
                      set(result_b.get("per_target_accuracy", {}).keys())
        for tid in all_targets:
            a_acc = result_a.get("per_target_accuracy", {}).get(tid, 0.0)
            b_acc = result_b.get("per_target_accuracy", {}).get(tid, 0.0)
            target_deltas[tid] = round(b_acc - a_acc, 4)

        return {
            "baseline": id_a,
            "candidate": id_b,
            "accuracy_delta": round(acc_delta, 4),
            "geo_confidence_delta": round(geo_delta, 4),
            "per_target_deltas": target_deltas,
            "regression_detected": acc_delta < -0.05,
            "baseline_accuracy": result_a["aggregate_accuracy"],
            "candidate_accuracy": result_b["aggregate_accuracy"],
        }

    # -- Internal helpers -----------------------------------------------------

    def _load_index(self) -> None:
        """Load the campaign index from disk."""
        index_path = os.path.join(self.campaigns_dir, self.INDEX_FILE)
        if os.path.exists(index_path):
            with open(index_path) as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        """Persist the campaign index to disk."""
        index_path = os.path.join(self.campaigns_dir, self.INDEX_FILE)
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _load_result(self, campaign_id: str) -> dict | None:
        """Load a campaign result JSON."""
        path = os.path.join(
            self.campaigns_dir, f"{campaign_id}_result.json",
        )
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
