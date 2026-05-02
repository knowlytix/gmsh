# SPDX-License-Identifier: Apache-2.0
"""GMSH Scenario Registry — authored, versioned test scenarios.

Provides a structured way to define, store, version, and search test
scenarios.  Scenarios can be manually authored (with ``question_text``)
or delegate to an existing GMS generator (with ``generator_name``).

Key capabilities:

- **CRUD** — add / get / update / delete with per-scenario YAML files
- **Versioning** — every update increments the version; old versions kept
- **Tag / tier search** — filter by tags or risk tier
- **Geometric search** — when a GMS store is available, scenarios are
  encoded as triples and searched via ``score_triple`` / ``tension_energy``
- **DOE bridge** — convert authored scenarios into ``GeneratedQuestion``
  objects compatible with the graphDOE question expansion pipeline
- **Failure promotion** — promote a failed ``DOEQuestionResult`` into a
  regression scenario with ``Scenario.from_failed_run()``
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import yaml

from knowlytix.benchmark.eval.generators.base import GeneratedQuestion

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Stop condition
# ---------------------------------------------------------------------------

@dataclass
class StopCondition:
    """Condition that can halt scenario execution early.

    Attributes
    ----------
    condition_type : str
        One of ``"max_turns"``, ``"max_latency_ms"``,
        ``"output_contains"``, ``"tool_called"``,
        ``"state_equals"``, ``"error_count"``.
    value : str | int | float
        The threshold or pattern for the condition.
    action : str
        What to do when triggered: ``"abort"`` (stop and fail),
        ``"flag"`` (continue but mark), ``"skip"`` (skip remaining).
    """

    condition_type: str = ""
    value: Any = ""
    action: str = "abort"  # "abort", "flag", "skip"

    @classmethod
    def from_dict(cls, d: dict) -> StopCondition:
        return cls(
            condition_type=d.get("condition_type", ""),
            value=d.get("value", ""),
            action=d.get("action", "abort"),
        )

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "value": self.value,
            "action": self.action,
        }


# ---------------------------------------------------------------------------
# Scenario data model
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A single authored or generated test scenario.

    Attributes
    ----------
    scenario_id : str
        Unique identifier (auto-generated as ``scn_<uuid8>`` if empty).
    name : str
        Short human-readable name.
    description : str
        Longer description of what the scenario tests.
    version : int
        Monotonically increasing version number.
    tags : list[str]
        Freeform tags for filtering (e.g. ``["compliance", "numeric"]``).
    risk_tier : str
        One of ``"advisory"``, ``"recommendation"``, ``"action_taking"``,
        or ``"standard"`` (default).
    generator_name : str
        If set, delegate question generation to this GMS generator
        (e.g. ``"plausibility"``).
    generator_params : dict
        Extra parameters forwarded to the generator.
    question_text : str
        For manually authored scenarios, the natural-language question.
    ground_truth : str
        Expected answer for manually authored scenarios.
    answer_type : str
        Answer type: ``"float"``, ``"int"``, ``"bool"``, ``"set_str"``,
        or ``"str"`` (default).
    metadata : dict
        Arbitrary key-value metadata.
    context_mode : str
        Context presentation mode (``"full"``, ``"section"``, etc.).
    fault_profile_refs : list[str]
        IDs of FaultProfile objects to inject during execution.
    policy_rule_refs : list[str]
        IDs of PolicyRule objects governing the scenario.
    required_tools : list[str]
        Tools the SUT *must* call (for process evaluation).
    forbidden_tools : list[str]
        Tools the SUT must *not* call.
    expected_tool_sequence : list[str]
        Expected order of tool calls.
    max_tool_calls : int
        Budget cap on tool invocations (0 = unlimited).
    expected_recovery : bool
        Whether the scenario expects graceful recovery from a fault.
    created_at : str
        ISO-ish creation timestamp.
    updated_at : str
        ISO-ish last-update timestamp.
    author : str
        Who authored this scenario.
    source_run_id : str
        If promoted from a failed run, the originating run ID.
    source_question_id : str
        If promoted from a failed run, the originating question ID.
    """

    scenario_id: str = ""
    name: str = ""
    description: str = ""
    version: int = 1
    tags: list[str] = field(default_factory=list)
    risk_tier: str = "standard"

    # Question definition (one of these approaches)
    generator_name: str = ""
    generator_params: dict = field(default_factory=dict)
    question_text: str = ""
    ground_truth: str = ""
    answer_type: str = "str"
    metadata: dict = field(default_factory=dict)

    # Execution context
    context_mode: str = "full"
    fault_profile_refs: list[str] = field(default_factory=list)
    policy_rule_refs: list[str] = field(default_factory=list)

    # Behavioral requirements (used by extended evaluators WS6)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_tool_sequence: list[str] = field(default_factory=list)
    max_tool_calls: int = 0
    expected_recovery: bool = False

    # Provenance
    created_at: str = ""
    updated_at: str = ""
    author: str = ""
    source_run_id: str = ""
    source_question_id: str = ""

    # Black-box testing enhancements
    oracle_definition: dict = field(default_factory=dict)
    expected_side_effects: dict = field(default_factory=dict)
    trace_pattern: str = ""
    constraints: list[dict] = field(default_factory=list)
    prohibited_outcomes: list[str] = field(default_factory=list)
    required_evidence_items: list[str] = field(default_factory=list)
    interaction_script: dict = field(default_factory=dict)
    environment_fixture_id: str = ""
    stop_conditions: list[dict] = field(default_factory=list)
    severity_overrides: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self):
        if not self.scenario_id:
            self.scenario_id = f"scn_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict (suitable for YAML / JSON)."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": list(self.tags),
            "risk_tier": self.risk_tier,
            "generator_name": self.generator_name,
            "generator_params": dict(self.generator_params),
            "question_text": self.question_text,
            "ground_truth": self.ground_truth,
            "answer_type": self.answer_type,
            "metadata": dict(self.metadata),
            "context_mode": self.context_mode,
            "fault_profile_refs": list(self.fault_profile_refs),
            "policy_rule_refs": list(self.policy_rule_refs),
            "required_tools": list(self.required_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "expected_tool_sequence": list(self.expected_tool_sequence),
            "max_tool_calls": self.max_tool_calls,
            "expected_recovery": self.expected_recovery,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "source_run_id": self.source_run_id,
            "source_question_id": self.source_question_id,
            "oracle_definition": dict(self.oracle_definition),
            "expected_side_effects": dict(self.expected_side_effects),
            "trace_pattern": self.trace_pattern,
            "constraints": list(self.constraints),
            "prohibited_outcomes": list(self.prohibited_outcomes),
            "required_evidence_items": list(self.required_evidence_items),
            "interaction_script": dict(self.interaction_script),
            "environment_fixture_id": self.environment_fixture_id,
            "stop_conditions": list(self.stop_conditions),
            "severity_overrides": dict(self.severity_overrides),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        """Construct a Scenario from a dict (e.g. parsed YAML).

        Unknown keys are silently ignored so that forward-compatible
        YAML files produced by newer versions do not break older code.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_yaml(self) -> str:
        """Serialize to a YAML string."""
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml_file(cls, path: str) -> Scenario:
        """Load a Scenario from a YAML file on disk.

        Parameters
        ----------
        path : str
            Path to the ``.yaml`` file.

        Returns
        -------
        Scenario
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Promotion from failed runs
    # ------------------------------------------------------------------

    @classmethod
    def from_failed_run(
        cls,
        run_id: str,
        question_result,
        failure_code: str = "",
    ) -> Scenario:
        """Promote a failed DOEQuestionResult into a regression scenario.

        This captures the failing question as an authored scenario so it
        can be included in future regression suites.

        Parameters
        ----------
        run_id : str
            The benchmark run that produced the failure.
        question_result : DOEQuestionResult
            The result object from the failing question.  Expected
            attributes: ``question``, ``ground_truth``, ``category``,
            ``doe_qid``, ``factor_assignment``.
        failure_code : str
            Optional failure code (e.g. ``"hallucination_detected"``).

        Returns
        -------
        Scenario
            A new scenario with question text, ground truth, and
            provenance fields populated from the failed result.
        """
        tags = []
        category = getattr(question_result, "category", "")
        if category:
            tags.append(category)
        if failure_code:
            tags.append(f"failure:{failure_code}")

        answer_type = "str"
        if hasattr(question_result, "factor_assignment"):
            answer_type = question_result.factor_assignment.get(
                "answer_type", "str"
            )

        metadata = {
            "promoted_from_run": run_id,
            "original_doe_qid": getattr(question_result, "doe_qid", ""),
            "failure_code": failure_code,
        }
        factor_assignment = getattr(question_result, "factor_assignment", {})
        if factor_assignment:
            metadata["original_factors"] = dict(factor_assignment)

        return cls(
            name=f"Regression: {category or 'unknown'} from {run_id[:12]}",
            description=(
                f"Auto-promoted from failed question "
                f"{getattr(question_result, 'doe_qid', '?')} "
                f"in run {run_id}."
            ),
            tags=tags,
            question_text=getattr(question_result, "question", ""),
            ground_truth=str(getattr(question_result, "ground_truth", "")),
            answer_type=answer_type,
            metadata=metadata,
            source_run_id=run_id,
            source_question_id=getattr(question_result, "doe_qid", ""),
        )


# ---------------------------------------------------------------------------
# Scenario Registry
# ---------------------------------------------------------------------------

class ScenarioRegistry:
    """Manages a directory of versioned YAML scenario files.

    Scenarios are stored as individual files named
    ``{scenario_id}_v{version}.yaml`` under *scenarios_dir*.  An
    ``index.json`` file tracks every scenario's latest version and
    metadata for fast look-ups without scanning the filesystem.

    Parameters
    ----------
    scenarios_dir : str
        Directory where YAML scenario files are stored.
    store : GMSExpertStore or None
        If provided, scenario metadata is encoded as GMS triples to
        enable geometric search.
    tracing : DOETracingGMS or None
        If provided, CRUD operations are traced.
    """

    def __init__(
        self,
        scenarios_dir: str,
        store: "GMSExpertStore | None" = None,
        tracing: "DOETracingGMS | None" = None,
    ):
        self.scenarios_dir = scenarios_dir
        self.store = store
        self.tracing = tracing
        self._index: dict = {}

        os.makedirs(scenarios_dir, exist_ok=True)
        self._load_index()

    # ==================================================================
    # Internal — index persistence
    # ==================================================================

    def _index_path(self) -> str:
        return os.path.join(self.scenarios_dir, "index.json")

    def _load_index(self) -> dict:
        """Load index.json into ``self._index`` and return it."""
        path = self._index_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                self._index = json.load(fh)
        else:
            self._index = {}
        return self._index

    def _save_index(self):
        """Persist ``self._index`` to index.json."""
        path = self._index_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._index, fh, indent=2, ensure_ascii=False)

    # ==================================================================
    # Internal — GMS triple encoding
    # ==================================================================

    def _encode_scenario_triples(self, scenario: Scenario):
        """Encode scenario metadata as GMS triples for geometric search.

        Encodes:
        - ``(scenario_id, "is_type", "test_scenario")``
        - ``(scenario_id, "tests_risk_tier", risk_tier)``
        - ``(scenario_id, "has_tag", tag)`` for each tag
        - ``(scenario_id, "has_name", name)`` if name is set
        - ``(scenario_id, "uses_generator", generator_name)`` if set
        """
        if self.store is None:
            return

        try:
            from knowlytix.knowledge.extract import ExtractedTriple
            from knowlytix.knowledge.learn import RuntimeLearner
        except ImportError:
            return

        triples = [
            ExtractedTriple(
                head=scenario.scenario_id,
                relation="is_type",
                tail="test_scenario",
            ),
            ExtractedTriple(
                head=scenario.scenario_id,
                relation="tests_risk_tier",
                tail=scenario.risk_tier,
            ),
        ]
        for tag in scenario.tags:
            triples.append(ExtractedTriple(
                head=scenario.scenario_id,
                relation="has_tag",
                tail=tag,
            ))
        if scenario.name:
            triples.append(ExtractedTriple(
                head=scenario.scenario_id,
                relation="has_name",
                tail=scenario.name,
            ))
        if scenario.generator_name:
            triples.append(ExtractedTriple(
                head=scenario.scenario_id,
                relation="uses_generator",
                tail=scenario.generator_name,
            ))

        try:
            learner = RuntimeLearner(self.store, self.store.config.learn)
            import torch
            device = next(
                (p.device for p in self.store.model.parameters()),
                torch.device("cpu"),
            )
            learner.learn_triples(triples, device)
        except Exception as exc:
            print(f"  [scenario-registry] GMS encode warning: {exc}")

    # ==================================================================
    # File naming
    # ==================================================================

    def _yaml_filename(self, scenario_id: str, version: int) -> str:
        return f"{scenario_id}_v{version}.yaml"

    def _yaml_path(self, scenario_id: str, version: int) -> str:
        return os.path.join(
            self.scenarios_dir,
            self._yaml_filename(scenario_id, version),
        )

    # ==================================================================
    # CRUD
    # ==================================================================

    def add(self, scenario: Scenario) -> str:
        """Save a new scenario, update the index, and encode GMS triples.

        Parameters
        ----------
        scenario : Scenario
            The scenario to add.  If ``scenario_id`` is empty it will
            be auto-generated by ``Scenario.__post_init__``.

        Returns
        -------
        str
            The ``scenario_id`` of the saved scenario.
        """
        sid = scenario.scenario_id
        scenario.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Write YAML
        path = self._yaml_path(sid, scenario.version)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(scenario.to_yaml())

        # Update index
        self._index[sid] = {
            "latest_version": scenario.version,
            "name": scenario.name,
            "risk_tier": scenario.risk_tier,
            "tags": list(scenario.tags),
            "updated_at": scenario.updated_at,
        }
        self._save_index()

        # Encode in GMS
        self._encode_scenario_triples(scenario)

        return sid

    def get(self, scenario_id: str) -> Scenario | None:
        """Load the latest version of a scenario by ID.

        Parameters
        ----------
        scenario_id : str
            The scenario identifier.

        Returns
        -------
        Scenario or None
            ``None`` if the scenario is not in the index.
        """
        entry = self._index.get(scenario_id)
        if entry is None:
            return None
        version = entry["latest_version"]
        path = self._yaml_path(scenario_id, version)
        if not os.path.exists(path):
            return None
        return Scenario.from_yaml_file(path)

    def update(self, scenario: Scenario):
        """Increment version, save a new YAML file, and update the index.

        Parameters
        ----------
        scenario : Scenario
            The scenario to update.  Its ``version`` will be incremented
            by one.  The previous version file is kept for history.
        """
        entry = self._index.get(scenario.scenario_id)
        if entry is not None:
            scenario.version = entry["latest_version"] + 1
        else:
            scenario.version = 1

        scenario.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        path = self._yaml_path(scenario.scenario_id, scenario.version)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(scenario.to_yaml())

        self._index[scenario.scenario_id] = {
            "latest_version": scenario.version,
            "name": scenario.name,
            "risk_tier": scenario.risk_tier,
            "tags": list(scenario.tags),
            "updated_at": scenario.updated_at,
        }
        self._save_index()

        self._encode_scenario_triples(scenario)

    def delete(self, scenario_id: str):
        """Remove a scenario from the index (YAML files are kept for history).

        Parameters
        ----------
        scenario_id : str
            The scenario to delete.
        """
        if scenario_id in self._index:
            del self._index[scenario_id]
            self._save_index()

    def list_all(self) -> list[Scenario]:
        """Return every scenario at its current (latest) version.

        Returns
        -------
        list[Scenario]
        """
        scenarios = []
        for sid, entry in self._index.items():
            path = self._yaml_path(sid, entry["latest_version"])
            if os.path.exists(path):
                scenarios.append(Scenario.from_yaml_file(path))
        return scenarios

    # ==================================================================
    # Versioning
    # ==================================================================

    def get_version(self, scenario_id: str, version: int) -> Scenario | None:
        """Load a specific version of a scenario.

        Parameters
        ----------
        scenario_id : str
            The scenario identifier.
        version : int
            Version number to retrieve.

        Returns
        -------
        Scenario or None
            ``None`` if the version file does not exist.
        """
        path = self._yaml_path(scenario_id, version)
        if not os.path.exists(path):
            return None
        return Scenario.from_yaml_file(path)

    def list_versions(self, scenario_id: str) -> list[int]:
        """List all available version numbers for a scenario.

        Parameters
        ----------
        scenario_id : str
            The scenario identifier.

        Returns
        -------
        list[int]
            Sorted list of version numbers found on disk.
        """
        versions = []
        prefix = f"{scenario_id}_v"
        for fname in os.listdir(self.scenarios_dir):
            if fname.startswith(prefix) and fname.endswith(".yaml"):
                version_str = fname[len(prefix):-len(".yaml")]
                try:
                    versions.append(int(version_str))
                except ValueError:
                    continue
        return sorted(versions)

    # ==================================================================
    # Search — tag / risk tier
    # ==================================================================

    def search_by_tag(self, tags: list[str]) -> list[Scenario]:
        """Find scenarios with ANY of the given tags.

        Parameters
        ----------
        tags : list[str]
            Tags to search for.

        Returns
        -------
        list[Scenario]
            Matching scenarios (latest versions).
        """
        tag_set = set(tags)
        results = []
        for sid, entry in self._index.items():
            if tag_set & set(entry.get("tags", [])):
                scn = self.get(sid)
                if scn is not None:
                    results.append(scn)
        return results

    def search_by_risk_tier(self, tier: str) -> list[Scenario]:
        """Find scenarios matching a specific risk tier.

        Parameters
        ----------
        tier : str
            Risk tier to match (e.g. ``"action_taking"``).

        Returns
        -------
        list[Scenario]
        """
        results = []
        for sid, entry in self._index.items():
            if entry.get("risk_tier") == tier:
                scn = self.get(sid)
                if scn is not None:
                    results.append(scn)
        return results

    # ==================================================================
    # Search — geometric (GMS-backed)
    # ==================================================================

    def search_geometric(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Scenario, float]]:
        """Search scenarios by geometric relevance via the GMS store.

        Uses ``fuzzy_match_entity`` to resolve the query to a known
        entity, then ``score_triple`` to rank each scenario by geodesic
        distance to that entity.  Lower distance = more relevant.

        Degrades gracefully to an empty list when no store is available.

        Parameters
        ----------
        query : str
            Free-text query (will be fuzzy-matched to an entity).
        top_k : int
            Maximum results to return.

        Returns
        -------
        list[tuple[Scenario, float]]
            ``(scenario, distance)`` pairs sorted by ascending distance.
        """
        if self.store is None:
            return []

        # Resolve query to an entity
        entity = self.store.fuzzy_match_entity(query)
        if entity is None:
            return []

        scored: list[tuple[Scenario, float]] = []
        for scn in self.list_all():
            dist = self.store.score_triple(
                scn.scenario_id, "is_type", "test_scenario"
            )
            if dist is None:
                continue
            # Also check relevance to the query entity
            query_dist = self.store.score_triple(
                scn.scenario_id, "has_tag", entity
            )
            # Use the query distance if available, fall back to type dist
            effective = query_dist if query_dist is not None else dist
            scored.append((scn, effective))

        scored.sort(key=lambda pair: pair[1])
        return scored[:top_k]

    def find_complementary(
        self,
        scenario_id: str,
        top_k: int = 5,
    ) -> list[tuple[Scenario, float]]:
        """Find scenarios that are complementary (test different things).

        Uses ``tension_energy()`` between the target scenario's entity
        and other scenario entities.  Tension of ~sqrt(2) indicates
        irrelevance / complementarity (neither agreeing nor contradicting),
        which is ideal for building a diverse test suite.

        Parameters
        ----------
        scenario_id : str
            The scenario to find complements for.
        top_k : int
            Maximum results to return.

        Returns
        -------
        list[tuple[Scenario, float]]
            ``(scenario, tension)`` pairs sorted by closeness to sqrt(2).
        """
        if self.store is None:
            return []

        target = math.sqrt(2)
        candidates: list[tuple[Scenario, float]] = []

        for scn in self.list_all():
            if scn.scenario_id == scenario_id:
                continue
            tension = self.store.tension_energy(
                scenario_id, scn.scenario_id
            )
            if tension is None:
                continue
            candidates.append((scn, tension))

        # Sort by closeness to sqrt(2) — the "complementary" sweet spot
        candidates.sort(key=lambda pair: abs(pair[1] - target))
        return candidates[:top_k]

    # ==================================================================
    # DOE Bridge — convert scenarios to GeneratedQuestion objects
    # ==================================================================

    def to_doe_questions(
        self,
        scenarios: list[Scenario],
        store: "GMSExpertStore",
    ) -> list[GeneratedQuestion]:
        """Convert authored scenarios to GeneratedQuestion objects.

        For scenarios with ``generator_name``, delegates to the matching
        GMS generator.  For scenarios with ``question_text``, creates a
        ``GeneratedQuestion`` directly with a lambda ``gms_answer_fn``
        that returns the ground truth.

        Parameters
        ----------
        scenarios : list[Scenario]
            Scenarios to convert.
        store : GMSExpertStore
            The GMS store used for ground-truth computation.

        Returns
        -------
        list[GeneratedQuestion]
            Questions ready for DOE expansion or direct evaluation.
        """

        questions: list[GeneratedQuestion] = []

        for scn in scenarios:
            if scn.generator_name:
                # Delegate to the named generator
                gq = self._generate_from_generator(scn, store)
                if gq:
                    questions.extend(gq)
            elif scn.question_text:
                # Manually authored — wrap into a GeneratedQuestion
                gt = scn.ground_truth
                questions.append(GeneratedQuestion(
                    qid=scn.scenario_id,
                    category=scn.tags[0] if scn.tags else "authored",
                    natural_language=scn.question_text,
                    gms_answer_fn=lambda _store, _gt=gt: _gt,
                    ground_truth=gt,
                    llm_prompt=scn.question_text,
                    answer_type=scn.answer_type,
                    metadata={
                        "scenario_id": scn.scenario_id,
                        "scenario_name": scn.name,
                        "risk_tier": scn.risk_tier,
                        **scn.metadata,
                    },
                ))

        return questions

    def _generate_from_generator(
        self,
        scenario: Scenario,
        store: "GMSExpertStore",
    ) -> list[GeneratedQuestion]:
        """Run a GMS generator for a scenario and return its questions.

        Parameters
        ----------
        scenario : Scenario
            Must have ``generator_name`` set.
        store : GMSExpertStore
            Ground-truth store.

        Returns
        -------
        list[GeneratedQuestion]
            Generated questions (may be empty if the generator produces
            nothing for the current store).
        """
        from .bridge import _GENERATOR_MAP

        gen_name = scenario.generator_name
        cls_name = _GENERATOR_MAP.get(gen_name)
        if cls_name is None:
            print(
                f"  [scenario-registry] Unknown generator "
                f"'{gen_name}' in scenario {scenario.scenario_id}"
            )
            return []

        # Geometric generators
        _GEOMETRIC_NAMES = {
            "plausibility", "tension", "holonomy",
            "link_prediction", "multi_hop_transport", "phase_comparison",
        }
        _EXTENDED_NAMES = {
            "exact_retrieval", "negative", "workflow_compliance",
        }

        try:
            if gen_name in _GEOMETRIC_NAMES:
                from .generators import geometric_generators
                gen_lookup = {
                    (type(g).category if hasattr(type(g), "category")
                     else g.category): g
                    for g in geometric_generators()
                }
                gen = gen_lookup.get(gen_name)
            elif gen_name in _EXTENDED_NAMES:
                from .test_generators import extended_generators
                gen_lookup = {
                    (type(g).category if hasattr(type(g), "category")
                     else g.category): g
                    for g in extended_generators()
                }
                gen = gen_lookup.get(gen_name)
            else:
                from knowlytix.benchmark.eval import generators as gen_mod
                cls = getattr(gen_mod, cls_name)
                gen = cls()

            if gen is None:
                return []

            max_q = scenario.generator_params.get("max_questions", 10)
            seed = scenario.generator_params.get("seed", 42)
            candidates = gen.generate(store)
            sampled = gen.sample(candidates, max_q, seed)

            # Inject scenario metadata into each question
            for q in sampled:
                q.metadata["scenario_id"] = scenario.scenario_id
                q.metadata["scenario_name"] = scenario.name

            return sampled

        except Exception as exc:
            print(
                f"  [scenario-registry] Generator '{gen_name}' error "
                f"for scenario {scenario.scenario_id}: {exc}"
            )
            return []
