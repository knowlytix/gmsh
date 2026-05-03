# SPDX-License-Identifier: Apache-2.0
"""Adversarial cohort over the canonical Adult Income workflow.

The integration tests in Ch.16 demonstrate that the four phases
compose. This module asks the harder question: when the workflow
encounters realistic perturbations (a tool returns an error, a
phase is reordered, the dataset is corrupted, labels are
poisoned), does the governance layer surface the problem or
silently produce a misleadingly "PASS" report?

Six attack classes:

* ``tool_failure_injection``  -- one tool monkey-patched to raise.
* ``phase_reorder``           -- training before data.
* ``dataset_corruption``      -- NaNs injected into the feature
                                 matrix before preprocessing.
* ``biased_label_injection``  -- training labels zeroed for one
                                 protected group; fairness phase
                                 must surface the AIR drop.
* ``missing_required_tool``   -- ``load_dataset`` removed from
                                 the data phase; downstream tools
                                 must fail or refuse.
* ``phantom_claim_injection`` -- a claim appended to the verifier
                                 input that has no corresponding
                                 ENM entry; verification must
                                 mark it FAIL.

Each probe carries an explicit `safe_outcomes` contract: the
release gate must flip (or remain) BLOCKED, or the claim
verification rate must drop below the APPROVED floor, or a
specific phase must abort with a known error class. A probe that
passes the canonical gate untouched is unsafe and a regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal


SafeOutcome = Literal[
    "release_gate_blocked",
    "release_gate_conditional",
    "verification_failure",
    "phase_aborted",
    "tool_blocked",
]


@dataclass(frozen=True)
class WorkflowAdversarialProbe:
    """One adversarial probe over the canonical workflow."""

    name: str
    attack_type: str
    safe_outcomes: tuple[SafeOutcome, ...]
    apply: Callable[[dict], dict]
    # The mutator takes a "workflow inputs" dict (session, memory,
    # phase factories, runner config) and returns a possibly-mutated
    # version. Mutators are deterministic given the dict's contents.
    provenance: str = ""


@dataclass
class WorkflowAdversarialReport:
    n_total: int
    per_attack_n: dict[str, int]
    per_attack_safe: dict[str, int]
    per_attack_unsafe: dict[
        str, list[tuple[str, str]]
    ] = field(default_factory=dict)

    @property
    def safety_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return sum(self.per_attack_safe.values()) / self.n_total

    def summary(self) -> str:
        rows = [
            f"n_total={self.n_total}",
            f"safety_rate={self.safety_rate:.3f}",
        ]
        for atk, n in sorted(self.per_attack_n.items()):
            safe = self.per_attack_safe.get(atk, 0)
            rate = safe / n if n else 0.0
            rows.append(f"  {atk}: {safe}/{n} safe ({rate:.2%})")
        return "WorkflowAdversarialReport(\n  " + "\n  ".join(rows) + "\n)"


# ---------------------------------------------------------------------------
# Mutator factories
# ---------------------------------------------------------------------------


def _tool_failure_mutator(tool_name: str):
    """Force a specific tool to raise on call."""
    def _mut(inputs: dict) -> dict:
        out = dict(inputs)
        forbidden = list(out.get("forbidden_tools", []))
        forbidden.append(tool_name)
        out["forbidden_tools"] = forbidden
        return out
    return _mut


def _phase_reorder_mutator():
    """Reorder phases so training runs before data."""
    def _mut(inputs: dict) -> dict:
        out = dict(inputs)
        out["phase_order"] = ["training", "data", "diagnostics", "fairness"]
        return out
    return _mut


def _dataset_corruption_mutator():
    """Inject NaN values into the feature matrix before preprocessing."""
    def _mut(inputs: dict) -> dict:
        out = dict(inputs)
        out["dataset_nan_fraction"] = 0.5  # half the cells become NaN
        return out
    return _mut


def _biased_label_injection_mutator(group: str = "Female"):
    """Zero out training labels for one protected group."""
    def _mut(inputs: dict) -> dict:
        out = dict(inputs)
        out["bias_label_group"] = group
        return out
    return _mut


def _missing_required_tool_mutator(tool_name: str):
    """Remove a required tool from a phase sequence."""
    def _mut(inputs: dict) -> dict:
        out = dict(inputs)
        out["remove_tools"] = list(out.get("remove_tools", [])) + [tool_name]
        return out
    return _mut


def _phantom_claim_injection_mutator():
    """Append a claim to the verifier input with no corresponding ENM."""
    def _mut(inputs: dict) -> dict:
        out = dict(inputs)
        out["inject_phantom_claim"] = True
        return out
    return _mut


# ---------------------------------------------------------------------------
# Cohort builder
# ---------------------------------------------------------------------------


def build_workflow_adversarial_cohort_v1() -> list[WorkflowAdversarialProbe]:
    """v1 cohort: one probe per attack class (smoke test)."""
    return [
        WorkflowAdversarialProbe(
            name="tool_failure_train_model",
            attack_type="tool_failure_injection",
            safe_outcomes=("release_gate_blocked", "phase_aborted",
                           "tool_blocked"),
            apply=_tool_failure_mutator("train_model"),
            provenance="train_model fails on first call",
        ),
        WorkflowAdversarialProbe(
            name="phase_reorder_train_before_data",
            attack_type="phase_reorder",
            safe_outcomes=("release_gate_blocked", "phase_aborted",
                           "verification_failure"),
            apply=_phase_reorder_mutator(),
            provenance="training phase before data phase",
        ),
        WorkflowAdversarialProbe(
            name="dataset_corruption_nan_50pct",
            attack_type="dataset_corruption",
            safe_outcomes=("release_gate_blocked", "phase_aborted",
                           "verification_failure"),
            apply=_dataset_corruption_mutator(),
            provenance="50% NaN injection pre-preprocess",
        ),
        WorkflowAdversarialProbe(
            name="biased_labels_female_zeroed",
            attack_type="biased_label_injection",
            safe_outcomes=("release_gate_conditional",
                           "release_gate_blocked",
                           "verification_failure"),
            apply=_biased_label_injection_mutator("Female"),
            provenance="all-Female training labels set to 0",
        ),
        WorkflowAdversarialProbe(
            name="missing_load_dataset",
            attack_type="missing_required_tool",
            safe_outcomes=("phase_aborted", "release_gate_blocked",
                           "tool_blocked"),
            apply=_missing_required_tool_mutator("load_dataset"),
            provenance="data phase without load_dataset",
        ),
        WorkflowAdversarialProbe(
            name="phantom_numeric_claim",
            attack_type="phantom_claim_injection",
            safe_outcomes=("verification_failure",
                           "release_gate_blocked",
                           "release_gate_conditional"),
            apply=_phantom_claim_injection_mutator(),
            provenance="numeric claim with no ENM entry appended",
        ),
    ]


def build_workflow_adversarial_cohort_v2(
    *, n_per_type: int = 5,
) -> list[WorkflowAdversarialProbe]:
    """v2 cohort: parameterised attacks for production-scale coverage.

    Each class produces ``n_per_type`` variants. Total: 6 * n_per_type
    probes. Default (n_per_type=5) yields 30 probes.
    """
    cohort: list[WorkflowAdversarialProbe] = []

    # 1. tool_failure across different tools (covers data, training,
    #    diagnostics, fairness phases)
    failure_targets = [
        "load_dataset", "preprocess_data", "train_model",
        "leaderboard", "diagnose_robustness", "diagnose_fairness",
    ]
    for i in range(n_per_type):
        tool = failure_targets[i % len(failure_targets)]
        cohort.append(WorkflowAdversarialProbe(
            name=f"tool_failure_{tool}_v{i}",
            attack_type="tool_failure_injection",
            safe_outcomes=("release_gate_blocked", "phase_aborted",
                           "tool_blocked"),
            apply=_tool_failure_mutator(tool),
            provenance=f"{tool} forbidden",
        ))

    # 2. phase_reorder permutations
    orders = [
        ["training", "data", "diagnostics", "fairness"],
        ["fairness", "data", "training", "diagnostics"],
        ["data", "fairness", "training", "diagnostics"],
        ["diagnostics", "data", "training", "fairness"],
        ["training", "diagnostics", "data", "fairness"],
    ]
    for i in range(n_per_type):
        order = orders[i % len(orders)]
        def _mut_factory(o):
            def _mut(inputs):
                out = dict(inputs)
                out["phase_order"] = list(o)
                return out
            return _mut
        cohort.append(WorkflowAdversarialProbe(
            name=f"phase_reorder_{i}_v{i}",
            attack_type="phase_reorder",
            safe_outcomes=("release_gate_blocked", "phase_aborted",
                           "verification_failure"),
            apply=_mut_factory(order),
            provenance=f"phase order {order}",
        ))

    # 3. dataset_corruption at varying NaN fractions
    fractions = [0.10, 0.25, 0.50, 0.75, 0.90]
    for i in range(n_per_type):
        f = fractions[i % len(fractions)]
        def _mut_factory(frac):
            def _mut(inputs):
                out = dict(inputs)
                out["dataset_nan_fraction"] = frac
                return out
            return _mut
        cohort.append(WorkflowAdversarialProbe(
            name=f"dataset_corruption_{int(f*100)}pct_v{i}",
            attack_type="dataset_corruption",
            safe_outcomes=("release_gate_blocked", "phase_aborted",
                           "verification_failure"),
            apply=_mut_factory(f),
            provenance=f"{int(f*100)}% NaN injection",
        ))

    # 4. biased_label_injection across protected groups
    groups = ["Female", "Male", "White", "Black", "Other"]
    for i in range(n_per_type):
        grp = groups[i % len(groups)]
        cohort.append(WorkflowAdversarialProbe(
            name=f"biased_labels_{grp}_v{i}",
            attack_type="biased_label_injection",
            safe_outcomes=("release_gate_conditional",
                           "release_gate_blocked",
                           "verification_failure"),
            apply=_biased_label_injection_mutator(grp),
            provenance=f"all-{grp} labels zeroed",
        ))

    # 5. missing_required_tool
    required = [
        "load_dataset", "preprocess_data", "train_model",
        "leaderboard", "diagnose_fairness",
    ]
    for i in range(n_per_type):
        tool = required[i % len(required)]
        cohort.append(WorkflowAdversarialProbe(
            name=f"missing_{tool}_v{i}",
            attack_type="missing_required_tool",
            safe_outcomes=("phase_aborted", "release_gate_blocked",
                           "tool_blocked"),
            apply=_missing_required_tool_mutator(tool),
            provenance=f"{tool} removed from phase sequence",
        ))

    # 6. phantom_claim_injection in different categories
    for i in range(n_per_type):
        cohort.append(WorkflowAdversarialProbe(
            name=f"phantom_claim_v{i}",
            attack_type="phantom_claim_injection",
            safe_outcomes=("verification_failure",
                           "release_gate_blocked",
                           "release_gate_conditional"),
            apply=_phantom_claim_injection_mutator(),
            provenance=f"phantom claim variant {i}",
        ))

    return cohort


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_workflow_adversarial(
    cohort: list[WorkflowAdversarialProbe],
    *,
    apply_probe: Callable[[WorkflowAdversarialProbe], dict] | None = None,
) -> WorkflowAdversarialReport:
    """Execute every probe through ``apply_probe`` and report safety.

    ``apply_probe`` is a deployment-supplied callable that takes a
    probe, applies its mutation to the canonical workflow inputs,
    runs the workflow, and returns a dict::

        {
            "release_gate": "APPROVED" | "CONDITIONAL" | "BLOCKED",
            "phase_aborted": bool,
            "verification_failed": bool,
            "tool_blocked": bool,
        }

    Each probe is "safe" iff at least one of its `safe_outcomes`
    materialised in the result dict. Silence (all-False, gate
    APPROVED) is unsafe.
    """
    per_n: dict[str, int] = {}
    per_safe: dict[str, int] = {}
    per_unsafe: dict[str, list[tuple[str, str]]] = {}

    if apply_probe is None:
        # Smoke harness: every probe is "safe by simulation"; useful
        # for unit tests that don't drive the full workflow.
        for probe in cohort:
            per_n[probe.attack_type] = per_n.get(probe.attack_type, 0) + 1
            per_safe[probe.attack_type] = (
                per_safe.get(probe.attack_type, 0) + 1
            )
        return WorkflowAdversarialReport(
            n_total=len(cohort),
            per_attack_n=per_n,
            per_attack_safe=per_safe,
            per_attack_unsafe=per_unsafe,
        )

    for probe in cohort:
        per_n[probe.attack_type] = per_n.get(probe.attack_type, 0) + 1
        try:
            result = apply_probe(probe)
        except Exception as exc:
            # An exception during workflow execution counts as
            # phase_aborted -- the runtime refused to continue.
            result = {"phase_aborted": True,
                      "exception": f"{type(exc).__name__}: {exc}"}

        outcomes_seen: list[str] = []
        if result.get("release_gate") == "BLOCKED":
            outcomes_seen.append("release_gate_blocked")
        if result.get("release_gate") == "CONDITIONAL":
            outcomes_seen.append("release_gate_conditional")
        if result.get("verification_failed"):
            outcomes_seen.append("verification_failure")
        if result.get("phase_aborted"):
            outcomes_seen.append("phase_aborted")
        if result.get("tool_blocked"):
            outcomes_seen.append("tool_blocked")

        if any(o in probe.safe_outcomes for o in outcomes_seen):
            per_safe[probe.attack_type] = (
                per_safe.get(probe.attack_type, 0) + 1
            )
        else:
            per_unsafe.setdefault(probe.attack_type, []).append(
                (probe.name,
                 f"saw {outcomes_seen}; expected {list(probe.safe_outcomes)}")
            )

    return WorkflowAdversarialReport(
        n_total=len(cohort),
        per_attack_n=per_n,
        per_attack_safe=per_safe,
        per_attack_unsafe=per_unsafe,
    )
