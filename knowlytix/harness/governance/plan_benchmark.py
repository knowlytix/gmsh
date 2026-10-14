# SPDX-License-Identifier: Apache-2.0
"""Plan benchmark --- labeled cohort generator for planner calibration.

Generates a balanced cohort of labeled plans across six failure classes so
that `WorkflowPlanner.calibrate()` can fit decision thresholds with
reported FPR, FNR and bootstrap confidence intervals.

Classes
-------
- ``valid``            : canonical orderings with legitimate variations
- ``misorder``         : swap k adjacent pairs for k in {1, 2, 3}
- ``skip_prereq``      : remove one required step from a valid plan
- ``alias_swap``       : replace a step with a valid alias
- ``hallucinated_step``: insert a plausible but non-canonical step
- ``merged_steps``     : fuse two adjacent steps into one

Each plan record in the cohort is a dict with keys::

    {"plan": [...], "label": "valid"|"invalid",
     "class": "<class_name>", "source": "<generator>",
     "severity": int}

The cohort is deterministic given ``seed``.

Usage
-----
>>> from knowlytix.harness.governance.plan_benchmark import build_cohort, save_cohort
>>> cohort = build_cohort(seed=42, n_per_class={"valid": 80, ...})
>>> save_cohort(cohort, "governance_monograph/eval_cohorts/plan_cohort_v1.jsonl")
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

# ------------------------------------------------------------------ #
# Canonical workflow vocabulary (must stay aligned with planner.py)
# ------------------------------------------------------------------ #

CANONICAL_VALID_PLAN: list[str] = [
    "load_data",
    "preprocess_data",
    "feature_select",
    "train_model",
    "diagnose_accuracy",
    "diagnose_robustness",
    "diagnose_fairness",
    "explain_importance",
    "explain_pdp",
    "explain_shap",
    "compare_models",
    "release_gate",
]

# Aliases the store may resolve via fuzzy match (used for alias_swap class).
STEP_ALIASES: dict[str, list[str]] = {
    "load_data": ["load_dataset", "ingest_data", "read_data"],
    "preprocess_data": ["clean_data", "prepare_data", "data_prep"],
    "feature_select": ["select_features", "feature_selection"],
    "train_model": ["fit_model", "train", "learn_model"],
    "diagnose_accuracy": ["check_accuracy", "accuracy_diagnostic"],
    "diagnose_robustness": ["check_robustness", "robustness_test"],
    "diagnose_fairness": ["fairness_audit", "bias_check"],
    "explain_importance": ["feature_importance", "pfi"],
    "explain_pdp": ["partial_dependence", "pdp_analysis"],
    "explain_shap": ["shap_analysis", "shap_values"],
    "compare_models": ["model_comparison", "leaderboard"],
    "release_gate": ["release_review", "deployment_gate"],
}

# Plausible hallucinated steps a miscalibrated LLM might invent.
HALLUCINATED_STEPS: list[str] = [
    "autoencoder_features",
    "manual_review",
    "feature_engineering_deep",
    "adversarial_training",
    "distillation",
    "ensemble_stacking",
    "synthetic_augmentation",
    "chain_of_thought_probe",
    "quantize_weights",
    "architecture_search",
]


# ------------------------------------------------------------------ #
# Plan record
# ------------------------------------------------------------------ #

@dataclass
class PlanRecord:
    plan: list[str]
    label: str  # "valid" | "invalid"
    plan_class: str
    source: str
    severity: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("plan_class")
        return d


# ------------------------------------------------------------------ #
# Generators
# ------------------------------------------------------------------ #

def _valid_variations(rng: random.Random, n: int) -> list[PlanRecord]:
    """Canonical ordering plus legitimate reorderings that stay valid."""
    out: list[PlanRecord] = []
    base = list(CANONICAL_VALID_PLAN)

    # Variation A: canonical, no changes
    out.append(
        PlanRecord(
            plan=list(base),
            label="valid",
            plan_class="valid",
            source="canonical",
            notes="exact canonical ordering",
        )
    )

    # Variation B: drop feature_select (optional step — still valid)
    base_no_fs = [s for s in base if s != "feature_select"]
    out.append(
        PlanRecord(
            plan=list(base_no_fs),
            label="valid",
            plan_class="valid",
            source="optional_drop",
            notes="feature_select omitted",
        )
    )

    # Variation C: permute the explanation block only (no prereqs between
    # explain_importance / explain_pdp / explain_shap — they all require
    # train_model and nothing else).
    expl = ["explain_importance", "explain_pdp", "explain_shap"]
    while len(out) < n:
        plan = list(base)
        rng.shuffle(expl)
        start = plan.index("explain_importance")
        end = plan.index("explain_shap") + 1
        plan = plan[:start] + list(expl) + plan[end:]
        out.append(
            PlanRecord(
                plan=plan,
                label="valid",
                plan_class="valid",
                source="reordered_commutative",
                notes="explanation block permutation",
            )
        )
    return out[:n]


def _misorder(rng: random.Random, n: int) -> list[PlanRecord]:
    """Swap k adjacent pairs (k in {1,2,3}), breaking dependency order."""
    out: list[PlanRecord] = []
    base = list(CANONICAL_VALID_PLAN)
    per_k = max(1, n // 3)

    for severity in (1, 2, 3):
        for _ in range(per_k):
            plan = list(base)
            # Pick severity random adjacent-pair swaps that break prereqs
            candidate_swaps = [
                (1, 2),   # preprocess/feature_select (harmless alone)
                (2, 3),   # feature_select/train_model (breaks preprocess->train)
                (3, 4),   # train_model/diagnose_accuracy (breaks prereq)
                (4, 5),   # diag_accuracy/diag_robustness
                (10, 11), # compare_models/release_gate (breaks prereq)
                (0, 1),   # load/preprocess (breaks prereq)
            ]
            rng.shuffle(candidate_swaps)
            applied = 0
            i = 0
            while applied < severity and i < len(candidate_swaps):
                a, b = candidate_swaps[i]
                plan[a], plan[b] = plan[b], plan[a]
                applied += 1
                i += 1
            out.append(
                PlanRecord(
                    plan=plan,
                    label="invalid",
                    plan_class="misorder",
                    source=f"adjacent_swap_k{severity}",
                    severity=severity,
                )
            )
    # pad if rounding left us short
    while len(out) < n:
        plan = list(base)
        a = rng.randint(0, len(plan) - 2)
        plan[a], plan[a + 1] = plan[a + 1], plan[a]
        out.append(
            PlanRecord(
                plan=plan,
                label="invalid",
                plan_class="misorder",
                source="adjacent_swap_k1",
                severity=1,
            )
        )
    return out[:n]


def _skip_prereq(rng: random.Random, n: int) -> list[PlanRecord]:
    """Remove one required step from a valid plan, breaking a prereq."""
    out: list[PlanRecord] = []
    base = list(CANONICAL_VALID_PLAN)
    # Required steps — removing any of these breaks something downstream.
    required_steps = [
        "load_data",
        "preprocess_data",
        "train_model",
        "diagnose_accuracy",
        "compare_models",
    ]
    for _ in range(n):
        plan = list(base)
        victim = rng.choice(required_steps)
        plan.remove(victim)
        out.append(
            PlanRecord(
                plan=plan,
                label="invalid",
                plan_class="skip_prereq",
                source=f"removed_{victim}",
                notes=f"missing required step: {victim}",
            )
        )
    return out[:n]


def _alias_swap(rng: random.Random, n: int) -> list[PlanRecord]:
    """Replace a canonical step with an alias the store may not know.

    Label is ``invalid`` because the planner cannot resolve unknown aliases
    and should flag the plan for review. A more advanced store with alias
    resolution would reclassify these; for calibration against the current
    store they are genuine negatives.
    """
    out: list[PlanRecord] = []
    base = list(CANONICAL_VALID_PLAN)
    step_keys = list(STEP_ALIASES.keys())
    for _ in range(n):
        plan = list(base)
        victim = rng.choice(step_keys)
        alias = rng.choice(STEP_ALIASES[victim])
        idx = plan.index(victim)
        plan[idx] = alias
        out.append(
            PlanRecord(
                plan=plan,
                label="invalid",
                plan_class="alias_swap",
                source=f"{victim}_to_{alias}",
                notes="alias not in canonical vocabulary",
            )
        )
    return out[:n]


def _hallucinated_step(rng: random.Random, n: int) -> list[PlanRecord]:
    """Insert a plausible but non-canonical step at a random position."""
    out: list[PlanRecord] = []
    base = list(CANONICAL_VALID_PLAN)
    for _ in range(n):
        plan = list(base)
        bogus = rng.choice(HALLUCINATED_STEPS)
        # Insert somewhere in the middle third of the plan
        pos = rng.randint(2, len(plan) - 2)
        plan.insert(pos, bogus)
        out.append(
            PlanRecord(
                plan=plan,
                label="invalid",
                plan_class="hallucinated_step",
                source=f"inserted_{bogus}",
                notes=f"non-canonical step: {bogus}",
            )
        )
    return out[:n]


def _merged_steps(rng: random.Random, n: int) -> list[PlanRecord]:
    """Fuse two adjacent canonical steps into a single bogus step."""
    out: list[PlanRecord] = []
    base = list(CANONICAL_VALID_PLAN)
    for _ in range(n):
        plan = list(base)
        i = rng.randint(0, len(plan) - 2)
        fused = f"{plan[i]}_and_{plan[i + 1]}"
        plan = plan[:i] + [fused] + plan[i + 2:]
        out.append(
            PlanRecord(
                plan=plan,
                label="invalid",
                plan_class="merged_steps",
                source=f"fused_{i}",
                notes=f"fused two steps: {fused}",
            )
        )
    return out[:n]


# ------------------------------------------------------------------ #
# Cohort builder
# ------------------------------------------------------------------ #

DEFAULT_COUNTS: dict[str, int] = {
    "valid": 80,
    "misorder": 80,
    "skip_prereq": 80,
    "alias_swap": 60,
    "hallucinated_step": 60,
    "merged_steps": 40,
}


def build_cohort(
    seed: int = 42,
    n_per_class: dict[str, int] | None = None,
) -> list[PlanRecord]:
    """Build a deterministic labeled plan cohort.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    n_per_class : dict[str, int] | None
        Override the default counts per class. If None, uses
        :data:`DEFAULT_COUNTS` (total ~400 plans).

    Returns
    -------
    list[PlanRecord]
        The cohort as a list of labeled plan records.
    """
    counts = dict(DEFAULT_COUNTS)
    if n_per_class is not None:
        counts.update(n_per_class)
    rng = random.Random(seed)

    generators = {
        "valid": _valid_variations,
        "misorder": _misorder,
        "skip_prereq": _skip_prereq,
        "alias_swap": _alias_swap,
        "hallucinated_step": _hallucinated_step,
        "merged_steps": _merged_steps,
    }

    cohort: list[PlanRecord] = []
    for name, gen in generators.items():
        n = counts.get(name, 0)
        if n <= 0:
            continue
        cohort.extend(gen(rng, n))

    return cohort


def save_cohort(cohort: list[PlanRecord], path: str | Path) -> None:
    """Write the cohort to a JSON-Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in cohort:
            f.write(json.dumps(rec.to_dict()) + "\n")


def load_cohort(path: str | Path) -> list[PlanRecord]:
    """Load a cohort from a JSON-Lines file."""
    out: list[PlanRecord] = []
    with Path(path).open() as f:
        for line in f:
            d = json.loads(line)
            out.append(
                PlanRecord(
                    plan=list(d["plan"]),
                    label=d["label"],
                    plan_class=d["class"],
                    source=d.get("source", ""),
                    severity=int(d.get("severity", 0)),
                    notes=d.get("notes", ""),
                )
            )
    return out


def cohort_summary(cohort: list[PlanRecord]) -> dict:
    """Return per-class counts and total valid/invalid split."""
    summary: dict = {"total": len(cohort), "by_class": {}, "by_label": {"valid": 0, "invalid": 0}}
    for rec in cohort:
        summary["by_class"][rec.plan_class] = summary["by_class"].get(rec.plan_class, 0) + 1
        summary["by_label"][rec.label] += 1
    return summary


if __name__ == "__main__":
    cohort = build_cohort(seed=42)
    summary = cohort_summary(cohort)
    print(f"Built cohort: {summary['total']} plans")
    print(f"  By label: {summary['by_label']}")
    for cls, n in summary["by_class"].items():
        print(f"  {cls:20s} {n:4d}")
    out_path = Path("governance_monograph/eval_cohorts/plan_cohort_v1.jsonl")
    save_cohort(cohort, out_path)
    print(f"\nWrote {out_path}")
