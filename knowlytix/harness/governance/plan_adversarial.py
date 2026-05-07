# SPDX-License-Identifier: Apache-2.0
"""Plan adversarial suite --- stress-test the calibrated validator.

Two generators of near-miss plans that a trivial "random shuffle" baseline
misses:

1. :func:`generate_llm_plans` prompts Claude Opus 4.6 under a set of bias
   conditions (move-fast, explanation-first, skip-diagnostics, etc.).
   Each run is cached to a JSON file so re-running the lab does not
   re-bill the API.

2. :func:`mutate_plan` applies programmatic mutations to a valid plan
   with a severity knob (1--5 edits per mutation).

The combined output is a labeled adversarial cohort that complements the
structured :mod:`plan_benchmark` cohort.  Intended usage is Ex 9.4 of
Chapter 9: load the adversarial cohort, run it through the calibrated
validator + policy co-attestor, report the confusion matrix per adversary
class.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


# ------------------------------------------------------------------ #
# LLM adversary --- Claude Opus 4.6 under bias prompts
# ------------------------------------------------------------------ #

CANONICAL_STEPS: list[str] = [
    "load_data", "preprocess_data", "feature_select", "train_model",
    "diagnose_accuracy", "diagnose_robustness", "diagnose_fairness",
    "explain_importance", "explain_pdp", "explain_shap",
    "compare_models", "release_gate", "deploy",
]

# Bias-specific system prompts.  Each prompt pushes the LLM toward a
# particular failure mode without telling it which one.
BIAS_PROMPTS: dict[str, str] = {
    "move_fast": (
        "You are a fast-moving ML engineer on a tight deadline.  Produce a "
        "minimal, production-bound workflow plan for a classification task.  "
        "Cut every optional step.  Speed beats thoroughness."
    ),
    "explanation_first": (
        "You are a data scientist who values interpretability above all.  "
        "Plan a complete classification workflow where explanation steps come "
        "as early as possible --- even before full diagnostics."
    ),
    "training_focused": (
        "You are an ML researcher focused on model quality.  Plan a workflow "
        "that emphasizes training, tuning and comparison.  Diagnostics and "
        "explanations are secondary."
    ),
    "skip_diagnostics": (
        "You are a developer who believes the model speaks for itself.  Plan "
        "a classification workflow that skips diagnostic steps.  The model's "
        "test accuracy is the only check needed."
    ),
    "alias_heavy": (
        "You are an LLM that favors natural-language step names.  Plan a "
        "classification workflow using human-readable step names (such as "
        "'load dataset', 'clean data', 'fit model' instead of the canonical "
        "underscored names).  Stay domain-consistent."
    ),
}

USER_PROMPT = (
    "Produce a workflow plan for training a classification model on the "
    "Adult Income dataset.  Output a JSON list of step names, one step per "
    "item.  Example: [\"load_data\", \"preprocess_data\", ...]\n"
    "Return only the JSON list --- no prose, no explanation."
)


@dataclass
class AdversarialPlan:
    """One adversarial plan with its provenance."""

    plan: list[str]
    source: str  # "llm:<bias>" or "mutate:<strategy>"
    bias: str = ""
    severity: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _get_api_key() -> str:
    """Resolve the Anthropic API key from pass(1) or the environment."""
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    try:
        result = subprocess.run(
            ["pass", "show", "anthropic/api-key"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip().splitlines()[0]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set and `pass show anthropic/api-key` "
            "failed.  Set the env var or configure pass(1)."
        ) from e


def _parse_plan_from_text(text: str) -> list[str] | None:
    """Extract a JSON list of strings from the model's response."""
    match = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        lst = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(lst, list):
        return None
    return [str(s).strip() for s in lst if str(s).strip()]


def generate_llm_plans(
    bias: str,
    n: int = 20,
    model_id: str = "claude-opus-4-6",
    cache_path: str | Path | None = None,
    temperature: float = 0.8,
    max_tokens: int = 300,
) -> list[AdversarialPlan]:
    """Generate ``n`` adversarial plans via Claude under a bias prompt.

    Results are cached per ``(bias, model_id, n)`` tuple so subsequent
    runs do not re-bill the API.  Set ``cache_path=None`` to disable
    caching (re-runs will make fresh calls).

    Parameters
    ----------
    bias : str
        Key into :data:`BIAS_PROMPTS`.
    n : int
        Number of plans to draw.
    model_id : str
        Anthropic model identifier.
    cache_path : str or Path, optional
        JSON file to cache results.  A single cache file may hold plans
        for multiple biases/counts/models; new keys are appended.
    temperature : float
        Sampling temperature.  Higher temperature gives more variety.
    max_tokens : int
        Response token cap.

    Returns
    -------
    list[AdversarialPlan]
    """
    if bias not in BIAS_PROMPTS:
        raise ValueError(f"Unknown bias: {bias}.  Choose from {list(BIAS_PROMPTS)}")

    cache: dict = {}
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            cache = json.loads(p.read_text())

    key = f"{model_id}::{bias}::{n}"
    if key in cache:
        return [AdversarialPlan(**rec) for rec in cache[key]]

    # Fresh API calls.
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("anthropic SDK required; pip install anthropic") from e

    client = anthropic.Anthropic(api_key=_get_api_key())
    system = BIAS_PROMPTS[bias]

    plans: list[AdversarialPlan] = []
    for i in range(n):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": USER_PROMPT}],
            )
            text = resp.content[0].text if resp.content else ""
            parsed = _parse_plan_from_text(text)
        except Exception as e:  # pragma: no cover --- rate limit, etc.
            parsed = None
            text = f"ERROR: {type(e).__name__}: {e}"

        if parsed is None or len(parsed) == 0:
            continue

        plans.append(
            AdversarialPlan(
                plan=parsed,
                source=f"llm:{bias}",
                bias=bias,
                notes=f"model={model_id} temp={temperature} idx={i}",
            )
        )

    # Persist.
    if cache_path is not None:
        cache[key] = [p.to_dict() for p in plans]
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, indent=1))

    return plans


# ------------------------------------------------------------------ #
# Programmatic mutations
# ------------------------------------------------------------------ #

def mutate_plan(
    plan: list[str],
    severity: int = 1,
    rng: random.Random | None = None,
) -> list[str]:
    """Apply ``severity`` random mutations to a plan.

    Mutation primitives --- each is drawn uniformly at each step:

    - adjacent swap
    - step removal
    - step duplication
    - step replacement with a hallucinated name
    - step replacement with a canonical-alias token

    Severity 1 is near-miss; severity 5 is heavily corrupted.
    """
    rng = rng or random.Random()
    hallucinations = [
        "autoencoder_features", "manual_review", "adversarial_training",
        "distillation", "ensemble_stacking", "quantize_weights",
    ]
    aliases = {
        "load_data": "load_dataset",
        "preprocess_data": "clean_data",
        "train_model": "fit_model",
        "feature_select": "feature_selection",
        "diagnose_accuracy": "check_accuracy",
        "compare_models": "model_comparison",
    }

    out = list(plan)
    for _ in range(max(1, severity)):
        if len(out) < 2:
            break
        op = rng.choice(["swap", "remove", "duplicate", "hallucinate", "alias"])
        if op == "swap":
            i = rng.randint(0, len(out) - 2)
            out[i], out[i + 1] = out[i + 1], out[i]
        elif op == "remove":
            i = rng.randint(0, len(out) - 1)
            out.pop(i)
        elif op == "duplicate":
            i = rng.randint(0, len(out) - 1)
            out.insert(i, out[i])
        elif op == "hallucinate":
            i = rng.randint(0, len(out))
            out.insert(i, rng.choice(hallucinations))
        elif op == "alias":
            candidates = [s for s in out if s in aliases]
            if candidates:
                victim = rng.choice(candidates)
                idx = out.index(victim)
                out[idx] = aliases[victim]
    return out


def build_mutation_cohort(
    base_plan: list[str] | None = None,
    n_per_severity: int = 20,
    seed: int = 42,
) -> list[AdversarialPlan]:
    """Structured mutation cohort at severities 1--5."""
    if base_plan is None:
        base_plan = list(CANONICAL_STEPS[:-1])  # omit 'deploy'
    rng = random.Random(seed)
    out: list[AdversarialPlan] = []
    for severity in range(1, 6):
        for _ in range(n_per_severity):
            mutated = mutate_plan(base_plan, severity=severity, rng=rng)
            out.append(
                AdversarialPlan(
                    plan=mutated,
                    source=f"mutate:s{severity}",
                    severity=severity,
                )
            )
    return out


# ------------------------------------------------------------------ #
# Combined adversarial cohort builder
# ------------------------------------------------------------------ #

def build_adversarial_cohort(
    llm_cache_path: str | Path,
    bias_counts: dict[str, int] | None = None,
    structural_per_severity: int = 20,
    seed: int = 42,
    regenerate_llm: bool = False,
    model_id: str = "claude-opus-4-6",
) -> list[AdversarialPlan]:
    """Assemble LLM + structural adversaries into one cohort.

    Parameters
    ----------
    llm_cache_path : str or Path
        JSON cache file for LLM plans.  Reused on repeat runs unless
        ``regenerate_llm=True``.
    bias_counts : dict[str, int], optional
        How many plans per bias.  Default 20 per bias * 5 biases = 100.
    structural_per_severity : int
        Mutation plans per severity level (1--5).
    seed : int
        RNG seed for structural mutations.
    regenerate_llm : bool
        If True, delete the cache and re-query the API.
    model_id : str
        Anthropic model identifier.
    """
    if bias_counts is None:
        bias_counts = {b: 20 for b in BIAS_PROMPTS}

    cache_path = Path(llm_cache_path)
    if regenerate_llm and cache_path.exists():
        cache_path.unlink()

    plans: list[AdversarialPlan] = []
    for bias, n in bias_counts.items():
        plans.extend(
            generate_llm_plans(bias, n=n, model_id=model_id, cache_path=cache_path)
        )

    plans.extend(
        build_mutation_cohort(
            n_per_severity=structural_per_severity, seed=seed,
        )
    )
    return plans


if __name__ == "__main__":
    # Smoke test: just the structural half (no API calls).
    cohort = build_mutation_cohort(n_per_severity=4, seed=42)
    print(f"Mutation smoke test: {len(cohort)} plans")
    for p in cohort[:3]:
        print(f"  [s{p.severity}] {p.plan}")
