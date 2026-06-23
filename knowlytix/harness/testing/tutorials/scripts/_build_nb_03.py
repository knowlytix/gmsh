# SPDX-License-Identifier: Apache-2.0
"""Build NB 03 — Designing Test Experiments (DOE) — from cell definitions, then execute.

Migrated from knowlytix/harness/testing/tutorials/optimal_design_doe.ipynb and realigned to
chapter-anchored template. Demonstrates the paper-style Sobol+Refine design on
the bank model-risk corpus, runs 60 real Sonnet 4.6 calls, and walks through
factor-effect analysis.

Cost: ~$0.30 / ~4 min wall clock.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_03.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "03_doe_design.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 03: Designing Test Experiments — Sobol+Refine on a Bank Corpus

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch03.tex` — Design of Experiments for Systematic Evaluation
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch03_design_of_experiments.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §6
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; ch04 fixture; NB 02 (geometric primitives)

**What you'll do.** Build a Design-of-Experiments matrix with the paper-style Sobol+Refine method, contrast it against vanilla Sobol on space-filling metrics (φ_p, min-dist, coverage), pull real graph-verified base questions from the bank model-risk corpus, run 60 Claude Sonnet 4.6 calls under the design, then analyze the factor effects with single-factor logistic regression + Benjamini-Hochberg FDR correction.

This is the methodology behind the FinStructBench case study (`graphdoe/papers/paper_jasa.tex`) applied to the bank inventory corpus.\
"""))

cells.append(md("""\
## 1. Setup

Standard preamble plus the design + analysis modules from `graphdoe`.\
"""))

cells.append(code("""\
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")

from _shared import (
    load_ch04_store, make_anthropic_client, seed_all,
    DEFAULT_TUTORIAL_MODEL,
)

import numpy as np
import pandas as pd
import torch

import graphdoe
from graphdoe.question_rephraser import QuestionRephraser
from graphdoe.analysis import DOEAnalyzer

seed_all()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")
print(f"graphdoe.design from: {graphdoe.design.__module__}")\
"""))

cells.append(md("""\
## 2. The Paper-Style API in Two Lines

`graphdoe.design()` accepts a list of base questions and the contextual-factor catalog selection. It returns a `DesignMatrix` whose `.matrix` DataFrame has one row per LLM call and the `question` column already chosen. No outer-product expansion — the total LLM-call budget is `n_runs`, *not* `n_questions × n_runs`.

For the demo we'll use 30 placeholder questions and the `core` preset (10 factors, defined in `graphdoe/factor_catalog.yaml`).\
"""))

cells.append(code("""\
PLACEHOLDER_QUESTIONS = [f"What is the value of metric {i}?" for i in range(30)]
CORE_FACTORS = [
    "clarity", "noise", "role", "answer_format", "reasoning_cue",
    "context_scope", "entity_aliasing", "instruction_conflict",
    "complexity", "style",
]

demo_design = graphdoe.design(
    questions=PLACEHOLDER_QUESTIONS,
    factor_names=CORE_FACTORS,
    n_runs=60,
    method="sobol+refine",
    seed=42,
    verbose=False,
)
print(f"shape: {demo_design.matrix.shape}")
print(f"columns: {list(demo_design.matrix.columns)}")
print()
print(demo_design.matrix.head(3).to_string())\
"""))

cells.append(md("""\
## 3. Sobol vs Sobol+Refine

Build both designs at the same `n_runs=60` and compare space-filling metrics. Pure local computation — no API spend in this section.\
"""))

cells.append(code("""\
N_RUNS_DEMO = 60

ds_sobol = graphdoe.design(
    questions=PLACEHOLDER_QUESTIONS, factor_names=CORE_FACTORS,
    n_runs=N_RUNS_DEMO, method="sobol", seed=42,
)
ds_refine = graphdoe.design(
    questions=PLACEHOLDER_QUESTIONS, factor_names=CORE_FACTORS,
    n_runs=N_RUNS_DEMO, method="sobol+refine", seed=42,
)
q_s, q_r = ds_sobol.quality(), ds_refine.quality()

rows = []
for k in ("phi_p", "min_dist", "mean_nn", "mir_avg", "mir_max", "coverage", "empty_cells"):
    s, r = q_s[k], q_r[k]
    rows.append({"metric": k, "sobol": s, "sobol+refine": r,
                 "ratio": r / s if s not in (0, 0.0) else float("nan")})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))\
"""))

cells.append(md("""\
**What to read off:**

- `phi_p` ↓ and `min_dist` ↑ — the headline space-filling improvement (paper reports ≈2.4× and 2.6× at $n=880, d=21$).
- `mir_avg` measures level imbalance per factor; goes up when the optimizer trades grid balance for spread.
- `coverage` is the fraction of marginal cells across all factors that get used at least once.\
"""))

cells.append(md("""\
## 4. Pull Real Graph-Verified Base Questions

Replace the placeholders with questions whose ground truth is derivable from triple traversal on the bank model-risk store. `bridge.generate_base_questions` walks the graph and emits questions that are **graph-verified, 100% correct by construction**.\
"""))

cells.append(code("""\
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig
from knowlytix.harness.testing.bridge import generate_base_questions

config = DOEHarnessConfig(
    markdown_path="knowlytix/harness/testing/tests/model_risk_assessment.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen",
    n_runs=N_RUNS_DEMO,
    method="sobol+refine",
    seed=42,
    enable_tracing=False,
    device=str(DEVICE),
)
bench = DOEGMSBenchmark(config)
import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    store = bench.ingest()

DESIRED_BASE_Q = 30
candidate_generators = [
    "exact_retrieval", "counting", "cross_reference", "multi_hop", "boundary",
]
base_qs = generate_base_questions(
    store=store,
    generator_names=candidate_generators,
    max_per_category=10,
    seed=42,
)

seen, unique_base = set(), []
for q in base_qs:
    if q.natural_language not in seen:
        seen.add(q.natural_language)
        unique_base.append(q)
    if len(unique_base) >= DESIRED_BASE_Q:
        break

print(f"Generated {len(base_qs)} base questions, kept {len(unique_base)} unique.")
print()
print("Sample base questions:")
for q in unique_base[:5]:
    print(f"  [{q.category}] {q.natural_language[:80]}  ->  gt={q.ground_truth}")\
"""))

cells.append(md("""\
## 5. Build the Paper-Style Design with Real Questions

Now `graphdoe.design()` with the real questions instead of placeholders. The `question` column already picks which base question each row asks; the other 10 columns specify the contextual factor levels for that row.\
"""))

cells.append(code("""\
design = graphdoe.design(
    questions=unique_base,
    factor_names=CORE_FACTORS,
    n_runs=N_RUNS_DEMO,
    method="sobol+refine",
    seed=42,
    verbose=False,
)
df = design.matrix
print(f"shape: {df.shape}     factors: {len(design.factors)}")
print(f"unique questions used: {df['question'].nunique()} / {len(unique_base)}")
print(f"avg reps per used question: {len(df) / df['question'].nunique():.1f}")
print()
print("design.quality():")
for k, v in design.quality().items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")\
"""))

cells.append(md("""\
## 6. Graph Baseline (Sanity Check)

Before paying for any LLM call, verify ground-truth integrity by traversing the graph for each base question. This must be 100% by construction — if it isn't, the ground truth itself is broken and the LLM run will be measuring nothing.\
"""))

cells.append(code("""\
from knowlytix.benchmark.scorers import score_answer

n_correct = 0
for q in unique_base:
    graph_answer = q.graph_answer_fn(store)
    s = score_answer(graph_answer, q.ground_truth)
    if s.correct:
        n_correct += 1
    else:
        print(f"  GT MISMATCH: {q.natural_language[:60]} "
              f"-> graph={graph_answer!r}  vs  gt={q.ground_truth!r}")

print(f"Graph baseline: {n_correct}/{len(unique_base)} ({n_correct/len(unique_base):.0%})")
assert n_correct == len(unique_base), "Graph baseline must be 100%"\
"""))

cells.append(md("""\
## 7. Run the LLM Under the Design

Iterate over the 60 design rows. For each: pick the base question, apply the row's 10 contextual-factor levels via `QuestionRephraser`, send to Claude Sonnet 4.6, score against ground truth.

We use a short system instruction that asks Sonnet to return just the answer value — no prose, no rounding — so exact-match scoring (`score_answer`) tracks the actual capability rather than the formatting style.\
"""))

cells.append(code("""\
import time
from knowlytix.benchmark.scorers import PARSERS

client = make_anthropic_client()
rephraser = QuestionRephraser(method="template")
base_q_lookup = {q.natural_language: q for q in unique_base}
with open(config.markdown_path) as f:
    document_context = f.read()

SYSTEM_INSTRUCTION = (
    "You are a model-risk analyst answering questions about a bank model "
    "inventory document. After any reasoning, write the final answer on a "
    "single line prefixed with 'ANSWER: ' — just the requested value (a number, "
    "a model identifier, or a one-word status). No units, no rounding."
)

factor_cols = [c for c in df.columns if c != "question"]
rows_out = []

t0 = time.time()
for i, row in df.iterrows():
    base_q = base_q_lookup[row["question"]]
    factor_assignment = {c: row[c] for c in factor_cols}
    rephrased = rephraser.rephrase(base_q, factor_assignment)

    resp = client.messages.create(
        model=DEFAULT_TUTORIAL_MODEL,
        max_tokens=256,
        system=SYSTEM_INSTRUCTION,
        messages=[{
            "role": "user",
            "content": f"<document>\\n{document_context}\\n</document>\\n\\n{rephrased.llm_prompt}",
        }],
    )
    raw = resp.content[0].text.strip()

    # Apply the question's type-aware parser before scoring — same code path
    # that DOEGMSBenchmark.run uses internally. This is the scorer of record;
    # raw exact-match on Claude's prose output is meaningless for factor analysis.
    parser = PARSERS.get(base_q.answer_type, PARSERS["str"])
    answer = parser(raw)
    s = score_answer(answer, base_q.ground_truth)

    rows_out.append({
        "row": i,
        "base_qid": base_q.qid,
        "category": base_q.category,
        "question": row["question"],
        "raw_answer": raw[:200],
        "parsed_answer": str(answer),
        "ground_truth": str(base_q.ground_truth),
        "correct": bool(s.correct),
        **factor_assignment,
    })

elapsed = time.time() - t0
results_df = pd.DataFrame(rows_out)
acc = results_df["correct"].mean()
print(f"  wall time : {elapsed:.1f}s")
print(f"  accuracy  : {results_df['correct'].sum()}/{len(results_df)} = {acc:.1%}")
print()
print("Per-category accuracy:")
for cat, g in results_df.groupby("category"):
    print(f"  {cat:20s} {g['correct'].sum():2d}/{len(g):2d} ({g['correct'].mean():.0%})")\
"""))

cells.append(md("""\
## 8. Factor Effect Analysis

Two complementary views — **single-factor** (which individual factors drive failures?) and **joint** (does the design as a whole explain variation in correctness?).

`DOEAnalyzer.run_logistic` returns a deviance table with one row per factor; `apply_fdr_correction` adds Benjamini-Hochberg-adjusted p-values across the family of factor tests so we control the false-discovery rate when scanning 10+ factors.\
"""))

cells.append(code("""\
factor_cols = [c for c in df.columns if c != "question"]
ana = DOEAnalyzer.from_dataframe(
    results_df, factors=factor_cols + ["question"]
)
deviance = ana.run_logistic(metric="correct")
cols = ["factor", "deviance", "p_value", "pseudo_r2", "significant", "n_levels"]
print("Single-factor analysis of deviance (top 10 by p-value):")
print(deviance.sort_values("p_value")[cols].head(10).to_string(index=False))\
"""))

cells.append(code("""\
deviance_adj = ana.apply_fdr_correction(deviance, alpha=0.05, method="fdr_bh")
print("Benjamini-Hochberg FDR-adjusted p-values:")
print()
cols = ["factor", "p_value", "p_value_adj", "significant_adj", "pseudo_r2"]
print(deviance_adj.sort_values("p_value_adj")[cols].head(10).to_string(index=False))\
"""))

cells.append(md("""\
The **joint model** fits all factors simultaneously and reports pseudo-R², AIC, BIC, ROC AUC, and the Hosmer-Lemeshow goodness-of-fit p-value. At `N_RUNS_DEMO=60` with 30 questions + 10 factors (~55 parameters), the joint model is *underdetermined* — the parameter-to-observation ratio is too low for a clean fit. Scaling to `N_RUNS_DEMO >= 300` (paper uses 880) lands inside the well-determined regime.\
"""))

cells.append(code("""\
try:
    joint = ana.run_joint_model(metric="correct")
    print("Joint logistic model (all factors fitted simultaneously):")
    for k, v in joint.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
except Exception as e:
    print(f"Joint model not fittable at N_RUNS_DEMO={N_RUNS_DEMO} (expected at small n): "
          f"{type(e).__name__}")
    print("Scale to N_RUNS_DEMO >= 300 for a well-determined joint logistic.")\
"""))

cells.append(md("""\
## 9. Take-Aways

1. **`graphdoe.design()` is one line.** It returns a `DesignMatrix` whose `.matrix` DataFrame has the question column pre-picked. Total LLM-call budget = `n_runs`, not `n_questions × n_runs`.
2. **Sobol+Refine beats vanilla Sobol** on `φ_p`, `min_dist`, and `coverage` at the same `n_runs`. The improvement scales with the dimension and cardinality of the factor space.
3. **Graph baselines are mandatory.** Always verify 100% on `graph_answer_fn` before paying for any LLM call — anything less means the ground truth is broken.
4. **FDR correction matters.** With 10+ factors, naive p-value thresholds will surface false positives. Apply Benjamini-Hochberg before reporting which factors "matter."
5. **Joint models need n_obs >> n_params.** Rule of thumb: 10× more observations than parameters. At 30 questions + 10 factors, the joint logistic needs ~550 runs minimum.\
"""))

cells.append(md("""\
## 10. Next Steps

- **NB 04** — `04_ingestion.ipynb` — how the ch04 ground-truth store was built (Ch04)
- **NB 05** — `05_question_generation.ipynb` — generators that produce graph-verified base questions (Ch05)
- **NB 07** — `07_statistical_analysis.ipynb` — joint-model AIC/BIC/ROC AUC at scale (Ch07)

### Scaling to paper settings

To replicate the FinStructBench case study, change three knobs:

```python
DESIRED_BASE_Q = 44             # paper uses 44 graph-verified base questions
CORE_FACTORS = [...]            # paper uses 20 contextual factors (factor_group="comprehensive")
N_RUNS_DEMO = 880               # paper budget; ≈45 min on Sonnet 4.6, ~$50-80
```

For the design-comparison contrast in the paper (Sobol vs Sobol+Refine), run sections 5–8 twice — once with `method="sobol"`, once with `method="sobol+refine"` — and contrast the joint-model metrics from §8.

### See also

- `gmsh_testing_ext/chapters/ch03.tex` — formal definitions, fill-distance argument, Fisher-information derivation
- `graphdoe/papers/paper_jasa.tex` — full methodology + 880-run case study
- `graphdoe/papers/simulation_power_study.py` — simulation behind the +22pp statistical-power claim under known ground truth
- `graphdoe.analysis.DOEAnalyzer` — single-factor logistic, Wilson CIs, odds ratios, two-way interactions, FDR correction\
"""))


def build_and_execute() -> None:
    nb = new_notebook()
    nb.cells = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (spark-venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    print(f"Writing {NB_PATH} ({len(cells)} cells)...")
    NB_PATH.write_text(nbformat.writes(nb))

    print("Executing notebook end-to-end...")
    t0 = time.time()
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  execution wall time: {time.time() - t0:.1f}s")

    print(f"Writing executed notebook back to {NB_PATH}...")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
