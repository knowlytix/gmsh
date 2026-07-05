# SPDX-License-Identifier: Apache-2.0
"""Build NB 00 Quick Start from cell definitions, then execute.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_00.py

Writes to: knowlytix/harness/testing/user_guide_tutorials/notebooks/00_quick_start.ipynb

Requires: ANTHROPIC_API_KEY (or pass(1) entry `anthropic/api-key`).
No mock-LLM fallback: missing key raises AnthropicAPIKeyError.
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "00_quick_start.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

cells = []

cells.append(md("""\
# Tutorial 00: Quick Start — Test an Agent Against a Bank Model-Risk Corpus

**Conceptual deep-dive:** `gmsh_testing_ext/preface.tex`, `gmsh_testing_ext/chapters/ch01.tex` — Framing the Testing Problem
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch01_agentic_testing_problem.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §3
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY` (env var or `pass show anthropic/api-key`); hardened `ch04_ground_truth` fixture (ships with the monograph)

**What you'll do.** Load the canonical bank model-risk store (MDL-001 through MDL-005: CreditScore-XGB, FraudNet-LSTM, AML-GBM, MarketVaR-GARCH, ChurnPredict-RF), generate test questions across thirteen geometric and taxonomy generators, score a real Claude Sonnet 4.6 evaluator end-to-end, and read off per-category accuracy plus typed verdicts ready for an audit report.

> This notebook is the **reference template** every other tutorial copies. It exercises the harness end-to-end with a real LLM and the canonical model-risk corpus — no mocks, no fallbacks. If the API key cannot be resolved, the notebook raises `AnthropicAPIKeyError` and stops, by design.\
"""))

cells.append(md("""\
## 1. Setup

Load shared helpers, resolve the Anthropic API key, pin random seeds, and configure plotting. The `_shared` module lives next to this notebook and centralizes fixture paths plus the canonical key resolver, so every notebook in this guide stays consistent.\
"""))

cells.append(code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from _shared import (
    KG_ROOT,
    CH04_STORE_PATH,
    DEFAULT_TUTORIAL_MODEL,
    load_ch04_store,
    make_anthropic_client,
    seed_all,
    setup_plotting,
    store_summary,
)

seed_all()
setup_plotting()
client = make_anthropic_client()             # raises AnthropicAPIKeyError if missing
print("kg-memory root :", KG_ROOT)
print("fixture        :", CH04_STORE_PATH)
print("evaluator model:", DEFAULT_TUTORIAL_MODEL)
print("client ready   :", client.__class__.__name__)\
"""))

cells.append(md("""\
## 2. Load the Ground-Truth Store

The `ch04_ground_truth` store is the canonical hardened fixture from the testing monograph. It encodes a bank model-inventory corpus — five production models across credit scoring, fraud detection, anti-money-laundering, market-risk VaR, and customer churn — together with their test metrics, risk tiers, governance status, and inter-model dependencies. The store ships trained with cap-admissibility (`cap_enabled=True`, `ρ_max = π/2`) so geometric scoring matches the monograph's blessed numerics.\
"""))

cells.append(code("""\
store = load_ch04_store()
stats = store_summary(store)
for k, v in stats.items():
    print(f"  {k:14s} {v}")\
"""))

cells.append(md("""\
We can probe the corpus directly through the store's geometric primitives. Each model in the inventory has a model-card row in ENM (exact numerical memory) — the harness uses these as ground truth for numerical questions.\
"""))

cells.append(code("""\
# Inspect a few model-inventory rows so you can see what the corpus encodes.
sample_keys = [k for k in store.enm.keys() if k.type == "model_inventory_summary"][:5]
for k in sample_keys:
    print(f"  [{k.type}] {k.id}")\
"""))

cells.append(md("""\
## 3. Configure the Benchmark

`DOEHarnessConfig` is the single dataclass that controls everything the benchmark does. For the Quick Start we set just the essentials:

- `factor_group="quick_screen"` — three Design-of-Experiments factors (clarity, entity aliasing, reasoning cue)
- `n_runs=2`, `max_per_category=3` — a small but representative sweep (~78 questions across 13 generators)
- `cap_enabled=True` — match the fixture's training geometry
- `enable_typed_verdicts=True` — emit Chapter-10 audit-shape verdicts per question
- `target_model="claude-sonnet-4-6"` — the SUT we're scoring\
"""))

cells.append(code("""\
import os
os.chdir(KG_ROOT)  # harness uses relative paths anchored at kg-memory/

from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig

config = DOEHarnessConfig(
    markdown_path="gmsh_testing_ext/lab_stores/ch04_ground_truth/documents/combined.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen",
    n_runs=2,
    max_per_category=3,
    cap_enabled=True,
    enable_typed_verdicts=True,
    enable_tracing=False,
    target_type="llm",
    target_model=DEFAULT_TUTORIAL_MODEL,
)
print(f"config: factor_group={config.factor_group} n_runs={config.n_runs} "
      f"max_per_category={config.max_per_category} model={config.target_model}")\
"""))

cells.append(md("""\
## 4. Load (or Build) the Store via the Benchmark

`bench.ingest()` is idempotent. Because the store already exists at `stores_dir/store_name/`, it loads in under a second instead of retraining. Re-running this cell in your own work is safe.\
"""))

cells.append(code("""\
import time
bench = DOEGMSBenchmark(config)
t0 = time.time()
bench_store = bench.ingest()
print(f"  load time: {time.time() - t0:.2f}s")
print(f"  store: {bench_store.stats()['entities']} entities, "
      f"{bench_store.stats()['triples']} triples, "
      f"{bench_store.stats()['enm_entries']} ENM rows")\
"""))

cells.append(md("""\
## 5. Generate Test Questions from Geometric Structure

`generate_questions()` runs every configured generator (eight geometric + five taxonomy generators by default) and folds the results through the Design-of-Experiments matrix. Each `DOEQuestion` wraps a `GeneratedQuestion` (with `.category` and `.natural_language`) plus the factor assignment that produced it.

See `gmsh_testing_ext/chapters/ch05.tex` for the derivation of each generator's geometric anchor, and NB 05 (`05_question_generation.ipynb`) for a per-generator walkthrough.\
"""))

cells.append(code("""\
questions = bench.generate_questions()
categories = sorted({q.question.category for q in questions})
print(f"generated {len(questions)} questions across {len(categories)} generators:")
for c in categories:
    n = sum(1 for q in questions if q.question.category == c)
    print(f"  {c:24s} ({n} questions)")
print()
print("sample question:")
print(f"  base_qid: {questions[0].base_qid}")
print(f"  category: {questions[0].question.category}")
print(f"  factors : {questions[0].factor_assignment}")
print(f"  text    : {questions[0].question.natural_language[:180]}")\
"""))

cells.append(md("""\
## 6. Build a Real LLM Evaluator

The harness contract is just a callable `evaluator(question, context) -> str`. The shipped `LLMEvaluator` wraps a raw Claude API call: it builds a prompt from the question text and the relevant document context, sends it to the configured model, and returns the answer string. No mock fallback — if the API key is unset the constructor raises.

For tool-using agents instead of bare LLM calls, use `AgentEvaluator` (NB 11) or `make_evaluator("agent", ...)` (NB 11/12).\
"""))

cells.append(code("""\
from knowlytix.harness.testing import LLMEvaluator

evaluator = LLMEvaluator(client=client, model=DEFAULT_TUTORIAL_MODEL)
# Smoke test on a hand-built question + context window.
test_context = (
    "Model MDL-001 (CreditScore-XGB) is a high-risk-tier credit-decisioning "
    "model. Its most recent test_auc was 0.8412 on the credit panel hold-out."
)
print(evaluator("What is the test_auc of MDL-001?", test_context))\
"""))

cells.append(md("""\
## 7. Run the Evaluation

`bench.run(evaluator=...)` walks every generated question, calls the evaluator with the question text and the relevant context window, and scores the answer against the geometric judge (`gmsh_testing_ext/chapters/ch06.tex`). With `enable_typed_verdicts=True` we also get a `TypedVerdict` per question — the structured form the audit pipeline (Chs 8–10) consumes.

This takes roughly 1–2 minutes of wall-clock time on a small sweep; the benchmark library emits one progress line per question which we redirect to a buffer so the cell output stays readable.\
"""))

cells.append(code("""\
import contextlib
import io

buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    result = bench.run(evaluator=evaluator)
elapsed = time.time() - t0

accuracy = result.total_correct / result.total_questions if result.total_questions else 0.0
print(f"  run time:        {elapsed:.1f}s")
print(f"  total questions: {result.total_questions}")
print(f"  correct:         {result.total_correct}")
print(f"  accuracy:        {accuracy:.1%}")\
"""))

cells.append(md("""\
## 8. Per-Category Breakdown

The benchmark records accuracy per generator category. Variation across categories is the diagnostic signal — a flat profile means the agent treats every probe type the same; a sharp drop on adversarial or boundary categories is a red flag for safety review.

For a production bank model-risk evaluation you'd render this same table into the audit report (NB 19) under risk-tiered cuts, where governance and recovery generators carry tighter pass thresholds than plausibility.\
"""))

cells.append(code("""\
import pandas as pd
from collections import defaultdict
agg = defaultdict(lambda: {"total": 0, "correct": 0})
for q in result.questions:
    agg[q.category]["total"] += 1
    agg[q.category]["correct"] += int(q.correct)

rows = [
    {"category": cat,
     "total": v["total"],
     "correct": v["correct"],
     "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
    for cat, v in agg.items()
]
df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
df\
"""))

cells.append(md("""\
## 9. Inspect Typed Verdicts

When `enable_typed_verdicts=True`, the benchmark emits a `TypedVerdict` per question keyed by `doe_qid` (Chapter 10's audit-shape verdict). Each verdict carries `correct` (the surface pass/fail), a list of `sub_verdicts` per claim, the geometric measurements (`geodesic_confidence`, `tension_score`, `holonomy_score`), and a list of `failures` with attribution.

> **Note.** Without an explicit calibration artifact loaded into the bench, the sub-verdicts come back tagged `NOT_CALIBRATED` with zero geometric scores — the surface `correct` flag still matches the per-category accuracy above, but the per-claim breakdown is uncalibrated. NB 20 (`20_calibration.ipynb`) shows how to thread a `CalibrationSnapshot` through so the sub-verdicts carry meaningful thresholds.\
"""))

cells.append(code("""\
verdicts = bench.typed_verdicts or {}
print(f"{len(verdicts)} typed verdicts keyed by doe_qid")
print()

# TypedVerdict carries: correct, n_claims, n_passed, n_failed, sub_verdicts,
# failures, geodesic_confidence, tension_score, holonomy_score, overall_confidence.
correct = [v for v in verdicts.values() if v.correct]
incorrect = [v for v in verdicts.values() if not v.correct]
print(f"  correct verdicts:   {len(correct)}")
print(f"  incorrect verdicts: {len(incorrect)}")

for label, sample_list in (("CORRECT", correct), ("INCORRECT", incorrect)):
    if sample_list:
        s = sample_list[0]
        print()
        print(f"{label} verdict sample (question_id={s.question_id}):")
        print(f"  correct             : {s.correct}")
        print(f"  n_claims            : {s.n_claims}")
        print(f"  n_passed / n_failed : {s.n_passed} / {s.n_failed}")
        print(f"  geodesic_confidence : {s.geodesic_confidence:.3f}")
        print(f"  tension_score       : {s.tension_score:.3f}")
        print(f"  holonomy_score      : {s.holonomy_score:.3f}")
        print(f"  overall_confidence  : {s.overall_confidence:.3f}")
        if s.failures:
            codes = [f.code.value for f in s.failures[:3]]
            print(f"  failures (top 3)    : {codes}")\
"""))

cells.append(md("""\
## 10. Visualize Per-Category Accuracy

A bar chart of per-category accuracy is the fastest read on which probe families the SUT is failing. In production you'd render the same data into the audit report (NB 19) or the live dashboard (NB 23) under risk-tiered cuts.\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.barh(df["category"], df["accuracy"], color="steelblue")
ax.set_xlim(0, 1)
ax.set_xlabel("accuracy")
ax.set_title(f"Quick Start: per-category accuracy on bank model-risk corpus\\n"
             f"SUT = {DEFAULT_TUTORIAL_MODEL}  ·  n = {result.total_questions} questions")
for i, (cat, acc) in enumerate(zip(df["category"], df["accuracy"])):
    ax.text(acc + 0.01, i, f"{acc:.0%}", va="center")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 11. Take-Aways

1. **Load, don't rebuild.** The `ch04_ground_truth` fixture ships hardened with the monograph — every tutorial in this guide loads it read-only via `_shared.load_ch04_store()`.
2. **`DOEGMSBenchmark` is the front door.** One config dataclass, one ingest call, one run call. Every tutorial in Parts II–V extends one or more of these three lines.
3. **Real LLM, no mock.** `LLMEvaluator` exercises the actual SUT against the bank model-risk corpus. The harness raises loudly when the key is missing — there is no fallback path that masks a broken integration.
4. **Typed verdicts are the audit shape.** Flip `enable_typed_verdicts=True` and the pipeline emits the Chapter-10 verdict form ready for risk-tiered release-gate evaluation (NB 19).
5. **Category accuracy is the diagnostic.** A flat profile across generators means the agent isn't reading the corpus; a sharp drop on adversarial / boundary categories means the agent is overfit to happy-path prompts and likely unsafe for production deployment.\
"""))

cells.append(md("""\
## 12. Next Steps

- **NB 01** — `01_framing.ipynb` — what the testing problem actually is, framed against a model-risk workflow (Ch01)
- **NB 02** — `02_gms_primitives.ipynb` — the six geometric primitives in 10 minutes, applied to the bank corpus (Ch02)
- **NB 11** — `11_sut_adapters.ipynb` — wire a tool-using agent through `AgentAdapter` instead of a bare LLM (Ch11)
- **NB 19** — `19_release_gates.ipynb` — risk-tiered cuts, audit report, and provenance chain (Ch19)

For theory:
- `gmsh_testing_ext/preface.tex` — why test agentic AI geometrically
- `gmsh_testing_ext/chapters/ch01.tex` — the testing problem
- `gmsh_testing_ext/chapters/ch04.tex` — how the bank model-risk fixture was constructed\
"""))


# ---------------------------------------------------------------------------
# Build and execute
# ---------------------------------------------------------------------------

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
