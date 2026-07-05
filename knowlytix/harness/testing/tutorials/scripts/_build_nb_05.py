# SPDX-License-Identifier: Apache-2.0
"""Build NB 05 — Generating Questions from Geometric Structure (Ch05)."""
from __future__ import annotations
import time
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "05_question_generation.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 05: Generating Questions from Geometric Structure

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch05.tex` — Generating Test Questions from Geometric Structure
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch05_question_generation.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §8
**Prereqs:** NB 02 (Primitives), NB 04 (Ingestion)

**What you'll do.** Generate test questions from the bank ground-truth store using the 13 available generator families. Inspect what each generator produces, what `claim_type` it targets, and what answer format the harness expects back. Each generated question carries a `graph_answer_fn` that derives the ground truth directly from store traversal — 100% correct by construction.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")
from _shared import load_ch04_store, seed_all
from knowlytix.harness.testing.bridge import generate_base_questions
seed_all()
store = load_ch04_store()
print(f"store: {store.stats()['entities']} entities, {store.stats()['triples']} triples")\
"""))

cells.append(md("""\
## 2. Concept

A generator is a function `store -> list[GeneratedQuestion]`. The harness ships 13 generators across two families:

**Geometric (8)** — questions whose answers reduce to a single GMS primitive:
- `plausibility` (score_triple)
- `tension` (tension_energy)
- `holonomy` (check_holonomy)
- `link_prediction` (link_predict)
- `multi_hop_transport` (composed rotors)
- `phase_comparison` (phase encoder ranges)
- `conditional` (conditional probability via graph)
- `procedural` (process-deviation traces)

**Taxonomy (5)** — questions targeting categorical SUT behaviors:
- `boundary` (edge cases)
- `adversarial` (jailbreak / prompt injection)
- `robustness` (paraphrase variants)
- `recovery_test` (fault-injection survival)
- `governance` (policy-relevant queries)

Every generated question carries a `graph_answer_fn(store)` that derives the ground truth programmatically — re-running it on the same store always yields the same answer.\
"""))

cells.append(md("""\
## 3. Generate from All 13 Families\
"""))

cells.append(code("""\
all_names = [
    "plausibility", "tension", "holonomy", "link_prediction",
    "multi_hop_transport", "phase_comparison", "conditional", "procedural",
    "boundary", "adversarial", "robustness", "recovery_test", "governance",
]
import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    questions = generate_base_questions(
        store=store, generator_names=all_names,
        max_per_category=3, seed=42,
    )
print(f"generated {len(questions)} questions across {len(set(q.category for q in questions))} families")\
"""))

cells.append(md("""\
## 4. Counts and Sample per Family\
"""))

cells.append(code("""\
from collections import defaultdict
by_cat = defaultdict(list)
for q in questions:
    by_cat[q.category].append(q)

for cat in sorted(by_cat.keys()):
    qs = by_cat[cat]
    sample = qs[0]
    print(f"[{cat:20s}]  n={len(qs)}  example: {sample.natural_language[:110]}")\
"""))

cells.append(md("""\
## 5. Inspect One Question End-to-End\
"""))

cells.append(code("""\
q = by_cat["plausibility"][0]
print(f"qid          : {q.qid}")
print(f"category     : {q.category}")
print(f"answer_type  : {q.answer_type}")
print(f"question     : {q.natural_language}")
print(f"ground_truth : {q.ground_truth!r}")
print()
print(f"graph_answer_fn(store) returns: {q.graph_answer_fn(store)!r}")
print(f"(this is the value the harness compares against the LLM answer)")\
"""))

cells.append(md("""\
## 6. Verify Graph Baseline — 100% by Construction\
"""))

cells.append(code("""\
from knowlytix.benchmark.scorers import score_answer

n_correct, n_skipped = 0, 0
for q in questions:
    try:
        ga = q.graph_answer_fn(store)
        s = score_answer(ga, q.ground_truth)
        if s.correct:
            n_correct += 1
    except Exception:
        n_skipped += 1
total = len(questions) - n_skipped
print(f"graph baseline: {n_correct}/{total} ({n_correct/total:.0%}) ({n_skipped} skipped)")\
"""))

cells.append(md("""\
## 7. Visualization — Question Counts by Family\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np
cats = sorted(by_cat.keys())
counts = [len(by_cat[c]) for c in cats]
colors = ["#3aa56b" if c in ("plausibility","tension","holonomy","link_prediction",
                              "multi_hop_transport","phase_comparison","conditional",
                              "procedural") else "#4a8cd1" for c in cats]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(cats, counts, color=colors)
ax.set_xlabel("# questions generated")
ax.set_title("Question yield per generator family (geometric=green, taxonomy=blue)")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 8. Take-Aways

1. **13 generators, two families.** Geometric questions test a single primitive in isolation; taxonomy questions test categorical SUT behaviors that any generator family might surface.
2. **`graph_answer_fn` is the ground-truth oracle.** No human annotation needed; the answer is derived from store traversal, which is what makes the generated benchmark reproducible.
3. **`max_per_category` caps the yield.** With max=3 we generate up to 39 questions; in a release-gate run you typically want max=10 (≈130 base questions × n_runs design points).
4. **The graph baseline is always 100%.** If it isn't, your ground truth is broken — abort the bench run.
5. **Generators are pluggable.** Build a custom generator by subclassing `BaseGenerator` and adding it to the registry; the harness picks it up via `generator_names=[...]`.\
"""))

cells.append(md("""\
## 9. Next Steps

- **NB 03** — `03_doe_design.ipynb` — design-of-experiments matrix over these base questions (Ch03)
- **NB 08** — `08_claim_decomposition.ipynb` — how claims are extracted from LLM answers to these questions (Ch08)

For theory: `gmsh_testing_ext/chapters/ch05.tex`.\
"""))


def build_and_execute():
    nb = new_notebook()
    nb.cells = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (spark-venv)",
                                      "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    print(f"Writing {NB_PATH} ({len(cells)} cells)...")
    NB_PATH.write_text(nbformat.writes(nb))
    print("Executing...")
    t0 = time.time()
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
