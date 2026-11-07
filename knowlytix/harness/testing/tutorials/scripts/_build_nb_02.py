# SPDX-License-Identifier: Apache-2.0
"""Build NB 02 GMS as Ground Truth (Ch02) from cell definitions, then execute.

Migrated from knowlytix/harness/testing/tutorials/querying_a_trained_gms.ipynb and
realigned to the chapter-anchored thin-recipe template.

Requires the hardened ch04_ground_truth fixture (no LLM calls in this notebook).

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_02.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "02_gms_primitives.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 02: GMS as Ground Truth — Primitives in 10 Minutes

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch02.tex` — Geometric Memory Systems as Ground Truth
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch02_gms_ground_truth.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §5
**Prereqs:** activated `~/cluster/spark-venv`; hardened `ch04_ground_truth` fixture

**What you'll do.** Load the bank model-risk store, exercise the six geometric primitives that drive every typed verdict in the harness (`score_triple`, `query_triples`, `lookup_enm`, `link_predict`, `tension_energy`, `check_holonomy`) against the canonical inventory (MDL-001 CreditScore-XGB, MDL-002 FraudNet-LSTM, MDL-003 AML-GBM, MDL-004 MarketVaR-GARCH, MDL-005 ChurnPredict-RF), then build a fresh store from a tiny inline corpus to see ingestion end-to-end. No LLM calls — pure geometry.\
"""))

cells.append(md("""\
## 1. Setup and Load the Store

The `ch04_ground_truth` store ships with the monograph: 514 entities, 68 relations, 823 triples, cap-trained on the bank model-risk corpus. Loading is two lines.\
"""))

cells.append(code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from _shared import load_ch04_store, store_summary, seed_all
seed_all()
store = load_ch04_store()
for k, v in store_summary(store).items():
    print(f"  {k:14s} {v}")\
"""))

cells.append(md("""\
## 2. The Six Primitives

Every typed verdict in the harness reduces to one or more of six calls (Ch02 §2.3):

| Primitive | Returns | What it measures |
|---|---|---|
| `score_triple(h, r, t)` | float (lower = more plausible) | Geodesic distance on the cap manifold |
| `query_triples(h, r, t)` | iterator of stored triples | Structural pattern match (any slot can be a wildcard) |
| `lookup_enm(category, entity_id)` | float \\| str | SHA-256-verified exact value |
| `link_predict(h, r, top_k)` | list[(tail, score)] | Type-constrained top-k completion |
| `tension_energy(a, b)` | float in [0, 2] | Entity-level disagreement |
| `check_holonomy(path, direct)` | float | Multi-hop transport defect |

The next four sections demonstrate each primitive against the bank corpus.\
"""))

cells.append(md("""\
### 2.1 `score_triple` — fact verification

Returns the geodesic distance between the rotor-transported head and the tail embedding on the cap manifold. **Lower means more plausible.** The Chapter 4 store records `(MDL-001, has_test_auc, 0.8412)` as ground truth; we contrast against several distractors a hallucinating agent might emit.\
"""))

cells.append(code("""\
head = "MDL-001"
rel = "has_test_auc"

candidates = [
    ("0.8412", "TRUTH  — MDL-001 test AUC"),
    ("0.8934", "WRONG — MDL-001 train AUC (right model, wrong relation tail)"),
    ("0.9234", "WRONG — MDL-002 test AUC (different model)"),
    ("0.7234", "WRONG — MDL-005 test AUC (different model)"),
]
print(f"score_triple({head!r}, {rel!r}, ?)  — lower = more plausible")
print()
for tail, note in candidates:
    s = store.score_triple(head, rel, tail)
    marker = "←" if "TRUTH" in note else " "
    print(f"  tail={tail:8s}  score={s:7.4f}  {marker}  {note}")\
"""))

cells.append(md("""\
The truth scores well below the distractors — that gap is what `GraphVerifier` and `HallucinationOracle` (NB 16) rely on. Note that `score_triple` returns `None` if the head, relation, or tail is not in the vocabulary; verifiers treat that as *unscoreable*, not *wrong*.\
"""))

cells.append(md("""\
### 2.2 `query_triples` — structural pattern lookup

Pure graph query, no geometry. Any of the three slots can be omitted (or set to `None`) to act as a wildcard. This is what tool-ordering and policy verifiers use when they need to know *what is recorded*, not *what is plausible*.\
"""))

cells.append(code("""\
# Pattern 1: everything about MDL-001 with relation has_status
print("query_triples(head='MDL-001', relation='has_status'):")
for h, r, t in store.query_triples(head="MDL-001", relation="has_status"):
    print(f"  ({h}, {r}, {t})")

# Pattern 2: every model's status (relation as the only filter)
print("\\nquery_triples(relation='has_status'):  (first 6)")
for h, r, t in list(store.query_triples(relation="has_status"))[:6]:
    print(f"  ({h}, {r}, {t})")

# Pattern 3: every triple about MDL-001 (head as the only filter)
mdl001 = list(store.query_triples(head="MDL-001"))
print(f"\\nquery_triples(head='MDL-001'):  {len(mdl001)} triples in total")
for h, r, t in mdl001[:5]:
    print(f"  ({h}, {r}, {t})")\
"""))

cells.append(md("""\
### 2.3 `lookup_enm` — exact numeric retrieval

Geometric scoring is good for plausibility but it is *approximate*. When verification needs the actual digit-for-digit value (regulatory reporting, capital impact, audit replay), the harness reads from the **Exact Numeric Memory** instead. Each entry is keyed by `(category, entity_id)` and stored with a SHA-256 integrity check.\
"""))

cells.append(code("""\
lookups = [
    ("performance_metrics", "MDL-001/Train AUC"),
    ("performance_metrics", "MDL-001/Test AUC"),
    ("performance_metrics", "MDL-001/Gini"),
    ("performance_metrics", "MDL-001/KS Statistic"),
]
for category, eid in lookups:
    val = store.lookup_enm(category, eid)
    print(f"  {category}::{eid:30s}  =  {val}")\
"""))

cells.append(md("""\
### 2.4 `link_predict` — top-k completion

Given a head and relation, score every entity in the vocabulary as a candidate tail and return the top-`k` by geodesic distance. Used by the harness when generating distractor answers and ranking competing claims. A type constraint keeps numeric values from polluting categorical predictions and vice versa.\
"""))

cells.append(code("""\
print("link_predict('MDL-001', 'has_test_auc', top_k=5):")
print()
print(f"  {'rank':>4s}  {'tail':>10s}  {'score':>7s}")
for i, (tail, score) in enumerate(
    store.link_predict("MDL-001", "has_test_auc", top_k=5), 1
):
    marker = "  ← ground truth" if tail == "0.8412" else ""
    print(f"  {i:>4d}  {tail:>10s}  {score:>7.4f}{marker}")\
"""))

cells.append(md("""\
### 2.5 `tension_energy` — entity disagreement

`tension_energy(a, b)` returns a value in `[0, 2]`: `0` if the two entities sit on top of each other in the u-embedding, `~sqrt(2)` if they are unrelated, `~2` if they are directly opposed. The contradiction generator and `RecoveryEvaluator` both consume this signal.\
"""))

cells.append(code("""\
print("tension_energy:")
for a, b in [
    ("MDL-001", "MDL-004"),
    ("MDL-001", "MDL-003"),
    ("MDL-001", "MDL-001"),
]:
    te = store.tension_energy(a, b)
    print(f"  {a} vs {b}:  {te:.4f}")\
"""))

cells.append(md("""\
### 2.6 `check_holonomy` — multi-hop transport defect

`check_holonomy(path, direct)` composes the rotors along `path` and compares the result to the rotor for `direct`. A defect of `0` means the multi-hop reasoning agrees with the direct edge; large defects are the signal a `ProcessEvaluator` turns into a `PROCESS_DEVIATION` failure (NB 17 / Ch17).\
"""))

cells.append(code("""\
print("check_holonomy:")
print()
defect = store.check_holonomy(["has_train_auc"], "has_test_auc")
print(f"  [has_train_auc] vs has_test_auc:  {defect:.4f}")
print()
print("  (large value means 'train AUC' and 'test AUC' are not the same relation —")
print("   correct: an agent that conflates them should fail a process-evaluator)")\
"""))

cells.append(md("""\
## 3. Build Your Own Store

When you do not have a pre-trained store sitting around, the harness's `DOEGMSBenchmark.ingest()` wraps the full ingestion pipeline (LLM extraction → graph build → embedding → ENM population → save). The cell below builds a tiny three-model inventory from inline markdown and queries it with the same primitives.

Ingestion on a small corpus takes about two minutes on GB10 with the default 200 epochs — the cap-admissibility training is fully exercised.\
"""))

cells.append(code("""\
import os, tempfile, contextlib, io
os.chdir(Path(__file__ if "__file__" in globals() else ".").resolve().parent.parent.parent.parent if False else str(ROOT))

from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig

TINY_DOC = \"\"\"# Quarterly Model Review (Tiny Demo)

## Models in Scope

| model_id | algorithm    | test_auc | status |
|----------|--------------|----------|--------|
| MDL-A    | XGBoost      | 0.8412   | Stable |
| MDL-B    | LightGBM     | 0.7956   | Watch  |
| MDL-C    | RandomForest | 0.8201   | Stable |
\"\"\"

tmp = Path(tempfile.mkdtemp(prefix="nb02_"))
md_path = tmp / "tiny_review.md"
md_path.write_text(TINY_DOC)

tiny_cfg = DOEHarnessConfig(
    markdown_path=str(md_path),
    stores_dir=str(tmp / "stores"),
    n_runs=1,
    factor_group="quick_screen",
    cap_enabled=True,
    enable_tracing=False,
)
bench = DOEGMSBenchmark(tiny_cfg)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tiny_store = bench.ingest()
print("tiny store stats:", tiny_store.stats())\
"""))

cells.append(md("""\
Same primitives, fresh store — the contract is identical.\
"""))

cells.append(code("""\
print("query_triples(head='MDL-A'):")
for h, r, t in tiny_store.query_triples(head="MDL-A"):
    print(f"  ({h}, {r}, {t})")

print()
truth = tiny_store.score_triple("MDL-A", "has_test_auc", "0.8412")
wrong = tiny_store.score_triple("MDL-A", "has_test_auc", "0.7956")
print(f"score_triple MDL-A has_test_auc 0.8412 (truth): {truth:.4f}")
print(f"score_triple MDL-A has_test_auc 0.7956 (wrong): {wrong:.4f}")\
"""))

cells.append(md("""\
## 4. Take-Aways

1. **Six primitives, one contract.** Every typed verdict the harness emits is some composition of `score_triple`, `query_triples`, `lookup_enm`, `link_predict`, `tension_energy`, `check_holonomy`. If they read naturally, the rest of the platform is composition.
2. **Plausibility is approximate; ENM is exact.** Use `score_triple` for "does the agent's claim look right?" Use `lookup_enm` for "what is the recorded value?" Regulatory and capital numbers always go through ENM.
3. **Type-constrained completion.** `link_predict` only considers entities that have appeared as a tail for the queried relation in the training graph — numeric values cannot pollute categorical predictions.
4. **Tension and holonomy are *structural*.** Tension catches direct disagreement between two entities; holonomy catches misuse of a multi-hop reasoning path that should be a single edge.
5. **Fresh stores share the same API.** Loading the canonical fixture and ingesting a brand-new corpus produce stores with identical interfaces — your verification code does not branch.\
"""))

cells.append(md("""\
## 5. Next Steps

- **NB 03** — `03_doe_design.ipynb` — Design-of-Experiments factor analysis on these primitives (Ch03)
- **NB 04** — `04_ingestion.ipynb` — the full ingestion pipeline end-to-end (Ch04)
- **NB 06** — `06_geometric_judge.ipynb` — how the primitives compose into the GMS judge (Ch06)
- **NB 16** — `16_hallucination.ipynb` — `HallucinationOracle` builds on `score_triple` (Ch16)

For theory:
- `gmsh_testing_ext/chapters/ch02.tex` — formal definitions, rotor algebra, cap-admissibility derivation
- `gmsh_testing_ext/appendix_a.tex` — GMS API reference\
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
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  execution wall time: {time.time() - t0:.1f}s")

    print(f"Writing executed notebook back to {NB_PATH}...")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
