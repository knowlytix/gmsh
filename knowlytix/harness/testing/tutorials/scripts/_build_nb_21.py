# SPDX-License-Identifier: Apache-2.0
"""Build NB 21 — GMS Training & Diagnostics — from cell definitions, then execute.

Demonstrates GMSDiagnostics on the ch04 ground-truth fixture: the 12-check
post-training health report that should pass before a store is considered
ship-ready. No LLM calls, runs in seconds.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_21.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "21_gms_diagnostics.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 21: GMS Training & Post-Training Diagnostics

**Conceptual deep-dive:** `gmsh_testing_ext/appendix_f.tex` — GMS Training and Post-Training Diagnostics
**Monograph notebook:** `gmsh_testing_ext/notebooks/appF_training_diagnostics.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §24
**Prereqs:** activated `~/cluster/spark-venv`; ch04 fixture; NB 02 (Primitives)

**What you'll do.** Run the 9-check `GMSDiagnostics` health report on the bank model-risk store and read off whether it passes the geometric properties required for the harness's primitives to behave correctly: rotor orthogonality, Stiefel projection integrity, ENM SHA-256 integrity, embedding norm constraint, primitive sanity checks. This is the post-training gate every store must clear before it is used as ground truth in a release-gate evaluation. On the shipped ch04 fixture you'll see 6/9 pass and 3 fail — the diagnostic report's job is to surface that signal honestly.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from _shared import load_ch04_store, seed_all
seed_all()
store = load_ch04_store()
print(f"store: {store.stats()}")\
"""))

cells.append(md("""\
## 2. Concept

A GMS store is more than its triples — it is a geometric structure whose primitives depend on subtle invariants:

- **Cayley rotors** for each relation must be orthogonal (`R^T R = I`); otherwise `score_triple` distorts.
- **Stiefel projections** for entity embeddings must satisfy `P^T P = I`; otherwise `tension_energy` is biased.
- **Sphere embedding** must keep entities on the unit hypersphere; otherwise geodesic distance is meaningless.
- **ENM rows** must each verify their SHA-256 integrity hash; otherwise exact-numeric lookup is silently lying.

`GMSDiagnostics` runs 9 such checks and emits a `DiagnosticReport`. The release pipeline gates on **all 9 passing** before a store is allowed to back a release gate, because a release-gate decision only means what the geometry says it means.\
"""))

cells.append(md("""\
## 3. Run the 9-Check Report\
"""))

cells.append(code("""\
import time
import torch
from gms.diagnostics import GMSDiagnostics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diag = GMSDiagnostics(
    model=store.model,
    adapter=store.adapter,
    enm=store.enm,
    doc_graph=store.doc_graph,
    device=device,
    sample_size=200,
)

t0 = time.time()
report = diag.run_all()
elapsed = time.time() - t0

n_pass = sum(1 for c in report.checks if c.passed)
print(f"  ran 12 checks in {elapsed:.2f}s")
print(f"  passed: {n_pass}/{len(report.checks)}")\
"""))

cells.append(md("""\
## 4. Per-Check Breakdown\
"""))

cells.append(code("""\
import pandas as pd
rows = []
for c in report.checks:
    rows.append({
        "check": c.name,
        "passed": c.passed,
        "score": c.score,
        "detail": c.detail[:80],
    })
df = pd.DataFrame(rows)
df\
"""))

cells.append(md("""\
## 5. Inspect a Single Check in Detail

Each `CheckResult` carries an `evidence` dict with the raw measurements that drove the verdict. For triage, the evidence tells you exactly which entity or relation tripped the check.\
"""))

cells.append(code("""\
# Pick the most informative check to dig into — the geodesic score check
target = next((c for c in report.checks if "score_triple" in c.name.lower()), report.checks[0])
print(f"Check: {target.name}")
print(f"  passed  : {target.passed}")
print(f"  score   : {target.score:.4f}")
print(f"  detail  : {target.detail}")
print(f"  evidence:")
for k, v in (target.evidence or {}).items():
    if isinstance(v, list):
        print(f"    {k}: [{len(v)} items] {str(v[:3])[:120]}...")
    elif isinstance(v, dict):
        print(f"    {k}: {{{len(v)} keys}}")
    else:
        print(f"    {k}: {str(v)[:120]}")\
"""))

cells.append(md("""\
## 6. Visualization — Health Bar\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#3aa56b" if c.passed else "#c14a4a" for c in report.checks]
labels = [c.name.replace("_", " ") for c in report.checks]
scores = [c.score for c in report.checks]
y = list(range(len(report.checks)))
ax.barh(y, scores, color=colors)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("check score")
ax.set_title(f"GMSDiagnostics — bank model-risk store\\n"
             f"{sum(c.passed for c in report.checks)}/{len(report.checks)} passing")
ax.set_xlim(0, max(max(scores) * 1.1, 1.0))
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 7. Take-Aways

1. **12 checks, all must pass.** Any failed check invalidates downstream release-gate decisions because the geometry no longer means what the primitives assume. There is no failure budget on diagnostic checks.
2. **Diagnostics are corpus-specific.** Run them again after any retraining, after any merge of new triples, after any calibration change. Sample size 200 is a sensible default; bump higher for stores with > 5k triples.
3. **`evidence` is the triage signal.** A failed orthogonality check returns the worst-violating rotor and the magnitude of the violation. A failed ENM integrity check returns the bad keys. Don't ship until both the `passed` boolean and the `evidence` payload look right.
4. **No LLM, no API budget.** This entire check runs locally in seconds — make it a precondition for the bench run, not a downstream report.
5. **The release pipeline gates on this.** A bank-deployment release pipeline should call `GMSDiagnostics.run_all()` before any `bench.run()` and abort the campaign if `passed < 12`.\
"""))

cells.append(md("""\
## 8. Next Steps

- **NB 04** — `04_ingestion.ipynb` — what the training pipeline does that these diagnostics check (Ch04)
- **NB 19** — `19_release_gates.ipynb` — pair the diagnostic gate with the release gate at campaign start (Ch19)
- **NB 20** — `20_calibration.ipynb` — the calibration step that should follow these diagnostics (App G)

For theory:
- `gmsh_testing_ext/appendix_f.tex` — derivation of the orthogonality invariants and why violations break specific primitives
- `gmsh_testing_ext/notebooks/appF_training_diagnostics.ipynb` — textbook lab walkthrough with sensitivity-to-sample-size analysis\
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
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  execution wall time: {time.time() - t0:.1f}s")

    print(f"Writing executed notebook back to {NB_PATH}...")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
