# SPDX-License-Identifier: Apache-2.0
"""Build NB 16 — Hallucination Detection (Ch16)."""
from __future__ import annotations
import time
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "16_hallucination.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 16: Hallucination Detection via Geodesic Distance

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch16.tex` — Hallucination Detection via Geodesic Distance
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch16_hallucination.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §19
**Prereqs:** NB 02 (Primitives), NB 09 (Verifier Routing), NB 20 (Calibration)

**What you'll do.** Build a `HallucinationOracle` over the bank model-risk store and test it on three handcrafted claims: a true triple `(MDL-001, has_test_auc, 0.8412)`, a single-edit hallucination `(MDL-001, has_test_auc, 0.9999)`, and a wrong-model hallucination `(MDL-007, has_test_auc, 0.8412)`. Read off the geodesic confidence for each, see how the calibrated τ_geo threshold separates them, and watch the `HALLUCINATION_DETECTED` failure code attach to the synthetic hallucinations.\
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
from knowlytix.harness.testing import HallucinationOracle
seed_all()
store = load_ch04_store()
print(f"store: {store.stats()['entities']} entities, {store.stats()['triples']} triples")\
"""))

cells.append(md("""\
## 2. Concept

`HallucinationOracle.assess_claim(head, relation, tail)` returns a Chapter-9 `SubVerdict`. Internally:

1. Call `store.score_triple(head, relation, tail)` to get the geodesic distance on the cap manifold.
2. Compare against `tau_geodesic` (passed at construction or loaded from `store/calibration.json`).
3. If `geodesic < tau_geo`, the triple is plausible → `passed=True`; otherwise `HALLUCINATION_DETECTED`.

The trained calibration artifact provides `tau_geo` per corpus. Without it, the oracle falls back to a conservative default (the chapter explains the choice). On the bank ch04 corpus, NB 20 produced `τ_geodesic = 0.876`.\
"""))

cells.append(md("""\
## 3. Train Calibration and Build the Oracle\
"""))

cells.append(code("""\
import contextlib, io
buf = io.StringIO()
# We need the GMSJudge from a benchmark to run calibrate(), which produces τ_geo.
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig
config = DOEHarnessConfig(
    markdown_path="gmsh_testing_ext/lab_stores/ch04_ground_truth/documents/combined.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen", n_runs=1, cap_enabled=True,
    enable_typed_verdicts=False, enable_tracing=False,
)
bench = DOEGMSBenchmark(config)
with contextlib.redirect_stdout(buf):
    bench.ingest()
    bench.judge.calibrate(seed=42)
tau_geo = bench.judge._thresholds.get("geodesic")
print(f"calibrated tau_geo: {tau_geo:.4f}")
oracle = HallucinationOracle(store=store, tau_geodesic=tau_geo)
print(f"oracle built with tau_geodesic={oracle.tau_geodesic:.4f}")\
"""))

cells.append(md("""\
## 4. Three Handcrafted Claims\
"""))

cells.append(code("""\
claims = [
    ("mdl-001", "has_test_auc", "0.8412",
     "TRUE — MDL-001 test AUC matches ground truth"),
    ("mdl-001", "has_test_auc", "0.9999",
     "HALLUCINATION — wrong AUC value for the right model"),
    ("mdl-007", "has_test_auc", "0.8412",
     "HALLUCINATION — wrong model attributed with MDL-001's AUC"),
]

print(f"{'head':10s}  {'relation':18s}  {'tail':12s}  {'verdict':9s}  {'geo':>7s}  note")
print("-" * 110)
for h, r, t, note in claims:
    sv = oracle.assess_claim(h, r, t)
    geo = sv.raw_score if sv.raw_score is not None else float("nan")
    verdict = "PASS" if sv.passed else "FAIL"
    print(f"{h:10s}  {r:18s}  {t:12s}  {verdict:9s}  {geo:>7.4f}  {note}")\
"""))

cells.append(md("""\
## 5. Visualization — Geodesic Distribution vs τ_geo\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

# Sample geodesic scores for a sweep of true triples vs random corruptions
true_scores = []
for h, r, t in list(store.doc_graph.triples)[:200]:
    s = store.score_triple(h, r, t)
    if s is not None:
        true_scores.append(s)

import random
rng = random.Random(42)
entities = list(store.adapter.entity_to_idx.keys())
fake_scores = []
for h, r, _ in list(store.doc_graph.triples)[:200]:
    fake_t = rng.choice(entities)
    s = store.score_triple(h, r, fake_t)
    if s is not None:
        fake_scores.append(s)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(true_scores, bins=30, alpha=0.6, color="#3aa56b", label=f"true triples (n={len(true_scores)})")
ax.hist(fake_scores, bins=30, alpha=0.6, color="#c14a4a", label=f"hallucinated triples (n={len(fake_scores)})")
ax.axvline(tau_geo, color="black", linestyle="--", linewidth=2, label=f"τ_geo = {tau_geo:.3f}")
ax.set_xlabel("geodesic distance")
ax.set_ylabel("count")
ax.set_title("Geodesic-distance distribution: true vs hallucinated triples (bank corpus)")
ax.legend()
plt.tight_layout()
plt.show()
print(f"true triples below τ_geo:        {sum(s < tau_geo for s in true_scores):>3d} / {len(true_scores)}")
print(f"hallucinated triples below τ_geo: {sum(s < tau_geo for s in fake_scores):>3d} / {len(fake_scores)}")\
"""))

cells.append(md("""\
## 6. Take-Aways

1. **τ_geo is a learned cut.** Don't hand-tune it; let `bench.judge.calibrate()` derive it from the corpus's positive/negative pairs.
2. **`HALLUCINATION_DETECTED` is `CRITICAL` at action_taking.** NB 22 spells out the tier escalation; a single hallucinated claim under an autonomous credit decision blocks the gate.
3. **The histogram is the calibration sanity check.** True and hallucinated distributions should separate cleanly around τ_geo. Heavy overlap means the embedding hasn't learned enough discrimination — go retrain.
4. **The oracle works on triples, not free text.** Pair with `ClaimCompiler` (NB 08) to extract triples from an LLM answer first; the oracle then judges each one individually.
5. **`evaluate(scenario, trace)` is the release-gate hook.** When pairing with `TieredReleaseGate` (NB 19), the oracle returns an `OracleVerdict` with the failure list pre-populated — exactly the shape the audit reporter consumes.\
"""))

cells.append(md("""\
## 7. Next Steps

- **NB 17** — `17_multi_turn.ipynb` — apply hallucination detection across a multi-turn dialog (Ch17)
- **NB 22** — `22_failure_codes.ipynb` — `HALLUCINATION_DETECTED` in the failure-code taxonomy (App C)

For theory: `gmsh_testing_ext/chapters/ch16.tex`.\
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
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
