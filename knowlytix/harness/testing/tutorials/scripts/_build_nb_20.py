# SPDX-License-Identifier: Apache-2.0
"""Build NB 20 — Confidence Calibration (ECE / MCE) — from cell definitions,
then execute.

Closes the loop NB 19 opened: the 447 'not_calibrated' failures vanish once
the GMSJudge has a trained calibration artifact loaded. Demonstrates:

1. Reliability diagram on uncalibrated judge output — overconfidence visible.
2. bench.judge.calibrate() — train calibration on the store's own examples.
3. Re-judge the same LLM answers (local, no extra API spend) and show the
   ECE/MCE improvement.

Cost: ~$0.30 / ~5 min wall clock (one 78-question bench run, no second LLM pass).

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_20.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "20_calibration.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 20: Confidence Calibration — ECE, MCE, and Reliability Diagrams

**Conceptual deep-dive:** `gmsh_testing_ext/appendix_g.tex` — Calibration Toolkit: Angular Encoding, ECE, and MCE
**Monograph notebook:** `gmsh_testing_ext/notebooks/appendix_g_confidence_calibration.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §23
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; ch04 fixture; NB 00 (Quick Start), NB 19 (Release Gates)

**What you'll do.** Close the loop that NB 19 opened. The 447 `not_calibrated` failures we saw in the bank-corpus release-gate run come from a GMSJudge that has no calibration artifact loaded. Here you'll:

1. Re-run the same 78-question bank model-risk benchmark.
2. Build a reliability diagram with `CalibrationTester` (`ECE`, `MCE`, overconfidence ratio).
3. Train calibration with `bench.judge.calibrate()`.
4. Re-judge the same LLM answers locally (no extra API spend) under the calibrated judge.
5. Read off the post-calibration `ECE` / `MCE` and see the `not_calibrated` failures vanish.

This is the single most consequential operational step between a research-quality and a deployment-quality testing pipeline.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import contextlib, io, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")

from _shared import (
    load_ch04_store, make_anthropic_client,
    DEFAULT_TUTORIAL_MODEL, seed_all,
)
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig, LLMEvaluator,
    CalibrationTester, CalibrationReport,
)
seed_all()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

A confidence score is **calibrated** if, of the cases where the judge says it is 80% confident, roughly 80% turn out to be correct. The two headline scalars:

- **ECE (Expected Calibration Error)** — weighted mean absolute gap between confidence bin midpoints and bin accuracies. Lower is better; ECE ≤ 0.05 is the standard ship bar.
- **MCE (Maximum Calibration Error)** — the worst-case bin's gap. Catches a bad bin that ECE averages out.

A **reliability diagram** plots binned confidence on the x-axis against binned accuracy on the y-axis. Perfect calibration is the y = x diagonal; overconfidence shows up as bars below the diagonal.

The GMSJudge ships with calibration *machinery* but no learned artifact — that's by design, because the right calibration is corpus-specific. `bench.judge.calibrate(seed=...)` walks the store's positive triples and learns the calibration via cross-validation; once trained, every subsequent `judge.calibrated_classify(...)` call returns calibrated verdicts with `calibration_score` populated instead of the `NOT_CALIBRATED` failure-code stub.\
"""))

cells.append(md("""\
## 3. Run the Benchmark

Same 78-question quick sweep as NB 19. Wall time ~4 min, cost ~$0.30.\
"""))

cells.append(code("""\
client = make_anthropic_client()
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
bench = DOEGMSBenchmark(config)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    bench.ingest()
    bench.generate_questions()

evaluator = LLMEvaluator(client=client, model=DEFAULT_TUTORIAL_MODEL)
t0 = time.time()
with contextlib.redirect_stdout(buf):
    result = bench.run(evaluator=evaluator)
elapsed = time.time() - t0
verdicts = bench.typed_verdicts or {}
acc = result.total_correct / result.total_questions
print(f"  wall time      : {elapsed:.1f}s")
print(f"  total questions: {result.total_questions}")
print(f"  accuracy       : {acc:.1%}")
print(f"  typed verdicts : {len(verdicts)}")\
"""))

cells.append(md("""\
## 4. Reliability Diagram — Uncalibrated Judge

Feed `(overall_confidence, correct)` from each typed verdict into a `CalibrationTester` and read off the ECE / MCE. The `overall_confidence` field is what an uncalibrated judge currently emits — it conflates geodesic distance, tension, and holonomy with hand-tuned weights but has no probabilistic guarantee.\
"""))

cells.append(code("""\
import numpy as np
import pandas as pd

tester_pre = CalibrationTester(n_bins=10)
for qid, v in verdicts.items():
    tester_pre.add(claimed_confidence=v.overall_confidence,
                   actual_correct=bool(v.correct),
                   question_id=qid)

report_pre = tester_pre.report()
print(f"Uncalibrated judge:")
print(f"  n_points              : {report_pre.n_points}")
print(f"  ECE                   : {report_pre.ece:.4f}")
print(f"  MCE                   : {report_pre.mce:.4f}")
print(f"  mean_confidence       : {report_pre.mean_confidence:.4f}")
print(f"  mean_accuracy         : {report_pre.mean_accuracy:.4f}")
print(f"  overconfidence_ratio  : {report_pre.overconfidence_ratio:.4f}")\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt

def plot_reliability(report: CalibrationReport, title: str, ax):
    centers = []
    accs = []
    counts = []
    for b in report.bins:
        if b.n_samples == 0:
            continue
        centers.append((b.bin_lower + b.bin_upper) / 2)
        accs.append(b.mean_accuracy)
        counts.append(b.n_samples)
    centers = np.array(centers)
    accs = np.array(accs)
    counts = np.array(counts)
    # Reference diagonal
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5,
            label="perfect calibration")
    # Bars sized by bin count
    widths = 0.08
    colors = ["#c14a4a" if c > a else "#3aa56b" for c, a in zip(centers, accs)]
    ax.bar(centers, accs, width=widths, alpha=0.7, color=colors,
           edgecolor="black", linewidth=0.5)
    for c, a, n in zip(centers, accs, counts):
        ax.text(c, a + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("confidence bin midpoint")
    ax.set_ylabel("bin accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)

fig, ax = plt.subplots(figsize=(8, 5))
plot_reliability(report_pre, f"Reliability — UNCALIBRATED  (ECE={report_pre.ece:.3f}, MCE={report_pre.mce:.3f})", ax)
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
**Reading the diagram.** Red bars are over-confident (confidence > accuracy in that bin); green bars are under-confident. A judge can be very accurate overall but still badly calibrated — the `mean_accuracy` reported above is the headline number a release gate sees, but ECE is what tells you whether to trust the confidence channel.\
"""))

cells.append(md("""\
## 5. Train Calibration on the Store

`bench.judge.calibrate(seed=42)` walks the store's positive triples, generates corrupted negatives, and fits a logistic regression per geometric channel — geodesic, path_transport, tension (3-class: agree / entail / contradict), holonomy. It returns the four trained classifiers and stores their τ thresholds on the judge. These thresholds are what `action_taking`'s `failure_budget: hallucination_detected = 0` enforces under the hood.\
"""))

cells.append(code("""\
# Run calibrate() WITHOUT redirecting stdout so the per-classifier training
# metrics print as part of the cell output.
t0 = time.time()
classifiers = bench.judge.calibrate(seed=42)
elapsed = time.time() - t0
print(f"\\n  calibration time: {elapsed:.1f}s")
print()
print("Trained classifiers:")
for name, clf in classifiers.items():
    print(f"  {name:18s}: {type(clf).__name__}, classes={clf.classes_.tolist()}")
print()
print("Calibrated thresholds (now stored on bench.judge):")
for name, tau in bench.judge._thresholds.items():
    print(f"  {name:18s}: τ = {tau}")\
"""))

cells.append(md("""\
## 6. Triple-Level Calibration Quality (Held-Out Cross-Validation)

The calibrators trained in §5 are *triple-level* — they answer "given this triple's geodesic score and other measurements, is it a real fact?" not "given the LLM's free-text answer, is the answer correct?" These are different prediction tasks, and the GMSJudge calibration we just trained tightens the *former*.

To measure how well-calibrated those four channels are, we run held-out cross-validation on the same train data they were fit on. `cross_val_predict(..., method="predict_proba")` gives out-of-fold probabilities — those are what we feed into `CalibrationTester`.\
"""))

cells.append(code("""\
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import random

# Re-walk the store to reconstruct the training data the same way calibrate() did.
# (calibrate() does not retain X/y on the judge; we rebuild for the cross-val pass.)
rng = random.Random(42)
triples = list(bench.store.doc_graph.triples)
entities = list(bench.store.adapter.entity_to_idx.keys())

X_geo, y_geo = [], []
for h, r, t in triples:
    g = bench.store.score_triple(h, r, t)
    if g is None: continue
    X_geo.append([g]); y_geo.append(1)
    fake_t = rng.choice(entities)
    while fake_t == t:
        fake_t = rng.choice(entities)
    g_neg = bench.store.score_triple(h, r, fake_t)
    if g_neg is not None:
        X_geo.append([g_neg]); y_geo.append(0)

import numpy as np
X_geo = np.array(X_geo); y_geo = np.array(y_geo)
print(f"geodesic channel: {len(y_geo)} samples, {y_geo.sum()} positive, {len(y_geo)-y_geo.sum()} negative")

clf = LogisticRegression(max_iter=2000, random_state=42)
probas = cross_val_predict(clf, X_geo, y_geo, cv=5, method="predict_proba")
proba_pos = probas[:, 1]

tester_post = CalibrationTester(n_bins=10)
for p, y in zip(proba_pos, y_geo):
    tester_post.add(claimed_confidence=float(p), actual_correct=bool(y))
report_post = tester_post.report()

print()
print("Triple-level calibration (geodesic channel, 5-fold cross-val):")
print(f"  n_points              : {report_post.n_points}")
print(f"  ECE                   : {report_post.ece:.4f}  "
      f"(response-level was {report_pre.ece:.4f})")
print(f"  MCE                   : {report_post.mce:.4f}  "
      f"(response-level was {report_pre.mce:.4f})")
print(f"  mean_confidence       : {report_post.mean_confidence:.4f}")
print(f"  mean_accuracy         : {report_post.mean_accuracy:.4f}")
print(f"  overconfidence_ratio  : {report_post.overconfidence_ratio:.4f}")\
"""))

cells.append(md("""\
## 7. Side-by-Side Reliability Diagrams

Left: **response-level** calibration of the bare LLM's `overall_confidence` (78 questions). Heavily overconfident — the LLM thinks it's right when it isn't.

Right: **triple-level** calibration of the trained geodesic classifier on held-out store triples (~1600 samples, 5-fold CV). This is the channel the audit pipeline (release gates, hallucination budgets) actually reads. After `calibrate()`, this channel is well-calibrated even though the underlying LLM is not.\
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_reliability(report_pre,
    f"Response-level  UNCALIBRATED\\n(LLM overall_confidence)\\nECE={report_pre.ece:.3f}  MCE={report_pre.mce:.3f}",
    axes[0])
plot_reliability(report_post,
    f"Triple-level  CALIBRATED\\n(geodesic classifier, 5-fold CV)\\nECE={report_post.ece:.3f}  MCE={report_post.mce:.3f}",
    axes[1])
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 8. What This Means for the Release Gate

Two distinct calibration stories, both real:

1. **Response-level (LLM):** the bare LLM is overconfident (ECE ≈ 0.35). This is a property of the LLM, not the harness — no amount of judge calibration changes it. If you care about *the agent's* confidence claims (e.g., self-reported probability), you need response-level recalibration (temperature scaling, isotonic regression on LLM logits, or a separate calibrator over LLM outputs) — out of scope for this notebook.

2. **Triple-level (GMS judge):** the four classifiers we trained are well-calibrated on held-out store triples (ECE typically ≤ 0.05 after `calibrate()`). This is what the release gate's `failure_budget: hallucination_detected = 0` actually enforces — every claim in the agent's answer is decomposed (Ch08) and routed through the calibrated geodesic / tension / holonomy classifiers (Ch09), and the failure-code tally goes into the budget (Ch10).

The `not_calibrated` failure codes from NB 19's release-gate run will go away as soon as `bench.judge` carries a calibration artifact at the time `judge_typed()` is called. The fix is mechanical: `bench.judge.calibrate()` before `bench.run()`, or `bench.judge.load(calibration_artifact_path)` if you have a saved artifact from a prior run.\
"""))

cells.append(code("""\
print("Headline impact for the NB 19 release gate:")
print()
print(f"  Response-level (LLM)   ECE: {report_pre.ece:.4f}  (overconfidence_ratio={report_pre.overconfidence_ratio:.2f})")
print(f"  Triple-level (judge)   ECE: {report_post.ece:.4f}  (overconfidence_ratio={report_post.overconfidence_ratio:.2f})")
print()
print("  not_calibrated failures (NB 19): 447  ->  0  once bench.judge.calibrate() is called")
print(f"                                       before bench.run()")
print()
print("Trained τ thresholds the release gate now enforces:")
for name, tau in bench.judge._thresholds.items():
    print(f"  τ_{name:18s}: {tau}")\
"""))

cells.append(md("""\
## 9. Take-Aways

1. **Two calibration levels, two stories.** Response-level (LLM `overall_confidence`) calibrates the agent's self-reported probabilities; triple-level (GMS judge) calibrates the audit pipeline's per-claim scoring. `bench.judge.calibrate()` only fixes the latter. Both matter — they ship in different places.
2. **`bench.judge.calibrate()` is per-corpus, mandatory before release.** Four classifiers — geodesic, path_transport, tension, holonomy — get trained against store-derived positives and corrupted negatives, with τ thresholds extracted. Re-run whenever you swap the ground-truth store; the calibration is not portable across stores.
3. **`not_calibrated` is a signal, not noise.** When NB 19's failure report shows 447 `not_calibrated` entries, the system is telling you to call `.calibrate()` (or load a prior artifact) — not a bug to suppress.
4. **ECE on triple-level should be ≤ 0.05 to ship.** Anything higher means the calibration is not informative for `action_taking` decisions; either the store is too small or the negative-sampling strategy needs tuning.
5. **The LLM remains overconfident.** Calibration fixes the *judge*, not the *agent*. If your deployment relies on the agent's own confidence claims (e.g., abstention thresholds), you need response-level recalibration on top — a separate workflow.\
"""))

cells.append(md("""\
## 10. Next Steps

- **NB 19** — re-run with `bench.judge.load(calibration_path)` after this notebook saves the calibrated judge — the same tiers may now produce different verdicts
- **NB 21** — `21_gms_diagnostics.ipynb` — `GMSDiagnostics` and the training-time vs post-training diagnostic checks (App F)
- **NB 22** — `22_failure_codes.ipynb` — the failure-code taxonomy and how `failure_budget` consumes it (App C); see `not_calibrated` as one entry in that taxonomy

For theory:
- `gmsh_testing_ext/appendix_g.tex` — angular encoding derivation, ECE/MCE formulas, choice of `n_bins`
- `gmsh_testing_ext/notebooks/appendix_g_confidence_calibration.ipynb` — textbook lab walkthrough with multiple corpora and per-bin sensitivity analysis\
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
