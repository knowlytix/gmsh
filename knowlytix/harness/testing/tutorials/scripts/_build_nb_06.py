# SPDX-License-Identifier: Apache-2.0
"""Build NB 06 — The Geometric Judge (Ch06)."""
from __future__ import annotations
import time
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "06_geometric_judge.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 06: The Geometric Judge — Per-Category Scoring

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch06.tex` — The Geometric Judge: Per-Category Scoring
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch06_geometric_judge.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §9
**Prereqs:** NB 02 (Primitives), NB 09 (Verifier Routing), NB 20 (Calibration)

**What you'll do.** Take five bank model-risk question-answer pairs (mix of correct, near-miss, and outright wrong answers), run them through `GMSJudge.judge()`, and inspect the per-component scores: `geodesic_confidence`, `tension_score`, `holonomy_score`, `phase_score`, `overall_confidence`. The judge fans the answer into per-category geometric measurements; the `overall_confidence` is the harness's headline number for "how plausible is this answer geometrically."\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys, os, contextlib, io
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")
from _shared import seed_all
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig
seed_all()

config = DOEHarnessConfig(
    markdown_path="gmsh_testing_ext/lab_stores/ch04_ground_truth/documents/combined.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen", n_runs=1, cap_enabled=True,
    enable_typed_verdicts=True, enable_tracing=False,
)
bench = DOEGMSBenchmark(config)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    bench.ingest()
    bench.judge.calibrate(seed=42)
judge = bench.judge
print(f"judge ready, calibrated thresholds: {list(judge._thresholds.keys())}")\
"""))

cells.append(md("""\
## 2. Concept

`GMSJudge.judge(llm_answer, ground_truth, exact_correct, exact_detail)` returns a `GeometricVerdict` with five score channels:

| Channel | Reads | Measures |
|---|---|---|
| `geodesic_confidence` | extracted triples | Per-claim plausibility on the cap manifold |
| `tension_score` | claim pairs | Cross-claim disagreement (post-hoc contradiction) |
| `holonomy_score` | multi-hop paths | Process consistency (did the reasoning compose correctly?) |
| `phase_score` | phase encoders | Whether numeric values land in the trained range |
| `overall_confidence` | weighted blend | Headline number combining the above |

These are post-extraction measurements over the LLM answer's *claims*, not over its raw prose. The judge first compiles claims (Ch08, NB 08), then scores each, then aggregates. After calibration (NB 20), each channel produces a probability; before calibration, channels report raw geometric measurements without a probabilistic interpretation.\
"""))

cells.append(md("""\
## 3. Five Bank Model-Risk Answer Variants\
"""))

cells.append(code("""\
samples = [
    {"q": "What is MDL-001's Test AUC?", "gt": "0.8412",
     "llm": "MDL-001 has a Test AUC of 0.8412. ANSWER: 0.8412",
     "label": "CORRECT — exact match"},
    {"q": "What is MDL-001's Test AUC?", "gt": "0.8412",
     "llm": "MDL-001 reports a Test AUC of approximately 0.84. ANSWER: 0.84",
     "label": "NEAR-MISS — rounded answer"},
    {"q": "What is MDL-001's Test AUC?", "gt": "0.8412",
     "llm": "MDL-001 has a Test AUC of 0.9234. ANSWER: 0.9234",
     "label": "WRONG — another model's AUC"},
    {"q": "What is MDL-003's status?", "gt": "Degraded",
     "llm": "MDL-003 is currently in a Degraded state. ANSWER: Degraded",
     "label": "CORRECT — categorical"},
    {"q": "What is MDL-003's status?", "gt": "Degraded",
     "llm": "MDL-003 is Stable. ANSWER: Stable",
     "label": "WRONG — categorical opposite"},
]
for s in samples:
    print(f"  [{s['label']:30s}]  Q: {s['q'][:50]:50s}  A: {s['llm'][-40:]!r}")\
"""))

cells.append(md("""\
## 4. Run Each Through GMSJudge\
"""))

cells.append(code("""\
from knowlytix.benchmark.scorers import score_answer, PARSERS
import pandas as pd

rows = []
for s in samples:
    parser = PARSERS["str"]
    parsed = parser(s["llm"])
    score = score_answer(parsed, s["gt"])
    v = judge.judge(
        llm_answer=s["llm"], ground_truth=s["gt"],
        exact_correct=score.correct, exact_detail=score.detail,
    )
    rows.append({
        "label": s["label"][:24],
        "exact": v.correct,
        "geodesic": round(v.geodesic_confidence, 3),
        "tension": round(v.tension_score, 3),
        "holonomy": round(v.holonomy_score, 3),
        "phase": round(v.phase_score, 3),
        "overall": round(v.overall_confidence, 3),
    })
df = pd.DataFrame(rows)
df\
"""))

cells.append(md("""\
## 5. Visualization — Per-Channel Profile per Answer\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

channels = ["geodesic", "tension", "holonomy", "phase", "overall"]
labels = [r["label"] for r in rows]
data = np.array([[r[ch] for ch in channels] for r in rows])

fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(labels))
w = 0.15
colors = ["#3aa56b", "#4a8cd1", "#c89438", "#a05a8a", "#000000"]
for i, ch in enumerate(channels):
    offset = (i - len(channels)/2) * w + w/2
    ax.bar(x + offset, data[:, i], w, label=ch, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("score (channel-dependent units)")
ax.set_title("GMSJudge per-channel scores across five bank-domain answer variants")
ax.legend(loc="upper right", ncol=5)
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 6. Take-Aways

1. **`overall_confidence` is a hand-tuned blend.** Pre-calibration it is *not* a probability. Use it as a relative ranking only; after `calibrate()` (NB 20) the per-channel classifiers produce calibrated probabilities.
2. **Per-channel scores diagnose differently.** A low `geodesic_confidence` with high `holonomy_score` says "the agent's individual facts are wrong but its reasoning chain is internally consistent" — diagnostic for fabrication. The reverse says "the facts are right but the reasoning path is novel" — possible distillation or recall failure.
3. **`exact` is binary; geometric channels are graded.** When `exact=True` and `overall_confidence` is low, you have a paraphrased correct answer; when `exact=False` and `overall_confidence` is high, you have a near-miss the geometric judge accepts.
4. **The judge runs claim-by-claim under the hood.** It uses `ClaimCompiler` (NB 08) + `VerificationRouter` (NB 09); what you see is the aggregate. For the per-claim trail, use `judge_typed()` (NB 10) instead.
5. **Calibration is the bridge to release-gates.** Without `bench.judge.calibrate()`, all the channels remain uncalibrated and `failure_budget` checks against them are meaningless. NB 20 walks the calibration step.\
"""))

cells.append(md("""\
## 7. Next Steps

- **NB 09** — `09_verifier_routing.ipynb` — what the judge calls under the hood (Ch09)
- **NB 10** — `10_verdicts_severity.ipynb` — `judge_typed()` for per-claim audit trails (Ch10)
- **NB 20** — `20_calibration.ipynb` — turn the per-channel scores into calibrated probabilities (App G)

For theory: `gmsh_testing_ext/chapters/ch06.tex`.\
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
