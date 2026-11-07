# SPDX-License-Identifier: Apache-2.0
"""Build NB 07 — Statistical Analysis of Factor Effects (Ch07)."""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "07_statistical_analysis.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 07: Statistical Analysis of Factor Effects

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch07.tex` — Statistical Analysis of Factor Effects
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch07_statistical_analysis.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §10
**Prereqs:** NB 03 (DOE Design)

**What you'll do.** Take a synthetic but realistic 300-row result table from a Sobol+Refine sweep over 10 contextual factors on the bank model-risk corpus, run `DOEAnalyzer.run_logistic` + `apply_fdr_correction` + `run_joint_model`, and read off which factors significantly impact correctness, with Benjamini-Hochberg-controlled false-discovery rate. This is the analysis layer that turns a DOE run into ship/no-ship signal at scale.

Synthetic input is used (no LLM calls) so the notebook runs locally in seconds while still exercising the full analyzer pipeline.\
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
from _shared import seed_all
seed_all()
from graphdoe.analysis import DOEAnalyzer
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

Three analyses:

1. **Single-factor logistic** — fit `correct ~ factor_level` for each factor independently; report deviance and per-factor pseudo-R². Best for cheap screening when n_obs is modest.
2. **FDR correction** — with 10+ factors, the family-wise false-discovery rate is high. Benjamini-Hochberg adjusts p-values so the expected proportion of false discoveries stays below α.
3. **Joint logistic** — fit all factors simultaneously; report pseudo-R², AIC, BIC, ROC AUC, Hosmer-Lemeshow goodness-of-fit p-value. Needs `n_obs >> n_params` (rule of thumb: 10× the parameter count).

For a paper-style 880-run design with 20 factors, the joint model is well-determined and produces the ship/no-ship verdict; for a 60-run smoke screen, the single-factor + FDR pass is the right tool.\
"""))

cells.append(md("""\
## 3. Build a Synthetic 300-Row Result Table

The synthetic generator: `correct` depends strongly on `question_difficulty`, mildly on `role` and `clarity`, and not at all on the other 7 factors. We will see whether the analyzer recovers this structure.\
"""))

cells.append(code("""\
import numpy as np
import pandas as pd
rng = np.random.default_rng(seed=42)

N = 300
FACTORS = {
    "clarity":       ["clear", "ambiguous"],
    "noise":         ["none", "high"],
    "role":          ["analyst", "auditor", "executive"],
    "answer_format": ["short", "long"],
    "reasoning_cue": ["yes", "no"],
    "context_scope": ["narrow", "wide"],
    "entity_aliasing":["full", "abbreviated"],
    "instruction_conflict": ["yes", "no"],
    "complexity":    ["simple", "compound"],
    "style":         ["formal", "casual"],
    "question_difficulty": ["easy", "medium", "hard"],
}

rows = []
for i in range(N):
    factors = {f: rng.choice(levels) for f, levels in FACTORS.items()}
    # Synthetic correctness model
    base = {"easy": 0.92, "medium": 0.65, "hard": 0.32}[factors["question_difficulty"]]
    if factors["role"] == "executive": base -= 0.10
    if factors["clarity"] == "ambiguous": base -= 0.05
    correct = rng.random() < max(0.05, min(0.95, base))
    rows.append({**factors, "correct": int(correct)})

df = pd.DataFrame(rows)
print(f"shape: {df.shape}")
print(f"mean correctness: {df['correct'].mean():.1%}")
print(df.head(4).to_string(index=False))\
"""))

cells.append(md("""\
## 4. Single-Factor Logistic + FDR\
"""))

cells.append(code("""\
ana = DOEAnalyzer.from_dataframe(df, factors=list(FACTORS.keys()))
deviance = ana.run_logistic(metric="correct")
deviance_adj = ana.apply_fdr_correction(deviance, alpha=0.05, method="fdr_bh")

print("Single-factor logistic with Benjamini-Hochberg FDR (sorted by p_value_adj):")
print()
cols = ["factor", "deviance", "p_value", "p_value_adj", "significant_adj", "pseudo_r2"]
print(deviance_adj.sort_values("p_value_adj")[cols].to_string(index=False))\
"""))

cells.append(md("""\
The expected pattern: `question_difficulty` significant after FDR (true effect, large size); `role` and `clarity` borderline significant depending on sample size; the other 7 factors not significant. The single-factor analysis isolates each factor without controlling for the others.\
"""))

cells.append(md("""\
## 5. Joint Logistic Model

The joint model fits all factors simultaneously. It controls for confounders the single-factor pass misses, but requires `n_obs >> n_params`. With 11 factors and ~22 parameters (after one-hot encoding), 300 obs gives ~14× — well-determined.\
"""))

cells.append(code("""\
try:
    joint = ana.run_joint_model(metric="correct")
    print("Joint logistic model:")
    for k, v in joint.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.4f}")
        else:
            print(f"  {k:18s}: {v}")
except Exception as e:
    print(f"Joint model failed: {type(e).__name__}: {e}")\
"""))

cells.append(md("""\
## 6. Visualization — Adjusted p-Values\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

sub = deviance_adj.sort_values("p_value_adj")
sig = sub["significant_adj"].fillna(False).astype(bool).values
colors = ["#c14a4a" if s else "#999999" for s in sig]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(sub["factor"], -np.log10(sub["p_value_adj"].fillna(1.0).clip(lower=1e-300)), color=colors)
ax.axvline(-np.log10(0.05), color="black", linestyle="--", label="α=0.05")
ax.set_xlabel("-log10(p_value_adj)")
ax.set_title("Single-factor effects with Benjamini-Hochberg FDR correction\\n"
             "(red = significant after correction)")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 7. Take-Aways

1. **Single-factor for screening, joint for shipping.** Single-factor analysis is cheap and unbiased per factor; joint analysis controls for confounders and is what a release report cites.
2. **FDR correction is mandatory at 10+ factors.** Without it, the family-wise false-positive rate exceeds α; with it, ≤ α of "significant" findings are spurious.
3. **`pseudo_r2` tells you which factor *matters most*.** p-values say "is there an effect"; pseudo-R² says "how much variance does it explain." Both are needed to write a useful report.
4. **Joint model needs `n_obs ≥ 10 × n_params`.** Under-resourced joint fits get `inf` pseudo-R² or singular-matrix errors. Scale n_runs or drop low-information factors.
5. **The harness's `DOEAnalyzer` is the entry point.** `.run_logistic`, `.apply_fdr_correction`, `.run_joint_model` cover the full screen-to-ship analysis flow. NB 03 §8 walks the real-data version on a Sobol+Refine bank corpus run.\
"""))

cells.append(md("""\
## 8. Next Steps

- **NB 03** — `03_doe_design.ipynb` — real Sobol+Refine sweep on the bank corpus (Ch03)
- **NB 19** — `19_release_gates.ipynb` — how the joint-model verdict feeds into the release report (Ch19)

For theory: `gmsh_testing_ext/chapters/ch07.tex`, `graphdoe/papers/paper_jasa.tex`.\
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
