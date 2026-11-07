# SPDX-License-Identifier: Apache-2.0
"""Build NB 10 — Verdicts, Failure Codes, and Severity — from cell definitions, then execute.

Pulls together NB 08 (ClaimCompiler) + NB 09 (VerificationRouter) into the
final audit-shape verdict: TypedVerdict. Demonstrates assembly, the
correct/n_passed/n_failed aggregation rule, FailureAttribution composition,
and SeverityClassifier applied at three risk tiers.

Cost: ~$0.05 / ~30 sec (5 small Sonnet calls).

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_10.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "10_verdicts_severity.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 10: Verdicts, Failure Codes, and Severity

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch10.tex` — Verdicts, Failures, and Severity Classification
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch10_verdicts_failures.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §13
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; ch04 fixture; NB 08 (Claims), NB 09 (Verifiers)

**What you'll do.** Close the typed-verdict pipeline. Watch `GMSJudge.judge_typed()` glue ClaimCompiler + VerificationRouter into a single `TypedVerdict` per question, see the aggregation rule that decides `TypedVerdict.correct`, inspect the `FailureAttribution` list that names which codes triggered, and apply `SeverityClassifier` at three risk tiers to see how the same TypedVerdict carries different severity profiles for advisory vs action_taking deployment.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")

from _shared import (
    load_ch04_store, make_anthropic_client,
    DEFAULT_TUTORIAL_MODEL, seed_all,
)
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig,
    SeverityClassifier, SeverityLevel, FailureCode,
)
seed_all()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept — `TypedVerdict` Assembly

`GMSJudge.judge_typed()` is the single call that produces a `TypedVerdict`:

```
LLM answer + ground truth
  -> ClaimCompiler.compile()       (NB 08)
       -> list[ClaimBase]
  -> VerificationRouter.route_all()  (NB 09)
       -> list[SubVerdict]
  -> aggregation rule
       -> TypedVerdict
```

The TypedVerdict carries:

| Field | Meaning |
|---|---|
| `correct: bool` | Aggregate pass/fail (defined below) |
| `n_claims, n_passed, n_failed` | Counts from the sub-verdicts |
| `sub_verdicts: list[SubVerdict]` | The audit trail — every per-claim decision is preserved |
| `failures: list[FailureAttribution]` | One entry per failing sub-verdict, carrying its `FailureCode` |
| `geodesic_confidence, tension_score, holonomy_score, overall_confidence` | Aggregate geometric measurements |

**The aggregation rule.** `TypedVerdict.correct = exact_correct` — the exact-match scorer drives correctness at the verdict level. Sub-verdict failures populate `n_failed` and the `failures` list but do **not** flip `correct` to False. Gating on sub-verdict failures happens one level up, at the **release gate** (NB 19), where `RiskTierProfile.failure_budget` translates failure-code counts into pass/fail per deployment tier. The verdict layer is *diagnostic*; the release-gate layer is *gating*. We verify this empirically in §5: you will see verdicts with `exact_correct=True` and `n_failed > 0` that still carry `correct=True`.\
"""))

cells.append(md("""\
## 3. Configure a Small Benchmark That Produces Typed Verdicts\
"""))

cells.append(code("""\
client = make_anthropic_client()
config = DOEHarnessConfig(
    markdown_path="gmsh_testing_ext/lab_stores/ch04_ground_truth/documents/combined.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen",
    n_runs=1,
    max_per_category=1,
    cap_enabled=True,
    enable_typed_verdicts=True,
    enable_tracing=False,
    target_type="llm",
    target_model=DEFAULT_TUTORIAL_MODEL,
)
import contextlib, io
buf = io.StringIO()
bench = DOEGMSBenchmark(config)
with contextlib.redirect_stdout(buf):
    bench.ingest()
    bench.generate_questions()

from knowlytix.harness.testing import LLMEvaluator
evaluator = LLMEvaluator(client=client, model=DEFAULT_TUTORIAL_MODEL)
t0 = time.time()
with contextlib.redirect_stdout(buf):
    result = bench.run(evaluator=evaluator)
elapsed = time.time() - t0
verdicts = bench.typed_verdicts or {}
print(f"  wall time      : {elapsed:.1f}s")
print(f"  total questions: {result.total_questions}")
print(f"  typed verdicts : {len(verdicts)}")
print(f"  accuracy       : {result.total_correct / result.total_questions:.1%}")\
"""))

cells.append(md("""\
## 4. Inspect a Single `TypedVerdict`\
"""))

cells.append(code("""\
sample_qid, sample = next(iter(verdicts.items()))
print(f"=== TypedVerdict for {sample_qid} ===")
print(f"  correct          : {sample.correct}")
print(f"  n_claims         : {sample.n_claims}")
print(f"  n_passed         : {sample.n_passed}")
print(f"  n_failed         : {sample.n_failed}")
print(f"  geodesic_conf    : {sample.geodesic_confidence:.3f}")
print(f"  tension_score    : {sample.tension_score:.3f}")
print(f"  holonomy_score   : {sample.holonomy_score:.3f}")
print(f"  overall_conf     : {sample.overall_confidence:.3f}")
print(f"  # sub_verdicts   : {len(sample.sub_verdicts)}")
print(f"  # failures       : {len(sample.failures)}")
print()
print("Sub-verdict trail (first 3):")
for sv in sample.sub_verdicts[:3]:
    status = "PASS" if sv.passed else "FAIL"
    print(f"  [{status}] {sv.verifier_name:18s} method={sv.gms_method_used:30s} score={sv.raw_score}")\
"""))

cells.append(md("""\
## 5. The Aggregation Rule on Real Data

For each verdict, check whether `correct == exact_correct` (regardless of `n_failed`). If a verdict's `correct` flips when sub-verdicts fail, the rule has been violated. The table below also lists `exact_correct AND (n_failed == 0)` as a column so you can see which verdicts have *all* sub-verdicts passing — a stronger property that the release-gate layer can choose to enforce.\
"""))

cells.append(code("""\
rows = []
for qid, v in verdicts.items():
    # Reconstruct exact_correct from result.questions (same qid)
    qr = next((q for q in result.questions if q.doe_qid == qid), None)
    exact_correct = bool(qr.correct) if qr else None
    all_sv_passed = (v.n_failed == 0)
    rows.append({
        "qid": qid[:30],
        "category": qr.category if qr else "",
        "exact_correct": exact_correct,
        "n_claims": v.n_claims,
        "n_failed": v.n_failed,
        "correct (TypedVerdict)": v.correct,
        "all_sub_pass": all_sv_passed,
        "matches_exact": v.correct == exact_correct,
    })
import pandas as pd
df_agg = pd.DataFrame(rows)
print(f"verdicts where TypedVerdict.correct == exact_correct: {df_agg['matches_exact'].sum()}/{len(df_agg)}")
print(f"verdicts where ALL sub-verdicts also passed (stricter):    {df_agg['all_sub_pass'].sum()}/{len(df_agg)}")
print()
print(df_agg.head(8).to_string(index=False))\
"""))

cells.append(md("""\
## 6. Failure-Code Census

Across all verdicts, group every `FailureAttribution.code` and count how many times each appeared. This is what `AuditReporter.generate_failure_report()` (NB 19 §8) does internally.\
"""))

cells.append(code("""\
from collections import Counter
code_counts = Counter()
for v in verdicts.values():
    for f in v.failures:
        code_counts[f.code.value] += 1

if code_counts:
    df_codes = pd.DataFrame(
        sorted(code_counts.items(), key=lambda x: -x[1]),
        columns=["failure_code", "count"],
    )
    print(df_codes.to_string(index=False))
else:
    print("(no failures recorded)")\
"""))

cells.append(md("""\
## 7. Apply `SeverityClassifier` at Three Risk Tiers

The same TypedVerdict's failure list classifies differently at advisory vs action_taking. NB 22 covers the full failure-code-to-severity map; this cell exercises the classifier on real verdict data.\
"""))

cells.append(code("""\
sc = SeverityClassifier()
all_failures = [f for v in verdicts.values() for f in v.failures]
print(f"Total failure attributions across all {len(verdicts)} verdicts: {len(all_failures)}")
print()
print(f"{'tier':16s}  " + "  ".join(f"{lvl.name:>8s}" for lvl in SeverityLevel))
print("-" * 70)
for tier in ["advisory", "recommendation", "action_taking"]:
    assessments = sc.classify_all(all_failures, risk_tier=tier)
    counts = sc.count_by_level(assessments)
    print(f"{tier:16s}  " +
          "  ".join(f"{counts.get(lvl, 0):>8d}" for lvl in SeverityLevel))\
"""))

cells.append(md("""\
## 8. Visualization — Severity Distribution by Tier\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

tiers_list = ["advisory", "recommendation", "action_taking"]
levels = list(SeverityLevel)
data = []
for tier in tiers_list:
    assessments = sc.classify_all(all_failures, risk_tier=tier)
    counts = sc.count_by_level(assessments)
    data.append([counts.get(lvl, 0) for lvl in levels])
data = np.array(data)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(tiers_list))
w = 0.2
colors = ["#7a1f1f", "#c14a4a", "#e89556", "#3aa56b"]
for i, lvl in enumerate(levels):
    ax.bar(x + i * w - 1.5*w, data[:, i], w, label=lvl.name, color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels(tiers_list)
ax.set_ylabel("# failures classified at this severity")
ax.set_title(f"{len(all_failures)} real failures from the bank-corpus bench run,\\n"
             f"classified at three risk tiers")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 9. Take-Aways

1. **`TypedVerdict.correct` is conservative.** A verdict is correct only if both the exact-match scorer AND every sub-verdict pass. This is by design — the release gate must err on the strict side.
2. **The audit trail is non-negotiable.** Every `TypedVerdict` carries the full `sub_verdicts` list and the `failures` list. Don't strip these — they are how a regulator (or release manager) reconstructs why a verdict failed two weeks after the run.
3. **`FailureAttribution` is the bridge to severity.** `SeverityClassifier.classify_all(failures, risk_tier=...)` is the canonical path; never hardcode severity in your reporting.
4. **The same verdict has three severities.** advisory / recommendation / action_taking each interpret the failure list differently. NB 22 details the escalation rules; this notebook just shows the end effect on real data.
5. **`overall_confidence` is post-aggregation.** It blends `geodesic_confidence`, `tension_score`, `holonomy_score` with hand-tuned weights. It is *not* calibrated — that's what NB 20's `bench.judge.calibrate()` adds on top.\
"""))

cells.append(md("""\
## 10. Next Steps

- **NB 19** — `19_release_gates.ipynb` — these typed verdicts feed straight into the release-gate audit (Ch19)
- **NB 20** — `20_calibration.ipynb` — calibrate the sub-verifiers so `geodesic_confidence` becomes a true probability (App G)
- **NB 22** — `22_failure_codes.ipynb` — full failure-code taxonomy and per-tier escalation (App C)

For theory:
- `gmsh_testing_ext/chapters/ch10.tex` — aggregation-rule soundness proof, severity-classification semantics
- `gmsh_testing_ext/notebooks/ch10_verdicts_failures.ipynb` — textbook lab walkthrough with edge-case verdict scenarios\
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
