# SPDX-License-Identifier: Apache-2.0
"""Build NB 09 — Routing Claims to Verifiers — from cell definitions, then execute.

Demonstrates VerificationRouter.route_all() on claims extracted in NB 08.
Each claim subclass routes to the right verifier (GraphVerifier,
NumericVerifier, ArtifactVerifier, LogicVerifier, PolicyVerifier) and emits
a SubVerdict with passed/confidence/failure attribution.

Cost: ~$0.05 / ~30 sec (re-runs the same 5 Sonnet calls as NB 08 for
self-containment, then routes locally).

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_09.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "09_verifier_routing.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 09: Routing Claims to Verifiers

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch09.tex` — Verifiers: Routing Claims to GMS Primitives
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch09_verifiers.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §12
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; ch04 fixture; NB 08 (Claim Decomposition)

**What you'll do.** Take claims extracted by `ClaimCompiler` and feed them to `VerificationRouter.route_all()`. Watch each claim subclass land at the right verifier (`GraphVerifier` for `FactClaim`, `NumericVerifier` for `ThresholdClaim`, `ArtifactVerifier` for `ArtifactClaim`, `LogicVerifier` for `RuleClaim`, `PolicyVerifier` for `ToolUseClaim`), see what `SubVerdict` each emits (`passed` / `confidence` / `gms_method_used` / `raw_score` / optional `FailureAttribution`), and read off the per-verifier-pass rate on real Sonnet 4.6 answers.\
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
from knowlytix.harness.testing import ClaimCompiler, VerificationRouter
seed_all()
store = load_ch04_store()
print(f"store: {store.stats()}")\
"""))

cells.append(md("""\
## 2. Concept — Five Verifier Types

The router inspects each claim's `claim_type` and dispatches to the corresponding verifier:

| Verifier | Handles | GMS primitive used |
|---|---|---|
| `GraphVerifier` | `FactClaim` (head, relation, tail) | `score_triple`, `query_triples` |
| `NumericVerifier` | `ThresholdClaim` (metric, op, value) | `lookup_enm` |
| `ArtifactVerifier` | `ArtifactClaim` (artifact_id, hash) | SHA-256 integrity check |
| `LogicVerifier` | `RuleClaim` (logical predicate over multiple claims) | `check_holonomy`, `tension_energy` |
| `PolicyVerifier` | `ToolUseClaim` (tool, args) | Policy graph traversal |

Each verifier returns a `SubVerdict` with:
- `passed: bool` — pass/fail verdict
- `confidence: float` — verifier-side confidence (0..1)
- `gms_method_used: str` — which primitive (or method) actually ran
- `raw_score: float` — the geometric measurement that drove the verdict
- `failure: FailureAttribution | None` — populated on failure with the failure code and rationale\
"""))

cells.append(md("""\
## 3. Run the Same 5 Bank-Domain Questions Through Sonnet 4.6

Re-run NB 08's 5 questions so this notebook is self-contained. Same questions, same model, same context.\
"""))

cells.append(code("""\
client = make_anthropic_client()
with open("knowlytix/harness/testing/tests/model_risk_assessment.md") as f:
    document_context = f.read()

QUESTIONS = [
    "What is the Test AUC of MDL-001?",
    "What is the current status of MDL-003?",
    "What model and algorithm does MDL-002 use and what is its risk rating?",
    "Which models have a Degraded status as of Q1 2026?",
    "What is MDL-005's Gini coefficient?",
]
SYSTEM = ("You are a model-risk analyst. Answer concisely, then write the "
          "final answer on a line prefixed with 'ANSWER: '.")

real_answers = []
t0 = time.time()
for q in QUESTIONS:
    r = client.messages.create(
        model=DEFAULT_TUTORIAL_MODEL, max_tokens=300, system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"<document>\\n{document_context}\\n</document>\\n\\n{q}"}],
    )
    real_answers.append({"question": q, "answer": r.content[0].text.strip()})
print(f"  wall time: {time.time() - t0:.1f}s")\
"""))

cells.append(md("""\
## 4. Compile + Route All Claims\
"""))

cells.append(code("""\
compiler = ClaimCompiler(store=store, semantic_mode="off")
router = VerificationRouter(store=store)

all_routed = []
for i, qa in enumerate(real_answers, 1):
    claims = compiler.compile(
        llm_answer=qa["answer"], question_text=qa["question"],
        run_id="nb09_demo", question_id=f"q{i}",
    )
    subverdicts = router.route_all(claims)
    all_routed.append({
        "question": qa["question"], "answer": qa["answer"],
        "claims": claims, "subverdicts": subverdicts,
    })
    n_pass = sum(1 for sv in subverdicts if sv.passed)
    print(f"[{i}] {len(claims)} claims  ->  {n_pass}/{len(subverdicts)} subverdicts passed")\
"""))

cells.append(md("""\
## 5. Per-Verifier Breakdown

Aggregate across all 5 questions and group by which verifier handled each claim.\
"""))

cells.append(code("""\
import pandas as pd
from collections import defaultdict
agg = defaultdict(lambda: {"total": 0, "passed": 0, "methods": set()})
for entry in all_routed:
    for sv in entry["subverdicts"]:
        agg[sv.verifier_name]["total"] += 1
        if sv.passed:
            agg[sv.verifier_name]["passed"] += 1
        if sv.gms_method_used:
            agg[sv.verifier_name]["methods"].add(sv.gms_method_used)

rows = []
for verifier, stats in agg.items():
    rows.append({
        "verifier": verifier,
        "total": stats["total"],
        "passed": stats["passed"],
        "pass_rate": stats["passed"] / stats["total"] if stats["total"] else 0.0,
        "gms_methods": ", ".join(sorted(stats["methods"])) or "—",
    })
df = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
df\
"""))

cells.append(md("""\
## 6. Inspect a Passing and a Failing SubVerdict

Both shapes are concrete — the `SubVerdict` schema makes pass and fail look the same; the difference is the `failure` attribute (`None` on pass, populated `FailureAttribution` on fail).\
"""))

cells.append(code("""\
all_subverdicts = [sv for entry in all_routed for sv in entry["subverdicts"]]
passed_sv = next((sv for sv in all_subverdicts if sv.passed), None)
failed_sv = next((sv for sv in all_subverdicts if not sv.passed), None)

for label, sv in [("PASSED", passed_sv), ("FAILED", failed_sv)]:
    if sv is None:
        print(f"{label}: no example available")
        continue
    print(f"=== {label} ===")
    print(f"  verifier_name   : {sv.verifier_name}")
    print(f"  claim_id        : {sv.claim_id}")
    print(f"  passed          : {sv.passed}")
    print(f"  confidence      : {sv.confidence:.3f}")
    print(f"  gms_method_used : {sv.gms_method_used}")
    print(f"  raw_score       : {sv.raw_score:.4f}" if sv.raw_score is not None else "  raw_score       : None")
    print(f"  detail          : {sv.detail[:120]}")
    if sv.failure is not None:
        print(f"  failure         :")
        print(f"     code         : {sv.failure.code.value}")
        print(f"     severity     : {sv.failure.severity}")
        print(f"     gms_method   : {sv.failure.gms_method}")
        print(f"     raw_score    : {sv.failure.raw_score}")
        print(f"     detail       : {sv.failure.detail[:100]}")
    print()\
"""))

cells.append(md("""\
## 7. Visualization — Verifier Workload + Pass Rate\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.barh(df["verifier"], df["total"], color="steelblue")
ax1.set_xlabel("# claims routed")
ax1.set_title("Verifier workload")
for i, n in enumerate(df["total"]):
    ax1.text(n + 0.3, i, str(n), va="center")

colors = ["#3aa56b" if r >= 0.5 else "#c14a4a" for r in df["pass_rate"]]
ax2.barh(df["verifier"], df["pass_rate"], color=colors)
ax2.set_xlim(0, 1)
ax2.set_xlabel("pass rate")
ax2.set_title("Verifier pass rate on bank corpus")
for i, r in enumerate(df["pass_rate"]):
    ax2.text(r + 0.02, i, f"{r:.0%}", va="center")

plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 8. Take-Aways

1. **One claim, one verifier.** No claim sees more than one verifier. The router's dispatch is type-driven; if `claim_type` is wrong, the wrong verifier runs.
2. **`raw_score` is the audit primitive.** Whether it's a geodesic distance, an ENM hash, or a holonomy defect, the number that drove the verdict is preserved — that's what the calibration step (NB 20) trains thresholds against.
3. **The failure carries the code, not the verdict.** `SubVerdict.passed` is the binary outcome; the `failure: FailureAttribution` payload carries the `FailureCode`, severity, GMS method, and detail. NB 22 walks the full code taxonomy.
4. **Coverage is the deployment story.** A bank-corpus deployment should aim for every claim to find a verifier. Claims that fall through (no verifier matched) are silent regressions — the router logs them, but they do not block the gate.
5. **`semantic_mode="off"` is the canonical setting.** Semantic claim extraction (mode="on") catches paraphrased claims but at the cost of noisier verifier inputs. Use it only when you've trained a calibration artifact that handles the noise.\
"""))

cells.append(md("""\
## 9. Next Steps

- **NB 10** — `10_verdicts_severity.ipynb` — aggregate SubVerdicts into a `TypedVerdict` and apply `SeverityClassifier` (Ch10)
- **NB 19** — `19_release_gates.ipynb` — the failure_budget the release gate enforces over these subverdicts (Ch19)
- **NB 20** — `20_calibration.ipynb` — the calibration step that tightens the raw_score thresholds (App G)

For theory:
- `gmsh_testing_ext/chapters/ch09.tex` — verifier composition lemmas, soundness/completeness of the router
- `gmsh_testing_ext/notebooks/ch09_verifiers.ipynb` — textbook lab walkthrough with custom-verifier exercises\
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
