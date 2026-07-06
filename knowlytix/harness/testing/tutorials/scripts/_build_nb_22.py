# SPDX-License-Identifier: Apache-2.0
"""Build NB 22 — Failure Code Taxonomy — from cell definitions, then execute.

Reference notebook for the 29 FailureCode enum members, the SeverityClassifier
mapping, risk-tier escalation, and how RiskTierProfile.failure_budget consumes
the codes. Pure reference content + a brief live classification demo on
synthetic verdicts. No LLM, no API budget.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_22.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "22_failure_codes.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 22: Failure Code Taxonomy and Severity

**Conceptual deep-dive:** `gmsh_testing_ext/appendix_c.tex` — Failure Code Taxonomy
**Monograph notebook:** *(reference appendix — no companion lab notebook)*
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §25
**Prereqs:** activated `~/cluster/spark-venv`; NB 10 (Verdicts & Severity), NB 19 (Release Gates)

**What you'll do.** Enumerate every `FailureCode` the harness's verifiers can emit, show how `SeverityClassifier` maps each to a `SeverityLevel`, demonstrate **risk-tier escalation** (the same code is `MODERATE` for advisory deployments and `CRITICAL` for action_taking), and trace how `RiskTierProfile.failure_budget` consumes the codes to decide which deployment tier a SUT can ship to. Pure reference content + a brief live classification demo on a synthetic failure set. No LLM calls.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from _shared import seed_all
seed_all()

from knowlytix.harness.testing import (
    FailureCode, SeverityClassifier, SeverityLevel, RiskTierProfile,
)
print(f"FailureCode members      : {len(list(FailureCode))}")
print(f"SeverityLevel members    : {[s.name for s in SeverityLevel]}")\
"""))

cells.append(md("""\
## 2. The 29 Failure Codes — Full Taxonomy

Every typed verdict the harness emits attributes its failure to one of these codes. The codes group into four families:

| Family | Codes | Verifier that emits them |
|---|---|---|
| **Claim extraction** | `SEMANTIC_EXTRACTION_FAILURE`, `SEMANTIC_MISMATCH`, `NO_CLAIMS_EXTRACTED`, `INCOMPLETE_ANSWER` | `ClaimCompiler` (Ch08) |
| **Fact verification** | `HALLUCINATION_DETECTED`, `ENTITY_NOT_FOUND`, `RELATION_NOT_FOUND`, `FACT_INCORRECT`, `EXACT_MISMATCH`, `THRESHOLD_VIOLATION`, `BOUNDARY_FAILURE` | `GraphVerifier` / `ENMVerifier` (Ch09) |
| **Geometric structure** | `PATH_INCONSISTENCY`, `CONTRADICTION_DETECTED`, `TENSION_EXCEEDED`, `HOLONOMY_DEFECT` | `MultiHopVerifier` / `ContradictionVerifier` (Ch09) |
| **Operational** | `TOOL_CALL_INVALID`, `POLICY_VIOLATION`, `POLICY_BLOCKED_PRE_EXECUTION`, `APPROVAL_REQUIRED`, `REQUIRED_TOOL_MISSING`, `FORBIDDEN_TOOL_CALLED`, `PROCESS_DEVIATION`, `TOOL_BUDGET_EXCEEDED`, `RECOVERY_FAILURE`, `ESCALATION_INAPPROPRIATE`, `CLARIFICATION_INSUFFICIENT` | `ProcessEvaluator` / `RecoveryEvaluator` / `PolicyEngine` (Ch12, Ch13, Ch17) |
| **Pipeline** | `VERIFICATION_ERROR`, `NOT_CALIBRATED`, `ARTIFACT_INTEGRITY_FAILURE` | Pipeline self-checks |\
"""))

cells.append(md("""\
## 3. Default Severity Map

`SeverityClassifier.classify_code(code)` returns a `SeverityAssessment` with the severity `level`, three component scores (`impact`, `recoverability`, `detectability`), a `composite_score`, and a `rationale`. Below is the full table at the default `risk_tier="standard"` setting.\
"""))

cells.append(code("""\
import pandas as pd
sc = SeverityClassifier()
rows = []
for fc in FailureCode:
    sa = sc.classify_code(fc)
    rows.append({
        "code": fc.value,
        "severity": sa.level.name,
        "impact": sa.impact_score,
        "recoverability": sa.recoverability_score,
        "detectability": sa.detectability_score,
        "composite": sa.composite_score,
    })
df = pd.DataFrame(rows).sort_values(["severity", "composite"], ascending=[True, False])
df.reset_index(drop=True, inplace=True)
df\
"""))

cells.append(md("""\
## 4. Risk-Tier Escalation

The same failure code carries different severity under different deployment tiers. `FACT_INCORRECT` is `HIGH` for advisory deployments but escalates to `CRITICAL` under `action_taking` — a fact-incorrect autonomous credit decision is materially worse than a fact-incorrect suggestion to a human analyst.

Below we compare three tiers across the four failure-code families.\
"""))

cells.append(code("""\
risk_tiers = ["advisory", "recommendation", "action_taking"]
spotlight_codes = [
    FailureCode.HALLUCINATION_DETECTED,
    FailureCode.FACT_INCORRECT,
    FailureCode.POLICY_VIOLATION,
    FailureCode.CONTRADICTION_DETECTED,
    FailureCode.TENSION_EXCEEDED,
    FailureCode.PROCESS_DEVIATION,
    FailureCode.RECOVERY_FAILURE,
    FailureCode.NOT_CALIBRATED,
]
rows = []
for fc in spotlight_codes:
    row = {"code": fc.value}
    for tier in risk_tiers:
        sa = sc.classify_code(fc, risk_tier=tier)
        row[tier] = sa.level.name
    rows.append(row)
pd.DataFrame(rows).set_index("code")\
"""))

cells.append(md("""\
## 5. `failure_budget` — How the Release Gate Consumes the Codes

`RiskTierProfile.failure_budget` is a dict mapping failure-code strings to maximum-allowed-count integers. The release gate (NB 19) sums each code's appearances across the typed verdicts and gates pass/fail on whether the count exceeds the budget.

For the three canonical tiers:\
"""))

cells.append(code("""\
tiers = {
    "advisory": RiskTierProfile.advisory(),
    "recommendation": RiskTierProfile.recommendation(),
    "action_taking": RiskTierProfile.action_taking(),
}
budget_rows = []
all_codes = set()
for t in tiers.values():
    all_codes.update(t.failure_budget.keys())

for code in sorted(all_codes):
    row = {"code": code}
    for name, t in tiers.items():
        row[name] = t.failure_budget.get(code, "∞")
    budget_rows.append(row)
pd.DataFrame(budget_rows).set_index("code") if budget_rows else "No explicit budgets set on the standard tiers"\
"""))

cells.append(md("""\
Read this table as: at `action_taking`, **zero** hallucinations, policy violations, and pre-execution policy blocks are tolerated. At `recommendation`, only hallucinations are zeroed. At `advisory`, the failure budgets are empty (∞) — operational issues are surfaced but do not block the deployment.

This is what gives the same SUT three independent release-gate verdicts on the same data.\
"""))

cells.append(md("""\
## 6. Live Classification Demo

Pass a synthetic list of failure-attribution records through `SeverityClassifier.classify_all()` and read off the level distribution. This mirrors what happens inside `AuditReporter.generate_failure_report()` (NB 19 §8).\
"""))

cells.append(code("""\
from knowlytix.harness.testing import FailureAttribution

# Synthetic verdict-level failures one might see from a real bench run
synthetic_failures = [
    FailureAttribution(code=FailureCode.HALLUCINATION_DETECTED, claim_id="c1",
                       verifier="GraphVerifier", severity="HIGH",
                       detail="Geodesic 0.91 > tau_geo 0.876",
                       gms_method="score_triple", raw_score=0.91),
    FailureAttribution(code=FailureCode.FACT_INCORRECT, claim_id="c2",
                       verifier="ENMVerifier", severity="HIGH",
                       detail="Expected 0.8412, got 0.84",
                       gms_method="lookup_enm", raw_score=0.0),
    FailureAttribution(code=FailureCode.CONTRADICTION_DETECTED, claim_id="c3",
                       verifier="ContradictionVerifier", severity="HIGH",
                       detail="tension 1.51 > tau_contra 1.43",
                       gms_method="tension_energy", raw_score=1.51),
    FailureAttribution(code=FailureCode.PROCESS_DEVIATION, claim_id="c4",
                       verifier="ProcessEvaluator", severity="MODERATE",
                       detail="Tool path differs from declared",
                       gms_method="check_holonomy", raw_score=10.5),
    FailureAttribution(code=FailureCode.NOT_CALIBRATED, claim_id="c5",
                       verifier="GMSJudge", severity="LOW",
                       detail="No calibration artifact loaded",
                       gms_method="", raw_score=0.0),
]

# Classify under three tiers
for tier in ["advisory", "recommendation", "action_taking"]:
    assessments = sc.classify_all(synthetic_failures, risk_tier=tier)
    counts = sc.count_by_level(assessments)
    print(f"  [{tier:14s}] " +
          "  ".join(f"{lvl.name}={counts.get(lvl, 0)}" for lvl in SeverityLevel))\
"""))

cells.append(md("""\
## 7. Visualization — Severity Distribution by Tier\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

tiers_list = ["advisory", "recommendation", "action_taking"]
levels = [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MODERATE, SeverityLevel.LOW]
data = []
for tier in tiers_list:
    assessments = sc.classify_all(synthetic_failures, risk_tier=tier)
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
ax.set_ylabel("count of failures")
ax.set_title("Same failure set, three deployment tiers — severity shifts up")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 8. Take-Aways

1. **29 codes, 4 families.** Claim extraction, fact verification, geometric structure, operational. Every typed verdict failure attributes to one. There is no "other" bucket — extend `FailureCode` if you need a new family.
2. **Severity is tier-dependent.** The same code is `HIGH` for advisory and `CRITICAL` for action_taking. The classifier handles this via `risk_tier=` — never hard-code severity in your reporting.
3. **`failure_budget` is the production gate.** Counts of specific codes vs hard limits. `hallucination_detected: 0` and `policy_violation: 0` at `action_taking` is the standard for autonomous bank decisions.
4. **`NOT_CALIBRATED` is `LOW`, not benign.** It is a `LOW` severity *signal*, but it pollutes every other verdict's calibrated-probability channel. NB 20 shows the fix (`bench.judge.calibrate()` before `bench.run()`).
5. **Composite scores let you rank.** The `composite_score` blends impact, recoverability, and detectability. Use it as the primary sort key when triaging a large failure list.\
"""))

cells.append(md("""\
## 9. Next Steps

- **NB 10** — `10_verdicts_severity.ipynb` — how typed verdicts are built (Ch10)
- **NB 19** — `19_release_gates.ipynb` — how `failure_budget` is consumed at release time (Ch19)
- **NB 20** — `20_calibration.ipynb` — make `NOT_CALIBRATED` disappear (App G)

For theory:
- `gmsh_testing_ext/appendix_c.tex` — full taxonomy derivation, escalation rules per tier, the rationale templates the classifier emits\
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
