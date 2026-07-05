# SPDX-License-Identifier: Apache-2.0
"""Build NB 01 — Framing the Testing Problem (Ch01)."""
from __future__ import annotations
import time
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "01_framing.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 01: Framing the Agentic Testing Problem

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch01.tex` — The Agentic AI Testing Problem
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch01_agentic_testing_problem.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §4
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; ch04 fixture

**What you'll do.** Ask Claude Sonnet 4.6 three bank model-risk questions where the *right* answer is precisely knowable from the model inventory but a confident-sounding wrong answer is easy to produce. Score the answers with three different schemes — exact match, BLEU-style overlap, and the harness's GMS judge — to see why surface-level scorers cannot detect the failures that matter for a production deployment.\
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
seed_all()
store = load_ch04_store()
client = make_anthropic_client()
print("imports OK")\
"""))

cells.append(md("""\
## 2. The Problem in One Sentence

> A confident-sounding wrong answer about a bank model is operationally worse than a hedged correct one — but every surface-level scorer rates them in the wrong order.

The harness is the geometric machinery for measuring whether an agent's answer is *consistent with what the model inventory actually records*, not whether it superficially looks like the answer.\
"""))

cells.append(md("""\
## 3. Three Bank Model-Risk Questions

Hand-picked so the right answer is unambiguous: a number, an identifier, a categorical status. Each can be looked up in the ch04 store directly.\
"""))

cells.append(code("""\
QUESTIONS = [
    {
        "q": "What is MDL-001's Test AUC?",
        "gt": "0.8412",
        "lookup": ("performance_metrics", "MDL-001/Test AUC"),
    },
    {
        "q": "What algorithm does MDL-002 (FraudNet) use?",
        "gt": "LSTM",
        "lookup": None,  # categorical, look up via triple query
    },
    {
        "q": "What is MDL-003's current operational status?",
        "gt": "Degraded",
        "lookup": None,
    },
]

for item in QUESTIONS:
    if item["lookup"]:
        cat, eid = item["lookup"]
        stored = store.lookup_enm(cat, eid)
        print(f"  Q: {item['q']}")
        print(f"     ground truth (ENM): {stored}")
    else:
        print(f"  Q: {item['q']}")
        print(f"     ground truth: {item['gt']}")
    print()\
"""))

cells.append(md("""\
## 4. Ask Sonnet 4.6 — Real Answers on the Bank Corpus\
"""))

cells.append(code("""\
with open("knowlytix/harness/testing/tests/model_risk_assessment.md") as f:
    context = f.read()

SYSTEM = ("You are a model-risk analyst answering questions about a bank "
          "model inventory. Be concise; end with a line 'ANSWER: <value>'.")

llm_answers = []
for item in QUESTIONS:
    r = client.messages.create(
        model=DEFAULT_TUTORIAL_MODEL, max_tokens=200, system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"<document>\\n{context}\\n</document>\\n\\n{item['q']}"}],
    )
    llm_answers.append(r.content[0].text.strip())

for item, ans in zip(QUESTIONS, llm_answers):
    print(f"Q: {item['q']}")
    print(f"  GT  : {item['gt']}")
    print(f"  LLM : ...{ans[-160:]!r}")
    print()\
"""))

cells.append(md("""\
## 5. Three Scoring Schemes\
"""))

cells.append(code("""\
import re

def score_exact(answer: str, gt: str) -> bool:
    # Extract value after 'ANSWER:' if present
    m = re.search(r"ANSWER:\\s*(.+?)$", answer, re.MULTILINE)
    payload = m.group(1).strip().strip("*'\\"") if m else answer.strip()
    return payload.lower() == gt.lower()

def score_bleu_like_overlap(answer: str, gt: str) -> float:
    # Crude word overlap (BLEU-1 unigram precision approximation)
    a_words = set(re.findall(r"\\b\\w+\\b", answer.lower()))
    g_words = set(re.findall(r"\\b\\w+\\b", gt.lower()))
    if not g_words: return 0.0
    return len(a_words & g_words) / len(g_words)

# GMS judge: ground-truth comparison via score_triple where applicable.
def score_gms_judge(answer: str, gt: str, head: str, relation: str) -> float | None:
    \"\"\"Geodesic distance for (head, relation, parsed_answer). Lower = better.\"\"\"
    m = re.search(r"ANSWER:\\s*(.+?)$", answer, re.MULTILINE)
    val = (m.group(1).strip().strip("*'\\"") if m else answer.strip()).split()[0]
    return store.score_triple(head, relation, val)

table_rows = []
for item, ans in zip(QUESTIONS, llm_answers):
    head, relation = {
        "What is MDL-001's Test AUC?": ("mdl-001", "has_test_auc"),
        "What algorithm does MDL-002 (FraudNet) use?": ("mdl-002", "has_algorithm"),
        "What is MDL-003's current operational status?": ("mdl-003", "has_status"),
    }[item["q"]]
    table_rows.append({
        "question": item["q"][:48],
        "ground_truth": item["gt"],
        "exact_match": score_exact(ans, item["gt"]),
        "word_overlap": round(score_bleu_like_overlap(ans, item["gt"]), 3),
        "gms_geodesic": round(score_gms_judge(ans, item["gt"], head, relation) or float("nan"), 3),
    })

import pandas as pd
df = pd.DataFrame(table_rows)
df\
"""))

cells.append(md("""\
## 6. Read the Table

- **`exact_match`** is binary: either the parsed answer string equals the ground truth or not.
- **`word_overlap`** rises with paraphrase ("0.8412 on the credit panel" overlaps with "0.8412") but is gameable by an agent that quotes long passages from the document.
- **`gms_geodesic`** scores against the actual store: numbers near 0 mean the (head, relation, parsed_answer) triple is geometrically close to the trained ground truth.

The story: surface scorers can't disambiguate between an agent that quotes the right value and an agent that paraphrases correctly without verification. The GMS judge can.\
"""))

cells.append(md("""\
## 7. Take-Aways

1. **Bank-domain testing has hard ground truth.** Numeric AUCs, identifier mappings, operational statuses — every right answer is precisely knowable from the model inventory.
2. **Confident wrong answers are the failure mode.** An LLM that emits `0.84` instead of `0.8412` on a credit decision is not "close enough"; it's wrong in a way that matters for capital impact.
3. **Surface scorers measure the wrong thing.** Exact match misses paraphrase; word overlap is gameable; both are blind to the underlying claim's truth.
4. **The geometric judge measures truth.** `score_triple(head, relation, tail)` returns geodesic distance on the cap manifold — a calibrated probability after `bench.judge.calibrate()` (NB 20).
5. **This guide is the practitioner-facing recipe.** The monograph chapter `gmsh_testing_ext/chapters/ch01.tex` makes the formal argument; everything that follows in this guide builds on top of geometric verification.\
"""))

cells.append(md("""\
## 8. Next Steps

- **NB 00** — `00_quick_start.ipynb` — run the full pipeline end-to-end (preface/Ch01)
- **NB 02** — `02_gms_primitives.ipynb` — the six geometric primitives that drive every verifier (Ch02)
- **NB 16** — `16_hallucination.ipynb` — the geodesic distance threshold in action (Ch16)

For theory: `gmsh_testing_ext/chapters/ch01.tex`.\
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
