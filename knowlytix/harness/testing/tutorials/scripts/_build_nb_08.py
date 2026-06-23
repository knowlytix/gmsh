# SPDX-License-Identifier: Apache-2.0
"""Build NB 08 — Decomposing Output into Claims — from cell definitions, then execute.

Demonstrates ClaimCompiler.compile() on real Sonnet 4.6 answers from the bank
model-risk corpus. Shows how a free-text LLM response becomes a list of
structured ClaimBase objects with raw_text + provenance + claim_type fields
that the downstream verifier router (NB 09) consumes.

Cost: ~$0.05 / ~30 sec wall clock (5 small Sonnet calls).

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/_build_nb_08.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "08_claim_decomposition.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 08: Decomposing LLM Output into Claims

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch08.tex` — Claims: Decomposing Agent Output into Verifiable Units
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch08_claims.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §11
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; ch04 fixture; NB 02 (Primitives)

**What you'll do.** Watch `ClaimCompiler.compile()` walk a free-text Sonnet 4.6 answer about the bank model inventory and emit a list of structured `ClaimBase` objects. Each claim carries its raw text, its `claim_type` (factual / numeric / structural / process / recovery / interaction), its provenance, and a confidence score. This is the bridge between the LLM's natural-language output and the verifier pipeline (NB 09) — without it, every downstream verifier would need its own ad-hoc parser.\
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
from knowlytix.harness.testing import ClaimCompiler
seed_all()
store = load_ch04_store()
print(f"store: {store.stats()}")\
"""))

cells.append(md("""\
## 2. Concept

An LLM answers in prose:

> *"MDL-001 (CreditScore-XGB) has a test AUC of 0.8412 on the credit panel. It is currently flagged as Degraded under the high risk tier."*

That single sentence carries three independently-verifiable claims:

1. `(MDL-001, has_test_auc, 0.8412)` — numeric fact, verifiable via `lookup_enm`
2. `(MDL-001, has_status, Degraded)` — categorical fact, verifiable via `query_triples`
3. `(MDL-001, has_risk_rating, High)` — categorical fact, verifiable via `query_triples`

`ClaimCompiler.compile(llm_answer, ...)` does this decomposition. It returns a `list[ClaimBase]` where each claim has:

- `claim_id` — UUID for cross-reference
- `raw_text` — the original sentence the claim was extracted from
- `claim_type` — one of `fact`, `threshold`, `rule`, `artifact`, `tool_use`, `semantic` (subclass: `FactClaim`, `ThresholdClaim`, `RuleClaim`, `ArtifactClaim`, `ToolUseClaim`, `SemanticClaim`)
- `provenance` — extraction metadata (extractor used, position in answer, etc.)
- `confidence` — extractor-side confidence (0..1)

The compiler is **type-aware**: a `FactClaim` carries `(head, relation, tail)` populated; a `ThresholdClaim` carries `(metric, operator, value)`; an `ArtifactClaim` carries `(artifact_id, expected_hash)`; etc. This shapes routing in NB 09 — each verifier handles one or two subclasses.\
"""))

cells.append(md("""\
## 3. Generate Real Sonnet 4.6 Answers

Five hand-picked questions covering different claim types, sent to Sonnet 4.6 with the bank model-risk corpus as context.\
"""))

cells.append(code("""\
client = make_anthropic_client()
with open("knowlytix/harness/testing/tests/model_risk_assessment.md") as f:
    document_context = f.read()

QUESTIONS = [
    # Numeric claim
    "What is the Test AUC of MDL-001?",
    # Categorical / structural
    "What is the current status of MDL-003?",
    # Multi-fact compound
    "What model and algorithm does MDL-002 use and what is its risk rating?",
    # Process-style
    "Which models have a Degraded status as of Q1 2026?",
    # Boundary / numeric near-miss
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
    real_answers.append({
        "question": q,
        "answer": r.content[0].text.strip(),
    })
print(f"  wall time: {time.time() - t0:.1f}s")
print()
for i, qa in enumerate(real_answers, 1):
    print(f"[{i}] Q: {qa['question']}")
    print(f"    A (last 160 chars): ...{qa['answer'][-160:]}")
    print()\
"""))

cells.append(md("""\
## 4. Compile Each Answer Into Claims\
"""))

cells.append(code("""\
compiler = ClaimCompiler(store=store, semantic_mode="off")
all_compiled = []
for i, qa in enumerate(real_answers, 1):
    claims = compiler.compile(
        llm_answer=qa["answer"],
        question_text=qa["question"],
        run_id="nb08_demo",
        question_id=f"q{i}",
    )
    all_compiled.append({"question": qa["question"], "answer": qa["answer"],
                         "claims": claims})
    print(f"[{i}] Q: {qa['question'][:60]}")
    print(f"    -> {len(claims)} claims extracted")
    for c in claims[:4]:
        print(f"       [{c.claim_type:12s}] conf={c.confidence:.2f} "
              f"raw='{c.raw_text[:80]}'")
    if len(claims) > 4:
        print(f"       ... ({len(claims) - 4} more)")
    print()\
"""))

cells.append(md("""\
## 5. Inspect Claim Subclasses

The compiler emits typed subclasses of `ClaimBase` — `FactClaim`, `NumericClaim`, `StructuralClaim`, etc. — each with extra fields the matching verifier needs. Below we look at one example of each subclass that appeared in §4.\
"""))

cells.append(code("""\
seen_types = {}
for entry in all_compiled:
    for c in entry["claims"]:
        if c.claim_type not in seen_types:
            seen_types[c.claim_type] = c

for ctype, sample in seen_types.items():
    print(f"=== {ctype} ===")
    print(f"  class       : {type(sample).__name__}")
    print(f"  claim_id    : {sample.claim_id}")
    print(f"  raw_text    : {sample.raw_text[:120]}")
    print(f"  confidence  : {sample.confidence}")
    extra = {k: v for k, v in sample.__dict__.items()
             if k not in {"claim_id", "source_run_id", "source_question_id",
                          "raw_text", "confidence", "provenance", "claim_type"}}
    if extra:
        print(f"  extra fields:")
        for k, v in extra.items():
            print(f"    {k:18s}: {str(v)[:100]}")
    print()\
"""))

cells.append(md("""\
## 6. Visualization — Claim Yield per Answer\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

n_claims_per_answer = [len(e["claims"]) for e in all_compiled]
labels = [f"Q{i+1}" for i in range(len(all_compiled))]
type_counts = {}
for e in all_compiled:
    for c in e["claims"]:
        type_counts.setdefault(c.claim_type, [0]*len(all_compiled))
for i, e in enumerate(all_compiled):
    for c in e["claims"]:
        type_counts[c.claim_type][i] += 1

fig, ax = plt.subplots(figsize=(9, 4.5))
bottom = np.zeros(len(all_compiled))
colors = {"factual": "#3aa56b", "numeric": "#4a8cd1", "structural": "#c89438",
          "process": "#a05a8a", "recovery": "#d16a4a", "interaction": "#6a6a6a"}
for ctype, counts in type_counts.items():
    ax.bar(labels, counts, bottom=bottom,
           label=ctype, color=colors.get(ctype, "#999999"))
    bottom += np.array(counts)
ax.set_ylabel("# claims extracted")
ax.set_title("Claim yield by type per LLM answer")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 7. Take-Aways

1. **One LLM sentence → many claims.** A compound bank-domain answer ("MDL-001 has AUC 0.8412 and is Degraded") yields 2-4 independent claims, each verifiable separately. The harness does not gate on "the answer was correct overall"; it gates on the typed verdicts each claim produces.
2. **`claim_type` drives routing.** `NumericClaim` ends up at `ENMVerifier`; `StructuralClaim` at `GraphVerifier`; `ProcessClaim` at `ProcessEvaluator`. The router (NB 09) reads this field — that is why the compiler must subclass.
3. **`raw_text` is the audit trail.** Every typed verdict carries the original sentence its claims were extracted from. When a failure report points to claim_id `c4`, you can read off the exact prose that triggered it.
4. **Confidence is *extraction* confidence.** Don't confuse it with verification confidence (NB 09's `SubVerdict.confidence`). A claim with confidence 0.5 means "we are not sure this is what the LLM meant" — high uncertainty here usually means the answer is paraphrastic / vague.
5. **`semantic_mode="off"` is the default.** Turn it on to extract claims the literal extractor misses (paraphrased facts, implicit numbers). Trade-off: extraction confidence drops because semantic parsing is fuzzier.\
"""))

cells.append(md("""\
## 8. Next Steps

- **NB 09** — `09_verifier_routing.ipynb` — feed these claims to `VerificationRouter` to produce `SubVerdict`s (Ch09)
- **NB 10** — `10_verdicts_severity.ipynb` — aggregate SubVerdicts into `TypedVerdict` + apply severity classification (Ch10)
- **NB 19** — `19_release_gates.ipynb` — the final consumer of these verdicts in the release-gate audit (Ch19)

For theory:
- `gmsh_testing_ext/chapters/ch08.tex` — formal claim taxonomy, paraphrase invariance proofs, semantic-mode trade-off analysis
- `gmsh_testing_ext/notebooks/ch08_claims.ipynb` — textbook lab walkthrough with adversarial-extraction exercises\
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
