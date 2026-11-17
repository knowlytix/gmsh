# Style Guide — GMSH Testing User Guide Tutorials

This document codifies writing conventions for `harness/doe/USER_GUIDE.md` (tutorial sections only — back-matter reference sections use their own conventions) and the companion `notebooks/NN_*.ipynb`.

The single most important rule:

> **The monograph teaches the *why*. The notebooks demonstrate the *how*. The guide is the bridge — short prose, minimum code, pointers to both.**

---

## 1. Tutorial Section Template (in USER_GUIDE.md)

Every tutorial section follows this exact skeleton. **Hard line budget: ≤ 60 lines.**

````markdown
## §N. Tutorial X: <Tutorial Title>

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/chNN.tex`](../../gmsh_testing_ext/chapters/chNN.tex) (<Chapter Title>)
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/chNN_<slug>.ipynb`](../../gmsh_testing_ext/notebooks/chNN_<slug>.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/NN_<slug>.ipynb`](user_guide_tutorials/notebooks/NN_<slug>.ipynb)
> **Key classes:** `PrimaryClass`, `helper_function()`
> **Prereqs:** `bench`, `store` (from NB 00 Quick Start)

**When to use this.** One paragraph (3–5 sentences) describing the concrete problem the reader faces and how this tutorial addresses it. Do *not* restate the chapter's motivation — point to the chapter and move on.

```python
# Minimum API skeleton — 5 to 15 lines, no inline commentary
from harness.doe import PrimaryClass
result = PrimaryClass(store=store).run(scenarios)
print(result.summary())
```

**Common variations.**
- `enable_X=True` — see notebook §3 for the parameter sweep
- `custom_Y=...` — see notebook §5 for the pluggable interface
- Batch mode: see §37 Common Recipes

**Edge cases.** One short bullet list of failure modes the reader should expect, each ending with a pointer to the notebook section that exercises it.
````

### What goes where

| Content type                          | Belongs in        |
|---------------------------------------|-------------------|
| Definitions, math, theorems           | **Monograph chapter** |
| Conceptual diagrams                   | **Monograph chapter** |
| Why this matters / motivation         | **Monograph chapter** |
| When to reach for this tool           | **Guide section** |
| Minimal API skeleton (5–15 lines)     | **Guide section** |
| Common variations bullet list         | **Guide section** |
| Runnable end-to-end example           | **Notebook**      |
| Parameter sweeps / sensitivity        | **Notebook**      |
| Outputs (tables, plots)               | **Notebook**      |
| Edge case demonstrations              | **Notebook**      |

If a tutorial section grows past 60 lines, move material to the notebook.

---

## 2. Notebook Skeleton

```
Cell 1  — markdown: title block (see template below)
Cell 2  — code: setup (imports + fixture load via _shared)
Cell 3  — markdown: concept recap (one paragraph; do not re-derive the chapter)
Cell 4+ — alternating code / markdown: main walkthrough
        - run a small end-to-end example
        - show outputs (DataFrames, matplotlib figures, prints)
        - vary one parameter; show the effect
Cell N-2 — code: predictable failure modes with clear error messages
Cell N-1 — markdown: take-aways (3–5 action-oriented bullets)
Cell N   — markdown: next steps (link to next NB + back to chapter)
```

**Target cell count: 15–40 cells.** Notebooks longer than 40 cells should split into sub-notebooks (`NN_<slug>_part_a.ipynb`, `NN_<slug>_part_b.ipynb`).

### Title cell template

```markdown
# Tutorial X: <Title>

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/chNN.tex` — <Chapter Title>
**Monograph notebook:** `gmsh_testing_ext/notebooks/chNN_<slug>.ipynb`
**Companion guide section:** `harness/doe/USER_GUIDE.md` §N
**Prereqs:** activated `~/cluster/spark-venv`, `ANTHROPIC_API_KEY` (if applicable), prior notebooks (list)

**What you'll do.** One sentence describing the runnable outcome of this notebook.
```

The three backlinks make the artifact triad explicit: the chapter `.tex` is theory, the monograph notebook is the textbook lab walkthrough (deeper, with exercises), and the user-guide notebook is the practitioner recipe (shorter, recipe-focused). Each artifact links to the other two so readers can navigate freely.

### Setup cell template

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))
from _shared import load_ch04_store, setup_plotting

store, bench = load_ch04_store()
setup_plotting()
```

---

## 3. Writing Voice

- **American English.** No Oxford comma. (Memory: `feedback_monograph_style`.)
- **Active voice.** "Configure the gateway with..." not "The gateway should be configured with..."
- **Direct.** "Use `TieredReleaseGate` for risk-tiered cuts." not "It may be useful in some cases to consider using..."
- **No filler.** Cut "simply," "just," "easy," "powerful," "intuitive."
- **Numbers in figures, not in prose.** Don't write "the accuracy went up by 3.2 points" — show the plot in the notebook.
- **Teach concept before API.** When a tutorial introduces a new class, lead with one sentence on what role it plays in the testing pipeline before showing its constructor. (Memory: `feedback_monograph_teaching_style`.)

### Banned framings

- ❌ "regulatory compliance" / "SR 11-7" / "EU AI Act" / "NIST AI RMF" / "BCBS 239"
  — unless the section is explicitly the regulatory-mapping section. Memory: `feedback_sr_11_7_rescinded`.
- ❌ "bank-grade" — describe the engineering property directly. Memory: `feedback_no_bank_grade`.
- ❌ Mock-LLM fallback. If a tutorial needs an LLM, raise `AnthropicAPIKeyError` on missing key. Memory: `feedback_no_mock_fallback`.

---

## 4. Code Conventions

### Imports
- Group standard library, third-party, local — separated by blank lines.
- Prefer specific imports: `from harness.doe.policy import PolicyEngine` over `import harness.doe`.

### Paths
- Absolute paths via `_shared.py` constants. No hard-coded `/home/asudjianto/...` in notebooks except in `_shared.py` itself.

### Fixtures
- All notebooks load via `_shared.load_ch04_store()` unless they need a sub-fixture.
- **Never mutate the ch04 store.** If a notebook needs a perturbed store, build in-memory and discard.

### Reproducibility
- Set `np.random.seed(42)` and `torch.manual_seed(42)` in the setup cell.
- For calibration / threshold sensitive results, also pin `random.seed(42)`.

### Output
- Use `pandas` DataFrames for tabular output (not raw dicts).
- Plots via `matplotlib`; one figure per cell; always set a title and axis labels.
- Errors should propagate — do not wrap in try/except just to print and continue.

---

## 5. Cross-Reference Convention

Every tutorial in `USER_GUIDE.md` and every notebook's title cell carries **two backlinks**:

1. **Conceptual deep-dive** → the monograph chapter (`gmsh_testing_ext/chapters/chNN.tex`)
2. **Companion** → the paired artifact (guide section ↔ notebook)

When referencing other tutorials, use the **section number** in the guide and the **notebook number** for code: "see §15 (NB 12) for fault injection."

When referencing monograph theorems, use the chapter file and label: "see `gmsh_testing_ext/chapters/ch06.tex` Theorem 6.3."

---

## 6. Calibration Artifact References

The hardened calibration artifacts live in the monograph:

- `gmsh_testing_ext/lab_stores/calibration_base/calibration.json` — base τ_geo, τ_ent, τ_contra
- `gmsh_testing_ext/lab_stores/ch18_doc_q1/calibration.json` — multi-document calibration

Notebooks load these read-only via `_shared.load_calibration()`. Do not regenerate calibration in tutorial notebooks (memory: `feedback_deterministic_calibration_for_audit` — high-stakes numbers must reproduce byte-identically).

---

## 7. Don't

- Don't paste theory paragraphs from the monograph into the guide.
- Don't show the same code in the guide and the notebook — the guide skeleton is *minimal*, the notebook is *full*.
- Don't write a tutorial without its deep-dive link.
- Don't write a notebook longer than 40 cells. Split it.
- Don't rebuild fixtures in tutorial notebooks.
- Don't add mock-LLM fallback paths.
- Don't introduce new monograph-style hardening artifacts (blessed fingerprints, FSM regression logs) in tutorial notebooks. Those belong in `gmsh_testing_ext/lab_runs/`.

---

## 8. Do

- Do read the matching chapter before drafting a tutorial.
- Do verify the API skeleton compiles against the current library.
- Do execute the notebook end-to-end and commit with outputs.
- Do link to specific notebook section numbers when promising "see notebook §X."
- Do raise on missing fixtures with a clear error pointing to the rebuild script.
- Do keep guide tutorials under 60 lines.
