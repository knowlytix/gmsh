# GMSH Testing Harness — User Guide + Tutorial Notebooks Master Plan

**Status:** Drafted 2026-05-11
**Owner:** Sudjianto + Claude agents
**Goal:** Restructure `harness/doe/USER_GUIDE.md` to align 1:1 with the `gmsh_testing_ext/` monograph, and deliver a matching set of executable Jupyter notebooks (one per tutorial) under `harness/doe/user_guide_tutorials/notebooks/`.

---

## 1. Goals and Non-Goals

### Goals
1. Restructure `harness/doe/USER_GUIDE.md` so every tutorial maps to exactly one monograph chapter or appendix, with an explicit *Conceptual deep-dive* link to that chapter.
2. Shrink each tutorial's prose to a **thin recipe**: when-to-use paragraph + minimum API skeleton + variations bullets + notebook pointer. The notebook does the heavy lifting; the monograph carries the theory.
3. Deliver `harness/doe/user_guide_tutorials/notebooks/NN_*.ipynb` — runnable, GPU-tested notebooks committed *with* outputs.
4. Reuse the hardened monograph fixture `gmsh_testing_ext/lab_stores/ch04_ground_truth/` (823 triples, 514 entities, cap_enabled=True) as the read-only ground-truth store for all chapter-aligned notebooks.
5. Each notebook executes end-to-end on `~/cluster/spark-venv` (CUDA 13.0, GPU GB10) in under 10 minutes.

### Non-Goals
- Do **not** duplicate the monograph's chapter lab notebooks under `gmsh_testing_ext/notebooks/chNN_*.ipynb`. Those already exist (Ch01–Ch19 + appF/G); each user-guide notebook *links* to its matching monograph notebook rather than reproducing its content. Pedagogical split: monograph notebook = textbook lab with Learning Objectives and exercises; user-guide notebook = practitioner recipe focused on a bank-domain workflow.
- Do **not** duplicate monograph theory in the guide. The monograph is the canonical source for *why*; the guide and notebooks are for *how*.
- Do **not** introduce mock-LLM fallback paths. Real-LLM tutorials raise `AnthropicAPIKeyError` when the key is absent (memory: `feedback_no_mock_fallback`).
- Do **not** use "bank-grade" language (memory: `feedback_no_bank_grade`).
- Do **not** frame governance as regulatory compliance unless the section explicitly handles regulatory mapping (memory: `feedback_sr_11_7_rescinded`).
- Do **not** rebuild the ch04 fixture from scratch in any notebook — load it read-only via `_shared.py`.

### The three-artifact triad
For every chapter `NN`, three artifacts coexist and cross-link:

1. **`gmsh_testing_ext/chapters/chNN.tex`** — theory, definitions, derivations.
2. **`gmsh_testing_ext/notebooks/chNN_<slug>.ipynb`** — textbook lab walkthrough paired with the chapter; Learning Objectives, exercises, full math.
3. **`harness/doe/user_guide_tutorials/notebooks/NN_<slug>.ipynb`** — practitioner recipe paired with `harness/doe/USER_GUIDE.md` §M; thin, runnable, bank-corpus focused.

Each artifact's first cell/section links to the other two. This is the canonical convention enforced by `STYLE.md`.

---

## 2. Directory Layout

```
kg-memory/harness/doe/user_guide_tutorials/
├── PLAN.md                       (this file)
├── README.md                     (user-facing index — Phase 0)
├── STYLE.md                      (writing + notebook conventions — Phase 0)
├── draft_sections/               (per-section drafts assembled into USER_GUIDE.md)
├── scripts/
│   ├── run_all_notebooks.py      (CI: nbconvert --execute on every NB)
│   └── validate_outputs.py       (post-execution assertions per NB)
└── notebooks/
    ├── _shared.py                (fixture loaders, common imports, plot helpers)
    ├── 00_quick_start.ipynb      (10-minute end-to-end pipeline)
    ├── 01_framing.ipynb          ← Ch01
    ├── 02_gms_primitives.ipynb   ← Ch02
    ├── 03_doe_design.ipynb       ← Ch03
    ├── 04_ingestion.ipynb        ← Ch04 (loads ch04 corpus, no rebuild)
    ├── 05_question_generation.ipynb  ← Ch05
    ├── 06_geometric_judge.ipynb  ← Ch06
    ├── 07_statistical_analysis.ipynb ← Ch07
    ├── 08_claim_decomposition.ipynb  ← Ch08
    ├── 09_verifier_routing.ipynb ← Ch09
    ├── 10_verdicts_severity.ipynb ← Ch10
    ├── 11_sut_adapters.ipynb     ← Ch11
    ├── 12_tool_gateway.ipynb     ← Ch12
    ├── 13_policy_engine.ipynb    ← Ch13
    ├── 14_environment_sim.ipynb  ← Ch14
    ├── 15_oracles.ipynb          ← Ch15
    ├── 16_hallucination.ipynb    ← Ch16
    ├── 17_multi_turn.ipynb       ← Ch17
    ├── 18_multi_agent.ipynb      ← Ch18
    ├── 19_release_gates.ipynb    ← Ch19
    ├── 20_calibration.ipynb      ← Appendix G
    ├── 21_gms_diagnostics.ipynb  ← Appendix F
    ├── 22_failure_codes.ipynb    ← Appendix C
    └── 23_operational_features.ipynb  (parallel, perf, cost, dashboard,
                                         paraphrase, streaming, browser,
                                         artifact versioning)
```

No `fixtures/` directory — we read directly from `gmsh_testing_ext/lab_stores/`.

---

## 3. Revised Section Map for USER_GUIDE.md

Legend: **NEW** = section did not exist · **REVISE** = lift + shrink + add deep-dive callout · **KEEP** = unchanged · **DROP** = remove from main flow (folded into ops annex or back matter)

### Front matter (3 sections)

| §  | Title                        | Status | Source                              | NB |
|----|------------------------------|--------|-------------------------------------|----|
| 1  | Prerequisites & Installation | KEEP   | existing §1 (light update for venv) | —  |
| 2  | Architecture Overview        | REVISE | existing §3 + layered-picture diagram | —  |
| 3  | Quick Start                  | REVISE | existing §2 trimmed to 10-min path  | 00 |

### Part I — Foundations (Ch01–Ch07 → §4–§10)

| §  | Title                                       | Chapter | Status  | NB |
|----|---------------------------------------------|---------|---------|----|
| 4  | Framing the Testing Problem                 | Ch01    | NEW     | 01 |
| 5  | GMS as Ground Truth — primitives in 10 min  | Ch02    | NEW     | 02 |
| 6  | Designing Test Experiments (DOE)            | Ch03    | REVISE  | 03 |
| 7  | Building a Ground-Truth Store (Ingestion)   | Ch04    | REVISE  | 04 |
| 8  | Generating Questions from Geometry          | Ch05    | REVISE  | 05 |
| 9  | The Geometric Judge — per-category scoring  | Ch06    | NEW     | 06 |
| 10 | Statistical Analysis of Factor Effects      | Ch07    | REVISE  | 07 |

### Part II — Claim Pipeline (Ch08–Ch10 → §11–§13)

| §  | Title                                  | Chapter | Status | NB |
|----|----------------------------------------|---------|--------|----|
| 11 | Decomposing Output into Claims         | Ch08    | NEW    | 08 |
| 12 | Routing Claims to Verifiers            | Ch09    | NEW    | 09 |
| 13 | Verdicts, Failure Codes, and Severity  | Ch10    | REVISE | 10 |

### Part III — SUT and Runtime Probes (Ch11–Ch15 → §14–§18)

| §  | Title                                          | Chapter | Status | NB |
|----|------------------------------------------------|---------|--------|----|
| 14 | Connecting Your SUT (Adapters, Targets, Traces)| Ch11    | REVISE | 11 |
| 15 | Tool Gateway and Fault Injection               | Ch12    | REVISE | 12 |
| 16 | Policy Engine                                  | Ch13    | REVISE | 13 |
| 17 | Environment Simulation                         | Ch14    | REVISE | 14 |
| 18 | Oracle Framework                               | Ch15    | REVISE | 15 |

### Part IV — Behavioral and Multi-Source (Ch16–Ch18 → §19–§21)

| §  | Title                                       | Chapter | Status | NB |
|----|---------------------------------------------|---------|--------|----|
| 19 | Hallucination Detection                     | Ch16    | REVISE | 16 |
| 20 | Stateful and Multi-Turn Testing             | Ch17    | REVISE | 17 |
| 21 | Multi-Agent & Cross-Document Consistency    | Ch18    | REVISE | 18 |

### Part V — Production Lifecycle (Ch19 → §22)

| §  | Title                                       | Chapter | Status | NB |
|----|---------------------------------------------|---------|--------|----|
| 22 | Campaigns, Release Gates, Provenance        | Ch19    | REVISE | 19 |

### Part VI — Calibration & Diagnostics (appendix-anchored, §23–§25)

| §  | Title                                       | Appendix | Status | NB |
|----|---------------------------------------------|----------|--------|----|
| 23 | Confidence Calibration (ECE / MCE)          | App G    | NEW    | 20 |
| 24 | GMS Training & Diagnostics                  | App F    | NEW    | 21 |
| 25 | Failure Code Taxonomy (reference + recipes) | App C    | NEW    | 22 |

### Part VII — Operational Annex (§26, single combined NB)

§26 *Operational Features* — single page covering parallel execution, performance profiling, cost & token tracking, browser/UI stubs, streaming consistency, live dashboard, paraphrase invariance, artifact versioning. **One shared notebook (NB 23).**

### Back Matter (§27–§31, no notebooks)

| §  | Title                  | Status |
|----|------------------------|--------|
| 27 | End-to-End Walkthrough | REVISE (trim, link to NB 00 + NB 19) |
| 28 | CLI Reference          | KEEP   |
| 29 | Configuration Reference| KEEP   |
| 30 | Troubleshooting        | KEEP   |
| 31 | Companion Resources    | NEW (links to monograph, governed_agent guide, API_REFERENCE.md) |

### Totals
- **25 tutorial sections** (22 chapter-anchored + 3 appendix-anchored) each with its own notebook.
- **1 ops-annex section** with 1 combined notebook.
- **24 notebooks** (NB 00 quickstart + NB 01–23).
- **5 back-matter sections** (reference material, no notebooks).

---

## 4. Per-Tutorial Section Skeleton

Every tutorial section in the rewritten `USER_GUIDE.md` follows this exact structure (see `STYLE.md` for the full template):

```markdown
## §N. Tutorial X: <Tutorial Title>

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/chNN.tex`](../../gmsh_testing_ext/chapters/chNN.tex) (<Chapter Title>)
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/chNN_<slug>.ipynb`](../../gmsh_testing_ext/notebooks/chNN_<slug>.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/NN_<slug>.ipynb`](user_guide_tutorials/notebooks/NN_<slug>.ipynb)
> **Key classes:** `ClassA`, `ClassB`, `function_c()`
> **Prereqs:** `bench`, `store` (NB 00 Quick Start)

**When to use this:** one paragraph — concrete problem the reader is trying to solve.

```python
# Minimum API skeleton — 5 to 15 lines, no commentary
from harness.doe import X
result = X(store=store).run(...)
```

**Common variations:**
- `option_a=True` to enable Y — see notebook §3
- `option_b=...` to plug in custom Z — see notebook §5
- For batch use: see `harness/doe/USER_GUIDE.md §37 Common Recipes`
```

**Hard line-budget per tutorial in USER_GUIDE.md: ≤ 60 lines.** Anything longer goes to the notebook.

---

## 5. Per-Notebook Skeleton

Every notebook follows this skeleton (codified in `STYLE.md`):

```
1. Title cell — markdown
   - "# Tutorial X: <title>"
   - Conceptual deep-dive link → chapter
   - Companion guide section link
   - Prereqs (env, fixture, prior notebooks)
2. Setup cell — code
   - from _shared import load_ch04_store, ...
   - store, bench = load_ch04_store()
3. Concept recap — markdown, one paragraph
   (NOT a re-derivation of the chapter — just enough framing to ground the code)
4. Main walkthrough — alternating code+markdown
   - Run a small end-to-end example
   - Show outputs (tables, figures)
   - Vary one parameter; show the effect
5. Edge cases / failure modes — code
   - One or two predictable failures with clear error messages
6. Take-aways — markdown
   - 3–5 bullets, action-oriented
7. Next steps — markdown
   - Link to next NB + back to the chapter for theory
```

Notebooks are committed with outputs (cell counts vary, target 15–40 cells).

---

## 6. Fixture Strategy

All chapter-aligned notebooks load the read-only ch04 fixture:

```python
# _shared.py
from pathlib import Path
ROOT = Path("/home/asudjianto/jupyterlab/kg-memory")
CH04_STORE = ROOT / "gmsh_testing_ext/lab_stores/ch04_ground_truth"
CALIBRATION = ROOT / "gmsh_testing_ext/lab_stores/calibration_base"

def load_ch04_store():
    """Load the canonical ground-truth store. Read-only. Do not mutate."""
    # ... loader implementation
```

**Properties of the fixture:**
- 823 triples, 514 entities, 68 relations
- `cap_enabled=True`, `cap_rho_max=π/2`, `cap_use_diag=True`
- Hardened during the Ch04 monograph sweep; do not retrain.
- For calibration thresholds (τ_geo, etc.), notebooks load from `gmsh_testing_ext/lab_stores/ch18_doc_q1/calibration.json` or `calibration_base/` as the chapter-specific tutorial requires.

**No new fixture builds.** If a notebook needs a small variant (e.g., contradicting corpus for Ch18), it builds in-memory and does not persist.

---

## 7. Execution Phases

### Phase 0 — Planning artifacts (this PR)
- `PLAN.md` (this file)
- `STYLE.md` (writing + notebook conventions)
- `README.md` (entry doc)
- Directory scaffold + `_shared.py`

### Phase 1 — Reference notebook + legacy migration
- NB 00 Quick Start (template for all subsequent NBs)
- Migrate `harness/doe/tutorials/connecting_a_sut.ipynb` → NB 11
- Migrate `harness/doe/tutorials/querying_a_trained_gms.ipynb` → NB 02
- Migrate `harness/doe/tutorials/optimal_design_doe.ipynb` → NB 03
- Remove `harness/doe/tutorials/`

### Phase 2 — Foundations (NB 01, 04–07)
Ch01 Framing, Ch04 Ingestion, Ch05 Question Generation, Ch06 Geometric Judge, Ch07 Statistical Analysis.

### Phase 3 — Claim Pipeline (NB 08–10)
Ch08 Claim Decomposition, Ch09 Verifier Routing, Ch10 Verdicts & Severity.

### Phase 4 — SUT & Runtime (NB 12–15)
Ch12 Tool Gateway, Ch13 Policy Engine, Ch14 Environment Simulation, Ch15 Oracle Framework.

### Phase 5 — Behavioral & Multi-Source (NB 16–18)
Ch16 Hallucination, Ch17 Multi-Turn, Ch18 Multi-Agent.

### Phase 6 — Production Lifecycle + Appendices (NB 19–22)
Ch19 Release Gates, App G Calibration, App F Diagnostics, App C Failure Codes.

### Phase 7 — Operational annex (NB 23)
Combined operational-features notebook (parallel, perf, cost, dashboard, paraphrase, streaming, browser, artifact versioning).

### Phase 8 — Guide rewrite
Rewrite `harness/doe/USER_GUIDE.md` to the new structure. Old guide content → `harness/doe/USER_GUIDE.legacy.md` for one release, then deleted.

### Phase 9 — CI validation
`scripts/run_all_notebooks.py` and `scripts/validate_outputs.py`. Full pass: all 24 notebooks PASS end-to-end on GB10.

---

## 8. Style Anchors (see STYLE.md for details)

- American English, no Oxford comma. (Memory: `feedback_monograph_style`.)
- Teach concept before API. (Memory: `feedback_monograph_teaching_style`.)
- Never frame as regulatory compliance unless the section is the regulatory-mapping section. (Memory: `feedback_sr_11_7_rescinded`.)
- No "bank-grade." (Memory: `feedback_no_bank_grade`.)
- No mock-LLM fallback. Raise `AnthropicAPIKeyError` on missing key. (Memory: `feedback_no_mock_fallback`.)
- Each tutorial in USER_GUIDE.md ≤ 60 lines. Heavy lifting goes in the notebook.

---

## 9. Open Questions

None at draft time. To be revisited per phase if anything surfaces.
