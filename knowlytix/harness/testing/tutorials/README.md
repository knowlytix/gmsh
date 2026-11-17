# GMSH Testing Harness — User Guide Tutorials

Companion notebooks for [`knowlytix/harness/testing/USER_GUIDE.md`](../USER_GUIDE.md). Each notebook pairs 1:1 with a guide tutorial section and a monograph chapter or appendix in [`gmsh_testing_ext/`](../../../../../gmsh_testing_ext/).

> **Status (2026-05-11):** ✅ Complete. All 24 notebooks (NB 00 + NB 01–23) execute end-to-end on the bank model-risk corpus with real Sonnet 4.6 where applicable. Paired with the rewritten [`knowlytix/harness/testing/USER_GUIDE.md`](../USER_GUIDE.md) via the three-artifact triad. See [PLAN.md](PLAN.md) §7 for the per-phase log.

## How the three artifacts work together

| Artifact | Role |
|---|---|
| `gmsh_testing_ext/chapters/chNN.tex` | **Theory** — derivations, theorems, motivation |
| `knowlytix/harness/testing/USER_GUIDE.md` §N | **Bridge** — thin recipe, when-to-use, minimum API skeleton |
| `user_guide_tutorials/notebooks/NN_*.ipynb` | **Practice** — runnable end-to-end example with outputs |

Each tutorial in the guide carries a *Conceptual deep-dive* link to the chapter and a *Notebook* link to the notebook. Each notebook's title cell links back to both.

## Layout

```
user_guide_tutorials/
├── PLAN.md             master plan (sections, notebooks, phases)
├── STYLE.md            writing + notebook conventions
├── README.md           this file
├── draft_sections/     per-section drafts before merge into USER_GUIDE.md
├── notebooks/
│   ├── _shared.py      fixture loaders, common imports, plot helpers
│   ├── 00_quick_start.ipynb
│   └── NN_*.ipynb      24 tutorial notebooks (chapter-aligned + appendix + ops annex)
└── scripts/
    ├── run_all_notebooks.py    CI: nbconvert --execute on every NB
    └── validate_outputs.py     post-execution assertions per NB
```

## Getting started

```bash
cd <repo-root>
jupyter lab knowlytix/harness/testing/user_guide_tutorials/notebooks/
# Open 00_quick_start.ipynb
```

The notebooks load the hardened ground-truth store at `gmsh_testing_ext/lab_stores/ch04_ground_truth/` read-only — no fixture build required.

## Runtime expectations

- Python 3.12 + PyTorch 2.10 on CUDA 13.0 (GPU GB10)
- Each notebook runs end-to-end in under 10 minutes
- `ANTHROPIC_API_KEY` required for notebooks that exercise real `ConstrainedLLM` (raises `AnthropicAPIKeyError` when absent — no mock fallback)

## Notebook index

| NB | Title | Monograph | Guide § |
|----|-------|-----------|---------|
| 00 | Quick Start | preface | §3 |
| 01 | Framing the Testing Problem | Ch01 | §4 |
| 02 | GMS as Ground Truth | Ch02 | §5 |
| 03 | Designing Test Experiments (DOE) | Ch03 | §6 |
| 04 | Building a Ground-Truth Store | Ch04 | §7 |
| 05 | Generating Questions from Geometry | Ch05 | §8 |
| 06 | The Geometric Judge | Ch06 | §9 |
| 07 | Statistical Analysis of Factor Effects | Ch07 | §10 |
| 08 | Decomposing Output into Claims | Ch08 | §11 |
| 09 | Routing Claims to Verifiers | Ch09 | §12 |
| 10 | Verdicts, Failure Codes, and Severity | Ch10 | §13 |
| 11 | Connecting Your SUT | Ch11 | §14 |
| 12 | Tool Gateway and Fault Injection | Ch12 | §15 |
| 13 | Policy Engine | Ch13 | §16 |
| 14 | Environment Simulation | Ch14 | §17 |
| 15 | Oracle Framework | Ch15 | §18 |
| 16 | Hallucination Detection | Ch16 | §19 |
| 17 | Stateful and Multi-Turn Testing | Ch17 | §20 |
| 18 | Multi-Agent & Cross-Document Consistency | Ch18 | §21 |
| 19 | Campaigns, Release Gates, Provenance | Ch19 | §22 |
| 20 | Confidence Calibration (ECE / MCE) | App G | §23 |
| 21 | GMS Training & Diagnostics | App F | §24 |
| 22 | Failure Code Taxonomy | App C | §25 |
| 23 | Operational Features (parallel, perf, cost, dashboard, …) | — | §26 |

## Cross-references

- [`knowlytix/harness/testing/USER_GUIDE.md`](../USER_GUIDE.md) — canonical written guide
- [`knowlytix/harness/testing/API_GUIDE.md`](../API_GUIDE.md) — full API signatures
- [`gmsh_testing_ext/`](../../../../../gmsh_testing_ext/) — long-form testing monograph (theory)
- [`knowlytix/harness/governance/USER_GUIDE.md`](../../governance/USER_GUIDE.md) — separate guide for the build side
- [`knowlytix/harness/governance/tutorials/notebooks/`](../../governance/tutorials/notebooks/) — the *build-side* user guide's notebooks (do not confuse with this one)
