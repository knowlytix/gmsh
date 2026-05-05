# Governed Agent — User Guide Tutorials

Companion notebooks for `kg-memory/USER_GUIDE.md`. One notebook per guide section that has runnable code; each notebook is self-contained, GPU-tested, and chains off a small set of shared fixtures.

> Status: ✅ complete. `kg-memory/USER_GUIDE.md` is assembled from 31 sections; all 27 tutorial notebooks execute end-to-end on GPU GB10 against the hybrid-mode fixture (500 ingest epochs). Last full validation: 2026-05-01, 27/27 PASS, ~36 min wall. See [PLAN.md](PLAN.md) for the per-phase log.

## Layout

```
user_guide_tutorials/
├── PLAN.md                  master plan (sections, notebooks, agent dispatch)
├── STYLE.md                 writing + notebook conventions
├── draft_sections/          per-section drafts assembled into USER_GUIDE.md
├── fixtures/                prebuilt GMS stores (built once, reused by notebooks)
├── notebooks/
│   ├── _shared.py           shared imports, plot helpers, fixture loaders
│   ├── _helpers_phase1.py   per-phase helper utilities (foundations)
│   ├── _helpers_phase4a.py  per-phase helper utilities (gateway + contracts)
│   ├── _helpers_phase4c.py  per-phase helper utilities (orchestrator + multi-agent)
│   ├── _helpers_phase5c.py  per-phase helper utilities (compliance)
│   └── NN_*.ipynb           27 tutorial notebooks in guide-section order
└── scripts/                 fixture builders, notebook execution, validation
```

## Getting started

```bash
source ~/cluster/spark-venv/bin/activate
cd ~/jupyterlab/kg-memory
python user_guide_tutorials/scripts/build_fixture.py        # one-time, ~10 min on GPU
jupyter lab user_guide_tutorials/notebooks/                 # then open 01_prerequisites.ipynb
```

## Runtime expectations

- Python 3.12 + PyTorch 2.10 on CUDA 13.0 (GB10 GPU)
- Each notebook runs end-to-end in under 10 minutes
- `ANTHROPIC_API_KEY` required for notebooks 06 (hybrid ingest) and 15 (real `ConstrainedLLM`); they raise loudly when absent

## Notebook index

See [PLAN.md §4](PLAN.md#4-notebook-specifications) for per-notebook specs.

## Cross-references

- `kg-memory/USER_GUIDE.md` — canonical written guide
- `kg-memory/gms_agent_builder_ext/` — long-form build monograph (deeper math, runbooks)
- `kg-memory/gmsh_testing_ext/` — long-form testing monograph
- `kg-memory/harness/doe/USER_GUIDE.md` — DOE / GMSH testing surface
