# knowlytix-gmsh tutorials

Jupyter notebooks accompanying the Geometric Memory Systems books. Two
tracks are available; both run against the same `knowlytix-harness`
wheel (no separate install).

## Tracks

* `agentic_testing_book/notebooks/` — **agentic testing book**. 22
  chapters + appendix F covering DOE-driven black-box agentic
  testing: oracles, generators, hallucination detection,
  calibration, multi-agent consensus, cross-document conflict,
  governance + audit operations. Pre-built lab stores at
  `agentic_testing_book/lab_stores/` so most chapters load without an
  LLM key; ch05/ch06 still need a provider key (see each chapter's
  Prerequisites cell).
* `user_guide/notebooks/` — **Governance USER_GUIDE companions**.
  27 short chapters (`01_prerequisites.ipynb` through
  `27_*.ipynb`) that walk through the same material as
  `knowlytix.harness.governance`'s `USER_GUIDE.md` (shipped in the
  `knowlytix-harness` wheel). Each notebook accompanies one section
  of the prose guide. Pre-built fixture stores at
  `user_guide/fixtures/` (`ds_practices_store`,
  `california_housing_store`, `MANIFEST.json`) so chapters chain off
  without rebuilding. Read the prose first; run the notebook to see
  the same flow end-to-end.

## Setup

```bash
pip install knowlytix-harness jupyterlab
jupyter lab
```

Open any `agentic_testing_book/notebooks/ch*.ipynb` or
`agentic_testing_book/notebooks/appF*.ipynb` to get started on the
agentic testing book track, or any `user_guide/notebooks/*.ipynb`
for the governance track. Each notebook is self-contained — pick
the chapter that matches what you want to learn rather than
reading in sequence.

> **Note**: notebooks ship with cell outputs cleared. Run them
> locally to populate outputs against your own environment.
