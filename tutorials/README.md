# knowlytix-gmsh tutorials

Jupyter notebooks accompanying the Geometric Memory Systems books.
Two tracks are available; both run against the same
`knowlytix-harness` wheel (no separate install).

## Tracks

* `agentic_testing_book/notebooks/` — **agentic testing book**. 22
  chapters + appendix F covering DOE-driven black-box agentic
  testing: oracles, generators, hallucination detection,
  calibration, multi-agent consensus, cross-document conflict,
  governance + audit operations. Pre-built lab stores at
  `agentic_testing_book/lab_stores/` so most chapters load without
  an LLM key; ch05/ch06 still need a provider key (see each
  chapter's Prerequisites cell).
* **Governance USER_GUIDE companions** — 27 short chapters
  (`01_prerequisites.ipynb` through `27_*.ipynb`) that accompany
  `knowlytix.harness.governance`'s `USER_GUIDE.md`. They ship inside
  the `knowlytix-harness` wheel at
  `knowlytix/harness/governance/tutorials/notebooks/` (visible in
  the mirror clone at the same path). Fixtures (`ds_practices_store`,
  `california_housing_store`) are built on first use via
  `python -m knowlytix.harness.governance.tutorials.notebooks.build_fixture`
  — about 10 min on GPU, then cached at
  `~/.knowlytix/governance_fixtures/`.

## Setup

```bash
pip install knowlytix-harness jupyterlab
jupyter lab
```

Open any `agentic_testing_book/notebooks/ch*.ipynb` or
`agentic_testing_book/notebooks/appF*.ipynb` for the agentic testing
book track, or any
`knowlytix/harness/governance/tutorials/notebooks/*.ipynb` for the
governance track. Each notebook is self-contained — pick the
chapter that matches what you want to learn rather than reading
in sequence.

> **Note**: notebooks ship with cell outputs cleared. Run them
> locally to populate outputs against your own environment.
