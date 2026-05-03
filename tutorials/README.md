# knowlytix-gmsh tutorials

Jupyter notebooks accompanying the Geometric Memory Systems books. Three
tracks are available; all run against the same `knowlytix-harness`
wheel (no separate install).

## Tracks

* `notebooks/` — **GMS-Harness book** (testing track). 22 chapters +
  appendix F covering DOE-driven black-box agentic testing: oracles,
  generators, hallucination detection, calibration, multi-agent
  consensus, cross-document conflict, governance + audit operations.
  Pre-built lab stores at `lab_stores/` so most chapters load without
  an LLM key; ch05/ch06 still need a provider key (see each chapter's
  Prerequisites cell).
* `agent-builder/notebooks/` — **GMS-Agent-Builder book** (governance
  track, GMS-219). 20 chapters + appendices covering the
  `knowlytix.harness.governance` runtime layer: triple-gate tool
  gateway, claim verification, behavioral FSM contracts, governance
  bundle signing, runtime gates, drift monitoring, regulatory
  compliance walkthroughs. Workshop-critical lab stores ship at
  `agent-builder/lab_stores/`; per-chapter stores can be regenerated
  via `scripts/regenerate_lab_stores.py` from the monorepo (see
  `--list` for the recipe table).
* `user-guide/notebooks/` — **Governance USER_GUIDE companions**.
  27 short chapters (`01_prerequisites.ipynb` through
  `27_*.ipynb`) that walk through the same material as
  `knowlytix.harness.governance`'s `USER_GUIDE.md` (shipped in the
  `knowlytix-harness` wheel). Each notebook accompanies one section
  of the prose guide. Pre-built fixture stores at
  `user-guide/fixtures/` (`ds_practices_store`,
  `california_housing_store`, `MANIFEST.json`) so chapters chain off
  without rebuilding. Read the prose first; run the notebook to see
  the same flow end-to-end.

## Setup

```bash
pip install knowlytix-harness jupyterlab
jupyter lab
```

Open any `notebooks/ch*.ipynb`, `notebooks/appF*.ipynb`, or
`agent-builder/notebooks/ch*.ipynb` to get started. Each notebook is
self-contained — pick the chapter that matches what you want to
learn rather than reading in sequence.
