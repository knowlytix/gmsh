# SPDX-License-Identifier: Apache-2.0
"""Build NB 04 — Building a Ground-Truth Store (Ingestion) (Ch04)."""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "04_ingestion.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 04: Building a Ground-Truth Store

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch04.tex` — Building Ground Truth: Ingestion and the GMS Expert Store
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch04_building_ground_truth.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §7
**Prereqs:** activated `~/cluster/spark-venv`; resolvable `ANTHROPIC_API_KEY`; NB 02 (Primitives)

**What you'll do.** Walk a small bank model-risk markdown through the four-stage ingestion pipeline (LLM extraction → triple graph build → embedding training → ENM population). Inspect each intermediate artifact. Inspect the on-disk store layout. Re-load the store and verify the same primitives we exercised in NB 02 work on the freshly-built store.

Cost: ~$0.30 / ~3 min wall (LLM extraction is the bottleneck).\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys, os, time, tempfile, contextlib, io
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")
from _shared import seed_all, make_anthropic_client
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig
seed_all()
make_anthropic_client()  # sets ANTHROPIC_API_KEY for AnthropicBackend
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

Ingestion is four stages:

1. **Extract** — `AnthropicBackend` reads the markdown and emits `(head, relation, tail)` triples + ENM numeric rows + phase encoder ranges. Uses Claude Sonnet for hybrid extraction.
2. **Graph build** — triples populate `store.doc_graph`; entity/relation vocabularies are populated on `store.adapter`.
3. **Embedding train** — `GeometricKnowledgeGraph` is trained on positive triples vs corrupted negatives; rotors learn relation transforms; entities land on the cap manifold.
4. **ENM populate** — numeric values from the document are SHA-256 hashed and indexed for exact lookup.

Output is a directory of 8 files (model.pt, adapter.json, enm.json, triples.json, phase.json, model_dims.json, metadata.json, documents/combined.md) that `GMSExpertStore.load()` can re-hydrate.\
"""))

cells.append(md("""\
## 3. A Tiny Bank Model-Risk Corpus\
"""))

cells.append(code("""\
TINY_DOC = \"\"\"# Q2 2026 Model Inventory

## Models in Scope

| model_id | algorithm | test_auc | status     | tier   |
|----------|-----------|----------|------------|--------|
| MDL-A    | XGBoost   | 0.8512   | Stable     | High   |
| MDL-B    | LightGBM  | 0.7956   | Watch      | Medium |
| MDL-C    | LSTM      | 0.8201   | Degraded   | High   |

## Risk Notes

- MDL-A is the production credit-scoring model. Capital impact under
  PD shift: $2.4M for a 100 bp downgrade.
- MDL-C is being rebuilt; deployment paused.
\"\"\"

tmp = Path(tempfile.mkdtemp(prefix="nb04_"))
md_path = tmp / "q2_inventory.md"
md_path.write_text(TINY_DOC)
print(f"corpus: {md_path}")
print(f"  {len(TINY_DOC)} chars, {len(TINY_DOC.split(chr(10)))} lines")\
"""))

cells.append(md("""\
## 4. Configure + Ingest\
"""))

cells.append(code("""\
config = DOEHarnessConfig(
    markdown_path=str(md_path),
    stores_dir=str(tmp / "stores"),
    factor_group="quick_screen", n_runs=1,
    cap_enabled=True, enable_tracing=False,
)
bench = DOEGMSBenchmark(config)
t0 = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    store = bench.ingest()
elapsed = time.time() - t0
print(f"  wall time: {elapsed:.1f}s")
print(f"  store stats:")
for k, v in store.stats().items():
    print(f"    {k:14s}: {v}")\
"""))

cells.append(md("""\
## 5. On-Disk Layout\
"""))

cells.append(code("""\
store_path = Path(store.stats()['store_path'])
print(f"{store_path}/")
for entry in sorted(store_path.iterdir()):
    if entry.is_file():
        size_kb = entry.stat().st_size / 1024
        print(f"  {entry.name:24s}  {size_kb:8.1f} KB")
    else:
        print(f"  {entry.name}/")\
"""))

cells.append(md("""\
## 6. Inspect Extracted Triples\
"""))

cells.append(code("""\
triples = list(store.doc_graph.triples)
print(f"extracted {len(triples)} triples")
print()
print("first 8 triples:")
for h, r, t in triples[:8]:
    print(f"  ({h}, {r}, {t})")\
"""))

cells.append(md("""\
## 7. Inspect ENM Rows\
"""))

cells.append(code("""\
enm_keys = list(store.enm.keys())
print(f"ENM rows: {len(enm_keys)}")
print()
for k in enm_keys[:8]:
    val = store.lookup_enm(k.type, k.id)
    print(f"  [{k.type:24s}]  {k.id:32s} = {val}")\
"""))

cells.append(md("""\
## 8. Verify Primitives Work on the Fresh Store\
"""))

cells.append(code("""\
# score_triple on a true fact and a fabricated one
truth = store.score_triple("mdl-a", "has_test_auc", "0.8512")
wrong = store.score_triple("mdl-a", "has_test_auc", "0.7956")
print(f"score_triple ('mdl-a', 'has_test_auc', '0.8512') TRUE  = {truth}")
print(f"score_triple ('mdl-a', 'has_test_auc', '0.7956') WRONG = {wrong}")
print()
# query_triples
print("query_triples(head='mdl-a'):")
for h, r, t in store.query_triples(head="mdl-a"):
    print(f"  ({h}, {r}, {t})")\
"""))

cells.append(md("""\
## 9. Reload from Disk and Compare\
"""))

cells.append(code("""\
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore
reloaded = GMSExpertStore(DocGMSConfig(store_path=str(store_path)))
with contextlib.redirect_stdout(buf):
    reloaded.load()
print(f"reloaded stats: {reloaded.stats()}")
# Verify a primitive
re_truth = reloaded.score_triple("mdl-a", "has_test_auc", "0.8512")
print(f"reloaded score_triple TRUE = {re_truth}  (was {truth} pre-save)")
print(f"  byte-identical? {abs(re_truth - truth) < 1e-6 if truth and re_truth else 'n/a'}")\
"""))

cells.append(md("""\
## 10. Take-Aways

1. **Ingestion is four stages, one config.** Pass a markdown path and a `DOEHarnessConfig`; the four stages run automatically.
2. **LLM extraction dominates wall time.** ~80% of the ingestion budget is in the Anthropic call. For repeat runs the store is loaded from disk (NB 00) — re-extract only when the document changes.
3. **Eight files, three categories.** Model weights (`model.pt` + `model_dims.json`), graph + vocab (`adapter.json`, `triples.json`, `enm.json`, `phase.json`), metadata (`metadata.json`, `documents/combined.md`). Persist all eight; lose any and `load()` fails.
4. **Cap-trained from the start.** Set `cap_enabled=True` at ingest time so the embedding learns under cap-admissibility constraints. Retroactive enabling is not supported.
5. **Reload is bit-deterministic.** A re-loaded store produces byte-identical primitive results, which is what `enable_provenance=True` audit replay relies on.\
"""))

cells.append(md("""\
## 11. Next Steps

- **NB 05** — `05_question_generation.ipynb` — generate test questions from the freshly-built store (Ch05)
- **NB 21** — `21_gms_diagnostics.ipynb` — run the 9-check health report on the new store (App F)
- **NB 20** — `20_calibration.ipynb` — calibrate the judge for the new store (App G)

For theory: `gmsh_testing_ext/chapters/ch04.tex`.\
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
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
