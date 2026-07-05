# SPDX-License-Identifier: Apache-2.0
"""Build NB 18 — Multi-Agent Consensus and Cross-Document Consistency (Ch18)."""
from __future__ import annotations
import time
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "18_multi_agent.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 18: Multi-Agent Consensus and Cross-Document Consistency

**Conceptual deep-dive:** `gmsh_testing_ext/chapters/ch18.tex` — Multi-Agent Consensus and Cross-Document Consistency
**Monograph notebook:** `gmsh_testing_ext/notebooks/ch18_multi_agent.ipynb`
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §21
**Prereqs:** NB 02 (Primitives), NB 14 (Environment Sim), NB 15 (Oracles)

**What you'll do.** Load three bank model-risk documents that disagree about specific facts (`ch18_doc_q1`, `q2`, `q3` — three quarterly reports for the same model inventory) into three GMS stores. Use `CrossDocumentAnalyzer` to find shared entities and detect conflicts between the documents. Use `ConsensusOracle.evaluate_consensus(entity)` to ask "what do all three sources agree about for MDL-001?" This is the multi-source verification surface a bank deployment needs when its agent reads model documentation from multiple business lines.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")
from _shared import seed_all
from knowlytix.harness.testing import ConsensusOracle, CrossDocumentAnalyzer
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore
seed_all()
print("imports OK")\
"""))

cells.append(md("""\
## 2. Concept

The harness models cross-document consistency as **multi-store geometry**: each document becomes its own `GMSExpertStore`; an entity that appears in multiple stores becomes a multi-source claim about the same real-world thing.

- `CrossDocumentAnalyzer.find_shared_entities()` lists entities present in 2+ stores
- `.find_conflicts()` flags entities whose attribute values disagree across stores (different `tension_energy` between corresponding embeddings)
- `ConsensusOracle.evaluate_consensus(entity)` returns an `InteractionResult` with the consensus verdict for that entity across all loaded stores

`ch18_doc_q1`, `ch18_doc_q2`, `ch18_doc_q3` are three quarterly model-risk reports that overlap on MDL-001 through MDL-005 but disagree on some metrics (statuses, AUCs) — a realistic stand-in for cross-business-line reporting drift.\
"""))

cells.append(md("""\
## 3. Load the Three Quarterly Stores\
"""))

cells.append(code("""\
import contextlib, io
buf = io.StringIO()

stores = {}
for tag in ["q1", "q2", "q3"]:
    path = f"gmsh_testing_ext/lab_stores/ch18_doc_{tag}"
    s = GMSExpertStore(DocGMSConfig(store_path=path))
    with contextlib.redirect_stdout(buf):
        s.load()
    stores[tag] = s
    stats = s.stats()
    print(f"  {tag}: {stats['entities']} entities, {stats['relations']} relations, "
          f"{stats['triples']} triples, {stats['enm_entries']} ENM")\
"""))

cells.append(md("""\
## 4. Find Shared Entities Across Documents\
"""))

cells.append(code("""\
analyzer = CrossDocumentAnalyzer(stores=stores)
shared = analyzer.find_shared_entities()
print(f"Total shared entities across q1/q2/q3: {len(shared)}")
print()
sample = list(shared)[:10] if not isinstance(shared, dict) else list(shared.items())[:10]
print("First 10 shared entities:")
for item in sample:
    print(f"  {item}")\
"""))

cells.append(md("""\
## 5. Detect Cross-Document Conflicts\
"""))

cells.append(code("""\
conflicts = analyzer.find_conflicts()
print(f"Detected conflicts: {len(conflicts)}")
print()
for c in conflicts[:10]:
    if hasattr(c, "entity"):
        print(f"  entity={c.entity} tension={getattr(c, 'tension_score', 'n/a')} "
              f"stores={getattr(c, 'stores', None)}")
    else:
        print(f"  {c}")\
"""))

cells.append(md("""\
## 6. Consensus on MDL-001 Across All Three Stores\
"""))

cells.append(code("""\
oracle = ConsensusOracle(stores=stores)
for entity in ["mdl-001", "mdl-002", "mdl-005"]:
    result = oracle.evaluate_consensus(entity)
    fields = [a for a in dir(result) if not a.startswith("_") and not callable(getattr(result, a))]
    print(f"=== consensus on {entity} ===")
    for f in fields[:6]:
        val = getattr(result, f)
        print(f"  {f:18s}: {str(val)[:80]}")
    print()\
"""))

cells.append(md("""\
## 7. Visualization — Conflict Density per Entity\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
from collections import Counter

if conflicts:
    entity_counts = Counter(getattr(c, "entity", str(c)[:40]) for c in conflicts)
    top = entity_counts.most_common(10)
    if top:
        names, counts = zip(*top)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(names)), counts, color="#c14a4a")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("# cross-document conflicts")
        ax.set_title("Top entities by cross-document conflict count")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
else:
    print("(no conflicts detected — stores agree)")\
"""))

cells.append(md("""\
## 8. Take-Aways

1. **One entity, N stores, one consensus question.** The multi-document picture is "do the sources agree about MDL-001's risk rating?" — a question the harness can answer via geometric tension across stores.
2. **Conflict detection is mechanical.** `find_conflicts()` runs `tension_energy` between corresponding embeddings; high tension = the two stores disagree about the entity. No human review required to flag the candidates.
3. **`ConsensusOracle` is the release-gate hook.** Wire it into a `CompositeOracle` (NB 15) for multi-source release decisions: passing the bench corpus is not enough; the agent's claims must also be consistent with all attested sources.
4. **Stores must share entity vocabulary.** If `q1` says "MDL-001" and `q2` says "MDL_001" (typo), the analyzer will treat them as separate entities. Normalize at ingest (see `feedback_ingest_canonicalize_entities`).
5. **Pair with `CrossStoreCalibration`.** When the stores were ingested at different times with different calibration artifacts, train a cross-store calibrator first (the resolved blocker in project memory `project_cross_store_calibration_blocker`).\
"""))

cells.append(md("""\
## 9. Next Steps

- **NB 19** — `19_release_gates.ipynb` — wire the consensus oracle into the release-gate audit (Ch19)

For theory: `gmsh_testing_ext/chapters/ch18.tex`.\
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
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
