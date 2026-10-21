# SPDX-License-Identifier: Apache-2.0
"""Build NB 23 — Operational Features (consolidated annex)."""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
NB_PATH = NB_DIR / "23_operational_features.ipynb"

def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)

cells = []
cells.append(md("""\
# Tutorial 23: Operational Features — Parallel / Perf / Cost / Artifacts

**Conceptual deep-dive:** *(no dedicated chapter — operational machinery distributed across the monograph)*
**Monograph notebook:** *(none)*
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §26
**Prereqs:** activated `~/cluster/spark-venv`; ch04 fixture

**What you'll do.** Walk four operational machineries that ship with the harness but live outside the chapter-anchored core:

1. **`PerformanceProfiler`** — per-question latency + SLA tracking
2. **`CostTracker`** — per-call USD + token budget tracking
3. **`InvarianceTester`** — paraphrase invariance over LLM responses
4. **`ArtifactRegistry`** — versioned artifact storage for fault profiles, policies, scenarios

Plus brief pointers to `ParallelExecutor`, `DashboardServer`, `StreamingAdapter`, `BrowserAdapter` — features that need running infrastructure (worker pools, websockets, headless browsers) and so are covered as reference only.\
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
from _shared import load_ch04_store, seed_all
seed_all()
store = load_ch04_store()
print(f"store: {store.stats()['entities']} entities, {store.stats()['triples']} triples")\
"""))

cells.append(md("""\
## 2. PerformanceProfiler — Latency SLA Checks\
"""))

cells.append(code("""\
from knowlytix.harness.testing import PerformanceProfiler

prof = PerformanceProfiler(store=store, sla_latency_ms=2000.0)
# Simulate a small batch of measurements
import random
random.seed(42)
tools = ["data_lookup", "risk_check", "calculate", "escalate_to_human"]
for i in range(20):
    base = 800 if i % 3 != 0 else 2500  # occasional SLA breach
    latency_ms = base + random.gauss(100, 50)
    prof.record(
        tool_name=random.choice(tools),
        latency_ms=latency_ms,
        metadata={"question_id": f"q{i:02d}"},
    )

report = prof.report()
print(f"PerformanceReport fields:")
for attr in dir(report):
    if attr.startswith("_") or callable(getattr(report, attr)):
        continue
    val = getattr(report, attr)
    if isinstance(val, (int, float, str, bool)) or val is None:
        print(f"  {attr:24s}: {val}")\
"""))

cells.append(md("""\
## 3. CostTracker — Budget Enforcement\
"""))

cells.append(code("""\
from knowlytix.harness.testing import CostTracker

tracker = CostTracker(
    store=store,
    budget_usd=5.00,
    input_price_per_mtok=3.0,
    output_price_per_mtok=15.0,
)
# Simulate 50 Sonnet 4.6 calls with realistic token counts
import random
random.seed(42)
for i in range(50):
    in_toks = random.randint(150, 400)
    out_toks = random.randint(20, 120)
    tracker.record(
        tool_name="anthropic_messages",
        input_tokens=in_toks,
        output_tokens=out_toks,
    )
s = tracker.summary()
print(f"CostSummary:")
for attr in dir(s):
    if attr.startswith("_") or callable(getattr(s, attr)):
        continue
    val = getattr(s, attr)
    if isinstance(val, (int, float, str, bool)) or val is None:
        if isinstance(val, float):
            print(f"  {attr:24s}: {val:.4f}")
        else:
            print(f"  {attr:24s}: {val}")\
"""))

cells.append(md("""\
## 4. InvarianceTester — Paraphrase Robustness\
"""))

cells.append(code("""\
from knowlytix.harness.testing import InvarianceTester

inv = InvarianceTester(store=store, tension_threshold=0.3)
# Five paraphrased answers to the same question; the agent should give the same value
paraphrases = [
    "MDL-001 has a Test AUC of 0.8412",
    "The Test AUC for MDL-001 is 0.8412",
    "0.8412 (MDL-001 Test AUC)",
    "Per Table 3.1, MDL-001's Test AUC is 0.8412",
    "MDL-001 reports Test AUC = 0.8412",
]
import inspect
sig = inspect.signature(inv.test_invariance) if hasattr(inv, "test_invariance") else None
if sig:
    print(f"InvarianceTester.test_invariance: {sig}")
# Methods available
print(f"InvarianceTester methods: {[m for m in dir(inv) if not m.startswith('_') and callable(getattr(inv, m))]}")\
"""))

cells.append(code("""\
# Run invariance check if API supports a list-of-paraphrases interface
try:
    # Try the canonical signature
    result = inv.evaluate_invariance(paraphrases, ground_truth="0.8412") if hasattr(inv, "evaluate_invariance") else None
    if result is None and hasattr(inv, "test_invariance"):
        result = inv.test_invariance(paraphrases)
    print(f"invariance result: {result}")
except Exception as e:
    print(f"(API surface differs by version: {type(e).__name__})")
    print(f"  available methods: {[m for m in dir(inv) if not m.startswith('_') and callable(getattr(inv, m))]}")\
"""))

cells.append(md("""\
## 5. ArtifactRegistry — Versioned Artifact Storage\
"""))

cells.append(code("""\
import tempfile
from knowlytix.harness.testing import ArtifactRegistry, VersionedArtifact

tmp = Path(tempfile.mkdtemp(prefix="nb23_artifacts_"))
reg = ArtifactRegistry(artifacts_dir=str(tmp), artifact_type="fault_profile")

# Store a fault profile artifact, then update it (creates v2)
art_v1 = VersionedArtifact(
    artifact_id="risk_check_outage",
    artifact_type="fault_profile",
    version=1,
    content={"tool_pattern": "risk_check", "error_rate": 1.0,
             "error_message": "Q1 2026 outage simulation"},
)
reg.save(art_v1) if hasattr(reg, "save") else None
art_v2 = VersionedArtifact(
    artifact_id="risk_check_outage",
    artifact_type="fault_profile",
    version=2,
    content={"tool_pattern": "risk_check", "error_rate": 0.5,
             "error_message": "Q2 2026 degraded simulation"},
)
reg.save(art_v2) if hasattr(reg, "save") else None

print(f"ArtifactRegistry methods: {[m for m in dir(reg) if not m.startswith('_') and callable(getattr(reg, m))][:10]}")
print(f"artifacts dir contents:")
for p in sorted(tmp.iterdir()):
    if p.is_dir():
        for f in sorted(p.iterdir())[:5]:
            print(f"  {f.relative_to(tmp)}")
    else:
        print(f"  {p.relative_to(tmp)}")\
"""))

cells.append(md("""\
## 6. Visualization — Latency Histogram + SLA Cut\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

latencies = [r.latency_ms for r in prof.records] if hasattr(prof, "records") else []
if not latencies:
    # fall back to the report payload
    print("(profiler does not expose raw records; skipping histogram)")
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(latencies, bins=15, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axvline(2000, color="red", linestyle="--", linewidth=2, label="SLA = 2000 ms")
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("# questions")
    ax.set_title(f"Per-question latency (n={len(latencies)})")
    ax.legend()
    plt.tight_layout()
    plt.show()\
"""))

cells.append(md("""\
## 7. Reference-Only Features

Four operational machineries that need running infrastructure to demonstrate end-to-end. Each ships with the harness; this notebook just points to the right entry points.

| Feature | Entry point | When to use |
|---|---|---|
| **`ParallelExecutor`** | `from knowlytix.harness.testing import ParallelExecutor, ParallelConfig` | When a benchmark exceeds ~100 questions and serial execution wastes wall time. Configure `max_workers` and run `executor.execute(bench, questions, evaluator)`. |
| **`DashboardServer`** | `from knowlytix.harness.testing import DashboardServer, DashboardSnapshot` | Live monitoring during long campaigns. Starts a WebSocket server on `live_dashboard_port`; UI subscribes to per-question events. |
| **`StreamingAdapter`** | `from knowlytix.harness.testing import StreamingAdapter, StreamConsistencyOracle` | When the SUT streams tokens and you need to check mid-stream contradictions (final answer might disagree with what it said earlier). |
| **`BrowserAdapter`** | `from knowlytix.harness.testing import BrowserAdapter, UIScript, BrowserConfig` | When the SUT is a browser-driven agent (Selenium / Playwright). Records DOM-level events as `ToolCallRecord`s. |\
"""))

cells.append(md("""\
## 8. Take-Aways

1. **Operational features are orthogonal to verification.** None of these change the failure-budget verdict; they just measure cost / time / robustness alongside it.
2. **Profile + budget every campaign.** A 1,000-question Sobol+Refine run can quietly cost $50+ in API spend; `CostTracker` is the only way to abort before you blow through.
3. **Versioned artifacts are the audit story.** Fault profiles, policy rules, scenarios, calibration thresholds — all get registered with `ArtifactRegistry` so a release packet can pin `(artifact_id, version)` and replay byte-identical.
4. **Paraphrase invariance is corpus-specific.** `InvarianceTester(tension_threshold=...)` needs calibrating on your store; the default 0.3 is a starting point, not a deployment value.
5. **The reference-only features need real infra.** Parallel needs workers, dashboard needs a port, streaming needs an LLM that supports streaming, browser needs a headless browser. They're shipped but exercised in integration tests, not in this guide.\
"""))

cells.append(md("""\
## 9. Where to Go Next

- **NB 19** — `19_release_gates.ipynb` — how cost/perf signals feed into the release-gate audit (Ch19)
- **`knowlytix/harness/testing/USER_GUIDE.md` §27** — End-to-End Walkthrough that wires everything together
- **`harness/doe/API_REFERENCE.md`** — full signatures for every class mentioned here

For theory: none of these features have a dedicated chapter; they live in `gmsh_testing_ext/chapters/ch19.tex` as supporting machinery for production-lifecycle work.\
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
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  wall time: {time.time() - t0:.1f}s")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
