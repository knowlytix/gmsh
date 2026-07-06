# SPDX-License-Identifier: Apache-2.0
"""Build NB 20b — Calibration with a Local LLM (Qwen2.5-7B-Instruct + vLLM)
— from cell definitions, then execute.

Companion to NB 20. Same calibration story; the LLM under test is swapped
from Claude Opus (API, sequential, ~$0.30) to Qwen2.5-7B-Instruct (local,
vLLM batched, free). The point is pedagogical: how do you wire a local
model into the DOE benchmark and into the `bench.judge.calibrate()` flow?

Wall clock (warm HF cache): ~30 s for vLLM weight load, ~30-60 s for
batched generation of the 78 questions, ~90 s for the calibration walks.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/tutorials/scripts/_build_nb_20b.py
"""
from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[5]
NB_DIR = ROOT / "knowlytix" / "harness" / "testing" / "tutorials" / "notebooks"
NB_PATH = NB_DIR / "20b_local_llm.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)


cells = []

cells.append(md("""\
# Tutorial 20b: Calibration with a Local LLM — Qwen2.5-7B-Instruct + vLLM

**Companion to:** `20_calibration.ipynb` (same story, Claude Opus over the Anthropic API)
**Conceptual deep-dive:** `gmsh_testing_ext/appendix_g.tex` — Calibration Toolkit
**Companion guide section:** `knowlytix/harness/testing/USER_GUIDE.md` §23
**Prereqs:** activated `~/cluster/spark-venv` with `vllm` and `transformers` installed; CUDA-capable GPU with ≥18 GB free; ch04 fixture; NB 20 read first

**What you'll do.** Re-run the same calibration story as NB 20, but swap the LLM under test from Claude Opus 4 to a local **Qwen2.5-7B-Instruct** served by **vLLM**. You'll learn:

1. How to write a `LocalLLMEvaluator` that conforms to the same `(question, context) -> str` interface as `LLMEvaluator`.
2. How to batch all 78 questions into one vLLM call instead of paying the per-question API latency 78 times.
3. How `bench.judge.calibrate()` is **LLM-agnostic** — the triple-level classifiers are trained on the GMS store, not on LLM responses, so the calibration story holds regardless of which model you point at the benchmark.
4. Where the bare-LLM accuracy / confidence story diverges between a frontier API model and a 7B local — and why the *audit pipeline* calibration is portable across both.

**Wall-clock budget.** Warm HF cache: vLLM weight-load is ~30 s, batched generation is ~30-60 s, calibration (§9-§10) is ~90 s. Total: **~3 min** vs. ~6 min for NB 20.\
"""))

cells.append(md("""\
## 1. Setup\
"""))

cells.append(code("""\
import contextlib, io, os, sys, time
from pathlib import Path

# vLLM forks its engine-core subprocess by default, which fails once any
# import has touched CUDA in the parent (knowlytix.harness.testing -> knowlytix.core -> torch.cuda
# does exactly that). 'spawn' makes the child do a clean CUDA init.
# MUST be set before `import vllm`.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, str(Path.cwd()))
os.chdir(str(ROOT))
sys.path.insert(0, ".")

from _shared import load_ch04_store, seed_all
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig,
    CalibrationTester, CalibrationReport,
)
from knowlytix.harness.testing.benchmark import _get_context, _build_section_map
from knowlytix.benchmark.eval.llm_caller import SYSTEM_PROMPT
import numpy as np
seed_all()

# Hard prereq: vLLM. No silent fallback — if vLLM isn't installed, the
# notebook should fail loudly here rather than later inside model load.
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
print(f"vllm        : {vllm.__version__}")
print(f"multiproc   : {os.environ['VLLM_WORKER_MULTIPROC_METHOD']}")
print(f"imports OK")\
"""))

cells.append(md("""\
## 2. Concept

**Why a local model?** Two reasons:

1. **Cost / iteration speed.** Each NB 20 run costs ~$0.30 and ~5 min wall time. For pedagogical iteration on the *calibration machinery* (not the LLM under test), you don't need a frontier model — you need *some* model whose answers you can route through the judge. A 7-8 B local model gives you that at $0 marginal cost.
2. **Batched throughput.** The Anthropic API is rate-limited per-request and the DOE orchestrator calls it sequentially (78 single-question calls). A local vLLM server handles all 78 prompts in one batched forward pass via **continuous batching** — different prompts at different decode positions sharing the same GPU step.

**Why Qwen2.5-7B-Instruct?** Strong factual-QA accuracy at the 7 B class, native chat template via `tokenizer.apply_chat_template`, terse instruction-following without a chain-of-thought preamble to parse around — exactly what we want when feeding answers into a regex-based ground-truth comparator.

**What does NOT change vs NB 20?** `bench.judge.calibrate()` and the triple-level cross-validation walk in §10 are **LLM-agnostic** — they train on the GMS store's positive triples and corrupted negatives, never touching LLM output. So the response-level calibration in §8 will look different (Qwen has different accuracy/confidence behavior than Opus), but the triple-level calibration in §10 will look essentially identical to NB 20 — the geodesic / tension / holonomy classifiers train on the same store.\
"""))

cells.append(md("""\
## 3. Build the Benchmark First (claim the GMS store's GPU memory)

On unified-memory devices like the GB10 (128 GB shared CPU/GPU), the order matters: vLLM's `gpu_memory_utilization` flag tells it to reserve a fraction of *free* memory at init, and once vLLM grabs that block, anything that wants GPU memory later (the GMS store's embedding matrices, the cap-on model) will OOM. So we load the GMS store first; vLLM then reserves whatever is left in §4.

Identical `DOEHarnessConfig` to NB 20 §3 except `target_model` — we point it at the Qwen2.5-7B-Instruct id so the recorded run metadata reflects what was actually evaluated.\
"""))

cells.append(code("""\
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

config = DOEHarnessConfig(
    markdown_path="gmsh_testing_ext/lab_stores/ch04_ground_truth/documents/combined.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen",
    n_runs=2,
    max_per_category=3,
    cap_enabled=True,
    enable_typed_verdicts=True,
    enable_tracing=False,
    target_type="llm",
    target_model=MODEL_ID,
)
bench = DOEGMSBenchmark(config)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    bench.ingest()
    bench.generate_questions()
print(f"questions   : {len(bench._doe_questions)}")
print(f"store size  : {len(bench.store.doc_graph.triples)} triples, "
      f"{len(bench.store.adapter.entity_to_idx)} entities")

import torch
free_after_gms, total = torch.cuda.mem_get_info()
print(f"GPU         : {free_after_gms/1e9:.1f}/{total/1e9:.1f} GB free after GMS load")\
"""))

cells.append(md("""\
## 4. Load Qwen2.5-7B-Instruct with vLLM

`LLM(...)` loads weights to GPU. The model is ~14 GB in bfloat16; weight-load on this GB10 is ~30 s once HF cache is warm. We set `gpu_memory_utilization=0.4` — Qwen 7B + KV cache for the 78 prompts comfortably fits in ~50 GB on this 128 GB unified-memory device, and the lower fraction leaves room for the GMS store loaded in §3 and headroom for the calibration walks in §9-§10.\
"""))

cells.append(code("""\
t0 = time.time()
llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    gpu_memory_utilization=0.4,
    max_model_len=8192,
    enforce_eager=False,
)
print(f"vLLM weight load: {time.time() - t0:.1f}s")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.0,   # deterministic for reproducible reliability diagrams
    top_p=1.0,
)
print(f"tokenizer       : {tokenizer.__class__.__name__}")
print(f"sampling        : temp=0.0, max_tokens=512")\
"""))

cells.append(md("""\
## 5. The `LocalLLMEvaluator` Class

The DOE benchmark calls `evaluator(question, context) -> str` once per question, sequentially. To get the batched win we need a two-phase pattern:

1. **`prefetch(pairs)`** — submit *all* `(question, context)` pairs to vLLM in one call. Cache the answers keyed by `(question, context)`.
2. **`__call__(q, ctx)`** — fast lookup from cache. Falls back to single-stream generation if a key is missing (so the evaluator still works as a normal `LLMEvaluator`-style callable).

The system prompt is the same `SYSTEM_PROMPT` used for the Opus path (from `knowlytix/benchmark/eval/llm_caller.py`) so the calibration comparison stays apples-to-apples.

**Chat template.** `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)` is the Qwen2.5-Instruct standard — it inserts the system + user roles and appends the assistant-turn prompt, returning a plain string that vLLM tokenizes internally.\
"""))

cells.append(code("""\
class LocalLLMEvaluator:
    \"\"\"vLLM-backed evaluator with batched-prefetch optimization.

    Same interface as :class:`knowlytix.harness.testing.LLMEvaluator` so it drops into
    :meth:`DOEGMSBenchmark.run` without changes.
    \"\"\"

    def __init__(self, llm, tokenizer, sampling_params,
                 system_prompt: str = SYSTEM_PROMPT):
        self.llm = llm
        self.tokenizer = tokenizer
        self.params = sampling_params
        self.system_prompt = system_prompt
        self._cache: dict[tuple[str, str], str] = {}

    def _format(self, question: str, context: str) -> str:
        \"\"\"Build a Qwen2.5-Instruct chat-template string for one (question, context) pair.\"\"\"
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content":
                f"<report>\\n{context}\\n</report>\\n\\n{question}"},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    def prefetch(self, pairs: list[tuple[str, str]]) -> float:
        \"\"\"Batch-generate all (question, context) pairs in one vLLM call.

        Returns wall-clock time for the batched generate() call.
        \"\"\"
        formatted = [self._format(q, ctx) for q, ctx in pairs]
        t0 = time.time()
        outputs = self.llm.generate(formatted, self.params, use_tqdm=False)
        elapsed = time.time() - t0
        for (q, ctx), out in zip(pairs, outputs):
            self._cache[(q, ctx)] = out.outputs[0].text.strip()
        return elapsed

    def __call__(self, question: str, context: str) -> str:
        key = (question, context)
        if key in self._cache:
            return self._cache[key]
        # Fallback: single-stream (slow path, only hits on cache miss)
        formatted = self._format(question, context)
        out = self.llm.generate([formatted], self.params, use_tqdm=False)
        ans = out[0].outputs[0].text.strip()
        self._cache[key] = ans
        return ans

    def __repr__(self) -> str:
        return (f"LocalLLMEvaluator(model={MODEL_ID!r}, "
                f"cached={len(self._cache)})")


evaluator = LocalLLMEvaluator(llm, tokenizer, sampling_params)
print(evaluator)\
"""))

cells.append(md("""\
## 6. Batched Prefetch — All 78 Questions in One vLLM Call

The DOE orchestrator's `run()` loop iterates `doe_questions` sequentially. To exploit vLLM continuous batching, we rebuild each question's context the same way `run()` does (via `_get_context`), then call `evaluator.prefetch()` once. After this cell, every subsequent `evaluator(q, ctx)` is an O(1) dict lookup.

Watch for the wall-time delta vs NB 20 §3's `bench.run()` Opus loop (~4-5 min): one batched vLLM call replaces 78 sequential API calls.\
"""))

cells.append(code("""\
# Rebuild contexts the same way bench.run() does internally.
section_map = _build_section_map(bench.markdown)
ctx_rng = np.random.default_rng(bench.config.seed)
pairs = []
for dq in bench._doe_questions:
    q = dq.question.llm_prompt
    ctx = _get_context(dq, bench.markdown, section_map,
                       bench.config.context_mode, ctx_rng)
    pairs.append((q, ctx))

print(f"prefetching {len(pairs)} (question, context) pairs in one batched vLLM call...")
prefetch_elapsed = evaluator.prefetch(pairs)
print(f"\\n  batched generate(): {prefetch_elapsed:.1f}s "
      f"({prefetch_elapsed * 1000 / len(pairs):.0f} ms/question amortized)")
print(f"  cache size         : {len(evaluator._cache)}")
print(f"  first answer sample: {list(evaluator._cache.values())[0][:160]!r}")\
"""))

cells.append(md("""\
## 7. Run the Benchmark (cache-served, no new LLM calls)

`bench.run()` iterates the questions, each call to `evaluator(q, ctx)` is now an O(1) cache hit, and the GMS judge produces typed verdicts as usual.\
"""))

cells.append(code("""\
buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    result = bench.run(evaluator=evaluator)
run_elapsed = time.time() - t0
verdicts = bench.typed_verdicts or {}
acc = result.total_correct / result.total_questions
print(f"  wall time      : {run_elapsed:.1f}s  (judging only; LLM was prefetched)")
print(f"  total questions: {result.total_questions}")
print(f"  accuracy       : {acc:.1%}")
print(f"  typed verdicts : {len(verdicts)}")\
"""))

cells.append(md("""\
## 8. Reliability Diagram — Uncalibrated Qwen2.5-7B-Instruct Judge Output

Same instrumentation as NB 20 §4. Expect a noticeably different shape from Opus: smaller 7B models on this domain are typically less accurate *and* less confident, so overconfidence may be milder but ECE need not be lower.\
"""))

cells.append(code("""\
tester_pre = CalibrationTester(n_bins=10)
for qid, v in verdicts.items():
    tester_pre.add(claimed_confidence=v.overall_confidence,
                   actual_correct=bool(v.correct),
                   question_id=qid)

report_pre = tester_pre.report()
print(f"Uncalibrated judge on Qwen2.5-7B-Instruct output:")
print(f"  n_points              : {report_pre.n_points}")
print(f"  ECE                   : {report_pre.ece:.4f}")
print(f"  MCE                   : {report_pre.mce:.4f}")
print(f"  mean_confidence       : {report_pre.mean_confidence:.4f}")
print(f"  mean_accuracy         : {report_pre.mean_accuracy:.4f}")
print(f"  overconfidence_ratio  : {report_pre.overconfidence_ratio:.4f}")\
"""))

cells.append(code("""\
import matplotlib.pyplot as plt

def plot_reliability(report: CalibrationReport, title: str, ax):
    centers, accs, counts = [], [], []
    for b in report.bins:
        if b.n_samples == 0:
            continue
        centers.append((b.bin_lower + b.bin_upper) / 2)
        accs.append(b.mean_accuracy)
        counts.append(b.n_samples)
    centers = np.array(centers); accs = np.array(accs); counts = np.array(counts)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5,
            label="perfect calibration")
    colors = ["#c14a4a" if c > a else "#3aa56b"
              for c, a in zip(centers, accs)]
    ax.bar(centers, accs, width=0.08, alpha=0.7, color=colors,
           edgecolor="black", linewidth=0.5)
    for c, a, n in zip(centers, accs, counts):
        ax.text(c, a + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
    ax.set_xlabel("confidence bin midpoint")
    ax.set_ylabel("bin accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)

fig, ax = plt.subplots(figsize=(8, 5))
plot_reliability(report_pre,
    f"Reliability — UNCALIBRATED (Qwen2.5-7B-Instruct)\\nECE={report_pre.ece:.3f}  MCE={report_pre.mce:.3f}",
    ax)
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 9. Train Calibration — `bench.judge.calibrate()`

Identical to NB 20 §5. The classifiers train on the **GMS store** (triples + corrupted negatives), not on Qwen's output — so this step is fully LLM-agnostic and you should see the same trained τ thresholds as NB 20.\
"""))

cells.append(code("""\
t0 = time.time()
classifiers = bench.judge.calibrate(seed=42)
calibrate_elapsed = time.time() - t0
print(f"\\n  calibration time: {calibrate_elapsed:.1f}s")
print()
print("Trained classifiers:")
for name, clf in classifiers.items():
    print(f"  {name:18s}: {type(clf).__name__}, classes={clf.classes_.tolist()}")
print()
print("Calibrated thresholds (now stored on bench.judge):")
for name, tau in bench.judge._thresholds.items():
    print(f"  {name:18s}: τ = {tau}")\
"""))

cells.append(md("""\
## 10. Triple-Level Held-Out Cross-Validation (Same as NB 20 §6)

The geodesic-channel cross-val from NB 20 §6, repeated here for the side-by-side reliability diagram in §11. Numbers should match NB 20 almost exactly — same store, same seed.\
"""))

cells.append(code("""\
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import random

rng = random.Random(42)
triples = list(bench.store.doc_graph.triples)
entities = list(bench.store.adapter.entity_to_idx.keys())

X_geo, y_geo = [], []
for h, r, t in triples:
    g = bench.store.score_triple(h, r, t)
    if g is None: continue
    X_geo.append([g]); y_geo.append(1)
    fake_t = rng.choice(entities)
    while fake_t == t:
        fake_t = rng.choice(entities)
    g_neg = bench.store.score_triple(h, r, fake_t)
    if g_neg is not None:
        X_geo.append([g_neg]); y_geo.append(0)

X_geo = np.array(X_geo); y_geo = np.array(y_geo)
print(f"geodesic channel: {len(y_geo)} samples, {y_geo.sum()} positive, "
      f"{len(y_geo)-y_geo.sum()} negative")

clf = LogisticRegression(max_iter=2000, random_state=42)
probas = cross_val_predict(clf, X_geo, y_geo, cv=5, method="predict_proba")
proba_pos = probas[:, 1]

tester_post = CalibrationTester(n_bins=10)
for p, y in zip(proba_pos, y_geo):
    tester_post.add(claimed_confidence=float(p), actual_correct=bool(y))
report_post = tester_post.report()

print()
print("Triple-level calibration (geodesic channel, 5-fold cross-val):")
print(f"  n_points              : {report_post.n_points}")
print(f"  ECE                   : {report_post.ece:.4f}  "
      f"(response-level was {report_pre.ece:.4f})")
print(f"  MCE                   : {report_post.mce:.4f}  "
      f"(response-level was {report_pre.mce:.4f})")
print(f"  mean_confidence       : {report_post.mean_confidence:.4f}")
print(f"  mean_accuracy         : {report_post.mean_accuracy:.4f}")
print(f"  overconfidence_ratio  : {report_post.overconfidence_ratio:.4f}")\
"""))

cells.append(md("""\
## 11. Side-by-Side: Qwen Response-Level vs. Triple-Level

Same two-panel reliability diagram as NB 20 §7. The **left** panel is *Qwen-specific* — it'll look different from NB 20's left panel because the LLM changed. The **right** panel is *LLM-agnostic* — it should match NB 20's right panel because the geodesic classifier was trained on the same store with the same seed.\
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_reliability(report_pre,
    f"Response-level  UNCALIBRATED\\n(Qwen2.5-7B overall_confidence)\\n"
    f"ECE={report_pre.ece:.3f}  MCE={report_pre.mce:.3f}",
    axes[0])
plot_reliability(report_post,
    f"Triple-level  CALIBRATED\\n(geodesic classifier, 5-fold CV)\\n"
    f"ECE={report_post.ece:.3f}  MCE={report_post.mce:.3f}",
    axes[1])
plt.tight_layout()
plt.show()\
"""))

cells.append(md("""\
## 12. Headline Comparison: Local vs API Path

Two numbers worth keeping side by side:

1. **Wall-clock end-to-end** for the LLM-under-test step. NB 20 §3 (78 sequential Opus calls): ~4-5 min. This notebook §6 (one batched Qwen vLLM call): ~30-60 s. The cost of *getting answers* drops by ~5-10× when you control the serving stack.
2. **Triple-level calibration is unchanged.** The geodesic / tension / holonomy classifiers don't care which LLM produced the answers — they're trained on the store. That's the operational point: **the audit pipeline's calibration is portable across LLMs**; only the response-level (LLM `overall_confidence`) needs re-calibration when you swap the model.\
"""))

cells.append(code("""\
print("Headline impact for the LLM-under-test swap:")
print()
print(f"  Local model               : {MODEL_ID}")
print(f"  Batched generate() (§6)   : {prefetch_elapsed:.1f}s for {len(pairs)} questions "
      f"({prefetch_elapsed*1000/len(pairs):.0f} ms/q amortized)")
print(f"  bench.run() loop (§7)     : {run_elapsed:.1f}s (cache hits + judging)")
print(f"  calibrate() (§9)          : {calibrate_elapsed:.1f}s (store walk, LLM-agnostic)")
print()
print(f"  Response-level (Qwen2.5)   ECE: {report_pre.ece:.4f}  "
      f"(overconfidence_ratio={report_pre.overconfidence_ratio:.2f})")
print(f"  Triple-level (GMS judge)   ECE: {report_post.ece:.4f}  "
      f"(overconfidence_ratio={report_post.overconfidence_ratio:.2f})")
print()
print("Trained τ thresholds (identical to NB 20 modulo store/seed):")
for name, tau in bench.judge._thresholds.items():
    print(f"  τ_{name:18s}: {tau}")\
"""))

cells.append(md("""\
## 13. Take-Aways

1. **The evaluator interface is the seam.** `(question, context) -> str` is all the DOE harness asks for. Anything that produces a string from a prompt + context plugs in: a frontier API model, a local vLLM server, a fine-tuned adapter, a tool-using agent. `LocalLLMEvaluator` in §5 is ~40 lines.
2. **Batched prefetch is the only thing that buys you the local-model speed advantage.** Single-stream local generation isn't faster than the API for short prompts — kernel-launch overhead per request dominates. Submit *all* prompts in one `llm.generate()` and vLLM's continuous batching does the rest.
3. **Triple-level calibration is portable, response-level is not.** When you swap LLMs, re-running `bench.judge.calibrate()` is a sanity check (you'll get nearly the same τ thresholds), but the *real* recalibration burden falls on response-level (LLM `overall_confidence`). The audit pipeline that NB 19's release gate enforces is LLM-agnostic.
4. **Chat templates are model-specific.** `tokenizer.apply_chat_template` handles the role-tag wrapping for whichever Qwen variant you load. If you swap to a model with a different template (Llama 3, Mistral, etc.), the same call adapts; if you swap to a model with a *thinking-mode* preamble (Qwen3, DeepSeek-R1), pass `enable_thinking=False` (or its equivalent) to keep answers terse for the regex parsers downstream.
5. **Determinism matters for reliability diagrams.** We set `temperature=0.0` so a re-run produces the same answers and the same ECE/MCE. With temperature > 0 you'd need to bootstrap multiple samples per question to get stable bin counts.

**When to prefer this path:**
- Iterating on the calibration machinery (this notebook's whole reason for existing)
- Stress-testing the judge against a *weaker* LLM (deliberate accuracy floor)
- Air-gapped or cost-sensitive environments

**When to keep the API path (NB 20):**
- The deployment target is a frontier API model — calibrate against what you actually ship
- You need response-level confidence claims from the model itself (no local equivalent)\
"""))

cells.append(md("""\
## 14. Next Steps

- **NB 19** — re-run the release gate with `bench.judge.calibrate()` called before `bench.run()`; the `not_calibrated` failures vanish whether the LLM is Opus or Qwen2.5-7B
- **NB 20** — the API-path version of this notebook (Claude Opus 4)
- **NB 21** — `21_gms_diagnostics.ipynb` — `GMSDiagnostics` and training-time vs post-training diagnostics

For the underlying serving stack:
- vLLM docs: https://docs.vllm.ai
- Qwen2.5 chat-template reference: `tokenizer.apply_chat_template(...)` documented in the Qwen2.5-Instruct model card on Hugging Face\
"""))


def build_and_execute() -> None:
    nb = new_notebook()
    nb.cells = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (spark-venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    print(f"Writing {NB_PATH} ({len(cells)} cells)...")
    NB_PATH.write_text(nbformat.writes(nb))

    print("Executing notebook end-to-end...")
    t0 = time.time()
    # 30-min timeout: weight load + 78-question batched generate + calibration walks.
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    print(f"  execution wall time: {time.time() - t0:.1f}s")

    print(f"Writing executed notebook back to {NB_PATH}...")
    nbformat.write(nb, NB_PATH)
    print("DONE.")


if __name__ == "__main__":
    build_and_execute()
