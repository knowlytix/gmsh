# GMSH Agentic Testing Harness — User Guide

A practitioner's guide to testing large-language-model agents on bank-domain ground truth with geometrically-grounded verification. Every tutorial pairs 1:1 with a chapter in `gmsh_testing_ext/` (theory) and a runnable notebook in `user_guide_tutorials/notebooks/` (practice).

> **Status (2026-05-11):** Rewritten to the three-artifact triad — every tutorial here links to its monograph chapter for theory and its companion notebook for runnable code. 24 paired notebooks ship with this guide; all execute end-to-end on the bank model-risk corpus with real Sonnet 4.6. See [`user_guide_tutorials/README.md`](user_guide_tutorials/README.md).
>
> **Addendum (2026-06-09):** §11–§13 and §23 now cover the answer-driven `GeometricClaimExtractor` (precision axis), the `CompletenessEvaluator` recall axis on `TypedVerdict`, and calibrated-threshold wiring with channel recusal. These are API additions ahead of their companion notebooks.

## The Three-Artifact Triad

Every chapter `NN` has three paired artifacts:

| Artifact | Role | Where |
|---|---|---|
| `gmsh_testing_ext/chapters/chNN.tex` | Theory: derivations, theorems, definitions | Monograph chapter |
| `gmsh_testing_ext/notebooks/chNN_*.ipynb` | Textbook lab with Learning Objectives and exercises | Monograph notebook |
| `user_guide_tutorials/notebooks/NN_*.ipynb` | Practitioner recipe on the bank corpus | User-guide notebook |

Each tutorial section below carries three backlinks so a reader can move between theory, lab walkthrough, and runnable recipe freely. The guide itself is the *bridge*: thin prose, minimum API skeleton, pointers to both ends.

## Table of Contents

**Front matter**

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Architecture Overview](#2-architecture-overview)
3. [Quick Start](#3-quick-start)

**Part I — Foundations**

4. [Tutorial 01: Framing the Testing Problem (Ch01)](#4-tutorial-01-framing-the-testing-problem)
5. [Tutorial 02: GMS as Ground Truth — Primitives in 10 Minutes (Ch02)](#5-tutorial-02-gms-as-ground-truth--primitives-in-10-minutes)
6. [Tutorial 03: Designing Test Experiments — Sobol+Refine (Ch03)](#6-tutorial-03-designing-test-experiments--sobolrefine)
7. [Tutorial 04: Building a Ground-Truth Store (Ch04)](#7-tutorial-04-building-a-ground-truth-store)
8. [Tutorial 05: Generating Questions from Geometry (Ch05)](#8-tutorial-05-generating-questions-from-geometry)
9. [Tutorial 06: The Geometric Judge — Per-Category Scoring (Ch06)](#9-tutorial-06-the-geometric-judge--per-category-scoring)
10. [Tutorial 07: Statistical Analysis of Factor Effects (Ch07)](#10-tutorial-07-statistical-analysis-of-factor-effects)

**Part II — Claim Pipeline**

11. [Tutorial 08: Decomposing Output into Claims (Ch08)](#11-tutorial-08-decomposing-output-into-claims)
12. [Tutorial 09: Routing Claims to Verifiers (Ch09)](#12-tutorial-09-routing-claims-to-verifiers)
13. [Tutorial 10: Verdicts, Failure Codes, and Severity (Ch10)](#13-tutorial-10-verdicts-failure-codes-and-severity)

**Part III — SUT and Runtime Probes**

14. [Tutorial 11: Connecting Your SUT — Adapters, Targets, Traces (Ch11)](#14-tutorial-11-connecting-your-sut--adapters-targets-traces)
15. [Tutorial 12: Tool Gateway and Fault Injection (Ch12)](#15-tutorial-12-tool-gateway-and-fault-injection)
16. [Tutorial 13: Policy Engine (Ch13)](#16-tutorial-13-policy-engine)
17. [Tutorial 14: Environment Simulation (Ch14)](#17-tutorial-14-environment-simulation)
18. [Tutorial 15: Oracle Framework (Ch15)](#18-tutorial-15-oracle-framework)

**Part IV — Behavioral and Multi-Source**

19. [Tutorial 16: Hallucination Detection (Ch16)](#19-tutorial-16-hallucination-detection)
20. [Tutorial 17: Stateful and Multi-Turn Testing (Ch17)](#20-tutorial-17-stateful-and-multi-turn-testing)
21. [Tutorial 18: Multi-Agent & Cross-Document Consistency (Ch18)](#21-tutorial-18-multi-agent--cross-document-consistency)

**Part V — Production Lifecycle**

22. [Tutorial 19: Release Gates and Audit Reports (Ch19)](#22-tutorial-19-release-gates-and-audit-reports)

**Part VI — Calibration & Diagnostics (Appendices)**

23. [Tutorial 20: Confidence Calibration — ECE / MCE (App G)](#23-tutorial-20-confidence-calibration--ece--mce)
24. [Tutorial 21: GMS Training & Post-Training Diagnostics (App F)](#24-tutorial-21-gms-training--post-training-diagnostics)
25. [Tutorial 22: Failure Code Taxonomy (App C)](#25-tutorial-22-failure-code-taxonomy)

**Part VII — Operational Annex**

26. [Operational Features — Parallel, Perf, Cost, Artifacts](#26-operational-features--parallel-perf-cost-artifacts)

**Back matter**

27. [End-to-End Walkthrough](#27-end-to-end-walkthrough)
28. [CLI Reference](#28-cli-reference)
29. [Configuration Reference](#29-configuration-reference)
30. [Troubleshooting](#30-troubleshooting)
31. [Companion Resources](#31-companion-resources)

---

## 1. Prerequisites & Installation

```bash
# Activate the project environment
source ~/cluster/spark-venv/bin/activate

# Verify the installation
python -c "from knowlytix.harness.testing import DOEGMSBenchmark; print('OK')"
```

**Environment requirements:**
- Python 3.12+ with PyTorch 2.10+ on CUDA 13.0 (GPU GB10 in the project's reference deployment)
- A resolvable `ANTHROPIC_API_KEY` (env var or `pass show anthropic/api-key`). Tutorials that exercise a real LLM raise `AnthropicAPIKeyError` when absent — there is no mock fallback.
- The hardened `ch04_ground_truth` fixture at `gmsh_testing_ext/lab_stores/ch04_ground_truth/` (ships with the monograph).

**API key resolution.** All notebooks call `_shared.make_anthropic_client()`, which delegates to `knowlytix.harness.governance.api_key.get_anthropic_api_key()`. Resolution order: `ANTHROPIC_API_KEY` env var → `pass show anthropic/api-key` → `AnthropicAPIKeyError`.

---

## 2. Architecture Overview

The harness is layered. Reading top to bottom, each layer consumes the previous and adds one diagnostic capability:

```
Production lifecycle    | Campaigns, release gates, audit packets, provenance       | Ch19 / §22
─────────────────────── + ────────────────────────────────────────────────────────  + ─────────
Behavioral & multi-src  | Hallucination, multi-turn, multi-agent, cross-document    | Ch16-18 / §19-§21
SUT runtime probes      | Adapters, gateway, policy, env-sim, oracles               | Ch11-15 / §14-§18
Claim pipeline          | ClaimCompiler → VerificationRouter → TypedVerdict         | Ch08-10 / §11-§13
DOE foundations         | Sobol+Refine design, generators, judge, statistics        | Ch03-07 / §6-§10
GMS ground truth        | Embedding manifold, ENM, six primitives                   | Ch02 / §5
Framing                 | Why surface scorers fail on agentic AI                    | Ch01 / §4
```

**Three calibration / diagnostic appendices** live alongside this layered stack:

- **App G** — Confidence calibration (ECE/MCE, reliability diagrams) — §23
- **App F** — Post-training geometric diagnostics — §24
- **App C** — Failure code taxonomy and tier escalation — §25

For a one-page architectural reference, see [`API_REFERENCE.md`](API_REFERENCE.md).

---

## 3. Quick Start

> **User-guide notebook:** [`user_guide_tutorials/notebooks/00_quick_start.ipynb`](user_guide_tutorials/notebooks/00_quick_start.ipynb)
> **Wall time:** ~7 min (78 real Sonnet 4.6 calls) · **Cost:** ~$0.50

Load the bank model-risk store, run a benchmark with a real LLM, read per-category accuracy. Three import lines, one config dataclass:

```python
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig, LLMEvaluator
from _shared import load_ch04_store, make_anthropic_client, DEFAULT_TUTORIAL_MODEL

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
    target_model=DEFAULT_TUTORIAL_MODEL,
)
bench = DOEGMSBenchmark(config)
bench.ingest()                # loads existing ch04 store; no retraining
bench.generate_questions()
evaluator = LLMEvaluator(client=make_anthropic_client(), model=DEFAULT_TUTORIAL_MODEL)
result = bench.run(evaluator=evaluator)
print(f"accuracy: {result.total_correct}/{result.total_questions}")
```

Expected output: ~50% accuracy across 78 questions, with per-category spread from 100% (holonomy, adversarial) to 0% (boundary, conditional). The notebook walks the result inspection and visualization.

**Where this lands you in the layered architecture:** front door (DOE foundations + a real evaluator). Every tutorial that follows extends one piece of this five-line flow.

---

## Part I — Foundations

## 4. Tutorial 01: Framing the Testing Problem

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch01.tex`](../../gmsh_testing_ext/chapters/ch01.tex) — The Agentic AI Testing Problem
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch01_agentic_testing_problem.ipynb`](../../gmsh_testing_ext/notebooks/ch01_agentic_testing_problem.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/01_framing.ipynb`](user_guide_tutorials/notebooks/01_framing.ipynb)
> **Key classes:** `store.score_triple`, `store.lookup_enm`
> **Prereqs:** None (entry point)

**When to use this.** Before writing any test scaffold, when you want to justify why the harness's geometric verification matters versus exact-match or BLEU-style scoring on agentic outputs.

```python
from _shared import load_ch04_store
store = load_ch04_store()
# Geodesic distance — the "did the agent really get it right?" signal
print(store.score_triple("mdl-001", "has_test_auc", "0.8412"))   # ~0.06 — truth
print(store.score_triple("mdl-001", "has_test_auc", "0.9234"))   # ~1.5  — wrong
```

**Common variations.**
- Replace `score_triple` with `lookup_enm` for exact numeric verification — see NB 01 §3.
- Compare against BLEU-1 / exact-match scorers on the same answers — NB 01 §5.

**Edge cases.** A near-miss like `"LSTM (3-layer)"` against ground truth `"LSTM"` exposes the scoring boundary: exact-match=False (substring mismatch), word_overlap=1.0 (word matches), gms_geodesic=NaN (parsed value not in vocab). See NB 01 §6.

## 5. Tutorial 02: GMS as Ground Truth — Primitives in 10 Minutes

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch02.tex`](../../gmsh_testing_ext/chapters/ch02.tex) — Geometric Memory Systems as Ground Truth
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch02_gms_ground_truth.ipynb`](../../gmsh_testing_ext/notebooks/ch02_gms_ground_truth.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/02_gms_primitives.ipynb`](user_guide_tutorials/notebooks/02_gms_primitives.ipynb)
> **Key classes:** `GMSExpertStore`
> **Prereqs:** NB 00 Quick Start

**When to use this.** When you need to exercise the six geometric primitives that every verifier in the harness reduces to: `score_triple`, `query_triples`, `lookup_enm`, `link_predict`, `tension_energy`, `check_holonomy`.

```python
from _shared import load_ch04_store
store = load_ch04_store()
# Six primitives, one contract
store.score_triple("mdl-001", "has_test_auc", "0.8412")     # plausibility on the cap manifold
list(store.query_triples(head="mdl-001"))                    # structural pattern lookup
store.lookup_enm("performance_metrics", "MDL-001/Test AUC")  # SHA-256-verified exact value
store.link_predict("mdl-001", "has_test_auc", top_k=5)      # type-constrained top-k
store.tension_energy("mdl-001", "mdl-004")                  # entity-level disagreement
store.check_holonomy(["has_train_auc"], "has_test_auc")     # multi-hop transport defect
```

**Common variations.**
- Build a fresh tiny store from inline markdown (NB 02 §3) and verify the same primitives work.
- For numeric audits use `lookup_enm` (digit-for-digit, SHA-256-verified); for plausibility use `score_triple` (geodesic).

**Edge cases.** `score_triple` returns `None` if any of head/relation/tail is missing from the vocabulary — verifiers treat that as *unscoreable*, not *wrong*.

## 6. Tutorial 03: Designing Test Experiments — Sobol+Refine

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch03.tex`](../../gmsh_testing_ext/chapters/ch03.tex) — Design of Experiments for Systematic Evaluation
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch03_design_of_experiments.ipynb`](../../gmsh_testing_ext/notebooks/ch03_design_of_experiments.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/03_doe_design.ipynb`](user_guide_tutorials/notebooks/03_doe_design.ipynb)
> **Key classes:** `graphdoe.design`, `DesignMatrix`, `DOEAnalyzer`, `QuestionRephraser`
> **Prereqs:** NB 02 (Primitives)

**When to use this.** When you have a fixed LLM-call budget and need a design matrix that gets the most information per call — for screening factors, building factor-effect reports, or replicating the FinStructBench paper-style flow.

```python
import graphdoe
from knowlytix.harness.testing.bridge import generate_base_questions

base = generate_base_questions(store=store,
    generator_names=["exact_retrieval", "counting", "cross_reference"],
    max_per_category=10, seed=42)
design = graphdoe.design(
    questions=base, factor_names=CORE_FACTORS, n_runs=60,
    method="sobol+refine", seed=42,
)
df = design.matrix  # 60 rows; the question column already picks one base question per row
```

**Common variations.**
- Compare `method="sobol"` vs `"sobol+refine"` and read the `quality()` deltas (`phi_p`, `min_dist`, `coverage`) — NB 03 §3.
- Scale to paper settings: `n_runs=880`, 20 contextual factors via `factor_group="comprehensive"`.
- Pair with `DOEAnalyzer.run_logistic` + `apply_fdr_correction` + `run_joint_model` for full factor-effect analysis — NB 03 §7-§8, §10.

**Edge cases.** Joint logistic at `n_obs < 10 × n_params` is under-determined; you'll see `inf` pseudo-R² or singular-matrix errors. Either scale `n_runs` or drop low-information factors. Use FDR correction whenever you screen ≥ 5 factors.

## 7. Tutorial 04: Building a Ground-Truth Store

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch04.tex`](../../gmsh_testing_ext/chapters/ch04.tex) — Building Ground Truth: Ingestion and the GMS Expert Store
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch04_building_ground_truth.ipynb`](../../gmsh_testing_ext/notebooks/ch04_building_ground_truth.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/04_ingestion.ipynb`](user_guide_tutorials/notebooks/04_ingestion.ipynb)
> **Key classes:** `DOEGMSBenchmark.ingest`, `DocGMSConfig`, `GMSExpertStore`
> **Prereqs:** NB 02 (Primitives)

**When to use this.** When the bank document you want to test against is *not* the shipped `ch04_ground_truth` fixture, and you need to build a fresh store from your own markdown corpus.

```python
config = DOEHarnessConfig(
    markdown_path="path/to/your/inventory.md",
    stores_dir="my_stores",
    cap_enabled=True,             # always set at ingest, not retroactively
    enable_tracing=False,
)
bench = DOEGMSBenchmark(config)
store = bench.ingest()            # LLM extraction → graph → embedding → ENM (~20s on a small corpus)
print(store.stats())
```

**Common variations.**
- Pin `ingest_mode="regex"` or `"hybrid"` (default) via `DOEHarnessConfig` for cheaper extraction on structured tables — NB 04 §4.
- Override loss/cap/train weights via `loss_config_path="path/to/loss.yaml"` — useful for corpus-specific tuning.

**Edge cases.** Ingestion is idempotent: if a store already exists at `stores_dir/store_name`, `bench.ingest()` loads it instead of retraining. Delete the directory to force re-ingest. A reload from disk produces byte-identical primitive results — verified in NB 04 §9.

## 8. Tutorial 05: Generating Questions from Geometry

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch05.tex`](../../gmsh_testing_ext/chapters/ch05.tex) — Generating Test Questions from Geometric Structure
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch05_question_generation.ipynb`](../../gmsh_testing_ext/notebooks/ch05_question_generation.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/05_question_generation.ipynb`](user_guide_tutorials/notebooks/05_question_generation.ipynb)
> **Key classes:** `bridge.generate_base_questions`, `GeneratedQuestion`
> **Prereqs:** NB 02 (Primitives), NB 04 (Ingestion)

**When to use this.** When you want to generate test questions directly from the store's structure — graph-verified by construction (100% correct ground truth, no human annotation needed) — across 13 generator families.

```python
from knowlytix.harness.testing.bridge import generate_base_questions
questions = generate_base_questions(store=store,
    generator_names=["plausibility", "tension", "boundary", "adversarial",
                     "exact_retrieval", "counting"],
    max_per_category=10, seed=42)
for q in questions[:3]:
    print(f"[{q.category}] {q.natural_language}  -> gt={q.ground_truth}")
```

**Common variations.**
- Use the 8 geometric generators (`plausibility`, `tension`, `holonomy`, `link_prediction`, `multi_hop_transport`, `phase_comparison`, `conditional`, `procedural`) for primitive-anchored testing.
- Use the 5 taxonomy generators (`boundary`, `adversarial`, `robustness`, `recovery_test`, `governance`) for SUT-behavior testing.

**Edge cases.** Always verify the graph baseline at 100% before paying for an LLM run. If `q.graph_answer_fn(store)` doesn't equal `q.ground_truth` for every question, the ground truth is broken and the LLM evaluation will measure nothing. NB 05 §6 walks the check.

## 9. Tutorial 06: The Geometric Judge — Per-Category Scoring

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch06.tex`](../../gmsh_testing_ext/chapters/ch06.tex) — The Geometric Judge: Per-Category Scoring
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch06_geometric_judge.ipynb`](../../gmsh_testing_ext/notebooks/ch06_geometric_judge.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/06_geometric_judge.ipynb`](user_guide_tutorials/notebooks/06_geometric_judge.ipynb)
> **Key classes:** `GMSJudge`, `GeometricVerdict`
> **Prereqs:** NB 02 (Primitives), NB 09 (Verifier Routing), NB 20 (Calibration)

**When to use this.** When you need to score a free-text LLM answer with the full five-channel geometric profile (`geodesic_confidence`, `tension_score`, `holonomy_score`, `phase_score`, `overall_confidence`) rather than a binary correct/incorrect.

```python
verdict = bench.judge.judge(
    llm_answer="MDL-001 has a Test AUC of 0.8412. ANSWER: 0.8412",
    ground_truth="0.8412",
    exact_correct=True, exact_detail="exact match",
)
print(f"geodesic={verdict.geodesic_confidence}  overall={verdict.overall_confidence}")
```

**Common variations.**
- `judge_typed(...)` returns a `TypedVerdict` with full per-claim sub-verdict trail — NB 10.
- After `bench.judge.calibrate()` (NB 20), `verdict.calibration_score` carries the calibrated probability instead of `None`.

**Edge cases.** Pre-calibration the `overall_confidence` is a hand-tuned blend, *not* a probability. Use as relative ranking only. A high `holonomy_score` with low `geodesic_confidence` says "internally consistent reasoning over wrong facts" — fabrication signature.

## 10. Tutorial 07: Statistical Analysis of Factor Effects

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch07.tex`](../../gmsh_testing_ext/chapters/ch07.tex) — Statistical Analysis of Factor Effects
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch07_statistical_analysis.ipynb`](../../gmsh_testing_ext/notebooks/ch07_statistical_analysis.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/07_statistical_analysis.ipynb`](user_guide_tutorials/notebooks/07_statistical_analysis.ipynb)
> **Key classes:** `DOEAnalyzer.run_logistic`, `apply_fdr_correction`, `run_joint_model`
> **Prereqs:** NB 03 (DOE Design)

**When to use this.** When you've run a Sobol+Refine sweep and need to extract which contextual factors significantly affect correctness — with Benjamini-Hochberg FDR control and a joint-model goodness-of-fit verdict.

```python
from graphdoe.analysis import DOEAnalyzer
ana = DOEAnalyzer.from_dataframe(results_df, factors=FACTOR_LIST)
dev = ana.run_logistic(metric="correct")
dev_adj = ana.apply_fdr_correction(dev, alpha=0.05, method="fdr_bh")
joint = ana.run_joint_model(metric="correct")  # needs n_obs >> n_params
print(f"joint pseudo_r2={joint['pseudo_r2']:.3f}  roc_auc={joint['roc_auc']:.3f}")
```

**Common variations.**
- `apply_fdr_correction` accepts `method="fdr_by"` (Benjamini-Yekutieli) for arbitrary-dependence FDR; default `"fdr_bh"` assumes independence.
- For two-way interactions, set `include_interactions=True` on `run_logistic` (warning: factorial blow-up).

**Edge cases.** Joint model requires `n_obs ≥ 10 × n_params` as a rule of thumb. At `n_runs=60` with 11 factors (~22 parameters) the joint fit may report `inf` pseudo-R² or fail with a singular matrix. Scale `n_runs` to ≥ 300, or drop low-information factors after the single-factor screen.

---

## Part II — Claim Pipeline

## 11. Tutorial 08: Decomposing Output into Claims

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch08.tex`](../../gmsh_testing_ext/chapters/ch08.tex) — Claims: Decomposing Agent Output into Verifiable Units
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch08_claims.ipynb`](../../gmsh_testing_ext/notebooks/ch08_claims.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/08_claim_decomposition.ipynb`](user_guide_tutorials/notebooks/08_claim_decomposition.ipynb)
> **Key classes:** `ClaimCompiler`, `ClaimBase`, `FactClaim`, `ThresholdClaim`, `ArtifactClaim`, `ClaimExtractor`, `RuleBasedClaimExtractor`, `GeometricClaimExtractor`
> **Prereqs:** NB 02 (Primitives)

**When to use this.** When you have a free-text LLM answer and need to extract structured claims (typed triples + provenance) for downstream verification.

```python
from knowlytix.harness.testing import ClaimCompiler
compiler = ClaimCompiler(store=store, semantic_mode="off")
claims = compiler.compile(
    llm_answer="MDL-001 has Test AUC 0.8412 and is currently Degraded.",
    question_text="What is MDL-001's status?", run_id="r1", question_id="q1",
)
for c in claims:
    print(f"[{c.claim_type}] {c.raw_text}  conf={c.confidence}")
```

**Common variations.**
- `semantic_mode="on"` catches paraphrased claims that the literal extractor misses; trades extraction confidence for coverage.
- Compound LLM answers (e.g., "MDL-001 has AUC 0.84 and is Degraded under high tier") fan out to 3–5 claims; each routes independently in NB 09.

**Edge cases.** `claim_type` drives downstream routing — get it wrong and the wrong verifier runs. Subclasses: `FactClaim` (head, relation, tail), `ThresholdClaim` (metric, op, value), `RuleClaim` (logical predicate), `ArtifactClaim` (artifact_id, hash), `ToolUseClaim` (tool, args), `SemanticClaim`.

**Choosing a claim extractor — rule-based vs answer-driven (the precision axis).** `ClaimCompiler` delegates extraction to a pluggable `ClaimExtractor`. The default `RuleBasedClaimExtractor` takes `head`/`relation` from the generator's schema and only scrapes the *tail* from the answer, so it can confirm only the single fact the question was built from. `GeometricClaimExtractor` is **answer-driven**: it reuses the RAG `GeometricQueryParser`'s bound vocabulary to find *every* entity the answer names, then emits a `FactClaim` for each `(head, relation, tail)` the answer actually asserts over the store's graph adjacency. Every claim is real store vocabulary, so `GraphVerifier.score_triple` yields a meaningful geodesic on each — this is the **precision (correctness)** axis: of the claims the answer makes, how many are grounded? Pair it with `CompletenessEvaluator` (the recall axis, §13) for the full picture.

```python
from knowlytix.harness.testing import GMSJudge, ClaimCompiler, GeometricClaimExtractor

# Select it on the judge (drives judge_typed end to end):
judge = GMSJudge(store=store, claim_extraction="geometric")   # vs the default "rule"

# …or wire the extractor directly into a standalone compiler:
compiler = ClaimCompiler(store=store, extractor=GeometricClaimExtractor(store))
```

Entity matching is tolerant of `snake_case` vs spaced forms; numeric tails match on value-equality (`35.0 == "35"`). Schema-driven numeric/threshold checks still run via the compiler's supplementary passes, so the geometric extractor *composes* with the rule-based checks rather than replacing them.

## 12. Tutorial 09: Routing Claims to Verifiers

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch09.tex`](../../gmsh_testing_ext/chapters/ch09.tex) — Verifiers: Routing Claims to GMS Primitives
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch09_verifiers.ipynb`](../../gmsh_testing_ext/notebooks/ch09_verifiers.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/09_verifier_routing.ipynb`](user_guide_tutorials/notebooks/09_verifier_routing.ipynb)
> **Key classes:** `VerificationRouter`, `SubVerdict`, `GraphVerifier`, `NumericVerifier`, `ArtifactVerifier`, `LogicVerifier`, `PolicyVerifier`
> **Prereqs:** NB 08 (Claims)

**When to use this.** When you have a list of `ClaimBase` from `ClaimCompiler` and need to dispatch each to its matching verifier and collect `SubVerdict`s.

```python
from knowlytix.harness.testing import VerificationRouter
router = VerificationRouter(store=store)
subverdicts = router.route_all(claims)
for sv in subverdicts:
    status = "PASS" if sv.passed else "FAIL"
    print(f"[{status}] {sv.verifier_name:18s} method={sv.gms_method_used} score={sv.raw_score}")
```

**Common variations.**
- Plug in custom verifiers via the `semantic_verifier`, `process_evaluator`, `recovery_evaluator`, `interaction_evaluator` kwargs.
- Pass `judge=calibrated_judge` to copy the judge's fitted per-channel cuts onto the verifiers whose decision axis matches: geodesic → `GraphVerifier.plausibility_threshold`, holonomy → `LogicVerifier.config.tau_path`, tension → `LogicVerifier.config.tau_ent`/`tau_contra`, policy geodesic → `PolicyVerifier.plausibility_threshold`. `GMSJudge.calibrate()` does this for you — it rebuilds the router with `judge=self` so the freshly-fit thresholds actually reach the verifiers (the router built at construction predates calibration and captured empty thresholds).

**Edge cases.** Before calibration (NB 20), `GraphVerifier` returns "GraphVerifier has no calibrated plausibility_threshold; run GMSJudge.calibrate() first and wire the geodesic key through" — that's the `not_calibrated` failure surfaced repeatedly through the audit pipeline. ArtifactVerifier is always available (SHA-256 doesn't need calibration). **After** calibration, a geometric channel the store *cannot* calibrate — no separable cohort for holonomy or tension — **recuses** itself: it returns a neutral sub-verdict carrying no failure attribution rather than a `not_calibrated` failure, letting the calibrated channels decide the answer while the dead channel abstains. An *asserted* multi-hop path is still confirmed via `query_triples` regardless of calibration (it holds by graph construction); recusal only applies when the channel can neither confirm nor refute.

## 13. Tutorial 10: Verdicts, Failure Codes, and Severity

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch10.tex`](../../gmsh_testing_ext/chapters/ch10.tex) — Verdicts, Failures, and Severity Classification
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch10_verdicts_failures.ipynb`](../../gmsh_testing_ext/notebooks/ch10_verdicts_failures.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/10_verdicts_severity.ipynb`](user_guide_tutorials/notebooks/10_verdicts_severity.ipynb)
> **Key classes:** `TypedVerdict`, `FailureAttribution`, `SeverityClassifier`, `SeverityLevel`, `CompletenessEvaluator`, `CompletenessReport`
> **Prereqs:** NB 08, NB 09

**When to use this.** When you need the audit-shape per-question verdict (`TypedVerdict`) that `AuditReporter` consumes downstream — `correct`, `n_claims`, `n_failed`, `sub_verdicts`, `failures`, geometric scores.

```python
verdict = bench.judge.judge_typed(
    llm_answer=raw_answer, ground_truth="0.8412",
    exact_correct=True, exact_detail="exact match",
    question_category="plausibility", run_id="r1", question_id="q1",
)
print(f"correct={verdict.correct}  n_failed={verdict.n_failed}  "
      f"failures={[f.code.value for f in verdict.failures]}")
```

**Common variations.**
- `SeverityClassifier.classify_all(failures, risk_tier="action_taking")` to assign per-tier severity to a failure list.
- Aggregate verdicts by category, code, or severity for failure-report tables.
- `verdict.completeness` carries the **recall** axis — `{expected, covered, missing, score}` — orthogonal to the correctness axis above. `judge_typed` populates it from `CompletenessEvaluator.evaluate(...)`, which keeps only the question's *store-grounded* ground-truth atoms (known entities + asked numeric values; margins, booleans, and schema bookkeeping are dropped so they don't inflate the denominator) and checks coverage with the same store-bound matching the `GeometricClaimExtractor` uses. The two axes diverge by design: a confident, fully-correct-but-*partial* answer scores high on correctness and low on completeness. `score == 1.0` when nothing was expected (vacuously complete); the dict is empty `{}` when completeness wasn't computed.

**Edge cases.** The aggregation rule is `TypedVerdict.correct = exact_correct` — sub-verdict failures populate `n_failed` and the `failures` list but do not flip `correct` to False. Gating happens at the release-gate layer (NB 19), not at the verdict layer. Empirically verified in NB 10 §5: 13/13 verdicts have `correct == exact_correct`.

---

## Part III — SUT and Runtime Probes

## 14. Tutorial 11: Connecting Your SUT — Adapters, Targets, Traces

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch11.tex`](../../gmsh_testing_ext/chapters/ch11.tex) — SUT Abstraction: Adapters, Targets, and Traces
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch11_sut_abstraction.ipynb`](../../gmsh_testing_ext/notebooks/ch11_sut_abstraction.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/11_sut_adapters.ipynb`](user_guide_tutorials/notebooks/11_sut_adapters.ipynb)
> **Key classes:** `SUTAdapter`, `LLMAdapter`, `AgentAdapter`, `RAGAdapter`, `TargetProfile`, `IsolationRequirements`, `SUTTrace`, `ToolCallRecord`
> **Prereqs:** NB 00 (Quick Start)

**When to use this.** When wiring a real System Under Test — agent, RAG pipeline, browser-driven UI, or bare LLM — through the harness's uniform `evaluate(question, context) -> SUTTrace` contract.

```python
from knowlytix.harness.testing import LLMAdapter, SUTTrace, TargetProfile, IsolationRequirements

class ClaudeBankAdapter(LLMAdapter):
    def __init__(self, client, model): self.client, self.model = client, model
    def evaluate(self, question, context) -> SUTTrace:
        r = self.client.messages.create(model=self.model, max_tokens=256,
            messages=[{"role": "user", "content": f"{context}\n\n{question}"}])
        return SUTTrace(raw_output=r.content[0].text, latency_ms=r.usage and 0.0,
                        input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)

target = TargetProfile(name="MyAgent", version="1.0.0", target_type="agent",
                       isolation=IsolationRequirements(network_isolated=True))
```

**Common variations.**
- Four integration levels: Level 1 passive observation, Level 2 gateway intercept (NB 12), Level 3 full simulation (NB 14), Level 4 real-Claude capstone (NB 11 §9).
- For tool-using agents, subclass `AgentAdapter` and populate `tool_calls` with `ToolCallRecord` entries.

**Edge cases.** `MockHarness` (in `demo.mock_agent`) is the canonical test fixture for exercising the gateway and policy engine deterministically. It's *not* a mock-LLM fallback — the harness raises loudly on missing API key by design.

## 15. Tutorial 12: Tool Gateway and Fault Injection

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch12.tex`](../../gmsh_testing_ext/chapters/ch12.tex) — The Tool Gateway: Fault Injection and Interception
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch12_tool_gateway.ipynb`](../../gmsh_testing_ext/notebooks/ch12_tool_gateway.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/12_tool_gateway.ipynb`](user_guide_tutorials/notebooks/12_tool_gateway.ipynb)
> **Key classes:** `ToolGateway`, `FaultProfile`
> **Prereqs:** NB 11 (SUT Adapters)

**When to use this.** When testing how a SUT recovers from tool faults (latency, errors, mocked responses) without modifying the underlying tool implementations.

```python
from knowlytix.harness.testing import ToolGateway, FaultProfile
gateway = ToolGateway(store=store, fault_profiles=[
    FaultProfile(tool_pattern="risk_check", error_rate=1.0, error_message="Simulated outage"),
    FaultProfile(tool_pattern="data_lookup", latency_ms=500.0),
])
adapter = AgentAdapter(harness, gateway=gateway)
trace = adapter.evaluate(question, context)
print(gateway.get_transcript().decisions)  # allow / inject_fault / block per call
```

**Common variations.**
- Glob patterns: `tool_pattern="risk_*"` matches any tool starting with `risk_`.
- Schedule a fault to a time window with `start_time` / `end_time` (paired with `ClockControl` from `EnvironmentSimulator`).
- Load fault profiles from YAML via `config.gateway_fault_profiles_path`.

**Edge cases.** Fault injection runs *after* policy checks (NB 13). If the policy engine blocks a call, no fault is exercised — useful for testing the policy without confounding from fault injection.

## 16. Tutorial 13: Policy Engine

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch13.tex`](../../gmsh_testing_ext/chapters/ch13.tex) — The Policy Engine: Runtime Rule Enforcement
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch13_policy_engine.ipynb`](../../gmsh_testing_ext/notebooks/ch13_policy_engine.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/13_policy_engine.ipynb`](user_guide_tutorials/notebooks/13_policy_engine.ipynb)
> **Key classes:** `PolicyEngine`, `PolicyRule`, `PolicyDecision`
> **Prereqs:** NB 12 (Tool Gateway)

**When to use this.** When you need runtime rule enforcement on tool calls: forbid destructive actions, require preconditions, or whitelist specific actions.

```python
from knowlytix.harness.testing import PolicyEngine, PolicyRule
rules = [
    PolicyRule(rule_id="no_delete", action="delete_records", rule_type="blocks"),
    PolicyRule(rule_id="risk_before_escalate", action="escalate_to_human",
               requirement="risk_check", rule_type="requires"),
    PolicyRule(rule_id="allow_lookup", action="data_lookup", rule_type="allows"),
]
engine = PolicyEngine(store=store, rules=rules)
decision = engine.check(action="delete_records", args={"id": "MDL-001"}, context="")
print(f"allowed={decision.allowed}  reason={decision.reason}")
```

**Common variations.**
- Valid `rule_type` values: `allows`, `blocks`, `requires`. Other strings fall through to the GMS-graph layer.
- YAML-load rules in production via `PolicyEngine.load_rules_from_yaml(path)`.
- Wire into the gateway via `ToolGateway(policy_engine=engine, ...)` so blocked calls never reach the underlying tool.

**Edge cases.** `check()` runs four layers: explicit-rule match, GMS-graph lookup, geometric plausibility (`score_triple`), contradiction detection (`tension_energy`). Even with no explicit rule, the GMS-graph layer may produce surprising verdicts on actions that have no support in the store. Use `engine.check(...)` to test before wiring through the gateway.

## 17. Tutorial 14: Environment Simulation

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch14.tex`](../../gmsh_testing_ext/chapters/ch14.tex) — Environment Simulation: Deterministic World-State
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch14_environment_simulation.ipynb`](../../gmsh_testing_ext/notebooks/ch14_environment_simulation.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/14_environment_sim.ipynb`](user_guide_tutorials/notebooks/14_environment_sim.ipynb)
> **Key classes:** `EnvironmentSimulator`, `ToolEndpointSpec`, `StateTransition`, `ClockControl`, `DataFixture`
> **Prereqs:** NB 11 (SUT Adapters), NB 12 (Gateway)

**When to use this.** When you need to replace real tool implementations entirely with deterministic synthetic endpoints — for audit-grade replay, side-effect oracles, or sandboxed isolation.

```python
from knowlytix.harness.testing import EnvironmentSimulator, ToolEndpointSpec, StateTransition, ClockControl
env = EnvironmentSimulator(tool_specs=[
    ToolEndpointSpec(tool_name="data_lookup", initial_state={"loaded": 0},
        default_response='{"status": "not_found"}',
        state_transitions=[StateTransition(trigger_action="data_lookup",
            trigger_condition="args.entity == 'MDL-001'",
            state_mutation={"loaded": 1},
            response_template='{"auc": 0.8412}')]),
], clock=ClockControl(mode="frozen", frozen_at="2026-04-01T09:00:00Z"))
resp = env.handle_tool_call("data_lookup", {"entity": "MDL-001"})
diff = env.get_state_diff(before, env.get_state())
```

**Common variations.**
- `ClockControl(mode="frozen", ...)` for byte-identical replay; `mode="real"` or `"ticking"` for forward simulation.
- `DataFixture(fixture_id="...", version=1, data={...})` for versioned data backing; the content_hash is auto-computed.

**Edge cases.** `trigger_condition` strings are informational; the matcher does not always enforce them strictly. State mutations always fire when a transition matches. For strict matching, validate inputs in the adapter before calling `handle_tool_call`.

## 18. Tutorial 15: Oracle Framework

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch15.tex`](../../gmsh_testing_ext/chapters/ch15.tex) — Oracles: Judging Agent Behavior Geometrically
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch15_oracles.ipynb`](../../gmsh_testing_ext/notebooks/ch15_oracles.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/15_oracles.ipynb`](user_guide_tutorials/notebooks/15_oracles.ipynb)
> **Key classes:** `OracleBase`, `ExactOutcomeOracle`, `TraceOracle`, `CompositeOracle`, `make_oracle`, `Scenario`, `OracleVerdict`
> **Prereqs:** NB 11 (SUT Adapters), NB 14 (Env Sim)

**When to use this.** When you need a pluggable `Scenario × SUTTrace → OracleVerdict` evaluator that composes multiple judgment dimensions (exact-outcome, trace pattern, side-effects) into a single pass/fail.

```python
from knowlytix.harness.testing import (
    Scenario, ExactOutcomeOracle, TraceOracle, CompositeOracle,
)
scenario = Scenario(scenario_id="s1", ground_truth="0.8412",
    trace_pattern="data_lookup > risk_check", required_tools=["data_lookup"])
oracles = [(ExactOutcomeOracle(store=store), 0.6),
           (TraceOracle(store=store), 0.4)]
composite = CompositeOracle(oracles=oracles, threshold=0.5,
                             fail_on_any_critical=True)
verdict = composite.evaluate(scenario, trace)
print(f"passed={verdict.passed}  conf={verdict.confidence}")
```

**Common variations.**
- Build custom oracles by subclassing `OracleBase`; the contract is `evaluate(scenario, trace, env_state_before, env_state_after) -> OracleVerdict`.
- `make_oracle(dict)` builds a composite from a YAML/dict definition.

**Edge cases.** `TraceOracle` reads `scenario.trace_pattern` and `scenario.expected_tool_sequence`; without either it returns `passed=True confidence=0.5 "No trace pattern defined"`. Always declare at least one to make the oracle discriminate.

---

## Part IV — Behavioral and Multi-Source

## 19. Tutorial 16: Hallucination Detection

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch16.tex`](../../gmsh_testing_ext/chapters/ch16.tex) — Hallucination Detection via Geodesic Distance
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch16_hallucination_detection.ipynb`](../../gmsh_testing_ext/notebooks/ch16_hallucination_detection.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/16_hallucination.ipynb`](user_guide_tutorials/notebooks/16_hallucination.ipynb)
> **Key classes:** `HallucinationOracle`, `HallucinationGenerator`
> **Prereqs:** NB 02 (Primitives), NB 09 (Verifier Routing), NB 20 (Calibration)

**When to use this.** When you need a binary `HALLUCINATION_DETECTED` verdict on extracted triples, calibrated against the store's positive/negative examples.

```python
from knowlytix.harness.testing import HallucinationOracle
tau = bench.judge._thresholds["geodesic"]   # learned via bench.judge.calibrate()
oracle = HallucinationOracle(store=store, tau_geodesic=tau)
sv = oracle.assess_claim("mdl-001", "has_test_auc", "0.8412")  # PASS, geo≈0.06
sv = oracle.assess_claim("mdl-007", "has_test_auc", "0.8412")  # FAIL, geo≈1.77
```

**Common variations.**
- `assess_triples(triples)` returns a Chapter-10 `TypedVerdict` over a batch — the audit-trail-ready form.
- `evaluate(scenario, trace)` returns an `OracleVerdict` so the oracle composes with `TieredReleaseGate`.

**Edge cases.** Without `bench.judge.calibrate()`, `tau_geodesic` falls back to a conservative default. On the bank ch04 corpus, NB 16 §5 verified 200/200 true triples and only 1/200 hallucinations land below the calibrated τ=0.876 — 99.5% precision.

## 20. Tutorial 17: Stateful and Multi-Turn Testing

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch17.tex`](../../gmsh_testing_ext/chapters/ch17.tex) — Stateful and Multi-Turn Testing
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch17_stateful_multiturn.ipynb`](../../gmsh_testing_ext/notebooks/ch17_stateful_multiturn.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/17_multi_turn.ipynb`](user_guide_tutorials/notebooks/17_multi_turn.ipynb)
> **Key classes:** `MultiTurnAdapter`, `InteractionScript`, `TurnSpec`
> **Prereqs:** NB 11 (SUT), NB 14 (Env Sim), NB 15 (Oracles)

**When to use this.** When testing whether an agent retains memory across turns, follows an expected dialog pattern, or makes downstream decisions correctly conditional on earlier turns.

```python
from knowlytix.harness.testing import MultiTurnAdapter, InteractionScript, TurnSpec
script = InteractionScript(script_id="bank_workflow", turns=[
    TurnSpec(turn_id="t1", user_message="Look up MDL-001.",
             expected_tool_calls=["data_lookup"], expected_contains=["MDL-001"]),
    TurnSpec(turn_id="t2", user_message="What is its status?",
             expected_contains=["degraded"]),
])
adapter = MultiTurnAdapter(harness=mock_harness, store=store)
result = adapter.execute_script(script=script, environment=env)
```

**Common variations.**
- `script.teardown_assertions` for invariants checked once after all turns.
- `TurnSpec.carry_forward` to pin state between turns explicitly.

**Edge cases.** `teardown_assertions` need a specific schema (a dict with a `type` key); a malformed assertion produces `"unknown assertion type"` in the result. Test custom assertion types against the script's `validate()` method before running long scripts.

## 21. Tutorial 18: Multi-Agent & Cross-Document Consistency

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch18.tex`](../../gmsh_testing_ext/chapters/ch18.tex) — Multi-Agent Consensus and Cross-Document Consistency
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch18_multi_agent_cross_doc.ipynb`](../../gmsh_testing_ext/notebooks/ch18_multi_agent_cross_doc.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/18_multi_agent.ipynb`](user_guide_tutorials/notebooks/18_multi_agent.ipynb)
> **Key classes:** `ConsensusOracle`, `AgentRole`, `CrossDocumentAnalyzer`
> **Prereqs:** NB 02 (Primitives), NB 15 (Oracles)

**When to use this.** When you have multiple documents (e.g., quarterly model-risk reports) covering the same entities and need to detect cross-source disagreement, or when you have multiple agents producing claims about the same entity and need a consensus verdict.

```python
from knowlytix.harness.testing import ConsensusOracle, CrossDocumentAnalyzer
stores = {"q1": store_q1, "q2": store_q2, "q3": store_q3}
analyzer = CrossDocumentAnalyzer(stores=stores)
print(analyzer.find_shared_entities())
print(analyzer.find_conflicts())
oracle = ConsensusOracle(stores=stores)
verdict = oracle.evaluate_consensus("mdl-001")
print(f"label={verdict.consensus_label}  mean_tension={verdict.mean_tension}")
```

**Common variations.**
- `find_conflicts()` returns geometric-tension-flagged entity divergences; pair with severity classification for triage.
- For multi-agent runs, register each agent's output to a per-agent store; `evaluate_consensus` reports per-pair and aggregate tension.

**Edge cases.** Stores must share an entity vocabulary. If "MDL-001" appears in `q1` and "MDL_001" in `q2`, the analyzer treats them as different entities. Normalize at ingest (memory: `feedback_ingest_canonicalize_entities`). Cross-store calibration may be needed when stores were trained at different times — see `project_cross_store_calibration_blocker`.

---

## Part V — Production Lifecycle

## 22. Tutorial 19: Release Gates and Audit Reports

> **Conceptual deep-dive:** [`gmsh_testing_ext/chapters/ch19.tex`](../../gmsh_testing_ext/chapters/ch19.tex) — Campaigns, Release Gates, and Provenance
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/ch19_governance_operations.ipynb`](../../gmsh_testing_ext/notebooks/ch19_governance_operations.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/19_release_gates.ipynb`](user_guide_tutorials/notebooks/19_release_gates.ipynb)
> **Key classes:** `AuditReporter`, `RiskTierProfile`, `TieredReleaseGate`
> **Prereqs:** NB 00 (Quick Start), NB 10 (Verdicts & Severity)

**When to use this.** When you have typed verdicts from a benchmark run and need a risk-tiered release verdict (`advisory` / `recommendation` / `action_taking`) plus an audit-ready summary.

```python
from knowlytix.harness.testing import AuditReporter, RiskTierProfile
reporter = AuditReporter(result=result, typed_verdicts=bench.typed_verdicts)
tiers = [RiskTierProfile.advisory(),
         RiskTierProfile.recommendation(),
         RiskTierProfile.action_taking()]
gates = reporter.generate_tiered_release_gate(tiers)
for tier, gate in zip(tiers, gates):
    print(f"{tier.tier_name:16s}  {'PASS' if gate.passed else 'FAIL'}  "
          f"blocking={gate.blocking_reasons[:1]}")
print(reporter.generate_tiered_executive_summary(gates))
print(reporter.generate_failure_report())  # dict keyed by code/category/severity
```

**Common variations.**
- Build custom `RiskTierProfile(tier_name=..., accuracy_threshold=..., failure_budget={...})` per deployment tier.
- YAML-load tiers via `config.risk_tier_profiles_path`.

**Edge cases.** Without `bench.judge.calibrate()` (NB 20), the `failure_report` is dominated by `not_calibrated` entries — 447 in NB 19's reference run. Calibrate before gating, or accept that the geometric channels under the release threshold are floor-noisy.

---

## Part VI — Calibration & Diagnostics

## 23. Tutorial 20: Confidence Calibration — ECE / MCE

> **Conceptual deep-dive:** [`gmsh_testing_ext/appendix_g.tex`](../../gmsh_testing_ext/chapters/appendix_g.tex) — Calibration Toolkit: Angular Encoding, ECE, and MCE
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/appendix_g_confidence_calibration.ipynb`](../../gmsh_testing_ext/notebooks/appendix_g_confidence_calibration.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/20_calibration.ipynb`](user_guide_tutorials/notebooks/20_calibration.ipynb)
> **Key classes:** `CalibrationTester`, `CalibrationReport`, `GMSJudge.calibrate`
> **Prereqs:** NB 00 (Quick Start), NB 19 (Release Gates)

**When to use this.** Before any release-gate evaluation that consumes geometric-confidence thresholds. Calibration is per-corpus and mandatory — without it, the `not_calibrated` failure code dominates the failure report.

```python
from knowlytix.harness.testing import CalibrationTester
# Train calibration on the store's positive/corrupted-negative examples
bench.judge.calibrate(seed=42)
# Inspect ECE/MCE on the resulting per-channel scores
tester = CalibrationTester(n_bins=10)
for qid, v in bench.typed_verdicts.items():
    tester.add(claimed_confidence=v.overall_confidence, actual_correct=bool(v.correct))
r = tester.report()
print(f"ECE={r.ece}  MCE={r.mce}  overconfidence_ratio={r.overconfidence_ratio}")
```

**Common variations.**
- Two calibration levels: *response-level* (LLM `overall_confidence`) and *triple-level* (the four trained classifiers — geodesic, path_transport, tension, holonomy). NB 20 shows both.
- After `calibrate()`, `bench.judge._thresholds` carries τ values per channel; these are what the release gate enforces.
- `calibrate()` also rebuilds the typed-verdict router in place (`judge=self`) so the freshly-fit cuts reach the verifiers — without this the router would keep the empty thresholds captured at construction and every fact claim would come back `not_calibrated`. A channel with no separable calibration cohort recuses (neutral, no failure attribution) instead of emitting `not_calibrated`; see the routing edge case in §12.

**Edge cases.** Response-level ECE measures the LLM's overconfidence and is *not* fixed by `bench.judge.calibrate()` — the judge calibration only tightens the triple-level channel. On the bank ch04 corpus: response-level ECE ≈ 0.35 (poor); triple-level (5-fold CV on the geodesic channel) ECE ≈ 0.012 (under the 0.05 ship bar).

## 24. Tutorial 21: GMS Training & Post-Training Diagnostics

> **Conceptual deep-dive:** [`gmsh_testing_ext/appendix_f.tex`](../../gmsh_testing_ext/chapters/appendix_f.tex) — GMS Training and Post-Training Diagnostics
> **Monograph notebook:** [`gmsh_testing_ext/notebooks/appF_training_diagnostics.ipynb`](../../gmsh_testing_ext/notebooks/appF_training_diagnostics.ipynb)
> **User-guide notebook:** [`user_guide_tutorials/notebooks/21_gms_diagnostics.ipynb`](user_guide_tutorials/notebooks/21_gms_diagnostics.ipynb)
> **Key classes:** `GMSDiagnostics`, `CheckResult`, `DiagnosticReport`
> **Prereqs:** NB 02 (Primitives)

**When to use this.** Before using a freshly-built (or freshly-retrained) store as ground truth. Runs nine geometric-invariant checks (rotor orthogonality, Stiefel projection, embedding norm, ENM SHA-256 integrity, primitive sanity) on the trained model.

```python
from gms.diagnostics import GMSDiagnostics
import torch
diag = GMSDiagnostics(model=store.model, adapter=store.adapter, enm=store.enm,
                      doc_graph=store.doc_graph,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      sample_size=200)
report = diag.run_all()
n_pass = sum(1 for c in report.checks if c.passed)
print(f"{n_pass}/{len(report.checks)} checks pass")
```

**Common variations.**
- Bump `sample_size` higher for stores with > 5k triples (default 200 is sensible for typical corpus sizes).
- Inspect `c.evidence` for failure triage — it carries the worst-violating rotor or the bad ENM keys.

**Edge cases.** On the shipped `ch04_ground_truth` fixture, 6/9 checks pass; `projection_integrity`, `tension_energy`, `check_holonomy` fail because of known geometric tension in the bank-corpus embedding. The primary primitive `score_triple` passes at 93.4%, so release-gate signals on this corpus remain meaningful. For a strict action_taking deployment, retrain with tighter loss weights or document the waiver.

## 25. Tutorial 22: Failure Code Taxonomy

> **Conceptual deep-dive:** [`gmsh_testing_ext/appendix_c.tex`](../../gmsh_testing_ext/chapters/appendix_c.tex) — Failure Code Taxonomy
> **Monograph notebook:** *(reference appendix — no companion lab notebook)*
> **User-guide notebook:** [`user_guide_tutorials/notebooks/22_failure_codes.ipynb`](user_guide_tutorials/notebooks/22_failure_codes.ipynb)
> **Key classes:** `FailureCode`, `SeverityClassifier`, `SeverityLevel`, `SeverityAssessment`, `RiskTierProfile`
> **Prereqs:** NB 10 (Verdicts & Severity), NB 19 (Release Gates)

**When to use this.** When you need to look up what a specific `FailureCode` means, how it maps to severity at each risk tier, and how `RiskTierProfile.failure_budget` consumes it.

```python
from knowlytix.harness.testing import FailureCode, SeverityClassifier
sc = SeverityClassifier()
for code in [FailureCode.HALLUCINATION_DETECTED, FailureCode.FACT_INCORRECT,
             FailureCode.NOT_CALIBRATED]:
    for tier in ["advisory", "recommendation", "action_taking"]:
        sa = sc.classify_code(code, risk_tier=tier)
        print(f"{code.value:30s} @ {tier:16s} -> {sa.level.name}")
```

**Common variations.**
- 29 codes across four families: claim extraction, fact verification, geometric structure, operational.
- `RiskTierProfile.action_taking().failure_budget = {hallucination_detected: 0, policy_violation: 0, policy_blocked_pre_execution: 0}` — autonomous-decision deployments tolerate zero of these.

**Edge cases.** Tier escalation is real: `fact_incorrect` is `HIGH` at `advisory` and `recommendation` but `CRITICAL` at `action_taking`. Never hardcode severity in your reporting — always go through `SeverityClassifier.classify_code(code, risk_tier=...)`.

---

## Part VII — Operational Annex

## 26. Operational Features — Parallel, Perf, Cost, Artifacts

> **User-guide notebook:** [`user_guide_tutorials/notebooks/23_operational_features.ipynb`](user_guide_tutorials/notebooks/23_operational_features.ipynb)
> **Key classes:** `PerformanceProfiler`, `CostTracker`, `InvarianceTester`, `ArtifactRegistry`, `ParallelExecutor`, `DashboardServer`, `StreamingAdapter`, `BrowserAdapter`

This annex covers operational machineries that live alongside the core verification pipeline. Most are reference-only here; NB 23 demos the four that exercise easily without standing up infrastructure.

**Demonstrated in NB 23:**

```python
from knowlytix.harness.testing import PerformanceProfiler, CostTracker, InvarianceTester, ArtifactRegistry
prof = PerformanceProfiler(store=store, sla_latency_ms=2000.0)
prof.record(tool_name="data_lookup", latency_ms=909.0)
print(prof.report())                # p50, p95, sla_tension

tracker = CostTracker(store=store, budget_usd=5.0, input_price_per_mtok=3.0, output_price_per_mtok=15.0)
tracker.record(tool_name="anthropic_messages", input_tokens=250, output_tokens=80)
print(tracker.summary())            # total_cost_usd, budget_utilization, budget_tension

reg = ArtifactRegistry(artifacts_dir="artifacts", artifact_type="fault_profile")
reg.add(versioned_artifact)         # registry-managed versioning + content_hash
```

**Reference-only — need running infrastructure:**

| Feature | Entry point | Why reference-only |
|---|---|---|
| `ParallelExecutor` | `from knowlytix.harness.testing import ParallelExecutor, ParallelConfig` | Needs a multi-worker pool; exercised in integration tests. |
| `DashboardServer` | `from knowlytix.harness.testing import DashboardServer, DashboardSnapshot` | Needs a WebSocket port (`live_dashboard_port`); UI subscribes to events. |
| `StreamingAdapter` | `from knowlytix.harness.testing import StreamingAdapter, StreamConsistencyOracle` | Needs an LLM that supports token streaming. |
| `BrowserAdapter` | `from knowlytix.harness.testing import BrowserAdapter, UIScript, BrowserConfig` | Needs a headless browser (Selenium / Playwright). |

---

## 27. End-to-End Walkthrough

A full pipeline run that exercises everything from ingestion to release-gate audit. This is the long-form version of NB 00's Quick Start plus the calibration step from NB 20 and the audit step from NB 19.

```python
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig, LLMEvaluator, AuditReporter, RiskTierProfile
from _shared import load_ch04_store, make_anthropic_client, DEFAULT_TUTORIAL_MODEL

# 1. Configure
config = DOEHarnessConfig(
    markdown_path="gmsh_testing_ext/lab_stores/ch04_ground_truth/documents/combined.md",
    stores_dir="gmsh_testing_ext/lab_stores",
    store_name="ch04_ground_truth",
    factor_group="quick_screen",
    n_runs=4,
    max_per_category=5,
    cap_enabled=True,
    enable_typed_verdicts=True,
    enable_tracing=False,
    target_type="llm",
    target_model=DEFAULT_TUTORIAL_MODEL,
)

# 2. Ingest (loads existing ch04 store)
bench = DOEGMSBenchmark(config)
bench.ingest()

# 3. Calibrate the judge — closes the not_calibrated failures
bench.judge.calibrate(seed=42)

# 4. Generate + evaluate
bench.generate_questions()
client = make_anthropic_client()
evaluator = LLMEvaluator(client=client, model=DEFAULT_TUTORIAL_MODEL)
result = bench.run(evaluator=evaluator)
verdicts = bench.typed_verdicts

# 5. Release-gate verdicts at three tiers
reporter = AuditReporter(result=result, typed_verdicts=verdicts)
tiers = [RiskTierProfile.advisory(),
         RiskTierProfile.recommendation(),
         RiskTierProfile.action_taking()]
gates = reporter.generate_tiered_release_gate(tiers)
for t, g in zip(tiers, gates):
    print(f"{t.tier_name:16s}  {'PASS' if g.passed else 'FAIL'}")

# 6. Executive summary + failure report
print(reporter.generate_tiered_executive_summary(gates))
fr = reporter.generate_failure_report()
print(f"failures by code: {fr['by_code']}")
```

The same flow runs end-to-end in NB 00 (Quick Start, light config) and NB 19 (Release Gates, with audit-report inspection).

---

## 28. CLI Reference

```bash
# Basic run
python -m knowlytix.harness.testing run report.md

# With GMSH features
python -m knowlytix.harness.testing run report.md \
    --typed-verdicts \
    --provenance \
    --snapshot-dir snapshots/ \
    --audit-report audit.html \
    --release-gate accuracy=0.8 geo=0.5 \
    --risk-tiers tier_profiles.yaml \
    --baseline-run run_abc12345 \
    --n-runs 64 \
    --target-model claude-sonnet-4-6

# From YAML config
python -m knowlytix.harness.testing run --config config.yaml
```

**Run management:**

```bash
python -m knowlytix.harness.testing runs list                  # list tracked runs
python -m knowlytix.harness.testing runs show run_abc12345     # show one run
```

**Regression and replay:**

```bash
python -m knowlytix.harness.testing regression promote run_abc12345
python -m knowlytix.harness.testing regression list
python -m knowlytix.harness.testing replay manifest.yaml
python -m knowlytix.harness.testing compare result_a.json result_b.json
```

The original single-command syntax still works: `python -m knowlytix.harness.testing report.md --n-runs 32`.

---

## 29. Configuration Reference

All configuration goes through `DOEHarnessConfig`. Set fields in Python or load from YAML via `from_yaml(path)`.

**YAML template:**

```yaml
document:
  markdown_path: "path/to/report.md"
  stores_dir: "gms_stores"

doe:
  factor_group: "quick_screen"
  n_runs: 32
  method: "sobol"
  seed: 42

target:
  target_type: "agent"
  target_model: "claude-sonnet-4-6"

gmsh:
  enable_typed_verdicts: true
  enable_provenance: true
  snapshot_dir: "snapshots"

agentic:
  enable_gateway: true
  gateway_fault_profiles_path: "fault_profiles.yaml"
  enable_policy_engine: true
  policy_rules_path: "policy_rules.yaml"

release:
  risk_tier_profiles_path: "tier_profiles.yaml"
  baseline_run_id: ""
```

**Frequently-used fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `markdown_path` | str | `""` | Path to the markdown document |
| `stores_dir` | str | `"gms_stores"` | Directory for GMS store files |
| `store_name` | str | `""` | Subdirectory under `stores_dir` (default: derived from markdown filename) |
| `factor_group` | str | `"quick_screen"` | DOE factor preset (`quick_screen`, `comprehensive`, etc.) |
| `n_runs` | int | `32` | DOE design rows |
| `max_per_category` | int | `10` | Max questions per generator |
| `cap_enabled` | bool | `False` | Cap-admissibility training (mandatory for monograph fixtures) |
| `target_type` | str | `"llm"` | `"llm"` or `"agent"` |
| `target_model` | str | `"claude-sonnet-4-20250514"` | Model name |
| `enable_typed_verdicts` | bool | `False` | Claim-level verification |
| `enable_tracing` | bool | `True` | Tracing store (disable in tutorial NBs) |
| `enable_gateway` | bool | `False` | Tool gateway |
| `enable_policy_engine` | bool | `False` | Policy engine |
| `enable_hallucination_testing` | bool | `False` | Hallucination oracle |
| `enable_calibration_testing` | bool | `False` | Calibration tester |
| `enable_multi_agent` | bool | `False` | Multi-agent consensus |
| `enable_cross_document` | bool | `False` | Cross-document conflict |
| `risk_tier_profiles_path` | str | `""` | YAML for `RiskTierProfile` set |
| `sla_latency_ms` | float | `5000.0` | SLA target |
| `cost_budget_usd` | float | `10.0` | Cost cap |
| `device` | str | `""` | `"cpu"`, `"cuda"`, or `""` (auto) |

See [`API_REFERENCE.md#doeharnessconfig`](API_REFERENCE.md#doeharnessconfig) for the full 60+ field reference.

---

## 30. Troubleshooting

### `AnthropicAPIKeyError`

**Symptom:** `AnthropicAPIKeyError: Anthropic API key not found.`

**Fix:** Set the env var or configure `pass(1)`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
pass insert anthropic/api-key
```

By design there is no mock-LLM fallback (memory: `feedback_no_mock_fallback`).

### CUDA assertion errors during tracing-store training

**Symptom:** `RuntimeError: CUDA error: device-side assert triggered`.

**Fix:** Tutorial NBs set `enable_tracing=False`. For production runs that need tracing, set `device="cpu"` for the tracing store.

### Index errors in `link_prediction` / `multi_hop_transport` generators

**Symptom:** `IndexError` during `bench.generate_questions()`.

**Fix:** These generators require a larger store. Either use a larger document, or exclude them: `generators=["plausibility", "tension", "exact_retrieval", "counting", "cross_reference", "boundary"]`.

### `bench.generate_questions()` returns 0 questions

**Fix:** Check that the store loaded (`store.stats()` shows entities > 0), the configured generators match the corpus structure, and `max_per_category > 0`.

### Policy engine blocking everything

**Symptom:** All tool calls blocked.

**Fix:** Test rules in isolation with `engine.check("tool_name", {}, "")` before wiring through the gateway. Valid `rule_type` values are `allows` / `blocks` / `requires`; other strings fall through to the GMS-graph layer (which may produce surprising verdicts).

### `not_calibrated` dominates the failure report

**Symptom:** `AuditReporter.generate_failure_report()` shows hundreds of `not_calibrated` entries.

**Fix:** Call `bench.judge.calibrate(seed=42)` before `bench.run(evaluator=...)`. This trains the four geometric-channel classifiers (geodesic, path_transport, tension, holonomy) and stores their τ thresholds on the judge. See NB 20.

### Replay fails with "Snapshot not found"

**Fix:** Snapshots are file-based — if the snapshot directory was deleted, the snapshot is gone. Use `snap_mgr.list_snapshots()` to see what's available.

### Import errors

**Fix:** Activate the project environment and verify the import:

```bash
cd ~/jupyterlab/kg-memory
source ~/cluster/spark-venv/bin/activate
python -c "from knowlytix.harness.testing import DOEGMSBenchmark; print('OK')"
```

For module-not-found on `_shared`, ensure the notebooks' `sys.path.insert(0, str(Path.cwd()))` runs first.

---

## 31. Companion Resources

- [`API_REFERENCE.md`](API_REFERENCE.md) — full dataclass + method signatures for every class mentioned here
- [`user_guide_tutorials/`](user_guide_tutorials/) — the 24 paired notebooks
- [`user_guide_tutorials/PLAN.md`](user_guide_tutorials/PLAN.md) — master plan with section map, fixture strategy, execution phases
- [`user_guide_tutorials/STYLE.md`](user_guide_tutorials/STYLE.md) — writing + notebook conventions enforced in every tutorial
- [`gmsh_testing_ext/`](../../gmsh_testing_ext/) — long-form testing monograph (Ch01–Ch19 + 7 appendices)
- [`gmsh_testing_ext/notebooks/`](../../gmsh_testing_ext/notebooks/) — monograph lab notebooks (per chapter + appendices F, G)
- [`governed_agent/USER_GUIDE.md`](../../governed_agent/USER_GUIDE.md) — separate guide for the *build* side of the stack
- [`knowlytix/harness/testing/USER_GUIDE.legacy.md`](USER_GUIDE.legacy.md) — the previous (4675-line) version of this guide, preserved for reference until the next release cycle

### How the docs fit together

```
You want:                                  Read:
─────────────────────────────────────      ──────────────────────────────────
Why this approach?                          gmsh_testing_ext/chapters/chNN.tex
How does the math work?                     gmsh_testing_ext/chapters/chNN.tex
How is the lab structured? Exercises?       gmsh_testing_ext/notebooks/chNN_*.ipynb
What do I run to test my agent?             this guide §N + user_guide_tutorials/notebooks/NN_*.ipynb
Where's the full API surface?               API_REFERENCE.md
What's the build-side story?                governed_agent/USER_GUIDE.md
```

Every tutorial section above links into the first four columns. Pick a starting point and follow the cross-references.
