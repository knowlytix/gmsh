# knowlytix-harness

> **G**eometric **M**emory **S**ystems **H**arness — DOE-driven, black-box
> agentic testing with graph-verified ground truth and Design-of-Experiments
> factor analysis. Provider-agnostic: swap between Anthropic, OpenAI,
> Bedrock, Azure, or local Ollama without touching code.

`knowlytix-harness` is the headline package in the [Geometric Memory Systems][gms-repo]
family. Use it to turn ad-hoc "does this agent work?" evaluations into
repeatable, statistically-grounded campaigns with typed verdicts, failure
taxonomy, cost tracking, and release gates. Bundles the runtime-governance
surface (`knowlytix.harness.governance`) for production-grade governed
agentic systems — same install, no extra step.

- **Package**: `knowlytix-harness`
- **License**: Apache-2.0
- **Python**: 3.12+
- **Status**: alpha (v0.x)

## Install

```bash
pip install knowlytix-harness
```

Pulls `knowlytix-core`, `knowlytix-knowledge`, and `knowlytix-benchmark` at
matching `~=0.1.0` versions (lockstep releases — no version mismatches). LLM
calls route through [LiteLLM][litellm]: one library, every provider.

## Provider setup (pick one)

The same `knowlytix-harness` wheel runs against any supported provider. Set the right env
vars and go — no code changes.

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GMS_LLM_MODEL=anthropic/claude-opus-4-6
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export GMS_LLM_MODEL=openai/gpt-4o-mini
```

### AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-west-2
export GMS_LLM_MODEL=bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
```

### Azure OpenAI

```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://your-resource.openai.azure.com
export AZURE_API_VERSION=2024-02-15-preview
export GMS_LLM_MODEL=azure/your-deployment-name
```

### Local Ollama (no API key)

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export GMS_LLM_MODEL=ollama/llama3
```

Full list including Google, Mistral, Cohere, Together, and more in
`.env.example` from the source repo.

## Post-install verification

After `pip install knowlytix-harness`, three commands confirm your stack is
healthy and open the human-facing exploration notebook:

```bash
pip install jupyterlab                       # one-time (JupyterLab)
knowlytix-smoke                                   # 5-step key-free assertion suite
jupyter lab $(knowlytix-smoke --notebook-path)    # interactive walkthrough
```

`knowlytix-smoke` exits 0 on a healthy install; exit 1 names which of the
5 checks failed (imports + `__all__`, Settings defaults,
`importlib.resources` fixture reachability, `knowlytix.benchmark.score_answer`
on shipped predictions, harness DOE fixture schema).  The notebook
is shipped as package data inside this wheel — no repo clone
needed — and its `--notebook-path` output is an absolute,
symlink-resolved filesystem path.

## CLI quickstart

```bash
# 1. Verify install
knowlytix-harness --help

# 2. Smoke test against the bundled fixture (no external data, no API key needed
#    if you use a dry-run evaluator)
knowlytix-harness run --fixture doe_smoke.json --dry-run

# 3. Live run with an LLM evaluator
knowlytix-harness run --markdown report.md --factor-group query_core --n-runs 32

# Alias — knowlytix-harness and knowlytix-testing are the same entry point
knowlytix-testing run --campaign campaigns/regression.yaml
```

## Programmatic quickstart — one DOE campaign end-to-end

```python
import os

from gms import get_llm, ModelPurpose
from knowlytix.harness.testing import (
    DOEGMSBenchmark, DOEHarnessConfig,
    make_evaluator, HallucinationOracle,
)

config = DOEHarnessConfig(
    markdown_path="report.md",      # document under test
    factor_group="query_core",      # DOE factor group
    n_runs=32,
    enable_hallucination_testing=True,
    enable_cost_tracking=True,
)

bench = DOEGMSBenchmark(config)
bench.ingest()

# make_evaluator(target_type, target_model, client=None, harness=None)
evaluator = make_evaluator(
    target_type="llm",
    target_model=os.environ["GMS_LLM_MODEL"],
    client=get_llm(ModelPurpose.DEFAULT),
)
result = bench.run(evaluator=evaluator)

analyzer = bench.analyze(result)   # returns a DOEAnalyzer (from graphdoe)
print(analyzer.summary())          # check the DOEAnalyzer API for exact method
```

## Configuration reference

### `GMSH_*` — harness tuning

| Variable | Default | Meaning |
|---|---|---|
| `GMSH_DOE_N_RUNS` | `32` | Runs per DOE campaign. |
| `GMSH_DOE_SEED` | `42` | RNG seed for run selection. |
| `GMSH_DOE_SLA_LATENCY_MS` | `5000` | Per-call latency SLA. |
| `GMSH_DOE_COST_BUDGET_USD` | `10.0` | Campaign-level USD ceiling. |
| `GMSH_DOE_HALLUCINATION_THRESHOLD` | `0.1` | Max tolerated hallucination rate. |
| `GMSH_MAX_WORKERS` | `4` | Parallel eval worker count. |
| `GMSH_QUESTION_TIMEOUT_S` | `60` | Per-question timeout. |
| `GMSH_MAX_RETRIES` | `2` | Retry count on evaluator error. |
| `GMSH_MAX_TURNS` | `8` | Multi-turn conversation cap. |
| `GMSH_TRUNCATE_RESULT_AT` | `10000` | Character cap on captured outputs. |
| `GMSH_STORES_DIR` | `./gms_stores` | Where ingested stores live. |
| `GMSH_TRACING_DIR` | `./doe_tracing_store` | Trace artifact root. |
| `GMSH_RUNS_DIR` | `./runs` | Run records output dir. |
| `GMSH_CAMPAIGNS_DIR` | `./campaigns` | Campaign manifests. |
| `GMSH_SESSION_STORE_PATH` | `./harness_session_store` | Session state. |
| `GMSH_LIVE_DASHBOARD_PORT` | `8765` | Live WebSocket dashboard port. |

Twenty-one `GMSH_ENABLE_*` feature flags toggle optional subsystems (typed
verdicts, provenance, gateway fault injection, policy engine, stateful
testing, hallucination oracle, calibration, multi-agent, cross-document,
disambiguation, invariance, streaming, live dashboard, and more). See the
`USER_GUIDE.md` shipped in the wheel for the full list.

### `GMS_LLM_*` — LLM routing

| Variable | Meaning |
|---|---|
| `GMS_LLM_MODEL` | Base LiteLLM model string. Required unless every purpose is overridden. |
| `GMS_LLM_MODEL_JUDGE` | Override for judge/verifier calls. |
| `GMS_LLM_MODEL_GENERATOR` | Override for question-generation. |
| `GMS_LLM_MODEL_SCORER` | Override for scoring. |
| `GMS_LLM_TIMEOUT_SECONDS` | Per-call timeout. Default `60`. |
| `GMS_LLM_MAX_RETRIES` | Transient retries. Default `2`. |
| `GMS_LLM_TEMPERATURE` | Sampling temperature. Default `0.0`. |

## Architecture in one paragraph

`knowlytix-harness` decomposes "did my agent behave correctly?" into (1) document
ingestion via `knowlytix-knowledge` → geometric memory store, (2) auto-generation of
graph-verified questions via `knowlytix-benchmark` + geometric generators, (3)
DOE factor-group sweep producing a structured run matrix, (4) typed verdict
verification against provable graph traversals, (5) failure taxonomy +
severity classification + cost/latency tracking, (6) release-gate decision
with audit packet. Every step is provider-agnostic — the same campaign YAML
runs unchanged against any supported LLM.

For **runtime governance** of agentic systems in production, the wheel also ships `knowlytix.harness.governance` (the *governed harness*): triple-gate tool gateway (schema + policy + plausibility), typed claim verification routed to GMS primitives, behavioral FSM contracts, governance bundle signing, runtime gates, and drift monitoring. Same wheel; same install — `pip install knowlytix-harness` gives you both the black-box testing and the governed-runtime surface.

## Public API

The wheel ships two subpackages — black-box testing (`knowlytix.harness.testing`) and runtime governance (`knowlytix.harness.governance`):

```python
# Black-box DOE testing (the headline product)
from knowlytix.harness.testing import (
    # Core
    DOEGMSBenchmark, DOEHarnessConfig, GMSHSettings,
    # Evaluators + judges
    LLMEvaluator, AgentEvaluator, make_evaluator, GMSJudge,
    # Oracles + taxonomy
    HallucinationOracle, SeverityClassifier, CompositeOracle,
    # Agentic testing
    ToolGateway, PolicyEngine, CampaignManager,
    # …195 symbols total in __all__
)

# Runtime governance — the governed harness
from knowlytix.harness.governance import (
    # Triple-gate tool gateway: schema validation + policy + GMS plausibility
    GovernedToolGateway,
    # Typed claim verification routed to GMS primitives
    ClaimRouter, TypedClaim,
    # Behavioral FSM contracts (advisory / recommendation / action-taking)
    BehavioralContract,
    # End-to-end orchestrator + lifecycle state machine
    GovernedOrchestrator,
    # Runtime gates + drift monitoring + bundle signing
    RuntimeGate, DriftMonitor, GovernanceBundle,
)
```

See the top of `harness/testing/__init__.py` and `harness/governance/__init__.py` for the full declarations or `USER_GUIDE.md` for task-oriented navigation.

## Related packages

| Package | Role |
|---|---|
| [`knowlytix-core`][knowlytix-core-pypi] | Geometric memory engine |
| [`knowlytix-knowledge`][knowlytix-knowledge-pypi] | Document ingest + query front-end |
| [`knowlytix-benchmark`][knowlytix-benchmark-pypi] | Structured-retrieval benchmark |

## Links

- Source: [knowlytix/gmsh][gms-repo]
- Book: _Geometric Memory Systems_ (forthcoming)
- Papers: _DOE-GMS Benchmark_, _GMSH Black-Box Agentic Testing_

[gms-repo]: https://github.com/knowlytix/gmsh
[knowlytix-core-pypi]: https://pypi.org/project/knowlytix-core/
[knowlytix-knowledge-pypi]: https://pypi.org/project/knowlytix-knowledge/
[knowlytix-benchmark-pypi]: https://pypi.org/project/knowlytix-benchmark/
[litellm]: https://docs.litellm.ai/
