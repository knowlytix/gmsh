# finstructbench

> Benchmark for **Struct**ured **R**etrieval from **Fin**ancial **Doc**uments.
> Auto-generates questions from document-graph topology and scores LLM
> predictions against a provably correct graph-traversal baseline.

`finstructbench` is one of four packages in the [Geometric Memory Systems][gms-repo]
family. Use it to answer questions like *"how close does my RAG pipeline get
to the graph-verified ground truth on this financial report?"* — with metrics
that resist gaming because the ground truth comes from graph operations, not
a held-out human-labeled test set.

- **Package**: `finstructbench`
- **License**: Apache-2.0
- **Python**: 3.12+
- **Status**: alpha (v0.x)

## Install

```bash
pip install finstructbench
```

Depends on [`gms-core`][gms-core-pypi] (pinned `~=0.1.0`). LLM-mode scoring
routes through [LiteLLM][litellm] — set `GMS_LLM_MODEL` plus your
provider's API key and any model works.

## Quickstart — score a prediction set

```python
import json
from importlib.resources import files

from finstructbench import score_answer

# Smoke fixtures shipped with the wheel:
questions = json.loads((files("finstructbench.fixtures.smoke") / "questions.json").read_text())
predictions = json.loads((files("finstructbench.fixtures.smoke") / "predictions.json").read_text())

by_id = {p["id"]: p["answer"] for p in predictions["predictions"]}
for q in questions["questions"]:
    result = score_answer(by_id[q["id"]], q["ground_truth"])
    mark = "correct" if result.correct else "wrong"
    print(f"{q['id']}: {mark}  partial={result.partial_score:.2f}  ({result.detail})")
```

## Quickstart — run the full benchmark

```python
from finstructbench import Benchmark, get_instance_path

bench = Benchmark(get_instance_path("model_validation"))
result = bench.run()          # graph-only mode (no LLM, no API key needed)
bench.print_results(result)
```

To evaluate an LLM against the same ground-truth graph:

```python
from finstructbench.llm_caller import create_client

client = create_client()       # reads GMS_LLM_MODEL_SCORER → GMS_LLM_MODEL
result = bench.run(llm_client=client)
bench.print_results(result)
```

## CLI

```bash
python -m finstructbench --instance model_validation
python -m finstructbench --instance credit_portfolio --llm-model anthropic/claude-opus-4-6
```

## Configuration

### `FINSTRUCTBENCH_*` — scoring tolerances

| Variable | Default | Meaning |
|---|---|---|
| `FINSTRUCTBENCH_FLOAT_TOL` | `1e-6` | Absolute tolerance for float comparisons. |
| `FINSTRUCTBENCH_CLOSE_THRESHOLD` | `0.01` | Relative tolerance for "close enough" financial values. |
| `FINSTRUCTBENCH_TUPLE_ELEMENT_TOL` | `1e-3` | Tolerance per element inside tuple answers. |

### `GMS_LLM_*` — LLM routing (only needed for LLM-mode scoring)

| Variable | Meaning |
|---|---|
| `GMS_LLM_MODEL` | Base LiteLLM model string. |
| `GMS_LLM_MODEL_SCORER` | Override for scoring calls (recommended). |
| `GMS_LLM_TIMEOUT_SECONDS` | Per-call timeout. Default `60`. |

See `.env.example` in the source repo for the full provider key reference.

## Included benchmark instances

Five synthetic financial-domain instances ship with the wheel:

| Instance | Topic |
|---|---|
| `basel_capital` | Bank capital adequacy under Basel III |
| `credit_portfolio` | Credit risk portfolio analysis |
| `fair_lending` | Fair lending compliance testing |
| `model_validation` | Model validation report (largest) |
| `stress_test` | Stress testing scenarios |

All synthetic — no real institution, person, or market event is depicted.

## Public API

```python
from finstructbench import (
    Benchmark, BenchmarkResult,
    DocumentGraph, ENMEntry, ENMKey, PhaseEncoder,
    FinStructBenchSettings,
    GeneratedQuestion, ScoreResult, score_answer,
    default_generators, get_instance_path, ingest_markdown, list_instances,
)
```

`GeneratedQuestion` is a stable contract consumed by `harness.doe.bridge` —
don't rename without coordinating (see `CLAUDE.md` §Coding standards).

## Related packages

| Package | Role |
|---|---|
| [`gms-core`][gms-core-pypi] | Geometric memory engine (required runtime dep) |
| [`docgms`][docgms-pypi] | Document-graph ingest + query front-end |
| [`gmsh`][gmsh-pypi] | DOE-driven testing harness (consumes `GeneratedQuestion`) |

## Links

- Source: [knowlytix/gmsh][gms-repo]
- Paper: _FinStructBench: Benchmarking Structured Retrieval from Financial Documents_

[gms-repo]: https://github.com/knowlytix/gmsh
[gms-core-pypi]: https://pypi.org/project/gms-core/
[docgms-pypi]: https://pypi.org/project/docgms/
[gmsh-pypi]: https://pypi.org/project/gmsh/
[litellm]: https://docs.litellm.ai/
