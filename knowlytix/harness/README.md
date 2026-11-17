# GMS Agent Harness

A standalone agent orchestration layer that owns the full execution loop — prompt construction, Claude API calls, tool dispatch, geometric memory integration, and policy enforcement — without requiring an external harness like Claude Code.

The Harness unifies **33 tools** across three categories:

- **GMS** (8 tools) — create, ingest, query, search, and manage geometric expert systems
- **MoDeva** (24 tools) — load data, train models, diagnose, explain, interpret, compare
- **Memory** (1 tool) — recall knowledge from the geometric memory system

Every tool execution can produce knowledge triples that are automatically learned into a persistent GMS store, enabling cross-session memory and contradiction detection via tension energy on $S^{m-1}$.

## Quick Start

```python
from knowlytix.harness import Harness, HarnessConfig

h = Harness(HarnessConfig(stores_dir="gms_stores"))
h.register_gms_tools()
h.register_modeva_tools()

response = h.run("Load BikeSharing, train XGB depth-2, run FANOVA, diagnose accuracy")
print(response)

h.save()
```

## Installation

The Harness lives inside the GMS project. No separate installation is needed — ensure the project root is on your `PYTHONPATH` and the dependencies are available:

```bash
# From the project root
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Provider keys and tunable env vars live in .env — copy the template:
cp .env.example .env
# then edit .env to set ANTHROPIC_API_KEY (or OPENAI_API_KEY,
# GOOGLE_API_KEY, OLLAMA_BASE_URL, …) and any GMS_*, GMSH_*, DOCGMS_*,
# FINSTRUCTBENCH_* overrides you need. `.env` is gitignored.

# Verify
python -c "from knowlytix.harness import Harness; print('OK')"
```

See [`.env.example`](../.env.example) at the repo root for the full list
of supported environment variables, grouped by domain (`GMS_LLM_*`,
`GMS_*`, `GMSH_*`, `DOCGMS_*`, `FINSTRUCTBENCH_*`).

### Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude Messages API client |
| `torch` | GPU device management, GMS model operations |
| `modeva` | Model development and validation toolkit |

All other dependencies (`docgms`, `gms`, `agents`) are internal to this project.

## Usage

### Programmatic API

#### Basic: single request

```python
from knowlytix.harness import Harness, HarnessConfig

config = HarnessConfig(
    model="claude-opus-4-7",
    stores_dir="gms_stores",
    session_store_path="my_session",
)

h = Harness(config)
h.register_gms_tools()
h.register_modeva_tools()

response = h.run(
    "Load BikeSharing dataset, train an XGBoost regressor with max_depth=2 "
    "named XGB2, run FANOVA interpretation and diagnose accuracy."
)
print(response)
h.save()
```

Output (abbreviated):

```
[Tool] load_dataset(name=BikeSharing, builtin=BikeSharing)
[Tool] preprocess_data(dataset_name=BikeSharing)
[Tool] add_model(dataset_name=BikeSharing, model_type=MoXGBRegressor, model_name=XGB2, params={'max_depth': 2})
[Tool] train_model(dataset_name=BikeSharing, model_name=XGB2)
[Tool] interpret_fi(dataset_name=BikeSharing, model_name=XGB2)
[Memory] Learned 12 triples from interpret_fi
[Tool] interpret_effects(dataset_name=BikeSharing, model_name=XGB2)
[Tool] diagnose_accuracy(dataset_name=BikeSharing, model_name=XGB2)
[Memory] Learned 9 triples from diagnose_accuracy

## Results Summary for XGB2 (XGBoost, max_depth=2)

### Accuracy
| Metric | Train  | Test   | Gap     |
|--------|--------|--------|---------|
| MSE    | 0.0064 | 0.0069 | 0.0005  |
| R²     | 0.8132 | 0.8035 | -0.0097 |

No overfitting detected — train-test gap < 1%.

### FANOVA Feature Importance
| Rank | Feature | Importance |
|------|---------|------------|
| 1    | hr      | 0.7411     |
| 2    | yr      | 0.0997     |
| 3    | temp    | 0.0569     |
...
```

#### Multi-turn conversation

```python
# First turn
response = h.run(
    "Load BikeSharing, train XGB2 (depth-2) and XGB6 (depth-6), compare accuracy"
)
print(response)

# Follow-up — message history is preserved
response2 = h.chat("Which model should I deploy? Consider accuracy vs robustness.")
print(response2)

# Follow-up — agent can recall memory from earlier turns
response3 = h.chat("What was the R² for XGB2?")
print(response3)

# Reset conversation (keeps GMS memory and MoDeva session state)
h.reset()
```

#### Session persistence across runs

```python
# Run 1: train and diagnose
h = Harness(HarnessConfig(session_store_path="credit_session"))
h.register_modeva_tools()
h.run("Load TaiwanCredit, train XGB depth-2, diagnose accuracy and robustness")
h.save()
# Session GMS: 55 entities, 42 triples

# Run 2 (later): memory persists
h2 = Harness(HarnessConfig(session_store_path="credit_session"))
h2.register_modeva_tools()
h2.run("What did we find about robustness in the previous session?")
# Agent uses recall_memory → retrieves robustness triples from session GMS
```

#### Custom tools

```python
import requests

def fetch_api(args: dict, ctx) -> tuple[str, None]:
    resp = requests.get(args["url"])
    return resp.text[:5000], None

h.register_tool(
    name="fetch_api_data",
    description="Fetch data from a REST API endpoint.",
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "API endpoint URL"},
        },
        "required": ["url"],
    },
    handler=fetch_api,
    category="custom",
)

# The LLM can now call fetch_api_data alongside GMS and MoDeva tools
h.run("Fetch the latest metrics from http://internal-api/metrics and analyze them")
```

#### GMS expert system tools

```python
h = Harness(HarnessConfig(stores_dir="gms_stores"))
h.register_gms_tools()

# Ingest a document into a new expert system
h.run("Ingest the file benchmark_paper.pdf into a new expert system called 'benchmark'")

# Query the expert system
h.run("What geometric scoring method is used for multi-hop reasoning?")

# Search for connections between concepts
h.run("Search for paths connecting 'cayley rotor' and 'holonomy defect'")
```

#### Observability hooks

```python
import json
from knowlytix.harness import Harness, HarnessConfig
from knowlytix.harness.core import HarnessHooks

log_file = open("harness_trace.jsonl", "a")

def on_tool(name, args):
    log_file.write(json.dumps({"event": "tool_call", "tool": name, "args": args}) + "\n")
    log_file.flush()

def on_result(name, result_str, raw):
    log_file.write(json.dumps({"event": "tool_result", "tool": name, "len": len(result_str)}) + "\n")

def on_learn(n_triples, tool_name):
    log_file.write(json.dumps({"event": "learn", "triples": n_triples, "from": tool_name}) + "\n")

hooks = HarnessHooks(
    on_tool_call=on_tool,
    on_tool_result=on_result,
    on_learn=on_learn,
)

h = Harness(HarnessConfig(), hooks=hooks)
```

#### Policy enforcement via hooks

```python
BLOCKED_TOOLS = {"gms_ingest"}  # prevent document ingestion

def policy_gate(tool_name: str, args: dict) -> bool:
    """Return False to block the tool call."""
    if tool_name in BLOCKED_TOOLS:
        print(f"  [POLICY] Blocked: {tool_name}")
        return False
    return True

hooks = HarnessHooks(before_tool=policy_gate)
h = Harness(HarnessConfig(), hooks=hooks)
```

### Command-Line Interface

```bash
# Interactive mode
python -m harness.run

# Single request
python -m harness.run -r "Load BikeSharing, train XGB depth-2, diagnose accuracy"

# GMS-only (no MoDeva tools)
python -m harness.run --no-modeva

# MoDeva-only (no GMS tools)
python -m harness.run --no-gms

# Disable auto-learning into session GMS
python -m harness.run --no-learn

# Custom stores directory
python -m harness.run --stores-dir ./my_stores --session-store ./my_session

# Use a different model
python -m harness.run --model claude-sonnet-4-20250514
```

Interactive commands:

| Command | Action |
|---------|--------|
| `tools` | List all registered tools with categories |
| `stats` | Show session GMS statistics |
| `save` | Persist session GMS to disk |
| `reset` | Clear conversation history (keeps memory and state) |
| `quit` | Save and exit |

Example interactive session:

```
============================================================
  GMS + MoDeva Agent Harness
============================================================
  [Harness] Session GMS: 33 entities, 21 triples
  [Harness] Model: claude-opus-4-7
  [Harness] Device: cuda
  [Harness] Ready.
  Tools registered: 32
  Categories: gms, modeva
============================================================

Interactive mode. Type 'quit' to exit, 'reset' to clear history.
Type 'tools' to list available tools, 'stats' for GMS stats.

You> Load TaiwanCredit and train a depth-2 XGBoost classifier
  [Tool] load_dataset(name=TaiwanCredit, builtin=TaiwanCredit)
  [Tool] add_model(dataset_name=TaiwanCredit, model_type=MoXGBClassifier, ...)
  [Tool] train_model(dataset_name=TaiwanCredit, model_name=XGB2)

Agent> Dataset loaded (30000 samples, 23 features, classification task).
       Model XGB2 trained. Ready for diagnostics.

You> diagnose accuracy and fairness on SEX feature
  [Tool] diagnose_accuracy(...)
  [Memory] Learned 9 triples from diagnose_accuracy
  [Tool] diagnose_fairness(..., group_feature=SEX)
  [Memory] Learned 1 triples from diagnose_fairness

Agent> Accuracy: AUC=0.772 (test), F1=0.467. No significant overfitting.
       Fairness: AIR for SEX=2 (Female) is 0.943 — above the 0.8 threshold.
       No disparate impact detected.

You> stats
  Session: {'entities': 55, 'relations': 19, 'triples': 42, 'enm_entries': 21}

You> save
  [Harness] Session saved: 55 entities, 42 triples

You> quit
  [Harness] Session saved. Goodbye.
```

## Configuration Reference

```python
@dataclass
class HarnessConfig:
    # LLM
    model: str = "claude-opus-4-7"         # Claude model for the ReAct loop
    max_tokens: int = 4096                  # Max tokens per LLM response
    max_turns: int = 30                     # Safety limit on tool iterations
    truncate_result_at: int = 8000          # Char limit for tool results

    # GMS stores
    stores_dir: str = "gms_stores"          # GMS expert system stores directory
    session_store_path: str = "harness_session_store"  # Mutable session GMS
    modeva_store_path: str = ""             # Static MoDeva reference GMS (optional)
    gms_model_name: str = "claude-sonnet-4-20250514"   # Model for GMS operations

    # Memory
    auto_learn: bool = True                 # Learn triples from tool results
    memory_context_lines: int = 30          # Max lines injected into system prompt

    # Policy
    enable_policy: bool = True              # Enable LogicVerifier checks
    policy_behaviors_dir: str = ""          # Behavior specification markdowns

    # Context management
    max_message_tokens: int = 180_000       # Compress context beyond this
    context_summary_model: str = ""         # Reserved for future summarization
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          Harness                             │
│                                                              │
│  ┌──────────────────┐         ┌────────────────────────┐    │
│  │   ToolRegistry    │         │    GMS Memory Layer     │    │
│  │                   │         │                         │    │
│  │  gms_*      (8)  │         │  Session Store (mutable) │    │
│  │  modeva_*  (24)  │         │  MoDeva Store  (static)  │    │
│  │  memory     (1)  │         │  RuntimeLearner          │    │
│  │  custom     (N)  │         │  ExactNumericalMemory    │    │
│  └────────┬─────────┘         └──────────┬───────────────┘    │
│           │                              │                    │
│  ┌────────▼─────────┐         ┌──────────▼──────────────┐    │
│  │   ReAct Loop      │◄───────►│  Policy Verification   │    │
│  │                   │         │                         │    │
│  │  Claude API       │         │  LogicVerifier          │    │
│  │  tool_use msgs    │         │  Geodesic scoring       │    │
│  │  Context mgmt     │         │  Tension energy gates   │    │
│  └────────┬─────────┘         └──────────────────────────┘    │
│           │                                                   │
│  ┌────────▼─────────┐                                        │
│  │  HarnessHooks     │                                       │
│  │  on_tool_call     │                                       │
│  │  on_tool_result   │                                       │
│  │  on_learn         │                                       │
│  │  before_tool      │                                       │
│  └──────────────────┘                                        │
└──────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
User Request
    │
    ▼
Build system prompt (base + GMS memory context)
    │
    ▼
┌─► Claude API call (model + system + tools + messages)
│       │
│   stop_reason?
│       │
│   end_turn ──► Return final text
│       │
│   tool_use ──► For each tool call:
│                   1. before_tool hook (can skip)
│                   2. LogicVerifier policy check
│                   3. registry.execute(name, args)
│                   4. Extract triples → RuntimeLearner
│                   5. Truncate if > 8000 chars
│                   6. on_tool_result hook
│                │
│                ▼
│            Append tool_results to messages
│            Refresh system prompt (new memory)
│                │
└────────────────┘
```

### Memory Pipeline

Tool results flow through deterministic triple extraction:

| Tool Result | Extracted Triples |
|-------------|-------------------|
| `diagnose_accuracy` | `(model, has_test_R2, 0.8035)`, `(model, train_test_gap_MSE, 0.0005)` |
| `interpret_fi` | `(model, feature_importance, hr)` with score 0.7411 |
| `diagnose_robustness` | `(model, robustness_at_0.1, 0.0117)` |
| `diagnose_fairness` | `(model, fairness_SEX, 0.943)` |
| `diagnose_weakness` | `(model, has_weakness_in, feature)`, `(model, n_weak_regions, 3)` |

Extracted triples are learned via `RuntimeLearner`:
- **Phase 1 (sync)**: Add entities/relations to the graph immediately
- **Phase 2 (async)**: Background Riemannian SGD refines embeddings (~6-8s on GPU)

Numeric values are stored in both the knowledge graph (as string tails) and the Exact Numerical Memory register (lossless IEEE 754 float64 with SHA-256 integrity).

## How It Relates to Other Components

| Component | Role | When to Use |
|-----------|------|-------------|
| **Harness** | Open-ended LLM agent with pluggable tools | Flexible exploration, natural language interaction |
| **Pipeline** (`orchestrator.py`) | Deterministic phase-based workflow | Reproducible, fixed-sequence analysis |
| **DSAgent** (`agent.py`) | MoDeva-specific ReAct agent | MoDeva-only tasks (predecessor to Harness) |
| **MCP Servers** | Tool access via Claude Code | When using Claude Code as the external harness |

## Tested Workflows

The following workflows have been verified end-to-end:

**Single request (7 tool calls):**
Load BikeSharing → preprocess → add XGB2 (depth-2) → train → FANOVA interpret_fi → interpret_effects → diagnose_accuracy. Learned 21 triples. Session grew from 33 to 55 entities.

**Multi-turn with model comparison (14 tool calls across 2 turns):**
Turn 1: Load → preprocess → add XGB2 + XGB6 → train_all → compare_accuracy → compare_robustness → diagnose_robustness (x2). Turn 2 (via `chat()`): diagnose_accuracy (x2) → diagnose_reliability (x2) → diagnose_resilience (x2) → recall_memory. Final session: 76 entities, 79 triples, 55 ENM entries.

**Session persistence:**
Run 1 saves session GMS with 42 triples. Run 2 loads the same store and retrieves prior findings via `recall_memory`.
