# Harness Technical Documentation

## 1. Overview

The Harness is a standalone agent orchestration layer that owns the full execution loop for LLM-powered agents backed by Geometric Memory Systems (GMS) and MoDeva model validation tools. It replaces external harnesses (such as Claude Code) with a self-contained runtime where user code controls the agent loop, tool routing, memory integration, and policy enforcement.

### 1.1 Design Goals

1. **Self-contained execution**: The harness manages the complete ReAct (Reason-Act) cycle internally --- prompt construction, Claude API calls, tool dispatch, result processing, and memory updates --- without requiring an external orchestrator.
2. **Pluggable tool architecture**: Tools from heterogeneous sources (GMS expert systems, MoDeva diagnostics, user-defined functions) are unified through a single registry with a common interface.
3. **Geometric memory integration**: Every tool execution can produce knowledge triples that are learned into a persistent GMS store, enabling cross-session memory and contradiction detection.
4. **Policy-governed execution**: Tool calls are verified against GMS geometry via the `LogicVerifier`, which scores action plausibility on the hypersphere $S^{m-1}$.
5. **Observable and controllable**: Callback hooks at every stage (tool call, result, learning, turn boundary) allow external systems to monitor, log, or override agent behavior.

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Harness                             │
│                                                             │
│  ┌─────────────────┐        ┌───────────────────────────┐  │
│  │   ToolRegistry   │        │      GMS Memory Layer     │  │
│  │                  │        │                           │  │
│  │  gms_*     (8)  │        │  Session Store (mutable)  │  │
│  │  modeva_*  (24) │        │  MoDeva Store  (static)   │  │
│  │  memory    (1)  │        │  RuntimeLearner           │  │
│  │  custom    (N)  │        │  ExactNumericalMemory     │  │
│  └────────┬────────┘        └─────────────┬─────────────┘  │
│           │                               │                 │
│  ┌────────▼────────┐        ┌─────────────▼─────────────┐  │
│  │   ReAct Loop     │◄──────►│   Policy Verification    │  │
│  │                  │        │                           │  │
│  │  Claude API      │        │  LogicVerifier            │  │
│  │  tool_use        │        │  Geodesic scoring         │  │
│  │  Context mgmt    │        │  Tension energy gates     │  │
│  └────────┬────────┘        └───────────────────────────┘  │
│           │                                                 │
│  ┌────────▼────────┐                                       │
│  │   HarnessHooks   │                                      │
│  │  on_tool_call    │                                      │
│  │  on_tool_result  │                                      │
│  │  on_learn        │                                      │
│  │  before_tool     │                                      │
│  └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Public API: `Harness`, `HarnessConfig`, `ToolRegistry`, `ToolDef` |
| `core.py` | Harness class --- ReAct loop, memory integration, policy, context management |
| `registry.py` | Unified tool registry --- GMS, MoDeva, and custom tool registration |
| `run.py` | CLI entry point --- interactive and single-request modes |
| `__main__.py` | Module entry point (`python -m harness`) |

---

## 2. ToolRegistry

The `ToolRegistry` provides a uniform interface over tools from multiple sources. Each tool is represented as a `ToolDef` dataclass:

```python
@dataclass
class ToolDef:
    name: str                                            # Tool identifier
    description: str                                     # LLM-visible description
    input_schema: dict                                   # JSON Schema for parameters
    handler: Callable[[dict, Any], tuple[str, Any|None]] # Execution function
    category: str = "custom"                             # Grouping label
```

### 2.1 Handler Contract

Every tool handler receives two arguments and returns a 2-tuple:

```python
def handler(args: dict, context: Any) -> tuple[str, Any | None]:
```

| Parameter | Description |
|-----------|-------------|
| `args` | Dictionary of tool arguments as provided by the LLM (matches `input_schema`) |
| `context` | Opaque context object passed through from `registry.execute()`. For MoDeva tools this is the `SessionState`; for GMS tools the context is unused (state is captured via closure). |
| **Return `[0]`** | String result sent back to the LLM as `tool_result` content |
| **Return `[1]`** | Optional raw result object used for triple extraction. `None` if the tool produces no structured output suitable for GMS learning. |

The raw result (index 1) is critical for the memory pipeline. For MoDeva tools, this is typically a `ValidationResult` object whose `.table`, `.value`, and `.key` attributes are inspected by `extract_triples_from_result()` to produce `ExtractedTriple` instances.

### 2.2 Bulk Registration

#### MoDeva Tools

`register_modeva_tools(state: SessionState)` iterates over the 24 tool definitions in `knowlytix.harness._modeva_tools.TOOL_DEFINITIONS` and creates a handler closure for each that delegates to `execute_tool(name, args, state)`. The `recall_memory` pseudo-tool is excluded --- it is handled directly by the harness. (MoDeva itself is an optional dep; if `modeva` isn't installed the tool list registers but invocation returns a clear "MoDeva not installed" error.)

The closure pattern ensures each handler captures its own tool name:

```python
def _make_handler(tool_name: str):
    def handler(args, ctx):
        return modeva_execute(tool_name, args, state)
    return handler
```

#### GMS Tools

`register_gms_tools(stores_dir, device, model_name)` creates an in-process equivalent of the `docgms.mcp_server` MCP server. Instead of communicating over stdio transport, all GMS operations execute directly in the harness process.

Shared mutable state is captured in a `gms_state` dictionary that mirrors the MCP server's `_active` module-level state:

```python
gms_state = {
    "name": None,          # Active expert system name
    "store": None,         # GMSExpertStore instance
    "engine": None,        # QueryEngine instance
    "config": None,        # DocGMSConfig instance
    "llm": None,           # LLMBackend instance
    "stores_dir": "...",   # Absolute path to stores directory
    "device": device,      # torch.device
    "model_name": "...",   # Claude model for LLM-based operations
}
```

Auto-loading behavior: if exactly one store exists in `stores_dir`, it is loaded automatically at registration time. If multiple stores exist, the most recently saved one is loaded.

### 2.3 Registered Tools

After calling both `register_gms_tools()` and `register_modeva_tools()`, the registry contains 33 tools (32 registered + 1 `recall_memory` added at first `run()` call):

| Category | Count | Tools |
|----------|-------|-------|
| `gms` | 8 | `gms_create`, `gms_list`, `gms_use`, `gms_ingest`, `gms_query`, `gms_search`, `gms_lookup`, `gms_status` |
| `modeva` | 24 | `load_dataset`, `data_summary`, `eda_correlation`, `preprocess_data`, `feature_engineer`, `feature_select`, `add_model`, `train_model`, `train_all_models`, `leaderboard`, `diagnose_accuracy`, `diagnose_robustness`, `diagnose_reliability`, `diagnose_resilience`, `diagnose_fairness`, `diagnose_weakness`, `explain_pfi`, `explain_pdp`, `explain_shap`, `explain_ale`, `compare_accuracy`, `compare_robustness`, `interpret_fi`, `interpret_effects` |
| `memory` | 1 | `recall_memory` |

### 2.4 Output Format

`to_anthropic_tools()` produces the Anthropic Messages API tool format:

```python
[
    {
        "name": "tool_name",
        "description": "Tool description for the LLM",
        "input_schema": { ... }   # JSON Schema
    },
    ...
]
```

This list is passed directly to `client.messages.create(tools=...)`.

---

## 3. Harness Core

### 3.1 Configuration

`HarnessConfig` is a dataclass with the following groups:

#### LLM Settings

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `"claude-opus-4-7"` | Claude model identifier for the ReAct loop |
| `max_tokens` | `4096` | Maximum tokens per LLM response |
| `max_turns` | `30` | Safety limit on tool call iterations per `run()` |
| `truncate_result_at` | `8000` | Character limit for tool results sent back to the LLM |

#### GMS Store Settings

| Field | Default | Description |
|-------|---------|-------------|
| `stores_dir` | `"gms_stores"` | Directory containing GMS expert system stores |
| `session_store_path` | `"harness_session_store"` | Path for the mutable session GMS store |
| `modeva_store_path` | `""` | Path to a pre-trained MoDeva reference store (static, read-only) |
| `gms_model_name` | `"claude-sonnet-4-20250514"` | Claude model used for GMS operations (triple extraction, query grounding) |

#### Memory Settings

| Field | Default | Description |
|-------|---------|-------------|
| `auto_learn` | `True` | Automatically extract and learn triples from tool results |
| `memory_context_lines` | `30` | Maximum lines of memory context injected into system prompt |

#### Policy Settings

| Field | Default | Description |
|-------|---------|-------------|
| `enable_policy` | `True` | Enable `LogicVerifier`-based policy checking on tool calls |
| `policy_behaviors_dir` | `""` | Directory containing behavior specification markdowns |

#### Context Management

| Field | Default | Description |
|-------|---------|-------------|
| `max_message_tokens` | `180,000` | Estimated token threshold triggering context compression |
| `context_summary_model` | `""` | Reserved for future LLM-based summarization |

### 3.2 Initialization

On construction, the harness performs:

1. **Device detection**: CUDA if available, else CPU.
2. **Anthropic client**: Created from `ANTHROPIC_API_KEY` environment variable.
3. **Tool registry**: Empty; tools are registered via `register_gms_tools()` and `register_modeva_tools()`.
4. **Session state**: A `SessionState` dataclass holding live MoDeva objects (`datasets`, `model_zoos`, `trained_models`, `results`).
5. **Session GMS**: Loaded from `session_store_path`. If no store exists, `bootstrap_session_store()` creates a minimal GMS with 21 seed triples covering common data science concepts and relations, then trains initial embeddings on GPU (200 epochs, ~2s on NVIDIA GB10).
6. **MoDeva GMS** (optional): If `modeva_store_path` points to a valid store, it is loaded as a static reference store. The harness queries it for methodology context but never writes to it.

### 3.3 ReAct Loop

The `run(user_request)` method implements the core agent loop:

```
                          ┌──────────────────┐
                          │  Build System    │
                          │  Prompt + GMS    │
                          │  Memory Context  │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                     ┌───►│  Claude API Call  │
                     │    │  messages.create  │
                     │    └────────┬─────────┘
                     │             │
                     │    ┌────────▼─────────┐
                     │    │  stop_reason?     │
                     │    └────────┬─────────┘
                     │             │
                     │     ┌───────┴──────┐
                     │     │              │
                     │  end_turn      tool_use
                     │     │              │
                     │     ▼              ▼
                     │  Return      ┌─────────────┐
                     │  final       │ For each     │
                     │  text        │ tool_use     │
                     │              │ block:       │
                     │              │              │
                     │              │ 1. Hook check│
                     │              │ 2. Policy    │
                     │              │ 3. Execute   │
                     │              │ 4. Learn     │
                     │              │ 5. Truncate  │
                     │              └──────┬──────┘
                     │                     │
                     │              ┌──────▼──────┐
                     │              │ Append       │
                     │              │ tool_results │
                     │              │ to messages  │
                     │              └──────┬──────┘
                     │                     │
                     │              ┌──────▼──────┐
                     │              │ Refresh      │
                     │              │ system prompt│
                     └──────────────┤ (new memory) │
                                    └─────────────┘
```

**Per-turn sequence for each tool call:**

1. **Hook check**: If `hooks.before_tool` is set and returns `False`, the tool call is skipped with a "skipped by policy hook" message.
2. **Policy verification**: `LogicVerifier.verify_action(context, tool_name)` scores the geodesic plausibility of calling this tool in the current context. Implausible calls are logged but not blocked (the LLM may have valid novel reasons).
3. **Execution**: `registry.execute(tool_name, args, context)` dispatches to the appropriate handler and catches exceptions.
4. **Learning**: If `auto_learn` is enabled and the handler returned a non-`None` raw result, `extract_triples_from_result()` produces `ExtractedTriple` instances that are passed to `RuntimeLearner.learn_triples()`. Learning is two-phase: Phase 1 (synchronous) adds entities/relations to the graph; Phase 2 (asynchronous background thread) runs Riemannian gradient descent on new embeddings.
5. **Truncation**: Results exceeding `truncate_result_at` characters are cut with a `[truncated]` marker.
6. **System prompt refresh**: After processing all tool calls in a turn, the system prompt is rebuilt with fresh GMS memory context reflecting any newly learned triples.

### 3.4 System Prompt Construction

The system prompt is assembled from three components:

1. **Base prompt**: Static instructions defining the agent's capabilities, principles (exact metrics, flag overfitting, FANOVA requires depth-2), and memory system description.
2. **MoDeva knowledge context**: Triples from the static MoDeva GMS store, queried by dataset name, model names, and keywords from the user request. Formatted as `[entity] relation → tail` lines inside a code block.
3. **Session memory context**: Triples from the mutable session GMS store, queried the same way. This contains findings from prior experiments (accuracy metrics, robustness scores, weakness regions, etc.).

Context gathering uses `gather_memory_context()` from `knowlytix.harness.memory`, which performs:
- Entity lookup for known dataset and model names
- Fuzzy matching for content words from the user question (words >3 chars, excluding stop words)
- ENM (Exact Numerical Memory) value lookup for matched entities
- Truncation to `memory_context_lines` (default 30)

### 3.5 The `recall_memory` Meta-Tool

The LLM can explicitly query memory mid-conversation by calling the `recall_memory` tool. This is handled specially by the harness (not dispatched through the registry) and returns formatted context from both GMS stores:

```
MODEVA KNOWLEDGE:
[entity] relation → tail
...

SESSION MEMORY:
[entity] relation → tail
...
```

This allows the LLM to selectively retrieve relevant knowledge when the automatically-injected system prompt context is insufficient.

### 3.6 Context Compression

When estimated token count exceeds `max_message_tokens` (default 180K), the harness compresses the conversation history:

1. Estimate tokens at ~4 characters per token across all messages.
2. If over threshold and more than 4 messages exist, retain the first message and last 4 messages.
3. Prepend a summary marker indicating compression occurred.

This is a simple windowing strategy. The `context_summary_model` field is reserved for future LLM-based summarization that would produce a semantic summary of dropped messages.

---

## 4. Memory Pipeline

### 4.1 Triple Extraction

Tool results flow through a deterministic (no LLM) triple extraction pipeline defined in `knowlytix.harness.memory.extract_triples_from_result()`. Extraction is keyed on the `ValidationResult.key` attribute:

| Result Key Pattern | Extracted Triples |
|-------------------|-------------------|
| `accuracy` (not compare) | `(model, has_{split}_{metric}, value)` for each metric/split; `(model, train_test_gap_{metric}, gap)` |
| `compare_accuracy` | `(model, has_test_{metric}, value)` per model; `(dataset, best_model_by_{metric}, model)` |
| `robustness` | `(model, robustness_at_{noise_level}, mean_score)` |
| `reliability` | `(model, reliability_{col}, value)` from conformal prediction table |
| `resilience` | `(model, resilience_worst_{col}, value)` from worst-subset row |
| `fairness` | `(model, fairness_{group}, metric_value)` |
| `weakness` | `(model, has_weakness_in, feature)` with JS divergence scores; `(model, n_weak_regions, count)` |
| `pfi`, `interpret_fi` | `(model, feature_importance, feature_name)` with importance scores |
| `summary` | `(dataset, has_features, count)`, `(dataset, summary_{key}, value)` |

All numeric values are stored in both the triple tail (as string) and the `ExtractedTriple.numeric_value` field for ENM storage.

### 4.2 RuntimeLearner

`RuntimeLearner` from `docgms.learn` handles two-phase learning:

- **Phase 1 (synchronous)**: New entities and relations are added to the `DocumentGraph`. New entity embeddings are initialized near related existing embeddings. The graph adjacency is updated. This completes before the next LLM turn.
- **Phase 2 (asynchronous)**: A background thread runs Riemannian SGD (sphere retraction + Stiefel retraction + skew-symmetric retraction) to refine embeddings for the newly added subgraph. This runs concurrently with the next LLM turn.

The `save()` method flushes pending Phase 2 training and persists the store to disk.

### 4.3 Dual-Store Architecture

| Store | Mutability | Purpose |
|-------|-----------|---------|
| **Session store** | Read-write | Accumulates knowledge from tool results across conversations. Grows monotonically. Used for experiment memory (metrics, weaknesses, model comparisons). |
| **MoDeva store** | Read-only | Pre-trained from MoDeva documentation. Provides methodology context (API usage, best practices, diagnostic interpretation). Never modified by the harness. |

Both stores are queried for system prompt context and `recall_memory` responses. Only the session store receives learned triples.

---

## 5. Policy Verification

### 5.1 LogicVerifier Integration

When `enable_policy` is `True`, the harness initializes a `LogicVerifier` backed by the session GMS store. On each tool call, `verify_action(context_entity, tool_name)` is invoked:

1. The current dataset name (or `"agent"` if none) is used as the context entity.
2. The tool name is used as the tail entity.
3. The verifier computes `score_triple(context, "should_call", tool)` --- the geodesic distance on $S^{m-1}$ between the context entity (rotated by the `should_call` relation rotor) and the tool entity.
4. If `score > min_plausibility` (from `VerifyConfig`), the tool call is flagged as implausible and logged.

**Current behavior**: Implausible tool calls are logged but not blocked. This is intentional --- the LLM may invoke tools in novel contexts not yet represented in the GMS. Over time, as the session store accumulates more triples linking contexts to tools, the policy becomes more informed.

### 5.2 Hook-Based Control

For harder policy enforcement, use the `before_tool` hook:

```python
def my_policy(tool_name: str, args: dict) -> bool:
    """Return False to block the tool call."""
    if tool_name == "gms_ingest" and "untrusted" in args.get("document_path", ""):
        return False
    return True

harness = Harness(hooks=HarnessHooks(before_tool=my_policy))
```

When `before_tool` returns `False`, the tool call is skipped and the LLM receives "Tool call skipped by policy hook." as the result.

---

## 6. Callback Hooks

`HarnessHooks` provides five optional callbacks:

| Hook | Signature | When Called |
|------|-----------|------------|
| `on_tool_call` | `(name: str, args: dict) -> None` | After a tool call is identified, before execution |
| `on_tool_result` | `(name: str, result: str, raw: Any) -> None` | After tool execution and learning |
| `before_tool` | `(name: str, args: dict) -> bool` | Before execution; return `False` to skip |
| `on_learn` | `(n_triples: int, tool_name: str) -> None` | After triples are successfully learned |
| `on_turn` | `(turn: int, phase: str) -> None` | At the start of each ReAct turn |

Example: logging all tool calls to a file:

```python
import json

log = open("harness_trace.jsonl", "a")

def trace_tool(name, args):
    log.write(json.dumps({"tool": name, "args": args}) + "\n")
    log.flush()

harness = Harness(hooks=HarnessHooks(on_tool_call=trace_tool))
```

---

## 7. Usage

### 7.1 Programmatic API

```python
from knowlytix.harness import Harness, HarnessConfig

# Configure
config = HarnessConfig(
    model="claude-opus-4-7",
    stores_dir="gms_stores",
    session_store_path="my_session",
    modeva_store_path="modeva_gms_store",  # optional
    auto_learn=True,
)

# Initialize
h = Harness(config)
h.register_gms_tools()
h.register_modeva_tools()

# Single request
response = h.run(
    "Load BikeSharing, train XGB depth-2, run FANOVA interpretation, "
    "then check robustness at noise levels 0.05, 0.1, 0.2"
)
print(response)

# Multi-turn conversation
response2 = h.chat("Now compare with a full-depth XGBoost model")
response3 = h.chat("Which model has better fairness on the Age feature?")

# Save session memory
h.save()

# Reset conversation (keeps memory)
h.reset()
```

### 7.2 Custom Tools

```python
def fetch_data_from_api(args: dict, ctx) -> tuple[str, None]:
    """Custom tool: fetch data from an internal API."""
    import requests
    url = args["url"]
    resp = requests.get(url)
    return resp.text[:5000], None

h.register_tool(
    name="fetch_api_data",
    description="Fetch data from an internal REST API endpoint.",
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "API endpoint URL"},
        },
        "required": ["url"],
    },
    handler=fetch_data_from_api,
    category="custom",
)
```

### 7.3 CLI

```bash
# Interactive mode
python -m harness.run

# Single request
python -m harness.run -r "Load TaiwanCredit, train XGB depth-2, diagnose accuracy"

# GMS-only (no MoDeva tools)
python -m harness.run --no-modeva

# MoDeva-only (no GMS tools)
python -m harness.run --no-gms

# Disable auto-learning
python -m harness.run --no-learn

# Custom stores
python -m harness.run --stores-dir ./my_stores --session-store ./my_session
```

Interactive commands:

| Command | Action |
|---------|--------|
| `tools` | List all registered tools with categories |
| `stats` | Show session GMS statistics |
| `save` | Persist session GMS to disk |
| `reset` | Clear conversation history (keeps memory and state) |
| `quit` | Save and exit |

---

## 8. Relationship to Existing Components

### 8.1 vs. DSAgent (`agents/ds_agent/agent.py`)

The `DSAgent` is a purpose-built data science agent with a hardcoded MoDeva tool set. The Harness generalizes this:

| Aspect | DSAgent | Harness |
|--------|---------|---------|
| Tools | 24 MoDeva + recall_memory (fixed) | Pluggable registry (GMS + MoDeva + custom) |
| Memory | Session + MoDeva stores | Same dual-store architecture |
| Policy | None | LogicVerifier + before_tool hook |
| Context mgmt | None | Token estimation + compression |
| Hooks | None | 5 callback hooks |
| Entry point | `DSAgent.run(request)` | `Harness.run(request)` |

### 8.2 vs. Pipeline (`agents/ds_agent/orchestrator.py`)

The `Pipeline` is a deterministic phase-based orchestrator (Data → Training → Interpret → Diagnose → Weakness → Fairness → Compare) with fixed control flow. The Harness is an open-ended agent --- the LLM decides which tools to call and in what order. They are complementary:

- Use **Pipeline** when you want reproducible, deterministic workflows with full tracing.
- Use **Harness** when you want flexible, LLM-guided exploration with natural language interaction.

### 8.3 vs. MCP Servers (`docgms/mcp_server.py`, `agents/ds_agent/mcp_server.py`)

The MCP servers expose the same tools over the Model Context Protocol stdio transport for use with Claude Code. The Harness registers equivalent tools in-process, avoiding the serialization overhead and process boundary of MCP. The tool logic is identical --- the Harness reuses the same underlying functions.

| Aspect | MCP Servers | Harness |
|--------|-------------|---------|
| Transport | stdio (JSON-RPC) | In-process function calls |
| Orchestrator | Claude Code (external) | `Harness.run()` (internal) |
| Memory | Not integrated | Session GMS + auto-learning |
| Policy | Not integrated | LogicVerifier + hooks |

---

## 9. Dependencies

| Package | Usage |
|---------|-------|
| `anthropic` | Claude Messages API client |
| `torch` | GPU device management, GMS model operations |
| `docgms` | GMS expert system store, query engine, runtime learner, LLM backend |
| `knowlytix.harness.memory` + `knowlytix.harness._modeva_tools` + `knowlytix.harness._verifier` | MoDeva tool definitions, session state, memory extraction, logic verification |
| `gms` | Core geometric library (embeddings, Cayley rotors, transport, search) |
| `modeva` | Model development and validation toolkit (DataSet, ModelZoo, TestSuite) |

The `ANTHROPIC_API_KEY` environment variable must be set for the Claude API client.

---

## 10. Data Flow Diagram

```
User Request (natural language)
    │
    ▼
┌───────────────────────────────┐
│ System Prompt Construction    │
│                               │
│ Base instructions             │
│ + MoDeva GMS context          │◄── MoDeva Store (read-only)
│ + Session GMS context         │◄── Session Store (read)
│ + Tool categories             │◄── ToolRegistry
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Claude API                    │
│ model + system + tools + msgs │
└───────────────┬───────────────┘
                │
        ┌───────┴───────┐
        │               │
    end_turn         tool_use
        │               │
        ▼               ▼
    Return text    ┌─────────────────────┐
                   │ Tool Dispatch       │
                   │                     │
                   │ 1. before_tool hook │
                   │ 2. LogicVerifier    │
                   │ 3. registry.execute │──► Handler function
                   │ 4. Raw result       │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Memory Pipeline     │
                   │                     │
                   │ extract_triples()   │
                   │ RuntimeLearner      │──► Session Store (write)
                   │ Phase 1: sync add   │
                   │ Phase 2: async SGD  │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Result Processing   │
                   │                     │
                   │ Truncate if > 8000  │
                   │ on_tool_result hook │
                   │ Append to messages  │
                   └─────────┬───────────┘
                             │
                             ▼
                   Refresh system prompt
                   (with new memory)
                             │
                             ▼
                   Next Claude API call
```
