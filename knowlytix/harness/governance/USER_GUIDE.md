# Knowlytix Harness Governance -- User Guide

This guide walks through the `knowlytix.harness.governance` subpackage
(shipped inside the `knowlytix-harness` wheel as of Phase 4.10), a
policy-governed framework for trustworthy agentic AI built on Geometric
Expert Systems (GMS).  Every governance decision reduces to a geometric
computation on the unit sphere $S^{m-1}$, producing auditable evidence
rather than probabilistic guardrails.

> **Runnable companions**: each numbered section below has a Jupyter
> notebook that walks through the same material end-to-end against the
> bundled fixture stores. They live in the public source mirror at
> [`knowlytix/gmsh`](https://github.com/knowlytix/gmsh) under
> `tutorials/user-guide/notebooks/` (one per section, `01_prerequisites.ipynb`
> through `27_*.ipynb`) and accompany the same `_shared.py` runtime helper
> + a `tutorials/user-guide/fixtures/` directory carrying the prebuilt
> `ds_practices_store` and `california_housing_store` so chapters chain
> off without rebuilding. Wheel-only customers can clone the mirror
> separately to run them; see the mirror's `tutorials/README.md` for
> the setup recipe.

---

## Contents

- [§1. Prerequisites](#1-prerequisites)
- [§2. Architecture overview](#2-architecture-overview)
- [§3. Quick start](#3-quick-start)
- [§4. GMS primitives reference](#4-gms-primitives-reference)
- [§5. GMS diagnostics](#5-gms-diagnostics)
- [§6. Knowledge ingestion](#6-knowledge-ingestion)
- [§7. Three-tier extraction](#7-three-tier-extraction)
- [§8. Authoring knowledge documents](#8-authoring-knowledge-documents)
- [§9. Tool catalog](#9-tool-catalog)
- [§10. AgentMemory](#10-agentmemory)
- [§11. GeometricReasoner](#11-geometricreasoner)
- [§12. Claim verification](#12-claim-verification)
- [§13. Runtime learning](#13-runtime-learning)
- [§14. Workflow planner](#14-workflow-planner)
- [§15. Constrained LLM (mock and real)](#15-constrained-llm-mock-and-real)
- [§16. LLM backends and API keys](#16-llm-backends-and-api-keys)
- [§17. Governed Tool Gateway](#17-governed-tool-gateway)
- [§18. Behavioral contracts](#18-behavioral-contracts)
- [§19. Policy engine and calibrated policy triple](#19-policy-engine-and-calibrated-policy-triple)
- [§20. Severity and FailureCode taxonomy](#20-severity-and-failurecode-taxonomy)
- [§21. Orchestrator](#21-orchestrator)
- [§22. Multi-agent](#22-multi-agent)
- [§23. Telemetry and tracing](#23-telemetry-and-tracing)
- [§24. Release gates and SPC](#24-release-gates-and-spc)
- [§25. Threshold calibration](#25-threshold-calibration)
- [§26. Runtime governance and bundles](#26-runtime-governance-and-bundles)
- [§27. Compliance](#27-compliance)
- [§28. Testing your agent](#28-testing-your-agent)
- [§29. Common patterns](#29-common-patterns)
- [§30. Mathematical appendix](#30-mathematical-appendix)
- [§31. Companion resources](#31-companion-resources)


---

## 1. Prerequisites

| Dependency | Purpose |
|-----------|---------|
| Python 3.12+ | Runtime |
| PyTorch 2.x (CUDA recommended) | GMS embeddings and training |
| `knowlytix-knowledge` | GMS expert store, runtime learner, extraction |
| `gms` | Geometric search, transport, ENM |
| `harness` | Policy engine, tool gateway, Design-of-Experiments (DOE) generators |
| scikit-learn | Model training and evaluation tools |

Activate the environment before running any code:

```bash
source ~/cluster/spark-venv/bin/activate
```

---

## 2. Architecture overview

Governance is layered because no single check can absorb every
failure mode an agent encounters. A schema gate cannot catch a
hallucinated fact, a fact verifier cannot catch a finite-state-machine
(FSM) illegal tool ordering, and an FSM cannot catch a quietly
drifting calibration. The
six layers below stack from cheap-fast structural checks to
geometry-driven evidence; together they produce one auditable trail
for every tool call.

### 2.1 Six-layer stack

```
Layer 1  Policy rules         PolicyEngine encodes responsible-ML rules
Layer 2  Formal contracts     BehavioralContract FSMs enforce workflows
Layer 3  Tool definitions     30+ sklearn tools with JSON schemas
Layer 4  Extraction           Three-tier knowledge extraction
Layer 5  Memory + reasoning   GMS store, AgentMemory, GeometricReasoner
Layer 6  Governance stack     Verification, tracing, SPC gates, compliance
```

Layers 1-3 are declarative: they refuse anything not listed. Layer 4
turns every tool result into knowledge triples + ENM entries. Layer 5
turns those triples into geometric evidence on $S^{m-1}$. Layer 6
records the evidence chain and runs release-time checks.

### 2.2 Three gates inside the GovernedToolGateway

Every tool call passes through `GovernedToolGateway.execute`, which
runs three gates in sequence and short-circuits on the first failure:

```
                       GovernedToolGateway.execute(tool, args, context)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  ┌──────────┐              ┌───────────┐            ┌──────────────┐
  │  Gate 1  │   pass       │   Gate 2  │   pass     │    Gate 3    │
  │  Schema  │ ───────────▶ │   Policy  │ ─────────▶ │ Plausibility │
  │ (JSON)   │              │  (rules)  │            │  (geodesic)  │
  └────┬─────┘              └─────┬─────┘            └──────┬───────┘
       │ fail                     │ fail                    │ fail
       ▼                          ▼                         ▼
   blocked=True              blocked=True             blocked=True
   reason="schema"           reason="policy:<rule>"   reason="plausibility"
```

| Gate | Check | Source of truth |
|------|-------|-----------------|
| Schema | type and required parameters against the tool's JSON schema | `TOOL_DEFINITIONS` |
| Policy | declarative rules (allowed tools, role boundaries, contradictions) | `PolicyEngine.check_all` |
| Plausibility | geodesic distance $d(R_r v_h, v_t)$ for `(context, should_call, tool)` | GMS store, threshold $\theta$ |

A `GatewayResult` carries `gate_results`, `policy_decision`,
`plausibility_score`, and `reason`, so a reviewer can read the full
gate-by-gate outcome from a single object.

> See `user_guide_tutorials/notebooks/02_architecture.ipynb` for an
> interactive walk through the layer stack and gate flow.

---

## 3. Quick start

> **Tip — skip the build.** A prebuilt fixture lives at
> `user_guide_tutorials/fixtures/ds_practices_store` (200 epochs over
> `data_science_practices.md`, $m=128$, 244 entities, 38 relations,
> 334 triples, 24 ENM entries). Every code example in §3-§5 of this
> guide runs against that fixture. To follow along without building
> from scratch:
>
> ```python
> from user_guide_tutorials.notebooks._shared import (
>     load_ds_practices_store,
> )
> store = load_ds_practices_store()
> print(store.stats())
> ```
>
> The fixture is rebuilt by
> `python -m user_guide_tutorials.scripts.build_fixture`. Identity
> and provenance live in `user_guide_tutorials/fixtures/MANIFEST.json`.

### 3.1 Build a knowledge store

```python
import warnings, sys, os
warnings.filterwarnings("ignore")
os.chdir("/path/to/kg-memory")
sys.path.insert(0, ".")

from docknowlytix.core.config import DocGMSConfig
from docknowlytix.core.store import GMSExpertStore
from knowlytix.core.config import GeometryConfig, TrainConfig

DEVICE = "cuda"
config = DocGMSConfig(
    store_path="my_store",
    knowledge_path="knowlytix/harness/governance/knowledge/data_science_practices.md",
    geometry=GeometryConfig(m=128),
    train=TrainConfig(epochs=200, device=DEVICE),
)
store = GMSExpertStore(config, device=DEVICE)
print(store.stats())
```

`store.stats()` returns a dict with `entities`, `relations`,
`triples`, `enm_entries`, `documents`, and `store_path`. If the
adapter holds entities the model has not yet embedded, the dict
also reports `unembedded_entities`.

### 3.1.1 Ingestion Modes — `regex` / `hybrid` / `llm_only`

Section 3.1 above creates an empty store. To populate it from a
markdown document, use `docknowlytix.core.ingest.ingest_document`. The
configuration knob is `DocGMSConfig.ingest_mode`:

```python
from docknowlytix.core.config import DocGMSConfig
from docknowlytix.core.llm_backend import AnthropicBackend
from docknowlytix.core.ingest import ingest_document

config = DocGMSConfig(
    store_path="my_store",
    ingest_mode="hybrid",   # default — see table below
)
llm = AnthropicBackend()    # required when ingest_mode != "regex"
result = ingest_document(
    store, "report.md", llm=llm, config=config, device=DEVICE,
)
```

| Mode | LLM | Recall vs regex | When to use |
|------|-----|-----------------|-------------|
| `"regex"` | not invoked | baseline (100%) | reproducible CI builds, offline fixtures |
| `"hybrid"` (default) | required | strict superset; +22% ENM, +14% triples; **0** value drift on shared keys | production |
| `"llm_only"` | required | ≈ 12.5% ENM recall, near-zero overlap | paper reproduction, debugging baseline |

**Misuse raises.** If `ingest_mode` is `"hybrid"` or `"llm_only"` and
`llm` is `None`, `ingest_document` raises `ValueError` rather than
silently falling back. Pass an `LLMBackend`, or set
`ingest_mode="regex"` if the large language model (LLM) is
genuinely unavailable.
Audit-defensible deployments should always set the mode explicitly
so a reviewer can tell from the configuration which path produced
the store.

The numeric-value contract is preserved across modes: every shared ENM
key carries the same SHA-256-hashed IEEE-754 value. The LLM
contributes only structural decisions (column classification, entity
detection) and prose-derived triples — never numbers that bypass the
register.

For a one-call alternative when you have only a string callable
(e.g. wrapping a non-Anthropic provider), use the standalone parser
directly:

```python
from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.benchmark.llm_caller import make_anthropic_callable

llm_callable = make_anthropic_callable()        # or any Callable[[str], str]
graph = ingest_markdown("report.md", mode="hybrid", llm_callable=llm_callable)
```

### 3.2 Set Up Session and Gateway

§3.1 produced a `GMSExpertStore` — the geometric memory the gates will
score against. The next two objects sit between an agent's intent
("call `load_dataset`") and the tool's effect: a session that
remembers per-run state, and a gateway that enforces the three gates
from §2.2 on every call.

```python
from knowlytix.harness.governance.session import SessionState
from knowlytix.harness.governance.governed_gateway import GovernedToolGateway
from knowlytix.harness.governance.tools import TOOL_DEFINITIONS

session = SessionState()
gateway = GovernedToolGateway(
    session=session,
    store=store,
    theta_plausibility=1.5,
    tool_schemas=TOOL_DEFINITIONS,
    device=DEVICE,
)
```

`SessionState` is a plain dataclass with five dicts — `datasets`,
`models`, `results`, `preprocessors`, `predictions` — that survive
across tool calls within a single run. `load_dataset` writes into
`session.datasets`, `train_model` reads from there and writes into
`session.models`, `predict` reads both. Nothing else in the stack
shares mutable state between calls; every other object in this
section either reads the session or is stateless.

`GovernedToolGateway` is the single chokepoint for tool dispatch.
Every call to `execute` runs schema validation against
`tool_schemas`, then a policy check against the rules registered on
its internal `PolicyEngine`, then a geodesic plausibility check
against `store` with `theta_plausibility` as the cut. The default
$\theta=1.5$ matches the calibrated policy threshold from
`gms_agent_builder_ext/chapters/ch07.tex` §7.4; §25 shows how to
derive a tighter cut from a labeled cohort. The full constructor
surface and the `GatewayResult` schema live in §17.

### 3.3 Execute a Tool Through the Gateway

A tool call is a single method invocation; everything that follows
— gate verdicts, audit fields, fault injection — is recorded on the
returned `GatewayResult`.

```python
result = gateway.execute("load_dataset", {"name": "iris"})
print(f"Blocked: {result.blocked}")
print(result.display)
```

`result.blocked` is the headline verdict. When it is `True`,
`result.reason` carries a one-line explanation naming the gate that
fired (`"Schema validation failed: …"`, `"Policy violation: Rule …"`,
`"Implausible tool call: …"`); `result.display` is empty because the
tool never ran. When `blocked` is `False`, `result.display` carries
the tool's human-readable summary and `result.raw` carries the
pure-Python dict that Tier-1 extraction (§7) reads. The full
eight-field schema — including `gate_results`, `policy_decision`,
`plausibility_score`, and `execution_time_ms` — is documented in
§17.3.

#### Worked example: an allowed call and a blocked call

The clearest way to see the gates is to run two calls back to back —
one that passes all three, and one the schema gate rejects before the
tool can execute. Both calls hit the same gateway built in §3.2; the
only difference is the arguments passed in.

```python
# (1) Allowed — load_dataset with the required `name` parameter.
allowed = gateway.execute(
    "load_dataset",
    {"name": "iris", "builtin": "iris"},
    context="data_loading",
)
print(f"blocked       = {allowed.blocked}")
print(f"reason        = {allowed.reason}")
print(f"gate_results  = {allowed.gate_results}")
print(f"display       = {allowed.display}")
```

Expected output:

```
blocked       = False
reason        = All gates passed
gate_results  = {'schema': True, 'policy': True, 'plausibility': True, 'fault_injection': True}
display       = Loaded 'iris': 150 samples, 4 features, target='target'
```

Every gate passed, so `execute_tool` ran and the dataset is now in
`session.datasets["iris"]`. The next call uses the same gateway but
omits a required argument:

```python
# (2) Blocked — train_model is called with no arguments.
#     The schema gate rejects before the tool runs.
blocked = gateway.execute(
    "train_model",
    {},                       # missing dataset, model_type, model_name
    context="training",
)
print(f"blocked       = {blocked.blocked}")
print(f"reason        = {blocked.reason}")
print(f"gate_results  = {blocked.gate_results}")
print(f"display       = {blocked.display!r}")
```

Expected output:

```
blocked       = True
reason        = Schema validation failed: Missing required parameter: dataset
gate_results  = {'schema': False}
display       = ''
```

Three details to notice. First, `gate_results` carries only one entry
— `schema: False` — because the gateway short-circuits on the first
failing gate; the policy and plausibility gates are not consulted at
all. Second, `display` is empty because the tool was never executed,
so nothing was added to `session.models`. Third, the reason names the
specific missing parameter, so a reviewer reading the audit log can
see exactly what the caller got wrong without re-deriving it from
the schema.

For the other two block reasons — `Policy violation: …` (a rule
fired in gate 2) and `Implausible tool call: distance(…) > theta=…`
(geodesic distance exceeded `theta_plausibility` in gate 3) — see
§17.4, which shows how to register a custom rule and how to tighten
the plausibility threshold against a held-out cohort. The two
calls above are also the runnable contents of notebook
`03_quick_start.ipynb`; notebook `16_tool_gateway.ipynb` extends
the same pattern with one demo cell per gate (schema, policy,
plausibility) plus a clean call for contrast.

### 3.4 Run a Full Governed Pipeline

The gateway in §3.2 governs *one* tool call. A real workflow is a
*sequence* of calls under a behavioral contract, with findings
written to durable memory and typed claims verified at the end.
`GovernedOrchestrator` is the harness that ties those pieces into
one auditable run; it composes the gateway with three further
objects.

```python
from knowlytix.harness.governance.memory import AgentMemory
from knowlytix.harness.governance.verifier import ClaimVerifier
from knowlytix.harness.governance.orchestrator import (
    GovernedOrchestrator, Scenario,
)

memory = AgentMemory(
    store_path="my_store",
    knowledge_path="knowlytix/harness/governance/knowledge/data_science_practices.md",
    device=DEVICE,
)
verifier = ClaimVerifier(store, device=DEVICE)
```

`AgentMemory` is the durable-storage facade over the store: it
routes a finding through graph insert, ENM write, and training
flush in the right order so a write does not return until every
component has caught up (§10). `ClaimVerifier` types each fact the
agent produces during the run and routes it to the right GMS
primitive — `score_triple` for facts, `lookup_enm` for numerics,
`check_holonomy` for reasoning chains — returning a typed verdict
the audit log can quote (§12).

```python
orchestrator = GovernedOrchestrator(
    gateway=gateway,
    memory=memory,
    verifier=verifier,
    session=session,
    store=store,
    device=DEVICE,
)
```

The orchestrator is the lifecycle manager. It owns no geometry of
its own: every gate verdict comes from `gateway`, every persisted
finding goes through `memory`, every claim is graded by `verifier`,
and the manifold lookups all hit the same `store`. This is what
makes the audit trail uniform — one object writes the trace, and
the trace's contents are the verdicts the four governance components
produced.

```python
scenario = Scenario(
    name="iris_evaluation",
    required_tools=["load_dataset", "train_model"],
    forbidden_tools=["delete_record"],
    max_tool_calls=20,
)
```

The `Scenario` is the run's behavioral contract. `required_tools`
must each appear in the plan — a plan that omits one is recorded as
a scenario violation during `PREPARING`. `forbidden_tools` are
filtered out before the gateway is even invoked, so the offending
call cannot happen and the audit log records why. `max_tool_calls`
caps the run; exceeding it during `EXECUTING` aborts to `ABORTED`
with the partial trace preserved. The full field semantics live in
§21.

```python
plan = [
    "load_dataset", "data_summary", "preprocess_data",
    "train_model", "diagnose_robustness", "explain_importance",
]

result = orchestrator.run(plan, scenario)
print(orchestrator.format_result(result))
```

`plan` is the ordered list of tool names to execute.
`orchestrator.run` walks the lifecycle FSM
`CREATED → PREPARING → EXECUTING → EVALUATING → COMPLETED`,
running each step through the gateway, recording its six-component
signal vector (§29), writing extracted findings to `memory`, and at the
end handing every typed claim to `verifier`. The returned
`OrchestratorResult` carries the lifecycle trace, the per-step
signal vectors, the scenario violations, and the verification
verdict; `format_result` renders all of that as the human-readable
audit summary an operator reads. §21 covers the lifecycle, the
two-attestor verifier pattern, and the tool-error abort path in
full.

---

## 4. GMS primitives reference

Every governance decision in this stack reduces to one of six
primitives on $S^{m-1}$. Each primitive takes structured input,
returns a single number with a well-defined semantic, and surfaces a
specific failure mode that no other primitive sees. Together they
form the smallest complete vocabulary the verifier and reasoner
need.

| Primitive | Returns | Verification use | Example |
|-----------|---------|------------------|---------|
| `score_triple` | geodesic $\geq 0$, lower is more plausible | fact plausibility | "GradientBoosting recommended for nonlinear data?" |
| `tension_energy` | scalar in $[0, 2]$ | contradiction detection | "Are 'Yes' and 'No' antipodal in this context?" |
| `check_holonomy` | defect $\geq 0$, lower is more consistent | reasoning-chain consistency | "Does the two-hop path equal the direct relation?" |
| `lookup_enm` | exact float (SHA-256 verified) | exact-numeric recall | "What is the AUC threshold for high-risk credit?" |
| `link_predict` | top-$k$ ranked tails | knowledge exploration | "Which models are recommended for nonlinear data?" |
| `query_triples` | list of matching triples | structural retrieval | "List every relation out of GradientBoosting" |

Math reminders:

- Geodesic distance on $S^{m-1}$: $d(R_r v_h, v_t) = \arccos(R_r v_h \cdot v_t)$
- Tension energy: $\sqrt{2 - 2\,a \cdot b}$
- Holonomy defect: $\|R_{\mathrm{path}} R_{\mathrm{direct}}^{-1} - I\|_F$

The examples below all run against the prebuilt
`ds_practices_store` fixture from §3.

### 4.1 `score_triple` — fact plausibility

A triple $(h, r, t)$ is plausible when the head, rotated by the
relation operator, lands close to the tail on the sphere. The
primitive returns the geodesic distance between $R_r v_h$ and $v_t$;
0 is exact match, larger is implausible.

```python
def score_triple(self, head: str, rel: str, tail: str) -> float | None:
    """Score plausibility of (head, rel, tail). Lower = more plausible."""
```

```python
score = store.score_triple(
    "Nonlinear relationships", "has_recommended_model", "GradientBoosting")
print(f"plausibility = {score:.4f}")   # close to 0 — well supported
```

**Interpretation.** A score below the calibrated plausibility
threshold (typically $\theta \approx 1.2$-$1.5$) means the store
supports the triple; a score well above it means the model has not
seen the triple and the verifier should refuse.

### 4.2 `tension_energy` — contradiction detection

Tension energy measures how aligned two entity embeddings are in the
*logical* view. It is the chord-length analogue of geodesic distance:
$\mathrm{tension}(a, b) = \sqrt{2 - 2\,a \cdot b}$, ranging from 0
(agree) through $\sqrt{2}$ (irrelevant) to 2 (contradict).

```python
def tension_energy(self, entity_a: str, entity_b: str) -> float | None:
    """Tension energy between two entities.
    0=agree, sqrt(2)=irrelevant, 2=contradict."""
```

```python
agree = store.tension_energy("RandomForest", "GradientBoosting")
mixed = store.tension_energy("RandomForest", "DecisionTree")
print(f"agree = {agree:.3f},  mixed = {mixed:.3f}")
```

**Interpretation.** Values below ~0.5 are agreement, ~0.5 to
$\sqrt{2}$ irrelevance, above $\sqrt{2}$ contradiction. The exact
band edges are calibrated per store via
`bootstrap_plausibility_threshold`.

### 4.3 `check_holonomy` — reasoning consistency

Holonomy compares a multi-step relational path against a direct
relation. If the path is internally consistent, walking the path
should land where the direct relation lands; the residual rotation
defect is small.

```python
def check_holonomy(self, relation_path: list[str],
                    direct_relation: str) -> float | None:
    """Holonomy defect for path vs direct edge. 0=consistent."""
```

```python
defect = store.check_holonomy(
    ["has_requires", "has_precedes"],   # multi-step path
    "has_phase",                         # direct relation
)
print(f"holonomy defect = {defect:.4f}")
```

**Interpretation.** Defects close to 0 indicate a consistent chain;
defects approaching the random-rotor baseline ($\sqrt{2m}$, e.g.
~16 for $m=128$) indicate the path and the direct relation
disagree, which is the structural signal of an inconsistent
multi-hop claim. **Closed loops** (paths that return to their
starting node, e.g. $a \to b \to c \to a$) are a special case
worth understanding before relying on holonomy in production —
see §11's [Worked example: loops and back-edges](#worked-example-loops-and-back-edges)
for end-to-end code that builds the topology, scores each edge,
and reads the holonomy verdict on closed vs open paths.

### 4.4 `lookup_enm` — exact numeric recall

Embeddings produce graded plausibility, but a model-risk report must
not round its area-under-the-curve (AUC). The Exact Numerical Memory
register stores numeric values losslessly and verifies them against a
SHA-256 hash on read.

```python
def lookup_enm(self, category: str, entity_id: str) -> float | None:
    """Exact numeric lookup with SHA-256 integrity."""
```

```python
auc_min = store.lookup_enm(
    "evaluation_standards",
    "High-risk (credit, lending)/Fairness, robustness, "
    "explanation, cross-validation/Minimum AUC",
)
print(f"minimum AUC = {auc_min}")    # 0.85, byte-identical to source
```

**Interpretation.** A returned value is byte-identical to the
ingested value; `None` means the key was not stored. Any corruption
fails the SHA-256 check inside the ENM register before this method
returns.

### 4.5 `link_predict` — knowledge exploration

Given a head and a relation, link prediction returns the
type-constrained top-$k$ tails ranked by score. It is the primitive
that powers distractor generation, oracle-neighborhood checks, and
"what model would you recommend" queries.

```python
def link_predict(self, head: str, relation: str,
                  top_k: int = 10) -> list[tuple[str, float]]:
    """Top-k tail predictions for (head, relation, ?)."""
```

```python
predictions = store.link_predict(
    "Nonlinear relationships", "has_recommended_model", top_k=3)
for entity, score in predictions:
    print(f"  {entity}: {score:.4f}")
```

**Interpretation.** The list is sorted lowest-score first (most
plausible). Self-entity is excluded and only entities that ever
appear as a tail of this relation in the training graph are scored,
so numeric entities cannot pollute categorical predictions.

### 4.6 `query_triples` — structural retrieval

Pure graph lookup against the document graph: deterministic, 100%
recall over stored triples. Any combination of `head`, `relation`,
`tail` filters is supported; omitted filters match all values.

```python
def query_triples(self, head=None, relation=None, tail=None):
    """Pattern-match triples in document graph."""
```

```python
triples = store.query_triples(head="GradientBoosting")
for h, r, t in triples[:5]:
    print(f"  ({h}, {r}, {t})")
```

**Interpretation.** Use this when the verifier needs an exact
ground-truth list (e.g. "every relation out of this entity") rather
than a graded plausibility score. It is the structural counterpart
of `score_triple`.

### 4.7 Cross-references

- Math reference: `gmsh_testing_ext/chapters/appendix_a.tex` §A.2
- Concept derivation: `gms_agent_builder_ext/chapters/ch02.tex`
- Notebook: `user_guide_tutorials/notebooks/04_gms_primitives.ipynb`

---

## 5. GMS diagnostics

`GMSDiagnostics` runs nine invariant checks on a built store: three
structural preconditions and one check per primitive. A single
failing check usually points at an upstream training or
configuration issue rather than at a knowledge gap, which is why the
diagnostics are kept separate from claim verification — they tell
you whether the store is *operable* before you ask it questions.

### 5.1 The nine checks

| # | Check | Purpose |
|---|-------|---------|
| S1 | `operator_orthogonality` | every relation rotor stays in $SO(m)$: $R^T R = I$, $\det(R) = 1$ |
| S2 | `projection_integrity` | both Stiefel projection matrices satisfy $PP^T = I$ |
| S3 | `embedding_norm` | every entity projects to the unit sphere ($\|v\|=1$, $\|u\|=1$) |
| P1 | `score_triple` | true triples score lower than type-matched fabricated tails ($\geq 90\%$) |
| P2 | `tension_energy` | agreement pairs $E < \sqrt 2$ ($\geq 80\%$); contradiction pairs $E > \sqrt 2$ ($\geq 70\%$); spectrum width $> 0.5$ |
| P3 | `check_holonomy` | self-identity defect $< 0.01$; triangle defect $<$ random baseline / 2 |
| P4 | `lookup_enm` | every ENM entry passes SHA-256 integrity (must be 100%) |
| P5 | `link_predict` | type-constrained mean reciprocal rank (MRR) $\geq 0.3$, Hits@5 $\geq 50\%$ |
| P6 | `query_triples` | every sampled structural query returns the expected hit |

> The class still exposes the original physics-flavored names
> (`cayley_orthogonality`, `stiefel_projection`, `sphere_embedding`)
> as aliases of S1-S3; user-facing notebooks use the neutral
> aliases above.

### 5.2 Running the report

```python
from knowlytix.core.diagnostics import GMSDiagnostics

diag = GMSDiagnostics(
    model=store.model,
    adapter=store.adapter,
    enm=store.enm,
    doc_graph=store.doc_graph,
    device=store.device,
)
report = diag.run_all()
GMSDiagnostics.print_report(report)
print(f"Pass: {report.pass_count}/{report.total}")
```

`run_all()` returns a `DiagnosticReport` with a `.checks` list of
`CheckResult(name, passed, score, detail, evidence)` entries plus
`pass_count`, `total`, and `all_passed`. `print_report` formats a
human-readable summary; the structured fields are what release-time
audits log.

### 5.3 What to do with a failure

| Failing check | Most likely upstream cause |
|---------------|----------------------------|
| S1 / S2 / S3 | training collapsed; check geometry config and learning rate |
| P1 (`score_triple`) | not enough training epochs; raise `train.epochs` |
| P2 (`tension_energy`) | corpus has no real contradictions; check ingestion mode and document content |
| P3 (`check_holonomy`) | rotor parameterization unstable; verify Cayley map and retraction step |
| P4 (`lookup_enm`) | hard data corruption — never run an audit against this store |
| P5 (`link_predict`) | tail-type table empty; rebuild adapter from the document graph |
| P6 (`query_triples`) | document graph not loaded; check store persistence |

A passing report is the precondition for every downstream claim
verification, calibration sweep, or release-gate run.

### 5.4 Cross-references

- Implementation: `gms/diagnostics.py:GMSDiagnostics.run_all`
- Notebook: `user_guide_tutorials/notebooks/05_gms_diagnostics.ipynb`

---

## 6. Knowledge ingestion

Section 3.1 created an empty `GMSExpertStore`. This section explains how to populate one from a source document and how the resulting store is laid out on disk. The single entry point is `docknowlytix.core.ingest.ingest_document`, which (a) converts the source to markdown via `docknowlytix.core.convert.convert_document` and (b) delegates to the markdown parser in `knowlytix.benchmark.ingest.ingest_markdown` over a configurable mode.

Two independent knobs control the pipeline:

1. **Conversion** — how the source file becomes markdown. PDF, TeX, MD and TXT are all accepted; PDF has two interchangeable backends. See [§Ingesting non-markdown sources](#ingesting-non-markdown-sources-pdf-tex-txt) below.
2. **Parsing** — how that markdown becomes triples + ENM entries: `regex`, `hybrid`, or `llm_only`. See [The three ingestion modes](#the-three-ingestion-modes) below.

Conversion runs first; the parsing mode then operates on the resulting markdown regardless of source format.

### Ingesting non-markdown sources (PDF, TeX, TXT)

`ingest_document` accepts more than `.md`. Step 1 of the pipeline calls `docknowlytix.core.convert.convert_document`, which dispatches on file extension and returns markdown that is then handed to the parser. The conversion is configured via `DocGMSConfig.convert` (a `ConvertConfig` dataclass).

| Extension          | Format    | Converter                                              |
|--------------------|-----------|--------------------------------------------------------|
| `.md`, `.markdown` | Markdown  | passthrough                                            |
| `.txt`             | Plain text | passthrough                                            |
| `.pdf`             | PDF       | OpenDataLoader **or** PyMuPDF (selectable)             |
| `.tex`, `.latex`   | LaTeX     | regex prose + LLM tabular (regex-only fallback)        |

#### PDF backends — pick one before ingest

PDF is the only format where two independent packages are available and the choice affects the resulting markdown. The selector is `ConvertConfig.pdf_backend`:

| Backend            | Reading order / tables                                              | Requirements                            | LLM in conversion |
|--------------------|---------------------------------------------------------------------|-----------------------------------------|-------------------|
| `"opendataloader"` | XY-Cut++ reading order, layout-based table detection, heading hierarchy | `pip install opendataloader-pdf`, Java 11+ | not used          |
| `"pymupdf"`        | per-page text extract; pages that *look* tabular routed through the LLM | `pip install pymupdf`                   | required for tables |
| `"auto"` (default) | OpenDataLoader if importable and Java 11+ on `PATH`, else PyMuPDF + LLM | best-effort                             | only if PyMuPDF wins |

`"opendataloader"` is the audit-defensible default for PDF: conversion is fully deterministic and contains no LLM calls. `"pymupdf"` is the no-Java fallback; it relies on the LLM to recover proper markdown tables from layout-flattened text.

> **Note.** The `pdf_backend` knob controls *conversion only*. The downstream parser still honors `ingest_mode` (`regex` / `hybrid` / `llm_only`). The two stages are independent — for example, `pdf_backend="opendataloader"` + `ingest_mode="regex"` gives a fully deterministic, LLM-free pipeline end-to-end.

#### Example: PDF ingest with OpenDataLoader (deterministic conversion)

```python
from docknowlytix.core.config import DocGMSConfig, ConvertConfig
from docknowlytix.core.ingest import ingest_document
from docknowlytix.core.llm_backend import AnthropicBackend
from docknowlytix.core.store import GMSExpertStore

config = DocGMSConfig(
    store_path="report_store",
    ingest_mode="hybrid",
    convert=ConvertConfig(pdf_backend="opendataloader"),
)
store = GMSExpertStore(config, device=DEVICE)
llm = AnthropicBackend()                              # required by ingest_mode="hybrid"
result = ingest_document(
    store, "report.pdf", llm=llm, config=config, device=DEVICE,
)
```

OpenDataLoader produces clean markdown with proper headings and tables without any LLM calls during conversion. The LLM is still required at parse time when `ingest_mode` is `"hybrid"` or `"llm_only"`.

#### Example: PDF ingest with PyMuPDF + LLM (no Java)

```python
config = DocGMSConfig(
    store_path="report_store",
    ingest_mode="hybrid",
    convert=ConvertConfig(pdf_backend="pymupdf", max_pages=50),
)
result = ingest_document(
    store, "report.pdf", llm=llm, config=config, device=DEVICE,
)
```

PyMuPDF extracts text per page; pages whose layout looks tabular (heuristic in `convert._looks_like_table`) are routed through the LLM with a "preserve all numbers exactly" system prompt. `max_pages` caps conversion cost on long PDFs; `chunk_size` (chars per LLM call) is also exposed on `ConvertConfig`.

#### Example: TeX ingest

```python
config = DocGMSConfig(
    store_path="paper_store",
    ingest_mode="hybrid",
    convert=ConvertConfig(),                          # defaults are fine for TeX
)
result = ingest_document(
    store, "paper.tex", llm=llm, config=config, device=DEVICE,
)
```

`convert_tex_to_markdown` extracts the document body, sends each `tabular` environment to the LLM for a clean markdown table (numbers preserved byte-for-byte), and converts surrounding prose with regex (`\section{...}` → `## ...`, `\textbf{...}` → `**...**`, etc.). With `llm=None`, the converter falls back to a regex-only TeX→MD path; tables degrade to raw `\begin{tabular}` blocks.

#### Skipping conversion

If you already have markdown produced by your own pipeline (a custom OCR step, a different PDF tool, etc.), write it to a `.md` file and call `ingest_document` directly — `convert_document` will detect the `.md` extension and passthrough.

### The three ingestion modes

`DocGMSConfig.ingest_mode` selects the extraction strategy. The mode is the only knob that decides whether and how an LLM is invoked during ingest. Misuse raises rather than silently degrading.

| Mode | LLM | Recall vs regex | When to use |
|------|-----|-----------------|-------------|
| `"regex"` | not invoked | baseline (100%) | reproducible CI builds, offline fixtures, audit-defensible deterministic builds |
| `"hybrid"` (default) | required | strict superset; +22% ENM, +14% triples; **0** value drift on shared keys | production |
| `"llm_only"` | required | ≈ 12.5% ENM recall, near-zero overlap | paper reproduction, debugging baseline |

`"hybrid"` is the default because it preserves every ENM key the regex parser finds, then augments the graph with prose-derived triples that regex cannot reach. The numeric-value contract is preserved across modes: every shared ENM key carries the same SHA-256-keyed IEEE-754 value. The LLM contributes only structural decisions (column classification, entity detection) and prose-derived triples — never numbers that bypass the register.

#### Regex mode

```python
from docknowlytix.core.config import DocGMSConfig
from docknowlytix.core.ingest import ingest_document
from docknowlytix.core.store import GMSExpertStore

config = DocGMSConfig(store_path="my_store", ingest_mode="regex")
store = GMSExpertStore(config, device=DEVICE)
result = ingest_document(
    store, "report.md", llm=None, config=config, device=DEVICE,
)
```

Regex mode never calls an LLM. It extracts from headings, markdown tables and `Key: Value` bullet patterns. It is the fastest mode and the only one that runs without `ANTHROPIC_API_KEY`. The two fixtures shipped with this tutorial (`ds_practices_store`, `california_housing_store`) are built in regex mode for exactly that reason.

#### Hybrid mode

```python
from docknowlytix.core.llm_backend import AnthropicBackend

config = DocGMSConfig(store_path="my_store", ingest_mode="hybrid")
llm = AnthropicBackend()    # raises AnthropicAPIKeyError if key missing
result = ingest_document(
    store, "report.md", llm=llm, config=config, device=DEVICE,
)
```

Hybrid runs the regex parser first, then asks the LLM at three injection points: column-type classification on ambiguous tables, entity detection on prose paragraphs and prose-triple extraction over sentences the regex parser left untouched. Every numeric value still flows through the regex pipeline, so the SHA-256-keyed ENM values are byte-identical to a regex-only build of the same document.

#### LLM-only mode

```python
config = DocGMSConfig(store_path="my_store", ingest_mode="llm_only")
result = ingest_document(
    store, "report.md", llm=llm, config=config, device=DEVICE,
)
```

LLM-only disables the regex backbone and asks the LLM to extract everything. It is shipped for paper reproduction and as a debugging baseline; it is not suitable for production because numeric-value recall drops to roughly 12.5%.

### `ingest_document()` signature and `IngestResult`

```python
from docknowlytix.core.ingest import ingest_document, IngestResult

result: IngestResult = ingest_document(
    store=store,                       # GMSExpertStore (created or loaded)
    document_path="report.pdf",        # PDF, TeX, MD, or TXT
    llm=AnthropicBackend(),            # required for hybrid / llm_only
    config=config,                     # carries ingest_mode + ConvertConfig
    device=DEVICE,
)
```

`IngestResult` is a small dataclass with these fields:

| Field             | Type     | Meaning                                                |
|-------------------|----------|--------------------------------------------------------|
| `document`        | `str`    | basename of the ingested document                      |
| `new_entities`    | `int`    | entities added to the adapter (first ingest only)      |
| `new_triples`     | `int`    | triples added to `doc_graph`                           |
| `new_enm`         | `int`    | ENM register entries written                           |
| `training_epochs` | `int`    | epochs run on first ingest (0 on incremental)          |
| `is_first`        | `bool`   | `True` on first ingest, `False` on incremental         |
| `elapsed_seconds` | `float`  | wall-clock time of the call                            |

A misconfigured call raises `ValueError` synchronously. If `ingest_mode` is `"hybrid"` or `"llm_only"` and `llm` is `None`, ingestion refuses to run rather than silently falling back to regex. Audit-defensible deployments should always set the mode explicitly so a reviewer can tell from the configuration which path produced the store.

### Persistence layout

`GMSExpertStore.save()` (called automatically at the end of `ingest_document`) writes a fixed set of files into `store_path/`:

| File                  | Format                  | Contents                                                   |
|-----------------------|-------------------------|------------------------------------------------------------|
| `model.pt`            | PyTorch state-dict      | trained `GeometricKnowledgeGraph` weights                  |
| `model_dims.json`     | JSON                    | `num_entities`, `num_relations`, cap-admissibility config  |
| `adapter.json`        | JSON                    | entity-to-index and relation-to-index vocabularies         |
| `triples.json`        | JSON                    | full list of `(head, relation, tail)` triples              |
| `phase.json`          | JSON                    | per-phase-encoder `v_min` and `v_max` ranges               |
| `enm.json`            | JSON                    | every ENM entry as `"<type>::<id>": value`                 |
| `metadata.json`       | JSON                    | ingestion history, timestamps, document list               |
| `documents/combined.md` | Markdown              | concatenated source markdown of every ingested document    |

Reload is the mirror of save: `GMSExpertStore.load()` reads `model_dims.json` first to recreate the architecture (including cap-admissibility flags), reconstructs `doc_graph` from `triples.json` plus `phase.json`, rebuilds the adapter, then loads `model.pt` weights into the matching tensor shapes.

### Incremental ingestion (Phase 1 sync, Phase 2 async)

Subsequent calls to `ingest_document` against a store that already has a model run the incremental path. The protocol is the two-phase pattern documented in `gms_agent_builder_ext/chapters/ch04.tex` and implemented in `knowlytix/knowledge/learn.py:RuntimeLearner`:

- **Phase 1 — synchronous, fast.** Diff the new document against the existing graph. New triples are appended to `doc_graph` immediately; new ENM entries are written immediately; the adapter vocabulary is expanded. Graph queries see the new facts on the next call. A contradiction gate (`tension_energy > sqrt(2)` against an existing fact) blocks acceptance before any of this is committed.
- **Phase 2 — background, asynchronous.** Newly added entities and relations need geometric coordinates. A background thread expands the embedding tensor (sentence-transformer init), adds new Cayley rotors for new relations, freezes the existing parameters, and runs Riemannian gradient descent on the new rows only. Existing knowledge is invariant under this growth.

Phase 1 always succeeds or raises; Phase 2 is observable via `RuntimeLearner.training_status()` and can be awaited with `RuntimeLearner.wait_for_training(timeout=...)` when a downstream step needs the new geometric coordinates to settle. The split keeps ingest interactive while still producing a fully geometric store.

#### Worked example: growing a store with a second markdown

The same `ingest_document` entry point dispatches first-document training and incremental growth. The store object decides which path to run by inspecting whether `store.model is None`:

```python
from docknowlytix.core.config import DocGMSConfig
from docknowlytix.core.ingest import ingest_document
from docknowlytix.core.store import GMSExpertStore

config = DocGMSConfig(store_path="rules_store", ingest_mode="regex")
store = GMSExpertStore(config, device=DEVICE)

# First document — full training pipeline.
result_a = ingest_document(
    store, "rules_v1.md", llm=None, config=config, device=DEVICE,
)
print(result_a.is_first, result_a.new_triples, result_a.new_entities)
# True   <triples in v1>   <entities in v1>

# Second document — incremental growth into the same store.
result_b = ingest_document(
    store, "rules_v2.md", llm=None, config=config, device=DEVICE,
)
print(result_b.is_first, result_b.new_triples, result_b.new_entities)
# False  <only the diff>   <only previously-unseen entities>

# Combined graph is the union of both documents.
print(len(store.doc_graph.triples))                  # v1 + v2 triples
print(store.doc_graph.query_triples(
    relation="has_imbalanced_classes"))              # v2-only fact visible
```

Three properties hold on this path:

- **Diff-only counts.** `IngestResult.new_triples` and `new_entities` count what was actually added, not what the second document parsed. Triples already in the graph are silently dropped, so re-ingesting the same document is a no-op (`new_triples=0`).
- **Phase 1 graph queries are immediate.** `store.doc_graph.query_triples(...)` and `store.doc_graph.lookup(...)` see the new facts as soon as `ingest_document` returns, regardless of whether Phase 2 has finished.
- **Contradiction gate is fail-closed.** If the second document asserts the negation of an existing fact (`tension_energy > sqrt(2)`), `learn_triples` rejects the entire batch with `accepted=False, reason="Contradiction gate: ..."` rather than partial-write.

`store.metadata["documents"]` records each ingested file with timestamp and per-document counts, so the audit trail names every source contributing to the merged graph.

> **Phase 2 caveat.** `ingest_document` constructs a transient `RuntimeLearner` for the incremental call and queues coordinate-growth jobs on it; the learner goes out of scope when the function returns. Callers who need the new entities to have *trained* geometric coordinates (not just adapter slots) should drive Phase 2 from a `RuntimeLearner` they own — call `learner.learn_triples(...)` directly, then `learner.flush_training()` and `learner.wait_for_training(timeout=...)`. The two-phase prose above documents the designed contract; the convenience wrapper exposes only Phase 1 cleanly.

### One-call alternative for non-Anthropic backends

When the only handle you have on the LLM is a `Callable[[str], str]`, the standalone parser bypasses `LLMBackend`:

```python
from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.benchmark.llm_caller import make_anthropic_callable

llm_callable = make_anthropic_callable()        # or any Callable[[str], str]
graph = ingest_markdown("report.md", mode="hybrid", llm_callable=llm_callable)
```

This produces a `DocumentGraph` only. Wrap it in a `GMSExpertStore` and run `populate_enm` plus `train_gms` from `knowlytix.core.train_finstructbench` to obtain the trained store.

### Cross-references

- USER_GUIDE.md §3.1 (Build a knowledge store)
- USER_GUIDE.md §13 (Runtime Learning) for Phase 2 details
- `knowlytix/knowledge/ingest.py:ingest_document` and `knowlytix/knowledge/store.py:GMSExpertStore.save`
- `knowlytix/knowledge/convert.py:convert_document` (PDF / TeX / MD / TXT dispatcher; `pdf_backend` selector)
- `gms_agent_builder_ext/chapters/ch04.tex` for the two-phase incremental protocol
- `knowlytix/benchmark/ingest.py:ingest_markdown` for the parser-mode dispatcher
- Notebook `notebooks/06_ingestion.ipynb`

---

## 7. Three-tier extraction

Section 6 covered ingest from a source document (markdown, PDF, or TeX). This section covers the runtime cousin: how the agent extracts triples and ENM entries from every tool result it observes during a session. The single entry point is `knowlytix.harness.governance.extractors.extract_from_tool`, and the design is a three-tier cascade: deterministic first, regex second, LLM only as a last resort.

### Concept: tier order matches accuracy order

Every governed-agent tool returns the dual-return pattern documented in `gms_agent_builder_ext/chapters/ch04.tex` §4.3: a tuple `(display_string, raw_dict)`. The display string is a human-readable summary; the raw dict is a pure-Python structure (strings, numbers, lists, nested dicts) that passes `json.dumps` without error. The two views carry the same information, but only the raw dict can be parsed deterministically.

The three tiers exploit that contract.

| Tier | Method                                  | Input            | Accuracy on structured tool output | When it fires                                  |
|------|-----------------------------------------|------------------|------------------------------------|------------------------------------------------|
| 1    | deterministic dict-read                 | `raw_result`     | 100% on registered tools           | always tried first when `raw_result` is a dict |
| 2    | regex over display string               | `display_string` | partial; depends on format         | Tier 1 unavailable or empty                    |
| 3    | LLM / structured-text parser            | `display_string` | 58 to 65% across categories        | Tier 2 also empty                              |

The Tier 1 / Tier 3 accuracy gap is the empirical finding from FinStructBench (`gms_agent_builder_ext/chapters/ch04.tex` §4.2). LLM extraction over display strings reaches only 58 to 65% accuracy across ten categories of increasing complexity — tables, nested dicts, scientific notation, financial-formatted numbers — because display strings are formatted for human consumption. The same values read directly from the raw dict are byte-identical to what the tool produced. The dual-return pattern is what makes Tier 1 possible.

**One-line guidance: structured tool output goes through Tier 1, always.** Skip the cascade by passing the `raw_result` dict; do not feed display strings to Tier 3 if a registered raw dict is available.

### `extract_from_tool` signature

```python
from knowlytix.harness.governance.extractors import extract_from_tool, ExtractionResult

extraction: ExtractionResult = extract_from_tool(
    tool_name="train_model",
    args={"model_type": "RandomForestClassifier",
          "model_name": "rf_iris", "dataset": "iris"},
    raw_result={                       # the dict returned alongside display
        "model_name": "rf_iris",
        "model_type": "RandomForestClassifier",
        "metrics": {"accuracy": 0.95, "f1": 0.94},
    },
    display_string="rf_iris  accuracy: 0.95  f1: 0.94",
)
```

The dispatcher tries Tier 1 first if `raw_result` is a non-empty dict and a Tier 1 extractor is registered for `tool_name` (registered tools listed below). It falls through to Tier 2 only if Tier 1 returns no facts or is unavailable, and to Tier 3 only if Tier 2 also returns nothing.

#### Tier 1 — registered tools

The Tier 1 dispatch table in `knowlytix/harness/governance/extractors.py` covers six tools today:

- `load_dataset`
- `train_model`
- `diagnose_robustness`
- `diagnose_fairness`
- `explain_importance`
- `leaderboard`

Each handler reads the raw dict by key, applies a defensive `_as_float` cast, and emits one `ExtractedTriple` per fact and one `ENMEntry` per numeric. There is no parsing and no guessing — every value comes directly from a known dict key.

#### Tier 2 — regex over display strings

Tier 2 fires when no Tier 1 extractor is registered for the tool, or when the registered handler returns zero facts (an `error` key, an empty result). It applies one of two regex patterns: a leaderboard row pattern when `tool_name == "leaderboard"`, and a generic `key: value` pattern otherwise. The output is structurally identical to Tier 1, only less reliable.

#### Tier 3 — structured-text parser (LLM-equivalent)

Tier 3 calls `docknowlytix.core.extract.extract_triples_simple`, which parses lines beginning with `TRIPLE:` and pipe-separated `head | relation | tail | numeric` quads. In production this slot is replaced with `docknowlytix.core.extract.extract_triples`, which prompts an LLM with the `_EXTRACT_SYSTEM` prompt and parses its `TRIPLE:`-prefixed output. The simple parser is the deterministic stand-in used here so the cascade has a working fallback when no LLM is configured.

### `ExtractedTriple` and `ENMEntry` fields

```python
from docknowlytix.core.extract import ExtractedTriple
from knowlytix.harness.governance.extractors import ENMEntry

ExtractedTriple(
    head="rf_iris",
    relation="has_accuracy",
    tail="0.95",
    numeric_value=0.95,    # set when the tail is numeric; None otherwise
)

ENMEntry(
    category="metric",          # "metric", "robustness", "fairness", ...
    entity_id="rf_iris_accuracy",
    value=0.95,
)
```

`ExtractionResult` wraps a list of triples, a list of ENM entries, the tier that produced them, the originating tool name, and a convenience `n_facts` total.

### When to cross-check Tier 1 against Tier 3

The `compare_extraction_accuracy` helper takes a ground-truth dict of expected metric values and an `ExtractionResult`, and reports per-key match / mismatch / missing counts. Run it occasionally as a sanity check against the registered Tier 1 extractors — a regression there is the most likely cause of a downstream metric drift.

### Cross-references

- USER_GUIDE.md §6 (Knowledge Ingestion) for the markdown counterpart
- `knowlytix/harness/governance/extractors.py` for the full Tier 1 dispatch table
- `knowlytix/knowledge/extract.py:extract_triples_simple` and `extract_triples`
- `gms_agent_builder_ext/chapters/ch04.tex` §4.2-4.3 for FinStructBench accuracy figures
- Notebook `notebooks/07_three_tier_extraction.ipynb`

---

## 8. Authoring knowledge documents

This section is for the author of a knowledge document. The previous two sections covered ingestion and extraction from the consumer side. This one covers what to write so the result ingests cleanly. Both shipped knowledge files (`knowlytix/harness/governance/knowledge/data_science_practices.md` and `knowlytix/harness/governance/knowledge/california_housing_domain.md`) follow the conventions below.

### What ingests cleanly

The regex parser in `knowlytix.benchmark.ingest.ingest_markdown` extracts triples and ENM entries from three constructs:

1. **Markdown headings** establish the section identity. A level-2 heading `## Model Selection Guidelines` becomes the section context for every table and bullet that follows. Heading hierarchy must be strict: only one `#` title, then `##` for top sections, then `###` for subsections.
2. **Markdown tables** are the highest-yield construct. Each row produces one or more triples; numeric cells additionally land in the ENM register with a SHA-256-keyed entry. Column headers should be terse and mostly entity-typed.
3. **Bullet lists with `Key: Value` patterns** produce one triple per line. The key becomes the relation, the value becomes the tail, and the section heading supplies the head.

What does **not** survive cleanly: free-text prose without entity mentions. A paragraph that explains a concept conversationally yields little or nothing in regex mode. Hybrid mode recovers some of this through the LLM prose-triple pass, but the deterministic baseline drops it. Author with this in mind: prefer bullets and tables when you want a fact to make it into the graph.

### A small example

The following thirty-line markdown is the kind of file that ingests cleanly. It describes a small model-selection rule set; replicate this shape in your own knowledge documents.

```markdown
# Iris Model Selection

## Classification Rules

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| Need interpretability | LogisticRegression | Transparent coefficients |
| Multiclass target | RandomForest | Native multiclass support |
| Small dataset below 200 rows | SVM | Kernel trick effective in low-sample regime |

### Performance Thresholds

| Risk Tier | Minimum Accuracy | Minimum F1 |
|---|---|---|
| Pilot | 0.85 | 0.80 |
| Production | 0.90 | 0.85 |

## Model Notes

- LogisticRegression: max_iter is 1000
- RandomForest: n_estimators is 100
- SVM: kernel is rbf
```

Ingest yield from this file in regex mode is approximately:

| Construct                  | Yield                                                       |
|----------------------------|-------------------------------------------------------------|
| H1 title                   | one triple `(Iris Model Selection, has_section, Classification Rules)` |
| H2 heading                 | one triple per child entity                                 |
| Classification table (3 rows) | three to six triples (`has_recommended_model`, `has_rationale`) |
| Threshold table (2 rows)   | two triples + four ENM entries (numeric cells)              |
| Bullet list (3 lines)      | three triples (`has_max_iter`, `has_n_estimators`, `has_kernel`) |

Total: roughly twelve to fifteen triples and four ENM entries from a thirty-line file. The shipped `data_science_practices.md` follows the same shape at scale and yields 334 triples plus 24 ENM entries on a regex build.

### Patterns drawn from `data_science_practices.md`

The shipped file uses three patterns worth copying:

1. **Recommendation table.** A three-column layout `Data Profile | Recommended Model | Rationale` produces `(Data Profile, has_recommended_model, Model)` plus `(Data Profile, has_rationale, Rationale)` triples. The same head can carry multiple recommendations because each row is independent.
2. **Capability matrix.** A wide table where each column is an attribute (`Handles Categoricals`, `Interpretable`, `Regularized`) yields one triple per cell. Boolean columns survive as `Yes` / `No` tails; numeric columns flow into ENM.
3. **Risk-tier threshold table.** A row-per-tier table with columns `Minimum AUC | Minimum F1 | Minimum Precision` yields ENM entries `tier_minimum_auc`, `tier_minimum_f1`, etc. These are the values the geometric reasoner consults when validating a claim of the form `(model, achieves, threshold)`.

### ENM round-trip

Numeric cells in markdown go through one well-defined path: the regex parser writes a `(type, id, value)` triple to the ENM register, the ENM hashes the key with SHA-256, and `lookup_enm(type, id)` returns the byte-identical IEEE-754 value on read. Floating-point drift is impossible: the value is stored, not regressed. This is what makes a claim of the form "the high-risk tier's minimum AUC is 0.85" auditable — the literal `0.85` from the markdown is what comes back from the register.

```python
store = load_my_store()
auc = store.lookup_enm("threshold", "high_risk_minimum_auc")
assert auc == 0.85   # byte-identical with the markdown source
```

### What does not survive prose

The following sentence yields zero triples in regex mode, because there is no table, no bullet, and no `Key: Value`:

> "RandomForest typically outperforms LogisticRegression on tabular data with mixed types, especially when the feature count exceeds twenty."

Hybrid mode recovers this through the LLM prose-triple pass and would emit something like `(RandomForest, outperforms, LogisticRegression)` plus a context entity for the condition. If your document has rationale that you genuinely need in the graph, write it in either of two forms:

- A table row where the rationale lives in a dedicated column, as `data_science_practices.md` does.
- A bullet `- RandomForest outperforms_on tabular_mixed_type` written in the same `Key: Value` style as the rest.

If the rationale is descriptive only and is not meant to be queryable, leave it as prose and accept that it will not surface in the graph.

### Authoring checklist

- One `#` title per file; nested `##` and `###` only.
- Tables for facts the agent must query.
- Bullets in `Key: Value` shape for short attributes.
- Numeric values in tables, not in prose.
- Entity names match across the document (case and spelling sensitive).
- Run `ingest_document` once and inspect `result.new_triples` and `result.new_enm`; if a section yields zero, it has no extractable facts and should be rewritten or accepted as prose-only.

### Cross-references

- USER_GUIDE.md §6 (Knowledge Ingestion)
- USER_GUIDE.md §7 (Three-Tier Extraction)
- `knowlytix/harness/governance/knowledge/data_science_practices.md` for the full reference document
- `knowlytix/benchmark/ingest.py:ingest_markdown` for the table and bullet parsers
- Notebook `notebooks/08_authoring_knowledge_documents.ipynb`

---

## 9. Tool catalog

Tools are JSON-schema'd Python functions that the agent invokes through the
`GovernedToolGateway`. Every tool listed in
`knowlytix/harness/governance/tools.py:TOOL_DEFINITIONS` has three pieces a caller must
understand: the **name** the gateway dispatches on, the **input schema** the
schema gate validates against, and the **raw dict** that
`execute_tool(name, args, state) -> (display_string, raw_dict)` returns. The
raw dict is required to be pure-Python (no numpy arrays, no DataFrames, no
sklearn estimators) so that Tier-1 extraction in §7 can SHA-256 it without
relying on object pickling. Calls flow through the triple-gate pipeline in
§17; the gate verdicts and the raw dict together form the audit record that
§23 hashes into the trace chain.

### Catalog

The 15 tools fall into five groups. Display output is a one-line excerpt of
the human-readable string the gateway returns alongside the raw dict.

| # | Group       | Tool name             | Required args                                | Description                                                                  | Sample display output                                          |
|---|-------------|-----------------------|----------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------|
| 1 | data        | `load_dataset`        | `name`                                       | Load sklearn built-in or CSV; train/test split into `state.datasets[name]`.  | `Loaded 'iris': 150 samples, 4 features, target='target'`      |
| 2 | data        | `data_summary`        | `dataset`                                    | `df.describe()`, dtypes, missing percentages.                                | `Data summary for 'iris': ... Missing %: {...}`                |
| 3 | data        | `eda_correlation`     | `dataset`                                    | Pearson or Spearman correlation matrix on numeric columns.                   | `Correlation matrix (pearson): ...`                            |
| 4 | data        | `preprocess_data`     | `dataset`                                    | Impute, scale, encode via sklearn `Pipeline`; stores fitted pipeline.        | `Preprocessing done: impute=mean, scale=standard, encode=...`  |
| 5 | model       | `feature_select`      | `dataset`, `method`                          | `SelectKBest` (`f_classif`, `mutual_info`) or tree-based importance.         | `Feature selection (mutual_info, k=5): petal length: 0.99 ...` |
| 6 | model       | `train_model`         | `dataset`, `model_type`, `model_name`        | Fit a registered sklearn estimator and compute train/test metrics + gap.    | `Trained 'rf_iris' (RandomForestClassifier): test_accuracy: ...`|
| 7 | model       | `predict`             | `dataset`, `model_name`                      | Generate predictions on the held-out test set.                              | `Predictions for 'rf_iris': 30 samples, unique values: 3`      |
| 8 | model       | `leaderboard`         | `dataset`                                    | Rank all trained models for a dataset by `test_accuracy` or `test_r2`.       | `Leaderboard: Rank Model Type test_accuracy ...`               |
| 9 | diagnostic  | `diagnose_robustness` | `dataset`, `model_name`                      | Add Gaussian noise scaled to per-feature std at multiple sigmas; reverdict.  | `Robustness ... Verdict: ROBUST/SENSITIVE/FRAGILE`             |
|10 | diagnostic  | `diagnose_fairness`   | `dataset`, `model_name`, `protected_attribute`| Adverse Impact Ratio + demographic parity across protected groups.          | `AIR: 0.83, Demographic parity diff: 0.07, AIR >= 0.8: True`   |
|11 | diagnostic  | `explain_importance`  | `dataset`, `model_name`                      | Permutation feature importance on the test set.                              | `Permutation importance for 'rf_iris': petal length: 0.34 ...` |
|12 | explanation | `explain_pdp`         | `dataset`, `model_name`, `features`          | Partial-dependence grid for one or two features.                             | `Partial dependence for 'rf_iris' on ['petal length']: ...`    |
|13 | explanation | `explain_shap`        | `dataset`, `model_name`                      | Mean absolute SHAP value per feature via `KernelExplainer`.                  | `Mean |SHAP| for 'rf_iris' (n=50): petal length: 0.21 ...`     |
|14 | memory      | `recall_memory`       | `entity`                                     | Stub for GMS recall (full surface lives in §10 `AgentMemory`).               | `[STUB] recall_memory(entity='LogisticRegression', ...)`       |
|15 | memory      | `save_finding`        | `triples`                                    | Stub for finding storage (full surface lives in §10 `AgentMemory`).          | `[STUB] save_finding: 3 triples received.`                     |

`recall_memory` and `save_finding` are intentional stubs so that the Layer-3
schema is uniform across the agent surface; the real recall and store paths
go through `AgentMemory.recall_entity`, `AgentMemory.compare_entities` and
`AgentMemory.store_findings` (see §10).

### Calling a tool through the gateway

Every tool call goes through `GovernedToolGateway.execute(name, args,
context)`. The gateway returns a `GatewayResult` whose `gate_results` field
records which of the three gates passed. The `context` argument is the
head entity for the gate-3 plausibility probe `(context,
plausibility_relation, tool_name)`; the §3.3 minimal call omits it and
gates 1 and 2 still run, but gate 3 only fires when the corpus has a
trained `should_call` (or whatever `plausibility_relation` is set to)
edge for that context. Production calls always pass an explicit
`context`.

```python
from knowlytix.harness.governance.session import SessionState
from knowlytix.harness.governance.governed_gateway import GovernedToolGateway
from knowlytix.harness.governance.tools import TOOL_DEFINITIONS

session = SessionState()
gateway = GovernedToolGateway(
    session=session,
    store=ds_practices_store,
    theta_plausibility=1.5,
    tool_schemas=TOOL_DEFINITIONS,
)

result = gateway.execute(
    "load_dataset",
    {"name": "iris", "builtin": "iris"},
    context="data_loading",
)
assert not result.blocked, result.reason
print(result.gate_results)  # {'schema': True, 'policy': True, 'plausibility': True, ...}
```

The raw dict is at `result.raw`; the display string is at `result.display`;
the per-gate verdicts are at `result.gate_results`. None of those four
fields depends on the tool's identity, which is what makes a downstream
verifier or tracer composable.

### Adding a custom tool

A new tool is a JSON-schema dict added to `TOOL_DEFINITIONS` plus a
dispatcher branch in `execute_tool`. The dispatcher must return
`(display_string, raw_dict)` where `raw_dict` is pure Python.

```python
# 1. Add the schema entry to TOOL_DEFINITIONS.
NEW_TOOL = {
    "name": "count_rows",
    "description": "Return the row count of a loaded dataset.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dataset": {"type": "string"},
        },
        "required": ["dataset"],
    },
}

# 2. Implement the handler.
def _count_rows(args, state):
    ds = state.datasets[args["dataset"]]
    n = len(ds["df"])
    return f"count_rows: {args['dataset']} has {n} rows", {"n_rows": n}

# 3. Wire it into the gateway via tool_schemas, and into execute_tool's
#    dispatcher (see the elif chain in knowlytix/harness/governance/tools.py:_dispatch).
gateway = GovernedToolGateway(
    session=session,
    store=ds_practices_store,
    tool_schemas=TOOL_DEFINITIONS + [NEW_TOOL],
)
```

Two requirements are non-negotiable. First, the schema must declare every
required parameter under `input_schema.required` so the schema gate can
block calls that omit them. Second, the raw dict must round-trip through
`json.dumps(default=str)`; if it does not, Tier-1 extraction in §7 cannot
hash the result and the audit chain breaks.

### Cross-references

- §17 Governed Tool Gateway — schema, policy, plausibility gate semantics
  and `GovernedToolGateway` wiring.
- §10 AgentMemory — the real surface behind the `recall_memory` and
  `save_finding` schema stubs.
- `gms_agent_builder_ext/chapters/ch03.tex` for the tool-catalog design
  rationale.
- `knowlytix/harness/governance/tools.py:execute_tool` — the canonical dispatcher.
- Notebook `09_tool_catalog.ipynb` exercises every tool through the gateway
  and renders a `(tool, gate)` heatmap.

---

## 10. AgentMemory

`AgentMemory` is the storage-layer abstraction that an agent (or notebook)
calls when it wants to record an experimental finding, recall what has been
recorded, or compare two entities geometrically. Internally it wraps three
pieces that §4 and §6 introduced separately: the persistent
`GMSExpertStore` (graph plus geometry on $S^{m-1}$), the `RuntimeLearner`
two-phase update (synchronous graph insert plus asynchronous Riemannian
gradient descent), and the lossless `ExactNumericalMemory` register. The
caller does not need to know which of those three components handles a
given operation; `AgentMemory` routes for them.

### Why a wrapper at all

The reason to wrap is that the three operations a working agent performs
constantly — store a finding, recall a finding, compare two findings — each
need to touch all three components in a specific order, and a naive caller
that touches them out of order will create silent bugs. A finding written
to the geometric graph but not the ENM register loses its exact numeric
value. A finding written to ENM but not the graph cannot be link-predicted.
A finding written to both but flushed before training completes will return
random embeddings on the next recall. `AgentMemory` enforces the order:
graph insert, then ENM write, then training flush, then save. Every
notebook in this guide that touches durable memory goes through it.

### The four operations

The four methods that matter for day-to-day agent code are
`store_findings`, `recall_entity`, `compare_entities` and
`check_contradiction`. Their signatures are stable and their semantics are
documented one method at a time below.

#### `store_findings(triples, enm_entries=None) -> LearnResult`

Insert a list of `(head, relation, tail)` triples into the graph and
embed them on the manifold. `enm_entries` is an optional list of
`(category, entity_id, value)` tuples written byte-identically into ENM
with a SHA-256 receipt. The call is synchronous on the graph side and
asynchronous on the training side; the returned `LearnResult` reports
whether a training pass was queued and waited for.

```python
from knowlytix.harness.governance.memory import AgentMemory

memory = AgentMemory(
    store_path="/path/to/findings_store",
    knowledge_path="/path/to/data_science_practices.md",
    device="cuda",
)
result = memory.store_findings(
    triples=[
        ("rf_iris", "has_metric", "test_accuracy"),
        ("rf_iris", "trained_on", "iris"),
    ],
    enm_entries=[("test_accuracy", "rf_iris", 0.973)],
)
```

#### `recall_entity(entity) -> dict`

Return everything the store knows about *entity*: triples where it is the
head, triples where it is the tail, and any ENM categories whose key
contains its name. Fuzzy entity matching is applied so a caller can pass
`"rf_iris"` and recall a stored entity named `"RandomForest/iris"`.

```python
view = memory.recall_entity("rf_iris")
print(view["triples_as_head"])   # [(head, relation, tail), ...]
print(view["enm_values"])        # {"test_accuracy::rf_iris": 0.973, ...}
```

For an exact numeric, prefer `recall_metric(category, entity_id) ->
float | None`, which is the byte-identical ENM lookup with SHA-256
verification.

#### `compare_entities(entity_a, entity_b) -> CompareResult`

Compute the tension energy between two entity embeddings and a
human-readable interpretation across the three bands set up in §4: under
$0.8$ counts as agreement, $0.8$ to $1.6$ counts as irrelevance, above
$1.6$ counts as contradiction. The return also carries `metric_deltas`,
which are the per-category numeric differences between the two entities'
ENM values when both entities share a category.

```python
cmp = memory.compare_entities("rf_iris", "logreg_iris")
print(cmp.tension_energy, cmp.interpretation)
# 0.42 'agreement'   if both classifiers landed near each other
print(cmp.metric_deltas)
# {'test_accuracy': 0.027}
```

#### `check_contradiction(head, relation, tail) -> float`

Probe whether storing the triple `(head, relation, tail)` would contradict
something already in the store. The implementation is the maximum
tension-energy between *tail* and the existing tails for `(head, relation,
?)`. A return value above $\sqrt{2} \approx 1.414$ falls in the
contradiction band of §4 and tells the caller the proposed write would
disagree with what is already there.

```python
existing_tension = memory.check_contradiction(
    head="rf_iris",
    relation="has_test_accuracy",
    tail="LowAccuracy",  # contradicts the stored "HighAccuracy"
)
if existing_tension > 1.4:
    raise ValueError(f"contradiction detected (tension={existing_tension:.2f})")
```

This is the contract that stops an agent from silently overwriting a fact
with its opposite. Whether the caller raises, logs or queues for review is
a policy decision; `AgentMemory` only reports.

### One observation about regex-mode fixtures

The contradiction probe relies on the trained store discriminating
opposite tails, which in turn relies on the ingestion mode. A fixture
built in regex-only mode (the default for the user-guide tutorials) has
looser primitive discrimination than a hybrid-mode fixture; tension
energies for known-contradictory pairs may sit closer to $\sqrt{2}$ than
to $2.0$. The notebook accompanying this section uses an empirical
threshold of $1.4$, which is comfortably above the irrelevance band but
below the regex-mode contradiction signal. A hybrid-mode rebuild
(`ANTHROPIC_API_KEY` set, `--rebuild` passed to `build_fixture.py`)
tightens the contradiction threshold toward $1.8$ and is what production
deployments should use.

### Cross-references

- §4 GMS Primitives Reference — definitions of `score_triple`,
  `tension_energy`, `lookup_enm` that `AgentMemory` builds on.
- §6 Knowledge Ingestion — how a fresh store is populated from a markdown
  knowledge document before findings are added on top.
- §11 GeometricReasoner — the reasoner uses an `AgentMemory` instance
  under the hood for plausibility, multi-hop chain validation and
  `explain_why` traces.
- `knowlytix/harness/governance/memory.py:AgentMemory` — source of record for every
  signature in this section.
- Notebook `10_agent_memory.ipynb` exercises the four methods end to end
  on synthetic findings from three classifiers.

---

## 11. GeometricReasoner

`GeometricReasoner` is the agent-facing surface over the six primitives in §4.
Where a primitive returns one number, the reasoner returns a verdict the agent
can act on. It wraps `score_triple`, `tension_energy`, `check_holonomy`,
`link_predict`, and `query_triples` (the structural lookup behind `find_path`)
into five verbs: `is_plausible`, `validate_chain`, `detect_contradictions`,
`recommend_next`, and `explain_why`. Every verdict it produces is gated on a
`CalibratedThresholds` bundle; there is no silent default. If calibration is
absent the reasoner refuses to answer, which is the right behavior for an
audit-defensible deployment.

### Building `CalibratedThresholds`

The reasoner constructor refuses any value that is not a
`CalibratedThresholds` dataclass. Two paths produce one:

1. **From a fitted `GMSJudge`.** `GMSJudge.calibrate()` (in `knowlytix/harness/testing/judge.py`)
   fits five per-measure logistic regressions (geodesic, path-transport,
   tension, holonomy, phase) and stores the cuts on `judge._thresholds`.
   `CalibratedThresholds.from_judge(judge)` lifts them into the bundle the
   reasoner expects.
2. **From `bootstrap_plausibility_threshold`.** This is the audit path. It is
   defined in `knowlytix/harness/governance/plausibility_calibration.py` and bootstraps the
   geodesic logistic regression over $K$ resamples of the (real,
   corrupted-tail) pairs to give a `ThresholdDistribution` with `mean`, `std`,
   and a 95% CI. Use the mean as `tau_plausibility` and surface the CI in the
   audit report so an auditor can see how stable the threshold is on the
   store. See §25 for the full calibration story including
   `CalibrationSnapshot` and cross-domain agreement.

```python
from knowlytix.harness.governance.reasoner import GeometricReasoner, CalibratedThresholds
from knowlytix.harness.governance.plausibility_calibration import bootstrap_plausibility_threshold
from knowlytix.harness.testing import GMSJudge

# Path 1: lift from a fitted judge (full five-measure calibration).
judge = GMSJudge(store)
judge.calibrate(seed=42)
thresholds = CalibratedThresholds.from_judge(judge)

# Path 2: bootstrap the plausibility cut alone (audit path).
dist = bootstrap_plausibility_threshold(store, n_bootstrap=200, seed=42)
thresholds_audit = CalibratedThresholds(
    tau_plausibility=dist.mean,
    tau_ent=None,
    tau_contra=None,
    tension_status="not_calibrated",
    diagnostics={"plausibility_ci": (dist.ci_low, dist.ci_hi),
                 "n_bootstrap": dist.n_bootstrap},
)

reasoner = GeometricReasoner(store, thresholds, device=DEVICE)
```

When `tension_status != "ok"` the reasoner's geometric contradiction arm
refuses to fire and `detect_contradictions` returns only schema-declared
oppositions (entries the corpus marked with `(r_a, opposite_of, r_b)`). Pass
`require_geometric=True` to make calibration failure surface as a
`CalibrationNotUsable` instead of silently degrading.

### Single-hop plausibility

This is the verb to reach for when the agent has one fact in hand
and wants a fast yes / no with a confidence number — typical use is
a pre-write check before storing a finding, or a sanity probe on a
hypothesis the LLM proposed.

`is_plausible(head, relation, tail)` scores the triple via `score_triple` and
compares against `tau_plausibility`. The return is `(plausible: bool, score:
float)` or `None` when either entity is unknown.

```python
ok, score = reasoner.is_plausible(
    "Nonlinear relationships",
    "has_recommended_model",
    "GradientBoosting",
)
print(f"plausible={ok}, score={score:.4f}")
```

`score` is the raw geodesic distance: lower is more plausible, and
the `plausible` boolean is just `score <= tau_plausibility`. A
`None` return is unknown-entity, not "implausible" — the caller
should refuse the action rather than treat it as a fail.

### Multi-hop chain validation

Single-hop checks miss the case where each step is plausible but
the chain as a whole is incoherent. Multi-hop validation catches
that: it grades every link, then asks whether the composed rotor
along the chain agrees with the chain's first relation as a direct
edge. The two answers together separate "every step is fine, the
chain is fine" from "every step is fine, but the chain is
inconsistent."

`validate_chain(chain)` scores every link individually and, when the relations
in the chain are amenable, computes a holonomy defect against the chain's
first relation as the heuristic direct edge. The returned `ChainValidation`
carries `links`, `mean_score`, `holonomy_defect`, `consistent`, and
`all_plausible`. Without a calibrated `tau_plausibility`, per-link
`plausible` stays `None` rather than defaulting to True.

```python
chain = [
    ("preprocess_data", "has_precedes", "feature_select"),
    ("feature_select", "has_precedes", "train_model"),
    ("train_model",    "has_precedes", "diagnose_accuracy"),
]
validation = reasoner.validate_chain(chain)
print(f"all plausible: {validation.all_plausible}")
print(f"mean score:    {validation.mean_score:.4f}")
if validation.holonomy_defect is not None:
    print(f"holonomy:      {validation.holonomy_defect:.4f}")
```

`all_plausible` is the per-link conjunction; `consistent` is the
holonomy verdict. A run that returns `all_plausible=True,
consistent=False` is the diagnostic signal for an inconsistent
chain — every step looks fine in isolation but the composition
disagrees with the direct relation. §14's `WorkflowPlanner` consumes
this surface to validate proposed plans.

### Worked example: loops and back-edges

Linear chains (`a → b → c → d`) are the easy case. Real workflows
also contain *back-edges* that close loops, and reasoning over those
needs an extra primitive. This section walks through a tiny graph
with two cycles, ingests it into a fresh store, and reads three
geometric verdicts: per-edge plausibility (`score_triple`),
path-vs-direct holonomy (`check_holonomy`), and **closed-loop
holonomy** (rotor composition vs identity, which `check_holonomy`
does not expose directly).

#### The graph

```
        ┌────────── shortcut ───────────┐
        │                               ▼
   ┌────┴───┐  follows  ┌────┐  follows ┌────┐  follows  ┌────┐
   │   a    │──────────▶│ b  │─────────▶│ c  │──────────▶│ d  │
   └────────┘           └─▲──┘          └─┬──┘           └─┬──┘
                          │               │                │
                          └─── shortcut ──┴────────────────┘
   reaches_in_two:  a → c,  b → d        (direct 2-step edges)
```

Five forward edges plus two back-edges; the back-edges close two
3-cycles: `a → b → c → a` (via `c → a`) and `b → c → d → b` (via
`d → b`). The `reaches_in_two` relation supplies an explicit
2-step direct edge so we can compare the composed `follows·follows`
rotor against a known-equivalent direct rotor.

#### Building the store

The regex parser produces `(head, has_<column>, tail)` triples
from each row, so a three-column markdown table per relation
gives us exactly the topology above:

```python
import torch
from docknowlytix.core.config import DocGMSConfig
from docknowlytix.core.ingest import ingest_document
from docknowlytix.core.store import GMSExpertStore
from knowlytix.core.config import GeometryConfig, TrainConfig

GRAPH_MD = (
    "# Loop Topology\n\n"
    "## Forward Edges\n\n"
    "| Node | follows |\n|------|---------|\n"
    "| a | b |\n| b | c |\n| c | d |\n\n"
    "## Back Edges\n\n"
    "| Node | shortcut |\n|------|----------|\n"
    "| c | a |\n| d | b |\n\n"
    "## Two Step Direct\n\n"
    "| Node | reaches_in_two |\n|------|----------------|\n"
    "| a | c |\n| b | d |\n"
)
# ... write GRAPH_MD to a file, then:
config = DocGMSConfig(
    store_path="loop_store",
    geometry=GeometryConfig(m=128),                 # tighter rotors
    train=TrainConfig(epochs=800, device=str(DEVICE)),
    ingest_mode="regex",
)
store = GMSExpertStore(config, device=DEVICE)
ingest_document(store, "loop_graph.md",
                llm=None, config=config, device=DEVICE)
```

A small corpus needs more training capacity than the §3.1 default
to produce compositionally consistent rotors. `m=128` and 800
epochs are both above the §3.1 settings; on this graph they are
what makes the relative ordering below visible.

#### Verdict 1 — every edge is plausible

```python
print(store.score_triple("a", "has_follows",       "b"))         # ≈ 0.01
print(store.score_triple("c", "has_shortcut",      "a"))         # ≈ 0.02
print(store.score_triple("a", "has_reaches_in_two","c"))         # ≈ 0.01
print(store.score_triple("c", "has_shortcut",      "d"))         # ≈ 1.85  (false edge — not in corpus)
```

`score_triple` correctly grades the five real edges as low geodesic
distance and a constructed false edge (`c → d` via `shortcut`) as
high. This is the per-edge baseline before we look at any path.

#### Verdict 2 — path-vs-direct holonomy via `check_holonomy`

`check_holonomy(path, direct)` measures
$\|R_\text{path} \cdot R_\text{direct}^{-1} - I\|_F$ — the residual
after walking the path and undoing the direct edge. Smaller is
more consistent.

```python
hol_consistent = store.check_holonomy(
    ["has_follows", "has_follows"], "has_reaches_in_two")
hol_off = store.check_holonomy(
    ["has_follows", "has_follows"], "has_shortcut")
print(f"hol([fwd,fwd] vs reaches_in_two) = {hol_consistent:.2f}")  # e.g. 3.54
print(f"hol([fwd,fwd] vs shortcut)       = {hol_off:.2f}")         # e.g. 4.48
```

The `reaches_in_two` direct relation is the trained 2-step
equivalent of `follows·follows`, so its holonomy is lower than
the unrelated `shortcut` direct. The signal is the **relative
ordering**, not closeness to zero — on a tiny corpus the rotors
can fit individual edges without their compositions hitting the
noise floor, so absolute holonomy stays moderate.

#### Verdict 3 — closed-loop holonomy via direct rotor composition

For a closed loop the natural check is "does the composed rotor
return to identity?", which is **not** what `check_holonomy`
measures (it always compares against another rotor). Reach for
`store.transport.compute_path_rotor` instead:

```python
fwd  = store.adapter.relation_to_idx["has_follows"]
back = store.adapter.relation_to_idx["has_shortcut"]
m    = store.config.geometry.m

I = torch.eye(m, device=DEVICE)

# Closed loop  a → b → c → a:  R_back ∘ R_fwd ∘ R_fwd
R_loop = store.transport.compute_path_rotor([fwd, fwd, back])

# Open path    a → b → c → d:  R_fwd ∘ R_fwd ∘ R_fwd  (does not return)
R_open = store.transport.compute_path_rotor([fwd, fwd, fwd])

loop_defect = ((R_loop - I) ** 2).sum().sqrt().item()    # ≈ 4.33
open_defect = ((R_open - I) ** 2).sum().sqrt().item()    # ≈ 5.89
print(f"closed loop  ||R - I||_F = {loop_defect:.2f}")
print(f"open path    ||R - I||_F = {open_defect:.2f}")
```

The closed-loop residual is consistently smaller than the open-path
residual: the cycle was attested in the corpus (`a → b`, `b → c`,
`c → a` are all real triples), so training pulled the composition
toward identity. The open path goes one step beyond the cycle and
has no such constraint, so the residual is larger.

#### Reading the three verdicts together

The three checks form a stack — each layer rules out a different
failure mode:

| Layer       | Verdict                                                       | What it catches                                  |
|-------------|---------------------------------------------------------------|--------------------------------------------------|
| Edge        | `score_triple(h, r, t) ≤ τ_plausibility`                       | A claimed edge is not in the trained corpus      |
| Path        | `check_holonomy(path, direct)` low vs alt direct relations    | The path and its direct-equivalent disagree      |
| Closed loop | `\|\|compute_path_rotor(loop) - I\|\|_F` low vs an open path  | A claimed cycle does not actually return         |

A run where all three pass is the "consistent multi-hop reasoning"
signal `WorkflowPlanner` and the §16 release gate consume. A run
where edges pass but paths fail is the inconsistent-chain
diagnostic from `validate_chain`. A run where edges and paths pass
but the closed-loop residual is comparable to an open path of the
same length is the signal that a claimed cycle is decorative — the
agent named a back-edge but the corpus does not actually close.

Cross-references: §4.3 `check_holonomy` is the path-level
primitive; `gms/graph/transport.py:compute_path_rotor` and
`triangle_holonomy` are the underlying rotor-composition utilities;
`gms_agent_builder_ext/chapters/ch06.tex` derives the
holonomy-defect formula from the Cayley parameterization.

### `explain_why`

The two verbs above return numbers; this one returns a *verdict* an
auditor can read. Use it when the output of the reasoning step
will land in a transcript or a release-gate report — the
`Explanation` object combines direct plausibility, path search, and
contradiction detection into one verdict string plus the supporting
evidence.

`explain_why(head, relation, tail)` combines direct plausibility with path
search and contradiction detection into a single `Explanation`. The verdict is
one of `supported`, `supported (direct only)`, `unsupported`, `contradicted`,
`unknown (entities not in store)`, or `uncalibrated (no plausibility
threshold)`. The two-provenance contradiction detector (schema and geometric)
is consulted so the explanation carries the same audit trail as standalone
`detect_contradictions` calls.

```python
explanation = reasoner.explain_why(
    "Nonlinear relationships",
    "has_recommended_model",
    "GradientBoosting",
)
print(f"verdict:       {explanation.verdict}")
print(f"direct score:  {explanation.direct_score:.4f}")
print(f"paths found:   {len(explanation.supporting_paths)}")
```

The six verdict values map cleanly onto operator action:
`supported` and `supported (direct only)` accept; `contradicted`
hard-rejects; `unsupported` routes to review; `unknown` and
`uncalibrated` refuse to answer. The last two are diagnostic, not
substantive — the caller's job is to widen the corpus or run
calibration, not to second-guess the verdict.

### `recommend_next`

The first three verbs grade an existing claim; this one *generates*
candidates. Plug it in when the question is "given that I just did
X, what should I do next?" — typically inside a workflow planner
that needs ranked suggestions rather than a single yes / no.

`recommend_next(context, relation, top_k)` wraps `link_predict` and is the
verb the workflow planner consumes (§14). The return is a list of
`(entity, geodesic)` pairs sorted lowest-distance first.

```python
nxt = reasoner.recommend_next(
    "preprocess_data", "has_precedes", top_k=3,
)
for entity, d in nxt:
    print(f"  d={d:.4f}  {entity}")
```

The list is sorted by geodesic distance (lowest first). The gap
between the first and second `d` is the confidence margin: a tight
gap means the top suggestion is barely ahead, which the planner
should treat as a tie. Numeric entities are excluded from the
ranking automatically — the type-constrained tail-table from §4.5
keeps "0.85" out of a `has_precedes` recommendation.

### Cross-references

- §4 GMS Primitives Reference — the primitives the reasoner composes.
- §12 Claim Verification — uses the same calibrated thresholds for the FACT
  and REASONING claim routes.
- §25 Threshold Calibration — `bootstrap_plausibility_threshold`,
  `CalibrationSnapshot`, cross-domain agreement.
- `knowlytix/harness/governance/reasoner.py:GeometricReasoner` — full source.
- `knowlytix/harness/governance/plausibility_calibration.py:bootstrap_plausibility_threshold`
  — bootstrap producer of `ThresholdDistribution`.
- `gms_agent_builder_ext/chapters/ch06.tex` — the chapter that introduces
  geometric reasoning end to end.
- Notebook `11_geometric_reasoner.ipynb` exercises every verb on the prebuilt
  fixture and plots holonomy defect against chain length.

---

## 12. Claim verification

`ClaimVerifier` turns an agent's natural-language output into a structured
verdict. The agent (or an upstream extractor) produces a list of typed
`Claim`s; the verifier routes each one to the right GMS primitive and emits
a `ClaimVerdict` carrying `passed`, `score`, `severity`, `evidence`, and
`raw_signal`. `verify_all` aggregates the per-claim results into an
`OverallVerdict` with a pass rate and a severity distribution that §20's
classifier and §23's tracer both consume.

### Five claim types and their primitive routes

| `ClaimType` | Verification primitive                 | `metadata` keys                                         |
|-------------|----------------------------------------|---------------------------------------------------------|
| `FACT`      | `query_triples` then `score_triple`    | `head`, `relation`, `tail`                              |
| `NUMERIC`   | `lookup_enm`                           | `category`, `entity_id`, `claimed_value`                |
| `THRESHOLD` | `lookup_enm` plus inequality check     | `category`, `entity_id`, `operator` (`ge`/`gt`/`le`/`lt`), `limit` |
| `COMPARISON`| `lookup_enm` for both entities         | `category`, `entity_a`, `entity_b`, `metric`, `direction` (`greater`/`less`) |
| `REASONING` | `check_holonomy`                       | `relation_path` (list of relation names), `direct_relation`         |

`FACT` runs the structural attestor first (`query_triples` exact graph match,
fuzzy-matched on entity strings). When the corpus already asserts the triple,
the verdict passes with confidence 0.95 regardless of geodesic stale-ness;
this is the runtime-extracted-fact path covered in §23. When the graph match
misses, geodesic falls through with `tau_plausibility` as the cut.

`NUMERIC` accepts a configurable relative tolerance (`numeric_tolerance`,
default 0.01 — explicitly *not* a learned cut, it is a precision policy).
When `numeric_range_index` is supplied the verifier runs a second attestor
(per-category `(min, max)` range plausibility) alongside the ENM exact-match
check; ENM-PASS plus range-FAIL routes the verdict to review rather than PASS.

`THRESHOLD` calls `lookup_enm` and applies the inequality. The signed margin
is exposed as `raw_signal` so dashboards can show distance-from-cut.

`COMPARISON` reads two ENM values for the same metric and applies the
direction. A missing ENM entry on either side fails closed.

`REASONING` calls `check_holonomy` on `relation_path` versus
`direct_relation`. The holonomy defect is exposed as `raw_signal`. Without a
calibrated `holonomy_threshold`, the verifier refuses the route — same rule
as the plausibility threshold.

### Building a verifier with calibrated thresholds

```python
from knowlytix.harness.governance.verifier import ClaimVerifier, Claim, ClaimType
from knowlytix.harness.governance.plausibility_calibration import bootstrap_plausibility_threshold
from knowlytix.harness.testing import GMSJudge

# Plausibility from bootstrap (audit path with CI).
dist = bootstrap_plausibility_threshold(store, n_bootstrap=200, seed=42)
plaus = dist.mean

# Holonomy from full GMSJudge.calibrate.
judge = GMSJudge(store)
judge.calibrate(seed=42)
holo = judge._thresholds.get("holonomy")

verifier = ClaimVerifier(
    store,
    plausibility_threshold=plaus,
    holonomy_threshold=holo,
    numeric_tolerance=0.01,
)
```

`plausibility_threshold` and `holonomy_threshold` have no default values; if
you pass `None` the corresponding routes (`FACT`, `REASONING`) return error
verdicts pointing the user at `GMSJudge.calibrate()`. This is deliberate —
silent defaults are not safe in audit-grade deployments (see §25).

### Running verification

A claim is a `Claim` dataclass — a `ClaimType`, the human-readable
text the agent produced, and a `metadata` dict carrying the
typed payload the route in §12.1 expects. The verifier never parses
`text`; it routes on `type` and reads `metadata`. Constructing the
list explicitly (rather than parsing free text) is what keeps the
verifier deterministic and the audit log replayable.

```python
claims = [
    Claim(
        type=ClaimType.FACT,
        text="GradientBoosting is recommended for nonlinear relationships",
        metadata={"head": "Nonlinear relationships",
                  "relation": "has_recommended_model",
                  "tail": "GradientBoosting"},
    ),
    Claim(
        type=ClaimType.THRESHOLD,
        text="High-risk credit AUC must be at least 0.85",
        metadata={
            "category": "evaluation_standards",
            "entity_id": "High-risk (credit, lending)/"
                         "Fairness, robustness, explanation, "
                         "cross-validation/Minimum AUC",
            "operator": "ge",
            "limit": 0.85,
        },
    ),
]
```

The two claims above exercise two different routes. The `FACT`
claim lands on `query_triples` then `score_triple`; the `THRESHOLD`
claim lands on `lookup_enm` plus the `ge` inequality with the
0.85 limit. Each route reads only the metadata keys named in §12.1's
table — extra keys are ignored, missing keys raise on construction.

```python
overall = verifier.verify_all(claims)
print(f"status={overall.overall_status} pass_rate={overall.pass_rate:.0%}")
for v in overall.claim_verdicts:
    print(f"  [{'+' if v.passed else 'X'}] {v.claim.text}")
    print(f"      {v.evidence}")
```

`verify_all` runs every claim through its route and aggregates the
verdicts into one `OverallVerdict`. Per-claim, `v.passed` is the
binary outcome, `v.evidence` is the human-readable trail (the
matched triple, the looked-up ENM value, the geodesic score), and
`v.severity` is the four-level base severity §20 promotes by tier.

`OverallVerdict` carries `n_passed`, `n_total`, `severity_distribution` (a
dict with counts per severity level), and `claim_verdicts`. The severity
levels — `info`, `warning`, `error`, `critical` — are the inputs to the §19
`SeverityClassifier` which can escalate further based on the agent's risk
tier (e.g. `action_taking` promotes any `fact_incorrect` to `CRITICAL`).

### Cross-references

- §4 GMS Primitives Reference — the five primitives this verifier composes.
- §11 GeometricReasoner — same calibrated bundle drives the reasoner.
- §20 Severity & FailureCode Taxonomy — consumes the verifier's per-claim
  severity field and tier-promotes it.
- §25 Threshold Calibration — produces `tau_plausibility` and `tau_holonomy`.
- `knowlytix/harness/governance/verifier.py:ClaimVerifier` — full source.
- `knowlytix/harness/testing/judge.py:GMSJudge.calibrate` — five-measure calibration.
- `gms_agent_builder_ext/chapters/ch08.tex` — the chapter that introduces
  typed claim verification end to end.
- Notebook `12_claim_verification.ipynb` exercises all five claim types and
  plots the per-type pass-rate bar chart.

---

## 13. Runtime learning

The store does not have to be frozen at ingest time. `RuntimeLearner`
adds new triples after the agent has started, while the rest of the
system keeps serving queries against the same `GMSExpertStore`. This
is how an agent absorbs verified findings from its own tool calls (a
trained model's accuracy, a dataset's row count, a comparison verdict)
without rebuilding the store from scratch.

### Two-phase protocol

Runtime growth is split into two phases so interactive use stays
responsive.

- **Phase 1 — synchronous, fast.** Verify triples against the
  contradiction gate, append them to the document graph, expand the
  adapter's entity and relation indices, and write any numeric values
  to ENM. Graph-level queries (`query_triples`, `lookup_enm`) work
  immediately after this phase returns.
- **Phase 2 — background, async.** Expand the embedding tables and
  add new Cayley rotors, then run Riemannian gradient descent on the
  new entities and rotors with the existing parameters frozen. This
  is what makes `score_triple` for the new triples converge below the
  plausibility threshold.

`learn_triples` returns a `LearnResult` immediately; queue Phase 2 by
calling `flush_training`, then block on completion with
`wait_for_training`. `wait_for_training` auto-flushes the queue, so a
caller who follows the natural pattern `learn_triples` then
`wait_for_training` is guaranteed to actually wait for every queued
job.

### Minimal API

The natural call pattern is `learn_triples` → check `training_queued`
→ `flush_training` + `wait_for_training`. The first call returns
the moment Phase 1 is done, so a caller that does not need the new
embeddings to settle (e.g., a batch ingest where downstream
verification will run after a separate flush) can skip the wait.
Every example in §10 and §13 that needs `score_triple` to converge
on the new entities follows the wait pattern below.

```python
from docknowlytix.core.learn import RuntimeLearner
from docknowlytix.core.extract import ExtractedTriple

learner = RuntimeLearner(store, config.learn)

triples = [
    ExtractedTriple(
        head="RandomForest", relation="trained_on",
        tail="iris_dataset"),
    ExtractedTriple(
        head="RandomForest", relation="has_accuracy",
        tail="1.0", numeric_value=1.0),
]

result = learner.learn_triples(triples, device=DEVICE)
if result.training_queued:
    learner.flush_training()
    learner.wait_for_training(timeout=120.0)

print(f"Triples added: {result.n_triples_added}")
```

The two `ExtractedTriple` rows show the two-form pattern from §7
(`ExtractedTriple` and `ENMEntry` fields). The first has a string
tail and goes only into the document graph. The second has a
string tail *and* a populated `numeric_value`,
which routes 1.0 byte-identically into the ENM register alongside
the graph triple. Always populate `numeric_value` on numeric
findings — without it the value is recoverable only as the string
`"1.0"`, and `lookup_enm` cannot return it as an exact float.

**Tip:** set `config.learn.n_steps = 200` (default is 50) so fact
triples converge below the plausibility threshold. Without enough
background training steps, `score_triple` may return geodesic
distances above $\theta = 1.2$ and fact claims that depend on the
new triples will fail.

### Pre-write contradiction gate

Phase 1 fires the contradiction gate before accepting writes. The
gate enforces two graph-declared schema rules: `opposite_of`
relations cannot both hold for the same head and tail, and
`is_functional` relations are single-valued per head. Both rules are
mined from the document graph itself, not hard-coded — so the gate
fires only on what the corpus declared. Geometric contradiction
detection over the manifold belongs to `GeometricReasoner.detect_contradictions`
at query time, with a calibrated `tau_contra` and recorded
provenance, not at write time.

> **Cap-trained runtime relation expansion.**
> `RuntimeLearner._expand_rotors` must grow `rho_raw` and `log_scale`
> when `model.cap_enabled`, otherwise `score_triple` after
> `PolicyEngine.encode_rules` will hit a CUDA out-of-bounds error.
> Inject seed triples into the adapter before main training instead
> of relying on `learn_triples` to pick up hard-negatives and tension
> pairs after the fact.

### Cross-references

- `knowlytix/knowledge/learn.py:RuntimeLearner.learn_triples` and
  `RuntimeLearner._expand_rotors`
- `gms_agent_builder_ext/chapters/ch08.tex §8.3` for the runtime
  growth protocol in book form
- §4.1 `score_triple` — how geodesic scores measure plausibility
- §6 Knowledge Ingestion — the offline counterpart to runtime learning

---

## 14. Workflow planner

A workflow planner answers four questions about a sequence of
governed-tool calls: what should run next, what does a full plan
toward a goal look like, is a proposed plan internally consistent,
and is a single step ready to run? `WorkflowPlanner` answers all
four by delegating to GMS primitives — it owns no geometry of its
own. `link_predict("precedes")` ranks candidate next steps,
`check_holonomy` measures whether the composed `precedes` rotor
along the plan agrees with the direct edge, and `query_triples` over
`requires` resolves prerequisites.

### Baseline API

The planner exposes three verbs corresponding to the four questions
above. `next_step` answers "what runs next given what I've done";
`full_plan` builds the sequence from the current context to a goal;
`validate_plan` grades a proposed sequence end-to-end. The three
share a constructor — the planner takes the same store the rest of
the stack uses, so the `precedes` and `requires` edges it consults
are the ones the corpus declared.

```python
from knowlytix.harness.governance.planner import WorkflowPlanner

planner = WorkflowPlanner(store, device=DEVICE)

# Recommend the single next step
next_step = planner.next_step(
    completed_steps=["load_data"], context="iris_pipeline")
print(f"  {next_step.tool} (conf={next_step.confidence:.2f})")

# Generate and validate a full plan
plan = planner.full_plan(goal="deploy", context="iris_pipeline", max_steps=10)
validation = planner.validate_plan([s.tool for s in plan])
print(f"Holonomy: {validation.holonomy:.4f}")
print(f"Prerequisites satisfied: {validation.prerequisites_satisfied}")
```

`next_step.tool` is the single ranked recommendation;
`next_step.confidence` is the geodesic-derived margin between it
and the runner-up (a tight margin signals an uncertain pick that
the orchestrator should treat as a tie). `full_plan` returns a
list of `PlanStep` objects, which is why `validate_plan` is
called on `[s.tool for s in plan]` — the validator grades sequences
of tool names, not step objects.

`PlanValidation` always carries three baseline fields: `holonomy`
(geometric defect of the `precedes` chain, lower is better),
`prerequisites_satisfied` (a hard boolean from the prerequisite
table), and `violations` (a list of human-readable strings naming
each prerequisite miss). These three are uncalibrated — the
calibrated `verdict` / `confidence` fields below appear only after
`planner.calibrate` has been run.

### Calibrated decision

Calling `planner.calibrate(cohort)` fits a logistic classifier on
six features extracted per plan: `holonomy`, `n_violations`,
`n_unknown_steps`, `core_coverage`, `mean_precedes_score`, and
`plan_length_delta`. The classifier produces a probability the plan
is valid; `tau_accept` and `tau_reject` partition that probability
into three verdicts. After calibration, `validate_plan` populates
three additional fields:

- `verdict` — `"accept"` if `prob_valid >= tau_accept`, `"reject"`
  if `prob_valid <= tau_reject`, otherwise `"review"`.
- `confidence` — the raw probability in $[0, 1]$.
- `calibrated_features` — the six-feature dict the classifier
  scored.

```python
from knowlytix.harness.governance.plan_benchmark import build_cohort

cohort = build_cohort(seed=42)
metrics = planner.calibrate(cohort, seed=42)
print(f"FPR={metrics['fpr']:.3f}, FNR={metrics['fnr']:.3f}")
```

`metrics` reports held-out false-positive rate (FPR; invalid plans
that were accepted) and false-negative rate (FNR; valid plans that
were rejected) with bootstrap 95 percent
confidence intervals. Calibrations persist via `save_calibration`
and `load_calibration` mirroring the `GMSJudge` format.

### Policy co-attestor

The geometric attestor catches subtle ordering and novelty failures
that a learned model will recognize. A separate declarative attestor
catches hard rules that should never be violated regardless of what
the geometry learned. `PlanPolicy` runs an AND-gated rule pack — if
the geometric verdict accepts but a policy rule rejects, the plan is
rejected.

```python
from knowlytix.harness.governance.plan_policy import PlanPolicy, data_science_rules

planner.attach_policy(PlanPolicy(data_science_rules()))
validation = planner.validate_plan(plan)
print(validation.verdict, validation.policy_verdict)
```

`data_science_rules()` ships eleven rules with stable IDs so audit
logs stay portable: vocabulary, no-duplicates, load-data-present,
preprocess-before-train, diagnostic-requires-train,
release-requires-compare, release-requires-fairness,
deploy-requires-release-gate, explain-requires-train,
plan-not-empty, plan-length-sane. Ten are blocking, the
length-sanity rule is advisory.

After `attach_policy`, `PlanValidation` carries two more fields:

- `policy_verdict` — `"accept"` if every blocking rule held,
  `"reject"` otherwise.
- `policy_reasons` — one human-readable string per rule that fired
  (blocking or advisory).

A plan is released for execution only when both the calibrated
verdict and the policy verdict are `accept`.

### Cross-references

- `knowlytix/harness/governance/planner.py:WorkflowPlanner` and
  `knowlytix/harness/governance/plan_policy.py:PlanPolicy`
- `gms_agent_builder_ext/chapters/ch09.tex` for the planner in book
  form
- §4.5 `link_predict` and §4.3 `check_holonomy` — the GMS primitives
  the planner consumes
- §18 Behavioral contracts — for the trace-level analogue of plan
  validation

---

## 15. Constrained LLM (mock and real)

A constrained LLM is the architectural answer to "let an LLM drive the agent without letting it run unchecked." The pattern, "LLM proposes, GMS approves," routes every LLM output through one of three governance channels. The interpretation channel is ungated and produces display-only narrative. The proposal channel is plausibility-gated through `LogicVerifier.verify_action`. The claim channel is contradiction-gated through `LogicVerifier.verify_domain_triples`. A fail-closed default rejects every proposal and every claim when the GMS store has no trained model, so an authority leak (LLM operating without geometric verification) cannot occur.

The library exposes two implementations of this contract. `MockConstrainedLLM` is deterministic, requires no API key, and is the right starting point for the tutorial. `ConstrainedLLM` is the production class wired to a real LLM backend and produces governance verdicts of the same shape. Both live in `knowlytix/harness/governance/governed_llm.py`.

### 15.A Mock walkthrough

`MockConstrainedLLM` uses pattern matching and template-based generation rather than actual LLM calls. Outputs are repeatable across runs, which is why it is the canonical surface for tutorials and CI:

```python
from docknowlytix.core.config import VerifyConfig
from agents.ds_agent.agents.base import LogicVerifier
from knowlytix.harness.governance.governed_llm import MockConstrainedLLM

verifier = LogicVerifier(store, VerifyConfig(min_plausibility=1.5))
llm = MockConstrainedLLM(store, verifier)

# Channel 1: ungated interpretation (display-only)
narrative = llm.interpret(
    "train_model",
    {"accuracy": 0.92, "f1": 0.91, "model_name": "RandomForest", "n_epochs": 100},
    context="training_phase",
)

# Channel 2: plausibility-gated proposals
proposals = llm.propose_next(
    context="training",
    completed=["load_dataset", "preprocess_data"],
)
for p in proposals:
    print(f"  {p.tool}: approved={p.approved}, score={p.score}")

# Channel 3: contradiction-gated claims
claims = llm.extract_claims(narrative)
for c in claims:
    print(f"  {c.text}: approved={c.approved}, tension={c.tension}")
```

The `Proposal` dataclass carries `tool`, `args`, `reason`, `approved`, `score`, and `rejection_reason`. The `GatedClaim` dataclass carries `text`, `triple`, `approved`, `tension`, and `rejection_reason`. Helpers `format_proposal_report` and `format_claim_report` produce a tabular audit view.

### 15.B Real `ConstrainedLLM` (Anthropic backend)

`ConstrainedLLM` (also in `knowlytix/harness/governance/governed_llm.py`) keeps the same three-channel contract but routes interpretation, proposal, and claim extraction through a real LLM backend. The expected backend is `AnthropicBackend` from `knowlytix/knowledge/llm_backend.py`, which resolves the API key via `knowlytix.harness.governance.api_key.get_anthropic_api_key`. There is no mock fallback: if the key is absent the constructor raises `AnthropicAPIKeyError` so deployments cannot accidentally exercise canned text instead of real LLM behavior.

```python
import os
from docknowlytix.core.config import VerifyConfig
from docknowlytix.core.llm_backend import AnthropicBackend
from agents.ds_agent.agents.base import LogicVerifier
from knowlytix.harness.governance.governed_llm import ConstrainedLLM
from knowlytix.harness.governance.api_key import AnthropicAPIKeyError

verifier = LogicVerifier(store, VerifyConfig(min_plausibility=1.5))

try:
    backend = AnthropicBackend(model="claude-opus-4-7")
    llm = ConstrainedLLM(
        store=store,
        verifier=verifier,
        llm=backend,
        allowed_tools=[
            "load_dataset", "preprocess_data", "train_model",
            "evaluate_model", "diagnose_accuracy", "explain_model",
        ],
    )
except AnthropicAPIKeyError as exc:
    print(f"No API key configured: {exc}")
    raise
```

The five rejection paths inside `ConstrainedLLM.propose_next` are exercised in order: fail-closed (no trained model), backend error (LLM call failed), forbidden tool (not in `allowed_tools`), co-attestor block (rule pack rejected the proposal), and geometric block (`LogicVerifier` rejected the proposal). Each path produces a `Proposal` with `approved=False` and a populated `rejection_reason`, so the audit log records the reason rather than receiving a silent empty list. `extract_claims` follows the same rejection contract over the contradiction gate.

The two classes are interchangeable wherever code consumes the `interpret`, `propose_next`, `extract_claims` interface: tests write against `MockConstrainedLLM`, deployments swap in `ConstrainedLLM`, and the gating layers downstream see the same `Proposal` and `GatedClaim` shapes.

### Cross-references

- `knowlytix/harness/governance/governed_llm.py:MockConstrainedLLM` and `knowlytix/harness/governance/governed_llm.py:ConstrainedLLM`
- `knowlytix/knowledge/llm_backend.py:AnthropicBackend`
- `knowlytix/harness/governance/api_key.py:AnthropicAPIKeyError`
- `gms_agent_builder_ext/chapters/ch11.tex` for the constrained-LLM chapter in book form

---

## 16. LLM backends and API keys

The library calls out to a Large Language Model in two places: hybrid-mode ingestion (§6) and the real `ConstrainedLLM` channels (§15.B). Both consume the same abstraction — `docknowlytix.core.llm_backend.LLMBackend` — so a single backend choice serves the whole pipeline. This section documents the supported backends, how to configure API keys for hosted providers, how to run open-weight models locally, and how to plug in a backend the library does not ship with (such as OpenAI or Gemini).

### The `LLMBackend` abstraction

The protocol is intentionally small so that a wrapper around any provider fits in ~30 lines:

```python
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    def call(self, system: str, user: str, max_tokens: int = 2048) -> str: ...
    @property
    @abstractmethod
    def model_name(self) -> str: ...
```

`call` accepts a system prompt and a user message and returns text. `model_name` is used in audit packets so a reviewer can tell which LLM produced a given verdict. Anything richer (streaming, tool-use schemas) is the caller's responsibility — keep the wrapper deterministic-by-default with `temperature=0`.

### Backends shipped with the library

#### 1. `AnthropicBackend` — hosted Claude

```python
from docknowlytix.core.llm_backend import AnthropicBackend

llm = AnthropicBackend(model="claude-opus-4-7")
text = llm.call(system="You answer in one sentence.", user="What is 2+2?")
```

`AnthropicBackend.__init__` resolves the API key by calling `knowlytix.harness.governance.api_key.get_anthropic_api_key()`, which checks two sources in order:

1. The `ANTHROPIC_API_KEY` environment variable.
2. The output of `pass show anthropic/api-key` (the Unix `pass(1)` password store).

If neither yields a key the constructor raises `AnthropicAPIKeyError` — a fatal exception that halts the run rather than degrading to canned responses. This is intentional: the governance layers downstream interpret the LLM's output and only produce meaningful audit evidence when they are exercised against real model behavior. See `knowlytix/harness/governance/api_key.py:24` for the exception and `knowlytix/harness/governance/api_key.py:35` for the resolver.

Two ways to set the key:

```bash
# Option A: export in the shell (per-session)
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B: store in `pass(1)` (recommended for developer workstations)
pass insert anthropic/api-key
```

Do not hard-code keys in notebooks, scripts, or YAML configs. Any file that lands in source control is one screen-share away from disclosure.

#### 2. `LocalTransformersBackend` — open-weight on local GPU

```python
from docknowlytix.core.llm_backend import LocalTransformersBackend

llm = LocalTransformersBackend(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
)
```

This backend uses HuggingFace Transformers and loads the model in `float16` on the requested device. The first call pays the model-load cost (a few seconds for an 8 B model on the GB10); subsequent calls reuse the cached weights. Pass any HuggingFace model identifier — small instruct-tuned models like `Qwen/Qwen2.5-7B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3` work well for governance probes. Greedy decoding (`do_sample=False`) is the default so verdicts are reproducible across reruns.

For models behind HuggingFace's gated-access program, set `HF_TOKEN` in the environment before instantiation:

```bash
export HF_TOKEN="hf_..."
```

The backend will not attempt to download or run a model when CUDA is unavailable; install the CPU-only path explicitly with `device="cpu"` if you need it (much slower, useful only for smoke tests).

#### 3. `create_backend(config)` — factory selection from config

`knowlytix/knowledge/llm_backend.py:create_backend` reads three fields from `ConvertConfig` (defined in `knowlytix/knowledge/config.py`):

| Field               | Default                          | Meaning                                                       |
|---------------------|----------------------------------|---------------------------------------------------------------|
| `llm_backend`       | `"anthropic"`                    | `"anthropic"` selects `AnthropicBackend`; `"local"` selects `LocalTransformersBackend` |
| `llm_model`         | `"claude-sonnet-4-20250514"`     | Model identifier passed to `AnthropicBackend`                  |
| `local_model_name`  | `""`                             | HuggingFace model identifier; required when `llm_backend == "local"` |

```python
from docknowlytix.core.config import ConvertConfig
from docknowlytix.core.llm_backend import create_backend

cfg = ConvertConfig(llm_backend="local", local_model_name="Qwen/Qwen2.5-7B-Instruct")
llm = create_backend(cfg)        # -> LocalTransformersBackend
```

Use the factory when the backend choice is configuration-driven (e.g., a YAML file that operators edit). Construct the backend class directly when you want explicit control in code.

### Backends not yet shipped — OpenAI, Gemini, OpenAI-compatible servers

The library does not ship adapters for OpenAI or Gemini. Both providers' SDKs are stable enough that a one-file subclass is the right level of effort. The pattern is the same in every case: instantiate the provider's client in `__init__`, translate the system + user pair into the provider's message format, and return the assistant text.

#### OpenAI (hosted)

```python
import os
from docknowlytix.core.llm_backend import LLMBackend


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or pass api_key=..."
            )
        self._client = OpenAI(api_key=key)
        self._model = model

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model
```

Set the key the same way as Anthropic:

```bash
export OPENAI_API_KEY="sk-..."
```

#### Google Gemini (hosted)

```python
import os
from docknowlytix.core.llm_backend import LLMBackend


class GeminiBackend(LLMBackend):
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        from google import genai
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Export it or pass api_key=..."
            )
        self._client = genai.Client(api_key=key)
        self._model = model

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            config={"system_instruction": system, "max_output_tokens": max_tokens, "temperature": 0},
            contents=user,
        )
        return resp.text

    @property
    def model_name(self) -> str:
        return self._model
```

Key:

```bash
export GOOGLE_API_KEY="..."
```

#### OpenAI-compatible local servers (vLLM, Ollama, llama.cpp server, LM Studio)

Many open-weight serving stacks expose an OpenAI-compatible `/v1/chat/completions` endpoint. Reuse the OpenAI SDK and override `base_url`:

```python
from openai import OpenAI

class LocalOpenAIBackend(OpenAIBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1"):
        from openai import OpenAI
        self._client = OpenAI(api_key="not-needed", base_url=base_url)
        self._model = model
```

Useful in deployment scenarios where the model lives in a container or a separate node — you can swap from `AnthropicBackend` to `LocalOpenAIBackend(model="Qwen2.5-7B-Instruct")` with a one-line change and exercise the same governance pipeline against an open-weight model.

### Choosing a backend

| Scenario                                      | Recommended backend                |
|-----------------------------------------------|-------------------------------------|
| Hosted, highest-quality verdicts, audited     | `AnthropicBackend("claude-opus-4-7")` |
| Hosted, alternative provider                  | Custom `OpenAIBackend` or `GeminiBackend` |
| Open-weight on the same machine               | `LocalTransformersBackend`         |
| Open-weight behind a serving endpoint         | `LocalOpenAIBackend` (custom)       |
| Reproducibility under audit                   | Any backend with `temperature=0` and a pinned model identifier |

Pin the exact model identifier (including the date suffix or version) in your `ConvertConfig`. Models silently rotate at hosted providers; an audit trail that says only `"claude"` cannot defend against "the answer changed because the model changed."

### Where the LLM is consumed

The same backend is plugged in at two sites:

1. **Hybrid ingest** — pass `llm` to `ingest_document(...)` (see §6). When the backend is `LocalTransformersBackend`, the first ingest pays the model-load cost; subsequent rebuilds reuse the in-memory weights if you keep the same Python process alive.
2. **`ConstrainedLLM`** — pass `llm` to the `ConstrainedLLM` constructor (see §15.B). All three channels (`interpret`, `propose_next`, `extract_claims`) flow through the single backend.

Because the abstraction is the same, switching backends is a one-line change at the call site. This is intentional: the governance layers do not depend on a specific provider, so reproducing a result against a second backend is a built-in cross-check.

### A note on key handling for shared environments

Tutorial notebooks in this repository load keys from the surrounding shell. Before publishing or sharing a notebook:

- Confirm no key string appears in cell outputs (`grep -r "sk-" notebooks/` is a quick check).
- Strip outputs from `.ipynb` files committed to source control: `jupyter nbconvert --clear-output --inplace <nb>`.
- For team or shared accounts, use a secrets manager (Vault, AWS Secrets Manager, Doppler) rather than per-user environment variables.

### See also

- `knowlytix/knowledge/llm_backend.py` — `LLMBackend`, `AnthropicBackend`, `LocalTransformersBackend`, `create_backend`
- `knowlytix/harness/governance/api_key.py:35` — `get_anthropic_api_key`, the canonical resolver
- §6 (Knowledge ingestion) — where the LLM is consumed during ingest
- §15.B (Real `ConstrainedLLM`) — where the LLM is consumed at inference time

---

## 17. Governed Tool Gateway

The `GovernedToolGateway` is the single point through which every tool
call flows. Its job is to enforce the three gates defined in §2.2 —
schema, policy, plausibility — in that order, short-circuit on the
first failure, and return a fully audit-ready `GatewayResult`. A
caller that wants to compose its own enforcement stack on top of the
governed agent does so by configuring this one object: rules go on
the `PolicyEngine`, fault profiles go on the underlying
`ToolGateway`, and the plausibility threshold lives directly on the
gateway. The result is that no tool execution path bypasses the
gates; an auditor reading the transcript sees a uniform
gate-by-gate record per call regardless of which tool ran.

### 17.1 Wiring a gateway

```python
from knowlytix.harness.governance.session import SessionState
from knowlytix.harness.governance.governed_gateway import GovernedToolGateway
from knowlytix.harness.governance.tools import TOOL_DEFINITIONS
from knowlytix.harness.testing import PolicyEngine

session = SessionState()
policy  = PolicyEngine(store=ds_practices_store)
gateway = GovernedToolGateway(
    session=session,
    store=ds_practices_store,
    policy=policy,
    theta_plausibility=1.5,
    tool_schemas=TOOL_DEFINITIONS,
    plausibility_relation="should_call",
    device="cuda",
)
```

The constructor takes seven positional or keyword arguments. The
`session` is the mutable scratchpad shared across calls. The `store`
is the GMS knowledge store used by gates 2 and 3. The `policy` is a
pre-configured `PolicyEngine`; pass `None` to get an empty engine
constructed from the same store. `theta_plausibility` sets the hard
geodesic threshold for gate 3 (default 1.5, matching the calibrated
policy threshold from `gms_agent_builder_ext/chapters/ch07.tex
§7.4`). `tool_schemas` is the JSON schema list gate 1 validates
against; if you omit it, the gateway imports
`knowlytix.harness.governance.tools.TOOL_DEFINITIONS`. `plausibility_relation`
selects the relation used for gate 3; the default `"should_call"`
matches the agent-centric convention.

### 17.2 The three gates in detail

Every call to `gateway.execute(tool_name, args, context)` runs the
three gates in sequence. Each gate has a single responsibility, a
defined input, and a hard-block path:

| # | Gate | Implementation | Block reason format |
|---|------|----------------|---------------------|
| 1 | Schema | `_validate_schema` against the tool's `input_schema` | `"Schema validation failed: Missing required parameter: <p>"` or `"Parameter '<p>' expected type '<t>'"` |
| 2 | Policy | `PolicyEngine.check(action=tool_name, args=args, context=context)` | `"Policy violation: Rule <id>: '<action>' blocked by '<req>' in context '<ctx>'"` |
| 3 | Plausibility | `store.score_triple(context, plausibility_relation, tool_name)` versus `theta_plausibility` | `"Implausible tool call: distance(<ctx>, should_call, <tool>) = <d> > theta=<θ>"` |

The schema gate is intentionally lightweight. It only checks
`required` and the JSON `type` of each provided value (string,
number, integer, boolean, array, object); it does not enforce
`enum` values, regex patterns, or nested object schemas. The trade
is determinism: the gate either passes or returns a single concrete
reason, no JSON-schema engine in the way. If you need stricter
validation, attach it as a custom policy rule (see §17.4) so that
the same gate-by-gate audit record covers the new check.

The policy gate delegates entirely to `PolicyEngine.check`, which
runs four internal layers (explicit rule match, GMS graph lookup,
geometric plausibility against the policy graph, contradiction via
tension energy). The full surface and the calibrated triple
(Layer 1 + Layer 2 + Layer 3) live in §19. From the gateway's point
of view there is one input — `PolicyDecision.allowed` — and the
gateway hard-blocks if it is `False`.

The plausibility gate scores
$d = d_{\text{geo}}(R_{r}\,v_{\text{ctx}},\,v_{\text{tool}})$
where $r$ is `plausibility_relation`. If the triple is not in the
store the score is `None` and the gate fails open (`plausible =
True`). This is deliberate so that early-stage stores with no
explicit `should_call` graph do not block every call; production
stores tighten the gate by ingesting an explicit policy graph and
calibrating $\theta$ on a held-out cohort (§25).

### 17.3 Reading a `GatewayResult`

A `GatewayResult` carries eight fields. The first three are the
business outcome:

```python
result.blocked            # bool — True if any gate fired
result.reason             # str  — human-readable explanation
result.display            # str  — tool's display string (empty if blocked)
```

The next three are the audit record:

```python
result.gate_results       # dict — {"schema": True, "policy": True, "plausibility": True, "fault_injection": True}
result.policy_decision    # PolicyDecision — full Layer-1..4 verdict from gate 2 (None if not reached)
result.plausibility_score # float — geodesic from gate 3 (None if not reached)
```

The last two cover fault injection and observability:

```python
result.fault_injected     # bool  — True if a FaultProfile fired after gate 3
result.execution_time_ms  # float — wall-clock time for the full execute() call
```

A reviewer reading `result.reason` gets the one-line summary, and a
reviewer reading `result.gate_results` together with
`result.policy_decision` gets the full per-gate trail. The
gateway's `transcript_summary()` aggregates these into per-gate block
counts, which the `release_gates_spc` runner in §24 consumes
directly.

### 17.4 Composing a custom gateway

Custom enforcement is layered on the same gateway by adding rules
to its `PolicyEngine` (gate 2), tightening `theta_plausibility`
(gate 3), or registering a `FaultProfile` for a tool name pattern.
Each path leaves the audit record uniform, so no downstream consumer
needs to special-case the new check.

```python
from knowlytix.harness.testing import PolicyRule, FaultProfile

# Gate 2: forbid a destructive tool by name.
gateway.add_policy_rule(PolicyRule(
    rule_id="rule-no-delete",
    action="delete_dataset",        # fnmatch-matched against tool_name
    rule_type="blocks",             # one of {"blocks", "requires", "allows"}
    requirement="any_context",
    context="*",
    severity="error",               # hard block; "warning" logs only
))

# Gate 3: tighten the plausibility threshold.
gateway.theta_plausibility = 1.2

# Inject latency into a specific tool family.
gateway.add_fault_profile(FaultProfile(
    tool_pattern="train_*",
    latency_ms=50,
    error_rate=0.0,
))
```

Adding a `requires` rule that the call cannot satisfy (e.g.
`requirement="human_approval"` with no approval triple in the
session context) hard-blocks at gate 2. Adding an `allows` rule
short-circuits the policy gate to PASS as long as no other rule
hard-blocks. Pattern syntax is `fnmatch`; `delete_*` matches
`delete_dataset`, `delete_model`, etc. See §19 for the calibrated
multi-layer pattern (Layer 1 declarative rules + Layer 2 GMS graph
+ Layer 3 contextual plausibility on the policy graph).

### 17.5 Cross-references

- §2.2 Three gates inside the GovernedToolGateway — architectural
  view.
- §3.2-3.3 Setup + minimal call.
- §9 Tool Catalog — the schemas gate 1 validates against.
- §19 Policy Engine + Calibrated Policy Triple — the full surface
  behind gate 2.
- §25 Threshold Calibration — how to derive a defensible
  `theta_plausibility` on a held-out cohort.
- `knowlytix/harness/governance/governed_gateway.py:GovernedToolGateway.execute`
  — implementation.
- `gms_agent_builder_ext/chapters/ch07.tex` — chapter form of
  the calibrated gateway with the same triple-gate semantics.
- Notebook `16_tool_gateway.ipynb` exercises each gate
  independently and renders a `(call × gate)` decision grid.

---

## 18. Behavioral contracts

A behavioral contract is a finite-state machine whose states
correspond to workflow milestones (`data_loaded`, `preprocessed`,
`model_trained`, `diagnosed`, `fair_checked`, `compared`,
`deployment_ready`, `human_review`) and whose transitions are the
**only** legal moves. Tool permissions are scoped to each state, so
a tool runs only when the FSM is in a state that explicitly allows
it. This is structural enforcement: a missing transition cannot be
taken, period. The contract pairs with §17's gate-2 policy check
to give the agent two complementary forms of process control —
declarative rules at gate 2, structural FSM enforcement after
execution.

### 18.1 The three risk-tier contracts

Three factories ship in `knowlytix/harness/governance/behaviors/contracts.py`.
Each is calibrated to a deployment risk tier. All three return a
fully populated `BehavioralContract` with explicit `transitions`,
`tool_permissions`, and `invariants`.

| Factory | Budget | Preprocess required | Fairness required | Robustness required | Human approval |
|---------|--------|---------------------|-------------------|---------------------|----------------|
| `advisory_contract()` | 50 | no | no | no | no |
| `recommendation_contract()` | 30 | yes (invariant) | no | no | no |
| `action_taking_contract()` | 20 | yes (invariant) | yes (invariant) | yes (invariant) | yes |

The advisory contract permits broad tool transitions and only
gates the deployment edge. The recommendation contract requires
preprocessing before training and pins diagnostic tools to the
post-training portion of the FSM. The action-taking contract is
strict: a single linear path
`idle → data_loaded → preprocessed → model_trained → diagnosed →
fair_checked → compared → human_review → deployment_ready`, with
mandatory `diagnose_robustness`, `diagnose_fairness`, and
human-review states. Budgets and invariants are taken verbatim from
`knowlytix/harness/governance/behaviors/contracts.py:advisory_contract` /
`recommendation_contract` / `action_taking_contract`.

### 18.2 Validating a tool sequence

`ProcessEvaluator` walks a sequence tool-by-tool, enforcing
permissions, transitions, the budget, and all invariants. The
result is a `SequenceValidation` with the verdict, the violations
list, the visited states, and the count of blocked tools.

```python
from knowlytix.harness.governance.behaviors.contracts import (
    ProcessEvaluator,
    advisory_contract,
    recommendation_contract,
    action_taking_contract,
)

contract  = action_taking_contract()
evaluator = ProcessEvaluator(contract)

valid_sequence = [
    "load_dataset", "preprocess_data", "train_model",
    "diagnose_robustness", "diagnose_fairness", "leaderboard",
]
v = evaluator.validate_sequence(valid_sequence)
print(v.valid, v.tools_blocked)
print([s.value for s in v.states_visited])
```

A sequence that omits `preprocess_data` triggers the
`_invariant_no_train_without_preprocess` invariant; a sequence
that omits `diagnose_fairness` and still reaches `deployment_ready`
triggers `_invariant_no_deploy_without_fairness`; a sequence that
exceeds the budget triggers a `budget exceeded` violation. The
budget number is `contract.max_tool_calls`, which is 50 for
advisory, 30 for recommendation, and 20 for action-taking — these
match the source-code defaults in
`knowlytix/harness/governance/behaviors/contracts.py`.

### 18.3 The pure-FSM and GMS holonomy paths

`ProcessEvaluator` exposes two validation paths.
`validate_sequence` is pure-FSM and has no GMS dependency: it
walks the transition table and the permission table directly. This
is the path used by the audit-defensible release gate; it produces
deterministic, replay-stable verdicts. `check_holonomy(actual,
expected, store=...)` is the geometric path: when the
`precedes` relation is trained in the store it composes rotors
along both sequences and returns the holonomy defect; otherwise
it falls back to a normalized Levenshtein distance scaled to
$[0, 2]$. The holonomy path is for diagnostic comparison, not for
gate enforcement — invariants and transitions remain the
load-bearing checks.

### 18.4 Cross-references

- §17 Governed Tool Gateway — the gate-2 policy check that
  complements behavioral contracts at the call site.
- §19 Policy Engine + Calibrated Policy Triple — the calibrated
  Layer 1+2+3 surface that contracts cross-reference for
  declarative rules.
- §13 of "How to Build Policy-Governed AI Agents"
  (`gms_agent_builder_ext/chapters/ch13.tex`) — chapter-form
  treatment of behavioral contracts with the same FSM and
  invariant set.
- `knowlytix/harness/governance/behaviors/contracts.py:BehavioralContract` —
  source of truth for the FSM + tool permissions + invariants.
- Notebook `17_behavioral_contracts.ipynb` validates one valid
  sequence and two failure modes against each of the three
  contracts.

---

## 19. Policy engine and calibrated policy triple

The `PolicyEngine` is the runtime authority that decides whether a tool call is permitted. It is the single audit-defensible chokepoint between an agent's intent (the proposed `(action, args, context)`) and the tool gateway's effect (the actual call). Where `BehavioralContract` (§18) verifies that a *trajectory* visits the required states, the policy engine refuses an *individual* call before it executes.

Three properties make the engine audit-defensible. First, every decision is structured as a `PolicyDecision` carrying `allowed`, `blocking_rules`, `gms_scores`, and a human-readable `reason` — the auditor never has to reconstruct *why* a call was rejected. Second, the engine fires four layers in a fixed order; a rejection at any layer short-circuits the rest, so the trail of which layer caught the violation is unambiguous. Third, the calibrated thresholds at Layers 3 and 4 come from a separate offline calibration phase (§25) and are sealed into a `PolicyBundle` whose hash-based message authentication code (HMAC-SHA-256) signature is verified before the runtime binds to it. A deployment that ships a tampered or mis-calibrated bundle fails at load time, not at decision time.

### The four layers of `PolicyEngine.check`

`PolicyEngine.check(action, args, context)` runs four geometry-backed layers in sequence:

| Layer | Mechanism                                          | What it catches                                              | Source signal                              |
|-------|----------------------------------------------------|--------------------------------------------------------------|--------------------------------------------|
| 1     | Declarative rule match (`PolicyRule` + `fnmatch`)  | Hard-coded refusals (`requires`, `blocks`, `allows`)         | YAML rule set                              |
| 2     | GMS graph lookup (`store.query_triples`)           | Stored `requires` / `blocks` triples on the action entity    | `doc_graph` edges                          |
| 3     | Geometric plausibility (`store.score_triple`)      | Action implausible in this context                           | geodesic distance ≤ `PLAUSIBILITY_THRESHOLD` |
| 4     | Contradiction detection (`store.tension_energy`)   | Action contradicts a blocking rule entity                    | tension energy ≤ `TENSION_THRESHOLD`        |

The first two layers are symbolic (no learned numerics). The last two reuse the same GMS primitives that §4 introduced — same store, same calibrated thresholds, same `CalibratedThresholds` snapshot used by `GeometricReasoner`. The engine never instantiates a separate model for policy checks.

```python
from knowlytix.harness.testing.policy import PolicyEngine, PolicyRule

rules = [
    PolicyRule(
        rule_id="RML-001",
        action="submit_escalation",
        requirement="human_approval",
        rule_type="requires",
    ),
    PolicyRule(
        rule_id="RML-002",
        action="delete_*",
        requirement="production",
        rule_type="blocks",
        context="production",
    ),
]
engine = PolicyEngine(store=ds_store, rules=rules)

decision = engine.check(
    action="delete_table",
    args={"table": "transactions"},
    context="production",
)
assert decision.allowed is False
assert "RML-002" in decision.blocking_rules
print(decision.reason)
```

### Calibrated thresholds at Layers 3 and 4

Layers 3 and 4 use class-level defaults (`PLAUSIBILITY_THRESHOLD = 1.5`, `TENSION_THRESHOLD = 1.7`) only when no calibrated `judge` is passed. Production deployments override both:

```python
from knowlytix.harness.governance.plausibility_calibration import bootstrap_plausibility_threshold

calibrated = bootstrap_plausibility_threshold(
    store, labeled_pairs, n_bootstrap=200,
)
# `calibrated.threshold` carries the empirically fit cut and CI
engine = PolicyEngine(
    store=ds_store,
    rules=rules,
    judge=calibrated_judge,         # exposes ._thresholds dict
    plausibility_relation="uses_tool",
)
```

When `judge._thresholds["policy_geodesic"]` is set, it replaces the Layer-3 threshold; when `judge._thresholds["tau_contra"]` is set, it replaces the Layer-4 threshold. The engine records the source of each threshold in `_calibration_source`, which is what an audit packet quotes.

`plausibility_relation` is the agent-centric probe relation. On the `data_science_practices` corpus it is `"uses_tool"`, so Layer 3 scores `(context, "uses_tool", action)` rather than the legacy `(action, "allowed_in", context)`. The choice is per-corpus and is sealed into the `PolicyBundle` so two stores cannot accidentally share a probe relation that has different semantics.

### `PolicyBundle`: the offline → runtime contract

The runtime engine never recalibrates. Calibration is offline, signed, and shipped as a `PolicyBundle`:

```python
from knowlytix.harness.governance.policy_bundle import PolicyBundle
from knowlytix.harness.governance.policy_multisign import sign_for, attach_signatures, verify_quorum

bundle = PolicyBundle(
    schema_version="1.0",
    bundle_id="ds-policy-2026-05",
    created_at="2026-05-01T00:00:00Z",
    source_store_path="fixtures/ds_practices_store",
    rules=[r.to_dict() for r in rules],
    rules_yaml_hash="...",
    policy_geodesic=calibrated.policy_geodesic,
    tau_contra=None,                    # corpus had no opposite_of triples
    plausibility_relation="uses_tool",
).freeze(key=os.environ["GOVERNED_AGENT_SIGNING_KEY"])
```

`freeze()` resolves the signing key from the explicit argument or from `GOVERNED_AGENT_SIGNING_KEY`, computes the canonical payload (every threshold, every rule, every fingerprint, sorted), HMAC-signs it, and stamps the signature into the bundle. `PolicyBundle.load()` re-verifies before returning the bundle; a tampered file raises `BundleSignatureError`. A bundle that ships `tau_contra=None` documents Layer 4 as `signal_unavailable` rather than fail-open at the class default — Ch 19's threshold calibration honours the same rule.

### Multi-signer k-of-n attestation

A high-stakes deployment never lets a single operator sign the policy bundle. `policy_multisign.py` formalises k-of-n attestation:

```python
from knowlytix.harness.governance.policy_multisign import (
    sign_for, attach_signatures, verify_quorum, assert_quorum,
)

sigs = [
    sign_for(bundle, signer_id="risk_lead", key=keys["risk_lead"]),
    sign_for(bundle, signer_id="ml_lead",   key=keys["ml_lead"]),
    sign_for(bundle, signer_id="ops_lead",  key=keys["ops_lead"]),
]
attach_signatures(bundle, sigs)

# at deployment load time
assert_quorum(bundle, key_registry=keys, k=2)   # raises if < 2 verify
```

`verify_quorum` rejects duplicate signers, unregistered signers, and bad signatures separately, so the audit log identifies *which* signature failed without leaking the others. The HMAC-SHA-256 primitive is a workflow stand-in: a deployment that needs threshold cryptography (Shamir, BLS) replaces the primitive while keeping the same `sign_for` / `verify_quorum` API.

### FSM invariants over the four-layer state space

`policy_invariants.verify_quorum_invariants(quorum_required=2)` enumerates every reachable layer-state combination and checks five invariants:

- **I1.** ALLOW iff zero blocking signals across all four layers
- **I2.** BLOCK requires ≥ `quorum_required` blocking signals
- **I3.** REVIEW iff between 1 and `quorum_required - 1` blocking signals
- **I4.** Layers in `signal_unavailable` are excluded from the quorum count, never counted as PASS
- **I5.** `record.replay()` reproduces the verdict from the `to_audit_record` payload alone

The function returns an `InvariantReport` whose `all_pass` property is the structural guarantee an auditor inspects: every verdict the engine can possibly produce satisfies I1-I5. Ch 12's hardening pass (`project_ch12_audit_defensible`) verified 36/36 reachable states; the same call wired into your CI is the regression guard.

### Cross-references

- USER_GUIDE.md §11 (GeometricReasoner) — the same `score_triple` + `tension_energy` primitives Layer 3-4 reuse
- USER_GUIDE.md §17 (Governed Tool Gateway) — where `PolicyEngine.check` is wired in as the `before_tool` hook
- USER_GUIDE.md §25 (Threshold Calibration) — how `policy_geodesic` and `tau_contra` are fit
- `knowlytix/harness/testing/policy.py:PolicyEngine.check` (the four-layer implementation)
- `knowlytix/harness/governance/policy_bundle.py:PolicyBundle.freeze` (offline-to-runtime contract)
- `knowlytix/harness/governance/policy_multisign.py` (k-of-n attestor signing)
- `knowlytix/harness/governance/policy_invariants.py:verify_quorum_invariants` (FSM property checks)
- `gms_agent_builder_ext/chapters/ch12.tex` §12.3-12.5 for the calibrated-policy-triple derivation
- Notebook `notebooks/18_policy_engine.ipynb`

---

## 20. Severity and FailureCode taxonomy

A failure is more useful when it is *named*. The `FailureCode` taxonomy in `knowlytix.harness.testing.failures` is the controlled vocabulary every verifier reaches for: each code identifies a specific GMS verification route that failed, so a release-gate report aggregates by root cause rather than by free-text strings. The companion `SeverityClassifier` in `knowlytix.harness.testing.severity` resolves each `(FailureCode, risk_tier)` pair to one of four `SeverityLevel`s, with the explicit promotion rule that `action_taking` tier escalates the moderate / high band by one level. This is the rule §18's `action_taking_contract` enforces at trajectory level; §20 enforces it at individual-failure level.

### The 29 `FailureCode` entries

`FailureCode` is a string enum partitioned by GMS verification route:

| Group              | Codes                                                                                                                                                              |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Content / semantic | `SEMANTIC_EXTRACTION_FAILURE`, `SEMANTIC_MISMATCH`, `HALLUCINATION_DETECTED`, `INCOMPLETE_ANSWER`, `NO_CLAIMS_EXTRACTED`                                            |
| Graph / structural | `ENTITY_NOT_FOUND`, `RELATION_NOT_FOUND`, `FACT_INCORRECT`, `PATH_INCONSISTENCY`                                                                                    |
| Logic              | `CONTRADICTION_DETECTED`, `TENSION_EXCEEDED`, `HOLONOMY_DEFECT`                                                                                                     |
| Numeric            | `THRESHOLD_VIOLATION`, `BOUNDARY_FAILURE`                                                                                                                           |
| Exactness (ENM)    | `EXACT_MISMATCH`, `ARTIFACT_INTEGRITY_FAILURE`                                                                                                                      |
| Agentic / policy   | `TOOL_CALL_INVALID`, `POLICY_VIOLATION`, `POLICY_BLOCKED_PRE_EXECUTION`, `APPROVAL_REQUIRED`                                                                        |
| Process            | `REQUIRED_TOOL_MISSING`, `FORBIDDEN_TOOL_CALLED`, `PROCESS_DEVIATION`, `TOOL_BUDGET_EXCEEDED`                                                                       |
| Recovery           | `RECOVERY_FAILURE`                                                                                                                                                  |
| Human interaction  | `ESCALATION_INAPPROPRIATE`, `CLARIFICATION_INSUFFICIENT`                                                                                                            |
| System             | `VERIFICATION_ERROR`, `NOT_CALIBRATED`                                                                                                                              |

The taxonomy is closed, not extensible. A novel failure mode is added by extending the enum (a vocabulary change everyone audits together), not by passing free-text. This is what makes cross-run aggregation byte-stable.

### `FailureAttribution`: the audit-packet record

Every failing verification produces a `FailureAttribution`:

```python
from knowlytix.harness.testing.failures import FailureAttribution, FailureCode, FailureSeverity

failure = FailureAttribution(
    code=FailureCode.FACT_INCORRECT,
    claim_id="claim_017",
    verifier="GraphVerifier",
    severity=FailureSeverity.ERROR,
    detail="(target_variable, IS, age) — not in store",
    gms_method="score_triple",
    raw_score=1.83,
    suggested_action="Re-run with corrected target variable",
)
```

The fields are exactly what an audit packet quotes: which claim failed, which verifier produced the failure, which GMS method generated the signal, what the raw score was, and what to do next. The `severity_assessment` slot is filled later by the classifier.

### `SeverityLevel` and the four-level scale

```python
class SeverityLevel(str, Enum):
    CRITICAL = "critical"    # safety-relevant, unrecoverable, hard to detect
    HIGH = "high"            # functionally wrong, potentially recoverable
    MODERATE = "moderate"    # partially wrong, recoverable
    LOW = "low"              # minor, cosmetic, or informational
```

`SeverityClassifier` maps each `FailureCode` to a base level via `CODE_SEVERITY_MAP`. The defaults reflect the GMS taxonomy's structure:

- **CRITICAL.** `HALLUCINATION_DETECTED`, `POLICY_VIOLATION`, `POLICY_BLOCKED_PRE_EXECUTION`, `APPROVAL_REQUIRED`
- **HIGH.** `FACT_INCORRECT`, `CONTRADICTION_DETECTED`, `THRESHOLD_VIOLATION`, `REQUIRED_TOOL_MISSING`, `FORBIDDEN_TOOL_CALLED`, `RECOVERY_FAILURE`, `PATH_INCONSISTENCY`
- **MODERATE.** `PROCESS_DEVIATION`, `TOOL_BUDGET_EXCEEDED`, `INCOMPLETE_ANSWER`, `ENTITY_NOT_FOUND`, `SEMANTIC_MISMATCH`, `BOUNDARY_FAILURE`, `TENSION_EXCEEDED`, `HOLONOMY_DEFECT`, `ESCALATION_INAPPROPRIATE`, `CLARIFICATION_INSUFFICIENT`, `TOOL_CALL_INVALID`
- **LOW.** `EXACT_MISMATCH`, `RELATION_NOT_FOUND`, `ARTIFACT_INTEGRITY_FAILURE`, `SEMANTIC_EXTRACTION_FAILURE`, `NO_CLAIMS_EXTRACTED`, `VERIFICATION_ERROR`

`NOT_CALIBRATED` defaults to `LOW` because it is a configuration signal, not a substantive failure.

### Risk-tier escalation: `action_taking` is special

`SeverityClassifier` reads a `risk_tier` argument on every call. The default escalation table is:

```python
_TIER_ESCALATION = {
    "action_taking":   {SeverityLevel.MODERATE, SeverityLevel.HIGH},
    "recommendation":  {SeverityLevel.MODERATE},
}
```

When a failure's base level is in the tier's set, `_escalate` promotes it one step. The consequence that matters in production is:

> An `action_taking` tier `FACT_INCORRECT` is **CRITICAL**, not HIGH.

This is the rule the testing monograph (`gmsh_testing_ext/chapters/appendix_c.tex`) and the Ch 8 hardening pass (`project_ch08_evaluation`) both turn on. An advisory tier can absorb a wrong number; a recommendation tier should re-derive; an action-taking tier that acts on a wrong number causes irrecoverable damage. The promotion encodes that asymmetry in one place.

```python
from knowlytix.harness.testing.severity import SeverityClassifier, SeverityLevel
from knowlytix.harness.testing.failures import FailureCode

classifier = SeverityClassifier()

# Same failure, three tiers
for tier in ("advisory", "recommendation", "action_taking"):
    a = classifier.classify_code(FailureCode.FACT_INCORRECT, risk_tier=tier)
    print(f"{tier:20s} -> {a.level.value}")
# advisory             -> high
# recommendation       -> high
# action_taking        -> critical
```

The classifier also computes three `[0, 1]` dimension scores — `impact`, `recoverability`, `detectability` — and a weighted `composite_score`. When the composite passes `_MODULATION_THRESHOLD = 0.85` the level is escalated one further step, capped at CRITICAL. This is the geometric modulation: a `HALLUCINATION_DETECTED` close to a real entity is harder to detect, which lowers detectability, which raises composite, which can promote a HIGH to CRITICAL even outside `action_taking`.

### The matrix: `FailureCode × tier → SeverityLevel`

The cleanest way to inspect the policy is to materialise the full matrix:

```python
import numpy as np

tiers = ["advisory", "recommendation", "action_taking"]
codes = list(FailureCode)
matrix = np.array([
    [classifier.classify_code(c, risk_tier=t).level for t in tiers]
    for c in codes
])
```

A heatmap of `matrix` (notebook NB 19) makes the `action_taking` column visibly darker on the rows with base level MODERATE or HIGH. The rows that are CRITICAL across *every* tier are the four codes the engine refuses without escalation; the rows that are LOW across every tier are the cosmetic ones.

### Configurable overrides via YAML

`SeverityClassifier.from_yaml` loads an override file:

```yaml
code_severity:
  exact_mismatch: high                 # promote from low
  tension_exceeded: high               # promote from moderate

tier_escalation:
  strict_regulated: [low, moderate, high]   # new tier

composite_weights:
  impact: 0.70
  recoverability: 0.10
  detectability: 0.20
  modulation_threshold: 0.85
```

Overrides are validated against the closed `FailureCode` and `SeverityLevel` vocabularies — an unknown code or level raises `ValueError` at load time, never silently. A deployment that calibrates its own table ships the YAML alongside the policy bundle so the auditor can re-derive the matrix exactly.

### Cross-references

- USER_GUIDE.md §18 (Behavioral Contracts) — where the same tier names parameterise trajectory-level checks
- USER_GUIDE.md §19 (Policy Engine) — `POLICY_VIOLATION` and `POLICY_BLOCKED_PRE_EXECUTION` originate there
- USER_GUIDE.md §24 (Release Gates & SPC) — release gate consumes the per-tier severity counts
- `knowlytix/harness/testing/failures.py:FailureCode` (29 entries) and `:FailureAttribution`
- `knowlytix/harness/testing/severity.py:SeverityClassifier`, `:CODE_SEVERITY_MAP`, `:_TIER_ESCALATION`
- `gmsh_testing_ext/chapters/appendix_c.tex` for the audit-defensible derivation
- Notebook `notebooks/19_severity_failure_codes.ipynb`

---

## 21. Orchestrator

The `GovernedOrchestrator` runs an ordered list of tool names through the
gateway end-to-end, enforces a behavioral contract at preparation time,
extracts knowledge from each step, verifies typed claims at evaluation
time, and returns a single `OrchestratorResult` carrying the lifecycle
trace, per-step signal vectors, scenario violations, and a verification
verdict. It is the layer that ties §17 (gateway), §10 (memory), §12 (claim
verification), and §18 (behavioral contracts) into one auditable run. The
orchestrator does not plan; planning is the job of `WorkflowPlanner` (§14).

### Lifecycle finite-state machine

A run flows through `CREATED` → `PREPARING` → `EXECUTING` → `EVALUATING`
→ `COMPLETED` (or `FAILED`). The lifecycle aborts to `ABORTED` when the
plan violates the scenario, when the budget is exhausted mid-execution,
or when an underlying tool returns an error rather than usable output.
Every transition is timestamped on the `RunManager`; an invalid
transition raises `InvalidTransitionError` rather than silently
corrupting state.

### Scenario field semantics

A `Scenario` is the behavioral contract. Four fields drive enforcement:

| Field                    | Type        | Behavior                                                                                                       |
|--------------------------|-------------|----------------------------------------------------------------------------------------------------------------|
| `required_tools`         | `list[str]` | Each name must appear somewhere in `plan`; missing entries are recorded as scenario violations during PREPARING. |
| `forbidden_tools`        | `list[str]` | Any name appearing in both `plan` and this list is recorded as a violation; the orchestrator filters these out before EXECUTING, so the forbidden tool never reaches the gateway. |
| `expected_tool_sequence` | `list[str]` | When non-empty, the executed tools must appear in this order as a subsequence (not necessarily contiguous). Checked in EVALUATING. |
| `max_tool_calls`         | `int`       | Hard cap on the number of tool calls. Exceeding it during EXECUTING transitions the run to `ABORTED`.           |

`forbidden_tools` enforcement happens in two places: a violation is
recorded in PREPARING, and the offending step is skipped before the
gateway is even called. The skip is what matters for safety; the
recorded violation is what matters for the audit trail.

### `OrchestratorPolicy` co-attestor

The geometric attestor inside the gateway (gates 1, 2, 3) and the
typed-claim verifier in EVALUATING share the same embedding store. To
cut the false-accept rate further, the orchestrator accepts an optional
second attestor — an `OrchestratorPolicy` rule pack that runs against
the `(plan, scenario)` tuple in PREPARING, *before* any tool fires. The
rule pack is pure Python: a list of declarative `OrchestratorRule`
predicates, each marked `blocking` or `advisory`. A blocking failure
transitions PREPARING → ABORTED with the reasons recorded in the trace.

Why a second attestor: the geometric gate and the structural rule pack
have uncorrelated blind spots. Unknown tool names, circular dependencies
(`train_model` before `load_dataset`), duplicate steps, and budget
underruns are caught structurally without ever consulting the manifold.
The default rule pack (`default_orchestrator_rules`) ships eight rules
covering these cases; production deployments extend it with
domain-specific predicates.

```python
from knowlytix.harness.governance.orchestrator_policy import (
    OrchestratorPolicy, default_orchestrator_rules,
)

policy = OrchestratorPolicy(default_orchestrator_rules())
orchestrator = GovernedOrchestrator(
    gateway=gateway, memory=memory, verifier=verifier,
    session=session, store=store, device=DEVICE,
    plan_policy=policy,   # the co-attestor
)
```

### Two-attestor verifier pattern

In EVALUATING the run carries two independent verifiers when bound to
a `GovernanceBundle`. The default `ClaimVerifier` scores claims via
geodesic plausibility on the memory store; the bundle's
`numeric_range_index` adds a second attestor that cross-checks every
NUMERIC claim against the calibrated plausible-range index. A claim is
PASSED only when both attestors agree. `from_bundle` rebuilds the
verifier so the second attestor binding is explicit at the audit-trail
level.

### Tool-error abort path

A tool can succeed at all three gates but still return an error in its
`raw["error"]` payload (for example, `load_dataset` called with an
unknown built-in). The gateway sees the call as completed; without an
explicit guard, downstream extraction would silently produce zero
claims, and the default PASS-on-empty-claims path would sign off on a
silent failure. The orchestrator instead aborts: when
`gw_result.raw.get("error")` is truthy, the run records a
`tool_error: …` violation, transitions to ABORTED, and returns the
partial trace. This converts a silent acceptance into a loud refusal.

### `format_result` output

`OrchestratorResult` round-trips to a human-readable summary via
`GovernedOrchestrator.format_result`. The summary shows the final state,
total wall-clock time, the lifecycle transition log, scenario
violations, the verification verdict (status, pass count, pass rate),
the unavailable-signal rate (steps where neither the policy store nor
the gateway returned a calibrated plausibility score), and a
per-step table with tool, status, extraction tier, fact count, and the
five-component signal vector. The same result also has `to_audit_log()`,
which emits a JSON-serializable record carrying every signal an auditor
needs to reconstruct the verdict — including the reason chain (incident
corpus triples + extracted triples per step) and, when present, the
bound `GovernanceBundle` id and signature.

### Cross-references

- §3.4 Quick-start orchestrator example
- §10 `AgentMemory` (the storage the orchestrator writes findings to)
- §12 Claim verification (the EVALUATING-phase attestor)
- §18 Behavioral contracts (`Scenario` is one form of contract)
- §26 Runtime governance (`GovernedOrchestrator.from_bundle`)
- `knowlytix/harness/governance/orchestrator.py:GovernedOrchestrator.run`
- `knowlytix/harness/governance/orchestrator_policy.py:OrchestratorPolicy`
- `gms_agent_builder_ext/chapters/ch10.tex` (full derivation)

---

## 22. Multi-agent

A `MultiAgentOrchestrator` coordinates several role-bound `GovernedAgent`s
that share a single `SessionState` but each own a private domain
`AgentMemory`. The model is structural, not advisory: an agent calling a
tool outside its allow-list raises `PermissionError` before the tool runs.
Adversarial probes (`role_hijack`, `frozen_role_mutation`, alias attacks,
cross-store writes, handoff forgery) target this surface; the library is
designed so each probe class is defeated by an immutability invariant
rather than by a runtime check the attacker can patch out.

### Identity-immutable `GovernedAgent`

`AgentRole` is a `frozen` dataclass with `frozenset` allowed and
forbidden tool sets. Three properties hold by construction:

- The dataclass is frozen, so `agent.role.name = "TrainingAgent"` raises
  `FrozenInstanceError`.
- `allowed_tools` is a `frozenset`, so `agent.role.allowed_tools.add(...)`
  raises `AttributeError`.
- `__post_init__` rejects roles whose allowed and forbidden sets overlap,
  surfacing configuration errors at construction.

`GovernedAgent` itself is identity-immutable. The attributes `role`,
`session`, and `memory` are set exactly once in `__init__` via
`object.__setattr__`; subsequent assignment is intercepted by a custom
`__setattr__` that raises `AttributeError` for any of those three
identity attributes. This defeats the `role_hijack` probe class —
`agent.role = TRAINING_AGENT_ROLE` cannot succeed at runtime.

```python
from knowlytix.harness.governance.agents.multi_agent import (
    AgentRole, GovernedAgent,
    DATA_AGENT_ROLE, TRAINING_AGENT_ROLE,
    DIAGNOSTICS_AGENT_ROLE, FAIRNESS_AGENT_ROLE,
)

# Custom roles are constructed the same way:
custom = AgentRole(
    name="Reporter",
    allowed_tools=frozenset({"leaderboard"}),
    forbidden_tools=frozenset({"train_model", "diagnose_fairness"}),
    description="Produces final reports only",
)
```

### Whitelist semantics

`GovernedAgent.can_execute` is a whitelist: a tool is permitted only if
its name is explicitly listed in `allowed_tools`. A name in neither set
is *forbidden* by default. This defeats the `tool_alias_attack` probe
class — registering a new tool name (`train_model_v2`) does not bypass
the role check, because that name is not in the allow-list.
`forbidden_tools` is retained for documentation and audit clarity: it
records *why* a tool is denied, and lets the audit log distinguish an
explicit deny from a default-deny.

### Domain-store isolation

Each `GovernedAgent` owns a private `AgentMemory`. The orchestrator's
`test_store_isolation` static method confirms that two agents have
distinct store objects (different `id()` for `agent.memory.store`). An
adversary that holds a reference to another agent's `memory.store` can
write directly to it — that is a Python-level invariant the library
cannot enforce — but the normal `execute()` path only writes to the
calling agent's own memory, and the audit fingerprint catches
out-of-band writes.

### Adversarial probes

Two adversarial classes exercise the immutability invariants directly.
The cohort `knowlytix/harness/governance/agents/multi_agent_adversarial.py` ships ten
classes; the two below are the smallest examples that exercise the
identity-immutable surface:

**`role_hijack`** — try to mutate `agent.role.name`, then try
`agent.role.allowed_tools.add("train_model")`. Both must raise; the
first is blocked by the frozen dataclass, the second by `frozenset`.
The audit log records `runtime_error` as the safe outcome.

**`frozen_role_mutation`** — three sub-kinds:
1. `add_to_set`: `agent.role.allowed_tools.add("train_model")` —
   blocked by `frozenset`.
2. `reassign_set`: `agent.role.allowed_tools = frozenset({"train_model"})`
   — blocked by the frozen dataclass.
3. `reassign_role`: `agent.role = TRAINING_AGENT_ROLE` — blocked by
   `GovernedAgent.__setattr__` (identity-immutable guard).

The full sweep `sweep_multi_agent_adversarial(orch, cohort)` executes
every probe and produces a `MultiAgentAdversarialReport` carrying a
single `safety_rate` plus per-class breakdowns. A safety rate below the
operating-point floor refuses to bind the orchestrator to a bundle.

### Pipeline execution

```python
from knowlytix.harness.governance.session import SessionState
from knowlytix.harness.governance.agents.multi_agent import (
    GovernedAgent, MultiAgentOrchestrator,
    DATA_AGENT_ROLE, TRAINING_AGENT_ROLE,
    DIAGNOSTICS_AGENT_ROLE, FAIRNESS_AGENT_ROLE,
)

shared = SessionState()
agents = {
    "data":        GovernedAgent(DATA_AGENT_ROLE,        shared, mem_data),
    "training":    GovernedAgent(TRAINING_AGENT_ROLE,    shared, mem_train),
    "diagnostics": GovernedAgent(DIAGNOSTICS_AGENT_ROLE, shared, mem_diag),
    "fairness":    GovernedAgent(FAIRNESS_AGENT_ROLE,    shared, mem_fair),
}
orch = MultiAgentOrchestrator(agents=agents, shared_session=shared)

phases = [
    ("data",        [("load_dataset",       {"name": "adult"}),
                     ("preprocess_data",    {"dataset": "adult"})]),
    ("training",    [("train_model",        {"dataset": "adult",
                                             "model_type": "RandomForestClassifier",
                                             "model_name": "rf"})]),
    ("diagnostics", [("diagnose_robustness", {"dataset": "adult", "model_name": "rf"})]),
    ("fairness",    [("diagnose_fairness",   {"dataset": "adult", "model_name": "rf",
                                              "protected_attribute": "sex"})]),
]
result = orch.run_pipeline(phases)
print(orch.format_handoffs())
print(orch.format_agent_stats())

# Probe a forbidden boundary directly:
blocked, msg = orch.test_boundary("data", "train_model", {"dataset": "adult"})
```

`run_pipeline` records a `HandoffRecord` between every consecutive
phase whose agent changes; per-agent statistics include
`tools_called`, `triples_learned`, and `blocked_attempts` (incremented
on every `PermissionError`).

### Cross-references

- §10 `AgentMemory` (each agent owns one)
- §17 Governed tool gateway (the gateway underneath every `execute`)
- §21 Orchestrator (single-agent variant)
- `knowlytix/harness/governance/agents/multi_agent.py:GovernedAgent`
- `knowlytix/harness/governance/agents/multi_agent_adversarial.py:sweep_multi_agent_adversarial`
- `gms_agent_builder_ext/chapters/ch17.tex` (identity-immutability hardening + 10 adversarial classes)

---

## 23. Telemetry and tracing

`GovernanceTracer` in `knowlytix/harness/governance/tracing.py` records every governed event (tool call, claim verification, state transition) into an in-memory audit trail and, optionally, into a `DOETracingGMS` store for geometric persistence. The trail is the substrate every other audit surface lives on: replay regression, drift monitoring, compliance evidence, and the hash-chain integrity check all consume the same `AuditPacket` list.

### Run lifecycle and basic recording

```python
from knowlytix.harness.governance.tracing import GovernanceTracer, AuditPacket

tracer = GovernanceTracer()
tracer.start_run("run_001", "Iris classification pipeline")

tracer.trace_tool_call(
    run_id="run_001", step_num=0,
    tool_name="load_dataset",
    args={"name": "iris"},
    result_summary="150 samples, 4 features",
    policy_decision="allowed",
    gms_scores={"plausibility": 0.41},
    gate_results={"schema": "PASS", "policy": "PASS", "plausibility": "PASS"},
    state_before="INITIAL", state_after="LOADED",
)

tracer.trace_verification(
    run_id="run_001", step_num=0,
    claims=[{"text": "iris has 150 samples"}],
    verdicts=[{"passed": True, "score": 0.95}],
)

tracer.trace_state_transition("run_001", "INITIAL", "LOADED")

tracer.end_run("run_001")
```

Three query methods read the trail back: `get_audit_trail(run_id)` returns the full chronological list of `AuditPacket`, `query_blocked_calls` and `query_allowed_calls` filter by `policy_decision`, and `get_verifications` / `get_transitions` return the verification and FSM-transition records for the run.

### `AuditPacket` schema

Each `AuditPacket` (defined in `knowlytix/harness/governance/tracing.py`) carries:

| Field              | Type            | Meaning                                                          |
|--------------------|-----------------|------------------------------------------------------------------|
| `step_num`         | int             | Sequential step number within the run                            |
| `tool_name`        | str             | Name of the tool that was called                                 |
| `policy_decision`  | str             | `"allowed"` or `"blocked"`                                       |
| `blocking_rules`   | list[str]       | Names of rules that blocked the call (empty if allowed)          |
| `gms_scores`       | dict[str,float] | GMS scores relevant to the decision (e.g., `plausibility`)       |
| `verification`     | dict or None    | Verification results stamped by `trace_verification`             |
| `state_before`     | str             | FSM state before the call                                        |
| `state_after`      | str             | FSM state after the call                                         |
| `timestamp`        | float           | Unix timestamp                                                   |
| `execution_time_ms`| float           | Wall-clock execution time in milliseconds                        |
| `gate_results`     | dict            | Per-gate pass/fail (`schema`, `policy`, `plausibility`)          |
| `args`             | dict            | Arguments passed to the tool                                     |
| `result_summary`   | str             | Brief summary of the tool result                                 |
| `prev_hash`        | str             | Hash of the previous sealed packet, `""` for the head            |
| `content_hash`     | str             | SHA-256 of this packet's unsealed payload (set by `seal_audit_trail`) |

`prev_hash` and `content_hash` are populated by `seal_audit_trail`. An unsealed packet keeps both fields as `""` so the trail is JSON-serializable in flight; `verify_audit_chain` reports an unsealed packet as a `missing_hash` break, not a silent pass.

### Hash-chain integrity

`knowlytix/harness/governance/tracing_hardening/hash_chain.py` ships the seal/verify routine. The chain is sealed once after the run completes (or after each checkpoint):

```python
from knowlytix.harness.governance.tracing_hardening import (
    seal_audit_trail, verify_audit_chain, compute_packet_content_hash,
    chain_head_hash,
)

packets = tracer.get_audit_trail("run_001")
seal_audit_trail(packets)               # mutates in place

# Independently recompute the hash of every packet and compare:
breaks = verify_audit_chain(packets)
assert breaks == []                     # intact chain

# The head hash is a single fixed-length summary of the entire trail:
head = chain_head_hash(packets)
```

`compute_packet_content_hash(packet)` is SHA-256 over the canonical JSON of the packet's unsealed fields (everything except `content_hash` and `prev_hash`, with `sort_keys=True`). `seal_audit_trail` walks the list in order, sets each packet's `prev_hash` to the previous packet's `content_hash` (or `""` for the head), then computes and stamps the new `content_hash`.

Once sealed, any post-hoc edit to a recorded field changes the canonical JSON, which changes that packet's recomputed `content_hash`. `verify_audit_chain` reports the offending position as a `ChainBreak(kind="content")`. An insertion, deletion, or reorder breaks `prev_hash` continuity at the next packet and surfaces as `kind="prev"`.

### Tail-truncation finding

`gms_agent_builder_ext/chapters/ch14.tex` §Lab 14.4 documents a structural gap: **the chain alone cannot detect truncation at the tail**. Removing the last packet leaves every surviving packet's `prev_hash` consistent with its predecessor's `content_hash`; there is no successor to break. `verify_audit_chain` returns an empty break list on the truncated trail.

The fix is a second, independent attestor that knows what the runtime decided. The library ships `knowlytix.harness.governance.tracing_hardening.cross_attestor.cross_attest_policy`; the same defence-in-depth pattern works locally with a `last_seen` register the runtime maintains alongside the tracer:

```python
class LastSeenRegister:
    def __init__(self):
        self.last_step_num = -1
        self.last_run_id = None

    def record(self, run_id: str, step_num: int) -> None:
        self.last_run_id = run_id
        self.last_step_num = step_num


def tail_truncation_detected(packets, register) -> bool:
    """True iff the audit trail's last step disagrees with last_seen."""
    if not packets:
        return register.last_step_num >= 0
    return packets[-1].step_num != register.last_step_num


# Runtime updates the register on every traced call:
register = LastSeenRegister()
for step in pipeline_steps:
    tracer.trace_tool_call(run_id="run_001", step_num=step.idx, ...)
    register.record("run_001", step.idx)

# After sealing, both attestors must agree:
seal_audit_trail(packets)
chain_intact = (verify_audit_chain(packets) == [])
truncated = tail_truncation_detected(packets, register)
assert chain_intact and not truncated
```

The `cross_attest_policy` and `cross_attest_combined` helpers generalize this to per-step matching against a runtime decision log; the `last_seen` register above is the minimal version that catches tail truncation alone.

### Reporting surfaces

`tracer.summary_report(run_id)` produces a human-readable report: tool-call totals, allowed / blocked split, total claims verified, average verification score, lowest-confidence step, and a per-violation breakdown listing each blocked step with the rule names that fired.

`tracer.dashboard_table(run_id)` returns a list of dicts, one row per packet, with columns suitable for a tabular display: `Step`, `Tool`, `Decision`, `Blocking Rules`, `Plausibility`, `Gate: Schema`, `Gate: Policy`, `Gate: Plausibility`, `Verification` (pass rate as percent), `Time (ms)`, `State` (`state_before -> state_after`).

### Cross-references

- `knowlytix/harness/governance/tracing.py:GovernanceTracer` — recording API
- `knowlytix/harness/governance/tracing_hardening/hash_chain.py:seal_audit_trail`, `verify_audit_chain`, `compute_packet_content_hash`, `ChainBreak`
- `knowlytix/harness/governance/tracing_hardening/cross_attestor.py:cross_attest_policy`, `cross_attest_combined` — independent attestor over runtime decision log
- `gms_agent_builder_ext/chapters/ch14.tex` §Lab 14.3-14.4 for the full hardening sweep, including the v2 60-probe adversarial cohort and the FSM-invariant enumeration where the tail-truncation gap is formalized
- Notebook `notebooks/22_telemetry_tracing.ipynb`

---

## 24. Release gates and SPC

`knowlytix/harness/governance/release_gates.py` ships a three-tier release gate plus a statistical-process-control (SPC) trend tracker. `knowlytix/harness/governance/release_gate_co_attestor.py` adds an SPC-slope co-attestor that returns a three-state verdict on `(metrics, tier)` tuples. The two surfaces address different failure modes: the tiered gate is a binary structural check on a single run, while the SPC co-attestor flags trend-degradation across runs even when the latest run still passes the gate.

### `RunMetrics` schema

Every gate decision and trend computation consumes a `RunMetrics` (defined in `release_gates.py`):

| Field             | Domain | Direction | Meaning                                                            |
|-------------------|--------|-----------|--------------------------------------------------------------------|
| `accuracy`        | [0,1]  | min       | Classification or regression accuracy                              |
| `confidence`      | [0,1]  | min       | Mean geometric confidence from claim verification                  |
| `robustness_drop` | [0,1]  | max       | Accuracy degradation under noise (0 = no drop, 1 = total failure)  |
| `fairness_air`    | [0,1]  | min       | Adverse-impact ratio, normalized to `min(rate_a/rate_b, rate_b/rate_a)` so a flipped-direction ratio cannot present as super-parity |
| `tool_calls`      | int    | —         | Number of tool calls executed                                      |
| `policy_blocks`   | int    | —         | Number of tool calls blocked by the policy engine                  |
| `run_id`          | str    | —         | Optional identifier for tracing and replay                         |

`RunMetrics.from_orchestrator_result(result, accuracy=..., robustness_drop=..., fairness_air=..., run_id=...)` is the production adapter. It computes `confidence` from the orchestrator's `claim_verifications` (mean geodesic-distance plausibility, normalized via $1/(1+d)$) and reads `tool_calls` and `policy_blocks` off the result. The external metrics — `accuracy`, `robustness_drop`, `fairness_air` — are the caller's responsibility because the orchestrator does not know its own ground-truth accuracy.

`metrics.is_finite()` and `metrics.is_well_formed()` validate the input before it reaches the gate. The gate itself fails closed on NaN/Inf (a non-finite value never silently passes regardless of comparison direction), but an explicit pre-check produces a clearer audit reason than a downstream finite-value block.

### `TieredReleaseGate`

Three tiers ship as class constants `TIERS`:

| Tier             | min_accuracy | min_confidence | max_robustness_drop | min_fairness_air |
|------------------|-------------:|---------------:|--------------------:|-----------------:|
| `advisory`       | 0.70         | 0.30           | 1.00 (unconstrained)| 0.00 (unconstrained) |
| `recommendation` | 0.80         | 0.50           | 0.10                | 0.00             |
| `action_taking`  | 0.85         | 0.70           | 0.05                | 0.80             |

```python
from knowlytix.harness.governance.release_gates import (
    TieredReleaseGate, RunMetrics, GateThresholds,
)

gate = TieredReleaseGate()
metrics = RunMetrics(
    accuracy=0.92, confidence=0.75,
    robustness_drop=0.05, fairness_air=0.95,
    tool_calls=15, policy_blocks=0,
    run_id="run_001",
)

# Per-tier:
result = gate.check(metrics, "action_taking")
print(result.passed, result.blocking_reasons, result.threshold_source)

# All three at once:
results = gate.check_all_tiers(metrics)
for tier, r in results.items():
    print(f"  {tier}: {'PASS' if r.passed else 'FAIL'}")
```

`GateResult` carries `tier`, `passed`, per-check `checks` (with `value`, `threshold`, `passed`, `finite`), `blocking_reasons` (human-readable), `is_calibrated`, and `threshold_source`. `is_calibrated=True` indicates the thresholds came from a fitted `ReleaseGateBundle`; `False` indicates the policy-declared class defaults.

### Monotonicity invariant

The gate constructor refuses to accept a tier set that violates monotonicity: stricter tiers must require at least as much on every directional metric. If a custom configuration sets, e.g., `recommendation.min_accuracy < advisory.min_accuracy`, `TieredReleaseGate.__init__` raises `ValueError` at construction time rather than letting the bug surface only when a divergent input arrives.

`gate.verify_monotonicity(metrics)` tests the runtime corollary: for any `RunMetrics`, if `action_taking` passes then `recommendation` must pass and `advisory` must pass. The check is a property test on outputs, distinct from the threshold-set check on inputs. `gms_agent_builder_ext/chapters/ch15.tex` §Lab 15.4 reports 14641/14641 monotonicity passes across an exhaustive grid sweep.

For high-stakes deployment, `TieredReleaseGate.from_bundle(bundle)` binds the gate to thresholds carried in a signed, fingerprinted `ReleaseGateBundle`. The bundle records bootstrap-fit thresholds calibrated on a labeled run cohort; the gate's `threshold_source` becomes `"bundle:<fingerprint>"` and surfaces in every `GateResult`.

### `RunTrend` for SPC

`RunTrend` accumulates `RunMetrics` chronologically and computes ordinary-least-squares (OLS) slopes per metric:

```python
from knowlytix.harness.governance.release_gates import RunTrend

trend = RunTrend()
for m in run_history:
    trend.metrics.append(m)

acc_slope = trend.slope("accuracy")     # OLS slope across run index
acc_mean  = trend.mean("accuracy")
fired = trend.alert("accuracy", threshold=-0.005)  # True if slope < -0.005
n_runs = trend.n_runs()
n_finite = trend.n_finite("accuracy")  # excludes NaN/Inf entries
```

NaN/Inf entries are dropped from the OLS fit (they would propagate into the slope estimate); `n_finite(field)` records how many runs were excluded so the audit log surfaces the exclusion count.

### `SPCSlopeCoAttestor`

A point-threshold gate cannot tell that a run is the third in a sequence whose accuracy is collapsing 1% per run. Conflating slope-based blocks into the gate either flickers on noisy series or hides the deviation signal in the audit trail. Following the Ch 13 `HolonomyCoAttestor` pattern, `SPCSlopeCoAttestor` returns a **three-state verdict** — `pass` / `flag` / `block` — distinct from the Ch 11 OR-semantics because the structural attestor is already binary.

```python
from knowlytix.harness.governance.release_gate_co_attestor import (
    SPCSlopeCoAttestor, SPCSlopeConfig,
)

attestor = SPCSlopeCoAttestor(
    gate=gate,
    config=SPCSlopeConfig(
        min_acc_slope=-0.005,
        min_conf_slope=-0.005,
        min_fair_slope=-0.005,
        max_rob_slope=+0.005,
        min_runs_for_alert=5,
    ),
)

verdict = attestor.evaluate(metrics, "action_taking", trend=prior_trend)
# verdict.verdict in {"pass", "flag", "block"}
# verdict.structural_passed, verdict.slope_alerted
# verdict.blocking_reasons, verdict.flag_reasons
# verdict.spc_slopes  # observed slope per metric
```

Decision lattice:

- `structural_passed = False` → `block`
- `structural_passed = True`, `slope_alerted = False` → `pass`
- `structural_passed = True`, `slope_alerted = True` → `flag`

For min-direction metrics (`accuracy`, `confidence`, `fairness_air`) the threshold is negative and the alert fires when slope drops below it; for `robustness_drop` (max-direction) the threshold is positive and the alert fires when slope rises above it.

### Calibrating slope thresholds against a healthy baseline

`attestor.calibrate_from_canonical(canonical_runs, target_quantile=0.05)` fits per-metric slope thresholds from a sequence of healthy-baseline runs:

```python
healthy_runs = [...]   # >= 5 RunMetrics from a known-good operating period
fitted_config = attestor.calibrate_from_canonical(
    healthy_runs, target_quantile=0.05,
)
# Inspect, then swap in if the fit is sound:
attestor.config = fitted_config
```

The method computes rolling-window slopes (window = 5) over the canonical trace and sets each per-metric threshold to the 5th percentile of that distribution (or 95th for `robustness_drop`). Only slopes worse than 95% of the healthy-baseline trip the flag, so natural variation in the trace does not over-alert. The fitted config is returned but not applied; the caller decides whether to swap it in.

### Cross-references

- `knowlytix/harness/governance/release_gates.py:TieredReleaseGate`, `RunMetrics`, `RunTrend`, `GateThresholds`, `GateResult`
- `knowlytix/harness/governance/release_gate_co_attestor.py:SPCSlopeCoAttestor`, `SPCSlopeConfig`, `CoAttestorVerdict`
- `knowlytix/harness/governance/release_gate_bundle.py:ReleaseGateBundle` for the calibrated, signed-bundle deployment path
- `gms_agent_builder_ext/chapters/ch15.tex` for the chapter-form exposition, the 200-probe adversarial cohort, and the 14641-point monotonicity sweep
- Notebook `notebooks/23_release_gates_spc.ipynb`

---

## 25. Threshold calibration

Thresholds are the boundary between geometric signal and a governed verdict. The geodesic plausibility cut $\theta_\mathrm{sim}$, the verifier confidence margin $\theta_\mathrm{margin}$, and the holonomy escalation cut $\theta_\mathrm{esc}$ each compress a continuous score into a binary pass / fail outcome, and each shifts the false-accept and false-reject rates of the entire agent. A "good" threshold for one corpus or claim cohort is rarely the right threshold for another. Calibration is the audit-defensible answer: pick thresholds against a labeled cohort, record the accuracy you achieved, and persist a deterministic snapshot the auditor can re-derive from frozen inputs.

### Calibration as a sweep

`ThresholdCalibrator` (in `knowlytix/harness/governance/calibration.py`) treats calibration as a sweep over `ThresholdConfig` vectors. Each configuration drives a fresh `ClaimVerifier`; the calibrator scores every labeled claim under that configuration and records `accuracy`, `false_accepts`, `false_rejects`, `tool_calls`. The two entry points are `sweep`, which varies one parameter at a time, and `grid_search`, which runs full factorial:

```python
from knowlytix.harness.governance.calibration import (
    ThresholdCalibrator, LabeledClaim, ThresholdConfig,
)

calibrator = ThresholdCalibrator(store, session=None, device="cuda")

sweep_results = calibrator.sweep(
    param_name="theta_sim",
    values=[1.0, 1.2, 1.5, 1.8, 2.0],
    test_claims=labeled,
)

grid_results = calibrator.grid_search(
    param_grid={
        "theta_sim":    [1.0, 1.2, 1.5, 1.8],
        "theta_margin": [0.10, 0.30, 0.50],
    },
    test_claims=labeled,
)

best = ThresholdCalibrator.best_config(grid_results)
print(f"opt theta_sim    = {best.config.theta_sim}")
print(f"opt theta_margin = {best.config.theta_margin}")
print(f"accuracy         = {best.accuracy:.3f}")
print(f"FA / FR          = {best.false_accepts} / {best.false_rejects}")

print(ThresholdCalibrator.sensitivity_analysis(grid_results))
```

`best_config` minimises `(-accuracy, false_accepts, false_rejects)` so ties go to the configuration with fewer dangerous accepts, then fewer rejections. `sensitivity_analysis` returns the per-parameter accuracy range and tells you which threshold is actually doing the work on this cohort.

### Bootstrap-calibrated plausibility threshold

A single fit produces one `theta_sim`, but the auditor's natural follow-up is "what would the threshold have been if you had drawn a different sample?" `bootstrap_plausibility_threshold` answers that question by refitting the geodesic logistic regression on $K$ resamples of the training pairs:

```python
from knowlytix.harness.governance.plausibility_calibration import (
    bootstrap_plausibility_threshold,
)

tau = bootstrap_plausibility_threshold(
    store, n_bootstrap=200, seed=42,
)

print(f"tau_plausibility = {tau.mean:.4f}")
print(f"95% CI           = [{tau.ci_low:.4f}, {tau.ci_hi:.4f}]")
print(f"std              = {tau.std:.4f}")
print(f"n_bootstrap      = {tau.n_bootstrap}")
```

The function returns a `ThresholdDistribution` (mean, std, 95% CI, n_bootstrap). The runtime verifier consumes `mean` as the operating threshold; the auditor reads `ci_low`/`ci_hi` whenever a verdict lands near the boundary. The function raises `RuntimeError` when the corpus has fewer than 30 (real, corrupted-tail) pairs or when the geodesic separation is too weak for at least $K/4$ resamples to converge — a thin store is told to widen the corpus rather than ship a degenerate threshold.

### CalibrationSnapshot — deterministic audit fingerprint

The sweep result by itself is not audit-grade: floating-point accuracy figures drift at the fourth or fifth decimal across CUDA reruns, so two consecutive runs of the same calibrator can disagree at byte level even when the governance answer is the same. `CalibrationSnapshot` (in `knowlytix/harness/governance/calibration_hardening/calibration_snapshot.py`) is the audit-defensible packaging:

```python
from knowlytix.harness.governance.calibration_hardening import (
    build_calibration_snapshot,
)

snapshot = build_calibration_snapshot(
    suite_id="ds_practices_v1",
    domain="ds_practices",
    grid_keys=list(param_grid.keys()),
    sweep_results=grid_results,
    optimum=best,
    ndigits=6,
)

print(snapshot.summary())
print(f"fingerprint = {snapshot.fingerprint()}")
```

The snapshot stores `suite_id`, `domain`, sorted `grid_keys`, `n_configs`, the canonical `CalibratedThreshold` optimum, and the full sweep sorted by `(theta_sim, theta_margin, theta_esc, d_max, budget)`. `canonical_payload` returns a sorted JSON byte string; `fingerprint` is the SHA-256 over that payload. Floating values are rounded to six decimals on the way in — that is the deliberate trade between absorbing CUDA non-determinism at the noise floor and preserving the deterministic governance content. Two reruns on the same store and cohort produce the same fingerprint; a corpus drift, a code change, or a numerics shift produces a different fingerprint and the auditor sees it.

### Cross-domain agreement

Calibration on one domain does not certify the threshold on another. The cross-domain agreement check answers "would the optimum I found on $A$ also win on $B$?" by comparing two `CalibrationSnapshot`s:

```python
from knowlytix.harness.governance.calibration_hardening import (
    compare_calibrations_across_domains,
)

snap_a = build_calibration_snapshot(..., domain="ds_practices",       ...)
snap_b = build_calibration_snapshot(..., domain="california_housing", ...)

report = compare_calibrations_across_domains(snap_a, snap_b)
print(report.summary())
for drift in report.drifts:
    print(f"  {drift.parameter}: {drift.delta:+.4f} "
          f"(domain_a={drift.value_a}, domain_b={drift.value_b})")
```

The report flags per-parameter drift between the two optima. Stable thresholds are a sign that the geometry is robust across corpora; unstable thresholds tell you to either domain-condition the calibration or widen the training cohort.

### Determinism for audit

The calibration notebook is one of two places in the user-guide stack with a hard determinism contract (the other is the policy snapshot). On a frozen cohort and a frozen store, two sequential runs must produce byte-identical snapshot fingerprints. The two practical rules:

1. Pin the seed everywhere the sweep touches stochasticity. `bootstrap_plausibility_threshold(seed=42)`, `grid_search` with a fixed claim ordering, `build_calibration_snapshot(ndigits=6)`. The default `ndigits=6` is the noise-floor rounding that absorbs CUDA non-determinism without losing governance signal.
2. Prefer the CPU logistic-regression pattern from `bootstrap_plausibility_threshold` over a CUDA-retraining loop. CUDA reductions are non-associative; running the same sweep twice on GPU drifts at the fourth decimal, which crosses the rounding boundary. Pull the geodesic scores once on GPU, fit the threshold on CPU, and the snapshot fingerprint is stable across reruns.

Cross-references: `knowlytix/harness/governance/calibration.py:ThresholdCalibrator`, `knowlytix/harness/governance/plausibility_calibration.py:bootstrap_plausibility_threshold`, `knowlytix/harness/governance/calibration_hardening/calibration_snapshot.py:CalibrationSnapshot`, `gms_agent_builder_ext/chapters/ch19.tex` for the calibration chapter in book form.

---

## 26. Runtime governance and bundles

A high-stakes deployment cannot recalibrate at request time. Bootstrap thresholds, adversarial sweeps, and out-of-distribution (OOD) validation are all heavy work; the runtime gate has to issue a verdict in microseconds with the bundle's signed contents as the only input. The bundle is the contract between offline validation (where the heavy work runs once) and the runtime gate (where the verdict is served). The library expresses that contract as a *taxonomy* of bundles, one per surface that needs runtime governance, all sharing the same shape: HMAC-SHA-256 signature over a canonical JSON payload, refusal at load when the signature does not verify, refusal at bind when the live snapshot diverges from the bundle's frozen values.

### Build, freeze, save, serve

The entry point is `build_bundle` in `knowlytix/harness/governance/offline_validator.py`. It runs bootstrap calibration, OOD sweep, drift baseline, and store fingerprint, then returns an unfrozen `GovernanceBundle`. Calling `freeze()` HMAC-signs the canonical payload with the key resolved by `bundle_signing.resolve_signing_key` (environment variable `GOVERNED_AGENT_SIGNING_KEY` or explicit argument); calling `save(path)` pickles the signed bundle. `serve(store, path)` is the convenience loader that verifies the signature and binds a `RuntimeGate`:

```python
from knowlytix.harness.governance.offline_validator import build_bundle
from knowlytix.harness.governance.governance_bundle import OperatingPoint, GovernanceBundle
from knowlytix.harness.governance.runtime_gate import serve, RuntimeGate

bundle = build_bundle(
    store,
    source_store_path="ds_practices_store",
    n_bootstrap=200,
    seed=42,
    operating_point=OperatingPoint(
        max_abstain_rate=0.20,
        max_review_rate=0.20,
    ),
).freeze()
bundle.save("bundles/ds_practices_v1.pkl")

gate = serve(store, "bundles/ds_practices_v1.pkl")
verdict = gate.verdict("RandomForest", "GradientBoosting")
print(f"{verdict.verdict}  p={verdict.p_contradiction}  "
      f"latency={verdict.latency_ms:.3f}ms  "
      f"bundle={verdict.bundle_signature[:12]}")
```

The runtime gate is read-only with respect to the bundle: any change to a frozen field requires producing and signing a new bundle. `GovernanceBundle.load` raises `BundleSignatureError` on a tampered or downgraded-signing bundle.

### The eight bundle types

The library ships eight bundles. Each freezes a different surface of the agent stack; each follows the same `freeze` / `save` / `load` / `verify` shape so an operator reasons about all eight artefacts the same way. Each row links to the chapter that hardened the surface and carries an audit-defensible hardening fingerprint.

| Bundle                           | Module                                                                  | Hardening chapter | Frozen contents                                                                                                                      |
|----------------------------------|-------------------------------------------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `GovernanceBundle`               | `knowlytix/harness/governance/governance_bundle.py:GovernanceBundle`                  | Ch 10             | tau_ent / tau_contra distributions, isotonic calibration grid, support index, opposition table, drift baseline, store fingerprint, operating point |
| `PolicyBundle`                   | `knowlytix/harness/governance/policy_bundle.py:PolicyBundle`                          | Ch 12             | YAML rule set + SHA-256 hash, layer-3 / layer-4 calibrated thresholds, FSM invariant proof hash, store fingerprint, k-of-n quorum    |
| `BehavioralContractBundle`       | `knowlytix/harness/governance/behaviors/contract_bundle.py:BehavioralContractBundle`  | Ch 13             | role-keyed contract fingerprints, verified-state lattice, holonomy budget, replay regression hash                                    |
| `AuditTrailBundle`               | `knowlytix/harness/governance/tracing_hardening/audit_bundle.py:AuditTrailBundle`     | Ch 14             | chain-head hash, metric fingerprint, n_packets, cohort safety rate, attestor coverage, multi-signer signatures                       |
| `ReleaseGateBundle`              | `knowlytix/harness/governance/release_gate_bundle.py:ReleaseGateBundle`               | Ch 15             | per-tier per-metric calibrated thresholds, policy-declared baseline, cohort hash, SPC slope config, monotonicity proof hash         |
| `WorkflowBundle`                 | `knowlytix/harness/governance/workflow_hardening/workflow_bundle.py:WorkflowBundle`   | Ch 16             | workflow snapshot fingerprint, release-gate verdict, pass rate, per-phase latency, accepted release-gate set                          |
| `MultiAgentBundle`               | `knowlytix/harness/governance/agents/multi_agent_bundle.py:MultiAgentBundle`          | Ch 17             | per-role fingerprints over allowed_tools / forbidden_tools, cohort hash, tool-frequency PSI baseline, handoff-latency baseline       |
| `AttackBundle`                   | `knowlytix/harness/governance/adversarial_hardening/attack_bundle.py:AttackBundle`    | Ch 18             | attack snapshot fingerprint, defense rate, n_attacks, cross-layer match rate, alt-validator agreement                                 |
| `CalibrationBundle`              | `knowlytix/harness/governance/calibration_hardening/calibration_bundle.py:CalibrationBundle` | Ch 19    | calibration snapshot fingerprint, optimum threshold vector, n_configs, alt-validator agreement, accuracy floor                       |
| `ComplianceBundle`               | `knowlytix/harness/governance/compliance_hardening/compliance_bundle.py:ComplianceBundle` | Ch 20         | compliance snapshot fingerprint, overall_status, pass_rate, per-regulation pass rates, alt-validator agreement                        |

The two bundles named in §11 of the legacy guide — `GovernanceBundle` (entity-comparison surface) and `ConstrainedLLMBundle` (constrained-LLM surface, in `knowlytix/harness/governance/llm_bundle.py`) — predate the harvest of hardening bundles in Chs 12-20; the table covers the eight surfaces called out in PLAN.md plus the canonical comparison and constrained-LLM bundles. Each bundle has a paired `bind_*_from_bundle` helper that refuses to bind when the live snapshot diverges from the frozen values.

### Drift monitoring

The bundle's `drift_baseline_te` is the tension-energy distribution from the calibration cohort. `DriftMonitor` (in `knowlytix/harness/governance/drift_monitor.py`) consumes runtime verdicts and reports population stability index against that baseline:

```python
from knowlytix.harness.governance.drift_monitor import DriftMonitor

monitor = DriftMonitor(
    bundle,
    window_size=500,
    alert_threshold=0.10,
    recalibrate_threshold=0.20,
    min_te_samples=50,
)

for v in stream_of_runtime_verdicts:
    monitor.record(v)

state = monitor.state()
print(state.summary())   # PSI, abstain_rate, severity, alert, recalibrate
```

Population-stability-index (PSI) bands follow standard model-risk practice: $\mathrm{PSI} < 0.10$ stable, $0.10$–$0.20$ minor shift (advisory), $> 0.20$ significant shift (recalibrate). The monitor also tracks abstain-rate drift and the bundle's `OperatingPoint` service-level objective (SLO); an abstain rate above `op.max_abstain_rate` raises `slo_violation` even when the tension distribution itself looks stable.

### Replay regression

`ReplayCohort` is the audit-grade equivalent of a regression test for the bundle. Build it once from labeled pairs, save it as JSON next to the bundle, and require any new bundle to reproduce the cohort verdict-for-verdict:

```python
from knowlytix.harness.governance.replay import (
    ReplayCohort, replay_bundle, compare_replays, assert_no_regression,
)
from knowlytix.harness.governance.runtime_gate import serve

cohort = ReplayCohort.from_labeled_pairs(
    labeled_pairs, name="ds_practices_v1", version="1",
)
cohort.save("bundles/ds_practices_v1.cohort.json")

old_gate = serve(store, "bundles/ds_practices_v1.pkl")
new_gate = serve(store, "bundles/ds_practices_v2.pkl")

old_result = replay_bundle(old_gate, cohort)
new_result = replay_bundle(new_gate, cohort)

delta = compare_replays(old_result, new_result)
print(delta.summary())
assert_no_regression(delta, allow_drift=False)
```

`assert_no_regression` raises `ReplayRegressionError` on any verdict change. The deployment team either updates the cohort (and signs off on the change) or rolls back the bundle.

### Reading order

A reader new to runtime governance should walk the surfaces in dependency order: §11 (GeometricReasoner, the offline calibration target), §19 (Policy Engine, the rule-layer co-attestor), §23 (Telemetry, the audit trail the bundle binds to), §24 (Release Gates, the deployment-decision surface), §25 (Threshold Calibration, the snapshot the bundle freezes), and finally this section, where all six surfaces become signed artefacts.

Cross-references: `knowlytix/harness/governance/governance_bundle.py:GovernanceBundle`, `knowlytix/harness/governance/runtime_gate.py:RuntimeGate`, `knowlytix/harness/governance/offline_validator.py:build_bundle`, `knowlytix/harness/governance/drift_monitor.py:DriftMonitor`, `knowlytix/harness/governance/replay.py:ReplayCohort`, `gms_agent_builder_ext/chapters/ch10.tex` (governance bundle and runtime gate), `gms_agent_builder_ext/chapters/ch12.tex` through `ch20.tex` (per-surface hardening bundles).

---

## 27. Compliance

This is the only section in the user guide where regulatory framing is appropriate. The governed agent stack is engineered against the principle that high-stakes decisions need the same kind of evidence trail the operating model already produces for software change control: who decided what, on which evidence, when, and how that decision can be reproduced. This section maps the engineering artifacts already produced by the rest of the stack — verifier verdicts, policy decisions, FSM contract transitions, telemetry packets, calibration snapshots — onto specific regulatory expectations from SR 11-7 (OCC/Federal Reserve) and the EU AI Act. Three components do the mapping: `ComplianceEvidenceGenerator` (rolls a pipeline-results dict into a typed evidence package), `SnapshotManager` (versions governance state for change management), and `ProvenanceRegistry` (records artifact lineage with SHA-256 fingerprints).

The framing is one-way. The library does not redefine its semantics to chase regulation. It records what it already did and presents the record under the article and expectation labels an examiner expects to see.

### 27.1 The compliance evidence package

`ComplianceEvidenceGenerator.generate_package(pipeline_results)` returns a `CompliancePackage` carrying typed `ComplianceCheck` rows. Each check names the regulation, the article or expectation, the architecture layer that produces evidence, the monograph chapter that demonstrates it, a status of `"satisfied"` / `"partial"` / `"not_applicable"`, and a one-paragraph evidence summary. The generator is stateless: every input is a key in the `pipeline_results` dict.

```python
from knowlytix.harness.governance.compliance import ComplianceEvidenceGenerator

generator = ComplianceEvidenceGenerator()
pipeline_results = {
    "models_trained": 3,
    "verification_pass_rate": 0.92,
    "policy_violations": 0,
    "contract_violations": 0,
    "spc_alerts": 0,
    "tracing_enabled": True,
    "snapshot_count": 2,
    "audit_entries": 15,
    "risk_assessment_done": True,
    "fairness_checked": True,
    "human_oversight_enabled": True,
}
package = generator.generate_package(pipeline_results)
print(package.overall_status, f"{package.pass_rate():.0%}")
print(package.to_markdown())
```

`overall_status` is `"COMPLIANT"` only when every check is `"satisfied"`; any `"partial"` or `"not_applicable"` row demotes the package to `"PARTIAL"`. `to_markdown()` renders the package grouped by regulation, with article, layer, chapter, status icon and evidence summary in a table — this is the artifact handed to a reviewer.

### 27.2 SR 11-7 expectations

`sr_11_7_checks` walks six expectations. Each expectation pulls a specific signal out of the pipeline-results dict and writes that signal into the evidence summary so the trail is verifiable.

| Expectation              | Architecture layer            | Evidence chapter                | Trigger for `"partial"`                |
|--------------------------|-------------------------------|---------------------------------|----------------------------------------|
| Conceptual Soundness     | Layer 2 (formal contracts)    | Ch 13 (Behavioral Contracts)    | `contract_violations > 0`              |
| Ongoing Monitoring       | Layer 6 (SPC + drift)         | Ch 15 (SPC and Release Gates)   | always satisfied; reports `spc_alerts` |
| Outcomes Analysis        | Layer 6 (decision traces)     | Ch 14 (Telemetry and Tracing)   | `tracing_enabled is False`             |
| Effective Challenge      | Layer 6 (verification)        | Ch 8 (Claim Verification)       | `verification_pass_rate == 0.0`        |
| Documentation            | Layer 6 (telemetry)           | Ch 14 (Telemetry and Tracing)   | `audit_entries == 0`                   |
| Change Management        | Layer 2 (contract versioning) | Ch 20 (Regulatory Alignment)    | `snapshot_count < 2`                   |

The chapter labels point into `gms_agent_builder_ext/chapters/`. The change-management row is what binds the snapshot manager (next subsection) to the compliance surface, not only the debugging surface.

### 27.3 EU AI Act articles

`eu_ai_act_checks` covers three articles directly relevant to high-stakes deployment.

| Article                                | Architecture layer         | Evidence chapter                |
|----------------------------------------|----------------------------|---------------------------------|
| Article 9 — Risk management system     | Layer 1 (policy rules)     | Ch 12 (Policy Encoding)         |
| Article 11 — Technical documentation   | Layer 6 (telemetry)        | Ch 14 (Telemetry and Tracing)   |
| Article 14 — Human oversight           | Layer 6 (release gates)    | Ch 15 (SPC and Release Gates)   |

Article 9 reads `risk_assessment_done` and `fairness_checked`; the evidence summary names the ten responsible-ML rules `PolicyEngine` enforces. Article 11 reads `tracing_enabled` and `audit_entries`; the summary points at the model inventory, training provenance, verification results, policy compliance matrix and release gate decisions in the audit trail. Article 14 reads `human_oversight_enabled` and `policy_violations`; the summary names the three-tier release gates (advisory, recommendation, action-taking) and the `human_approval_for_deploy` policy rule.

The cross-walk is intentional: every regulatory row references an architecture layer the user guide already explains and a monograph chapter the reader can follow for depth. The compliance package is *generated* evidence, not bespoke documentation.

### 27.4 Snapshot management for change control

`SnapshotManager.take_snapshot(label, state)` deep-copies a governance-state dict and stamps it with an ISO 8601 timestamp and a SHA-256 fingerprint over the serialized state. `compare(snap_a, snap_b)` returns a dict with `added`, `removed`, `changed`, `unchanged` and a one-line `summary`.

```python
from knowlytix.harness.governance.compliance import SnapshotManager

mgr = SnapshotManager()
mgr.take_snapshot("before_retrain", {
    "models": ["RandomForest_v1"],
    "policy_version": "1.0",
    "contract_violations": 0,
})
# ... retrain happens ...
mgr.take_snapshot("after_retrain", {
    "models": ["RandomForest_v1", "RandomForest_v2"],
    "policy_version": "1.0",
    "contract_violations": 0,
})
diff = mgr.compare(0, 1)
print(diff["summary"])
print("changed:", list(diff["changed"].keys()))
```

The fingerprint is the audit anchor: any reviewer with the snapshot can recompute it, and an integrity break shows up immediately. The diff is what populates the change-management row of the compliance package.

### 27.5 Provenance registry

`ProvenanceRegistry.register(name, artifact_type, source, metadata, parent)` records a row with a 16-character SHA-256 fingerprint over the serialized record. Artifacts have a `parent` pointer that the registry walks back to the root via `get_lineage(name)`. Recognized `artifact_type` values are `"dataset"`, `"model"`, `"metric"`, `"verification"`, `"policy"`, `"snapshot"`.

```python
from knowlytix.harness.governance.compliance import ProvenanceRegistry

registry = ProvenanceRegistry()
registry.register("iris_dataset", "dataset", "sklearn",
                  {"n_samples": 150, "n_features": 4})
registry.register("RandomForest_v1", "model", "training",
                  {"accuracy": 0.97}, parent="iris_dataset")
registry.register("accuracy_metric", "metric", "diagnostics",
                  {"value": 0.97}, parent="RandomForest_v1")

lineage = registry.get_lineage("accuracy_metric")
for rec in lineage:
    print(f"  {rec['type']:8s} {rec['name']:24s} {rec['fingerprint']}")
```

`get_lineage` returns the chain in root-to-leaf order, so the first entry is always the root dataset and the last is the named artifact. This is the structure NB 26 visualizes as a left-to-right DAG.

### 27.6 The full provenance chain: document → verdict

The compliance surface above is one half of the provenance picture. The other half lives on the testing side, in the harness, and connects an examiner from a *raw document on disk* through the GMS primitives all the way to a typed `Verdict` produced by the verifier. The chain has six stages:

| Stage    | Carrier object                                      | Source                                           |
|----------|-----------------------------------------------------|--------------------------------------------------|
| document | doc_id (SHA-256 of the source markdown)             | `knowlytix/harness/governance/knowledge/*.md`                  |
| section  | section_tag (normalized heading path)               | `knowlytix/knowledge/ingest.py` section parser                |
| triple   | `(head, relation, tail)` written into `doc_graph`   | `knowlytix/knowledge/ingest.py` extractor                     |
| truth    | `EntityTruth` / `RelationTruth` / `ThresholdTruth` / `ArtifactTruth` | `knowlytix/harness/testing/truth.py:TruthStore` |
| claim    | `ClaimBase` (typed: fact, numeric, threshold, comparison, reasoning) | `knowlytix/harness/testing/claims.py` |
| verdict  | `TypedVerdict` with `geodesic`, `tension_energy`, `holonomy_defect` | `knowlytix/harness/testing/router.py:VerificationRouter` |

Stages 1-3 live inside `knowlytix-knowledge`; stages 4-6 live in `knowlytix/harness/testing`. The split is deliberate. Compliance evidence (`knowlytix/harness/governance/compliance.py`) summarizes the *aggregate* trail — pass rates, audit-entry counts, snapshot diffs — while the per-claim-trace surface (`knowlytix/harness/testing/`) makes each individual verdict reproducible. An examiner who wants "show me the path from this regulatory finding back to the source document" walks `ComplianceCheck.evidence_summary` → `audit_entries` → `TypedVerdict` → `ClaimBase` → `TruthBase.doc_id` → markdown file.

The four `TruthStore` types make the middle of the chain typed:

| Type             | Links into                              | Carries                                           |
|------------------|-----------------------------------------|---------------------------------------------------|
| `EntityTruth`    | `adapter.entity_to_idx[entity]`         | embedding index in the manifold                   |
| `RelationTruth`  | `store.score_triple(head, relation, tail)` | geodesic score at ingest time                  |
| `ThresholdTruth` | `doc_graph.phase_encoders[encoder]`     | encoder name, value, phase angle                  |
| `ArtifactTruth`  | `store.lookup_enm(category, entity_id)` | exact value, SHA-256 of the stored value          |

`TruthStore.build_from_store` scans a populated `GMSExpertStore` and constructs one truth per primitive. The `truth_id` is a UUID prefix; `doc_id` is the source-document hash; `section_tag` is the normalized heading. `get_truth_for_claim(claim)` resolves any `ClaimBase` to its most specific truth, which is the join point between the verifier's typed claims and the document chain.

This split is why §27 references both `knowlytix/harness/governance/compliance.py` *and* `knowlytix/harness/testing/truth.py`. The compliance generator depends on the testing harness for the per-claim provenance backing; the testing harness depends on the compliance generator for the regulatory roll-up. Notebook 26 demonstrates the compliance side end-to-end and prints the `TruthStore` types in a sidecar cell so the reader can see both halves of the chain in one place.

### 27.7 Cross-references

- §23 Telemetry & Tracing — where `audit_entries` come from (`AuditPacket` with `prev_hash` / `content_hash`).
- §24 Release Gates & SPC — where `spc_alerts` come from.
- §18 Behavioral Contracts — where `contract_violations` come from (FSM enforcement).
- §19 Policy Engine — where `policy_violations` come from.
- §25 Threshold Calibration — where `verification_pass_rate` is anchored to a calibrated cohort.
- `knowlytix/harness/governance/compliance.py:ComplianceEvidenceGenerator.generate_package` — implementation.
- `knowlytix/harness/governance/compliance.py:SnapshotManager.compare` — diff semantics.
- `knowlytix/harness/governance/compliance.py:ProvenanceRegistry.get_lineage` — lineage traversal.
- `knowlytix/harness/testing/truth.py:TruthStore.build_from_store` — typed truth construction.
- `knowlytix/harness/testing/truth.py:TruthStore.get_truth_for_claim` — claim → truth resolution.
- `gms_agent_builder_ext/chapters/ch20.tex §20.2` — SR 11-7 mapping; `§20.3` — EU AI Act mapping; `§20.5` — generator API.
- `gmsh_testing_ext/chapters/ch20.tex` — testing-monograph chapter that reads the same compliance package from the harness side.
- Notebook `26_compliance.ipynb` walks `generate_package`, `SnapshotManager.compare` and `ProvenanceRegistry.get_lineage` end-to-end and renders the lineage DAG.

---

## 28. Testing your agent

Sections 1 through 27 of this guide build a governed agent. Once a build is signed and frozen the question shifts: how do you test it? Testing a governed agent uses a separate surface in `knowlytix/harness/testing/` rather than the build-side modules under `knowlytix/harness/governance/` and `gms/`. The build-side guide ends at deployment; the test-side guide picks up there. The test surface is GMSH (Geometric Memory System Harness), and the deeper material on it lives in the testing monograph `gmsh_testing_ext/` and in `knowlytix/harness/testing/USER_GUIDE.md`. This section is a pointer, not a duplicate.

### What testing adds on top of the build surface

The build-side primitives `score_triple`, `tension_energy`, `check_holonomy`, and `lookup_enm` (§4) are the same primitives the test harness uses to grade an agent's answers. The harness adds three things the build surface does not own. First, *systematic question generation*: `DOEGMSBenchmark` walks the GMS store and produces typed test questions (plausibility, tension, holonomy, link-prediction, multi-hop, phase, conditional, procedural) covering every primitive automatically. Second, *Design of Experiments factor sweeps*: the same base questions are rephrased and replicated across context, distractor, and difficulty factors so a regression can be attributed to a factor rather than to noise. Third, *typed verdicts and release gates*: every answer is decomposed into typed claims (`FactClaim`, `ThresholdClaim`, `RuleClaim`, ...), each verified by a dedicated verifier, then aggregated into a `TypedVerdict` that feeds a `TieredReleaseGate`.

### A 30-line end-to-end example

The minimum viable run takes a markdown source, ingests it, generates DOE questions, evaluates against any callable, and produces a `DOEBenchmarkResult` with a release gate decision. The example below uses the prebuilt regex-mode fixture and the bundled `MockHarness` evaluator from `demo/mock_agent.py` so it runs without an API key.

```python
from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig
from demo.mock_agent import create_mock_evaluator

FIXTURE = "user_guide_tutorials/fixtures"  # parent of ds_practices_store
KNOWLEDGE = "knowlytix/harness/governance/knowledge/data_science_practices.md"

config = DOEHarnessConfig(
    markdown_path=KNOWLEDGE,
    stores_dir=FIXTURE,
    store_name="ds_practices_store",
    factor_group="quick_screen",
    n_runs=4,
    max_per_category=2,
    target_type="agent",
    target_model="mock-agent-v1",
    enable_typed_verdicts=True,
    enable_tracing=True,
)

bench = DOEGMSBenchmark(config)
bench.ingest()                             # loads existing fixture; no LLM
bench.generate_questions()
result = bench.run(evaluator=create_mock_evaluator(recovery_mode="fallback"))
analyzer = bench.analyze(result)
print(f"accuracy = {result.accuracy:.1%}   ({result.total_correct}/{result.total_questions})")
gate = bench.run_release_gate(result, analyzer, accuracy_threshold=0.50)
print(f"release_gate = {'PASS' if gate.passed else 'FAIL'}")
```

The `create_mock_evaluator()` callable is `(question, context) -> answer`; swap in a real agent or LLM by passing a different callable that respects the same signature. The release gate raises a structured `ReleaseGate` decision rather than a Boolean, so an audit trail of *why* a release passed or failed is always attached.

### Where to go for depth

The harness has 19 chapters of tutorial material in `gmsh_testing_ext/` and 38 sections in `knowlytix/harness/testing/USER_GUIDE.md`. Use this guide as the entry point, then read the testing monograph end-to-end if your role includes signing release gates or attesting to GMSTO numbers.

| Topic | Depth lives in |
|-------|----------------|
| Tool gateway and fault injection | `knowlytix/harness/testing/USER_GUIDE.md` §5; `gmsh_testing_ext/chapters/ch08.tex` |
| Policy engine for test-time enforcement | `knowlytix/harness/testing/USER_GUIDE.md` §6; `gmsh_testing_ext/chapters/ch09.tex` |
| Authored scenarios and replay regression | `knowlytix/harness/testing/USER_GUIDE.md` §7-9; `gmsh_testing_ext/chapters/ch10-11.tex` |
| Process / recovery / interaction evaluators | `knowlytix/harness/testing/USER_GUIDE.md` §10; `gmsh_testing_ext/chapters/ch12.tex` |
| Risk-tiered release gates | `knowlytix/harness/testing/USER_GUIDE.md` §11; `gmsh_testing_ext/chapters/ch13.tex` |
| Severity classification and failure taxonomy | `knowlytix/harness/testing/USER_GUIDE.md` §15; `gmsh_testing_ext/chapters/appendix_c.tex` |
| Campaign management for long test runs | `knowlytix/harness/testing/USER_GUIDE.md` §17; `gmsh_testing_ext/chapters/ch14.tex` |
| Adversarial and boundary generators | `knowlytix/harness/testing/USER_GUIDE.md` §18; `gmsh_testing_ext/chapters/ch15.tex` |
| Hallucination detection (geodesic band) | `knowlytix/harness/testing/USER_GUIDE.md` §27; `gmsh_testing_ext/chapters/ch18.tex` |
| Confidence calibration on a labelled cohort | `knowlytix/harness/testing/USER_GUIDE.md` §28; `gmsh_testing_ext/chapters/ch19.tex` |
| End-to-end walkthrough | `knowlytix/harness/testing/USER_GUIDE.md` §34 |

The companion notebook NB 27 in `user_guide_tutorials/notebooks/` runs the example above end-to-end and prints the factor-effect summary so you can see the harness output before opening the deeper material.

---

## 29. Common patterns

### Fresh store for deterministic results

Delete stale state before rebuilding to avoid accumulated triples
from prior runs:

```python
import shutil, contextlib, io, sys

@contextlib.contextmanager
def quiet():
    """Suppress stdout during GMS training/ingestion."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old

def fresh_memory(store_path, knowledge_path, device):
    """Build a fresh AgentMemory (delete prior state)."""
    shutil.rmtree(store_path, ignore_errors=True)
    with quiet():
        mem = AgentMemory(
            store_path=store_path,
            knowledge_path=knowledge_path,
            device=device,
        )
    return mem
```

### Signal vector interpretation

Each orchestrator step produces a six-component signal vector
$(\text{sim}, \text{ver}, \text{depth}, \text{nov}, \text{margin}, \text{signal\_unavailable})$.
`as_tuple()` returns the five numeric components:

| Component | Range | Meaning |
|-----------|-------|---------|
| `sim` | [0, 1] | Plausibility (inverted geodesic distance) |
| `ver` | {0, 1} | 1 if extraction produced structured triples |
| `depth` | [0, 1] | Normalized step progress (step / total) |
| `nov` | {0, 1} | 1 if new triples were extracted |
| `margin` | [0, 1] | Confidence margin from plausibility gate |
| `signal_unavailable` | bool | True when neither a `policy_store` nor a gateway plausibility score was available; downstream consumers should treat the numeric signal as missing rather than trusted |

### Tension energy interpretation

Tension energy between two entities on $S^{m-1}$:

| Range | Interpretation |
|-------|---------------|
| 0 -- 0.5 | Agreement |
| 0.5 -- 1.414 | Irrelevance |
| 1.414 -- 2.0 | Contradiction |

---

## 30. Mathematical appendix

This appendix collects the geometric definitions used throughout the guide. Every primitive in §4 reduces to one of the equations below. The full derivations and constraint-aware optimizer details live in `gmsh_testing_ext/chapters/appendix_a.tex` and `gms_agent_builder_ext/chapters/appendix_c.tex`; this section is the operating reference.

### The unit hypersphere

Every entity embedding lives on the unit hypersphere

$$
S^{m-1} = \{\, v \in \mathbb{R}^m \mid \|v\|_2 = 1 \,\}, \qquad m = 128
$$

at machine precision ($\max |\|v\| - 1| < 10^{-6}$). Three properties motivate the choice. Fixed-norm embeddings give every entity equal geometric weight, so no entity can dominate by scale alone. Rotations on $S^{m-1}$ form a closed group $SO(m)$, so composition of relations stays a valid relation. The geodesic distance is bounded in $[0, \pi]$, which gives a calibratable range for plausibility.

### Geodesic distance

For $a, b \in S^{m-1}$ the geodesic distance is the angle between them:

$$
d(a, b) = \arccos(a \cdot b) \in [0, \pi].
$$

This is the score returned by `score_triple(h, r, t) = d(R_r \tilde h, \tilde t)`. Lower means more plausible.

### Cayley rotors

Each relation $r$ is represented by a learned rotation $R_r \in SO(m)$ parameterized through the Cayley transform of an antisymmetric generator $W_r = -W_r^\top$:

$$
R_r = (I - W_r)(I + W_r)^{-1}.
$$

The Cayley transform of any antisymmetric $W$ is automatically orthogonal and has determinant $+1$, so $R_r R_r^\top = I$ and $\det R_r = +1$ are *structural* rather than penalized. Composition $R_{r_2} R_{r_1}$ is therefore another valid rotor — that closure is what makes multi-hop verification well-defined.

### Tension energy

Contradiction lives in the second embedding view. For two logical embeddings $a, b \in S^{m-1}$:

$$
\mathrm{tension}(a, b) = \sqrt{2 - 2\, a \cdot b} \in [0, 2].
$$

Three bands carry distinct semantic meaning:

| Range | Geometry | Interpretation |
|-------|----------|----------------|
| $[0, 0.5)$ | $\theta \approx 0$ | Agreement |
| $[0.5, \sqrt{2})$ | $\theta \approx \pi/2$ | Irrelevance |
| $[\sqrt{2}, 2]$ | $\theta \approx \pi$ | Contradiction |

Three bands, not two. Collapsing irrelevance and contradiction is what cosine-similarity stores do; doing so loses the distinction between *unrelated* and *actively disagreeing*.

### Holonomy defect

For a relation chain $r_1 \cdots r_k$ versus a direct relation $r_{\mathrm{dir}}$ the holonomy defect is the Frobenius norm of the residual rotor:

$$
H = \big\| R_{r_k} \cdots R_{r_1}\, R_{r_{\mathrm{dir}}}^{-1} - I \big\|_F.
$$

$H \approx 0$ means the chain and the direct relation agree as transformations. $H \gg 0$ means the chain is internally inconsistent. This is the operating signal for multi-hop verification (FM-4 in the testing monograph).

### Phase encoding on $S^1$

Each entity carries an optional phase $\phi \in [0, 2\pi)$ embedded as a unit vector on the circle:

$$
\Phi(\phi) = (\cos\phi,\; \sin\phi) \in S^1.
$$

Phase distance $d_\Phi(\phi_a, \phi_b) = \arccos(\Phi(\phi_a) \cdot \Phi(\phi_b))$ is the same geodesic primitive at $m = 2$ and is what drives lifecycle, ordering, and temporal-precedence checks. Phase composes additively under known relations: a `precedes` rotor adds a fixed offset on $S^1$.

### Stiefel projection and the dual $u$/$v$ split

Every entity $e$ has two complementary views:

$$
\tilde e = (u_e, v_e), \qquad u_e \in S^{m-1},\; v_e \in S^{m-1}.
$$

The split is realized by a Stiefel projection: the constraint-aware optimizer keeps the stacked column matrix $[u_e \mid v_e] \in \mathbb{R}^{m \times 2}$ on the Stiefel manifold $V_2(\mathbb{R}^m) = \{X \in \mathbb{R}^{m \times 2} \mid X^\top X = I_2\}$, so $u_e \perp v_e$ and both are unit-norm. The semantic view $u$ carries relation transport (used by `score_triple`); the logical view $v$ carries logical commitments (used by `tension_energy`). Natural-language-inference (NLI) style claim-claim entailment uses $v$; relational fact verification uses $u$. This separation is why GMS can score *plausibility* and detect *contradiction* in the same store without those signals collapsing onto a single axis.

### Exact numeric memory (ENM)

Geometric scores are calibrated approximations. Numeric values that must be exact — test AUC, threshold parameters, calibrated $\theta$ — are stored in a separate side register, the *Exact Numerical Memory*. ENM stores the IEEE-754 64-bit byte representation of every numeric value alongside a SHA-256 key derived from `(category, entity_path)`:

$$
\mathrm{key}(\mathrm{cat}, \mathrm{path}) = \mathrm{SHA\text{-}256}(\mathrm{cat} \,\|\, \mathrm{path}),
$$

$$
\mathrm{ENM}[\mathrm{key}] = (\,\mathrm{IEEE\text{-}754\ bytes},\ \mathrm{SHA\text{-}256\ checksum}\,).
$$

A `lookup_enm(category, entity_path)` call rehydrates the IEEE-754 bytes byte-identically and re-verifies the checksum on every read. Approximate similarity is in the geometry; lossless precision is in ENM. Audit-defensible numbers should always go through ENM rather than through `score_triple`.

---

## 31. Companion resources

| Resource | Description |
|----------|-------------|
| `user_guide_tutorials/` | Executable companion to this guide. Notebooks NB 01-27 cover sections §1-28 (excluding §16, which is conceptual only) with runnable code, GPU-tested visualizations, and per-notebook assertions. Start with `user_guide_tutorials/README.md` |
| `gms_agent_builder_ext/` | "How to Build Policy-Governed AI Agents" -- external (operator encoding) monograph, 20 chapters + 6 appendices, LaTeX source |
| `gms_agent_builder_int/` | Internal (rotor/phase encoding) edition with full math derivations |
| `gms_agent_builder_ext/notebooks/` | Executable Jupyter notebooks covering every chapter |
| `gmsh_testing_ext/` | "How to Test and Validate Agentic AI Systems" -- testing monograph, 19 chapters + 7 appendices, LaTeX source. The depth reference for §28 |
| `knowlytix/harness/testing/USER_GUIDE.md` | Step-by-step walkthrough of the GMSH testing harness, 38 sections; entry point for testers |
| `knowlytix/harness/testing/API_GUIDE.md` | Full constructor signatures, dataclass fields, return types, and enumerations for every public class in `knowlytix.harness.testing` |
| `knowlytix/harness/governance/knowledge/data_science_practices.md` | Default domain knowledge for GMS ingestion |
| `knowlytix/harness/governance/knowledge/california_housing_domain.md` | California Housing reference ontology used by the complete-workflow chapter |
| `knowlytix/harness/governance/` | Library source code (~70 modules organized into core surface plus runtime-governance, policy-, plan-, orchestrator-, release-gate-, LLM-, behavioral-contract-, and multi-agent-hardening families) |
| `demo/` | Example scripts and mock agents |
| `API_GUIDE.md` | Complete API reference |
