# DOE-GMS Benchmark: Technical Reference

## 1. System Overview

The DOE-GMS Benchmark is a systematic framework for evaluating LLM document comprehension using Geometric Memory Systems (GMS) as both ground truth and geometric judge. It combines three subsystems:

1. **DocGMS** -- converts markdown documents into trained GMS expert stores on the hypersphere $S^{m-1}$ ($m = 128$)
2. **Question Generators** -- twelve generators (six classical, six geometric) that produce questions exercising distinct GMS primitives
3. **graphDOE** -- Design of Experiments engine that varies question presentation factors via Sobol quasi-random sequences and analyzes results via logistic regression

The pipeline produces three outputs for each evaluation: (a) binary correctness via traditional scoring, (b) a `GeometricVerdict` with continuous confidence scores and (c) a tracing GMS store recording all events as geometric triples.

### 1.1 Module Layout

```
harness/doe/
    __init__.py         # Public API exports
    __main__.py         # python -m harness.doe entry
    config.py           # DOEHarnessConfig dataclass
    benchmark.py        # DOEGMSBenchmark orchestrator (7-stage pipeline)
    bridge.py           # Adapter: eval generators <-> graphDOE question format
    generators.py       # 6 geometric question generators
    judge.py            # GMSJudge producing GeometricVerdict
    evaluators.py       # LLMEvaluator, AgentEvaluator wrappers
    tracing.py          # DOETracingGMS (evaluation audit log)
    cli.py              # CLI entry point (gms-doe command)
    tests/
        model_risk_assessment.md   # Test document (Meridian Financial)
        test_e2e.py                # End-to-end validation
        run_llm_benchmark.py       # Sonnet vs Opus benchmark script
        results/                   # Saved benchmark JSON results
```

### 1.2 Dependencies

| Package | Role |
|---------|------|
| `gms` | Core geometric knowledge graph (embeddings, rotors, transport) |
| `docgms` | Document ingestion, triple extraction, GMS training |
| `eval` | Classic question generators and type-aware scorers |
| `finstructbench` | Graph-verifiable question format and document graph |
| `graphdoe` | Design matrix (Sobol/LHS/Grid), question rephraser, DOE analyzer |
| `anthropic` | Claude API client (optional, for LLM evaluation) |
| `torch` | Geometric computation backend |

---

## 2. Configuration (`config.py`)

### 2.1 DOEHarnessConfig

A `@dataclass` centralizing all parameters. Fields are grouped by function:

**Document ingestion:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `markdown_path` | `str` | `""` | Source markdown document |
| `stores_dir` | `str` | `"gms_stores"` | Directory for persisted GMS stores |
| `store_name` | `str` | `""` | Store name; auto-derived from filename if empty |
| `ingest_epochs` | `int` | `200` | GMS training epochs during ingestion |

**DOE design:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `factor_group` | `str` | `"quick_screen"` | Factor group from graphDOE catalog |
| `factor_names` | `list[str]` | `[]` | Specific factor names; empty = all in group |
| `n_runs` | `int` | `32` | Number of design matrix runs |
| `method` | `str` | `"sobol"` | Design method: `"sobol"`, `"lhs"`, `"grid"` |
| `seed` | `int` | `42` | Random seed for reproducibility |

**Question generation:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rephrase_method` | `str` | `"template"` | Rephrasing: `"template"` (deterministic) or `"llm"` |
| `rephrase_model` | `str` | `"claude-sonnet-4-20250514"` | Model for LLM-based rephrasing |
| `max_per_category` | `int` | `10` | Maximum base questions per generator |
| `generators` | `list[str]` | `[]` | Generator names; empty = all 12 generators |

**Evaluation target:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_type` | `str` | `"llm"` | Target type: `"llm"` or `"agent"` |
| `target_model` | `str` | `"claude-sonnet-4-20250514"` | Model identifier |
| `context_mode` | `str` | `"full"` | Context: `"full"`, `"oracle_section"`, `"noisy_section"` |

**Tracing and device:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tracing_dir` | `str` | `"doe_tracing_store"` | Directory for tracing GMS |
| `enable_tracing` | `bool` | `True` | Whether to trace evaluation events |
| `device` | `str` | `""` | PyTorch device; empty = auto-detect CUDA |

### 2.2 Methods

**`from_yaml(path: str) -> DOEHarnessConfig`** (classmethod)

Loads a YAML config file, flattens nested sections into a flat dict and filters to known dataclass fields:

```python
with open(path) as f:
    raw = yaml.safe_load(f)
flat = {}
for section in raw.values():
    if isinstance(section, dict):
        flat.update(section)
known = {f.name for f in cls.__dataclass_fields__.values()}
return cls(**{k: v for k, v in flat.items() if k in known})
```

**`resolve_store_name() -> str`**

Returns `store_name` if set, otherwise derives from the markdown filename: strip extension, replace spaces with underscores, lowercase.

---

## 3. Benchmark Orchestrator (`benchmark.py`)

### 3.1 DOEGMSBenchmark

The central class managing the seven-stage pipeline. Constructor accepts a `DOEHarnessConfig` and initializes:

- `device`: resolved `torch.device` (CUDA if available, else CPU)
- `factors`: loaded from `FactorLoader` with the configured group/names
- `design_matrix`: `DesignMatrix(factors, method, n_runs, seed)`
- `rephraser`: `QuestionRephraser(rephrase_method, rephrase_model)`
- `tracing`: `DOETracingGMS(stores_dir, device)` if `enable_tracing=True`
- `store`, `judge`, `markdown`, `_doe_questions`: initialized to `None`

### 3.2 Stage 1: Ingestion

**Method:** `ingest(markdown_path: str | None = None) -> GMSExpertStore`

1. Resolves markdown path from argument or `config.markdown_path`
2. Creates `DocGMSConfig` with `store_path = stores_dir/store_name` and `epochs = ingest_epochs`
3. Initializes `GMSExpertStore(config, device)`
4. If store exists on disk: loads it. Otherwise: creates `AnthropicBackend()` and calls `ingest_document(store, path, llm, doc_config, device)` which extracts triples, trains dual embeddings on $S^{m-1}$ for the configured epochs and populates the ENM
5. Reads raw markdown text into `self.markdown`
6. Creates `GMSJudge(store)` for geometric scoring
7. Prints store statistics (entities, relations, triples, ENM entries)

### 3.3 Stage 2: Question Generation

**Method:** `generate_questions() -> list[DOEQuestion]`

1. Calls `bridge.generate_base_questions(store, generator_names, max_per_category, seed)`:
   - Loads classic generators via `eval.generators.default_generators()`
   - Loads geometric generators via `generators.geometric_generators()`
   - Filters by `generator_names` if specified (empty = all 12)
   - For each generator: `gen.generate(store)` -> `gen.sample(max_per_category, seed)` -> validate -> adapt to graphDOE format
2. Generates design matrix: `self.design_matrix.generate()` -> DataFrame of factor assignments
3. Expands base questions: `self.rephraser.expand(base_questions, design_matrix)`
   - Each base question is rephrased under each row of the design matrix
   - Total DOE questions = `len(base_questions) * n_runs`
   - Ground truth is **never** altered during rephrasing
4. Caches result in `self._doe_questions`

### 3.4 Stage 3-5: Evaluation, Scoring and Judging

**Method:** `run(evaluator=None, llm_client=None, model="") -> DOEBenchmarkResult`

Must provide either `evaluator` (callable `(question, context) -> str`) or `llm_client` (Anthropic client).

**Per-question loop:**

```
for each DOE question dq:
    1. Extract context via _get_context(dq, markdown, section_map, context_mode, rng)
    2. Evaluate: raw_answer = evaluator(dq.question.llm_prompt, context)
    3. Parse: answer = PARSERS[answer_type](raw_answer)
    4. Score: score = score_answer(answer, ground_truth)
    5. Judge: verdict = self.judge.judge(raw_answer, ground_truth, score.correct, ...)
    6. Inject geometric scores into factor_assignment:
         _geo_confidence, _tension, _holonomy, _overall_confidence
    7. Trace question via DOETracingGMS
    8. Append DOEQuestionResult to result
```

**Context extraction** (`_get_context`):

| Mode | Behavior |
|------|----------|
| `"full"` | Return entire markdown document |
| `"oracle_section"` | Return only the section containing the answer |
| `"noisy_section"` | Return relevant section + 1-2 random distractor sections, shuffled |

Section inference (`_section_for_question`) checks metadata fields `enm_type`, `type1` and regex patterns in the prompt, falling back to the full document if no section can be inferred.

**Error handling:** any exception during evaluation creates a failed `DOEQuestionResult` with `answer="ERROR: {e}"` and `correct=False`. Execution continues to the next question.

### 3.5 Stage 5 (Alternative): GMS Baseline

**Method:** `run_gms_baseline() -> DOEBenchmarkResult`

Validates ground truth by calling `q.graph_answer_fn(None)` for each DOE question (the closure ignores the graph argument and re-derives the answer from the GMS store). Expected accuracy: 100%. Any deviation indicates a ground truth issue.

### 3.6 Stage 6: Statistical Analysis

**Method:** `analyze(result: DOEBenchmarkResult) -> DOEAnalyzer`

1. Creates `DOEAnalyzer(result)` from graphDOE
2. Runs `analyzer.run_logistic()`: per-factor logistic regression with analysis of deviance
3. Traces analysis results to `DOETracingGMS`
4. Saves tracing store

### 3.7 Stage 7: Tracing Persistence

**Method:** `save_tracing()`

Saves the tracing GMS store to disk and prints entity/triple counts.

### 3.8 Results Display

**Method:** `print_results(result: DOEBenchmarkResult)`

Prints per-category table with columns: Category, Correct, Total, Rate, GeoConf, Tension. Then prints aggregate geometric summary: mean geodesic confidence, mean tension score, mean holonomy score, mean overall confidence.

---

## 4. Question Bridge (`bridge.py`)

### 4.1 Problem

The eval generators produce `eval.generators.base.GeneratedQuestion` objects with a `gms_answer_fn(store) -> answer` method. The graphDOE rephraser expects `finstructbench.generators.base.GeneratedQuestion` objects with a `graph_answer_fn(graph) -> answer` method. These two interfaces are incompatible.

### 4.2 Solution: Closure-Based Adapter

**Function:** `adapt_question(eq: EvalQuestion, store: GMSExpertStore) -> FSBQuestion`

```python
FSBQuestion(
    qid=eq.qid,
    category=eq.category,
    natural_language=eq.natural_language,
    graph_answer_fn=lambda _graph, _s=store, _fn=eq.gms_answer_fn: _fn(_s),
    ground_truth=eq.ground_truth,
    llm_prompt=eq.llm_prompt,
    answer_type=eq.answer_type,
    metadata=eq.metadata,
)
```

The closure captures `store` and `gms_answer_fn` via default argument binding. When `graph_answer_fn(_graph)` is called (by graphDOE), it ignores `_graph` and calls `gms_answer_fn(store)`.

### 4.3 Generator Orchestration

**Function:** `generate_base_questions(store, generator_names=None, max_per_category=10, seed=42) -> list[FSBQuestion]`

1. If `generator_names` is empty/None: loads all classic generators + all geometric generators
2. If `generator_names` is specified: loads only the named generators from `_GENERATOR_MAP`:

```python
_GENERATOR_MAP = {
    # Classic
    "exact_recall", "threshold", "cross_reference",
    "contradiction", "multi_hop", "counting",
    # Geometric
    "plausibility", "tension", "holonomy",
    "link_prediction", "multi_hop_transport", "phase_comparison",
}
```

3. For each generator: generate candidates -> sample up to `max_per_category` -> validate -> adapt via `adapt_question`

---

## 5. Geometric Question Generators (`generators.py`)

Six generators that exercise distinct GMS geometric primitives. All inherit from `eval.generators.base.QuestionGenerator` and implement `generate(store) -> list[GeneratedQuestion]`.

### 5.1 PlausibilityGenerator

**Category:** `"plausibility"`

**GMS Primitive:** `store.score_triple(h, r, t)` -- geodesic distance on $S^{m-1}$

**Question subtypes:**

*Comparative plausibility* -- For each known triple $(h, r, t)$, sample a random wrong tail $t'$ from the entity set. Compute geodesic distances $d_{\text{real}} = d_{\text{geo}}(h, r, t)$ and $d_{\text{wrong}} = d_{\text{geo}}(h, r, t')$. Include the question only if $d_{\text{real}} < d_{\text{wrong}} - 0.1$ (clear geometric separation). Randomize A/B ordering. The LLM must choose which statement is more plausible. Answer type: `str` (A or B).

*Absolute plausibility* -- Classify a triple as plausible if $d_{\text{geo}}(h, r, t) < 0.5$. The LLM must judge whether a statement is supported by the document. Answer type: `bool`.

**Sampling:** first 50 triples for comparative, first 30 for absolute.

**Thresholds:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Comparative margin | 0.1 | Minimum geodesic difference between real and wrong |
| Absolute plausibility | 0.5 | Geodesic below this = plausible |

### 5.2 TensionGenerator

**Category:** `"tension"`

**GMS Primitive:** `store.tension_energy(a, b)` -- Clifford tension energy $E = 2\sin(\theta/2) = \sqrt{2 - 2\,\mathbf{u}_a \cdot \mathbf{u}_b}$

**Physics:**

| Energy range | Interpretation |
|---|---|
| $E \approx 0$ | Agreement/entailment |
| $E \approx \sqrt{2} \approx 1.41$ | Irrelevance |
| $E \approx 2$ | Contradiction |

**Algorithm:**
1. Sample up to 200 entity pairs with deduplication
2. Compute tension energy for each pair
3. Sort by tension

**Question subtypes:**

*Classification* -- For each entity pair, assign a label based on tension:

```
if E < 0.5:   label = "agree"
if E > 1.5:   label = "contradict"
otherwise:    label = "unrelated"
```

The LLM must identify the relationship. Answer type: `str` (agree/contradict/unrelated).

*Comparative* -- Pair a low-tension pair ($E < 0.5$) with a high-tension pair ($E > 1.0$). The LLM must identify which pair has greater contradiction. Randomize A/B positioning. Answer type: `str` (A or B).

**Thresholds:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Agree threshold | 0.5 | Below this = agreement |
| Contradict threshold | 1.5 | Above this = contradiction |
| Comparative high threshold | 1.0 | Minimum tension for high pair |

### 5.3 HolonomyGenerator

**Category:** `"holonomy"`

**GMS Primitive:** `store.check_holonomy([r1, r2], r_direct)` -- rotor composition consistency

**Mathematical basis:** Given an indirect path $h \xrightarrow{r_1} m \xrightarrow{r_2} t$ and a direct edge $h \xrightarrow{r_d} t$, the holonomy defect is:

$$\delta = \|R_{r_2} R_{r_1} R_{r_d}^{-1} - I\|_F$$

**Algorithm:**
1. Build adjacency map from all triples: `head -> [(relation, tail), ...]`
2. Find triangles: for each direct edge $(h, r_d, t)$, search for a two-hop path $h \to m \to t$ via the adjacency map
3. Compute holonomy defect for each triangle
4. Generate question: "Is the indirect path from $h$ through $m$ to $t$ consistent with the direct relationship?"
5. Ground truth: `consistent = (defect <= 0.5)`

**Threshold:** defect $\leq 0.5$ = consistent. Answer type: `bool`.

### 5.4 LinkPredictionGenerator

**Category:** `"link_prediction"`

**GMS Primitive:** `store.link_predict(head, relation, top_k=5)` -- beam search ranking all entities by geodesic distance after applying the relation rotor

**Algorithm:**
1. Shuffle triples, iterate unique $(h, r)$ pairs
2. For each pair, get top-5 predictions
3. **Fairness filter:** include the question only if the ground truth tail appears in the top-3 predictions
4. Question: "What is the `<relation>` of `<head>`?"
5. Ground truth: the actual tail entity

**Limit:** up to 40 questions after filtering. Answer type: `str`.

### 5.5 MultiHopTransportGenerator

**Category:** `"multi_hop_transport"`

**GMS Primitive:** `store.transport.path_score(h_idx, [r1_idx, r2_idx], t_idx)` -- composes relational rotors along a chain and measures geodesic distance to the target

**Algorithm:**
1. Build adjacency map from all triples
2. Find 2-hop chains: $h \xrightarrow{r_1} m \xrightarrow{r_2} t$ where $m \neq h$, $t \neq h$, $t \neq m$
3. Compute path score: $s = \arccos(\mathbf{v}_{\text{transported}}^\top \tilde{\mathbf{v}}_t)$ where $\mathbf{v}_{\text{transported}} = R_{r_2} R_{r_1} \tilde{\mathbf{v}}_h$
4. **Distance filter:** skip paths with $s > 2.0$ (geometrically unreachable)
5. Question: "Starting from $h$, follow $r_1$ then $r_2$. What do you reach?"
6. Ground truth: the tail entity

**Limit:** up to 30 questions after filtering. Answer type: `str`.

**Threshold:** path geodesic $\leq 2.0$.

### 5.6 PhaseComparisonGenerator

**Category:** `"phase_comparison"`

**GMS Primitive:** Exact Numerical Memory (ENM) lookup via `store.query_exact(category, entity_id)`. Phase encoding on $S^1$ is the underlying mechanism; the answer function uses exact ENM values for ground truth precision.

**Algorithm:**
1. Extract all ENM entries grouped by type/category: `key.type -> [(key.id, value), ...]`
2. Within each category, sort entries by numeric value
3. Generate comparison pairs within sliding windows of size 4
4. Require $|v_a - v_b| > 10^{-9}$ for valid comparison
5. Randomize A/B positioning
6. Question: "Which has a higher `<metric>`: $A$ or $B$?"
7. Ground truth: `"A"` if `val_a > val_b` else `"B"`

**Limit:** up to 8 pairs per category. Answer type: `str`.

---

## 6. GMS Geometric Judge (`judge.py`)

### 6.1 GeometricVerdict

A `@dataclass` extending binary correctness with four continuous geometric scores:

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `correct` | `bool` | -- | Traditional exact-match result |
| `geodesic_confidence` | `float` | $[0, 1]$ | Manifold fit: $e^{-\bar{d}}$ |
| `tension_score` | `float` | $[0, 2]$ | Contradiction level: 0 = agree, 2 = contradict |
| `holonomy_score` | `float` | $[0, \infty)$ | Path consistency: 0 = consistent |
| `overall_confidence` | `float` | $[0, 1]$ | Weighted composite |
| `extracted_triples` | `list[ExtractedTriple]` | -- | Triples found in answer |
| `scored_triples` | `list[ScoredTriple]` | -- | Triples scored against manifold |
| `contradictions` | `list[ContradictionResult]` | -- | Detected contradictions |
| `inconsistencies` | `list[HolonomyResult]` | -- | Detected path inconsistencies |

### 6.2 GMSJudge

**Constructor:** `GMSJudge(store, verify_config=None, weights=None)`

**Default weights:**

```python
DEFAULT_WEIGHTS = {
    "geodesic": 0.3,   # manifold fit
    "tension":  0.2,   # consistency with known facts
    "holonomy": 0.2,   # reasoning path coherence
    "exact":    0.3,   # traditional exact-match
}
```

If the GMS model is trained, the constructor also builds a `GeometricSearch` instance with adjacency from `doc_graph.triples`, enabling triple-set confidence scoring as a fallback.

### 6.3 Judging Algorithm

**Method:** `judge(llm_answer, ground_truth, exact_correct, exact_detail, question_metadata) -> GeometricVerdict`

**Step 1: Triple extraction** (`_extract_triples`)

Two sources, in priority order:

1. **Metadata triples:** Extract `(head, relation, tail)` from question metadata fields. For multi-hop questions, also extract intermediate steps via `mid`, `r1`, `r2`.
2. **Text-matched triples:** Find entities mentioned in the LLM answer (case-insensitive substring matching against all store entities with length > 2). For each entity pair found, check if a known relation exists in `doc_graph.triples`.

If no triples are extracted, fall back to `_score_from_metadata` (Section 6.5).

**Step 2: Triple scoring** (`_score_triples`)

For each extracted triple:
1. Fuzzy-match head and tail against store entities (exact -> case-insensitive -> substring)
2. If both match: `gms_score = store.score_triple(head_match, relation, tail_match)`
3. Compute geodesic confidence: $\text{geo\_conf} = \exp(-\bar{d})$ where $\bar{d}$ is the mean geodesic distance across all scored triples

**Step 3: Tension verification**

1. Gather entity pairs from scored triples
2. Run `check_contradictions(entity_pairs, store, config)` from `docgms.verify`
3. Compute mean tension energy: $\bar{E} = \text{mean}\{E_i\}$

**Step 4: Holonomy checking**

1. Run `check_holonomy(scored_triples, store, config)` from `docgms.verify`
2. Compute mean holonomy defect: $\bar{\delta} = \text{mean}\{\delta_j\}$

**Step 5: Overall confidence**

$$s_{\text{overall}} = w_e \cdot c + w_g \cdot e^{-\bar{d}} + w_t \cdot \max\!\left(1 - \frac{\bar{E}}{2},\; 0\right) + w_h \cdot e^{-\bar{\delta}}$$

where $c = 1$ if correct else $0$. The result is clamped to $[0, 1]$.

| Component | Weight | Signal | Transformation |
|-----------|--------|--------|----------------|
| Exact match | 0.3 | Binary correctness | $c \in \{0, 1\}$ |
| Geodesic | 0.3 | Mean geodesic distance | $e^{-\bar{d}} \in [0, 1]$ |
| Tension | 0.2 | Mean tension energy | $(1 - \bar{E}/2) \in [0, 1]$ |
| Holonomy | 0.2 | Mean holonomy defect | $e^{-\bar{\delta}} \in [0, 1]$ |

### 6.4 Detail String

The verdict detail combines the traditional scoring detail with geometric metrics:

```
{exact_detail} | GMS: geo_conf=0.550 tension=1.407 holonomy=0.230 overall=0.490
```

Contradictions and inconsistencies are appended if present.

### 6.5 Metadata-Based Fallback

**Method:** `_score_from_metadata(verdict, llm_answer, ground_truth, metadata) -> GeometricVerdict`

When no triples can be extracted from the answer text, the judge scores the ground truth triple from metadata instead:

1. Score `(head, relation, tail)` from metadata via `store.score_triple`
2. Extract tension energy and holonomy defect from metadata if present (generators embed these)
3. Use `GeometricSearch.score_triple_set` for additional confidence if available
4. Compute overall confidence via the same weighted formula

This ensures every question produces geometric signals even when the LLM answer contains no recognizable entities.

---

## 7. Evaluators (`evaluators.py`)

### 7.1 LLMEvaluator

Wraps direct Claude API calls:

```python
class LLMEvaluator:
    def __init__(self, client: anthropic.Anthropic, model: str)
    def __call__(self, question: str, context: str) -> str:
        return call_llm(self.client, context, question, self.model)
```

Uses `eval.llm_caller.call_llm` which sends the context as a system-level document and the question as the user message.

### 7.2 AgentEvaluator

Wraps the Harness agent loop:

```python
class AgentEvaluator:
    def __init__(self, harness: Harness)
    def __call__(self, question: str, context: str) -> str:
        self.harness.messages = []    # Reset conversation
        self.harness._turn = 0
        prompt = f"<report>\n{context}\n</report>\n\n{question}"
        return self.harness.run(prompt)
```

The agent can use registered tools (GMS query, MoDeva diagnostics, etc.) during evaluation. Conversation state is reset between questions.

### 7.3 Factory

```python
def make_evaluator(target_type, target_model, client=None, harness=None):
    if target_type == "llm":
        if client is None:
            client = anthropic.Anthropic()
        return LLMEvaluator(client, target_model)
    if target_type == "agent":
        return AgentEvaluator(harness)  # harness required
```

---

## 8. DOE Methodology

### 8.1 Factor Catalog

The graphDOE factor catalog (`factor_catalog.yaml`) defines 50+ factors across 10 categories. The `quick_screen` group provides a fast 3-factor screening:

| Factor | Levels | Description |
|--------|--------|-------------|
| `clarity` | Clear, Ambiguous, Misleading | Question wording clarity |
| `entity_aliasing` | Exact, Synonym, Abbreviated | How entity names are presented |
| `reasoning_cue` | Direct, Step_By_Step, No_Cue | Chain-of-thought prompting |

### 8.2 Sobol Quasi-Random Design

The `DesignMatrix` generates space-filling experimental designs using scrambled Sobol sequences from `scipy.stats.qmc.Sobol`:

1. Generate $2^{\lceil \log_2 N \rceil}$ points in $[0, 1]^d$ (d = number of factors)
2. Truncate to $N$ runs
3. Map unit hypercube values to categorical levels via uniform quantile binning

The Sobol sequence provides better space-filling properties than simple random sampling or Latin Hypercube designs, ensuring balanced coverage of the factor space.

### 8.3 Question Rephrasing

The `QuestionRephraser` applies factor assignments to base questions via deterministic template transformations. The invariant:

$$\forall q,\; \forall f \in \mathcal{F}: \quad \text{ground\_truth}(\text{rephrase}(q, f)) = \text{ground\_truth}(q)$$

Examples:
- `clarity=Ambiguous`: entity names may be paraphrased or hedged
- `entity_aliasing=Abbreviated`: "MDL-001" stays as-is, full names shortened
- `reasoning_cue=Step_By_Step`: prepends "Let's think step by step."

### 8.4 Statistical Analysis

The `DOEAnalyzer` performs logistic regression with analysis of deviance. For each factor $f_j$:

1. Fit a binomial logistic model: $\log\frac{P(\text{correct})}{1 - P(\text{correct})} = \beta_0 + \sum_k \beta_{jk} \cdot \mathbb{1}[f_j = \ell_k]$
2. Compute likelihood-ratio statistic: $G^2 = -2(\mathcal{L}_{\text{null}} - \mathcal{L}_{\text{model}})$
3. Under $H_0$: $G^2 \sim \chi^2_{k-1}$
4. Report McFadden's pseudo-$R^2 = 1 - \mathcal{L}_{\text{model}} / \mathcal{L}_{\text{null}}$
5. Compute odds ratios per factor level relative to reference
6. Report failure rates with Wilson 95% confidence intervals

Per-category analysis runs separate logistic regressions for each question category, identifying localized factor effects.

---

## 9. Geometric Tracing (`tracing.py`)

### 9.1 DOETracingGMS

Records every evaluation event as geometric triples in a dedicated GMS store, following the TracingGMS pattern from the DS agent architecture.

**Bootstrap:** seed triples define the DOE evaluation ontology:

```
(doe_benchmark,   is_type,       evaluation)
(doe_question,    has_property,  factor_assignment)
(doe_question,    has_property,  score)
(doe_question,    has_property,  ground_truth)
(doe_question,    has_property,  answer)
(doe_run,         is_part_of,    doe_benchmark)
(doe_analysis,    is_type,       statistical_analysis)
(correct,         is_type,       score_outcome)
(incorrect,       is_type,       score_outcome)
(logistic_regression, is_type,   analysis_method)
```

Plus factor category seeds: `query_factor`, `persona_factor`, `reasoning_factor`, `context_factor`, `entity_factor`, `adversarial_factor`, `variability_factor`, `robustness_factor` -- each `is_type doe_factor`.

### 9.2 Traced Events

**Run start** (`trace_run_start`):

| Triple | Description |
|--------|-------------|
| `(run_id, is_type, doe_run)` | Run type declaration |
| `(run_id, document, filename)` | Source document |
| `(run_id, target_model, model_name)` | Model under test |
| `(run_id, context_mode, mode)` | Context presentation |
| `(run_id, n_design_runs, N)` | Design size |
| `(run_id, n_base_questions, N)` | Base question count |
| `(run_id, started_at, timestamp)` | Start time |
| `(run_id, tests_factor, factor_name)` | One per factor |

**Question evaluation** (`trace_question`):

| Triple | Description |
|--------|-------------|
| `(doe_qid, is_type, doe_question)` | Question type |
| `(doe_qid, base_question, base_qid)` | Base question link |
| `(doe_qid, target_model, model)` | Model evaluated |
| `(doe_qid, score_correct, true/false)` | Binary result |
| `(doe_qid, score_detail, detail)` | Scoring detail (truncated to 200 chars) |
| `(doe_qid, elapsed_seconds, N)` | Wall-clock time |
| `(doe_qid, run, run_id)` | Parent run |
| `(doe_qid, timestamp, ts)` | Evaluation time |
| `(doe_qid, factor_{name}, level)` | One per factor in assignment |

Geometric scores (`_geo_confidence`, `_tension`, `_holonomy`, `_overall_confidence`) are included in the factor assignment dict and traced as individual factor triples.

**Run end** (`trace_run_end`):

| Triple | Description |
|--------|-------------|
| `(run_id, total_questions, N)` | Total evaluated |
| `(run_id, total_correct, N)` | Total correct |
| `(run_id, accuracy, pct)` | Overall accuracy |
| `(run_id, finished_at, timestamp)` | Completion time |

**Analysis** (`trace_analysis`):

| Triple | Description |
|--------|-------------|
| `(analysis_id, is_type, doe_analysis)` | Analysis type |
| `(analysis_id, run, run_id)` | Parent run |
| `(analysis_id, factor_{name}_p_value, p)` | Per-factor p-value |
| `(analysis_id, factor_{name}_deviance, G2)` | Per-factor deviance |
| `(analysis_id, significant_factor, name)` | If $p < 0.05$ |

### 9.3 Query Methods

| Method | Purpose |
|--------|---------|
| `query_run(run_id)` | All triples where `run_id` appears as head or tail |
| `query_by_factor(factor_name, level)` | All questions with `factor_{name} = level` |
| `query_question(doe_qid)` | All triples for a specific question |
| `stats()` | Entity, triple and ENM counts |
| `save()` | Persist to disk |

Because the tracing store is itself a trained GMS, queries benefit from geometric scoring. The geodesic distance between two evaluation events measures their similarity in the evaluation space.

---

## 10. Answer Parsing and Scoring

### 10.1 Answer Parsers (`eval/scorers.py`)

The `PARSERS` dict maps answer types to parsing functions that extract structured answers from free-form LLM text:

| Key | Parser | Strategy |
|-----|--------|----------|
| `"float"` | `parse_float` | Regex `ANSWER: ([\d.eE+-]+)` or last decimal number found |
| `"int"` | `parse_int` | Regex `ANSWER: (\d+)` or last reasonable number (1-999) |
| `"bool"` | `parse_bool` | Regex `ANSWER: (true\|false\|yes\|no)` or heuristic (yes/passes -> True) |
| `"set_str"` | `parse_set_str` | Regex `ANSWER: (.+)` then comma-split, strip quotes, filter < 100 chars |
| `"str"` | `parse_str` | Regex `ANSWER: (.+)` or full text |

### 10.2 Score Function

`score_answer(parsed, ground_truth) -> ScoreResult(correct, partial_score, detail)`

| Ground truth type | Scoring | Tolerance |
|---|---|---|
| `float` | Exact: $|a - g| < 10^{-6}$; Close: $|a - g| < 0.01$ (0.5 points) | $10^{-6}$ exact, $0.01$ close |
| `int` | Exact equality | -- |
| `bool` | Exact equality | -- |
| `str` | Case-insensitive equality | -- |
| `set` | Partial credit: $|\text{intersection}| / |\text{ground\_truth}|$ | -- |
| `tuple` | Element-wise with per-type tolerances | Per-element |

---

## 11. CLI Reference

### 11.1 Command Line

```
gms-doe [markdown] [options]
```

Or:

```
python -m harness.doe [markdown] [options]
```

### 11.2 Options

| Option | Default | Description |
|--------|---------|-------------|
| `markdown` | (positional) | Path to markdown document |
| `--config` | -- | YAML config file (overrides other flags) |
| `--factor-group` | `quick_screen` | Factor group from catalog |
| `--factor-names` | -- | Specific factor names (space-separated) |
| `--n-runs` | `32` | Number of design runs |
| `--method` | `sobol` | Design method: `sobol`, `lhs`, `grid` |
| `--seed` | `42` | Random seed |
| `--rephrase-method` | `template` | `template` or `llm` |
| `--max-per-category` | `10` | Base questions per generator |
| `--generators` | -- | Generator names (space-separated) |
| `--target-type` | `llm` | `llm` or `agent` |
| `--target-model` | `claude-sonnet-4-20250514` | Model identifier |
| `--context-mode` | `full` | `full`, `oracle_section`, `noisy_section` |
| `--no-trace` | `False` | Disable tracing |
| `--tracing-dir` | `doe_tracing_store` | Tracing store directory |
| `--output` | -- | Save results JSON to this path |
| `--stores-dir` | `gms_stores` | GMS store directory |
| `--device` | -- | `cuda` or `cpu` (auto-detect if omitted) |
| `--baseline-only` | `False` | Run GMS baseline only (no LLM) |

### 11.3 Examples

```bash
# Quick baseline validation (no API calls)
gms-doe report.md --baseline-only --device cpu

# Full benchmark with Opus
gms-doe report.md \
    --target-model claude-opus-4-7 \
    --factor-group quick_screen \
    --n-runs 8 \
    --max-per-category 5 \
    --output results.json

# Specific generators only
gms-doe report.md \
    --generators plausibility tension holonomy \
    --n-runs 16

# From YAML config
gms-doe --config doe_config.yaml
```

---

## 12. Data Flow Summary

```
Markdown Document
      |
      v
[DocGMS Ingestion] ---- AnthropicBackend (LLM triple extraction)
      |                  200 training epochs on S^{m-1}
      v
GMSExpertStore
  - 508 entities, 66 relations, 889 triples (for test doc)
  - 261 ENM entries
  - Trained dual embeddings + Cayley rotors
      |
      v
[12 Question Generators]
  Classic: exact_recall, threshold, cross_reference,
           contradiction, multi_hop, counting
  Geometric: plausibility, tension, holonomy,
             link_prediction, multi_hop_transport, phase_comparison
      |
      v
[Bridge Adapter] ---- closure: gms_answer_fn(store) -> graph_answer_fn(_)
      |
      v
Base Questions (max_per_category each)
      |
      v
[DOE Design Matrix] ---- Sobol quasi-random, 3 factors x 3 levels
      |
      v
[Question Rephraser] ---- template transforms, ground truth invariant
      |
      v
DOE Questions (base x n_runs = 320 for quick_screen with max 5/cat and 8 runs)
      |
      v
[LLM/Agent Evaluator] ---- Claude API with full document context
      |
      v
Raw Answers
      |
      +---> [PARSERS] ---> [score_answer] ---> binary correct/incorrect
      |
      +---> [GMSJudge]
               |
               +---> Extract triples (metadata + text matching)
               +---> Score triples (fuzzy match + store.score_triple)
               +---> Check tensions (check_contradictions)
               +---> Check holonomy (check_holonomy)
               +---> Compute overall (weighted: 0.3 exact + 0.3 geo + 0.2 tension + 0.2 holonomy)
               |
               v
            GeometricVerdict
               |
      +--------+--------+
      |                 |
      v                 v
[DOETracingGMS]    [DOEBenchmarkResult]
  16,791 triples        |
  2,768 entities        v
                   [DOEAnalyzer]
                     Logistic regression
                     Analysis of deviance
                     Odds ratios
                     Wilson CIs
```

---

## 13. Benchmark Results Reference

### 13.1 Test Document

Model Risk Assessment Report for Meridian Financial Technologies, Inc. Eight sections covering 12 ML models across credit decisioning, fraud detection, AML, market risk and customer analytics. Contains performance metrics (AUC, Gini, KS), robustness tables (noise perturbation 1-20%), fairness metrics (Adverse Impact Ratio), data quality indicators (PSI, missing rates), risk findings, model dependencies with financial impacts ($12.3M-$89.1M) and remediation tracking.

**GMS store:** 508 entities, 66 relations, 889 triples, 261 ENM entries.

### 13.2 Results: Sonnet 4 vs Opus 4.6

Configuration: `quick_screen` factors, 8 Sobol runs, 5 base questions per category, `context_mode=full`.

**Per-category accuracy (40 questions each):**

| Category | Sonnet 4 | Opus 4.6 | Tier |
|---|---|---|---|
| cross_reference | 70% | 62% | Accessible |
| counting | 52% | 60% | Accessible |
| holonomy | 2% | 12% | Marginal |
| plausibility | 10% | 5% | Marginal |
| tension | 10% | 10% | Marginal |
| link_prediction | 0% | 0% | Unreachable |
| multi_hop | 0% | 0% | Unreachable |
| multi_hop_transport | 0% | 0% | Unreachable |
| **Overall** | **18.1%** | **18.8%** | |

**Geometric scores (mean across 320 questions):**

| Metric | Sonnet 4 | Opus 4.6 |
|---|---|---|
| Geodesic confidence | 0.257 | 0.259 |
| Tension score | 1.401 | 1.414 |
| Holonomy score | 0.605 | 0.580 |
| Overall confidence | 0.312 | 0.316 |

**DOE factor significance:** No factor reached $p < 0.05$ at the aggregate level. Localized effects: `reasoning_cue` affects holonomy ($R^2 = 0.21$) and tension ($R^2 = 0.33$) for Opus; `clarity` affects tension ($R^2 = 0.33$) for Sonnet.

**Interpretation:** The mean tension score ($\approx 1.4 \approx \sqrt{2}$) indicates incorrect answers are predominantly *irrelevant* (orthogonal in logical space) rather than actively contradictory. The holonomy scores ($\approx 0.6$, above the 0.5 consistency threshold) confirm systematic multi-hop reasoning failures. Question category dominates performance far more than presentation factors.
