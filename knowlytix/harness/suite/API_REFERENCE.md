# knowlytix.harness.suite — API Reference

Public surface of `knowlytix.harness.suite`. For the conceptual overview see
[README.md](README.md); for prose and recipes see [USER_GUIDE.md](USER_GUIDE.md).
`License-Identifier: Apache-2.0`.

The package re-exports its whole public surface from the top level:

```python
__all__ = [
    "BaseSpec", "FactorSpec", "Profile", "Catalog",
    "ResolvedSuite", "resolve", "resolve_profile", "applicable_factors",
    "QAItem", "BaseSource", "UserBaseSource", "SeedCaseSource", "CatalogBaseSource",
    "Scenario", "compose", "simple_design", "graphdoe_design",
    "emit_classifier_sft", "emit_draft_sft", "draft_prompt", "passes_contract", "to_jsonl",
    "evaluate",
]
```

Convention used throughout these docs:

```python
from knowlytix.harness import suite as S
```

So `S.Catalog`, `S.resolve_profile`, `S.compose`, `S.evaluate.run`, and so on.
`evaluate` is a re-exported *module* (`S.evaluate`), not a function.

| symbol | role |
| ------ | ---- |
| `Catalog`, `BaseSpec`, `FactorSpec`, `Profile` | the three catalogs and their entries |
| `resolve`, `resolve_profile`, `applicable_factors`, `ResolvedSuite` | expand a selection into a concrete suite |
| `QAItem`, `BaseSource`, `UserBaseSource`, `SeedCaseSource`, `CatalogBaseSource` | where base (query, answer) pairs come from |
| `Scenario`, `compose`, `simple_design`, `graphdoe_design` | enrich bases with a presentation design |
| `emit_classifier_sft`, `emit_draft_sft`, `draft_prompt`, `passes_contract`, `to_jsonl` | SFT / fine-tuning corpora |
| `evaluate` (`run`, `summary`, `weak_link`, `attribute`, `SUTResult`) | run a SUT, score, attribute failures |

---

## Catalogs — `catalogs`

The catalogs live in `suite/catalogs/` as three YAML files: `base_catalog.yaml`
(WHAT is asked + WHAT is true), `factor_catalog.yaml` (HOW it is presented), and
`profiles.yaml` (named base+factor+mode bundles).

### `BaseSpec` (dataclass, frozen)

One base question type — defines content and its ground truth.

```python
@dataclass(frozen=True)
class BaseSpec:
    name: str
    family: str
    category: str
    description: str
    requires: list[str]
    generator: str | None
    answer_type: str
    ground_truth: str
    multiplicity: str
    applicability: list[str]
    status: str
    source: list[str]
```

`generator` is the `knowlytix.benchmark.generators` class name driving
`CatalogBaseSource`, or `None` when the category has no generator yet (`status`
`build`/`partial`). `family` is one of the five base families
(`retrieval_recall`, `comparison_ranking`, `reasoning_composition`,
`consistency_integrity`, `behavioral_operational`).

### `FactorSpec` (dataclass, frozen)

One enrichment factor — presentation only, ground-truth invariant.

```python
@dataclass(frozen=True)
class FactorSpec:
    name: str
    section: str
    cardinality: int
    levels: list[str]
    description: str
    gt_invariant: bool
    applies_families: list[str]      # ["*"] means all
    applies_answer_types: list[str]  # ["*"] means all

    def applies_to(self, base: "BaseSpec") -> bool: ...
```

`applies_to(base)` is the enrichment-enriches-the-base predicate: `True` when
`base.family` is in `applies_families` (or `applies_families` is `["*"]`) **and**
`base.answer_type` is in `applies_answer_types` (or `["*"]`). `levels` is loaded
from the YAML `categories` key; `gt_invariant` is enforced `True` by the catalog
validator.

### `Profile` (dataclass, frozen)

A pick-and-choose bundle: base selection + factor selection + mode.

```python
@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    bases: list[str]
    factors: list[str]
    mode: str  # "cross" | "embedded"
```

`bases` and `factors` are selection tokens (a name, a family/group name, or
`"*"`); they are expanded at resolution time. `mode` defaults to `"embedded"`
when omitted in YAML.

### `Catalog` (dataclass)

```python
@dataclass
class Catalog:
    bases: dict[str, BaseSpec]
    factors: dict[str, FactorSpec]
    factor_groups: dict[str, list[str]]
    profiles: dict[str, Profile]
    families: dict[str, list[str]] = {}
```

`families` maps each family name to the names of its bases; `factor_groups` maps
each group name (e.g. `quick_screen`, `query_core`) to its member factor names.

#### `Catalog.load(catalogs_dir: str | Path | None = None) -> Catalog`

Load the three YAML files. `catalogs_dir` defaults to the package's own
`catalogs/` directory, so `S.Catalog.load()` loads the shipped catalog; pass a
path to load a custom catalog set. `profiles.yaml` is optional (missing → no
profiles). Raises on missing `base_catalog.yaml` / `factor_catalog.yaml`.

#### `Catalog.summary() -> str`

One-line census, e.g. `"18 base categories in 5 families, 40 factors in 8 groups,
7 profiles"`.

---

## Resolution — `resolve`

Expand a profile or explicit selection into a concrete `ResolvedSuite`.

Expansion rules (both `bases` and `factors`): `"*"` → all; a family name (bases)
or factor-group name (factors) → its members; otherwise a literal entry name. An
unknown token raises `KeyError`. After expansion, factors are filtered by
`applies_to` against the selected bases (unless `filter_factors=False`); dropped
factors are recorded on the suite for transparency.

### `ResolvedSuite` (dataclass)

```python
@dataclass
class ResolvedSuite:
    bases: list[BaseSpec]
    factors: list[FactorSpec]
    mode: str
    dropped_factors: list[tuple[str, str]] = []
    profile: str | None = None

    @property
    def base_names(self) -> list[str]: ...      # [b.name for b in bases]

    @property
    def factor_names(self) -> list[str]: ...     # [f.name for f in factors]

    def summary(self) -> str: ...
```

`dropped_factors` is a list of `(factor_name, reason)` pairs for factors that
`applies_to` matched no selected base. `factor_names` is exactly what you pass to
`evaluate.attribute`. `summary()` →
`"suite[numeric_screen]: 5 bases x 11 factors, mode=embedded (2 factors dropped
by applies_to)"`.

### `resolve(catalog, bases, factors, mode="embedded", filter_factors=True) -> ResolvedSuite`

```python
def resolve(
    catalog: Catalog,
    bases: list[str],
    factors: list[str],
    mode: str = "embedded",
    filter_factors: bool = True,
) -> ResolvedSuite: ...
```

`mode` must be `"cross"` or `"embedded"` (else `ValueError`). `bases` / `factors`
are selection-token lists. The returned suite's `profile` is `None`.

### `resolve_profile(catalog, name, filter_factors=True) -> ResolvedSuite`

```python
def resolve_profile(catalog: Catalog, name: str, filter_factors: bool = True) -> ResolvedSuite: ...
```

Resolve a named profile from `catalog.profiles` (unknown name → `KeyError`). Uses
the profile's `bases`, `factors`, and `mode`; stamps `suite.profile = name`.

### `applicable_factors(factors, bases) -> tuple[list[FactorSpec], list[tuple[str, str]]]`

```python
def applicable_factors(
    factors: list[FactorSpec], bases: list[BaseSpec]
) -> tuple[list[FactorSpec], list[tuple[str, str]]]: ...
```

Split `factors` into `(kept, dropped)`: a factor is kept when it `applies_to` at
least one base in `bases`. `dropped` is a list of `(name, reason)` pairs. Called
internally by `resolve` when `filter_factors=True`; exposed for inspection.

---

## Base sources — `sources`

Three sources, all yielding `list[QAItem]` from `.items(suite=None)`. The
enrichment layer treats all three identically.

### `QAItem` (dataclass)

```python
@dataclass
class QAItem:
    qid: str
    query: str
    answer: Any
    answer_type: str = "str"
    base: str | None = None                  # base-category name, or None (user)
    components: dict[str, Any] = {}           # per-tool ground truth
    metadata: dict[str, Any] = {}
```

`components` carries per-component ground truth (the per-tool labels used by
`evaluate` for component scoring and `weak_link`); `base` is `None` for
user-supplied items.

### `BaseSource` (Protocol, runtime-checkable)

```python
@runtime_checkable
class BaseSource(Protocol):
    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]: ...
```

Any object with a conforming `items` method is a base source.

### `UserBaseSource(pairs)`

```python
class UserBaseSource:
    def __init__(self, pairs: list[dict[str, Any]]): ...
    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]: ...
```

Bring-your-own `(query, answer)` pairs; no GMS, no generation. Each dict in
`pairs` uses keys `query`, `answer`, optional `answer_type` (default `"str"`),
optional `qid` (default `user-NNNN`), and optional `metadata`. The user's
`answer` is the ground truth. `suite` is ignored.

### `SeedCaseSource(cases, query_key="message", answer_key="expected_classification", id_key="id", component_keys=(...))`

```python
class SeedCaseSource:
    def __init__(
        self,
        cases: list[dict[str, Any]],
        query_key: str = "message",
        answer_key: str = "expected_classification",
        id_key: str = "id",
        component_keys: tuple[str, ...] = (
            "expected_classification", "expected_product", "expected_issue",
            "expected_policy", "expected_regulation",
            "expected_escalation", "expected_escalate",
        ),
    ): ...
    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]: ...
```

Hand-labeled cases carrying per-component ground truth (the capstone's labeled
complaint cases). For each case: `query` ← `case[query_key]`, `answer` ←
`case.get(answer_key)`, `qid` ← `str(case.get(id_key, "case-NNNN"))`. Every
`component_keys` key present in the case is copied into `QAItem.components`;
remaining keys (minus `query_key`/`id_key`/components) go to `metadata`.
`answer_type` is fixed to `"decision"` and `base` to `"seed_case"`. `suite` is
ignored.

### `CatalogBaseSource(store, max_per_category=10, seed=42)`

```python
class CatalogBaseSource:
    def __init__(self, store: Any, max_per_category: int = 10, seed: int = 42): ...
    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]: ...
```

Mine a GMS store using the selected base categories' generators. **`suite` is
required** (`items(suite=None)` → `ValueError`): it names which bases to
generate. For each base whose `BaseSpec.generator` resolves to a
`knowlytix.benchmark.generators` class, the generator is instantiated and run
against `store.doc_graph` (falling back to `store`); candidates are sampled to
`max_per_category` (using `gen.sample(..., seed=seed)` when available, else a
head slice). Each `GeneratedQuestion`'s `natural_language` becomes the query and
its `ground_truth` the answer — graph-derived and provably correct. Bases with no
generator (or an unresolvable name) are skipped.

---

## Composition — `compose`

Enrich base items with a presentation design, holding ground truth invariant.

A *design function* maps `(level_specs, n_runs, seed)` → list of rows, where each
row is `{factor_name: level}` and `level_specs` is a list of `(name, levels)`
pairs:

```python
LevelSpec = list[tuple[str, list[str]]]
DesignFn = Callable[[LevelSpec, int, int], list[dict[str, str]]]
```

### `Scenario` (dataclass)

```python
@dataclass
class Scenario:
    sid: str
    base: QAItem
    factor_levels: dict[str, str]
    mode: str
    metadata: dict[str, Any] = {}
```

`sid` is `"{qid}~{run:04d}"`; `factor_levels` is one assignment of the
presentation factors; `mode` is `"cross"` or `"embedded"`.

### `simple_design(level_specs, n_runs, seed=42) -> list[dict[str, str]]`

```python
def simple_design(level_specs: LevelSpec, n_runs: int, seed: int = 42) -> list[dict[str, str]]: ...
```

Deterministic per-factor random sampling (one independent `rng.choice` per
factor per run). No external dependencies — the default `design_fn`, good for
tests and offline use. Returns `[{}]` when `level_specs` is empty.

### `graphdoe_design(level_specs, n_runs, seed=42, method="sobol+refine") -> list[dict[str, str]]`

```python
def graphdoe_design(level_specs: LevelSpec, n_runs: int, seed: int = 42,
                    method: str = "sobol+refine") -> list[dict[str, str]]: ...
```

Space-filling design via `knowlytix.harness.graphdoe.DesignMatrix` (each factor
emitted as `{"name", "type": "categorical", "categories"}`). `method` is the
graphdoe design method (`"sobol+refine"` default). Returns `[{}]` when
`level_specs` is empty. Note: `compose` calls `design_fn(level_specs, n_runs,
seed)` with three positional args, so to override `method` pass a
`functools.partial` as `design_fn`.

### `compose(suite, base_items, n_runs=64, seed=42, design_fn=simple_design, balance_base=True, level_overrides=None) -> list[Scenario]`

```python
def compose(
    suite: ResolvedSuite,
    base_items: list[QAItem],
    n_runs: int = 64,
    seed: int = 42,
    design_fn: DesignFn = simple_design,
    balance_base: bool = True,
    level_overrides: dict[str, list[str]] | None = None,
) -> list[Scenario]: ...
```

Returns `[]` for empty `base_items`. The presentation level-specs are
`[(f.name, level_overrides.get(f.name, f.levels)) for f in suite.factors]`.

**`mode="cross"` (Option 1 — base × design).** One design over the presentation
factors only; every base item is crossed with every design row. `balance_base`
is ignored. `#scenarios = #base_items × n_runs`.

**`mode="embedded"` (Option 2 — base is a factor).**

- `balance_base=True` (default): the presentation factors are space-filled by
  `design_fn`, and the base question is assigned as a **balanced blocking
  factor** — each base id appears ~`n_runs / k` times, seed-shuffled — so every
  base is exercised equally. Robust at small `n`. `#scenarios = n_runs`.
- `balance_base=False`: `base_question` is added to the design matrix as a
  categorical factor whose levels are the base ids, sampled **jointly** with the
  presentation factors (pure Option 2 / JASA Table 1). Best at large `n`.
  `#scenarios = n_runs`.

`level_overrides` lets an application supply its own level vocabulary for a
catalog factor (e.g. `{"entity_aliasing": ["canonical", "alias"]}`) without
editing the domain-neutral catalog.

---

## Emission — `emit`

SFT / fine-tuning emitters (the Chapter 15 consumer). Text generation is
pluggable via `materialize_fn` / `response_fn`:

```python
MaterializeFn = Callable[[Scenario], str]
ResponseFn = Callable[[Scenario], str]
```

The defaults are dependency-light: `materialize_fn` returns the base query and
`label_fn` returns `scenario.base.components.get("expected_classification",
scenario.base.answer)`.

### `emit_classifier_sft(scenarios, materialize_fn=_default_materialize, label_fn=_default_label) -> list[dict[str, Any]]`

```python
def emit_classifier_sft(
    scenarios: list[Scenario],
    materialize_fn: MaterializeFn = _default_materialize,
    label_fn: Callable[[Scenario], Any] = _default_label,
) -> list[dict[str, Any]]: ...
```

Label-preserving `(message, label)` records for classifier SFT. Each record is
`{"message": materialize_fn(s), "label": label_fn(s), "_seed": <query or qid>,
"_factors": dict(s.factor_levels)}`. `_seed` is the base query when
`base.base is None` (user source), else the `qid`. The label is invariant under
the presentation factors — enrichment varies the surface only.

### `draft_prompt(complaint, issue, summary) -> str`

```python
def draft_prompt(complaint: str, issue: str, summary: str) -> str: ...
```

The exact inference-time template:
`"Complaint: {complaint}\nIssue: {issue}\nPolicy: {summary}\nResponse:"`.

### `passes_contract(response, golden) -> bool`

```python
def passes_contract(response: str, golden: dict[str, Any]) -> bool: ...
```

The MeMo strictness contract. Returns `True` only when `response`:

- contains every keyword in `golden["citation_keywords"]` (case-insensitive;
  `golden["required_keywords"]` is accepted as an alias),
- contains every value in `golden["required_numbers"]` byte-exact (substring of
  the raw `response`, not lowercased — numbers must match exactly),
- contains **no** phrase in `golden["forbidden_phrases"]` (case-insensitive).

Golden schema (matches the capstone `draft_response_goldens.json`):

```python
golden = {
    "citation_keywords": ["overdraft"],   # or "required_keywords"
    "required_numbers":  ["35"],
    "forbidden_phrases": ["we will refund", "guaranteed"],
    "one_line_summary":  "...",            # used by emit_draft_sft's Policy line
}
```

### `emit_draft_sft(scenarios, response_fn, issues=None, summaries=None, goldens=None, validate=True, source="gmstest") -> tuple[list[dict], list[dict]]`

```python
def emit_draft_sft(
    scenarios: list[Scenario],
    response_fn: ResponseFn,
    issues: dict[str, str] | None = None,
    summaries: dict[str, str] | None = None,
    goldens: dict[str, dict] | None = None,
    validate: bool = True,
    source: str = "gmstest",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...
```

Grounded `(user, assistant)` pairs. **Returns `(kept, dropped)`.** Each record is
`{"user": draft_prompt(complaint, issue, summary), "assistant": response_fn(s),
"policy_id": pid, "issue": issue, "source": source}`. The policy id `pid` is
`s.base.metadata["policy_id"]` (else `metadata["category"]`, else `""`); `issue`
is `issues.get(pid, s.base.metadata.get("issue", ""))`; `summary` is
`summaries.get(pid)` else `goldens[pid]["one_line_summary"]`. When
`validate=True` and `pid in goldens`, a record failing `passes_contract` goes to
`dropped` instead of `kept`; otherwise everything is kept.

### `to_jsonl(records, path) -> int`

```python
def to_jsonl(records: list[dict[str, Any]], path: str | Path) -> int: ...
```

Write one JSON object per line (creating parent directories). Returns the record
count.

---

## Evaluation — `evaluate`

Re-exported as a module: `S.evaluate.run`, `S.evaluate.summary`,
`S.evaluate.weak_link`, `S.evaluate.attribute`, `S.evaluate.SUTResult`. Runs
composed scenarios through a System Under Test and scores at three levels
(outcome / trajectory / process).

A SUT is any callable `sut(query, context="") -> answer | SUTResult`. Returning a
bare value scores outcome only; returning a `SUTResult` unlocks trajectory and
per-component scoring.

### `SUTResult` (dataclass)

```python
@dataclass
class SUTResult:
    answer: Any = None
    components: dict[str, Any] = {}                  # tool -> predicted
    trajectory: list[dict[str, Any]] = []            # ordered steps
    status: str = "ok"                               # ok|escalated|failed
    escalation_trigger: str | None = None
```

Each trajectory step is a dict with at least `"tool"` and optional `"ok"`
(`False` marks a tool failure). A judge function (for groundedness) has type
`JudgeFn = Callable[[str], tuple[str, float]]` (`draft -> (tier, score)`).

### `run(scenarios, sut, judge_fn=None, workflow_order=None) -> list[dict[str, Any]]`

```python
def run(
    scenarios: list[Scenario],
    sut: Callable[..., Any],
    judge_fn: JudgeFn | None = None,
    workflow_order: list[str] | None = None,
) -> list[dict[str, Any]]: ...
```

Execute every scenario (calling `sut(scn.base.query,
context=scn.base.metadata.get("context", ""))`) and return one result row per
scenario. Each row carries:

- `sid`, `seed` (base qid), `base` (base-category name);
- `f_<factor>` for each factor level (the prefix `evaluate.attribute` expects);
- `correct` (`0/1` outcome) and `status`.

Outcome is correct when, for a labeled seed case, the predicted
`classification` matches `expected_classification` **and** escalation matches
(`expected_escalation`/`expected_escalate`); otherwise the answer equals the base
ground truth. When the SUT returns a trajectory the row also gets
`tool_sequence`, `n_steps`, `tool_failures`, and — if `workflow_order` is given —
`workflow_adherent` (`0/1`). For each `expected_<x>` component present in both
truth and `res.components`, the row gets `c_<x>` (`0/1`). With a `judge_fn` and a
non-empty string answer, the row gets `draft_tier` and `draft_score`.

### `summary(rows) -> dict[str, Any]`

```python
def summary(rows: list[dict[str, Any]]) -> dict[str, Any]: ...
```

`{"n", "accuracy"}`, plus `"workflow_adherence"` when any row carries
`workflow_adherent`.

### `weak_link(rows, workflow_order) -> dict[str, int]`

```python
def weak_link(rows: list[dict[str, Any]], workflow_order: list[str]) -> dict[str, int]: ...
```

For each failing row (`correct == 0`), attribute the failure to the **first** tool
in `workflow_order` whose component score `c_<tool>` is `0`. Returns
`{tool: failure_count}` — which step in the workflow most often breaks first.

### `attribute(rows, factor_names, metric="correct", alpha=0.05) -> list[dict[str, Any]]`

```python
def attribute(
    rows: list[dict[str, Any]],
    factor_names: list[str],
    metric: str = "correct",
    alpha: float = 0.05,
) -> list[dict[str, Any]]: ...
```

Logistic factor attribution via `knowlytix.harness.graphdoe.DOEAnalyzer`. The
`f_<factor>` columns are renamed back to bare factor names, fed to
`DOEAnalyzer.from_dataframe(..., factors=...)`, scored with `run_logistic(metric=
metric)`, and Benjamini-Hochberg corrected at `alpha`. Pass
`suite.factor_names`. Returns a list of per-factor records (coefficient,
p-value, FDR flag). `metric` selects the binary outcome column to model
(`"correct"` default; any `0/1` row column such as `"workflow_adherent"` works).

---

The catalogs are domain-neutral; applications supply level vocabulary via
`compose(..., level_overrides=...)`. To check catalog consistency:
`python -m knowlytix.harness.suite._validate`.
