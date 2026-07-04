# knowlytix.harness.suite — User Guide

Catalog-driven construction of **test suites** and **training corpora** over GMS
ground truth. You pick *what* is asked (a base taxonomy whose answers are
provably correct), enrich it with *how* it is presented (a DoE layer of
presentation factors), and consume the composed scenarios two ways: as an SFT
corpus or as an evaluation run with logistic factor attribution.

For the conceptual overview and package layout see [README.md](README.md); for
function-level detail see [API_REFERENCE.md](API_REFERENCE.md).

Convention in this guide:

```python
from knowlytix.harness import suite as S
```

## What this module is for

Testing and training an agentic system needs many *variants* of a question whose
**answer does not change**. A complaint about an overdraft fee is still an
overdraft complaint whether it is terse or verbose, polite or hostile, written by
a clerk or a regulator. The right answer is fixed by the document/graph; only the
surface varies.

That is the **ground-truth-invariant enrichment principle** the whole module is
built on. The base catalog defines a question type *and its true answer* (drawn
from GMS primitives, labeled seed cases, or you). The factor catalog defines
presentation factors that vary the query without touching the answer — every
factor is `gt_invariant` and is filtered by `applies_to` to the bases it
actually enriches. Composition crosses the two; evaluation then attributes any
*failure* back to the presentation factors that caused it, because the answer was
never supposed to move.

The five short sections below follow the call flow; §6 stitches them together.

---

## 1. Loading and exploring a catalog

**When.** Always first — everything resolves against a `Catalog`.

```python
from knowlytix.harness import suite as S

cat = S.Catalog.load()          # the shipped catalogs/ directory
print(cat.summary())            # "18 base categories in 5 families, 40 factors in 8 groups, 7 profiles"

print(sorted(cat.profiles))     # ['everything', 'integrity_audit', 'jasa_reproduction', ...]
print(cat.families["retrieval_recall"])   # base names in that family
print(cat.factor_groups["quick_screen"])  # ['clarity', 'entity_aliasing', 'reasoning_cue']
```

**Inspecting an entry.** Each base and factor is a frozen dataclass:

```python
b = cat.bases["exact_recall"]
print(b.family, b.answer_type, b.generator)   # retrieval_recall str <GeneratorClass or None>

f = cat.factors["clarity"]
print(f.levels, f.gt_invariant)               # ['terse', 'verbose', ...] True
print(f.applies_to(b))                         # True if this factor enriches this base
```

**Common variations.** Load a custom catalog set with
`S.Catalog.load("/path/to/catalogs")`. The directory needs `base_catalog.yaml`
and `factor_catalog.yaml`; `profiles.yaml` is optional.

**Edge cases.** A missing base/factor file raises; a missing `profiles.yaml`
just yields an empty `profiles` dict. Validate a custom catalog before trusting
it: `python -m knowlytix.harness.suite._validate` enforces the
base↔enrichment contract (every factor `gt_invariant: true`, `applies_to`
references real families/answer-types, snake_case names, no cross-catalog name
collisions).

---

## 2. Sourcing base items

**When.** After resolving a suite (§3) — but the choice of source is independent
of the suite, so it is worth understanding first. All three sources yield
`list[QAItem]` from `.items(suite=None)` and are interchangeable downstream.

| source | where ground truth comes from | needs a suite? |
| ------ | ----------------------------- | -------------- |
| `UserBaseSource` | you supply `(query, answer)` directly; no GMS | no |
| `SeedCaseSource` | hand-labeled cases with per-component truth | no |
| `CatalogBaseSource` | mined from a GMS store via benchmark generators | **yes** |

**UserBaseSource** — bring your own pairs:

```python
src = S.UserBaseSource([
    {"query": "What is the overdraft fee?", "answer": "$35", "answer_type": "str"},
    {"query": "Which division has the highest revenue?", "answer": "Consumer"},
])
items = src.items()             # answer IS ground truth; queries are NOT expanded
```

**SeedCaseSource** — labeled cases carrying per-tool ground truth (the capstone's
complaint cases). The `component_keys` become `QAItem.components`, which unlocks
component scoring and `weak_link` in evaluation:

```python
cases = [
    {"id": "c001", "message": "You charged me $35 twice!",
     "expected_classification": "complaint", "expected_product": "checking",
     "expected_escalation": True},
]
src = S.SeedCaseSource(cases)               # defaults: query_key="message", answer_key="expected_classification"
items = src.items()
print(items[0].components)                  # {'expected_classification': ..., 'expected_product': ..., ...}
```

Override the keys when your schema differs:
`S.SeedCaseSource(cases, query_key="text", answer_key="label",
component_keys=("label", "expected_route"))`.

**CatalogBaseSource** — mine a GMS store. Ground truth is graph-derived (provably
correct). It **requires the resolved suite** so it knows which generators to run:

```python
src = S.CatalogBaseSource(store, max_per_category=10, seed=42)
items = src.items(suite)        # suite from §3; ValueError if omitted
```

**Edge cases.** `CatalogBaseSource` silently skips any selected base whose
`BaseSpec.generator` is `None` or unresolvable (status `build`/`partial`) — so an
empty `items` list usually means the chosen bases have no generators yet, not a
store problem. `UserBaseSource` and `SeedCaseSource` ignore the `suite` argument.

---

## 3. Composing scenarios

**When.** Once you have a `ResolvedSuite` and base items, cross them into
`Scenario`s.

Resolve a suite from a named profile, or build one explicitly:

```python
suite = S.resolve_profile(cat, "numeric_screen")     # bases + factors + mode, from profiles.yaml
print(suite.summary())                                # "suite[numeric_screen]: 5 bases x 11 factors, mode=embedded ..."
print(suite.dropped_factors)                          # factors filtered out by applies_to

# or, explicitly:
suite = S.resolve(cat, bases=["exact_recall", "ranking"],
                  factors=["query_core", "robustness"], mode="embedded")
```

Then compose:

```python
items = S.UserBaseSource(pairs).items()               # any source from §2
scns  = S.compose(suite, items, n_runs=120)
```

**cross vs embedded** (the suite's `mode`):

- `"cross"` (Option 1) — one design over the presentation factors only; *every*
  base item is crossed with *every* design row. `#scenarios = #items × n_runs`.
  Use it to stress each base exhaustively across the same design grid.
- `"embedded"` (Option 2) — the base question is itself a factor in the design.
  `#scenarios = n_runs`. This is the JASA Table 1 construction and the default
  for most profiles.

**blocking vs joint base** (embedded mode only, via `balance_base`):

- `balance_base=True` (default) — presentation factors are space-filled by the
  design; the base id is assigned as a **balanced blocking factor**, so every
  base appears ~`n_runs / k` times. Robust at small `n` — a joint optimizer would
  otherwise starve a high-cardinality `base_question`.
- `balance_base=False` — `base_question` joins the design matrix as a joint
  factor (pure Option 2 / JASA Table 1). Best at large `n`.

**design_fn choice:**

```python
from knowlytix.harness.suite.compose import graphdoe_design

scns = S.compose(suite, items, n_runs=120, design_fn=S.simple_design)    # default: deterministic, no deps
scns = S.compose(suite, items, n_runs=120, design_fn=graphdoe_design)    # Sobol+refine space-filling
```

`simple_design` is dependency-light and deterministic (good for tests/offline);
`graphdoe_design` uses the `graphdoe` Sobol+refine generator for real
space-filling coverage. To pass a non-default `method`, wrap it — `compose` calls
`design_fn(level_specs, n_runs, seed)` positionally:

```python
import functools
sobol = functools.partial(graphdoe_design, method="sobol")
scns  = S.compose(suite, items, design_fn=sobol)
```

**Common variations.** Supply application-specific level vocabulary without
editing the domain-neutral catalog:

```python
scns = S.compose(suite, items,
                 level_overrides={"entity_aliasing": ["canonical", "alias"]})
```

**Edge cases.** Empty `base_items` → `[]`. A suite with no factors composes
base-only scenarios (one design row of `{}` per run). `compose` does not mutate
the suite or the items.

---

## 4. Emitting an SFT corpus

**When.** You want fine-tuning data, not an evaluation — the Chapter 15 consumer.
Two emitters mirror the two shipped capstone pipelines.

**Classifier SFT** — label-preserving `(message, label)` pairs. Enrichment varies
the surface; the label is invariant:

```python
recs = S.emit_classifier_sft(scns)
# [{"message": ..., "label": ..., "_seed": ..., "_factors": {...}}, ...]
S.to_jsonl(recs, "out/classifier_sft.jsonl")
```

Plug in a real rewriter (a Qwen materialization in production) via
`materialize_fn`, and a custom label via `label_fn`:

```python
recs = S.emit_classifier_sft(scns, materialize_fn=qwen_rewrite, label_fn=lambda s: s.base.answer)
```

**Draft SFT** — grounded `(user, assistant)` pairs under the **MeMo strictness
contract**. A pair is kept only if the response cites the policy keyword, holds
the byte-exact required numbers, and uses no forbidden phrase. `emit_draft_sft`
**returns `(kept, dropped)`**:

```python
goldens = {
    "overdraft_policy": {
        "citation_keywords": ["overdraft"],
        "required_numbers":  ["35"],
        "forbidden_phrases": ["we will refund", "guaranteed"],
        "one_line_summary":  "Overdraft fee is $35 per item.",
    },
}
kept, dropped = S.emit_draft_sft(scns, response_fn=qwen_draft, goldens=goldens)
print(len(kept), len(dropped))      # dropped = pairs that failed the contract
S.to_jsonl(kept, "out/draft_sft.jsonl")
```

The contract itself is reusable standalone:

```python
S.passes_contract("Per the overdraft policy, the $35 fee applies.", goldens["overdraft_policy"])  # True
```

**Common variations.** `issues=` / `summaries=` keyed by policy id override the
`Issue:` / `Policy:` lines of the prompt template (`draft_prompt`); `source=`
stamps the corpus tag (default `"gmstest"`); `validate=False` keeps every pair
(no contract filtering).

**Edge cases.** Numbers in `required_numbers` are matched byte-exact against the
raw response (not lowercased) — `"$35"` and `"35.00"` are different surfaces.
Validation only runs for a scenario whose `policy_id` is present in `goldens`;
scenarios with no matching golden are always kept.

---

## 5. Evaluating through a SUT and attributing factors

**When.** You want to *test* a system and learn which presentation factors break
it — the Chapter 16 consumer.

A SUT is any callable `sut(query, context="") -> answer | SUTResult`. Return a
bare value to score outcome only; return a `SUTResult` to unlock trajectory and
per-component scoring:

```python
def my_sut(query, context=""):
    return S.evaluate.SUTResult(
        answer="...",
        components={"classification": "complaint"},
        trajectory=[{"tool": "classify_complaint", "ok": True}, {"tool": "draft_response", "ok": True}],
        status="ok",
    )
```

The four-step flow — **run → summary → weak_link → attribute**:

```python
order = ["classify_complaint", "extract_facts", "search_policy", "flag_regulatory", "draft_response"]

rows  = S.evaluate.run(scns, my_sut, workflow_order=order)
print(S.evaluate.summary(rows))                 # {"n": ..., "accuracy": ..., "workflow_adherence": ...}
print(S.evaluate.weak_link(rows, order))        # {tool: failures} — first tool to err per failure
table = S.evaluate.attribute(rows, suite.factor_names)   # logistic + Benjamini-Hochberg
for r in table:
    print(r)                                    # per-factor coefficient, p-value, FDR flag
```

`run` emits one row per scenario with `correct` (`0/1`), `status`, an `f_<factor>`
column per presentation factor, and — when the SUT returns a trajectory —
`tool_sequence`, `n_steps`, `tool_failures`, and `workflow_adherent`. Each
labeled component yields a `c_<component>` column. `attribute` expects the
`f_<factor>` columns and `suite.factor_names`; it renames them and runs them
through `graphdoe.DOEAnalyzer` — so the factors that *cause* failures fall out
with corrected p-values.

**Common variations.** Score groundedness of a textual draft with a
`judge_fn=draft -> (tier, score)` (e.g. `store.score_triple`), which adds
`draft_tier` / `draft_score` columns. Attribute a different binary outcome with
`metric="workflow_adherent"` instead of the default `"correct"`. Tighten or
loosen the FDR with `alpha=`.

**Edge cases.** `weak_link` only attributes a failure when the failing row has a
`c_<tool>` component column equal to `0`; an outcome-only SUT (bare-value return,
no `components`) yields no weak-link counts. `workflow_adherent` appears only
when both a trajectory and a `workflow_order` are supplied.

---

## 6. End to end

Stitching the full call flow — catalog → suite → store-mined bases → space-filling
composition → evaluation → attribution:

```python
from knowlytix.harness import suite as S
from knowlytix.harness.suite.compose import graphdoe_design

# 1. Load the catalog and resolve a profile.
cat   = S.Catalog.load()
suite = S.resolve_profile(cat, "numeric_screen")      # bases + factors + embedded mode
print(suite.summary())

# 2. Mine base questions from a GMS store (ground truth is graph-derived).
items = S.CatalogBaseSource(store, max_per_category=12).items(suite)

# 3. Enrich with a Sobol+refine presentation design (answers held invariant).
scns  = S.compose(suite, items, n_runs=200, design_fn=graphdoe_design)

# 4. Run the system under test and score it.
rows  = S.evaluate.run(scns, my_sut, workflow_order=workflow)
print(S.evaluate.summary(rows))                       # overall accuracy + workflow adherence

# 5. Attribute failures to the presentation factors that caused them.
table = S.evaluate.attribute(rows, suite.factor_names)
for r in table:
    print(r)                                          # which factor moved a fixed-answer question
```

Swap step 2 for `S.SeedCaseSource(cases).items()` or `S.UserBaseSource(pairs).items()`
to evaluate without a GMS store, and swap step 4–5 for the §4 emitters to produce
a training corpus from the same composed scenarios instead. The catalog stays
domain-neutral throughout; supply application level vocabulary at composition time
via `compose(..., level_overrides=...)`.
