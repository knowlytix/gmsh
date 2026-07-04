# knowlytix.harness.suite

Catalog-driven construction of **test suites** and **training corpora** over GMS
ground truth. The package unifies three previously separate concerns behind one
interface:

- a **base taxonomy** of question types whose answers are computed by GMS
  primitives (provably correct), drawn from a store, labeled seed cases, or
  supplied by the user;
- a **DoE enrichment** layer of presentation factors that vary a question
  without changing its answer (`gt_invariant`), filtered to the bases they
  apply to (`applies_to`);
- two **consumers** of the composed scenarios: supervised fine-tuning corpora
  (`emit_classifier_sft`, `emit_draft_sft`) and evaluation with logistic factor
  attribution (`evaluate.run`, `evaluate.attribute`, `evaluate.weak_link`).

It composes with the rest of the harness rather than replacing it:
`compose(..., design_fn=graphdoe_design)` uses the `graphdoe` Sobol+refine
designs, `CatalogBaseSource` drives the `benchmark` generators, and
`evaluate.attribute` wraps `graphdoe.DOEAnalyzer`.

## Layout

```
suite/
  catalogs/   base_catalog.yaml · factor_catalog.yaml · profiles.yaml
  catalogs.py resolve.py sources.py compose.py emit.py evaluate.py
  _validate.py   consistency checks (python -m knowlytix.harness.suite._validate)
```

## Quickstart

```python
from knowlytix.harness import suite as S
from knowlytix.harness.suite.compose import graphdoe_design

cat   = S.Catalog.load()
suite = S.resolve_profile(cat, 'numeric_screen')          # bases + factors + mode
items = S.CatalogBaseSource(store).items(suite)           # or SeedCaseSource / UserBaseSource
scns  = S.compose(suite, items, n_runs=120, design_fn=graphdoe_design)

recs  = S.emit_classifier_sft(scns)                       # SFT corpus
rows  = S.evaluate.run(scns, sut)                         # evaluation
table = S.evaluate.attribute(rows, suite.factor_names)    # logistic attribution
```

The catalogs are domain-neutral; applications supply level vocabulary via
`compose(..., level_overrides=...)`.
