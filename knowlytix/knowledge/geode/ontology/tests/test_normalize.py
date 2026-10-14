# SPDX-License-Identifier: Apache-2.0
"""Tier-1 deterministic normalization tests.

Covers the safety contract: merge only observed pairs, never rename a singleton,
never touch a value tail, veto a conflicting lexical group, and keep provenance
origins resolvable.
"""

from __future__ import annotations

from knowlytix.knowledge.geode.ontology import normalize_graph, singularize
from knowlytix.knowledge.geode.ontology.map import OntologyMap


def test_singularize_regular_and_guarded():
    assert singularize("policies") == "policy"
    assert singularize("taxes") == "tax"
    assert singularize("boxes") == "box"
    assert singularize("batches") == "batch"
    assert singularize("fees") == "fee"
    assert singularize("models") == "model"
    assert singularize("keys") == "key"
    # guarded / non-plural: unchanged
    assert singularize("status") == "status"
    assert singularize("class") == "class"      # -ss
    assert singularize("analysis") == "analysis"  # -is
    assert singularize("gas") == "gas"          # short
    assert singularize("series") == "series"    # stopword


def test_merges_observed_entity_plural_pair():
    triples = [
        ("overdraft fees", "has_amount", "35"),
        ("overdraft fee", "has_amount", "35"),
        ("overdraft fee", "has_waivable", "yes"),
    ]
    out, omap = normalize_graph(triples)
    # both surfaces collapse to one canonical (the more frequent "overdraft fee")
    assert omap.entity.get("overdraft fees") == "overdraft fee"
    heads = {h for h, _r, _t in out}
    assert "overdraft fees" not in heads and "overdraft fee" in heads
    # exact-duplicate fact collapsed
    assert out.count(("overdraft fee", "has_amount", "35")) == 1


def test_singleton_never_renamed():
    # "status" would mis-singularize as a key, but with no pair it must not change
    triples = [("account status", "has_value", "open"),
               ("account tier", "has_value", "gold")]
    out, omap = normalize_graph(triples)
    assert omap.is_noop()
    assert set(out) == set(triples)


def test_value_tail_untouched():
    # "checks" only ever appears as a TAIL value, never a head -> not a candidate
    triples = [("low-risk", "has_required_checks", "checks"),
               ("low-risk", "has_min_auc", "0.65")]
    out, omap = normalize_graph(triples)
    tails = {t for _h, _r, t in out}
    assert "checks" in tails               # value preserved verbatim
    assert omap.is_noop()


def test_conflict_veto_holds_out_member():
    # same lexical key ("fee"/"fees") but conflicting shared attribute -> not merged
    triples = [
        ("service fees", "has_amount", "10"),
        ("service fee", "has_amount", "99"),   # disagrees on the same attribute
    ]
    out, omap = normalize_graph(triples)
    assert omap.is_noop()                       # merge vetoed
    assert any(f["reason"] == "attribute_conflict" for f in omap.flagged)


def test_relation_plural_merge_and_functional_conflict():
    merged = normalize_graph([
        ("a", "has_fees", "1"), ("b", "has_fee", "2")])[1]
    assert merged.relation.get("has_fees") == "has_fee"

    # functional conflict: one head asserts both relations with different tails
    conflicted = normalize_graph([
        ("a", "has_fees", "1"), ("a", "has_fee", "2")])[1]
    assert not conflicted.relation      # not merged
    assert any(f["reason"] == "functional_conflict" for f in conflicted.flagged)


def test_provenance_origins_recorded():
    triples = [("overdraft fees", "has_amount", "35"),
               ("overdraft fee", "has_amount", "35")]
    out, omap = normalize_graph(triples)
    ct = ("overdraft fee", "has_amount", "35")
    origins = omap.origins(ct)
    assert ("overdraft fees", "has_amount", "35") in origins


def test_map_roundtrips_through_records():
    _out, omap = normalize_graph([
        ("overdraft fees", "has_amount", "35"),
        ("overdraft fee", "has_amount", "35")])
    back = OntologyMap.from_records(omap.to_records())
    assert back.entity == omap.entity
    assert back.origins(("overdraft fee", "has_amount", "35")) == \
        omap.origins(("overdraft fee", "has_amount", "35"))
