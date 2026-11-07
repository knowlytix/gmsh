"""Regression test for PolicyEngine Layer 2 triple-shape handling.

``GMSExpertStore.query_triples`` (via ``knowlytix.benchmark.graph.DocumentGraph``)
returns a list of ``(head, relation, tail)`` **tuples** --- see
``docgms/DESIGN.md`` §4.4.  An earlier implementation of
:meth:`PolicyEngine.check` unpacked results via ``triple.get("relation")``
or ``getattr(triple, "relation")`` and therefore silently skipped every
hit when the store returned tuples.  The practical effect was that
Layer 2 (graph lookup) of the four-layer check was a no-op across the
entire banking governance surface, even though Lab 12.1 / Lab 12.3 in the
monograph claim to exercise it.

This test pins the tuple code path so the regression cannot recur
without the test failing.  A dict-form fixture is included to keep
alternative store implementations compatible.
"""

from __future__ import annotations

from types import SimpleNamespace

from knowlytix.harness.testing.policy import PolicyEngine


class _RecordingLearner:
    """No-op RuntimeLearner so PolicyEngine init succeeds."""

    def learn_triples(self, triples, device):
        return SimpleNamespace(n_triples_added=len(triples))

    def flush_training(self):
        pass

    def wait_for_training(self, timeout=None):
        pass


class _StubStore:
    """Return whatever triple-shape the test parameter specifies.

    PolicyEngine.check() also touches fuzzy_match_entity(),
    score_triple() and tension_energy(); they are stubbed to
    unambiguous non-firing answers so the test isolates Layer 2.
    """

    def __init__(self, triples, *, entities=None):
        self._triples = list(triples)
        self._entities = set(entities or [])
        self.adapter = SimpleNamespace(
            entity_to_idx={e: i for i, e in enumerate(self._entities)},
            relation_to_idx={},
        )
        self.config = None

    def query_triples(self, head=None, relation=None, tail=None):
        return [
            t for t in self._triples
            if (head is None or t[0] == head)
            and (relation is None or t[1] == relation)
            and (tail is None or t[2] == tail)
        ]

    def fuzzy_match_entity(self, name):
        return name if name in self._entities else None

    def score_triple(self, *args, **kwargs):
        return None

    def tension_energy(self, *args, **kwargs):
        return None

    def stats(self):
        return {"entities": len(self._entities), "relations": 0,
                "triples": len(self._triples)}


def _build_engine(triples, entities):
    """Construct a PolicyEngine without side effects."""
    store = _StubStore(triples, entities=entities)
    engine = PolicyEngine.__new__(PolicyEngine)
    engine.store = store
    engine.tracing = None
    engine.run_id = ""
    engine._rules = []
    engine._learner = _RecordingLearner()
    engine._plausibility_relation = None
    engine._judge = None
    engine._calibration_source = {"plausibility": "default",
                                  "tension": "default"}
    engine.PLAUSIBILITY_THRESHOLD = 1.5
    engine.TENSION_THRESHOLD = 1.7
    return engine


def test_layer2_fires_on_tuple_triples():
    """Tuple form: (head, relation, tail). The production store shape."""
    engine = _build_engine(
        triples=[("delete_raw_data", "blocks", "raw_data_protection")],
        entities={"delete_raw_data", "raw_data_protection"},
    )

    decision = engine.check("delete_raw_data", args={}, context="")

    assert not decision.allowed, "Layer 2 must block on blocks triple"
    graph_rule_ids = [r for r in decision.blocking_rules
                      if r.startswith("gms_graph:")]
    assert graph_rule_ids == [
        "gms_graph:delete_raw_data->blocks->raw_data_protection"
    ], decision.blocking_rules


def test_layer2_fires_on_dict_triples():
    """Dict form: compat path for alternative stores."""
    engine = _build_engine(
        triples=[{"head": "delete_raw_data", "relation": "blocks",
                  "tail": "raw_data_protection"}],
        entities={"delete_raw_data", "raw_data_protection"},
    )

    # _StubStore.query_triples returns whatever list it was given;
    # the dict path needs a slightly different indexing, so adapt:
    def _query(head=None, relation=None, tail=None):
        return [t for t in engine.store._triples
                if (head is None or t["head"] == head)]
    engine.store.query_triples = _query

    decision = engine.check("delete_raw_data", args={}, context="")
    assert not decision.allowed
    graph_rule_ids = [r for r in decision.blocking_rules
                      if r.startswith("gms_graph:")]
    assert graph_rule_ids == [
        "gms_graph:delete_raw_data->blocks->raw_data_protection"
    ]


def test_layer2_fires_on_requires_triple():
    """Requires relation: the other Layer 2 branch."""
    engine = _build_engine(
        triples=[("deploy_model", "requires", "fairness_checked")],
        entities={"deploy_model", "fairness_checked"},
    )

    decision = engine.check("deploy_model", args={}, context="")
    assert not decision.allowed
    graph_rule_ids = [r for r in decision.blocking_rules
                      if r.startswith("gms_graph:")]
    assert graph_rule_ids == [
        "gms_graph:deploy_model->requires->fairness_checked"
    ]


def test_layer2_noop_on_unrelated_relation():
    """Non-blocks/non-requires triples must not add to blocking_rules."""
    engine = _build_engine(
        triples=[("deploy_model", "is_type", "action")],
        entities={"deploy_model", "action"},
    )

    decision = engine.check("deploy_model", args={}, context="")
    assert decision.allowed, "No blocks/requires relation → no gate fires"
    assert decision.blocking_rules == []
