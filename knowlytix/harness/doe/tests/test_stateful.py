"""Real-store tests for StatefulAgentTester and MemoryRecallOracle.

Everything runs against the ingested ``ch04_ground_truth`` store
(``conftest.py::ch04_store``).  No duck-typing of geometric primitives
— ``score_triple`` and ``tension_energy`` hit the trained manifold
directly.

What each test pins:

1. ``build_probes`` really samples from the loaded GMS and every
   teach/recall prompt refers to a triple the store can score.
2. ``score_recall`` fills ``recall_geodesic`` with a real float when
   the agent echoes a known tail and a larger one when it drifts.
3. The full teach → delay → recall cycle against ``FullMockHarness``
   yields a geodesic consistent with perfect in-memory recall.
4. ``MemoryRecallOracle.evaluate`` emits the audit triple
   (``geodesic_expected``, ``geodesic_recalled``, tension) for both
   the perfect-recall and drifted-recall cases, and the passed flag
   tracks the calibrated threshold loaded from ``calibration.json``
   next to the store.
5. ``assess`` aggregates real scores against the calibrated geodesic
   cut (not an ad-hoc 0.5) and the 1.2× degradation rule fires when
   late-session tension rises.
6. Calibration values from ``calibration.json`` are the single source
   of truth — tester/oracle/generator/adapter all consume the same
   numbers; tests assert equality with the on-disk JSON.
7. ``StatefulAgentGenerator`` only emits questions for triples that
   clear the calibrated plausibility cut.

Unit-level dataclass round-trips (``MemoryAssessment.to_dict``) are
kept in place since they don't need the store.
"""

from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Calibration source of truth — read once, reused by every calibrated test.
# ---------------------------------------------------------------------------

def _calibration_thresholds(store) -> dict:
    """Return the ``thresholds`` block from the store's calibration.json.

    Tests use this rather than hard-coded numbers so a re-calibration of
    ch04_ground_truth automatically flows into the expected values.
    """
    path = os.path.join(store.store_path, "calibration.json")
    assert os.path.isfile(path), f"calibration.json missing at {path}"
    with open(path) as f:
        return json.load(f)["thresholds"]


# ---------------------------------------------------------------------------
# build_probes — against the real store
# ---------------------------------------------------------------------------

def test_tester_loads_calibrated_thresholds(ch04_store):
    """Constructing ``StatefulAgentTester`` must load tau_geodesic and
    tau_tension from calibration.json.  These attributes are what
    ``assess()`` consumes — a drift here means the harness is silently
    falling back to hand-picked constants."""
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    cal = _calibration_thresholds(ch04_store)
    assert tester.recall_geodesic_threshold == pytest.approx(cal["geodesic"])
    assert tester.tension_threshold == pytest.approx(cal["tension"])


def test_build_probes_uses_real_store_triples(ch04_store):
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    probes = tester.build_probes(n_probes=5, delay_turns=3, seed=42)

    store_triples = set(ch04_store.doc_graph.triples)
    assert len(probes) == 5
    for p in probes:
        assert p.fact_triple in store_triples
        assert p.expected_answer == p.fact_triple[2]
        assert p.delay_turns == 3
        h, r, _t = p.fact_triple
        assert h in p.teach_prompt and r in p.teach_prompt
        assert h in p.recall_prompt and r in p.recall_prompt


def test_build_probes_deterministic_with_seed(ch04_store):
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    a = StatefulAgentTester(store=ch04_store).build_probes(n_probes=5, seed=7)
    b = StatefulAgentTester(store=ch04_store).build_probes(n_probes=5, seed=7)
    assert [p.fact_triple for p in a] == [p.fact_triple for p in b]


def test_build_probes_caps_at_triple_count(ch04_store):
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    total = len(list(ch04_store.doc_graph.triples))
    probes = StatefulAgentTester(store=ch04_store).build_probes(
        n_probes=total + 50, seed=0,
    )
    assert len(probes) == total


# ---------------------------------------------------------------------------
# score_recall — real geodesic distances
# ---------------------------------------------------------------------------

def test_score_recall_with_real_tail_returns_finite_float(ch04_store):
    """When the recalled string is an entity the manifold knows, the
    score is a real, non-negative float."""
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    probes = tester.build_probes(n_probes=3, seed=42)
    for p in probes:
        _h, _r, t = p.fact_triple
        tester.score_recall(p, t)  # perfect recall: recalled == taught
        assert p.recall_geodesic is not None
        assert isinstance(p.recall_geodesic, float)
        assert p.recall_geodesic >= 0.0


def test_score_recall_drift_raises_geodesic(ch04_store):
    """A drifted recall produces a geodesic strictly larger than the
    perfect-recall geodesic on the same (h, r)."""
    from knowlytix.harness.doe.stateful import AgentMemoryProbe, StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    # Pick the triple with the lowest geodesic — the manifold's best fit.
    scored = [
        (ch04_store.score_triple(*tr), tr)
        for tr in ch04_store.doc_graph.triples
    ]
    scored = [(s, tr) for s, tr in scored if s is not None]
    scored.sort(key=lambda x: x[0])
    good_triple = scored[0][1] if scored else None
    assert good_triple is not None, "real store has no scorable triples"

    h, r, t = good_triple
    perfect = AgentMemoryProbe(fact_triple=good_triple, expected_answer=t)
    drifted = AgentMemoryProbe(fact_triple=good_triple, expected_answer=t)
    tester.score_recall(perfect, t)
    # Use an entity the manifold knows but is structurally unrelated.
    drifted_tail = next(
        (trip[0] for trip in ch04_store.doc_graph.triples if trip[0] != h),
        "Organization",
    )
    tester.score_recall(drifted, drifted_tail)

    assert perfect.recall_geodesic is not None
    assert drifted.recall_geodesic is not None
    assert drifted.recall_geodesic > perfect.recall_geodesic


def test_measure_session_tension_returns_real_float(ch04_store):
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    triples = list(ch04_store.doc_graph.triples)
    a, b = triples[0][0], triples[1][0]
    t = tester.measure_session_tension(a, b)
    assert t is not None
    assert isinstance(t, float)


# ---------------------------------------------------------------------------
# End-to-end teach → delay → recall against FullMockHarness
# ---------------------------------------------------------------------------

def test_full_teach_delay_recall_cycle(ch04_store, full_mock_harness):
    """FullMockHarness.teach() plants a fact; after distractor turns,
    recall() still returns it exactly.  The probe's geodesic for the
    perfectly-recalled tail must be finite and equal to the raw
    ``score_triple`` call on the same (h, r, taught) triple."""
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    probes = tester.build_probes(n_probes=3, delay_turns=2, seed=1)

    for probe in probes:
        h, r, t = probe.fact_triple
        full_mock_harness.teach(f"{h}_{r}", t)
        for _ in range(probe.delay_turns):
            full_mock_harness.run("What is the test AUC of MDL-001?")
        recalled = full_mock_harness.recall(f"{h}_{r}")
        assert recalled == t, "FullMockHarness lost a taught fact"
        tester.score_recall(probe, recalled)
        expected_geo = ch04_store.score_triple(h, r, t)
        assert probe.recall_geodesic == pytest.approx(expected_geo)


# ---------------------------------------------------------------------------
# MemoryRecallOracle — real primitives, both branches
# ---------------------------------------------------------------------------

def test_oracle_perfect_recall_against_real_store(ch04_store):
    """When expected_tail == recalled_answer, tension collapses toward
    zero and both geodesics are equal."""
    from knowlytix.harness.doe.stateful import MemoryRecallOracle

    triples = list(ch04_store.doc_graph.triples)
    # Use a triple the store actually scores (not all tails are entities).
    picked = None
    for h, r, t in triples:
        if ch04_store.score_triple(h, r, t) is not None:
            picked = (h, r, t)
            break
    assert picked is not None

    h, r, t = picked
    v = MemoryRecallOracle(store=ch04_store, geodesic_threshold=5.0).evaluate(
        head=h, relation=r, expected_tail=t, recalled_answer=t,
    )
    assert v["oracle_type"] == "memory_recall"
    assert v["geodesic_expected"] is not None
    assert v["geodesic_recalled"] is not None
    assert v["geodesic_expected"] == v["geodesic_recalled"]
    # Perfect match: tension should be ~0 (same entity, zero drift).
    if v["tension_expected_vs_recalled"] is not None:
        assert v["tension_expected_vs_recalled"] < 0.01


def test_oracle_drifted_recall_raises_tension(ch04_store):
    """A recalled answer that is a different entity produces higher
    tension than a perfect recall."""
    from knowlytix.harness.doe.stateful import MemoryRecallOracle

    triples = list(ch04_store.doc_graph.triples)
    picked, drifted = None, None
    for h, r, t in triples:
        if ch04_store.score_triple(h, r, t) is not None:
            picked = (h, r, t)
            break
    for h2, _, _ in triples:
        if h2 != picked[0] and h2 != picked[2]:
            drifted = h2
            break
    assert picked is not None and drifted is not None

    h, r, t = picked
    oracle = MemoryRecallOracle(store=ch04_store, geodesic_threshold=5.0)
    perfect = oracle.evaluate(h, r, t, t)
    drift = oracle.evaluate(h, r, t, drifted)

    # Recalled geodesic must move when the tail changes.
    assert drift["geodesic_recalled"] != perfect["geodesic_recalled"]
    # Tension for the drifted case is higher than for the perfect case.
    if (perfect["tension_expected_vs_recalled"] is not None
            and drift["tension_expected_vs_recalled"] is not None):
        assert (drift["tension_expected_vs_recalled"]
                >= perfect["tension_expected_vs_recalled"])


def test_oracle_auto_loads_calibrated_threshold(ch04_store):
    """Constructing ``MemoryRecallOracle`` without an explicit threshold
    must pull tau_geodesic from calibration.json next to the store —
    same source as every other calibrated oracle."""
    from knowlytix.harness.doe.stateful import MemoryRecallOracle

    oracle = MemoryRecallOracle(store=ch04_store)
    cal = _calibration_thresholds(ch04_store)
    assert oracle.geodesic_threshold == pytest.approx(cal["geodesic"])


def test_oracle_threshold_gate(ch04_store):
    """Passed iff geodesic_recalled <= threshold.  Setting threshold
    *just above* the perfect-recall geodesic passes; just below, it
    fails.  The gate is inclusive — calibrated thresholds are
    boundary values, so equality on tau must count as a pass."""
    from knowlytix.harness.doe.stateful import MemoryRecallOracle

    triples = list(ch04_store.doc_graph.triples)
    h, r, t = next(trip for trip in triples
                   if ch04_store.score_triple(*trip) is not None)
    geo = ch04_store.score_triple(h, r, t)

    above = MemoryRecallOracle(
        store=ch04_store, geodesic_threshold=geo + 0.1,
    ).evaluate(h, r, t, t)
    below = MemoryRecallOracle(
        store=ch04_store, geodesic_threshold=max(geo - 0.1, 0.0),
    ).evaluate(h, r, t, t)

    assert above["passed"] is True
    assert below["passed"] is False
    # The audit trail must carry the threshold used — otherwise the
    # verdict cannot be reproduced offline.
    assert above["tau_geodesic"] == pytest.approx(geo + 0.1)


# ---------------------------------------------------------------------------
# assess — aggregation on real scored probes
# ---------------------------------------------------------------------------

def test_assess_counts_real_recalls_at_calibrated_threshold(
    ch04_store, full_mock_harness,
):
    """End-to-end: build 5 probes, teach all, recall all, assess.
    ``n_recalled`` must match the count of probes whose geodesic is
    ``<=`` the calibrated tau_geodesic — the same cut the tester
    reports on ``recall_geodesic_threshold``."""
    from knowlytix.harness.doe.stateful import StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    tau = tester.recall_geodesic_threshold
    cal = _calibration_thresholds(ch04_store)
    # The tester's live threshold must equal the on-disk calibration.
    assert tau == pytest.approx(cal["geodesic"])

    probes = tester.build_probes(n_probes=5, delay_turns=0, seed=42)
    for p in probes:
        h, r, t = p.fact_triple
        full_mock_harness.teach(f"{h}_{r}", t)
        recalled = full_mock_harness.recall(f"{h}_{r}")
        tester.score_recall(p, recalled)

    a = tester.assess(probes)
    expected_recalled = sum(
        1 for p in probes
        if p.recall_geodesic is not None and p.recall_geodesic <= tau
    )
    assert a.n_recalled == expected_recalled
    assert a.n_probes == 5


def test_assess_degradation_rule_on_real_tensions(ch04_store):
    """Construct a probe list with real session-tension values that
    rise across the midpoint — the 1.2× rule (stateful.py:272) must
    fire.  Uses the real tension primitive to generate the numbers."""
    from knowlytix.harness.doe.stateful import AgentMemoryProbe, StatefulAgentTester

    tester = StatefulAgentTester(store=ch04_store)
    triples = list(ch04_store.doc_graph.triples)
    # Low-tension pair: same entity with itself.
    low_entity = triples[0][0]
    low_t = ch04_store.tension_energy(low_entity, low_entity)
    # Higher-tension pair: two different entities.
    hi_t = None
    for h1, _, _ in triples:
        if h1 != low_entity:
            hi_t = ch04_store.tension_energy(low_entity, h1)
            if hi_t is not None and hi_t > low_t * 2:
                break
    if hi_t is None or not (hi_t > low_t * 1.5):
        pytest.skip("no pair with sufficient tension spread in this store")

    def _p(s):
        p = AgentMemoryProbe()
        p.session_tension = float(s)
        return p

    # Early-half low, late-half high — should trigger degradation.
    probes = [_p(low_t), _p(low_t), _p(low_t),
              _p(hi_t), _p(hi_t), _p(hi_t)]
    assert tester.assess(probes).degradation_detected is True

    # All-low: no degradation.
    flat = [_p(low_t) for _ in range(6)]
    assert tester.assess(flat).degradation_detected is False


def test_memory_assessment_to_dict_round_trip():
    """Unit-level — dataclass serialization."""
    from knowlytix.harness.doe.stateful import MemoryAssessment

    d = MemoryAssessment(
        n_probes=4, n_recalled=3, mean_recall_geodesic=0.42,
        mean_session_tension=0.01, degradation_detected=False,
    ).to_dict()
    assert d["recall_rate"] == pytest.approx(0.75)
    assert d["mean_recall_geodesic"] == pytest.approx(0.42)
    assert d["degradation_detected"] is False


# ---------------------------------------------------------------------------
# StatefulAgentGenerator — against real store
# ---------------------------------------------------------------------------

def test_generator_only_emits_plausible_triples(ch04_store):
    """Every generated question must correspond to a triple whose
    real geodesic score is within the calibrated tau_geodesic.  The
    generator samples 20 triples internally and filters — on stores
    where few triples clear the cut, the output may be empty; the
    invariant is that anything it *does* emit passes the calibrated
    cut, not an ad-hoc 1.0."""
    from knowlytix.harness.doe.stateful import StatefulAgentGenerator

    tau = _calibration_thresholds(ch04_store)["geodesic"]
    qs = StatefulAgentGenerator().generate(ch04_store)
    for q in qs:
        h = q.metadata["head"]
        r = q.metadata["relation"]
        t = q.metadata["expected_tail"]
        score = ch04_store.score_triple(h, r, t)
        assert score is not None
        assert score <= tau


def test_generator_returns_empty_when_graph_missing(ch04_store):
    """Unit-level guard: no doc_graph → no questions."""
    from knowlytix.harness.doe.stateful import StatefulAgentGenerator

    # Temporarily hide the graph.
    saved = ch04_store.doc_graph
    ch04_store.doc_graph = None
    try:
        assert StatefulAgentGenerator().generate(ch04_store) == []
    finally:
        ch04_store.doc_graph = saved


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
