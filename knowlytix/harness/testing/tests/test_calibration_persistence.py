"""Round-trip test for judge calibration persistence and oracle auto-load.

Exercises three guarantees:

1. :meth:`GMSJudge.save` → :meth:`GMSJudge.load` restores ``_thresholds``
   and ``_calibrations`` exactly.
2. Oracles discover ``calibration.json`` next to the store and apply the
   calibrated cut instead of the hand-picked default.
3. With no ``calibration.json`` present, oracles silently fall back to
   hand-picked defaults (objective O7).

No real store is trained — a duck-typed stub gives the oracles the
surface they need without the cost of loading a full GMS.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Stub store — enough surface for the oracles we exercise here.
# ---------------------------------------------------------------------------

class _StubStore:
    """Minimal store with the attributes oracles and judge touch.

    Oracles call ``score_triple`` / ``check_holonomy`` / ``tension_energy``
    and read ``store_path`` for the calibration loader.  Everything else
    is a no-op.
    """

    def __init__(self, store_path: str, geodesic: float = 0.0):
        self.store_path = store_path
        self._fixed_geodesic = geodesic
        # Attributes the judge constructor inspects before calibration.
        self.model = None
        self.transport = None
        self.adapter = None
        self.doc_graph = None
        self.enm = None
        self.device = "cpu"

    def score_triple(self, head, rel, tail):
        return self._fixed_geodesic

    def check_holonomy(self, expected, actual):
        # Return a tiny, well-below-threshold value; actual oracle
        # tests override this when they want a violation.
        return 0.0

    def tension_energy(self, *args, **kwargs):
        return 0.0


# ---------------------------------------------------------------------------
# Judge round-trip
# ---------------------------------------------------------------------------

def test_judge_save_load_roundtrip():
    """Thresholds and classifiers survive save → load."""
    from sklearn.linear_model import LogisticRegression

    from knowlytix.harness.testing.judge import GMSJudge

    with tempfile.TemporaryDirectory() as tmpdir:
        store = _StubStore(tmpdir)
        judge = GMSJudge(store)

        # Hand-populate calibration state as if calibrate() had run.
        judge._thresholds = {
            "geodesic": 0.35,
            "policy_geodesic": 0.28,
            "holonomy": 1.82,
            "tension": (0.95, 1.58),
            "tau_ent": 0.95,
            "tau_contra": 1.58,
            "phase": 0.1,
        }
        clf = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
        judge._calibrations = {"geodesic": clf}

        judge.save()
        assert os.path.isfile(os.path.join(tmpdir, "calibration.json"))
        assert os.path.isfile(os.path.join(tmpdir, "calibration.pkl"))

        # Fresh judge, no in-memory state.
        judge2 = GMSJudge(store)
        assert judge2.load() is True

        # Scalar thresholds round-trip as floats.
        assert judge2._thresholds["geodesic"] == pytest.approx(0.35)
        assert judge2._thresholds["tau_ent"] == pytest.approx(0.95)
        assert judge2._thresholds["tau_contra"] == pytest.approx(1.58)

        # Tuple threshold round-trips back to tuple.
        assert isinstance(judge2._thresholds["tension"], tuple)
        assert judge2._thresholds["tension"] == pytest.approx((0.95, 1.58))

        # Classifier reloaded; prediction is stable.
        reloaded = judge2._calibrations["geodesic"]
        assert reloaded.predict([[0.0]])[0] == clf.predict([[0.0]])[0]


def test_judge_load_missing_returns_false():
    """No calibration.json → load() returns False, leaves judge blank."""
    from knowlytix.harness.testing.judge import GMSJudge

    with tempfile.TemporaryDirectory() as tmpdir:
        store = _StubStore(tmpdir)
        judge = GMSJudge(store)
        assert judge.load() is False
        assert not getattr(judge, "_thresholds", None)


def test_judge_load_tolerates_missing_pickle():
    """calibration.json without calibration.pkl still restores thresholds."""
    from knowlytix.harness.testing.judge import GMSJudge

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "schema_version": 1,
            "thresholds": {"geodesic": 0.4},
            "weights": {"geodesic": 0.3, "tension": 0.2,
                        "holonomy": 0.2, "exact": 0.3},
        }
        with open(os.path.join(tmpdir, "calibration.json"), "w") as f:
            json.dump(payload, f)

        store = _StubStore(tmpdir)
        judge = GMSJudge(store)
        assert judge.load() is True
        assert judge._thresholds["geodesic"] == pytest.approx(0.4)
        # No classifiers loaded; calibrated_classify() would raise if called.
        assert getattr(judge, "_calibrations", {}) == {}


# ---------------------------------------------------------------------------
# Oracle auto-load
# ---------------------------------------------------------------------------

def _write_calibration(path: str, thresholds: dict) -> None:
    payload = {"schema_version": 1, "thresholds": thresholds,
               "weights": {"geodesic": 0.3}}
    with open(os.path.join(path, "calibration.json"), "w") as f:
        json.dump(payload, f)


def _make_scenario(ground_truth: str = "High"):
    return SimpleNamespace(
        ground_truth=ground_truth,
        metadata={},
        constraints=[],
        trace_pattern="",
        expected_tool_sequence=[],
        expected_side_effects={},
    )


def _make_trace(raw_output: str = "wrong"):
    return SimpleNamespace(raw_output=raw_output, tool_calls=[], latency_ms=0.0)


def test_exact_outcome_reads_calibration():
    """ExactOutcomeOracle uses calibrated geodesic cut when present."""
    from knowlytix.harness.testing.oracles import ExactOutcomeOracle

    with tempfile.TemporaryDirectory() as tmpdir:
        # A geodesic of 0.3 passes when tau = 0.5 but fails the default
        # 0.7 confidence cut (since 1 - 0.3/2 = 0.85 > 0.7 — both pass here).
        # Use a harder case: geodesic 0.45, tau 0.5 → calibrated passes;
        # default linear transform → conf = 0.775 → also passes.  So the
        # interesting divergence is the other way: geodesic 0.55, tau 0.5
        # → calibrated fails; default conf = 0.725 → default passes.
        _write_calibration(tmpdir, {"geodesic": 0.5})
        store = _StubStore(tmpdir, geodesic=0.55)

        oracle = ExactOutcomeOracle(store=store)
        verdict = oracle.evaluate(_make_scenario(), _make_trace())

        assert verdict.passed is False, (
            "calibrated cut should fail geodesic=0.55 vs tau=0.5"
        )
        assert verdict.raw_scores["tau_geodesic"] == pytest.approx(0.5)
        assert verdict.raw_scores["gms_geodesic"] == pytest.approx(0.55)


def test_exact_outcome_default_when_no_calibration():
    """Without calibration.json, the hand-picked 0.7 cut is used."""
    from knowlytix.harness.testing.oracles import ExactOutcomeOracle

    with tempfile.TemporaryDirectory() as tmpdir:
        # No calibration.json written.
        store = _StubStore(tmpdir, geodesic=0.55)
        oracle = ExactOutcomeOracle(store=store)
        verdict = oracle.evaluate(_make_scenario(), _make_trace())

        # Linear fallback: conf = 1 - 0.55/2 = 0.725 > 0.7 → passes.
        assert verdict.passed is True
        assert "tau_geodesic" not in verdict.raw_scores


def test_acceptable_set_reads_calibration():
    """AcceptableSetOracle uses the same calibrated geodesic cut."""
    from knowlytix.harness.testing.oracles import AcceptableSetOracle

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_calibration(tmpdir, {"geodesic": 0.5})
        store = _StubStore(tmpdir, geodesic=0.55)

        scenario = _make_scenario()
        scenario.metadata = {"acceptable_set": ["Critical", "Severe"]}
        trace = _make_trace("the risk is elevated")

        oracle = AcceptableSetOracle(store=store)
        verdict = oracle.evaluate(scenario, trace)

        # Calibrated path: best geodesic 0.55 > tau 0.5 → fails.
        assert verdict.passed is False
        assert verdict.raw_scores["tau_geodesic"] == pytest.approx(0.5)


def test_trace_oracle_flags_holonomy_overshoot():
    """TraceOracle fails when holonomy exceeds the calibrated cut."""
    from knowlytix.harness.testing.oracles import TraceOracle

    class HighHolonomyStore(_StubStore):
        def check_holonomy(self, expected, actual):
            return 2.5  # deliberately high

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_calibration(tmpdir, {"holonomy": 1.5})
        store = HighHolonomyStore(tmpdir)

        scenario = _make_scenario()
        scenario.trace_pattern = "a, b"
        scenario.expected_tool_sequence = ["a", "b"]
        trace = SimpleNamespace(
            tool_calls=[SimpleNamespace(tool_name="a"),
                        SimpleNamespace(tool_name="b")],
            raw_output="",
            latency_ms=0.0,
        )

        oracle = TraceOracle(store=store)
        verdict = oracle.evaluate(scenario, trace)

        # Symbolic check passes (both tools present), but holonomy
        # 2.5 > tau 1.5 forces a fail.
        assert verdict.passed is False
        assert verdict.raw_scores["holonomy_defect"] == pytest.approx(2.5)
        assert verdict.raw_scores["tau_holonomy"] == pytest.approx(1.5)


def test_loader_handles_malformed_calibration_json():
    """Corrupt calibration.json does not crash the oracle."""
    from knowlytix.harness.testing.oracles import ExactOutcomeOracle

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "calibration.json"), "w") as f:
            f.write("{not valid json")

        store = _StubStore(tmpdir, geodesic=0.1)
        oracle = ExactOutcomeOracle(store=store)
        # Should construct without error and fall back to defaults.
        assert oracle._thresholds == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
