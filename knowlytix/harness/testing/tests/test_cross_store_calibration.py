"""Round-trip + oracle-wiring tests for cross-store tension calibration.

Exercises four guarantees:

1. Synthetic 3-feature samples feed through
   :func:`calibrate_cross_store_tension` and produce a fitted
   :class:`CalibratedClassifierCV`, a sane
   ``projection_thresholds`` block
   (``0 <= tau_minor_upper <= tau_contra_lower``), and a metrics bundle.
2. :func:`save_cross_store_calibration` writes three files with the
   v2 schema; :func:`load_cross_store_calibration` round-trips them.
3. :class:`CrossDocumentAnalyzer` discovers the artifacts via
   ``calibration_dir=`` and classifies per-triple through the 3-D
   calibrated classifier (not the per-claim tau).
4. :class:`ConsensusOracle` does the same -- the worst-triple 3-D
   feature vector drives both label and confidence.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _make_synthetic_samples(rng_seed: int = 7):
    """Generate a clean 3-class dataset the calibrator can actually fit.

    Features mirror the real manifest-driven distribution in
    ``knowlytix/harness/testing/tests/model_risk_assessment_calibration_manifest.json``:

      UNCHANGED    -- delta ~ 0.05, tail_missing=0, has_replacement=0
      MODIFIED     -- tail orphaned in amended -> delta = 0 by
                      construction, tail_missing=1, has_replacement=1
      CONTRADICTED -- tail preserved, delta ~ 0.90, has_replacement=1

    MODIFIED rows are hard for a delta-only model (no signal); the
    tail_missing / has_replacement features are what carry the class.
    """
    from knowlytix.harness.testing.cross_store_calibration import CrossStoreSample
    rng = np.random.default_rng(rng_seed)
    samples = []
    for i in range(120):
        samples.append(CrossStoreSample(
            head=f"ent_{i}", relation="has_x", tail=f"v_{i}",
            score_base=1.0, score_amended=0.95,
            delta=float(max(0.0, rng.normal(0.05, 0.02))),
            tail_missing=0, has_replacement=0,
            label=0, label_name="UNCHANGED",
            provenance="unchanged_in_both",
        ))
    for i in range(40):
        samples.append(CrossStoreSample(
            head=f"ent_{i+200}", relation="has_x", tail=f"v_{i}",
            score_base=1.0, score_amended=None,
            delta=0.0,                          # tail orphaned -> delta=0
            tail_missing=1, has_replacement=1,
            label=1, label_name="MODIFIED",
            provenance="manifest:MODIFIED",
        ))
    for i in range(40):
        samples.append(CrossStoreSample(
            head=f"ent_{i+400}", relation="has_x", tail=f"v_{i}",
            score_base=1.0, score_amended=0.1,
            delta=float(max(0.0, rng.normal(0.90, 0.05))),
            tail_missing=0, has_replacement=1,
            label=2, label_name="CONTRADICTED",
            provenance="manifest:CONTRADICTED",
        ))
    return samples


# ---------------------------------------------------------------------------
# Calibration fit
# ---------------------------------------------------------------------------

def test_calibrate_produces_ordered_projection_thresholds():
    from knowlytix.harness.testing.cross_store_calibration import calibrate_cross_store_tension
    samples = _make_synthetic_samples()
    result = calibrate_cross_store_tension(samples, seed=13)
    pt = result.projection_thresholds
    assert "tau_minor_upper" in pt
    assert "tau_contra_lower" in pt
    tau_minor = pt["tau_minor_upper"]
    tau_contra = pt["tau_contra_lower"]
    assert 0.0 <= tau_minor <= tau_contra

    metrics = result.metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix_labels"] == [
        "UNCHANGED", "MODIFIED", "CONTRADICTED",
    ]
    # 3-D predict_proba
    X = [
        [0.05, 0.0, 0.0],   # UNCHANGED cell
        [0.40, 1.0, 1.0],   # MODIFIED cell
        [0.90, 0.0, 1.0],   # CONTRADICTED cell
    ]
    probs = result.classifier.predict_proba(X)
    assert probs.shape == (3, 3)
    assert np.argmax(probs[0]) == 0
    assert np.argmax(probs[2]) == 2


def test_calibrate_reports_per_class_support():
    from knowlytix.harness.testing.cross_store_calibration import calibrate_cross_store_tension
    samples = _make_synthetic_samples()
    result = calibrate_cross_store_tension(samples, seed=13)
    per_class = result.metrics["per_class"]
    for name in ("UNCHANGED", "MODIFIED", "CONTRADICTED"):
        assert per_class[name]["support"] > 0
        assert 0.0 <= per_class[name]["precision"] <= 1.0
        assert 0.0 <= per_class[name]["recall"] <= 1.0
    assert 0.0 <= result.metrics["expected_calibration_error"] <= 1.0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    from knowlytix.harness.testing.cross_store_calibration import (
        calibrate_cross_store_tension,
        save_cross_store_calibration,
        load_cross_store_calibration,
        SCHEMA_VERSION,
    )
    samples = _make_synthetic_samples()
    result = calibrate_cross_store_tension(samples, seed=13)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_cross_store_calibration(result, Path(tmpdir))
        assert paths["header"].is_file()
        assert paths["classifier"].is_file()
        assert paths["metrics"].is_file()

        payload = json.loads(paths["header"].read_text())
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["feature_names"] == [
            "delta", "tail_missing", "has_replacement",
        ]
        assert "projection_thresholds" in payload
        assert "sample_counts" in payload
        assert "splits" in payload

        loaded = load_cross_store_calibration(Path(tmpdir))
        saved_pt = loaded["header"]["projection_thresholds"]
        assert saved_pt["tau_minor_upper"] == pytest.approx(
            result.projection_thresholds["tau_minor_upper"]
        )
        # Classifier prediction stable on a 3-D vector
        X = [[0.3, 0.0, 1.0]]
        p_new = loaded["classifier"].predict_proba(X)
        p_old = result.classifier.predict_proba(X)
        assert np.allclose(p_new, p_old)


# ---------------------------------------------------------------------------
# CrossDocumentAnalyzer auto-load
# ---------------------------------------------------------------------------

class _StubStore:
    """Minimal store surface for the analyzer to walk."""
    def __init__(
        self,
        store_path: str = "",
        entities: list[str] | None = None,
        triples: list[tuple[str, str, str]] | None = None,
        scores: dict | None = None,
    ):
        self.store_path = store_path
        self._scores = scores or {}
        ent_list = entities if entities is not None else ["ModelA", "ModelB", "ModelC"]
        self.adapter = SimpleNamespace(
            entity_to_idx={e: i for i, e in enumerate(ent_list)},
        )
        self.doc_graph = SimpleNamespace(triples=triples or [])

    def score_triple(self, head, rel, tail):
        return self._scores.get((head, rel, tail))

    def tension_energy(self, *args, **kwargs):
        return None


def _write_cross_store_calibration(target_dir: Path) -> None:
    from knowlytix.harness.testing.cross_store_calibration import (
        calibrate_cross_store_tension,
        save_cross_store_calibration,
    )
    samples = _make_synthetic_samples()
    result = calibrate_cross_store_tension(samples, seed=13)
    save_cross_store_calibration(result, target_dir)


def test_analyzer_uses_cross_store_classifier():
    from knowlytix.harness.testing.cross_document import CrossDocumentAnalyzer

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_cross_store_calibration(Path(tmpdir))

        # Three distinct regimes mirroring the training feature cells:
        #   ModelA  -- (delta=0.03, tm=0, hr=0)  UNCHANGED, filtered out
        #   ModelB  -- tail "Medium" orphaned in B, replacement "Low"
        #              (delta=0, tm=1, hr=1)  MODIFIED cell
        #   ModelC  -- tail "High" retained in B, replacement "Medium"
        #              (delta=0.9, tm=0, hr=1) CONTRADICTED cell
        triples_a = [
            ("ModelA", "has_rating", "High"),
            ("ModelB", "has_rating", "Medium"),
            ("ModelC", "has_rating", "High"),
        ]
        triples_b = [
            ("ModelA", "has_rating", "High"),
            ("ModelB", "has_rating", "Low"),
            ("ModelC", "has_rating", "High"),
            ("ModelC", "has_rating", "Medium"),
        ]
        sa = _StubStore(
            store_path="",
            entities=["ModelA", "ModelB", "ModelC", "High", "Medium"],
            triples=triples_a,
            scores={
                ("ModelA", "has_rating", "High"): 1.00,
                ("ModelB", "has_rating", "Medium"): 1.00,
                ("ModelC", "has_rating", "High"): 1.00,
            },
        )
        sb = _StubStore(
            store_path="",
            # "Medium" intentionally orphaned in store B so the ModelB
            # triple hits the (tail_missing=1, has_replacement=1) cell.
            entities=["ModelA", "ModelB", "ModelC", "High", "Low"],
            triples=triples_b,
            scores={
                ("ModelA", "has_rating", "High"): 0.97,   # delta ~0.03
                ("ModelC", "has_rating", "High"): 0.10,   # delta ~0.90
            },
        )
        analyzer = CrossDocumentAnalyzer(
            stores={"a": sa, "b": sb},
            calibration_dir=tmpdir,
        )
        assert analyzer._cross_store_classifier is not None

        conflicts = analyzer.find_conflicts()
        by_entity = {c.entity: c for c in conflicts}
        assert "ModelA" not in by_entity          # UNCHANGED filtered out
        assert "ModelB" in by_entity
        assert "ModelC" in by_entity
        # ModelC's (delta=0.9, tm=0, hr=1) should classify as contradiction.
        assert by_entity["ModelC"].conflict_type == "contradiction"
        # ModelB's (delta=0, tm=1, hr=1) should surface (orphaned tail).
        assert by_entity["ModelB"].tail_missing == 1
        assert by_entity["ModelB"].has_replacement == 1
        for c in conflicts:
            assert 0.0 <= c.confidence <= 1.0
            assert c.relation == "has_rating"


def test_conflict_type_prefers_cross_store_bands():
    from knowlytix.harness.testing.cross_document import _conflict_type
    # Cross-store bands dominate
    assert _conflict_type(0.1, tau=0.9,
                          tau_minor_upper=0.2, tau_contra_lower=0.6) \
        == "minor_variance"
    assert _conflict_type(0.3, tau=0.9,
                          tau_minor_upper=0.2, tau_contra_lower=0.6) \
        == "disagreement"
    assert _conflict_type(0.8, tau=0.9,
                          tau_minor_upper=0.2, tau_contra_lower=0.6) \
        == "contradiction"
    # Falls back to tau when cross-store bands absent
    assert _conflict_type(0.1, tau=0.5) == "minor_variance"
    assert _conflict_type(0.6, tau=0.5) == "disagreement"


# ---------------------------------------------------------------------------
# ConsensusOracle wiring
# ---------------------------------------------------------------------------

def test_consensus_oracle_uses_cross_store_classifier():
    from knowlytix.harness.testing.multi_agent import ConsensusOracle

    triples = [
        ("ModelA", "has_rating", "High"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_cross_store_calibration(Path(tmpdir))

        # Small delta -> consensus_reached = True, label minor_variance.
        low_a = _StubStore(
            entities=["ModelA", "High"],
            triples=triples,
            scores={("ModelA", "has_rating", "High"): 1.00},
        )
        low_b = _StubStore(
            entities=["ModelA", "High"],
            triples=triples,
            scores={("ModelA", "has_rating", "High"): 0.97},
        )
        oracle = ConsensusOracle(
            stores={"agent_a": low_a, "agent_b": low_b},
            calibration_dir=tmpdir,
        )
        r = oracle.evaluate_consensus("ModelA")
        assert r.consensus_reached is True
        assert r.consensus_label == "minor_variance"
        assert r.tau_minor_upper is not None
        assert r.tau_contra_lower is not None

        # Large delta with replacement tail -> CONTRADICTED cell
        # (tm=0, hr=1), feeding the classifier a vector it was
        # actually trained on.
        hi_a = _StubStore(
            entities=["ModelA", "High", "Medium"],
            triples=triples,
            scores={("ModelA", "has_rating", "High"): 1.00},
        )
        hi_b = _StubStore(
            entities=["ModelA", "High", "Medium"],
            triples=[
                ("ModelA", "has_rating", "High"),
                ("ModelA", "has_rating", "Medium"),   # replacement
            ],
            scores={("ModelA", "has_rating", "High"): 0.10},
        )
        oracle2 = ConsensusOracle(
            stores={"agent_a": hi_a, "agent_b": hi_b},
            calibration_dir=tmpdir,
        )
        r2 = oracle2.evaluate_consensus("ModelA")
        assert r2.consensus_reached is False
        assert r2.consensus_label in {"disagreement", "contradiction"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
