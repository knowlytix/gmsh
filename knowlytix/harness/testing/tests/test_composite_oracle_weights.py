"""Tests for CompositeOracle weights, YAML loader, and audit trail.

Exercises the three guarantees added to :mod:`knowlytix.harness.testing.oracles`:

1. ``DEFAULT_COMPOSITE_ORACLE_WEIGHTS`` keys stay in lockstep with
   each oracle's ``oracle_type`` class attribute, so a rename cannot
   drift the YAML schema silently.
2. ``load_composite_oracle_weights`` round-trips, merges over
   defaults, and raises on unknown keys or malformed YAML.
3. ``CompositeOracle.from_weights`` skips zero-weight entries, and
   the per-sub-oracle audit keys are sufficient to reconstruct the
   composite confidence by hand --- the aggregate must not be a
   black box.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Stub store — enough surface for the oracles the composite may instantiate.
# ---------------------------------------------------------------------------

class _StubStore:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.model = None
        self.transport = None
        self.adapter = None
        self.doc_graph = None
        self.enm = None
        self.device = "cpu"

    # Geometric primitives the oracles may call.
    def score_triple(self, h, r, t):
        return 0.0

    def check_holonomy(self, expected, actual):
        return 0.0

    def tension_energy(self, *args, **kwargs):
        return 0.0


def _scenario(**overrides):
    base = dict(
        ground_truth="High",
        question_text="q",
        metadata={},
        constraints=[],
        trace_pattern="",
        expected_tool_sequence=[],
        expected_side_effects={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _trace(raw_output: str = "High"):
    return SimpleNamespace(
        raw_output=raw_output,
        tool_calls=[],
        latency_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

def test_default_weights_cover_every_oracle_type():
    """Default-weights keys must match each oracle's oracle_type.

    Pins the contract: renaming ``ExactOutcomeOracle.oracle_type`` or
    adding a new oracle without updating DEFAULT_COMPOSITE_ORACLE_WEIGHTS
    becomes a test failure rather than a silent YAML-schema drift.
    """
    from knowlytix.harness.testing import DEFAULT_COMPOSITE_ORACLE_WEIGHTS
    from knowlytix.harness.testing.hallucination import HallucinationOracle
    from knowlytix.harness.testing.oracles import (
        AcceptableSetOracle,
        ConstraintOracle,
        ExactOutcomeOracle,
        HumanReviewedOracle,
        SideEffectOracle,
        TraceOracle,
    )

    expected = {
        ExactOutcomeOracle.oracle_type,
        AcceptableSetOracle.oracle_type,
        ConstraintOracle.oracle_type,
        TraceOracle.oracle_type,
        SideEffectOracle.oracle_type,
        HumanReviewedOracle.oracle_type,
        HallucinationOracle.oracle_type,
    }
    assert set(DEFAULT_COMPOSITE_ORACLE_WEIGHTS) == expected


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _write_yaml(tmpdir: str, body: str) -> str:
    path = os.path.join(tmpdir, "config.yaml")
    with open(path, "w") as f:
        f.write(body)
    return path


def test_loader_merges_over_defaults():
    """Unlisted keys keep their defaults; listed keys override them."""
    from knowlytix.harness.testing import (
        DEFAULT_COMPOSITE_ORACLE_WEIGHTS,
        load_composite_oracle_weights,
    )

    with tempfile.TemporaryDirectory() as td:
        path = _write_yaml(td, (
            "composite_oracle_weights:\n"
            "  exact_outcome: 0.75\n"
            "  trace: 0.05\n"
        ))
        w = load_composite_oracle_weights(path)

    assert w["exact_outcome"] == pytest.approx(0.75)
    assert w["trace"] == pytest.approx(0.05)
    # Unlisted keys keep their defaults exactly.
    for k, v in DEFAULT_COMPOSITE_ORACLE_WEIGHTS.items():
        if k in ("exact_outcome", "trace"):
            continue
        assert w[k] == pytest.approx(v), k


def test_loader_returns_defaults_when_section_missing():
    """A YAML file with no composite_oracle_weights key returns defaults."""
    from knowlytix.harness.testing import (
        DEFAULT_COMPOSITE_ORACLE_WEIGHTS,
        load_composite_oracle_weights,
    )

    with tempfile.TemporaryDirectory() as td:
        path = _write_yaml(td, "code_severity:\n  fact_incorrect: high\n")
        w = load_composite_oracle_weights(path)

    assert w == DEFAULT_COMPOSITE_ORACLE_WEIGHTS


def test_loader_rejects_unknown_key():
    from knowlytix.harness.testing import load_composite_oracle_weights

    with tempfile.TemporaryDirectory() as td:
        path = _write_yaml(td, (
            "composite_oracle_weights:\n"
            "  bogus: 0.5\n"
        ))
        with pytest.raises(ValueError, match="unknown keys"):
            load_composite_oracle_weights(path)


def test_loader_rejects_non_mapping_top_level():
    from knowlytix.harness.testing import load_composite_oracle_weights

    with tempfile.TemporaryDirectory() as td:
        path = _write_yaml(td, "- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_composite_oracle_weights(path)


def test_loader_rejects_non_mapping_section():
    from knowlytix.harness.testing import load_composite_oracle_weights

    with tempfile.TemporaryDirectory() as td:
        path = _write_yaml(td, (
            "composite_oracle_weights:\n"
            "  - exact_outcome\n"
            "  - 0.5\n"
        ))
        with pytest.raises(ValueError, match="must be a mapping"):
            load_composite_oracle_weights(path)


# ---------------------------------------------------------------------------
# Example YAML — surface contract
# ---------------------------------------------------------------------------

def test_example_yaml_loads_and_validates():
    """The example config shipped with the repo must round-trip cleanly."""
    from knowlytix.harness.testing import load_composite_oracle_weights

    path = os.path.join(
        os.path.dirname(__file__),
        "..", "severity_config.example.yaml",
    )
    assert os.path.isfile(path), path
    w = load_composite_oracle_weights(path)
    # Every weight must be a finite non-negative float.
    for k, v in w.items():
        assert isinstance(v, float), (k, type(v))
        assert v >= 0, (k, v)


# ---------------------------------------------------------------------------
# _oracle_from_type dispatcher
# ---------------------------------------------------------------------------

def test_oracle_from_type_builds_every_known_type():
    """Every key in DEFAULT_COMPOSITE_ORACLE_WEIGHTS must be dispatchable."""
    from knowlytix.harness.testing import DEFAULT_COMPOSITE_ORACLE_WEIGHTS
    from knowlytix.harness.testing.oracles import _oracle_from_type

    with tempfile.TemporaryDirectory() as td:
        store = _StubStore(td)
        for otype in DEFAULT_COMPOSITE_ORACLE_WEIGHTS:
            oracle = _oracle_from_type(otype, store, environment=None)
            assert oracle.oracle_type == otype, otype


def test_oracle_from_type_rejects_unknown():
    from knowlytix.harness.testing.oracles import _oracle_from_type

    with pytest.raises(ValueError, match="Unknown oracle type"):
        _oracle_from_type("not_a_real_oracle", None, None)


# ---------------------------------------------------------------------------
# CompositeOracle.from_weights
# ---------------------------------------------------------------------------

def test_from_weights_skips_zero_weight_entries():
    """Zero-weight oracles must not be instantiated."""
    from knowlytix.harness.testing.oracles import CompositeOracle

    with tempfile.TemporaryDirectory() as td:
        store = _StubStore(td)
        oracle = CompositeOracle.from_weights(
            {
                "exact_outcome": 0.5,
                "constraint": 0.5,
                "trace": 0.0,  # skipped
                "acceptable_set": 0.0,  # skipped
                "human_reviewed": 0.0,  # skipped
                "hallucination": 0.0,  # skipped
                "side_effect": 0.0,  # skipped
            },
            store=store,
        )
    active = [o.oracle_type for o, _ in oracle._oracles]
    assert sorted(active) == ["constraint", "exact_outcome"]


def test_from_weights_preserves_weight_values():
    from knowlytix.harness.testing.oracles import CompositeOracle

    with tempfile.TemporaryDirectory() as td:
        store = _StubStore(td)
        oracle = CompositeOracle.from_weights(
            {"exact_outcome": 0.7, "constraint": 0.2, "trace": 0.1,
             "acceptable_set": 0.0, "human_reviewed": 0.0,
             "hallucination": 0.0, "side_effect": 0.0},
            store=store,
        )
    weights = {o.oracle_type: w for o, w in oracle._oracles}
    assert weights["exact_outcome"] == pytest.approx(0.7)
    assert weights["constraint"] == pytest.approx(0.2)
    assert weights["trace"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Audit-trail reconstruction invariant
# ---------------------------------------------------------------------------

def test_audit_trail_reconstructs_composite_confidence():
    """The reported confidence must equal sum(c_i * w_i) / sum(w_i).

    This is the user-facing guarantee: a reviewer can check the
    aggregate by hand from the raw_scores triples.  Any divergence
    would signal a bug in CompositeOracle.evaluate().
    """
    from knowlytix.harness.testing.oracles import CompositeOracle

    with tempfile.TemporaryDirectory() as td:
        store = _StubStore(td)
        oracle = CompositeOracle.from_weights(
            {"exact_outcome": 0.6, "constraint": 0.3, "trace": 0.1,
             "acceptable_set": 0.0, "human_reviewed": 0.0,
             "hallucination": 0.0, "side_effect": 0.0},
            store=store,
            threshold=0.5,
            fail_on_any_critical=False,
        )
        # Use a permissive scenario so sub-oracles return nontrivial
        # confidences we can recombine.
        scenario = _scenario(
            ground_truth="High",
            constraints=[{"type": "must_contain", "value": "High"}],
            trace_pattern="",
            expected_tool_sequence=[],
        )
        verdict = oracle.evaluate(scenario, _trace("The rating is High."))

    # Reconstruct the weighted confidence from the audit triples.
    scores = verdict.raw_scores
    types = ["exact_outcome", "constraint", "trace"]
    numer = sum(scores[f"{t}.confidence"] * scores[f"{t}.weight"]
                for t in types)
    denom = sum(scores[f"{t}.weight"] for t in types)
    reconstructed = numer / denom
    assert reconstructed == pytest.approx(verdict.confidence, abs=1e-9)


def test_audit_trail_exposes_passed_as_float():
    """Each sub-oracle's passed flag must appear as 0.0 / 1.0."""
    from knowlytix.harness.testing.oracles import CompositeOracle

    with tempfile.TemporaryDirectory() as td:
        store = _StubStore(td)
        oracle = CompositeOracle.from_weights(
            {"exact_outcome": 0.5, "constraint": 0.5,
             "trace": 0.0, "acceptable_set": 0.0,
             "human_reviewed": 0.0, "hallucination": 0.0,
             "side_effect": 0.0},
            store=store,
        )
        scenario = _scenario(ground_truth="High")
        verdict = oracle.evaluate(scenario, _trace("High is the answer."))

    for t in ("exact_outcome", "constraint"):
        flag = verdict.raw_scores[f"{t}.passed"]
        assert flag in (0.0, 1.0), (t, flag)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
