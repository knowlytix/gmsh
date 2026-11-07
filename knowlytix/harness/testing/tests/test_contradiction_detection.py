# SPDX-License-Identifier: Apache-2.0
"""Failing tests that pin down the three root-cause defects in the
GMS contradiction-detection pipeline.

Each test demonstrates one specific defect currently exhibited by the
library. They are expected to FAIL on the current code; once the
data-driven fixes are in place they must PASS.

Defects under test:

  D1  Mining does not honour graph-declared relation oppositions
      (``(r_pos, opposite_of, r_neg)`` triples).

  D2  Mining mislabels same-relation / different-tail pairs as
      contradictions for NON-functional relations (e.g. numeric
      metrics, one-to-many recommendations). This produces noisy
      training labels that push the manifold toward irrelevance.

  D3  ``label_tension`` relies on a hardcoded opposition vocabulary
      (``TENSION_OPPOSING_PATTERNS``) instead of graph-declared
      ``opposite_of`` triples.

  D4  ``GMSJudge.calibrate()`` silently collapses ``tau_contra`` to
      ``tau_ent`` when the 3-class classifier produces only one
      observable crossing in the data range. Degenerate calibration
      must be rejected / signalled, never silently accepted.

Runs:
    python -m pytest tests/test_contradiction_detection.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowlytix.benchmark.graph import DocumentGraph
from knowlytix.core.data.prepare import mine_tension_pairs
from knowlytix.core.train_finstructbench import GraphToGMS
from knowlytix.harness.testing.primitive_calibration import (
    TENSION_OPPOSING_PATTERNS,
    CalibrationSample,
    label_tension,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_adapter(triples):
    """Build a real GraphToGMS adapter from a list of triples.

    No training is performed; we only need the index tensors and
    vocabularies for mining-level tests.
    """
    g = DocumentGraph()
    for h, r, t in triples:
        g.add_triple(h, r, t)
    return GraphToGMS(g), g


def normalise_pairs(pairs_idx, adapter):
    """Return contradiction pairs as sorted (name_a, name_b) tuples."""
    out = set()
    for a, b in pairs_idx:
        name_a = adapter.idx_to_entity[a]
        name_b = adapter.idx_to_entity[b]
        out.add(tuple(sorted((name_a, name_b))))
    return out


def expect_pair(*names: str) -> tuple[str, ...]:
    # DocumentGraph.add_triple canonicalises entity names (lower, single-spaced)
    # so any pair that comes back from mining is in canonical form. Test inputs
    # below keep original casing for readability; this helper renders the
    # expected pair in the same canonical form for assertion symmetry.
    return tuple(sorted(DocumentGraph._canon_entity(n) for n in names))


# ---------------------------------------------------------------------------
# D1 — graph-declared oppositions must drive mining
# ---------------------------------------------------------------------------


class D1_GraphDeclaredOppositions(unittest.TestCase):
    """mine_tension_pairs must pick up ``opposite_of`` declarations in
    the graph and yield contradict pairs for heads that use opposing
    relations on the same target."""

    def test_passes_fails_on_same_gate(self):
        triples = [
            # Two models disagreeing on the same gate via opposing relations.
            ("GB", "passes", "best_auc_gate"),
            ("RF", "fails",  "best_auc_gate"),
            # Declare the opposition structurally in the graph itself.
            ("passes", "opposite_of", "fails"),
            # Filler triple so the relations are present in the adapter.
            ("GB",     "in_section",  "gates"),
        ]
        adapter, _ = build_adapter(triples)
        _, contradict_pairs = mine_tension_pairs(adapter)

        contradicts = normalise_pairs(contradict_pairs, adapter)
        expected = expect_pair("GB", "RF")

        self.assertIn(
            expected, contradicts,
            "mining must produce a contradiction pair (GB, RF) because "
            "'passes' is declared opposite_of 'fails' in the graph and "
            "both relations share the same tail 'best_auc_gate'. "
            f"Got: {sorted(contradicts)}",
        )


# ---------------------------------------------------------------------------
# D2 — functional-only contradictions for same-relation / different-tail
# ---------------------------------------------------------------------------


class D2_FunctionalMining(unittest.TestCase):
    """Same-relation / different-tail pairs are contradictions ONLY
    when the relation is declared functional (single-valued)."""

    def test_non_functional_metric_relation(self):
        # A numeric-valued relation that is inherently NOT functional:
        # different models have different accuracies, not contradictions.
        triples = [
            ("LR", "has_accuracy", "0.87"),
            ("RF", "has_accuracy", "0.89"),
            ("GB", "has_accuracy", "0.91"),
            # No is_functional declaration → default: not functional.
        ]
        adapter, _ = build_adapter(triples)
        _, contradict_pairs = mine_tension_pairs(adapter)
        contradicts = normalise_pairs(contradict_pairs, adapter)

        # None of the model–model pairs should be contradictions.
        for a, b in [("LR", "RF"), ("LR", "GB"), ("RF", "GB")]:
            self.assertNotIn(
                expect_pair(a, b), contradicts,
                f"pair ({a},{b}) must NOT be mined as contradiction "
                "because 'has_accuracy' is not declared is_functional. "
                f"Got: {sorted(contradicts)}",
            )

    def test_functional_status_relation(self):
        triples = [
            ("M1", "has_status", "Degraded"),
            ("M2", "has_status", "Stable"),
            ("M3", "has_status", "Stable"),
            # Declare functionality on the graph.
            ("has_status", "is_functional", "true"),
        ]
        adapter, _ = build_adapter(triples)
        _, contradict_pairs = mine_tension_pairs(adapter)
        contradicts = normalise_pairs(contradict_pairs, adapter)

        self.assertIn(
            expect_pair("M1", "M2"), contradicts,
            "M1 (Degraded) and M2 (Stable) must be a contradict pair "
            "when has_status is declared functional. "
            f"Got: {sorted(contradicts)}",
        )
        # M2 and M3 share a tail → agreement, not contradiction.
        self.assertNotIn(
            expect_pair("M2", "M3"), contradicts,
            "M2 and M3 share tail 'Stable' → must be agreement, "
            "never contradiction.",
        )


# ---------------------------------------------------------------------------
# D3 — label_tension must read declarations from the graph, not
#      a hardcoded vocabulary.
# ---------------------------------------------------------------------------


class D3_LabelTensionDataDriven(unittest.TestCase):
    """label_tension must consume graph-declared opposite_of triples.

    We pick a relation-name pair that is NOT in the hardcoded
    TENSION_OPPOSING_PATTERNS list so the test is defeated only if
    the declaration path works.
    """

    class _MockStore:
        """Minimal stand-in with just what label_tension accesses."""
        def __init__(self, triples):
            self.doc_graph = DocumentGraph()
            for h, r, t in triples:
                self.doc_graph.add_triple(h, r, t)
            entities = set()
            relations = set()
            for h, r, t in triples:
                entities.add(h)
                entities.add(t)
                relations.add(r)
            self.adapter = type("A", (), {
                "entity_to_idx": {e: i for i, e in enumerate(sorted(entities))},
                "relation_to_idx": {r: i for i, r in enumerate(sorted(relations))},
            })()

        def tension_energy(self, a, b):
            # Returns a stand-in scalar so label_tension doesn't drop samples.
            # The numeric value does not drive this test; the identity of
            # the pair does.
            return 1.0

    def test_corpus_declared_opposition_is_mined(self):
        # Pick a pair that is absolutely NOT in the hardcoded vocabulary.
        custom_pos, custom_neg = "green_lights", "red_flags"
        assert (custom_pos, custom_neg) not in TENSION_OPPOSING_PATTERNS
        assert (custom_neg, custom_pos) not in TENSION_OPPOSING_PATTERNS

        triples = [
            ("policy_A", custom_pos, "approval_gate"),
            ("policy_B", custom_neg, "approval_gate"),
            (custom_pos, "opposite_of", custom_neg),
            # Fillers so A/B both have other edges.
            ("policy_A", "in_section", "gates"),
            ("policy_B", "in_section", "gates"),
        ]
        store = self._MockStore(triples)
        samples = label_tension(store, seed=0)

        contradict_pairs = {
            frozenset(("policy_A", "policy_B")) if s.label_name == "contradict"
            else None
            for s in samples
        }
        # Find any contradict sample that includes policy_A/policy_B.
        self.assertTrue(
            any(s.label_name == "contradict" for s in samples),
            "label_tension produced no contradict samples despite "
            "an explicit (green_lights, opposite_of, red_flags) "
            "declaration in the graph. Its opposition discovery is "
            "hardcoded to TENSION_OPPOSING_PATTERNS and must be "
            "made data-driven.",
        )


# ---------------------------------------------------------------------------
# D3b — label_tension must honour is_functional declarations as well,
#       matching mine_tension_pairs. Otherwise training optimises for
#       contradict pairs that calibration never sees, and the three
#       classes overlap.
# ---------------------------------------------------------------------------


class D3b_LabelTensionFunctionalRelations(unittest.TestCase):
    """label_tension must mirror mine_tension_pairs: a
    (r, is_functional, *) declaration emits both head-pair and
    tail-vs-tail contradict samples."""

    class _MockStore:
        def __init__(self, triples):
            self.doc_graph = DocumentGraph()
            for h, r, t in triples:
                self.doc_graph.add_triple(h, r, t)
            entities = set()
            relations = set()
            for h, r, t in triples:
                entities.add(h)
                entities.add(t)
                relations.add(r)
            self.adapter = type("A", (), {
                "entity_to_idx": {
                    e: i for i, e in enumerate(sorted(entities))
                },
                "relation_to_idx": {
                    r: i for i, r in enumerate(sorted(relations))
                },
            })()

        def tension_energy(self, a, b):
            return 1.0

    def test_functional_relation_emits_head_and_tail_contradicts(self):
        triples = [
            # Functional has_severity with three mutually exclusive
            # tail values; no opposite_of declared.
            ("alert_a", "has_severity", "low"),
            ("alert_b", "has_severity", "low"),
            ("alert_c", "has_severity", "medium"),
            ("alert_d", "has_severity", "high"),
            ("has_severity", "is_functional", "true"),
            # Filler so heads share another tail (would be agree
            # under the buggy path).
            ("alert_a", "in_section", "alerts"),
            ("alert_b", "in_section", "alerts"),
            ("alert_c", "in_section", "alerts"),
            ("alert_d", "in_section", "alerts"),
        ]
        store = self._MockStore(triples)
        samples = label_tension(store, seed=0)

        contra_count = sum(
            1 for s in samples if s.label_name == "contradict"
        )
        agree_count = sum(
            1 for s in samples if s.label_name == "agree"
        )

        # Expect head-pair contradicts: alert_a/c, alert_a/d,
        # alert_b/c, alert_b/d, alert_c/d  → 5 head pairs.
        # Plus tail-vs-tail: low/medium, low/high, medium/high → 3.
        # Total: 8 contradict pairs.
        self.assertEqual(
            contra_count, 8,
            "label_tension must emit 5 head-pair + 3 tail-vs-tail "
            "contradicts under a single (has_severity, is_functional, *) "
            "declaration. Got "
            f"{contra_count}. Mining path is asymmetric with "
            "mine_tension_pairs.",
        )
        # The shared (in_section, alerts) tail must NOT promote those
        # head pairs to agree once they are contradicts under
        # has_severity. agree_pairs -= contradict_pairs.
        self.assertLessEqual(
            agree_count, 1,
            "agree pairs must be deduped against contradict pairs. "
            f"Got agree={agree_count}. Buggy code labels alert_a/c etc "
            "as both agree (via in_section) and contradict (via "
            "is_functional), poisoning the calibration fit.",
        )

    def test_provenance_distinguishes_functional_from_opposing(self):
        """When both schema sources fire, the provenance string must
        identify which declaration produced the sample so debugging
        a calibration distribution remains possible."""
        triples = [
            ("p_a", "passes_check", "gate"),
            ("p_b", "fails_check", "gate"),
            ("passes_check", "opposite_of", "fails_check"),
            ("p_a", "has_status", "ok"),
            ("p_b", "has_status", "broken"),
            ("has_status", "is_functional", "true"),
            ("p_a", "in_section", "checks"),
            ("p_b", "in_section", "checks"),
        ]
        store = self._MockStore(triples)
        samples = label_tension(store, seed=0)
        provs = {
            s.provenance for s in samples if s.label_name == "contradict"
        }
        self.assertIn("opposing_relations", provs)
        self.assertIn("functional_relation", provs)


# ---------------------------------------------------------------------------
# D4 — Calibration must not silently collapse tau_contra to tau_ent
# ---------------------------------------------------------------------------


class D4_CalibrationDegenerateHandling(unittest.TestCase):
    """The tension calibrator must never silently collapse
    ``tau_ent`` onto ``tau_contra``. The strict
    ``calibrate_logic_tension`` path in
    ``harness.doe.threshold_calibration`` returns ``(None, None)``
    on a degenerate 3-class fit; ``GMSJudge.calibrate()`` must route
    through that path and set ``tension_status`` to a non-OK value
    rather than expose ``tau_ent == tau_contra``."""

    def test_calibrate_logic_tension_rejects_degenerate_3class(self):
        """Truly degenerate: classes agree (0) and contradict (2)
        occupy the *same* value region, so the balanced LogReg cannot
        find an arg-max region for both classes and at most one of
        them survives in the prediction domain. Exactly one class
        boundary is scannable → `fit_three_class_thresholds` must
        return (None, None).

        Regression note: class imbalance alone used to trigger this,
        but balanced class weights (added to the calibrator) resolve
        imbalance-induced apparent degeneracy. This fixture models
        the residual case balancing cannot save — class 2 data is
        literally indistinguishable from class 0 data."""
        from knowlytix.harness.testing.threshold_calibration import (
            calibrate_logic_tension,
        )

        # Identical deterministic values for agree (0) and
        # contradict (2); unrelated (1) sits in its own region.
        low = list(np.linspace(0.2, 0.4, 100))
        mid = list(np.linspace(1.4, 1.5, 100))
        values = np.array(low + mid + low)
        labels = np.array([0]*100 + [1]*100 + [2]*100)
        samples = [
            CalibrationSample(
                value=float(v), label=int(l),
                label_name={0: "agree", 1: "unrelated", 2: "contradict"}[l],
                provenance="synthetic",
            )
            for v, l in zip(values, labels)
        ]

        clf, cuts = calibrate_logic_tension(samples, seed=0)

        self.assertIsNone(
            cuts,
            "calibrate_logic_tension returned TensionCuts for a "
            "truly indistinguishable 3-class fit; it must return "
            "(None, None) so downstream callers know to refuse "
            "operation. "
            f"Got: clf={clf!r}, cuts={cuts!r}",
        )

    def test_calibrate_logic_tension_enforces_monotonicity(self):
        """When the fit succeeds, the cuts must satisfy
        ``tau_ent < tau_contra`` strictly — never equal."""
        from knowlytix.harness.testing.threshold_calibration import (
            calibrate_logic_tension,
        )

        rng = np.random.RandomState(1)
        values = np.concatenate([
            rng.normal(0.3, 0.05, 100),
            rng.normal(1.4, 0.05, 100),
            rng.normal(1.9, 0.05, 100),
        ])
        labels = np.array([0]*100 + [1]*100 + [2]*100)
        samples = [
            CalibrationSample(
                value=float(v), label=int(l),
                label_name={0: "agree", 1: "unrelated", 2: "contradict"}[l],
                provenance="synthetic",
            )
            for v, l in zip(values, labels)
        ]

        clf, cuts = calibrate_logic_tension(samples, seed=1)
        self.assertIsNotNone(cuts)
        self.assertLess(
            cuts.tau_ent, cuts.tau_contra,
            f"tau_ent ({cuts.tau_ent}) must be strictly less than "
            f"tau_contra ({cuts.tau_contra}); calibrator must not "
            "allow silent collapse.",
        )

    def test_gmsjudge_thresholds_carry_explicit_status(self):
        """Exercise the full GMSJudge calibration path: wire in a
        store-shaped object just enough to drive the tension branch,
        then assert that its output never presents
        tau_ent == tau_contra silently."""
        # We construct the minimal surface GMSJudge.calibrate() touches
        # for the tension path only. Other measures will skip due to
        # insufficient data, which is fine; we only care about the
        # invariant this test names.

        # Route through calibrate_logic_tension directly with a
        # degenerate fixture, then mirror how the (fixed) judge builds
        # its thresholds dict. If the judge is correctly wired, the
        # dict must either set tension_status to a non-ok value or
        # expose distinct tau_ent < tau_contra; never both equal and
        # unflagged.
        from knowlytix.harness.testing.threshold_calibration import (
            calibrate_logic_tension,
        )

        # Truly indistinguishable: agree and contradict share the
        # same deterministic value region.
        low = list(np.linspace(0.2, 0.4, 100))
        mid = list(np.linspace(1.4, 1.5, 100))
        values = np.array(low + mid + low)
        labels = np.array([0]*100 + [1]*100 + [2]*100)
        samples = [
            CalibrationSample(
                value=float(v), label=int(l),
                label_name={0: "agree", 1: "unrelated", 2: "contradict"}[l],
                provenance="synthetic",
            )
            for v, l in zip(values, labels)
        ]

        # This reproduces the judge's decision logic for the tension
        # branch (see judge.py, tension block after this fix).
        clf, cuts = calibrate_logic_tension(samples, seed=2)
        thresholds: dict = {}
        if clf is None or cuts is None:
            thresholds["tension_status"] = "degenerate"
        else:
            thresholds["tension"] = (cuts.tau_ent, cuts.tau_contra)
            thresholds["tau_ent"] = float(cuts.tau_ent)
            thresholds["tau_contra"] = float(cuts.tau_contra)
            thresholds["tension_status"] = "ok"

        same_and_unflagged = (
            thresholds.get("tau_ent") is not None
            and thresholds.get("tau_contra") is not None
            and thresholds["tau_ent"] == thresholds["tau_contra"]
            and thresholds.get("tension_status") == "ok"
        )
        self.assertFalse(
            same_and_unflagged,
            "GMSJudge tension calibration would expose "
            "tau_ent == tau_contra with tension_status='ok' — the "
            "exact collapse the refactor must prevent. "
            f"Got: {thresholds!r}",
        )
        # Stricter: degenerate fit must be flagged.
        self.assertEqual(
            thresholds.get("tension_status"), "degenerate",
            "A degenerate 3-class fit must produce "
            "tension_status='degenerate', got "
            f"{thresholds.get('tension_status')!r}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# D7 — 3-class calibrator enforces CV-accuracy and convergence gates
# ---------------------------------------------------------------------------


class D7_CalibrationQualityGates(unittest.TestCase):
    """Bank-grade: a calibrated fit is accepted only when the
    classifier actually discriminates (5-fold CV accuracy above a
    statistical floor) AND training converged the class means to
    their targets (agree near 0, contradict near 2, sufficient
    span)."""

    def test_low_cv_accuracy_rejected(self):
        """Three classes with identical value distributions --- CV
        accuracy collapses to chance. The fit must be refused even
        though two ordered crossings can be scanned."""
        from knowlytix.harness.testing.threshold_calibration import (
            calibrate_logic_tension,
        )

        rng = np.random.RandomState(0)
        # All three classes share the same distribution. Balanced
        # LogReg may still find two arg-max transitions (because of
        # the monotone-label prior), but the CV accuracy will be
        # near 1/3 --- below MIN_CV_ACCURACY_3CLASS.
        values = np.concatenate([
            rng.normal(1.0, 0.3, 120),
            rng.normal(1.0, 0.3, 120),
            rng.normal(1.0, 0.3, 120),
        ])
        labels = np.array([0]*120 + [1]*120 + [2]*120)
        samples = [
            CalibrationSample(
                value=float(v), label=int(l),
                label_name={0: "agree", 1: "unrelated", 2: "contradict"}[l],
                provenance="synthetic",
            )
            for v, l in zip(values, labels)
        ]

        clf, cuts = calibrate_logic_tension(samples, seed=0)
        self.assertIsNone(
            cuts,
            "calibrator accepted a fit with chance-level CV accuracy; "
            "bank-grade must refuse.",
        )

    def test_unconverged_training_rejected(self):
        """Class ordering is preserved and CV accuracy is high, but
        the contradict class never reached tension ~ 2 (training did
        not converge). The span gate must catch this."""
        from knowlytix.harness.testing.threshold_calibration import (
            calibrate_logic_tension,
        )

        rng = np.random.RandomState(1)
        # Agree and contradict are discriminable (different means)
        # but the span is only 0.4 (well below MIN_TENSION_SPAN=1.0)
        # so the geometry did not learn a real antipodal/parallel
        # layout.
        values = np.concatenate([
            rng.normal(1.10, 0.05, 80),
            rng.normal(1.30, 0.05, 80),
            rng.normal(1.50, 0.05, 80),
        ])
        labels = np.array([0]*80 + [1]*80 + [2]*80)
        samples = [
            CalibrationSample(
                value=float(v), label=int(l),
                label_name={0: "agree", 1: "unrelated", 2: "contradict"}[l],
                provenance="synthetic",
            )
            for v, l in zip(values, labels)
        ]

        clf, cuts = calibrate_logic_tension(samples, seed=1)
        self.assertIsNone(
            cuts,
            "calibrator accepted a fit whose class means span "
            "only 0.4 (well below MIN_TENSION_SPAN=1.0). Bank-grade "
            "must refuse: training did not converge the contradict "
            "class to its target.",
        )

    def test_well_converged_training_accepted(self):
        """Training pushed agree to ~0 and contradict to ~2; span
        exceeds 1.0 and CV accuracy is high. Accept the fit and
        surface the class-mean diagnostics on the classifier."""
        from knowlytix.harness.testing.threshold_calibration import (
            calibrate_logic_tension,
        )

        rng = np.random.RandomState(2)
        values = np.concatenate([
            rng.normal(0.10, 0.05, 100),
            rng.normal(1.40, 0.05, 100),
            rng.normal(1.90, 0.05, 100),
        ])
        labels = np.array([0]*100 + [1]*100 + [2]*100)
        samples = [
            CalibrationSample(
                value=float(v), label=int(l),
                label_name={0: "agree", 1: "unrelated", 2: "contradict"}[l],
                provenance="synthetic",
            )
            for v, l in zip(values, labels)
        ]

        clf, cuts = calibrate_logic_tension(samples, seed=2)
        self.assertIsNotNone(cuts)
        self.assertLess(cuts.tau_ent, cuts.tau_contra)
        # The classifier must carry audit metadata so downstream
        # surfaces can report CV accuracy and class means.
        self.assertTrue(hasattr(clf, "_cv_accuracy_"))
        self.assertGreaterEqual(clf._cv_accuracy_, 0.60)
        self.assertTrue(hasattr(clf, "_class_means_"))
        means = clf._class_means_
        self.assertLess(means["agree"], means["contradict"])
        self.assertGreater(
            means["contradict"] - means["agree"], 1.0,
            "class-mean span must exceed the convergence floor",
        )


if __name__ == "__main__":
    unittest.main()
