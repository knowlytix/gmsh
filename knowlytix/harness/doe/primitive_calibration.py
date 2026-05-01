# SPDX-License-Identifier: Apache-2.0
"""Task-appropriate calibration labelers for GMS primitives.

The original ``judge.calibrate()`` used a single real-vs-corrupted-tail
signal for every measure.  That works for geodesic (where "is this a
real triple?" is the natural question) but is the wrong training signal
for tension and holonomy.  Tension answers "do these two entities
agree in polarity?"; holonomy answers "is this multi-hop path
geometrically consistent with the direct route?"  Corrupting a tail
entity does not perturb either quantity in a discriminative way.

This module provides per-primitive labelers that generate samples
from the store using the same structural cue the primitive is meant
to detect.  Each labeler returns a list of ``CalibrationSample``
objects suitable for fitting a scikit-learn classifier.

Pattern mirrors the question generators (``PlausibilityGenerator``,
``LinkPredictionGenerator``): derive labels from graph structure,
no LLM calls, deterministic under a seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CalibrationSample:
    """One labeled measurement for a primitive calibrator.

    value       : raw geometric quantity (e.g. tension energy, defect)
    label       : integer class label
    label_name  : human-readable class name ("agree", "twisted", ...)
    provenance  : short string explaining how the sample was built
    """

    value: float
    label: int
    label_name: str
    provenance: str

    def as_feature(self) -> list[float]:
        return [self.value]


# ---------------------------------------------------------------------------
# Tension
# ---------------------------------------------------------------------------


# Relation-name patterns that encode explicit opposition were
# previously declared here as a module-level constant. That was
# removed for auditability reasons: every opposition must now be
# declared in the corpus itself via ``(r_a, opposite_of, r_b)``
# triples, which ``label_tension`` reads from the graph below.
#
# The symbol is retained as an empty list so existing imports do not
# break, but no code path should append to it. Declare oppositions
# in the corpus markdown instead.
TENSION_OPPOSING_PATTERNS: list[tuple[str, str]] = []

_OPPOSITE_OF_RELATION = "opposite_of"


def _read_opposing_relations(
    triples: list[tuple[str, str, str]],
    relations: set[str],
) -> set[tuple[str, str]]:
    """Extract symmetric ``(r_a, r_b)`` opposition pairs from
    corpus-declared ``(r_a, opposite_of, r_b)`` triples.

    A declaration only counts when both ``r_a`` and ``r_b`` appear
    as relations elsewhere in the graph — otherwise the pair refers
    to entity names that happen to share the relation vocabulary.
    """
    opposing: set[tuple[str, str]] = set()
    for h, r, t in triples:
        if r != _OPPOSITE_OF_RELATION:
            continue
        if h in relations and t in relations and h != t:
            opposing.add((h, t))
            opposing.add((t, h))
    return opposing


def label_tension(
    store: "GMSExpertStore",
    *,
    seed: int = 42,
    max_unrelated: int = 500,
    min_agree_support: int = 2,
) -> list[CalibrationSample]:
    """Build agree / unrelated / contradict pairs from graph structure.

    Labels:
      0  agree       — pairs sharing at least one (relation, target)
      1  unrelated   — random pairs with no shared target and no
                       direct edge between them
      2  contradict  — pairs linked via opposing relations on a
                       shared target, where the opposition is
                       declared in the graph via
                       ``(r_a, opposite_of, r_b)`` triples. Empty
                       when the corpus declares no oppositions.
    """
    triples = list(store.doc_graph.triples)
    entities = list(store.adapter.entity_to_idx.keys())
    relations = set(store.adapter.relation_to_idx.keys())

    opposing_relations = _read_opposing_relations(triples, relations)

    # Schema-declaration relations should not participate in
    # agreement / contradiction mining themselves.
    schema_relations = {_OPPOSITE_OF_RELATION, "is_functional"}

    # Index: (relation, target) -> [heads]
    shared_target: dict[tuple[str, str], list[str]] = defaultdict(list)
    for h, r, t in triples:
        if r in schema_relations:
            continue
        shared_target[(r, t)].append(h)

    # Agree: heads that share at least one (r, target).
    agree_pairs: set[tuple[str, str]] = set()
    for (_, _), heads in shared_target.items():
        uniq = sorted(set(heads))
        if len(uniq) < min_agree_support:
            continue
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                agree_pairs.add(tuple(sorted((uniq[i], uniq[j]))))

    # Contradict: opposing relations on a shared target.
    target_rel_heads: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    for h, r, t in triples:
        if r in schema_relations:
            continue
        target_rel_heads[t][r].add(h)

    contradict_pairs: set[tuple[str, str]] = set()
    for pos, neg in opposing_relations:
        for _, rel_map in target_rel_heads.items():
            for a in rel_map.get(pos, set()):
                for b in rel_map.get(neg, set()):
                    if a != b:
                        contradict_pairs.add(tuple(sorted((a, b))))

    # Contradict dominates agreement. Two heads that share a tail
    # on some non-opposing relation but also carry declared
    # opposition on another relation are contradict pairs, not
    # agreement pairs; labelling them as both fits a classifier
    # against conflicting targets. Mirrors the dedup in
    # ``gms.data.prepare.mine_tension_pairs``.
    agree_pairs -= contradict_pairs

    # Unrelated: random pairs with no direct edge and not already
    # classified as agree or contradict.
    direct_edges: set[tuple[str, str]] = set()
    for h, _, t in triples:
        direct_edges.add(tuple(sorted((h, t))))
    rng = random.Random(seed)
    unrelated_pairs: set[tuple[str, str]] = set()
    attempts = 0
    budget = 40 * max_unrelated
    while len(unrelated_pairs) < max_unrelated and attempts < budget:
        attempts += 1
        if len(entities) < 2:
            break
        a, b = rng.sample(entities, 2)
        key = tuple(sorted((a, b)))
        if (key in direct_edges or key in agree_pairs
                or key in contradict_pairs):
            continue
        unrelated_pairs.add(key)

    samples: list[CalibrationSample] = []

    def _append(pairs: set, label: int, name: str, prov: str) -> None:
        for a, b in pairs:
            te = store.tension_energy(a, b)
            if te is None:
                continue
            samples.append(CalibrationSample(
                value=float(te), label=label, label_name=name,
                provenance=prov,
            ))

    _append(agree_pairs, 0, "agree", "shared_target")
    _append(unrelated_pairs, 1, "unrelated", "random_disjoint")
    _append(contradict_pairs, 2, "contradict", "opposing_relations")

    return samples


# ---------------------------------------------------------------------------
# Holonomy
# ---------------------------------------------------------------------------


def label_holonomy(
    store: "GMSExpertStore",
    *,
    seed: int = 42,
    max_paths: int = 400,
) -> list[CalibrationSample]:
    """Build consistent / twisted path samples.

    Consistent : a real two-hop path (h, r1, m), (m, r2, t) where a
                 direct edge (h, r_direct, t) also exists.  The
                 defect ``||R_{r2} R_{r1} R_{r_direct}^{-1} - I||_F``
                 reflects how cleanly the composition matches the
                 direct relation.
    Twisted    : same (h, m, t) path but r2 is replaced with a random
                 different relation, breaking the geometric composition.

    Labels: 1 = consistent, 0 = twisted.
    """
    triples = list(store.doc_graph.triples)
    relations = list(store.adapter.relation_to_idx.keys())

    triples_by_head: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for h, r, t in triples:
        triples_by_head[h].append((r, t))

    rng = np.random.RandomState(seed)
    samples: list[CalibrationSample] = []
    count = 0

    for h, r_direct, t in triples:
        if h not in triples_by_head:
            continue
        for r1, m in triples_by_head[h]:
            if m in (t, h) or m not in triples_by_head:
                continue
            for r2, tt in triples_by_head[m]:
                if tt != t:
                    continue
                defect_pos = store.check_holonomy([r1, r2], r_direct)
                if defect_pos is None:
                    continue
                samples.append(CalibrationSample(
                    value=float(defect_pos), label=1,
                    label_name="consistent",
                    provenance=f"path:{h}->{r1}->{m}->{r2}->{t}",
                ))
                # Twisted: replace r2 with a random different relation
                alt = [r for r in relations if r != r2]
                if not alt:
                    continue
                fake_r2 = str(rng.choice(alt))
                defect_neg = store.check_holonomy([r1, fake_r2], r_direct)
                if defect_neg is None:
                    continue
                samples.append(CalibrationSample(
                    value=float(defect_neg), label=0,
                    label_name="twisted",
                    provenance=f"swap:{r2}->{fake_r2}",
                ))
                count += 1
                if count >= max_paths:
                    return samples
                break
            if count >= max_paths:
                return samples
    return samples


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def samples_to_arrays(
    samples: list[CalibrationSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert samples to (X, y) ndarrays for scikit-learn."""
    if not samples:
        return np.zeros((0, 1)), np.zeros((0,), dtype=int)
    X = np.array([s.as_feature() for s in samples], dtype=float)
    y = np.array([s.label for s in samples], dtype=int)
    return X, y


def summarize_samples(samples: list[CalibrationSample]) -> dict[str, dict]:
    """Per-class mean/std/count summary for a list of samples."""
    by_class: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        by_class[s.label_name].append(s.value)
    summary: dict[str, dict] = {}
    for name, vals in by_class.items():
        arr = np.array(vals, dtype=float)
        summary[name] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
        }
    return summary
