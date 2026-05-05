# SPDX-License-Identifier: Apache-2.0
"""Data-driven calibration of the support gate's minimum thresholds.

The runtime gate's hardcoded ``min_triples=2, min_relations=1`` is
a heuristic floor. For high-stakes deployment, the floor must be
*calibrated*: at what support level does verdict reliability drop?
This module sweeps candidate floors against the same labeled
cohort the calibration uses, and selects the smallest floor that
preserves a target accuracy on non-abstained verdicts.

The result is stored on the bundle. The runtime gate reads it and
overrides the heuristic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.governance.governance_bundle import EntitySupport


_OPPOSITE_OF_RELATION = "opposite_of"


@dataclass(frozen=True)
class CalibratedSupportThresholds:
    """Floors selected from labeled-cohort support distribution.

    ``min_triples``/``min_relations`` are the smallest values such that
    pairs above the floor maintain ``coverage_accuracy`` >= the
    configured target.  ``n_pairs`` / ``coverage_pairs`` summarize how
    many labeled pairs the calibration evaluated and how many lay
    above the chosen floor (the rest are routed to abstain).
    """

    min_triples: int
    min_relations: int
    target_accuracy: float
    coverage_accuracy: float
    n_pairs: int
    coverage_pairs: int
    rationale: str

    @property
    def coverage_rate(self) -> float:
        return self.coverage_pairs / self.n_pairs if self.n_pairs else 0.0


def _build_labeled_pairs(
    store: "GMSExpertStore", *, seed: int = 42, max_unrelated: int = 200,
    min_agree_support: int = 2,
):
    """Mirror label_tension's pair-construction so we can attach
    entity names to each labeled sample."""
    import random

    triples = list(store.doc_graph.triples)
    entities = list(store.adapter.entity_to_idx.keys())
    relations = set(store.adapter.relation_to_idx.keys())
    schema = {_OPPOSITE_OF_RELATION, "is_functional"}

    opposing_relations: set[tuple[str, str]] = set()
    for h, r, t in triples:
        if r != _OPPOSITE_OF_RELATION:
            continue
        if h in relations and t in relations and h != t:
            opposing_relations.add((h, t))
            opposing_relations.add((t, h))

    shared_target: dict[tuple[str, str], list[str]] = defaultdict(list)
    for h, r, t in triples:
        if r in schema:
            continue
        shared_target[(r, t)].append(h)

    agree_pairs: set[tuple[str, str]] = set()
    for (_, _), heads in shared_target.items():
        uniq = sorted(set(heads))
        if len(uniq) < min_agree_support:
            continue
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                agree_pairs.add(tuple(sorted((uniq[i], uniq[j]))))

    target_rel_heads: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for h, r, t in triples:
        if r in schema:
            continue
        target_rel_heads[t][r].add(h)

    contradict_pairs: set[tuple[str, str]] = set()
    for pos, neg in opposing_relations:
        for _, rel_map in target_rel_heads.items():
            for a in rel_map.get(pos, set()):
                for b in rel_map.get(neg, set()):
                    if a != b:
                        contradict_pairs.add(tuple(sorted((a, b))))

    agree_pairs -= contradict_pairs

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

    out: list[tuple[str, str, int]] = []
    for a, b in agree_pairs:
        out.append((a, b, 0))
    for a, b in unrelated_pairs:
        out.append((a, b, 1))
    for a, b in contradict_pairs:
        out.append((a, b, 2))
    return out


def calibrate_support_thresholds(
    store: "GMSExpertStore",
    support_index: dict[str, "EntitySupport"],
    tau_ent: float,
    tau_contra: float,
    *,
    target_accuracy: float = 0.85,
    seed: int = 42,
    fallback_min_triples: int = 2,
    fallback_min_relations: int = 1,
) -> CalibratedSupportThresholds:
    """Sweep candidate floors; pick the smallest preserving accuracy.

    Decision under floor T: pairs with min(support_a, support_b) < T
    abstain; pairs above the floor are predicted by tension band.
    Accuracy is measured only on non-abstained pairs.
    """
    pairs = _build_labeled_pairs(store, seed=seed)
    if not pairs:
        return CalibratedSupportThresholds(
            min_triples=fallback_min_triples,
            min_relations=fallback_min_relations,
            target_accuracy=target_accuracy,
            coverage_accuracy=0.0,
            n_pairs=0, coverage_pairs=0,
            rationale="no labeled pairs; using heuristic floor",
        )

    pair_te = []
    pair_min_triples = []
    pair_min_relations = []
    pair_label = []
    for a, b, label in pairs:
        sup_a = support_index.get(a)
        sup_b = support_index.get(b)
        if sup_a is None or sup_b is None:
            continue
        te = store.tension_energy(a, b)
        if te is None:
            continue
        pair_te.append(float(te))
        pair_min_triples.append(min(sup_a.triple_count, sup_b.triple_count))
        pair_min_relations.append(min(sup_a.relation_count, sup_b.relation_count))
        pair_label.append(label)

    if not pair_te:
        return CalibratedSupportThresholds(
            min_triples=fallback_min_triples,
            min_relations=fallback_min_relations,
            target_accuracy=target_accuracy,
            coverage_accuracy=0.0,
            n_pairs=0, coverage_pairs=0,
            rationale="no scoreable pairs; using heuristic floor",
        )

    te_arr = np.asarray(pair_te)
    mt_arr = np.asarray(pair_min_triples)
    mr_arr = np.asarray(pair_min_relations)
    y_arr = np.asarray(pair_label)

    def _predict(te: float) -> int:
        if te <= tau_ent:
            return 0
        if te >= tau_contra:
            return 2
        return 1
    pred_arr = np.asarray([_predict(t) for t in te_arr])
    correct = (pred_arr == y_arr)

    # Candidate floors: every distinct triple count seen in the cohort.
    candidate_triples = sorted(set(mt_arr.tolist()))
    candidate_relations = sorted(set(mr_arr.tolist()))

    best_t = max(fallback_min_triples, candidate_triples[0])
    best_r = max(fallback_min_relations, candidate_relations[0])
    best_acc = 0.0
    best_cov = 0
    rationale = (
        f"sweeping triple floors {candidate_triples} and relation floors "
        f"{candidate_relations} against {len(y_arr)} labeled pairs"
    )

    # Greedy: smallest triple floor that achieves target on triples;
    # then smallest relation floor that preserves it.
    for T in candidate_triples:
        mask = (mt_arr >= T)
        if mask.sum() < 5:
            continue
        acc = correct[mask].mean()
        if acc >= target_accuracy:
            best_t = max(T, fallback_min_triples)
            best_acc = float(acc)
            best_cov = int(mask.sum())
            break

    if best_acc == 0.0:
        # Could not meet target at any floor — pick the floor with
        # highest accuracy among those with sufficient coverage.
        per_floor = []
        for T in candidate_triples:
            mask = (mt_arr >= T)
            if mask.sum() < 5:
                continue
            per_floor.append((float(correct[mask].mean()), T, int(mask.sum())))
        if per_floor:
            best_acc, best_t, best_cov = max(per_floor)
            best_t = max(best_t, fallback_min_triples)
            rationale += (
                f"; target accuracy {target_accuracy:.2f} unattainable "
                f"-- selected best-effort floor"
            )

    # Now sweep relation floors above best_t, preserving accuracy.
    mask_t = (mt_arr >= best_t)
    if mask_t.any():
        for R in candidate_relations:
            mask = mask_t & (mr_arr >= R)
            if mask.sum() < 5:
                continue
            acc = correct[mask].mean()
            if acc >= target_accuracy or R == candidate_relations[0]:
                best_r = max(R, fallback_min_relations)
                best_acc = float(acc)
                best_cov = int(mask.sum())
                if acc >= target_accuracy:
                    break

    return CalibratedSupportThresholds(
        min_triples=int(best_t),
        min_relations=int(best_r),
        target_accuracy=target_accuracy,
        coverage_accuracy=best_acc,
        n_pairs=int(len(y_arr)),
        coverage_pairs=best_cov,
        rationale=rationale,
    )
