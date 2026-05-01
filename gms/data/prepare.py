# SPDX-License-Identifier: LicenseRef-Knowlytix-GMS-EULA
"""GMS training data preparation.

Prepares all data artifacts required by the GMS loss function:
  - Triple dataset with type-aware hard negative sampling
  - Tension pairs (agreement / contradiction) mined from KG structure
  - Path training data (relation triangles) for holonomy consistency

This module is the single source of truth for data creation.
The training loop should call ``prepare_training_data`` and consume
the returned ``TrainingData`` bundle without further data logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Training data bundle
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    """All data artifacts required for GMS training.

    Attributes:
        dataset: PyTorch Dataset yielding (h, r, t, neg_tails) per triple.
        agree_pairs: List of (entity_a, entity_b) pairs sharing a (relation, tail).
        contra_pairs: List of (entity_a, entity_b) pairs with different tails
            for the same relation, plus tail-vs-tail semantic opposites.
        relation_triangles: List of (r_ij, r_jk, r_ik) tuples for holonomy
            consistency training.  Each triangle means the composition
            R_jk @ R_ij should approximate R_ik.
        n_entities: Total entity count (for random irrelevance sampling).
    """
    dataset: Dataset
    agree_pairs: list[tuple[int, int]]
    contra_pairs: list[tuple[int, int]]
    relation_triangles: list[tuple[int, int, int]]
    n_entities: int


# ---------------------------------------------------------------------------
# Triple dataset with type-aware negative sampling
# ---------------------------------------------------------------------------

class TripleDataset(Dataset):
    """Dataset of (head, relation, tail) with type-aware negative sampling.

    Negative sampling strategy (per positive triple):
      - 1 self-negative: head entity as tail.  Forces R_r(v_h) away from v_h,
        preventing the identity rotor from dominating link prediction.
      - (num_neg - 1) // 2 hard negatives: sampled from the same relation's
        observed tail pool, excluding the true tail.  These are entities the
        model could plausibly confuse with the correct answer.
      - Remaining slots: uniform random negatives for broad coverage.

    Why this mix:
      - Pure random negatives are trivially easy in a typed KG (e.g., a model
        name will never be confused with a numeric threshold).
      - Pure hard negatives from the same type pool can starve gradient on
        the broader embedding manifold.
      - The 50/50 split balances discrimination sharpness with manifold coverage.
    """

    def __init__(self, adapter, num_neg: int = 32):
        self.adapter = adapter
        self.num_neg = num_neg
        self.n_entities = adapter.num_entities

        # Precompute per-relation tail lists for O(1) sampling
        self._rel_tails: dict[int, list[int]] = {}
        for r_idx, tail_set in adapter.tail_per_relation.items():
            self._rel_tails[r_idx] = list(tail_set)

    def __len__(self):
        return len(self.adapter.heads)

    def __getitem__(self, idx):
        h = self.adapter.heads[idx]
        r = self.adapter.relations[idx]
        t = self.adapter.tails[idx]

        r_int = r.item()
        t_int = t.item()
        h_int = h.item()

        # Reserve 1 slot for the self-negative
        n_hard = (self.num_neg - 1) // 2
        n_rand = self.num_neg - 1 - n_hard

        # Hard negatives: same relation's tail pool, excluding true tail
        rel_tails = self._rel_tails.get(r_int)
        if rel_tails and len(rel_tails) > 1:
            filtered = [x for x in rel_tails if x != t_int]
            if filtered:
                hard_idx = torch.randint(0, len(filtered), (n_hard,))
                hard_negs = torch.tensor(
                    [filtered[i] for i in hard_idx], dtype=torch.long
                )
            else:
                hard_negs = torch.randint(0, self.n_entities, (n_hard,))
        else:
            hard_negs = torch.randint(0, self.n_entities, (n_hard,))

        rand_negs = torch.randint(0, self.n_entities, (n_rand,))
        self_neg = torch.tensor([h_int], dtype=torch.long)

        neg_tails = torch.cat([self_neg, hard_negs, rand_negs])
        return h, r, t, neg_tails


# ---------------------------------------------------------------------------
# Tension pair mining
# ---------------------------------------------------------------------------

def mine_tension_pairs(
    adapter,
    max_per_group: int = 3,
    max_tail_values: int = 10,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Mine agreement and contradiction pairs from KG structure.

    The tension energy E(u_a, u_b) in u-space captures logical relationships:
      - Agreement  (E -> 0):   entities sharing the same (relation, tail) value
      - Contradiction (E -> 2): entities with *different* tails for the same
        relation, plus the tail entities themselves (semantic opposites)
      - Irrelevance (E -> sqrt2): random pairs (sampled at training time)

    Args:
        adapter: GraphToGMS instance with heads, relations, tails tensors and
            entity/relation vocabularies.
        max_per_group: Cap on entities sampled per tail group to prevent
            quadratic explosion on large groups.
        max_tail_values: Maximum distinct tail values per relation for
            tail-vs-tail contradiction mining.

    Returns:
        Tuple of (agreement_pairs, contradiction_pairs) where each pair is
        (entity_idx_a, entity_idx_b) with a < b (deduplicated).
    """
    # Group: relation -> tail -> [head entities]
    rel_tail_to_heads: dict[int, dict[int, list[int]]] = {}
    for h, r, t in zip(
        adapter.heads.tolist(),
        adapter.relations.tolist(),
        adapter.tails.tolist(),
    ):
        rel_tail_to_heads.setdefault(r, {}).setdefault(t, []).append(h)

    agreement: list[tuple[int, int]] = []
    contradiction: list[tuple[int, int]] = []

    for r, tail_groups in rel_tail_to_heads.items():
        groups = list(tail_groups.values())
        tails = list(tail_groups.keys())

        # Agreement: entities sharing the same (relation, tail) value
        # e.g., MDL-001 and MDL-003 both have has_status = Degraded
        for group in groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    agreement.append((group[i], group[j]))

        # Contradiction (head-head): entities with different tails
        # e.g., MDL-001 (Degraded) vs MDL-004 (Stable) for has_status
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                for a in groups[i][:max_per_group]:
                    for b in groups[j][:max_per_group]:
                        contradiction.append((a, b))

        # Contradiction (tail-tail): the tail entities themselves
        # e.g., (Degraded, Stable) — these are semantic opposites within
        # the same relation's value space
        if 1 < len(tails) <= max_tail_values:
            for i in range(len(tails)):
                for j in range(i + 1, len(tails)):
                    contradiction.append((tails[i], tails[j]))

    # Deduplicate with canonical ordering (a < b)
    agreement = list({(min(a, b), max(a, b)) for a, b in agreement})
    contradiction = list({(min(a, b), max(a, b)) for a, b in contradiction})

    return agreement, contradiction


# ---------------------------------------------------------------------------
# Relation triangle mining (for holonomy / path consistency)
# ---------------------------------------------------------------------------

def mine_relation_triangles(adapter) -> list[tuple[int, int, int]]:
    """Find relation triangles (r_ij, r_jk, r_ik) from the KG.

    A triangle exists when entity i connects to j via r1, j connects to k
    via r2, and i connects to k via r3.  The holonomy loss penalises
    ||R_{r2} R_{r1} - R_{r3}||_F, enforcing relational composition
    consistency on the manifold.

    Returns:
        List of (r_ij, r_jk, r_ik) relation-index tuples.
    """
    # Build adjacency: head -> [(relation, tail)]
    adj: dict[int, list[tuple[int, int]]] = {}
    for h, r, t in zip(
        adapter.heads.tolist(),
        adapter.relations.tolist(),
        adapter.tails.tolist(),
    ):
        adj.setdefault(h, []).append((r, t))

    # Also build reverse map: tail -> [(relation, head)] for closing triangles
    tail_adj: dict[int, list[tuple[int, int]]] = {}
    for h, r, t in zip(
        adapter.heads.tolist(),
        adapter.relations.tolist(),
        adapter.tails.tolist(),
    ):
        tail_adj.setdefault(t, []).append((r, h))

    triangles: set[tuple[int, int, int]] = set()

    # For each edge (i -r1-> j), find j -r2-> k and i -r3-> k
    for i, edges_i in adj.items():
        for r1, j in edges_i:
            if j not in adj:
                continue
            for r2, k in adj[j]:
                if k == i:
                    continue  # skip self-loops
                # Check if i -r3-> k exists
                for r3, k2 in adj.get(i, []):
                    if k2 == k and r3 != r1:
                        triangles.add((r1, r2, r3))

    return list(triangles)


# ---------------------------------------------------------------------------
# Tension batch sampler (used inside the training loop)
# ---------------------------------------------------------------------------

def sample_tension_batch(
    agree_pairs: list[tuple[int, int]],
    contra_pairs: list[tuple[int, int]],
    n_entities: int,
    batch_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a balanced batch of tension pairs for u-space training.

    Returns equal thirds of agreement (label 0), contradiction (label 2),
    and irrelevance (label 1) pairs.

    Args:
        agree_pairs: Mined agreement pairs.
        contra_pairs: Mined contradiction pairs.
        n_entities: Total entities for random irrelevance sampling.
        batch_size: Total pairs to sample.

    Returns:
        Tuple of (entity_a, entity_b, labels) tensors.
    """
    n_agree = batch_size // 3
    n_contra = batch_size // 3
    n_irrel = batch_size - n_agree - n_contra

    pair_a: list[int] = []
    pair_b: list[int] = []
    labels: list[int] = []

    # Agreement pairs (label 0)
    if agree_pairs:
        idxs = torch.randint(0, len(agree_pairs), (n_agree,))
        for i in idxs:
            a, b = agree_pairs[i.item()]
            pair_a.append(a)
            pair_b.append(b)
            labels.append(0)

    # Contradiction pairs (label 2)
    if contra_pairs:
        idxs = torch.randint(0, len(contra_pairs), (n_contra,))
        for i in idxs:
            a, b = contra_pairs[i.item()]
            pair_a.append(a)
            pair_b.append(b)
            labels.append(2)

    # Irrelevance pairs (label 1) — random entity pairs
    for _ in range(n_irrel):
        pair_a.append(random.randint(0, n_entities - 1))
        pair_b.append(random.randint(0, n_entities - 1))
        labels.append(1)

    return (
        torch.tensor(pair_a, dtype=torch.long),
        torch.tensor(pair_b, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Top-level preparation function
# ---------------------------------------------------------------------------

def prepare_training_data(
    adapter,
    num_neg: int = 32,
    max_per_group: int = 3,
    max_tail_values: int = 10,
) -> TrainingData:
    """Prepare all training data artifacts from a GraphToGMS adapter.

    This is the main entry point. Call once before training begins.

    Args:
        adapter: GraphToGMS instance.
        num_neg: Negative samples per positive triple.
        max_per_group: Cap for tension pair mining (see mine_tension_pairs).
        max_tail_values: Cap for tail-vs-tail contradiction mining.

    Returns:
        TrainingData bundle ready for the training loop.
    """
    dataset = TripleDataset(adapter, num_neg=num_neg)
    agree_pairs, contra_pairs = mine_tension_pairs(
        adapter,
        max_per_group=max_per_group,
        max_tail_values=max_tail_values,
    )
    triangles = mine_relation_triangles(adapter)

    print("  Training data prepared:")
    print(f"    Triples:              {len(dataset)}")
    print(f"    Agreement pairs:      {len(agree_pairs)}")
    print(f"    Contradiction pairs:  {len(contra_pairs)}")
    print(f"    Relation triangles:   {len(triangles)}")

    return TrainingData(
        dataset=dataset,
        agree_pairs=agree_pairs,
        contra_pairs=contra_pairs,
        relation_triangles=triangles,
        n_entities=adapter.num_entities,
    )
