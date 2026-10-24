# SPDX-License-Identifier: Apache-2.0
"""Design quality metrics: phi_p, min distance, imbalance."""

import numpy as np
import torch


def compute_quality(raw, cardinalities, p=50):
    """Compute quality metrics for a continuous design.

    Args:
        raw: (n, m) array in [0, 1]^m.
        cardinalities: (m,) array of factor cardinalities.
        p: phi_p exponent.

    Returns:
        dict with phi_p, min_dist, mean_nn, mir_avg, mir_max, empty_cells.
    """
    cards = np.asarray(cardinalities, dtype=np.float64)
    weights = cards / cards.mean()
    n, m = raw.shape

    X = torch.tensor(raw, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    Xw = X * w.unsqueeze(0)

    D = torch.cdist(Xw, Xw)
    D_safe = D.clone()
    D_safe.clamp_(min=1e-30)
    D_safe.log_()
    D_safe.mul_(-p)
    D_safe.fill_diagonal_(float("-inf"))
    import math
    log_phi = (torch.logsumexp(D_safe.reshape(-1), 0) - math.log(2.0)) / p
    phi_p = log_phi.exp().item()

    D.fill_diagonal_(float("inf"))
    min_dist = D.min().item()
    mean_nn = D.min(dim=1).values.mean().item()

    levels = np.floor(raw * cards[np.newaxis, :]).astype(int)
    levels = np.minimum(levels, (cards - 1).astype(int)[np.newaxis, :])

    mirs = []
    empty = 0
    total = 0
    for j in range(m):
        c = int(cards[j])
        counts = np.bincount(levels[:, j], minlength=c)
        mirs.append((counts.max() - counts.min()) / (n / c))
        empty += int(np.sum(counts == 0))
        total += c

    return {
        "phi_p": phi_p,
        "min_dist": min_dist,
        "mean_nn": mean_nn,
        "mir_avg": float(np.mean(mirs)),
        "mir_max": float(np.max(mirs)),
        "empty_cells": empty,
        "total_cells": total,
        "coverage": 1.0 - empty / total if total > 0 else 1.0,
    }
