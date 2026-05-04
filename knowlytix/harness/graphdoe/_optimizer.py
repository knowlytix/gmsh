# SPDX-License-Identifier: Apache-2.0
"""GPU-accelerated phi_p space-filling design optimization.

Minimizes phi_p over free point locations in [0,1]^m using PyTorch.
Analytic gradients (no autograd) keep peak memory at O(n^2).
"""

import math
import numpy as np
import torch
from time import perf_counter


def _loss_and_grad(Z, p, weights=None):
    X = torch.sigmoid(Z)
    n = X.shape[0]

    Xw = X * weights.unsqueeze(0) if weights is not None else X

    M = torch.cdist(Xw, Xw)
    M.clamp_(min=1e-30)
    M.log_()
    M.mul_(-p)
    M.fill_diagonal_(float("-inf"))

    log_phi_full = torch.logsumexp(M.reshape(-1), dim=0)
    loss = (log_phi_full - math.log(2.0)) / p

    log_phi_sum = log_phi_full - math.log(2.0)
    M.mul_((p + 2.0) / p)
    M.sub_(log_phi_sum)
    M.exp_()
    M.fill_diagonal_(0.0)

    G_row_sum = M.sum(dim=1, keepdim=True)
    grad_X = torch.mm(M, X) - G_row_sum * X
    if weights is not None:
        grad_X = grad_X * (weights.unsqueeze(0) ** 2)
    grad_Z = grad_X * X * (1.0 - X)

    return loss, grad_Z


def _get_device():
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def optimize(
    n,
    m,
    *,
    p_final=50,
    p_start=5,
    n_stages=5,
    iters_per_stage=None,
    lr=None,
    use_lbfgs=True,
    device=None,
    dtype=torch.float64,
    seed=None,
    X_init=None,
    verbose=False,
    dim_weights=None,
):
    """Optimize n points in [0,1]^m for space-filling (phi_p criterion).

    Returns:
        (X, phi_p, info) where X is (n, m) ndarray in [0,1]^m.
    """
    if device is None:
        device = _get_device()
    if lr is None:
        lr = 1.0 if use_lbfgs else 0.01
    if iters_per_stage is None:
        iters_per_stage = 200 if use_lbfgs else 500
    if seed is not None:
        torch.manual_seed(seed)

    w_t = None
    if dim_weights is not None:
        w_t = torch.tensor(dim_weights, device=device, dtype=dtype)

    if X_init is not None:
        X0 = torch.tensor(X_init, device=device, dtype=dtype).clamp(0.02, 0.98)
    else:
        X0 = torch.rand(n, m, device=device, dtype=dtype) * 0.96 + 0.02
    Z = torch.logit(X0)
    Z.requires_grad_(True)

    ps = np.geomspace(p_start, p_final, n_stages).tolist()
    t0 = perf_counter()

    for stage, p in enumerate(ps):
        if use_lbfgs:
            opt = torch.optim.LBFGS(
                [Z], lr=lr, max_iter=iters_per_stage,
                line_search_fn="strong_wolfe",
            )

            def closure():
                with torch.no_grad():
                    loss, grad = _loss_and_grad(Z, p, weights=w_t)
                    Z.grad = grad
                return loss

            opt.step(closure)
        else:
            opt = torch.optim.Adam([Z], lr=lr)
            for _ in range(iters_per_stage):
                with torch.no_grad():
                    loss, grad = _loss_and_grad(Z, p, weights=w_t)
                    Z.grad = grad
                opt.step()

        if verbose:
            with torch.no_grad():
                X = torch.sigmoid(Z)
                D = torch.cdist(X, X)
                D.fill_diagonal_(float("inf"))
                print(
                    f"  Stage {stage+1}/{n_stages}: p={p:6.1f} | "
                    f"min_dist={D.min().item():.6f}"
                )

    elapsed = perf_counter() - t0

    with torch.no_grad():
        X_final = torch.sigmoid(Z)
        Xw = X_final * w_t.unsqueeze(0) if w_t is not None else X_final
        D = torch.cdist(Xw, Xw)
        D.clamp_(min=1e-30)
        D.log_()
        D.mul_(-p_final)
        D.fill_diagonal_(float("-inf"))
        log_phi_full = torch.logsumexp(D.reshape(-1), dim=0)
        log_phi_sum = log_phi_full - math.log(2.0)
        phi_p = (log_phi_sum / p_final).exp().item()
        D = torch.cdist(Xw, Xw)
        D.fill_diagonal_(float("inf"))
        min_dist = D.min().item()

    X_np = X_final.cpu().numpy()

    if verbose:
        print(
            f"  phi_p={phi_p:.6f} | min_dist={min_dist:.6f} | "
            f"{elapsed:.2f}s | {device}"
        )

    return X_np, phi_p, {"time": elapsed, "min_dist": min_dist, "device": str(device)}
