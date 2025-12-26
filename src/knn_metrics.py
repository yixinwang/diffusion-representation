
"""
KNN graph construction with multiple distance/metric options.

Supports:
- Euclidean distance on input vectors
- Score-function induced local Mahalanobis metric (approx pullback metric)
- "Initial-noise" embedding trick inspired by the observation that (for DDIM)
  generated samples can vary approximately linearly with initial noise at
  small/moderate scales (see arXiv:2502.04670).

Notes
-----
This module is intentionally dependency-light (numpy + torch).
For large N, consider swapping the exact distance computation with FAISS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch


def _pred_noise(diffusion_model, x: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
    """Return epsilon_theta(x,t). Supports MNISTDiffusion (has .model) or bare Unet."""
    if hasattr(diffusion_model, "model") and callable(getattr(diffusion_model, "model")):
        return diffusion_model.model(x, t_batch)
    if callable(diffusion_model):
        return diffusion_model(x, t_batch)
    raise TypeError("diffusion_model must be MNISTDiffusion-like (with .model) or a callable Unet")

Array = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Array) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def _flatten(x: Array) -> Array:
    if isinstance(x, np.ndarray):
        return x.reshape(x.shape[0], -1)
    return x.view(x.shape[0], -1)


@torch.no_grad()
def ddim_initial_noise_embedding(
    diffusion_model,
    x0: torch.Tensor,
    t: int,
) -> torch.Tensor:
    """
    Approximate an "initial noise" (epsilon) embedding for x0 at timestep t:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps
    =>  eps ≈ (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1-alpha_bar_t)

    We synthesize x_t using fresh noise eps, then recover eps.
    This produces a stable embedding (given the same random seed) that can be
    used for neighbor search.

    Parameters
    ----------
    diffusion_model : MNISTDiffusion-like
        Must expose buffers `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`.
    x0 : (N,C,H,W) tensor in [-1,1]
    t : int
        Diffusion timestep.

    Returns
    -------
    eps_hat : (N, C*H*W) tensor
    """
    assert isinstance(x0, torch.Tensor)
    device = x0.device
    t_batch = torch.full((x0.shape[0],), int(t), device=device, dtype=torch.long)
    eps = torch.randn_like(x0)
    a = diffusion_model.sqrt_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    s = diffusion_model.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    x_t = a * x0 + s * eps
    eps_hat = (x_t - a * x0) / (s + 1e-12)
    return eps_hat.view(eps_hat.shape[0], -1)


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """
    Exact pairwise squared Euclidean distances. O(N^2 d).
    X: (N,d)
    Returns: (N,N)
    """
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    XX = np.sum(X * X, axis=1, keepdims=True)  # (N,1)
    D = XX + XX.T - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    return D


@torch.no_grad()
def score_local_metric(
    diffusion_model,
    x: torch.Tensor,
    t: int,
    *,
    jacobian_rank: int = 8,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Build a per-point local metric G_i ≈ J_i^T J_i + eps I,
    where J_i is the Jacobian of the (flattened) predicted noise w.r.t x.
    We approximate J_i^T J_i using a low-rank randomized estimator with
    Jacobian-vector products.

    Parameters
    ----------
    diffusion_model : MNISTDiffusion-like
        Must have `model(x_t, t)` that returns predicted noise with same shape as x.
    x : (N, C, H, W) tensor
        Input point(s). Typically you pass x_t for a fixed t.
    t : int
    jacobian_rank : int
        Number of probe vectors for low-rank approximation.
    eps : float
        Diagonal jitter.

    Returns
    -------
    G : (N, d, d) tensor, where d=C*H*W
        WARNING: This is dxd per point; only use for small d (MNIST) / small N.
    """
    # For MNIST (1,28,28) => d=784; storing N*d*d can still be heavy.
    # Keep this as a reference implementation; for larger d, use a diagonal/low-rank form.
    N = x.shape[0]
    device = x.device
    x = x.requires_grad_(True)
    t_batch = torch.full((N,), int(t), device=device, dtype=torch.long)

    d = int(np.prod(x.shape[1:]))
    probes = torch.randn((jacobian_rank, d), device=device)
    probes = probes / (probes.norm(dim=1, keepdim=True) + 1e-12)

    # compute Jv for each probe, then accumulate (Jv)(Jv)^T in output space,
    # but we need J^T J in input space; we can estimate via v' (J^T J) v = ||J v||^2
    # and then reconstruct full matrix is costly. Here we do exact via autograd loop
    # for MNIST-size: compute gradients of dot(pred, u) for basis u = e_k is worse.
    #
    # Instead: approximate G as diagonal using squared gradients of scalar probes.
    # We'll output diagonal metric as (N,d) to be used in Mahalanobis distance.
    raise NotImplementedError("Use score_diagonal_metric for scalable neighbor search.")


def score_diagonal_metric(
    diffusion_model,
    x_t: torch.Tensor,
    t: int,
    *,
    probes: int = 4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Scalable diagonal approximation to the pullback metric:
        G_i ≈ diag( E_u [ (∂ <eps_pred, u> / ∂x)^2 ] ) + eps

    This uses a small number of random probe vectors u in output space.
    Works for moderate d and N.

    Returns
    -------
    gdiag : (N, d) tensor, positive.
    """
    N = x_t.shape[0]
    device = x_t.device
    x = x_t.detach().requires_grad_(True)
    t_batch = torch.full((N,), int(t), device=device, dtype=torch.long)
    pred = _pred_noise(diffusion_model, x, t_batch)  # predicted noise (epsilon_theta)
    pred_flat = pred.view(N, -1)
    d = pred_flat.shape[1]


    # Convert epsilon prediction to score if diffusion buffers are present:
    # score(x_t,t) ≈ -eps_theta(x_t,t) / sqrt(1 - alpha_bar_t)
    if hasattr(diffusion_model, "sqrt_one_minus_alphas_cumprod"):
        sigma = diffusion_model.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(N, 1, 1, 1)
        sigma = sigma.clamp_min(1e-12)
        pred_flat = (-pred / sigma).view(N, -1)

    gdiag = torch.zeros((N, d), device=device)
    for _ in range(int(probes)):
        u = torch.randn((N, d), device=device)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-12)
        s = (pred_flat * u).sum(dim=1)  # (N,)
        grads = torch.autograd.grad(
            outputs=s.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )[0].view(N, -1)
        gdiag += grads * grads
    gdiag = gdiag / float(probes)
    gdiag = gdiag + eps
    return gdiag.detach()


def mahalanobis_sq_dists_diag(X: np.ndarray, gdiag: np.ndarray) -> np.ndarray:
    """
    Pairwise squared distances using per-point diagonal metric:
        d(i,j)^2 = (x_i-x_j)^T diag(g_i) (x_i-x_j)
    We symmetrize with g_ij = 0.5*(g_i + g_j).

    X: (N,d)
    gdiag: (N,d)
    Returns: (N,N)
    """
    N, d = X.shape
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        diff = X - X[i]  # (N,d)
        gij = 0.5 * (gdiag + gdiag[i:i+1])
        D[i] = np.sum(gij * (diff * diff), axis=1)
    return D


def knn_graph(
    X: Array,
    k: int,
    *,
    metric: Literal["euclidean", "score_riemannian_diag", "initial_noise"] = "euclidean",
    diffusion_model=None,
    t: Optional[int] = None,
    candidate_k: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph (indices + squared distances).

    Parameters
    ----------
    X : (N, ...) array/tensor
        Data points.
    k : int
        Number of neighbors (excluding self).
    metric :
        - "euclidean": exact Euclidean in flattened X.
        - "score_riemannian_diag": uses a diagonal pullback metric estimated from
          diffusion_model at timestep t on x_t (you provide X as x_t).
        - "initial_noise": maps x0 -> eps_hat embedding at timestep t (requires diffusion_model)
          then does Euclidean in that embedding.
    diffusion_model :
        Needed for score-based metrics.
    t :
        Needed for score-based metrics.
    candidate_k :
        If set and metric is expensive, first preselect candidate_k by Euclidean and
        then rerank using the chosen metric (cheap-to-expensive two-stage).
    device :
        Torch device to use.

    Returns
    -------
    nn_idx : (N,k) int64
    nn_d2  : (N,k) float64 squared distances
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_k is not None and candidate_k < k:
        raise ValueError("candidate_k must be >= k")

    if metric in ("score_riemannian_diag", "initial_noise") and (diffusion_model is None or t is None):
        raise ValueError(f"metric='{metric}' requires diffusion_model and t")

    # Prepare embedding Z (N,d) in numpy
    if metric == "initial_noise":
        if not isinstance(X, torch.Tensor):
            X_t = torch.as_tensor(_to_numpy(X))
        else:
            X_t = X
        if device is not None:
            X_t = X_t.to(device)
            diffusion_model = diffusion_model.to(device)
        Z_t = ddim_initial_noise_embedding(diffusion_model, X_t, int(t))
        Z = _to_numpy(Z_t)
    else:
        Z = _to_numpy(_flatten(X))

    # Euclidean preselection if requested
    if candidate_k is not None:
        D_e = pairwise_sq_dists(Z)
        nn0 = np.argsort(D_e, axis=1)[:, 1:candidate_k+1]
    else:
        nn0 = None

    if metric == "euclidean" or metric == "initial_noise":
        D = pairwise_sq_dists(Z)
    elif metric == "score_riemannian_diag":
        # compute diagonal metric gdiag in torch on X (assumed x_t)
        if not isinstance(X, torch.Tensor):
            X_t = torch.as_tensor(_to_numpy(X))
        else:
            X_t = X
        if device is not None:
            X_t = X_t.to(device)
            diffusion_model = diffusion_model.to(device)
        gdiag_t = score_diagonal_metric(diffusion_model, X_t, int(t))
        gdiag = _to_numpy(gdiag_t)
        D = mahalanobis_sq_dists_diag(Z, gdiag)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    N = D.shape[0]
    nn_idx = np.empty((N, k), dtype=np.int64)
    nn_d2 = np.empty((N, k), dtype=np.float64)

    if nn0 is None:
        order = np.argsort(D, axis=1)[:, 1:k+1]
        nn_idx[:] = order
        nn_d2[:] = np.take_along_axis(D, order, axis=1)
    else:
        # rerank within candidates only (include self protection)
        for i in range(N):
            cand = nn0[i]
            cand_d = D[i, cand]
            pick = np.argsort(cand_d)[:k]
            nn_idx[i] = cand[pick]
            nn_d2[i] = cand_d[pick]

    return nn_idx, nn_d2
