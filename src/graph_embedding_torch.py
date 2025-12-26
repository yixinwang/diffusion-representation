from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class GraphEmbedResult:
    Y: np.ndarray
    evals: np.ndarray
    kind: str
    diagnostics: dict


def _lobpcg_call(A: torch.Tensor, k: int, X: torch.Tensor, largest: bool, iters: int, tol: float = 1e-6):
    """Compatibility wrapper: torch.lobpcg API differs across torch versions."""
    # Newer API: maxiter keyword
    try:
        return torch.lobpcg(A, k=k, B=None, X=X, largest=largest, maxiter=iters, tol=tol)
    except TypeError:
        pass
    # Older API: niter keyword
    try:
        return torch.lobpcg(A, k=k, B=None, X=X, largest=largest, niter=iters, tol=tol)
    except TypeError:
        pass
    # Oldest API: no iteration keyword (uses default iters)
    return torch.lobpcg(A, k=k, B=None, X=X, largest=largest, tol=tol)


def build_dense_affinity(nn_idx: np.ndarray, nn_d2: np.ndarray, eps: Optional[float] = None) -> tuple[torch.Tensor, float]:
    """Dense symmetric kNN affinity W (N,N) on CPU (float64)."""
    nn_idx = np.asarray(nn_idx, dtype=np.int64)
    nn_d2 = np.asarray(nn_d2, dtype=np.float64)
    N, k = nn_idx.shape
    if eps is None:
        eps = float(np.median(nn_d2))
        eps = max(eps, 1e-12)

    W = torch.zeros((N, N), dtype=torch.float64)
    rows = np.repeat(np.arange(N), k)
    cols = nn_idx.reshape(-1)
    vals = np.exp(-nn_d2.reshape(-1) / eps)

    W[rows, cols] = torch.from_numpy(vals).to(torch.float64)
    W = torch.maximum(W, W.T)
    W.fill_diagonal_(0.0)
    return W, eps


def laplacian_eigenmap_from_knn_torch(
    nn_idx: np.ndarray,
    nn_d2: np.ndarray,
    out_dim: int = 8,
    eps: Optional[float] = None,
    iters: int = 200,
) -> GraphEmbedResult:
    """
    Laplacian eigenmaps on dense kNN graph using torch.lobpcg.

    L_sym = I - D^{-1/2} W D^{-1/2} (symmetric PSD).
    Return smallest non-trivial eigenvectors (drop the constant eigenvector).
    """
    W, eps_used = build_dense_affinity(nn_idx, nn_d2, eps=eps)
    d = W.sum(dim=1).clamp_min(1e-12)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(d))
    N = W.shape[0]
    L = torch.eye(N, dtype=torch.float64) - D_inv_sqrt @ W @ D_inv_sqrt

    k = min(int(out_dim) + 1, N - 1)
    X = torch.randn((N, k), dtype=torch.float64)

    evals, evecs = _lobpcg_call(L, k=k, X=X, largest=False, iters=iters, tol=1e-6)
    order = torch.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    Y = evecs[:, 1:out_dim + 1].detach().cpu().numpy()

    return GraphEmbedResult(
        Y=Y,
        evals=evals.detach().cpu().numpy(),
        kind="laplacian",
        diagnostics={"eps": eps_used, "degree_median": float(torch.median(d).item()), "k": int(nn_idx.shape[1])},
    )


def diffusion_map_knn_torch(
    nn_idx: np.ndarray,
    nn_d2: np.ndarray,
    out_dim: int = 8,
    eps: Optional[float] = None,
    alpha: float = 0.5,
    iters: int = 200,
) -> GraphEmbedResult:
    """
    Diffusion-map eigenvectors via symmetric normalization (no SciPy).

    Build W (symmetric), alpha-normalize:
        q_i = sum_j W_ij
        W_alpha = W_ij / (q_i^alpha q_j^alpha)
    Then symmetric operator:
        S = D^{-1/2} W_alpha D^{-1/2}
    Compute largest eigenpairs of S; drop the top (trivial) eigenvector.
    """
    W, eps_used = build_dense_affinity(nn_idx, nn_d2, eps=eps)
    q = W.sum(dim=1).clamp_min(1e-12)
    qa = q.pow(-alpha)
    Walpha = (qa[:, None] * W) * qa[None, :]

    d = Walpha.sum(dim=1).clamp_min(1e-12)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(d))
    S = D_inv_sqrt @ Walpha @ D_inv_sqrt

    N = S.shape[0]
    k = min(int(out_dim) + 1, N - 1)
    X = torch.randn((N, k), dtype=torch.float64)

    evals, evecs = _lobpcg_call(S, k=k, X=X, largest=True, iters=iters, tol=1e-6)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    Y = evecs[:, 1:out_dim + 1].detach().cpu().numpy()

    return GraphEmbedResult(
        Y=Y,
        evals=evals.detach().cpu().numpy(),
        kind="diffusion_sym",
        diagnostics={"eps": eps_used, "alpha": float(alpha), "degree_median": float(torch.median(d).item()), "k": int(nn_idx.shape[1])},
    )


def participation_ratio_from_evals(evals: np.ndarray, drop_first: bool = True) -> float:
    mu = np.asarray(evals, dtype=np.float64)
    mu = np.sort(mu)[::-1]
    if drop_first and mu.size > 0:
        mu = mu[1:]
    if mu.size == 0:
        return float("nan")
    num = (mu * mu).sum()
    den = (mu * mu * mu * mu).sum()
    return float((num * num) / den) if den > 0 else float("nan")
