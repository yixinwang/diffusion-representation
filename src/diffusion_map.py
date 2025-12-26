
"""
Diffusion Maps embedding.

Reference: Coifman & Lafon (2006), "Diffusion Maps".

Given pairwise squared distances D2, builds an RBF kernel and computes
the leading diffusion coordinates.

This file is kept separate from LTSA for clarity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DiffusionMapResult:
    Y: np.ndarray            # (N, out_dim)
    evals: np.ndarray        # (out_dim,) diffusion eigenvalues (descending)
    evecs: np.ndarray        # (N, out_dim) corresponding eigenvectors
    epsilon: float
    diagnostics: dict


def diffusion_map(
    D2: np.ndarray,
    out_dim: int,
    *,
    epsilon: Optional[float] = None,
    alpha: float = 0.5,
    t: int = 1,
) -> DiffusionMapResult:
    """
    Parameters
    ----------
    D2 : (N,N) array
        Pairwise squared distances.
    out_dim : int
        Output embedding dimension.
    epsilon : float, optional
        Kernel bandwidth. If None, median of nonzero distances is used.
    alpha : float
        Density normalization parameter (0: none, 1: full).
    t : int
        Diffusion time; coordinates scale as lambda^t * psi.

    Returns
    -------
    DiffusionMapResult
    """
    D2 = np.asarray(D2, dtype=np.float64)
    N = D2.shape[0]

    if epsilon is None:
        # median of upper triangle (excluding diagonal)
        tri = D2[np.triu_indices(N, k=1)]
        tri = tri[np.isfinite(tri)]
        tri = tri[tri > 0]
        if tri.size == 0:
            raise ValueError("D2 appears to be all zeros.")
        epsilon = float(np.median(tri))

    K = np.exp(-D2 / (epsilon + 1e-12))  # RBF

    # alpha-normalization (Coifman-Lafon)
    q = K.sum(axis=1)
    q_alpha = np.power(q, -alpha)
    K_tilde = (q_alpha[:, None] * K) * q_alpha[None, :]

    d = K_tilde.sum(axis=1)
    P = K_tilde / (d[:, None] + 1e-12)  # row-stochastic

    # Eigen-decomposition of P^T (or symmetric conjugate). We'll use symmetric normalization:
    # S = D^{-1/2} K_tilde D^{-1/2} (similar to P)
    d_sqrt_inv = 1.0 / np.sqrt(d + 1e-12)
    S = (d_sqrt_inv[:, None] * K_tilde) * d_sqrt_inv[None, :]

    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(evals)[::-1]  # descending
    evals = evals[order]
    evecs = evecs[:, order]

    # Map back to right eigenvectors of P
    psi = (d_sqrt_inv[:, None] * evecs)

    # drop trivial first eigenpair (lambda=1)
    lam = evals[1:out_dim+1]
    psi = psi[:, 1:out_dim+1]

    Y = psi * (lam ** t)[None, :]

    diagnostics = {
        "kernel_epsilon": epsilon,
        "alpha": alpha,
        "diffusion_time": t,
        "lambda1": float(evals[0]),
        "lambda2": float(evals[1]) if evals.size > 1 else None,
    }
    return DiffusionMapResult(Y=Y, evals=lam, evecs=psi, epsilon=epsilon, diagnostics=diagnostics)


def estimate_intrinsic_dim_from_spectrum(
    evals: np.ndarray,
    *,
    threshold: float = 0.05,
    max_dim: int = 20,
) -> int:
    """
    Heuristic: count eigenvalues (excluding the trivial 1) above threshold.

    For diffusion maps, a sharp drop in eigenvalues suggests intrinsic dim.
    """
    evals = np.asarray(evals)
    m = min(int(max_dim), evals.size)
    return int(np.sum(evals[:m] > threshold))
