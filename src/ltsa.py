
"""
Local Tangent Space Alignment (LTSA) embedding.

Reference: Zhang & Zha (2004), "Principal Manifolds and Nonlinear Dimension Reduction
via Local Tangent Space Alignment".

This implementation assumes you already have a kNN graph (indices + distances).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

Array = np.ndarray


@dataclass
class LTSAResult:
    Y: Array                 # (N, out_dim) embedding
    eigvals: Array           # (out_dim+1,) smallest non-trivial eigenvalues
    intrinsic_dim: Optional[int] = None
    diagnostics: Optional[dict] = None


def _pca_tangent_basis(Xn: Array, d: int) -> Tuple[Array, Array]:
    """
    Xn: (k, D) neighborhood points
    Returns: (U, mu)
      U: (D, d) top-d right singular vectors (tangent basis)
      mu: (D,) neighborhood mean
    """
    mu = Xn.mean(axis=0)
    Z = Xn - mu
    # SVD on k x D; right singular vectors shape D x D
    # We need top-d in ambient space.
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    U = Vt[:d].T  # (D,d)
    return U, mu


def ltsa(
    X: Array,
    nn_idx: Array,
    out_dim: int,
    *,
    local_dim: Optional[int] = None,
    ridge: float = 1e-6,
) -> LTSAResult:
    """
    Parameters
    ----------
    X : (N, D)
        Data matrix (ambient coordinates).
    nn_idx : (N, k)
        Neighbor indices (excluding self).
    out_dim : int
        Target embedding dimension.
    local_dim : int, optional
        Tangent dimension used per neighborhood. Default: out_dim.
    ridge : float
        Numerical ridge added to local alignment.

    Returns
    -------
    LTSAResult
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    k = nn_idx.shape[1]
    d = int(local_dim) if local_dim is not None else int(out_dim)
    if d > D:
        raise ValueError("local_dim cannot exceed ambient dimension")

    # Build alignment matrix B (N x N) as sum_i G_i G_i^T where G_i spans local coordinates.
    B = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        idx = nn_idx[i]
        Xi = np.vstack([X[i:i+1], X[idx]])  # (k+1, D), include center first
        U, mu = _pca_tangent_basis(Xi, d=d)  # (D,d)
        Z = (Xi - mu) @ U  # (k+1, d) local coords

        # Construct local alignment matrix:
        # Gi = [1, Z] then project out the span of Gi from I
        # Following LTSA: Wi = I - Gi (Gi^T Gi)^{-1} Gi^T
        ones = np.ones((k+1, 1), dtype=np.float64)
        Gi = np.hstack([ones, Z])  # (k+1, d+1)
        GTG = Gi.T @ Gi
        GTG = GTG + ridge * np.eye(d+1)
        inv = np.linalg.inv(GTG)
        Wi = np.eye(k+1) - Gi @ inv @ Gi.T  # (k+1,k+1)

        # Scatter-add into global B
        # Indices in global space:
        I = np.concatenate([[i], idx])
        for a in range(k+1):
            ia = I[a]
            B[ia, I] += Wi[a]

    # Solve smallest eigenvectors of B subject to constant vector; embedding are eigenvectors 2..out_dim+1.
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # drop first eigenvector (constant)
    Y = eigvecs[:, 1:out_dim+1].copy()
    ev = eigvals[1:out_dim+1].copy()

    diagnostics = {
        "B_trace": float(np.trace(B)),
        "B_min_eig": float(eigvals[0]),
        "B_gap_after_const": float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else None,
        "eigvals_full": eigvals.copy(),
    }
    return LTSAResult(Y=Y, eigvals=ev, diagnostics=diagnostics)
