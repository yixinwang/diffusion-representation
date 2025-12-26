"""
Local Tangent Space Alignment (LTSA), Euclidean and diagonal-Riemannian versions.

Reference:
Zhang & Zha (2004), "Principal Manifolds and Nonlinear Dimension Reduction
via Local Tangent Space Alignment".
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

Array = np.ndarray


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass
class LTSAResult:
    Y: Array                 # (N, out_dim) embedding
    eigvals: Array           # (out_dim,) smallest non-trivial eigenvalues
    intrinsic_dim: Optional[int] = None
    diagnostics: Optional[dict] = None


# ---------------------------------------------------------------------
# Euclidean LTSA (baseline)
# ---------------------------------------------------------------------
def _pca_tangent_basis(Xn: Array, d: int) -> Tuple[Array, Array]:
    """
    Xn: (k+1, D) neighborhood points (center included)
    Returns:
      U  : (D, d) ambient tangent basis
      mu : (D,) neighborhood mean
    """
    mu = Xn.mean(axis=0)
    Z = Xn - mu
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    U = Vt[:d].T
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
    Standard Euclidean LTSA.
    """
    X = np.asarray(X, dtype=np.float64)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)

    N, D = X.shape
    k = nn_idx.shape[1]
    d = int(local_dim) if local_dim is not None else int(out_dim)
    if d > D:
        raise ValueError("local_dim cannot exceed ambient dimension")

    B = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        idx = nn_idx[i]
        I = np.concatenate([[i], idx])
        Xi = X[I]

        U, mu = _pca_tangent_basis(Xi, d=d)
        Zloc = (Xi - mu) @ U  # (k+1, d)

        ones = np.ones((k+1, 1))
        Gi = np.hstack([ones, Zloc])
        GTG = Gi.T @ Gi + ridge * np.eye(d + 1)
        Wi = np.eye(k + 1) - Gi @ np.linalg.inv(GTG) @ Gi.T

        for a in range(k + 1):
            ia = I[a]
            B[ia, I] += Wi[a]

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    Y = eigvecs[:, 1:out_dim + 1].copy()
    ev = eigvals[1:out_dim + 1].copy()

    diagnostics = {
        "B_trace": float(np.trace(B)),
        "B_min_eig": float(eigvals[0]),
        "B_gap_after_const": float(eigvals[1] - eigvals[0]),
        "metric": "euclidean",
    }

    return LTSAResult(Y=Y, eigvals=ev, diagnostics=diagnostics)


# ---------------------------------------------------------------------
# Diagonal-Riemannian LTSA (metric-aware)
# ---------------------------------------------------------------------
def ltsa_riemannian(
    X: np.ndarray,
    nn_idx: np.ndarray,
    gdiag: np.ndarray,
    out_dim: int,
    *,
    local_dim: Optional[int] = None,
    ridge: float = 1e-6,
    normalize_metric: bool = True,
    eps: float = 1e-12,
) -> LTSAResult:
    """
    Metric-aware LTSA with diagonal Riemannian metric.

    For each point i, tangent estimation is performed in locally whitened
    coordinates:
        (x_j - x_i) -> sqrt(gdiag_i) ⊙ (x_j - x_i)

    Alignment follows standard LTSA over (k+1) points including the center.
    """
    X = np.asarray(X, dtype=np.float64)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)
    gdiag = np.asarray(gdiag, dtype=np.float64)

    N, D = X.shape
    k = nn_idx.shape[1]
    d = int(local_dim) if local_dim is not None else int(out_dim)
    if d > D:
        raise ValueError("local_dim cannot exceed ambient dimension")
    if gdiag.shape != (N, D):
        raise ValueError(f"gdiag must have shape {(N, D)}, got {gdiag.shape}")

    g = np.clip(gdiag, eps, None)

    # Remove arbitrary per-point scaling (important!)
    if normalize_metric:
        logg = np.log(g)
        logg -= logg.mean(axis=1, keepdims=True)  # geometric mean = 1
        g = np.exp(logg)

    sqrtg = np.sqrt(g)

    B = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        idx = nn_idx[i]
        I = np.concatenate([[i], idx])
        Xi = X[I]

        # Metric-aware tangent estimation
        mu = Xi.mean(axis=0)
        Z = Xi - mu
        Zw = Z * sqrtg[i][None, :]   # whiten by center metric

        _, _, Vt = np.linalg.svd(Zw, full_matrices=False)
        U = Vt[:d].T                 # (D, d)

        Zloc = (Xi - mu) @ U         # local coordinates

        ones = np.ones((k + 1, 1))
        Gi = np.hstack([ones, Zloc])
        GTG = Gi.T @ Gi + ridge * np.eye(d + 1)
        Wi = np.eye(k + 1) - Gi @ np.linalg.inv(GTG) @ Gi.T

        for a in range(k + 1):
            ia = I[a]
            B[ia, I] += Wi[a]

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    Y = eigvecs[:, 1:out_dim + 1].copy()
    ev = eigvals[1:out_dim + 1].copy()

    diagnostics = {
        "B_trace": float(np.trace(B)),
        "B_min_eig": float(eigvals[0]),
        "B_gap_after_const": float(eigvals[1] - eigvals[0]),
        "metric": "riemannian_diag",
        "normalize_metric": bool(normalize_metric),
    }

    return LTSAResult(Y=Y, eigvals=ev, diagnostics=diagnostics)
