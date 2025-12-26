
"""
Intrinsic dimension estimators to sanity-check "genuinely low-dimensional" structure.

Includes:
- Levina-Bickel MLE (kNN-based)
- Local PCA participation ratio

Both are heuristics; use multiple checks and look for agreement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class IntrinsicDimReport:
    lb_mle: float
    lb_mle_std: float
    local_pca_pr: float
    local_pca_pr_std: float
    details: dict


def levina_bickel_mle(nn_dists: np.ndarray) -> np.ndarray:
    """
    Levina & Bickel (2005) MLE intrinsic dimension estimator.

    Parameters
    ----------
    nn_dists : (N, k) array
        Distances (NOT squared) to k nearest neighbors ordered ascending.

    Returns
    -------
    m_hat : (N,) per-point estimates using all k neighbors.
    """
    nn_dists = np.asarray(nn_dists, dtype=np.float64)
    if nn_dists.ndim != 2:
        raise ValueError("nn_dists must be (N,k)")
    N, k = nn_dists.shape
    if k < 2:
        raise ValueError("Need k>=2 for Levina-Bickel")
    # avoid log(0)
    rk = nn_dists[:, -1] + 1e-12
    logs = np.log(rk[:, None] / (nn_dists[:, :-1] + 1e-12))
    denom = np.mean(logs, axis=1)
    m_hat = 1.0 / (denom + 1e-12)
    return m_hat


def local_pca_participation_ratio(
    X: np.ndarray,
    nn_idx: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    """
    Local PCA participation ratio (PR) per point:
        PR = (sum λ)^2 / (sum λ^2)
    where λ are eigenvalues of the local covariance.

    Returns
    -------
    pr : (N,) float
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    k = nn_idx.shape[1]
    pr = np.zeros(N, dtype=np.float64)
    for i in range(N):
        Xi = X[nn_idx[i]]
        if center:
            Xi = Xi - Xi.mean(axis=0, keepdims=True)
        C = (Xi.T @ Xi) / max(1, k - 1)
        w = np.linalg.eigvalsh(C)
        w = np.maximum(w, 0.0)
        s1 = np.sum(w)
        s2 = np.sum(w * w)
        pr[i] = (s1 * s1) / (s2 + 1e-12)
    return pr


def intrinsic_dim_report(
    X: np.ndarray,
    nn_idx: np.ndarray,
    nn_d2: np.ndarray,
) -> IntrinsicDimReport:
    """
    Convenience wrapper that computes:
    - LB MLE from neighbor distances
    - Local PCA participation ratio
    """
    nn_d = np.sqrt(np.maximum(nn_d2, 0.0))
    lb = levina_bickel_mle(nn_d)
    pr = local_pca_participation_ratio(X, nn_idx)

    return IntrinsicDimReport(
        lb_mle=float(np.mean(lb)),
        lb_mle_std=float(np.std(lb)),
        local_pca_pr=float(np.mean(pr)),
        local_pca_pr_std=float(np.std(pr)),
        details={
            "lb_per_point": lb,
            "pr_per_point": pr,
        },
    )
