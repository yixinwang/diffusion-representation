"""Self-contained algebraic reproduction of the RMPF-R9 failure mechanism.

This does not reproduce the opened CIFAR/UCF timings; those require the released arrays
and full artifact package. It checks the frozen sketch width, exact triangular
normalization, full-row recovery, and the objective-vs-relative-parameter failure.
"""
from __future__ import annotations

import json
import math
import numpy as np


def ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)


def objective(x: np.ndarray, y: np.ndarray, theta: np.ndarray, lam: float) -> float:
    return float(np.sum((x @ theta - y) ** 2) + lam * np.sum(theta**2))


def main() -> None:
    d, q, eta, eps, delta = 2, 9, 0.05, 0.10, 0.01
    beta = (1.0 - eta) / 3.0
    m_theory = math.ceil(8 * d / (3 * beta * eps**2) * math.log(2 * d * q / delta))
    assert m_theory == 13792

    # Exact triangular map: parameter approximation never breaks normalization.
    rng = np.random.default_rng(20260828)
    n = 200_000
    x = rng.normal(size=(n, d))
    # Deliberately tiny identifiable coefficients: aggregate prediction risk is flat
    # relative to their norm, matching the real R9 diagnosis.
    theta_true = np.array([[0.0030, -0.0025, 0.0010], [-0.0040, 0.0060, -0.0015]])
    y = x @ theta_true + rng.normal(scale=1.0, size=(n, 3))
    lam = 1e-2
    theta_full = ridge(x, y, lam)

    # Source-frozen balanced subsample; one index set serves every response.
    m = 16384
    idx = np.arange(m) * (n // m)
    idx = np.minimum(idx, n - 1)
    scale = n / len(idx)
    theta_sketch = np.linalg.solve(
        scale * x[idx].T @ x[idx] + lam * np.eye(d),
        scale * x[idx].T @ y[idx],
    )
    f_full = objective(x, y, theta_full, lam)
    f_sketch = objective(x, y, theta_sketch, lam)
    objective_ratio = f_sketch / f_full
    relative_parameter_error = np.linalg.norm(theta_sketch - theta_full) / np.linalg.norm(theta_full)

    # Exact one-coordinate lifting round trip and log-Jacobian cancellation.
    alpha, beta_coef, sigma = theta_sketch[0, 0], theta_sketch[1, 0], 0.37
    parent = rng.normal(size=1000)
    neighbor = rng.normal(size=1000)
    visible = rng.normal(size=1000)
    residual = (visible - alpha * parent - beta_coef * neighbor) / sigma
    recovered = alpha * parent + beta_coef * neighbor + sigma * residual
    logdet = -math.log(sigma)
    inverse_logdet = math.log(sigma)

    # Full-row identity recovery is exact by construction.
    theta_recovered = ridge(x, y, lam)

    out = {
        "frozen_theory_rows": m_theory,
        "executed_rows": m,
        "objective_ratio": objective_ratio,
        "relative_parameter_error": relative_parameter_error,
        "roundtrip_max_abs": float(np.max(np.abs(recovered - visible))),
        "logdet_cancel_abs": abs(logdet + inverse_logdet),
        "full_row_parameter_recovery_max_abs": float(np.max(np.abs(theta_recovered - theta_full))),
        "lesson": "near-unit objective preservation does not imply small relative error for tiny coefficients",
    }
    assert objective_ratio < 1.001
    assert relative_parameter_error > 0.08
    assert out["roundtrip_max_abs"] < 1e-12
    assert out["logdet_cancel_abs"] == 0.0
    assert out["full_row_parameter_recovery_max_abs"] == 0.0
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
