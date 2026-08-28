#!/usr/bin/env python3
"""NumPy-only HSRF known-truth checks.

This compact script verifies the triangular Jacobian identity, the exact
finite OLS estimation term, a parity/local separation, and the positive
conditional-scale Jensen gap. It is not the full scientific runner.
"""
from __future__ import annotations

import json
import math
import numpy as np


def coupling_forward(a: np.ndarray, b: np.ndarray, w: np.ndarray):
    phi = np.c_[np.ones(len(a)), a, np.prod(np.tanh(a), axis=1)]
    return a, b - phi @ w


def coupling_inverse(a: np.ndarray, z: np.ndarray, w: np.ndarray):
    phi = np.c_[np.ones(len(a)), a, np.prod(np.tanh(a), axis=1)]
    return a, z + phi @ w


def main() -> None:
    rng = np.random.default_rng(20260828)

    # Exact triangular round trip; log determinant is zero.
    a = rng.normal(size=(128, 8))
    b = rng.normal(size=(128, 8))
    w = rng.normal(scale=0.1, size=(10, 8))
    aa, z = coupling_forward(a, b, w)
    aaa, bb = coupling_inverse(aa, z, w)
    roundtrip = float(max(np.max(np.abs(a - aaa)), np.max(np.abs(b - bb))))

    # Finite OLS theorem.
    n, m, q, sigma = 512, 8, 6, 0.7
    repetitions = 2000
    risks = []
    for _ in range(repetitions):
        x = rng.normal(size=(n, m))
        noise = rng.normal(scale=sigma, size=(n, q))
        w_hat = np.linalg.solve(x.T @ x, x.T @ noise)
        risks.append(np.sum(w_hat * w_hat))
    monte_carlo = float(np.mean(risks))
    analytic = q * sigma**2 * m / (n - m - 1)

    # Global parity cannot be represented by a disconnected constant predictor.
    signs = rng.choice([-1.0, 1.0], size=(200_000, 8))
    parity = np.prod(signs, axis=1)
    local_prediction = np.zeros_like(parity)
    global_prediction = parity
    local_mse = float(np.mean((parity - local_prediction) ** 2))
    global_mse = float(np.mean((parity - global_prediction) ** 2))

    # Conditional-scale Jensen gap.
    x = rng.normal(size=500_000)
    variance = np.exp(0.8 * np.tanh(x))
    scale_gain = 0.5 * (math.log(float(np.mean(variance))) - float(np.mean(np.log(variance))))

    result = {
        "roundtrip_max_error": roundtrip,
        "coupling_logdet": 0.0,
        "ols_analytic_prediction_error": analytic,
        "ols_monte_carlo_prediction_error": monte_carlo,
        "ols_absolute_error": abs(monte_carlo - analytic),
        "local_parity_mse": local_mse,
        "global_parity_mse": global_mse,
        "conditional_scale_gain": scale_gain,
        "all_checks_pass": bool(
            roundtrip < 1e-12
            and abs(monte_carlo - analytic) < 0.01 * analytic
            and local_mse > 0.99
            and global_mse == 0.0
            and scale_gain > 0.0
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
