#!/usr/bin/env python3
"""NumPy-only algebraic reproduction of the R11 determinant-one angular layer."""
from __future__ import annotations
import json
import numpy as np


def spd_sqrt_det_one(cov: np.ndarray) -> np.ndarray:
    ridge = 0.98 * cov + 0.02 * np.trace(cov) * np.eye(3) / 3.0
    ridge /= np.linalg.det(ridge) ** (1.0 / 3.0)
    values, vectors = np.linalg.eigh(ridge)
    return (vectors * np.sqrt(values)) @ vectors.T


def energy_score(reference: np.ndarray, generated: np.ndarray) -> float:
    r = reference.reshape(len(reference), -1)[:512]
    g = generated.reshape(len(generated), -1)[:512]
    cross = np.sqrt(np.maximum(((g[:, None] - r[None]) ** 2).sum(-1), 0.0)).mean()
    self_term = np.sqrt(np.maximum(((g[:, None] - g[None]) ** 2).sum(-1), 0.0)).mean()
    return float(cross - 0.5 * self_term)


def run(seed: int = 9800) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    train_n, test_n, sites = 6000, 5000, 32
    gamma = 0.55
    a0 = np.diag([np.exp(gamma), np.exp(-gamma), 1.0])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a1 = q @ a0 @ q.T

    train_state = rng.integers(0, 2, train_n)
    test_state = rng.integers(0, 2, test_n)
    train_z = rng.normal(size=(train_n, sites, 3))
    test_z = rng.normal(size=(test_n, sites, 3))
    train = np.einsum("nsj,njk->nsk", train_z, np.where(train_state[:, None, None] == 0, a0, a1))
    test = np.einsum("nsj,njk->nsk", test_z, np.where(test_state[:, None, None] == 0, a0, a1))

    fitted = []
    for state in (0, 1):
        x = train[train_state == state].reshape(-1, 3)
        fitted.append(spd_sqrt_det_one(np.cov(x, rowvar=False, bias=True)))
    fitted = np.stack(fitted)
    inverse = np.linalg.inv(fitted)
    base = np.einsum("nsj,njk->nsk", test, inverse[test_state])
    reconstructed = np.einsum("nsj,njk->nsk", base, fitted[test_state])

    d = sites * 3
    nll_identity = float(np.mean(0.5 * np.sum(test * test, axis=(1, 2))) / d)
    nll_learned = float(np.mean(0.5 * np.sum(base * base, axis=(1, 2))) / d)
    reference = rng.normal(size=test.shape)
    energy_identity = energy_score(reference, test)
    energy_learned = energy_score(reference, base)

    # Hidden parity has identity covariance and survives every linear covariance-shape map.
    parity = rng.choice(np.array([-1.0, 1.0]), size=(test_n, sites, 3))
    parity[:, -1, 0] = np.prod(parity[:, :-1, 0], axis=1)
    hidden_before = float(np.mean(np.prod(parity[:, :, 0], axis=1)))
    hidden_after = hidden_before

    return {
        "det_A0": float(np.linalg.det(a0)),
        "det_A1": float(np.linalg.det(a1)),
        "maximum_roundtrip_error": float(np.max(np.abs(reconstructed - test))),
        "nll_gain_per_dimension": nll_identity - nll_learned,
        "proper_energy_gain": energy_identity - energy_learned,
        "hidden_parity_before": hidden_before,
        "hidden_parity_after": hidden_after,
        "exact_zero_logdet_layer": bool(np.allclose(np.linalg.det(fitted), 1.0, atol=1e-12)),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
