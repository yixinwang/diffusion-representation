"""Minimal exact RMPF-R13 spherical-feedback algebra check.

This compact file verifies inverse and log-Jacobian formulas only. The complete
local release contains the full known-truth, systems and real-smoke code/results.
"""
from __future__ import annotations
import numpy as np


def mobius(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    aa = float(a @ a)
    au = u @ a
    den = 1.0 + aa + 2.0 * au
    return ((1.0 - aa) * u + 2.0 * (1.0 + au)[:, None] * a) / den[:, None]


def logdet(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    aa = float(a @ a)
    return 7.0 * (np.log1p(-aa) - np.log(1.0 + aa + 2.0 * (u @ a)))


def main() -> None:
    rng = np.random.default_rng(20260830)
    u = rng.normal(size=(10000, 8))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    a = rng.normal(size=8)
    a *= 0.65 / np.linalg.norm(a)
    v = mobius(u, a)
    recovered = mobius(v, -a)
    error = float(np.max(np.abs(recovered - u)))
    cancellation = float(np.max(np.abs(logdet(u, a) + logdet(v, -a))))
    assert error < 1e-12
    assert cancellation < 1e-11
    zero = np.zeros(8)
    assert np.array_equal(mobius(u, zero), u)
    print({"roundtrip_max_abs": error, "logdet_cancel_max_abs": cancellation, "identity_recovery": True})


if __name__ == "__main__":
    main()
