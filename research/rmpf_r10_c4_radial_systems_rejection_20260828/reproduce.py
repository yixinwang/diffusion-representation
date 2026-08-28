"""Minimal algebraic reproduction for the RMPF-R10-C4 radial layer."""
from __future__ import annotations
import numpy as np


def forward(z, h, dh):
    r = np.linalg.norm(z, axis=-1)
    out = z.copy(); ld = np.zeros_like(r)
    mask = r > 1e-15
    t = np.log(r[mask])
    v = h(t)
    rho = np.exp(v)
    ratio = rho / r[mask]
    out[mask] = z[mask] * ratio[:, None]
    ld[mask] = np.log(dh(t)) + z.shape[-1] * np.log(ratio)
    return out, ld


def inverse(y, hinv, dhinverse):
    rho = np.linalg.norm(y, axis=-1)
    out = y.copy(); ld = np.zeros_like(rho)
    mask = rho > 1e-15
    v = np.log(rho[mask])
    t = hinv(v)
    r = np.exp(t)
    ratio = r / rho[mask]
    out[mask] = y[mask] * ratio[:, None]
    ld[mask] = np.log(dhinverse(v)) + y.shape[-1] * np.log(ratio)
    return out, ld


if __name__ == "__main__":
    a = 0.2
    h = lambda t: t + a
    dh = lambda t: np.ones_like(t)
    hinv = lambda v: v - a
    dhinverse = lambda v: np.ones_like(v)
    rng = np.random.default_rng(7)
    z = rng.normal(size=(1000, 3))
    y, lf = forward(z, h, dh)
    back, li = inverse(y, hinv, dhinverse)
    assert np.max(np.abs(back-z)) < 1e-12
    assert np.max(np.abs(lf+li)) < 1e-12
    assert np.array_equal(np.sign(y), np.sign(z))
    print({"roundtrip": float(np.max(np.abs(back-z))),
           "logdet_cancel": float(np.max(np.abs(lf+li)))})
