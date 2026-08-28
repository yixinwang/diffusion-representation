#!/usr/bin/env python3
"""Minimal exact RMPF known-truth reproduction.

No VAE, encoder, decoder, discarded coordinate, or stochastic fiber is used.
Every visible coordinate is retained by an orthonormal Haar map.  The script
checks exact inversion/Jacobian accounting and the hidden global parity/copula
adversary, then reproduces the strength/rank transition on fixed samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
import numpy as np


def haar(n: int) -> np.ndarray:
    if n < 2 or n & (n - 1):
        raise ValueError("n must be a power of two")
    rows = [np.ones(n) / math.sqrt(n)]
    scale = n
    while scale >= 2:
        for start in range(0, n, scale):
            row = np.zeros(n)
            half = scale // 2
            row[start:start + half] = 1 / math.sqrt(scale)
            row[start + half:start + scale] = -1 / math.sqrt(scale)
            rows.append(row)
        scale //= 2
    H = np.stack(rows[:n])
    assert np.max(np.abs(H @ H.T - np.eye(n))) < 1e-12
    return H


def tree_feature(x: np.ndarray, perm: np.ndarray, signs: np.ndarray) -> np.ndarray:
    h = np.tanh(1.35 * x[:, perm] * signs[None])
    while h.shape[1] > 1:
        h = np.tanh(2.15 * h[:, 0::2] * h[:, 1::2])
    return h[:, 0]


@dataclass(frozen=True)
class Config:
    dim: int = 32
    stages: int = 16
    rank: int = 4
    strength: float = 12.0
    tau: float = 0.60
    step: float = 1 / 16
    seed: int = 20260827
    global_stages: tuple[int, ...] = (3, 7, 11, 15)


class RMPF:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.d = cfg.dim // 2
        self.H = haar(cfg.dim)
        rng = np.random.default_rng(cfg.seed)
        self.scale = np.exp(rng.normal(0.0, 0.015, size=(cfg.stages, self.d)))
        self.shift = rng.normal(0.0, 0.015, size=(cfg.stages, self.d))
        self.local = rng.normal(0.0, 0.16, size=(cfg.stages, self.d, 4))
        self.perm = np.stack([rng.permutation(self.d) for _ in range(cfg.rank)])
        self.signs = rng.choice([-1.0, 1.0], size=(cfg.rank, self.d))
        self.W = rng.normal(0.0, 1.0 / math.sqrt(self.d), size=(cfg.stages, self.d, cfg.rank))
        # Orthogonalize output directions so rank truncation has a clean meaning.
        for k in cfg.global_stages:
            q, _ = np.linalg.qr(self.W[k])
            self.W[k] = 1.58 * q[:, :cfg.rank]

    def texture(self, z: np.ndarray) -> np.ndarray:
        return np.sinh(self.cfg.tau * z) / self.cfg.tau

    def inv_texture(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh(self.cfg.tau * x) / self.cfg.tau

    def phi(self, a: np.ndarray) -> np.ndarray:
        return np.stack([tree_feature(a, self.perm[j], self.signs[j]) for j in range(self.cfg.rank)], 1)

    def local_increment(self, k: int, a: np.ndarray) -> np.ndarray:
        f = np.stack([np.ones_like(a), np.tanh(a), np.tanh(np.roll(a, -1, 1)), a], -1)
        return np.einsum("ndf,df->nd", f, self.local[k])

    def forward_base(self, z: np.ndarray, active_rank: int | None = None) -> np.ndarray:
        active_rank = self.cfg.rank if active_rank is None else active_rank
        a = self.texture(z[:, :self.d])
        b = self.texture(z[:, self.d:])
        for k in range(self.cfg.stages):
            a = a * self.scale[k] + self.shift[k]
            inc = self.local_increment(k, a)
            if k in self.cfg.global_stages and active_rank:
                p = self.phi(a)[:, :active_rank]
                inc = inc + self.cfg.strength * p @ self.W[k, :, :active_rank].T
            b = b + self.cfg.step * inc
        return np.concatenate([a, b], 1) @ self.H

    def inverse(self, x: np.ndarray, active_rank: int | None = None) -> np.ndarray:
        active_rank = self.cfg.rank if active_rank is None else active_rank
        y = x @ self.H.T
        aK, bK = y[:, :self.d], y[:, self.d:]
        # Recover all coarse states exactly.
        states = [None] * (self.cfg.stages + 1)
        states[-1] = aK
        for k in range(self.cfg.stages - 1, -1, -1):
            states[k] = (states[k + 1] - self.shift[k]) / self.scale[k]
        shift = np.zeros_like(bK)
        for k in range(self.cfg.stages):
            a = states[k + 1]
            inc = self.local_increment(k, a)
            if k in self.cfg.global_stages and active_rank:
                p = self.phi(a)[:, :active_rank]
                inc = inc + self.cfg.strength * p @ self.W[k, :, :active_rank].T
            shift += self.cfg.step * inc
        b0 = bK - shift
        return np.concatenate([self.inv_texture(states[0]), self.inv_texture(b0)], 1)

    def logpdf(self, x: np.ndarray, active_rank: int | None = None) -> np.ndarray:
        z = self.inverse(x, active_rank)
        log_base = -0.5 * np.sum(z * z, 1) - 0.5 * self.cfg.dim * math.log(2 * math.pi)
        log_texture = np.sum(np.log(np.cosh(self.cfg.tau * z)), 1)
        log_coarse = np.sum(np.log(self.scale))
        return log_base - log_texture - log_coarse


def dense_jacobian_check() -> dict[str, float]:
    cfg = Config(dim=8, stages=4, rank=1, strength=2.5, global_stages=(3,), seed=17)
    model = RMPF(cfg)
    z = np.random.default_rng(3).normal(size=(1, cfg.dim))
    eps = 1e-6
    J = np.empty((cfg.dim, cfg.dim))
    for j in range(cfg.dim):
        zp, zm = z.copy(), z.copy()
        zp[0, j] += eps
        zm[0, j] -= eps
        J[:, j] = (model.forward_base(zp)[0] - model.forward_base(zm)[0]) / (2 * eps)
    numeric = np.linalg.slogdet(J)[1]
    analytic = np.sum(np.log(np.cosh(cfg.tau * z))) + np.sum(np.log(model.scale))
    return {"finite_difference_logdet": float(numeric), "analytic_logdet": float(analytic),
            "absolute_error": float(abs(numeric - analytic))}


def parity_adversary(m: int = 8) -> dict[str, float]:
    cube = np.array(np.meshgrid(*([[-1.0, 1.0]] * m), indexing="ij")).reshape(m, -1).T
    parity = np.prod(cube, 1)
    even, odd = cube[parity == 1], cube[parity == -1]
    covariance_gap = float(np.max(np.abs(np.cov(even, rowvar=False, bias=True) - np.cov(odd, rowvar=False, bias=True))))
    subset_gap = 0.0
    for j in range(m):
        ec = np.unique(np.delete(even, j, 1), axis=0, return_counts=True)[1]
        oc = np.unique(np.delete(odd, j, 1), axis=0, return_counts=True)[1]
        subset_gap = max(subset_gap, float(np.max(np.abs(ec / ec.sum() - oc / oc.sum()))))
    return {"pairwise_covariance_gap": covariance_gap, "m_minus_1_subset_mass_gap": subset_gap,
            "global_parity_mean_separation": float(abs(np.prod(even, 1).mean() - np.prod(odd, 1).mean()))}


def study(seed: int = 9100, n: int = 4000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 32))
    strength_rows = []
    for strength in (0, 1, 4, 8, 12, 20):
        model = RMPF(Config(strength=float(strength)))
        x = model.forward_base(z)
        target = -model.logpdf(x).mean() / 32
        no_global = -model.logpdf(x, active_rank=0).mean() / 32
        strength_rows.append({"strength": strength, "nll_no_global_minus_rmpf": float(no_global - target)})
    true = RMPF(Config(strength=12.0))
    x = true.forward_base(z)
    target_nll = -true.logpdf(x).mean() / 32
    rank_rows = []
    for rank in (0, 1, 2, 4):
        rank_rows.append({"active_rank": rank,
                          "excess_nll_per_dim": float((-true.logpdf(x, active_rank=rank).mean() / 32) - target_nll)})
    t0 = time.perf_counter(); x = true.forward_base(z); forward_seconds = time.perf_counter() - t0
    zhat = true.inverse(x)
    return {
        "roundtrip_max_abs_error": float(np.max(np.abs(z - zhat))),
        "logpdf_all_finite": bool(np.isfinite(true.logpdf(x)).all()),
        "strength_sweep": strength_rows,
        "rank_sweep": rank_rows,
        "forward_seconds_4000": forward_seconds,
        "jacobian": dense_jacobian_check(),
        "parity_adversary": parity_adversary(),
        "equivalence_copy_mismatch": float(np.max(np.abs(x - true.forward_base(z)))),
    }


if __name__ == "__main__":
    result = study()
    print(json.dumps(result, indent=2, sort_keys=True))
