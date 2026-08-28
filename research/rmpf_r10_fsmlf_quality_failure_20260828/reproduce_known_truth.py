#!/usr/bin/env python3
"""Standalone RMPF-R10 hidden-copula known-truth reproduction (NumPy only)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

LOG2PI = math.log(2.0 * math.pi)
H4 = np.array([
    [1, 1, 1, 1],
    [1, -1, 1, -1],
    [1, 1, -1, -1],
    [1, -1, -1, 1],
], dtype=np.float64) / 2.0


def dct_basis(dim: int, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    n = np.arange(dim, dtype=np.float64)[:, None]
    k = indices.astype(np.float64)[None, :]
    out = math.sqrt(2.0 / dim) * np.cos(math.pi * (n + 0.5) * k / dim)
    if np.any(indices == 0):
        out[:, indices == 0] /= math.sqrt(2.0)
    np.testing.assert_allclose(out.T @ out, np.eye(len(indices)), atol=2e-10, rtol=0)
    return out


@dataclass
class Endpoint:
    indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    active: np.ndarray
    a_dim: int
    b_dim: int

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.indices) != 4:
            raise ValueError("This compact reproduction freezes rank four.")
        return dct_basis(self.b_dim, self.indices), H4

    @staticmethod
    def state(c: np.ndarray) -> np.ndarray:
        return (np.prod(np.where(c >= 0, 1, -1), axis=1) > 0).astype(np.int64)

    def transform(self, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.asarray(source, dtype=np.float64).copy()
        ld = np.zeros(len(out), dtype=np.float64)
        u, h = self.basis()
        b = out[:, self.a_dim:]
        selected = b @ u
        p = selected @ h.T
        m = 2
        state = self.state(p[:, :m])
        y = p.copy()
        for j in range(m):
            mu = self.means[j, state] if self.active[j] else 0.0
            sc = self.scales[j, state] if self.active[j] else 1.0
            y[:, m + j] = (p[:, m + j] - mu) / sc
            ld -= np.log(sc)
        out[:, self.a_dim:] = b + (y @ h - selected) @ u.T
        return out, ld

    def inverse(self, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.asarray(base, dtype=np.float64).copy()
        ld = np.zeros(len(out), dtype=np.float64)
        u, h = self.basis()
        b = out[:, self.a_dim:]
        selected = b @ u
        y = selected @ h.T
        m = 2
        state = self.state(y[:, :m])
        p = y.copy()
        for j in range(m):
            mu = self.means[j, state] if self.active[j] else 0.0
            sc = self.scales[j, state] if self.active[j] else 1.0
            p[:, m + j] = y[:, m + j] * sc + mu
            ld += np.log(sc)
        out[:, self.a_dim:] = b + (p @ h - selected) @ u.T
        return out, ld

    def copy(self) -> "Endpoint":
        return Endpoint(self.indices.copy(), self.means.copy(), self.scales.copy(), self.active.copy(), self.a_dim, self.b_dim)


def nll(endpoint: Endpoint, x: np.ndarray) -> float:
    z, ld = endpoint.transform(x)
    return float(np.mean(0.5 * (np.sum(z * z, axis=1) + z.shape[1] * LOG2PI) - ld) / z.shape[1])


def coordinate_control(x: np.ndarray, a_dim: int, rank: int) -> Endpoint:
    means = np.zeros((rank // 2, 2))
    scales = np.ones((rank // 2, 2))
    active = np.ones(rank // 2, dtype=bool)
    ep = Endpoint(np.arange(rank), means, scales, active, a_dim, x.shape[1] - a_dim)
    u, h = ep.basis()
    p = (x[:, a_dim:] @ u) @ h.T
    m = rank // 2
    for j in range(m):
        mu = float(np.mean(p[:, m + j]))
        sc = float(np.clip(np.std(p[:, m + j], ddof=1), 0.25, 4.0))
        means[j, :] = mu
        scales[j, :] = sc
    return ep


def parity_gap(endpoint: Endpoint, x: np.ndarray) -> float:
    z, _ = endpoint.transform(x)
    u, h = endpoint.basis()
    p = (z[:, endpoint.a_dim:] @ u) @ h.T
    state = endpoint.state(p[:, :2])
    target = np.where(p[:, 2] >= 0, 1.0, -1.0)
    return float(abs(np.mean(target[state == 1]) - np.mean(target[state == 0])))


def run_seed(seed: int) -> dict[str, float | int]:
    a_dim, b_dim, rank = 8, 72, 4
    teacher = Endpoint(
        np.arange(rank),
        np.array([[-1.3, 1.3], [-0.8, 0.8]], dtype=np.float64),
        np.array([[0.75, 1.25], [1.20, 0.70]], dtype=np.float64),
        np.ones(2, dtype=bool), a_dim, b_dim,
    )
    fit_base = np.random.default_rng(seed).normal(size=(24000, a_dim + b_dim))
    fit_x, _ = teacher.inverse(fit_base)
    test_base = np.random.default_rng(seed + 1000).normal(size=(40000, a_dim + b_dim))
    test_x, _ = teacher.inverse(test_base)

    u, h = teacher.basis()
    p = (fit_x[:, a_dim:] @ u) @ h.T
    state = teacher.state(p[:, :2])
    means = np.zeros((2, 2)); scales = np.ones((2, 2))
    for j in range(2):
        for s in (0, 1):
            mask = state == s
            means[j, s] = np.mean(p[mask, 2 + j])
            scales[j, s] = np.std(p[mask, 2 + j], ddof=1)
    candidate = Endpoint(np.arange(rank), means, scales, np.ones(2, dtype=bool), a_dim, b_dim)
    coordinate = coordinate_control(fit_x, a_dim, rank)
    identity = Endpoint(np.arange(rank), np.zeros((2, 2)), np.ones((2, 2)), np.zeros(2, dtype=bool), a_dim, b_dim)
    copied = candidate.copy()
    z, ld = candidate.transform(test_x)
    x2, ild = candidate.inverse(z)
    return {
        "seed": seed,
        "identity_nll": nll(identity, test_x),
        "coordinate_nll": nll(coordinate, test_x),
        "candidate_nll": nll(candidate, test_x),
        "coordinate_minus_candidate_nll": nll(coordinate, test_x) - nll(candidate, test_x),
        "identity_minus_candidate_nll": nll(identity, test_x) - nll(candidate, test_x),
        "candidate_parity_gap": parity_gap(candidate, test_x),
        "coordinate_parity_gap": parity_gap(coordinate, test_x),
        "copy_mismatch": float(np.max(np.abs(candidate.transform(test_x)[0] - copied.transform(test_x)[0]))),
        "roundtrip_max_abs_error": float(np.max(np.abs(test_x - x2))),
        "logdet_cancel_max_abs_error": float(np.max(np.abs(ld + ild))),
    }


def main() -> None:
    rows = [run_seed(seed) for seed in range(9600, 9605)]
    diff = np.array([row["coordinate_minus_candidate_nll"] for row in rows])
    mean = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
    t4 = 2.7764451051977987
    gates = {
        "nll_every_seed": bool(np.all(diff >= 0.01)),
        "parity_every_seed": bool(all(row["candidate_parity_gap"] <= 0.05 for row in rows)),
        "copy_exact": bool(all(row["copy_mismatch"] == 0 for row in rows)),
        "roundtrip": bool(all(row["roundtrip_max_abs_error"] <= 1e-12 for row in rows)),
        "logdet": bool(all(row["logdet_cancel_max_abs_error"] <= 1e-12 for row in rows)),
    }
    verdict = {
        "known_truth_pass": bool(all(gates.values())),
        "gates": gates,
        "coordinate_minus_candidate_mean": mean,
        "paired_95_ci": [mean - t4 * se, mean + t4 * se],
        "rows": rows,
        "confirmation_opened": False,
    }
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r10_known_truth_verdict.json")
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if not verdict["known_truth_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
