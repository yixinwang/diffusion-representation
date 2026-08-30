#!/usr/bin/env python3
"""Self-contained deterministic/oracle reproduction for the compact RMPF-R19 milestone.

Requires numpy and scipy. It reproduces the five-seed exact-oracle proper-energy gate
and checks inverse/Jacobian cancellation for the periodic rational-quadratic map.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.distance import cdist

PI = math.pi
TWOPI = 2.0 * PI
SEEDS = range(10030, 10035)
D = 16
TRUE_VECTOR = np.array([4.0, -3.0, 2.2, -1.6, 0.85, -0.45, 0.22, -0.10], float)
TRUE_VECTOR /= np.linalg.norm(TRUE_VECTOR)
TRUE_BETA = np.array([[0.9, -0.5], [-0.7, 0.8]], float)
TRUE_KNOTS = (
    np.array([-PI, -3.135, -3.0, -0.01, 0.01, 3.0, PI], float),
    np.array([-PI, -3.135, -2.8, -0.01, 0.01, 2.8, PI], float),
)


def wrap(x):
    x = np.asarray(x, float)
    return (x + PI) % TWOPI - PI


def derivatives(x, y):
    slopes = np.diff(y) / np.diff(x)
    d = np.ones_like(x)
    for j in range(1, len(x) - 1):
        d[j] = 2.0 / (1.0 / slopes[j - 1] + 1.0 / slopes[j])
    d = np.clip(d, 0.05, 50.0)
    d[0] = d[-1] = 1.0
    return d


class RQS:
    def __init__(self, x):
        self.x = np.asarray(x, float)
        self.y = np.linspace(-PI, PI, len(self.x))
        self.d = derivatives(self.x, self.y)

    @staticmethod
    def _idx(v, knots):
        return np.clip(np.searchsorted(knots, v, side="right") - 1, 0, len(knots) - 2)

    def forward(self, value):
        x = wrap(value)
        i = self._idx(x, self.x)
        x0, x1 = self.x[i], self.x[i + 1]
        y0, y1 = self.y[i], self.y[i + 1]
        d0, d1 = self.d[i], self.d[i + 1]
        width, height = x1 - x0, y1 - y0
        delta = height / width
        theta = (x - x0) / width
        omt = 1.0 - theta
        tomt = theta * omt
        common = d0 + d1 - 2.0 * delta
        den = delta + common * tomt
        y = y0 + height * (delta * theta * theta + d0 * tomt) / den
        num = delta * delta * (d1 * theta * theta + 2.0 * delta * tomt + d0 * omt * omt)
        deriv = num / (den * den)
        return wrap(y), np.log(deriv)

    def inverse(self, value):
        y = wrap(value)
        i = self._idx(y, self.y)
        x0, x1 = self.x[i], self.x[i + 1]
        y0, y1 = self.y[i], self.y[i + 1]
        d0, d1 = self.d[i], self.d[i + 1]
        width, height = x1 - x0, y1 - y0
        delta = height / width
        yrel = y - y0
        common = d0 + d1 - 2.0 * delta
        a = yrel * common + height * (delta - d0)
        b = height * d0 - yrel * common
        c = -delta * yrel
        disc = np.maximum(b * b - 4.0 * a * c, 0.0)
        root = np.sqrt(disc)
        theta = np.empty_like(y)
        linear = np.abs(a) < 1e-12
        theta[linear] = -c[linear] / np.where(np.abs(b[linear]) > 1e-12, b[linear], 1.0)
        theta[~linear] = 2.0 * c[~linear] / (-b[~linear] - root[~linear])
        theta = np.clip(theta, 0.0, 1.0)
        x = x0 + theta * width
        omt = 1.0 - theta
        tomt = theta * omt
        den = delta + common * tomt
        num = delta * delta * (d1 * theta * theta + 2.0 * delta * tomt + d0 * omt * omt)
        deriv = num / (den * den)
        return wrap(x), -np.log(deriv)


SPLINES = (RQS(TRUE_KNOTS[0]), RQS(TRUE_KNOTS[1]))


def phase(phi, beta):
    return beta[0] * np.sin(phi) + beta[1] * np.cos(phi)


def transform(samples, inverse=False):
    out = np.asarray(samples, float).copy()
    logdet = np.zeros(len(out))
    state = (out[:, :8] @ TRUE_VECTOR > 0).astype(int)
    cond = out[:, 8:12]
    target = out[:, 12:16]
    for plane in (0, 1):
        k = 2 * plane
        ca = np.arctan2(cond[:, k + 1], cond[:, k])
        ta = np.arctan2(target[:, k + 1], target[:, k])
        radius = np.linalg.norm(target[:, k:k + 2], axis=1)
        mapped = ta.copy()
        for s in (0, 1):
            mask = state == s
            residual = wrap(ta[mask] - phase(ca[mask], TRUE_BETA[s]))
            z, ld = SPLINES[s].inverse(residual) if inverse else SPLINES[s].forward(residual)
            mapped[mask] = wrap(z + phase(ca[mask], TRUE_BETA[s]))
            logdet[mask] += ld
        target[:, k] = radius * np.cos(mapped)
        target[:, k + 1] = radius * np.sin(mapped)
    out[:, 12:16] = target
    return out, logdet


def energy(generated, reference):
    return float(np.mean(cdist(generated, reference)) - 0.5 * np.mean(cdist(generated, generated)))


def interval(values):
    x = np.asarray(values, float)
    mean = x.mean()
    half = 2.7764451051977987 * x.std(ddof=1) / math.sqrt(len(x))
    return mean, mean - half, mean + half


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(512, D))
    z, ld = transform(x, inverse=False)
    xr, ild = transform(z, inverse=True)
    print("roundtrip", float(np.max(np.abs(xr - x))))
    print("logdet_cancel", float(np.max(np.abs(ld + ild))))

    gains = []
    for seed in SEEDS:
        r = np.random.default_rng(seed)
        test_base = r.normal(size=(4096, D))
        reference, _ = transform(test_base, inverse=True)
        reference = reference[:1024]
        generation_base = r.normal(size=(1024, D))
        oracle, _ = transform(generation_base, inverse=True)
        gain = energy(generation_base, reference) - energy(oracle, reference)
        gains.append(gain)
        print(seed, f"{gain:.12f}")
    mean, lower, upper = interval(gains)
    print("oracle_energy_gain", mean, lower, upper)
    assert lower >= 0.005


if __name__ == "__main__":
    main()
