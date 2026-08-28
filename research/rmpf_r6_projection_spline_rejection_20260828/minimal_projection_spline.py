"""Minimal exact all-coordinate projection RQ-spline reproduction.

This file checks the algebraic RMPF-R6 endpoint layer only.  The complete
opened-development release, hashes, controls, and failed child rounds are kept
in the accompanying research artifact bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np


@dataclass(frozen=True)
class RQ:
    x: np.ndarray
    y: np.ndarray
    d: np.ndarray
    bound: float

    def _bin(self, values: np.ndarray, knots: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(knots, values, side="right") - 1, 0, len(knots) - 2)

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        output = values.copy()
        logdet = np.zeros_like(values)
        mask = (values > -self.bound) & (values < self.bound)
        if not np.any(mask):
            return output, logdet
        v = values[mask]
        k = self._bin(v, self.x)
        x0, x1 = self.x[k], self.x[k + 1]
        y0, y1 = self.y[k], self.y[k + 1]
        d0, d1 = self.d[k], self.d[k + 1]
        width, height = x1 - x0, y1 - y0
        slope = height / width
        theta = (v - x0) / width
        one_minus = 1.0 - theta
        product = theta * one_minus
        denominator = slope + (d0 + d1 - 2.0 * slope) * product
        output[mask] = y0 + height * (slope * theta**2 + d0 * product) / denominator
        derivative = slope**2 * (
            d1 * theta**2 + 2.0 * slope * product + d0 * one_minus**2
        ) / denominator**2
        if np.any(derivative <= 0):
            raise FloatingPointError("nonpositive derivative")
        logdet[mask] = np.log(derivative)
        return output, logdet

    def inverse(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        output = values.copy()
        logdet = np.zeros_like(values)
        mask = (values > -self.bound) & (values < self.bound)
        if not np.any(mask):
            return output, logdet
        v = values[mask]
        k = self._bin(v, self.y)
        x0, x1 = self.x[k], self.x[k + 1]
        y0, y1 = self.y[k], self.y[k + 1]
        d0, d1 = self.d[k], self.d[k + 1]
        width, height = x1 - x0, y1 - y0
        slope = height / width
        yrel = v - y0
        common = d0 + d1 - 2.0 * slope
        a = yrel * common + height * (slope - d0)
        b = height * d0 - yrel * common
        c = -slope * yrel
        disc = np.maximum(b * b - 4.0 * a * c, 0.0)
        theta = np.empty_like(v)
        linear = np.abs(a) < 1e-12
        theta[linear] = -c[linear] / np.where(np.abs(b[linear]) > 1e-12, b[linear], 1.0)
        theta[~linear] = 2.0 * c[~linear] / (-b[~linear] - np.sqrt(disc[~linear]))
        theta = np.clip(theta, 0.0, 1.0)
        output[mask] = x0 + theta * width
        one_minus = 1.0 - theta
        product = theta * one_minus
        denominator = slope + common * product
        derivative = slope**2 * (
            d1 * theta**2 + 2.0 * slope * product + d0 * one_minus**2
        ) / denominator**2
        logdet[mask] = -np.log(derivative)
        return output, logdet


@dataclass(frozen=True)
class ProjectionLayer:
    basis: np.ndarray                 # shape (detail_dimension, rank)
    splines: tuple[RQ, ...]
    active_dimension: int

    def transform(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        detail = state[:, self.active_dimension:]
        projected = detail @ self.basis
        moved = projected.copy()
        logdet = np.zeros(len(state))
        for j, spline in enumerate(self.splines):
            moved[:, j], ld = spline.forward(projected[:, j])
            logdet += ld
        result = state.copy()
        result[:, self.active_dimension:] = detail + (moved - projected) @ self.basis.T
        return result, logdet

    def inverse(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        detail = state[:, self.active_dimension:]
        projected = detail @ self.basis
        restored = projected.copy()
        logdet = np.zeros(len(state))
        for j, spline in enumerate(self.splines):
            restored[:, j], ld = spline.inverse(projected[:, j])
            logdet += ld
        result = state.copy()
        result[:, self.active_dimension:] = detail + (restored - projected) @ self.basis.T
        return result, logdet


def demo() -> dict[str, float]:
    rng = np.random.default_rng(20260828)
    q, rank, active = 9, 3, 4
    basis, _ = np.linalg.qr(rng.normal(size=(q, rank)))
    knots = np.linspace(-4.0, 4.0, 9)
    # Nonidentity but monotone map with exact identity tails.
    target = knots + 0.18 * np.sin(np.pi * knots / 4.0)
    target[0], target[-1] = -4.0, 4.0
    derivative = np.ones_like(knots)
    splines = tuple(RQ(knots.copy(), target.copy(), derivative.copy(), 4.0) for _ in range(rank))
    layer = ProjectionLayer(basis, splines, active)
    source = rng.normal(size=(256, active + q))
    transformed, ld = layer.transform(source)
    restored, ild = layer.inverse(transformed)

    # Dense finite-difference determinant for one point.
    epsilon = 1e-6
    x = source[:1]
    jac = np.empty((x.shape[1], x.shape[1]))
    for j in range(x.shape[1]):
        plus, minus = x.copy(), x.copy()
        plus[0, j] += epsilon
        minus[0, j] -= epsilon
        yp, _ = layer.transform(plus)
        ym, _ = layer.transform(minus)
        jac[:, j] = (yp[0] - ym[0]) / (2.0 * epsilon)
    sign, numeric_logdet = np.linalg.slogdet(jac)
    result = {
        "roundtrip_max": float(np.max(np.abs(restored - source))),
        "logdet_cancel_max": float(np.max(np.abs(ld + ild))),
        "dense_jacobian_sign": float(sign),
        "dense_logdet_error": float(abs(numeric_logdet - ld[0])),
    }
    if result["roundtrip_max"] >= 1e-9 or result["dense_logdet_error"] >= 1e-6:
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
