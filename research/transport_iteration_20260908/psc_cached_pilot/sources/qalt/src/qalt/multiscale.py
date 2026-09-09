from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


LOG_2PI = math.log(2.0 * math.pi)


def _check_even(array: np.ndarray, axes: tuple[int, ...]) -> None:
    if any(array.shape[axis] % 2 for axis in axes):
        raise ValueError("every transformed axis must have even length")


def haar_forward(array: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Apply a local orthonormal Haar transform along the declared axes."""
    result = np.asarray(array, dtype=float).copy()
    _check_even(result, axes)
    root_two = math.sqrt(2.0)
    for axis in axes:
        even = np.take(result, np.arange(0, result.shape[axis], 2), axis=axis)
        odd = np.take(result, np.arange(1, result.shape[axis], 2), axis=axis)
        result = np.concatenate(((even + odd) / root_two, (even - odd) / root_two), axis=axis)
    return result


def haar_inverse(coefficients: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Invert :func:`haar_forward` using the same local lifting operations."""
    result = np.asarray(coefficients, dtype=float).copy()
    _check_even(result, axes)
    root_two = math.sqrt(2.0)
    for axis in reversed(axes):
        half = result.shape[axis] // 2
        low = np.take(result, np.arange(half), axis=axis)
        high = np.take(result, np.arange(half, 2 * half), axis=axis)
        even = (low + high) / root_two
        odd = (low - high) / root_two
        shape = list(result.shape)
        restored = np.empty(shape, dtype=float)
        index = [slice(None)] * result.ndim
        index[axis] = slice(0, None, 2)
        restored[tuple(index)] = even
        index[axis] = slice(1, None, 2)
        restored[tuple(index)] = odd
        result = restored
    return result


def band_slices(shape: tuple[int, ...], axes: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """Return the coarse band followed by all detail bands."""
    bands: list[tuple[slice, ...]] = []
    for mask in range(2 ** len(axes)):
        index = [slice(None)] * len(shape)
        for bit, axis in enumerate(axes):
            half = shape[axis] // 2
            index[axis] = slice(half, None) if mask & (1 << bit) else slice(0, half)
        bands.append(tuple(index))
    return bands


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def parent_features(parent: np.ndarray) -> np.ndarray:
    flat = np.asarray(parent, dtype=float).reshape(-1)
    return np.column_stack((np.ones_like(flat), np.tanh(flat), np.sin(flat)))


@dataclass(frozen=True)
class MixtureFiber:
    logistic: np.ndarray
    scales: np.ndarray

    def probabilities(self, parent: np.ndarray) -> np.ndarray:
        return sigmoid(parent_features(parent) @ self.logistic)

    def log_prob(self, detail: np.ndarray, parent: np.ndarray) -> np.ndarray:
        values = np.asarray(detail, dtype=float).reshape(-1)
        probability = self.probabilities(parent)
        component_logs = []
        for weight, scale in ((1.0 - probability, self.scales[0]), (probability, self.scales[1])):
            component_logs.append(
                np.log(np.maximum(weight, 1e-15))
                - math.log(scale)
                - 0.5 * (LOG_2PI + (values / scale) ** 2)
            )
        stacked = np.stack(component_logs)
        maximum = stacked.max(axis=0)
        return maximum + np.log(np.exp(stacked - maximum).sum(axis=0))

    def sample(self, parent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probability = self.probabilities(parent)
        component = rng.random(len(probability)) < probability
        scales = self.scales[component.astype(int)]
        return (scales * rng.normal(size=len(scales))).reshape(np.shape(parent))


def fit_mixture_fiber(detail: np.ndarray, parent: np.ndarray, iterations: int = 30) -> MixtureFiber:
    """Fit a two-scale, parent-gated zero-mean Gaussian mixture by EM."""
    values = np.asarray(detail, dtype=float).reshape(-1)
    features = parent_features(parent)
    if len(values) != len(features):
        raise ValueError("detail and parent must have the same number of coefficients")
    variance = max(float(np.mean(values**2)), 1e-6)
    scales = np.sqrt(variance) * np.array([0.55, 1.45])
    logistic = np.zeros(features.shape[1])
    ridge = 1e-5 * np.eye(features.shape[1])
    for _ in range(iterations):
        probability = sigmoid(features @ logistic)
        log_small = np.log(np.maximum(1.0 - probability, 1e-15)) - np.log(scales[0]) - 0.5 * (values / scales[0]) ** 2
        log_large = np.log(np.maximum(probability, 1e-15)) - np.log(scales[1]) - 0.5 * (values / scales[1]) ** 2
        responsibility = sigmoid(log_large - log_small)
        for _ in range(5):
            fitted = sigmoid(features @ logistic)
            gradient = features.T @ (responsibility - fitted) - ridge @ logistic
            weights = np.maximum(fitted * (1.0 - fitted), 1e-6)
            hessian = features.T @ (features * weights[:, None]) + ridge
            logistic += np.linalg.solve(hessian, gradient)
        small_weight = np.maximum(1.0 - responsibility, 1e-12)
        large_weight = np.maximum(responsibility, 1e-12)
        scales = np.sqrt(
            [
                np.sum(small_weight * values**2) / np.sum(small_weight),
                np.sum(large_weight * values**2) / np.sum(large_weight),
            ]
        )
        scales = np.maximum(scales, 1e-4)
        if scales[0] > scales[1]:
            scales = scales[::-1]
            logistic = -logistic
    return MixtureFiber(logistic=logistic, scales=scales)


def diagonal_scale(detail: np.ndarray) -> float:
    return math.sqrt(max(float(np.mean(np.asarray(detail, dtype=float) ** 2)), 1e-12))


def gaussian_log_prob(detail: np.ndarray, scale: float) -> np.ndarray:
    values = np.asarray(detail, dtype=float)
    return -math.log(scale) - 0.5 * (LOG_2PI + (values / scale) ** 2)


def euler_scale(scale: np.ndarray | float, steps: int) -> np.ndarray:
    target = np.asarray(scale, dtype=float)
    base = 1.0 + np.log(target) / steps
    if steps <= 0 or np.any(base <= 0):
        raise ValueError("Euler scale is unstable")
    return base**steps


def token_benchmark(shape: tuple[int, ...], steps: int, repeats: int = 9) -> dict[str, float]:
    """Time matched local samplers through the shared inverse Haar decoder."""
    total_tokens = int(np.prod(shape))
    active_tokens = total_tokens // (2 ** len(shape))
    if not 0 < active_tokens < total_tokens:
        raise ValueError("active token count must be between zero and total tokens")
    rng = np.random.default_rng(91)
    axes = tuple(range(1, len(shape) + 1))
    full_initial = rng.normal(size=(8, *shape))
    coarse_shape = (8, *(length // 2 for length in shape))
    active_initial = rng.normal(size=coarse_shape)

    def full_timed() -> float:
        values = full_initial.copy()
        started = time.perf_counter()
        for _ in range(steps):
            scratch = np.tanh(values)
            values += 0.01 * scratch
        output = haar_inverse(values, axes)
        return time.perf_counter() - started + 0.0 * float(output.reshape(-1)[0])

    def qalt_timed() -> float:
        active = active_initial.copy()
        started = time.perf_counter()
        for _ in range(steps):
            scratch = np.tanh(active)
            active += 0.01 * scratch
        coefficients = np.tanh(full_initial)
        coarse_index = band_slices(coefficients.shape, axes)[0]
        coefficients[coarse_index] = active
        output = haar_inverse(coefficients, axes)
        return time.perf_counter() - started + 0.0 * float(output.reshape(-1)[0])

    full_times = [full_timed() for _ in range(repeats)]
    qalt_times = [qalt_timed() for _ in range(repeats)]
    item = full_initial.dtype.itemsize * len(full_initial)
    return {
        "full_seconds": float(np.median(full_times)),
        "qalt_seconds": float(np.median(qalt_times)),
        "latency_ratio": float(np.median(qalt_times) / np.median(full_times)),
        "full_peak_working_bytes": float(2 * total_tokens * item),
        "qalt_peak_working_bytes": float((total_tokens + 2 * active_tokens) * item),
        "memory_ratio": float((total_tokens + 2 * active_tokens) / (2 * total_tokens)),
        "full_token_updates": float(steps * total_tokens),
        "qalt_token_updates": float(steps * active_tokens + total_tokens - active_tokens),
    }
