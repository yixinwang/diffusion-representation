from __future__ import annotations

import numpy as np


def sample_active(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a non-Gaussian, heavy-tailed active mixture."""
    component = rng.integers(0, 3, size=n)
    locations = np.stack(
        [
            np.linspace(-2.0, 0.5, dim),
            np.linspace(0.2, 2.2, dim),
            np.sin(np.arange(dim) + 0.5),
        ]
    )
    scales = np.array([0.35, 0.55, 0.8])
    return locations[component] + scales[component, None] * rng.standard_t(5, size=(n, dim))


def _decoder_shift(z_dim: int, r: np.ndarray) -> np.ndarray:
    shift = np.empty((len(r), z_dim), dtype=r.dtype)
    r_mean = r.mean(axis=1)
    for j in range(z_dim):
        shift[:, j] = 0.3 * np.tanh(r[:, j % r.shape[1]]) + 0.1 * np.sin(r_mean + j)
    return shift


def decoder(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Apply a nonlinear triangular bijection with unit Jacobian determinant."""
    if z.ndim != 2 or r.ndim != 2 or len(z) != len(r):
        raise ValueError("z and r must be aligned matrices")
    if z.shape[1] == 0:
        return r.copy()
    return np.concatenate([z + _decoder_shift(z.shape[1], r), r], axis=1)


def inverse_decoder(x: np.ndarray, active_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`decoder` exactly."""
    if x.ndim != 2 or not 0 < active_dim < x.shape[1]:
        raise ValueError("x must be [n, d+q] with 0 < d < d+q")
    r = x[:, active_dim:].copy()
    z = x[:, :active_dim] - _decoder_shift(active_dim, r)
    return z, r


def euler_scale(target_scale: np.ndarray | float, steps: int) -> np.ndarray:
    """Endpoint scale from Euler integration of r' = log(s) r."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    scale = np.asarray(target_scale, dtype=float)
    base = 1.0 + np.log(scale) / steps
    if np.any(base <= 0):
        raise ValueError("Euler stability factor must be positive")
    return base**steps


def fiber_kl(target_scale: np.ndarray | float, model_scale: np.ndarray | float) -> float:
    """Forward KL between centered diagonal Gaussians."""
    target = np.asarray(target_scale, dtype=float)
    model = np.asarray(model_scale, dtype=float)
    if np.any(target <= 0) or np.any(model <= 0):
        raise ValueError("scales must be positive")
    ratio = (target / model) ** 2
    return float(0.5 * np.sum(ratio - 1.0 - np.log(ratio)))


def fiber_w2_squared(target_scale: np.ndarray | float, model_scale: np.ndarray | float) -> float:
    """Squared Wasserstein distance between centered diagonal Gaussians."""
    target = np.asarray(target_scale, dtype=float)
    model = np.asarray(model_scale, dtype=float)
    return float(np.sum((target - model) ** 2))


def pooled_variance(samples: np.ndarray) -> np.ndarray:
    """Fit one shared zero-mean variance and repeat it over coordinates."""
    if samples.ndim != 2:
        raise ValueError("samples must be [n, q]")
    value = float(np.mean(samples**2))
    return np.full(samples.shape[1], value)


def separate_variances(samples: np.ndarray) -> np.ndarray:
    """Fit one zero-mean variance for each coordinate."""
    if samples.ndim != 2:
        raise ValueError("samples must be [n, q]")
    return np.mean(samples**2, axis=0)
