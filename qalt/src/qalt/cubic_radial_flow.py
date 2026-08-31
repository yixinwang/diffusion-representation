"""Exact dimension-preserving cubic radial flow used by the aligned toy.

The module is deliberately independent of data loading and model fitting.
Arrays use the final axis as the transformed vector dimension.
"""

from __future__ import annotations

import math

import numpy as np


LOG_2PI = math.log(2.0 * math.pi)


def _validated_vectors(values: np.ndarray, name: str) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] < 1:
        raise ValueError(f"{name} must have a nonempty final vector axis")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must be finite")
    return vectors


def _validated_parameters(a: float, scale: float) -> tuple[float, float]:
    coefficient = float(a)
    radial_scale = float(scale)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("a must be finite and nonnegative")
    if not math.isfinite(radial_scale) or radial_scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return coefficient, radial_scale


def forward(
    z: np.ndarray,
    a: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map base vectors forward and return the exact log determinant."""

    base = _validated_vectors(z, "z")
    coefficient, radial_scale = _validated_parameters(a, scale)
    radius_squared = np.sum(base * base, axis=-1)
    multiplier = radial_scale * (1.0 + coefficient * radius_squared)
    values = multiplier[..., None] * base
    dimension = base.shape[-1]
    log_det = (
        dimension * math.log(radial_scale)
        + (dimension - 1) * np.log1p(coefficient * radius_squared)
        + np.log1p(3.0 * coefficient * radius_squared)
    )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(log_det)):
        raise FloatingPointError("cubic radial forward map produced a nonfinite value")
    return values, log_det


def inverse(
    x: np.ndarray,
    a: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the cubic radial map and return the inverse log determinant."""

    values = _validated_vectors(x, "x")
    coefficient, radial_scale = _validated_parameters(a, scale)
    output_radius = np.linalg.norm(values, axis=-1) / radial_scale
    if coefficient == 0.0:
        radius = output_radius
    else:
        root = math.sqrt(3.0 * coefficient)
        radius = (2.0 / root) * np.sinh(
            np.arcsinh(1.5 * root * output_radius) / 3.0
        )
    denominator = radial_scale * (1.0 + coefficient * radius * radius)
    base = values / denominator[..., None]
    _, forward_log_det = forward(base, coefficient, radial_scale)
    return base, -forward_log_det


def standard_normal_log_prob(z: np.ndarray) -> np.ndarray:
    """Return the rowwise standard-normal log density."""

    base = _validated_vectors(z, "z")
    return -0.5 * (
        base.shape[-1] * LOG_2PI + np.sum(base * base, axis=-1)
    )


def log_prob(x: np.ndarray, a: float, scale: float = 1.0) -> np.ndarray:
    """Evaluate the exactly normalized pushforward density."""

    base, inverse_log_det = inverse(x, a, scale)
    return standard_normal_log_prob(base) + inverse_log_det


def best_affine_variance(a: float, scale: float = 1.0, dimension: int = 9) -> float:
    """Return the population variance of one target coordinate."""

    coefficient, radial_scale = _validated_parameters(a, scale)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    factor = (
        1.0
        + 2.0 * coefficient * (dimension + 2)
        + coefficient * coefficient * (dimension + 2) * (dimension + 4)
    )
    return radial_scale * radial_scale * factor


def isotropic_normal_log_prob(x: np.ndarray, variance: float) -> np.ndarray:
    """Evaluate a centered isotropic Gaussian with a supplied variance."""

    values = _validated_vectors(x, "x")
    value = float(variance)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("variance must be finite and positive")
    return -0.5 * (
        values.shape[-1] * (LOG_2PI + math.log(value))
        + np.sum(values * values, axis=-1) / value
    )


def jensen_gap_lower_bound(a: float, dimension: int = 9) -> float:
    """Return the checked KL lower bound per coordinate against affine Gaussians."""

    coefficient, _ = _validated_parameters(a, 1.0)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    variance_factor = best_affine_variance(coefficient, 1.0, dimension)
    return (
        0.5 * math.log(variance_factor)
        - ((dimension - 1) / dimension) * math.log1p(coefficient * dimension)
        - math.log1p(3.0 * coefficient * dimension) / dimension
    )


def band_energy_correlation(a: float, dimension: int = 9) -> float:
    """Return the exact correlation of two 3D band energies for dimension nine."""

    coefficient, _ = _validated_parameters(a, 1.0)
    if dimension != 9:
        raise ValueError("the registered three-band formula requires dimension=9")

    def chi_square_moment(order: int) -> float:
        value = 1.0
        for index in range(order):
            value *= dimension + 2 * index
        return value

    mean_radial_energy = (
        chi_square_moment(1)
        + 2.0 * coefficient * chi_square_moment(2)
        + coefficient * coefficient * chi_square_moment(3)
    )
    second_radial_energy = sum(
        multiplier
        * coefficient**power
        * chi_square_moment(power + 2)
        for power, multiplier in enumerate((1.0, 4.0, 6.0, 4.0, 1.0))
    )
    covariance = second_radial_energy / 11.0 - mean_radial_energy**2 / 9.0
    variance = 5.0 * second_radial_energy / 33.0 - mean_radial_energy**2 / 9.0
    return covariance / variance
