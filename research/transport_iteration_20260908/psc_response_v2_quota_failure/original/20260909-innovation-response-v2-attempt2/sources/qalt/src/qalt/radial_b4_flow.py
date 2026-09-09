"""Exact reversible coordinates and radial children for fitted B4 blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import (
    betainc,
    betaincinv,
    gammainc,
    gammaincc,
    gammainccinv,
    gammaincinv,
    gammaln,
    log_ndtr,
    logsumexp,
)

from qalt.cubic_radial_flow import (
    best_affine_variance,
    forward as cubic_forward,
    inverse as cubic_inverse,
    log_prob as cubic_log_prob,
    standard_normal_log_prob,
)
from qalt.rgb_block import FixedShapeGSM


RGB_DIMENSION = 3
JOINT_DIMENSION = 9
LOG_2PI = math.log(2.0 * math.pi)
LOG_TWO = math.log(2.0)
LOG_SQRT_TWO_OVER_PI = 0.5 * math.log(2.0 / math.pi)


def _as_vectors(values: np.ndarray, dimension: int, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != dimension:
        raise ValueError(f"{name} must have final dimension {dimension}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.reshape(-1, dimension), array.shape[:-1]


def _chi_radius_from_probabilities(
    cdf: np.ndarray,
    survival: np.ndarray,
    dimension: int,
) -> np.ndarray:
    output = np.empty_like(cdf)
    lower = cdf <= 0.5
    if np.any((cdf[lower] < 0.0) | (cdf[lower] > 1.0)):
        raise FloatingPointError("invalid lower-tail probability")
    if np.any((survival[~lower] <= 0.0) | (survival[~lower] > 1.0)):
        raise FloatingPointError("upper-tail probability underflow")
    output[lower] = np.sqrt(2.0 * gammaincinv(0.5 * dimension, cdf[lower]))
    output[~lower] = np.sqrt(2.0 * gammainccinv(0.5 * dimension, survival[~lower]))
    return output


def _beta_lower_tail_inverse(
    shape_a: float,
    shape_b: float,
    probability: np.ndarray,
) -> np.ndarray:
    """Invert a beta lower tail without forming its complementary quantile."""

    values = np.asarray(probability, dtype=np.float64)
    if np.any((values < 0.0) | (values > 1.0) | ~np.isfinite(values)):
        raise FloatingPointError("invalid beta lower-tail probability")
    quantile = np.asarray(betaincinv(shape_a, shape_b, values), dtype=np.float64)
    fallback = (values > 0.0) & (values < 1.0) & (
        ~np.isfinite(quantile) | (quantile <= 0.0) | (quantile >= 1.0)
    )
    if np.any(fallback):
        selected = values[fallback]
        log_lower = np.full(
            len(selected), math.log(np.nextafter(0.0, 1.0)), dtype=np.float64
        )
        log_upper = np.zeros(len(selected), dtype=np.float64)
        for _ in range(80):
            log_midpoint = 0.5 * (log_lower + log_upper)
            midpoint_cdf = betainc(shape_a, shape_b, np.exp(log_midpoint))
            below = midpoint_cdf < selected
            log_lower = np.where(below, log_midpoint, log_lower)
            log_upper = np.where(below, log_upper, log_midpoint)
        quantile[fallback] = np.exp(0.5 * (log_lower + log_upper))
    return quantile


def _chi3_log_survival(radius: np.ndarray) -> np.ndarray:
    values = np.asarray(radius, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("chi radii must be finite and nonnegative")
    first = LOG_TWO + log_ndtr(-values)
    second = np.full_like(values, -math.inf)
    positive = values > 0.0
    second[positive] = (
        LOG_SQRT_TWO_OVER_PI
        + np.log(values[positive])
        - 0.5 * values[positive] ** 2
    )
    return np.logaddexp(first, second)


def _mixture_chi3_log_survival(
    radius: np.ndarray,
    mixture: FixedShapeGSM,
) -> np.ndarray:
    scaled = np.asarray(radius, dtype=np.float64)[:, None] / mixture.scales[None, :]
    return logsumexp(
        np.log(mixture.weights)[None, :] + _chi3_log_survival(scaled), axis=1
    )


def _chi3_radius_from_log_survival(log_survival: np.ndarray) -> np.ndarray:
    target = np.asarray(log_survival, dtype=np.float64)
    if np.any(~np.isfinite(target)) or np.any(target >= 0.0):
        raise ValueError("log survival probabilities must be finite and negative")
    lower = np.zeros_like(target)
    upper = np.maximum(1.0, np.sqrt(np.maximum(0.0, -2.0 * target)) + 2.0)
    for _ in range(8):
        needs_growth = _chi3_log_survival(upper) > target
        upper = np.where(needs_growth, 2.0 * upper, upper)
    if np.any(_chi3_log_survival(upper) > target):
        raise FloatingPointError("failed to bracket a chi-3 log-survival quantile")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        below = _chi3_log_survival(midpoint) > target
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    return 0.5 * (lower + upper)


def gaussianize_gsm(
    mixture: FixedShapeGSM,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map an elliptical RGB GSM exactly to a standard three-normal block."""

    rows, leading = _as_vectors(residual, RGB_DIMENSION, "residual")
    whitened = solve_triangular(
        mixture.cholesky, rows.T, lower=True, check_finite=False
    ).T
    radius = np.linalg.norm(whitened, axis=1)
    scaled_square = radius[:, None] ** 2 / (2.0 * mixture.scales[None, :] ** 2)
    cdf = np.sum(mixture.weights[None, :] * gammainc(1.5, scaled_square), axis=1)
    log_survival = _mixture_chi3_log_survival(radius, mixture)
    target_radius = np.empty_like(radius)
    lower_tail = cdf <= 0.5
    target_radius[lower_tail] = np.sqrt(
        2.0 * gammaincinv(1.5, cdf[lower_tail])
    )
    target_radius[~lower_tail] = _chi3_radius_from_log_survival(
        log_survival[~lower_tail]
    )
    nonzero = radius > 0.0
    multiplier = np.empty_like(radius)
    multiplier[nonzero] = target_radius[nonzero] / radius[nonzero]
    multiplier[~nonzero] = float(
        np.sum(mixture.weights / mixture.scales**RGB_DIMENSION)
        ** (1.0 / RGB_DIMENSION)
    )
    gaussian = whitened * multiplier[:, None]
    log_det = mixture.log_prob(rows) - standard_normal_log_prob(gaussian)
    if not np.all(np.isfinite(gaussian)) or not np.all(np.isfinite(log_det)):
        raise FloatingPointError("GSM Gaussianization produced a nonfinite value")
    return gaussian.reshape((*leading, RGB_DIMENSION)), log_det.reshape(leading)


def invert_gaussianized_gsm(
    mixture: FixedShapeGSM,
    gaussian: np.ndarray,
    iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`gaussianize_gsm` by a stable bracketed radial solve."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    rows, leading = _as_vectors(gaussian, RGB_DIMENSION, "gaussian")
    target_radius = np.linalg.norm(rows, axis=1)
    target_square = 0.5 * target_radius**2
    target_cdf = gammainc(1.5, target_square)
    target_log_survival = _chi3_log_survival(target_radius)
    lower = float(np.min(mixture.scales)) * target_radius
    upper = float(np.max(mixture.scales)) * target_radius
    use_lower_tail = target_cdf <= 0.5
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        scaled_square = midpoint[:, None] ** 2 / (2.0 * mixture.scales[None, :] ** 2)
        mixture_cdf = np.sum(
            mixture.weights[None, :] * gammainc(1.5, scaled_square), axis=1
        )
        mixture_log_survival = _mixture_chi3_log_survival(midpoint, mixture)
        below = np.where(
            use_lower_tail,
            mixture_cdf < target_cdf,
            mixture_log_survival > target_log_survival,
        )
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    source_radius = 0.5 * (lower + upper)
    nonzero = target_radius > 0.0
    multiplier = np.empty_like(target_radius)
    multiplier[nonzero] = source_radius[nonzero] / target_radius[nonzero]
    multiplier[~nonzero] = float(
        np.sum(mixture.weights / mixture.scales**RGB_DIMENSION)
        ** (-1.0 / RGB_DIMENSION)
    )
    whitened = rows * multiplier[:, None]
    residual = whitened @ mixture.cholesky.T
    inverse_log_det = standard_normal_log_prob(rows) - mixture.log_prob(residual)
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(inverse_log_det)):
        raise FloatingPointError("inverse GSM Gaussianization produced a nonfinite value")
    return residual.reshape((*leading, RGB_DIMENSION)), inverse_log_det.reshape(leading)


def covariance_normalized_cubic_scale(a: float) -> float:
    return 1.0 / math.sqrt(best_affine_variance(a, dimension=JOINT_DIMENSION))


def cubic_to_base(values: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    rows, leading = _as_vectors(values, JOINT_DIMENSION, "values")
    base, log_det = cubic_inverse(rows, a, covariance_normalized_cubic_scale(a))
    return base.reshape((*leading, JOINT_DIMENSION)), log_det.reshape(leading)


def cubic_from_base(base: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    rows, leading = _as_vectors(base, JOINT_DIMENSION, "base")
    values, log_det = cubic_forward(rows, a, covariance_normalized_cubic_scale(a))
    return values.reshape((*leading, JOINT_DIMENSION)), log_det.reshape(leading)


def cubic_log_ratio(values: np.ndarray, a: float) -> np.ndarray:
    rows, leading = _as_vectors(values, JOINT_DIMENSION, "values")
    ratio = cubic_log_prob(rows, a, covariance_normalized_cubic_scale(a)) - standard_normal_log_prob(rows)
    return ratio.reshape(leading)


def _validated_tau(tau: float) -> float:
    value = float(tau)
    if not math.isfinite(value) or value < 0.0 or value > 1.0 / 2.1:
        raise ValueError("tau must be finite and lie in [0, 1/2.1]")
    return value


def student_log_prob(values: np.ndarray, tau: float) -> np.ndarray:
    rows, leading = _as_vectors(values, JOINT_DIMENSION, "values")
    inverse_df = _validated_tau(tau)
    if inverse_df == 0.0:
        return standard_normal_log_prob(rows).reshape(leading)
    degrees = 1.0 / inverse_df
    radius_squared = np.sum(rows * rows, axis=1)
    log_density = (
        gammaln(0.5 * (degrees + JOINT_DIMENSION))
        - gammaln(0.5 * degrees)
        - 0.5 * JOINT_DIMENSION * math.log(math.pi * (degrees - 2.0))
        - 0.5
        * (degrees + JOINT_DIMENSION)
        * np.log1p(radius_squared / (degrees - 2.0))
    )
    return log_density.reshape(leading)


def student_log_ratio(values: np.ndarray, tau: float) -> np.ndarray:
    rows, leading = _as_vectors(values, JOINT_DIMENSION, "values")
    return (student_log_prob(rows, tau) - standard_normal_log_prob(rows)).reshape(leading)


def student_from_base(base: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    rows, leading = _as_vectors(base, JOINT_DIMENSION, "base")
    inverse_df = _validated_tau(tau)
    if inverse_df == 0.0:
        return rows.reshape((*leading, JOINT_DIMENSION)).copy(), np.zeros(leading)
    degrees = 1.0 / inverse_df
    base_radius = np.linalg.norm(rows, axis=1)
    base_square = 0.5 * base_radius**2
    cdf = gammainc(0.5 * JOINT_DIMENSION, base_square)
    survival = gammaincc(0.5 * JOINT_DIMENSION, base_square)
    lower = cdf <= 0.5
    target_radius = np.empty_like(base_radius)
    beta_quantile = _beta_lower_tail_inverse(
        0.5 * JOINT_DIMENSION, 0.5 * degrees, cdf[lower]
    )
    target_radius[lower] = (
        math.sqrt(degrees - 2.0)
        * np.sqrt(beta_quantile)
        / np.sqrt(1.0 - beta_quantile)
    )
    upper_survival = survival[~lower]
    if np.any(upper_survival <= 0.0):
        raise FloatingPointError("student upper-tail probability underflow")
    beta_complement = _beta_lower_tail_inverse(
        0.5 * degrees, 0.5 * JOINT_DIMENSION, upper_survival
    )
    if np.any(beta_complement <= 0.0):
        raise FloatingPointError("student beta-complement quantile underflow")
    target_radius[~lower] = (
        math.sqrt(degrees - 2.0)
        * np.sqrt(1.0 - beta_complement)
        / np.sqrt(beta_complement)
    )
    nonzero = base_radius > 0.0
    multiplier = np.empty_like(base_radius)
    multiplier[nonzero] = target_radius[nonzero] / base_radius[nonzero]
    log_phi_zero = -0.5 * JOINT_DIMENSION * LOG_2PI
    log_student_zero = float(student_log_prob(np.zeros((1, JOINT_DIMENSION)), inverse_df)[0])
    multiplier[~nonzero] = math.exp(
        (log_phi_zero - log_student_zero) / JOINT_DIMENSION
    )
    values = rows * multiplier[:, None]
    log_det = standard_normal_log_prob(rows) - student_log_prob(values, inverse_df)
    return values.reshape((*leading, JOINT_DIMENSION)), log_det.reshape(leading)


def student_to_base(values: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    rows, leading = _as_vectors(values, JOINT_DIMENSION, "values")
    inverse_df = _validated_tau(tau)
    if inverse_df == 0.0:
        return rows.reshape((*leading, JOINT_DIMENSION)).copy(), np.zeros(leading)
    degrees = 1.0 / inverse_df
    value_radius = np.linalg.norm(rows, axis=1)
    radius_squared = value_radius**2
    denominator = radius_squared + degrees - 2.0
    beta_argument = radius_squared / denominator
    beta_complement = (degrees - 2.0) / denominator
    cdf = betainc(0.5 * JOINT_DIMENSION, 0.5 * degrees, beta_argument)
    survival = betainc(0.5 * degrees, 0.5 * JOINT_DIMENSION, beta_complement)
    target_radius = _chi_radius_from_probabilities(cdf, survival, JOINT_DIMENSION)
    nonzero = value_radius > 0.0
    multiplier = np.empty_like(value_radius)
    multiplier[nonzero] = target_radius[nonzero] / value_radius[nonzero]
    log_phi_zero = -0.5 * JOINT_DIMENSION * LOG_2PI
    log_student_zero = float(student_log_prob(np.zeros((1, JOINT_DIMENSION)), inverse_df)[0])
    multiplier[~nonzero] = math.exp(
        (log_student_zero - log_phi_zero) / JOINT_DIMENSION
    )
    base = rows * multiplier[:, None]
    inverse_log_det = student_log_prob(rows, inverse_df) - standard_normal_log_prob(base)
    return base.reshape((*leading, JOINT_DIMENSION)), inverse_log_det.reshape(leading)


@dataclass(frozen=True)
class SearchResult:
    parameter: float
    objective: float
    evaluations: int
    grid_parameters: tuple[float, ...]
    grid_objectives: tuple[float, ...]
    refined_candidates: tuple[tuple[float, float], ...]


def _grid_refine_maximum(
    score: Callable[[float], float],
    grid: np.ndarray,
) -> SearchResult:
    parameters = np.asarray(grid, dtype=np.float64)
    if parameters.ndim != 1 or len(parameters) < 3 or np.any(np.diff(parameters) <= 0.0):
        raise ValueError("search grid must be strictly increasing with at least three points")
    evaluations = 0

    def evaluated(value: float) -> float:
        nonlocal evaluations
        evaluations += 1
        result = float(score(float(value)))
        if not math.isfinite(result):
            raise FloatingPointError("radial search produced a nonfinite objective")
        return result

    grid_values = np.array([evaluated(value) for value in parameters])
    candidate_indices = {0, len(parameters) - 1}
    candidate_indices.update(
        index
        for index in range(1, len(parameters) - 1)
        if grid_values[index] >= grid_values[index - 1]
        and grid_values[index] >= grid_values[index + 1]
    )
    candidates = [(float(parameters[index]), float(grid_values[index])) for index in candidate_indices]
    for index in sorted(candidate_indices):
        left_index = max(0, index - 1)
        right_index = min(len(parameters) - 1, index + 1)
        result = minimize_scalar(
            lambda value: -evaluated(value),
            bounds=(float(parameters[left_index]), float(parameters[right_index])),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 200},
        )
        if not result.success:
            raise RuntimeError(f"radial refinement failed: {result.message}")
        candidates.append((float(result.x), -float(result.fun)))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    best_parameter, best_objective = candidates[0]
    return SearchResult(
        best_parameter,
        best_objective,
        evaluations,
        tuple(float(value) for value in parameters),
        tuple(float(value) for value in grid_values),
        tuple(candidates),
    )


def fit_cubic_parameter(values: np.ndarray) -> SearchResult:
    rows, _ = _as_vectors(values, JOINT_DIMENSION, "values")
    return _grid_refine_maximum(
        lambda coefficient: float(np.mean(cubic_log_ratio(rows, coefficient))),
        np.linspace(0.0, 0.1, 101),
    )


def fit_student_parameter(values: np.ndarray) -> SearchResult:
    rows, _ = _as_vectors(values, JOINT_DIMENSION, "values")
    eta_grid = np.linspace(math.log(0.1), math.log(998.0), 101)
    finite = _grid_refine_maximum(
        lambda eta: float(
            np.mean(student_log_ratio(rows, 1.0 / (2.0 + math.exp(eta))))
        ),
        eta_grid,
    )
    gaussian_objective = float(np.mean(student_log_ratio(rows, 0.0)))
    gaussian_candidate = (math.inf, gaussian_objective)
    finite_candidates = tuple(
        (2.0 + math.exp(eta), objective)
        for eta, objective in finite.refined_candidates
    )
    candidates = (*finite_candidates, gaussian_candidate)
    best_degrees, best_objective = sorted(
        candidates, key=lambda item: (-item[1], item[0])
    )[0]
    parameter = 0.0 if math.isinf(best_degrees) else 1.0 / best_degrees
    return SearchResult(
        parameter,
        best_objective,
        finite.evaluations + 1,
        tuple(1.0 / (2.0 + math.exp(eta)) for eta in finite.grid_parameters) + (0.0,),
        finite.grid_objectives + (gaussian_objective,),
        tuple(
            (0.0 if math.isinf(degrees) else 1.0 / degrees, objective)
            for degrees, objective in candidates
        ),
    )
