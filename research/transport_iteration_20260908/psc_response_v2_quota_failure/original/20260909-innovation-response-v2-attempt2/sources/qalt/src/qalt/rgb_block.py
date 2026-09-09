"""Numerically stable three-dimensional Gaussian scale-mixture blocks.

The module is intentionally independent of the historical scalar routing code.
It implements only the fixed-shape RGB block model and exact deterministic
views required by the registered B1-v2 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri


LOG_2PI = math.log(2.0 * math.pi)
RGB_DIMENSION = 3


def _as_rgb_rows(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != RGB_DIMENSION:
        raise ValueError("expected an array whose final dimension is three")
    if not np.all(np.isfinite(array)):
        raise ValueError("RGB values must be finite")
    return array.reshape(-1, RGB_DIMENSION), array.shape[:-1]


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def project_log_eigenvalues(
    eigenvalues: np.ndarray,
    lower: float = 0.1,
    upper: float = 10.0,
) -> np.ndarray:
    """Project log eigenvalues onto a determinant-one bounded spectrum.

    This is the Euclidean projection onto
    ``{x: sum(x)=0, log(lower)<=x_i<=log(upper)}``.  Its scalar Lagrange
    multiplier is found by bisection.  The result is invariant to multiplying
    every input eigenvalue by a common positive constant.
    """

    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("eigenvalues must be a finite positive vector")
    if not 0.0 < lower <= 1.0 <= upper or not lower < upper:
        raise ValueError("bounds must satisfy 0 < lower <= 1 <= upper")

    logs = np.log(values)
    low_log, high_log = math.log(lower), math.log(upper)
    left = float(np.min(logs - high_log))
    right = float(np.max(logs - low_log))
    for _ in range(100):
        midpoint = 0.5 * (left + right)
        total = float(np.sum(np.clip(logs - midpoint, low_log, high_log)))
        if total > 0.0:
            left = midpoint
        else:
            right = midpoint
    projected_logs = np.clip(logs - 0.5 * (left + right), low_log, high_log)

    # Remove the last few ulps of bisection error without leaving the box.
    error = float(np.sum(projected_logs))
    if error != 0.0:
        candidates = np.flatnonzero(
            (projected_logs - error >= low_log) & (projected_logs - error <= high_log)
        )
        if candidates.size:
            projected_logs[candidates[0]] -= error
    return np.exp(projected_logs)


def fit_determinant_one_shape(
    residual: np.ndarray,
    shrinkage: float = 0.001,
    eigenvalue_lower: float = 0.1,
    eigenvalue_upper: float = 10.0,
) -> np.ndarray:
    """Fit the registered uncentered, shrunk determinant-one RGB shape."""

    rows, _ = _as_rgb_rows(residual)
    if len(rows) == 0:
        raise ValueError("at least one residual is required")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    scatter = rows.T @ rows / len(rows)
    isotropic_variance = float(np.trace(scatter) / RGB_DIMENSION)
    if not np.isfinite(isotropic_variance) or isotropic_variance <= 0.0:
        raise ValueError("residual scatter must have positive trace")
    regularized = (1.0 - shrinkage) * scatter + shrinkage * isotropic_variance * np.eye(RGB_DIMENSION)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (regularized + regularized.T))
    projected = project_log_eigenvalues(eigenvalues, eigenvalue_lower, eigenvalue_upper)
    shape = (eigenvectors * projected) @ eigenvectors.T
    return 0.5 * (shape + shape.T)


def water_filled_weights(masses: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Maximize ``sum masses[k] log(weight[k])`` on a floored simplex."""

    values = np.asarray(masses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("masses must be a finite nonnegative vector")
    if not 0.0 <= floor < 1.0 / values.size:
        raise ValueError("weight floor must satisfy 0 <= floor < 1/K")
    if float(np.sum(values)) <= 0.0:
        raise ValueError("at least one mass must be positive")

    active = np.ones(values.size, dtype=bool)
    weights = np.full(values.size, floor, dtype=np.float64)
    while True:
        inactive_count = int(np.sum(~active))
        remaining = 1.0 - floor * inactive_count
        active_mass = float(np.sum(values[active]))
        if active_mass <= 0.0:
            raise FloatingPointError("positive mass disappeared from active set")
        proposal = remaining * values[active] / active_mass
        below = proposal < floor
        if not np.any(below):
            weights[active] = proposal
            break
        active_indices = np.flatnonzero(active)
        active[active_indices[below]] = False

    # Assign rounding residue to an interior coordinate.
    interior = np.flatnonzero(weights > floor)
    weights[interior[-1]] += 1.0 - float(np.sum(weights))
    return weights


def recycle_component_normal(base_normal: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use one normal coordinate for both a categorical draw and fresh normal.

    The returned component and remapped normal are independent in the exact
    continuous construction.  No auxiliary random coordinate is consumed.
    """

    values = np.asarray(base_normal, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("base normals must be finite")
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or not np.isclose(np.sum(probabilities), 1.0, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("weights must be finite, positive, and sum to one")

    zero_open = np.nextafter(0.0, 1.0)
    one_open = np.nextafter(1.0, 0.0)
    uniforms = np.clip(ndtr(values), zero_open, one_open)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    # The registered intervals are (P[k-1], P[k]], so exact boundaries belong
    # to the lower component.  The choice is distributionally null but fixes
    # deterministic finite-precision copy controls.
    component = np.searchsorted(cumulative, uniforms, side="left")
    component = np.minimum(component, len(probabilities) - 1)
    previous = np.where(component == 0, 0.0, cumulative[np.maximum(component - 1, 0)])
    within = (uniforms - previous) / probabilities[component]
    within = np.clip(within, zero_open, one_open)
    return component.astype(np.int64), ndtri(within)


@dataclass(frozen=True)
class FixedShapeGSM:
    """Three-dimensional zero-mean Gaussian scale mixture with fixed shape."""

    weights: np.ndarray
    scales: np.ndarray
    shape: np.ndarray
    cholesky: np.ndarray = field(init=False, repr=False)
    log_det_cholesky: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).copy()
        scales = np.asarray(self.scales, dtype=np.float64).copy()
        shape = np.asarray(self.shape, dtype=np.float64).copy()
        if (
            weights.ndim != 1
            or scales.shape != weights.shape
            or weights.size == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ValueError("weights must be positive and sum to one")
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("scales must be finite and positive")
        if shape.shape != (RGB_DIMENSION, RGB_DIMENSION) or np.any(~np.isfinite(shape)):
            raise ValueError("shape must be a finite 3x3 matrix")
        shape = 0.5 * (shape + shape.T)
        cholesky = np.linalg.cholesky(shape)
        log_det_cholesky = float(np.sum(np.log(np.diag(cholesky))))
        if abs(2.0 * log_det_cholesky) > 1e-8:
            raise ValueError("shape determinant must equal one")
        weights.setflags(write=False)
        scales.setflags(write=False)
        shape.setflags(write=False)
        cholesky.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "cholesky", cholesky)
        object.__setattr__(self, "log_det_cholesky", log_det_cholesky)

    def _mahalanobis(self, residual: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        rows, leading_shape = _as_rgb_rows(residual)
        whitened = solve_triangular(self.cholesky, rows.T, lower=True, check_finite=False).T
        return np.sum(whitened**2, axis=1), leading_shape

    def component_log_prob(self, residual: np.ndarray) -> np.ndarray:
        radius, leading_shape = self._mahalanobis(residual)
        logs = (
            np.log(self.weights)[None, :]
            - 0.5 * RGB_DIMENSION * LOG_2PI
            - self.log_det_cholesky
            - RGB_DIMENSION * np.log(self.scales)[None, :]
            - 0.5 * radius[:, None] / self.scales[None, :] ** 2
        )
        return logs.reshape((*leading_shape, len(self.weights)))

    def log_prob(self, residual: np.ndarray) -> np.ndarray:
        return _logsumexp(self.component_log_prob(residual), axis=-1)

    def marginal_log_prob(self, residual: np.ndarray) -> np.ndarray:
        rows, leading_shape = _as_rgb_rows(residual)
        standard_deviations = self.scales[:, None] * np.sqrt(np.diag(self.shape))[None, :]
        logs = (
            np.log(self.weights)[None, :, None]
            - 0.5 * LOG_2PI
            - np.log(standard_deviations)[None, :, :]
            - 0.5 * (rows[:, None, :] / standard_deviations[None, :, :]) ** 2
        )
        return _logsumexp(logs, axis=1).reshape((*leading_shape, RGB_DIMENSION))

    def product_log_prob(self, residual: np.ndarray) -> np.ndarray:
        """Log density of the no-refit product of exact fitted marginals."""

        return np.sum(self.marginal_log_prob(residual), axis=-1)

    def zero_correlation(self) -> "FixedShapeGSM":
        """Remove off-diagonal covariance while preserving component marginals."""

        diagonal = np.diag(self.shape)
        determinant_scale = float(np.prod(diagonal) ** (1.0 / RGB_DIMENSION))
        zero_shape = np.diag(diagonal / determinant_scale)
        zero_scales = self.scales * math.sqrt(determinant_scale)
        return FixedShapeGSM(self.weights, zero_scales, zero_shape)

    def exact_scalar_log_prob(self, residual: np.ndarray) -> np.ndarray:
        """Return the three exact chain-rule conditional log densities."""

        rows, leading_shape = _as_rgb_rows(residual)
        output = np.empty((len(rows), RGB_DIMENSION), dtype=np.float64)
        log_weights = np.log(self.weights)
        log_scales = np.log(self.scales)
        for coordinate in range(RGB_DIMENSION):
            if coordinate == 0:
                means = np.zeros(len(rows), dtype=np.float64)
                variance_factor = float(self.shape[0, 0])
                posterior_logs = np.broadcast_to(log_weights, (len(rows), len(self.weights)))
            else:
                leading = self.shape[:coordinate, :coordinate]
                beta = np.linalg.solve(leading, self.shape[:coordinate, coordinate])
                means = rows[:, :coordinate] @ beta
                variance_factor = float(self.shape[coordinate, coordinate] - self.shape[coordinate, :coordinate] @ beta)
                leading_cholesky = np.linalg.cholesky(leading)
                whitened = solve_triangular(
                    leading_cholesky,
                    rows[:, :coordinate].T,
                    lower=True,
                    check_finite=False,
                ).T
                parent_radius = np.sum(whitened**2, axis=1)
                unnormalized = (
                    log_weights[None, :]
                    - coordinate * log_scales[None, :]
                    - 0.5 * parent_radius[:, None] / self.scales[None, :] ** 2
                )
                posterior_logs = unnormalized - _logsumexp(unnormalized, axis=1)[:, None]
            if variance_factor <= 0.0:
                raise FloatingPointError("nonpositive Schur complement")
            deviations = rows[:, coordinate] - means
            component_logs = (
                posterior_logs
                - 0.5 * LOG_2PI
                - log_scales[None, :]
                - 0.5 * math.log(variance_factor)
                - 0.5 * deviations[:, None] ** 2 / (self.scales[None, :] ** 2 * variance_factor)
            )
            output[:, coordinate] = _logsumexp(component_logs, axis=1)
        return output.reshape((*leading_shape, RGB_DIMENSION))

    def sample(self, base_normal: np.ndarray) -> np.ndarray:
        """Sample the joint GSM using exactly the supplied three normals."""

        rows, leading_shape = _as_rgb_rows(base_normal)
        component, remapped_first = recycle_component_normal(rows[:, 0], self.weights)
        standard = rows.copy()
        standard[:, 0] = remapped_first
        residual = (standard @ self.cholesky.T) * self.scales[component, None]
        return residual.reshape((*leading_shape, RGB_DIMENSION))

    def sample_product(self, base_normal: np.ndarray) -> np.ndarray:
        """Sample the product of fitted marginals from the same three normals."""

        rows, leading_shape = _as_rgb_rows(base_normal)
        output = np.empty_like(rows)
        marginal_shape = np.sqrt(np.diag(self.shape))
        for coordinate in range(RGB_DIMENSION):
            component, remapped = recycle_component_normal(rows[:, coordinate], self.weights)
            output[:, coordinate] = remapped * self.scales[component] * marginal_shape[coordinate]
        return output.reshape((*leading_shape, RGB_DIMENSION))


@dataclass(frozen=True)
class GSMFitDiagnostics:
    log_likelihood: tuple[float, ...]
    converged: bool
    initialization: int
    final_log_likelihoods: tuple[float, ...]


def fit_fixed_shape_gsm(
    residual: np.ndarray,
    shape: np.ndarray,
    components: int,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    weight_floor: float = 1e-4,
    scale_floor: float = 0.05,
    scale_ceiling: float = 2.0,
) -> tuple[FixedShapeGSM, GSMFitDiagnostics]:
    """Fit only radial weights and scales with the registered shape frozen."""

    rows, _ = _as_rgb_rows(residual)
    if len(rows) < components or components < 1:
        raise ValueError("the sample must contain at least one row per component")
    if max_iterations < 1 or tolerance < 0.0:
        raise ValueError("invalid EM stopping rule")
    if not 0.0 < weight_floor < 1.0 / components:
        raise ValueError("weight floor must satisfy 0 < floor < 1/K")
    if not 0.0 < scale_floor <= scale_ceiling:
        raise ValueError("invalid scale interval")

    template = FixedShapeGSM(np.array([1.0]), np.array([1.0]), shape)
    radius, _ = template._mahalanobis(rows)
    root_mean_scale = math.sqrt(float(np.mean(radius)) / RGB_DIMENSION)
    geometric = np.clip(
        root_mean_scale * np.geomspace(0.5, 1.8, components),
        scale_floor,
        scale_ceiling,
    )
    quantiles = (np.arange(components, dtype=np.float64) + 0.5) / components
    radial = np.clip(
        np.sqrt(np.quantile(radius / RGB_DIMENSION, quantiles)),
        scale_floor,
        scale_ceiling,
    )
    starts = [geometric]
    if components > 1 and not np.array_equal(geometric, radial):
        starts.append(radial)

    fitted: list[tuple[FixedShapeGSM, tuple[float, ...], bool]] = []
    for initial_scales in starts:
        weights = np.full(components, 1.0 / components, dtype=np.float64)
        scales = np.sort(initial_scales.astype(np.float64, copy=True))
        trace: list[float] = []
        converged = False
        for _ in range(max_iterations):
            model = FixedShapeGSM(weights, scales, shape)
            component_logs = model.component_log_prob(rows)
            point_logs = _logsumexp(component_logs, axis=1)
            old_likelihood = float(np.sum(point_logs))
            if not trace:
                trace.append(old_likelihood)
            responsibilities = np.exp(component_logs - point_logs[:, None])
            masses = np.sum(responsibilities, axis=0)
            new_weights = water_filled_weights(masses, weight_floor)
            new_variances = np.empty(components, dtype=np.float64)
            for component in range(components):
                if masses[component] <= np.finfo(np.float64).tiny:
                    new_variances[component] = scales[component] ** 2
                else:
                    new_variances[component] = float(
                        np.sum(responsibilities[:, component] * radius)
                        / (RGB_DIMENSION * masses[component])
                    )
            new_scales = np.sqrt(np.clip(new_variances, scale_floor**2, scale_ceiling**2))
            order = np.argsort(new_scales)
            weights, scales = new_weights[order], new_scales[order]
            updated = FixedShapeGSM(weights, scales, shape)
            new_likelihood = float(np.sum(updated.log_prob(rows)))
            numerical_tolerance = 1e-10 * (1.0 + abs(old_likelihood))
            if new_likelihood < old_likelihood - numerical_tolerance:
                raise FloatingPointError("constrained EM decreased observed likelihood")
            trace.append(new_likelihood)
            improvement = new_likelihood - old_likelihood
            if improvement <= tolerance * (1.0 + abs(old_likelihood)):
                converged = True
                break
        fitted.append((FixedShapeGSM(weights, scales, shape), tuple(trace), converged))

    final_likelihoods = tuple(trace[-1] for _, trace, _ in fitted)
    best = int(np.argmax(final_likelihoods))
    model, trace, converged = fitted[best]
    return model, GSMFitDiagnostics(trace, converged, best, final_likelihoods)


def within_band_parent_masks(bands: int = 3, colors: int = 3) -> tuple[tuple[int, ...], ...]:
    """Parents for color-autoregressive bands that remain mutually parallel."""

    if bands < 1 or colors < 1:
        raise ValueError("bands and colors must be positive")
    return tuple(
        tuple(band * colors + earlier for earlier in range(color))
        for band in range(bands)
        for color in range(colors)
    )


def full_parent_masks(dimension: int = 9) -> tuple[tuple[int, ...], ...]:
    """Parents for a global scalar autoregressive order."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    return tuple(tuple(range(coordinate)) for coordinate in range(dimension))
